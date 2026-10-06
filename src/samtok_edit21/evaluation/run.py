"""Run one model on the v2 benchmark cases (torchrun: one process per GPU).

Settings (plan section 8.7; region kind from the compiled edit type):

=========  ==========================================  ==========================================
setting    mask regions (remove/replace/attribute/...)  box regions (add)
=========  ==========================================  ==========================================
mask       codec(user mask); blend: user mask           box of user mask; blend: box +10%
box        best SAM2(box) mask -> codec; blend: box     user box; blend: box +10%
point      best SAM2(point) mask -> codec; blend: mask  median training add box at the point (D5)
text       pass 1 (ref prompt); blend: decoded regions  pass 1; blend: decoded boxes +10%
text_plain the plain instruction, no region (add only, D9)
=========  ==========================================  ==========================================

The DiT binding always reads the regions decoded from the prompt's tokens
(D8); blending uses the user's raw region. ``--stock`` runs stock
Qwen-Image-2.1 with the benchmark's two-image locator protocol instead.
Outputs: ``<output>/<setting>[+blend]/<case_id>.png`` (source size, RGB,
white-composited) with a JSON sidecar; finished samples are skipped.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from samtok_edit21.data.io import write_json
from samtok_edit21.data.protocol import box_coords, box_of, pixel_box, relative_box
from samtok_edit21.evaluation.cases import BENCHMARK_REPO, BENCHMARK_ROOT, SETTINGS

DEFAULT_ADD_BOX = (214, 267)  # median width/height of the v2 training add boxes (0-1000)


def official_size(image, area=1024 * 1024):
    ratio = image.width / image.height
    width = round(math.sqrt(area * ratio) / 32) * 32
    height = round(math.sqrt(area / ratio) / 32) * 32
    return max(32, width), max(32, height)


def best_candidate(masks, scores):
    order = [i for i in np.argsort(scores)[::-1] if np.asarray(masks[i]).any()]
    if not order:
        raise ValueError("SAM2 produced no nonempty candidate")
    return np.asarray(masks[order[0]]) > 0


def point_box(point, width, height, size=DEFAULT_ADD_BOX):
    cx, cy = 1000 * point[0] / width, 1000 * point[1] / height
    w, h = size
    x1, y1 = min(max(0, round(cx - w / 2)), 1000 - w), min(max(0, round(cy - h / 2)), 1000 - h)
    return box_of((x1, y1, x1 + w, y1 + h))


def region_inputs(case, setting, kind, source, codec):
    """(region token, blend region in source pixels, details) for one interactive setting."""
    from samtok_edit21.models.binding import box_pixels
    from samtok_edit21.regions.selection import segment

    width, height = source.size
    region = case["region"]
    if setting == "mask":
        mask = np.asarray(Image.open(region["mask"]).convert("L")) > 0
        if mask.shape != (height, width):
            raise ValueError(f"{case['case_id']}: region mask is not in source pixels")
        if kind == "box":
            token = box_of(pixel_box(mask))
            return token, box_pixels(token, width, height, expand=0.1), {}
        return codec.encode(source, [mask])[0][0], mask, {}
    if setting == "box":
        x1, y1, x2, y2 = (float(v) for v in region["box"])
        x2, y2 = min(max(x2, x1 + 1), width), min(max(y2, y1 + 1), height)
        box = box_of(relative_box((x1, y1, x2, y2), width, height))
        if kind == "box":
            return box, box_pixels(box, width, height, expand=0.1), {}
        mask = best_candidate(*segment(codec, source, box=(x1, y1, x2, y2)))
        return codec.encode(source, [mask])[0][0], box_pixels(box, width, height), {"sam2_area": float(mask.mean())}
    if setting == "point":
        x, y = region["point"]
        if kind == "box":
            box = point_box((x, y), width, height)
            return box, box_pixels(box, width, height, expand=0.1), {"default_box": list(box_coords(box))}
        mask = best_candidate(*segment(codec, source, points=[(float(x), float(y), 1)]))
        return codec.encode(source, [mask])[0][0], mask, {"sam2_area": float(mask.mean())}
    raise ValueError(setting)


def export(image, source):
    """Benchmark view: alpha over white, RGB, resized to the source size."""
    rgba = image.convert("RGBA")
    rgb = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB")
    return rgb.resize(source.size, Image.Resampling.LANCZOS)


def load_stock(args, device):
    from glob import glob
    from diffsynth.core import ModelConfig
    from diffsynth.pipelines.qwen_image_21 import QwenImage21Pipeline
    return QwenImage21Pipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device=device,
        model_configs=[ModelConfig(path=sorted(glob(str(Path(args.qwen) / component / "*.safetensors"))))
                       for component in ("transformer", "text_encoder", "vae")],
        processor_config=ModelConfig(path=str(Path(args.qwen) / "processor")))


def stock_inputs(case, setting, source):
    """The benchmark's frozen two-image locator protocol for one atomic case."""
    if str(BENCHMARK_REPO) not in sys.path:
        sys.path.insert(0, str(BENCHMARK_REPO))
    from evaluation.common import render_annotation, two_image_locator_prompt
    if setting == "text":
        return [source], case["instruction"]
    region = {"mask": case["region"]["mask"], "box": case["region"]["box"], "point": case["region"]["point"]}
    locator = render_annotation(source, [region], Path(BENCHMARK_ROOT), setting).convert("RGB")
    return [source, locator], two_image_locator_prompt(case["region_instruction"], 1, setting)


def stock_generate(pipe, prompt, images, seed, size, steps):
    # The official TE wrapper leaves a norm hook per call; drop only new ones.
    norm = pipe.text_encoder.model.model.language_model.norm
    previous = set(norm._forward_hooks)
    try:
        width, height = size
        return pipe(prompt, edit_image=images, seed=seed, num_inference_steps=steps, cfg_scale=1.0,
                    height=height, width=width, use_kv_cache=True, rand_device="cpu")
    finally:
        for key in set(norm._forward_hooks) - previous:
            del norm._forward_hooks[key]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cases", required=True)
    parser.add_argument("--compiled", help="compiled.jsonl (required except for --stock)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--settings", nargs="+", choices=SETTINGS, default=list(SETTINGS))
    parser.add_argument("--blend", choices=("off", "on", "both"), default="both")
    parser.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    parser.add_argument("--datasets", nargs="+", help="Restrict to these source datasets")
    parser.add_argument("--limit", type=int, help="First N cases after filtering (smoke)")
    parser.add_argument("--qwen", default="/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen-Image-2.1")
    parser.add_argument("--samtok", default="/mnt/bn/strategy-mllm-train/user/tanyue/models/SAMTok/Qwen3-VL-8B-SAMTok")
    parser.add_argument("--te-adapter", help="Stage 1 localization adapter (pass 1)")
    parser.add_argument("--dit-adapter", help="Stage 2 DiT adapter")
    parser.add_argument("--binding", default="adapter",
                        choices=("adapter", "none", "bias_span", "bias_clause", "region_rope", "region_embed"))
    parser.add_argument("--binding-beta", type=float)
    parser.add_argument("--binding-eps", type=float)
    parser.add_argument("--stock", action="store_true", help="Stock Qwen-Image-2.1, two-image protocol")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args(argv)
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local}"
    torch.cuda.set_device(local)
    cases = [json.loads(line) for line in Path(args.cases).read_text().splitlines() if line.strip()]
    cases = [c for c in cases if args.split == "all" or c["split"] == args.split]
    if args.datasets:
        cases = [c for c in cases if c["source_dataset"] in args.datasets]
    cases = cases[:args.limit] if args.limit else cases
    compiled = {}
    if not args.stock:
        if not args.compiled or not args.dit_adapter:
            raise SystemExit("A v2 run needs --compiled and --dit-adapter")
        compiled = {e["case_id"]: e for e in map(json.loads, Path(args.compiled).read_text().splitlines()) if e}
    if args.stock and (args.blend != "off" or set(args.settings) - {"mask", "box", "point", "text"}):
        raise SystemExit("--stock reproduces the benchmark protocol: --blend off, mask/box/point/text")
    output = Path(args.output)
    jobs = []
    for case in cases:
        kind = "stock" if args.stock else compiled[case["case_id"]]["region_kind"]
        for setting in args.settings:
            if setting == "text_plain" and kind != "box":
                continue  # D9: plain text-only is the add comparison
            blends = [False] if setting == "text_plain" or args.blend == "off" else \
                     [True] if args.blend == "on" else [False, True]
            jobs += [(case, setting, blend) for blend in blends]
    jobs = jobs[rank::world]
    if args.stock:
        pipe, codec, config = load_stock(args, device), None, {}
    else:
        from samtok_edit21.data.provenance import pass2_text_encoder
        from samtok_edit21.models.binding import BindingConfig
        from samtok_edit21.models.codec import SamtokCodec
        from samtok_edit21.models.pipeline import load_pipeline
        from samtok_edit21.training.objectives import adapter_binding, load_adapter
        config = json.loads((Path(args.dit_adapter) / "adapter.json").read_text())
        pass2 = pass2_text_encoder(config["conditioning_identity"], args.qwen, args.samtok, args.te_adapter)
        binding = adapter_binding(config)
        if args.binding != "adapter":
            if (args.binding == "region_embed") != (binding.mode == "region_embed"):
                raise SystemExit("region_embed exists only in an adapter trained with it")
            binding = BindingConfig(args.binding, binding.beta if args.binding_beta is None else args.binding_beta,
                                    binding.eps if args.binding_eps is None else args.binding_eps, binding.rank)
        pipe = load_pipeline(args.qwen, args.samtok, device=device)
        if args.te_adapter:
            load_adapter(pipe.text_encoder, args.te_adapter)
        load_adapter(pipe.dit, args.dit_adapter)
        pipe.eval()
        codec = SamtokCodec(Path(args.samtok) / "sam2.1_hiera_large.pt",
                            Path(args.samtok) / "mask_tokenizer_256x2.pth", device=device)
    if rank == 0:
        write_json(output / "run.json", {"args": vars(args), "world_size": world, "jobs": len(jobs) * world,
                                         "dit_adapter_config": config.get("binding") if config else "stock"})
    from samtok_edit21.models.pipeline import edit
    for done, (case, setting, blend) in enumerate(jobs, 1):
        directory = output / (setting + ("+blend" if blend else ""))
        path = directory / f"{case['case_id']}.png"
        if path.is_file() and path.with_suffix(".json").is_file():
            continue
        source = Image.open(case["source_image"]).convert("RGB")
        width, height = official_size(source)
        started = time.perf_counter()
        record = {"case_id": case["case_id"], "setting": setting, "blend": blend, "split": case["split"],
                  "source_dataset": case["source_dataset"], "benchmark_type": case["benchmark_type"],
                  "native_size": [width, height], "seed": args.seed}
        if args.stock:
            images, prompt = stock_inputs(case, setting, source)
            image = stock_generate(pipe, prompt, images, args.seed, (width, height), args.steps)
            record.update(prompt=prompt, inputs=len(images))
        else:
            entry = compiled[case["case_id"]]
            kwargs = dict(height=height, width=width, num_inference_steps=args.steps, cfg_scale=1.0,
                          seed=args.seed, use_kv_cache=True, pass2_te=pass2, binding=binding, codec=codec)
            details = {}
            if setting in {"mask", "box", "point"}:
                token, region, details = region_inputs(case, setting, entry["region_kind"], source, codec)
                prompt, mode = entry["noref_template"].replace("{region}", token), "inline"
                kwargs["blend_region"] = region if blend else None
            elif setting == "text":
                prompt, mode = case["instruction"], "online"
                kwargs.update(blend_region="prompt" if blend else None, max_new_tokens=args.max_new_tokens)
            else:
                prompt, mode = case["instruction"], "direct"
            image, report = edit(pipe, prompt, [source.convert("RGBA")], mode=mode, **kwargs)
            record.update(prompt=prompt, mode=mode, edit_type=entry["edit_type"], region_kind=entry["region_kind"],
                          report=report, **details)
        directory.mkdir(parents=True, exist_ok=True)
        export(image, source).save(path)
        record["seconds"] = time.perf_counter() - started
        write_json(path.with_suffix(".json"), record)
        print(json.dumps({"rank": rank, "done": done, "of": len(jobs), "case": case["case_id"],
                          "setting": setting, "blend": blend, "seconds": round(record["seconds"], 1)}), flush=True)


if __name__ == "__main__":
    main()
