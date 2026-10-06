"""Eight GPU replicas exercise every v2 inference path on a trained smoke run.

Each rank runs one configuration with the run's Stage 2 adapter and the
localization adapter: direct, inline (mask / box, ref / noref), online two-pass,
oracle (mask / box), an inference-time binding override and an uncached
denoising loop, with and without latent blending.  Outputs must be finite,
non-constant RGBA images; every report is kept next to its image.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from accelerate.utils import set_seed

from samtok_edit21.data.io import write_json
from samtok_edit21.data.protocol import REGION_KIND
from samtok_edit21.data.provenance import pass2_text_encoder
from samtok_edit21.models.binding import BindingConfig
from samtok_edit21.models.codec import SamtokCodec
from samtok_edit21.models.pipeline import edit, load_pipeline
from samtok_edit21.training.objectives import adapter_binding, load_adapter


def pick(rows, sample_type, kind, variant=None):
    for row in rows:
        if (row["sample_type"] == sample_type and REGION_KIND.get(row["edit_type"]) == kind
                and row.get("instr_variant") == variant):
            return row
    raise ValueError(f"Smoke data lacks a {sample_type}/{kind}/{variant} row")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--te-adapter", required=True)
    parser.add_argument("--qwen", required=True)
    parser.add_argument("--samtok", required=True)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    set_seed(100 + rank)
    root, data = Path(args.run_root), Path(args.data)
    out = root / "inference"
    out.mkdir(exist_ok=True)
    stage1 = [json.loads(line) for line in (data / "stage1.jsonl").read_text().splitlines()]
    stage2 = [json.loads(line) for line in (data / "stage2.jsonl").read_text().splitlines()]
    dit = root / "stage2" / "adapter"
    config = json.loads((dit / "adapter.json").read_text())
    pass2 = pass2_text_encoder(config["conditioning_identity"], args.qwen, args.samtok, args.te_adapter)
    trained = adapter_binding(config)
    override = BindingConfig("bias_clause" if trained.mode == "none" else "none", trained.beta, trained.eps, trained.rank)
    if trained.mode == "region_embed":
        override = trained  # region_embed cannot be switched at inference
    ntp_mask, ntp_box = pick(stage1, "edit_ntp", "mask"), pick(stage1, "edit_ntp", "box")
    cases = [
        dict(name="direct", row=pick(stage2, "edit", "mask"), mode="direct"),
        dict(name="inline-ref-mask", row=pick(stage2, "edit_umt", "mask", "ref"), mode="inline"),
        dict(name="inline-noref-box-blend", row=pick(stage2, "edit_umt", "box", "noref"), mode="inline", blend=True),
        dict(name="online-box-blend", row=ntp_box, mode="online", blend=True),
        dict(name="oracle-mask-blend", row=ntp_mask, mode="oracle", blend=True),
        dict(name="oracle-box", row=ntp_box, mode="oracle"),
        dict(name="inline-noref-mask-override-blend", row=pick(stage2, "edit_umt", "mask", "noref"),
             mode="inline", blend=True, binding=override),
        dict(name="inline-ref-box-uncached", row=pick(stage2, "edit_umt", "box", "ref"), mode="inline", kv_cache=False),
    ]
    case = cases[rank]
    row = case["row"]
    pipe = load_pipeline(args.qwen, args.samtok, device=f"cuda:{rank}")
    load_adapter(pipe.text_encoder, args.te_adapter)
    load_adapter(pipe.dit, dit)
    pipe.eval()
    codec = SamtokCodec(Path(args.samtok) / "sam2.1_hiera_large.pt", Path(args.samtok) / "mask_tokenizer_256x2.pth",
                        device=f"cuda:{rank}")
    image, report = edit(
        pipe, row["prompt"], [Image.open(row["edit_image"]).convert("RGBA")], mode=case["mode"],
        cot=row.get("mt_cot") if case["mode"] == "oracle" else None, variant="ref", max_new_tokens=192,
        pass2_te=pass2, binding=case.get("binding", trained), codec=codec,
        blend_region="prompt" if case.get("blend") else None,
        height=256, width=256, num_inference_steps=4, cfg_scale=1.0, seed=100 + rank,
        use_kv_cache=case.get("kv_cache", True))
    array = np.asarray(image)
    if image.size != (256, 256) or image.mode != "RGBA" or not np.isfinite(array).all() or array[..., :3].std() <= 0:
        raise AssertionError(f"Invalid smoke image for {case['name']}")
    image.save(out / f"rank{rank}-{case['name']}.png")
    write_json(out / f"rank{rank}.json", {
        "rank": rank, "case": case["name"], "mode": case["mode"], "prompt": row["prompt"],
        "edit_type": row["edit_type"], "report": report, "rgb_std": float(array[..., :3].std()),
        "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30})
    print(json.dumps({"rank": rank, "case": case["name"], "passed": True,
                      "fallback_reason": report["fallback_reason"], "binding": report["binding"],
                      "blend": report["blend"]}), flush=True)


if __name__ == "__main__":
    main()
