"""Token-grounding evaluation (v3): does the model act on the region a mask token names, given only the token?

Every probe feeds the model exactly what inference feeds it - the source image and a prompt containing the token -
with no codec-decoded region, no attention bias and no latent blending.  Masks appear only afterwards, to score.

* highlight      - "Apply a <colour> semi-transparent mask to <X>": read the painted region back from the output and
                   compare it with the token's region (IoU).  Items that carry ``other`` (a second token on the same
                   image) also report which of the two regions the output follows.
* remove probe   - "Remove the object in this region <M>." for the two tokens of a pair: the change must land on the
                   token's own instance, not on the other one.
* edit locality  - benchmark instructions with the token: how much changes inside and far outside the region.

Run on all GPUs of all nodes::

    python -m torch.distributed.run --nproc_per_node 8 -m samtok_edit21 ground-eval \\
        --data <data_dir> --adapter <adapter_dir> --output <eval_dir>

Without ``--adapter`` the stock DiT is evaluated (the base of every comparison).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

from samtok_edit21.data.highlight import PALETTE, REMOVE_PROBE, extract_mask, highlight_prompt, iou, pick_color

EVAL_PROTOCOL = "/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v2/eval/protocol_v2_001"
CANVAS_AREA = 1024 * 1024
FAR_MARGIN = 31          # pixels: "far from the region" excludes this dilation of it
MIN_PROBE_PIXELS = 30    # a pair is scored only when both instances keep this many exclusive pixels at 512 px


def canvas(image, area=CANVAS_AREA):
    """Generation size (height, width): about ``area`` pixels at the source aspect ratio, multiples of 32."""
    ratio = image.width / image.height
    return (max(32, round(math.sqrt(area / ratio) / 32) * 32), max(32, round(math.sqrt(area * ratio) / 32) * 32))


def export(image, size):
    """Model output -> RGB at the source size (white under transparency), as the benchmark scores it."""
    rgba = image.convert("RGBA")
    flat = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB")
    return flat.resize(size, Image.Resampling.LANCZOS)


def grid_mask(mask, height_width):
    """Pixel mask -> boolean mask on the target token grid of a ``height_width`` canvas."""
    height, width = height_width
    image = Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255).resize((width, height), Image.BILINEAR)
    weight = torch.as_tensor(np.asarray(image, np.float32) / 255.0)
    return (F.avg_pool2d(weight[None, None], 16, 16)[0, 0] > 0.5).numpy()


@torch.no_grad()
def generate(pipe, codec, prompt, source_rgb, *, steps=20, seed=0):
    """Token-only inference: inline region tokens, raw text encoder, no binding, no blending, CFG 1."""
    from samtok_edit21.models.binding import BindingConfig
    from samtok_edit21.models.pipeline import edit

    height, width = canvas(source_rgb)
    image, _ = edit(pipe, prompt, [source_rgb.convert("RGBA")], mode="inline", height=height, width=width,
                    num_inference_steps=steps, cfg_scale=1.0, seed=seed, use_kv_cache=True, pass2_te="raw",
                    binding=BindingConfig("none"), codec=codec, blend_region=None)
    return export(image, source_rgb.size), (height, width)


def item_span(codec, source_rgb, item):
    """The item's token: given, or encoded from a user mask (``gt_mask``) the way an interactive user would."""
    if "gt_mask" not in item:
        return item["span"], None
    truth = np.asarray(Image.open(item["gt_mask"]).convert("L")) > 0
    return codec.encode(source_rgb.convert("RGBA"), [truth])[0][0], truth


@torch.no_grad()
def highlight_item(pipe, codec, item, *, steps=20):
    """Returns ``(record, output image, read-back mask)`` for one highlight probe."""
    source = Image.open(item["source"]).convert("RGB")
    record = {key: item[key] for key in ("set", "variant", "case_id", "pair", "tok", "unseen", "edit_type") if key in item}
    span, truth = item_span(codec, source, item)
    color = pick_color(source)
    output, height_width = generate(pipe, codec, highlight_prompt(span, item.get("phrase"), color), source, steps=steps)
    predicted = extract_mask(source, output, PALETTE[color])
    decoded = codec.decode_strict(source.convert("RGBA"), span)[0] > 0
    record.update(color=color, area=float(decoded.mean()), pred_area=float(predicted.mean()),
                  iou_codec=iou(predicted, decoded),
                  iou_codec_grid=iou(grid_mask(predicted, height_width), grid_mask(decoded, height_width)))
    if truth is not None:
        record.update(area=float(truth.mean()), iou_gt=iou(predicted, truth),
                      iou_gt_grid=iou(grid_mask(predicted, height_width), grid_mask(truth, height_width)),
                      codec_vs_gt=iou(decoded, truth))
    if "other" in item:
        other = codec.decode_strict(source.convert("RGBA"), item["other"])[0] > 0
        shared = decoded & other
        own, rival = decoded & ~shared, other & ~shared
        record.update(iou_self=iou(predicted, own), iou_other=iou(predicted, rival),
                      valid=bool(own.sum() > 0 and rival.sum() > 0))
    return record, output, predicted


@torch.no_grad()
def remove_probe_item(pipe, codec, item, *, steps=20):
    """Mean absolute change on the token's own instance, on the other instance, and elsewhere."""
    source = Image.open(item["source"]).convert("RGB")
    output, _ = generate(pipe, codec, REMOVE_PROBE.format(span=item["span"]), source, steps=steps)
    size = (512, round(512 * source.height / source.width))
    before, after = np.asarray(source.resize(size), np.float32), np.asarray(output.resize(size), np.float32)

    def region(span):
        mask = codec.decode_strict(source.convert("RGBA"), span)[0].astype(np.uint8) * 255
        return np.asarray(Image.fromarray(mask).resize(size, Image.NEAREST)) > 127

    own, other = region(item["span"]), region(item["other"])
    shared = own & other
    own, other = own & ~shared, other & ~shared
    change = np.abs(after - before).mean(-1)
    record = {"pair": item["pair"], "tok": item["tok"], "unseen": item.get("unseen"),
              "valid": bool(own.sum() >= MIN_PROBE_PIXELS and other.sum() >= MIN_PROBE_PIXELS)}
    if record["valid"]:
        record.update(change_own=float(change[own].mean()), change_other=float(change[other].mean()),
                      change_far=float(change[~(own | other)].mean()))
    return record, output


@torch.no_grad()
def edit_locality_item(pipe, codec, case, compiled, *, steps=20):
    """A benchmark instruction (phrase + token): mean change inside the user's region and far outside it."""
    source = Image.open(case["source_image"]).convert("RGB")
    truth = np.asarray(Image.open(case["region"]["mask"]).convert("L")) > 0
    span = codec.encode(source.convert("RGBA"), [truth])[0][0]
    template = compiled.get("ref_template") or compiled["noref_template"]
    output, _ = generate(pipe, codec, template.replace("{region}", span), source, steps=steps)
    change = np.abs(np.asarray(output, np.float32) - np.asarray(source, np.float32)).mean(-1)
    near = np.asarray(Image.fromarray(truth.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(FAR_MARGIN))) > 0
    far = ~near
    record = {"case_id": case["case_id"], "edit_type": compiled["edit_type"], "area": float(truth.mean()),
              "diff_in": float(change[truth].mean()), "diff_out": float(change[far].mean()) if far.any() else 0.0}
    return record, output


def metric_key(record):
    """IoU against the user's mask when the item has one, otherwise against the token's decoded region."""
    return "iou_gt" if "iou_gt" in record else "iou_codec"


def pair_follow(records, own="iou_self", other="iou_other"):
    """Per pair: both tokens select their own instance.  Returns ``{pair: bool}`` over pairs with two valid probes."""
    pairs = {}
    for record in records:
        if "pair" in record and record.get("valid"):
            pairs.setdefault(record["pair"], []).append(record[own] > record[other])
    return {pair: all(votes) for pair, votes in pairs.items() if len(votes) == 2}


def quick_summary(records):
    """Small-sample summary logged during training (held-out / dev, without and with the phrase, pair follow)."""

    def mean(name, variant):
        values = [r[metric_key(r)] for r in records if r["set"] == name and r["variant"] == variant]
        return round(float(np.mean(values)), 3) if values else None

    follow = list(pair_follow([r for r in records if r["set"] == "pair"]).values())
    return {"n": len(records), "ho_noref": mean("ho", "noref"), "ho_ref": mean("ho", "ref"),
            "dev_noref": mean("dev", "noref"), "dev_ref": mean("dev", "ref"),
            "pair_follow": round(float(np.mean(follow)), 3) if follow else None, "pairs": len(follow)}


def thumbnail(image, side=768):
    scale = side / max(image.size)
    if scale >= 1:
        return image
    return image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)


def build_jobs(items, cases, *, has_adapter, remove_pairs, limit):
    """(kind, index, model) triples in a fixed order, so every rank derives the same list."""
    main = "trained" if has_adapter else "base"
    counts, jobs = {}, []
    for index, item in enumerate(items):
        counts[item["set"]] = counts.get(item["set"], 0) + 1
        if not limit or counts[item["set"]] <= limit:
            jobs.append(("highlight", index, main))
    probes = [index for index, item in enumerate(items) if item["set"] == "pair" and item["pair"] < remove_pairs]
    models = ("trained", "base") if has_adapter else ("base",)
    jobs += [("remove", index, model) for model in models for index in probes]
    jobs += [("edit", index, model) for model in models for index in range(len(cases))]
    return jobs


def load_edit_cases(protocol, count):
    protocol = Path(protocol)
    compiled = {row["case_id"]: row for row in map(json.loads, open(protocol / "compiled.jsonl"))}
    cases = sorted(map(json.loads, open(protocol / "cases.jsonl")), key=lambda case: case["case_id"])
    cases = [c for c in cases if c["split"] == "dev" and compiled[c["case_id"]]["region_kind"] == "mask"][:count]
    return cases, compiled


def parser():
    from samtok_edit21.models.pipeline import DEFAULT_QWEN, DEFAULT_SAMTOK

    p = argparse.ArgumentParser(prog="samtok-edit ground-eval", description=__doc__.split("\n\n")[0])
    p.add_argument("--data", required=True, help="Directory with eval.json")
    p.add_argument("--output", required=True)
    p.add_argument("--adapter", help="v3 adapter directory; omit to evaluate the stock DiT")
    p.add_argument("--qwen", default=DEFAULT_QWEN)
    p.add_argument("--samtok", default=DEFAULT_SAMTOK)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--remove-pairs", type=int, default=120, help="Pairs used for the remove probe")
    p.add_argument("--edit-cases", type=int, default=48, help="Benchmark dev cases used for edit locality")
    p.add_argument("--eval-protocol", default=EVAL_PROTOCOL, help="Directory with cases.jsonl and compiled.jsonl")
    p.add_argument("--limit", type=int, default=0, help="Smoke test: at most this many highlight items per set")
    return p


def main(argv=None):
    import torch.distributed as dist

    from samtok_edit21.models.code_injector import trained_weights
    from samtok_edit21.models.codec import load_codec
    from samtok_edit21.models.pipeline import load_pipeline
    from samtok_edit21.training.grounding import load_grounding_adapter

    args = parser().parse_args(argv)
    dist.init_process_group("nccl", timeout=timedelta(hours=2))
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    device = f"cuda:{local}"
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    pipe = load_pipeline(args.qwen, args.samtok, device=device)
    codec = load_codec(args.samtok, device)
    if args.adapter:
        load_grounding_adapter(pipe, codec, args.adapter)
    pipe.eval()
    items = json.load(open(Path(args.data) / "eval.json"))
    cases, compiled = load_edit_cases(args.eval_protocol, args.edit_cases)
    jobs = build_jobs(items, cases, has_adapter=bool(args.adapter), remove_pairs=args.remove_pairs, limit=args.limit)
    mine = jobs[rank::world]
    results = {"recs": [], "remove_pairs": [], "edits": []}
    started = time.time()
    for done, (kind, index, model) in enumerate(mine):
        try:
            with trained_weights(pipe, model == "trained"):
                if kind == "highlight":
                    record, image, predicted = highlight_item(pipe, codec, items[index], steps=args.steps)
                    record["idx"] = index
                    folder = out / "images" / items[index]["set"]
                    folder.mkdir(parents=True, exist_ok=True)
                    small = thumbnail(image)
                    small.save(folder / f"{index:05d}.jpg", quality=88)
                    Image.fromarray(predicted.astype(np.uint8) * 255).resize(small.size, Image.NEAREST).save(
                        folder / f"{index:05d}_pred.png")
                    results["recs"].append(record)
                elif kind == "remove":
                    record, image = remove_probe_item(pipe, codec, items[index], steps=args.steps)
                    record.update(idx=index, model=model)
                    folder = out / "remove_pairs" / model
                    folder.mkdir(parents=True, exist_ok=True)
                    thumbnail(image).save(folder / f"{index:05d}.jpg", quality=88)
                    results["remove_pairs"].append(record)
                else:
                    case = cases[index]
                    record, image = edit_locality_item(pipe, codec, case, compiled[case["case_id"]], steps=args.steps)
                    record["model"] = model
                    folder = out / "edits" / model
                    folder.mkdir(parents=True, exist_ok=True)
                    thumbnail(image).save(folder / f"{case['case_id']}.jpg", quality=88)
                    results["edits"].append(record)
        except Exception as error:  # one unreadable image must not lose a multi-hour evaluation
            print(f"[rank {rank}] eval skip {kind} {index} ({model}): {error!r}"[:400], flush=True)
        if rank == 0 and done % 10 == 0:
            print(json.dumps({"done": done, "of": len(mine), "sec": round(time.time() - started)}), flush=True)
    gathered = [None] * world
    dist.all_gather_object(gathered, results)
    if rank == 0:
        merged = {key: [record for part in gathered for record in part[key]] for key in results}
        merged["recs"].sort(key=lambda record: record["idx"])
        json.dump({"adapter": str(args.adapter) if args.adapter else None, "steps": args.steps,
                   "jobs": len(jobs), **merged}, open(out / "eval.json", "w"))
        (out / "EVAL_DONE").write_text("ok")
        print("wrote", out / "eval.json", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
