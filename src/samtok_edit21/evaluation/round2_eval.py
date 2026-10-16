"""Round-2 evaluation (docs/08 section 5.1): token-only probes on the held-out source images.

Every probe feeds the model what inference feeds it - the source image and a prompt with the token(s) - and scores
the output afterwards against masks the model never saw:

* single  - one highlight; IoU of the painted region with the token's decoded region (``iou``) and with the dense
            annotation it was encoded from (``iou_gt``);
* multi   - 2-4 tokens in one prompt, one colour each; per-region IoU, all-correct, and colour painted into another
            token's region (cross-talk);
* pair    - the same blind highlight prompt with token A, then token B (instances of one category, or a part and
            its host); a pair follows when both outputs overlap their own region more than the other one;
* edit    - SA-1B 50k dev edits with the blind no-ref sentence + token: change outside the support, distance to
            the teacher target inside it, change inside the really changed pixels;
* remove  - "Remove the selected region <M>." on both instances of a pair: the change must land on the own one.

    python -m torch.distributed.run --nproc_per_node 8 -m samtok_edit21 round2-eval \\
        --data <round2 dir> --output <eval dir> [--adapter <adapter dir>]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter

from samtok_edit21.data.highlight import OVERLAY_ALPHA, PALETTE, extract_mask, iou, pick_color
from samtok_edit21.evaluation.grounding import generate, thumbnail
from samtok_edit21.preparation import instruction_protocol_v2 as p2
from samtok_edit21.preparation import round2_data as rd

EVAL_SIDE = 512          # edit and remove metrics are computed at this width
MIN_PROBE_PIXELS = 30


def extract_masks(source_rgb, output_rgb, colors, min_energy=0.02):
    """Per-colour read-back of a multi-colour highlight: each pixel is assigned to the closest model among
    "unchanged" and "50 % overlay of colour k" (pixels where colour k is too close to the source are not assigned
    to it), then lightly smoothed."""
    s = np.asarray(source_rgb, np.float32) / 255.0
    o = np.asarray(output_rgb.resize(source_rgb.size, Image.BILINEAR), np.float32) / 255.0
    dist = [np.linalg.norm(o - s, axis=-1)]
    for color in colors:
        c = np.asarray(PALETTE[color], np.float32)[None, None] / 255.0
        d = np.linalg.norm(o - (s + OVERLAY_ALPHA * (c - s)), axis=-1)
        d[((c - s) ** 2).sum(-1) < min_energy] = np.inf
        dist.append(d)
    label = np.argmin(np.stack(dist), axis=0)
    masks = []
    for k in range(1, len(colors) + 1):
        smooth = Image.fromarray(((label == k) * 255).astype(np.uint8)).filter(ImageFilter.BoxBlur(2))
        masks.append(np.asarray(smooth, np.float32) / 255.0 > 0.5)
    return masks


def entry_masks(entry, index):
    m = entry["masks"][index]
    return rd.decode_rle(m["dec"]), rd.decode_rle(m["rle"]), m


def single_item(pipe, codec, item, pool, *, steps=20):
    entry = pool(item["source_id"])
    source = Image.open(entry["image"]).convert("RGB")
    decoded, truth, m = entry_masks(entry, item["mask"])
    color = pick_color(source)
    prompt = rd.highlight_prompt_r2(m["span"], color, item.get("phrase"))
    output, _ = generate(pipe, codec, prompt, source, steps=steps)
    predicted = extract_mask(source, output, PALETTE[color])
    record = {"set": "single", "variant": item["variant"], "area": m["area"], "color": color,
              "iou": iou(predicted, decoded), "iou_gt": iou(predicted, truth), "pred_area": float(predicted.mean())}
    return record, output, predicted


def multi_item(pipe, codec, item, pool, *, steps=20):
    entry = pool(item["source_id"])
    source = Image.open(entry["image"]).convert("RGB")
    decoded = [entry_masks(entry, i)[0] for i in item["masks"]]
    spans = [entry["masks"][i]["span"] for i in item["masks"]]
    output, _ = generate(pipe, codec, rd.multi_prompt(spans, item["colors"]), source, steps=steps)
    predicted = extract_masks(source, output, item["colors"])
    ious = [iou(p, d) for p, d in zip(predicted, decoded)]
    cross = []
    for k, p in enumerate(predicted):
        others = np.zeros_like(p)
        for j, d in enumerate(decoded):
            if j != k:
                others |= d
        cross.append(float((p & others).sum()) / max(float(p.sum()), 1.0))
    record = {"set": "multi", "n": len(spans), "ious": ious, "all_correct": bool(min(ious) >= 0.5),
              "cross_talk": float(np.mean(cross)), "areas": [entry["masks"][i]["area"] for i in item["masks"]]}
    return record, output, predicted


def pair_item(pipe, codec, item, pool, *, steps=20):
    entry = pool(item["source_id"])
    source = Image.open(entry["image"]).convert("RGB")
    own, _, m = entry_masks(entry, item["mask"])
    other = entry_masks(entry, item["other"])[0]
    color = pick_color(source)
    output, _ = generate(pipe, codec, rd.highlight_prompt_r2(m["span"], color), source, steps=steps)
    predicted = extract_mask(source, output, PALETTE[color])
    shared = own & other
    record = {"set": "pair", "pair": item["pair"], "pair_kind": item["pair_kind"], "tok": item["tok"],
              "iou_self": iou(predicted, own & ~shared), "iou_other": iou(predicted, other & ~shared),
              "valid": bool((own & ~shared).sum() > 0 and (other & ~shared).sum() > 0)}
    if item["pair_kind"] == "part_host":
        # the part lies inside its host: a correct part highlight covers the part and little of the rest
        record.update(iou_self=iou(predicted, own), iou_other=iou(predicted, other), valid=True)
    return record, output, predicted


def small(image, side=EVAL_SIDE):
    return image.resize((side, max(1, round(side * image.height / image.width))), Image.BILINEAR)


def small_mask(mask, size):
    return np.asarray(Image.fromarray(np.asarray(mask, np.uint8) * 255).resize(size, Image.NEAREST)) > 127


def edit_item(pipe, codec, item, ctx, *, steps=20):
    record = ctx.release[item["sample_id"]]
    token = ctx.tokens[item["sample_id"]]
    root = rd.RELEASE
    source = Image.open(root / record["source_image"]).convert("RGB")
    target = Image.open(root / record["edited_image"]).convert("RGB")
    prompt = p2.render(record["instruction_spec"], False, selector_token=token["span"])
    output, _ = generate(pipe, codec, prompt, source, steps=steps)
    s, t, o = small(source), small(target), small(output)
    size = s.size
    support = small_mask(np.asarray(Image.open(root / record["support_mask"]).convert("L")) > 0, size)
    changed = record["edited_region"]
    changed_px = rd.decode_rle({"size": changed["rle_size"], "counts": changed["rle_counts"]})
    changed = small_mask(changed_px, size)
    S, T, O = (np.asarray(x, np.float32) for x in (s, t, o))
    diff_src, diff_tgt = np.abs(O - S).mean(-1), np.abs(O - T).mean(-1)
    out = {"set": "edit", "sample_id": item["sample_id"], "operation": item["operation"], "prompt": prompt,
           "out_change": float(diff_src[~support].mean()) if (~support).any() else 0.0,
           "in_target_err": float(diff_tgt[support].mean()),
           "in_change": float(diff_src[changed].mean()) if changed.any() else 0.0,
           "teacher_in_change": float(np.abs(T - S).mean(-1)[changed].mean()) if changed.any() else 0.0}
    return out, output, None


def remove_item(pipe, codec, item, pool, *, steps=20):
    entry = pool(item["source_id"])
    source = Image.open(entry["image"]).convert("RGB")
    own, _, m = entry_masks(entry, item["mask"])
    other = entry_masks(entry, item["other"])[0]
    output, _ = generate(pipe, codec, f"Remove the selected region {m['span']}.", source, steps=steps)
    s, o = small(source), small(output)
    own, other = small_mask(own, s.size), small_mask(other, s.size)
    shared = own & other
    own, other = own & ~shared, other & ~shared
    change = np.abs(np.asarray(o, np.float32) - np.asarray(s, np.float32)).mean(-1)
    rec = {"set": "remove", "pair": item["pair"], "tok": item["tok"],
           "valid": bool(own.sum() >= MIN_PROBE_PIXELS and other.sum() >= MIN_PROBE_PIXELS)}
    if rec["valid"]:
        rec.update(change_own=float(change[own].mean()), change_other=float(change[other].mean()),
                   change_far=float(change[~(own | other)].mean()))
    return rec, output, None


def run_item(pipe, codec, item, ctx, steps=20):
    kind = item["set"]
    if kind == "single":
        return single_item(pipe, codec, item, ctx.pool, steps=steps)
    if kind == "multi":
        return multi_item(pipe, codec, item, ctx.pool, steps=steps)
    if kind == "pair":
        return pair_item(pipe, codec, item, ctx.pool, steps=steps)
    if kind == "edit":
        return edit_item(pipe, codec, item, ctx, steps=steps)
    return remove_item(pipe, codec, item, ctx.pool, steps=steps)


def pair_follow(records, own="iou_self", other="iou_other"):
    pairs = {}
    for r in records:
        if r.get("valid"):
            pairs.setdefault(r["pair"], []).append(r[own] > r[other])
    return {pair: all(v) for pair, v in pairs.items() if len(v) == 2}


def summarize(records):
    """Headline numbers of an evaluation (also used for the in-training quick evaluation)."""
    out = {"n": len(records)}
    single = [r for r in records if r["set"] == "single"]
    for variant in ("noref", "ref"):
        values = [r["iou"] for r in single if r["variant"] == variant]
        out[f"single_{variant}"] = round(float(np.mean(values)), 3) if values else None
    small_noref = [r["iou"] for r in single if r["variant"] == "noref" and r["area"] < 0.005]
    out["single_noref_small"] = round(float(np.mean(small_noref)), 3) if small_noref else None
    multi = [r for r in records if r["set"] == "multi"]
    if multi:
        out["multi_all_correct"] = round(float(np.mean([r["all_correct"] for r in multi])), 3)
        out["multi_region_iou"] = round(float(np.mean([v for r in multi for v in r["ious"]])), 3)
        out["multi_cross_talk"] = round(float(np.mean([r["cross_talk"] for r in multi])), 3)
    pairs = [r for r in records if r["set"] == "pair"]
    for kind in ("instance", "part_host"):
        follow = list(pair_follow([r for r in pairs if r["pair_kind"] == kind]).values())
        out[f"pair_follow_{kind}"] = round(float(np.mean(follow)), 3) if follow else None
    edits = [r for r in records if r["set"] == "edit"]
    if edits:
        out["edit_out_change"] = round(float(np.mean([r["out_change"] for r in edits])), 2)
        out["edit_in_target_err"] = round(float(np.mean([r["in_target_err"] for r in edits])), 2)
    removes = [r for r in records if r["set"] == "remove"]
    if removes:
        follow = list(pair_follow(removes, "change_own", "change_other").values())
        out["remove_pair_follow"] = round(float(np.mean(follow)), 3) if follow else None
    return out


def select(items, limit=0):
    """Indices to run: everything, or for a smoke test the first ``limit`` items of each set (both tokens of the
    first ``limit // 2`` pairs)."""
    if not limit:
        return list(range(len(items)))
    seen, kept = {}, []
    for index, item in enumerate(items):
        if item["set"] in ("pair", "remove"):
            if item["pair"] < max(1, limit // 2):
                kept.append(index)
        else:
            seen[item["set"]] = seen.get(item["set"], 0) + 1
            if seen[item["set"]] <= limit:
                kept.append(index)
    return kept


def parser():
    from samtok_edit21.models.pipeline import DEFAULT_QWEN, DEFAULT_SAMTOK
    p = argparse.ArgumentParser(prog="samtok-edit round2-eval", description=__doc__.split("\n\n")[0])
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--adapter", help="Adapter directory (round 1 or 2); omit for the stock DiT")
    p.add_argument("--items", default="eval_c1.json")
    p.add_argument("--qwen", default=DEFAULT_QWEN)
    p.add_argument("--samtok", default=DEFAULT_SAMTOK)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--limit", type=int, default=0, help="Smoke test: at most this many items per set")
    return p


def main(argv=None):
    import torch.distributed as dist
    from samtok_edit21.models.codec import load_codec
    from samtok_edit21.models.pipeline import load_pipeline
    from samtok_edit21.training.grounding import load_grounding_adapter
    from samtok_edit21.training.round2 import Context

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
    ctx = Context(args.data)
    items = json.load(open(Path(args.data) / args.items))
    jobs = select(items, args.limit)
    mine = jobs[rank::world]
    records, started = [], time.time()
    for done, index in enumerate(mine):
        item = items[index]
        try:
            record, image, _ = run_item(pipe, codec, item, ctx, steps=args.steps)
            record["idx"] = index
            records.append(record)
            folder = out / "images" / item["set"]
            folder.mkdir(parents=True, exist_ok=True)
            thumbnail(image).save(folder / f"{index:05d}.jpg", quality=88)
        except Exception as error:
            print(f"[rank {rank}] eval skip {item['set']} {index}: {error!r}"[:400], flush=True)
        if rank == 0 and done % 20 == 0:
            print(json.dumps({"done": done, "of": len(mine), "sec": round(time.time() - started)}), flush=True)
    gathered = [None] * world
    dist.all_gather_object(gathered, records)
    if rank == 0:
        merged = sorted((r for part in gathered for r in part), key=lambda r: r["idx"])
        summary = summarize(merged)
        json.dump({"adapter": args.adapter, "steps": args.steps, "jobs": len(jobs), "summary": summary, "records": merged},
                  open(out / "eval.json", "w"))
        (out / "EVAL_DONE").write_text(json.dumps(summary))
        print(json.dumps(summary), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
