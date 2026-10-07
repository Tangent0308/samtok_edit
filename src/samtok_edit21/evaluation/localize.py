"""Pass-1 localization benchmark (E1): Stage 1 region tokens vs the benchmark region.

``run`` (torchrun, one process per GPU) asks the TE for each case's region
exactly as the text setting's pass 1 does (``models.pipeline.localize`` on
the same canvas), decodes every region with the decoder that training and
inference use (``binding.unit_masks``: codec for mask spans, outward raster
for boxes) and writes one record per case; finished cases are skipped.

``summarize`` reports, per split and compiled edit type: parse rate (pass 1
bound a region), format rate (the expected region kind: a box for add, a mask
span otherwise), mean mask IoU, mean box IoU and box Acc@0.5 against the
benchmark region.  A failed parse scores IoU 0.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from samtok_edit21.data.io import write_json
from samtok_edit21.data.protocol import REGION_RE, region_kind
from samtok_edit21.evaluation.run import official_size

TYPES = ("add", "remove", "replace", "attribute", "text")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def pixel_bbox(mask):
    ys, xs = np.nonzero(mask)
    return (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1) if len(xs) else None


def box_iou(a, b):
    if a is None or b is None:
        return 0.0
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def score_case(case, entry, result, masks):
    """Record for one case from the localization result and decoded unit masks."""
    gt = np.asarray(Image.open(case["region"]["mask"]).convert("L")) > 0
    expected = entry["region_kind"]
    parsed = result["actual_variant"] != "plain"
    regions = [m.group() for m in REGION_RE.finditer(result["conditioning_prompt"])] if parsed else []
    kinds = sorted({region_kind(r) for r in regions})
    pred = np.zeros_like(gt)
    for mask in masks:
        pred |= mask
    union = np.logical_or(pred, gt).sum()
    x1, y1, x2, y2 = (float(v) for v in case["region"]["box"])
    return {"case_id": case["case_id"], "split": case["split"], "source_dataset": case["source_dataset"],
            "benchmark_type": case["benchmark_type"], "edit_type": entry["edit_type"], "expected_kind": expected,
            "parsed": parsed, "fallback_reason": result.get("fallback_reason"), "raw": result["raw"],
            "units": len(masks), "regions": len(regions), "kinds": kinds,
            "format_ok": parsed and kinds == [expected],
            "mask_iou": float(np.logical_and(pred, gt).sum() / union) if parsed and union else 0.0,
            "box_iou": box_iou(pixel_bbox(pred), (x1, y1, x2, y2)) if parsed else 0.0,
            "pred_area": float(pred.mean()), "gt_area": float(gt.mean())}


def run(args):
    import torch
    from samtok_edit21.models.binding import unit_masks
    from samtok_edit21.models.codec import SamtokCodec
    from samtok_edit21.models.pipeline import load_pipeline, localize
    from samtok_edit21.training.objectives import load_adapter

    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local}"
    torch.cuda.set_device(local)
    cases = [c for c in read_jsonl(args.cases) if args.split == "all" or c["split"] == args.split]
    cases = cases[:args.limit] if args.limit else cases
    compiled = {e["case_id"]: e for e in read_jsonl(args.compiled)}
    records = Path(args.output) / "records"
    records.mkdir(parents=True, exist_ok=True)
    jobs = [c for c in cases[rank::world] if not (records / f"{c['case_id']}.json").is_file()]
    pipe = load_pipeline(args.qwen, args.samtok, device=device, components=("text_encoder",))
    if args.te_adapter:
        load_adapter(pipe.text_encoder, args.te_adapter)
    pipe.eval()
    codec = SamtokCodec(Path(args.samtok) / "sam2.1_hiera_large.pt",
                        Path(args.samtok) / "mask_tokenizer_256x2.pth", device=device)
    if rank == 0:
        write_json(Path(args.output) / "run.json", {"args": vars(args), "world_size": world, "cases": len(cases)})
    for done, case in enumerate(jobs, 1):
        started = time.perf_counter()
        source = Image.open(case["source_image"]).convert("RGB")
        width, height = official_size(source)
        with torch.no_grad():
            result = localize(pipe, case["instruction"], [source.convert("RGBA")], height=height, width=width,
                              max_new_tokens=args.max_new_tokens, variant="ref")
        masks = []
        if result["actual_variant"] != "plain":
            try:
                masks = unit_masks(result["conditioning_prompt"], source, codec)[1]
            except ValueError as exc:  # undecodable spans count as a failed parse
                result.update(actual_variant="plain", fallback_reason=f"decode: {exc}")
        record = score_case(case, compiled[case["case_id"]], result, masks)
        record["seconds"] = round(time.perf_counter() - started, 2)
        write_json(records / f"{case['case_id']}.json", record)
        print(json.dumps({"rank": rank, "done": done, "of": len(jobs), "case": case["case_id"],
                          "parsed": record["parsed"], "mask_iou": round(record["mask_iou"], 3)}), flush=True)


def summary_rows(records):
    groups = defaultdict(list)
    for r in records:
        for split in (r["split"], "all"):
            for typ in (r["edit_type"], "all"):
                groups[(split, typ)].append(r)
    rows = {}
    for (split, typ), group in groups.items():
        n = len(group)
        parsed = [r for r in group if r["parsed"]]
        rows[f"{split}/{typ}"] = {
            "n": n, "parse_rate": len(parsed) / n, "format_rate": sum(r["format_ok"] for r in group) / n,
            "mask_iou": sum(r["mask_iou"] for r in group) / n,
            "mask_iou_parsed": sum(r["mask_iou"] for r in parsed) / len(parsed) if parsed else None,
            "box_iou": sum(r["box_iou"] for r in group) / n,
            "acc50": sum(r["box_iou"] >= 0.5 for r in group) / n,
            "single_unit_rate": sum(r["units"] == 1 for r in group) / n}
    return rows


def summarize(args):
    runs = dict(item.split("=", 1) for item in args.runs)
    cases = {c["case_id"]: c for c in read_jsonl(args.cases)}
    report = {}
    for label, directory in runs.items():
        records = [json.loads(p.read_text()) for p in sorted((Path(directory) / "records").glob("*.json"))]
        missing = sorted(set(cases) - {r["case_id"] for r in records})
        report[label] = {"records": len(records), "missing": len(missing), "rows": summary_rows(records)}
    write_json(args.output, report)
    keys = [f"{s}/{t}" for s in ("dev", "all") for t in (*TYPES, "all")]
    print("| run | split/type | n | parse | format | mask IoU | box IoU | Acc@0.5 |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for label, item in report.items():
        for key in keys:
            row = item["rows"].get(key)
            if row:
                print(f"| {label} | {key} | {row['n']} | {row['parse_rate']:.2f} | {row['format_rate']:.2f} | "
                      f"{row['mask_iou']:.3f} | {row['box_iou']:.3f} | {row['acc50']:.2f} |")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--cases", required=True)
    run_parser.add_argument("--compiled", required=True)
    run_parser.add_argument("--output", required=True)
    run_parser.add_argument("--te-adapter", help="Stage 1 localization adapter; omit for the raw SAMTok TE")
    run_parser.add_argument("--split", choices=("dev", "test", "all"), default="all")
    run_parser.add_argument("--limit", type=int)
    run_parser.add_argument("--max-new-tokens", type=int, default=256)
    run_parser.add_argument("--qwen", default="/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen-Image-2.1")
    run_parser.add_argument("--samtok", default="/mnt/bn/strategy-mllm-train/user/tanyue/models/SAMTok/Qwen3-VL-8B-SAMTok")
    summary = sub.add_parser("summarize")
    summary.add_argument("--cases", required=True)
    summary.add_argument("--runs", nargs="+", required=True, help="label=<run output dir>")
    summary.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    run(args) if args.command == "run" else summarize(args)


if __name__ == "__main__":
    main()
