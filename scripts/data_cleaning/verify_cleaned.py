#!/usr/bin/env python3
"""Verify a cleaned pair dataset written by clean_pairs.py.

Every kept pair: both image files exist, have the manifest size (so target and source share one frame), unchanged
images are hard links and repaired ones PNG; every instance mask decodes to the image size, is non-empty and its
box is the outward-rounded 0-1000 box of the mask; the union mask is the union of the instances; the recorded
statistics meet the keeping standard.  A random sample is re-measured from the written files.

usage: verify_cleaned.py --out DIR [--sample N] [--workers N]
"""
import argparse
import json
import os
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clean_pairs as cp  # noqa: E402

OUT = None


def decode(item):
    return mask_utils.decode({"size": item["rle_size"], "counts": item["rle_counts"].encode()}).astype(bool)


def check(line):
    row = json.loads(line)
    problems = []
    width, height = row["size"]
    stat = {}
    for key in ("source_image", "target_image"):
        path = f"{OUT}/{row[key]}"
        try:
            with Image.open(path) as im:
                if im.size != (width, height):
                    problems.append(f"{key} size {im.size} != {(width, height)}")
                stat[key] = (im.format, os.stat(path).st_nlink)
        except OSError as error:
            problems.append(f"{key} unreadable: {error}")
    if "source_image" in stat and stat["source_image"][1] < 2:
        problems.append("source is not a hard link")
    if "target_image" in stat:
        linked = row["treatment"] == "link"
        if linked and stat["target_image"][1] < 2:
            problems.append("unchanged target is not a hard link")
        if not linked and stat["target_image"][0] != "PNG":
            problems.append("processed target is not PNG")
    union = np.zeros((height, width), bool)
    for inst in row["instances"]:
        mask = decode(inst)
        if mask.shape != (height, width) or not mask.any():
            problems.append(f"instance {inst.get('instance_id')}: bad mask")
            continue
        if cp.pixel_box(mask) != inst["box_1000"]:
            problems.append(f"instance {inst.get('instance_id')}: box does not match its mask")
        if inst["moved"] and not (inst.get("mapped_from_target") and row["decision"] == "fix_register"):
            problems.append(f"instance {inst.get('instance_id')}: moved without registration of a target-grounded instance")
        union |= mask
    region = decode(row["region"])
    if region.shape != union.shape or (region != union).any() or cp.pixel_box(region) != row["region"]["box_1000"]:
        problems.append("region is not the union of the instances")
    after = row["metrics"]["after"]
    if after["far_out"] > cp.GATE["far_out_final"] or after["fill"] < cp.GATE["fill_final"]:
        problems.append(f"keeping standard not met: {after}")
    return row["id"], row["dataset"], row["type"], row["decision"], row["treatment"], sum(i["moved"] for i in row["instances"]), problems


def remeasure(line):
    """far_out / fill recomputed from the written files against the recorded values."""
    row = json.loads(line)
    src = np.asarray(Image.open(f"{OUT}/{row['source_image']}").convert("RGB"))
    tgt = np.asarray(Image.open(f"{OUT}/{row['target_image']}").convert("RGB"))
    got = cp.stats(src, tgt, decode(row["region"]))
    want = row["metrics"]["after"]
    return row["id"], row["treatment"], abs(got["far_out"] - want["far_out"]), abs(got["fill"] - want["fill"])


def _init(out):
    global OUT
    OUT = out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sample", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()
    out = Path(args.out)
    lines = (out / "manifest.jsonl").read_text().splitlines()
    decisions = Counter()
    with (out / "decisions.jsonl").open() as stream:
        ids = []
        for line in stream:
            res = json.loads(line)
            ids.append(res["id"])
            decisions[res["decision"]] += 1
    kept = sum(decisions[d] for d in cp.KEPT)
    report = {"pairs": len(ids), "duplicate_ids": len(ids) - len(set(ids)), "decisions": dict(decisions),
              "manifest_rows": len(lines), "manifest_matches_kept": len(lines) == kept}
    bad, table, moved_pairs, moved_instances = [], Counter(), 0, 0
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(str(out),)) as pool:
        for rid, dataset, kind, decision, treatment, moved, problems in pool.map(check, lines, chunksize=64):
            table[treatment] += 1
            moved_pairs += moved > 0
            moved_instances += moved
            if problems:
                bad.append({"id": rid, "problems": problems})
        sample = random.Random(0).sample(lines, min(args.sample, len(lines)))
        worst = Counter()
        drift = []
        for rid, treatment, d_far, d_fill in pool.map(remeasure, sample, chunksize=16):
            worst[treatment] = max(worst[treatment], d_far, d_fill)
            if max(d_far, d_fill) > 1e-6:
                drift.append({"id": rid, "treatment": treatment, "far_out_diff": d_far, "fill_diff": d_fill})
    report.update(treatments=dict(table), pairs_with_moved_instances=moved_pairs, moved_instances=moved_instances,
                  pairs_with_problems=len(bad), problems=bad[:50], remeasured=len(sample),
                  remeasured_max_abs_diff={k: float(v) for k, v in worst.items()}, remeasured_differing=len(drift),
                  remeasured_examples=drift[:20])
    (out / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("problems", "remeasured_examples")}, ensure_ascii=False, indent=1))
    print("OK" if not bad and report["manifest_matches_kept"] and not report["duplicate_ids"] and not drift else "PROBLEMS FOUND")


if __name__ == "__main__":
    main()
