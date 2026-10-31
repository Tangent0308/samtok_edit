"""Descriptive and paired source-image bootstrap statistics, no oracle rematching."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def describe(rows):
    targets = [t for r in rows for t in r["targets"]]
    unknown = [t for t in targets if t["locator"] != "mask"]
    known = [t for t in targets if t["locator"] == "mask"]

    def mean(ts, key):
        return float(np.mean([t[key] for t in ts])) if ts else None

    def group(ts):
        return {
            "n": len(ts),
            "iou": mean(ts, "iou"),
            "strict_iou": mean(ts, "strict_iou"),
            "compatible_iou": mean(ts, "compatible_iou"),
            "iou50": float(np.mean([t["strict_iou"] >= 0.5 for t in ts]))
            if ts
            else None,
            "compatible_iou50": float(np.mean([t["compatible_iou"] >= 0.5 for t in ts]))
            if ts
            else None,
            "canonical_iou": mean(ts, "canonical_iou"),
            "exact_code": mean(ts, "exact_code"),
            "possible_id_swap": mean(ts, "possible_id_swap"),
        }

    return {
        "requests": len(rows),
        "targets": len(targets),
        "strict_rate": mean(rows, "strict"),
        "compatible_rate": mean(rows, "compatible"),
        "all_targets_iou50": mean(rows, "all_iou50"),
        "compatible_all_targets_iou50": mean(rows, "compatible_all_iou50"),
        "unknown": group(unknown),
        "known": group(known),
        "all": group(targets),
        "multi_unknown": group([t for t in unknown if t["K"] >= 2]),
        "single_unknown": group([t for t in unknown if t["K"] == 1]),
        "by_mode": {
            k: group([t for t in targets if t["mode"] == k])
            for k in sorted({t["mode"] for t in targets})
        },
        "by_locator": {
            k: group([t for t in targets if t["locator"] == k])
            for k in sorted({t["locator"] for t in targets})
        },
        "by_K_unknown": {
            str(k): group([t for t in unknown if t["K"] == k])
            for k in sorted({t["K"] for t in targets})
        },
        "by_operation_unknown": {
            k: group([t for t in unknown if t["operation"] == k])
            for k in sorted({t["operation"] for t in targets})
        },
        "by_area_unknown": {
            k: group([t for t in unknown if lo <= t["area"] < hi])
            for k, lo, hi in [
                ("tiny", 0, 0.01),
                ("medium", 0.01, 0.05),
                ("large", 0.05, 1),
            ]
        },
        "point_hit": mean([t for t in targets if t["locator"] == "point"], "point_hit"),
        "minimum_canonical_cached_agreement": min(
            t["canonical_cached_agreement"] for t in targets
        ),
    }


def paired(base, adapted, seed=2026103002, metric="compatible_iou"):
    if {r["id"] for r in base} != {r["id"] for r in adapted}:
        raise ValueError("Mismatched source requests")
    b = {r["id"]: r for r in base}
    a = {r["id"]: r for r in adapted}
    rng = np.random.default_rng(seed)
    output = {}
    for loc in ["unknown", "point", "box", "none", "mask"]:
        groups = []
        for uid in sorted(b):
            bb = [
                t
                for t in b[uid]["targets"]
                if (t["locator"] != "mask" if loc == "unknown" else t["locator"] == loc)
            ]
            aa = [
                t
                for t in a[uid]["targets"]
                if (t["locator"] != "mask" if loc == "unknown" else t["locator"] == loc)
            ]
            if bb:
                if [t["label"] for t in bb] != [t["label"] for t in aa]:
                    raise ValueError("Mismatched target IDs")
                groups.append(([t[metric] for t in bb], [t[metric] for t in aa]))
        if not groups:
            continue
        sums = np.array([[np.sum(x), np.sum(y), len(x)] for x, y in groups])
        d = sums.sum(0)
        draws = rng.integers(0, len(groups), size=(5000, len(groups)))
        samples = sums[draws].sum(1)
        deltas = (samples[:, 1] - samples[:, 0]) / samples[:, 2]
        output[loc] = {
            "source_images": len(groups),
            "metric": metric,
            "target_delta": float((d[1] - d[0]) / d[2]),
            "paired_source_bootstrap_95ci": np.quantile(
                deltas, [0.025, 0.975]
            ).tolist(),
        }
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--baseline")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    rows = [json.loads(x) for x in Path(args.scores).read_text().splitlines()]
    out = describe(rows)
    if args.baseline:
        base = [json.loads(x) for x in Path(args.baseline).read_text().splitlines()]
        out["paired"] = paired(base, rows)
        out["paired_strict"] = paired(base, rows, metric="strict_iou")
        out["paired_multi"] = paired(
            [r for r in base if len(r["targets"]) >= 2],
            [r for r in rows if len(r["targets"]) >= 2],
        )
    Path(args.output).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
