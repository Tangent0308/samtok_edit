"""Judge manifest for v2 benchmark outputs (the unchanged pair_v2 judge).

Rows follow the judge's own manifest contract (same fields and input digest
as its prepare step). Stock outputs of the unchanged single-region cases are
linked from the existing qwen21_656 run; split MIRAGE cases need the v2
``--stock`` run. The v2 dev/test split is kept in ``v2_split``; the judge's
own ``split`` stays ``unassigned``, so launch it with ``--split all``.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

from samtok_edit21.data.io import file_hash

JUDGE_CODE = ("/mnt/bn/strategy-mllm-train/intern/users/tanyue/experiments/SAMTokEdit/"
              "qwen21_stage2_benchmark_noref_aligned_20261004/judge/code")
STOCK_656 = "/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/qwen21_656/inference/qwen21"
STOCK_SETTINGS = {"text": "text_only", "mask": "mask_annotation", "box": "box_annotation", "point": "point_annotation"}


def judge_row(case, method, setting, output, digest, hashes):
    def sha(path):
        if path not in hashes:
            hashes[path] = file_hash(path)
        return hashes[path]

    with Image.open(case["source_image"]) as image:
        size = list(image.size)
    region = case["region"]
    exists = Path(output).is_file()
    row = {"sample_id": f"{case['case_id']}/{method}/{setting}", "case_id": case["case_id"],
           "eval_index": case["eval_index"], "method": method, "setting": setting,
           "source_dataset": case["source_dataset"], "edit_type": case["benchmark_type"],
           "source_image": case["source_image"], "source_sha256": sha(case["source_image"]), "source_size": size,
           "output_image": str(output), "output_sha256": sha(str(output)) if exists else None,
           "instruction": case["instruction"], "region_instruction": case["region_instruction"],
           "regions": [{"box": region["box"], "point": region["point"], "mask": region["mask"],
                        "mask_sha256": sha(region["mask"])}],
           "annotation_status": "imported_not_human_calibrated", "protocol": "samtok_v2_atomic",
           "prepared_sha256": "", "cohort": "v2", "split": "unassigned", "v2_split": case["split"],
           "control": None, "expected": {}, "label_provenance": "unlabeled",
           "delivery_status": "available" if exists else "missing_output"}
    row["input_digest"] = digest(row)
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cases", required=True)
    parser.add_argument("--run", action="append", default=[], metavar="METHOD=DIR",
                        help="A v2 run directory (from evaluation.run) under a method name; repeatable")
    parser.add_argument("--stock", metavar="DIR", help="v2 --stock run for split MIRAGE cases; adds method 'stock'")
    parser.add_argument("--stock-656", default=STOCK_656)
    parser.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    parser.add_argument("--judge-code", default=JUDGE_CODE)
    parser.add_argument("--output", required=True)
    parser.add_argument("--compiled", help="compiled.jsonl; with it, text_plain is expected only for add (D9)")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    sys.path.insert(0, args.judge_code)
    from evaluation.metrics.protocol import digest

    cases = [json.loads(line) for line in Path(args.cases).read_text().splitlines() if line.strip()]
    cases = [c for c in cases if args.split == "all" or c["split"] == args.split]
    kinds = {}
    if args.compiled:
        kinds = {e["case_id"]: e["region_kind"] for e in map(json.loads, Path(args.compiled).read_text().splitlines()) if e}
    rows, hashes = [], {}
    for spec in args.run:
        method, _, directory = spec.partition("=")
        directory = Path(directory)
        for setting_dir in sorted(p for p in directory.iterdir() if p.is_dir()):
            for case in cases:
                if setting_dir.name == "text_plain" and kinds.get(case["case_id"], "box") != "box":
                    continue
                output = setting_dir / f"{case['case_id']}.png"
                if output.is_file() or args.require_complete:
                    rows.append(judge_row(case, method, setting_dir.name, output, digest, hashes))
    if args.stock:
        for setting, old in STOCK_SETTINGS.items():
            for case in cases:
                if case["part"] is None:  # unchanged single-region row: reuse the 656-case output
                    output = Path(args.stock_656) / old / f"{case['eval_index']:04d}.png"
                else:
                    output = Path(args.stock) / setting / f"{case['case_id']}.png"
                rows.append(judge_row(case, "stock", setting, output, digest, hashes))
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate sample ids")
    missing = [row["sample_id"] for row in rows if row["delivery_status"] != "available"]
    if args.require_complete and missing:
        raise SystemExit(f"{len(missing)} outputs missing, e.g. {missing[:5]}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
    print(json.dumps({"rows": len(rows), "missing": len(missing),
                      "by_method_setting": dict(Counter(f"{r['method']}/{r['setting']}" for r in rows))}))


if __name__ == "__main__":
    main()
