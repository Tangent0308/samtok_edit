"""v2 benchmark protocol (plan section 8.7): atomic cases, settings and regions.

The benchmark's single-region rows are used as they are; MIRAGE two-region
rows are split into one atomic edit per region (D4). Every case keeps one
region; the other region of a split MIRAGE case is only a preservation check.
Cases are split 30% dev / 70% test by source image, so tuning never sees a
test image.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

BENCHMARK_ROOT = "/mnt/bn/strategy-mllm-train/user/tanyue/datasets/samtok_edit_benchmark"
BENCHMARK_REPO = "/opt/tiger/tanyue/samtok_edit_benchmark"
MIRAGE_SELECTION = BENCHMARK_REPO + "/selection/mirage_selected_regions.jsonl"
SETTINGS = ("mask", "box", "point", "text", "text_plain")
DEV_FRACTION = 0.3


def split_of(source_image, dev_fraction=DEV_FRACTION):
    """Deterministic dev/test assignment by source image (never by case)."""
    value = int(hashlib.sha256(source_image.encode()).hexdigest()[:8], 16) / 16**8
    return "dev" if value < dev_fraction else "test"


def _sentence(text):
    text = text.strip()
    text = text[0].upper() + text[1:]
    return text if text.endswith((".", "!", "?")) else text + "."


def mirage_atomic(row, selection):
    """One atomic case per region of a two-region MIRAGE row.

    The with-location instruction is split on ", and " (clause k edits region
    k, the order MIRAGE writes both instructions in); the region-only
    instruction on its "For {region_k}," clauses.
    """
    clauses = re.split(r",\s+and\s+", row["instruction"]["with_location_reference"].strip().rstrip("."))
    region_only = re.findall(r"For \{region_(\d+)\}, (.+?)(?=\s*For \{region_\d+\},|$)",
                             row["instruction"]["region_only"])
    edits = selection["selected_atomic_edits"]
    if not len(clauses) == len(region_only) == len(edits) == len(row["regions"]):
        raise ValueError(f"{row['id']}: cannot split into one clause per region")
    cases = []
    for k, (clause, (index, text), edit) in enumerate(zip(clauses, region_only, edits), 1):
        if int(index) != k or edit["region"] != k:
            raise ValueError(f"{row['id']}: region order differs between instructions")
        refer = edit["refer_object"].replace('\\"', '"').lower()
        cases.append({
            "part": k, "benchmark_type": edit["operation"],
            "instruction": _sentence(clause),
            "region_instruction": "For {region_1}, " + text.strip(),
            "region": row["regions"][k - 1],
            "referent_in_clause": refer in clause.lower() or refer.removeprefix("the ") in clause.lower(),
        })
    return cases


def build_cases(benchmark_root=BENCHMARK_ROOT, mirage_selection=MIRAGE_SELECTION, dev_fraction=DEV_FRACTION):
    root = Path(benchmark_root)
    rows = [json.loads(line) for line in (root / "benchmark.jsonl").read_text().splitlines() if line.strip()]
    selections = {json.loads(line)["candidate_id"]: json.loads(line)
                  for line in Path(mirage_selection).read_text().splitlines() if line.strip()}
    cases = []
    for index, row in enumerate(rows):
        common = {"eval_index": index, "benchmark_id": row["id"], "source_dataset": row["source_dataset"],
                  "source_image": str(root / row["source_image"]),
                  "split": split_of(row["source_image"], dev_fraction)}
        if len(row["regions"]) == 1:
            parts = [{"part": None, "benchmark_type": row["edit_type"],
                      "instruction": row["instruction"]["with_location_reference"],
                      "region_instruction": row["instruction"]["region_only"],
                      "region": row["regions"][0], "referent_in_clause": True}]
        elif row["source_dataset"] == "mirage":
            parts = mirage_atomic(row, selections[row["id"]])
        else:
            continue  # Two-region CompBench rows are out of scope (no atomic split).
        for part in parts:
            region = part.pop("region")
            case_id = row["id"] if part["part"] is None else f"{row['id']}#r{part['part']}"
            cases.append({**common, **part, "case_id": case_id,
                          "region": {"mask": str(root / region["mask"]), "box": region["box"],
                                     "point": region["point"]}})
    if len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("Duplicate case ids")
    return cases


def main(argv=None):
    import argparse
    from collections import Counter
    from samtok_edit21.data.io import write_json

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", required=True, help="cases.jsonl")
    parser.add_argument("--benchmark-root", default=BENCHMARK_ROOT)
    parser.add_argument("--mirage-selection", default=MIRAGE_SELECTION)
    args = parser.parse_args(argv)
    cases = build_cases(args.benchmark_root, args.mirage_selection)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases))
    summary = {"cases": len(cases), "by_dataset_type": dict(Counter(
        f"{c['source_dataset']}:{c['benchmark_type']}" for c in cases)),
        "by_split": dict(Counter(c["split"] for c in cases)),
        "sources_by_split": dict(Counter(split for _, split in {(c["source_image"], c["split"]) for c in cases})),
        "mirage_referent_mismatch": [c["case_id"] for c in cases if not c["referent_in_clause"]]}
    write_json(output.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
