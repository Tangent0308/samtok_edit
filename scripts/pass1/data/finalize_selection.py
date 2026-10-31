"""Create a new audited selection version; never mutate the input selection."""

import argparse
import collections
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from samtok_edit21.pass1.data.selection import (
    canonicalize_teachers,
    describe,
    rebalance_modes,
    region_key,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    src = args.selection
    out = args.output
    if out.exists():
        raise FileExistsError(out)
    out.mkdir()
    config = json.loads((args.config).read_text())
    manifest = json.loads((src / "manifest.json").read_text())
    audit = json.loads((args.audit / "source_audit.json").read_text())
    bad = set(audit["quarantine_sources"])
    region_table = pq.read_table(src / "selected_regions.parquet")
    region_map = {r["key"]: r for r in region_table.to_pylist()}
    source_table = pq.read_table(src / "selected_sources.parquet")
    sources = {r["source"]: r for r in source_table.to_pylist()}
    hashes = {
        r["path"]: r
        for r in pq.read_table(args.audit / "image_hashes.parquet").to_pylist()
    }
    if any(s["image"] not in hashes for s in sources.values()):
        raise ValueError("Unhashed selected source")
    all_rows = {}
    removed = {}
    stats = {}
    file_hashes = {}
    for name in ["warmup", "main", "dev", "test"]:
        rows = [
            json.loads(x)
            for x in (src / (name + ".selection.jsonl")).read_text().splitlines()
        ]
        removed[name] = [r["id"] for r in rows if r["source"] in bad]
        rows = [r for r in rows if r["source"] not in bad]
        if name in ["dev", "test"]:
            rebalance_modes(rows, region_map, config, "eval")
        for row in rows:
            for t in row["targets"]:
                if t["mode"].endswith("_box"):
                    t["locator_policy"] = "tight"
        all_rows[name] = rows
        stats[name] = describe(rows, region_map, sources)
        path = out / (name + ".selection.jsonl")
        with path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r, separators=(",", ":")) + "\n")
        file_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    keys = {
        t["region_key"]
        for rows in all_rows.values()
        for r in rows
        for t in r["targets"]
    }
    selected = [region_map[k] for k in sorted(keys)]
    teacher_conflicts = canonicalize_teachers(selected)
    region_schema = (
        region_table.schema
        if "canonical_label_key" in region_table.schema.names
        else region_table.schema.append(pa.field("canonical_label_key", pa.string()))
    )
    pq.write_table(
        pa.Table.from_pylist(selected, schema=region_schema),
        out / "selected_regions.parquet",
        compression="zstd",
    )
    (out / "canonical_teacher_resolution.json").write_text(
        json.dumps(teacher_conflicts, ensure_ascii=False, indent=2) + "\n"
    )
    source_keys = {r["source"] for rows in all_rows.values() for r in rows}
    pq.write_table(
        pa.Table.from_pylist(
            [sources[k] for k in sorted(source_keys)], schema=source_table.schema
        ),
        out / "selected_sources.parquet",
        compression="zstd",
    )
    reserve = pq.read_table(src / "small_lowq_reserve.parquet")
    reserve_rows = [r for r in reserve.to_pylist() if r["source"] not in bad]
    pq.write_table(
        pa.Table.from_pylist(reserve_rows, schema=reserve.schema),
        out / "small_lowq_reserve.parquet",
        compression="zstd",
    )
    uses = collections.Counter(
        r["source"] for name in ["warmup", "main"] for r in all_rows[name]
    )
    regions = collections.Counter(
        region_key(region_map[t["region_key"]])
        for name in ["warmup", "main"]
        for r in all_rows[name]
        for t in r["targets"]
    )
    pairs = collections.Counter(
        r["pair_id"] for r in all_rows["main"] if r.get("pair_id")
    )
    if any(n != 2 for n in pairs.values()):
        raise ValueError("Incomplete pair after quarantine")
    split_sets = {name: {r["source"] for r in rows} for name, rows in all_rows.items()}
    trainset = split_sets["warmup"] | split_sets["main"]
    assert not (trainset & (split_sets["dev"] | split_sets["test"])) and not (
        split_sets["dev"] & split_sets["test"]
    )
    remaining_issues = []
    for issue in audit["issues"]:
        if issue.get("kind") == "heldout_vs_train":
            survives = (
                issue["query"] in source_keys and issue["candidate"] in source_keys
            )
        else:
            survives = issue.get("query", issue.get("source")) in source_keys
        if survives:
            remaining_issues.append(issue)
    if remaining_issues:
        raise ValueError("Unresolved source audit issue")
    manifest.update(
        {
            "status": "source_audited_selection_pending_full_geometry_materialization",
            "config": config,
            "summary": stats,
            "selected_unique_regions": len({region_key(r) for r in selected}),
            "selected_annotation_rows": len(selected),
            "train_sources": len(trainset),
            "max_train_requests_per_source": max(uses.values()),
            "max_train_region_exposures": max(regions.values()),
            "reserved_lowq_rows": len(reserve_rows),
            "reserved_lowq_sources": len({r["source"] for r in reserve_rows}),
            "selection_sha256": file_hashes,
            "quarantined_sources": sorted(bad),
            "removed_request_counts": {k: len(v) for k, v in removed.items()},
            "remaining_audit_flags": 0,
            "audit_hashed_image_paths": audit["hashed_paths"],
            "pool_aliases_checked": audit["pool_aliases_checked"],
            "audit_path": str(args.audit / "source_audit.json"),
            "pending_checks": [
                "Full point/box geometry materialization and resulting public-input conflict audit; bounded preflight is separate",
                "Full training JSONL context-length check with native processor",
            ],
            "reproduction": "Run formal_select with the frozen config, then apply the frozen source quarantine and rebalance eval modes; audit outputs are retained.",
            "no_model_training_or_inference_performed": True,
        }
    )
    manifest["counterfactual"].update(
        {
            "selected_pairs": len(pairs),
            "rows": 2 * len(pairs),
            "shortfall_pairs": manifest["counterfactual"]["requested_pairs"]
            - len(pairs),
        }
    )
    manifest["counterfactual"]["modes"] = dict(
        collections.Counter(
            next(t["mode"] for t in r["targets"] if t["label"] == r["contrast_label"])
            for r in all_rows["main"]
            if r.get("pair_role") == "a"
        )
    )
    d1_groups = {
        r["group"]
        for r in pq.read_table(
            args.inventory / "merged_regions.parquet",
            columns=["group", "family", "valid", "roundtrip", "area"],
        ).to_pylist()
        if r["family"] == "D1"
        and r["valid"]
        and r["roundtrip"] >= 0.7
        and 0.00003 <= r["area"] <= 0.3
    }
    manifest["d1_source_requests"] = sum(n for s, n in uses.items() if s in d1_groups)
    manifest["d1_source_requests_definition"] = (
        "Training requests whose source has eligible core D1 annotations in the inventory; includes source aliases to another family."
    )
    manifest["base_pool_source_requests"] = sum(
        n for s, n in uses.items() if sources[s].get("pool_origin") == "base"
    )
    manifest["reserve_status"] = (
        "Unselected optional candidate bank; not approved for training and not covered by the selected-source audit in full."
    )
    manifest["canonical_teacher_groups_resolved"] = len(teacher_conflicts)
    manifest["teacher_resolution_policy"] = (
        "For identical source and S digest with different cached spans, use the existing teacher with highest roundtrip; preserve original instruction and raw cache provenance."
    )
    real_targets = [
        region_map[t["region_key"]]
        for r in all_rows["main"]
        for t in r["targets"]
        if t["region_key"].startswith("M:")
    ]
    payload = collections.Counter(
        (r.get("payload") or "").strip().lower()
        for r in real_targets
        if r.get("payload")
    )
    instructions = collections.Counter(
        r.get("noref_instruction") for r in real_targets if r.get("noref_instruction")
    )
    manifest["main_repetition_audit"] = {
        "real_edit_targets": len(real_targets),
        "payload_top20": payload.most_common(20),
        "noref_instruction_top20": instructions.most_common(20),
        "payload_policy": "Report repetition, enforce source<=8 and region<=8. DiT payload<=50 is not inherited as a Pass1 hard cap.",
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    (out / "selection_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n"
    )
    (out / "removed_requests.json").write_text(json.dumps(removed, indent=2) + "\n")
    print(
        json.dumps(
            {
                "summary": stats,
                "counterfactual": manifest["counterfactual"],
                "removed": manifest["removed_request_counts"],
                "train_sources": len(trainset),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
