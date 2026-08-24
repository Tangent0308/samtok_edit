#!/usr/bin/env python3
"""Remove validation-content duplicates and rebalance Stage-2 source pools."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve_image(base: Path, path: str) -> Path:
    image_path = Path(path)
    return image_path if image_path.is_absolute() else base / image_path


def sha256_file(path: Path) -> tuple[Path, str]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return path, digest.hexdigest()


def stat_file(path: Path) -> tuple[Path, int]:
    return path, path.stat().st_size


def source_identity(row: dict) -> tuple[str, int]:
    provenance = row.get("provenance") or {}
    source_parquet = provenance.get("source_parquet")
    row_idx = provenance.get("row_idx")
    if source_parquet is None or row_idx is None:
        raise ValueError(f"Row lacks source provenance: {row}")
    return Path(str(source_parquet)).stem.replace(" ", "_") + ".parquet", int(
        row_idx
    )


def write_jsonl_atomic(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def write_json_atomic(payload: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edit_mt_jsonl", type=Path, required=True)
    parser.add_argument("--edit_jsonl", type=Path, required=True)
    parser.add_argument("--training_base", type=Path, required=True)
    parser.add_argument("--validation_jsonl", type=Path, required=True)
    parser.add_argument("--validation_base", type=Path, required=True)
    parser.add_argument("--output_edit_mt_jsonl", type=Path, required=True)
    parser.add_argument("--output_edit_jsonl", type=Path, required=True)
    parser.add_argument("--report_json", type=Path, required=True)
    parser.add_argument("--target_edit_rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--io_workers", type=int, default=64)
    args = parser.parse_args()
    if args.io_workers < 1:
        raise ValueError("io_workers must be positive")

    edit_mt = read_jsonl(args.edit_mt_jsonl)
    edit = read_jsonl(args.edit_jsonl)
    validation = read_jsonl(args.validation_jsonl)
    if any(row.get("sample_type") != "edit_mt" for row in edit_mt):
        raise ValueError("Every edit_mt source row must have sample_type=edit_mt")
    if any(row.get("sample_type") != "edit" for row in edit):
        raise ValueError("Every edit source row must have sample_type=edit")

    validation_paths = {
        resolve_image(args.validation_base, row[key])
        for row in validation
        for key in ("edit_image", "image")
    }
    with concurrent.futures.ThreadPoolExecutor(args.io_workers) as executor:
        validation_hash_pairs = list(executor.map(sha256_file, validation_paths))
    validation_hashes = {digest for _, digest in validation_hash_pairs}
    validation_sizes = {path.stat().st_size for path in validation_paths}

    training_paths = {
        resolve_image(args.training_base, row[key])
        for row in edit_mt + edit
        for key in ("edit_image", "image")
    }
    with concurrent.futures.ThreadPoolExecutor(args.io_workers) as executor:
        training_sizes = dict(executor.map(stat_file, training_paths))
    size_matched_paths = {
        path for path, size in training_sizes.items() if size in validation_sizes
    }
    with concurrent.futures.ThreadPoolExecutor(args.io_workers) as executor:
        training_hash_pairs = list(executor.map(sha256_file, size_matched_paths))
    unsafe_paths = {
        path for path, digest in training_hash_pairs if digest in validation_hashes
    }

    def is_unsafe(row: dict) -> bool:
        return any(
            resolve_image(args.training_base, row[key]) in unsafe_paths
            for key in ("edit_image", "image")
        )

    unsafe_mt = [row for row in edit_mt if is_unsafe(row)]
    unsafe_edit = [row for row in edit if is_unsafe(row)]
    safe_mt = [row for row in edit_mt if not is_unsafe(row)]
    safe_edit = [row for row in edit if not is_unsafe(row)]
    safe_mt_ids = {source_identity(row) for row in safe_mt}
    if len(safe_mt_ids) != len(safe_mt):
        raise ValueError("Safe edit_mt pool contains duplicate source identities")

    target_edit_rows = args.target_edit_rows
    if target_edit_rows is None:
        target_edit_rows = (len(safe_mt) + 1) // 2
    if target_edit_rows < 1:
        raise ValueError("target_edit_rows must be positive")
    preferred = [row for row in safe_edit if source_identity(row) not in safe_mt_ids]
    overlap = [row for row in safe_edit if source_identity(row) in safe_mt_ids]
    rng = random.Random(args.seed)
    if target_edit_rows <= len(preferred):
        selected_edit = rng.sample(preferred, target_edit_rows)
    else:
        needed_overlap = target_edit_rows - len(preferred)
        if needed_overlap > len(overlap):
            raise ValueError(
                f"Need {needed_overlap} overlap rows but only {len(overlap)} are safe"
            )
        selected_edit = preferred + rng.sample(overlap, needed_overlap)
        rng.shuffle(selected_edit)

    selected_edit_ids = [source_identity(row) for row in selected_edit]
    if len(set(selected_edit_ids)) != len(selected_edit_ids):
        raise ValueError("Selected edit pool contains duplicate source identities")
    cross_type_overlap = len(set(selected_edit_ids) & safe_mt_ids)
    theoretical_minimum_overlap = max(0, target_edit_rows - len(preferred))
    if cross_type_overlap != theoretical_minimum_overlap:
        raise RuntimeError(
            f"Cross-type overlap {cross_type_overlap} is not minimal "
            f"({theoretical_minimum_overlap})"
        )

    write_jsonl_atomic(safe_mt, args.output_edit_mt_jsonl)
    write_jsonl_atomic(selected_edit, args.output_edit_jsonl)
    report = {
        "validation": {
            "metadata": str(args.validation_jsonl.resolve()),
            "rows": len(validation),
            "unique_image_hashes": len(validation_hashes),
        },
        "content_audit": {
            "training_unique_paths": len(training_paths),
            "size_matched_paths_hashed": len(size_matched_paths),
            "unsafe_training_paths": len(unsafe_paths),
        },
        "edit_mt": {
            "input_rows": len(edit_mt),
            "content_excluded_rows": len(unsafe_mt),
            "output_rows": len(safe_mt),
            "excluded_edit_types": dict(
                Counter((row.get("provenance") or {}).get("edit_type", "") for row in unsafe_mt)
            ),
        },
        "edit": {
            "input_rows": len(edit),
            "content_excluded_rows": len(unsafe_edit),
            "safe_input_rows": len(safe_edit),
            "safe_preferred_non_mt_rows": len(preferred),
            "safe_overlap_rows_available": len(overlap),
            "output_rows": len(selected_edit),
            "cross_type_source_identity_overlap": cross_type_overlap,
            "theoretical_minimum_overlap": theoretical_minimum_overlap,
            "excluded_edit_types": dict(
                Counter((row.get("provenance") or {}).get("edit_type", "") for row in unsafe_edit)
            ),
        },
        "outputs": {
            "edit_mt": str(args.output_edit_mt_jsonl.resolve()),
            "edit": str(args.output_edit_jsonl.resolve()),
        },
        "seed": args.seed,
    }
    write_json_atomic(report, args.report_json)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
