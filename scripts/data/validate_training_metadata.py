#!/usr/bin/env python3
"""Validate composed SAMTokEdit training metadata and referenced images."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "DiffSynth-Studio"))

from diffsynth.core.data.samtok_dataset import (  # noqa: E402
    parse_and_canonicalize_mt_cot,
    to_cot,
)


REQUIRED_FIELDS = {
    "edit_mt": {"image", "edit_image", "prompt", "mt_cot", "sample_type"},
    "edit_ntp": {"edit_image", "prompt", "mt_cot", "sample_type"},
    "edit": {"image", "edit_image", "prompt", "sample_type"},
}


def parse_expected_counts(text: str | None) -> dict[str, int]:
    if not text:
        return {}
    counts = {}
    for part in text.split(","):
        name, value = part.split(":", 1)
        counts[name.strip()] = int(value)
    return counts


def resolve_image(path: str, base_path: Path) -> Path:
    image_path = Path(path)
    return image_path if image_path.is_absolute() else base_path / image_path


def verify_image(path: Path) -> None:
    with Image.open(path) as image:
        image.verify()


def write_json_atomic(payload: dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata_jsonl", type=Path, required=True)
    parser.add_argument("--base_path", type=Path, required=True)
    parser.add_argument(
        "--expected_counts",
        default=None,
        help="Comma-separated exact counts, for example edit_mt:20000,edit_ntp:10000,edit:10000",
    )
    parser.add_argument("--require_ascii", action="store_true")
    parser.add_argument("--check_paths", action="store_true")
    parser.add_argument("--decode_image_sample", type=int, default=0)
    parser.add_argument("--io_workers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report_json", type=Path, default=None)
    args = parser.parse_args()
    if args.io_workers < 1:
        raise ValueError("io_workers must be positive")

    counts = Counter()
    edit_types = Counter()
    qc_flags = Counter()
    image_paths = set()
    empty_cot = 0
    digest = hashlib.sha256()

    with args.metadata_jsonl.open("rb") as handle:
        for row_id, line in enumerate(handle):
            digest.update(line)
            if not line.strip():
                continue
            row = json.loads(line)
            sample_type = row.get("sample_type")
            if sample_type not in REQUIRED_FIELDS:
                raise ValueError(f"Row {row_id} has invalid sample_type={sample_type!r}")
            missing = REQUIRED_FIELDS[sample_type] - row.keys()
            if missing:
                raise ValueError(f"Row {row_id} is missing required fields: {missing}")
            if not isinstance(row["prompt"], str) or not row["prompt"].strip():
                raise ValueError(f"Row {row_id} has an empty or non-string prompt")
            if args.require_ascii and not row["prompt"].isascii():
                raise ValueError(f"Row {row_id} prompt contains non-ASCII text")

            if sample_type in {"edit_mt", "edit_ntp"}:
                canonical = parse_and_canonicalize_mt_cot(row["mt_cot"])
                if canonical != row["mt_cot"]:
                    raise ValueError(f"Row {row_id} has non-canonical mt_cot")
                if args.require_ascii and not row["mt_cot"].isascii():
                    raise ValueError(f"Row {row_id} mt_cot contains non-ASCII text")
                if row["mt_cot"] == to_cot([]):
                    empty_cot += 1

            for key in ("image", "edit_image"):
                if key in row:
                    image_paths.add(resolve_image(row[key], args.base_path))
            provenance = row.get("provenance") or {}
            if provenance.get("edit_type") is not None:
                edit_types[str(provenance["edit_type"])] += 1
            if provenance.get("qc_flag") is not None:
                qc_flags[str(provenance["qc_flag"])] += 1
            counts[sample_type] += 1

    expected_counts = parse_expected_counts(args.expected_counts)
    if expected_counts and dict(counts) != expected_counts:
        raise ValueError(f"Counts {dict(counts)} do not match expected {expected_counts}")

    sorted_paths = sorted(image_paths)
    if args.check_paths:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.io_workers
        ) as executor:
            exists = executor.map(Path.is_file, sorted_paths)
            missing_paths = [
                str(path) for path, is_file in zip(sorted_paths, exists) if not is_file
            ]
        if missing_paths:
            raise FileNotFoundError(
                f"{len(missing_paths)} referenced images are missing; first={missing_paths[0]}"
            )

    decoded = 0
    if args.decode_image_sample:
        if args.decode_image_sample < 0:
            raise ValueError("decode_image_sample cannot be negative")
        sample_size = min(args.decode_image_sample, len(image_paths))
        sampled_paths = random.Random(args.seed).sample(sorted_paths, sample_size)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.io_workers
        ) as executor:
            for _ in executor.map(verify_image, sampled_paths):
                decoded += 1

    report = {
        "metadata": str(args.metadata_jsonl.resolve()),
        "sha256": digest.hexdigest(),
        "rows": sum(counts.values()),
        "counts": dict(counts),
        "empty_cot": empty_cot,
        "unique_referenced_images": len(image_paths),
        "all_image_paths_exist": args.check_paths,
        "decoded_image_sample": decoded,
        "ascii_text_required": args.require_ascii,
        "edit_types": dict(edit_types),
        "qc_flags": dict(qc_flags),
    }
    if args.report_json is not None:
        write_json_atomic(report, args.report_json)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
