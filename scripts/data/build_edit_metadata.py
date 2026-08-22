#!/usr/bin/env python3
"""Materialize plain ``edit`` rows from the original CrispEdit parquets."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_edit_mt_metadata import (  # noqa: E402
    DEFAULT_CRISPEDIT,
    DEFAULT_MASKS,
    _decode_image,
    _image_bytes,
    _write_bytes_once,
)


def write_atomic(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crispedit_dir", type=Path, default=DEFAULT_CRISPEDIT)
    parser.add_argument(
        "--output_root",
        type=Path,
        default=_REPO_ROOT / "data" / "crispedit_samtok",
    )
    parser.add_argument("--output_jsonl", type=Path, default=None)
    parser.add_argument(
        "--filter_with_mask_parquets",
        action="store_true",
        help="Only keep rows whose paired mask parquet has filter_decision=keep",
    )
    parser.add_argument("--mask_dir", type=Path, default=DEFAULT_MASKS)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--max_rows", type=int, default=None)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    output_jsonl = args.output_jsonl or output_root / "edit_all.jsonl"
    raw_paths = sorted(args.crispedit_dir.glob("*.parquet"))
    if args.max_files is not None:
        raw_paths = raw_paths[: args.max_files]
    rows = []
    for raw_path in tqdm(raw_paths, desc="CrispEdit edit shards"):
        allowed = None
        if args.filter_with_mask_parquets:
            mask_path = args.mask_dir / raw_path.name
            mask_rows = pq.read_table(
                mask_path, columns=["row_idx", "filter_decision"]
            ).to_pylist()
            allowed = {
                int(row["row_idx"])
                for row in mask_rows
                if row["filter_decision"] == "keep"
            }
        raw_rows = pq.read_table(
            raw_path, columns=["input_img", "instruction", "output_img", "type"]
        ).to_pylist()
        for row_idx, raw in enumerate(raw_rows):
            if allowed is not None and row_idx not in allowed:
                continue
            source_bytes = _image_bytes(raw["input_img"], raw_path.parent)
            target_bytes = _image_bytes(raw["output_img"], raw_path.parent)
            _, source_ext = _decode_image(source_bytes)
            _, target_ext = _decode_image(target_bytes)
            stem = raw_path.stem.replace(" ", "_")
            source_rel = Path("images") / stem / f"{row_idx:06d}_source.{source_ext}"
            target_rel = Path("images") / stem / f"{row_idx:06d}_target.{target_ext}"
            _write_bytes_once(output_root / source_rel, source_bytes)
            _write_bytes_once(output_root / target_rel, target_bytes)
            rows.append(
                {
                    "image": target_rel.as_posix(),
                    "edit_image": source_rel.as_posix(),
                    "prompt": raw["instruction"],
                    "sample_type": "edit",
                }
            )
            if args.max_rows is not None and len(rows) >= args.max_rows:
                break
        if args.max_rows is not None and len(rows) >= args.max_rows:
            break
    write_atomic(rows, output_jsonl)
    print(
        json.dumps(
            {
                "output": str(output_jsonl),
                "rows": len(rows),
                "filtered": args.filter_with_mask_parquets,
                "dataset_base_path": str(output_root),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
