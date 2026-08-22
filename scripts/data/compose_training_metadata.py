#!/usr/bin/env python3
"""Compose Stage-1 and Stage-2 JSONL files from the three sample sources."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "DiffSynth-Studio"))

from diffsynth.core.data.samtok_dataset import (  # noqa: E402
    parse_and_canonicalize_mt_cot,
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate(rows: list[dict], expected_type: str):
    for row_id, row in enumerate(rows):
        if row.get("sample_type") != expected_type:
            raise ValueError(
                f"{expected_type} input row {row_id} has sample_type={row.get('sample_type')!r}"
            )
        required = {"edit_image", "prompt"}
        if expected_type != "edit_ntp":
            required.add("image")
        missing = required - row.keys()
        if missing:
            raise ValueError(f"{expected_type} input row {row_id} missing {missing}")
        if expected_type in {"edit_mt", "edit_ntp"}:
            canonical = parse_and_canonicalize_mt_cot(row.get("mt_cot"))
            if canonical != row.get("mt_cot"):
                raise ValueError(f"{expected_type} input row {row_id} has non-canonical mt_cot")


def cap(rows: list[dict], maximum: int | None, rng: random.Random) -> list[dict]:
    if maximum is None or maximum >= len(rows):
        return rows[:]
    return rng.sample(rows, maximum)


def write_atomic(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edit_mt_jsonl", type=Path, required=True)
    parser.add_argument("--edit_ntp_jsonl", type=Path, required=True)
    parser.add_argument("--edit_jsonl", type=Path, required=True)
    parser.add_argument("--stage1_output", type=Path, required=True)
    parser.add_argument("--stage2_output", type=Path, required=True)
    parser.add_argument("--max_edit_mt", type=int, default=None)
    parser.add_argument("--max_edit_ntp", type=int, default=None)
    parser.add_argument("--max_edit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    edit_mt = read_jsonl(args.edit_mt_jsonl)
    edit_ntp = read_jsonl(args.edit_ntp_jsonl)
    edit = read_jsonl(args.edit_jsonl)
    validate(edit_mt, "edit_mt")
    validate(edit_ntp, "edit_ntp")
    validate(edit, "edit")
    edit_mt = cap(edit_mt, args.max_edit_mt, rng)
    edit_ntp = cap(edit_ntp, args.max_edit_ntp, rng)
    edit = cap(edit, args.max_edit, rng)

    stage1 = edit_mt + edit_ntp + edit
    stage2 = edit_mt + edit
    rng.shuffle(stage1)
    rng.shuffle(stage2)
    write_atomic(stage1, args.stage1_output)
    write_atomic(stage2, args.stage2_output)
    print(
        json.dumps(
            {
                "stage1": {
                    "path": str(args.stage1_output),
                    "rows": len(stage1),
                    "edit_mt": len(edit_mt),
                    "edit_ntp": len(edit_ntp),
                    "edit": len(edit),
                    "note": "The Stage-1 dataset schedule enforces the runtime 2:1:1 ratio.",
                },
                "stage2": {
                    "path": str(args.stage2_output),
                    "rows": len(stage2),
                    "edit_mt": len(edit_mt),
                    "edit": len(edit),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
