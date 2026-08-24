#!/usr/bin/env python3
"""Compose Stage-1 and/or Stage-2 JSONL files from validated sample sources."""

from __future__ import annotations

import argparse
from collections import Counter
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


def arrange_stage2_rows(
    edit_mt: list[dict],
    edit: list[dict],
    rng: random.Random,
    num_shards: int | None,
    pad_to_shards: bool = False,
) -> tuple[list[dict], list[dict[str, int]] | None, dict[str, int]]:
    if num_shards is None:
        rows = edit_mt + edit
        rng.shuffle(rows)
        return rows, None, {}
    if num_shards < 1:
        raise ValueError("stage2_num_shards must be positive")
    padding = Counter()
    if pad_to_shards:
        # Preserve every source row, then find the smallest final 2:1 counts
        # whose two types are independently divisible by the strided shard
        # count.  This also handles an odd all-eligible edit_mt pool without
        # silently deleting a real row.
        minimum_edit = max(len(edit), (len(edit_mt) + 1) // 2)
        final_edit = (
            (minimum_edit + num_shards - 1) // num_shards * num_shards
        )
        final_mt = 2 * final_edit
        edit_padding = final_edit - len(edit)
        mt_padding = final_mt - len(edit_mt)
        if mt_padding or edit_padding:
            if not edit_mt or not edit:
                raise ValueError("Cannot pad an empty Stage-2 source")

            def padded_copy(row: dict, ordinal: int) -> dict:
                copied = row.copy()
                copied["schedule_padding"] = {
                    "reason": "exact_2_to_1_ratio_and_strided_shard_divisibility",
                    "ordinal": ordinal,
                }
                return copied

            def padding_choices(rows: list[dict], count: int) -> list[dict]:
                if count <= len(rows):
                    return rng.sample(rows, count)
                return rng.choices(rows, k=count)

            mt_choices = padding_choices(edit_mt, mt_padding)
            edit_choices = padding_choices(edit, edit_padding)
            edit_mt = edit_mt + [
                padded_copy(row, ordinal)
                for ordinal, row in enumerate(mt_choices)
            ]
            edit = edit + [
                padded_copy(row, ordinal)
                for ordinal, row in enumerate(edit_choices)
            ]
            if mt_padding:
                padding["edit_mt"] = mt_padding
            if edit_padding:
                padding["edit"] = edit_padding
    if len(edit_mt) % num_shards or len(edit) % num_shards:
        raise ValueError(
            "Stage-2 type counts must each be divisible by stage2_num_shards; "
            f"got edit_mt={len(edit_mt)}, edit={len(edit)}, shards={num_shards}"
        )

    shuffled_mt = edit_mt[:]
    shuffled_edit = edit[:]
    rng.shuffle(shuffled_mt)
    rng.shuffle(shuffled_edit)
    mt_per_shard = len(shuffled_mt) // num_shards
    edit_per_shard = len(shuffled_edit) // num_shards
    shards = []
    for shard_id in range(num_shards):
        shard = (
            shuffled_mt[
                shard_id * mt_per_shard : (shard_id + 1) * mt_per_shard
            ]
            + shuffled_edit[
                shard_id * edit_per_shard : (shard_id + 1) * edit_per_shard
            ]
        )
        rng.shuffle(shard)
        shards.append(shard)

    rows = [
        shards[shard_id][position]
        for position in range(len(shards[0]))
        for shard_id in range(num_shards)
    ]
    shard_counts = [
        dict(Counter(row["sample_type"] for row in rows[shard_id::num_shards]))
        for shard_id in range(num_shards)
    ]
    return rows, shard_counts, dict(padding)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edit_mt_jsonl", type=Path, required=True)
    parser.add_argument("--edit_ntp_jsonl", type=Path, default=None)
    parser.add_argument("--edit_jsonl", type=Path, required=True)
    parser.add_argument("--stage1_output", type=Path, default=None)
    parser.add_argument("--stage2_output", type=Path, default=None)
    parser.add_argument("--max_edit_mt", type=int, default=None)
    parser.add_argument("--max_edit_ntp", type=int, default=None)
    parser.add_argument("--max_edit", type=int, default=None)
    parser.add_argument(
        "--stage2_num_shards",
        type=int,
        default=None,
        help="Arrange Stage-2 rows so every strided distributed shard has the same type ratio",
    )
    parser.add_argument(
        "--pad_stage2_to_shards",
        action="store_true",
        help=(
            "Deterministically duplicate the minimum 2:1 block needed for exact "
            "strided-shard divisibility; duplicated rows carry schedule_padding metadata"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.stage1_output is None and args.stage2_output is None:
        parser.error("At least one of --stage1_output or --stage2_output is required")
    if args.stage1_output is not None and args.edit_ntp_jsonl is None:
        parser.error("--edit_ntp_jsonl is required when composing Stage 1")

    rng = random.Random(args.seed)
    edit_mt = read_jsonl(args.edit_mt_jsonl)
    edit = read_jsonl(args.edit_jsonl)
    validate(edit_mt, "edit_mt")
    validate(edit, "edit")
    edit_mt = cap(edit_mt, args.max_edit_mt, rng)
    edit_ntp = None
    if args.stage1_output is not None:
        edit_ntp = read_jsonl(args.edit_ntp_jsonl)
        validate(edit_ntp, "edit_ntp")
        edit_ntp = cap(edit_ntp, args.max_edit_ntp, rng)
    edit = cap(edit, args.max_edit, rng)

    report = {}
    if args.stage1_output is not None:
        stage1 = edit_mt + edit_ntp + edit
        rng.shuffle(stage1)
        write_atomic(stage1, args.stage1_output)
        report["stage1"] = {
            "path": str(args.stage1_output),
            "rows": len(stage1),
            "edit_mt": len(edit_mt),
            "edit_ntp": len(edit_ntp),
            "edit": len(edit),
            "note": "The Stage-1 dataset schedule enforces the runtime 2:1:1 ratio.",
        }
    if args.stage2_output is not None:
        source_counts = {"edit_mt": len(edit_mt), "edit": len(edit)}
        stage2, shard_counts, padding = arrange_stage2_rows(
            edit_mt,
            edit,
            rng,
            args.stage2_num_shards,
            pad_to_shards=args.pad_stage2_to_shards,
        )
        write_atomic(stage2, args.stage2_output)
        final_counts = Counter(row["sample_type"] for row in stage2)
        report["stage2"] = {
            "path": str(args.stage2_output),
            "rows": len(stage2),
            "source_counts": source_counts,
            "edit_mt": final_counts["edit_mt"],
            "edit": final_counts["edit"],
            "schedule_padding": padding,
            "ratio": (
                "edit_mt:2,edit:1"
                if final_counts["edit_mt"] == 2 * final_counts["edit"]
                else None
            ),
            "num_strided_shards": args.stage2_num_shards,
            "strided_shard_counts": shard_counts,
        }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
