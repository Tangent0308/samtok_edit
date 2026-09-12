#!/usr/bin/env python3
"""Audit the exact Stage-1 schedule without loading images or models."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "DiffSynth-Studio"))

from diffsynth.core.data.samtok_dataset import SamtokEditingDataset  # noqa: E402


def write_atomic(payload: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata_jsonl", type=Path, required=True)
    parser.add_argument("--base_path", type=Path, required=True)
    parser.add_argument("--world_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report_json", type=Path, required=True)
    args = parser.parse_args()

    dataset = SamtokEditingDataset(
        base_path=str(args.base_path),
        metadata_path=str(args.metadata_jsonl),
        repeat=args.repeat,
        data_file_keys=[],
        type_ratio="edit_mt:4,edit_ntp:2,edit:1,edit_umt:1",
        num_processes=args.world_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        seed=args.seed,
    )
    if dataset.schedule is None:
        raise RuntimeError("Stage-1 schedule was not constructed")

    schedule = dataset.schedule
    if args.gradient_accumulation_steps % 8:
        raise ValueError("gradient_accumulation_steps must be a multiple of 8")
    positions_per_step = args.world_size * args.gradient_accumulation_steps
    if len(schedule) % positions_per_step:
        raise RuntimeError("Schedule length is not divisible by one optimizer step")
    ratio_repeats = args.gradient_accumulation_steps // 8
    expected_step_counts = Counter(
        edit_mt=4 * args.world_size * ratio_repeats,
        edit_ntp=2 * args.world_size * ratio_repeats,
        edit=args.world_size * ratio_repeats,
        edit_umt=args.world_size * ratio_repeats,
    )
    global_counts = Counter()
    step_count = len(schedule) // positions_per_step
    for step in range(step_count):
        indices = schedule[
            step * positions_per_step : (step + 1) * positions_per_step
        ]
        types = [dataset.data[index]["sample_type"] for index in indices]
        counts = Counter(types)
        if counts != expected_step_counts:
            raise RuntimeError(f"Bad type ratio in optimizer step {step}: {counts}")
        rank_sequences = [types[rank :: args.world_size] for rank in range(args.world_size)]
        if any(sequence != rank_sequences[0] for sequence in rank_sequences[1:]):
            raise RuntimeError(f"Rank type sequences diverge in optimizer step {step}")
        for micro_start in range(0, positions_per_step, args.world_size):
            if len(set(types[micro_start : micro_start + args.world_size])) != 1:
                raise RuntimeError(
                    f"Ranks receive mixed sample types in step {step}, "
                    f"micro-step {micro_start // args.world_size}"
                )
        global_counts.update(types)

    uses = Counter(schedule)
    expected_uses = args.repeat
    if len(uses) != len(dataset.data) or set(uses.values()) != {expected_uses}:
        raise RuntimeError(
            "Schedule does not consume every metadata row exactly repeat times: "
            f"unique={len(uses)}/{len(dataset.data)}, uses={sorted(set(uses.values()))}"
        )

    report = {
        "passed": True,
        "metadata": str(args.metadata_jsonl.resolve()),
        "metadata_rows": len(dataset.data),
        "schedule_rows": len(schedule),
        "world_size": args.world_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "optimizer_steps": step_count,
        "repeat": args.repeat,
        "global_sample_type_counts": dict(global_counts),
        "per_optimizer_step_counts": dict(expected_step_counts),
        "rank_type_sequences_equal": True,
        "rank_micro_steps_homogeneous": True,
        "unique_metadata_rows_consumed": len(uses),
        "uses_per_metadata_row": sorted(set(uses.values())),
        "implicit_schedule_recycling": False,
    }
    write_atomic(report, args.report_json)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
