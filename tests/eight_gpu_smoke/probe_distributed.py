"""Check Accelerate device binding and per-rank schedule composition."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

from accelerate import Accelerator

from samtok_edit21.data.io import make_schedule, read_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--stage", choices=("stage1", "stage2"), required=True)
    parser.add_argument("--accumulation", type=int, required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260925)
    args = parser.parse_args()
    accelerator = Accelerator()
    rows = read_rows(args.metadata)
    schedule, report = make_schedule(
        rows,
        args.stage,
        accelerator.num_processes,
        args.accumulation,
        steps=args.steps,
        seed=args.seed,
    )
    local = schedule[accelerator.process_index :: accelerator.num_processes]
    local_types = Counter(rows[index].get("sample_type", "cache") for index in local)
    print(
        json.dumps(
            {
                "rank": accelerator.process_index,
                "world_size": accelerator.num_processes,
                "local_rank": os.environ.get("LOCAL_RANK"),
                "device": str(accelerator.device),
                "local_length": len(local),
                "local_types": dict(local_types),
                "global_length": len(schedule),
                "report": report,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
