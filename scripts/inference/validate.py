#!/usr/bin/env python3
"""Run a small JSONL validation set through one loaded SAMTokEdit pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from infer_samtok_edit import (
    DEFAULT_QWEN_2511,
    DEFAULT_SAMTOK_TE,
    _REPO_ROOT,
    build_pipeline,
    run_edit,
)


def resolve(path: str, base_path: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else base_path / path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--dataset_base", type=Path, default=Path("."))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--qwen_2511_dir", type=Path, default=DEFAULT_QWEN_2511)
    parser.add_argument("--samtok_te_dir", type=Path, default=DEFAULT_SAMTOK_TE)
    parser.add_argument("--merged_te_dir", type=Path, default=_REPO_ROOT / "models" / "merged_samtok_te")
    parser.add_argument("--te_lora", type=Path, default=None)
    parser.add_argument("--dit_lora", type=Path, default=None)
    parser.add_argument("--max_samples", type=int, default=32)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_gt_cot", action="store_true")
    parser.add_argument("--disable_cot", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    pipe = build_pipeline(
        args.qwen_2511_dir,
        args.samtok_te_dir,
        args.merged_te_dir,
        args.te_lora,
        args.dit_lora,
        args.device,
    )
    with args.metadata.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    rows = rows[: args.max_samples]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, row in enumerate(rows):
        source = Image.open(resolve(row["edit_image"], args.dataset_base)).convert("RGB")
        output = run_edit(
            pipe,
            source,
            row["prompt"],
            seed=args.seed + index,
            num_inference_steps=args.num_inference_steps,
            cfg_scale=args.cfg_scale,
            mt_cot=row.get("mt_cot") if args.use_gt_cot else None,
            enable_samtok_cot=not args.disable_cot,
        )
        save_path = args.output_dir / f"{index:04d}.png"
        output.save(save_path)
        records.append(
            {
                "index": index,
                "prompt": row["prompt"],
                "output": str(save_path),
                "mt_cot": pipe.last_mt_cot,
                "parse_layer": pipe.last_parse_layer,
            }
        )
    (args.output_dir / "results.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
