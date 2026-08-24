#!/usr/bin/env python3
"""Audit completed Stage 1 five-setting evaluation artifacts.

This analysis is intentionally model-free.  It validates sidecars and images,
measures online/GT CoT agreement, and checks whether settings accidentally
produced byte-identical or nearly identical outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


DEFAULT_ROOT = Path(
    "/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/"
    "stage1_evaluation/five_settings"
)
SETTINGS = [
    "s1_qwen2511_stock",
    "s2_samtok_initial_direct",
    "s3_stage1_te_direct",
    "s4_stage1_te_online_cot",
    "s5_stage1_te_gt_cot",
]
SIDECAR_PATTERN = re.compile(r"^\d{4}\.json$")
FENCED_JSON_PATTERN = re.compile(r"```json\n(.*)\n```", re.DOTALL)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="default: ROOT/analysis/quantitative_audit.json",
    )
    return parser.parse_args()


def load_records(root: Path) -> dict[str, dict[int, dict]]:
    records: dict[str, dict[int, dict]] = {}
    for setting in SETTINGS:
        setting_dir = root / setting
        sidecars = sorted(
            path for path in setting_dir.glob("*.json") if SIDECAR_PATTERN.match(path.name)
        )
        setting_records = {}
        for path in sidecars:
            record = json.loads(path.read_text(encoding="utf-8"))
            index = int(record["metadata_index"])
            if index in setting_records:
                raise ValueError(f"duplicate metadata_index={index} in {setting_dir}")
            setting_records[index] = record
        records[setting] = setting_records
    index_sets = {setting: set(items) for setting, items in records.items()}
    if len({frozenset(indices) for indices in index_sets.values()}) != 1:
        raise ValueError(f"setting index sets differ: {index_sets}")
    return records


def parse_cot(value: str | None):
    if value is None:
        return None
    match = FENCED_JSON_PATTERN.fullmatch(value)
    return json.loads(match.group(1) if match else value)


def cot_flags(record: dict) -> dict[str, bool]:
    gt = parse_cot(record["gt_mt_cot"])
    predicted = parse_cot(record["conditioned_mt_cot"])
    nonempty_pair = bool(gt) and bool(predicted) and len(gt) == len(predicted)
    return {
        "exact_canonical": record["conditioned_mt_cot"] == record["gt_mt_cot"],
        "both_empty": gt == [] and predicted == [],
        "gt_empty_predicted_nonempty": gt == [] and predicted != [],
        "gt_nonempty_predicted_empty": gt != [] and predicted == [],
        "object_count_match": len(gt) == len(predicted),
        "mask_exact_nonempty": nonempty_pair
        and all(a["mask_2d"] == b["mask_2d"] for a, b in zip(gt, predicted)),
        "label_exact_nonempty": nonempty_pair
        and all(a["label"] == b["label"] for a, b in zip(gt, predicted)),
    }


def summarize_cot(records: dict[str, dict[int, dict]], indices: list[int]):
    online = records[SETTINGS[3]]
    gt_oracle = records[SETTINGS[4]]
    overall = Counter()
    per_type: defaultdict[str, Counter] = defaultdict(Counter)
    parse_layers = Counter()
    nonempty_gt = 0
    empty_gt = 0
    for index in indices:
        record = online[index]
        flags = cot_flags(record)
        edit_type = record["provenance"]["edit_type"]
        overall["total"] += 1
        per_type[edit_type]["total"] += 1
        gt = parse_cot(record["gt_mt_cot"])
        if gt:
            nonempty_gt += 1
            per_type[edit_type]["nonempty_gt"] += 1
        else:
            empty_gt += 1
            per_type[edit_type]["empty_gt"] += 1
        parse_layers[record["parse_layer"]] += 1
        for name, value in flags.items():
            overall[name] += int(value)
            per_type[edit_type][name] += int(value)
    return {
        "online_parse_layers": dict(sorted(parse_layers.items())),
        "gt_nonempty": nonempty_gt,
        "gt_empty": empty_gt,
        "overall_counts": dict(overall),
        "by_edit_type_counts": {
            edit_type: dict(counts)
            for edit_type, counts in sorted(per_type.items())
        },
        "gt_oracle_exact_conditioning": sum(
            gt_oracle[index]["conditioned_mt_cot"] == gt_oracle[index]["gt_mt_cot"]
            for index in indices
        ),
        "direct_settings_with_null_conditioning": {
            setting: sum(
                records[setting][index]["conditioned_mt_cot"] is None
                and records[setting][index]["pass1_raw"] is None
                and records[setting][index]["parse_layer"] is None
                for index in indices
            )
            for setting in SETTINGS[:3]
        },
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_outputs(records: dict[str, dict[int, dict]], indices: list[int]):
    hashes: dict[str, dict[int, str]] = {setting: {} for setting in SETTINGS}
    decodable = Counter()
    sizes: dict[str, Counter] = {setting: Counter() for setting in SETTINGS}
    for setting in SETTINGS:
        for index in indices:
            path = Path(records[setting][index]["output"])
            try:
                with Image.open(path) as image:
                    image.load()
                    sizes[setting][f"{image.width}x{image.height}"] += 1
            except (OSError, UnidentifiedImageError) as error:
                raise ValueError(f"invalid output image: {path}") from error
            hashes[setting][index] = sha256(path)
            decodable[setting] += 1

    pairwise = {}
    for left_pos, left in enumerate(SETTINGS):
        for right in SETTINGS[left_pos + 1 :]:
            normalized_mae = []
            for index in indices:
                left_image = np.asarray(
                    Image.open(records[left][index]["output"]).convert("RGB"),
                    dtype=np.float32,
                )
                right_image = np.asarray(
                    Image.open(records[right][index]["output"]).convert("RGB"),
                    dtype=np.float32,
                )
                if left_image.shape != right_image.shape:
                    raise ValueError(
                        f"shape mismatch at index={index}: "
                        f"{left}={left_image.shape}, {right}={right_image.shape}"
                    )
                normalized_mae.append(float(np.mean(np.abs(left_image - right_image)) / 255.0))
            pairwise[f"{left}__vs__{right}"] = {
                "byte_identical": sum(
                    hashes[left][index] == hashes[right][index] for index in indices
                ),
                "normalized_rgb_mae_mean": float(np.mean(normalized_mae)),
                "normalized_rgb_mae_min": min(normalized_mae),
                "normalized_rgb_mae_max": max(normalized_mae),
            }
    return {
        "decodable": dict(decodable),
        "unique_output_hashes": {
            setting: len(set(setting_hashes.values()))
            for setting, setting_hashes in hashes.items()
        },
        "output_sizes": {
            setting: dict(sorted(setting_sizes.items()))
            for setting, setting_sizes in sizes.items()
        },
        "pairwise": pairwise,
        "metric_note": (
            "normalized RGB MAE only detects output changes; it is not a semantic "
            "image-editing quality metric"
        ),
    }


def main():
    args = parse_args()
    output = args.output or args.root / "analysis" / "quantitative_audit.json"
    records = load_records(args.root)
    indices = sorted(records[SETTINGS[0]])
    consistency_fields = [
        "seed",
        "num_inference_steps",
        "cfg_scale",
        "output_size",
        "prompt",
        "source",
        "target",
        "gt_mt_cot",
        "world_size",
    ]
    report = {
        "status": "complete",
        "root": str(args.root),
        "sample_count": len(indices),
        "settings": {setting: len(records[setting]) for setting in SETTINGS},
        "configuration_consistency": {
            field: sum(
                len(
                    {
                        json.dumps(records[setting][index][field], sort_keys=True)
                        for setting in SETTINGS
                    }
                )
                == 1
                for index in indices
            )
            for field in consistency_fields
        },
        "cot": summarize_cot(records, indices),
        "outputs": summarize_outputs(records, indices),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
