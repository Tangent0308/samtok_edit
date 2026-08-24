#!/usr/bin/env python3
"""Audit, compare, and visualize the completed SAMTokEdit S1-S8 evaluation."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError


DEFAULT_EXPERIMENT_ROOT = Path(
    "/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit"
)
DEFAULT_STAGE1_ROOT = DEFAULT_EXPERIMENT_ROOT / "stage1_evaluation/five_settings"
DEFAULT_STAGE2_ROOT = DEFAULT_EXPERIMENT_ROOT / "stage2_evaluation/three_settings"
DEFAULT_OUTPUT_DIR = DEFAULT_EXPERIMENT_ROOT / "stage2_evaluation/eight_settings"

STAGE1_SETTINGS = (
    "s1_qwen2511_stock",
    "s2_samtok_initial_direct",
    "s3_stage1_te_direct",
    "s4_stage1_te_online_cot",
    "s5_stage1_te_gt_cot",
)
STAGE2_SETTINGS = (
    "s6_stage2_direct",
    "s7_stage2_online_cot",
    "s8_stage2_gt_cot",
)
SETTINGS = STAGE1_SETTINGS + STAGE2_SETTINGS
SETTING_LABELS = {
    "s1_qwen2511_stock": "S1 Stock 2511",
    "s2_samtok_initial_direct": "S2 Initial direct",
    "s3_stage1_te_direct": "S3 Stage-1 direct",
    "s4_stage1_te_online_cot": "S4 Stage-1 online CoT",
    "s5_stage1_te_gt_cot": "S5 Stage-1 GT CoT",
    "s6_stage2_direct": "S6 Stage-2 direct",
    "s7_stage2_online_cot": "S7 Stage-2 online CoT",
    "s8_stage2_gt_cot": "S8 Stage-2 GT CoT",
}
SETTING_ROOT_KIND = {
    **{setting: "stage1" for setting in STAGE1_SETTINGS},
    **{setting: "stage2" for setting in STAGE2_SETTINGS},
}
SIDECAR_PATTERN = re.compile(r"^\d{4}\.json$")
FENCED_JSON_PATTERN = re.compile(r"```json\n(.*)\n```", re.DOTALL)
FOCUSED_PAIRS = {
    "stage1_to_stage2_direct": (
        "s3_stage1_te_direct",
        "s6_stage2_direct",
    ),
    "stage1_to_stage2_online_cot": (
        "s4_stage1_te_online_cot",
        "s7_stage2_online_cot",
    ),
    "stage1_to_stage2_gt_cot": (
        "s5_stage1_te_gt_cot",
        "s8_stage2_gt_cot",
    ),
    "stage1_direct_to_online_cot": (
        "s3_stage1_te_direct",
        "s4_stage1_te_online_cot",
    ),
    "stage2_direct_to_online_cot": (
        "s6_stage2_direct",
        "s7_stage2_online_cot",
    ),
    "stage1_online_to_gt_cot": (
        "s4_stage1_te_online_cot",
        "s5_stage1_te_gt_cot",
    ),
    "stage2_online_to_gt_cot": (
        "s7_stage2_online_cot",
        "s8_stage2_gt_cot",
    ),
}


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1_root", type=Path, default=DEFAULT_STAGE1_ROOT)
    parser.add_argument("--stage2_root", type=Path, default=DEFAULT_STAGE2_ROOT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--make_panels", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args(argv)


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_records(
    stage1_root: Path,
    stage2_root: Path,
) -> tuple[dict[str, dict[int, dict]], dict[str, dict]]:
    roots = {"stage1": stage1_root, "stage2": stage2_root}
    records: dict[str, dict[int, dict]] = {}
    configs: dict[str, dict] = {}
    for setting in SETTINGS:
        root = roots[SETTING_ROOT_KIND[setting]]
        setting_dir = root / setting
        config_path = setting_dir / "run_config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"missing setting config: {config_path}")
        configs[setting] = json.loads(config_path.read_text(encoding="utf-8"))
        sidecars = sorted(
            path
            for path in setting_dir.glob("*.json")
            if SIDECAR_PATTERN.fullmatch(path.name)
        )
        setting_records: dict[int, dict] = {}
        for path in sidecars:
            record = json.loads(path.read_text(encoding="utf-8"))
            index = int(record["metadata_index"])
            if record.get("setting") != setting:
                raise ValueError(
                    f"sidecar setting mismatch in {path}: {record.get('setting')!r}"
                )
            if index in setting_records:
                raise ValueError(f"duplicate metadata_index={index} in {setting_dir}")
            setting_records[index] = record
        records[setting] = setting_records

    index_sets = {setting: set(items) for setting, items in records.items()}
    if len({frozenset(indices) for indices in index_sets.values()}) != 1:
        counts = {setting: len(indices) for setting, indices in index_sets.items()}
        raise ValueError(f"setting index sets differ: {counts}")
    if not next(iter(index_sets.values())):
        raise ValueError("no completed evaluation sidecars found")
    return records, configs


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


def summarize_online_cot(
    records: dict[str, dict[int, dict]],
    indices: list[int],
    setting: str,
) -> dict:
    overall = Counter()
    per_type: defaultdict[str, Counter] = defaultdict(Counter)
    parse_layers = Counter()
    for index in indices:
        record = records[setting][index]
        flags = cot_flags(record)
        edit_type = record["provenance"]["edit_type"]
        gt = parse_cot(record["gt_mt_cot"])
        overall["total"] += 1
        overall["nonempty_gt"] += int(bool(gt))
        overall["empty_gt"] += int(not gt)
        per_type[edit_type]["total"] += 1
        per_type[edit_type]["nonempty_gt"] += int(bool(gt))
        per_type[edit_type]["empty_gt"] += int(not gt)
        parse_layers[str(record["parse_layer"])] += 1
        for name, value in flags.items():
            overall[name] += int(value)
            per_type[edit_type][name] += int(value)
    return {
        "setting": setting,
        "parse_layers": dict(sorted(parse_layers.items())),
        "overall_counts": dict(overall),
        "by_edit_type_counts": {
            edit_type: dict(counts)
            for edit_type, counts in sorted(per_type.items())
        },
    }


def summarize_cot(records: dict[str, dict[int, dict]], indices: list[int]) -> dict:
    direct_settings = (
        "s1_qwen2511_stock",
        "s2_samtok_initial_direct",
        "s3_stage1_te_direct",
        "s6_stage2_direct",
    )
    stage1_online = "s4_stage1_te_online_cot"
    stage2_online = "s7_stage2_online_cot"
    compared_fields = ("conditioned_mt_cot", "parse_layer", "pass1_raw")
    return {
        "stage1_online": summarize_online_cot(records, indices, stage1_online),
        "stage2_online": summarize_online_cot(records, indices, stage2_online),
        "stage1_vs_stage2_online_same_te": {
            field: sum(
                records[stage1_online][index][field]
                == records[stage2_online][index][field]
                for index in indices
            )
            for field in compared_fields
        },
        "gt_oracle_exact_conditioning": {
            setting: sum(
                records[setting][index]["conditioned_mt_cot"]
                == records[setting][index]["gt_mt_cot"]
                for index in indices
            )
            for setting in ("s5_stage1_te_gt_cot", "s8_stage2_gt_cot")
        },
        "direct_settings_with_null_conditioning": {
            setting: sum(
                records[setting][index]["conditioned_mt_cot"] is None
                and records[setting][index]["pass1_raw"] is None
                and records[setting][index]["parse_layer"] is None
                for index in indices
            )
            for setting in direct_settings
        },
    }


def validate_cot_routing(cot_report: dict, sample_count: int) -> None:
    direct_counts = cot_report["direct_settings_with_null_conditioning"]
    invalid_direct = {
        setting: count
        for setting, count in direct_counts.items()
        if count != sample_count
    }
    if invalid_direct:
        raise ValueError(
            "direct settings unexpectedly used CoT/pass-1 telemetry: "
            f"{invalid_direct}"
        )

    same_te = cot_report["stage1_vs_stage2_online_same_te"]
    mismatched_te = {
        field: count for field, count in same_te.items() if count != sample_count
    }
    if mismatched_te:
        raise ValueError(
            "S4/S7 use the same deterministic Stage-1 TE but pass-1 outputs differ: "
            f"{mismatched_te}"
        )

    oracle_counts = cot_report["gt_oracle_exact_conditioning"]
    invalid_oracle = {
        setting: count
        for setting, count in oracle_counts.items()
        if count != sample_count
    }
    if invalid_oracle:
        raise ValueError(
            "GT-CoT settings did not condition on the exact validation CoT: "
            f"{invalid_oracle}"
        )


def _summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def summarize_outputs(
    records: dict[str, dict[int, dict]], indices: list[int]
) -> dict:
    hashes: dict[str, dict[int, str]] = {setting: {} for setting in SETTINGS}
    sizes: dict[str, Counter[str]] = {setting: Counter() for setting in SETTINGS}
    pair_values: dict[tuple[str, str], list[float]] = {
        pair: [] for pair in itertools.combinations(SETTINGS, 2)
    }
    focused_by_type: dict[str, defaultdict[str, list[float]]] = {
        name: defaultdict(list) for name in FOCUSED_PAIRS
    }

    for index in indices:
        arrays = {}
        shapes = {}
        for setting in SETTINGS:
            path = Path(records[setting][index]["output"])
            if not path.is_file():
                raise FileNotFoundError(f"missing output image: {path}")
            try:
                with Image.open(path) as image:
                    image.load()
                    rgb = image.convert("RGB")
                    sizes[setting][f"{rgb.width}x{rgb.height}"] += 1
                    arrays[setting] = np.asarray(rgb, dtype=np.float32)
                    shapes[setting] = arrays[setting].shape
            except (OSError, UnidentifiedImageError) as error:
                raise ValueError(f"invalid output image: {path}") from error
            hashes[setting][index] = sha256(path)
        if len(set(shapes.values())) != 1:
            raise ValueError(f"output shape mismatch at index={index}: {shapes}")

        for pair in pair_values:
            left, right = pair
            mae = float(np.mean(np.abs(arrays[left] - arrays[right])) / 255.0)
            pair_values[pair].append(mae)
        edit_type = records[SETTINGS[0]][index]["provenance"]["edit_type"]
        for name, pair in FOCUSED_PAIRS.items():
            left, right = pair
            mae = float(np.mean(np.abs(arrays[left] - arrays[right])) / 255.0)
            focused_by_type[name][edit_type].append(mae)

    pairwise = {}
    for (left, right), values in pair_values.items():
        pairwise[f"{left}__vs__{right}"] = {
            "byte_identical": sum(
                hashes[left][index] == hashes[right][index] for index in indices
            ),
            "normalized_rgb_mae": _summary(values),
        }
    focused = {}
    for name, pair in FOCUSED_PAIRS.items():
        key = f"{pair[0]}__vs__{pair[1]}"
        focused[name] = {
            "settings": list(pair),
            **pairwise[key],
            "by_edit_type_normalized_rgb_mae": {
                edit_type: _summary(values)
                for edit_type, values in sorted(focused_by_type[name].items())
            },
        }
    return {
        "decodable": {setting: len(indices) for setting in SETTINGS},
        "unique_output_hashes": {
            setting: len(set(setting_hashes.values()))
            for setting, setting_hashes in hashes.items()
        },
        "output_sizes": {
            setting: dict(sorted(setting_sizes.items()))
            for setting, setting_sizes in sizes.items()
        },
        "focused_comparisons": focused,
        "pairwise_all_28": pairwise,
        "metric_note": (
            "Normalized RGB MAE and byte identity only measure whether outputs "
            "changed. They are integrity/sensitivity diagnostics, not semantic "
            "image-editing quality metrics."
        ),
    }


def summarize_configs(configs: dict[str, dict]) -> dict:
    expected_modes = {
        "s1_qwen2511_stock": "disabled",
        "s2_samtok_initial_direct": "disabled",
        "s3_stage1_te_direct": "disabled",
        "s4_stage1_te_online_cot": "online",
        "s5_stage1_te_gt_cot": "ground_truth",
        "s6_stage2_direct": "disabled",
        "s7_stage2_online_cot": "online",
        "s8_stage2_gt_cot": "ground_truth",
    }
    for setting, config in configs.items():
        specs = config.get("settings", [])
        if len(specs) != 1 or specs[0].get("key") != setting:
            raise ValueError(
                f"setting config does not describe exactly {setting}: {specs}"
            )
        if specs[0].get("cot_mode") != expected_modes[setting]:
            raise ValueError(
                f"unexpected cot_mode for {setting}: {specs[0].get('cot_mode')!r}"
            )

    metadata_hashes = {
        setting: config["data"]["metadata_sha256"]
        for setting, config in configs.items()
    }
    if len(set(metadata_hashes.values())) != 1:
        raise ValueError(f"metadata hashes differ across settings: {metadata_hashes}")
    stage2_dit_hashes = {
        configs[setting]["models"]["dit_lora_sha256"]
        for setting in STAGE2_SETTINGS
    }
    if len(stage2_dit_hashes) != 1:
        raise ValueError(f"Stage-2 settings use different DiT LoRAs: {stage2_dit_hashes}")
    stage1_te_hashes = {
        configs[setting]["models"]["stage1_te_lora_sha256"]
        for setting in STAGE1_SETTINGS[2:] + STAGE2_SETTINGS
    }
    if len(stage1_te_hashes) != 1:
        raise ValueError(
            f"Stage-1/Stage-2 method settings use different TE LoRAs: {stage1_te_hashes}"
        )
    return {
        "metadata_sha256": next(iter(metadata_hashes.values())),
        "stage1_te_lora_sha256": next(iter(stage1_te_hashes)),
        "stage2_dit_lora_sha256": next(iter(stage2_dit_hashes)),
        "stage2_checkpoint": configs[STAGE2_SETTINGS[0]]["models"]["dit_lora"],
        "stage2_checkpoint_step": configs[STAGE2_SETTINGS[0]]["models"][
            "checkpoint_step"
        ],
        "stage2_samples_consumed_with_repeat": configs[STAGE2_SETTINGS[0]][
            "models"
        ]["samples_consumed_with_repeat"],
        "world_sizes": {
            setting: config["parallelism"]["world_size"]
            for setting, config in configs.items()
        },
    }


def _font(size: int, *, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for path in (
        Path("/usr/share/fonts/truetype/dejavu") / name,
        Path("/usr/share/fonts/dejavu") / name,
    ):
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _fit_cell(image: Image.Image, cell_size: tuple[int, int]) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail(cell_size, Image.Resampling.LANCZOS)
    cell = Image.new("RGB", cell_size, "white")
    cell.paste(
        image,
        ((cell_size[0] - image.width) // 2, (cell_size[1] - image.height) // 2),
    )
    return cell


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
    lines = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and draw.textbbox((0, 0), candidate, font=font)[2] > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or [""]


def make_panels(
    records: dict[str, dict[int, dict]], indices: list[int], output_dir: Path
) -> dict:
    panel_dir = output_dir / "panels_with_instruction"
    panel_dir.mkdir(parents=True, exist_ok=True)
    cell_size = (288, 288)
    label_height = 52
    instruction_font = _font(21, bold=True)
    label_font = _font(15, bold=True)
    labels = ["Source", "Target"] + [SETTING_LABELS[setting] for setting in SETTINGS]
    manifest = []
    for index in indices:
        anchor = records[SETTINGS[0]][index]
        paths = [Path(anchor["source"]), Path(anchor["target"])] + [
            Path(records[setting][index]["output"]) for setting in SETTINGS
        ]
        cells = []
        for path in paths:
            with Image.open(path) as image:
                cells.append(_fit_cell(image, cell_size))
        panel_width = cell_size[0] * len(cells)
        edit_type = anchor["provenance"]["edit_type"]
        heading = f"#{index:04d} | {edit_type} | Instruction: {anchor['prompt'].strip()}"
        scratch = Image.new("RGB", (panel_width, 1), "white")
        heading_lines = _wrap(
            ImageDraw.Draw(scratch), heading, instruction_font, panel_width - 32
        )
        line_height = 29
        header_height = max(58, 18 + line_height * len(heading_lines))
        panel = Image.new(
            "RGB",
            (panel_width, header_height + label_height + cell_size[1]),
            "white",
        )
        draw = ImageDraw.Draw(panel)
        draw.rectangle((0, 0, panel_width, header_height), fill=(22, 34, 52))
        for line_index, line in enumerate(heading_lines):
            draw.text(
                (16, 10 + line_index * line_height),
                line,
                font=instruction_font,
                fill="white",
            )
        for column, (cell, label) in enumerate(zip(cells, labels)):
            left = column * cell_size[0]
            label_lines = label.split(" ", 1)
            for line_index, line in enumerate(label_lines):
                bbox = draw.textbbox((0, 0), line, font=label_font)
                text_width = bbox[2] - bbox[0]
                draw.text(
                    (left + max(4, (cell_size[0] - text_width) // 2), header_height + 5 + 20 * line_index),
                    line,
                    font=label_font,
                    fill="black",
                )
            panel.paste(cell, (left, header_height + label_height))
        panel_path = panel_dir / f"{index:04d}.jpg"
        panel.save(panel_path, quality=93)
        manifest.append(
            {
                "metadata_index": index,
                "edit_type": edit_type,
                "instruction": anchor["prompt"],
                "panel": str(panel_path.resolve()),
            }
        )
    _atomic_write_jsonl(panel_dir / "manifest.jsonl", manifest)

    representatives = []
    seen_types = set()
    for item in manifest:
        if item["edit_type"] not in seen_types:
            seen_types.add(item["edit_type"])
            representatives.append(Path(item["panel"]))
    images = []
    for path in representatives:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    overview_path = panel_dir / "overview_representative_7types.jpg"
    if images:
        overview = Image.new(
            "RGB",
            (max(image.width for image in images), sum(image.height for image in images)),
            "white",
        )
        top = 0
        for image in images:
            overview.paste(image, (0, top))
            top += image.height
        overview.save(overview_path, quality=93)
    return {
        "panel_count": len(manifest),
        "panel_dir": str(panel_dir.resolve()),
        "overview": str(overview_path.resolve()) if images else None,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    records, configs = load_records(args.stage1_root, args.stage2_root)
    indices = sorted(records[SETTINGS[0]])
    consistency_fields = (
        "seed",
        "num_inference_steps",
        "cfg_scale",
        "output_size",
        "prompt",
        "source",
        "target",
        "gt_mt_cot",
        "world_size",
    )
    consistency = {
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
    }
    if any(count != len(indices) for count in consistency.values()):
        raise ValueError(f"cross-setting configuration mismatch: {consistency}")

    cot_report = summarize_cot(records, indices)
    validate_cot_routing(cot_report, len(indices))
    report = {
        "status": "complete",
        "protocol": "samtok_edit_eight_setting_comparison_v1",
        "stage1_root": str(args.stage1_root.resolve()),
        "stage2_root": str(args.stage2_root.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "sample_count": len(indices),
        "settings": {setting: len(records[setting]) for setting in SETTINGS},
        "configuration_consistency": consistency,
        "checkpoints": summarize_configs(configs),
        "cot": cot_report,
        "outputs": summarize_outputs(records, indices),
    }
    if args.make_panels:
        report["visualization"] = make_panels(records, indices, args.output_dir)
    report_path = args.output_dir / "analysis" / "eight_setting_audit.json"
    _atomic_write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
