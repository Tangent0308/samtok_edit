#!/usr/bin/env python3
"""Compare refined Stage-2 checkpoints trained on 8 and 32 GPUs."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in [
    REPO_ROOT / "scripts" / "data",
    REPO_ROOT / "scripts" / "eval",
]:
    sys.path.insert(0, str(path))

from run_stage1_eval import (  # noqa: E402
    _atomic_write_json,
    _atomic_write_jsonl,
    _fit_panel_cell,
    _panel_font,
    _wrap_panel_text,
    resolve_data_path,
    sha256_file,
)
from run_stage2_eval import (  # noqa: E402
    DEFAULT_DATASET_BASE,
    DEFAULT_EXPERIMENT_ROOT,
    DEFAULT_SAMTOK_TE,
    DEFAULT_VALSET,
    load_and_validate_rows,
)
from samtok_codec import SPAN_RE, SamtokCodec  # noqa: E402


DEFAULT_SINGLE_OUTPUT = DEFAULT_EXPERIMENT_ROOT / "three_settings"
DEFAULT_FOUR_NODE_OUTPUT = DEFAULT_EXPERIMENT_ROOT / "four_node_settings"
DEFAULT_OUTPUT = DEFAULT_EXPERIMENT_ROOT / "single_vs_four_node"
STOCK_KEY = "s1_qwen2511_stock"
ONLINE_KEY = "s2_stage2_online_cot"
UMT_KEY = "s3_stage2_edit_umt"


def read_records(root: Path, setting: str, rows: list[dict]) -> dict[int, dict]:
    setting_root = root / setting
    records = {}
    for row in rows:
        index = int(row["eval_index"])
        path = setting_root / f"{index:04d}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing evaluation record: {path}")
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("eval_index") != index or record.get("eval_id") != row["eval_id"]:
            raise ValueError(f"Evaluation identity mismatch: {path}")
        if record.get("setting") != setting:
            raise ValueError(f"Evaluation setting mismatch: {path}")
        output = Path(record["output"])
        if not output.is_file():
            raise FileNotFoundError(f"Missing generated image: {output}")
        with Image.open(output) as image:
            image.verify()
        records[index] = record
    return records


def read_setting_config(root: Path, setting: str) -> dict:
    path = root / setting / "run_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing setting config: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_protocol(
    rows: list[dict], single_root: Path, four_root: Path, metadata_hash: str
) -> dict:
    configs = {
        "single_stock": read_setting_config(single_root, STOCK_KEY),
        "single_online": read_setting_config(single_root, ONLINE_KEY),
        "single_umt": read_setting_config(single_root, UMT_KEY),
        "four_node_online": read_setting_config(four_root, ONLINE_KEY),
        "four_node_umt": read_setting_config(four_root, UMT_KEY),
    }
    generation_fields = [
        "seed_rule",
        "num_inference_steps",
        "cfg_scale",
        "samtok_max_new_tokens",
        "dtype",
        "edit_image_auto_resize",
        "zero_cond_t",
        "height_width",
    ]
    reference = configs["single_stock"]["generation"]
    for name, config in configs.items():
        if config["data"]["metadata_sha256"] != metadata_hash:
            raise ValueError(f"{name} used a different validation metadata hash")
        for field in generation_fields:
            if config["generation"][field] != reference[field]:
                raise ValueError(f"{name} differs in generation field {field}")
        if int(config["data"]["selected_rows"]) != len(rows):
            raise ValueError(f"{name} did not evaluate all selected rows")
    single_models = configs["single_online"]["models"]
    four_models = configs["four_node_online"]["models"]
    if configs["single_umt"]["models"] != single_models:
        raise ValueError("Single-node online and UMT settings used different models")
    if configs["four_node_umt"]["models"] != four_models:
        raise ValueError("Four-node online and UMT settings used different models")
    return {
        "generation": {field: reference[field] for field in generation_fields},
        "single_node_models": single_models,
        "four_node_models": four_models,
        "inference_world_sizes": {
            name: int(config["parallelism"]["world_size"])
            for name, config in configs.items()
        },
    }


def overlay(image: Image.Image, mask: np.ndarray) -> Image.Image:
    array = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    selected = np.asarray(mask, dtype=bool)
    array[selected] = array[selected] * 0.45 + np.asarray([245, 62, 72]) * 0.55
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def read_binary_mask(path: Path, source_size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        if image.size != source_size:
            raise ValueError(f"Mask geometry mismatch: {path}: {image.size} != {source_size}")
        return np.asarray(image.convert("L")) > 0


def decode_online_mask(codec: SamtokCodec, source: Image.Image, record: dict):
    cot = record.get("conditioned_mt_cot")
    masks = codec.decode(source, cot) if cot else []
    if not masks:
        return None, 0
    return np.logical_or.reduce([np.asarray(mask, dtype=bool) for mask in masks]), len(masks)


def mask_metrics(reference: np.ndarray, prediction: np.ndarray | None) -> dict:
    if prediction is None:
        return {
            "decodable": False,
            "iou": None,
            "dice": None,
            "precision": None,
            "recall": None,
            "area_fraction": None,
        }
    reference = np.asarray(reference, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    intersection = int(np.logical_and(reference, prediction).sum())
    union = int(np.logical_or(reference, prediction).sum())
    reference_area = int(reference.sum())
    prediction_area = int(prediction.sum())
    return {
        "decodable": True,
        "iou": intersection / union if union else 1.0,
        "dice": (2 * intersection) / (reference_area + prediction_area)
        if reference_area + prediction_area
        else 1.0,
        "precision": intersection / prediction_area if prediction_area else 0.0,
        "recall": intersection / reference_area if reference_area else 0.0,
        "area_fraction": prediction_area / prediction.size,
    }


def centered_label(draw, left: int, width: int, top: int, text: str, font) -> None:
    text_width = draw.textbbox((0, 0), text, font=font)[2]
    draw.text((left + max(5, (width - text_width) // 2), top), text, font=font, fill="black")


def make_panel(
    cells: list[Image.Image], labels: list[str], heading: str, cell_size=(320, 320)
) -> Image.Image:
    label_height = 48
    width = cell_size[0] * len(cells)
    heading_font, label_font = _panel_font(20, bold=True), _panel_font(15, bold=True)
    scratch = ImageDraw.Draw(Image.new("RGB", (width, 1), "white"))
    lines = _wrap_panel_text(scratch, heading, heading_font, width - 32)
    header_height = max(58, 18 + 28 * len(lines))
    panel = Image.new("RGB", (width, header_height + label_height + cell_size[1]), "white")
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, width, header_height), fill=(22, 34, 52))
    for line_index, line in enumerate(lines):
        draw.text((16, 10 + line_index * 28), line, font=heading_font, fill="white")
    for column, (cell, label) in enumerate(zip(cells, labels)):
        left = column * cell_size[0]
        centered_label(draw, left, cell_size[0], header_height + 12, label, label_font)
        panel.paste(_fit_panel_cell(cell, cell_size), (left, header_height + label_height))
    return panel


def build_overviews(by_category: dict[str, list[Path]], root: Path) -> dict:
    overviews = {}
    for category, paths in sorted(by_category.items()):
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        overview = Image.new(
            "RGB",
            (max(image.width for image in images), sum(image.height for image in images)),
            "white",
        )
        top = 0
        for image in images:
            overview.paste(image, (0, top))
            top += image.height
        path = root / f"overview_{category}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        overview.save(path, quality=92)
        overviews[category] = str(path.resolve())
    return overviews


def numeric_summary(records: list[dict], field: str) -> dict:
    values = [record[field] for record in records if record[field] is not None]
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def summarize_masks(manifest: list[dict], prefix: str) -> dict:
    records = [record[prefix] for record in manifest]
    return {
        "decodable": sum(record["decodable"] for record in records),
        "iou": numeric_summary(records, "iou"),
        "dice": numeric_summary(records, "dice"),
        "precision": numeric_summary(records, "precision"),
        "recall": numeric_summary(records, "recall"),
        "area_fraction": numeric_summary(records, "area_fraction"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--valset", type=Path, default=DEFAULT_VALSET)
    parser.add_argument("--dataset_base", type=Path, default=DEFAULT_DATASET_BASE)
    parser.add_argument("--single_output_dir", type=Path, default=DEFAULT_SINGLE_OUTPUT)
    parser.add_argument("--four_node_output_dir", type=Path, default=DEFAULT_FOUR_NODE_OUTPUT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sam2_ckpt", type=Path, default=DEFAULT_SAMTOK_TE / "sam2.1_hiera_large.pt")
    parser.add_argument(
        "--mask_tokenizer_ckpt",
        type=Path,
        default=DEFAULT_SAMTOK_TE / "mask_tokenizer_256x2.pth",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    rows, data_report = load_and_validate_rows(args.valset, args.dataset_base)
    metadata_hash = sha256_file(args.valset)
    single = {
        key: read_records(args.single_output_dir, key, rows)
        for key in (STOCK_KEY, ONLINE_KEY, UMT_KEY)
    }
    four = {
        key: read_records(args.four_node_output_dir, key, rows)
        for key in (ONLINE_KEY, UMT_KEY)
    }
    protocol = validate_protocol(
        rows, args.single_output_dir, args.four_node_output_dir, metadata_hash
    )

    codec = SamtokCodec(
        args.sam2_ckpt,
        args.mask_tokenizer_ckpt,
        device=args.device,
        dtype=torch.float32,
    )
    edit_root = args.output_dir / "visualizations" / "edit_comparisons"
    mask_root = args.output_dir / "visualizations" / "mask_comparisons"
    edit_by_category: defaultdict[str, list[Path]] = defaultdict(list)
    mask_by_category: defaultdict[str, list[Path]] = defaultdict(list)
    edit_manifest, mask_manifest = [], []
    parse_layers = {"single": Counter(), "four_node": Counter()}

    for position, row in enumerate(rows, 1):
        index = int(row["eval_index"])
        source_path = resolve_data_path(row["edit_image"], args.dataset_base)
        target_path = resolve_data_path(row["image"], args.dataset_base)
        with Image.open(source_path) as image:
            source = image.convert("RGB")
        with Image.open(target_path) as image:
            target = image.convert("RGB")
        raw_mask = read_binary_mask(
            resolve_data_path(row["gt_mask"], args.dataset_base), source.size
        )
        gt_decoded = read_binary_mask(
            resolve_data_path(row["gt_decoded_mask"], args.dataset_base), source.size
        )
        single_online = single[ONLINE_KEY][index]
        four_online = four[ONLINE_KEY][index]
        single_mask, single_spans = decode_online_mask(codec, source, single_online)
        four_mask, four_spans = decode_online_mask(codec, source, four_online)
        parse_layers["single"][str(single_online.get("parse_layer"))] += 1
        parse_layers["four_node"][str(four_online.get("parse_layer"))] += 1
        metrics = {
            "gt_token_decode": mask_metrics(raw_mask, gt_decoded),
            "single_online": mask_metrics(raw_mask, single_mask),
            "four_node_online": mask_metrics(raw_mask, four_mask),
        }
        single_spans_text = [
            match.group(0)
            for match in SPAN_RE.finditer(single_online.get("conditioned_mt_cot") or "")
        ]
        four_spans_text = [
            match.group(0)
            for match in SPAN_RE.finditer(four_online.get("conditioned_mt_cot") or "")
        ]
        metrics["single_online"]["exact_gt_span"] = single_spans_text == [row["gt_mask_span"]]
        metrics["single_online"]["decoded_span_count"] = single_spans
        metrics["four_node_online"]["exact_gt_span"] = four_spans_text == [row["gt_mask_span"]]
        metrics["four_node_online"]["decoded_span_count"] = four_spans

        generated_paths = [
            Path(single[STOCK_KEY][index]["output"]),
            Path(single_online["output"]),
            Path(single[UMT_KEY][index]["output"]),
            Path(four_online["output"]),
            Path(four[UMT_KEY][index]["output"]),
        ]
        generated = []
        for path in generated_paths:
            with Image.open(path) as image:
                generated.append(image.convert("RGB"))
        heading = (
            f"#{index:04d} | {row['primary_category']} | Instruction: {row['prompt']}"
        )
        edit_panel = make_panel(
            [source, target, *generated],
            [
                "Source",
                "GT edited",
                "Stock 2511",
                "8-GPU online CoT",
                "8-GPU edit_umt",
                "32-GPU online CoT",
                "32-GPU edit_umt",
            ],
            heading,
        )
        category = row["primary_category"]
        edit_path = edit_root / category / f"{index:04d}.jpg"
        edit_path.parent.mkdir(parents=True, exist_ok=True)
        edit_panel.save(edit_path, quality=92)
        edit_by_category[category].append(edit_path)

        single_iou = metrics["single_online"]["iou"]
        four_iou = metrics["four_node_online"]["iou"]
        metric_heading = (
            f"{heading} | IoU: GT-token={format_metric(metrics['gt_token_decode']['iou'])}, "
            f"8-GPU={format_metric(single_iou)}, 32-GPU={format_metric(four_iou)}"
        )
        mask_panel = make_panel(
            [
                overlay(source, raw_mask),
                overlay(source, gt_decoded),
                overlay(source, single_mask) if single_mask is not None else source,
                overlay(source, four_mask) if four_mask is not None else source,
            ],
            [
                "GT raw mask",
                "GT token decode",
                "8-GPU online decode",
                "32-GPU online decode",
            ],
            metric_heading,
        )
        mask_path = mask_root / category / f"{index:04d}.jpg"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        mask_panel.save(mask_path, quality=92)
        mask_by_category[category].append(mask_path)

        edit_manifest.append(
            {
                "eval_index": index,
                "eval_id": row["eval_id"],
                "category": category,
                "instruction": row["prompt"],
                "panel": str(edit_path.resolve()),
            }
        )
        mask_manifest.append(
            {
                "eval_index": index,
                "eval_id": row["eval_id"],
                "category": category,
                "instruction": row["prompt"],
                "single_parse_layer": single_online.get("parse_layer"),
                "four_node_parse_layer": four_online.get("parse_layer"),
                **metrics,
                "panel": str(mask_path.resolve()),
            }
        )
        print(
            f"[comparison] {position}/{len(rows)} eval_index={index} "
            f"single_iou={format_metric(single_iou)} "
            f"four_node_iou={format_metric(four_iou)}",
            flush=True,
        )

    edit_overviews = build_overviews(edit_by_category, edit_root)
    mask_overviews = build_overviews(mask_by_category, mask_root)
    _atomic_write_jsonl(edit_root / "manifest.jsonl", edit_manifest)
    _atomic_write_jsonl(mask_root / "manifest.jsonl", mask_manifest)
    single_wins = sum(
        row["single_online"]["iou"] > row["four_node_online"]["iou"]
        for row in mask_manifest
    )
    four_wins = sum(
        row["four_node_online"]["iou"] > row["single_online"]["iou"]
        for row in mask_manifest
    )
    ties = len(mask_manifest) - single_wins - four_wins
    report = {
        "status": "complete",
        "protocol": "refined Stage-2 8-GPU versus 32-GPU ScaleEdit comparison",
        "data": data_report,
        "inputs": {
            "single_output_dir": str(args.single_output_dir.resolve()),
            "four_node_output_dir": str(args.four_node_output_dir.resolve()),
            **protocol,
        },
        "coverage": {
            "rows": len(rows),
            "settings_per_row": 7,
            "existing_generated_settings": 3,
            "new_four_node_generated_settings": 2,
            "generated_images_verified": len(rows) * 5,
        },
        "online_parse_layers": {
            key: dict(sorted(value.items())) for key, value in parse_layers.items()
        },
        "mask_metrics_against_raw_gt": {
            "gt_token_decode": summarize_masks(mask_manifest, "gt_token_decode"),
            "single_online": summarize_masks(mask_manifest, "single_online"),
            "four_node_online": summarize_masks(mask_manifest, "four_node_online"),
            "iou_case_wins": {
                "single": single_wins,
                "four_node": four_wins,
                "ties": ties,
            },
            "exact_gt_span": {
                "single": sum(row["single_online"]["exact_gt_span"] for row in mask_manifest),
                "four_node": sum(
                    row["four_node_online"]["exact_gt_span"] for row in mask_manifest
                ),
            },
        },
        "visualizations": {
            "edit_per_case": len(edit_manifest),
            "edit_category_overviews": edit_overviews,
            "mask_per_case": len(mask_manifest),
            "mask_category_overviews": mask_overviews,
        },
    }
    _atomic_write_json(args.output_dir / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
