#!/usr/bin/env python3
"""Validate, quantify, and visualize paired DiT mask-token interventions."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "eval"))

from run_eval import (  # noqa: E402
    _atomic_write_json,
    _atomic_write_jsonl,
    _fit_panel_cell,
    _panel_font,
    _verify_image,
    _wrap_panel_text,
    sha256_file,
)


DEFAULT_EXPERIMENT_ROOT = Path(
    "/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/"
    "crispedit_refined/interpretability/dit_mask_token_counterfactual"
)


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def read_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        if image.size != size:
            raise ValueError(f"Mask geometry mismatch: {path}: {image.size} != {size}")
        return np.asarray(image.convert("L")) > 0


def resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255)
    return np.asarray(image.resize((shape[1], shape[0]), Image.Resampling.NEAREST)) > 0


def binary_iou(first: np.ndarray, second: np.ndarray) -> float:
    union = np.logical_or(first, second).sum()
    return float(np.logical_and(first, second).sum() / union) if union else 1.0


def top_area_mask(heatmap: np.ndarray, target_area: int) -> np.ndarray:
    flat = np.asarray(heatmap).reshape(-1)
    count = min(max(1, int(target_area)), flat.size)
    indices = np.argpartition(flat, -count)[-count:]
    selected = np.zeros(flat.size, dtype=bool)
    selected[indices] = True
    return selected.reshape(heatmap.shape)


def centroid(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    total = values.sum()
    if total <= 0:
        return math.nan, math.nan
    y, x = np.indices(values.shape)
    return float((x * values).sum() / total), float((y * values).sum() / total)


def attention_metrics(
    heatmaps: np.ndarray,
    raw_target: np.ndarray,
    decoded_target: np.ndarray,
    raw_other: np.ndarray,
    decoded_other: np.ndarray,
) -> dict:
    shape = tuple(heatmaps.shape[-2:])
    masks = {
        "raw_target": resize_mask(raw_target, shape),
        "decoded_target": resize_mask(decoded_target, shape),
        "raw_other": resize_mask(raw_other, shape),
        "decoded_other": resize_mask(decoded_other, shape),
    }
    per_map = []
    for heatmap in heatmaps:
        heatmap = heatmap / max(float(heatmap.sum()), 1e-12)
        row = {}
        peak = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
        for name, mask in masks.items():
            selected = top_area_mask(heatmap, int(mask.sum()))
            row[f"{name}_mass"] = float(heatmap[mask].sum())
            row[f"{name}_top_area_iou"] = binary_iou(selected, mask)
            row[f"{name}_peak_inside"] = bool(mask[peak])
        per_map.append(row)
    aggregate = heatmaps.mean(axis=0)
    aggregate /= max(float(aggregate.sum()), 1e-12)
    metrics = {
        "num_layer_step_maps": int(len(heatmaps)),
        "source_grid": [int(shape[1]), int(shape[0])],
        "aggregate_attention_centroid_xy": list(centroid(aggregate)),
        "per_layer_step": per_map,
    }
    for name, mask in masks.items():
        area_fraction = float(mask.mean())
        metrics[f"{name}_area_fraction"] = area_fraction
        metrics[f"{name}_mass_mean"] = float(
            np.mean([row[f"{name}_mass"] for row in per_map])
        )
        metrics[f"{name}_top_area_iou_mean"] = float(
            np.mean([row[f"{name}_top_area_iou"] for row in per_map])
        )
        metrics[f"{name}_peak_inside_rate"] = float(
            np.mean([row[f"{name}_peak_inside"] for row in per_map])
        )
        metrics[f"{name}_attention_density"] = (
            metrics[f"{name}_mass_mean"] / area_fraction
            if area_fraction > 0
            else None
        )
        # If the selected top-k cells and target are independent and have the
        # same area fraction p, p/(2-p) is the plug-in chance IoU baseline.
        metrics[f"{name}_top_area_iou_chance"] = (
            area_fraction / (2.0 - area_fraction) if area_fraction > 0 else None
        )
        metrics[f"{name}_top_area_iou_lift"] = (
            metrics[f"{name}_top_area_iou_mean"]
            / metrics[f"{name}_top_area_iou_chance"]
            if metrics[f"{name}_top_area_iou_chance"]
            else None
        )
    metrics["decoded_routing_margin"] = (
        metrics["decoded_target_mass_mean"] - metrics["decoded_other_mass_mean"]
    )
    metrics["raw_routing_margin"] = (
        metrics["raw_target_mass_mean"] - metrics["raw_other_mass_mean"]
    )
    metrics["decoded_density_routing_margin"] = (
        metrics["decoded_target_attention_density"]
        - metrics["decoded_other_attention_density"]
    )
    metrics["raw_density_routing_margin"] = (
        metrics["raw_target_attention_density"]
        - metrics["raw_other_attention_density"]
    )
    return metrics | {"aggregate_heatmap": aggregate}


def overlay_mask(source: Image.Image, mask: np.ndarray, color=(239, 68, 68)) -> Image.Image:
    array = np.asarray(source.convert("RGB"), dtype=np.float32).copy()
    selected = np.asarray(mask, dtype=bool)
    array[selected] = array[selected] * 0.42 + np.asarray(color, dtype=np.float32) * 0.58
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def colorize_heatmap(heatmap: np.ndarray, size: tuple[int, int]) -> Image.Image:
    heatmap = np.asarray(heatmap, dtype=np.float32)
    positive = heatmap[heatmap > 0]
    scale = float(np.percentile(positive, 99.5)) if positive.size else 1.0
    value = np.clip(heatmap / max(scale, 1e-12), 0, 1) ** 0.55
    stops = np.asarray(
        [[15, 23, 42], [29, 78, 216], [34, 211, 238], [250, 204, 21], [239, 68, 68]],
        dtype=np.float32,
    )
    position = value * (len(stops) - 1)
    low = np.floor(position).astype(np.int64)
    high = np.minimum(low + 1, len(stops) - 1)
    fraction = (position - low)[..., None]
    rgb = stops[low] * (1 - fraction) + stops[high] * fraction
    return Image.fromarray(rgb.astype(np.uint8)).resize(size, Image.Resampling.BILINEAR)


def overlay_attention(source: Image.Image, heatmap: np.ndarray) -> Image.Image:
    source_array = np.asarray(source.convert("RGB"), dtype=np.float32)
    colors = np.asarray(colorize_heatmap(heatmap, source.size), dtype=np.float32)
    return Image.fromarray(np.clip(source_array * 0.42 + colors * 0.58, 0, 255).astype(np.uint8))


def overlay_attention_delta(source: Image.Image, delta: np.ndarray) -> Image.Image:
    source_array = np.asarray(source.convert("RGB"), dtype=np.float32)
    image = Image.fromarray(np.asarray(delta, dtype=np.float32), mode="F").resize(
        source.size, Image.Resampling.BILINEAR
    )
    values = np.asarray(image, dtype=np.float32)
    nonzero = np.abs(values[np.abs(values) > 0])
    scale = float(np.percentile(nonzero, 99.0)) if nonzero.size else 1.0
    strength = np.clip(np.abs(values) / max(scale, 1e-12), 0, 1) ** 0.55
    positive = np.asarray([239, 68, 68], dtype=np.float32)
    negative = np.asarray([37, 99, 235], dtype=np.float32)
    colors = np.where((values >= 0)[..., None], positive, negative)
    alpha = (0.72 * strength)[..., None]
    return Image.fromarray(
        np.clip(source_array * (1 - alpha) + colors * alpha, 0, 255).astype(np.uint8)
    )


def make_panel(cells: list[Image.Image], labels: list[str], heading: str) -> Image.Image:
    columns, cell_size = 4, (300, 300)
    rows = math.ceil(len(cells) / columns)
    width = columns * cell_size[0]
    heading_font = _panel_font(20, bold=True)
    label_font = _panel_font(14, bold=True)
    scratch = ImageDraw.Draw(Image.new("RGB", (width, 1), "white"))
    lines = _wrap_panel_text(scratch, heading, heading_font, width - 32)
    header_height = max(62, 16 + 28 * len(lines))
    label_height = 48
    panel = Image.new(
        "RGB", (width, header_height + rows * (label_height + cell_size[1])), "white"
    )
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, width, header_height), fill=(20, 31, 49))
    for index, line in enumerate(lines):
        draw.text((16, 10 + 28 * index), line, font=heading_font, fill="white")
    for index, (cell, label) in enumerate(zip(cells, labels)):
        row, column = divmod(index, columns)
        left = column * cell_size[0]
        top = header_height + row * (label_height + cell_size[1])
        label_lines = _wrap_panel_text(draw, label, label_font, cell_size[0] - 12)[:2]
        for line_index, line in enumerate(label_lines):
            box = draw.textbbox((0, 0), line, font=label_font)
            line_width = box[2] - box[0]
            draw.text(
                (left + max(6, (cell_size[0] - line_width) // 2), top + 5 + 19 * line_index),
                line,
                font=label_font,
                fill="black",
            )
        panel.paste(
            _fit_panel_cell(cell.convert("RGB"), cell_size),
            (left, top + label_height),
        )
    return panel


def cosine(first: np.ndarray, second: np.ndarray) -> float | None:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.dot(first, second) / denominator) if denominator else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    args = parser.parse_args()
    manifest_path = args.experiment_root / "data" / "interventions.jsonl"
    rows = read_rows(manifest_path)
    if not rows:
        raise ValueError("Intervention manifest is empty")

    metric_rows, panel_paths = [], []
    for row in rows:
        case_root = args.experiment_root / "runs" / f"case_{int(row['case_index']):02d}_{row['benchmark_id']}"
        records = {}
        attentions = {}
        for condition in ("original", "alternate"):
            record_path = case_root / f"{condition}_record.json"
            if not record_path.is_file():
                raise FileNotFoundError(f"Missing inference record: {record_path}")
            record = json.loads(record_path.read_text(encoding="utf-8"))
            _verify_image(Path(record["output"]), "output", int(row["case_index"]) + 1)
            archive = np.load(record["attention"])
            mask_to_source_heatmaps = archive["mask_query_to_source_heatmaps"]
            source_to_mask_heatmaps = archive["source_query_to_mask_heatmaps"]
            expected_maps = len(set(archive["layer_ids"].tolist())) * len(
                set(archive["progress_ids"].tolist())
            )
            for heatmaps in (mask_to_source_heatmaps, source_to_mask_heatmaps):
                if len(heatmaps) != expected_maps or not np.allclose(
                    heatmaps.sum(axis=(1, 2)), 1.0, atol=1e-5
                ):
                    raise ValueError(f"Invalid attention archive: {record['attention']}")
            records[condition] = record
            attentions[condition] = {
                "mask_query_to_source_heatmaps": mask_to_source_heatmaps,
                "source_query_to_mask_heatmaps": source_to_mask_heatmaps,
                "mean_raw_source_attention_mass": float(
                    archive["raw_source_attention_mass"].mean()
                ),
                "mean_raw_mask_key_attention_mass": float(
                    archive["raw_mask_key_attention_mass"].mean()
                ),
            }

        if records["original"]["seed"] != records["alternate"]["seed"]:
            raise ValueError(f"{row['benchmark_id']}: paired conditions used different seeds")
        with Image.open(row["source_image"]) as handle:
            source = handle.convert("RGB")
        original_raw = read_mask(Path(row["original_mask"]), source.size)
        alternate_raw = read_mask(Path(row["alternate_mask"]), source.size)
        original_decoded = read_mask(Path(row["original_token_decode"]), source.size)
        alternate_decoded = read_mask(Path(row["alternate_token_decode"]), source.size)

        condition_metrics = {}
        aggregate_heatmaps = {}
        for condition, target_raw, target_decoded, other_raw, other_decoded in (
            ("original", original_raw, original_decoded, alternate_raw, alternate_decoded),
            ("alternate", alternate_raw, alternate_decoded, original_raw, original_decoded),
        ):
            condition_metrics[condition] = {}
            aggregate_heatmaps[condition] = {}
            for direction, archive_key in (
                ("mask_query_to_source", "mask_query_to_source_heatmaps"),
                ("source_query_to_mask", "source_query_to_mask_heatmaps"),
            ):
                values = attention_metrics(
                    attentions[condition][archive_key],
                    target_raw,
                    target_decoded,
                    other_raw,
                    other_decoded,
                )
                aggregate_heatmaps[condition][direction] = values.pop("aggregate_heatmap")
                condition_metrics[condition][direction] = values
            condition_metrics[condition]["mean_raw_source_attention_mass"] = attentions[
                condition
            ]["mean_raw_source_attention_mass"]
            condition_metrics[condition]["mean_raw_mask_key_attention_mass"] = attentions[
                condition
            ]["mean_raw_mask_key_attention_mass"]

        original_metrics = condition_metrics["original"]
        alternate_metrics = condition_metrics["alternate"]
        original_heatmap = aggregate_heatmaps["original"]["mask_query_to_source"]
        alternate_heatmap = aggregate_heatmaps["alternate"]["mask_query_to_source"]
        original_reverse_heatmap = aggregate_heatmaps["original"]["source_query_to_mask"]
        alternate_reverse_heatmap = aggregate_heatmaps["alternate"]["source_query_to_mask"]

        mask_shift = np.asarray(centroid(resize_mask(alternate_decoded, original_heatmap.shape))) - np.asarray(
            centroid(resize_mask(original_decoded, original_heatmap.shape))
        )
        direction_shifts = {}
        for direction in ("mask_query_to_source", "source_query_to_mask"):
            attention_shift = np.asarray(
                alternate_metrics[direction]["aggregate_attention_centroid_xy"]
            ) - np.asarray(original_metrics[direction]["aggregate_attention_centroid_xy"])
            original_direction_heatmap = aggregate_heatmaps["original"][direction]
            alternate_direction_heatmap = aggregate_heatmaps["alternate"][direction]
            original_decoded_grid = resize_mask(
                original_decoded, original_direction_heatmap.shape
            )
            alternate_decoded_grid = resize_mask(
                alternate_decoded, original_direction_heatmap.shape
            )
            alternate_target_gain = float(
                alternate_direction_heatmap[alternate_decoded_grid].sum()
                - original_direction_heatmap[alternate_decoded_grid].sum()
            )
            original_target_loss = float(
                original_direction_heatmap[original_decoded_grid].sum()
                - alternate_direction_heatmap[original_decoded_grid].sum()
            )
            direction_shifts[direction] = {
                "attention_centroid_shift_xy": attention_shift.tolist(),
                "attention_mask_shift_cosine": cosine(attention_shift, mask_shift),
                "alternate_target_attention_gain": alternate_target_gain,
                "original_target_attention_loss": original_target_loss,
                "counterfactual_switch_score": alternate_target_gain
                + original_target_loss,
            }
        case_metrics = {
            "case_index": row["case_index"],
            "benchmark_id": row["benchmark_id"],
            "edit_type": row["edit_type"],
            "location_free_template": row["location_free_template"],
            "seed": records["original"]["seed"],
            "original": original_metrics,
            "alternate": alternate_metrics,
            "paired": {
                "both_decoded_routing_margins_positive": (
                    original_metrics["mask_query_to_source"]["decoded_routing_margin"] > 0
                    and alternate_metrics["mask_query_to_source"]["decoded_routing_margin"] > 0
                ),
                "mask_centroid_shift_xy": mask_shift.tolist(),
                "directions": direction_shifts,
            },
        }
        metric_rows.append(case_metrics)

        with Image.open(records["original"]["output"]) as handle:
            original_output = handle.convert("RGB")
        with Image.open(records["alternate"]["output"]) as handle:
            alternate_output = handle.convert("RGB")
        cells = [
            overlay_mask(source, original_raw),
            overlay_mask(source, alternate_raw, color=(37, 99, 235)),
            overlay_mask(source, original_decoded),
            overlay_mask(source, alternate_decoded, color=(37, 99, 235)),
            original_output,
            alternate_output,
            overlay_attention(source, original_heatmap),
            overlay_attention(source, alternate_heatmap),
            overlay_attention(source, original_reverse_heatmap),
            overlay_attention(source, alternate_reverse_heatmap),
            overlay_attention_delta(source, alternate_heatmap - original_heatmap),
            overlay_attention_delta(
                source, alternate_reverse_heatmap - original_reverse_heatmap
            ),
        ]
        labels = [
            "Benchmark mask A (raw)",
            "SAM2 alternate mask B (raw)",
            "Mask-token A decode",
            "Mask-token B decode",
            "Output conditioned on tokens A",
            "Output conditioned on tokens B",
            f"A mask-query -> source | margin={original_metrics['mask_query_to_source']['decoded_routing_margin']:.3f}",
            f"B mask-query -> source | margin={alternate_metrics['mask_query_to_source']['decoded_routing_margin']:.3f}",
            f"Source-query -> A mask-key | margin={original_metrics['source_query_to_mask']['decoded_routing_margin']:.3f}",
            f"Source-query -> B mask-key | margin={alternate_metrics['source_query_to_mask']['decoded_routing_margin']:.3f}",
            "B - A delta: mask-query -> source (red gains)",
            "B - A delta: source-query -> mask-key (red gains)",
        ]
        panel = make_panel(
            cells,
            labels,
            f"Case {row['case_index']:02d} | {row['benchmark_id']} | location-free prompt: "
            f"{row['location_free_template']}",
        )
        panel_path = args.experiment_root / "visualizations" / f"case_{int(row['case_index']):02d}.jpg"
        panel_path.parent.mkdir(parents=True, exist_ok=True)
        panel.save(panel_path, quality=94)
        panel_paths.append(panel_path)
        case_metrics["visualization"] = str(panel_path.resolve())

    metrics_path = args.experiment_root / "analysis" / "metrics.jsonl"
    _atomic_write_jsonl(metrics_path, metric_rows)
    direction_summary = {}
    for direction in ("mask_query_to_source", "source_query_to_mask"):
        routing_margins = [
            condition[direction]["decoded_routing_margin"]
            for row in metric_rows
            for condition in (row["original"], row["alternate"])
        ]
        top_iou = [
            condition[direction]["decoded_target_top_area_iou_mean"]
            for row in metric_rows
            for condition in (row["original"], row["alternate"])
        ]
        top_iou_chance = [
            condition[direction]["decoded_target_top_area_iou_chance"]
            for row in metric_rows
            for condition in (row["original"], row["alternate"])
        ]
        top_iou_lift = [
            condition[direction]["decoded_target_top_area_iou_lift"]
            for row in metric_rows
            for condition in (row["original"], row["alternate"])
        ]
        density_routing_margins = [
            condition[direction]["decoded_density_routing_margin"]
            for row in metric_rows
            for condition in (row["original"], row["alternate"])
        ]
        shift_cosines = [
            row["paired"]["directions"][direction]["attention_mask_shift_cosine"]
            for row in metric_rows
            if row["paired"]["directions"][direction]["attention_mask_shift_cosine"]
            is not None
        ]
        direction_summary[direction] = {
            "conditions_with_positive_decoded_routing_margin": sum(
                margin > 0 for margin in routing_margins
            ),
            "total_conditions": len(routing_margins),
            "cases_with_both_decoded_routing_margins_positive": sum(
                row["original"][direction]["decoded_routing_margin"] > 0
                and row["alternate"][direction]["decoded_routing_margin"] > 0
                for row in metric_rows
            ),
            "mean_decoded_routing_margin": float(np.mean(routing_margins)),
            "conditions_with_positive_decoded_density_routing_margin": sum(
                margin > 0 for margin in density_routing_margins
            ),
            "mean_decoded_density_routing_margin": float(
                np.mean(density_routing_margins)
            ),
            "mean_decoded_target_top_area_iou": float(np.mean(top_iou)),
            "mean_decoded_target_top_area_iou_chance": float(
                np.mean(top_iou_chance)
            ),
            "mean_decoded_target_top_area_iou_lift": float(np.mean(top_iou_lift)),
            "mean_attention_mask_shift_cosine": float(np.mean(shift_cosines)),
            "cases_with_positive_counterfactual_switch_score": sum(
                row["paired"]["directions"][direction][
                    "counterfactual_switch_score"
                ]
                > 0
                for row in metric_rows
            ),
            "mean_counterfactual_switch_score": float(
                np.mean(
                    [
                        row["paired"]["directions"][direction][
                            "counterfactual_switch_score"
                        ]
                        for row in metric_rows
                    ]
                )
            ),
        }
    report = {
        "status": "complete",
        "protocol": (
            "Paired edit_umt inference with identical source, location-free text, seed, and "
            "generation settings; only the four SAMTok mask tokens change. Both directions of "
            "the exact post-RoPE joint-attention probability are retained: mask-token query to "
            "source-image key, and source-image query to mask-token key. Spatial maps are "
            "normalized within the source and averaged equally over 4 tokens, 24 heads, "
            "selected layers, and selected denoising steps."
        ),
        "num_cases": len(rows),
        "num_inference_conditions": 2 * len(rows),
        "num_generated_images": 2 * len(rows),
        "num_attention_maps": sum(
            row[condition][direction]["num_layer_step_maps"]
            for row in metric_rows
            for condition in ("original", "alternate")
            for direction in ("mask_query_to_source", "source_query_to_mask")
        ),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "probe": {
            "zero_based_dit_layers": [5, 15, 30, 45, 59],
            "zero_based_denoising_steps": [0, 10, 20, 30, 39],
            "attention_heads": 24,
            "mask_tokens": 4,
        },
        "metrics": str(metrics_path.resolve()),
        "visualizations": [str(path.resolve()) for path in panel_paths],
        "summary": direction_summary,
        "interpretation_limits": [
            "Attention is descriptive evidence, not by itself a causal proof of editing behavior.",
            "The primary overlap uses codec-decoded masks because those are the regions represented by the four tokens; raw-mask overlap is also retained.",
            "Top-area IoU thresholds each heatmap to the same latent-cell area as its target, avoiding an arbitrary global heatmap threshold.",
            "Mass routing margin is area-dependent, so density-normalized routing margin and an equal-area chance IoU baseline are also reported.",
            "Generated images have no counterfactual ground truth and are intended for paired visual inspection.",
        ],
    }
    report_path = args.experiment_root / "analysis" / "report.json"
    _atomic_write_json(report_path, report)

    images = [Image.open(path).convert("RGB") for path in panel_paths]
    overview = Image.new(
        "RGB", (max(image.width for image in images), sum(image.height for image in images)), "white"
    )
    top = 0
    for image in images:
        overview.paste(image, (0, top))
        top += image.height
    overview_path = args.experiment_root / "visualizations" / "overview.jpg"
    overview.save(overview_path, quality=92)
    report["overview"] = str(overview_path.resolve())
    _atomic_write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
