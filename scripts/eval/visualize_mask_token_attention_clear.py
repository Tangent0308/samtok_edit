#!/usr/bin/env python3
"""Render high-contrast, high-resolution DiT mask-token attention diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "eval"))

from run_eval import (  # noqa: E402
    _atomic_write_json,
    _fit_panel_cell,
    _panel_font,
    _verify_image,
    _wrap_panel_text,
)
from summarize_mask_token_interpretability import (  # noqa: E402
    read_mask,
    read_rows,
)


DEFAULT_EXPERIMENT_ROOT = Path(
    "/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/"
    "crispedit_refined/interpretability/dit_mask_token_counterfactual"
)
A_COLOR = np.asarray([255, 55, 55], dtype=np.float32)
B_COLOR = np.asarray([0, 220, 255], dtype=np.float32)


def resize_float(values: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(np.asarray(values, dtype=np.float32), mode="F")
    return np.asarray(image.resize(size, Image.Resampling.BILINEAR), dtype=np.float32)


def resize_bool(values: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(np.asarray(values, dtype=np.uint8) * 255)
    return np.asarray(image.resize(size, Image.Resampling.NEAREST)) > 0


def interpolate_colors(values: np.ndarray, stops: list[tuple[int, int, int]]) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float32), 0, 1)
    palette = np.asarray(stops, dtype=np.float32)
    position = values * (len(palette) - 1)
    lower = np.floor(position).astype(np.int64)
    upper = np.minimum(lower + 1, len(palette) - 1)
    fraction = (position - lower)[..., None]
    return palette[lower] * (1 - fraction) + palette[upper] * fraction


def sequential_colors(values: np.ndarray) -> np.ndarray:
    # Near-black -> purple -> red -> orange -> yellow -> white.
    return interpolate_colors(
        values,
        [
            (4, 7, 18),
            (45, 15, 92),
            (145, 20, 115),
            (225, 55, 45),
            (252, 165, 28),
            (255, 247, 170),
            (255, 255, 255),
        ],
    )


def attention_scale(*heatmap_groups: np.ndarray) -> float:
    values = np.concatenate(
        [np.asarray(group, dtype=np.float32).reshape(-1) for group in heatmap_groups]
    )
    grid_area = int(np.asarray(heatmap_groups[0]).shape[-2] * np.asarray(heatmap_groups[0]).shape[-1])
    enrichment = values * grid_area
    return max(2.0, float(np.percentile(enrichment, 99.5)))


def boundary(mask: np.ndarray, width: int) -> np.ndarray:
    size = max(3, 2 * int(width) + 1)
    if size % 2 == 0:
        size += 1
    image = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255)
    outer = np.asarray(image.filter(ImageFilter.MaxFilter(size))) > 0
    inner = np.asarray(image.filter(ImageFilter.MinFilter(size))) > 0
    return np.logical_and(outer, np.logical_not(inner))


def paint_boundary(
    image: np.ndarray,
    mask: np.ndarray,
    color: np.ndarray,
    width: int,
) -> None:
    image[boundary(mask, width)] = color


def add_sequential_colorbar(
    canvas: Image.Image,
    low: float,
    high: float,
    label: str = "attention / uniform",
) -> None:
    width, height = canvas.size
    left, right = 24, width - 24
    top, bottom = height - 29, height - 15
    ramp = np.linspace(0, 1, max(2, right - left), dtype=np.float32)[None, :]
    colors = sequential_colors(ramp)
    colors = np.repeat(colors, bottom - top, axis=0).astype(np.uint8)
    canvas.paste(Image.fromarray(colors), (left, top))
    draw = ImageDraw.Draw(canvas)
    font = _panel_font(13, bold=True)
    draw.text((left, 4), f"{label}: {low:.1f}x", font=font, fill="white", stroke_width=2, stroke_fill="black")
    text = f"{high:.1f}x"
    box = draw.textbbox((0, 0), text, font=font)
    draw.text(
        (right - (box[2] - box[0]), 4),
        text,
        font=font,
        fill="white",
        stroke_width=2,
        stroke_fill="black",
    )


def render_attention_focus(
    source: Image.Image,
    heatmap: np.ndarray,
    scale_high: float,
    size: tuple[int, int],
) -> Image.Image:
    """Render attention alone over a deliberately dark source-image reference."""

    content_height = size[1] - 42
    content_size = (size[0], content_height)
    source_array = np.asarray(
        source.convert("RGB").resize(content_size, Image.Resampling.LANCZOS),
        dtype=np.float32,
    )
    enrichment = resize_float(
        np.asarray(heatmap, dtype=np.float32) * heatmap.shape[-2] * heatmap.shape[-1],
        content_size,
    )
    strength = np.clip((enrichment - 0.5) / max(scale_high - 0.5, 1e-6), 0, 1) ** 0.52
    colors = sequential_colors(strength)
    glow = strength[..., None]
    # Low-attention pixels retain only a dark scene reference.  High-attention
    # pixels approach the saturated magma color, so hotspots remain visible
    # without any mask outline or overlay.
    dark_source = source_array * 0.20
    output = dark_source * (1 - 0.55 * glow) + colors * (0.10 + 0.90 * glow)
    canvas = Image.new("RGB", size, (5, 8, 15))
    canvas.paste(Image.fromarray(np.clip(output, 0, 255).astype(np.uint8)), (0, 0))
    add_sequential_colorbar(canvas, 0.5, scale_high)
    return canvas


def render_mask_overlay(
    source: Image.Image,
    mask: np.ndarray,
    color: np.ndarray,
    size: tuple[int, int],
) -> Image.Image:
    array = np.asarray(source.convert("RGB").resize(size, Image.Resampling.LANCZOS), dtype=np.float32)
    resized = resize_bool(mask, size)
    array[resized] = array[resized] * 0.25 + color * 0.75
    paint_boundary(array, resized, color, 5)
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def make_labeled_panel(
    cells: list[Image.Image],
    labels: list[str],
    heading: str,
    columns: int = 4,
    cell_size: tuple[int, int] = (480, 480),
) -> Image.Image:
    rows = math.ceil(len(cells) / columns)
    width = columns * cell_size[0]
    heading_font = _panel_font(27, bold=True)
    label_font = _panel_font(20, bold=True)
    scratch = ImageDraw.Draw(Image.new("RGB", (width, 1), "white"))
    heading_lines = []
    for paragraph in heading.splitlines():
        heading_lines.extend(
            _wrap_panel_text(scratch, paragraph, heading_font, width - 40)
        )
    header_height = max(100, 24 + 38 * len(heading_lines))
    label_height = 78
    panel = Image.new(
        "RGB",
        (width, header_height + rows * (label_height + cell_size[1])),
        "white",
    )
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, width, header_height), fill=(13, 24, 42))
    for index, line in enumerate(heading_lines):
        draw.text((20, 16 + 38 * index), line, font=heading_font, fill="white")
    for index, (cell, label) in enumerate(zip(cells, labels)):
        row, column = divmod(index, columns)
        left = column * cell_size[0]
        top = header_height + row * (label_height + cell_size[1])
        label_lines = _wrap_panel_text(draw, label, label_font, cell_size[0] - 20)[:3]
        for line_index, line in enumerate(label_lines):
            box = draw.textbbox((0, 0), line, font=label_font)
            draw.text(
                (left + max(10, (cell_size[0] - (box[2] - box[0])) // 2), top + 7 + 24 * line_index),
                line,
                font=label_font,
                fill=(15, 23, 42),
            )
        panel.paste(_fit_panel_cell(cell.convert("RGB"), cell_size), (left, top + label_height))
    return panel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--output_dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.experiment_root / "visualizations_clear"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_rows(args.experiment_root / "data" / "interventions.jsonl")
    summary_paths = []
    prompt_records = []
    cell_size = (420, 420)
    for row in manifest:
        case_index = int(row["case_index"])
        case_root = args.experiment_root / "runs" / f"case_{case_index:02d}_{row['benchmark_id']}"
        with Image.open(row["source_image"]) as handle:
            source = handle.convert("RGB")
        raw_a = read_mask(Path(row["original_mask"]), source.size)
        raw_b = read_mask(Path(row["alternate_mask"]), source.size)
        decoded_a = read_mask(Path(row["original_token_decode"]), source.size)
        decoded_b = read_mask(Path(row["alternate_token_decode"]), source.size)
        with Image.open(case_root / "original_output.png") as handle:
            output_a = handle.convert("RGB")
        with Image.open(case_root / "alternate_output.png") as handle:
            output_b = handle.convert("RGB")
        archive_a = np.load(case_root / "original_attention.npz")
        archive_b = np.load(case_root / "alternate_attention.npz")

        maps_a = archive_a["mask_query_to_source_heatmaps"]
        maps_b = archive_b["mask_query_to_source_heatmaps"]
        m2s_a, m2s_b = maps_a.mean(axis=0), maps_b.mean(axis=0)
        shared_scale = attention_scale(maps_a, maps_b)
        cells = [
            source,
            render_mask_overlay(source, raw_a, A_COLOR, cell_size),
            render_mask_overlay(source, decoded_a, A_COLOR, cell_size),
            render_mask_overlay(source, raw_b, B_COLOR, cell_size),
            render_mask_overlay(source, decoded_b, B_COLOR, cell_size),
            render_attention_focus(
                source, m2s_a, shared_scale, cell_size
            ),
            render_attention_focus(
                source, m2s_b, shared_scale, cell_size
            ),
            output_a,
            output_b,
        ]
        labels = [
            "Original source image",
            "Original benchmark mask A overlay",
            "Mask-token A decoded mask overlay",
            "SAM2-modified mask B overlay",
            "Mask-token B decoded mask overlay",
            "Mask-token A attention region | mask-query -> source",
            "Mask-token B attention region | mask-query -> source",
            "Output using mask tokens A",
            "Output using mask tokens B",
        ]
        summary = make_labeled_panel(
            cells,
            labels,
            (
                f"Case {case_index:02d} | {row['benchmark_id']} | location-free paired intervention\n"
                f"Full conditioning prompt A encoded for DiT: {row['original_prompt']}\n"
                f"Full conditioning prompt B encoded for DiT: {row['alternate_prompt']}\n"
                "Attention panels contain no mask overlay; yellow/white means stronger "
                "mask-token attention; A/B share one scale"
            ),
            columns=9,
            cell_size=cell_size,
        )
        summary_path = output_dir / f"case_{case_index:02d}_summary.jpg"
        summary.save(summary_path, quality=95, subsampling=0)
        summary_paths.append(summary_path)
        prompt_records.append(
            {
                "case_index": case_index,
                "benchmark_id": row["benchmark_id"],
                "full_conditioning_prompt_a": row["original_prompt"],
                "full_conditioning_prompt_b": row["alternate_prompt"],
                "mask_span_a": row["original_mask_span"],
                "mask_span_b": row["alternate_mask_span"],
                "shared_attention_scale_high_x_uniform": shared_scale,
            }
        )
        print(f"[clear-visualization] case={case_index} nine-panel summary complete", flush=True)

    summaries = [Image.open(path).convert("RGB") for path in summary_paths]
    overview = Image.new(
        "RGB",
        (max(image.width for image in summaries), sum(image.height for image in summaries)),
        "white",
    )
    top = 0
    for image in summaries:
        overview.paste(image, (0, top))
        top += image.height
    overview_path = output_dir / "overview_clear.jpg"
    overview.save(overview_path, quality=93, subsampling=0)

    for index, path in enumerate(summary_paths + [overview_path], start=1):
        _verify_image(path, "clear attention visualization", index)
    report = {
        "status": "complete",
        "num_cases": len(manifest),
        "num_summary_panels": len(summary_paths),
        "summary_panels": [str(path.resolve()) for path in summary_paths],
        "overview": str(overview_path.resolve()),
        "prompts": prompt_records,
        "rendering": {
            "attention_value": "source-space attention probability divided by uniform probability",
            "attention_scale": "shared between A and B within each case and direction",
            "summary_layout": "source, raw mask A overlay, decoded mask A overlay, raw mask B overlay, decoded mask B overlay, mask-token A attention, mask-token B attention, output A, output B",
            "summary_attention": "darkened source with high-contrast magma intensity and no mask overlay or boundary",
        },
    }
    report_path = output_dir / "report.json"
    _atomic_write_json(report_path, report)

    # Replace only the obsolete visualization products owned by this
    # experiment.  Raw attention archives, generated outputs, and metrics are
    # deliberately untouched.
    for row in manifest:
        case_index = int(row["case_index"])
        for direction in ("mask_query_to_source", "source_query_to_mask"):
            obsolete_grid = output_dir / f"case_{case_index:02d}_{direction}_grid.jpg"
            obsolete_grid.unlink(missing_ok=True)
    obsolete_dir = args.experiment_root / "visualizations"
    for row in manifest:
        (obsolete_dir / f"case_{int(row['case_index']):02d}.jpg").unlink(missing_ok=True)
    (obsolete_dir / "overview.jpg").unlink(missing_ok=True)
    if obsolete_dir.is_dir() and not any(obsolete_dir.iterdir()):
        obsolete_dir.rmdir()

    analysis_report_path = args.experiment_root / "analysis" / "report.json"
    if analysis_report_path.is_file():
        analysis_report = json.loads(analysis_report_path.read_text(encoding="utf-8"))
        analysis_report["visualizations"] = [
            str(path.resolve()) for path in summary_paths
        ]
        analysis_report["overview"] = str(overview_path.resolve())
        analysis_report["visualization_report"] = str(report_path.resolve())
        _atomic_write_json(analysis_report_path, analysis_report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
