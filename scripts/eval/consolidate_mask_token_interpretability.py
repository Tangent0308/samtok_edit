#!/usr/bin/env python3
"""Build one audited 12-case entry point from completed interpretability jobs.

The source experiment directories remain the reproducible owners of inputs,
generated images, attention archives, and sidecars.  This script only creates
a compact index: unified manifests/metrics, symlinks to the final nine-panel
figures, one vertical overview, and a combined report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import fmean

from PIL import Image, ImageDraw, ImageFont


DEFAULT_INTERPRETABILITY_ROOT = Path(
    "/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/"
    "crispedit_refined/interpretability"
)
DEFAULT_SOURCE_ROOTS = (
    DEFAULT_INTERPRETABILITY_ROOT / "dit_mask_token_counterfactual",
    DEFAULT_INTERPRETABILITY_ROOT / "dit_mask_token_counterfactual_additional",
)
DEFAULT_OUTPUT_ROOT = DEFAULT_INTERPRETABILITY_ROOT / "dit_mask_token_counterfactual_12case"

CURATED_CATEGORIES = {
    "cb_train-00002-of-00007_0206": "zebra",
    "cb_train-00002-of-00007_0246": "elephant",
    "cb_train-00005-of-00007_0348": "cat",
}
DIRECTIONS = ("mask_query_to_source", "source_query_to_mask")
UNIFIED_BANNER_HEIGHT = 72


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_symlink(target: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    temporary = link.with_name(link.name + ".tmp")
    if temporary.is_symlink() or temporary.exists():
        temporary.unlink()
    temporary.symlink_to(target.resolve())
    os.replace(temporary, link)


def summarize_metrics(metric_rows: list[dict]) -> dict:
    """Recompute the same aggregate fields used by the per-job summarizer."""
    result = {}
    for direction in DIRECTIONS:
        conditions = [
            row[condition][direction]
            for row in metric_rows
            for condition in ("original", "alternate")
        ]
        routing = [item["decoded_routing_margin"] for item in conditions]
        density = [item["decoded_density_routing_margin"] for item in conditions]
        top_iou = [item["decoded_target_top_area_iou_mean"] for item in conditions]
        chance = [item["decoded_target_top_area_iou_chance"] for item in conditions]
        lift = [item["decoded_target_top_area_iou_lift"] for item in conditions]
        cosines = [
            row["paired"]["directions"][direction]["attention_mask_shift_cosine"]
            for row in metric_rows
        ]
        cosines = [value for value in cosines if value is not None]
        switches = [
            row["paired"]["directions"][direction]["counterfactual_switch_score"]
            for row in metric_rows
        ]
        result[direction] = {
            "conditions_with_positive_decoded_routing_margin": sum(
                value > 0 for value in routing
            ),
            "total_conditions": len(routing),
            "cases_with_both_decoded_routing_margins_positive": sum(
                row["original"][direction]["decoded_routing_margin"] > 0
                and row["alternate"][direction]["decoded_routing_margin"] > 0
                for row in metric_rows
            ),
            "mean_decoded_routing_margin": fmean(routing),
            "conditions_with_positive_decoded_density_routing_margin": sum(
                value > 0 for value in density
            ),
            "mean_decoded_density_routing_margin": fmean(density),
            "mean_decoded_target_top_area_iou": fmean(top_iou),
            "mean_decoded_target_top_area_iou_chance": fmean(chance),
            "mean_decoded_target_top_area_iou_lift": fmean(lift),
            "mean_attention_mask_shift_cosine": fmean(cosines),
            "cases_with_positive_counterfactual_switch_score": sum(
                value > 0 for value in switches
            ),
            "mean_counterfactual_switch_score": fmean(switches),
        }
    return result


def validate_source(root: Path) -> tuple[list[dict], list[dict], dict, list[Path]]:
    manifest_path = root / "data" / "interventions.jsonl"
    metrics_path = root / "analysis" / "metrics.jsonl"
    report_path = root / "analysis" / "report.json"
    visual_report_path = root / "visualizations_clear" / "report.json"
    for path in (manifest_path, metrics_path, report_path, visual_report_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = read_jsonl(manifest_path)
    metrics = read_jsonl(metrics_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    visual_report = json.loads(visual_report_path.read_text(encoding="utf-8"))
    panels = [Path(path) for path in visual_report["summary_panels"]]
    count = len(manifest)
    if report.get("status") != "complete" or visual_report.get("status") != "complete":
        raise ValueError(f"Incomplete source experiment: {root}")
    if not (count == len(metrics) == len(panels) == report.get("num_cases")):
        raise ValueError(f"Source count mismatch: {root}")
    for intervention, metric, panel in zip(manifest, metrics, panels):
        if intervention["benchmark_id"] != metric["benchmark_id"]:
            raise ValueError(f"Manifest/metric mismatch: {root}")
        if not panel.is_file() or panel.stat().st_size == 0:
            raise FileNotFoundError(panel)
        with Image.open(panel) as image:
            if image.size != (3780, 674):
                raise ValueError(f"Unexpected nine-panel geometry: {panel}: {image.size}")
    return manifest, metrics, report, panels


def unified_case_label(case_index: int, total_cases: int, row: dict) -> str:
    category = str(row["semantic_category"]).upper()
    benchmark_id = row["benchmark_id"]
    return (
        f"UNIFIED CASE {case_index:02d} / {total_cases - 1:02d}"
        f"   |   {category}   |   {benchmark_id}"
    )


def overview_font(size: int = 34):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def build_overview(panel_paths: list[Path], rows: list[dict], output_path: Path) -> None:
    if len(panel_paths) != len(rows):
        raise ValueError(
            f"Overview panel/metadata mismatch: {len(panel_paths)} != {len(rows)}"
        )
    images = []
    try:
        for path in panel_paths:
            images.append(Image.open(path).convert("RGB"))
        overview = Image.new(
            "RGB",
            (
                max(image.width for image in images),
                sum(image.height + UNIFIED_BANNER_HEIGHT for image in images),
            ),
            "white",
        )
        draw = ImageDraw.Draw(overview)
        font = overview_font()
        top = 0
        for case_index, (image, row) in enumerate(zip(images, rows)):
            draw.rectangle(
                (0, top, overview.width, top + UNIFIED_BANNER_HEIGHT),
                fill=(8, 24, 48),
            )
            draw.rectangle(
                (
                    0,
                    top + UNIFIED_BANNER_HEIGHT - 6,
                    overview.width,
                    top + UNIFIED_BANNER_HEIGHT,
                ),
                fill=(0, 220, 255),
            )
            draw.text(
                (26, top + 13),
                unified_case_label(case_index, len(rows), row),
                font=font,
                fill=(255, 255, 255),
            )
            panel_top = top + UNIFIED_BANNER_HEIGHT
            overview.paste(image, (0, panel_top))
            top = panel_top + image.height
        temporary = output_path.with_suffix(".tmp.jpg")
        overview.save(temporary, quality=90, optimize=True)
        temporary.replace(output_path)
    finally:
        for image in images:
            image.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source_root",
        action="append",
        type=Path,
        dest="source_roots",
        help="Completed source experiment root; pass once per source job.",
    )
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_roots = tuple(args.source_roots or DEFAULT_SOURCE_ROOTS)
    output_root = args.output_root
    visual_root = output_root / "visualizations"
    visual_root.mkdir(parents=True, exist_ok=True)

    combined_manifest: list[dict] = []
    combined_metrics: list[dict] = []
    source_reports = []
    unified_panels: list[Path] = []
    seen_ids: set[str] = set()
    seen_categories: set[str] = set()

    for source_root in source_roots:
        manifest, metrics, report, panels = validate_source(source_root)
        source_reports.append(
            {
                "root": str(source_root.resolve()),
                "num_cases": len(manifest),
                "manifest_sha256": report["manifest_sha256"],
                "report": str((source_root / "analysis" / "report.json").resolve()),
            }
        )
        for intervention, metric, source_panel in zip(manifest, metrics, panels):
            benchmark_id = intervention["benchmark_id"]
            category = intervention.get("semantic_category") or CURATED_CATEGORIES.get(
                benchmark_id
            )
            if not category:
                raise ValueError(f"Missing semantic category for {benchmark_id}")
            if benchmark_id in seen_ids or category in seen_categories:
                raise ValueError(f"Duplicate benchmark/category: {benchmark_id}/{category}")
            seen_ids.add(benchmark_id)
            seen_categories.add(category)

            unified_index = len(combined_manifest)
            panel = visual_root / f"case_{unified_index:02d}_{category}.jpg"
            atomic_symlink(source_panel, panel)
            unified_panels.append(panel)

            intervention = dict(intervention)
            intervention.update(
                {
                    "case_index": unified_index,
                    "semantic_category": category,
                    "source_case_index": metric["case_index"],
                    "source_experiment_root": str(source_root.resolve()),
                    "summary_panel": str(panel),
                }
            )
            combined_manifest.append(intervention)

            metric = dict(metric)
            metric.update(
                {
                    "case_index": unified_index,
                    "semantic_category": category,
                    "source_case_index": metric["case_index"],
                    "source_experiment_root": str(source_root.resolve()),
                    "visualization": str(panel),
                }
            )
            combined_metrics.append(metric)

    if len(combined_manifest) != 12:
        raise ValueError(f"Expected exactly 12 cases, found {len(combined_manifest)}")

    expected_names = {path.name for path in unified_panels}
    for stale in visual_root.glob("case_*.jpg"):
        if stale.name not in expected_names:
            stale.unlink()

    manifest_path = output_root / "manifest.jsonl"
    metrics_path = output_root / "metrics.jsonl"
    overview_path = visual_root / "overview_12case.jpg"
    report_path = output_root / "report.json"
    atomic_write_jsonl(manifest_path, combined_manifest)
    atomic_write_jsonl(metrics_path, combined_metrics)
    build_overview(unified_panels, combined_manifest, overview_path)

    num_attention_maps = sum(
        row[condition][direction]["num_layer_step_maps"]
        for row in combined_metrics
        for condition in ("original", "alternate")
        for direction in DIRECTIONS
    )
    report = {
        "status": "complete",
        "protocol": (
            "Unified index over 12 paired edit_umt mask-token interventions. Source, "
            "location-free text, seed, checkpoint, and generation settings are fixed within "
            "each pair; only the canonical four-token mask span changes."
        ),
        "num_cases": len(combined_manifest),
        "num_inference_conditions": 2 * len(combined_manifest),
        "num_generated_images": 2 * len(combined_manifest),
        "num_attention_maps": num_attention_maps,
        "categories": [row["semantic_category"] for row in combined_manifest],
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "metrics": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
        "summary": summarize_metrics(combined_metrics),
        "summary_panels": [str(path) for path in unified_panels],
        "overview": str(overview_path),
        "overview_rendering": {
            "rows": len(combined_manifest),
            "banner_height_px": UNIFIED_BANNER_HEIGHT,
            "banner_fields": [
                "unified_case_index",
                "semantic_category",
                "benchmark_id",
            ],
            "note": (
                "Every overview row has a canonical UNIFIED CASE 00-11 banner. "
                "The source panel's internal case number is source-job provenance only."
            ),
        },
        "source_experiments": source_reports,
        "storage": (
            "The unified case panels are symlinks. Original inference records, generated "
            "images, attention NPZ archives, masks, and logs remain in source_experiments."
        ),
    }
    atomic_write_json(report_path, report)

    readme = output_root / "README.md"
    temporary = readme.with_suffix(".md.tmp")
    temporary.write_text(
        "# DiT mask-token counterfactual analysis: unified 12-case index\n\n"
        "This directory is the canonical result entry point. It combines all 12 reviewed "
        "same-class multi-instance cases without duplicating the original attention archives.\n\n"
        "- `visualizations/overview_12case.jpg`: all 12 final nine-panel summaries; every "
        "row starts with a canonical `UNIFIED CASE 00-11` banner, category, and benchmark ID.\n"
        "- `visualizations/case_*.jpg`: symlinks to each final full-resolution summary.\n"
        "- `manifest.jsonl`: unified intervention and provenance index.\n"
        "- `metrics.jsonl`: unified per-case attention metrics.\n"
        "- `report.json`: aggregate metrics and source experiment roots.\n\n"
        "The source roots listed in `report.json` remain authoritative for masks, generated "
        "outputs, inference sidecars, attention NPZ files, and logs.\n",
        encoding="utf-8",
    )
    temporary.replace(readme)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
