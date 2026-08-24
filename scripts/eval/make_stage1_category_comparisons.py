#!/usr/bin/env python3
"""Build model-free, per-category Stage 1 evaluation comparison sheets.

The final-result sheets read the completed S1-S5 PNGs directly. The mask sheets
use the raw CrispEdit raster mask and the independent GT/online decoded overlay
cells already produced by analyze_stage1_cot_masks.py. No model is loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "eval"))

from analyze_stage1_cot_masks import (  # noqa: E402
    fit_cell,
    font,
    overlay,
    parse_cot,
    raw_annotation_mask,
    wrap_text,
    write_json,
)


DEFAULT_ROOT = Path(
    "/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/"
    "stage1_evaluation/five_settings"
)
DEFAULT_RAW_MASK_DIR = Path(
    "/mnt/bn/strategy-mllm-train/user/tanyue/datasets/"
    "CrispEdit-2M-mask-parquet-101697"
)
ONLINE_SETTING = "s4_stage1_te_online_cot"
SETTINGS = (
    "s1_qwen2511_stock",
    "s2_samtok_initial_direct",
    "s3_stage1_te_direct",
    "s4_stage1_te_online_cot",
    "s5_stage1_te_gt_cot",
)
FINAL_LABELS = (
    "Original",
    "GT edited image",
    "S1 Stock 2511",
    "S2 Initial direct",
    "S3 Stage-1 direct",
    "S4 Online CoT",
    "S5 GT CoT",
)
MASK_LABELS = (
    "GT raster mask (blue)",
    "GT token decode (green)",
    "Online token decode (red)",
)
FINAL_CELL_SIZE = (280, 280)
MASK_CELL_SIZE = (320, 320)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--raw_mask_dir", type=Path, default=DEFAULT_RAW_MASK_DIR)
    parser.add_argument(
        "--decoded_panel_dir",
        type=Path,
        default=None,
        help="default: ROOT/analysis/decoded_mask_overlap/panels",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="default: ROOT/analysis/category_comparisons",
    )
    parser.add_argument("--jpeg_quality", type=int, default=92)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_online_records(root: Path) -> list[dict]:
    sidecars = sorted((root / ONLINE_SETTING).glob("[0-9][0-9][0-9][0-9].json"))
    records = [json.loads(path.read_text(encoding="utf-8")) for path in sidecars]
    indices = [int(record["metadata_index"]) for record in records]
    if len(records) != 64 or indices != list(range(64)):
        raise ValueError(
            f"Expected metadata indices 0..63 in {root / ONLINE_SETTING}, got {indices}"
        )
    for record in records:
        index = int(record["metadata_index"])
        expected = root / ONLINE_SETTING / f"{index:04d}.png"
        if Path(record["output"]).resolve() != expected.resolve():
            raise ValueError(f"Online output path mismatch at index={index}")
    return records


def load_mask_metrics(root: Path) -> dict[int, dict]:
    path = root / "analysis" / "decoded_mask_overlap" / "report.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(row["metadata_index"]): row for row in payload["samples"]}


def load_fitted(path: Path, size: tuple[int, int]) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        image.load()
        return fit_cell(image, size)


def add_empty_badge(image: Image.Image, text: str) -> Image.Image:
    image = image.copy()
    draw = ImageDraw.Draw(image)
    badge_font = font(16, bold=True)
    bbox = draw.textbbox((0, 0), text, font=badge_font)
    width = bbox[2] - bbox[0] + 20
    height = bbox[3] - bbox[1] + 14
    draw.rounded_rectangle(
        (10, 10, 10 + width, 10 + height),
        radius=7,
        fill=(20, 28, 38),
        outline="white",
        width=2,
    )
    draw.text((20, 16), text, font=badge_font, fill="white")
    return image


def crop_decoded_mask_cells(panel: Image.Image) -> tuple[Image.Image, Image.Image]:
    """Return independent GT and online cells from an audited five-cell panel."""

    panel = panel.convert("RGB")
    expected_width = MASK_CELL_SIZE[0] * 5
    if panel.width != expected_width or panel.height < MASK_CELL_SIZE[1]:
        raise ValueError(
            f"Unexpected decoded panel size {panel.size}; expected width={expected_width}"
        )
    top = panel.height - MASK_CELL_SIZE[1]
    online = panel.crop(
        (
            MASK_CELL_SIZE[0],
            top,
            MASK_CELL_SIZE[0] * 2,
            panel.height,
        )
    )
    gt = panel.crop(
        (
            MASK_CELL_SIZE[0] * 2,
            top,
            MASK_CELL_SIZE[0] * 3,
            panel.height,
        )
    )
    return gt, online


def centered_text(draw, box, text: str, text_font, fill="black"):
    left, top, right, bottom = box
    lines = wrap_text(draw, text, text_font, right - left - 12)
    line_height = max(20, text_font.size + 4) if hasattr(text_font, "size") else 22
    total_height = line_height * len(lines)
    y = top + max(0, (bottom - top - total_height) // 2)
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=text_font)
        width = bbox[2] - bbox[0]
        draw.text(
            (left + max(6, (right - left - width) // 2), y),
            line,
            font=text_font,
            fill=fill,
        )
        y += line_height


def make_sheet(
    *,
    category: str,
    records: list[dict],
    labels: tuple[str, ...],
    cell_size: tuple[int, int],
    cell_builder,
    path: Path,
    quality: int,
):
    width = cell_size[0] * len(labels)
    title_height = 64
    column_height = 56
    row_font = font(18, bold=True)
    title_font = font(25, bold=True)
    label_font = font(17, bold=True)
    scratch = ImageDraw.Draw(Image.new("RGB", (width, 1)))
    row_specs = []
    for record in records:
        heading = (
            f"#{int(record['metadata_index']):04d} | "
            f"Instruction: {record['prompt'].strip()}"
        )
        lines = wrap_text(scratch, heading, row_font, width - 32)
        heading_height = max(48, 14 + 25 * len(lines))
        row_specs.append((record, lines, heading_height))
    height = title_height + column_height + sum(
        heading_height + cell_size[1] + 8
        for _, _, heading_height in row_specs
    )
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.rectangle((0, 0, width, title_height), fill=(17, 31, 49))
    centered_text(
        draw,
        (0, 0, width, title_height),
        f"Stage 1 evaluation | category: {category} | samples: {len(records)}",
        title_font,
        fill="white",
    )
    for column, label in enumerate(labels):
        left = column * cell_size[0]
        fill = (231, 237, 244) if column % 2 == 0 else (241, 245, 249)
        draw.rectangle(
            (left, title_height, left + cell_size[0], title_height + column_height),
            fill=fill,
        )
        centered_text(
            draw,
            (left, title_height, left + cell_size[0], title_height + column_height),
            label,
            label_font,
        )

    top = title_height + column_height
    for row_number, (record, lines, heading_height) in enumerate(row_specs):
        row_fill = (248, 250, 252) if row_number % 2 == 0 else (237, 242, 247)
        draw.rectangle((0, top, width, top + heading_height), fill=row_fill)
        for line_number, line in enumerate(lines):
            draw.text(
                (16, top + 7 + 25 * line_number),
                line,
                font=row_font,
                fill=(18, 28, 40),
            )
        cells = cell_builder(record)
        if len(cells) != len(labels):
            raise ValueError(
                f"Cell count mismatch for index={record['metadata_index']}: "
                f"{len(cells)} != {len(labels)}"
            )
        image_top = top + heading_height
        for column, cell in enumerate(cells):
            if cell.size != cell_size:
                raise ValueError(
                    f"Cell size mismatch for index={record['metadata_index']}: "
                    f"{cell.size} != {cell_size}"
                )
            left = column * cell_size[0]
            sheet.paste(cell, (left, image_top))
            draw.rectangle(
                (
                    left,
                    image_top,
                    left + cell_size[0] - 1,
                    image_top + cell_size[1] - 1,
                ),
                outline=(170, 179, 190),
                width=1,
            )
        top = image_top + cell_size[1] + 8

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    sheet.save(temporary, format="JPEG", quality=quality, subsampling=0)
    temporary.replace(path)
    return sheet.size


def main():
    args = parse_args()
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg_quality must be in [1, 100]")
    output_dir = args.output_dir or args.root / "analysis" / "category_comparisons"
    decoded_panel_dir = (
        args.decoded_panel_dir
        or args.root / "analysis" / "decoded_mask_overlap" / "panels"
    )
    records = load_online_records(args.root)
    metrics = load_mask_metrics(args.root)
    by_type: defaultdict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_type[record["provenance"]["edit_type"]].append(record)

    raw_mask_cache: dict[Path, dict[int, bytes]] = {}
    mask_audit: dict[int, dict] = {}

    def final_cells(record: dict) -> list[Image.Image]:
        index = int(record["metadata_index"])
        paths = [Path(record["source"]), Path(record["target"])] + [
            args.root / setting / f"{index:04d}.png" for setting in SETTINGS
        ]
        return [load_fitted(path, FINAL_CELL_SIZE) for path in paths]

    def mask_cells(record: dict) -> list[Image.Image]:
        index = int(record["metadata_index"])
        gt_items = parse_cot(record["gt_mt_cot"])
        online_items = parse_cot(record["conditioned_mt_cot"])
        with Image.open(record["source"]) as image:
            source = image.convert("RGB")
        raw_mask = raw_annotation_mask(
            record,
            source.size,
            args.raw_mask_dir,
            raw_mask_cache,
        )
        raw_cell = fit_cell(overlay(source, raw_mask, (60, 120, 255)), MASK_CELL_SIZE)
        if not raw_mask.any():
            raw_cell = add_empty_badge(raw_cell, "EMPTY RAW MASK")

        if gt_items:
            if not online_items:
                raise ValueError(f"Online CoT is empty for nonempty GT at index={index}")
            panel_path = decoded_panel_dir / f"{index:04d}.jpg"
            if not panel_path.is_file():
                raise FileNotFoundError(
                    f"Missing audited decoded panel for nonempty CoT: {panel_path}"
                )
            with Image.open(panel_path) as panel:
                gt_cell, online_cell = crop_decoded_mask_cells(panel)
            if index not in metrics:
                raise ValueError(f"Missing decoded mask metrics for index={index}")
        else:
            if online_items:
                raise ValueError(f"Online CoT is nonempty for empty GT at index={index}")
            clean_source = fit_cell(source, MASK_CELL_SIZE)
            gt_cell = add_empty_badge(clean_source, "EMPTY GT TOKEN MASK")
            online_cell = add_empty_badge(clean_source, "EMPTY ONLINE TOKEN MASK")
            if index in metrics:
                raise ValueError(f"Unexpected decoded mask metrics for empty GT index={index}")

        mask_audit[index] = {
            "gt_cot_nonempty": bool(gt_items),
            "online_cot_nonempty": bool(online_items),
            "raw_mask_nonempty": bool(raw_mask.any()),
            "decoded_panel": (
                str(decoded_panel_dir / f"{index:04d}.jpg") if gt_items else None
            ),
            "mask_iou": metrics.get(index, {}).get("mask_iou"),
        }
        return [raw_cell, gt_cell, online_cell]

    manifest = {
        "status": "complete",
        "root": str(args.root),
        "output_dir": str(output_dir),
        "generation": "model-free aggregation of completed S1-S5 outputs and audited decoded-mask panels",
        "final_columns": list(FINAL_LABELS),
        "mask_columns": list(MASK_LABELS),
        "mask_layout_note": (
            "The three masks are separate source-image overlay panels; no cell "
            "combines GT, GT-token decode, and online-token decode."
        ),
        "decoded_panel_dir": str(decoded_panel_dir),
        "raw_mask_dir": str(args.raw_mask_dir),
        "categories": {},
    }
    for category, category_records in sorted(by_type.items()):
        final_path = output_dir / f"{category}_final_results.jpg"
        mask_path = output_dir / f"{category}_mask_comparison.jpg"
        final_size = make_sheet(
            category=category,
            records=category_records,
            labels=FINAL_LABELS,
            cell_size=FINAL_CELL_SIZE,
            cell_builder=final_cells,
            path=final_path,
            quality=args.jpeg_quality,
        )
        mask_size = make_sheet(
            category=category,
            records=category_records,
            labels=MASK_LABELS,
            cell_size=MASK_CELL_SIZE,
            cell_builder=mask_cells,
            path=mask_path,
            quality=args.jpeg_quality,
        )
        indices = [int(record["metadata_index"]) for record in category_records]
        manifest["categories"][category] = {
            "count": len(indices),
            "metadata_indices": indices,
            "gt_cot_nonempty": sum(mask_audit[index]["gt_cot_nonempty"] for index in indices),
            "raw_mask_nonempty": sum(mask_audit[index]["raw_mask_nonempty"] for index in indices),
            "final_results": {
                "path": str(final_path),
                "size": list(final_size),
                "sha256": sha256_file(final_path),
            },
            "mask_comparison": {
                "path": str(mask_path),
                "size": list(mask_size),
                "sha256": sha256_file(mask_path),
            },
        }
        print(
            f"[category] {category}: count={len(indices)} "
            f"final={final_path.name} masks={mask_path.name}"
        )

    manifest["sample_mask_audit"] = {
        str(index): mask_audit[index] for index in sorted(mask_audit)
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k != "sample_mask_audit"}, indent=2))


if __name__ == "__main__":
    main()
