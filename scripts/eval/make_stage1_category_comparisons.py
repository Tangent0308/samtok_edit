#!/usr/bin/env python3
"""Build model-free, per-category Stage 1 or multi-checkpoint comparison sheets.

The final-result sheets read completed evaluation PNGs directly. The mask
sheets use the raw CrispEdit raster mask and the independent GT/online decoded
overlay cells already produced by analyze_stage1_cot_masks.py. Passing one or
more Stage 2 result roots enables checkpoint comparison layouts. No model is
loaded.
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
STAGE1_ONLINE_SETTING = "s4_stage1_te_online_cot"
STAGE2_ONLINE_SETTING = "s7_stage2_online_cot"
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
STAGE1_FINAL_LABELS = (
    "Original",
    "GT edited image",
    "S1 Stock 2511",
    "S2 Initial direct",
    "S3 Stage-1 direct",
    "S4 Online CoT",
    "S5 GT CoT",
)
EIGHT_FINAL_LABELS = STAGE1_FINAL_LABELS + (
    "S6 Stage-2 direct",
    "S7 Stage-2 online CoT",
    "S8 Stage-2 GT CoT",
)
STAGE1_MASK_LABELS = (
    "GT raster mask (blue)",
    "GT token decode (green)",
    "Online token decode (red)",
)
EIGHT_MASK_LABELS = (
    "GT raster mask (blue)",
    "GT token decode (green)",
    "Stage-1 online token decode (red)",
    "Stage-2 online token decode (red)",
)
STAGE2_VARIANT_LABELS = (
    "direct",
    "online CoT",
    "GT CoT",
)
FINAL_CELL_SIZE = (280, 280)
MASK_CELL_SIZE = (320, 320)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--stage2_root",
        type=Path,
        default=None,
        help=(
            "primary completed S6-S8 result root; when set, append its three "
            "settings to the backward-compatible Stage 1 layout"
        ),
    )
    parser.add_argument(
        "--additional_stage2_root",
        type=Path,
        action="append",
        default=[],
        help=(
            "additional completed S6-S8 result root to append as three new "
            "comparison settings; may be passed more than once"
        ),
    )
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
        help=(
            "default: ROOT/analysis/category_comparisons for Stage 1, or "
            "the newest Stage 2 root's sibling N_settings_comparison/analysis/"
            "category_comparisons for checkpoint comparisons"
        ),
    )
    parser.add_argument("--jpeg_quality", type=int, default=92)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_online_records(root: Path, setting: str) -> list[dict]:
    sidecars = sorted((root / setting).glob("[0-9][0-9][0-9][0-9].json"))
    records = [json.loads(path.read_text(encoding="utf-8")) for path in sidecars]
    indices = [int(record["metadata_index"]) for record in records]
    if len(records) != 64 or indices != list(range(64)):
        raise ValueError(
            f"Expected metadata indices 0..63 in {root / setting}, got {indices}"
        )
    for record in records:
        index = int(record["metadata_index"])
        expected = root / setting / f"{index:04d}.png"
        if Path(record["output"]).resolve() != expected.resolve():
            raise ValueError(f"Online output path mismatch at index={index}")
    return records


def validate_matching_online_records(
    stage1_records: list[dict], stage2_records: list[dict]
) -> dict:
    """Prove that S4 and S7 use the same Stage 1 TE pass-1 result."""

    identity_fields = (
        "metadata_index",
        "source",
        "target",
        "prompt",
        "gt_mt_cot",
        "conditioned_mt_cot",
        "pass1_raw",
        "parse_layer",
    )
    if len(stage1_records) != len(stage2_records):
        raise ValueError(
            f"S4/S7 record count mismatch: {len(stage1_records)} != {len(stage2_records)}"
        )
    for stage1, stage2 in zip(stage1_records, stage2_records, strict=True):
        for field in identity_fields:
            if stage1.get(field) != stage2.get(field):
                raise ValueError(
                    "S4/S7 online record mismatch at "
                    f"index={stage1.get('metadata_index')} field={field}"
                )
    return {
        "checked_records": len(stage1_records),
        "matching_records": len(stage1_records),
        "fields": list(identity_fields),
        "all_equal": True,
    }


def load_checkpoint_step(root: Path) -> int:
    path = root / "preflight.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    step = int(payload["models"]["checkpoint_step"])
    checkpoint = Path(payload["models"]["dit_lora"])
    if checkpoint.stem != f"step-{step}":
        raise ValueError(
            f"Stage 2 preflight checkpoint mismatch in {path}: {checkpoint} vs step={step}"
        )
    return step


def build_comparison_labels(
    checkpoint_steps: list[int],
) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    """Return final/mask columns and title for zero or more Stage 2 roots."""

    if not checkpoint_steps:
        return STAGE1_FINAL_LABELS, STAGE1_MASK_LABELS, "Stage 1 evaluation"
    if len(checkpoint_steps) == 1:
        return EIGHT_FINAL_LABELS, EIGHT_MASK_LABELS, "S1-S8 evaluation"

    final_labels = list(STAGE1_FINAL_LABELS)
    mask_labels = list(STAGE1_MASK_LABELS)
    for checkpoint_number, step in enumerate(checkpoint_steps):
        setting_start = 6 + checkpoint_number * len(STAGE2_SETTINGS)
        final_labels.extend(
            f"S{setting_start + offset} Stage-2 step-{step} {variant}"
            for offset, variant in enumerate(STAGE2_VARIANT_LABELS)
        )
        mask_labels.append(f"Stage-2 step-{step} online token decode (red)")
    total_settings = len(STAGE1_SETTINGS) + len(STAGE2_SETTINGS) * len(checkpoint_steps)
    return (
        tuple(final_labels),
        tuple(mask_labels),
        f"S1-S{total_settings} checkpoint comparison",
    )


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
    evaluation_title: str,
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
        f"{evaluation_title} | category: {category} | samples: {len(records)}",
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
    if args.additional_stage2_root and args.stage2_root is None:
        raise ValueError("--additional_stage2_root requires --stage2_root")
    stage2_roots = (
        [args.stage2_root, *args.additional_stage2_root]
        if args.stage2_root is not None
        else []
    )
    resolved_stage2_roots = [root.resolve() for root in stage2_roots]
    if len(set(resolved_stage2_roots)) != len(resolved_stage2_roots):
        raise ValueError("Stage 2 result roots must be unique")
    checkpoint_steps = [load_checkpoint_step(root) for root in stage2_roots]
    if len(set(checkpoint_steps)) != len(checkpoint_steps):
        raise ValueError(f"Stage 2 checkpoint steps must be unique: {checkpoint_steps}")
    final_labels, mask_labels, evaluation_title = build_comparison_labels(
        checkpoint_steps
    )
    if args.output_dir is not None:
        output_dir = args.output_dir
    elif stage2_roots:
        total_settings = len(STAGE1_SETTINGS) + len(STAGE2_SETTINGS) * len(stage2_roots)
        comparison_name = (
            "eight_settings_comparison"
            if len(stage2_roots) == 1
            else (
                "eleven_settings_comparison"
                if total_settings == 11
                else f"{total_settings}_settings_comparison"
            )
        )
        output_dir = (
            stage2_roots[-1].parent
            / comparison_name
            / "analysis"
            / "category_comparisons"
        )
    else:
        output_dir = args.root / "analysis" / "category_comparisons"
    decoded_panel_dir = (
        args.decoded_panel_dir
        or args.root / "analysis" / "decoded_mask_overlap" / "panels"
    )
    records = load_online_records(args.root, STAGE1_ONLINE_SETTING)
    stage2_record_sets = [
        load_online_records(root, STAGE2_ONLINE_SETTING) for root in stage2_roots
    ]
    online_match_audits = [
        validate_matching_online_records(records, stage2_records)
        for stage2_records in stage2_record_sets
    ]
    stage2_by_index_sets = [
        {int(record["metadata_index"]): record for record in stage2_records}
        for stage2_records in stage2_record_sets
    ]
    metrics = load_mask_metrics(args.root)
    by_type: defaultdict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_type[record["provenance"]["edit_type"]].append(record)

    raw_mask_cache: dict[Path, dict[int, bytes]] = {}
    mask_audit: dict[int, dict] = {}

    def final_cells(record: dict) -> list[Image.Image]:
        index = int(record["metadata_index"])
        paths = [Path(record["source"]), Path(record["target"])] + [
            args.root / setting / f"{index:04d}.png" for setting in STAGE1_SETTINGS
        ]
        paths.extend(
            stage2_root / setting / f"{index:04d}.png"
            for stage2_root in stage2_roots
            for setting in STAGE2_SETTINGS
        )
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
            online_badge = (
                "EMPTY STAGE-1 ONLINE MASK"
                if stage2_roots
                else "EMPTY ONLINE TOKEN MASK"
            )
            online_cell = add_empty_badge(clean_source, online_badge)
            if index in metrics:
                raise ValueError(f"Unexpected decoded mask metrics for empty GT index={index}")

        cells = [raw_cell, gt_cell, online_cell]
        stage2_mask_audit = {}
        for step, stage2_by_index in zip(
            checkpoint_steps, stage2_by_index_sets, strict=True
        ):
            stage2_record = stage2_by_index[index]
            stage2_items = parse_cot(stage2_record["conditioned_mt_cot"])
            if gt_items:
                # Every Stage 2 online setting uses the same frozen Stage 1 TE
                # as S4. The per-root full-record equality audits prove these
                # are exactly the same tokens, hence the same codec decode.
                # Keep a distinct cell for every checkpoint comparison column.
                stage2_online_cell = online_cell.copy()
            else:
                stage2_online_cell = add_empty_badge(
                    fit_cell(source, MASK_CELL_SIZE),
                    f"EMPTY S2-{step} ONLINE MASK",
                )
            cells.append(stage2_online_cell)
            stage2_mask_audit[f"step-{step}"] = {
                "online_cot_nonempty": bool(stage2_items),
                "stage1_online_cot_equal": (
                    record["conditioned_mt_cot"]
                    == stage2_record["conditioned_mt_cot"]
                ),
            }

        mask_audit[index] = {
            "gt_cot_nonempty": bool(gt_items),
            "stage1_online_cot_nonempty": bool(online_items),
            "stage2_checkpoints": stage2_mask_audit,
            "stage2_online_cot_nonempty": (
                next(iter(stage2_mask_audit.values()))["online_cot_nonempty"]
                if len(stage2_mask_audit) == 1
                else None
            ),
            "stage1_stage2_online_cot_equal": (
                next(iter(stage2_mask_audit.values()))["stage1_online_cot_equal"]
                if len(stage2_mask_audit) == 1
                else None
            ),
            "raw_mask_nonempty": bool(raw_mask.any()),
            "decoded_panel": (
                str(decoded_panel_dir / f"{index:04d}.jpg") if gt_items else None
            ),
            "mask_iou": metrics.get(index, {}).get("mask_iou"),
        }
        return cells

    manifest = {
        "status": "complete",
        "root": str(args.root),
        "stage2_root": str(args.stage2_root) if args.stage2_root else None,
        "stage2_roots": [str(root) for root in stage2_roots],
        "stage2_checkpoint_steps": checkpoint_steps,
        "output_dir": str(output_dir),
        "generation": (
            f"model-free aggregation of completed S1-S{len(STAGE1_SETTINGS) + len(STAGE2_SETTINGS) * len(stage2_roots)} "
            "outputs and audited decoded-mask panels"
            if stage2_roots
            else "model-free aggregation of completed S1-S5 outputs and audited decoded-mask panels"
        ),
        "final_columns": list(final_labels),
        "mask_columns": list(mask_labels),
        "mask_layout_note": (
            f"The {len(mask_labels)} masks are separate source-image overlay panels; "
            "no cell combines multiple masks."
        ),
        "stage2_online_decode_note": (
            "Every Stage 2 online setting uses the same frozen Stage 1 TE as S4. "
            "For every checkpoint, all 64 source/target/prompt/GT/conditioned/"
            "pass1/parser fields were proven equal, so the S4 codec decode is "
            "reused in a distinct overlay cell for each checkpoint."
            if stage2_roots
            else None
        ),
        "stage1_stage2_online_match_audits": {
            f"step-{step}": audit
            for step, audit in zip(checkpoint_steps, online_match_audits, strict=True)
        },
        "stage1_stage2_online_match_audit": (
            online_match_audits[0] if len(online_match_audits) == 1 else None
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
            labels=final_labels,
            cell_size=FINAL_CELL_SIZE,
            cell_builder=final_cells,
            path=final_path,
            quality=args.jpeg_quality,
            evaluation_title=evaluation_title,
        )
        mask_size = make_sheet(
            category=category,
            records=category_records,
            labels=mask_labels,
            cell_size=MASK_CELL_SIZE,
            cell_builder=mask_cells,
            path=mask_path,
            quality=args.jpeg_quality,
            evaluation_title=evaluation_title,
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
