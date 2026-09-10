#!/usr/bin/env python3
"""Decode Stage 2 online mask tokens and build visual, metric-free comparisons."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in [REPO_ROOT / "scripts/data", Path(__file__).resolve().parent]:
    sys.path.insert(0, str(path))

from run_stage1_eval import (  # noqa: E402
    _atomic_write_json,
    _atomic_write_jsonl,
    _fit_panel_cell,
    _panel_font,
    _wrap_panel_text,
    resolve_data_path,
)
from run_stage2_eval import (  # noqa: E402
    DEFAULT_DATASET_BASE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SAMTOK_TE,
    DEFAULT_VALSET,
    load_and_validate_rows,
)
from samtok_codec import SamtokCodec  # noqa: E402


ONLINE_SETTING = "s2_stage2_online_cot"


def overlay(image: Image.Image, mask: np.ndarray) -> Image.Image:
    array = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    selected = np.asarray(mask, dtype=bool)
    array[selected] = array[selected] * 0.45 + np.asarray([245, 62, 72]) * 0.55
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def unavailable(source: Image.Image, message: str, size=(320, 320)) -> Image.Image:
    cell = _fit_panel_cell(source, size)
    shade = Image.new("RGBA", size, (20, 28, 40, 180))
    cell = Image.alpha_composite(cell.convert("RGBA"), shade).convert("RGB")
    draw = ImageDraw.Draw(cell)
    font = _panel_font(18, bold=True)
    lines = _wrap_panel_text(draw, message, font, size[0] - 36)
    top = (size[1] - 25 * len(lines)) // 2
    for index, line in enumerate(lines):
        width = draw.textbbox((0, 0), line, font=font)[2]
        draw.text(((size[0] - width) // 2, top + 25 * index), line, font=font, fill="white")
    return cell


def read_binary_mask(path: Path, source_size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        if image.size != source_size:
            raise ValueError(f"Mask geometry mismatch: {path}: {image.size} != {source_size}")
        return np.asarray(image.convert("L")) > 0


def make_panel(
    row: dict,
    source: Image.Image,
    raw_mask: np.ndarray,
    gt_decoded: np.ndarray,
    online_mask: np.ndarray | None,
    parse_layer: str | None,
) -> Image.Image:
    cell_size, label_height = (320, 320), 42
    cells = [
        _fit_panel_cell(overlay(source, raw_mask), cell_size),
        _fit_panel_cell(overlay(source, gt_decoded), cell_size),
        (
            _fit_panel_cell(overlay(source, online_mask), cell_size)
            if online_mask is not None
            else unavailable(source, "No valid online mask span", cell_size)
        ),
    ]
    labels = ["GT raw mask", "GT mask-token decode", "Online mask-token decode"]
    width = cell_size[0] * len(cells)
    heading_font, label_font = _panel_font(20, bold=True), _panel_font(16, bold=True)
    scratch = ImageDraw.Draw(Image.new("RGB", (width, 1), "white"))
    heading = (
        f"#{row['eval_index']:04d} | {row['primary_category']} | "
        f"Online parser: {parse_layer or 'none'} | Instruction: {row['prompt']}"
    )
    lines = _wrap_panel_text(scratch, heading, heading_font, width - 28)
    header_height = max(58, 16 + 28 * len(lines))
    panel = Image.new("RGB", (width, header_height + label_height + 320), "white")
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, width, header_height), fill=(22, 34, 52))
    for index, line in enumerate(lines):
        draw.text((14, 9 + index * 28), line, font=heading_font, fill="white")
    for column, (cell, label) in enumerate(zip(cells, labels)):
        left = column * 320
        label_width = draw.textbbox((0, 0), label, font=label_font)[2]
        draw.text(
            (left + max(5, (320 - label_width) // 2), header_height + 9),
            label,
            font=label_font,
            fill="black",
        )
        panel.paste(cell, (left, header_height + label_height))
    return panel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--valset", type=Path, default=DEFAULT_VALSET)
    parser.add_argument("--dataset_base", type=Path, default=DEFAULT_DATASET_BASE)
    parser.add_argument("--eval_output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--sam2_ckpt", type=Path, default=DEFAULT_SAMTOK_TE / "sam2.1_hiera_large.pt")
    parser.add_argument(
        "--mask_tokenizer_ckpt",
        type=Path,
        default=DEFAULT_SAMTOK_TE / "mask_tokenizer_256x2.pth",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    rows, _ = load_and_validate_rows(args.valset, args.dataset_base)
    output_dir = args.output_dir or args.eval_output_dir / "visualizations/mask_comparisons"
    result_paths = sorted((args.eval_output_dir / ONLINE_SETTING).glob("[0-9][0-9][0-9][0-9].json"))
    records = {
        int(record["eval_index"]): record
        for record in (
            json.loads(path.read_text(encoding="utf-8")) for path in result_paths
        )
    }
    expected = {int(row["eval_index"]) for row in rows}
    if set(records) != expected:
        raise ValueError(
            f"Online result coverage mismatch: missing={sorted(expected - set(records))}, "
            f"extra={sorted(set(records) - expected)}"
        )
    codec = SamtokCodec(
        args.sam2_ckpt,
        args.mask_tokenizer_ckpt,
        device=args.device,
        dtype=torch.float32,
    )
    by_category: defaultdict[str, list[Path]] = defaultdict(list)
    manifest, parse_layers = [], Counter()
    for position, row in enumerate(rows, 1):
        index = int(row["eval_index"])
        record = records[index]
        source_path = resolve_data_path(row["edit_image"], args.dataset_base)
        with Image.open(source_path) as image:
            source = image.convert("RGB")
        raw_mask = read_binary_mask(
            resolve_data_path(row["gt_mask"], args.dataset_base), source.size
        )
        gt_decoded = read_binary_mask(
            resolve_data_path(row["gt_decoded_mask"], args.dataset_base), source.size
        )
        online_masks = (
            codec.decode(source, record["conditioned_mt_cot"])
            if record.get("conditioned_mt_cot")
            else []
        )
        online_mask = (
            np.logical_or.reduce([np.asarray(mask, dtype=bool) for mask in online_masks])
            if online_masks
            else None
        )
        parse_layer = record.get("parse_layer")
        parse_layers[str(parse_layer)] += 1
        panel = make_panel(row, source, raw_mask, gt_decoded, online_mask, parse_layer)
        category = row["primary_category"]
        path = output_dir / category / f"{index:04d}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        panel.save(path, quality=92)
        by_category[category].append(path)
        manifest.append(
            {
                "eval_index": index,
                "eval_id": row["eval_id"],
                "category": category,
                "instruction": row["prompt"],
                "parse_layer": parse_layer,
                "online_generated_span_count": len(online_masks),
                "online_mask_decodable": online_mask is not None,
                "panel": str(path.resolve()),
            }
        )
        print(
            f"[mask visualization] {position}/{len(rows)} eval_index={index} "
            f"parse={parse_layer} decoded_spans={len(online_masks)}",
            flush=True,
        )

    overviews = {}
    for category, paths in sorted(by_category.items()):
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        overview = Image.new(
            "RGB", (max(image.width for image in images), sum(image.height for image in images)), "white"
        )
        top = 0
        for image in images:
            overview.paste(image, (0, top))
            top += image.height
        path = output_dir / f"overview_{category}.jpg"
        overview.save(path, quality=92)
        overviews[category] = str(path.resolve())
    _atomic_write_jsonl(output_dir / "manifest.jsonl", manifest)
    report = {
        "status": "complete",
        "protocol": "metric-free visual comparison of raw GT, GT token decode, and online token decode",
        "rows": len(rows),
        "online_decodable_rows": sum(row["online_mask_decodable"] for row in manifest),
        "parse_layers": dict(sorted(parse_layers.items())),
        "per_case_panels": len(manifest),
        "category_overviews": overviews,
    }
    _atomic_write_json(output_dir / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
