#!/usr/bin/env python3
"""Decode Stage 1 online/GT CoT masks and measure spatial overlap."""

from __future__ import annotations

import argparse
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "data"))

from samtok_codec import SamtokCodec  # noqa: E402


DEFAULT_ROOT = Path(
    "/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/"
    "stage1_evaluation/five_settings"
)
DEFAULT_SAMTOK_DIR = Path(
    "/mnt/bn/strategy-mllm-train/user/tanyue/models/SAMTok/"
    "Qwen2.5-VL-7B-SAMTok-gres-ft"
)
DEFAULT_RAW_MASK_DIR = Path(
    "/mnt/bn/strategy-mllm-train/user/tanyue/datasets/"
    "CrispEdit-2M-mask-parquet-101697"
)
ONLINE_SETTING = "s4_stage1_te_online_cot"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--sam2_ckpt",
        type=Path,
        default=DEFAULT_SAMTOK_DIR / "sam2.1_hiera_large.pt",
    )
    parser.add_argument(
        "--mask_tokenizer_ckpt",
        type=Path,
        default=DEFAULT_SAMTOK_DIR / "mask_tokenizer_256x2.pth",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--raw_mask_dir", type=Path, default=DEFAULT_RAW_MASK_DIR)
    parser.add_argument("--output_dir", type=Path, default=None)
    return parser.parse_args()


def parse_cot(value: str) -> list[dict]:
    prefix, suffix = "```json\n", "\n```"
    if value.startswith(prefix) and value.endswith(suffix):
        value = value[len(prefix) : -len(suffix)]
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise ValueError("CoT payload must be a JSON list")
    return parsed


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else 1.0


def mask_dice(left: np.ndarray, right: np.ndarray) -> float:
    denominator = left.sum() + right.sum()
    return float(2 * np.logical_and(left, right).sum() / denominator) if denominator else 1.0


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    y, x = np.nonzero(mask)
    if not len(x):
        return None
    return int(x.min()), int(y.min()), int(x.max()), int(y.max())


def bbox_iou(left, right) -> float:
    if left is None or right is None:
        return float(left is None and right is None)
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    intersection = max(0, min(lx2, rx2) - max(lx1, rx1) + 1) * max(
        0, min(ly2, ry2) - max(ly1, ry1) + 1
    )
    left_area = (lx2 - lx1 + 1) * (ly2 - ly1 + 1)
    right_area = (rx2 - rx1 + 1) * (ry2 - ry1 + 1)
    return float(intersection / (left_area + right_area - intersection))


def centroid_distance(left: np.ndarray, right: np.ndarray) -> float:
    left_y, left_x = np.nonzero(left)
    right_y, right_x = np.nonzero(right)
    if not len(left_x) or not len(right_x):
        return 1.0
    distance = np.hypot(left_x.mean() - right_x.mean(), left_y.mean() - right_y.mean())
    diagonal = np.hypot(left.shape[1], left.shape[0])
    return float(distance / diagonal)


def union_masks(masks: list[np.ndarray]) -> np.ndarray:
    if not masks:
        raise ValueError("Cannot union an empty mask list")
    return np.logical_or.reduce([np.asarray(mask, dtype=bool) for mask in masks])


def raw_annotation_mask(
    record: dict,
    image_size: tuple[int, int],
    raw_mask_dir: Path,
    cache: dict[Path, dict[int, bytes]],
) -> np.ndarray:
    provenance = record["provenance"]
    parquet_path = raw_mask_dir / provenance["source_parquet"]
    if parquet_path not in cache:
        cache[parquet_path] = {
            int(row["row_idx"]): row["mask_png"]
            for row in pq.read_table(
                parquet_path, columns=["row_idx", "mask_png"]
            ).to_pylist()
        }
    data = cache[parquet_path][int(provenance["row_idx"])]
    with Image.open(io.BytesIO(data)) as image:
        mask = image.convert("L")
        if mask.size != image_size:
            mask = mask.resize(image_size, Image.Resampling.NEAREST)
        return np.asarray(mask) > 0


def font(size: int, *, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for base in [
        Path("/usr/share/fonts/truetype/dejavu"),
        Path("/usr/share/fonts/dejavu"),
    ]:
        path = base / name
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def wrap_text(draw, text: str, text_font, width: int) -> list[str]:
    lines, current = [], ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and draw.textbbox((0, 0), candidate, font=text_font)[2] > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or [""]


def overlay(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int]):
    array = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    selected = np.asarray(mask, dtype=bool)
    array[selected] = array[selected] * 0.45 + np.asarray(color) * 0.55
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def overlap_overlay(image: Image.Image, predicted: np.ndarray, gt: np.ndarray):
    array = np.asarray(image.convert("RGB"), dtype=np.float32).copy() * 0.45
    predicted = np.asarray(predicted, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    intersection = predicted & gt
    predicted_only = predicted & ~gt
    gt_only = gt & ~predicted
    array[intersection] = (45, 220, 90)
    array[predicted_only] = (245, 65, 65)
    array[gt_only] = (60, 120, 255)
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def fit_cell(image: Image.Image, size: tuple[int, int] = (320, 320)):
    image = image.convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    cell = Image.new("RGB", size, "white")
    cell.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return cell


def make_panel(
    record: dict,
    source: Image.Image,
    predicted,
    gt,
    raw_annotation,
    metrics: dict,
    path: Path,
):
    cells = [
        fit_cell(source),
        fit_cell(overlay(source, predicted, (245, 65, 65))),
        fit_cell(overlay(source, gt, (60, 210, 90))),
        fit_cell(overlay(source, raw_annotation, (60, 120, 255))),
        fit_cell(overlap_overlay(source, predicted, gt)),
    ]
    labels = [
        "Source",
        "Online mask (red)",
        "GT decoded (green)",
        "Raw annotation (blue)",
        "Overlap: common=green, online=red, GT=blue",
    ]
    width = 320 * len(cells)
    heading_font, label_font = font(20, bold=True), font(16, bold=True)
    scratch = ImageDraw.Draw(Image.new("RGB", (width, 1)))
    heading = (
        f"#{record['metadata_index']:04d} | {record['provenance']['edit_type']} | "
        f"IoU {metrics['mask_iou']:.3f} | Dice {metrics['dice']:.3f} | "
        f"Instruction: {record['prompt']}"
    )
    lines = wrap_text(scratch, heading, heading_font, width - 28)
    header_height = max(54, 14 + len(lines) * 27)
    label_height = 36
    panel = Image.new("RGB", (width, header_height + label_height + 320), "white")
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, width, header_height), fill=(22, 34, 52))
    for line_index, line in enumerate(lines):
        draw.text((14, 8 + line_index * 27), line, font=heading_font, fill="white")
    for column, (cell, label) in enumerate(zip(cells, labels)):
        left = column * 320
        label_width = draw.textbbox((0, 0), label, font=label_font)[2]
        draw.text(
            (left + max(5, (320 - label_width) // 2), header_height + 7),
            label,
            font=label_font,
            fill="black",
        )
        panel.paste(cell, (left, header_height + label_height))
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(path, quality=92)


def metric_summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p25": float(np.quantile(array, 0.25)),
        "p75": float(np.quantile(array, 0.75)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def summarize_rows(rows: list[dict]) -> dict:
    summary = {
        "count": len(rows),
        "mask_iou": metric_summary([row["mask_iou"] for row in rows]),
        "dice": metric_summary([row["dice"] for row in rows]),
        "bbox_iou": metric_summary([row["bbox_iou"] for row in rows]),
        "normalized_centroid_distance": metric_summary(
            [row["normalized_centroid_distance"] for row in rows]
        ),
        "predicted_to_gt_area_ratio": metric_summary(
            [row["predicted_to_gt_area_ratio"] for row in rows]
        ),
        "iou_threshold_counts": {
            threshold: sum(row["mask_iou"] >= float(threshold) for row in rows)
            for threshold in ["0.25", "0.50", "0.75"]
        },
    }
    for name in [
        "online_to_raw_iou",
        "gt_decoded_to_raw_iou",
        "online_to_raw_dice",
        "gt_decoded_to_raw_dice",
    ]:
        if all(name in row for row in rows):
            summary[name] = metric_summary([row[name] for row in rows])
    return summary


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    args = parse_args()
    output_dir = args.output_dir or args.root / "analysis" / "decoded_mask_overlap"
    sidecars = sorted((args.root / ONLINE_SETTING).glob("[0-9][0-9][0-9][0-9].json"))
    records = [json.loads(path.read_text(encoding="utf-8")) for path in sidecars]
    nonempty_records = [record for record in records if parse_cot(record["gt_mt_cot"])]
    codec = SamtokCodec(
        args.sam2_ckpt,
        args.mask_tokenizer_ckpt,
        device=args.device,
        dtype=torch.float32,
    )
    results = []
    raw_mask_cache: dict[Path, dict[int, bytes]] = {}
    for position, record in enumerate(nonempty_records, 1):
        predicted_items = parse_cot(record["conditioned_mt_cot"])
        gt_items = parse_cot(record["gt_mt_cot"])
        if len(predicted_items) != len(gt_items):
            raise ValueError(
                f"object-count mismatch at metadata_index={record['metadata_index']}"
            )
        predicted_spans = [item["mask_2d"] for item in predicted_items]
        gt_spans = [item["mask_2d"] for item in gt_items]
        with Image.open(record["source"]) as image:
            source = image.convert("RGB")
        if predicted_spans == gt_spans:
            predicted_masks = codec.decode(source, "".join(predicted_spans))
            gt_masks = [mask.copy() for mask in predicted_masks]
        else:
            decoded = codec.decode(source, "".join(predicted_spans + gt_spans))
            predicted_masks = decoded[: len(predicted_spans)]
            gt_masks = decoded[len(predicted_spans) :]
        if len(predicted_masks) != len(predicted_spans) or len(gt_masks) != len(gt_spans):
            raise RuntimeError(
                f"decode count mismatch at metadata_index={record['metadata_index']}"
            )
        predicted_union, gt_union = union_masks(predicted_masks), union_masks(gt_masks)
        raw_annotation = raw_annotation_mask(
            record, source.size, args.raw_mask_dir, raw_mask_cache
        )
        if not raw_annotation.any():
            raise ValueError(
                f"empty raw annotation at metadata_index={record['metadata_index']}"
            )
        gt_area = int(gt_union.sum())
        metrics = {
            "metadata_index": record["metadata_index"],
            "edit_type": record["provenance"]["edit_type"],
            "instruction": record["prompt"],
            "token_exact": predicted_spans == gt_spans,
            "mask_iou": mask_iou(predicted_union, gt_union),
            "dice": mask_dice(predicted_union, gt_union),
            "bbox_iou": bbox_iou(mask_bbox(predicted_union), mask_bbox(gt_union)),
            "normalized_centroid_distance": centroid_distance(predicted_union, gt_union),
            "predicted_area_pixels": int(predicted_union.sum()),
            "gt_area_pixels": gt_area,
            "predicted_to_gt_area_ratio": float(predicted_union.sum() / gt_area),
            "raw_annotation_area_pixels": int(raw_annotation.sum()),
            "online_to_raw_iou": mask_iou(predicted_union, raw_annotation),
            "gt_decoded_to_raw_iou": mask_iou(gt_union, raw_annotation),
            "online_to_raw_dice": mask_dice(predicted_union, raw_annotation),
            "gt_decoded_to_raw_dice": mask_dice(gt_union, raw_annotation),
            "per_object_iou": [
                mask_iou(predicted, target)
                for predicted, target in zip(predicted_masks, gt_masks)
            ],
            "predicted_spans": predicted_spans,
            "gt_spans": gt_spans,
        }
        results.append(metrics)
        make_panel(
            record,
            source,
            predicted_union,
            gt_union,
            raw_annotation,
            metrics,
            output_dir / "panels" / f"{record['metadata_index']:04d}.jpg",
        )
        print(
            f"[{position:02d}/{len(nonempty_records)}] "
            f"index={record['metadata_index']:04d} iou={metrics['mask_iou']:.4f}"
        )

    by_type: defaultdict[str, list[dict]] = defaultdict(list)
    for row in results:
        by_type[row["edit_type"]].append(row)
    report = {
        "status": "complete",
        "reference": "GT CoT decoded on the same source image with released VQ-SAM2",
        "online_setting": ONLINE_SETTING,
        "sam2_ckpt": str(args.sam2_ckpt),
        "mask_tokenizer_ckpt": str(args.mask_tokenizer_ckpt),
        "raw_mask_dir": str(args.raw_mask_dir),
        "total_eval_rows": len(records),
        "nonempty_gt_rows": len(results),
        "overall": summarize_rows(results),
        "token_exact_subset": summarize_rows([row for row in results if row["token_exact"]]),
        "token_nonexact_subset": summarize_rows(
            [row for row in results if not row["token_exact"]]
        ),
        "by_edit_type": {
            edit_type: summarize_rows(rows) for edit_type, rows in sorted(by_type.items())
        },
        "samples": results,
    }
    write_json(output_dir / "report.json", report)

    ordered = sorted(results, key=lambda row: row["mask_iou"])
    selected = ordered[:3] + [ordered[len(ordered) // 2]] + ordered[-3:]
    images = []
    for row in selected:
        with Image.open(output_dir / "panels" / f"{row['metadata_index']:04d}.jpg") as image:
            images.append(image.convert("RGB"))
    overview = Image.new(
        "RGB", (max(image.width for image in images), sum(image.height for image in images)), "white"
    )
    top = 0
    for image in images:
        overview.paste(image, (0, top))
        top += image.height
    overview.save(output_dir / "overview_iou_low_median_high.jpg", quality=92)
    print(json.dumps({key: value for key, value in report.items() if key != "samples"}, indent=2))


if __name__ == "__main__":
    main()
