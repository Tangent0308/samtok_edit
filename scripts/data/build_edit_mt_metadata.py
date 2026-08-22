#!/usr/bin/env python3
"""Join CrispEdit image/mask parquets, encode masks, and materialize metadata."""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "DiffSynth-Studio"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from diffsynth.core.data.samtok_dataset import make_labels, sanitize_label, to_cot  # noqa: E402
from samtok_codec import SamtokCodec  # noqa: E402


DEFAULT_CRISPEDIT = Path(
    "/mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M"
)
DEFAULT_MASKS = Path(
    "/mnt/bn/strategy-mllm-train/user/tanyue/datasets/"
    "CrispEdit-2M-mask-parquet-101697"
)
DEFAULT_SAMTOK_DIR = Path(
    "/mnt/bn/strategy-mllm-train/user/tanyue/models/SAMTok/"
    "Qwen2.5-VL-7B-SAMTok-gres-ft"
)


def _image_bytes(cell: dict, parquet_dir: Path) -> bytes:
    data = cell.get("bytes") if isinstance(cell, dict) else None
    if data:
        return data
    path = cell.get("path") if isinstance(cell, dict) else None
    if path:
        path = Path(path)
        if not path.is_absolute():
            path = parquet_dir / path
        return path.read_bytes()
    raise ValueError("Image parquet cell contains neither bytes nor a path")


def _decode_image(data: bytes) -> tuple[Image.Image, str]:
    with Image.open(io.BytesIO(data)) as image:
        image_format = (image.format or "PNG").lower()
        if image_format == "jpeg":
            image_format = "jpg"
        return image.convert("RGB"), image_format


def _decode_mask(data: bytes, image_size: tuple[int, int]) -> np.ndarray:
    if not data:
        return np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
    with Image.open(io.BytesIO(data)) as mask_image:
        if mask_image.size != image_size:
            mask_image = mask_image.resize(image_size, Image.Resampling.NEAREST)
        return (np.asarray(mask_image.convert("L")) > 0).astype(np.uint8)


def _write_bytes_once(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.stat().st_size != len(data):
            raise ValueError(f"Existing materialized image has a different size: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _label_for_row(phrases: dict, canonical_type: str, instruction: str) -> str:
    if canonical_type == "add":
        label = phrases.get("target") or phrases.get("source")
    else:
        label = phrases.get("source") or phrases.get("target")
    return sanitize_label(label or instruction)


def _atomic_jsonl(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _combine_shards(shard_paths, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("wb") as output:
        for shard_path in shard_paths:
            with shard_path.open("rb") as shard:
                while chunk := shard.read(8 * 1024 * 1024):
                    output.write(chunk)
    os.replace(temporary, output_path)


def process_parquet_pair(
    raw_path: Path,
    mask_path: Path,
    output_root: Path,
    codec: SamtokCodec,
    codec_batch_size: int,
    remaining_rows: int | None,
):
    raw_rows = pq.read_table(
        raw_path, columns=["input_img", "instruction", "output_img", "type"]
    ).to_pylist()
    mask_rows = pq.read_table(mask_path).to_pylist()
    selected = []
    stats = Counter()

    for mask_row in mask_rows:
        if mask_row.get("filter_decision") != "keep":
            stats["filter_drop"] += 1
            continue
        if remaining_rows is not None and len(selected) >= remaining_rows:
            break
        row_idx = int(mask_row["row_idx"])
        if not 0 <= row_idx < len(raw_rows):
            raise IndexError(f"{mask_path.name}: row_idx {row_idx} outside raw parquet")
        raw = raw_rows[row_idx]
        if raw["instruction"] != mask_row["instruction"]:
            raise ValueError(f"Instruction mismatch in {mask_path.name} row {row_idx}")

        source_bytes = _image_bytes(raw["input_img"], raw_path.parent)
        target_bytes = _image_bytes(raw["output_img"], raw_path.parent)
        source, source_ext = _decode_image(source_bytes)
        _, target_ext = _decode_image(target_bytes)
        stem = raw_path.stem.replace(" ", "_")
        source_rel = Path("images") / stem / f"{row_idx:06d}_source.{source_ext}"
        target_rel = Path("images") / stem / f"{row_idx:06d}_target.{target_ext}"
        _write_bytes_once(output_root / source_rel, source_bytes)
        _write_bytes_once(output_root / target_rel, target_bytes)

        try:
            phrases = json.loads(mask_row.get("phrases_json") or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"Bad phrases_json in {mask_path.name} row {row_idx}") from exc
        canonical_type = mask_row.get("canonical_type") or raw.get("type") or ""
        is_empty_cot = bool(phrases.get("is_global") or phrases.get("is_noop"))
        mask = _decode_mask(mask_row.get("mask_png") or b"", source.size)
        if mask.sum() == 0:
            is_empty_cot = True
            stats["empty_mask"] += 1

        selected.append(
            {
                "source": source,
                "mask": mask,
                "label": _label_for_row(
                    phrases, canonical_type, raw["instruction"]
                ),
                "empty_cot": is_empty_cot,
                "edit_row": {
                    "image": target_rel.as_posix(),
                    "edit_image": source_rel.as_posix(),
                    "prompt": raw["instruction"],
                    "sample_type": "edit",
                },
                "provenance": {
                    "source_parquet": raw_path.name,
                    "row_idx": row_idx,
                    "edit_type": canonical_type,
                    "qc_flag": mask_row.get("qc_flag"),
                },
            }
        )

    local = [row for row in selected if not row["empty_cot"]]
    for start in range(0, len(local), codec_batch_size):
        batch = local[start : start + codec_batch_size]
        spans = codec.encode_single_batch(
            (row["source"], row["mask"]) for row in batch
        )
        for row, span in zip(batch, spans):
            row["mt_cot"] = to_cot([(span, make_labels(row["label"], 1)[0])])
            stats["encoded_mask"] += 1

    edit_mt_rows, edit_rows = [], []
    for row in selected:
        mt_cot = to_cot([]) if row["empty_cot"] else row["mt_cot"]
        edit_row = row["edit_row"]
        edit_rows.append(edit_row)
        edit_mt_rows.append(
            {
                **edit_row,
                "mt_cot": mt_cot,
                "sample_type": "edit_mt",
                "provenance": row["provenance"],
            }
        )
        if row["empty_cot"]:
            stats["empty_cot"] += 1
    stats["kept"] = len(selected)
    return edit_mt_rows, edit_rows, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crispedit_dir", type=Path, default=DEFAULT_CRISPEDIT)
    parser.add_argument("--mask_dir", type=Path, default=DEFAULT_MASKS)
    parser.add_argument(
        "--output_root",
        type=Path,
        default=_REPO_ROOT / "data" / "crispedit_samtok",
    )
    parser.add_argument(
        "--edit_mt_jsonl", type=Path, default=None, help="Defaults under output_root"
    )
    parser.add_argument(
        "--edit_jsonl", type=Path, default=None, help="Defaults under output_root"
    )
    parser.add_argument(
        "--sam2_ckpt", type=Path, default=DEFAULT_SAMTOK_DIR / "sam2.1_hiera_large.pt"
    )
    parser.add_argument(
        "--mask_tokenizer_ckpt",
        type=Path,
        default=DEFAULT_SAMTOK_DIR / "mask_tokenizer_256x2.pth",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="float32")
    parser.add_argument("--codec_batch_size", type=int, default=4)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.codec_batch_size < 1:
        raise ValueError("codec_batch_size must be positive")
    if args.resume and args.max_rows is not None:
        raise ValueError("--resume and --max_rows cannot be combined")
    output_root = args.output_root.resolve()
    edit_mt_path = args.edit_mt_jsonl or output_root / "edit_mt.jsonl"
    edit_path = args.edit_jsonl or output_root / "edit.jsonl"
    shard_dir = output_root / "metadata_shards"

    mask_paths = sorted(args.mask_dir.glob("*.parquet"))
    if args.max_files is not None:
        mask_paths = mask_paths[: args.max_files]
    pairs = []
    for mask_path in mask_paths:
        raw_path = args.crispedit_dir / mask_path.name
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        pairs.append((raw_path, mask_path))

    codec = SamtokCodec(
        args.sam2_ckpt,
        args.mask_tokenizer_ckpt,
        device=args.device,
        dtype=getattr(torch, args.dtype),
    )
    totals = Counter()
    completed_mt, completed_edit = [], []
    remaining = args.max_rows
    for raw_path, mask_path in tqdm(pairs, desc="CrispEdit parquet shards"):
        mt_shard = shard_dir / f"{mask_path.stem}.edit_mt.jsonl"
        edit_shard = shard_dir / f"{mask_path.stem}.edit.jsonl"
        if args.resume and mt_shard.is_file() and edit_shard.is_file():
            completed_mt.append(mt_shard)
            completed_edit.append(edit_shard)
            totals["resumed_shards"] += 1
            continue
        edit_mt_rows, edit_rows, stats = process_parquet_pair(
            raw_path,
            mask_path,
            output_root,
            codec,
            args.codec_batch_size,
            remaining,
        )
        _atomic_jsonl(edit_mt_rows, mt_shard)
        _atomic_jsonl(edit_rows, edit_shard)
        completed_mt.append(mt_shard)
        completed_edit.append(edit_shard)
        totals.update(stats)
        if remaining is not None:
            remaining -= len(edit_mt_rows)
            if remaining <= 0:
                break

    _combine_shards(completed_mt, edit_mt_path)
    _combine_shards(completed_edit, edit_path)
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "edit_mt_metadata": str(edit_mt_path),
                "edit_metadata": str(edit_path),
                "stats": dict(totals),
                "important": "SAMTok codes were encoded on input/source images",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
