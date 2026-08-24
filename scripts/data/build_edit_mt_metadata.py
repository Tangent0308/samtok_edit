#!/usr/bin/env python3
"""Join CrispEdit image/mask parquets, encode masks, and materialize metadata."""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import sys
from collections import Counter, defaultdict
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


def canonical_source_parquet(name: str) -> str:
    """Match parquet provenance to the image-directory filename normalization."""

    return Path(name).stem.replace(" ", "_") + ".parquet"


def source_identity_from_row(row: dict) -> tuple[str, int] | None:
    """Recover the original CrispEdit parquet row from generated metadata."""

    provenance = row.get("provenance") or {}
    source_parquet = provenance.get("source_parquet")
    row_idx = provenance.get("row_idx")
    if source_parquet is not None and row_idx is not None:
        return canonical_source_parquet(str(source_parquet)), int(row_idx)

    edit_image = row.get("edit_image")
    if not isinstance(edit_image, str):
        return None
    path = Path(edit_image)
    row_prefix, separator, _ = path.name.partition("_source.")
    if not separator or not row_prefix.isdigit() or not path.parent.name:
        return None
    return f"{path.parent.name}.parquet", int(row_prefix)


def load_excluded_source_ids(paths: list[Path]) -> tuple[set[tuple[str, int]], Counter]:
    """Load source identities already used by one or more metadata files."""

    excluded = set()
    stats = Counter()
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                stats["metadata_rows"] += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
                identity = source_identity_from_row(row)
                if identity is None:
                    stats["unidentifiable_rows"] += 1
                    continue
                if identity in excluded:
                    stats["duplicate_identities"] += 1
                else:
                    excluded.add(identity)
                    stats["source_identities"] += 1
                stats[f"identified_type:{row.get('sample_type', 'unknown')}"] += 1
    return excluded, stats


def collect_candidates(
    pairs: list[tuple[Path, Path]],
    ascii_only: bool,
    excluded_source_ids: set[tuple[str, int]] | None = None,
) -> tuple[list[tuple[str, int, str]], Counter]:
    """Collect globally sampleable rows without decoding images or loading the codec."""
    candidates = []
    stats = Counter()
    for raw_path, mask_path in tqdm(pairs, desc="Scanning eligible CrispEdit rows"):
        raw_rows = pq.read_table(raw_path, columns=["instruction", "type"]).to_pylist()
        mask_rows = pq.read_table(
            mask_path,
            columns=[
                "row_idx",
                "instruction",
                "filter_decision",
                "phrases_json",
                "canonical_type",
            ],
        ).to_pylist()
        for mask_row in mask_rows:
            if mask_row.get("filter_decision") != "keep":
                stats["filter_drop"] += 1
                continue
            row_idx = int(mask_row["row_idx"])
            source_identity = (canonical_source_parquet(mask_path.name), row_idx)
            if excluded_source_ids and source_identity in excluded_source_ids:
                stats["excluded_source"] += 1
                continue
            if not 0 <= row_idx < len(raw_rows):
                raise IndexError(
                    f"{mask_path.name}: row_idx {row_idx} outside raw parquet"
                )
            raw = raw_rows[row_idx]
            instruction = raw["instruction"]
            if instruction != mask_row["instruction"]:
                raise ValueError(
                    f"Instruction mismatch in {mask_path.name} row {row_idx}"
                )
            try:
                phrases = json.loads(mask_row.get("phrases_json") or "{}")
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Bad phrases_json in {mask_path.name} row {row_idx}"
                ) from exc
            canonical_type = mask_row.get("canonical_type") or raw.get("type") or ""
            is_empty_cot = bool(phrases.get("is_global") or phrases.get("is_noop"))
            label = _label_for_row(phrases, canonical_type, instruction)
            if ascii_only and (
                not instruction.isascii() or (not is_empty_cot and not label.isascii())
            ):
                stats["non_ascii_drop"] += 1
                continue
            candidates.append((mask_path.name, row_idx, canonical_type))
            stats["eligible"] += 1
            stats[f"eligible_type:{canonical_type}"] += 1
    return candidates, stats


def sample_candidates(
    candidates: list[tuple[str, int, str]],
    sample_rows: int,
    seed: int,
) -> list[tuple[str, int, str]]:
    if sample_rows < 1:
        raise ValueError("sample_rows must be positive")
    if sample_rows > len(candidates):
        raise ValueError(
            f"Requested {sample_rows} rows from only {len(candidates)} eligible rows"
        )
    return random.Random(seed).sample(candidates, sample_rows)


def partition_pairs(
    pairs: list[tuple[Path, Path]], num_workers: int, worker_index: int
) -> list[tuple[Path, Path]]:
    if num_workers < 1:
        raise ValueError("num_workers must be positive")
    if not 0 <= worker_index < num_workers:
        raise ValueError("worker_index must be in [0, num_workers)")
    return pairs[worker_index::num_workers]


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
    selected_row_indices: set[int] | None = None,
):
    raw_rows = pq.read_table(
        raw_path, columns=["input_img", "instruction", "output_img", "type"]
    ).to_pylist()
    mask_rows = pq.read_table(mask_path).to_pylist()
    selected = []
    stats = Counter()

    for mask_row in mask_rows:
        row_idx = int(mask_row["row_idx"])
        if selected_row_indices is not None and row_idx not in selected_row_indices:
            continue
        if mask_row.get("filter_decision") != "keep":
            stats["filter_drop"] += 1
            continue
        if remaining_rows is not None and len(selected) >= remaining_rows:
            break
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
    parser.add_argument(
        "--sample_rows",
        type=int,
        default=None,
        help="Globally sample this many eligible rows with --seed instead of prefix truncation",
    )
    parser.add_argument(
        "--all_eligible",
        action="store_true",
        help=(
            "Build every globally eligible row after ASCII and metadata exclusions. "
            "Unlike legacy prefix mode, this performs the same preflight checks as "
            "--sample_rows before loading the codec."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--ascii_only",
        action="store_true",
        help="Exclude rows whose prompt or non-empty CoT label contains non-ASCII text",
    )
    parser.add_argument(
        "--exclude_metadata_jsonl",
        type=Path,
        action="append",
        default=[],
        help=(
            "Metadata whose original CrispEdit source rows must be excluded. "
            "May be provided multiple times."
        ),
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of independent parquet workers sharing output_root",
    )
    parser.add_argument(
        "--worker_index",
        type=int,
        default=0,
        help="This worker's zero-based parquet partition index",
    )
    parser.add_argument(
        "--skip_combine",
        action="store_true",
        help="Write atomic metadata shards only; combine them in a final --combine_only run",
    )
    parser.add_argument(
        "--combine_only",
        action="store_true",
        help="Verify and combine all expected metadata shards without loading the codec",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.codec_batch_size < 1:
        raise ValueError("codec_batch_size must be positive")
    if args.max_rows is not None and (args.sample_rows is not None or args.all_eligible):
        raise ValueError("--max_rows cannot be combined with global eligible selection")
    if args.sample_rows is not None and args.all_eligible:
        raise ValueError("--sample_rows and --all_eligible cannot be combined")
    if args.resume and args.max_rows is not None:
        raise ValueError("--resume and --max_rows cannot be combined")
    if (args.ascii_only or args.exclude_metadata_jsonl) and not (
        args.sample_rows is not None or args.all_eligible
    ):
        raise ValueError(
            "--ascii_only and --exclude_metadata_jsonl require --sample_rows or "
            "--all_eligible so filtering is applied globally"
        )
    if args.num_workers > 1 and not args.skip_combine:
        raise ValueError("Multi-worker construction requires --skip_combine")
    output_root = args.output_root.resolve()
    edit_mt_path = args.edit_mt_jsonl or output_root / "edit_mt.jsonl"
    edit_path = args.edit_jsonl or output_root / "edit.jsonl"
    shard_dir = output_root / "metadata_shards"

    mask_paths = sorted(args.mask_dir.glob("*.parquet"))
    if args.max_files is not None:
        mask_paths = mask_paths[: args.max_files]
    all_pairs = []
    for mask_path in mask_paths:
        raw_path = args.crispedit_dir / mask_path.name
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        all_pairs.append((raw_path, mask_path))

    if args.combine_only:
        mt_shards = [
            shard_dir / f"{mask_path.stem}.edit_mt.jsonl"
            for _, mask_path in all_pairs
        ]
        edit_shards = [
            shard_dir / f"{mask_path.stem}.edit.jsonl"
            for _, mask_path in all_pairs
        ]
        missing = [path for path in mt_shards + edit_shards if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Cannot combine: {len(missing)} metadata shards are missing; first={missing[0]}"
            )
        _combine_shards(mt_shards, edit_mt_path)
        _combine_shards(edit_shards, edit_path)
        print(
            json.dumps(
                {
                    "output_root": str(output_root),
                    "edit_mt_metadata": str(edit_mt_path),
                    "edit_metadata": str(edit_path),
                    "combined_parquet_shards": len(all_pairs),
                },
                indent=2,
            )
        )
        return

    selection_stats = Counter()
    excluded_source_ids, exclusion_stats = load_excluded_source_ids(
        args.exclude_metadata_jsonl
    )
    selected_by_file = None
    if args.sample_rows is not None or args.all_eligible:
        candidates, selection_stats = collect_candidates(
            all_pairs,
            args.ascii_only,
            excluded_source_ids=excluded_source_ids,
        )
        selected_candidates = (
            sample_candidates(candidates, args.sample_rows, args.seed)
            if args.sample_rows is not None
            else candidates
        )
        selected_by_file = defaultdict(set)
        for filename, row_idx, canonical_type in selected_candidates:
            selected_by_file[filename].add(row_idx)
            selection_stats[f"selected_type:{canonical_type}"] += 1
        selection_stats["selected"] = len(selected_candidates)
        selection_stats["selected_parquet_files"] = len(selected_by_file)
    pairs = partition_pairs(all_pairs, args.num_workers, args.worker_index)

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
            (
                selected_by_file.get(mask_path.name, set())
                if selected_by_file is not None
                else None
            ),
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

    if not args.skip_combine:
        _combine_shards(completed_mt, edit_mt_path)
        _combine_shards(completed_edit, edit_path)
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "edit_mt_metadata": str(edit_mt_path),
                "edit_metadata": str(edit_path),
                "combined": not args.skip_combine,
                "worker": {
                    "index": args.worker_index,
                    "count": args.num_workers,
                    "assigned_parquet_shards": len(pairs),
                },
                "selection": {
                    "mode": (
                        "global_random"
                        if args.sample_rows is not None
                        else "all_eligible"
                        if args.all_eligible
                        else "prefix"
                    ),
                    "seed": args.seed if args.sample_rows is not None else None,
                    "ascii_only": args.ascii_only,
                    "stats": dict(selection_stats),
                },
                "exclusion": {
                    "metadata": [str(path) for path in args.exclude_metadata_jsonl],
                    "stats": dict(exclusion_stats),
                },
                "stats": dict(totals),
                "important": "SAMTok codes were encoded on input/source images",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
