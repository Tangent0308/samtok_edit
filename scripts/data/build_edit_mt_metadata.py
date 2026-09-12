#!/usr/bin/env python3
"""Build refined CrispEdit ``edit_mt`` and ``edit_umt`` metadata.

The refined mask release stores one row for every original CrispEdit row.  Only
``filter_decision=keep`` rows with a non-empty raster mask are valid training
targets.  Empty/global/no-op CoT fallbacks are intentionally forbidden.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import re
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
    "CrispEdit-2M-mask"
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
        existing = path.read_bytes()
        if existing != data:
            raise ValueError(
                "Existing materialized image differs from the source parquet: "
                f"{path} (existing_sha256={hashlib.sha256(existing).hexdigest()}, "
                f"source_sha256={hashlib.sha256(data).hexdigest()})"
            )
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _unique_text(values):
    output = []
    seen = set()
    for value in values:
        value = sanitize_label(value or "")
        key = value.casefold()
        if value and key not in seen:
            seen.add(key)
            output.append(value)
    return output


def _label_for_row(mask_row: dict, canonical_type: str, instruction: str) -> str:
    """Derive one short English label for the union edit-region mask."""

    if canonical_type in {"style", "background"}:
        return canonical_type

    instances = mask_row.get("instance_masks") or []
    edit_instances = [
        item for item in instances if (item or {}).get("role") == "edit_region"
    ]
    preferred_side = "target" if canonical_type == "add" else "source"
    preferred = [
        item.get("ref")
        for item in edit_instances
        if item.get("grounding_image") == preferred_side
    ]
    fallback = [item.get("ref") for item in edit_instances]
    labels = _unique_text(preferred or fallback)

    if not labels:
        try:
            ground = json.loads(mask_row.get("ground_json") or "{}")
        except json.JSONDecodeError:
            ground = {}
        changes = ((ground.get("observation") or {}).get("parsed") or {}).get(
            "changes"
        ) or []
        key_order = (
            ("target_ref", "sam_ref", "source_ref")
            if canonical_type == "add"
            else ("source_ref", "sam_ref", "target_ref")
        )
        labels = _unique_text(
            change.get(key)
            for change in changes
            for key in key_order
            if isinstance(change, dict)
        )

    return sanitize_label(" and ".join(labels) if labels else instruction)


_UMT_STYLE_REF_RE = re.compile(
    r"\b(?:(?:(?:this|the|entire|whole)\s+)?"
    r"(?:image|photo(?:graph)?|picture)|this)\b",
    re.I,
)
_UMT_BACKGROUND_REF_RE = re.compile(r"\b(?:the\s+)?background\b", re.I)
_UMT_REPLACE_RE = re.compile(r"\breplace\s+(?P<ref>.+?)\s+with\b", re.I)
_UMT_COLOR_PATTERNS = [
    re.compile(r"\bchange\s+the\s+colou?r\s+of\s+(?P<ref>.+?)\s+to\b", re.I),
    re.compile(r"\b(?:turn|change|convert)\s+(?P<ref>.+?)\s+(?:into|to)\b", re.I),
]
_UMT_REMOVE_RE = re.compile(
    r"\b(?:remove|delete|erase)\s+(?P<ref>.+?)(?=\s*[.!?]?\s*$)", re.I
)
_UMT_ADD_LOCATION_RE = re.compile(
    r"\b(?P<prep>next\s+to|in\s+front\s+of|onto|near|beside|behind|within|inside|"
    r"across|along|around|at|on|to|in|by)\s+(?P<ref>.+?)(?=\s*[.!?]?\s*$)",
    re.I,
)
_MOTION_VERBS = (
    "adjusts|bends|changes|clasps|closes|crosses|extends|faces|gestures|holds|"
    "is|joins|leans|lifts|looks|lowers|moves|opens|picks|points|raises|reaches|"
    "rests|rotates|runs|shifts|sits|smiles|spreads|stands|stretches|swings|"
    "transitions|turns|walks|widens"
)
_UMT_MOTION_RE = re.compile(
    rf"^(?P<ref>(?:the|a|an)\s+.+?)\s+(?P<verb>{_MOTION_VERBS})\b", re.I
)


def _replace_group(prompt: str, match: re.Match, span: str, method: str):
    start, stop = match.span("ref") if "ref" in match.groupdict() else match.span()
    original = prompt[start:stop]
    rewritten = prompt[:start] + span + prompt[stop:]
    if rewritten.count("<|mt_start|>") != 1 or rewritten.count("<|mt_end|>") != 1:
        return None
    return rewritten, original, method


def build_umt_prompt(prompt: str, canonical_type: str, span: str):
    """Replace one syntactic edit-region reference with the mask span.

    Conservative templates are used instead of inventing a phrase.  Rows that
    cannot be transformed unambiguously remain valid ``edit_mt`` examples but
    are not emitted as ``edit_umt``.
    """

    if "<|mt_" in prompt:
        return None
    if canonical_type == "style":
        match = _UMT_STYLE_REF_RE.search(prompt)
        return _replace_group(prompt, match, span, "style_image_ref") if match else None
    if canonical_type == "background":
        match = _UMT_BACKGROUND_REF_RE.search(prompt)
        return _replace_group(prompt, match, span, "background_ref") if match else None
    if canonical_type == "replace":
        match = _UMT_REPLACE_RE.search(prompt)
        return _replace_group(prompt, match, span, "replace_object") if match else None
    if canonical_type == "color":
        for index, pattern in enumerate(_UMT_COLOR_PATTERNS):
            match = pattern.search(prompt)
            if match:
                return _replace_group(prompt, match, span, f"color_object_{index}")
        return None
    if canonical_type == "remove":
        match = _UMT_REMOVE_RE.search(prompt)
        return _replace_group(prompt, match, span, "remove_object") if match else None
    if canonical_type == "motion":
        match = _UMT_MOTION_RE.search(prompt)
        return _replace_group(prompt, match, span, "motion_subject") if match else None
    if canonical_type == "add":
        matches = list(_UMT_ADD_LOCATION_RE.finditer(prompt))
        if not matches:
            return None
        # The final locative complement is normally the placement/reference
        # region; earlier prepositions often belong to the added object itself.
        return _replace_group(prompt, matches[-1], span, "add_placement")
    return None


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
                "canonical_type",
                "mask_png",
                "mask_sum",
                "qc_flag",
                "instance_masks",
                "ground_json",
            ],
        ).to_pylist()
        for mask_row in mask_rows:
            if mask_row.get("filter_decision") != "keep":
                stats["filter_drop"] += 1
                continue
            if not mask_row.get("mask_png") or int(mask_row.get("mask_sum") or 0) <= 0:
                stats["empty_mask_drop"] += 1
                stats[f"empty_mask_qc:{mask_row.get('qc_flag')}"] += 1
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
            canonical_type = mask_row.get("canonical_type") or raw.get("type") or ""
            label = _label_for_row(mask_row, canonical_type, instruction)
            if ascii_only and (not instruction.isascii() or not label.isascii()):
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
        if not mask_row.get("mask_png") or int(mask_row.get("mask_sum") or 0) <= 0:
            stats["empty_mask_drop"] += 1
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

        canonical_type = mask_row.get("canonical_type") or raw.get("type") or ""
        mask = _decode_mask(mask_row.get("mask_png") or b"", source.size)
        if mask.sum() == 0:
            raise ValueError(
                f"Refined keep row has an empty decoded mask: {mask_path.name}:{row_idx}"
            )

        selected.append(
            {
                "source": source,
                "mask": mask,
                "label": _label_for_row(mask_row, canonical_type, raw["instruction"]),
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
                    "mask_sum": int(mask_row.get("mask_sum") or 0),
                },
            }
        )

    for start in range(0, len(selected), codec_batch_size):
        batch = selected[start : start + codec_batch_size]
        spans = codec.encode_single_batch(
            (row["source"], row["mask"]) for row in batch
        )
        for row, span in zip(batch, spans):
            row["mask_span"] = span
            row["mt_cot"] = to_cot([(span, make_labels(row["label"], 1)[0])])
            stats["encoded_mask"] += 1

    edit_mt_rows, edit_umt_rows = [], []
    for row in selected:
        edit_row = row["edit_row"]
        edit_mt_rows.append(
            {
                **edit_row,
                "mt_cot": row["mt_cot"],
                "sample_type": "edit_mt",
                "provenance": row["provenance"],
            }
        )
        transformed = build_umt_prompt(
            edit_row["prompt"], row["provenance"]["edit_type"], row["mask_span"]
        )
        if transformed is None:
            stats["umt_rewrite_drop"] += 1
            stats[f"umt_rewrite_drop_type:{row['provenance']['edit_type']}"] += 1
            continue
        umt_prompt, replaced_text, method = transformed
        provenance = {
            **row["provenance"],
            "original_prompt": edit_row["prompt"],
            "umt_replaced_text": replaced_text,
            "umt_rewrite_method": method,
        }
        edit_umt_rows.append(
            {
                "image": edit_row["image"],
                "edit_image": edit_row["edit_image"],
                "prompt": umt_prompt,
                "sample_type": "edit_umt",
                "provenance": provenance,
            }
        )
        stats["umt_emitted"] += 1
        stats[f"umt_emitted_type:{row['provenance']['edit_type']}"] += 1
        stats[f"umt_rewrite_method:{method}"] += 1
    stats["kept"] = len(selected)
    return edit_mt_rows, edit_umt_rows, stats


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
        "--edit_umt_jsonl", type=Path, default=None, help="Defaults under output_root"
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
    edit_umt_path = args.edit_umt_jsonl or output_root / "edit_umt.jsonl"
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
        edit_umt_shards = [
            shard_dir / f"{mask_path.stem}.edit_umt.jsonl"
            for _, mask_path in all_pairs
        ]
        missing = [path for path in mt_shards + edit_umt_shards if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Cannot combine: {len(missing)} metadata shards are missing; first={missing[0]}"
            )
        _combine_shards(mt_shards, edit_mt_path)
        _combine_shards(edit_umt_shards, edit_umt_path)
        print(
            json.dumps(
                {
                    "output_root": str(output_root),
                    "edit_mt_metadata": str(edit_mt_path),
                    "edit_umt_metadata": str(edit_umt_path),
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
    prepartitioned_pairs = None
    if args.sample_rows is not None or args.all_eligible:
        # Global random sampling must inspect the complete pool.  In all-eligible
        # mode, however, workers can scan only their own deterministic parquet
        # partition.  This avoids reading the lightweight columns eight times
        # before an eight-GPU build while producing the identical union.
        candidate_pairs = all_pairs
        if args.all_eligible and args.num_workers > 1:
            prepartitioned_pairs = partition_pairs(
                all_pairs, args.num_workers, args.worker_index
            )
            candidate_pairs = prepartitioned_pairs
        candidates, selection_stats = collect_candidates(
            candidate_pairs,
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
    work_pairs = (
        [pair for pair in all_pairs if pair[1].name in selected_by_file]
        if selected_by_file is not None
        else all_pairs
    )
    pairs = (
        prepartitioned_pairs
        if prepartitioned_pairs is not None
        else partition_pairs(work_pairs, args.num_workers, args.worker_index)
    )

    codec = SamtokCodec(
        args.sam2_ckpt,
        args.mask_tokenizer_ckpt,
        device=args.device,
        dtype=getattr(torch, args.dtype),
    )
    totals = Counter()
    completed_mt, completed_umt = [], []
    remaining = args.max_rows
    for raw_path, mask_path in tqdm(pairs, desc="CrispEdit parquet shards"):
        mt_shard = shard_dir / f"{mask_path.stem}.edit_mt.jsonl"
        umt_shard = shard_dir / f"{mask_path.stem}.edit_umt.jsonl"
        if args.resume and mt_shard.is_file() and umt_shard.is_file():
            completed_mt.append(mt_shard)
            completed_umt.append(umt_shard)
            totals["resumed_shards"] += 1
            continue
        edit_mt_rows, edit_umt_rows, stats = process_parquet_pair(
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
        _atomic_jsonl(edit_umt_rows, umt_shard)
        completed_mt.append(mt_shard)
        completed_umt.append(umt_shard)
        totals.update(stats)
        if remaining is not None:
            remaining -= len(edit_mt_rows)
            if remaining <= 0:
                break

    if not args.skip_combine:
        _combine_shards(completed_mt, edit_mt_path)
        _combine_shards(completed_umt, edit_umt_path)
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "edit_mt_metadata": str(edit_mt_path),
                "edit_umt_metadata": str(edit_umt_path),
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
