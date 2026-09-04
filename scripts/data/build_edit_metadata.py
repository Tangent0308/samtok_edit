#!/usr/bin/env python3
"""Select fact-prefiltered plain ``edit`` rows and materialize exact images."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_edit_mt_metadata import (  # noqa: E402
    DEFAULT_CRISPEDIT,
    _atomic_jsonl,
    _combine_shards,
    _decode_image,
    _image_bytes,
    _write_bytes_once,
    canonical_source_parquet,
    load_excluded_source_ids,
    partition_pairs,
)


DEFAULT_FILTER_MANIFEST = Path(
    "/mnt/bn/strategy-mllm-train/user/tanyue/datasets/"
    "CrispEdit-2M-fact-prefilter/manifest"
)


def collect_candidates(
    raw_paths: list[Path],
    *,
    ascii_only: bool,
    excluded_source_ids: set[tuple[str, int]],
    deprioritized_source_ids: set[tuple[str, int]],
    filter_manifest_dir: Path,
) -> tuple[list[tuple[str, int, str, bool]], Counter]:
    """Collect lightweight candidates without decoding or materializing images."""

    candidates = []
    stats = Counter()
    for raw_path in tqdm(raw_paths, desc="Scanning plain-edit candidates"):
        manifest_path = filter_manifest_dir / raw_path.name
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest_rows = pq.read_table(
            manifest_path, columns=["row_idx", "filter_decision"]
        ).to_pylist()
        allowed = {
            int(row["row_idx"])
            for row in manifest_rows
            if row["filter_decision"] == "keep"
        }
        raw_rows = pq.read_table(raw_path, columns=["instruction", "type"]).to_pylist()
        if len(manifest_rows) != len(raw_rows):
            raise ValueError(
                f"Manifest/raw row-count mismatch for {raw_path.name}: "
                f"{len(manifest_rows)} != {len(raw_rows)}"
            )
        if any(int(row["row_idx"]) != index for index, row in enumerate(manifest_rows)):
            raise ValueError(f"Manifest row_idx is not aligned for {raw_path.name}")
        source_parquet = canonical_source_parquet(raw_path.name)
        for row_idx, raw in enumerate(raw_rows):
            stats["scanned"] += 1
            if row_idx not in allowed:
                stats["prefilter_drop"] += 1
                continue
            identity = (source_parquet, row_idx)
            if identity in excluded_source_ids:
                stats["excluded_source"] += 1
                continue
            instruction = raw.get("instruction") or ""
            if not instruction:
                stats["empty_prompt_drop"] += 1
                continue
            if ascii_only and not instruction.isascii():
                stats["non_ascii_drop"] += 1
                continue
            edit_type = raw.get("type") or ""
            deprioritized = identity in deprioritized_source_ids
            candidates.append((raw_path.name, row_idx, edit_type, deprioritized))
            stats["eligible"] += 1
            stats[f"eligible_type:{edit_type}"] += 1
            stats[
                "eligible_deprioritized" if deprioritized else "eligible_preferred"
            ] += 1
    return candidates, stats


def select_candidates(
    candidates: list[tuple[str, int, str, bool]],
    sample_rows: int | None,
    seed: int,
) -> list[tuple[str, int, str, bool]]:
    """Randomly select rows, exhausting non-deprioritized rows before fallback."""

    if sample_rows is None:
        return candidates[:]
    if sample_rows < 1:
        raise ValueError("sample_rows must be positive")
    if sample_rows > len(candidates):
        raise ValueError(
            f"Requested {sample_rows} rows from only {len(candidates)} eligible rows"
        )
    rng = random.Random(seed)
    preferred = [candidate for candidate in candidates if not candidate[3]]
    fallback = [candidate for candidate in candidates if candidate[3]]
    if sample_rows <= len(preferred):
        selected = rng.sample(preferred, sample_rows)
    else:
        selected = preferred + rng.sample(fallback, sample_rows - len(preferred))
        rng.shuffle(selected)
    return selected


def materialize_parquet(
    raw_path: Path,
    output_root: Path,
    selected_row_indices: set[int] | None,
    remaining_rows: int | None,
) -> tuple[list[dict], Counter]:
    raw_rows = pq.read_table(
        raw_path, columns=["input_img", "instruction", "output_img", "type"]
    ).to_pylist()
    rows = []
    stats = Counter()
    for row_idx, raw in enumerate(raw_rows):
        if selected_row_indices is not None and row_idx not in selected_row_indices:
            continue
        if remaining_rows is not None and len(rows) >= remaining_rows:
            break
        source_bytes = _image_bytes(raw["input_img"], raw_path.parent)
        target_bytes = _image_bytes(raw["output_img"], raw_path.parent)
        _, source_ext = _decode_image(source_bytes)
        _, target_ext = _decode_image(target_bytes)
        stem = raw_path.stem.replace(" ", "_")
        source_rel = Path("images") / stem / f"{row_idx:06d}_source.{source_ext}"
        target_rel = Path("images") / stem / f"{row_idx:06d}_target.{target_ext}"
        _write_bytes_once(output_root / source_rel, source_bytes)
        _write_bytes_once(output_root / target_rel, target_bytes)
        rows.append(
            {
                "image": target_rel.as_posix(),
                "edit_image": source_rel.as_posix(),
                "prompt": raw["instruction"],
                "sample_type": "edit",
                "provenance": {
                    "source_parquet": raw_path.name,
                    "row_idx": row_idx,
                    "edit_type": raw.get("type") or "",
                },
            }
        )
        stats["materialized"] += 1
        stats[f"materialized_type:{raw.get('type') or ''}"] += 1
    return rows, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crispedit_dir", type=Path, default=DEFAULT_CRISPEDIT)
    parser.add_argument(
        "--filter_manifest_dir", type=Path, default=DEFAULT_FILTER_MANIFEST
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=_REPO_ROOT / "data" / "crispedit_samtok",
    )
    parser.add_argument("--output_jsonl", type=Path, default=None)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument(
        "--sample_rows",
        type=int,
        default=None,
        help="Globally sample exactly this many eligible rows with --seed",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--ascii_only", action="store_true", help="Exclude rows with non-ASCII prompts"
    )
    parser.add_argument(
        "--exclude_metadata_jsonl",
        type=Path,
        action="append",
        default=[],
        help="Hard-exclude source identities in this metadata; may be repeated",
    )
    parser.add_argument(
        "--deprioritize_metadata_jsonl",
        type=Path,
        action="append",
        default=[],
        help=(
            "Prefer source identities absent from this metadata, then use the minimum "
            "random fallback needed to reach --sample_rows; may be repeated"
        ),
    )
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--worker_index", type=int, default=0)
    parser.add_argument("--skip_combine", action="store_true")
    parser.add_argument("--combine_only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.max_rows is not None and args.sample_rows is not None:
        raise ValueError("--max_rows and --sample_rows cannot be combined")
    if args.resume and args.max_rows is not None:
        raise ValueError("--resume and --max_rows cannot be combined")
    if args.num_workers > 1 and not args.skip_combine:
        raise ValueError("Multi-worker construction requires --skip_combine")
    if args.deprioritize_metadata_jsonl and args.sample_rows is None:
        raise ValueError("--deprioritize_metadata_jsonl requires --sample_rows")

    output_root = args.output_root.resolve()
    output_jsonl = args.output_jsonl or output_root / "edit_all.jsonl"
    shard_dir = output_root / "edit_metadata_shards"
    raw_paths = sorted(args.crispedit_dir.glob("*.parquet"))
    if args.max_files is not None:
        raw_paths = raw_paths[: args.max_files]

    if args.combine_only:
        shard_paths = [shard_dir / f"{path.stem}.edit.jsonl" for path in raw_paths]
        missing = [path for path in shard_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Cannot combine: {len(missing)} metadata shards are missing; first={missing[0]}"
            )
        _combine_shards(shard_paths, output_jsonl)
        print(
            json.dumps(
                {
                    "output_root": str(output_root),
                    "edit_metadata": str(output_jsonl),
                    "combined_parquet_shards": len(shard_paths),
                },
                indent=2,
            )
        )
        return

    hard_exclusions, hard_exclusion_stats = load_excluded_source_ids(
        args.exclude_metadata_jsonl
    )
    deprioritized, deprioritized_stats = load_excluded_source_ids(
        args.deprioritize_metadata_jsonl
    )
    selection_stats = Counter()
    candidates, selection_stats = collect_candidates(
        raw_paths,
        ascii_only=args.ascii_only,
        excluded_source_ids=hard_exclusions,
        deprioritized_source_ids=deprioritized,
        filter_manifest_dir=args.filter_manifest_dir,
    )
    if args.sample_rows is not None:
        selected = select_candidates(candidates, args.sample_rows, args.seed)
        selection_mode = "global_random"
    elif args.max_rows is not None:
        selected = candidates[: args.max_rows]
        selection_mode = "eligible_prefix"
    else:
        selected = candidates
        selection_mode = "all_eligible"
    selected_by_file = defaultdict(set)
    for filename, row_idx, edit_type, was_deprioritized in selected:
        selected_by_file[filename].add(row_idx)
        selection_stats[f"selected_type:{edit_type}"] += 1
        selection_stats[
            "selected_deprioritized" if was_deprioritized else "selected_preferred"
        ] += 1
    selection_stats["selected"] = len(selected)
    selection_stats["selected_parquet_files"] = len(selected_by_file)

    assigned_pairs = partition_pairs(
        [(path, path) for path in raw_paths], args.num_workers, args.worker_index
    )
    totals = Counter()
    completed = []
    for raw_path, _ in tqdm(assigned_pairs, desc="CrispEdit edit shards"):
        shard_path = shard_dir / f"{raw_path.stem}.edit.jsonl"
        if args.resume and shard_path.is_file():
            completed.append(shard_path)
            totals["resumed_shards"] += 1
            continue
        selected_indices = selected_by_file.get(raw_path.name, set())
        if selected_indices:
            rows, stats = materialize_parquet(
                raw_path,
                output_root,
                selected_indices,
                None,
            )
        else:
            rows, stats = [], Counter()
        _atomic_jsonl(rows, shard_path)
        completed.append(shard_path)
        totals.update(stats)

    if not args.skip_combine:
        _combine_shards(completed, output_jsonl)
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "edit_metadata": str(output_jsonl),
                "combined": not args.skip_combine,
                "worker": {
                    "index": args.worker_index,
                    "count": args.num_workers,
                    "assigned_parquet_shards": len(assigned_pairs),
                },
                "selection": {
                    "mode": selection_mode,
                    "seed": args.seed if args.sample_rows is not None else None,
                    "ascii_only": args.ascii_only,
                    "stats": dict(selection_stats),
                },
                "fact_prefilter_manifest": str(args.filter_manifest_dir),
                "hard_exclusion": {
                    "metadata": [str(path) for path in args.exclude_metadata_jsonl],
                    "stats": dict(hard_exclusion_stats),
                },
                "deprioritized": {
                    "metadata": [str(path) for path in args.deprioritize_metadata_jsonl],
                    "stats": dict(deprioritized_stats),
                },
                "stats": dict(totals),
                "dataset_base_path": str(output_root),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
