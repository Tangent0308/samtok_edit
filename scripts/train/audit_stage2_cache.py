#!/usr/bin/env python3
"""Audit Stage-2 cache completeness, provenance, structure, dtype, and finiteness."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch


def parse_counts(value):
    counts = Counter()
    for item in value.split(","):
        name, count = item.split(":", 1)
        counts[name] = int(count)
    return counts


def visit_tensors(value, path, callback):
    if torch.is_tensor(value):
        callback(path, value)
    elif isinstance(value, dict):
        for key, item in value.items():
            visit_tensors(item, f"{path}.{key}", callback)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            visit_tensors(item, f"{path}[{index}]", callback)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_root", type=Path, required=True)
    parser.add_argument("--expected_counts", default="edit_mt:16,edit:8")
    parser.add_argument("--world_size", type=int, default=8)
    parser.add_argument("--expected_te_lora", type=Path, required=True)
    parser.add_argument("--report_json", type=Path, required=True)
    args = parser.parse_args()

    expected_counts = parse_counts(args.expected_counts)
    expected_rows = sum(expected_counts.values())
    if expected_rows % args.world_size:
        raise ValueError("Expected cache rows must be divisible by world size")
    expected_rank_counts = Counter(
        {key: value // args.world_size for key, value in expected_counts.items()}
    )
    if any(
        expected_rank_counts[key] * args.world_size != value
        for key, value in expected_counts.items()
    ):
        raise ValueError("Every expected sample-type count must divide across ranks")

    cache_files = sorted(args.cache_root.glob("*/*.pth"))
    sidecar_files = sorted(args.cache_root.glob("*/*.json"))
    errors = []
    global_counts = Counter()
    per_rank_counts = defaultdict(Counter)
    metadata_indices = []
    tensor_dtypes = Counter()
    tensor_paths = Counter()
    total_cache_bytes = 0

    if len(cache_files) != expected_rows:
        errors.append(f"cache file count {len(cache_files)} != {expected_rows}")
    if len(sidecar_files) != expected_rows:
        errors.append(f"sidecar file count {len(sidecar_files)} != {expected_rows}")

    expected_lora = str(args.expected_te_lora.resolve())
    for sidecar_path in sidecar_files:
        record = json.loads(sidecar_path.read_text(encoding="utf-8"))
        rank = int(sidecar_path.parent.name)
        sample_type = record.get("sample_type")
        global_counts[sample_type] += 1
        per_rank_counts[rank][sample_type] += 1
        metadata_indices.append(int(record.get("metadata_index", -1)))
        if record.get("worker_rank") != rank:
            errors.append(f"worker rank mismatch: {sidecar_path}")
        if record.get("world_size") != args.world_size:
            errors.append(f"world size mismatch: {sidecar_path}")
        if str(Path(record.get("preset_te_lora_path", "")).resolve()) != expected_lora:
            errors.append(f"TE LoRA mismatch: {sidecar_path}")
        summary_text = json.dumps(record.get("cache_summary"))
        if "samtok_cot_hidden" in summary_text or "samtok_cot_labels" in summary_text:
            errors.append(f"NTP-only tensor leaked into cache: {sidecar_path}")
        if '"finite": false' in summary_text:
            errors.append(f"sidecar reports non-finite tensor: {sidecar_path}")

    for cache_path in cache_files:
        total_cache_bytes += cache_path.stat().st_size
        try:
            cached_inputs = torch.load(cache_path, map_location="cpu", weights_only=False)
        except Exception as error:
            errors.append(f"cannot load {cache_path}: {error}")
            continue
        if not isinstance(cached_inputs, tuple) or len(cached_inputs) != 3:
            errors.append(f"bad cache tuple: {cache_path}")
            continue
        shared, positive, negative = cached_inputs
        if not {"latents", "input_latents", "edit_latents"}.issubset(shared):
            errors.append(f"missing shared latent: {cache_path}")
        if "prompt_emb" not in positive or "prompt_emb" not in negative:
            errors.append(f"missing positive/negative prompt embedding: {cache_path}")

        def audit_tensor(path, tensor):
            dtype = str(tensor.dtype).removeprefix("torch.")
            tensor_dtypes[dtype] += 1
            tensor_paths[path] += 1
            if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                errors.append(f"non-finite tensor {cache_path}:{path}")
            if tensor.is_floating_point() and tensor.dtype != torch.bfloat16:
                errors.append(f"unexpected floating dtype {cache_path}:{path}={dtype}")

        for name, value in zip(("shared", "positive", "negative"), cached_inputs):
            visit_tensors(value, name, audit_tensor)

    expected_indices = set(range(expected_rows))
    if len(metadata_indices) != expected_rows or set(metadata_indices) != expected_indices:
        errors.append("metadata indices are not exactly 0..N-1")
    if global_counts != expected_counts:
        errors.append(f"global sample counts {global_counts} != {expected_counts}")
    for rank in range(args.world_size):
        if per_rank_counts[rank] != expected_rank_counts:
            errors.append(
                f"rank {rank} sample counts {per_rank_counts[rank]} != {expected_rank_counts}"
            )

    digest = hashlib.sha256()
    for cache_path in cache_files:
        digest.update(cache_path.relative_to(args.cache_root).as_posix().encode())
        with cache_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)

    report = {
        "passed": not errors,
        "cache_root": str(args.cache_root),
        "cache_files": len(cache_files),
        "sidecar_files": len(sidecar_files),
        "total_cache_bytes": total_cache_bytes,
        "combined_cache_sha256": digest.hexdigest(),
        "metadata_indices_unique": len(set(metadata_indices)),
        "metadata_index_range": (
            [min(metadata_indices), max(metadata_indices)] if metadata_indices else None
        ),
        "global_sample_type_counts": dict(global_counts),
        "per_rank_sample_type_counts": {
            str(rank): dict(per_rank_counts[rank]) for rank in range(args.world_size)
        },
        "tensor_dtype_counts": dict(tensor_dtypes),
        "tensor_path_counts": dict(sorted(tensor_paths.items())),
        "expected_te_lora": expected_lora,
        "errors": errors,
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
