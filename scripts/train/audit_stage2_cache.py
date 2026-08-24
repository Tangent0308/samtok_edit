#!/usr/bin/env python3
"""Audit Stage-2 cache completeness, provenance, structure, dtype, and finiteness."""

import argparse
import hashlib
import io
import json
import multiprocessing
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch


_EXPECTED_TE_LORA = ""
_WORLD_SIZE = 1


def parse_counts(value):
    counts = Counter()
    for item in value.split(","):
        name, count = item.split(":", 1)
        counts[name] = int(count)
    return counts


def normalize_path(value):
    return os.path.abspath(os.path.expanduser(str(value)))


def visit_tensors(value, path, callback):
    if torch.is_tensor(value):
        callback(path, value)
    elif isinstance(value, dict):
        for key, item in value.items():
            visit_tensors(item, f"{path}.{key}", callback)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            visit_tensors(item, f"{path}[{index}]", callback)


def initialize_worker(expected_te_lora, world_size, torch_threads):
    global _EXPECTED_TE_LORA, _WORLD_SIZE
    _EXPECTED_TE_LORA = normalize_path(expected_te_lora)
    _WORLD_SIZE = world_size
    torch.set_num_threads(torch_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def discover_cache_tasks(cache_root, world_size, expected_rows):
    """Discover cache pairs with one directory scan per rank."""

    errors = []
    tasks = []
    cache_files = 0
    sidecar_files = 0
    expected_per_rank = expected_rows // world_size
    expected_ids = set(range(expected_per_rank))

    try:
        root_entries = list(os.scandir(cache_root))
    except OSError as error:
        return [], 0, 0, [f"cannot scan cache root {cache_root}: {error}"]
    extra_rank_dirs = sorted(
        entry.name
        for entry in root_entries
        if entry.is_dir(follow_symlinks=False)
        and entry.name.isdigit()
        and int(entry.name) not in range(world_size)
    )
    if extra_rank_dirs:
        errors.append(f"unexpected rank directories: {extra_rank_dirs}")

    for rank in range(world_size):
        rank_dir = cache_root / str(rank)
        pth_ids = set()
        json_ids = set()
        try:
            entries = os.scandir(rank_dir)
        except OSError as error:
            errors.append(f"cannot scan rank directory {rank_dir}: {error}")
            continue
        with entries:
            for entry in entries:
                suffix = Path(entry.name).suffix
                if suffix not in {".pth", ".json"}:
                    continue
                stem = Path(entry.name).stem
                if not stem.isdigit():
                    errors.append(f"non-numeric cache filename: {entry.path}")
                    continue
                local_id = int(stem)
                if suffix == ".pth":
                    pth_ids.add(local_id)
                    cache_files += 1
                else:
                    json_ids.add(local_id)
                    sidecar_files += 1

        missing_pth = sorted(expected_ids - pth_ids)
        missing_json = sorted(expected_ids - json_ids)
        extra_pth = sorted(pth_ids - expected_ids)
        extra_json = sorted(json_ids - expected_ids)
        if missing_pth:
            errors.append(
                f"rank {rank} missing {len(missing_pth)} pth files; "
                f"first={missing_pth[:10]}"
            )
        if missing_json:
            errors.append(
                f"rank {rank} missing {len(missing_json)} json files; "
                f"first={missing_json[:10]}"
            )
        if extra_pth:
            errors.append(
                f"rank {rank} has {len(extra_pth)} extra pth files; "
                f"first={extra_pth[:10]}"
            )
        if extra_json:
            errors.append(
                f"rank {rank} has {len(extra_json)} extra json files; "
                f"first={extra_json[:10]}"
            )

        for local_id in sorted(pth_ids):
            cache_path = rank_dir / f"{local_id}.pth"
            sidecar_path = rank_dir / f"{local_id}.json"
            tasks.append(
                (
                    len(tasks),
                    rank,
                    local_id,
                    str(cache_path),
                    str(sidecar_path),
                    cache_path.relative_to(cache_root).as_posix(),
                )
            )
    return tasks, cache_files, sidecar_files, errors


def audit_cache_pair(task):
    """Read each sidecar/cache pair once and return a compact audit result."""

    task_index, rank, local_id, cache_name, sidecar_name, relative_name = task
    cache_path = Path(cache_name)
    sidecar_path = Path(sidecar_name)
    errors = []
    sample_type = None
    metadata_index = None
    tensor_dtypes = Counter()
    tensor_paths = Counter()
    cache_bytes = 0
    file_sha256 = None

    try:
        record = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception as error:
        errors.append(f"cannot load sidecar {sidecar_path}: {error}")
        record = None
    if record is not None:
        sample_type = record.get("sample_type")
        try:
            metadata_index = int(record.get("metadata_index", -1))
        except (TypeError, ValueError):
            metadata_index = -1
            errors.append(f"bad metadata index: {sidecar_path}")
        if record.get("worker_rank") != rank:
            errors.append(f"worker rank mismatch: {sidecar_path}")
        if record.get("world_size") != _WORLD_SIZE:
            errors.append(f"world size mismatch: {sidecar_path}")
        recorded_lora = normalize_path(record.get("preset_te_lora_path", ""))
        if recorded_lora != _EXPECTED_TE_LORA:
            errors.append(f"TE LoRA mismatch: {sidecar_path}")
        summary_text = json.dumps(record.get("cache_summary"), separators=(",", ":"))
        if "samtok_cot_hidden" in summary_text or "samtok_cot_labels" in summary_text:
            errors.append(f"NTP-only tensor leaked into cache: {sidecar_path}")
        if '"finite":false' in summary_text:
            errors.append(f"sidecar reports non-finite tensor: {sidecar_path}")

    try:
        payload = cache_path.read_bytes()
        cache_bytes = len(payload)
        file_sha256 = hashlib.sha256(payload).hexdigest()
        cached_inputs = torch.load(
            io.BytesIO(payload), map_location="cpu", weights_only=False
        )
    except Exception as error:
        errors.append(f"cannot load {cache_path}: {error}")
        cached_inputs = None

    if cached_inputs is not None:
        if not isinstance(cached_inputs, tuple) or len(cached_inputs) != 3:
            errors.append(f"bad cache tuple: {cache_path}")
        else:
            shared, positive, negative = cached_inputs
            if not isinstance(shared, dict) or not {
                "latents",
                "input_latents",
                "edit_latents",
            }.issubset(shared):
                errors.append(f"missing shared latent: {cache_path}")
            if not isinstance(positive, dict) or "prompt_emb" not in positive:
                errors.append(f"missing positive prompt embedding: {cache_path}")
            if not isinstance(negative, dict) or "prompt_emb" not in negative:
                errors.append(f"missing negative prompt embedding: {cache_path}")
            forbidden = {"samtok_cot_hidden", "samtok_cot_labels"}
            for container_name, container in (
                ("shared", shared),
                ("positive", positive),
                ("negative", negative),
            ):
                if isinstance(container, dict) and forbidden.intersection(container):
                    errors.append(
                        f"NTP-only tensor in {container_name} cache: {cache_path}"
                    )

            def audit_tensor(path, tensor):
                dtype = str(tensor.dtype).removeprefix("torch.")
                tensor_dtypes[dtype] += 1
                tensor_paths[path] += 1
                if tensor.is_floating_point():
                    if tensor.dtype != torch.bfloat16:
                        errors.append(
                            f"unexpected floating dtype {cache_path}:{path}={dtype}"
                        )
                    if not bool(torch.isfinite(tensor).all().item()):
                        errors.append(f"non-finite tensor {cache_path}:{path}")

            for name, value in zip(
                ("shared", "positive", "negative"), cached_inputs
            ):
                visit_tensors(value, name, audit_tensor)

    return {
        "task_index": task_index,
        "rank": rank,
        "local_id": local_id,
        "relative_name": relative_name,
        "sample_type": sample_type,
        "metadata_index": metadata_index,
        "cache_bytes": cache_bytes,
        "file_sha256": file_sha256,
        "tensor_dtypes": dict(tensor_dtypes),
        "tensor_paths": dict(tensor_paths),
        "errors": errors,
    }


def format_duration(seconds):
    if seconds is None:
        return "unknown"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def log_progress(phase, completed, total, start_time, error_count, total_bytes):
    elapsed = max(time.monotonic() - start_time, 1e-9)
    rate = completed / elapsed
    eta = (total - completed) / rate if rate > 0 else None
    percent = 100.0 * completed / total if total else 100.0
    gib = total_bytes / (1024**3)
    print(
        f"[Stage2CacheAudit][{phase}] processed={completed}/{total} "
        f"percent={percent:.3f} rate={rate:.2f}_cache_per_s "
        f"read_gib={gib:.2f} elapsed={format_duration(elapsed)} "
        f"eta={format_duration(eta)} errors={error_count}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_root", type=Path, required=True)
    parser.add_argument("--expected_counts", default="edit_mt:16,edit:8")
    parser.add_argument("--world_size", type=int, default=8)
    parser.add_argument("--expected_te_lora", type=Path, required=True)
    parser.add_argument("--report_json", type=Path, required=True)
    parser.add_argument(
        "--workers", type=int, default=min(32, os.cpu_count() or 1)
    )
    parser.add_argument("--torch_threads_per_worker", type=int, default=1)
    parser.add_argument("--chunksize", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=500)
    parser.add_argument("--progress_interval", type=float, default=30.0)
    parser.add_argument(
        "--start_method", choices=("fork", "spawn", "forkserver"), default="fork"
    )
    args = parser.parse_args()

    if args.workers < 1 or args.torch_threads_per_worker < 1:
        raise ValueError("workers and torch_threads_per_worker must be positive")
    if args.log_every < 1 or args.progress_interval <= 0:
        raise ValueError("log_every and progress_interval must be positive")

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

    audit_start = time.monotonic()
    print(
        f"[Stage2CacheAudit][start] cache_root={args.cache_root} "
        f"expected_rows={expected_rows} workers={args.workers} "
        f"torch_threads_per_worker={args.torch_threads_per_worker} "
        f"chunksize={args.chunksize}",
        flush=True,
    )
    discovery_start = time.monotonic()
    tasks, cache_file_count, sidecar_file_count, discovery_errors = (
        discover_cache_tasks(args.cache_root, args.world_size, expected_rows)
    )
    print(
        f"[Stage2CacheAudit][discovery] cache_files={cache_file_count} "
        f"sidecar_files={sidecar_file_count} tasks={len(tasks)} "
        f"elapsed={format_duration(time.monotonic() - discovery_start)} "
        f"errors={len(discovery_errors)}",
        flush=True,
    )

    errors = list(discovery_errors)
    error_count = len(errors)
    max_reported_errors = 1000
    if cache_file_count != expected_rows:
        error_count += 1
        if len(errors) < max_reported_errors:
            errors.append(f"cache file count {cache_file_count} != {expected_rows}")
    if sidecar_file_count != expected_rows:
        error_count += 1
        if len(errors) < max_reported_errors:
            errors.append(
                f"sidecar file count {sidecar_file_count} != {expected_rows}"
            )

    global_counts = Counter()
    per_rank_counts = defaultdict(Counter)
    metadata_indices = []
    tensor_dtypes = Counter()
    tensor_paths = Counter()
    total_cache_bytes = 0
    file_hashes = [None] * len(tasks)
    completed = 0
    processing_start = time.monotonic()
    last_log_time = processing_start
    context = multiprocessing.get_context(args.start_method)
    with context.Pool(
        processes=args.workers,
        initializer=initialize_worker,
        initargs=(
            str(args.expected_te_lora),
            args.world_size,
            args.torch_threads_per_worker,
        ),
    ) as pool:
        for result in pool.imap_unordered(
            audit_cache_pair, tasks, chunksize=args.chunksize
        ):
            completed += 1
            sample_type = result["sample_type"]
            rank = result["rank"]
            metadata_index = result["metadata_index"]
            global_counts[sample_type] += 1
            per_rank_counts[rank][sample_type] += 1
            metadata_indices.append(metadata_index)
            tensor_dtypes.update(result["tensor_dtypes"])
            tensor_paths.update(result["tensor_paths"])
            total_cache_bytes += result["cache_bytes"]
            if result["file_sha256"] is not None:
                file_hashes[result["task_index"]] = (
                    result["relative_name"],
                    result["file_sha256"],
                )
            result_errors = result["errors"]
            error_count += len(result_errors)
            if len(errors) < max_reported_errors:
                errors.extend(
                    result_errors[: max_reported_errors - len(errors)]
                )
            now = time.monotonic()
            if (
                completed % args.log_every == 0
                or now - last_log_time >= args.progress_interval
                or completed == len(tasks)
            ):
                log_progress(
                    "cache",
                    completed,
                    len(tasks),
                    processing_start,
                    error_count,
                    total_cache_bytes,
                )
                last_log_time = now

    expected_indices = set(range(expected_rows))
    valid_metadata_indices = [
        index for index in metadata_indices if isinstance(index, int)
    ]
    observed_indices = set(valid_metadata_indices)
    if len(metadata_indices) != expected_rows or observed_indices != expected_indices:
        error_count += 1
        if len(errors) < max_reported_errors:
            errors.append("metadata indices are not exactly 0..N-1")
    if global_counts != expected_counts:
        error_count += 1
        if len(errors) < max_reported_errors:
            errors.append(
                f"global sample counts {global_counts} != {expected_counts}"
            )
    for rank in range(args.world_size):
        if per_rank_counts[rank] != expected_rank_counts:
            error_count += 1
            if len(errors) < max_reported_errors:
                errors.append(
                    f"rank {rank} sample counts {per_rank_counts[rank]} "
                    f"!= {expected_rank_counts}"
                )
    if any(item is None for item in file_hashes):
        error_count += 1
        if len(errors) < max_reported_errors:
            errors.append("one or more cache files have no SHA256")

    manifest_digest = hashlib.sha256()
    for item in file_hashes:
        if item is None:
            continue
        relative_name, file_sha256 = item
        manifest_digest.update(relative_name.encode("utf-8"))
        manifest_digest.update(b"\0")
        manifest_digest.update(bytes.fromhex(file_sha256))

    elapsed = time.monotonic() - audit_start
    report = {
        "passed": error_count == 0,
        "cache_root": str(args.cache_root),
        "cache_files": cache_file_count,
        "sidecar_files": sidecar_file_count,
        "total_cache_bytes": total_cache_bytes,
        "combined_cache_sha256": manifest_digest.hexdigest(),
        "combined_cache_sha256_algorithm": (
            "sha256(concat(relative_path, NUL, raw_file_sha256))"
        ),
        "metadata_indices_unique": len(observed_indices),
        "metadata_index_range": (
            [min(observed_indices), max(observed_indices)]
            if observed_indices
            else None
        ),
        "global_sample_type_counts": dict(global_counts),
        "per_rank_sample_type_counts": {
            str(rank): dict(per_rank_counts[rank])
            for rank in range(args.world_size)
        },
        "tensor_dtype_counts": dict(tensor_dtypes),
        "tensor_path_counts": dict(sorted(tensor_paths.items())),
        "expected_te_lora": normalize_path(args.expected_te_lora),
        "workers": args.workers,
        "torch_threads_per_worker": args.torch_threads_per_worker,
        "chunksize": args.chunksize,
        "elapsed_seconds": elapsed,
        "average_cache_per_second": completed / elapsed if elapsed else None,
        "error_count": error_count,
        "errors_truncated": error_count > len(errors),
        "errors": errors,
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = args.report_json.with_suffix(args.report_json.suffix + ".tmp")
    temporary_report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_report.replace(args.report_json)
    print(
        f"[Stage2CacheAudit][complete] passed={report['passed']} "
        f"processed={completed} elapsed={format_duration(elapsed)} "
        f"errors={error_count} report={args.report_json}",
        flush=True,
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if error_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
