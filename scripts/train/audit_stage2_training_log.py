#!/usr/bin/env python3
"""Strictly audit a debug Stage-2 DiT-LoRA training run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import torch
from safetensors import safe_open


TYPE_TO_ID = {"edit_mt": 0, "edit": 1, "edit_umt": 2}
TARGET_FAMILIES = {
    "to_q",
    "to_k",
    "to_v",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_out.0",
    "to_add_out",
    "img_mlp.net.2",
    "img_mod.1",
    "txt_mlp.net.2",
    "txt_mod.1",
}
EXPECTED_UNUSED_ZERO_TENSORS = {
    "transformer_blocks.59.attn.add_q_proj.lora_B.default.weight",
    "transformer_blocks.59.attn.to_add_out.lora_B.default.weight",
    "transformer_blocks.59.txt_mlp.net.2.lora_B.default.weight",
}


def parse_counts(text: str) -> dict[str, int]:
    counts = {}
    for item in text.split(","):
        name, value = item.split(":", 1)
        counts[name.strip()] = int(value)
    if set(counts) != set(TYPE_TO_ID) or any(value <= 0 for value in counts.values()):
        raise ValueError(f"Expected positive counts for {sorted(TYPE_TO_ID)}; got {counts}")
    return counts


def describe(values) -> dict[str, float | int]:
    values = [float(value) for value in values]
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def require(condition: bool, message: str, errors: list[str]):
    if not condition:
        errors.append(message)


def extract_debug_records(text: str):
    records = defaultdict(list)
    for line in text.replace("\r", "\n").splitlines():
        marker = "[SamtokDebug]["
        start = line.find(marker)
        if start < 0:
            continue
        tag_end = line.find("]", start + len(marker))
        if tag_end < 0:
            continue
        tag = line[start + len(marker) : tag_end]
        payload = line[tag_end + 1 :].strip()
        records[tag].append(json.loads(payload))
    return records


def audit_checkpoint(path: Path, errors: list[str]) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)

    dtype_tensor_counts = Counter()
    zero_names = []
    nonfinite_names = []
    invalid_names = []
    parameter_count = 0
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        for name in keys:
            tensor = handle.get_tensor(name)
            parameter_count += tensor.numel()
            dtype_tensor_counts[str(tensor.dtype).removeprefix("torch.")] += 1
            if ".lora_A." not in name and ".lora_B." not in name:
                invalid_names.append(name)
            if not bool(torch.isfinite(tensor).all()):
                nonfinite_names.append(name)
            if not bool(torch.count_nonzero(tensor)):
                zero_names.append(name)

    require(len(keys) == 1440, f"checkpoint tensor count {len(keys)} != 1440", errors)
    require(
        parameter_count == 235_929_600,
        f"checkpoint parameter count {parameter_count} != 235929600",
        errors,
    )
    require(
        dtype_tensor_counts == {"bfloat16": 1440},
        f"checkpoint dtypes changed: {dict(dtype_tensor_counts)}",
        errors,
    )
    require(not invalid_names, f"non-LoRA checkpoint keys: {invalid_names[:5]}", errors)
    require(not nonfinite_names, f"non-finite checkpoint tensors: {nonfinite_names[:5]}", errors)
    require(
        set(zero_names) == EXPECTED_UNUSED_ZERO_TENSORS,
        f"unexpected all-zero checkpoint tensors: {zero_names}",
        errors,
    )
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "tensors": len(keys),
        "parameters": parameter_count,
        "dtype_tensor_counts": dict(dtype_tensor_counts),
        "invalid_key_count": len(invalid_names),
        "nonfinite_tensor_count": len(nonfinite_names),
        "zero_tensor_count": len(zero_names),
        "zero_tensor_names": zero_names,
    }


def audit_csv(
    path: Path,
    steps: list[dict],
    expected_steps: int,
    errors: list[str],
) -> dict:
    values = defaultdict(dict)
    row_count = 0
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            step = int(row["step"])
            key = row["key"]
            value = float(row["value"])
            row_count += 1
            require(math.isfinite(value), f"non-finite CSV value at {step}:{key}", errors)
            require(key not in values[step], f"duplicate CSV value at {step}:{key}", errors)
            values[step][key] = value

    expected_keys = {
        "loss",
        "loss_fm",
        "debug/grad_norm_before_clip",
        "debug/trainable_grad_tensors",
        "debug/nonzero_grad_tensors",
        "debug/frozen_grad_tensors",
        "debug/gradients_finite",
        "debug/optimizer_step",
        "debug/learning_rate_used",
        "debug/learning_rate_next",
        "debug/probe_update_l2_norm",
        "debug/sync_gradients",
    }
    require(
        sorted(values) == list(range(1, expected_steps + 1)),
        "CSV optimizer steps are missing, duplicated, or non-contiguous",
        errors,
    )
    for optimizer_step, record in enumerate(steps, start=1):
        row = values.get(optimizer_step, {})
        require(set(row) == expected_keys, f"CSV keys changed at step {optimizer_step}", errors)
        if row:
            require(
                abs(row["loss"] - row["loss_fm"]) <= 1e-12,
                f"loss != loss_fm at CSV step {optimizer_step}",
                errors,
            )
            require(
                abs(row["loss"] - float(record["rank_loss_fm"][0])) <= 1e-7,
                f"CSV loss does not match rank-0 debug loss at step {optimizer_step}",
                errors,
            )
    return {
        "rows": row_count,
        "keys": dict(
            Counter(key for step_values in values.values() for key in step_values)
        ),
        "rank0_loss_matches_debug": not any(
            "CSV loss does not match" in error for error in errors
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--expected_counts", default="edit_mt:16,edit:8,edit_umt:8")
    parser.add_argument("--expected_repeat", type=int, default=2)
    parser.add_argument("--expected_epochs", type=int, default=5)
    parser.add_argument("--world_size", type=int, default=8)
    parser.add_argument("--report_json", type=Path, required=True)
    args = parser.parse_args()

    expected_counts = parse_counts(args.expected_counts)
    expected_steps_per_epoch = (
        sum(expected_counts.values()) * args.expected_repeat // args.world_size
    )
    expected_steps = expected_steps_per_epoch * args.expected_epochs
    errors = []
    text = args.log_path.read_text(errors="replace")
    records = extract_debug_records(text)
    parameter_rows = records["stage2_parameter_audit"]
    runtime_rows = records["stage2_runtime_audit"]
    steps = records["stage2_step"]
    epoch_rows = records["stage2_epoch_audit"]

    require(len(parameter_rows) == 1, "expected one Stage-2 parameter audit", errors)
    require(len(runtime_rows) == 1, "expected one Stage-2 runtime audit", errors)
    require(len(steps) == expected_steps, f"debug step count {len(steps)} != {expected_steps}", errors)
    require(
        len(epoch_rows) == args.expected_epochs,
        f"epoch audit count {len(epoch_rows)} != {args.expected_epochs}",
        errors,
    )

    parameter_audit = parameter_rows[0] if parameter_rows else {}
    require(parameter_audit.get("trainable_tensors") == 1440, "trainable tensor count changed", errors)
    require(
        parameter_audit.get("trainable_parameters") == 235_929_600,
        "trainable parameter count changed",
        errors,
    )
    require(
        parameter_audit.get("trainable_parameter_dtypes") == {"bfloat16": 235_929_600},
        "Stage-2 trainable parameters are not all bf16",
        errors,
    )
    require(parameter_audit.get("dit_trainable_parameters") == 235_929_600, "DiT trainable boundary changed", errors)
    require(parameter_audit.get("text_encoder_trainable_parameters") == 0, "text encoder is trainable in Stage 2b", errors)
    require(parameter_audit.get("vae_trainable_parameters") == 0, "VAE is trainable in Stage 2b", errors)
    require(not parameter_audit.get("invalid_trainable_names"), "invalid trainable parameter names", errors)
    require(
        parameter_audit.get("target_family_trainable_tensors")
        == {name: 120 for name in sorted(TARGET_FAMILIES)},
        "official DiT LoRA target-family coverage changed",
        errors,
    )

    runtime = runtime_rows[0] if runtime_rows else {}
    expected_runtime = {
        "world_size": args.world_size,
        "physical_cache_rows": sum(expected_counts.values()),
        "physical_cache_type_counts": expected_counts,
        "dataset_repeat": args.expected_repeat,
        "dataset_rows_per_epoch": sum(expected_counts.values()) * args.expected_repeat,
        "micro_steps_per_rank_per_epoch": expected_steps_per_epoch,
        "num_epochs": args.expected_epochs,
        "total_optimizer_steps": expected_steps,
        "gradient_accumulation_steps": 1,
        "effective_global_batch_size": args.world_size,
        "pipeline_dtype": "bfloat16",
        "base_learning_rate": 1e-4,
        "weight_decay": 0.01,
        "zero_cond_t": True,
        "gradient_checkpointing": True,
        "find_unused_parameters": True,
        "gradient_clipping_enabled": False,
    }
    for key, expected in expected_runtime.items():
        require(runtime.get(key) == expected, f"runtime {key}={runtime.get(key)!r} != {expected!r}", errors)
    require(runtime.get("optimizer_betas") == [0.9, 0.999], "AdamW betas changed", errors)
    require(runtime.get("scheduler") == "ConstantLR", "Stage-2 scheduler changed", errors)

    losses = []
    losses_by_type = defaultdict(list)
    timesteps = []
    grad_norms = []
    probe_updates = []
    probe_norms = []
    learning_rates = []
    trainable_grad_counts = Counter()
    nonzero_grad_counts = Counter()
    zero_loss_positions = []
    epoch_type_counts = defaultdict(Counter)
    epoch_source_counts = defaultdict(Counter)
    id_to_type = {value: key for key, value in TYPE_TO_ID.items()}
    expected_shape_dtypes = {
        "input_latents_dtype": "bfloat16",
        "noise_dtype": "bfloat16",
        "noise_pred_dtype": "bfloat16",
        "training_target_dtype": "bfloat16",
        "loss_fm_dtype": "float32",
    }
    for index, record in enumerate(steps, start=1):
        require(record.get("optimizer_step") == index, f"optimizer step mismatch at {index}", errors)
        epoch = int(record.get("epoch", -1))
        require(0 <= epoch < args.expected_epochs, f"invalid epoch at step {index}", errors)
        require(
            record.get("micro_step_in_epoch") == (index - 1) % expected_steps_per_epoch + 1,
            f"micro-step position mismatch at optimizer step {index}",
            errors,
        )
        for key, expected in expected_shape_dtypes.items():
            require(record.get(key) == expected, f"{key} changed at step {index}", errors)
        shapes = {
            tuple(record.get("input_latents_shape", [])),
            tuple(record.get("noise_pred_shape", [])),
            tuple(record.get("training_target_shape", [])),
        }
        require(len(shapes) == 1 and () not in shapes, f"FM tensor shapes differ at step {index}", errors)
        require(record.get("gradients_finite") is True, f"non-finite gradient at step {index}", errors)
        require(record.get("frozen_grad_tensors") == 0, f"frozen gradient at step {index}", errors)
        require(record.get("nonzero_grad_tensors", 0) > 0, f"zero trainable gradients at step {index}", errors)
        require(record.get("sync_gradients") == 1, f"unexpected accumulation state at step {index}", errors)

        type_ids = record.get("rank_sample_type_ids", [])
        source_ids = record.get("rank_source_row_ids", [])
        rank_losses = record.get("rank_loss_fm", [])
        rank_timesteps = record.get("rank_timesteps", [])
        rank_grad_norms = record.get("rank_grad_norm_before_clip", [])
        rank_updates = record.get("rank_probe_update_l2_norm", [])
        rank_norms = record.get("rank_probe_parameter_l2_norm", [])
        for name, values in {
            "types": type_ids,
            "sources": source_ids,
            "losses": rank_losses,
            "timesteps": rank_timesteps,
            "grad_norms": rank_grad_norms,
            "probe_updates": rank_updates,
            "probe_norms": rank_norms,
        }.items():
            require(len(values) == args.world_size, f"rank {name} length mismatch at step {index}", errors)
        require(all(math.isfinite(float(value)) and float(value) >= 0 for value in rank_losses), f"bad FM loss at step {index}", errors)
        require(all(math.isfinite(float(value)) for value in rank_timesteps), f"bad timestep at step {index}", errors)
        require(all(math.isfinite(float(value)) for value in rank_grad_norms), f"bad gradient norm at step {index}", errors)
        require(all(float(value) > 0 for value in rank_updates), f"zero/non-finite probe update at step {index}", errors)
        require(
            rank_norms and max(rank_norms) - min(rank_norms) <= 1e-5 * max(1.0, max(rank_norms)),
            f"DDP probe parameters diverged at step {index}",
            errors,
        )
        for type_id, source_id, loss, timestep in zip(type_ids, source_ids, rank_losses, rank_timesteps):
            sample_type = id_to_type.get(int(type_id))
            require(sample_type is not None, f"invalid type id at step {index}", errors)
            if sample_type is None:
                continue
            epoch_type_counts[epoch][sample_type] += 1
            epoch_source_counts[epoch][int(source_id)] += 1
            losses.append(float(loss))
            losses_by_type[sample_type].append(float(loss))
            timesteps.append(float(timestep))
            if float(loss) == 0:
                zero_loss_positions.append(
                    {"optimizer_step": index, "metadata_index": int(source_id), "timestep": float(timestep)}
                )
        grad_norms.append(float(record["grad_norm_before_clip"]))
        probe_updates.append(float(record["probe_update_l2_norm"]))
        probe_norms.append(float(record["rank_probe_parameter_l2_norm"][0]))
        learning_rates.append(float(record["learning_rate_used"]))
        trainable_grad_counts[int(record["trainable_grad_tensors"])] += 1
        nonzero_grad_counts[int(record["nonzero_grad_tensors"])] += 1

    expected_epoch_types = Counter(
        {name: count * args.expected_repeat for name, count in expected_counts.items()}
    )
    for epoch in range(args.expected_epochs):
        require(epoch_type_counts[epoch] == expected_epoch_types, f"type counts changed in epoch {epoch}", errors)
        require(
            set(epoch_source_counts[epoch]) == set(range(sum(expected_counts.values())))
            and set(epoch_source_counts[epoch].values()) == {args.expected_repeat},
            f"cache coverage changed in epoch {epoch}",
            errors,
        )
    for epoch, record in enumerate(epoch_rows):
        require(record.get("epoch") == epoch, f"epoch audit ordering changed at {epoch}", errors)
        require(record.get("sample_type_counts") == dict(expected_epoch_types), f"epoch audit types changed at {epoch}", errors)
        require(record.get("unique_metadata_rows") == sum(expected_counts.values()), f"epoch audit coverage changed at {epoch}", errors)
        require(record.get("uses_per_metadata_row") == [args.expected_repeat], f"epoch audit repeat changed at {epoch}", errors)

    if learning_rates:
        require(abs(learning_rates[0] - 1e-4 / 3) < 1e-10, "first ConstantLR value changed", errors)
        require(all(abs(value - 1e-4) < 1e-10 for value in learning_rates[1:]), "post-warmup ConstantLR values changed", errors)

    checkpoint = args.checkpoint or args.output_path / f"step-{expected_steps}.safetensors"
    require(checkpoint.is_file(), f"missing final checkpoint: {checkpoint}", errors)
    checkpoint_report = audit_checkpoint(checkpoint, errors) if checkpoint.is_file() else {}
    csv_path = args.output_path / "loss.csv"
    require(csv_path.is_file(), f"missing loss CSV: {csv_path}", errors)
    csv_report = audit_csv(csv_path, steps, expected_steps, errors) if csv_path.is_file() else {}

    error_patterns = {
        "traceback": r"(?i)traceback",
        "oom": r"(?i)out of memory",
        "cuda_error": r"(?i)cuda (?:runtime )?error",
        "nccl_error": r"(?i)nccl[^\n\r]*error",
        "nan_token": r"(?i)(?<![A-Za-z])nan(?![A-Za-z])",
        "segfault": r"(?i)segmentation fault",
    }
    error_pattern_counts = {
        name: len(re.findall(pattern, text)) for name, pattern in error_patterns.items()
    }
    require(not any(error_pattern_counts.values()), f"training log error patterns: {error_pattern_counts}", errors)
    require(text.count("Destroy COMPLETE") == args.world_size, "not every NCCL rank exited cleanly", errors)
    require("Waiting for W&B process to finish... (success)." in text, "W&B did not finish successfully", errors)
    synced = re.findall(r"wandb: Synced .*?: (https?://\S+)", text)
    local_runs = sorted((args.output_path / "wandb_log" / "wandb").glob("run-*"))

    report = {
        "passed": not errors,
        "errors": errors,
        "log_path": str(args.log_path.resolve()),
        "output_path": str(args.output_path.resolve()),
        "expected_counts": expected_counts,
        "parameter_audit": parameter_audit,
        "runtime_audit": runtime,
        "optimizer_steps": len(steps),
        "epoch_audits": epoch_rows,
        "rank_level_fm_loss": describe(losses),
        "rank_level_fm_loss_by_type": {
            name: describe(values) for name, values in sorted(losses_by_type.items())
        },
        "timestep": describe(timesteps),
        "grad_norm_before_clip": describe(grad_norms),
        "probe_update_l2_norm": describe(probe_updates),
        "probe_parameter_l2_norm_first_last": (
            [probe_norms[0], probe_norms[-1]] if probe_norms else []
        ),
        "learning_rate_used": learning_rates,
        "trainable_grad_tensor_count_frequency": dict(trainable_grad_counts),
        "nonzero_grad_tensor_count_frequency": dict(nonzero_grad_counts),
        "zero_loss_positions": zero_loss_positions,
        "checkpoint": checkpoint_report,
        "csv": csv_report,
        "log_error_pattern_counts": error_pattern_counts,
        "wandb": {
            "finish_success": "Waiting for W&B process to finish... (success)." in text,
            "url": synced[-1] if synced else None,
            "local_dir": str(local_runs[-1]) if local_runs else None,
        },
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report_json.with_suffix(args.report_json.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.report_json)
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
