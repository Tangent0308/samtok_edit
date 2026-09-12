#!/usr/bin/env python3
"""Turn a debug Stage-1 training log into a strict, machine-readable audit."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import re
from pathlib import Path


MARKER = re.compile(r"\[SamtokDebug\]\[(?P<kind>[^]]+)\] (?P<payload>\{.*\})")
TYPE_IDS = {"edit_mt": 0, "edit_ntp": 1, "edit": 2, "edit_umt": 3}
EXPECTED_LOSSES = {
    "edit_mt": {"loss_ntp", "loss_fm"},
    "edit_ntp": {"loss_ntp"},
    "edit": {"loss_fm"},
    "edit_umt": {"loss_fm"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_log(path: Path) -> dict[str, list[dict]]:
    records: dict[str, list[dict]] = defaultdict(list)
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, 1):
            match = MARKER.search(line)
            if not match:
                continue
            try:
                payload = json.loads(match.group("payload"))
            except json.JSONDecodeError as error:
                raise ValueError(f"Bad debug JSON at line {line_number}: {error}") from error
            records[match.group("kind")].append(payload)
    return records


def add_check(checks: dict[str, bool], errors: list[str], name: str, passed: bool):
    checks[name] = bool(passed)
    if not passed:
        errors.append(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--report_json", type=Path, required=True)
    parser.add_argument("--expected_world_size", type=int, default=8)
    parser.add_argument("--expected_accumulation_steps", type=int, default=8)
    parser.add_argument("--expected_repeat", type=int, default=2)
    parser.add_argument("--expected_source_rows", type=int, default=128)
    parser.add_argument("--wandb_run_id", default=None)
    parser.add_argument("--wandb_url", default=None)
    args = parser.parse_args()

    records = parse_log(args.log)
    micro = records.get("micro_step", [])
    parameter = records.get("parameter_audit", [])
    runtime = records.get("runtime_audit", [])
    checks: dict[str, bool] = {}
    errors: list[str] = []

    add_check(checks, errors, "one_parameter_audit", len(parameter) == 1)
    add_check(checks, errors, "one_runtime_audit", len(runtime) == 1)
    runtime_row = runtime[0] if len(runtime) == 1 else {}
    parameter_row = parameter[0] if len(parameter) == 1 else {}
    expected_micro = int(runtime_row.get("micro_steps_per_rank", -1))
    add_check(checks, errors, "all_micro_steps_logged", len(micro) == expected_micro)
    add_check(
        checks,
        errors,
        "world_size",
        runtime_row.get("world_size") == args.expected_world_size,
    )
    add_check(
        checks,
        errors,
        "accumulation_steps",
        runtime_row.get("gradient_accumulation_steps")
        == args.expected_accumulation_steps,
    )
    add_check(
        checks,
        errors,
        "schedule_size",
        runtime_row.get("dataset_schedule_rows_global")
        == args.expected_source_rows * args.expected_repeat,
    )
    add_check(
        checks,
        errors,
        "configured_type_ratio",
        runtime_row.get("sample_type_ratio")
        == "edit_mt:4,edit_ntp:2,edit:1,edit_umt:1",
    )
    add_check(checks, errors, "pipeline_bfloat16", runtime_row.get("pipeline_dtype") == "bfloat16")
    add_check(
        checks,
        errors,
        "loss_weights",
        runtime_row.get("ntp_loss_weight") == 0.05
        and runtime_row.get("fm_loss_weight") == 1.0,
    )
    add_check(checks, errors, "learning_rate", runtime_row.get("learning_rate") == 4e-5)
    add_check(checks, errors, "weight_decay", runtime_row.get("weight_decay") == 0.05)
    add_check(checks, errors, "max_grad_norm", runtime_row.get("max_grad_norm") == 1.0)
    add_check(
        checks,
        errors,
        "te_only_trainable",
        parameter_row.get("text_encoder_trainable_parameters", 0) > 0
        and parameter_row.get("dit_trainable_parameters") == 0
        and parameter_row.get("vae_trainable_parameters") == 0
        and not parameter_row.get("invalid_trainable_names"),
    )
    add_check(
        checks,
        errors,
        "trainable_fp32",
        set(parameter_row.get("trainable_parameter_dtypes", {})) == {"float32"},
    )
    add_check(
        checks,
        errors,
        "frozen_bfloat16",
        set(parameter_row.get("frozen_parameter_dtypes", {})) == {"bfloat16"},
    )

    local_counts = Counter(row.get("sample_type") for row in micro)
    global_counts = Counter()
    component_values: dict[str, list[float]] = defaultdict(list)
    type_totals: dict[str, list[float]] = defaultdict(list)
    window_counts: list[dict[str, int]] = []
    all_finite = True
    loss_dispatch = True
    ddp_types = True
    provenance = True
    targets = True
    cot_routing = True
    umt_spans = True
    ntp_alignment = True
    precision = True
    gradient_routing = True
    sync_pattern = True
    rank_parameter_sync = True
    positive_updates = []

    for offset in range(0, len(micro), args.expected_accumulation_steps):
        block = micro[offset : offset + args.expected_accumulation_steps]
        window_counts.append(dict(Counter(row.get("sample_type") for row in block)))
    expected_window = {"edit_mt": 4, "edit_ntp": 2, "edit": 1, "edit_umt": 1}
    add_check(
        checks,
        errors,
        "ratio_per_accumulation_window",
        all(item == expected_window for item in window_counts),
    )

    for index, row in enumerate(micro, 1):
        sample_type = row.get("sample_type")
        rank_types = row.get("rank_sample_type_ids", [])
        global_counts[sample_type] += len(rank_types)
        expected_id = TYPE_IDS.get(sample_type)
        ddp_types &= len(rank_types) == args.expected_world_size and set(
            rank_types
        ) == {expected_id}
        provenance &= (
            row.get("ddp_sample_type_consistent") == 1
            and row.get("ddp_schedule_group_consecutive") == 1
            and len(row.get("rank_source_row_ids", [])) == args.expected_world_size
        )

        observed_losses = set()
        if row.get("has_ntp_loss"):
            observed_losses.add("loss_ntp")
        if row.get("has_fm_loss"):
            observed_losses.add("loss_fm")
        loss_dispatch &= observed_losses == EXPECTED_LOSSES.get(sample_type)
        loss_total = row.get("loss_total")
        all_finite &= isinstance(loss_total, (int, float)) and math.isfinite(loss_total)
        type_totals[sample_type].extend(row.get("rank_loss_total", []))
        for component, values in row.get("rank_component_losses", {}).items():
            component_values[f"{sample_type}.{component}"].extend(values)
        expected_total = (row.get("loss_ntp") or 0.0) * runtime_row.get(
            "ntp_loss_weight", 0.0
        ) + (row.get("loss_fm") or 0.0) * runtime_row.get(
            "fm_loss_weight", 0.0
        )
        all_finite &= abs(loss_total - expected_total) <= 2e-6
        all_finite &= row.get("loss_identity_error", float("inf")) <= 2e-8

        targets &= row.get("conditioning_is_metadata_edit_image") is True
        targets &= row.get("fm_target_is_metadata_image") is (sample_type != "edit_ntp")
        if sample_type in {"edit_mt", "edit_ntp"}:
            cot_routing &= row.get("cot_tokens", 0) > 0
            cot_routing &= row.get("user_mask_span_count") == 0
            ntp_alignment &= row.get("ntp_shift_alignment_ok") is True
        else:
            cot_routing &= row.get("cot_tokens") == 0
            cot_routing &= (
                row.get("cot_hidden_shape") is None
                and row.get("cot_label_shape") is None
            )
        if sample_type == "edit_umt":
            spans = row.get("user_mask_span_token_ids", [])
            umt_spans &= (
                row.get("user_mask_span_count") == 1
                and len(spans) == 1
                and len(spans[0]) == 4
            )
            umt_spans &= (
                row.get("user_mask_spans_atomic") is True
                and row.get("user_mask_spans_in_template") is True
            )
        elif sample_type != "edit_umt":
            umt_spans &= row.get("user_mask_span_count") == 0

        precision &= (
            row.get("loss_total_dtype") == "float32"
            and row.get("prompt_emb_dtype") == "bfloat16"
        )
        if row.get("has_ntp_loss"):
            precision &= (
                row.get("loss_ntp_dtype") == "float32"
                and row.get("cot_hidden_dtype") == "bfloat16"
                and row.get("cot_label_dtype") == "int64"
            )
        if row.get("has_fm_loss"):
            precision &= (
                row.get("loss_fm_dtype") == "float32"
                and row.get("target_latent_dtype") == "bfloat16"
            )

        gradient_routing &= (
            row.get("gradients_finite") is True
            and row.get("nonzero_grad_tensors", 0) > 0
            and row.get("frozen_grad_tensors") == 0
        )
        slot = (index - 1) % args.expected_accumulation_steps + 1
        should_sync = slot == args.expected_accumulation_steps
        sync_pattern &= (
            row.get("accumulation_slot") == slot
            and bool(row.get("sync_gradients")) is should_sync
        )
        if should_sync:
            update = row.get("probe_update_l2_norm")
            positive_updates.append(update)
            norms = row.get("rank_probe_parameter_l2_norm", [])
            updates = row.get("rank_probe_update_l2_norm", [])
            rank_parameter_sync &= (
                len(norms) == args.expected_world_size
                and max(norms) - min(norms) <= 1e-7
            )
            rank_parameter_sync &= (
                len(updates) == args.expected_world_size
                and max(updates) - min(updates) <= 1e-7
            )
        else:
            sync_pattern &= row.get("probe_update_l2_norm") is None

    expected_global = {
        "edit_mt": args.expected_source_rows * args.expected_repeat // 2,
        "edit_ntp": args.expected_source_rows * args.expected_repeat // 4,
        "edit": args.expected_source_rows * args.expected_repeat // 8,
        "edit_umt": args.expected_source_rows * args.expected_repeat // 8,
    }
    add_check(checks, errors, "global_type_counts", dict(global_counts) == expected_global)
    add_check(checks, errors, "ddp_type_homogeneity", ddp_types)
    add_check(checks, errors, "ddp_provenance", provenance)
    add_check(checks, errors, "loss_dispatch", loss_dispatch)
    add_check(checks, errors, "loss_finite_and_weighted_identity", all_finite)
    add_check(checks, errors, "source_and_target_routing", targets)
    add_check(checks, errors, "cot_routing", cot_routing)
    add_check(checks, errors, "umt_atomic_prompt_span", umt_spans)
    add_check(checks, errors, "ntp_shift_alignment", ntp_alignment)
    add_check(checks, errors, "precision", precision)
    add_check(checks, errors, "gradient_routing", gradient_routing)
    add_check(checks, errors, "gradient_accumulation_sync_pattern", sync_pattern)
    add_check(checks, errors, "rank_parameter_sync", rank_parameter_sync)
    add_check(
        checks,
        errors,
        "every_optimizer_step_updates_lora",
        len(positive_updates) == runtime_row.get("optimizer_steps")
        and all(value is not None and value > 0 for value in positive_updates),
    )

    checkpoints = sorted(args.output_dir.glob("step-*.safetensors"))
    add_check(
        checks,
        errors,
        "checkpoint_written",
        len(checkpoints) == 1 and checkpoints[0].stat().st_size > 0,
    )
    loss_csv = args.output_dir / "loss.csv"
    training_args = args.output_dir / "training_args.json"
    add_check(checks, errors, "artifacts_written", loss_csv.is_file() and training_args.is_file())

    def summary(values: list[float]) -> dict[str, float | int | None]:
        return {
            "count": len(values),
            "mean": sum(values) / len(values) if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }

    report = {
        "passed": not errors,
        "errors": errors,
        "checks": checks,
        "log": str(args.log),
        "output_dir": str(args.output_dir),
        "debug_records": {key: len(value) for key, value in records.items()},
        "runtime_audit": runtime_row,
        "parameter_audit": parameter_row,
        "local_micro_step_type_counts": dict(local_counts),
        "global_sample_type_counts": dict(global_counts),
        "accumulation_window_type_counts": window_counts,
        "loss_total_by_type": {key: summary(value) for key, value in sorted(type_totals.items())},
        "component_losses": {
            key: summary(value) for key, value in sorted(component_values.items())
        },
        "optimizer_probe_update_l2_norms": positive_updates,
        "max_loss_identity_error": max(
            (row.get("loss_identity_error", 0.0) for row in micro), default=None
        ),
        "checkpoint": (
            {
                "path": str(checkpoints[0]),
                "bytes": checkpoints[0].stat().st_size,
                "sha256": sha256(checkpoints[0]),
            }
            if len(checkpoints) == 1
            else None
        ),
        "wandb": {
            "run_id": args.wandb_run_id,
            "url": args.wandb_url,
        },
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report_json.with_suffix(args.report_json.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.report_json)
    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
