"""Gradient diagnostics that permit connected, finite zero-gradient samples."""
from __future__ import annotations

import json
import math
import os
from collections import Counter
from pathlib import Path

import torch


def validate_gradient_record(record):
    """Zero is valid; missing gradients, nonfinite values and frozen grads are not."""
    if record.get("status", "ok") != "ok" or record.get("errors"):
        raise ValueError("Gradient audit recorded an error")
    for key in ("grad_norm_before_clip", "current_backward_grad_peak"):
        value = record.get(key)
        if not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid {key}")
    if record.get("trainable_grad_tensors", 0) <= 0:
        raise ValueError("Missing trainable gradients")
    # Read compatibility for historical diagnostics without the hook count.
    if record.get("current_backward_hook_tensors", 1) <= 0:
        raise ValueError("Current backward did not reach trainable parameters")
    if record.get("frozen_grad_tensors") != 0:
        raise ValueError("Frozen parameters received gradients")


def audit_backward(module):
    trainable = [p for p in module.parameters() if p.requires_grad]
    frozen = sum(p.grad is not None for p in module.parameters() if not p.requires_grad)
    norms = [p.grad.detach().float().norm() for p in trainable if p.grad is not None]
    peaks = module._branch_grad_peaks
    total = float(torch.stack(norms).norm()) if norms else 0.0
    peak = float(torch.stack(peaks).amax()) if peaks else 0.0
    current = module.pending_metrics[-1]
    zero = peak == 0 and bool(peaks)
    zero_reason = None
    if zero:
        zero_reason = ("fm_scheduler_weight_zero" if current.get("training_weight") == 0
                       else "finite_zero_backward")
    module._audit_microsteps = getattr(module, "_audit_microsteps", 0) + 1
    errors = []
    if not norms:
        errors.append("missing_trainable_gradients")
    if not peaks:
        errors.append("missing_current_backward_hooks")
    if not math.isfinite(total) or not math.isfinite(peak):
        errors.append("nonfinite_gradient")
    if frozen:
        errors.append("frozen_parameters_with_gradients")
    record = {
        "status": "error" if errors else "ok", "errors": errors,
        "microstep": module._audit_microsteps,
        "completed_optimizer_updates": module.completed_updates,
        "branch": current["_branch"], "row_sha256": current.get("_row_sha256"),
        "grad_norm_before_clip": total if math.isfinite(total) else None,
        "current_backward_grad_peak": peak if math.isfinite(peak) else None,
        "trainable_grad_tensors": len(norms),
        "current_backward_hook_tensors": len(peaks),
        "nonzero_grad_tensors": sum(float(n) > 0 for n in norms),
        "frozen_grad_tensors": frozen, "current_backward_zero": zero,
        "accumulated_grad_zero": total == 0, "zero_gradient_reason": zero_reason,
        **{key: current[key] for key in ("weighted_total", "timestep", "timestep_index",
                                        "timestep_before_cast", "training_weight",
                                        "bound_units", "loss_fm", "loss_ntp") if key in current},
    }
    # Publish failures too, before raising; JSON remains valid for NaN/Inf errors.
    with (Path(module.args.output) / f"gradients-rank{os.environ.get('RANK', '0')}.jsonl").open("a") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")
    if errors:
        raise RuntimeError("Invalid gradient audit: " + json.dumps(record, allow_nan=False))
    validate_gradient_record(record)
    current.update(zero_backward=int(zero), zero_weight_fm=int(current.get("training_weight") == 0))
    if zero_reason:
        current["_gradient_zero_reason"] = zero_reason
    return record


def audit_gradient_logs(directory, world_size, steps, accumulation, ratio):
    """Stream post-run diagnostics with the same validity rules as training."""
    report = {}
    for rank in range(world_size):
        count, zeros, reasons, branches = 0, 0, Counter(), Counter()
        path = Path(directory) / f"gradients-rank{rank}.jsonl"
        with path.open() as stream:
            for line in stream:
                record = json.loads(line)
                validate_gradient_record(record)
                count += 1
                if record.get("microstep", count) != count:
                    raise ValueError(f"{path}: reordered microstep")
                branches[record["branch"]] += 1
                if record["current_backward_grad_peak"] == 0:
                    zeros += 1
                    reasons[record.get("zero_gradient_reason") or "finite_zero_backward"] += 1
        if count != steps * accumulation or branches != {k: v * steps for k, v in ratio.items()}:
            raise ValueError(f"{path}: microstep count/branch ratio mismatch")
        report[str(rank)] = {"microsteps": count, "zero_backwards": zeros,
                             "zero_reasons": dict(reasons)}
    return report
