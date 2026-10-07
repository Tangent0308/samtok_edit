"""Compare a cached-input and an on-the-fly Stage 2 run of the same schedule.

Everything that depends on the training computation must be identical:
per-update metrics (loss, timestep, sampled rows), optimizer log, per-rank
gradient logs, the trainable-parameter hash on every rank, every checkpoint
and the final adapter with its recorded conditioning identity.  Only wall
time and peak memory may differ.

    python scripts/diagnostics/compare_stage2_runs.py <cached run dir> <on-the-fly run dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file


def records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def same_tensors(a, b):
    first, second = load_file(str(a)), load_file(str(b))
    return set(first) == set(second) and all(torch.equal(first[k], second[k]) for k in first), len(first)


def compare(cached, online):
    cached, online = Path(cached), Path(online)
    checks = {}
    for name in ("training_metrics.jsonl", "optimizer_steps.jsonl"):
        checks[name] = records(cached / name) == records(online / name)
    gradient_logs = sorted(p.name for p in cached.glob("gradients-rank*.jsonl"))
    checks["gradient_logs"] = bool(gradient_logs) and gradient_logs == sorted(
        p.name for p in online.glob("gradients-rank*.jsonl")) and all(
        records(cached / n) == records(online / n) for n in gradient_logs)
    ranks = [json.loads((d / "rank_parameters.json").read_text()) for d in (cached, online)]
    checks["rank_parameter_hashes"] = [(r["rank"], r["sha256"], r["trainable_parameters"]) for r in ranks[0]] == \
        [(r["rank"], r["sha256"], r["trainable_parameters"]) for r in ranks[1]]
    steps = sorted(p.name for p in cached.glob("step-*.safetensors"))
    checks["checkpoints"] = bool(steps) and steps == sorted(p.name for p in online.glob("step-*.safetensors")) and all(
        same_tensors(cached / n, online / n)[0] for n in steps)
    equal, tensors = same_tensors(cached / "adapter/adapter.safetensors", online / "adapter/adapter.safetensors")
    checks["adapter_tensors"] = equal
    configs = [json.loads((d / "adapter/adapter.json").read_text()) for d in (cached, online)]
    checks["adapter_config"] = configs[0] == configs[1]
    metrics = records(cached / "training_metrics.jsonl")
    return {"cached": str(cached), "on_the_fly": str(online), "identical": all(checks.values()), "checks": checks,
            "updates": len(metrics), "adapter_tensors": tensors, "checkpoints": steps,
            "loss_fm": [round(m["metrics"]["loss_fm"], 6) for m in metrics],
            "region_row_share": [m["metrics"].get("region_row") for m in metrics],
            "bound_units": [m["metrics"].get("bound_units") for m in metrics],
            "peak_memory_gib": {"cached": max(r["peak_memory_gib"] for r in ranks[0]),
                                "on_the_fly": max(r["peak_memory_gib"] for r in ranks[1])}}


if __name__ == "__main__":
    report = compare(sys.argv[1], sys.argv[2])
    print(json.dumps(report, indent=1))
    sys.exit(0 if report["identical"] else 1)
