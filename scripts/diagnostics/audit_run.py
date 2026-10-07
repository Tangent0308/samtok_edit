"""Fail-closed audit of a v2 run: every executed phase (stage1, cache, stage2).

Checks optimizer-update counts, exact per-rank branch ratios, gradient logs,
identical trainable weights on all ranks, finite adapters, W&B completion,
the adapter's binding recipe and the conditioning cache (all rows for small
caches, a deterministic sample for full ones; Stage 2 startup already verified
every row in parallel).  Stage 2 without a cache computes its inputs on the
fly; its adapter must then record the raw-TE identity of this run's rows.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

import torch
from safetensors.torch import load_file

from samtok_edit21.data.io import RATIOS, file_hash, write_json
from samtok_edit21.data.provenance import FORMAT, PREPROCESSING, _verify_cache_row, validate_cache_manifest
from samtok_edit21.training.gradient_audit import audit_gradient_logs

ACCUMULATION = {"stage1": 8, "stage2": 4}


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def audit_stage(root, stage, steps, world, mode):
    directory = root / stage
    accumulation, ratio = ACCUMULATION[stage], RATIOS[stage]
    metrics, updates = lines(directory / "training_metrics.jsonl"), lines(directory / "optimizer_steps.jsonl")
    if len(metrics) != steps or len(updates) != steps:
        raise ValueError(f"{stage}: expected {steps} optimizer updates")
    for index, entry in enumerate(metrics, 1):
        if entry["optimizer_step"] != index or entry["skipped"]:
            raise ValueError(f"{stage}: skipped or reordered optimizer update {index}")
        if entry["samples"] != world * accumulation:
            raise ValueError(f"{stage}: global sample count mismatch at update {index}")
        if entry["rank_samples"] != {str(rank): accumulation for rank in range(world)}:
            raise ValueError(f"{stage}: rank sample count mismatch at update {index}")
        if entry["branches"] != {kind: count * world for kind, count in ratio.items()}:
            raise ValueError(f"{stage}: branch ratio mismatch at update {index}")
        if not all(math.isfinite(float(value)) for value in entry["metrics"].values()):
            raise ValueError(f"{stage}: nonfinite metric at update {index}")
    parameters = json.loads((directory / "rank_parameters.json").read_text())
    if len(parameters) != world or len({row["sha256"] for row in parameters}) != 1:
        raise ValueError(f"{stage}: trainable parameters diverged across ranks")
    adapter_dir = directory / "adapter"
    state = load_file(str(adapter_dir / "adapter.safetensors"))
    config = json.loads((adapter_dir / "adapter.json").read_text())
    if not state or not all(torch.isfinite(value).all() for value in state.values()):
        raise ValueError(f"{stage}: missing or nonfinite adapter")
    if not any(value.count_nonzero() for key, value in state.items() if "lora_B" in key):
        raise ValueError(f"{stage}: LoRA B never moved from zero")
    embed = sorted(key for key in state if key.startswith("region_embed."))
    binding = config.get("binding", {"mode": "none"})
    if stage == "stage2" and (binding["mode"] != mode or bool(embed) != (mode == "region_embed")):
        raise ValueError(f"{stage}: adapter binding recipe differs from the run arguments")
    wandb = json.loads((directory / "wandb.json").read_text())
    if wandb.get("status") != "finished":
        raise ValueError(f"{stage}: W&B did not finish")
    return {"updates": steps, "world_size": world, "accumulation": accumulation,
            "gradients": audit_gradient_logs(directory, world, steps, accumulation, ratio),
            "parameter_hash": parameters[0]["sha256"], "adapter_sha256": file_hash(adapter_dir / "adapter.safetensors"),
            "adapter_tensors": len(state), "region_embed_tensors": embed, "binding": binding,
            "final_metrics": metrics[-1]["metrics"], "wandb": wandb}


def audit_cache(cache_dir, sample=1000):
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    identity, rows, ordered = validate_cache_manifest(manifest)
    indices = list(range(len(rows)))
    if len(rows) > 5 * sample:
        indices = sorted(random.Random(0).sample(indices, sample))
    kinds, units, empty = Counter(), Counter(), 0
    for index in indices:
        _verify_cache_row(cache_dir, rows[index], identity, ordered)
        payload = torch.load(cache_dir / rows[index]["_cache_file"], map_location="cpu", weights_only=True)
        binding = payload["inputs"].get("region_binding")
        kinds[rows[index]["sample_type"] + ":" + rows[index]["edit_type"]] += 1
        if binding is not None:
            units[len(binding["units"])] += 1
            empty += sum(binding["empty"])
    return {"rows": len(rows), "verified_rows": len(indices), "manifest_sha256": file_hash(cache_dir / "manifest.json"),
            "verified_kinds": dict(kinds), "units_per_region_row": dict(units), "empty_region_units": empty,
            "identity_binding": identity["binding"], "te_adapter": identity["te_adapter"]}


def audit_on_the_fly(root, run, data_hashes):
    identity = json.loads((root / "stage2" / "adapter" / "adapter.json").read_text())["conditioning_identity"]
    if ((identity.get("schema"), identity.get("preprocessing")) != (FORMAT, PREPROCESSING)
            or identity.get("te_adapter") is not None or not identity.get("binding")):
        raise ValueError("stage2: conditioning is not the v2 raw-TE protocol")
    if identity.get("max_pixels") != run["max_pixels"] or identity.get("metadata_sha256") != data_hashes["stage2.jsonl"]:
        raise ValueError("stage2: conditioning identity does not match this run's rows or resolution")
    return {"mode": "on_the_fly", "metadata_sha256": identity["metadata_sha256"],
            "max_pixels": identity["max_pixels"], "identity_binding": identity["binding"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    run, world = manifest["args"], manifest["world_size"]
    report = {"world_size": world, "phases": run["phases"], "stages": {}}
    for rank in range(world):
        probe = json.loads((root / "collectives" / f"rank{rank}.json").read_text())
        if probe["rank"] != rank or probe["world_size"] != world or probe["all_reduce"] != world * (world + 1) / 2:
            raise ValueError(f"Collective probe failed on rank {rank}")
    for stage in ("stage1", "stage2"):
        if stage in run["phases"]:
            report["stages"][stage] = audit_stage(root, stage, run[f"{stage}_steps"], world, run["binding"])
    if "cache" in run["phases"] or ("stage2" in run["phases"] and run.get("cache")):
        report["cache"] = audit_cache(root / "cache")
    elif "stage2" in run["phases"]:
        report["conditioning"] = audit_on_the_fly(root, run, manifest["data"])
    report["passed"] = True
    write_json(root / "audit.json", report)
    print(json.dumps({"passed": True, "world_size": world, "phases": run["phases"],
                      "cache_rows": report.get("cache", {}).get("rows")}))


if __name__ == "__main__":
    main()
