"""Fail-closed post-run audit for the full four-node training path."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from safetensors.torch import load_file

from samtok_edit21.data import file_hash
from samtok_edit21.provenance import verify_cache
from samtok_edit21.training_core.gradient_audit import audit_gradient_logs


def lines(path):
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    root, data = Path(args.run_root).resolve(), Path(args.data).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    common, world = manifest["args"], manifest["world_size"]
    result = {"world_size": world, "data": str(data), "stages": {}}
    for stage, accumulation, expected in (
        ("stage1", 8, {"edit_ntp": 3, "edit_umt:ref": 2, "edit_umt:noref": 2, "edit": 1}),
        ("stage2", 4, {"edit_umt:ref": 1, "edit_umt:noref": 2, "edit": 1}),
    ):
        directory = root / stage
        metrics = lines(directory / "training_metrics.jsonl")
        updates = lines(directory / "optimizer_steps.jsonl")
        steps = common[f"{stage}_steps"]
        if len(metrics) != steps or len(updates) != steps:
            raise ValueError(f"{stage}: expected {steps} optimizer updates")
        for index, entry in enumerate(metrics, 1):
            if entry["optimizer_step"] != index or entry["skipped"]:
                raise ValueError(f"{stage}: skipped or reordered optimizer update {index}")
            if entry["samples"] != world * accumulation:
                raise ValueError(f"{stage}: global sample count mismatch")
            if entry["rank_samples"] != {str(rank): accumulation for rank in range(world)}:
                raise ValueError(f"{stage}: rank sample count mismatch")
            if entry["branches"] != {kind: count * world for kind, count in expected.items()}:
                raise ValueError(f"{stage}: branch ratio mismatch at update {index}")
            if not all(math.isfinite(float(value)) for value in entry["metrics"].values()):
                raise ValueError(f"{stage}: nonfinite metric at update {index}")
        parameters = json.loads((directory / "rank_parameters.json").read_text())
        if len(parameters) != world or len({row["sha256"] for row in parameters}) != 1:
            raise ValueError(f"{stage}: trainable parameters diverged across ranks")
        adapter = directory / "adapter" / "adapter.safetensors"
        if not adapter.is_file() or not load_file(str(adapter)):
            raise ValueError(f"{stage}: missing adapter")
        wandb = json.loads((directory / "wandb.json").read_text())
        if wandb.get("status") != "finished" or wandb.get("mode") != common["wandb_mode"]:
            raise ValueError(f"{stage}: W&B did not finish in requested mode")
        result["stages"][stage] = {"steps": steps, "metrics": len(metrics),
                                    "gradients": audit_gradient_logs(directory, world, steps, accumulation, expected),
                                    "parameter_hash": parameters[0]["sha256"],
                                    "adapter_sha256": file_hash(adapter)}
    cache = json.loads((root / "cache" / "manifest.json").read_text())
    verify_cache(root / "cache", cache)
    result["cache"] = {"rows": cache["row_count"], "verified": True,
                        "manifest_sha256": file_hash(root / "cache" / "manifest.json")}
    result["passed"] = True
    (root / "audit_full.json.tmp").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (root / "audit_full.json.tmp").replace(root / "audit_full.json")
    print(json.dumps({"passed": True, "world_size": world, "cache_rows": cache["row_count"]}))


if __name__ == "__main__":
    main()
