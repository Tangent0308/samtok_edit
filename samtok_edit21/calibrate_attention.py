"""Measure A/C gradients on frozen conditioning cache; no optimizer updates."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import torch
from accelerate.utils import set_seed

from .data import write_json
from .attention_supervision import require_attention_backend
from .model import DEFAULT_QWEN, DEFAULT_SAMTOK, load_pipeline
from .provenance import assert_models_match, cache_path, verify_cache
from .region_supervision import validate_supervision
from .training import adapter_identity, add_adapter, flow_loss, load_adapter


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--qwen", default=DEFAULT_QWEN)
    p.add_argument("--samtok", default=DEFAULT_SAMTOK)
    p.add_argument("--device", default="cuda")
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--init-adapter")
    p.add_argument("--samples", type=int, default=8, help="Use the first N eligible rows in manifest order")
    p.add_argument("--timesteps", nargs="+", type=int, default=[100, 500, 900])
    p.add_argument("--seed", type=int, default=20260920)
    p.add_argument("--target-ratio", type=float, default=0.2)
    p.add_argument("--region-weight", type=float, default=0.5)
    p.add_argument("--region-n-min", type=float, default=16.0)
    p.add_argument("--attention-read-weight", type=float, default=0.5)
    p.add_argument("--attention-layers", nargs="+", type=int, default=[7,11,15,19,23])
    args = p.parse_args(argv)
    require_attention_backend()
    if Path(args.output).exists():
        raise ValueError("Calibration output must be fresh")
    if args.samples < 1 or not math.isfinite(args.target_ratio) or not 0 < args.target_ratio <= 1 or any(not 0 <= t < 1000 for t in args.timesteps):
        raise ValueError("Invalid calibration samples/ratio/timesteps")
    args.attention_layers = sorted(args.attention_layers)
    if len(set(args.attention_layers)) != len(args.attention_layers) or any(not 0 <= i < 32 for i in args.attention_layers):
        raise ValueError("Attention layers must be unique indices in [0,31]")
    if any(not math.isfinite(v) or v < 0 for v in (args.region_weight, args.attention_read_weight)) or not math.isfinite(args.region_n_min) or args.region_n_min <= 0:
        raise ValueError("Invalid calibration loss coefficients")
    manifest = json.loads((Path(args.cache) / "manifest.json").read_text())
    verify_cache(args.cache, manifest)
    assert_models_match(manifest["identity"], args.qwen, args.samtok)
    if not manifest.get("supervision_identity"):
        raise ValueError("Calibration requires region supervision cache")
    set_seed(args.seed)
    pipe = load_pipeline(args.qwen, args.samtok, device=args.device, components=("dit",))
    pipe.scheduler.set_timesteps(1000, training=True)
    if args.init_adapter:
        config = load_adapter(pipe.dit, args.init_adapter, trainable=True)
        if config.get("stage") != "stage2" or config.get("conditioning_identity") != manifest["identity"]:
            raise ValueError("Calibration warm-start conditioning mismatch")
    else:
        add_adapter(pipe.dit, "stage2", args.rank, 0)
    pipe.dit.train()
    params = [v for v in pipe.dit.parameters() if v.requires_grad]

    def device(value):
        if isinstance(value, torch.Tensor): return value.to(args.device)
        if isinstance(value, dict): return {k: device(v) for k, v in value.items()}
        if isinstance(value, list): return [device(v) for v in value]
        return value

    records, used = [], 0
    for row in manifest["rows"]:
        inputs = torch.load(cache_path(args.cache, row["_cache_file"]), map_location="cpu", weights_only=True)["inputs"]
        validate_supervision(inputs.get("region_supervision"), inputs, row, require_positions=True)
        if not inputs["region_supervision"]["eligible"]: continue
        inputs = device(inputs)
        for timestep in args.timesteps:
            # Compiled FlexAttention may donate backward buffers, which rules
            # out retain_graph=True. Recompute each objective with the SAME
            # noise and RNG state (including any warm-start LoRA dropout).
            noise = torch.randn_like(inputs["input_latents"])
            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state(args.device)
            norms = {}
            for name in ("basic_fm", "fm", "attention"):
                torch.set_rng_state(cpu_rng)
                torch.cuda.set_rng_state(cuda_rng, args.device)
                total, metrics, parts = flow_loss(pipe, inputs, timestep_index=timestep, noise=noise,
                    region_weight=args.region_weight, region_n_min=args.region_n_min,
                    attention_weight=1, attention_layers=args.attention_layers,
                    attention_read_weight=args.attention_read_weight, return_components=True)
                grads = torch.autograd.grad(parts[name], params, allow_unused=True)
                norm = torch.stack([g.float().square().sum() for g in grads if g is not None]).sum().sqrt().item()
                if not math.isfinite(norm) or norm <= 0:
                    raise ValueError(f"Cannot calibrate zero/nonfinite {name} gradients")
                norms[name] = norm
                del grads, parts, total
            records.append({"row_hash": inputs["region_supervision"]["row_hash"], "timestep_index": timestep,
                            "norms": norms, "suggested_weight": args.target_ratio * norms["fm"] / norms["attention"],
                            "metrics": metrics})
        used += 1
        if used >= args.samples: break
    if not records:
        raise ValueError("No eligible calibration rows")
    coefficient = statistics.median(r["suggested_weight"] for r in records)
    write_json(args.output, {"args": vars(args), "actual_samples": used,
                            "conditioning_identity": manifest["identity"],
                            "supervision_identity": manifest["supervision_identity"],
                            "init_adapter": adapter_identity(args.init_adapter) if args.init_adapter else None,
                            "attention_weight": coefficient, "records": records,
                            "measured_ratios": [coefficient * r["norms"]["attention"] / r["norms"]["fm"] for r in records]})
    print(json.dumps({"attention_weight": coefficient, "measurements": len(records)}))


if __name__ == "__main__":
    main()
