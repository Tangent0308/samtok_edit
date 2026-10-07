"""Region-token sensitivity of Stage 2 adapters: does the DiT use the region tokens?

For held-out region rows of stage2.jsonl (rows the run's training schedule
never sampled), the flow-matching velocity error is measured inside the row's
true edit region (its target coverage map) and outside it, once with the row's
own region tokens and once with region tokens taken from another row (same
region kind, a different place).  Timesteps and noise are fixed per row, so
every adapter sees identical inputs.  A DiT that uses the region tokens
predicts the edited region worse when the tokens point elsewhere.

    torchrun --nproc_per_node 8 scripts/diagnostics/region_sensitivity.py run \
        --metadata <data>/stage2.jsonl --adapter <run>/stage2/adapter --label b0 --output <dir>
    python scripts/diagnostics/region_sensitivity.py summarize --output <dir> --labels b0 e6 e7
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from types import SimpleNamespace

TYPES = ("add", "remove", "replace", "attribute")
TIMESTEP_INDICES = (250, 500, 750)


def held_out_rows(rows, seed, steps, world=32, per_type=48):
    """Single-unit region rows never sampled by the training schedule, and a donor for each."""
    from samtok_edit21.data.io import make_schedule
    from samtok_edit21.data.protocol import REGION_RE, region_kind, regions_in
    schedule, _ = make_schedule(rows, "stage2", world, 4, steps=steps, seed=seed, type_weights="main4")
    seen = set(schedule)
    pools = {t: [] for t in TYPES}
    for index, row in enumerate(rows):
        if index in seen or row["sample_type"] != "edit_umt" or row["edit_type"] not in pools:
            continue
        regions = regions_in(row["prompt"])
        if len(regions) == 1 and len(REGION_RE.findall(row["prompt"])) == 1:
            pools[row["edit_type"]].append(index)
    rng = random.Random(0)
    chosen = []
    for typ in TYPES:
        rng.shuffle(pools[typ])
        chosen += pools[typ][:per_type]
    by_kind = {}
    for index in chosen:
        by_kind.setdefault(region_kind(REGION_RE.search(rows[index]["prompt"]).group()), []).append(index)
    pairs = []
    for index in chosen:
        kind = region_kind(REGION_RE.search(rows[index]["prompt"]).group())
        donors = [d for d in by_kind[kind] if rows[d]["edit_image"] != rows[index]["edit_image"]]
        pairs.append((index, random.Random(index).sample(donors, min(5, len(donors)))))
    return pairs


def swapped(row, donor_row):
    from samtok_edit21.data.protocol import REGION_RE
    region = REGION_RE.search(donor_row["prompt"]).group()
    return {**row, "prompt": REGION_RE.sub(lambda _: region, row["prompt"], count=1)}


def errors(pipe, inputs, binding, coverage, index, timestep_index):
    """Per-region squared velocity error at a fixed timestep with row-seeded noise."""
    import torch
    from samtok_edit21.models.binding import RegionBinding
    x = inputs["input_latents"]
    generator = torch.Generator(device="cpu").manual_seed(1000003 * index + timestep_index)
    noise = torch.randn(x.shape, generator=generator, dtype=torch.float32).to(device=x.device, dtype=x.dtype)
    t = pipe.scheduler.timesteps[torch.tensor([timestep_index])].to(device=pipe.device, dtype=pipe.torch_dtype)
    noisy = pipe.scheduler.add_noise(x, noise, t)
    target = pipe.scheduler.training_target(x, noise, t)
    payload = inputs.get("region_binding")
    region_binding = None
    if binding.mode == "region_embed" or (binding.mode != "none" and payload is not None):
        region_binding = RegionBinding(binding, payload, getattr(pipe.dit, "region_embed", None)
                                       if binding.mode == "region_embed" else None)
    model_inputs = {k: v for k, v in inputs.items() if k != "region_binding"}
    pred = pipe.model_fn(dit=pipe.dit, latents=noisy, timestep=t, **model_inputs, kv_cache=None,
                         use_gradient_checkpointing=False, region_binding=region_binding)
    err = (pred.float() - target.float()).pow(2).mean(dim=1)[0]
    inside, outside = coverage.to(err.device), 1 - coverage.to(err.device)
    return {"in": float((err * inside).sum() / inside.sum()), "out": float((err * outside).sum() / outside.sum()),
            "all": float(err.mean())}


def run(args):
    import torch
    from samtok_edit21.data.io import read_rows
    from samtok_edit21.models.pipeline import load_pipeline
    from samtok_edit21.training.engine import OnlineConditioning
    from samtok_edit21.training.objectives import adapter_binding, load_adapter

    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local}"
    torch.cuda.set_device(local)
    rows = read_rows(args.metadata)
    pairs = held_out_rows(rows, args.seed, args.steps, per_type=args.per_type)[rank::world]
    conditioning = OnlineConditioning(SimpleNamespace(qwen=args.qwen, samtok=args.samtok, device=device,
                                                      base_path=".", max_pixels=args.max_pixels))
    pipe = load_pipeline(args.qwen, args.samtok, device=device, components=("dit",))
    pipe.scheduler.set_timesteps(1000, training=True)
    config = json.loads((Path(args.adapter) / "adapter.json").read_text())
    binding = adapter_binding(config)
    load_adapter(pipe.dit, args.adapter)
    pipe.eval()
    out = Path(args.output) / args.label
    out.mkdir(parents=True, exist_ok=True)
    with torch.no_grad(), (out / f"rank-{rank:02d}.jsonl").open("w") as stream:
        for index, donors in pairs:
            row = rows[index]
            true_inputs = conditioning(row)
            coverage = true_inputs["region_binding"]["target"][0]
            swap_inputs, overlap = None, None
            for donor in donors:  # first donor whose region barely overlaps the true one
                candidate = conditioning(swapped(row, rows[donor]))
                other = candidate["region_binding"]["target"][0]
                overlap = float(torch.minimum(coverage, other).sum() / torch.maximum(coverage, other).sum())
                if overlap < 0.2:
                    swap_inputs = candidate
                    break
            if swap_inputs is None:
                continue
            record = {"row": index, "edit_type": row["edit_type"], "variant": row.get("instr_variant"),
                      "area": float(coverage.mean()), "swap_overlap": overlap, "timesteps": {}}
            for ti in TIMESTEP_INDICES:
                record["timesteps"][str(ti)] = {"true": errors(pipe, true_inputs, binding, coverage, index, ti),
                                                "swap": errors(pipe, swap_inputs, binding, coverage, index, ti)}
            stream.write(json.dumps(record) + "\n")
            stream.flush()
    print(json.dumps({"rank": rank, "label": args.label, "rows": len(pairs)}), flush=True)


def summarize(args):
    rng = random.Random(0)
    def boot(values, n=4000):
        means = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(n))
        return means[int(0.025 * n)], means[int(0.975 * n)]
    data = {}
    for label in args.labels:
        records = [json.loads(line) for path in sorted((Path(args.output) / label).glob("rank-*.jsonl"))
                   for line in path.read_text().splitlines() if line.strip()]
        data[label] = {r["row"]: r for r in records}
    common = sorted(set.intersection(*(set(d) for d in data.values())))
    def stat(label, rows, key):
        values = []
        for r in rows:
            for ts in data[label][r]["timesteps"].values():
                values.append(key(ts))
        return values
    report = {"rows": len(common), "by_label": {}}
    for label in args.labels:
        entry = {}
        for typ in (*TYPES, "all"):
            rows = [r for r in common if typ == "all" or data[label][r]["edit_type"] == typ]
            if not rows:
                continue
            per_row = []
            for r in rows:
                ts = data[label][r]["timesteps"].values()
                per_row.append({k: sum(t[c][s] for t in ts) / len(ts)
                                for k, (c, s) in {"in_true": ("true", "in"), "in_swap": ("swap", "in"),
                                                  "out_true": ("true", "out"), "out_swap": ("swap", "out")}.items()})
            gain = [(p["in_swap"] - p["in_true"]) / p["in_true"] for p in per_row]
            entry[typ] = {"n": len(rows), **{k: sum(p[k] for p in per_row) / len(per_row) for k in per_row[0]},
                          "in_rise_rel": sum(gain) / len(gain), "in_rise_rel_ci": boot(gain)}
        report["by_label"][label] = entry
    Path(args.output, "summary.json").write_text(json.dumps(report, indent=1))
    print(f"held-out region rows: {report['rows']} (x {len(TIMESTEP_INDICES)} timesteps)")
    print("| adapter | type | n | in-region error (true tokens) | with swapped tokens | rise [95% CI] | outside error true / swapped |")
    print("|---|---|---:|---:|---:|---|---|")
    for label, entry in report["by_label"].items():
        for typ, e in entry.items():
            lo, hi = e["in_rise_rel_ci"]
            print(f"| {label} | {typ} | {e['n']} | {e['in_true']:.4f} | {e['in_swap']:.4f} | "
                  f"{e['in_rise_rel'] * 100:+.1f}% [{lo * 100:+.1f}, {hi * 100:+.1f}] | {e['out_true']:.4f} / {e['out_swap']:.4f} |")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--metadata", required=True)
    r.add_argument("--adapter", required=True)
    r.add_argument("--label", required=True)
    r.add_argument("--output", required=True)
    r.add_argument("--seed", type=int, default=20261006, help="Training seed whose schedule defines held-out rows")
    r.add_argument("--steps", type=int, default=1000)
    r.add_argument("--per-type", type=int, default=48)
    r.add_argument("--max-pixels", type=int, default=1048576)
    r.add_argument("--qwen", default="/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen-Image-2.1")
    r.add_argument("--samtok", default="/mnt/bn/strategy-mllm-train/user/tanyue/models/SAMTok/Qwen3-VL-8B-SAMTok")
    s = sub.add_parser("summarize")
    s.add_argument("--output", required=True)
    s.add_argument("--labels", nargs="+", required=True)
    args = parser.parse_args()
    run(args) if args.command == "run" else summarize(args)


if __name__ == "__main__":
    main()
