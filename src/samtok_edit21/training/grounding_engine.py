"""Stage A trainer (v3): stock Qwen-Image-2.1 DiT + fresh LoRA + code injector, trained on token-grounding rows.

No region binding, no latent blending, no pixel mask in the forward pass - the model sees what inference sees.  Runs
under torchrun on one or several nodes; every rank processes its own rows, gradients are averaged across ranks and
every rank applies the same optimizer step, so the effective batch is ``world_size * accumulation``::

    python -m torch.distributed.run --nproc_per_node 8 -m samtok_edit21 ground-train \\
        --data <data_dir> --output <run_dir>/train

``<data_dir>`` is written by ``samtok-edit ground-data``: ``epoch_rows.jsonl`` (rows), ``schedule.json`` (the global
row order, consumed ``world_size`` rows at a time), ``eval.json`` and ``eval_quick_idx.json`` (in-training checks).

All ranks must hold the same weights.  LoRA matrices are initialised randomly, so after building the model rank 0's
trainable parameters are broadcast to every rank, and the checkpoints verify that the ranks still agree.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch

from samtok_edit21.training.routing import DEFAULT_LAYERS


def parser():
    from samtok_edit21.models.pipeline import DEFAULT_QWEN, DEFAULT_SAMTOK

    integers = lambda text: tuple(int(x) for x in text.split(",") if x.strip())  # noqa: E731
    p = argparse.ArgumentParser(prog="samtok-edit ground-train", description=__doc__.split("\n\n")[0])
    p.add_argument("--data", required=True, help="Directory written by `samtok-edit ground-data`")
    p.add_argument("--output", required=True, help="Training output directory")
    p.add_argument("--qwen", default=DEFAULT_QWEN)
    p.add_argument("--samtok", default=DEFAULT_SAMTOK)
    p.add_argument("--updates", type=int, default=1300, help="Optimizer steps")
    p.add_argument("--accumulation", type=int, default=4, help="Rows per rank per optimizer step")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=33, help="Linear warm-up, in optimizer steps")
    p.add_argument("--weight-decay", type=float, default=0.01, help="LoRA only; injector and routing affine use 0")
    p.add_argument("--rank", type=int, default=32, help="DiT LoRA rank of a fresh adapter")
    p.add_argument("--init-adapter", help="Warm start from a v3 adapter directory instead of a fresh LoRA")
    p.add_argument("--lambda-route", type=float, default=0.05, help="Attention-routing loss weight; 0 disables it")
    p.add_argument("--lambda-region", type=float, default=1.0, help="Region-normalised FM term weight")
    p.add_argument("--lambda-distractor", type=float, default=1.0, help="Other-instance FM term weight")
    p.add_argument("--route-layers", type=integers, default=DEFAULT_LAYERS, help="Comma-separated DiT block indices")
    p.add_argument("--no-code-injector", action="store_true", help="Ablation: train without the code-vector channel")
    p.add_argument("--checkpoints", type=integers, default=(325, 650, 975, 1300), help="Updates at which to save")
    p.add_argument("--quick-eval", type=integers, default=(0, 325, 650, 975, 1300), help="Updates at which to run it")
    p.add_argument("--quick-limit", type=int, default=999, help="Quick-eval items per rank (smoke tests)")
    p.add_argument("--max-pixels", type=int, default=1024 * 1024)
    p.add_argument("--seed", type=int, default=2000, help="Rank r draws timesteps and noise from seed + r")
    p.add_argument("--resume", action="store_true", help="Continue from <output>/resume.pt")
    p.add_argument("--unsynced-init", action="store_true",
                   help="Regression only: keep each rank's own random LoRA init, as E-A1/E-A2 were run")
    return p


def broadcast_parameters(parameters, dist):
    """Give every rank rank 0's values (LoRA A matrices are random and would otherwise differ per rank)."""
    flat = torch.cat([p.detach().flatten().float() for p in parameters])
    dist.broadcast(flat, src=0)
    offset = 0
    with torch.no_grad():
        for p in parameters:
            p.copy_(flat[offset:offset + p.numel()].view_as(p).to(p.dtype))
            offset += p.numel()


def rank_divergence(parameters, dist, world):
    """Largest relative gap between any rank's parameter checksum and rank 0's (0 when the ranks agree)."""
    checksum = float(torch.stack([p.detach().double().abs().sum() for p in parameters]).sum())
    gathered = [None] * world
    dist.all_gather_object(gathered, checksum)
    return max(abs(value - gathered[0]) for value in gathered) / max(abs(gathered[0]), 1e-12)


def average_gradients(parameters, dist, world):
    """Replace each rank's gradients by their mean over ranks."""
    grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in parameters]
    flat = torch.cat([g.flatten().float() for g in grads])
    dist.all_reduce(flat)
    flat /= world
    offset = 0
    for p, g in zip(parameters, grads):
        p.grad = flat[offset:offset + g.numel()].view_as(p).to(p.dtype).clone()
        offset += g.numel()


def main(argv=None):
    import torch.distributed as dist

    from samtok_edit21.evaluation.grounding import highlight_item, quick_summary
    from samtok_edit21.models.code_injector import (
        CodeInjector, attach_code_injector, has_code_injector, load_code_injector, save_code_injector,
    )
    from samtok_edit21.models.codec import load_codec
    from samtok_edit21.models.pipeline import load_pipeline
    from samtok_edit21.training.grounding import grounding_loss, prepare_sample
    from samtok_edit21.training.objectives import add_adapter, load_adapter, save_adapter
    from samtok_edit21.training.routing import AttentionRouter

    args = parser().parse_args(argv)
    if args.updates < 1 or args.accumulation < 1:
        raise ValueError("--updates and --accumulation must be positive")
    dist.init_process_group("nccl", timeout=timedelta(hours=2))
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    device = f"cuda:{local}"
    out, data = Path(args.output), Path(args.data)
    out.mkdir(parents=True, exist_ok=True)
    checkpoints, quick_at = set(args.checkpoints), set(args.quick_eval)

    def log(name, message):
        if rank == 0:
            print(json.dumps(message), flush=True)
            with open(out / name, "a") as handle:
                handle.write(json.dumps(message) + "\n")

    # One random stream per rank for timesteps and noise.  The model built below also draws from it, which is why
    # the trainable weights are broadcast from rank 0 afterwards.
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)

    pipe = load_pipeline(args.qwen, args.samtok, device=device)
    pipe.scheduler.set_timesteps(1000, training=True)
    if args.init_adapter:
        load_adapter(pipe.dit, args.init_adapter, trainable=True)
    else:
        add_adapter(pipe.dit, "stage2", args.rank, 0.0)
    pipe.dit.train()
    codec = load_codec(args.samtok, device)
    injector = None
    if not args.no_code_injector:
        injector = attach_code_injector(pipe, CodeInjector.from_codec(codec, pipe.processor.tokenizer))
        if args.init_adapter and has_code_injector(args.init_adapter):
            load_code_injector(injector, args.init_adapter)
    router = AttentionRouter(pipe.dit, args.route_layers).to(device) if args.lambda_route > 0 else None

    lora = [p for p in pipe.dit.parameters() if p.requires_grad]
    extra = (list(injector.proj.parameters()) if injector is not None else []) + ([router.affine] if router else [])
    parameters = lora + extra
    groups = [{"params": lora, "weight_decay": args.weight_decay}]
    if extra:
        groups.append({"params": extra, "weight_decay": 0.0})
    optimizer = torch.optim.AdamW(groups, lr=args.lr)

    rows = [json.loads(line) for line in open(data / "epoch_rows.jsonl")]
    schedule = json.load(open(data / "schedule.json"))
    needed = args.updates * args.accumulation * world
    order = (schedule * (needed // len(schedule) + 1))[:needed]
    mine = order[rank::world]
    items = json.load(open(data / "eval.json"))
    quick = [items[index] | {"idx": index} for index in json.load(open(data / "eval_quick_idx.json"))]
    quick = quick[rank::world][:args.quick_limit]

    start, resume_path = 0, out / "resume.pt"
    if args.resume and resume_path.exists():
        state = torch.load(resume_path, map_location="cpu")
        with torch.no_grad():
            for p, value in zip(parameters, state["params"]):
                p.copy_(value.to(p.device))
        optimizer.load_state_dict(state["opt"])
        start = state["update"]
        del state
    elif not args.unsynced_init:
        broadcast_parameters(parameters, dist)
    divergence = None if args.unsynced_init else rank_divergence(parameters, dist, world)
    if divergence is not None and divergence > 1e-6:
        raise RuntimeError(f"Ranks start from different weights (relative checksum gap {divergence:.3g})")

    recipe = {"updates": args.updates, "accumulation": args.accumulation, "world": world, "lr": args.lr,
              "warmup": args.warmup, "weight_decay": args.weight_decay, "lambda_route": args.lambda_route,
              "lambda_region": args.lambda_region, "lambda_distractor": args.lambda_distractor,
              "route_layers": list(args.route_layers) if router else [], "code_injector": injector is not None,
              "seed": args.seed, "synced_init": not args.unsynced_init, "init_adapter": args.init_adapter,
              "data": str(data)}
    log("train_metrics.jsonl", {"start": start, "world": world, "updates": args.updates, "acc": args.accumulation,
                               "lr": args.lr, "lambda_route": args.lambda_route,
                               "trainable": sum(p.numel() for p in parameters), "rows_per_rank": len(mine),
                               "synced_init": not args.unsynced_init, "recipe": recipe})
    started = time.time()

    def quick_eval(tag):
        pipe.dit.eval()
        records = []
        for position, item in enumerate(quick):
            try:
                record, image, _ = highlight_item(pipe, codec, item, steps=20)
                record["idx"] = item["idx"]
                records.append(record)
                if position < 1:
                    folder = out / "quick_images" / f"step{tag}"
                    folder.mkdir(parents=True, exist_ok=True)
                    image.resize((512, round(512 * image.height / image.width))).save(
                        folder / f"r{rank}_{item['set']}_{item['variant']}_{item['idx']}.jpg", quality=88)
            except Exception as error:
                print(f"[rank {rank}] quick skip {item['idx']}: {error!r}"[:400], flush=True)
        pipe.scheduler.set_timesteps(1000, training=True)   # generation switched the scheduler to inference
        pipe.dit.train()
        gathered = [None] * world
        dist.all_gather_object(gathered, records)
        if rank == 0:
            merged = [record for part in gathered for record in part]
            json.dump(merged, open(out / f"quick_{tag}.json", "w"))
            summary = quick_summary(merged)
            log("quick_summary.jsonl", {"quick": tag, "n": summary.pop("n"), **summary,
                                       "sec": round(time.time() - started)})

    if start == 0 and 0 in quick_at:
        quick_eval(0)
    # loss, rows, mse, region, distractor, routing, highlight rows, skipped rows, rows with a routing term
    totals = torch.zeros(9, device=device)
    for update in range(start, args.updates):
        for group in optimizer.param_groups:
            group["lr"] = args.lr * min(1.0, (update + 1) / max(1, args.warmup))
        optimizer.zero_grad(set_to_none=True)
        for step in range(args.accumulation):
            row = rows[mine[update * args.accumulation + step]]
            try:
                sample = prepare_sample(pipe, codec, row, data_root=data, max_pixels=args.max_pixels)
                flow, routing, metrics = grounding_loss(
                    pipe, sample, router, lambda_region=args.lambda_region, lambda_distractor=args.lambda_distractor)
                loss = flow + (args.lambda_route * routing if routing is not None else 0.0)
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite loss")
                (loss / args.accumulation).backward()
                totals[0] += float(loss.detach())
                totals[1] += 1
                totals[2] += metrics["mse"]
                totals[3] += metrics["reg"]
                totals[4] += metrics["dis"]
                if routing is not None:
                    totals[5] += metrics["route"]
                    totals[8] += 1
                totals[6] += row["kind"] == "hl"
            except Exception as error:   # a broken image must not end a multi-hour run; skips are counted and logged
                totals[7] += 1
                print(f"[rank {rank}] skip {row['kind']} {str(row['edit_image'])[-60:]}: {error!r}"[:400], flush=True)
            finally:
                if router is not None:
                    router.disarm()      # only now: checkpointing re-runs the hooks during backward
        average_gradients(parameters, dist, world)
        optimizer.step()
        done = update + 1
        if done % 10 == 0 or done == args.updates:
            summed = totals.clone()
            dist.all_reduce(summed)
            totals.zero_()
            count = max(float(summed[1]), 1.0)
            norm = 0.0 if injector is None else float(sum(p.detach().float().norm() ** 2 for p in injector.proj.parameters()) ** 0.5)
            log("train_metrics.jsonl", {
                "update": done, "loss": round(float(summed[0]) / count, 5), "mse": round(float(summed[2]) / count, 5),
                "reg": round(float(summed[3]) / count, 5), "dis": round(float(summed[4]) / count, 5),
                "route": round(float(summed[5]) / max(float(summed[8]), 1.0), 5), "samples": int(summed[1]),
                "hl": int(summed[6]), "skipped": int(summed[7]), "sec": round(time.time() - started),
                "lr": optimizer.param_groups[0]["lr"], "inj_norm": round(norm, 4),
                "route_affine": [] if router is None else [round(float(x), 3) for x in router.affine.detach().flatten()[:4]]})
        if done in checkpoints or done == args.updates:
            dist.barrier()
            divergence = None if args.unsynced_init else rank_divergence(parameters, dist, world)
            if divergence is not None and divergence > 1e-4:
                raise RuntimeError(f"Ranks diverged at update {done} (relative checksum gap {divergence:.3g})")
            if rank == 0:
                folder = out / f"adapter_{done}"
                affine = None if router is None else [[float(x) for x in pair] for pair in router.affine.detach().cpu()]
                save_adapter(pipe.dit, folder, {"stage": "stage2", "binding": {"mode": "none"},
                                                "grounding": {**recipe, "update": done, "route_affine": affine,
                                                              "rank_divergence": divergence}})
                if injector is not None:
                    save_code_injector(injector, folder)
                torch.save({"params": [p.detach().cpu() for p in parameters], "opt": optimizer.state_dict(),
                            "update": done}, out / "resume.tmp")
                os.replace(out / "resume.tmp", resume_path)
            dist.barrier()
        if done in quick_at:
            quick_eval(done)
    dist.barrier()
    if rank == 0:
        (out / "TRAIN_DONE").write_text(json.dumps({"updates": args.updates, "adapter": f"adapter_{args.updates}",
                                                    "sec": round(time.time() - started)}))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
