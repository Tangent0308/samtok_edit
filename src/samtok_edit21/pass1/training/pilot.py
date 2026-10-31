"""torchrun-compatible assistant-only Pass1 LoRA pilot. No DiT is loaded."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset

from samtok_edit21.pass1.model import (
    encode,
    frozen_fingerprint,
    grouped_loss,
    load_model,
    processor_for,
)
from samtok_edit21.pass1.training.checkpoint import pass2_probe, save_adapter, singleton


class Requests(Dataset):
    def __init__(self, path, max_pixels):
        self.path = Path(path)
        self.root = self.path.parent
        self.rows = [json.loads(x) for x in self.path.read_text().splitlines()]
        self.processor = processor_for(max_pixels=max_pixels)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        with Image.open(self.root / r["image"]) as im:
            x, p, y, g = encode(self.processor, im, r["request"], r["answer"])
        if x.input_ids.shape[1] > 8192:
            raise ValueError(f"Context overflow: {r['id']}")
        return x, p, y, g, r["id"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--save-steps", type=int, nargs="+", default=[64, 128])
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--max-pixels", type=int, default=2097152)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2026103001)
    args = ap.parse_args()
    local = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local)
    device = f"cuda:{local}"
    if world > 1:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
    else:
        rank = 0

    def barrier():
        if world > 1:
            dist.barrier()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    output = Path(args.output)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
        (output / "config.json").write_text(
            json.dumps(
                vars(args)
                | {"world_size": world, "effective_batch": world * args.accum},
                indent=2,
            )
            + "\n"
        )
    barrier()
    data = Requests(args.data, args.max_pixels)
    need = args.steps * args.accum * world
    if need > len(data):
        raise ValueError(f"Need {need} requests, have {len(data)}")
    indices = list(range(need))[rank::world]
    loader = DataLoader(
        data,
        batch_size=1,
        sampler=indices,
        num_workers=args.workers,
        collate_fn=singleton,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    model = load_model(device, train=True, rank=args.rank)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if any("lora_" not in n for n, p in model.named_parameters() if p.requires_grad):
        raise ValueError("Non-LoRA trainable parameter")
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    model.config.use_cache = False
    before_hash = probe = None
    if rank == 0:
        before_hash = frozen_fingerprint(model)
        probe = pass2_probe(model, data, device)
        print(
            json.dumps(
                {
                    "trainable": sum(p.numel() for p in trainable),
                    "frozen": sum(
                        p.numel() for p in model.parameters() if not p.requires_grad
                    ),
                    "frozen_sha256_before": before_hash,
                }
            ),
            flush=True,
        )
    barrier()
    ddp = (
        DDP(
            model,
            device_ids=[local],
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
        if world > 1
        else model
    )
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8
    )
    optimizer.zero_grad(set_to_none=True)
    log = (output / "train.jsonl").open("w") if rank == 0 else None
    start = time.time()
    step = 0
    stats_sum = {}
    tokens = 0
    loss_sum = 0.0
    max_context = 0
    for micro, (inputs, positions, labels, groups, uid) in enumerate(loader):
        end = (micro + 1) % args.accum == 0
        step_idx = micro // args.accum + 1
        warmup = max(1, round(args.steps * 0.03))
        scale = (
            step_idx / warmup
            if step_idx <= warmup
            else 0.1
            + 0.9
            * 0.5
            * (
                1
                + math.cos(math.pi * (step_idx - warmup) / max(1, args.steps - warmup))
            )
        )
        for group in optimizer.param_groups:
            group["lr"] = args.lr * scale
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
        positions = positions.to(device)
        labels = labels.to(device)
        groups = groups.to(device)
        max_context = max(max_context, inputs["input_ids"].shape[1])
        tokens += inputs["input_ids"].numel()
        sync = contextlib.nullcontext() if end or world == 1 else ddp.no_sync()
        model.model.rope_deltas = None
        with sync:
            out = ddp(**inputs, logits_to_keep=positions, use_cache=False)
            loss, stats = grouped_loss(out.logits[0], labels, groups)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss {uid}")
            (loss / args.accum).backward()
        loss_sum += float(loss.detach())
        for k, v in stats.items():
            stats_sum[k] = stats_sum.get(k, 0) + v / args.accum
        if end:
            norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite gradients")
            if step_idx == 1 and float(norm) == 0:
                raise RuntimeError("Zero initial LoRA gradients")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step = step_idx
            mean_loss = torch.tensor(loss_sum / args.accum, device=device)
            if world > 1:
                dist.all_reduce(mean_loss)
                mean_loss /= world
            if rank == 0:
                row = {
                    "step": step,
                    "loss": float(mean_loss),
                    "rank0_group_loss": stats_sum,
                    "lr": optimizer.param_groups[0]["lr"],
                    "grad_norm": float(norm),
                    "elapsed_s": time.time() - start,
                    "max_context_rank0": max_context,
                    "peak_gpu_gb_rank0": torch.cuda.max_memory_allocated() / 1e9,
                    "requests_seen": step * args.accum * world,
                }
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(json.dumps(row), flush=True)
            loss_sum = 0.0
            stats_sum = {}
            if step in args.save_steps or step == args.steps:
                barrier()
                if rank == 0:
                    save_adapter(
                        model,
                        output / f"step_{step:04d}",
                        vars(args)
                        | {"step": step, "requests_seen": step * args.accum * world},
                    )
                barrier()
    barrier()
    if rank == 0:
        after = frozen_fingerprint(model)
        after_probe = pass2_probe(model, data, device)
        checks = {
            k: {
                "equal": torch.equal(probe[k], after_probe[k]),
                "max_abs_diff": float(
                    (probe[k].float() - after_probe[k].float()).abs().max()
                ),
            }
            for k in probe
        }
        audit = {
            "frozen_sha256_before": before_hash,
            "frozen_sha256_after": after,
            "all_frozen_parameters_unchanged": after == before_hash,
            "adapter_off_probe": checks,
            "steps": step,
            "duration_s": time.time() - start,
        }
        (output / "freeze_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        print(json.dumps(audit), flush=True)
        log.close()
        if before_hash != after or any(not c["equal"] for c in checks.values()):
            raise RuntimeError("Frozen/adapter-off invariance failed")
    barrier()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
