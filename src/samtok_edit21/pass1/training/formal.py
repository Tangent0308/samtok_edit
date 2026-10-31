"""Resumable Pass1 LM-LoRA training over a frozen, explicitly batched curriculum.

Only LoRA weights are optimized. Padding slots have exactly zero loss; the last
partial batch is normalized by its real request count. Checkpoints include the
optimizer and every rank's RNG state, and become resumable only after COMPLETE.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
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

from samtok_edit21.pass1.io import file_hash
from samtok_edit21.pass1.model import (
    encode,
    frozen_fingerprint,
    grouped_loss,
    load_model,
    processor_for,
)
from samtok_edit21.pass1.training.checkpoint import pass2_probe, save_adapter, singleton


def lr_for(step, total, peak):
    warmup = max(1, round(total * 0.03))
    scale = (
        step / warmup
        if step <= warmup
        else 0.1
        + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
    )
    return peak * scale


def rank_samples(batches, start, stop, rank, world, accum):
    """Return (row index, gradient multiplier); DDP averages across ranks."""
    result = []
    capacity = world * accum
    for batch in batches[start:stop]:
        ids = batch["indices"]
        real = sum(i >= 0 for i in ids)
        if len(ids) != capacity or real == 0:
            raise ValueError("Invalid explicit global batch")
        for micro in range(accum):
            index = ids[micro * world + rank]
            result.append((max(0, index), capacity / real if index >= 0 else 0.0))
    return result


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(),
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"])


class Requests(Dataset):
    def __init__(self, path, max_pixels):
        self.path = Path(path)
        self.rows = [json.loads(x) for x in self.path.read_text().splitlines()]
        self.processor = processor_for(max_pixels=max_pixels)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, item):
        index, weight = item if isinstance(item, tuple) else (item, 1.0)
        row = self.rows[index]
        with Image.open(row["image"]) as im:
            x, p, y, g = encode(self.processor, im, row["request"], row["answer"])
        if x.input_ids.shape[1] > 8192:
            raise ValueError("Context overflow " + row["id"])
        return x, p, y, g, row["id"], weight


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--batch-plan", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--resume")
    ap.add_argument("--stop-after-step", type=int)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--max-pixels", type=int, default=2097152)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=2026103005)
    ap.add_argument("--save-every", type=int, default=100)
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

    plan = json.loads(Path(args.batch_plan).read_text())
    batches = plan["batches"]
    total = len(batches)
    if plan["world_size"] != world or plan["accumulation"] != args.accum:
        raise ValueError("Batch plan topology mismatch")
    output = Path(args.output)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        if not args.resume and (output / "config.json").exists():
            raise FileExistsError("Fresh training cannot overwrite an existing run")
        if file_hash(args.data) != plan["data_sha256"]:
            raise ValueError("Prepared training file changed")
    barrier()
    state = None
    start_step = 0
    if args.resume:
        checkpoint = Path(args.resume)
        if not (checkpoint / "COMPLETE.json").is_file():
            raise ValueError("Incomplete checkpoint")
        complete = json.loads((checkpoint / "COMPLETE.json").read_text())
        if complete["batch_plan_sha256"] != file_hash(args.batch_plan):
            raise ValueError("Batch plan changed on resume")
        state = torch.load(
            checkpoint / "training_state.pt", map_location="cpu", weights_only=False
        )
        start_step = state["step"]
        if len(state["rng_states"]) != world:
            raise ValueError("Resume requires the same number of ranks")
    stop = min(args.stop_after_step or total, total)
    if not 0 <= start_step < stop <= total:
        raise ValueError("Invalid resume/stop interval")
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    dataset = Requests(args.data, args.max_pixels)
    samples = rank_samples(batches, start_step, stop, rank, world, args.accum)
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=samples,
        num_workers=args.workers,
        collate_fn=singleton,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        generator=torch.Generator().manual_seed(args.seed + rank),
    )
    model = load_model(device, adapter=args.resume, train=True, rank=args.rank)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if any("lora_" not in n for n, p in model.named_parameters() if p.requires_grad):
        raise ValueError("Non-LoRA parameter trainable")
    if any(p.dtype != torch.float32 for p in trainable):
        raise ValueError("LoRA parameters must stay FP32")
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    model.config.use_cache = False
    frozen_before = probe_before = None
    if rank == 0:
        frozen_before = frozen_fingerprint(model)
        probe_before = pass2_probe(model, dataset, device)
        if state:
            if frozen_before != state["frozen_sha256"]:
                raise RuntimeError("Frozen base changed across resume")
            if any(
                not torch.equal(probe_before[k], state["base_probe"][k])
                for k in probe_before
            ):
                raise RuntimeError("Adapter-off probe changed across resume")
        else:
            (output / "config.json").write_text(
                json.dumps(
                    vars(args)
                    | {
                        "world_size": world,
                        "total_steps": total,
                        "effective_batch": world * args.accum,
                        "planned_real_requests": len(dataset),
                        "trainable_parameters": sum(p.numel() for p in trainable),
                        "batch_plan_sha256": file_hash(args.batch_plan),
                        "frozen_sha256": frozen_before,
                    },
                    indent=2,
                )
                + "\n"
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
    if state:
        optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng_states"][rank])
    optimizer.zero_grad(set_to_none=True)
    attempt = f"{start_step:04d}_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    log = (output / "train.jsonl").open("a") if rank == 0 else None
    started = time.time()
    loss_sum = 0.0
    count = 0
    max_context = 0
    requests_seen = sum(sum(i >= 0 for i in b["indices"]) for b in batches[:start_step])
    if rank == 0:
        print(
            json.dumps(
                {
                    "event": "training_started",
                    "start_step": start_step,
                    "stop_step": stop,
                    "total_steps": total,
                    "resume": args.resume,
                    "attempt": attempt,
                }
            ),
            flush=True,
        )

    def checkpoint(step):
        barrier()
        local_rng = rng_state()
        rngs = [None] * world
        if world > 1:
            dist.all_gather_object(rngs, local_rng)
        else:
            rngs = [local_rng]
        if rank == 0:
            path = output / f"step_{step:04d}"
            temporary = output / f".step_{step:04d}_{attempt}.partial"
            if path.exists():
                raise FileExistsError(path)
            save_adapter(
                model,
                temporary,
                vars(args)
                | {
                    "step": step,
                    "total_steps": total,
                    "requests_seen": requests_seen,
                    "stage": batches[step - 1]["stage"],
                },
            )
            torch.save(
                {
                    "step": step,
                    "optimizer": optimizer.state_dict(),
                    "rng_states": rngs,
                    "frozen_sha256": frozen_before,
                    "base_probe": probe_before,
                },
                temporary / "training_state.pt",
            )
            complete = {
                "step": step,
                "requests_seen": requests_seen,
                "batch_plan_sha256": file_hash(args.batch_plan),
                "adapter_sha256": file_hash(temporary / "adapter_model.safetensors"),
                "training_state_bytes": (temporary / "training_state.pt")
                .stat()
                .st_size,
                "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
            (temporary / "COMPLETE.json").write_text(
                json.dumps(complete, indent=2) + "\n"
            )
            temporary.rename(path)
            print(
                json.dumps(
                    {"event": "checkpoint_saved", "step": step, "path": str(path)}
                ),
                flush=True,
            )
        barrier()

    for micro, (inputs, positions, labels, groups, uid, weight) in enumerate(loader):
        step = start_step + micro // args.accum + 1
        end = (micro + 1) % args.accum == 0
        for group in optimizer.param_groups:
            group["lr"] = lr_for(step, total, args.lr)
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
        positions = positions.to(device)
        labels = labels.to(device)
        groups = groups.to(device)
        max_context = max(max_context, inputs["input_ids"].shape[1])
        model.model.rope_deltas = None
        with contextlib.nullcontext() if end or world == 1 else ddp.no_sync():
            out = ddp(**inputs, logits_to_keep=positions, use_cache=False)
            loss, _ = grouped_loss(out.logits[0], labels, groups)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss " + uid)
            (loss * weight / args.accum).backward()
        if weight > 0:
            loss_sum += float(loss.detach())
            count += 1
        if not end:
            continue
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        if not torch.isfinite(norm) or (step == 1 and float(norm) == 0):
            raise FloatingPointError("Invalid gradient norm")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        totals = torch.tensor([loss_sum, count], device=device, dtype=torch.float64)
        if world > 1:
            dist.all_reduce(totals)
        real = sum(i >= 0 for i in batches[step - 1]["indices"])
        requests_seen += real
        if int(totals[1]) != real:
            raise RuntimeError("Real batch accounting mismatch")
        if rank == 0:
            row = {
                "step": step,
                "total_steps": total,
                "stage": batches[step - 1]["stage"],
                "loss": float(totals[0] / totals[1]),
                "lr": optimizer.param_groups[0]["lr"],
                "grad_norm": float(norm),
                "elapsed_s": time.time() - started,
                "attempt": attempt,
                "attempt_start_step": start_step,
                "requests_seen": requests_seen,
                "real_batch_size": real,
                "max_context_rank0": max_context,
                "peak_gpu_gb_rank0": torch.cuda.max_memory_allocated() / 1e9,
                "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(json.dumps(row), flush=True)
        loss_sum = 0.0
        count = 0
        if step % args.save_every == 0 or step == stop:
            checkpoint(step)
    barrier()
    if rank == 0:
        frozen_after = frozen_fingerprint(model)
        probe_after = pass2_probe(model, dataset, device)
        checks = {
            k: {
                "equal": torch.equal(probe_before[k], probe_after[k]),
                "max_abs_diff": float(
                    (probe_before[k].float() - probe_after[k].float()).abs().max()
                ),
            }
            for k in probe_before
        }
        audit = {
            "start_step": start_step,
            "stop_step": stop,
            "total_steps": total,
            "all_frozen_parameters_unchanged": frozen_before == frozen_after,
            "frozen_sha256_before": frozen_before,
            "frozen_sha256_after": frozen_after,
            "adapter_off_probe": checks,
            "duration_s": time.time() - started,
            "attempt": attempt,
        }
        (output / f"freeze_audit_{stop:04d}.json").write_text(
            json.dumps(audit, indent=2) + "\n"
        )
        log.close()
        if not audit["all_frozen_parameters_unchanged"] or any(
            not x["equal"] for x in checks.values()
        ):
            raise RuntimeError("Frozen/Pass2 invariance failed")
    barrier()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
