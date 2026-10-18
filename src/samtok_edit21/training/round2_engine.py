"""Round-2 trainer (docs/08 sections 5.3 and 7): stock Qwen-Image-2.1 DiT + LoRA + code injector, C1 or C1b rows.

C1 starts a fresh LoRA; C1b warm-starts from a C1 adapter (``--init-adapter``) with the C1b rows, a cosine decay
and heavier hard negatives::

    ... round2-train --data <round2 dir> --output <run dir>/train --rows rows_c1b.jsonl \\
        --init-adapter <C1 run>/train/adapter_1500 --updates 800 --warmup 20 --lr-schedule cosine --heavy-weight 8

The model sees what inference sees (source image + prompt with tokens); regions enter only through the loss
(``training.round2``).  Same distributed recipe as the round-1 trainer: every rank processes its own rows, gradients
are averaged, all ranks start from rank 0's weights and the checkpoints verify that they still agree::

    python -m torch.distributed.run --nproc_per_node 8 -m samtok_edit21 round2-train \\
        --data <round2 dir> --output <run dir>/train

Global row ``(update * accumulation + step) * world + rank`` of the rows file is processed by ``rank`` at that
step, so the batch of every update is fixed by the data, whatever the number of GPUs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from datetime import timedelta
from pathlib import Path

import torch

from samtok_edit21.training.grounding_engine import average_gradients, broadcast_parameters, rank_divergence
from samtok_edit21.training.routing import DEFAULT_LAYERS

KIND_INDEX = {"hl": 0, "hl_ref": 1, "hlm": 2, "edit": 3, "edit_v1": 4, "edit_multi": 5, "edit_multi_v1": 6}


def parser():
    from samtok_edit21.models.pipeline import DEFAULT_QWEN, DEFAULT_SAMTOK

    integers = lambda text: tuple(int(x) for x in text.split(",") if x.strip())  # noqa: E731
    p = argparse.ArgumentParser(prog="samtok-edit round2-train", description=__doc__.split("\n\n")[0])
    p.add_argument("--data", required=True, help="Directory written by `samtok-edit round2-data`")
    p.add_argument("--output", required=True)
    p.add_argument("--rows", default="rows_c1.jsonl")
    p.add_argument("--eval-items", default="eval_c1.json")
    p.add_argument("--quick-items", default="eval_c1_quick_idx.json")
    p.add_argument("--init-adapter", help="Warm start: LoRA, code injector and routing calibrations of a round-2 adapter")
    p.add_argument("--qwen", default=DEFAULT_QWEN)
    p.add_argument("--samtok", default=DEFAULT_SAMTOK)
    p.add_argument("--updates", type=int, default=2000)
    p.add_argument("--accumulation", type=int, default=4, help="Rows per rank per optimizer step")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant",
                   help="After warm-up: constant, or cosine from --lr down to --lr-min at the last update")
    p.add_argument("--lr-min", type=float, default=1e-5)
    p.add_argument("--heavy-weight", type=float, default=3.0,
                   help="Weight of hard negatives (distractors, other spans) in both routing terms")
    p.add_argument("--weight-decay", type=float, default=0.01, help="LoRA only; injector and routing calibration use 0")
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--lambda-route", type=float, default=0.05, help="Write-side routing weight")
    p.add_argument("--lambda-read", type=float, default=0.05, help="Read-side routing weight; 0 disables it")
    p.add_argument("--lambda-region", type=float, default=1.0)
    p.add_argument("--lambda-distractor", type=float, default=1.0)
    p.add_argument("--route-layers", type=integers, default=DEFAULT_LAYERS)
    p.add_argument("--checkpoints", type=integers, default=(500, 1000, 1500, 2000))
    p.add_argument("--quick-eval", type=integers, default=(0, 500, 1000, 1500, 2000))
    p.add_argument("--quick-limit", type=int, default=999, help="Quick-eval items per rank (smoke tests)")
    p.add_argument("--max-pixels", type=int, default=1024 * 1024)
    p.add_argument("--seed", type=int, default=2000, help="Rank r draws timesteps and noise from seed + r")
    p.add_argument("--resume", action="store_true")
    return p


def learning_rate(args, update):
    """Linear warm-up, then constant or cosine decay to ``lr_min`` at the last update (0-based ``update``)."""
    if update < args.warmup:
        return args.lr * (update + 1) / max(1, args.warmup)
    if args.lr_schedule == "constant":
        return args.lr
    progress = (update - args.warmup) / max(1, args.updates - 1 - args.warmup)
    return args.lr_min + 0.5 * (args.lr - args.lr_min) * (1 + math.cos(math.pi * min(1.0, progress)))


def main(argv=None):
    import torch.distributed as dist

    from samtok_edit21.evaluation.round2_eval import run_item, summarize
    from samtok_edit21.models.code_injector import CodeInjector, attach_code_injector, load_code_injector, save_code_injector
    from samtok_edit21.models.codec import load_codec
    from samtok_edit21.models.pipeline import load_pipeline
    from samtok_edit21.training.objectives import add_adapter, load_adapter, save_adapter
    from samtok_edit21.training.round2 import Context, prepare_round2, round2_loss
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

    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    pipe = load_pipeline(args.qwen, args.samtok, device=device)
    pipe.scheduler.set_timesteps(1000, training=True)
    if args.init_adapter:
        init_config = load_adapter(pipe.dit, args.init_adapter, trainable=True)
    else:
        add_adapter(pipe.dit, "stage2", args.rank, 0.0)
    pipe.dit.train()
    codec = load_codec(args.samtok, device)
    injector = attach_code_injector(pipe, CodeInjector.from_codec(codec, pipe.processor.tokenizer))
    router = AttentionRouter(pipe.dit, args.route_layers, read=args.lambda_read > 0).to(device)
    if args.init_adapter:
        load_code_injector(injector, args.init_adapter)
        injector.requires_grad_(True)
        grounding = init_config.get("grounding", {})
        if list(grounding.get("route_layers", args.route_layers)) != list(args.route_layers):
            raise ValueError("--route-layers differ from the warm-start adapter's routing layers")
        with torch.no_grad():
            router.affine.copy_(torch.tensor(grounding["route_affine"], dtype=router.affine.dtype))
            if router.read_affine is not None and grounding.get("read_affine") is not None:
                router.read_affine.copy_(torch.tensor(grounding["read_affine"], dtype=router.read_affine.dtype))

    lora = [p for p in pipe.dit.parameters() if p.requires_grad]
    extra = list(injector.proj.parameters()) + [router.affine] + ([router.read_affine] if router.read_affine is not None else [])
    parameters = lora + extra
    optimizer = torch.optim.AdamW([{"params": lora, "weight_decay": args.weight_decay},
                                   {"params": extra, "weight_decay": 0.0}], lr=args.lr)

    ctx = Context(data)
    rows = [json.loads(line) for line in open(data / args.rows)]
    needed = args.updates * args.accumulation * world
    if needed > len(rows):
        raise ValueError(f"{needed} rows needed, {args.rows} has {len(rows)}")
    items = json.load(open(data / args.eval_items))
    quick = [items[i] | {"idx": i} for i in json.load(open(data / args.quick_items))]
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
    else:
        broadcast_parameters(parameters, dist)
    divergence = rank_divergence(parameters, dist, world)
    if divergence > 1e-6:
        raise RuntimeError(f"Ranks start from different weights (relative checksum gap {divergence:.3g})")

    recipe = {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(args).items()
              if k not in ("output", "resume", "quick_limit")} | {"world": world, "synced_init": True}
    log("train_metrics.jsonl", {"start": start, "recipe": recipe, "trainable": sum(p.numel() for p in parameters),
                               "rows": len(rows), "needed": needed})
    started = time.time()

    def quick_eval(tag):
        pipe.dit.eval()
        records = []
        for position, item in enumerate(quick):
            try:
                record, image, _ = run_item(pipe, codec, item, ctx, steps=20)
                record["idx"] = item["idx"]
                records.append(record)
                if position < 2:
                    folder = out / "quick_images" / f"step{tag}"
                    folder.mkdir(parents=True, exist_ok=True)
                    image.resize((512, round(512 * image.height / image.width))).save(
                        folder / f"r{rank}_{item['set']}_{item['idx']}.jpg", quality=88)
            except Exception as error:
                print(f"[rank {rank}] quick skip {item['idx']}: {error!r}"[:400], flush=True)
        pipe.scheduler.set_timesteps(1000, training=True)
        pipe.dit.train()
        gathered = [None] * world
        dist.all_gather_object(gathered, records)
        if rank == 0:
            merged = [r for part in gathered for r in part]
            json.dump(merged, open(out / f"quick_{tag}.json", "w"))
            log("quick_summary.jsonl", {"quick": tag, **summarize(merged), "sec": round(time.time() - started)})

    if start == 0 and 0 in quick_at:
        quick_eval(0)
    # loss, rows, mse, region, distractor, write, rows with write, read, rows with read, skipped, + per kind counts
    totals = torch.zeros(10 + len(KIND_INDEX), device=device)
    for update in range(start, args.updates):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(args, update)
        optimizer.zero_grad(set_to_none=True)
        for step in range(args.accumulation):
            row = rows[(update * args.accumulation + step) * world + rank]
            try:
                sample = prepare_round2(pipe, ctx, row, max_pixels=args.max_pixels)
                flow, write, read, metrics = round2_loss(pipe, sample, router, lambda_region=args.lambda_region,
                                                         lambda_distractor=args.lambda_distractor,
                                                         heavy_weight=args.heavy_weight)
                loss = flow
                if write is not None:
                    loss = loss + args.lambda_route * write
                if read is not None and args.lambda_read > 0:
                    loss = loss + args.lambda_read * read
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite loss")
                (loss / args.accumulation).backward()
                totals[0] += float(loss.detach())
                totals[1] += 1
                totals[2] += metrics["mse"]
                totals[3] += metrics["reg"]
                totals[4] += metrics["dis"]
                if write is not None:
                    totals[5] += metrics["write"]
                    totals[6] += 1
                if read is not None:
                    totals[7] += metrics["read"]
                    totals[8] += 1
                totals[10 + KIND_INDEX[row["kind"]]] += 1
            except Exception as error:   # counted and logged; a broken image must not end a long run
                totals[9] += 1
                print(f"[rank {rank}] skip {row['kind']} {row.get('source_id')} {row.get('sample_id', '')}: {error!r}"[:400],
                      flush=True)
            finally:
                router.disarm()          # only now: checkpointing re-runs the hooks during backward
        average_gradients(parameters, dist, world)
        optimizer.step()
        done = update + 1
        if done % 10 == 0 or done == args.updates:
            summed = totals.clone()
            dist.all_reduce(summed)
            totals.zero_()
            count = max(float(summed[1]), 1.0)
            norm = float(sum(p.detach().float().norm() ** 2 for p in injector.proj.parameters()) ** 0.5)
            log("train_metrics.jsonl", {
                "update": done, "loss": round(float(summed[0]) / count, 5), "mse": round(float(summed[2]) / count, 5),
                "reg": round(float(summed[3]) / count, 5), "dis": round(float(summed[4]) / count, 5),
                "write": round(float(summed[5]) / max(float(summed[6]), 1.0), 5),
                "read": round(float(summed[7]) / max(float(summed[8]), 1.0), 5),
                "samples": int(summed[1]), "skipped": int(summed[9]),
                "kinds": {k: int(summed[10 + i]) for k, i in KIND_INDEX.items()},
                "sec": round(time.time() - started), "lr": optimizer.param_groups[0]["lr"], "inj_norm": round(norm, 4),
                "affine": [round(float(x), 3) for x in router.affine.detach().flatten()[:2]],
                "read_affine": [] if router.read_affine is None else [round(float(x), 3) for x in router.read_affine.detach().flatten()[:2]]})
        if done in checkpoints or done == args.updates:
            dist.barrier()
            divergence = rank_divergence(parameters, dist, world)
            if divergence > 1e-4:
                raise RuntimeError(f"Ranks diverged at update {done} (relative checksum gap {divergence:.3g})")
            if rank == 0:
                folder = out / f"adapter_{done}"
                affine = [[float(x) for x in pair] for pair in router.affine.detach().cpu()]
                read_affine = None if router.read_affine is None else [[float(x) for x in pair] for pair in router.read_affine.detach().cpu()]
                save_adapter(pipe.dit, folder, {"stage": "stage2", "binding": {"mode": "none"},
                                                "grounding": {**recipe, "round": 2, "update": done, "route_affine": affine,
                                                              "read_affine": read_affine, "rank_divergence": divergence}})
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
