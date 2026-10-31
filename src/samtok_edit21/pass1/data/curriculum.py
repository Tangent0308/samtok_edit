"""Validate every public request/context and freeze a one-pass curriculum."""

from __future__ import annotations

import argparse
import collections
import importlib
import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

from samtok_edit21.pass1.io import file_hash
from samtok_edit21.pass1.model import encode, prefix_text, processor_for
from samtok_edit21.pass1.protocol import parse_answer, validate_request


def pack_batches(rows, indices, capacity, seed, stage):
    """Keep both counterfactual members within a global optimizer update."""
    rng = random.Random(seed)
    pairs = collections.defaultdict(list)
    blocks = []
    for index in indices:
        row = rows[index]
        if row.get("pair_id"):
            pairs[row["pair_id"]].append(index)
        else:
            blocks.append([index])
    if any(len(v) != 2 for v in pairs.values()):
        raise ValueError("Incomplete counterfactual pair")
    blocks += list(pairs.values())
    rng.shuffle(blocks)
    batches = []
    current = []
    for block in blocks:
        if len(current) + len(block) > capacity:
            batches.append(
                {"stage": stage, "indices": current + [-1] * (capacity - len(current))}
            )
            current = []
        current.extend(block)
    if current:
        batches.append(
            {"stage": stage, "indices": current + [-1] * (capacity - len(current))}
        )
    flat = [i for b in batches for i in b["indices"] if i >= 0]
    if sorted(flat) != sorted(indices):
        raise AssertionError("Batch packing dropped or duplicated requests")
    return batches


def context_length(processor, row):
    ip = processor.image_processor
    native = importlib.import_module(type(ip).__module__)
    w, h = row["size"]
    hh, ww = native.smart_resize(
        height=h,
        width=w,
        factor=ip.patch_size * ip.merge_size,
        min_pixels=ip.size["shortest_edge"],
        max_pixels=ip.size["longest_edge"],
    )
    vision_tokens = (hh // ip.patch_size) * (ww // ip.patch_size) // (ip.merge_size**2)
    suffix = (
        json.dumps(row["answer"], ensure_ascii=False, separators=(",", ":"))
        + "<|im_end|>"
    )
    text = prefix_text(processor, row["request"]) + suffix
    ids = processor.tokenizer.encode(text, add_special_tokens=False)
    image_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    if ids.count(image_id) != 1:
        raise ValueError("Expected one source image placeholder")
    suffix_ids = processor.tokenizer.encode(suffix, add_special_tokens=False)
    if ids[-len(suffix_ids) :] != suffix_ids:
        raise ValueError("Assistant suffix boundary mismatch")
    return len(ids) + vision_tokens - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--materialized", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=2026103005)
    ap.add_argument("--world-size", type=int, default=8)
    ap.add_argument("--accum", type=int, default=8)
    args = ap.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.materialized / "manifest.json").read_text())
    rejected = manifest["planned_requests"] - manifest["accepted_requests"]
    if rejected / max(1, manifest["planned_requests"]) > 0.01:
        raise ValueError("More than 1% of requests rejected; inspect before training")
    started = time.time()
    proc = processor_for()
    by_split = {}
    context_stats = {}
    native_examples = []
    ids = set()
    for name in ["warmup", "main", "dev", "test"]:
        rows = []
        lengths = []
        with (args.materialized / (name + ".jsonl")).open() as f:
            for line in f:
                row = json.loads(line)
                if row["id"] in ids:
                    raise ValueError("Duplicate request ID")
                ids.add(row["id"])
                validate_request(row["request"])
                if not parse_answer(json.dumps(row["answer"]), row["request"])[
                    "complete"
                ]:
                    raise ValueError("Invalid teacher answer")
                length = context_length(proc, row)
                if length > 8192:
                    raise ValueError("Context overflow " + row["id"])
                lengths.append(length)
                if (
                    len(rows) < 4
                    or len(row["answer"]) == 5
                    and sum(len(x["answer"]) == 5 for x in native_examples) < 16
                ):
                    native_examples.append(row)
                minimal = {
                    k: row[k]
                    for k in [
                        "id",
                        "group",
                        "image",
                        "image_sha256",
                        "size",
                        "request",
                        "answer",
                    ]
                }
                minimal["stage"] = name
                for k in ["pair_id", "pair_role", "contrast_label"]:
                    if k in row:
                        minimal[k] = row[k]
                rows.append(minimal)
                if len(rows) % 10000 == 0:
                    print(
                        json.dumps(
                            {
                                "phase": name,
                                "validated": len(rows),
                                "elapsed_s": time.time() - started,
                            }
                        ),
                        flush=True,
                    )
        if not rows:
            raise ValueError("Empty split " + name)
        by_split[name] = rows
        context_stats[name] = {
            "requests": len(rows),
            "min": min(lengths),
            "max": max(lengths),
            "mean": float(np.mean(lengths)),
            "p95": float(np.quantile(lengths, 0.95)),
        }
    # Cross-check the analytical expansion against the real native image processor.
    audited = []
    for row in native_examples:
        with Image.open(row["image"]) as im:
            x, _, _, _ = encode(proc, im, row["request"], row["answer"])
        expected = context_length(proc, row)
        actual = x.input_ids.shape[1]
        if actual != expected:
            raise ValueError(
                f"Native context expansion differs: {row['id']} {expected} != {actual}"
            )
        audited.append({"id": row["id"], "native_tokens": actual})
    groups = {k: {r["group"] for r in rows} for k, rows in by_split.items()}
    train_groups = groups["warmup"] | groups["main"]
    if (
        train_groups & (groups["dev"] | groups["test"])
        or groups["dev"] & groups["test"]
    ):
        raise ValueError("Source split leakage")
    rows = by_split["warmup"] + by_split["main"]
    warm_n = len(by_split["warmup"])
    path = args.output / "train.jsonl"
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    capacity = args.world_size * args.accum
    warm_batches = pack_batches(
        rows, list(range(warm_n)), capacity, args.seed, "warmup"
    )
    main_batches = pack_batches(
        rows, list(range(warm_n, len(rows))), capacity, args.seed + 1, "main"
    )
    plan = {
        "world_size": args.world_size,
        "accumulation": args.accum,
        "data_sha256": file_hash(path),
        "seed": args.seed,
        "warmup_requests": warm_n,
        "main_requests": len(rows) - warm_n,
        "warmup_steps": len(warm_batches),
        "main_steps": len(main_batches),
        "batches": warm_batches + main_batches,
    }
    (args.output / "batch_plan.json").write_text(
        json.dumps(plan, separators=(",", ":")) + "\n"
    )
    for name in ["dev", "test"]:
        # Keep full GT only for offline scoring. Infer reads image + request only.
        import shutil

        shutil.copyfile(
            args.materialized / (name + ".jsonl"), args.output / (name + ".jsonl")
        )
    total = len(plan["batches"])
    warm = len(warm_batches)
    candidates = sorted(set([warm, *range(warm + 500, total, 500), total]))
    summary = {
        "status": "ready",
        "training_requests": len(rows),
        "train_sources": len(train_groups),
        "context": context_stats,
        "native_processor_checks": audited,
        "total_steps": total,
        "warmup_steps": warm,
        "main_steps": len(main_batches),
        "dev_candidate_steps": candidates,
        "masked_padding_slots": total * capacity - len(rows),
        "all_real_requests_used_exactly_once": True,
        "counterfactual_pairs_co_batched": True,
        "data_sha256": plan["data_sha256"],
        "batch_plan_sha256": file_hash(args.output / "batch_plan.json"),
        "selection_metric": "dev unknown.strict_iou; exact ties choose earlier step; test only after selection",
        "materialization_rejections": rejected,
        "elapsed_s": time.time() - started,
    }
    (args.output / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    # Two real global batches for the resumability smoke; these weights are discarded.
    smoke_indices = [i for b in plan["batches"][:2] for i in b["indices"] if i >= 0]
    smoke = args.output / "smoke"
    smoke.mkdir()
    with (smoke / "train.jsonl").open("w") as f:
        for i in smoke_indices:
            f.write(
                json.dumps(rows[i], ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    smoke_batches = [
        {"stage": "smoke", "indices": list(range(j * capacity, (j + 1) * capacity))}
        for j in range(2)
    ]
    if len(smoke_indices) != 2 * capacity:
        raise ValueError("Smoke expects two full batches")
    (smoke / "batch_plan.json").write_text(
        json.dumps(
            {
                "world_size": args.world_size,
                "accumulation": args.accum,
                "data_sha256": file_hash(smoke / "train.jsonl"),
                "batches": smoke_batches,
            },
            separators=(",", ":"),
        )
        + "\n"
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
