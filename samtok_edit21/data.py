"""Validated metadata, exact global-batch ratios, and reproducible cache IO."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from .protocol import TYPE_WEIGHTS, validate_row

EXPERIMENT_ROOT = (
    "/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/qwen_image_2_1"
)
RATIOS = {
    "stage1": {"edit_ntp": 3, "edit_umt:ref": 2, "edit_umt:noref": 2, "edit": 1},
    "stage2": {"edit_umt:ref": 1, "edit_umt:noref": 2, "edit": 1},
}


def row_kind(row):
    return row["sample_type"] + (
        ":" + row["instr_variant"] if row["sample_type"] == "edit_umt" else ""
    )


def read_rows(path):
    rows = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(validate_row(json.loads(line)))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{i}: {exc}") from exc
    if not rows:
        raise ValueError(f"Empty metadata: {path}")
    return rows


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w") as f:
        for row in rows:
            validate_row(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def row_hash(row):
    return hashlib.sha256(
        json.dumps(row, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def dimensions(image, max_pixels):
    scale = min(1.0, math.sqrt(max_pixels / (image.width * image.height)))
    return max(32, round(image.width * scale / 32) * 32), max(
        32, round(image.height * scale / 32) * 32
    )


def load_images(row, base_path, max_pixels):
    def load(p):
        with Image.open(Path(base_path) / p) as im:
            return im.convert("RGBA")

    sources = row["edit_image"]
    sources = [sources] if isinstance(sources, str) else sources
    images = [load(p) for p in sources]
    target = load(row["image"]) if "image" in row else None
    width, height = dimensions(target or images[0], max_pixels)
    if target is not None:
        target = target.resize((width, height), Image.Resampling.LANCZOS)
    return images, target, height, width


def make_schedule(rows, stage, world_size, accumulation, *, steps=None, seed=0):
    """Return global position-major row indices; slice [rank::world_size].

    Homogeneous rank steps are preferred when accumulation allows them; otherwise
    the exact ratio is distributed across ranks. Stage 2 always follows its ratio,
    including ref:noref=1:2, rather than globally shuffling an imbalanced cache.
    """
    if world_size < 1 or accumulation < 1 or (steps is not None and steps < 1):
        raise ValueError("world_size, accumulation and steps must be positive")
    ratio = RATIOS[stage]
    pools = defaultdict(lambda: defaultdict(list))
    for i, row in enumerate(rows):
        kind = row_kind(row)
        if kind in ratio:
            pools[kind][row["edit_type"]].append(i)
        elif stage == "stage2":
            raise ValueError("Stage 2 metadata/cache must exclude NTP")
    missing = set(ratio) - set(pools)
    if missing:
        raise ValueError(f"Requested sampling pools are absent: {sorted(missing)}")
    global_batch = world_size * accumulation
    block_len = sum(ratio.values())
    if global_batch % block_len:
        raise ValueError(f"world_size * accumulation must be divisible by {block_len}")
    per_step = {k: v * global_batch // block_len for k, v in ratio.items()}
    if steps is None:
        steps = max(
            math.ceil(sum(map(len, pools[k].values())) / per_step[k]) for k in ratio
        )
    rng, queues = random.Random(seed), {}

    def draw(kind):
        types = sorted(pools[kind])
        weights = [TYPE_WEIGHTS[t] for t in types]
        if kind == "edit":
            weights = [len(pools[kind][t]) for t in types]
            # Natural distribution with individual background/global cap 15%.
            weights = capped_plain_weights(types, weights)
        typ = rng.choices(types, weights=weights)[0]
        key = (kind, typ)
        if not queues.get(key):
            queues[key] = pools[kind][typ].copy()
            rng.shuffle(queues[key])
        return queues[key].pop()

    schedule = []
    for _ in range(steps):
        if accumulation % block_len == 0:
            kinds = [
                k
                for k, n in ratio.items()
                for _ in range(n * accumulation // block_len)
            ]
            rng.shuffle(kinds)
            kinds = [k for k in kinds for _ in range(world_size)]
        else:
            kinds = [k for k, n in per_step.items() for _ in range(n)]
            rng.shuffle(kinds)
        schedule.extend(draw(k) for k in kinds)
    draw_counts = Counter(schedule)

    def exposure(indices):
        counts = [draw_counts[i] for i in indices]
        draws = sum(counts)
        unique = sum(count > 0 for count in counts)
        return {
            "source_rows": len(indices),
            "draws": draws,
            "mean_draws_per_row": draws / len(indices),
            "unique_rows": unique,
            "unseen_rows": len(indices) - unique,
            "min_draws_per_row": min(counts),
            "max_draws_per_row": max(counts),
        }

    pool_exposure = {}
    for kind, by_type in pools.items():
        indices = [i for group in by_type.values() for i in group]
        pool_exposure[kind] = {
            **exposure(indices),
            "by_edit_type": {
                edit_type: exposure(group)
                for edit_type, group in sorted(by_type.items())
            },
        }
    report = {
        "steps": steps,
        "global_batch": global_batch,
        "per_step": per_step,
        "realized": dict(Counter(row_kind(rows[i]) for i in schedule)),
        "edit_types": dict(Counter(rows[i]["edit_type"] for i in schedule)),
        "unique_rows": len(draw_counts),
        "pool_exposure": pool_exposure,
        "draws": len(schedule),
        "absent_edit_types": {
            k: sorted(set(TYPE_WEIGHTS) - set(pools[k])) for k in ratio
        },
        "sampling": "weighted with shuffled per-type pools; pool recycling is explicit",
    }
    return schedule, report


def capped_plain_weights(types, weights):
    """Cap each rare type's sampling probability, renormalizing uncapped types."""
    if not any(t not in {"background", "global"} for t in types):
        raise ValueError(
            "Plain edit needs a non-background/global pool to satisfy the 15% caps"
        )
    fixed = {}
    while True:
        remaining = 1.0 - sum(fixed.values())
        total = sum(w for i, w in enumerate(weights) if i not in fixed)
        probs = [fixed.get(i, remaining * w / total) for i, w in enumerate(weights)]
        over = [
            i
            for i, t in enumerate(types)
            if t in {"background", "global"} and i not in fixed and probs[i] > 0.15
        ]
        if not over:
            return probs
        fixed.update({i: 0.15 for i in over})
