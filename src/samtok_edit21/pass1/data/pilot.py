"""Read-only merged-data sampler; all new files go into an explicit output root."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from pycocotools import mask as mask_utils
from scipy.ndimage import distance_transform_edt

from samtok_edit21.pass1.protocol import (
    VERSION,
    box_normalized,
    point_normalized,
    point_pixels,
    valid_span,
    validate_request,
)

MODES = [
    (True, "none"),
    (True, "point"),
    (True, "box"),
    (True, "mask"),
    (False, "point"),
    (False, "box"),
    (False, "mask"),
]


def decode_rle(r):
    return mask_utils.decode(
        {"size": r["size"], "counts": r["counts"].encode()}
    ).astype(bool)


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def geometry(unit, size, rng):
    s = unit["selection"]
    rle = {"size": s["rle_size"], "counts": s["rle_counts"]}
    mask = decode_rle(rle)
    if tuple(mask.shape) != tuple(size[::-1]) or not mask.any():
        raise ValueError("empty_or_wrong_size")
    area = float(mask.mean())
    if not 0.0005 <= area <= 0.30:
        raise ValueError("area_filter")
    dec = decode_rle(unit["token"]["dec"])
    if dec.shape != mask.shape:
        raise ValueError("decode_wrong_size")
    # Quantization-safe positive clicks: the public click must still lie in S,
    # and its normalized round-trip must be inside the supervised token mask.
    ys, xs = np.where(mask & dec)
    if not len(xs):
        raise ValueError("no_shared_click")
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)
    sy, sx = np.where(mask)
    box = box_normalized(
        [int(sx.min()), int(sy.min()), int(sx.max() + 1), int(sy.max() + 1)], size
    )
    crop = mask[y0:y1, x0:x1]
    dist = distance_transform_edt(np.pad(crop, 1))[1:-1, 1:-1]
    order = np.argsort(-dist[ys - y0, xs - x0])
    candidates = list(order[:32]) + rng.sample(range(len(xs)), min(128, len(xs)))
    clicks = []
    for k in candidates:
        p = point_normalized([int(xs[k]), int(ys[k])], size)
        xx, yy = point_pixels(p, size)
        if mask[yy, xx] and dec[yy, xx] and p not in clicks:
            clicks.append(p)
            if len(clicks) >= 8:
                break
    if not clicks:
        raise ValueError("no_quantization_safe_click")
    return {
        "selection": rle,
        "canonical_dec": unit["token"]["dec"],
        "area": area,
        "box": box,
        "points": clicks,
        "span": unit["token"]["span"],
        "roundtrip": float(
            np.logical_and(mask, dec).sum() / np.logical_or(mask, dec).sum()
        ),
    }


def make_request(group, split, index, rng):
    available = group["units"]
    # 20% single-target rehearsal; otherwise use up to five real same-image units.
    if split == "train" and rng.random() < 0.20:
        chosen = rng.sample(available, 1)
    else:
        chosen = rng.sample(available, min(5, len(available)))
    targets, answers, truth = [], [], []
    for j, u in enumerate(chosen):
        has_ref, loc = MODES[(index + j) % len(MODES)]
        if not has_ref and not u["noref"]:
            has_ref = True
        label = f"U{j + 1}"
        t = {
            "label": label,
            "instruction": u["ref"] if has_ref else u["noref"],
            "has_ref": has_ref,
            "locator": loc,
        }
        if loc == "point":
            t["region"] = {
                "point": rng.choice(u["points"]) if split == "train" else u["points"][0]
            }
        elif loc == "box":
            t["region"] = {"box": u["box"]}
        elif loc == "mask":
            t["region"] = {"mask_2d": u["span"]}
        targets.append(t)
        answers.append({"label": label, "mask_2d": u["span"]})
        truth.append(
            {
                k: u[k]
                for k in [
                    "selection",
                    "canonical_dec",
                    "area",
                    "span",
                    "roundtrip",
                    "family",
                    "record",
                    "original_uid",
                    "operation",
                ]
            }
            | {"label": label}
        )
    request = validate_request(
        {
            "protocol": VERSION,
            "coordinate_system": "normalized_0_1000",
            "targets": targets,
        }
    )
    return {
        "id": f"{split}-{index:05d}",
        "split": split,
        "group": group["group"],
        "image": group["image"],
        "image_sha256": group["image_sha256"],
        "size": group["size"],
        "request": request,
        "answer": answers,
        "truth": truth,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=2026103001)
    ap.add_argument("--train-requests", type=int, default=8192)
    args = ap.parse_args()
    root = Path(args.output).resolve()
    source = Path(args.source).resolve()
    if root == source or source in root.parents or (root / "manifest.json").exists():
        raise ValueError("Require a fresh separate output directory")
    (root / "images").mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    bench = [json.loads(x) for x in Path(args.benchmark).read_text().splitlines()]
    excluded_hashes = {x["source"]["sha256"] for x in bench}
    excluded_ids = {x["source"]["image_id"] for x in bench}
    cols = ["record", "uid", "family", "default", "split", "group"]
    rows = pq.read_table(source / "units.parquet", columns=cols).to_pylist()
    family_groups = collections.defaultdict(set)
    for r in rows:
        if r["default"] and r["split"] == "train":
            family_groups[r["family"]].add(r["group"])
    # D2 dense-pool expansion is deliberately deferred in this first pilot.
    # D1 combines distinct selections on the same source image; D12/D15 join pairs.
    quotas = {
        "D1": 120,
        "D3": 100,
        "D4": 250,
        "D10": 160,
        "D11": 240,
        "D12": 240,
        "D13": 730,
        "D14": 120,
        "D15": 280,
    }
    candidates = set()
    for family, count in quotas.items():
        g = sorted(family_groups[family])
        rng.shuffle(g)
        candidates.update(g[: count + 80])
    wanted = {
        r["record"]
        for r in rows
        if r["default"] and r["split"] == "train" and r["group"] in candidates
    }
    valid_units = {
        (r["record"], r["uid"])
        for r in rows
        if r["default"] and r["split"] == "train" and r["group"] in candidates
    }
    del rows
    grouped = collections.defaultdict(list)
    rejected = collections.Counter()
    with (source / "records.jsonl").open() as f:
        for line in f:
            # IDs are the first field; avoid parsing the whole 5 GB archive.
            head = line.split('"', 4)
            if len(head) < 4 or head[3] not in wanted:
                continue
            r = json.loads(line)
            for u in r["units"]:
                if (r["id"], u["uid"]) not in valid_units:
                    continue
                if (
                    not valid_span(u.get("token", {}).get("span"))
                    or (u.get("token", {}).get("roundtrip") or 0) < 0.5
                ):
                    rejected["token_quality"] += 1
                    continue
                if not u.get("instruction_ref") or "<|mt_" in u["instruction_ref"]:
                    rejected["instruction"] += 1
                    continue
                grouped[r["group"]].append((r, u))
    print(
        json.dumps({"candidate_groups": len(grouped), "records": len(wanted)}),
        flush=True,
    )
    chosen = []
    for family, count in quotas.items():
        gs = [g for g in sorted(grouped) if grouped[g][0][0]["family"] == family]
        rng.shuffle(gs)
        accepted = 0
        for g in gs:
            if accepted >= count:
                break
            pairs = grouped[g]
            r0 = pairs[0][0]
            src = source / r0["source_image"]
            if src.stem in excluded_ids:
                rejected["benchmark_id"] += 1
                continue
            image_hash = hashlib.sha256(src.read_bytes()).hexdigest()
            if image_hash in excluded_hashes:
                rejected["benchmark_hash"] += 1
                continue
            units = []
            seen = set()
            # Preserve original records' target groups, then add distinct same-image selections.
            pairs.sort(key=lambda ru: (-len(ru[0]["units"]), ru[0]["id"], ru[1]["uid"]))
            for r, u in pairs:
                span = u["token"]["span"]
                if span in seen:
                    continue
                try:
                    geo = geometry(u, r["size"], rng)
                except ValueError as e:
                    rejected[str(e)] += 1
                    continue
                noref = (
                    u.get("instruction_noref")
                    if not u.get("noref_ineligible")
                    else None
                )
                units.append(
                    geo
                    | {
                        "ref": u["instruction_ref"],
                        "noref": noref,
                        "family": r["family"],
                        "record": r["id"],
                        "original_uid": u["uid"],
                        "operation": u["op"],
                    }
                )
                seen.add(span)
                if len(units) >= 5:
                    break
            if not units:
                rejected["no_usable_units"] += 1
                continue
            relative = "images/" + image_hash + src.suffix.lower()
            dest = root / relative
            if not dest.exists():
                shutil.copyfile(src, dest)
            chosen.append(
                {
                    "group": g,
                    "family": family,
                    "image": relative,
                    "original_image": str(src),
                    "image_sha256": image_hash,
                    "size": r0["size"],
                    "units": units,
                }
            )
            accepted += 1
        print(f"{family}: {accepted}/{count} groups", flush=True)
    if len(chosen) < 2240:
        raise RuntimeError(
            f"Not enough groups: {len(chosen)}; adjust sampling explicitly"
        )
    # Stratified family splits with disjoint source hashes; split before request expansion.
    splits = {"train": [], "dev": [], "test": []}
    for family in quotas:
        gs = [g for g in chosen if g["family"] == family]
        rng.shuffle(gs)
        nd = round(len(gs) * 64 / 2240)
        nt = round(len(gs) * 128 / 2240)
        splits["dev"] += gs[:nd]
        splits["test"] += gs[nd : nd + nt]
        splits["train"] += gs[nd + nt :]
    for name, n in [("dev", 64), ("test", 128)]:
        while len(splits[name]) < n:
            splits[name].append(splits["train"].pop())
        while len(splits[name]) > n:
            splits["train"].append(splits[name].pop())
    sha_sets = [{g["image_sha256"] for g in splits[s]} for s in splits]
    if any(sha_sets[i] & sha_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Source hash crosses splits")
    summary = {}
    all_rows = {}
    for name, gs in splits.items():
        rng.shuffle(gs)
        n = args.train_requests if name == "train" else len(gs)
        result = [make_request(gs[i % len(gs)], name, i, rng) for i in range(n)]
        if name == "train":
            rng.shuffle(result)
        out = root / (name + ".jsonl")
        with out.open("w") as f:
            for r in result:
                f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
        summary[name] = {
            "requests": len(result),
            "source_groups": len(gs),
            "K": dict(
                collections.Counter(len(x["request"]["targets"]) for x in result)
            ),
            "modes": dict(
                collections.Counter(
                    ("ref" if t["has_ref"] else "noref") + "_" + t["locator"]
                    for x in result
                    for t in x["request"]["targets"]
                )
            ),
            "families": dict(collections.Counter(g["family"] for g in gs)),
            "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        }
        all_rows[name] = [
            {
                "group": g["group"],
                "image": g["image"],
                "image_sha256": g["image_sha256"],
                "original_image": g["original_image"],
                "family": g["family"],
            }
            for g in gs
        ]
    dump_json(root / "split_sources.json", all_rows)
    dump_json(
        root / "manifest.json",
        {
            "seed": args.seed,
            "source": str(source),
            "benchmark": args.benchmark,
            "benchmark_sha256": hashlib.sha256(
                Path(args.benchmark).read_bytes()
            ).hexdigest(),
            "benchmark_hash_overlap": 0,
            "split_hash_overlap": 0,
            "summary": summary,
            "rejected": dict(rejected),
            "selection": "Default train units only; S masks; token roundtrip >= .5; S area [.0005,.30]; distinct same-image tokens up to 5.",
            "point_policy": "Up to 8 normalized, S-and-canonical-dec positive clicks; heldout first deep click. This is a teacher-consistent pilot, not benchmark click distribution.",
            "known_limitations": [
                "No D2 dense-pool expansion in pilot",
                "K=5 and noref-part coverage may be limited",
                "Same-image edit targets may originate from separate records; no joint edited image is used",
            ],
        },
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
