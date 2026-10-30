"""Round-2 data (docs/01): the development split, the SA-1B mask pool and token caches, the training rows of C1 to C3
and of the merged dataset, and the development evaluation set.

Sources (protocol 2.0, read only):
* the SA-1B 50k release: ``annotations/train_protocol_v2.jsonl`` (edits, support and changed regions);
* the SA-1B dense annotations of the same 1,132 source images: ``<run root>/sources/<id>/dense.json``.

Steps (outputs go to one data directory; nothing is written next to the sources):

    python -m torch.distributed.run --nproc_per_node 8 -m samtok_edit21 round2-data pool --out <dir>
    python -m samtok_edit21 round2-data rows --out <dir>
    python -m samtok_edit21 round2-data eval --out <dir>
    python -m samtok_edit21 round2-data rows-c2 --out <dir>     # stage B (C2)
    python -m samtok_edit21 round2-data rows-c3m --out <dir>    # stage C without new data (C3-M)
    python -m samtok_edit21 round2-data rows-c3d --out <dir>    # stage C with the extedit batch-1 data (D10, C3-D)
    python -m samtok_edit21 round2-data rows-merged --out <dir> --spec <spec.json>   # merged data, OPSD rows too

``pool`` encodes every usable dense mask (and every 50k source mask) with SAMTok and decodes the token again; the
decoded region is what the token means and is the supervision target, exactly as in round 1.  ``rows`` writes the
C1 training rows in their training order; ``eval`` the C1 evaluation items, all on the held-out source images.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import re

import numpy as np

SA1B = Path("/mnt/bn/strategy-mllm-train/user/tanyue/datasets/sa1b_samtok_edit_50k_20261014")
RELEASE = SA1B / "release"
# the production run of the release keeps the per-source dense annotations (sources/<id>/dense.json)
SA1B_RUN = Path("/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/data_pipeline/02_production_runs/"
                "sa1b_samtok_edit_50k_20261014")
V1_ROOT = Path("/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTokEdit_Cleaned_20261009")
V1_MANIFEST = V1_ROOT / "protocol_adapt_qwen38_27b_v1/protocol_v2_blind/manifest_protocol_v2.jsonl"

AREA_MIN, AREA_MAX = 0.0005, 0.30       # smaller: below what SAMTok and a 16 px token grid resolve; larger: background
MIN_ROUNDTRIP = 0.5                     # a token whose decoded region misses its mask this much does not name it
DEV_IMAGES_50K, DEV_RECORDS_V1 = 60, 40
# Area buckets (fraction of the image) and the share of single-token highlight regions drawn from each: close to
# benchmark v2 (45 % under 0.5 %), with the large regions kept that the earlier data was made of.
BUCKETS = (0.0005, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 1.01)
BUCKET_SHARE = (0.05, 0.20, 0.20, 0.20, 0.15, 0.12, 0.08)
SAME_OBJECT_IOU = 0.8                   # two dense masks this similar are one object labelled twice
PART_INSIDE, HOST_FACTOR = 0.9, 2.0     # A is a part of B: 90 % of A lies in B and B is at least twice A
DISJOINT = 0.02                         # multi-token regions overlap at most 2 % of the smaller one


def rle_area_frac(rle):
    from pycocotools import mask as mu
    h, w = rle["size"]
    return float(mu.area(rle)) / (h * w)


def as_rle(rle):
    counts = rle["counts"]
    return {"size": list(rle["size"]), "counts": counts.encode() if isinstance(counts, str) else counts}


def encode_rle(mask):
    from pycocotools import mask as mu
    rle = mu.encode(np.asfortranarray(np.asarray(mask, dtype=np.uint8)))
    return {"size": [int(v) for v in rle["size"]], "counts": rle["counts"].decode()}


def decode_rle(rle):
    from pycocotools import mask as mu
    return mu.decode(as_rle(rle)).astype(bool)


def category(name):
    """The most specific word of a dense label such as "person,woman"."""
    words = [w.strip() for w in str(name or "").split(",") if w.strip()]
    return re.sub(r"\s+", " ", words[-1]).lower() if words else ""


def bucket(area):
    for index in range(len(BUCKETS) - 1):
        if BUCKETS[index] <= area < BUCKETS[index + 1]:
            return index
    return None


def stable_order(keys, salt):
    return sorted(keys, key=lambda key: hashlib.sha256(f"{salt}:{key}".encode()).hexdigest())


def dev_split(release_rows, v1_rows):
    """60 SA-1B source images and 40 v1 records held out of all training (docs/01)."""
    sources = stable_order({r["source_id"] for r in release_rows}, "round2-dev-50k")[:DEV_IMAGES_50K]
    v1 = stable_order({r["id"] for r in v1_rows}, "round2-dev-v1")[:DEV_RECORDS_V1]
    return {"sa1b_sources": sorted(sources), "v1_ids": sorted(v1)}


def relations(masks):
    """Same-category, part/host and overlap relations between the dense masks of one image (by RLE)."""
    from pycocotools import mask as mu
    rles = [as_rle(m["rle"]) for m in masks]
    areas = np.array([float(mu.area(r)) for r in rles])
    inter = np.zeros((len(rles), len(rles)))
    boxes = mu.toBbox(rles) if rles else np.zeros((0, 4))
    for i in range(len(rles)):
        for j in range(i + 1, len(rles)):
            a, b = boxes[i], boxes[j]
            if a[0] > b[0] + b[2] or b[0] > a[0] + a[2] or a[1] > b[1] + b[3] or b[1] > a[1] + a[3]:
                continue
            inter[i, j] = inter[j, i] = float(mu.area(mu.merge([rles[i], rles[j]], intersect=True)))
    out = []
    for i, m in enumerate(masks):
        same, hosts, parts, touching, duplicate = [], [], [], [], []
        for j, other in enumerate(masks):
            if i == j:
                continue
            union = areas[i] + areas[j] - inter[i, j]
            if union and inter[i, j] / union >= SAME_OBJECT_IOU:
                duplicate.append(j)
                continue
            if inter[i, j] >= PART_INSIDE * areas[i] and areas[j] >= HOST_FACTOR * areas[i]:
                hosts.append(j)
            elif inter[i, j] >= PART_INSIDE * areas[j] and areas[i] >= HOST_FACTOR * areas[j]:
                parts.append(j)
            elif inter[i, j] > 0:
                touching.append(j)
            if m["category"] and other["category"] == m["category"] and inter[i, j] <= DISJOINT * min(areas[i], areas[j]):
                same.append(j)
        out.append({"same": same, "hosts": hosts, "parts": parts, "touching": touching, "duplicate": duplicate})
    return out


# ---------------------------------------------------------------- pool (GPU, torchrun)

def encode_masks(codec, image, masks, batch=16):
    """SAMTok spans of ``masks`` in input order (``codec.encode`` sorts them; the order is mapped back)."""
    spans = [None] * len(masks)
    for start in range(0, len(masks), batch):
        chunk = masks[start:start + batch]
        encoded, order = codec.encode(image, chunk)
        for position, original in enumerate(order):
            spans[start + original] = encoded[position]
    return spans


def decode_spans(codec, image, spans, batch=16):
    out = []
    for start in range(0, len(spans), batch):
        out += [m > 0 for m in codec.decode_strict(image, "".join(spans[start:start + batch]))]
    return out


def pool_image(codec, source_id, image_path, dense_path):
    from PIL import Image
    from pycocotools import mask as mu
    with Image.open(image_path) as im:
        image = im.convert("RGB")
    dense = json.load(open(dense_path))
    objects = dense["objects_json"]
    objects = json.loads(objects) if isinstance(objects, str) else objects
    masks, arrays = [], []
    for ob in objects:
        rle = ob["segmentation"]
        if tuple(rle["size"]) != (image.height, image.width):
            continue
        area = float(mu.area(as_rle(rle))) / (image.width * image.height)
        if not AREA_MIN <= area <= AREA_MAX:
            continue
        masks.append({"oid": str(ob["object_id"]), "category": category(ob.get("category")), "area": round(area, 6),
                      "rle": {"size": list(rle["size"]), "counts": rle["counts"] if isinstance(rle["counts"], str)
                              else rle["counts"].decode()}})
        arrays.append(mu.decode(as_rle(rle)).astype(bool))
    if not masks:
        return {"source_id": source_id, "image": str(image_path), "size": [image.width, image.height], "masks": []}
    spans = encode_masks(codec, image, arrays)
    decoded = decode_spans(codec, image, spans)
    for m, array, span, dec in zip(masks, arrays, spans, decoded):
        union = float((array | dec).sum())
        m.update(span=span, dec=encode_rle(dec), dec_area=round(float(dec.mean()), 6),
                 roundtrip=round(float((array & dec).sum()) / union, 4) if union else 0.0)
    for m, rel in zip(masks, relations(masks)):
        m["rel"] = rel
    return {"source_id": source_id, "image": str(image_path), "size": [image.width, image.height], "masks": masks}


def pool_edit_tokens(codec, rows):
    """Token of every 50k source mask (protocol 2.0 mask) and its round trip."""
    from PIL import Image
    out = []
    for row in rows:
        with Image.open(RELEASE / row["source_image"]) as im:
            image = im.convert("RGB")
        with Image.open(RELEASE / row["mask"]) as im:
            mask = np.asarray(im.convert("L")) > 0
        span = encode_masks(codec, image, [mask])[0]
        dec = decode_spans(codec, image, [span])[0]
        union = float((mask | dec).sum())
        out.append({"sample_id": row["sample_id"], "span": span, "dec": encode_rle(dec),
                    "roundtrip": round(float((mask & dec).sum()) / union, 4) if union else 0.0})
    return out


def run_pool(args):
    import torch
    import torch.distributed as dist
    from samtok_edit21.models.codec import load_codec
    from samtok_edit21.models.pipeline import DEFAULT_SAMTOK
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    out = Path(args.out)
    (out / "pool").mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in open(RELEASE / "annotations/train_protocol_v2.jsonl")]
    v1 = [json.loads(line) for line in open(V1_MANIFEST)]
    if rank == 0 and not (out / "dev_split.json").exists():
        (out / "dev_split.json").write_text(json.dumps(dev_split(rows, v1), indent=1))
    dist.barrier()
    codec = load_codec(DEFAULT_SAMTOK, f"cuda:{int(os.environ['LOCAL_RANK'])}")
    sources = sorted({r["source_id"]: r["source_image"] for r in rows}.items())
    for index, (source_id, image) in enumerate(sources[rank::world]):
        path = out / "pool" / f"{source_id}.json"
        if path.exists():
            continue
        entry = pool_image(codec, source_id, RELEASE / image, SA1B_RUN / "sources" / source_id / "dense.json")
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(entry))
        tmp.rename(path)
        if rank == 0 and index % 20 == 0:
            print(f"pool {index * world}/{len(sources)}", flush=True)
    part = out / f"edit_tokens.part{rank}.jsonl"
    if not part.exists():
        tokens = pool_edit_tokens(codec, rows[rank::world])
        tmp = part.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(t) + "\n" for t in tokens))
        tmp.rename(part)
    dist.barrier()
    if rank == 0:
        merged = {}
        for r in range(world):
            for line in open(out / f"edit_tokens.part{r}.jsonl"):
                t = json.loads(line)
                merged[t["sample_id"]] = t
        with open(out / "edit_tokens.jsonl", "w") as f:
            for row in rows:
                f.write(json.dumps(merged[row["sample_id"]]) + "\n")
        for r in range(world):
            (out / f"edit_tokens.part{r}.jsonl").unlink()
        print("pool done", flush=True)
    dist.destroy_process_group()


# ---------------------------------------------------------------- scaled pool (GPU, one worker per process)
#
# The two-stage retrain (docs/03) reads tokens on many more images than the 1,132 of the 50k: every D12 image (twins,
# two confusable targets per image) and D15 images in a fixed order, 20,000 in all, with the same per-image entry as
# ``pool``.  The data directory gets the round-2 files the trainer reads (hard links) and the old pool entries;
# workers on any number of machines share the shards through claims on NAS:
#
#     python -m samtok_edit21 round2-data pool-ext-prepare --out <dir> --base <round2 dir>
#     CUDA_VISIBLE_DEVICES=g python -m samtok_edit21 round2-data pool-ext-work --out <dir> --name <worker>
#     python -m samtok_edit21 round2-data pool-ext-check --out <dir>

POOL_EXT_IMAGES, POOL_EXT_SHARD = 20000, 40
POOL_EXT_FAMILIES = ("D12", "D15")              # in this order: all of D12, then D15 up to the count
BASE_FILES = ("dev_split.json", "edit_tokens.jsonl", "support_cache.jsonl", "eval_c1.json", "eval_c1_quick_idx.json",
              "eval_c2_quick_idx.json", "v1_tokens.jsonl", "d4_tokens.jsonl")


def pool_ext_images(merged=None, count=POOL_EXT_IMAGES):
    """[(source_id, image, dense)] of the scaled pool: default-view images of POOL_EXT_FAMILIES (every image of the
    first family, the next in a fixed order), none that shares a source with the benchmark or the dev split."""
    from samtok_edit21.preparation import sa1b_groups as sg
    merged = Path(merged or MERGED_ROOT)
    images, blocked = defaultdict(dict), set()
    for line in open(merged / "records.jsonl"):
        if line[8:line.index('"', 8)].split(":", 1)[0] not in POOL_EXT_FAMILIES:   # {"id": "<family>:...
            continue
        r = json.loads(line)
        sid = r["sa1b"]["source_id"]
        if "benchmark_overlap" in r["exclude"] or "dev_overlap" in r["exclude"]:
            blocked.add(sid)
        elif r["default"]:
            images[r["family"]][sid] = str(merged / r["source_image"])
    dense = {}
    for production, _, kinds in sg.SETS.values():
        for job in sg.jobs(production, kinds):
            dense.setdefault(job["source_id"], job["dense"])
    chosen, out = set(), []
    for family in POOL_EXT_FAMILIES:
        for sid in stable_order(sorted(set(images[family]) - blocked - chosen), f"pool-ext-{family}"):
            if len(out) == count:
                return out
            chosen.add(sid)
            out.append((sid, images[family][sid], dense[sid]))
    return out


def pool_ext_prepare(out, base, merged=None, count=POOL_EXT_IMAGES, shard=POOL_EXT_SHARD):
    out, base, merged = Path(out), Path(base), Path(merged or MERGED_ROOT)
    (out / "pool").mkdir(parents=True, exist_ok=True)
    for name in BASE_FILES:                       # hard links: the trainer and the evaluation read them here
        if not (out / name).exists():
            os.link(base / name, out / name)
    dev = set(json.load(open(base / "dev_split.json"))["sa1b_sources"])
    for path in sorted((base / "pool").glob("*.json")):
        if not (out / "pool" / path.name).exists():
            os.link(path, out / "pool" / path.name)
    images = [i for i in pool_ext_images(merged, count) if i[0] not in dev]
    run = out / "pool_ext"
    run.mkdir(exist_ok=True)
    info = {"base": str(base), "merged": str(merged), "families": POOL_EXT_FAMILIES, "count": len(images),
            "by_family_rule": "all D12 default-view images, then D15 in stable order", "shard_images": shard,
            "images": images, "shards": [[s, min(s + shard, len(images))] for s in range(0, len(images), shard)]}
    (run / "jobs.json").write_text(json.dumps(info))
    return {k: v for k, v in info.items() if k not in ("images", "shards")} | {"shards": len(info["shards"]),
                                                                                "base_pool": len(list((base / "pool").glob("*.json")))}


def pool_ext_work(out, name, limit_shards=0):
    """Pool entries of unclaimed shards (``pool/<source_id>.json``) until none is left."""
    import time
    import torch
    from samtok_edit21.models.codec import load_codec
    from samtok_edit21.models.pipeline import DEFAULT_SAMTOK
    from samtok_edit21.preparation.sa1b_groups import Claims
    out = Path(out)
    run = out / "pool_ext"
    info = json.load(open(run / "jobs.json"))
    claims = Claims(run, "pool", name)
    torch.set_num_threads(4)
    codec = load_codec(DEFAULT_SAMTOK, "cuda")
    (run / "logs").mkdir(exist_ok=True)
    (run / "done").mkdir(exist_ok=True)
    log, done = open(run / "logs" / f"work-{name}.log", "a"), 0
    for index, (start, end) in enumerate(info["shards"]):
        marker = run / "done" / f"{index:04d}.json"
        if marker.exists() or not claims.take(index):
            continue
        stop = claims.beating(index)
        started, errors, masks = time.time(), [], 0
        try:
            for sid, image, dense in info["images"][start:end]:
                path = out / "pool" / f"{sid}.json"
                if path.exists():
                    continue
                try:
                    entry = pool_image(codec, sid, image, dense)
                except Exception as exc:          # one broken image must not stop the shard; the check lists it
                    errors.append({"source_id": sid, "error": f"{type(exc).__name__}: {exc}"[:300]})
                    continue
                masks += len(entry["masks"])
                tmp = path.with_suffix(f".{name}.tmp")
                tmp.write_text(json.dumps(entry))
                os.replace(tmp, path)
            tmp = marker.with_suffix(f".{name}.tmp")
            tmp.write_text(json.dumps({"shard": index, "worker": name, "errors": errors, "masks": masks,
                                       "sec": round(time.time() - started, 1)}))
            os.replace(tmp, marker)
        finally:
            stop.set()
        done += 1
        log.write(json.dumps({"shard": index, "images": end - start, "errors": len(errors), "masks": masks,
                              "sec": round(time.time() - started, 1), "time": time.strftime("%H:%M:%S")}) + "\n")
        log.flush()
        if limit_shards and done >= limit_shards:
            break
    return done


def pool_ext_check(out):
    """Progress, or once every shard is done the pool's statistics (``pool_ext/summary.json``)."""
    out = Path(out)
    run = out / "pool_ext"
    info = json.load(open(run / "jobs.json"))
    markers = [json.load(open(p)) for p in sorted((run / "done").glob("*.json"))] if (run / "done").exists() else []
    status = {"shards": len(info["shards"]), "done": len(markers), "errors": sum(len(m["errors"]) for m in markers)}
    if len(markers) < len(info["shards"]):
        return status
    dev = set(json.load(open(out / "dev_split.json"))["sa1b_sources"])
    new = {sid for sid, _, _ in info["images"]}
    counts, buckets, roundtrips = Counter(), Counter(), []
    for path in sorted((out / "pool").glob("*.json")):
        entry = json.load(open(path))
        part = "new" if path.stem in new else ("base_dev" if path.stem in dev else "base_train")
        counts[f"{part}_images"] += 1
        counts[f"{part}_masks"] += len(entry["masks"])
        for m in entry["masks"]:
            if usable(m):
                counts[f"{part}_usable"] += 1
                if part == "new":
                    buckets[bucket(m["area"])] += 1
            if part == "new" and "roundtrip" in m:
                roundtrips.append(m["roundtrip"])
    summary = status | dict(counts) | {
        "missing_new": len(new - {p.stem for p in (out / "pool").glob("*.json")}),
        "new_usable_by_bucket": {f"{BUCKETS[b]}-{BUCKETS[b + 1]}": buckets[b] for b in sorted(buckets)},
        "new_roundtrip_median": round(float(np.median(roundtrips)), 4) if roundtrips else None,
        "new_roundtrip_low_share": round(float(np.mean(np.asarray(roundtrips) < MIN_ROUNDTRIP)), 4) if roundtrips else None,
        "errors_list": [e for m in markers for e in m["errors"]]}
    (run / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


# ---------------------------------------------------------------- C1 training rows (CPU)

HL_SINGLE = (
    "Apply {a} {color} semi-transparent mask to {x} and keep everything else unchanged.",
    "Highlight {x} with {a} {color} semi-transparent mask; leave the rest of the image unchanged.",
    "Cover {x} with {a} {color} semi-transparent overlay and keep everything else unchanged.",
    "Mark the pixels of {x} with {a} {color} semi-transparent mask and leave the rest unchanged.",
)
HL_MULTI = (
    "Apply semi-transparent masks: {items}. Keep everything else unchanged.",
    "Highlight each region with its own semi-transparent color: {items}; leave the rest of the image unchanged.",
    "Cover {items} with semi-transparent overlays and keep everything else unchanged.",
)
ITEM_FORMS = ("{color} to {x}", "{color} for {x}", "{x} with {color}")   # one per HL_MULTI template
MULTI_COLORS = ("red", "green", "blue", "magenta", "yellow", "cyan")
C1_MIX = {"hl": 0.40, "hl_ref": 0.15, "hlm": 0.30, "edit": 0.15}
EDIT_FORMS = {"noref_token": 0.65, "ref_token": 0.20, "ref_only": 0.15}


def article(color):
    return "an" if color[0] in "aeiou" else "a"


def join_items(parts):
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def highlight_prompt_r2(span, color, phrase=None, template=0):
    x = f"the {phrase} {span}" if phrase else f"the selected region {span}"
    return HL_SINGLE[template].format(a=article(color), color=color, x=x)


def multi_prompt(spans, colors, template=0):
    form = ITEM_FORMS[template]
    items = join_items([form.format(color=c, x=f"the selected region {s}") for s, c in zip(spans, colors)])
    return HL_MULTI[template].format(items=items)


def mean_color(image_rgb, mask):
    return np.asarray(image_rgb, np.float32)[mask].mean(0) if mask.any() else np.zeros(3, np.float32)


def usable(m):
    return "span" in m and m["roundtrip"] >= MIN_ROUNDTRIP and m["dec_area"] > 0


def negatives_of(entry, index):
    """Masks that must stay unchanged when ``index`` is highlighted: same-category instances and its hosts."""
    rel = entry["masks"][index]["rel"]
    return sorted(set(rel["same"][:3] + rel["hosts"][:1]))


def pick_regions(entries, rng, count, *, replace=True, shares=BUCKET_SHARE):
    """``count`` (image, mask) draws following ``shares`` of BUCKETS; every other draw, where possible, has a negative.

    Without ``replace`` (evaluation) a region is drawn at most once and a short bucket gives what it has.
    """
    by_bucket, with_negative = defaultdict(list), defaultdict(list)
    for e_index, entry in enumerate(entries):
        for m_index, m in enumerate(entry["masks"]):
            b = bucket(m["area"]) if usable(m) else None
            if b is None:
                continue
            by_bucket[b].append((e_index, m_index))
            if negatives_of(entry, m_index):
                with_negative[b].append((e_index, m_index))
    draws, taken = [], set()
    for b, share in enumerate(shares):
        for k in range(round(count * share)):
            pool = with_negative[b] if (k % 2 == 0 and with_negative[b]) else by_bucket[b]
            if not replace:
                pool = [d for d in pool if d not in taken] or [d for d in by_bucket[b] if d not in taken]
                if not pool:
                    break
            draw = rng.choice(pool)
            taken.add(draw)
            draws.append(draw)
    rng.shuffle(draws)
    return draws[:count]


def multi_group(entry, rng, size, start=()):
    """``size`` usable, pairwise disjoint, non-duplicate masks of one image (decoded regions disjoint too),
    beginning with ``start`` when given."""
    candidates = [i for i, m in enumerate(entry["masks"]) if usable(m) and bucket(m["area"]) is not None]
    rng.shuffle(candidates)
    chosen = list(start)
    decoded = [decode_rle(entry["masks"][i]["dec"]) for i in chosen]
    if len(chosen) >= size:
        return chosen[:size]
    for i in candidates:
        if i in chosen:
            continue
        rel = entry["masks"][i]["rel"]
        if any(j in chosen for j in rel["hosts"] + rel["parts"] + rel["duplicate"] + rel["touching"]):
            continue
        dec = decode_rle(entry["masks"][i]["dec"])
        if any((dec & d).sum() > DISJOINT * min(dec.sum(), d.sum()) for d in decoded):
            continue
        chosen.append(i)
        decoded.append(dec)
        if len(chosen) == size:
            return chosen
    return None


def build_rows(out, seed=20261016, updates=2000, batch=32):
    """C1 rows in training order: highlight (single, single with the category name, multi) and edits."""
    rng = random.Random(seed)
    split = json.load(open(out / "dev_split.json"))
    dev = set(split["sa1b_sources"])
    entries = [json.load(open(p)) for p in sorted((out / "pool").glob("*.json"))]
    train_entries = [e for e in entries if e["source_id"] not in dev]
    total = updates * batch
    counts = {kind: round(total * share) for kind, share in C1_MIX.items()}
    counts["edit"] = total - counts["hl"] - counts["hl_ref"] - counts["hlm"]
    rows = []
    singles = pick_regions(train_entries, rng, counts["hl"] + counts["hl_ref"])
    named = [k for k, (e, m) in enumerate(singles) if train_entries[e]["masks"][m]["category"]]
    with_name = set(rng.sample(named, min(counts["hl_ref"], len(named))))
    for k, (e_index, m_index) in enumerate(singles):
        entry = train_entries[e_index]
        phrase = entry["masks"][m_index]["category"] if k in with_name else None
        rows.append({"kind": "hl_ref" if phrase else "hl", "source_id": entry["source_id"], "masks": [m_index],
                     "negatives": [negatives_of(entry, m_index)], "colors": [rng.choice(MULTI_COLORS)],
                     "template": rng.randrange(len(HL_SINGLE)), "phrase": phrase})
    # multi-token rows: 2-5 regions, sizes weighted towards 2-3
    sizes = rng.choices([2, 3, 4, 5], weights=[0.35, 0.30, 0.20, 0.15], k=counts["hlm"])
    for size in sizes:
        for _ in range(20):
            entry = rng.choice(train_entries)
            group = multi_group(entry, rng, size)
            if group:
                break
        else:
            raise RuntimeError("no disjoint multi-token group found")
        colors = rng.sample(MULTI_COLORS, size)
        rows.append({"kind": "hlm", "source_id": entry["source_id"], "masks": group,
                     "negatives": [negatives_of(entry, i) for i in group], "colors": colors,
                     "template": rng.randrange(len(HL_MULTI))})
    # edit rows from the 50k (training sources only); token rows need a no-ref sentence and a faithful token
    release = [json.loads(line) for line in open(RELEASE / "annotations/train_protocol_v2.jsonl")]
    tokens = {t["sample_id"]: t for t in map(json.loads, open(out / "edit_tokens.jsonl"))}
    train_edits = [r for r in release if r["source_id"] not in dev]
    token_ok = [r for r in train_edits if r["instruction_noref"] and tokens[r["sample_id"]]["roundtrip"] >= MIN_ROUNDTRIP]
    forms = rng.choices(list(EDIT_FORMS), weights=list(EDIT_FORMS.values()), k=counts["edit"])
    for form in forms:
        r = rng.choice(train_edits if form == "ref_only" else token_ok)
        rows.append({"kind": "edit", "form": form, "sample_id": r["sample_id"], "source_id": r["source_id"]})
    rng.shuffle(rows)
    with open(out / "rows_c1.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    stats = {"rows": len(rows), "kinds": dict(Counter(r["kind"] for r in rows)),
             "edit_forms": dict(Counter(r["form"] for r in rows if r["kind"] == "edit")),
             "multi_sizes": dict(Counter(len(r["masks"]) for r in rows if r["kind"] == "hlm")),
             "train_images": len(train_entries), "dev_images": len(dev),
             "usable_masks_train": sum(usable(m) for e in train_entries for m in e["masks"])}
    (out / "rows_c1_stats.json").write_text(json.dumps(stats, indent=1))
    return stats


# ---------------------------------------------------------------- C1b rows: hard negatives next to the target (CPU)
#
# C1 took its negatives from the same-category list in annotation order: the nearest same-category instance was a
# negative in only half of the rows that had one, a different-category neighbour never was, and the read side had
# no heavy negatives.  C1's near misses landed on a different-category neighbour (37 %), the host of a part
# (29 %) or a same-category neighbour (24 %).  C1b therefore takes as negatives the nearest annotated regions of
# any category plus the hosts, adds counterfactual pairs of adjacent regions (and of a part and its host), and builds
# part of the multi-token rows around an adjacent pair (docs/02 section 3.7).

NEAR_GAP = 0.05            # a neighbour: box gap at most 5 % of the image diagonal
MAX_NEAR, MAX_HOSTS = 6, 2
PAIR_SHARE = 0.30          # share of the single-token highlight rows that come as counterfactual pairs
PAIR_PART_HOST = 0.30      # of the pairs: a part and its host; the rest are adjacent regions of similar size
PAIR_AREA_RATIO = 4.0
MULTI_ADJACENT = 0.40      # share of the multi-token rows built around an adjacent pair


def geometry(entry):
    """Boxes (x, y, w, h) of the decoded regions and of the annotations, and the image diagonal (cached)."""
    if "_geo" not in entry:
        from pycocotools import mask as mu
        masks = entry["masks"]
        dec = np.asarray(mu.toBbox([as_rle(m["dec"]) for m in masks]), float).reshape(-1, 4) if masks else np.zeros((0, 4))
        ann = np.asarray(mu.toBbox([as_rle(m["rle"]) for m in masks]), float).reshape(-1, 4) if masks else np.zeros((0, 4))
        entry["_geo"] = (dec, ann, float(np.hypot(*entry["size"])))
    return entry["_geo"]


def box_gap(a, b):
    """Euclidean gap between two (x, y, w, h) boxes; 0 when they touch or overlap."""
    dx = max(0.0, max(a[0], b[0]) - min(a[0] + a[2], b[0] + b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[1] + a[3], b[1] + b[3]))
    return float(np.hypot(dx, dy))


def centre_distance(a, b):
    return float(np.hypot(a[0] + a[2] / 2 - b[0] - b[2] / 2, a[1] + a[3] / 2 - b[1] - b[3] / 2))


def overlap_frac(entry, i, j):
    """Intersection of the decoded regions of i and j over the smaller one."""
    from pycocotools import mask as mu
    a, b = as_rle(entry["masks"][i]["dec"]), as_rle(entry["masks"][j]["dec"])
    return float(max(mu.iou([a], [b], [1])[0, 0], mu.iou([b], [a], [1])[0, 0]))


def near_negatives(entry, index, *, target_box=None, exclude=()):
    """Annotated regions that must stay unchanged when region ``index`` is the target: the nearest objects of
    any category (box gap at most NEAR_GAP of the diagonal, closest first, up to MAX_NEAR) and its hosts.

    Never negatives: the target, its parts (they lie inside it) and duplicate annotations of it.  For an edit
    whose region is not a pool region, ``index`` is None and ``target_box`` gives the region's box.
    """
    dec, ann, diag = geometry(entry)
    if index is None:
        rel, box = {"hosts": [], "parts": [], "duplicate": []}, target_box
    else:
        rel, box = entry["masks"][index]["rel"], dec[index] if target_box is None else target_box
    skip = set(rel["parts"]) | set(rel["duplicate"]) | set(exclude) | ({index} if index is not None else set())
    ranked = sorted((box_gap(box, ann[j]), centre_distance(box, ann[j]), j)
                    for j in range(len(entry["masks"])) if j not in skip)
    near = [j for gap, _, j in ranked if gap <= NEAR_GAP * diag][:MAX_NEAR]
    hosts = [j for j in rel["hosts"] if j not in skip][:MAX_HOSTS]
    return sorted(set(near) | set(hosts))


def edit_negatives(entry, token_dec):
    """Near negatives of an SA-1B edit: pool regions around its token's decoded region, without the edited
    object itself (IoU at least 0.5) and its parts (at least 90 % inside the edited region)."""
    from pycocotools import mask as mu
    if not entry["masks"]:
        return []
    target = as_rle(token_dec)
    anns = [as_rle(m["rle"]) for m in entry["masks"]]
    same = mu.iou(anns, [target], [0])[:, 0]
    inside = mu.iou(anns, [target], [1])[:, 0]        # crowd: intersection over the area of the annotation
    exclude = [j for j in range(len(anns)) if same[j] >= 0.5 or inside[j] >= 0.9]
    return near_negatives(entry, None, target_box=np.asarray(mu.toBbox([target]), float)[0], exclude=exclude)


def pair_partner(entry, index, rng, part_host):
    """The second region of a counterfactual pair: the host of a part or a part of a host (``part_host`` True or
    "part_host"), another instance of the same category and similar size ("same"), else an adjacent usable region of
    similar size (any category).  Decoded regions disjoint except for part and host.  None if there is none."""
    dec, _, diag = geometry(entry)
    m = entry["masks"][index]
    similar = lambda o: max(m["area"], o["area"]) <= PAIR_AREA_RATIO * min(m["area"], o["area"])  # noqa: E731
    if part_host is True or part_host == "part_host":
        options = [j for j in m["rel"]["hosts"] + m["rel"]["parts"] if usable(entry["masks"][j])]
    elif part_host == "same":
        options = [j for j in m["rel"]["same"] if usable(entry["masks"][j]) and similar(entry["masks"][j])]
        rng.shuffle(options)
        options = [j for j in options[:8] if overlap_frac(entry, index, j) <= DISJOINT]
    else:
        skip = set(m["rel"]["hosts"]) | set(m["rel"]["parts"]) | set(m["rel"]["duplicate"]) | {index}
        options = [j for j, o in enumerate(entry["masks"])
                   if j not in skip and usable(o) and similar(o) and box_gap(dec[index], dec[j]) <= NEAR_GAP * diag]
        rng.shuffle(options)
        options = [j for j in options[:8] if overlap_frac(entry, index, j) <= DISJOINT]
    return rng.choice(options) if options else None


def multi_group_adjacent(entry, rng, size):
    """A multi-token group built around an adjacent pair of usable, disjoint regions (None if there is none)."""
    dec, _, diag = geometry(entry)
    candidates = [i for i, m in enumerate(entry["masks"]) if usable(m) and bucket(m["area"]) is not None]
    rng.shuffle(candidates)
    for i in candidates[:20]:
        rel = entry["masks"][i]["rel"]
        blocked = set(rel["hosts"]) | set(rel["parts"]) | set(rel["duplicate"]) | set(rel["touching"]) | {i}
        near = [j for j in candidates if j not in blocked and box_gap(dec[i], dec[j]) <= NEAR_GAP * diag]
        rng.shuffle(near)
        for j in near[:5]:
            if overlap_frac(entry, i, j) <= DISJOINT:
                group = multi_group(entry, rng, size, start=(i, j))
                if group:
                    return group
    return None


def place_in_slots(units, rng, updates, world, accumulation):
    """Rows in training order: global row (u * accumulation + s) * world + r is processed by rank r at step s of
    update u.  A pair occupies steps s, s + 1 (s even) of one rank, so both members share an update; singles fill
    the remaining slots two by two.  OPSD rows (kind opsd_merged) cost several other rows each, and the ranks wait for
    the slowest one at every gradient average: they fill whole step pairs, two rows on every rank, spread evenly over
    the run; a remainder short of a step pair joins the other singles.  Without OPSD rows the placement (and the use
    of ``rng``) is the one of the earlier rows files."""
    if accumulation % 2:
        raise ValueError("pairs need an even accumulation")
    pairs = [u for u in units if len(u) == 2]
    singles = [u[0] for u in units if len(u) == 1 and u[0].get("kind") != "opsd_merged"]
    heavy = [u[0] for u in units if len(u) == 1 and u[0].get("kind") == "opsd_merged"]
    steps = []                                      # OPSD step pairs: rows 2r, 2r + 1 go to rank r
    if heavy:
        rng.shuffle(heavy)
        full = len(heavy) - len(heavy) % (2 * world)
        steps = [heavy[k:k + 2 * world] for k in range(0, full, 2 * world)]
        singles += heavy[full:]
    rng.shuffle(singles)
    blocks = pairs + [singles[k:k + 2] for k in range(0, len(singles), 2)]
    rng.shuffle(blocks)
    groups = updates * accumulation // 2
    if len(blocks) + len(steps) * world != groups * world or any(len(b) != 2 for b in blocks):
        raise ValueError("units do not fill the slots exactly")
    at = [g for g in range(groups) if (g + 1) * len(steps) // groups > g * len(steps) // groups]
    opsd_at = {g: h for h, g in enumerate(at)}
    rows, k = [None] * (updates * world * accumulation), 0
    for u in range(updates):
        for r in range(world):
            for s in range(0, accumulation, 2):
                g = (u * accumulation + s) // 2
                if g in opsd_at:
                    block = steps[opsd_at[g]][2 * r:2 * r + 2]
                else:
                    block, k = blocks[k], k + 1
                rows[(u * accumulation + s) * world + r], rows[(u * accumulation + s + 1) * world + r] = block
    return rows


def highlight_pairs(train, rng, n_pairs, key_prefix, shares=BUCKET_SHARE, kinds=None):
    """Counterfactual highlight pairs: same prompt, template and colour, an adjacent region or a part and its host
    (PAIR_PART_HOST of the pairs), or with ``kinds`` (weights of "adjacent", "same", "part_host") each pair's kind
    drawn by weight; an anchor without a partner of its kind is skipped."""
    units = []
    for e_index, m_index in pick_regions(train, rng, 4 * n_pairs, shares=shares):
        if len(units) == n_pairs:
            break
        entry = train[e_index]
        if kinds is None:
            part_host = rng.random() < PAIR_PART_HOST
            kind = "part_host" if part_host else "adjacent"
        else:
            kind = part_host = rng.choices(list(kinds), weights=list(kinds.values()))[0]
        partner = pair_partner(entry, m_index, rng, part_host)
        if partner is None:
            continue
        template, color, key = rng.randrange(len(HL_SINGLE)), rng.choice(MULTI_COLORS), f"{key_prefix}-{len(units)}"
        members = []
        for a, b in ((m_index, partner), (partner, m_index)):
            negatives = set(near_negatives(entry, a))
            if b not in entry["masks"][a]["rel"]["parts"]:          # the other member, unless it lies inside a
                negatives.add(b)
            members.append({"kind": "hl", "source_id": entry["source_id"], "masks": [a], "negatives": [sorted(negatives)],
                            "neg_ann": True, "colors": [color], "template": template, "phrase": None,
                            "pair_key": key, "pair_kind": kind})
        units.append(members)
    if len(units) < n_pairs:
        raise RuntimeError(f"only {len(units)} of {n_pairs} counterfactual pairs found")
    return units


def highlight_singles(train, rng, need, n_named, shares=BUCKET_SHARE):
    """``need`` single-token highlights with near negatives; ``n_named`` of them name the object (``hl_ref``)."""
    singles = pick_regions(train, rng, need + 16, shares=shares)[:need]   # per-bucket rounding can fall a few short
    if len(singles) != need:
        raise RuntimeError(f"{len(singles)} of {need} single-token regions drawn")
    named = [k for k, (e, m) in enumerate(singles) if train[e]["masks"][m]["category"]]
    with_name = set(rng.sample(named, min(n_named, len(named))))
    units = []
    for k, (e_index, m_index) in enumerate(singles):
        entry = train[e_index]
        phrase = entry["masks"][m_index]["category"] if k in with_name else None
        units.append([{"kind": "hl_ref" if phrase else "hl", "source_id": entry["source_id"], "masks": [m_index],
                       "negatives": [near_negatives(entry, m_index)], "neg_ann": True, "colors": [rng.choice(MULTI_COLORS)],
                       "template": rng.randrange(len(HL_SINGLE)), "phrase": phrase}])
    return units


def highlight_multis(train, rng, count):
    """``count`` multi-token highlights of 2-5 regions, MULTI_ADJACENT of them built around an adjacent pair."""
    units = []
    for size in rng.choices([2, 3, 4, 5], weights=[0.35, 0.30, 0.20, 0.15], k=count):
        adjacent = rng.random() < MULTI_ADJACENT
        for _ in range(50):
            entry = rng.choice(train)
            group = multi_group_adjacent(entry, rng, size) if adjacent else multi_group(entry, rng, size)
            if group:
                break
        else:
            raise RuntimeError("no multi-token group found")
        units.append([{"kind": "hlm", "source_id": entry["source_id"], "masks": group,
                       "negatives": [near_negatives(entry, i, exclude=group) for i in group], "neg_ann": True,
                       "colors": rng.sample(MULTI_COLORS, size), "template": rng.randrange(len(HL_MULTI)),
                       "adjacent": adjacent}])
    return units


def build_rows_c1b(out, seed=20261018, updates=800, world=8, accumulation=4):
    """C1b rows (same kinds and shares as C1) with near negatives, counterfactual pairs and adjacent multi groups."""
    rng = random.Random(seed)
    dev = set(json.load(open(out / "dev_split.json"))["sa1b_sources"])
    train = [json.load(open(p)) for p in sorted((out / "pool").glob("*.json")) if p.stem not in dev]
    by_id = {e["source_id"]: e for e in train}
    total = updates * world * accumulation
    counts = {kind: round(total * share) for kind, share in C1_MIX.items()}
    counts["edit"] = total - counts["hl"] - counts["hl_ref"] - counts["hlm"]
    n_pairs = round(PAIR_SHARE * (counts["hl"] + counts["hl_ref"]) / 2)
    units = highlight_pairs(train, rng, n_pairs, "c1b-pair")
    units += highlight_singles(train, rng, counts["hl"] + counts["hl_ref"] - 2 * n_pairs, counts["hl_ref"])
    units += highlight_multis(train, rng, counts["hlm"])
    release = [json.loads(line) for line in open(RELEASE / "annotations/train_protocol_v2.jsonl")]
    tokens = {t["sample_id"]: t for t in map(json.loads, open(out / "edit_tokens.jsonl"))}
    train_edits = [r for r in release if r["source_id"] not in dev]
    token_ok = [r for r in train_edits if r["instruction_noref"] and tokens[r["sample_id"]]["roundtrip"] >= MIN_ROUNDTRIP]
    for form in rng.choices(list(EDIT_FORMS), weights=list(EDIT_FORMS.values()), k=counts["edit"]):
        r = rng.choice(train_edits if form == "ref_only" else token_ok)
        token = tokens[r["sample_id"]]
        negatives = edit_negatives(by_id[r["source_id"]], token["dec"]) if token["roundtrip"] >= MIN_ROUNDTRIP else []
        units.append([{"kind": "edit", "form": form, "sample_id": r["sample_id"], "source_id": r["source_id"],
                       "negatives": negatives}])
    rows = place_in_slots(units, rng, updates, world, accumulation)
    with open(out / "rows_c1b.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    hl = [r for r in rows if r["kind"] in ("hl", "hl_ref")]
    stats = {"rows": len(rows), "updates": updates, "world": world, "accumulation": accumulation,
             "kinds": dict(Counter(r["kind"] for r in rows)),
             "pairs": {k: v // 2 for k, v in Counter(r["pair_kind"] for r in rows if r.get("pair_key")).items()},
             "pair_rows": sum(1 for r in rows if r.get("pair_key")),
             "multi_adjacent": round(float(np.mean([r["adjacent"] for r in rows if r["kind"] == "hlm"])), 3),
             "single_rows_with_negatives": round(float(np.mean([bool(r["negatives"][0]) for r in hl])), 3),
             "negatives_per_single_row": round(float(np.mean([len(r["negatives"][0]) for r in hl])), 2),
             "edit_rows_with_negatives": round(float(np.mean([bool(r["negatives"]) for r in rows if r["kind"] == "edit"])), 3),
             "edit_forms": dict(Counter(r["form"] for r in rows if r["kind"] == "edit"))}
    (out / "rows_c1b_stats.json").write_text(json.dumps(stats, indent=1))
    return stats


# ---------------------------------------------------------------- C2 rows: stage B, editing with tokens (CPU)
#
# Stage B (C2; docs/02 sections 3.7 to 3.9): editing becomes the main task, starting from C1b, and every usable edit
# is used.
# * every 50k record whose token names its region, and every cleaned v1 record likewise, appears once; add records
#   (the weakest operation after stage A) appear twice;
# * v1 targets that also change the image outside the edit are kept, with the loss restricted to the edit's
#   neighbourhood (``loss_region: local``), instead of being left out;
# * natural twins of the 50k (same image, same blind sentence, another region) are paired inside their group and
#   both rows of a pair go into one update;
# * the cleaned v1 multi-object pairs (D4, real joint targets) appear once; joined 50k edits of one image add more
#   multi-target rows; highlights as in C1b keep the localisation.
# Every edit carries its token, as a blind sentence + token or ref + token: the method always has a token.  The loss
# weights the pixels that really changed (``region: changed``) rather than the whole execution region.

C2_SHARE = {"edit_multi": 0.10, "hl": 0.06, "hlm": 0.04}      # of all rows; the rest are the real edits above
C2_REPEAT = {"add": 2, "replace": 1, "attribute": 1, "remove": 1}
C2_FORMS = {"noref_token": 0.65, "ref_token": 0.35}
C2_OPS = {"add": 0.30, "replace": 0.25, "attribute": 0.25, "remove": 0.20}   # operation draw of the joined multi rows
C2_MULTI_SIZES = {2: 0.40, 3: 0.35, 4: 0.25}
C2_HL_NAMED = 0.27            # share of the single-token highlights that name the object (C1b: 15 of 55)
V1_MAX_OUT_DIFF, V1_MAX_OUT_SHARE = 1.0, 0.02   # above either, a target changes the image outside the edit
D4_MANIFEST = V1_ROOT / "protocol_multi_qwen38_27b_v1/protocol_v2_blind/manifest_multi_protocol_v2.jsonl"
MULTI_EDIT_INTRO = "Execute all numbered instructions and keep everything else unchanged."


def multi_edit_prompt(sentences):
    """Several edits in one prompt, one numbered line each (the v2 benchmark numbers its units U1, U2, ...)."""
    return MULTI_EDIT_INTRO + "\n" + "\n".join(f"U{k + 1}: {sentence}" for k, sentence in enumerate(sentences))


def drifts(stats):
    """The target also changes the image outside the edit (v1 sources are other editors, not a paste-back)."""
    return stats["out_diff"] > V1_MAX_OUT_DIFF or stats["out_share8"] > V1_MAX_OUT_SHARE


def v1_usable(rows, tokens, dev_ids):
    """Cleaned v1 records usable for stage B: the token names the source mask and the target has the source's size.
    Records without a blind sentence are usable with ref + token."""
    return [r for r in rows if r["id"] not in dev_ids and r["id"] in tokens and tokens[r["id"]].get("size_ok")
            and tokens[r["id"]]["roundtrip"] >= MIN_ROUNDTRIP]


def d4_usable(pairs, tokens):
    """Multi-object pairs whose every unit's token names its source mask."""
    return [p for p in pairs if p["id"] in tokens and tokens[p["id"]].get("size_ok")
            and all(u["roundtrip"] >= MIN_ROUNDTRIP for u in tokens[p["id"]]["units"])]


def pair_within(group):
    """Pairs of one twin group, nearest boxes first; an odd record is returned alone."""
    left = list(group)
    pairs = []
    while len(left) > 1:
        a, b = min(((x, y) for i, x in enumerate(left) for y in left[i + 1:]), key=lambda p: record_gap(*p))
        pairs.append((a, b))
        left = [r for r in left if r is not a and r is not b]
    return pairs, left


def record_gap(a, b):
    """Gap between the (x0, y0, x1, y1) boxes of two 50k records."""
    (ax0, ay0, ax1, ay1), (bx0, by0, bx1, by1) = a["bbox"], b["bbox"]
    return float(np.hypot(max(0, max(ax0, bx0) - min(ax1, bx1)), max(0, max(ay0, by0) - min(ay1, by1))))


def natural_twins(edits):
    """Groups of 50k records on one image with the same blind sentence: only the token tells them apart."""
    groups = defaultdict(list)
    for r in edits:
        groups[(r["source_id"], r["instruction_noref"])].append(r)
    return [g for g in groups.values() if len(g) > 1]


def twin_pair(group, rng, close):
    if close:
        return min(((a, b) for i, a in enumerate(group) for b in group[i + 1:]), key=lambda p: record_gap(*p))
    return tuple(rng.sample(group, 2))


def weighted_distinct(rng, weights, k):
    """``k`` different keys of ``weights``, each draw proportional to the remaining weights."""
    pool, out = dict(weights), []
    for _ in range(k):
        names = list(pool)
        pick = rng.choices(names, weights=[pool[n] for n in names])[0]
        out.append(pick)
        del pool[pick]
    return out


class CoarseSupports:
    """Execution regions of the 50k records on 32 px cells (``support_cache.jsonl``), dilated by one cell."""

    def __init__(self, path):
        self.rows = {r["sample_id"]: r for r in map(json.loads, open(path))}
        self.cache = {}

    def __call__(self, sample_id):
        if sample_id not in self.cache:
            from pycocotools import mask as mu
            from scipy.ndimage import binary_dilation
            grid = mu.decode(as_rle(self.rows[sample_id]["coarse"])).astype(bool)
            self.cache[sample_id] = binary_dilation(grid, iterations=1)
        return self.cache[sample_id]


def composite_group(records, supports, rng, size):
    """``size`` records of one image with different operations whose execution regions stay a cell apart."""
    present = {r["operation"] for r in records}
    if len(present) < size:
        return None
    chosen, occupied = [], None
    for op in weighted_distinct(rng, {o: w for o, w in C2_OPS.items() if o in present}, size):
        options = [r for r in records if r["operation"] == op]
        rng.shuffle(options)
        for r in options[:20]:
            region = supports(r["sample_id"])
            if occupied is None or (region.shape == occupied.shape and not (region & occupied).any()):
                chosen.append(r)
                occupied = region if occupied is None else occupied | region
                break
        else:
            return None
    return chosen


def build_rows_c2(out, seed=20261019, world=8, accumulation=4):
    """Stage-B rows in training order (same order rule as C1/C1b) and their statistics.  The number of updates
    follows from the data: every usable edit once (add twice) plus the joined multi-target rows and highlights."""
    rng = random.Random(seed)
    split = json.load(open(out / "dev_split.json"))
    dev, dev_v1 = set(split["sa1b_sources"]), set(split["v1_ids"])
    train = [json.load(open(p)) for p in sorted((out / "pool").glob("*.json")) if p.stem not in dev]
    by_id = {e["source_id"]: e for e in train}
    release = [json.loads(line) for line in open(RELEASE / "annotations/train_protocol_v2.jsonl")]
    tokens = {t["sample_id"]: t for t in map(json.loads, open(out / "edit_tokens.jsonl"))}
    edits = [r for r in release if r["source_id"] not in dev and tokens[r["sample_id"]]["roundtrip"] >= MIN_ROUNDTRIP]
    negatives = lambda r: edit_negatives(by_id[r["source_id"]], tokens[r["sample_id"]]["dec"])  # noqa: E731

    def form(record_noref):
        if not record_noref:
            return "ref_token"
        return rng.choices(list(C2_FORMS), weights=list(C2_FORMS.values()))[0]

    units, counts = [], Counter()
    # 50k: twins paired inside their group (blind sentence + token for both), every other record alone
    paired = set()
    for k, group in enumerate(natural_twins([r for r in edits if r["instruction_noref"]])):
        pairs, _ = pair_within(group)
        for a, b in pairs:
            key = f"c2-twin-{len(paired) // 2}"
            units.append([{"kind": "edit", "form": "noref_token", "sample_id": x["sample_id"], "source_id": x["source_id"],
                           "negatives": negatives(x), "partners": [y["sample_id"]], "region": "changed",
                           "pair_key": key, "pair_kind": "twin"} for x, y in ((a, b), (b, a))])
            paired |= {a["sample_id"], b["sample_id"]}
    counts["edit_twin_rows"] = len(paired)
    for r in edits:
        for repeat in range(C2_REPEAT[r["operation"]]):
            if repeat == 0 and r["sample_id"] in paired:
                continue
            units.append([{"kind": "edit", "form": form(r["instruction_noref"]), "sample_id": r["sample_id"],
                           "source_id": r["source_id"], "negatives": negatives(r), "region": "changed"}])
    # cleaned v1: every usable record; drifting targets keep the loss near the edit
    v1_rows = [json.loads(line) for line in open(V1_MANIFEST)]
    v1_tokens = {t["id"]: t for t in map(json.loads, open(out / "v1_tokens.jsonl"))}
    v1_ok = v1_usable(v1_rows, v1_tokens, dev_v1)
    for r in v1_ok:
        for _ in range(C2_REPEAT[r["instruction_spec"]["operation"]]):
            row = {"kind": "edit_v1", "form": form(r["instruction_noref"]), "id": r["id"]}
            if drifts(v1_tokens[r["id"]]):
                row["loss_region"] = "local"
            units.append([row])
    # D4: the real multi-object pairs, units in a random order
    d4_pairs = [json.loads(line) for line in open(D4_MANIFEST)]
    d4_tokens = {t["id"]: t for t in map(json.loads, open(out / "d4_tokens.jsonl"))}
    d4_ok = d4_usable(d4_pairs, d4_tokens)
    for p in d4_ok:
        order = list(range(len(p["units"])))
        rng.shuffle(order)
        row = {"kind": "edit_multi_v1", "id": p["id"], "order": order,
               "forms": [form(p["units"][i]["instruction_noref"]) for i in order]}
        if drifts(d4_tokens[p["id"]]):
            row["loss_region"] = "local"
        units.append([row])
    real = sum(len(u) for u in units)
    total = -(-round(real / (1 - sum(C2_SHARE.values()))) // (world * accumulation)) * world * accumulation
    updates = total // (world * accumulation)
    n_multi, n_hlm = round(total * C2_SHARE["edit_multi"]), round(total * C2_SHARE["hlm"])
    n_hl = total - real - n_multi - n_hlm
    # joined multi-target rows from non-overlapping 50k edits of one image
    supports = CoarseSupports(out / "support_cache.jsonl")
    by_image = defaultdict(list)
    for r in edits:
        if r["instruction_noref"]:
            by_image[r["source_id"]].append(r)
    images = sorted(by_image)
    sizes = Counter()
    for size in rng.choices(list(C2_MULTI_SIZES), weights=list(C2_MULTI_SIZES.values()), k=n_multi):
        for _ in range(200):
            group = composite_group(by_image[rng.choice(images)], supports, rng, size)
            if group:
                break
        else:
            raise RuntimeError(f"no {size}-unit composite found")
        rng.shuffle(group)
        sizes[size] += 1
        units.append([{"kind": "edit_multi", "source_id": group[0]["source_id"], "sample_ids": [r["sample_id"] for r in group],
                       "forms": [form(True) for _ in group], "negatives": sorted({j for r in group for j in negatives(r)})}])
    # highlights, as in C1b
    n_hl_pairs = round(PAIR_SHARE * n_hl / 2)
    units += highlight_pairs(train, rng, n_hl_pairs, "c2-hlpair")
    units += highlight_singles(train, rng, n_hl - 2 * n_hl_pairs, round(C2_HL_NAMED * n_hl))
    units += highlight_multis(train, rng, n_hlm)

    rows = place_in_slots(units, rng, updates, world, accumulation)
    with open(out / "rows_c2.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    release_op = {r["sample_id"]: r["operation"] for r in release}
    v1_op = {r["id"]: r["instruction_spec"]["operation"] for r in v1_rows}
    stats = {"rows": len(rows), "updates": updates, "world": world, "accumulation": accumulation,
             "kinds": dict(Counter(r["kind"] for r in rows)),
             "ops_50k": dict(Counter(release_op[r["sample_id"]] for r in rows if r["kind"] == "edit")),
             "ops_v1": dict(Counter(v1_op[r["id"]] for r in rows if r["kind"] == "edit_v1")),
             "ops_joined_multi_units": dict(Counter(release_op[s] for r in rows if r["kind"] == "edit_multi" for s in r["sample_ids"])),
             "forms_single": dict(Counter(r["form"] for r in rows if r["kind"] in ("edit", "edit_v1"))),
             "records_50k_usable": len(edits), "records_50k_used": len({r["sample_id"] for r in rows if r["kind"] == "edit"}),
             "twin_rows": counts["edit_twin_rows"], "twin_pairs": sum(1 for r in rows if r.get("pair_kind") == "twin") // 2,
             "v1_usable": len(v1_ok), "v1_used": len({r["id"] for r in rows if r["kind"] == "edit_v1"}),
             "v1_rows_local_loss": sum(1 for r in rows if r["kind"] == "edit_v1" and r.get("loss_region") == "local"),
             "d4_pairs_usable": len(d4_ok), "d4_pairs": len(d4_pairs),
             "d4_rows_local_loss": sum(1 for r in rows if r["kind"] == "edit_multi_v1" and r.get("loss_region") == "local"),
             "d4_units": dict(Counter(len(p["units"]) for p in d4_ok)),
             "joined_multi_sizes": dict(sizes), "highlight_pairs": n_hl_pairs,
             "edit_rows_with_negatives": round(float(np.mean([bool(r["negatives"]) for r in rows if r["kind"] == "edit"])), 3)}
    (out / "rows_c2_stats.json").write_text(json.dumps(stats, indent=1))
    return stats


# Stage C (C3; docs/03): 1,500 updates continued from C2.  The C2 mix is kept by drawing its rows (pairs stay
# pairs); "half done" targets are left out first; the data variant replaces a quarter of the rows by extedit batch 1.
EXT_ROOT = Path("/mnt/bn/strategy-mllm-train/user/tanyue/datasets/ExtEdit_Protocol2_20261021")
# docs/01.  The 10-29 build adds D12-D14 and keeps every record of the 10-26 build byte for byte (same ids, same
# relative image paths), so rows written against SAMTokEdit_Merged_20261024 read the same records here.
MERGED_ROOT = Path("/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTokEdit_Merged_20261029")
MERGED_FORMS = ("noref_token", "ref_token")   # how a merged unit with a blind sentence can read
EXT_MANIFEST = EXT_ROOT / "manifest_protocol_v2.jsonl"
EXT_MULTI_MANIFEST = EXT_ROOT / "manifest_multi_protocol_v2.jsonl"
C3_UPDATES = 1500
C3_NEW_SHARE = 0.25
# extedit rows of the data variant, aimed at what C2 loses (docs/03, the C2 analyses): replacements whose old object is
# fully gone, whole-object attribute changes, text changes, real multi-unit pairs
C3_EXT = {"replace": 5000, "attribute": 4000, "text": 1500, "multi": 1500}
C3_EXT_REPLACE_COVERED = 0.85
C3_EXT_ATTRIBUTE_COVERED, C3_EXT_ATTRIBUTE_AREA = 0.90, 0.01
HALF_DONE, ANCHOR_DAMAGED = 0.5, 0.6     # replace / attribute changing < half the target; add changing the anchor
C2_CATEGORIES = {"edit": "edit_50k", "edit_v1": "edit_v1", "edit_multi_v1": "edit_multi_v1", "edit_multi": "edit_multi",
                 "hl": "highlight", "hl_ref": "highlight", "hlm": "highlight"}


def spec_op(spec):
    """add / remove / replace / attribute, with text changes (attribute.text) as "text"."""
    if spec["operation"] == "attribute" and spec["attributes"] and spec["attributes"][0].get("property") == "text":
        return "text"
    return spec["operation"]


def half_done(op, covered):
    """A target that did only part of its edit (docs/03, analysis/c3_plan_20261020): a replace or attribute
    change covering less than half of the target, or an add that changed most of its anchor."""
    if covered is None:
        return False
    return (op in ("replace", "attribute") and covered < HALF_DONE) or (op == "add" and covered > ANCHOR_DAMAGED)


def c2_units(rows):
    """The units of a rows file: the two rows of a pair (same ``pair_key``) together, every other row alone."""
    units, pairs = [], {}
    for row in rows:
        key = row.get("pair_key")
        if key is None:
            units.append([row])
        elif key in pairs:
            pairs[key].append(row)
        else:
            pairs[key] = [row]
            units.append(pairs[key])
    if any(len(u) > 2 for u in units):
        raise ValueError("a pair key is used by more than two rows")
    return units


def draw_units(units, need, rng):
    """Units in random order until their rows reach ``need`` (a pair that would overshoot is skipped)."""
    order = list(units)
    rng.shuffle(order)
    out, count = [], 0
    for unit in order:
        if count + len(unit) <= need:
            out.append(unit)
            count += len(unit)
        if count == need:
            break
    if count != need:
        raise ValueError(f"cannot draw {need} rows from {sum(map(len, units))}")
    return out


def ext_rows(rng, need=C3_EXT):
    """Rows of the extedit batch-1 data (D10) for the data variant: drawn per group, forms as in C2."""
    tokens = {t["id"]: t for t in map(json.loads, open(EXT_ROOT / "tokens.jsonl"))}
    multi_tokens = {t["id"]: t for t in map(json.loads, open(EXT_ROOT / "multi_tokens.jsonl"))}
    groups = defaultdict(list)
    for r in map(json.loads, open(EXT_MANIFEST)):
        t = tokens.get(r["id"])
        if not t or not t.get("size_ok") or t["roundtrip"] < MIN_ROUNDTRIP:
            continue
        op, region = spec_op(r["instruction_spec"]), r["edited_region"]
        covered = region["stats"]["s_covered"]
        if op == "replace" and covered >= C3_EXT_REPLACE_COVERED:
            groups["replace"].append(r)
        elif op == "attribute" and covered >= C3_EXT_ATTRIBUTE_COVERED and region["area_frac"] >= C3_EXT_ATTRIBUTE_AREA:
            groups["attribute"].append(r)
        elif op == "text":
            groups["text"].append(r)
    groups["multi"] = [p for p in map(json.loads, open(EXT_MULTI_MANIFEST))
                       if p["id"] in multi_tokens and multi_tokens[p["id"]].get("size_ok")
                       and all(u["roundtrip"] >= MIN_ROUNDTRIP for u in multi_tokens[p["id"]]["units"])]

    def form(noref):
        return rng.choices(list(C2_FORMS), weights=list(C2_FORMS.values()))[0] if noref else "ref_token"

    units, available = [], {g: len(v) for g, v in groups.items()}
    for group, n in need.items():
        for r in rng.sample(groups[group], n):
            if group == "multi":
                order = list(range(len(r["units"])))
                rng.shuffle(order)
                units.append([{"kind": "edit_multi_ext", "id": r["id"], "order": order,
                               "forms": [form(r["units"][i]["instruction_noref"]) for i in order], "ext_group": group}])
            else:
                units.append([{"kind": "edit_ext", "form": form(r["instruction_noref"]), "id": r["id"], "ext_group": group}])
    return units, available


def build_rows_c3(out, variant, seed=20261023, world=8, accumulation=4):
    """Stage-C rows (``variant`` "m": the C2 mix only; "d": a quarter replaced by extedit batch 1) and their
    statistics: C2's rows without the half-done targets, drawn per category in C2's proportions."""
    rng = random.Random(f"{seed}:{variant}")
    total = C3_UPDATES * world * accumulation
    release = {r["sample_id"]: r for r in map(json.loads, open(RELEASE / "annotations/train_protocol_v2.jsonl"))}
    v1 = {r["id"]: r for r in map(json.loads, open(V1_MANIFEST))}
    dropped_50k = {i for i, r in release.items()
                   if half_done(r["operation"], (r.get("edited_region") or {}).get("stats", {}).get("s_covered"))}
    # v1 adds are kept: their selection is sometimes the surface the new object stands on (analysis/c3_plan_20261020)
    dropped_v1 = {i for i, r in v1.items() if spec_op(r["instruction_spec"]) in ("replace", "attribute")
                  and half_done(r["instruction_spec"]["operation"], (r.get("edited_region") or {}).get("stats", {}).get("s_covered"))}
    c2 = [json.loads(line) for line in open(out / "rows_c2.jsonl")]
    shares = Counter(C2_CATEGORIES[r["kind"]] for r in c2)

    def clean(unit):
        for row in unit:
            if row.get("sample_id") in dropped_50k or row.get("id") in dropped_v1 \
                    or any(s in dropped_50k for s in row.get("sample_ids", ())):
                return False
        return True

    units = [u for u in c2_units(c2) if clean(u)]
    by_category = defaultdict(list)
    for unit in units:
        by_category[C2_CATEGORIES[unit[0]["kind"]]].append(unit)
    new_units, ext_available = ext_rows(rng) if variant == "d" else ([], {})
    base = total - sum(len(u) for u in new_units)
    need = {c: round(base * n / len(c2)) for c, n in shares.items()}
    need["edit_50k"] += base - sum(need.values())          # rounding goes to the largest category
    drawn = [u for c in sorted(need) for u in draw_units(by_category[c], need[c], rng)] + new_units
    rows = place_in_slots(drawn, rng, C3_UPDATES, world, accumulation)   # the total is even, so are the singles
    name = f"rows_c3{variant}"
    with open(out / f"{name}.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    (out / f"{name}_dropped.json").write_text(json.dumps({"50k": sorted(dropped_50k), "v1": sorted(dropped_v1)}))
    ops = Counter()
    for row in rows:
        if row["kind"] == "edit":
            ops[f"50k:{release[row['sample_id']]['operation']}"] += 1
        elif row["kind"] == "edit_v1":
            ops[f"v1:{spec_op(v1[row['id']]['instruction_spec'])}"] += 1
        elif row["kind"] == "edit_ext":
            ops[f"ext:{row['ext_group']}"] += 1
    stats = {"variant": variant, "rows": len(rows), "updates": C3_UPDATES, "world": world, "accumulation": accumulation,
             "kinds": dict(Counter(r["kind"] for r in rows)),
             "categories": dict(Counter(C2_CATEGORIES.get(r["kind"], "ext") for r in rows)),
             "single_edit_operations": dict(sorted(ops.items())),
             "dropped_half_done": {"50k_records": len(dropped_50k), "v1_records": len(dropped_v1),
                                   "c2_units_left_out": len(c2_units(c2)) - len(units)},
             "ext_available": ext_available, "ext_rows": dict(C3_EXT) if variant == "d" else {},
             "pairs": sum(1 for u in drawn if len(u) == 2)}
    (out / f"{name}_stats.json").write_text(json.dumps(stats, indent=1))
    return stats


# Rows from the merged dataset (docs/01).  A spec (JSON) lists draws over the default-view records, then the shares
# of highlight rows and of joined 50k multi-target rows; forms, the near negatives and twins of the SA-1B records as
# in C2.  A draw: {"family": "D11" or [...], "op": "add" | "remove" | "replace" | "attribute" | "text" | "multi" |
# "single", "text_role": "change" | "render" | "remove", and "rows": N or "share": fraction of all rows or "rest": true};
# with "opsd": true its rows are on-policy self-distillation rows (kind opsd_merged, training/opsd.py: no target image
# is read, no negatives or twins; place_in_slots gives them whole step pairs on every rank); "forms" gives the draw its
# own weights of the two forms for units with a blind sentence (default: the spec's "forms"); "area_max" keeps the
# records whose every unit changes at most this share of the image (edited_region.area_frac); "pool_only" keeps the
# SA-1B records whose source image is in the mask pool (so they get near negatives); "pairs": true draws D12 twins as
# pairs (both members, blind sentence + token, each the other's S as a negative, one update; the draw's rows are
# twice its pairs).  Records are drawn without replacement over all draws; a draw larger than what is left repeats
# its records.  The spec's "exclude" names a file of record ids (one per line, or a JSON list) kept out of every draw,
# for example held-out rows.
#
# Spec options for the two-stage retrain (docs/03): "negatives" lists the families whose single-unit records get near
# negatives when their image is in the pool (default ["D1"]); "payload_cap": {"limit": 50, "families": [...]} applies
# the merge's new-object description cap (merge_data.apply_caps: multi-unit records counted first and kept, single
# records in a fixed order) to families the merge did not cap; "highlights" may give "pairs" (share of rows that are
# counterfactual highlight pairs; without it they are PAIR_SHARE of the single share), "buckets" (shares of BUCKETS),
# "named" (share of the single highlights that name the object) and "pair_kinds" (weights of "adjacent", "same",
# "part_host").
def merged_match(record, draw):
    families = draw.get("family")
    if families is not None and record["family"] not in ([families] if isinstance(families, str) else families):
        return False
    op = draw.get("op")
    if op == "multi" and len(record["units"]) < 2:
        return False
    if op == "single" and len(record["units"]) != 1:
        return False
    if op not in (None, "multi", "single") and (len(record["units"]) != 1 or record["units"][0]["op"] != op):
        return False
    area = draw.get("area_max")
    if area is not None and any((u.get("edited_region") or {}).get("area_frac") is None
                                or u["edited_region"]["area_frac"] > area for u in record["units"]):
        return False
    role = draw.get("text_role")
    return role is None or any(u["text"]["role"] == role for u in record["units"])


def excluded_ids(spec):
    """Record ids a rows spec keeps out of every draw: ``"exclude"`` names a file, one id per line or a JSON list."""
    if not spec.get("exclude"):
        return set()
    text = Path(spec["exclude"]).read_text()
    return set(json.loads(text)) if text.lstrip().startswith("[") else {line.strip() for line in text.splitlines() if line.strip()}


def payload_capped(records, cap):
    """Ids of the records of ``cap["families"]`` over the new-object description cap (``cap["limit"]`` records per
    description, counted as merge_data.apply_caps does: multi-unit records first and never dropped, then the single
    records in the merge's fixed order)."""
    from samtok_edit21.preparation.merge_data import payload_key, stable
    live = [r for r in records if r["family"] in cap["families"]]
    counts, out = Counter(), set()
    for r in live:
        if len(r["units"]) > 1:
            for u in r["units"]:
                key = payload_key(u["instruction_spec"])
                if key:
                    counts[key] += 1
    for r in sorted((r for r in live if len(r["units"]) == 1), key=lambda r: stable(r["id"])):
        key = payload_key(r["units"][0]["instruction_spec"])
        if key is None:
            continue
        if counts[key] >= cap["limit"]:
            out.add(r["id"])
        else:
            counts[key] += 1
    return out


def twin_pairs(options):
    """D12 twins among ``options`` whose partner is there too, as (member, partner) pairs in id order."""
    by_id = {r["id"]: r for r in options}
    return [(r, by_id[r["twin"]["partner"]]) for r in sorted(by_id.values(), key=lambda r: r["id"])
            if r.get("twin") and r["twin"]["partner"] in by_id and r["id"] < r["twin"]["partner"]]


def build_rows_merged(out, spec_path):
    spec = json.load(open(spec_path))
    name, rng = spec["name"], random.Random(spec.get("seed", 20261025))
    world, accumulation, updates = spec.get("world", 8), spec.get("accumulation", 4), spec["updates"]
    total = updates * world * accumulation
    forms = spec.get("forms", C2_FORMS)
    for weights in [forms] + [d["forms"] for d in spec["draws"] if "forms" in d]:
        if not weights or set(weights) - set(MERGED_FORMS):
            raise ValueError(f"forms must weigh {MERGED_FORMS}, got {weights}")
    records = [r for r in map(json.loads, open(Path(spec.get("merged", MERGED_ROOT)) / "records.jsonl")) if r["default"]]
    capped = payload_capped(records, spec["payload_cap"]) if spec.get("payload_cap") else set()
    records = [r for r in records if r["id"] not in capped]
    dev = set(json.load(open(out / "dev_split.json"))["sa1b_sources"])
    train = [json.load(open(p)) for p in sorted((out / "pool").glob("*.json")) if p.stem not in dev]
    pool = {e["source_id"]: e for e in train}
    negative_families = set(spec.get("negatives", ["D1"] if spec.get("d1_negatives", True) else []))
    hl = spec.get("highlights", {})
    n_hl, n_hlm = round(total * hl.get("single", 0.0)), round(total * hl.get("multi", 0.0))
    n_pairs = round(total * hl["pairs"] / 2) if "pairs" in hl else round(PAIR_SHARE * n_hl / 2)
    n_named = round(hl.get("named", C2_HL_NAMED) * n_hl)
    if "pairs" in hl:
        n_hl += 2 * n_pairs                   # the single-token highlight rows: singles and both members of the pairs
    shares = tuple(hl.get("buckets", BUCKET_SHARE))
    if len(shares) != len(BUCKETS) - 1 or abs(sum(shares) - 1) > 1e-6:
        raise ValueError(f"highlight buckets must give {len(BUCKETS) - 1} shares summing to 1, got {shares}")
    n_join = round(total * spec.get("joined_multi", 0.0))
    draws = spec["draws"]
    sizes = [d.get("rows", round(total * d.get("share", 0.0))) for d in draws]
    sizes = [s - s % 2 if d.get("pairs") else s for s, d in zip(sizes, draws)]
    rest = [k for k, d in enumerate(draws) if d.get("rest")]
    left = total - n_hl - n_hlm - n_join - sum(s for k, s in enumerate(sizes) if k not in rest)
    if len(rest) > 1 or left < 0 or (not rest and left) or any(draws[k].get("pairs") for k in rest):
        raise ValueError(f"draws do not add up: {left} rows left for {len(rest)} 'rest' draws")
    for k in rest:
        sizes[k] = left

    def form(unit, weights=None):
        if not unit.get("instruction_noref"):
            return "ref_token"
        weights = weights or forms
        return rng.choices(list(weights), weights=list(weights.values()))[0]

    def negatives(r):
        if r["family"] not in negative_families or not r.get("sa1b") or r["sa1b"]["source_id"] not in pool:
            return None
        return edit_negatives(pool[r["sa1b"]["source_id"]], r["units"][0]["token"]["dec"])

    used, singles, units, report = excluded_ids(spec), [], [], []
    for k, (draw, need) in enumerate(zip(draws, sizes)):
        options = [r for r in records if r["id"] not in used and merged_match(r, draw)
                   and (not draw.get("pool_only") or (r.get("sa1b") and r["sa1b"]["source_id"] in pool))]
        if draw.get("pairs"):
            pairs = twin_pairs(options)
            if not pairs and need:
                raise ValueError(f"no twin pair for draw {draw}")
            picks = rng.sample(pairs, min(need // 2, len(pairs)))
            while len(picks) < need // 2:
                picks += rng.sample(pairs, min(need // 2 - len(picks), len(pairs)))
            used |= {r["id"] for pair in picks for r in pair}
            report.append({"draw": draw, "rows": need, "available_pairs": len(pairs), "repeated_pairs": max(0, need // 2 - len(pairs))})
            for j, (a, b) in enumerate(picks):
                key = f"{name}-d{k}-pair-{j}"
                members = []
                for x, y in ((a, b), (b, a)):
                    row = {"kind": "edit_merged", "id": x["id"], "form": "noref_token", "partners": [y["id"]],
                           "pair_key": key, "pair_kind": "twin"}
                    found = negatives(x)
                    if found:
                        row["negatives"] = found
                    members.append(row)
                units.append(members)
            continue
        if not options and need:
            raise ValueError(f"no record for draw {draw}")
        picks = rng.sample(options, min(need, len(options)))
        while len(picks) < need:               # oversample: the draw needs more than is left
            picks += rng.sample(options, min(need - len(picks), len(options)))
        used |= {r["id"] for r in picks}
        report.append({"draw": draw, "rows": need, "available": len(options), "repeated": max(0, need - len(options))})
        for r in picks:
            row = {"kind": "opsd_merged" if draw.get("opsd") else "edit_merged", "id": r["id"]}
            if len(r["units"]) > 1:
                order = list(range(len(r["units"])))
                rng.shuffle(order)
                row.update(order=order, forms=[form(r["units"][i], draw.get("forms")) for i in order])
                units.append([row])
                continue
            row["form"] = form(r["units"][0], draw.get("forms"))
            if row["kind"] == "opsd_merged":
                units.append([row])
                continue
            found = negatives(r)
            if found is not None:
                row["negatives"] = found
            singles.append((r, row))
    # SA-1B twins among the drawn rows: same image, same blind sentence; both read blind with their token
    twins = defaultdict(list)
    if spec.get("d1_twins", True):
        for r, row in singles:
            if r["family"] == "D1" and r["units"][0].get("instruction_noref"):
                twins[(r["sa1b"]["source_id"], r["units"][0]["instruction_noref"])].append((r, row))
    paired = set()
    for k, group in enumerate(g for g in twins.values() if len(g) > 1):
        items = [{"bbox": r["sa1b"]["bbox"], "row": row} for r, row in group]
        pairs, _ = pair_within(items)
        for j, (a, b) in enumerate(pairs):
            key = f"{name}-twin-{k}-{j}"
            ra, rb = a["row"], b["row"]
            if ra["id"] == rb["id"]:
                continue
            ra.update(form="noref_token", partners=[rb["id"]], pair_key=key, pair_kind="twin")
            rb.update(form="noref_token", partners=[ra["id"]], pair_key=key, pair_kind="twin")
            units.append([ra, rb])
            paired |= {id(ra), id(rb)}
    units += [[row] for _, row in singles if id(row) not in paired]
    # joined 50k multi-target rows (as C2), from the default-view SA-1B records
    if n_join:
        release = [json.loads(line) for line in open(RELEASE / "annotations/train_protocol_v2.jsonl")]
        ok = {r["sa1b"]["sample_id"] for r in records if r["family"] == "D1"}
        supports = CoarseSupports(out / "support_cache.jsonl")
        by_image = defaultdict(list)
        for r in release:
            if r["sample_id"] in ok and r["instruction_noref"]:
                by_image[r["source_id"]].append(r)
        images = sorted(by_image)
        dec = {r["sa1b"]["sample_id"]: r["units"][0]["token"]["dec"] for r in records if r["family"] == "D1"}
        for size in rng.choices(list(C2_MULTI_SIZES), weights=list(C2_MULTI_SIZES.values()), k=n_join):
            for _ in range(200):
                group = composite_group(by_image[rng.choice(images)], supports, rng, size)
                if group:
                    break
            else:
                raise RuntimeError(f"no {size}-unit composite found")
            rng.shuffle(group)
            entry = pool.get(group[0]["source_id"])
            negatives_ = sorted({j for r in group for j in edit_negatives(entry, dec[r["sample_id"]])}) if entry else []
            units.append([{"kind": "edit_multi", "source_id": group[0]["source_id"], "sample_ids": [r["sample_id"] for r in group],
                           "forms": [form(r) for r in group], "negatives": negatives_}])
    units += highlight_pairs(train, rng, n_pairs, f"{name}-hlpair", shares=shares, kinds=hl.get("pair_kinds"))
    units += highlight_singles(train, rng, n_hl - 2 * n_pairs, n_named, shares=shares)
    units += highlight_multis(train, rng, n_hlm)
    rows = place_in_slots(units, rng, updates, world, accumulation)
    with open(out / f"{name}.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    family = {r["id"]: r["family"] for r in records}
    by_id = {r["id"]: r for r in records}
    merged_rows = [r for r in rows if r["kind"] == "edit_merged"]
    single_hl = [r for r in rows if r["kind"] in ("hl", "hl_ref")]
    hl_area = Counter(bucket(pool[r["source_id"]]["masks"][r["masks"][0]]["area"]) for r in single_hl)
    stats = {"name": name, "rows": len(rows), "updates": updates, "world": world, "accumulation": accumulation,
             "kinds": dict(Counter(r["kind"] for r in rows)), "draws": report,
             "merged_rows_by_family": dict(Counter(family[r["id"]] for r in merged_rows)),
             "merged_rows_multi": sum(1 for r in merged_rows if len(by_id[r["id"]]["units"]) > 1),
             "merged_units_by_op": dict(Counter(u["op"] for r in merged_rows for u in by_id[r["id"]]["units"])),
             "opsd_rows_by_family": dict(Counter(family[r["id"]] for r in rows if r["kind"] == "opsd_merged")),
             "merged_records_distinct": len({r["id"] for r in merged_rows}),
             "twin_pairs": sum(1 for r in rows if r.get("pair_kind") == "twin") // 2,
             "twin_pairs_by_family": {f: n // 2 for f, n in Counter(family[r["id"]] for r in merged_rows if r.get("pair_kind") == "twin").items()},
             "rows_with_negatives_by_family": dict(Counter(family[r["id"]] for r in merged_rows if r.get("negatives"))),
             "d1_rows_with_negatives": sum(1 for r in merged_rows if r.get("negatives") and family[r["id"]] == "D1"),
             "forms": dict(Counter(r.get("form") for r in merged_rows if "form" in r)),
             "payload_capped_records": len(capped),
             "highlight_pairs_by_kind": {k: n // 2 for k, n in Counter(r["pair_kind"] for r in rows if r["kind"] == "hl" and r.get("pair_key")).items()},
             "highlight_single_named": sum(1 for r in single_hl if r["kind"] == "hl_ref"),
             "highlight_single_by_bucket": {f"{BUCKETS[b]}-{BUCKETS[b + 1]}": hl_area[b] for b in sorted(hl_area)},
             "highlight_images": len({r["source_id"] for r in rows if r["kind"] in ("hl", "hl_ref", "hlm")}),
             "pool_train_images": len(train)}
    (out / f"{name}_stats.json").write_text(json.dumps(stats, indent=1))
    return stats


def build_quick_c2(out, seed=20261019):
    """Items checked during stage-B training: 40 edits (10 per operation), 10 remove pairs, 24 + 8 single
    highlights, 8 multi-token highlights, 4 instance pairs and 2 part/host pairs (112 items)."""
    rng = random.Random(seed)
    items = json.load(open(out / "eval_c1.json"))
    pick = []
    for op in ("add", "remove", "replace", "attribute"):
        pick += rng.sample([i for i, it in enumerate(items) if it["set"] == "edit" and it["operation"] == op], 10)
    for kind, n_pairs in (("remove", 10), ("instance", 4), ("part_host", 2)):
        group = "remove" if kind == "remove" else "pair"
        pairs = sorted({it["pair"] for it in items if it["set"] == group and (group == "remove" or it["pair_kind"] == kind)})
        chosen = set(rng.sample(pairs, n_pairs))
        pick += [i for i, it in enumerate(items) if it["set"] == group and it["pair"] in chosen
                 and (group == "remove" or it["pair_kind"] == kind)]
    for variant, n in (("noref", 24), ("ref", 8)):
        pick += rng.sample([i for i, it in enumerate(items) if it["set"] == "single" and it["variant"] == variant], n)
    pick += rng.sample([i for i, it in enumerate(items) if it["set"] == "multi"], 8)
    json.dump(sorted(pick), open(out / "eval_c2_quick_idx.json", "w"))
    return {"items": len(pick), "sets": dict(Counter(items[i]["set"] for i in pick))}


# ---------------------------------------------------------------- C1 evaluation items (CPU)

def build_eval(out, seed=20261017):
    """Highlight (single, with name), multi-token, counterfactual pairs, edits and the remove probe on dev images."""
    rng = random.Random(seed)
    split = json.load(open(out / "dev_split.json"))
    dev = set(split["sa1b_sources"])
    entries = [json.load(open(out / "pool" / f"{s}.json")) for s in sorted(dev)]
    items = []
    singles = pick_regions(entries, rng, 600, replace=False)
    named = [k for k, (e, m) in enumerate(singles) if entries[e]["masks"][m]["category"]]
    with_name = set(rng.sample(named, min(200, len(named))))
    for k, (e_index, m_index) in enumerate(singles):
        entry = entries[e_index]
        phrase = entry["masks"][m_index]["category"] if k in with_name else None
        items.append({"set": "single", "variant": "ref" if phrase else "noref", "source_id": entry["source_id"],
                      "image": entry["image"], "mask": m_index, "phrase": phrase})
    for k in range(100):
        size = [2, 3, 4][k % 3]
        for _ in range(50):
            entry = rng.choice(entries)
            group = multi_group(entry, rng, size)
            if group:
                break
        items.append({"set": "multi", "source_id": entry["source_id"], "image": entry["image"], "masks": group,
                      "colors": rng.sample(MULTI_COLORS, size)})
    pairs = []
    for entry in entries:
        for i, m in enumerate(entry["masks"]):
            if not usable(m):
                continue
            for j in m["rel"]["same"]:
                if j > i and usable(entry["masks"][j]) and max(m["area"], entry["masks"][j]["area"]) <= 3 * min(m["area"], entry["masks"][j]["area"]):
                    pairs.append(("instance", entry, i, j))
            for j in m["rel"]["hosts"][:1]:
                if usable(entry["masks"][j]):
                    pairs.append(("part_host", entry, i, j))
    rng.shuffle(pairs)
    chosen = [p for p in pairs if p[0] == "instance"][:150] + [p for p in pairs if p[0] == "part_host"][:50]
    for number, (kind, entry, i, j) in enumerate(chosen):
        for tok, (a, b) in (("a", (i, j)), ("b", (j, i))):
            items.append({"set": "pair", "pair": number, "pair_kind": kind, "tok": tok, "source_id": entry["source_id"],
                          "image": entry["image"], "mask": a, "other": b})
    release = [json.loads(line) for line in open(RELEASE / "annotations/train_protocol_v2.jsonl")]
    tokens = {t["sample_id"]: t for t in map(json.loads, open(out / "edit_tokens.jsonl"))}
    dev_edits = [r for r in release if r["source_id"] in dev and r["instruction_noref"]
                 and tokens[r["sample_id"]]["roundtrip"] >= MIN_ROUNDTRIP and r["edited_region"]["area_frac"] > 0]
    for op in ("add", "remove", "replace", "attribute"):
        for r in rng.sample([r for r in dev_edits if r["operation"] == op], 25):
            items.append({"set": "edit", "sample_id": r["sample_id"], "operation": op})
    instance_pairs = [p for p in chosen if p[0] == "instance"][:60]
    for number, (kind, entry, i, j) in enumerate(instance_pairs):
        for tok, (a, b) in (("a", (i, j)), ("b", (j, i))):
            items.append({"set": "remove", "pair": number, "tok": tok, "source_id": entry["source_id"],
                          "image": entry["image"], "mask": a, "other": b})
    (out / "eval_c1.json").write_text(json.dumps(items))
    quick = ([k for k, it in enumerate(items) if it["set"] == "single"][:64]
             + [k for k, it in enumerate(items) if it["set"] == "multi"][:16]
             + [k for k, it in enumerate(items) if it["set"] == "pair" and it["pair"] < 16])
    (out / "eval_c1_quick_idx.json").write_text(json.dumps(quick))
    stats = dict(Counter(it["set"] for it in items))
    stats["pair_kinds"] = dict(Counter(p[0] for p in chosen))
    stats["single_variants"] = dict(Counter(it["variant"] for it in items if it["set"] == "single"))
    (out / "eval_c1_stats.json").write_text(json.dumps(stats, indent=1))
    return stats


def main(argv=None):
    p = argparse.ArgumentParser(prog="samtok-edit round2-data", description=__doc__.split("\n\n")[0])
    p.add_argument("step", choices=["pool", "rows", "rows-c1b", "rows-c2", "rows-c3m", "rows-c3d", "rows-merged",
                                    "quick-c2", "eval", "pool-ext-prepare", "pool-ext-work", "pool-ext-check"])
    p.add_argument("--out", required=True)
    p.add_argument("--spec", help="rows-merged: the JSON spec of the draws (see build_rows_merged)")
    p.add_argument("--base", help="pool-ext-prepare: the round-2 data directory whose files and pool are linked")
    p.add_argument("--count", type=int, default=POOL_EXT_IMAGES, help="pool-ext-prepare: new images")
    p.add_argument("--name", help="pool-ext-work: worker name (claims and logs)")
    p.add_argument("--limit-shards", type=int, default=0, help="pool-ext-work: stop after this many shards (tests)")
    args = p.parse_args(argv)
    out = Path(args.out)
    if args.step == "pool":
        run_pool(args)
    elif args.step == "pool-ext-prepare":
        print(json.dumps(pool_ext_prepare(out, args.base, count=args.count), indent=1))
    elif args.step == "pool-ext-work":
        import socket
        print(json.dumps({"shards": pool_ext_work(out, args.name or f"{socket.gethostname()}-{os.getpid()}",
                                                  args.limit_shards)}))
    elif args.step == "pool-ext-check":
        print(json.dumps(pool_ext_check(out), indent=1)[:4000])
    elif args.step == "rows":
        print(json.dumps(build_rows(out), indent=1))
    elif args.step == "rows-c1b":
        print(json.dumps(build_rows_c1b(out), indent=1))
    elif args.step == "rows-c2":
        print(json.dumps(build_rows_c2(out), indent=1))
    elif args.step in ("rows-c3m", "rows-c3d"):
        print(json.dumps(build_rows_c3(out, args.step[-1]), indent=1))
    elif args.step == "rows-merged":
        print(json.dumps(build_rows_merged(out, args.spec), indent=1))
    elif args.step == "quick-c2":
        print(json.dumps(build_quick_c2(out), indent=1))
    else:
        print(json.dumps(build_eval(out), indent=1))


if __name__ == "__main__":
    main()
