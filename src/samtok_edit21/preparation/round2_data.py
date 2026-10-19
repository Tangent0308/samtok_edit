"""Round-2 data (docs/archive/v3_process/08 section 5): the mask pool, C1 training rows and the C1 evaluation set.

Sources (protocol 2.0, read only):
* the SA-1B 50k release: ``annotations/train_protocol_v2.jsonl`` (edits, support and changed regions);
* the SA-1B dense annotations of the same 1,132 source images: ``<run root>/sources/<id>/dense.json``.

Steps (outputs go to one data directory; nothing is written next to the sources):

    python -m torch.distributed.run --nproc_per_node 8 -m samtok_edit21 round2-data pool --out <dir>
    python -m samtok_edit21 round2-data rows --out <dir>
    python -m samtok_edit21 round2-data eval --out <dir>
    python -m samtok_edit21 round2-data rows-c2 --out <dir>     # stage B (docs/archive/v3_process/08 section 9)

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
    """60 SA-1B source images and 40 v1 records held out of all training (docs/archive/v3_process/08 section 2)."""
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
        entry = pool_image(codec, source_id, RELEASE / image, SA1B / "sources" / source_id / "dense.json")
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


def pick_regions(entries, rng, count, *, replace=True):
    """``count`` (image, mask) draws following BUCKET_SHARE; every other draw, where possible, has a negative.

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
    for b, share in enumerate(BUCKET_SHARE):
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
    from PIL import Image
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
# part of the multi-token rows around an adjacent pair (docs/archive/v3_process/08 section 6.3).

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
    """The second region of a counterfactual pair: the host of a part or a part of a host (``part_host``), else
    an adjacent usable region of similar size (any category, decoded regions disjoint).  None if there is none."""
    dec, _, diag = geometry(entry)
    m = entry["masks"][index]
    if part_host:
        options = [j for j in m["rel"]["hosts"] + m["rel"]["parts"] if usable(entry["masks"][j])]
    else:
        skip = set(m["rel"]["hosts"]) | set(m["rel"]["parts"]) | set(m["rel"]["duplicate"]) | {index}
        options = [j for j, o in enumerate(entry["masks"])
                   if j not in skip and usable(o) and max(m["area"], o["area"]) <= PAIR_AREA_RATIO * min(m["area"], o["area"])
                   and box_gap(dec[index], dec[j]) <= NEAR_GAP * diag]
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
    the remaining slots two by two."""
    if accumulation % 2:
        raise ValueError("pairs need an even accumulation")
    pairs = [u for u in units if len(u) == 2]
    singles = [u[0] for u in units if len(u) == 1]
    rng.shuffle(singles)
    blocks = pairs + [singles[k:k + 2] for k in range(0, len(singles), 2)]
    rng.shuffle(blocks)
    if len(blocks) != updates * world * accumulation // 2 or any(len(b) != 2 for b in blocks):
        raise ValueError("units do not fill the slots exactly")
    rows, k = [None] * (updates * world * accumulation), 0
    for u in range(updates):
        for r in range(world):
            for s in range(0, accumulation, 2):
                rows[(u * accumulation + s) * world + r], rows[(u * accumulation + s + 1) * world + r] = blocks[k]
                k += 1
    return rows


def highlight_pairs(train, rng, n_pairs, key_prefix):
    """Counterfactual highlight pairs: same prompt, template and colour, an adjacent region or a part and its host."""
    units = []
    for e_index, m_index in pick_regions(train, rng, 4 * n_pairs):
        if len(units) == n_pairs:
            break
        entry = train[e_index]
        part_host = rng.random() < PAIR_PART_HOST
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
                            "pair_key": key, "pair_kind": "part_host" if part_host else "adjacent"})
        units.append(members)
    if len(units) < n_pairs:
        raise RuntimeError(f"only {len(units)} of {n_pairs} counterfactual pairs found")
    return units


def highlight_singles(train, rng, need, n_named):
    """``need`` single-token highlights with near negatives; ``n_named`` of them name the object (``hl_ref``)."""
    singles = pick_regions(train, rng, need + 16)[:need]      # the per-bucket rounding can fall a few draws short
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
# Stage B (docs/archive/v3_process/08 section 9): editing becomes the main task, starting from C1b, and every usable edit is used.
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
    p.add_argument("step", choices=["pool", "rows", "rows-c1b", "rows-c2", "quick-c2", "eval"])
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    out = Path(args.out)
    if args.step == "pool":
        run_pool(args)
    elif args.step == "rows":
        print(json.dumps(build_rows(out), indent=1))
    elif args.step == "rows-c1b":
        print(json.dumps(build_rows_c1b(out), indent=1))
    elif args.step == "rows-c2":
        print(json.dumps(build_rows_c2(out), indent=1))
    elif args.step == "quick-c2":
        print(json.dumps(build_quick_c2(out), indent=1))
    else:
        print(json.dumps(build_eval(out), indent=1))


if __name__ == "__main__":
    main()
