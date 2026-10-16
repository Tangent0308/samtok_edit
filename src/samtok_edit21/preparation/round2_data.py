"""Round-2 data (docs/08 section 5): the mask pool, C1 training rows and the C1 evaluation set.

Sources (protocol 2.0, read only):
* the SA-1B 50k release: ``annotations/train_protocol_v2.jsonl`` (edits, support and changed regions);
* the SA-1B dense annotations of the same 1,132 source images: ``<run root>/sources/<id>/dense.json``.

Steps (outputs go to one data directory; nothing is written next to the sources):

    python -m torch.distributed.run --nproc_per_node 8 -m samtok_edit21 round2-data pool --out <dir>
    python -m samtok_edit21 round2-data rows --out <dir>
    python -m samtok_edit21 round2-data eval --out <dir>

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
V1_MANIFEST = Path("/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTokEdit_Cleaned_20261009/"
                   "protocol_adapt_qwen38_27b_v1/protocol_v2_blind/manifest_protocol_v2.jsonl")

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
    """60 SA-1B source images and 40 v1 records held out of all training (docs/08 section 2)."""
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


def multi_group(entry, rng, size):
    """``size`` usable, pairwise disjoint, non-duplicate masks of one image (decoded regions disjoint too)."""
    candidates = [i for i, m in enumerate(entry["masks"]) if usable(m) and bucket(m["area"]) is not None]
    rng.shuffle(candidates)
    chosen, decoded = [], []
    for i in candidates:
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
    p.add_argument("step", choices=["pool", "rows", "eval"])
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    out = Path(args.out)
    if args.step == "pool":
        run_pool(args)
    elif args.step == "rows":
        print(json.dumps(build_rows(out), indent=1))
    else:
        print(json.dumps(build_eval(out), indent=1))


if __name__ == "__main__":
    main()
