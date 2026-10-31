"""Read-only source inventory. Write only to --output; no model calls."""

import argparse
import collections
import concurrent.futures
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pycocotools import mask as mu

POOL = None
MERGED = None
PILOT = None
BENCH = None
SPAN = re.compile(r"<\|mt_start\|><\|mt_([0-9]{4})\|><\|mt_([0-9]{4})\|><\|mt_end\|>")


def valid(s):
    m = SPAN.fullmatch(s or "")
    return bool(m and int(m[1]) < 256 and 256 <= int(m[2]) < 512)


def rle(r):
    return {
        "size": r["size"],
        "counts": r["counts"].encode() if isinstance(r["counts"], str) else r["counts"],
    }


def sid_of(p):
    m = re.search(r"(sa_[0-9]+)", str(p))
    return m[1] if m else ""


def exclusions():
    pilot = json.loads((PILOT / "data/split_sources.json").read_text())
    old = json.loads((POOL / "dev_split.json").read_text())
    bench = [json.loads(x) for x in BENCH.read_text().splitlines()]
    result = {
        "benchmark_sids": {x["source"]["image_id"] for x in bench},
        "old_dev_sids": set(old["sa1b_sources"]),
        "pilot_train_sids": {sid_of(x["original_image"]) for x in pilot["train"]},
        "pilot_holdout_sids": {
            sid_of(x["original_image"]) for k in ["dev", "test"] for x in pilot[k]
        },
        "pilot_train_groups": {x["group"] for x in pilot["train"]},
        "pilot_holdout_groups": {x["group"] for k in ["dev", "test"] for x in pilot[k]},
    }
    for v in result.values():
        v.discard("")
    return result


def bucket(a):
    return int(
        np.searchsorted(
            [0.0005, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.3], a, side="right"
        )
    )


def inventory_pool(out):
    start = time.time()
    jobs = json.loads((POOL / "pool_ext/jobs.json").read_text())
    new = {x[0]: x for x in jobs["images"]}
    ex = exclusions()
    fields = [
        ("key", pa.string()),
        ("sid", pa.string()),
        ("image", pa.string()),
        ("dense", pa.string()),
        ("pool_file", pa.string()),
        ("origin", pa.string()),
        ("w", pa.int32()),
        ("h", pa.int32()),
        ("index", pa.int32()),
        ("oid", pa.string()),
        ("category", pa.string()),
        ("area", pa.float64()),
        ("dec_area", pa.float64()),
        ("roundtrip", pa.float64()),
        ("span", pa.string()),
        ("valid", pa.bool_()),
        ("bbox", pa.list_(pa.int32())),
        ("s_digest", pa.string()),
        ("bucket", pa.int32()),
        ("same", pa.list_(pa.int32())),
        ("hosts", pa.list_(pa.int32())),
        ("parts", pa.list_(pa.int32())),
        ("touching", pa.list_(pa.int32())),
        ("duplicate", pa.list_(pa.int32())),
        ("old_dev", pa.bool_()),
        ("benchmark", pa.bool_()),
        ("pilot_train", pa.bool_()),
        ("pilot_holdout", pa.bool_()),
        ("span_multiplicity", pa.int32()),
    ]
    schema = pa.schema(fields)
    files = sorted((POOL / "pool").glob("*.json"))
    counts = collections.Counter()
    quality = collections.Counter()
    relations = collections.Counter()
    image_rows = []

    def one(path):
        raw = path.read_bytes()
        o = json.loads(raw)
        sid = o["source_id"]
        ms = o["masks"]
        w, h = o["size"]
        rs = [rle(m["rle"]) for m in ms]
        # This installed pycocotools uses uint8 for a list count; chunk large images.
        areas = (
            np.concatenate(
                [mu.area(rs[j : j + 128]) for j in range(0, len(rs), 128)]
            ).astype(float)
            / (w * h)
            if rs
            else []
        )
        boxes = (
            np.concatenate([mu.toBbox(rs[j : j + 128]) for j in range(0, len(rs), 128)])
            if rs
            else []
        )
        spans = collections.Counter(m.get("span") for m in ms)
        origin = (
            "new_" + ("D12" if "/D12/" in o["image"] else "D15")
            if sid in new
            else "base"
        )
        flags = {
            "old_dev": sid in ex["old_dev_sids"],
            "benchmark": sid in ex["benchmark_sids"],
            "pilot_train": sid in ex["pilot_train_sids"],
            "pilot_holdout": sid in ex["pilot_holdout_sids"],
        }
        common = {
            "sid": sid,
            "image": o["image"],
            "dense": new.get(sid, ["", "", ""])[2],
            "pool_file": str(path),
            "origin": origin,
            "w": w,
            "h": h,
        } | flags
        rows = []
        for i, (m, a, b) in enumerate(zip(ms, areas, boxes)):
            q = m.get("roundtrip", 0)
            s = m.get("span", "")
            rels = {
                k: m.get("rel", {}).get(k, [])
                for k in ["same", "hosts", "parts", "touching", "duplicate"]
            }
            rows.append(
                common
                | {
                    "key": f"P:{sid}:{i}",
                    "index": i,
                    "oid": m["oid"],
                    "category": m.get("category", ""),
                    "area": float(a),
                    "dec_area": m.get("dec_area", 0),
                    "roundtrip": q,
                    "span": s,
                    "valid": valid(s),
                    "bucket": bucket(a),
                    "bbox": [int(b[0]), int(b[1]), int(b[0] + b[2]), int(b[1] + b[3])],
                    "s_digest": hashlib.sha1(rs[i]["counts"]).hexdigest(),
                    "span_multiplicity": spans[s],
                }
                | rels
            )
        image = common | {
            "masks": len(rows),
            "q50": sum(
                r["valid"] and r["roundtrip"] >= 0.5 and r["dec_area"] > 0 for r in rows
            ),
            "q70": sum(
                r["valid"] and r["roundtrip"] >= 0.7 and r["dec_area"] > 0 for r in rows
            ),
            "pool_sha256": hashlib.sha256(raw).hexdigest(),
        }
        return rows, image

    writer = pq.ParquetWriter(out / "pool_regions.parquet", schema, compression="zstd")
    buffer = []
    with concurrent.futures.ThreadPoolExecutor(8) as executor:
        for j, (rows, im) in enumerate(executor.map(one, files), 1):
            buffer += rows
            image_rows.append(im)
            counts[im["origin"] + "_images"] += 1
            eligible = not (im["old_dev"] or im["benchmark"] or im["pilot_holdout"])
            for r in rows:
                counts[im["origin"] + "_masks"] += 1
                if r["valid"] and r["dec_area"] > 0:
                    for q in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]:
                        if r["roundtrip"] >= q:
                            quality[(r["origin"], eligible, q, r["bucket"])] += 1
                if eligible and r["valid"] and r["roundtrip"] >= 0.7:
                    for k in ["same", "hosts", "parts", "touching", "duplicate"]:
                        relations[(k, bool(r[k]))] += 1
                    relations[("span_collision", r["span_multiplicity"] > 1)] += 1
            if len(buffer) >= 10000:
                writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
                buffer = []
            if j % 1000 == 0:
                print(
                    json.dumps(
                        {"images": j, "elapsed_s": round(time.time() - start, 1)}
                    ),
                    flush=True,
                )
    if buffer:
        writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
    writer.close()
    pq.write_table(
        pa.Table.from_pylist(image_rows),
        out / "pool_images.parquet",
        compression="zstd",
    )
    summary = {
        "counts": dict(counts),
        "quality": [
            {
                "origin": k[0],
                "eligible_train": k[1],
                "threshold": k[2],
                "bucket": k[3],
                "n": v,
            }
            for k, v in sorted(quality.items())
        ],
        "relations": [
            {"kind": k[0], "present": k[1], "n": v}
            for k, v in sorted(relations.items())
        ],
        "excluded_image_counts": {
            k: sum(im[k] for im in image_rows)
            for k in ["old_dev", "benchmark", "pilot_holdout", "pilot_train"]
        },
        "images_q70_ge5_eligible": sum(
            im["q70"] >= 5
            and not (im["old_dev"] or im["benchmark"] or im["pilot_holdout"])
            for im in image_rows
        ),
        "pool_source_summary": json.loads((POOL / "pool_ext/summary.json").read_text()),
        "elapsed_s": time.time() - start,
    }
    (out / "pool_inventory.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["counts"]), flush=True)


def inventory_merged(out):
    start = time.time()
    ex = exclusions()
    index = pq.read_table(
        MERGED / "units.parquet", columns=["record", "uid", "default", "split", "group"]
    )
    index = index.filter(pc.and_(index["default"], pc.equal(index["split"], "train")))
    allowed = {(x["record"], x["uid"]) for x in index.to_pylist()}
    records = {x[0] for x in allowed}
    strings = [
        "key",
        "record",
        "uid",
        "family",
        "group",
        "sid",
        "image",
        "span",
        "ref_instruction",
        "noref_instruction",
        "ref",
        "scope",
        "target_part",
        "operation",
        "payload",
        "selection_source",
        "anchor_mode",
        "source_role",
        "s_digest",
    ]
    fields = [(k, pa.string()) for k in strings] + [
        (k, pa.bool_())
        for k in [
            "valid",
            "noref_eligible",
            "old_dev",
            "benchmark",
            "pilot_train",
            "pilot_holdout",
            "half_done",
            "has_site",
        ]
    ]
    fields += [(k, pa.float64()) for k in ["area", "roundtrip", "dec_area"]] + [
        ("w", pa.int32()),
        ("h", pa.int32()),
        ("bucket", pa.int32()),
        ("bbox", pa.list_(pa.int32())),
        ("record_K", pa.int32()),
    ]
    schema = pa.schema(fields)
    writer = pq.ParquetWriter(
        out / "merged_regions.parquet", schema, compression="zstd"
    )
    buffer = []
    counts = collections.Counter()
    qualities = collections.Counter()
    images = {}
    examples = {}
    num = 0
    with (MERGED / "records.jsonl").open() as f:
        for line in f:
            head = line.split('"', 4)
            if len(head) < 4 or head[3] not in records:
                continue
            r = json.loads(line)
            sid = sid_of(r["source_image"])
            w, h = r["size"]
            g = r["group"]
            flags = {
                "old_dev": sid in ex["old_dev_sids"],
                "benchmark": sid in ex["benchmark_sids"],
                "pilot_train": g in ex["pilot_train_groups"]
                or bool(sid and sid in ex["pilot_train_sids"]),
                "pilot_holdout": g in ex["pilot_holdout_groups"]
                or bool(sid and sid in ex["pilot_holdout_sids"]),
            }
            images[g] = {
                "group": g,
                "sid": sid,
                "image": str(MERGED / r["source_image"]),
                "family": r["family"],
                "w": w,
                "h": h,
            } | flags
            for u in r["units"]:
                if (r["id"], u["uid"]) not in allowed:
                    continue
                sel = u.get("selection", {})
                tok = u.get("token") or {}
                spec = u.get("instruction_spec") or {}
                sem = u.get("mask_semantics") or {}
                if not sel.get("rle_size") or not sel.get("rle_counts"):
                    continue
                s = rle({"size": sel["rle_size"], "counts": sel["rle_counts"]})
                a = float(mu.area(s)) / (w * h)
                box = mu.toBbox(s)
                dec = tok.get("dec")
                da = float(mu.area(rle(dec))) / (w * h) if dec else 0.0
                rt = tok.get("roundtrip") or 0
                span = tok.get("span", "")
                row = {
                    "key": f"M:{r['id']}:{u['uid']}",
                    "record": r["id"],
                    "uid": u["uid"],
                    "family": r["family"],
                    "group": g,
                    "sid": sid,
                    "image": str(MERGED / r["source_image"]),
                    "span": span,
                    "ref_instruction": u.get("instruction_ref") or "",
                    "noref_instruction": u.get("instruction_noref") or "",
                    "ref": spec.get("ref") or "",
                    "scope": spec.get("scope") or "",
                    "target_part": spec.get("target_part") or "",
                    "operation": u.get("op") or "",
                    "payload": spec.get("payload") or "",
                    "selection_source": sel.get("source") or "",
                    "anchor_mode": str(sem.get("anchor_mode") or ""),
                    "source_role": str(sem.get("source_role") or ""),
                    "s_digest": hashlib.sha1(s["counts"]).hexdigest(),
                    "area": a,
                    "roundtrip": rt,
                    "dec_area": da,
                    "valid": valid(span),
                    "noref_eligible": not bool(u.get("noref_ineligible"))
                    and bool(u.get("instruction_noref")),
                    "half_done": bool(u.get("half_done")),
                    "has_site": bool(sel.get("site")),
                    "w": w,
                    "h": h,
                    "bucket": bucket(a),
                    "record_K": len(r["units"]),
                    "bbox": [
                        int(box[0]),
                        int(box[1]),
                        int(box[0] + box[2]),
                        int(box[1] + box[3]),
                    ],
                } | flags
                buffer.append(row)
                num += 1
                counts[(r["family"], row["operation"], row["selection_source"])] += 1
                if valid(span) and not (
                    flags["benchmark"] or flags["old_dev"] or flags["pilot_holdout"]
                ):
                    for q in [0.5, 0.7, 0.8, 0.9]:
                        if rt >= q:
                            qualities[(r["family"], q, bucket(a))] += 1
                eg = (r["family"], row["operation"], row["selection_source"])
                if eg not in examples:
                    examples[eg] = {"record": r["id"], "unit": u, "size": r["size"]}
            if len(buffer) >= 10000:
                writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
                buffer = []
                print(
                    json.dumps(
                        {"units": num, "elapsed_s": round(time.time() - start, 1)}
                    ),
                    flush=True,
                )
    if buffer:
        writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
    writer.close()
    pq.write_table(
        pa.Table.from_pylist(list(images.values())),
        out / "merged_images.parquet",
        compression="zstd",
    )
    summary = {
        "default_train_units": num,
        "images": len(images),
        "counts": [
            {"family": k[0], "op": k[1], "selection_source": k[2], "n": v}
            for k, v in sorted(counts.items())
        ],
        "quality": [
            {"family": k[0], "threshold": k[1], "bucket": k[2], "n": v}
            for k, v in sorted(qualities.items())
        ],
        "excluded_image_counts": {
            k: sum(im[k] for im in images.values())
            for k in ["old_dev", "benchmark", "pilot_holdout", "pilot_train"]
        },
        "elapsed_s": time.time() - start,
    }
    (out / "merged_inventory.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "semantic_examples.json").write_text(
        json.dumps(list(examples.values()), ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({"units": num, "images": len(images)}), flush=True)


def main():
    global POOL, MERGED, PILOT, BENCH
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pool", "merged"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--pool-root", type=Path, required=True)
    ap.add_argument("--merged-source", type=Path, required=True)
    ap.add_argument("--pilot-root", type=Path, required=True)
    ap.add_argument("--benchmark-manifest", type=Path, required=True)
    a = ap.parse_args()
    POOL, MERGED, PILOT, BENCH = (
        a.pool_root,
        a.merged_source,
        a.pilot_root,
        a.benchmark_manifest,
    )
    o = Path(a.output)
    o.mkdir(exist_ok=True, parents=True)
    target = o / f"{a.mode}_regions.parquet"
    if target.exists():
        raise FileExistsError(target)
    {"pool": inventory_pool, "merged": inventory_merged}[a.mode](o)


if __name__ == "__main__":
    main()
