"""Merge of the protocol-2.0 edit data into one training set (2026-10-24; docs/01 section 2.13).

Sources (read only):
* D1   SA-1B 50k self-distilled edits (release format: S and the support region as PNG masks), tokens in
       ``edit_tokens.jsonl`` of the round-2 data directory;
* D3   cleaned v1 single-unit edits, D4 cleaned v1 multi-unit pairs (protocol 2.0 manifests), tokens in
       ``v1_tokens.jsonl`` / ``d4_tokens.jsonl``;
* D10, D11  the extedit batches, tokens shipped with them; their source-image SHA-1 come from the ingest rows of
       the production runs;
* the unified text re-check (``verdicts.jsonl``: role, requested text and verdict of every text unit);
* the SA-1B region pool D2 and the 50k support cache, for the near negatives, twins and joined multi-target rows of
  the D1 records.
Output, one folder with every image hard-linked (same file system, no extra space):
* ``records.jsonl``  one record per pair in the D3 / D4 unit schema; every record has ``units`` (a single edit has
  one), each unit its token inline and a ``text`` tag; the record its source-image ``group`` (SHA-1), ``drift`` and
  ``loss_region``, ``split`` and ``exclude`` (the reasons it is not in the default training view);
* ``units.parquet`` one row per unit for filtering; ``views/default_train.txt``, ``views/dev.txt``;
  ``stats.json``, ``build.json``;
* ``images/<family>/...``, ``pool/``, ``support_cache.jsonl``.

    python -m samtok_edit21 merge-data --out <dataset dir>
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np

from samtok_edit21.preparation import round2_data as rd

DS = Path("/mnt/bn/strategy-mllm-train/user/tanyue/datasets")
R2 = Path("/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v3/data/round2_20261016")
RUNS = Path("/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/data_pipeline/02_production_runs")
TEXT_VERDICTS = RUNS / "merge_text_recheck_20261024/admin/calibration/verdicts.jsonl"
BENCHMARK_ASSETS = DS / "samtok_edit_benchmark_v2/assets"
EXT = {"D10": (DS / "ExtEdit_Protocol2_20261021", [RUNS / "extedit_protocol2_100k_local8_20261021/node0"]),
       "D11": (DS / "ExtEdit_Protocol2_20261022", [RUNS / "extedit_protocol2_batch2_local8_20261022/node0"]
               + [RUNS / f"extedit_protocol2_batch2_4n_20261023/node{k}" for k in range(4)])}
D4_MANIFEST = rd.D4_MANIFEST
NEAR_DHASH = 6                 # as the extedit ingest: a benchmark source image within 6 bits of dHash is a copy
PAYLOAD_CAP = 50               # uses of one new-object description (add / replace), as the extedit publish
PER_IMAGE_CAP = 3              # records per source image outside D1 (D1 edits one image many times by design)
FAMILY_ORDER = ("D10", "D11", "D3", "D4", "D1")   # who keeps a capped slot first
TEXT_SCOPE = re.compile(r"\b(text|lettering|letters|words?|logo|logos|inscription|writing|caption|watermark|graffiti|"
                        r"number|numbers|label|slogan|title|headline|name)\b", re.I)


# ----------------------------------------------------------------------------------------------- helpers

def spec_op(spec):
    """add / remove / replace / attribute, a text change (attribute.text) as "text"."""
    if spec["operation"] == "attribute" and spec.get("attributes") and spec["attributes"][0].get("property") == "text":
        return "text"
    return spec["operation"]


def covered(region):
    return ((region or {}).get("stats") or {}).get("s_covered")


def half_done(spec, region, family):
    """A target that did part of its edit (docs/03, C3): a replace or attribute change covering less than half
    of S; for SA-1B 50k adds also an anchor changed by more than 60 % (v1 add selections can be the support itself).
    A text change is not measured this way (as in C3's v1 rows): only its glyphs change, a small part of S."""
    op = spec_op(spec)
    return rd.half_done(op, covered(region)) if (family == "D1" or op != "add") else False


def rle_only(region):
    if not region or not region.get("rle_counts"):
        return None
    return {k: region[k] for k in ("rle_size", "rle_counts", "box_1000") if k in region}


def png_rle(path):
    from PIL import Image
    mask = np.asarray(Image.open(path).convert("L")) > 0
    out = rd.encode_rle(mask)
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    box = [int(xs.min() * 1000 // w), int(ys.min() * 1000 // h), int(-(-(xs.max() + 1) * 1000 // w)),
           int(-(-(ys.max() + 1) * 1000 // h))] if xs.size else None
    return {"rle_size": list(out["size"]), "rle_counts": out["counts"], "box_1000": box}


def _d1_masks(job):
    root, mask, support = job
    return png_rle(root / mask), png_rle(root / support)


def dhash(data, size=8):
    """64-bit difference hash (gray, (size+1) x size), the extedit ingest's near-copy test."""
    from PIL import Image
    with Image.open(io.BytesIO(data)) as im:
        g = im.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    px = list(g.getdata())
    bits = 0
    for y in range(size):
        for x in range(size):
            bits = (bits << 1) | (px[y * (size + 1) + x] > px[y * (size + 1) + x + 1])
    return bits


def _digest(path):
    data = Path(path).read_bytes()
    return str(path), hashlib.sha1(data).hexdigest(), dhash(data)


def near(h, hashes, limit=NEAR_DHASH):
    return any(bin(h ^ other).count("1") <= limit for other in hashes)


def text_tag(family, record_id, k, spec, verdicts):
    """The unit's text tag: role change / render / remove / None, the requested text and the re-check verdict."""
    v = verdicts.get((family, record_id, k))
    if v and v["text_role"]:
        return {"role": v["text_role"], "requested": v["requested"], "verdict": v["verdict"], "reason": v["reason"]}
    if spec["operation"] == "remove" and TEXT_SCOPE.search(spec.get("scope") or ""):
        return {"role": "remove", "requested": None, "verdict": None, "reason": None}
    return {"role": None, "requested": None, "verdict": None, "reason": None}


def unit_fields(unit):
    """The D3 unit fields a merged unit keeps (regions as RLE)."""
    keep = ("instruction_spec", "instruction_ref", "instruction_noref", "noref_ineligible", "mask_semantics", "flags",
            "comparable", "audit", "text_check", "text_payload", "protocol_v2_change", "instruction_protocol_version")
    out = {k: unit[k] for k in keep if k in unit}
    out["selection"] = {**{k: v for k, v in (unit.get("selection") or {}).items() if k != "instances"}}
    out["execution_region"] = rle_only(unit.get("execution_region"))
    out["edited_region"] = unit.get("edited_region")
    return out


def token_of(tok):
    return {"span": tok["span"], "dec": tok["dec"], "roundtrip": tok["roundtrip"]}


# ----------------------------------------------------------------------------------------------- sources

def d1_records(verdicts, workers):
    root = rd.RELEASE
    tokens = {t["sample_id"]: t for t in map(json.loads, open(R2 / "edit_tokens.jsonl"))}
    dev = set(json.load(open(R2 / "dev_split.json"))["sa1b_sources"])
    rows = [json.loads(line) for line in open(root / "annotations/train_protocol_v2.jsonl")]
    with ProcessPoolExecutor(workers) as pool:
        masks = list(pool.map(_d1_masks, [(root, r["mask"], r["support_mask"]) for r in rows], chunksize=64))
    for r, (s_rle, support_rle) in zip(rows, masks):
        spec = r["instruction_spec"]
        tok = tokens.get(r["sample_id"])
        unit = {"uid": "U1", "instruction_spec": spec, "instruction_ref": r["instruction"],
                "instruction_noref": r.get("instruction_noref"), "noref_ineligible": r.get("noref_ineligible"),
                "mask_semantics": r.get("mask_semantics"), "flags": [], "audit": r.get("audit"),
                "selection": {"source": "mask", **s_rle}, "execution_region": {"source": "support", **support_rle},
                "edited_region": r["edited_region"], "op": spec_op(spec),
                "token": token_of(tok) if tok else None, "half_done": half_done(spec, r["edited_region"], "D1"),
                "text": text_tag("D1", r["sample_id"], 0, spec, verdicts)}
        yield {"id": f"D1:{r['sample_id']}", "family": "D1", "origin": "sa1b", "source_ref": r["sample_id"],
               "group": None, "source_image": str(root / r["source_image"]),
               "target_image": str(root / r["edited_image"]), "size": list(r["image_size"]), "units": [unit],
               "drift": None, "loss_region": None, "size_ok": True, "split": "dev" if r["source_id"] in dev else "train",
               "sa1b": {"source_id": r["source_id"], "sample_id": r["sample_id"], "object_id": r.get("object_id"),
                        "bbox": r.get("bbox"), "operation": r["operation"]},
               "provenance": {"job_id": r.get("job_id"), "case_id": r.get("case_id"), "unit_id": r.get("unit_id")}}


def manifest_records(family, root, files, single_tokens, multi_tokens, verdicts, dev_ids=(), shas=None):
    """D3 / D4 / D10 / D11 records (the D3 / D4 manifest schema)."""
    for rel, multi in files:
        for line in open(root / rel):
            r = json.loads(line)
            tok = (multi_tokens if multi else single_tokens).get(r["id"])
            units_in = r["units"] if multi else [r]
            units = []
            for k, u in enumerate(units_in):
                spec = u["instruction_spec"]
                t = None
                if tok:
                    t = tok["units"][k] if multi else tok
                    if multi and t["uid"] != u["id"].rsplit("#", 1)[-1]:
                        raise ValueError(f"{family} token units do not match {r['id']}")
                units.append({"uid": f"U{k + 1}", **unit_fields(u), "op": spec_op(spec),
                              "token": token_of(t) if t else None,
                              "half_done": half_done(spec, u.get("edited_region"), family),
                              "text": text_tag(family, r["id"], k, spec, verdicts)})
            size = units[0]["selection"].get("rle_size")
            drift = {k: tok[k] for k in ("out_diff", "out_share8", "in_diff") if tok and k in tok} or None
            drifting = bool(drift) and rd.drifts(drift)
            rec = {"id": f"{family}:{r['id']}", "family": family, "origin": r.get("dataset"), "source_ref": r["id"],
                   "group": None, "source_image": str(root / r["source_image"]),
                   "target_image": str(root / r["target_image"]), "size": [size[1], size[0]] if size else None,
                   "units": units, "drift": drift, "loss_region": "local" if drifting else None,
                   "size_ok": bool(tok and tok.get("size_ok", True)), "split": "dev" if r["id"] in dev_ids else "train"}
            for key in ("type_before", "instruction_before", "observed", "treatment", "provenance", "naturalness_check"):
                if key in r:
                    rec[key] = r[key]
            if shas is not None and r["id"] in shas:
                rec["group"] = f"sha1:{shas[r['id']]}"
            yield rec


def ext_shas(runs):
    out = {}
    for run in runs:
        for line in open(run / "ingest" / "rows.jsonl"):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[row["id"]] = row["source_sha1"]
    return out


def benchmark_index(workers):
    paths = sorted(BENCHMARK_ASSETS.glob("*/source.*"))
    with ThreadPoolExecutor(workers) as pool:
        digests = list(pool.map(_digest, paths))
    return {d for _, d, _ in digests}, [h for _, _, h in digests]


# ----------------------------------------------------------------------------------------------- views

def payload_key(spec):
    """The normalised new-object description of an add / replace (the extedit publish rule), else None."""
    if spec["operation"] not in ("add", "replace"):
        return None
    text = " ".join(str(spec.get("payload") or "").lower().replace(",", " ").split())
    for article in ("a ", "an ", "some ", "one "):
        if text.startswith(article):
            text = text[len(article):]
            break
    return text or None


def hard_exclusions(rec):
    """Reasons a record leaves the default view, from its own fields."""
    out = []
    if any(u["token"] is None for u in rec["units"]):
        out.append("no_token")
    elif any(u["token"]["roundtrip"] < rd.MIN_ROUNDTRIP for u in rec["units"]):
        out.append("roundtrip_low")
    if not rec["size_ok"]:
        out.append("size_mismatch")
    if any(u["half_done"] for u in rec["units"]):
        out.append("half_done")
    if any(u["text"]["verdict"] == "fail" for u in rec["units"]):
        out.append("text_fail")
    return out


def stable(key):
    return hashlib.sha1(key.encode()).hexdigest()


def apply_caps(records):
    """Payload cap over every family (multi-unit pairs counted first and never dropped, then single records in a
    fixed pseudo-random order) and the per-image cap outside D1 (in FAMILY_ORDER, then the same order); only records
    still in the default view take part."""
    live = [r for r in records if r["split"] == "train" and not r["exclude"]]
    counts = Counter()
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
        if counts[key] >= PAYLOAD_CAP:
            r["exclude"].append("payload_cap")
        else:
            counts[key] += 1
    per_group = Counter()
    rank = {f: i for i, f in enumerate(FAMILY_ORDER)}
    for r in sorted((r for r in live if not r["exclude"] and r["family"] != "D1"),
                    key=lambda r: (rank[r["family"]], stable(r["id"]))):
        per_group[r["group"]] += 1
        if per_group[r["group"]] > PER_IMAGE_CAP:
            r["exclude"].append("per_image_cap")


# ----------------------------------------------------------------------------------------------- build

def link(job):
    src, dst = job
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if os.path.samefile(src, dst):
            return False
        raise FileExistsError(dst)
    os.link(src, dst)
    return True


def image_name(rec, kind, src):
    family, ref = rec["family"], rec["source_ref"]
    if family == "D1":
        if kind == "source":
            return f"images/D1/source/{rec['sa1b']['source_id']}{Path(src).suffix}"
        return f"images/D1/target/{ref[:2]}/{ref}{Path(src).suffix}"
    return f"images/{family}/{ref[-2:]}/{ref}.{kind}{Path(src).suffix}"


def build(out, workers=32, limit=0):
    """Build the merged dataset in ``out`` (``limit``: only the first records of every family, for a smoke build)."""
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    verdicts = {}
    for v in map(json.loads, open(TEXT_VERDICTS)):
        verdicts[(v["source"], v["record"], v["unit"])] = v
    dev_v1 = set(json.load(open(R2 / "dev_split.json"))["v1_ids"])
    v1 = rd.V1_ROOT
    records = list(d1_records(verdicts, workers))
    v1_tok = {t["id"]: t for t in map(json.loads, open(R2 / "v1_tokens.jsonl"))}
    d4_tok = {t["id"]: t for t in map(json.loads, open(R2 / "d4_tokens.jsonl"))}
    records += manifest_records("D3", v1, [(rd.V1_MANIFEST.relative_to(v1), False)], v1_tok, {}, verdicts, dev_v1)
    records += manifest_records("D4", v1, [(D4_MANIFEST.relative_to(v1), True)], {}, d4_tok, verdicts)
    for family, (root, runs) in EXT.items():
        single = {t["id"]: t for t in map(json.loads, open(root / "tokens.jsonl"))}
        multi = {t["id"]: t for t in map(json.loads, open(root / "multi_tokens.jsonl"))}
        records += manifest_records(family, root, [("manifest_protocol_v2.jsonl", False),
                                                   ("manifest_multi_protocol_v2.jsonl", True)],
                                    single, multi, verdicts, shas=ext_shas(runs))
    if limit:
        kept = Counter()
        records = [r for r in records if (kept.update([r["family"]]) or kept[r["family"]] <= limit)]
    # source-image identity of the v1 records and the benchmark test (D10 / D11 were tested at ingest)
    bench_sha, bench_hash = benchmark_index(workers)
    to_hash = sorted({r["source_image"] for r in records if r["family"] in ("D1", "D3", "D4")})
    with ThreadPoolExecutor(workers) as pool:
        digests = {p: (s, h) for p, s, h in pool.map(_digest, to_hash)}
    for r in records:
        r["exclude"] = []
        if r["family"] in ("D1", "D3", "D4"):
            sha, h = digests[r["source_image"]]
            r["group"] = f"sha1:{sha}"
            if sha in bench_sha or near(h, bench_hash):
                r["exclude"].append("benchmark_overlap")
        r["exclude"] += hard_exclusions(r)
    dev_groups = {r["group"] for r in records if r["split"] == "dev"}
    for r in records:
        if r["split"] == "train" and r["group"] in dev_groups:
            r["exclude"].append("dev_overlap")
    apply_caps(records)
    # images: hard links, paths relative to the dataset folder
    jobs, seen = [], {}
    for r in records:
        for kind, key in (("source", "source_image"), ("target", "target_image")):
            dst = image_name(r, kind, r[key])
            if dst not in seen:
                seen[dst] = r[key]
                jobs.append((Path(r[key]), out / dst))
            elif seen[dst] != r[key]:
                raise ValueError(f"two images would share {dst}")
            r[key] = dst
    for name in sorted(os.listdir(R2 / "pool"))[:limit or None]:
        jobs.append((R2 / "pool" / name, out / "pool" / name))
    jobs.append((R2 / "support_cache.jsonl", out / "support_cache.jsonl"))
    with ThreadPoolExecutor(workers) as pool:
        made = sum(pool.map(link, jobs))
    for r in records:
        r["default"] = r["split"] == "train" and not r["exclude"]
    with open(out / "records.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "views").mkdir(exist_ok=True)
    (out / "views" / "default_train.txt").write_text("".join(r["id"] + "\n" for r in records if r["default"]))
    (out / "views" / "dev.txt").write_text("".join(r["id"] + "\n" for r in records if r["split"] == "dev"))
    write_units(out, records)
    stats = statistics(records)
    stats["links"] = {"files": len(jobs), "made": made}
    (out / "stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False))
    write_build_info(out, {"workers": workers, "limit": limit}, started, workers=workers)
    return stats


def inputs():
    """The files the build reads; the images, the pool entries and the benchmark sources are only counted."""
    files = [rd.RELEASE / "annotations/train_protocol_v2.jsonl", R2 / "edit_tokens.jsonl", R2 / "dev_split.json",
             rd.V1_MANIFEST, D4_MANIFEST, R2 / "v1_tokens.jsonl", R2 / "d4_tokens.jsonl", TEXT_VERDICTS,
             R2 / "support_cache.jsonl"]
    for root, runs in EXT.values():
        files += [root / name for name in ("manifest_protocol_v2.jsonl", "manifest_multi_protocol_v2.jsonl",
                                           "tokens.jsonl", "multi_tokens.jsonl")]
        files += [run / "ingest" / "rows.jsonl" for run in runs]
    return files


def _file_entry(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return {"path": str(path), "bytes": os.path.getsize(path), "sha1": h.hexdigest()}


def write_build_info(out, arguments, started, finished=None, commit=None, note=None, workers=16):
    """``build.json``: code commit (and uncommitted files under src/), arguments, times, every input with its size
    and SHA-1, and the counts of the linked folders."""
    repo = Path(__file__).resolve().parents[3]

    def git(*args):
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True).stdout.strip()

    info = {"command": "python -m samtok_edit21 merge-data", "arguments": arguments, "started": started,
            "finished": finished or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "commit": commit or git("rev-parse", "HEAD"),
            "uncommitted": [] if commit else git("status", "--porcelain", "--", "src").splitlines()}
    if note:
        info["note"] = note
    with ThreadPoolExecutor(workers) as pool:
        info["inputs"] = list(pool.map(_file_entry, inputs()))
    info["counted"] = {"sa1b_pool_entries": len(os.listdir(R2 / "pool")),
                       "benchmark_sources": len(list(BENCHMARK_ASSETS.glob("*/source.*")))}
    (Path(out) / "build.json").write_text(json.dumps(info, indent=1, ensure_ascii=False))
    return info


def write_units(out, records):
    import pyarrow as pa
    import pyarrow.parquet as pq
    cols = defaultdict(list)
    for r in records:
        for u in r["units"]:
            cols["record"].append(r["id"])
            cols["uid"].append(u["uid"])
            cols["family"].append(r["family"])
            cols["origin"].append(r["origin"])
            cols["multi"].append(len(r["units"]) > 1)
            cols["op"].append(u["op"])
            cols["text_role"].append(u["text"]["role"])
            cols["text_verdict"].append(u["text"]["verdict"])
            cols["area"].append((u.get("edited_region") or {}).get("area_frac"))
            cols["roundtrip"].append(u["token"]["roundtrip"] if u["token"] else None)
            cols["half_done"].append(u["half_done"])
            cols["noref"].append(u.get("instruction_noref") is not None)
            cols["loss_local"].append(r["loss_region"] == "local")
            cols["split"].append(r["split"])
            cols["default"].append(r["default"])
            cols["exclude"].append(",".join(r["exclude"]))
            cols["group"].append(r["group"])
    pq.write_table(pa.table(dict(cols)), out / "units.parquet")


def statistics(records):
    out = {"records": len(records), "units": sum(len(r["units"]) for r in records)}
    default = [r for r in records if r["default"]]
    out["default_records"] = len(default)
    out["default_units"] = sum(len(r["units"]) for r in default)
    out["dev_records"] = sum(r["split"] == "dev" for r in records)
    fam = defaultdict(Counter)
    for r in records:
        fam[r["family"]]["records"] += 1
        fam[r["family"]]["units"] += len(r["units"])
        fam[r["family"]]["multi_records"] += len(r["units"]) > 1
        if r["default"]:
            fam[r["family"]]["default_records"] += 1
            fam[r["family"]]["default_units"] += len(r["units"])
    out["by_family"] = {f: dict(c) for f, c in sorted(fam.items())}
    ops = Counter(u["op"] for r in default for u in r["units"])
    out["default_units_by_op"] = dict(ops.most_common())
    out["default_units_by_family_op"] = {f: dict(Counter(u["op"] for r in default if r["family"] == f for u in r["units"]))
                                         for f in sorted(fam)}
    out["default_text_roles"] = dict(Counter(u["text"]["role"] for r in default for u in r["units"] if u["text"]["role"]))
    out["exclusions"] = dict(Counter(x for r in records for x in r["exclude"]).most_common())
    out["default_loss_local_records"] = sum(r["loss_region"] == "local" for r in default)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(prog="samtok-edit merge-data", description=__doc__.split("\n\n")[0])
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--limit", type=int, default=0, help="first records of every family only (smoke build)")
    args = p.parse_args(argv)
    print(json.dumps(build(args.out, args.workers, args.limit), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
