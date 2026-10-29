"""The data line's twins / multi-object / salvaged single-target production as merge families D12, D13, D14 (docs/01).

Source (read only): the production of 2026-10-25..28 (labeling repo ``Tangent0308/sam3-crispedit``, branch
``sa1b_single_twins_target``, ``docs/TWINS_MULTI_ZH.md``), ``PRODUCTION/{twins_balanced,multi_balanced,singles}.jsonl``
with absolute paths into the production folder:
* D12 twins: 12,511 groups of two members, one SA-1B image, the same operation on two objects, each member edited
  from the source on its own (one record per member, ``twin`` names the partner);
* D13 multi-object: 39,814 images with 2-4 targets of different operations fused into one target image;
* D14 salvaged singles: 12,363 members / targets of groups that did not ship as groups (user 10-29: use them too).
Every target is pasted back: outside the supports of its units it equals the source.

Stages, in the run folder ``RUN`` (multi-machine: README there):
1. ``prepare``: one job per source image (``jobs.jsonl``), split into shards (``shards.json``).
2. ``work --stage tokens`` (one process per GPU): SAMTok token of every unit's selection S as delivered and, for an
   add, of its host object's SA-1B mask.  An add's S is the attachment spot found from the edit; the codec misses
   most of these spots (round trip < 0.5 for 64 % of a sample), so ``assemble`` uses the host instead whenever the
   spot fails (user 10-29); the source image's SHA-1 and dHash for the merge's identity and benchmark tests.
   ``work --stage regions`` (CPU processes): every unit's edited region (D1's method, ``regions/edited.py``) and
   each target's change outside its supports.
   Shards are claimed through atomic mkdir on NAS; a claim whose heartbeat is older than ``STALE`` is taken over.
3. ``assemble``: records in the merge schema (``records_D12.jsonl`` ...), read by ``merge_data``.  A multi-object
   record whose unit fails (round trip, half done) keeps its other units: the failing unit's whole generation
   component is restored to source pixels in a new target image (``recomposed/``), exactly as the production drops
   a failed target; the units it loses are listed in ``pruned``.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import socket
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from samtok_edit21.preparation import merge_data as md
from samtok_edit21.preparation import round2_data as rd

PRODUCTION = md.RUNS / "sa1b_samtok_group_twins50k_multi100k_20261025/final"
RUN = md.RUNS / "merge_sa1b_groups_20261029"
FILES = {"twins": "twins_balanced.jsonl", "multi": "multi_balanced.jsonl", "singles": "singles.jsonl"}
FAMILY = {"twins": "D12", "multi": "D13", "singles": "D14"}
SHARD_JOBS = 250
STALE = 1200          # seconds without heartbeat before a claim is taken over
SEED = 20261029


# ----------------------------------------------------------------------------------------------- jobs

def _unit(u, target=None):
    out = {"uid": u["unit_id"], "op": u["operation"], "mask": u["mask"], "support": u["support_mask"],
           "object_id": str(u.get("object_id"))}
    if target:
        out["target"] = target
    return out


def jobs(production=PRODUCTION):
    """One job per source image: its units, masks, supports, targets and dense labels (for the add hosts)."""
    out = []
    for line in open(production / FILES["twins"]):
        g = json.loads(line)
        out.append({"kind": "twins", "rid": g["twin_id"], "source_id": g["source_id"], "source": g["source_image"],
                    "dense": g["source_provenance"]["dense_record"],
                    "units": [_unit(m, m["edited_image"]) for m in g["members"]]})
    for line in open(production / FILES["multi"]):
        g = json.loads(line)
        out.append({"kind": "multi", "rid": g["multi_id"], "source_id": g["source_id"], "source": g["source_image"],
                    "dense": g["source_provenance"]["dense_record"], "target": g["joint_target"],
                    "units": [_unit(u) for u in g["units"]]})
    for line in open(production / FILES["singles"]):
        s = json.loads(line)
        out.append({"kind": "singles", "rid": s["job_id"], "source_id": s["source_id"], "source": s["source_image"],
                    "dense": str(Path(s["source_image"]).parent / "dense.json"),
                    "units": [{"uid": s["unit_id"], "op": s["operation"], "mask": s["mask"],
                               "support": s["support_mask"], "object_id": str(s.get("object_id")),
                               "target": s["edited_image"]}]})
    random.Random(SEED).shuffle(out)       # shards of similar cost
    return out


def prepare(run=RUN, production=PRODUCTION, shard_jobs=SHARD_JOBS):
    run.mkdir(parents=True, exist_ok=True)
    js = jobs(production)
    with open(run / "jobs.jsonl", "w") as f:
        for j in js:
            f.write(json.dumps(j) + "\n")
    shards = [[start, min(start + shard_jobs, len(js))] for start in range(0, len(js), shard_jobs)]
    info = {"jobs": len(js), "units": sum(len(j["units"]) for j in js), "by_kind": dict(Counter(j["kind"] for j in js)),
            "shards": shards, "shard_jobs": shard_jobs, "production": str(production),
            "files": {k: md._file_entry(production / v) for k, v in FILES.items()}}
    (run / "shards.json").write_text(json.dumps(info, indent=1))
    return info


# ----------------------------------------------------------------------------------------------- per image

def load_mask(path):
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im.convert("L")) > 0


def host_mask(dense, object_id, shape):
    """The SA-1B dense-label mask of ``object_id`` (an add's host object), or None."""
    from pycocotools import mask as mu
    objects = dense["objects_json"]
    objects = json.loads(objects) if isinstance(objects, str) else objects
    for ob in objects:
        if str(ob["object_id"]) == str(object_id):
            rle = ob["segmentation"]
            if tuple(rle["size"]) != tuple(shape):
                return None
            return mu.decode(rd.as_rle(rle)).astype(bool)
    return None


def token_job(codec, job):
    """Tokens of the units' S (and add hosts), the source image's SHA-1, dHash and size."""
    from PIL import Image
    data = Path(job["source"]).read_bytes()
    image = Image.open(io.BytesIO(data)).convert("RGB")
    w, h = image.size
    out = {"rid": job["rid"], "kind": job["kind"], "sha1": hashlib.sha1(data).hexdigest(),
           "dhash": f"{md.dhash(data):016x}", "size": [w, h], "units": []}
    masks, keys, dense = [], [], None
    for u in job["units"]:
        entry = {"uid": u["uid"], "site": None, "host": None}
        s = load_mask(u["mask"])
        if s.shape != (h, w) or not s.any():
            entry["error"] = f"selection {s.shape} empty={not s.any()}"
        else:
            masks.append(s)
            keys.append((len(out["units"]), "site"))
        if u["op"] == "add":
            if dense is None:
                dense = json.load(open(job["dense"]))
            host = host_mask(dense, u["object_id"], (h, w))
            if host is not None and host.any():
                masks.append(host)
                keys.append((len(out["units"]), "host"))
        out["units"].append(entry)
    spans = rd.encode_masks(codec, image, masks) if masks else []
    decoded = rd.decode_spans(codec, image, spans) if spans else []
    for (k, which), mask, span, dec in zip(keys, masks, spans, decoded):
        union = float((mask | dec).sum())
        entry = {"span": span, "dec": rd.encode_rle(dec), "area": round(float(mask.mean()), 6),
                 "roundtrip": round(float((mask & dec).sum()) / union, 4) if union else 0.0}
        if which == "host":
            from samtok_edit21.regions.edited import encode
            entry["rle"] = encode(mask)
        out["units"][k][which] = entry
    return out


def region_job(job):
    """Edited region of every unit (clipped to its support) and each target's change outside its supports."""
    from PIL import Image
    from samtok_edit21.regions.edited import color_difference, encode, sa1b_edited_region, SA1B_EDITED_PARAMS
    with Image.open(job["source"]) as im:
        source = im.convert("RGB")
    src = np.asarray(source)
    shape = src.shape[:2]
    out = {"rid": job["rid"], "kind": job["kind"], "targets": [], "units": {}}
    masks = {}
    for u in job["units"]:
        s, sup = load_mask(u["mask"]), load_mask(u["support"])
        ok = s.shape == shape and sup.shape == shape
        masks[u["uid"]] = (s, sup, ok)
        out["units"][u["uid"]] = {"uid": u["uid"], "shape_ok": ok,
                                  "selection": encode(s) if ok else None, "support": encode(sup) if ok else None}
    groups = defaultdict(list)          # target path -> units edited into it
    for u in job["units"]:
        groups[u.get("target") or job["target"]].append(u["uid"])
    for path, uids in groups.items():
        with Image.open(path) as im:
            target = im.convert("RGB")
        tgt = np.asarray(target)
        info = {"path": path, "uids": uids, "size_ok": tgt.shape == src.shape}
        if info["size_ok"]:
            d = color_difference(source, target, SA1B_EDITED_PARAMS["blur_sigma"])
            union = np.zeros(shape, bool)
            for uid in uids:
                if masks[uid][2]:
                    union |= masks[uid][1]
            diff = np.abs(tgt.astype(np.int16) - src.astype(np.int16)).max(-1)
            outside = ~union
            info.update(outside_changed_frac=round(float((diff[outside] > 0).mean()), 7),
                        outside_changed_gt2_frac=round(float((diff[outside] > 2).mean()), 7),
                        outside_max_diff=int(diff[outside].max()) if outside.any() else 0,
                        outside_strong_frac=round(float((d[outside] > SA1B_EDITED_PARAMS["t_high"]).mean()), 7))
            for uid in uids:
                s, sup, ok = masks[uid]
                if ok:
                    field, _ = sa1b_edited_region(source, target, s, sup, d=d)
                    out["units"][uid]["edited_region"] = field
        out["targets"].append(info)
    out["units"] = list(out["units"].values())
    return out


# ----------------------------------------------------------------------------------------------- workers

class Claims:
    """Shard claims through atomic mkdir on the shared disk, with a heartbeat while a shard is processed."""

    def __init__(self, run, stage, me):
        self.root, self.stage, self.me = run / "claims" / stage, stage, me
        self.root.mkdir(parents=True, exist_ok=True)
        (run / "claims" / "stale").mkdir(parents=True, exist_ok=True)
        self.stale = run / "claims" / "stale"

    def take(self, shard):
        d = self.root / f"{shard:04d}"
        try:
            d.mkdir()
        except FileExistsError:
            beat = d / "heartbeat"
            last = max(beat.stat().st_mtime if beat.exists() else 0.0, d.stat().st_mtime)
            if time.time() - last < STALE:
                return False
            try:
                d.rename(self.stale / f"{self.stage}-{shard:04d}-{int(time.time())}-{self.me}")
                d.mkdir()
            except OSError:
                return False
        (d / "owner").write_text(f"{self.me} {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
        (d / "heartbeat").touch()
        return True

    def beating(self, shard):
        stop = threading.Event()

        def beat():
            while not stop.wait(60):
                try:
                    (self.root / f"{shard:04d}" / "heartbeat").touch()
                except OSError:
                    pass
        threading.Thread(target=beat, daemon=True).start()
        return stop


def work(stage, run=RUN, device="cuda", name=None, limit_shards=0):
    """Process unclaimed shards of ``stage`` until none is left (or ``limit_shards`` are done by this worker)."""
    me = name or f"{socket.gethostname()}-{os.getpid()}"
    info = json.load(open(run / "shards.json"))
    all_jobs = [json.loads(line) for line in open(run / "jobs.jsonl")]
    out_dir = run / stage
    out_dir.mkdir(parents=True, exist_ok=True)
    claims = Claims(run, stage, me)
    codec = None
    if stage == "tokens":
        import torch
        from samtok_edit21.models.codec import load_codec
        from samtok_edit21.models.pipeline import DEFAULT_SAMTOK
        torch.set_num_threads(4)             # two token workers per GPU share the machine's cores
        codec = load_codec(DEFAULT_SAMTOK, device)
    else:
        import cv2
        cv2.setNumThreads(1)                 # many region workers: one thread each
    (run / "logs").mkdir(exist_ok=True)
    log = open(run / "logs" / f"{stage}-{me}.log", "a")
    done = 0
    for shard, (start, end) in enumerate(info["shards"]):
        target = out_dir / f"{shard:04d}.jsonl"
        if target.exists() or not claims.take(shard):
            continue
        stop = claims.beating(shard)
        started, errors, lines = time.time(), 0, []
        try:
            for job in all_jobs[start:end]:
                try:
                    res = token_job(codec, job) if stage == "tokens" else region_job(job)
                except Exception as exc:          # one bad image must not stop the shard; assemble reports it
                    errors += 1
                    res = {"rid": job["rid"], "kind": job["kind"], "error": f"{type(exc).__name__}: {exc}"}
                lines.append(json.dumps(res))
            tmp = out_dir / f".{shard:04d}.{me}.tmp"
            tmp.write_text("\n".join(lines) + "\n")
            os.replace(tmp, target)
        finally:
            stop.set()
        done += 1
        log.write(json.dumps({"shard": shard, "jobs": end - start, "errors": errors,
                              "sec": round(time.time() - started, 1), "time": time.strftime("%H:%M:%S")}) + "\n")
        log.flush()
        if limit_shards and done >= limit_shards:
            break
    return done


# ----------------------------------------------------------------------------------------------- records

def compact_audit(audit):
    keep = ("decision", "operation_correct", "target_done", "non_target_preserved", "natural", "host_integrity",
            "reference_unique", "location_correct", "audit_mode")
    out = {k: audit[k] for k in keep if k in (audit or {})}
    verifier = (audit or {}).get("verifier")
    if isinstance(verifier, dict):
        out["verifier"] = {"version": verifier.get("version"), "pass": verifier.get("pass")}
    return out


def text_tag(unit, spec):
    """change (attribute.text) / render (add, replace with requested wording) / remove (text-like scope) / None;
    rendered and changed text was read back by the production's OCR check."""
    req = ((unit.get("generation_context") or {}).get("text_verification") or {})
    mode = (req.get("requirement") or {}).get("mode")
    if mode == "exact":
        role = "change" if md.spec_op(spec) == "text" else "render"
        return {"role": role, "requested": req["requirement"].get("text"),
                "verdict": "pass" if req.get("decision") == "pass" else "fail",
                "reason": f"production OCR check {req.get('version')}"}
    if spec["operation"] == "remove" and md.TEXT_SCOPE.search(spec.get("scope") or ""):
        return {"role": "remove", "requested": None, "verdict": None, "reason": None}
    return {"role": None, "requested": None, "verdict": None, "reason": None}


def merged_unit(family, unit, spec, tok, reg, audit):
    """One unit in the merge schema; an add whose spot token fails takes its host object as S."""
    site, host = (tok or {}).get("site"), (tok or {}).get("host")
    selection = {"source": "mask", **reg["selection"]} if reg.get("selection") else None
    chosen = site
    if spec["operation"] == "add" and selection is not None:
        selection["source"] = "site"
        if (not site or site["roundtrip"] < rd.MIN_ROUNDTRIP) and host and host["roundtrip"] >= rd.MIN_ROUNDTRIP:
            chosen = host
            selection = {"source": "host", **host["rle"], "site": {**reg["selection"],
                         "roundtrip": site["roundtrip"] if site else None}}
    region = reg.get("edited_region")
    if region and chosen is host and host is not None and region.get("rle_counts"):
        # s_covered against the S used (the host), as the field is read for half-done tests
        changed = rd.decode_rle({"size": region["rle_size"], "counts": region["rle_counts"]})
        h_mask = rd.decode_rle({"size": host["rle"]["rle_size"], "counts": host["rle"]["rle_counts"]})
        region = {**region, "s_covered": round(float((changed & h_mask).sum()) / max(int(h_mask.sum()), 1), 3)}
    out = {"uid": None, "production_uid": unit["unit_id"], "instruction_spec": spec,
           "instruction_ref": unit.get("instruction"), "instruction_noref": unit.get("instruction_noref"),
           "noref_ineligible": None if unit.get("instruction_noref") else (unit.get("blind_ineligible_reason")
                                                                         or "no blind sentence"),
           "mask_semantics": unit.get("mask_semantics"), "flags": [], "audit": compact_audit(audit),
           "selection": selection, "execution_region": {"source": "support", **reg["support"]} if reg.get("support") else None,
           "edited_region": region, "op": md.spec_op(spec),
           "token": md.token_of(chosen) if chosen else None,
           "half_done": md.half_done(spec, region, family), "text": text_tag(unit, spec),
           "object_id": str(unit.get("object_id")), "point": unit.get("point"), "bbox": unit.get("bbox")}
    return out


def record(family, rid, ref, src, target, size, units, extra):
    rec = {"id": f"{family}:{ref}", "family": family, "origin": "sa1b", "source_ref": ref, "group": None,
           "source_image": src, "target_image": target, "size": size, "units": units, "drift": None,
           "loss_region": None, "size_ok": True, "split": "train"}
    for k, u in enumerate(units):
        u["uid"] = f"U{k + 1}"
    rec.update(extra)
    return rec


def load_stage(run, stage):
    out = {}
    for path in sorted((run / stage).glob("*.jsonl")):
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                out[(r["kind"], r["rid"])] = r
    return out


def bad_unit(unit):
    """Why a unit cannot train with its token (the merge's hard exclusions at unit level)."""
    why = []
    if unit["token"] is None:
        why.append("no_token")
    elif unit["token"]["roundtrip"] < rd.MIN_ROUNDTRIP:
        why.append("roundtrip_low")
    if unit["half_done"]:
        why.append("half_done")
    if unit["text"]["verdict"] == "fail":
        why.append("text_fail")
    if unit["edited_region"] is None or unit["selection"] is None or unit["execution_region"] is None:
        why.append("size_mismatch")       # an image or mask that does not match the source, or missing results
    elif not unit["edited_region"].get("rle_counts"):
        why.append("no_change")           # the target equals the source inside the support
    return why


LOCAL = ("size_mismatch", "no_change")   # unit reasons the merge's own checks do not see: carried as prior_exclude


def recompose(source_path, target_path, removed_supports, out_path):
    """The target with the removed units' supports restored to source pixels (the production's own way of
    dropping a failed target from a fused image); returns the number of restored pixels."""
    from PIL import Image
    union = None
    for rle in removed_supports:
        mask = rd.decode_rle({"size": rle["rle_size"], "counts": rle["rle_counts"]})
        union = mask if union is None else union | mask
    if Path(out_path).exists():          # written by an earlier assemble: the same pixels
        return int(union.sum())
    with Image.open(source_path) as im:
        src = np.asarray(im.convert("RGB"))
    with Image.open(target_path) as im:
        tgt = np.asarray(im.convert("RGB")).copy()
    tgt[union] = src[union]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(f".{out_path.name}.{os.getpid()}.tmp.png")
    Image.fromarray(tgt).save(tmp)
    os.replace(tmp, out_path)
    return int(union.sum())


def outside_ok(target_info):
    """A pasted-back target: nothing outside the supports changed by more than decoder rounding."""
    return target_info.get("size_ok") and target_info.get("outside_changed_gt2_frac", 1.0) == 0.0


def _recompose_job(job):
    return recompose(*job)


def assemble(run=RUN, production=PRODUCTION, bench=None, dev=None, workers=32):
    """Merge-schema records of the three families (``records_D12/13/14.jsonl``) and ``assemble.json``."""
    toks, regs = load_stage(run, "tokens"), load_stage(run, "regions")
    bench_ids = bench if bench is not None else {json.loads(l)["source"]["image_id"]
                                                 for l in open(md.DS / "samtok_edit_benchmark_v2/benchmark/cases.jsonl")}
    dev_ids = dev if dev is not None else set(json.load(open(md.R2 / "dev_split.json"))["sa1b_sources"])
    stats = defaultdict(Counter)
    prod = production.name if production.name != "final" else production.parent.name

    def base(kind, g, unit_list, tok, reg):
        spec_units, why = [], {}
        for u in unit_list:
            t = next((x for x in tok.get("units", []) if x["uid"] == u["unit_id"]), None) if tok else None
            r = next((x for x in reg.get("units", []) if x["uid"] == u["unit_id"]), None) if reg else None
            spec = u["instruction_spec"]
            mu = merged_unit(FAMILY[kind], u, spec, t, r or {}, u.get("semantic_audit") or u.get("audit") or {})
            spec_units.append(mu)
            why[u["unit_id"]] = bad_unit(mu)
            stats[kind]["units"] += 1
            if mu["selection"] and mu["selection"]["source"] == "host":
                stats[kind]["add_host"] += 1
            elif u["operation"] == "add":
                stats[kind]["add_site"] += 1
        return spec_units, why

    def prior(source_id):
        return ["benchmark_overlap"] if source_id in bench_ids else []

    files, recompose_jobs = {}, []
    for kind in ("twins", "multi", "singles"):
        files[kind] = open(run / f"records_{FAMILY[kind]}.jsonl.tmp", "w")
    try:
        for line in open(production / FILES["twins"]):
            g = json.loads(line)
            tok, reg = toks.get(("twins", g["twin_id"]), {}), regs.get(("twins", g["twin_id"]), {})
            units, why = base("twins", g, g["members"], tok, reg)
            same = g["members"][0].get("instruction_noref") == g["members"][1].get("instruction_noref")
            refs = [f"{g['twin_id']}-{m['unit_id']}" for m in g["members"]]
            for k, (m, u) in enumerate(zip(g["members"], units)):
                tinfo = next((t for t in reg.get("targets", []) if m["unit_id"] in t["uids"]), {})
                pre = (prior(g["source_id"]) + ([] if outside_ok(tinfo) else ["outside_change"])
                       + [w for w in why[m["unit_id"]] if w in LOCAL])
                rec = record("D12", g["twin_id"], refs[k], g["source_image"], m["edited_image"], tok.get("size"), [u],
                             {"group": f"sha1:{tok['sha1']}" if tok.get("sha1") else None, "dhash": tok.get("dhash"),
                              "split": "dev" if g["source_id"] in dev_ids else "train", "prior_exclude": pre,
                              "sa1b": {"source_id": g["source_id"], "object_id": str(m.get("object_id")),
                                       "operation": g["operation"]},
                              "twin": {"group": g["twin_id"], "partner": f"D12:{refs[1 - k]}", "same_noref": same},
                              "size_ok": bool(tinfo.get("size_ok")) and "size_mismatch" not in why[m["unit_id"]],
                              "production": {"run": prod, "file": FILES["twins"], "id": g["twin_id"],
                                             "unit_id": m["unit_id"]}})
                files["twins"].write(json.dumps(rec, ensure_ascii=False) + "\n")
                stats["twins"]["records"] += 1
                stats["twins"]["bad_units"] += bool(why[m["unit_id"]])
        for line in open(production / FILES["singles"]):
            s = json.loads(line)
            # a salvaged single keeps no generation context; its generation folder does (the OCR text check):
            # generation.json (twin members) or render.json (multi-object steps)
            folder = Path(s["edited_image"]).parent
            for name in ("generation.json", "render.json"):
                if (folder / name).exists():
                    info = json.load(open(folder / name))
                    context = (info.get("row") or {}).get("generation_context") or info.get("generation_context")
                    if context:
                        s["generation_context"] = context
                        stats["singles"]["generation_context"] += 1
                        break
            tok, reg = toks.get(("singles", s["job_id"]), {}), regs.get(("singles", s["job_id"]), {})
            units, why = base("singles", s, [s], tok, reg)
            tinfo = (reg.get("targets") or [{}])[0]
            ref = s["job_id"].replace("/", "-")
            rec = record("D14", s["job_id"], ref, s["source_image"], s["edited_image"], tok.get("size"), units,
                         {"group": f"sha1:{tok['sha1']}" if tok.get("sha1") else None, "dhash": tok.get("dhash"),
                          "split": "dev" if s["source_id"] in dev_ids else "train",
                          "prior_exclude": (prior(s["source_id"]) + ([] if outside_ok(tinfo) else ["outside_change"])
                                            + [w for w in why[s["unit_id"]] if w in LOCAL]),
                          "size_ok": bool(tinfo.get("size_ok")) and "size_mismatch" not in why[s["unit_id"]],
                          "sa1b": {"source_id": s["source_id"], "object_id": str(s.get("object_id")),
                                   "operation": s["operation"]},
                          "salvage": s.get("provenance"),
                          "production": {"run": prod, "file": FILES["singles"], "id": s["job_id"]}})
            files["singles"].write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats["singles"]["records"] += 1
            stats["singles"]["bad_units"] += bool(why[s["unit_id"]])
        for line in open(production / FILES["multi"]):
            g = json.loads(line)
            tok, reg = toks.get(("multi", g["multi_id"]), {}), regs.get(("multi", g["multi_id"]), {})
            units, why = base("multi", g, g["units"], tok, reg)
            tinfo = (reg.get("targets") or [{}])[0]
            pre = prior(g["source_id"]) + ([] if outside_ok(tinfo) else ["outside_change"])
            target, pruned = g["joint_target"], None
            bad = {uid for uid, w in why.items() if w}
            if bad and not pre:
                # components over the units the production kept (it lists the planned ones, some of which it dropped)
                present = {u["unit_id"] for u in g["units"]}
                comps = [[uid for uid in c if uid in present] for c in g["parallel"]["components"]]
                drop = sorted({uid for c in comps if bad & set(c) for uid in c})
                keep = [u for u in units if u["production_uid"] not in drop]
                if keep:
                    supports = [u["execution_region"] for u in units if u["production_uid"] in drop]
                    out_path = run / "recomposed" / f"{g['multi_id'][-2:]}" / f"{g['multi_id']}.png"
                    recompose_jobs.append((g["source_image"], g["joint_target"], supports, out_path))
                    target = str(out_path)
                    pruned = {"units": drop, "reasons": {uid: why[uid] for uid in drop if why[uid]},
                              "kept": [u["production_uid"] for u in keep],
                              "target": "recomposed from the production's joint target"}
                    stats["multi"]["pruned_records"] += 1
                    stats["multi"]["pruned_units"] += len(drop)
                    stats["multi"]["pruned_good_units"] += len([u for u in drop if not why[u]])
                    units = keep
            left = sorted({w for u in units for w in why[u["production_uid"]] if w in LOCAL})
            rec = record("D13", g["multi_id"], g["multi_id"], g["source_image"], target, tok.get("size"), units,
                         {"group": f"sha1:{tok['sha1']}" if tok.get("sha1") else None, "dhash": tok.get("dhash"),
                          "split": "dev" if g["source_id"] in dev_ids else "train", "prior_exclude": pre + left,
                          "size_ok": bool(tinfo.get("size_ok")) and "size_mismatch" not in left,
                          "sa1b": {"source_id": g["source_id"],
                                   "object_ids": [u["object_id"] for u in units]},
                          "multi": {"n_targets_production": len(g["units"]), "operations": g.get("operations"),
                                    "components": g["parallel"]["components"]},
                          "pruned": pruned,
                          "production": {"run": prod, "file": FILES["multi"], "id": g["multi_id"]}})
            files["multi"].write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats["multi"]["records"] += 1
            stats["multi"]["units_kept"] += len(units)
    finally:
        for f in files.values():
            f.close()
    # the new target images of pruned records, in parallel, before the records that point at them are published
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(workers) as pool:
        restored = list(pool.map(_recompose_job, recompose_jobs, chunksize=4))
    stats["multi"]["restored_pixels"] = int(sum(restored))
    for kind in FILES:
        os.replace(run / f"records_{FAMILY[kind]}.jsonl.tmp", run / f"records_{FAMILY[kind]}.jsonl")
    summary = {k: dict(v) for k, v in stats.items()}
    summary["missing"] = {"tokens": sum(1 for j in map(json.loads, open(run / "jobs.jsonl"))
                                        if (j["kind"], j["rid"]) not in toks),
                          "regions": sum(1 for j in map(json.loads, open(run / "jobs.jsonl"))
                                         if (j["kind"], j["rid"]) not in regs)}
    summary["errors"] = {"tokens": sum("error" in r for r in toks.values()),
                         "regions": sum("error" in r for r in regs.values())}
    (run / "assemble.json").write_text(json.dumps(summary, indent=1))
    return summary


# ----------------------------------------------------------------------------------------------- main

def main(argv=None):
    p = argparse.ArgumentParser(prog="samtok-edit sa1b-groups", description=__doc__.split("\n\n")[0])
    p.add_argument("command", choices=("prepare", "work", "assemble", "status"))
    p.add_argument("--run", type=Path, default=RUN)
    p.add_argument("--stage", choices=("tokens", "regions"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--name")
    p.add_argument("--limit-shards", type=int, default=0)
    p.add_argument("--workers", type=int, default=32, help="assemble: processes that write the recomposed targets")
    a = p.parse_args(argv)
    if a.command == "prepare":
        print(json.dumps({k: v for k, v in prepare(a.run).items() if k != "shards"}, indent=1))
    elif a.command == "work":
        print(json.dumps({"stage": a.stage, "shards_done": work(a.stage, a.run, a.device, a.name, a.limit_shards)}))
    elif a.command == "assemble":
        print(json.dumps(assemble(a.run, workers=a.workers), indent=1))
    else:
        info = json.load(open(a.run / "shards.json"))
        n = len(info["shards"])
        print(json.dumps({s: f"{len(list((a.run / s).glob('*.jsonl')))}/{n}" for s in ("tokens", "regions")}))
    return 0


if __name__ == "__main__":
    main()
