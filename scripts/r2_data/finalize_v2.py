"""Self-distillation v2 targets: whole-image base-model edit inside the edit region, source restored outside it.

For every completed generation (pilot / ext / v2 batches) not yet finalized:
  composite = alpha * raw + (1 - alpha) * source, alpha = Gaussian-feathered restore region R
  R = pixels within 2% of the image diagonal of (selection | execution_region | edited_region)
Gates (on the raw output, before restore):
  phase-correlation shift == 0 and peak >= 0.3      (no re-framing)
  outside-R change (|diff| > 30) <= 5%              (the edit happened in the region, not elsewhere)
  |median drift| of unchanged outside pixels <= 3 and |drift in the ring just outside R| <= 3  (no seam after restore)
  in-region change >= 0.3 (add: >= 0.15 over the union region); total change <= 40%
  colour attribute edits: detail ratio (Laplacian std composite / source inside the selection) >= 0.7  (not flat)
Writes v2_final/{targets,masks}/<hash[:2]>/<hash>.png, metrics.jsonl (every item), pool_token.jsonl and pool_plain.jsonl
(accepted items).  Incremental: items already in metrics.jsonl are skipped.

usage: finalize_v2.py --batches <dir> [<dir> ...] --out <v2_final dir> [--workers 32] [--limit N]
"""
import argparse, hashlib, json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from distill_metrics import region, phase_shift  # noqa: E402

ROOT = Path("/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTokEdit_Merged_20261029")


def one(job):
    spec, unit, raw_path, out = job
    out = Path(out)
    rid = spec["id"]
    h = hashlib.sha1(rid.encode()).hexdigest()
    rec = {"id": rid, "op": spec["op"], "origin": spec["origin"], "family": spec["family"]}
    try:
        raw = Image.open(raw_path).convert("RGB")
        W, H = raw.size
        src0 = Image.open(ROOT / spec["source_image"])
        shape = (src0.height, src0.width)
        src = src0.convert("RGB").resize((W, H), Image.Resampling.LANCZOS)
        rs = lambda m: np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)) > 0
        masks = {k: region(unit.get(k), shape) for k in ("selection", "execution_region", "edited_region")}
        union = np.zeros((H, W), bool)
        for m in masks.values():
            if m is not None:
                union |= rs(m)
        sel = rs(masks["selection"]) if masks["selection"] is not None else union
        if union.sum() < 100:
            return rec | {"status": "no_region"}
        diag = (W * W + H * H) ** 0.5
        dist = ndimage.distance_transform_edt(~union)
        R = dist <= 0.02 * diag
        ring = (dist > 0.02 * diag) & (dist <= 0.04 * diag)
        x = np.asarray(src, np.int16); y = np.asarray(raw, np.int16)
        d = np.abs(y - x).mean(2); sd = (y - x).mean(2)
        outside = ~R
        unch = outside & (d <= 30)
        ring_u = ring & (d <= 30)
        gx = np.asarray(src.convert("L").resize((512, max(16, round(512 * H / W))) if W >= H else (max(16, round(512 * W / H)), 512)), float)
        gy = np.asarray(raw.convert("L").resize((gx.shape[1], gx.shape[0])), float)
        shift, peak = phase_shift(gx, gy)
        in_mask = union if spec["op"] == "add" else sel
        m_in = float((d[in_mask] > 30).mean()) if in_mask.any() else 0.0
        rec.update(out30=float((d[outside] > 30).mean()) if outside.any() else 0.0,
                   drift=float(np.median(sd[unch])) if unch.any() else 99.0,
                   ring_drift=float(np.median(sd[ring_u])) if ring_u.any() else 0.0,
                   in30=m_in, total30=float((d > 30).mean()), shift=shift, peak=peak, area=float(union.mean()))
        alpha = np.clip(ndimage.gaussian_filter(R.astype(np.float32), 0.005 * diag), 0, 1)[..., None]
        comp = (alpha * y + (1 - alpha) * x).round().clip(0, 255).astype(np.uint8)
        colour = spec["op"] == "attribute" and any(p in ("color", "tint") for p in (spec.get("property") or []))
        detail = None
        if colour and sel.sum() >= 200:
            lc = ndimage.laplace(np.asarray(Image.fromarray(comp).convert("L"), np.float32))
            ls = ndimage.laplace(np.asarray(src.convert("L"), np.float32))
            detail = float(lc[sel].std() / max(ls[sel].std(), 1e-3))
        rec["detail"] = detail
        gates = {"aligned": shift == 0 and peak >= 0.3, "local": rec["out30"] <= 0.05,
                 "no_seam": abs(rec["drift"]) <= 3 and abs(rec["ring_drift"]) <= 3,
                 "executed": m_in >= (0.15 if spec["op"] == "add" else 0.3), "bounded": rec["total30"] <= 0.40,
                 "not_flat": detail is None or detail >= 0.7}
        rec["gates"] = gates
        rec["accepted"] = all(gates.values())
        if rec["accepted"]:
            tp = out / "targets" / h[:2] / f"{h}.png"; mp = out / "masks" / h[:2] / f"{h}.png"
            tp.parent.mkdir(parents=True, exist_ok=True); mp.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(comp).save(tp)
            dc = ndimage.gaussian_filter(np.abs(comp.astype(np.int16) - x).mean(2), 1.0)
            change = ndimage.binary_closing((dc > 15) & R, iterations=2)
            Image.fromarray(change.astype(np.uint8) * 255).save(mp, optimize=True)
            rec.update(target=str(tp), mask=str(mp), change_frac=float(change.mean()))
        rec["status"] = "ok"
        return rec
    except Exception as e:  # noqa: BLE001
        return rec | {"status": "error", "error": repr(e)[:300]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    mpath = out / "metrics.jsonl"
    seen = {json.loads(l)["id"] for l in open(mpath)} if mpath.exists() else set()
    todo = {}
    for b in map(Path, a.batches):
        specs = {json.loads(l)["id"]: json.loads(l) for l in open(b / "specs.jsonl")}
        for p in sorted((b / "shards").glob("shard_*.jsonl")):
            for l in open(p):
                r = json.loads(l)
                if r.get("status") == "completed" and r["id"] not in seen and r["id"] in specs:
                    todo[r["id"]] = (specs[r["id"]], str(b / r["target"]))
    ids = sorted(todo)[: a.limit] if a.limit else sorted(todo)
    want = set(ids); units = {}
    if want:
        with open(ROOT / "records.jsonl") as f:
            for line in f:
                rid = line[8:line.index('"', 8)]
                if rid in want:
                    units[rid] = json.loads(line)["units"][0]
                    if len(units) == len(want):
                        break
    jobs = [(todo[i][0], units[i], todo[i][1], str(out)) for i in ids if i in units]
    print(f"to finalize: {len(jobs)} (already done {len(seen)})", flush=True)
    n_ok = 0
    with ProcessPoolExecutor(a.workers) as pool, open(mpath, "a") as fh:
        for k, rec in enumerate(pool.map(one, jobs, chunksize=8)):
            fh.write(json.dumps(rec) + "\n")
            n_ok += bool(rec.get("accepted"))
            if k % 2000 == 0:
                fh.flush(); print(f"{k}/{len(jobs)} accepted so far {n_ok}", flush=True)
    # rebuild pools from every accepted item
    rows = [json.loads(l) for l in open(mpath)]
    acc = [r for r in rows if r.get("accepted")]
    specs_all = {}
    for b in map(Path, a.batches):
        for l in open(b / "specs.jsonl"):
            s = json.loads(l); specs_all[s["id"]] = s
    need = {r["id"] for r in acc}; meta = {}
    with open(ROOT / "records.jsonl") as f:
        for line in f:
            rid = line[8:line.index('"', 8)]
            if rid in need:
                rr = json.loads(line); u = rr["units"][0]
                meta[rid] = {"noref": not u.get("noref_ineligible"), "twin": (rr.get("twin") or {}).get("partner")}
                if len(meta) == len(need):
                    break
    with open(out / "pool_token.jsonl", "w") as ft, open(out / "pool_plain.jsonl", "w") as fp:
        for r in acc:
            s = specs_all[r["id"]]; m = meta.get(r["id"], {})
            ft.write(json.dumps({"id": r["id"], "target": r["target"], "mask": r["mask"], "noref": m.get("noref", True),
                                 "twin": m.get("twin"), "family": r["family"], "op": r["op"], "group": s["group"]}) + "\n")
            fp.write(json.dumps({"id": "sd:" + r["id"], "source": str(ROOT / s["source_image"]), "target": r["target"],
                                 "mask": r["mask"], "prompt": s["prompt"], "category": r["op"], "group": s["group"]}, ensure_ascii=False) + "\n")
    print(f"metrics {len(rows)}, accepted {len(acc)} ({len(acc) / max(len(rows), 1):.3f})", flush=True)


if __name__ == "__main__":
    main()
