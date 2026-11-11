"""Offline preservation/execution metrics of self-distilled targets. Regions are used only for filtering."""
import argparse, json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
ROOT = Path("/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTokEdit_Merged_20261029")
def decode_rle(size, counts):
    """COCO compressed RLE -> bool mask (pure Python decoder)."""
    h, w = size; cnts = []; p = 0; s = counts.encode() if isinstance(counts, str) else counts
    while p < len(s):
        x = 0; k = 0; more = True
        while more:
            c = s[p] - 48; x |= (c & 0x1f) << (5 * k); more = bool(c & 0x20); p += 1; k += 1
            if not more and (c & 0x10): x |= -1 << (5 * k)
        if len(cnts) > 2: x += cnts[-2]
        cnts.append(x)
    arr = np.zeros(h * w, bool); pos = 0; v = False
    for c in cnts:
        if v: arr[pos:pos + c] = True
        pos += c; v = not v
    return arr.reshape((w, h)).T
def region(field, shape):
    if not field or not field.get("rle_counts"): return None
    m = decode_rle(field["rle_size"], field["rle_counts"])
    if m.shape != shape: m = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((shape[1], shape[0]), Image.NEAREST)) > 0
    return m
def phase_shift(a, b):
    f = np.fft.fft2(a - a.mean()) * np.conj(np.fft.fft2(b - b.mean())); f /= np.abs(f) + 1e-9
    c = np.fft.ifft2(f).real; i = np.unravel_index(np.argmax(c), c.shape)
    dy, dx = [(x if x <= s // 2 else x - s) for x, s in zip(i, c.shape)]
    return float(np.hypot(dx, dy)), float(c.max())
def one(job):
    spec, rec_unit, tgt = job
    src = Image.open(ROOT / spec["source_image"]).convert("RGB"); out = Image.open(tgt).convert("RGB")
    W, H = out.size; shape = (src.height, src.width)
    masks = [region(rec_unit.get(k), shape) for k in ("selection", "execution_region", "edited_region")]
    s_m = masks[0]; a = np.zeros(shape, bool)
    for m in masks:
        if m is not None: a |= m
    rs = lambda m: np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)) > 0
    a = rs(a); s_r = rs(s_m)
    x = np.asarray(src.resize((W, H), Image.Resampling.LANCZOS), np.int16); y = np.asarray(out, np.int16)
    d = np.abs(y - x).mean(2); sd = (y - x).mean(2)
    diag = (W * W + H * H) ** 0.5
    outside = ndimage.distance_transform_edt(~a) > 0.02 * diag
    unch = outside & (d <= 30)
    gx = np.asarray(src.convert("L").resize((512, round(512 * H / W))), float) if W >= H else np.asarray(src.convert("L").resize((round(512 * W / H), 512)), float)
    gy = np.asarray(out.convert("L").resize((gx.shape[1], gx.shape[0])), float)
    shift, peak = phase_shift(gx, gy)
    return {"id": spec["id"], "op": spec["op"], "origin": spec["origin"], "area": spec["area"], "outside_frac": float(outside.mean()),
            "out30": float((d[outside] > 30).mean()) if outside.any() else None, "out15": float((d[outside] > 15).mean()) if outside.any() else None,
            "drift": float(np.median(sd[unch])) if unch.any() else None, "mad_unch": float(d[unch].mean()) if unch.any() else None,
            "in15": float((d[s_r] > 15).mean()), "in30": float((d[s_r] > 30).mean()), "total30": float((d > 30).mean()), "shift": shift, "peak": peak}
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args(); d = Path(a.dir)
    specs = {json.loads(l)["id"]: json.loads(l) for l in open(d / "specs.jsonl")}
    done = [json.loads(l) for p in sorted((d / "shards").glob("shard_*.jsonl")) for l in open(p)]
    done = [r for r in done if r["status"] == "completed"]
    want = {r["id"] for r in done}; recs = {}
    with open(ROOT / "records.jsonl") as f:
        for line in f:
            rid = line[8:line.index('"', 8)]
            if rid in want:
                recs[rid] = json.loads(line)["units"][0]
                if len(recs) == len(want): break
    jobs = [(specs[r["id"]], recs[r["id"]], str(d / r["target"])) for r in done]
    with ProcessPoolExecutor(a.workers) as pool:
        rows = list(pool.map(one, jobs, chunksize=4))
    (d / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print("metrics", len(rows))
if __name__ == "__main__":
    main()
