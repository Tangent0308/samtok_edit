"""Extract filtered local photographic NHR-Edit pairs from finished parquet shards. One process per shard; idempotent."""
import argparse, io, json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from scipy import ndimage

LOCAL = {"Remove object", "Add object", "Change object", "Change color", "Change material or texture", "Reshape object",
         "Change haircut", "Change beard or moustache", "Remove glasses", "Add glasses"}
PHOTO = {"standard", "dslr", "photo", "realistic", "close-up", "drone", "portrait", "macro", "snapshot", " mm", "wide-angle", "low-angle",
         "realistic shot", "panorama", "overhead shot", "photoreal", "telephoto", "panoramic", "realism", "aerial", "still life", "nighttime",
         "polaroid", "documentary photograph", "magazine photograph", "overhead view", "ultra-wide", "closeup", "birds-eye", "hyperreal",
         "middle-distance shot", "fashion photograph", "drone still"}

def phase(a, b):
    f = np.fft.fft2(a - a.mean()) * np.conj(np.fft.fft2(b - b.mean())); f /= np.abs(f) + 1e-9
    c = np.fft.ifft2(f).real; i = np.unravel_index(np.argmax(c), c.shape)
    dy, dx = [(x if x <= s // 2 else x - s) for x, s in zip(i, c.shape)]
    return float(np.hypot(dx, dy)), float(c.max())

def metrics(src, tgt):
    x = np.asarray(src.convert("RGB"), np.int16); y = np.asarray(tgt.convert("RGB"), np.int16)
    d = np.abs(y - x).mean(2); sd = (y - x).mean(2)
    un = d <= 30
    scale = 512 / max(src.size)
    size = (max(16, round(src.width * scale)), max(16, round(src.height * scale)))
    shift, peak = phase(np.asarray(src.convert("L").resize(size), float), np.asarray(tgt.convert("L").resize(size), float))
    mask = ndimage.binary_closing(ndimage.gaussian_filter(d, 1.0) > 15, iterations=2)
    return {"total30": float((d > 30).mean()), "drift": float(np.median(sd[un])) if un.any() else 99.0, "shift": shift, "peak": peak,
            "mask_frac": float(mask.mean())}, mask

def work(job):
    path, ok_ids, out = job
    out = Path(out); kept = []; seen = 0
    pf = pq.ParquetFile(path)
    for rg in range(pf.num_row_groups):
        ids = pf.read_row_group(rg, columns=["sample_id"]).column(0).to_pylist()
        if not any(i in ok_ids for i in ids): continue
        tb = pf.read_row_group(rg, columns=["sample_id", "source_image", "edited_image", "edit_instruction", "augmented_instructions", "category", "kind", "style", "original_seed"])
        for r in tb.to_pylist():
            sid = r["sample_id"]
            if sid not in ok_ids: continue
            seen += 1
            try:
                sb, tb_ = r["source_image"]["bytes"], r["edited_image"]["bytes"]
                src = Image.open(io.BytesIO(sb)); tgt = Image.open(io.BytesIO(tb_)); src.load(); tgt.load()
                if src.size != tgt.size: continue
                m, mask = metrics(src, tgt)
            except Exception:
                continue
            if not (m["shift"] == 0 and m["peak"] >= 0.3 and abs(m["drift"]) <= 2 and 0.001 <= m["total30"] <= 0.35 and m["mask_frac"] >= 0.001):
                continue
            sub = out / "images" / f"{sid % 256:02x}"; sub.mkdir(parents=True, exist_ok=True)
            ext = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}.get(src.format, "bin")
            ps = sub / f"{sid}_src.{ext}"; pt = sub / f"{sid}_tgt.{ext}"; pm = sub / f"{sid}_mask.png"
            ps.write_bytes(sb); pt.write_bytes(tb_)
            Image.fromarray(mask.astype(np.uint8) * 255).save(pm, optimize=True)
            kept.append({"id": sid, "source": str(ps), "target": str(pt), "mask": str(pm), "prompt": r["edit_instruction"], "augmented": r["augmented_instructions"],
                         "category": r["category"], "kind": r["kind"], "style": r["style"], "seed": r["original_seed"], "size": list(src.size), **m})
    name = Path(path).stem
    (out / "pool_parts").mkdir(exist_ok=True)
    tmp = out / "pool_parts" / f"{name}.tmp"
    tmp.write_text("".join(json.dumps(k, ensure_ascii=False) + "\n" for k in kept)); tmp.replace(out / "pool_parts" / f"{name}.jsonl")
    return name, seen, len(kept)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=12); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(); root = Path(a.root); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ok = set()
    for l in open(root / "metadata_only/metadata.jsonl"):
        r = json.loads(l)
        if r["style"] in PHOTO and all(p.strip() in LOCAL for p in r["category"].split(" and ")):
            ok.add(r["sample_id"])
    print("candidates by metadata:", len(ok), flush=True)
    done = {p.stem for p in (out / "pool_parts").glob("*.jsonl")} if (out / "pool_parts").exists() else set()
    shards = [p for p in sorted(root.glob("data-*.parquet")) if (p.with_suffix(".parquet.ok")).exists() and p.stem not in done]
    if a.limit: shards = shards[:a.limit]
    print("shards to process:", len(shards), flush=True)
    with ProcessPoolExecutor(a.workers) as pool:
        for name, seen, kept in pool.map(work, [(str(p), ok, str(out)) for p in shards]):
            print(f"{name}: candidates {seen}, kept {kept}", flush=True)

if __name__ == "__main__":
    main()
