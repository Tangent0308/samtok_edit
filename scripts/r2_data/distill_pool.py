"""Accepted self-distilled pairs -> change masks and pools (plain and token forms)."""
import json, hashlib, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
from scipy import ndimage
sys.path.insert(0, str(Path(__file__).parent))
from distill_metrics import region, ROOT
P = Path(sys.argv[1])
def one(job):
    s, u, tgt, mask_path = job
    src = Image.open(ROOT / s["source_image"]).convert("RGB"); out = Image.open(tgt).convert("RGB"); W, H = out.size; shape = (src.height, src.width)
    a = np.zeros(shape, bool)
    for k in ("selection", "execution_region", "edited_region"):
        m = region(u.get(k), shape)
        if m is not None: a |= m
    a = np.asarray(Image.fromarray(a.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)) > 0
    near = ndimage.distance_transform_edt(~a) <= 0.02 * (W * W + H * H) ** 0.5
    x = np.asarray(src.resize((W, H), Image.Resampling.LANCZOS), np.int16); y = np.asarray(out, np.int16)
    d = ndimage.gaussian_filter(np.abs(y - x).mean(2), 1.0)
    mask = ndimage.binary_closing((d > 15) & near, iterations=2)
    Path(mask_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path, optimize=True)
    return float(mask.mean())
def main():
    m = pd.DataFrame([json.loads(l) for l in open(P / "metrics.jsonl")])
    m = m[(m["out30"] <= 0.015) & (m["drift"].abs() <= 3) & (m["shift"] == 0) & (m["peak"] >= 0.3) & (m["in30"] >= 0.3) & (m["total30"] <= 0.35)]
    specs = {json.loads(l)["id"]: json.loads(l) for l in open(P / "specs.jsonl")}
    want = set(m["id"]); recs = {}
    with open(ROOT / "records.jsonl") as f:
        for line in f:
            rid = line[8:line.index('"', 8)]
            if rid in want:
                recs[rid] = json.loads(line)
                if len(recs) == len(want): break
    jobs = []; plain = []; token = []
    for i in m["id"]:
        h = hashlib.sha1(i.encode()).hexdigest(); tgt = str(P / "targets" / h[:2] / f"{h}.png"); mp = str(P / "masks" / h[:2] / f"{h}.png")
        s = specs[i]; u = recs[i]["units"][0]
        jobs.append((s, u, tgt, mp))
        plain.append({"id": f"sd:{i}", "source": str(ROOT / s["source_image"]), "target": tgt, "mask": mp, "prompt": s["prompt"], "category": s["op"], "group": s["group"]})
        token.append({"id": i, "target": tgt, "mask": mp, "noref": not u.get("noref_ineligible")})
    with ProcessPoolExecutor(16) as pool:
        fr = list(pool.map(one, jobs, chunksize=4))
    (P / "pool_plain.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in plain))
    (P / "pool_token.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in token))
    print("accepted", len(plain), "mask area median", round(float(np.median(fr)), 4))
if __name__ == "__main__":
    main()
