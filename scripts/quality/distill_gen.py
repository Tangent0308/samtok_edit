"""Generate plain edit targets with the original Qwen-Image-2.1 (no adapter, no paste-back)."""
import argparse, hashlib, json, time, traceback
from pathlib import Path
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--shards", type=int, default=1); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import torch
    from PIL import Image
    from samtok_edit21.data.io import dimensions
    from samtok_edit21.models.pipeline import load_pipeline, edit
    root = Path("/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTokEdit_Merged_20261029")
    qwen = "/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen-Image-2.1"
    specs = [json.loads(l) for l in open(a.specs)][a.shard::a.shards]
    out = Path(a.out); (out / "shards").mkdir(parents=True, exist_ok=True)
    log = open(out / "shards" / f"shard_{a.shard:02d}.jsonl", "a")
    done = {json.loads(l)["id"] for l in open(log.name)} if Path(log.name).stat().st_size else set()
    pipe = load_pipeline(qwen, None, device="cuda"); pipe.eval()
    for s in specs:
        if s["id"] in done: continue
        name = hashlib.sha1(s["id"].encode()).hexdigest()
        target = out / "targets" / name[:2] / f"{name}.png"; target.parent.mkdir(parents=True, exist_ok=True)
        tick = time.monotonic(); row = {"id": s["id"], "target": str(target.relative_to(out)), "status": "failed"}
        try:
            image = Image.open(root / s["source_image"]).convert("RGBA")
            w, h = dimensions(image, 1024 * 1024)
            torch._dynamo.reset(); torch.manual_seed(a.seed)
            with torch.inference_mode():
                raw, _ = edit(pipe, s["prompt"], [image], mode="direct", pass2_te="raw", num_inference_steps=40, seed=a.seed,
                              width=w, height=h, use_kv_cache=True, cfg_scale=1.0)
            raw.convert("RGB").save(target)
            row.update(status="completed", width=w, height=h)
        except Exception as e:
            row.update(error=repr(e), traceback=traceback.format_exc())
        row["seconds"] = time.monotonic() - tick
        log.write(json.dumps(row, ensure_ascii=False) + "\n"); log.flush()
        print(f"{s['id']}: {row['status']} {row['seconds']:.1f}s", flush=True)
if __name__ == "__main__":
    main()
