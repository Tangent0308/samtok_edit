"""Read-only byte/hash and near-duplicate checks for a frozen selection."""

import argparse
import concurrent.futures
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--pilot-splits", type=Path, required=True)
    ap.add_argument("--benchmark-manifest", type=Path, required=True)
    ap.add_argument("--benchmark-dataset", type=Path, required=True)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    sources = pq.read_table(a.selection / "selected_sources.parquet").to_pylist()
    pilot = a.pilot_splits
    prior = json.loads(pilot.read_text())
    bench = a.benchmark_manifest
    cases = [json.loads(x) for x in bench.read_text().splitlines()]
    paths = {s["image"] for s in sources} | {
        s["pool_image"] for s in sources if s.get("pool_image")
    }
    paths |= {s["original_image"] for rows in prior.values() for s in rows}
    bench_paths = {
        x["source"]["image_id"]: str(bench.parent / x["source"]["image"]) for x in cases
    }
    if not all(Path(p).exists() for p in bench_paths.values()):
        root = a.benchmark_dataset
        bench_paths = {
            x["source"]["image_id"]: str(root / x["source"]["image"]) for x in cases
        }
    if not all(Path(p).exists() for p in bench_paths.values()):
        raise FileNotFoundError(
            "Benchmark images unavailable for independent near-duplicate audit"
        )
    paths |= set(bench_paths.values())
    start = time.time()

    def one(path):
        raw = Path(path).read_bytes()
        with Image.open(io.BytesIO(raw)) as im:
            size = im.size
            arr = np.asarray(im.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
        bits = (arr[:, 1:] > arr[:, :-1]).flatten()
        h = 0
        for bit in bits:
            h = (h << 1) | int(bit)
        return {
            "path": path,
            "sha1": hashlib.sha1(raw).hexdigest(),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "dhash": h,
            "w": size[0],
            "h": size[1],
            "bytes": len(raw),
        }

    rows = []
    with concurrent.futures.ThreadPoolExecutor(32) as executor:
        for i, x in enumerate(executor.map(one, sorted(paths)), 1):
            rows.append(x)
            if i % 2000 == 0:
                print(
                    json.dumps(
                        {
                            "hashed": i,
                            "total": len(paths),
                            "elapsed_s": round(time.time() - start, 1),
                        }
                    ),
                    flush=True,
                )
    schema = pa.schema(
        [
            ("path", pa.string()),
            ("sha1", pa.string()),
            ("sha256", pa.string()),
            ("dhash", pa.uint64()),
            ("w", pa.int32()),
            ("h", pa.int32()),
            ("bytes", pa.int64()),
        ]
    )
    pq.write_table(
        pa.Table.from_pylist(rows, schema=schema),
        a.output / "image_hashes.parquet",
        compression="zstd",
    )
    by_path = {r["path"]: r for r in rows}
    issues = []
    quarantine = set()
    for s in sources:
        h = by_path[s["image"]]
        reasons = []
        if s["source"].startswith("sha1:") and h["sha1"] != s["source"][5:]:
            reasons.append("merged_sha1_mismatch")
        if (h["w"], h["h"]) != (s["w"], s["h"]):
            reasons.append("source_size_mismatch")
        if s.get("pool_image") and by_path[s["pool_image"]]["sha256"] != h["sha256"]:
            reasons.append("pool_alias_bytes_mismatch")
        if reasons:
            issues.append({"source": s["source"], "reasons": reasons})
            quarantine.add(s["source"])
    train = [s for s in sources if s["split"] == "train"]
    dev = [s for s in sources if s["split"] == "dev"]
    test = [s for s in sources if s["split"] == "test"]

    def near(query, candidates, kind, ban_candidates=False):
        dh = np.array(
            [by_path[s["image"]]["dhash"] for s in candidates], dtype=np.uint64
        )
        for s in query:
            h = by_path[s["image"]]
            dist = np.bitwise_count(dh ^ np.uint64(h["dhash"]))
            hits = np.where(dist <= 6)[0]
            for idx in hits:
                t = candidates[int(idx)]
                if s["source"] == t["source"]:
                    continue
                issues.append(
                    {
                        "kind": kind,
                        "query": s["source"],
                        "candidate": t["source"],
                        "dhash_distance": int(dist[idx]),
                        "same_bytes": h["sha256"] == by_path[t["image"]]["sha256"],
                    }
                )
                quarantine.add(t["source"] if ban_candidates else s["source"])

    near(dev + test, train, "heldout_vs_train", True)
    near(test, dev, "test_vs_dev")
    prior_rows = [
        {"source": "pilot:" + s["group"], "image": s["original_image"]}
        for values in prior.values()
        for s in values
    ]
    near(dev + test, prior_rows, "fresh_holdout_vs_pilot")
    prior_held = [
        {"source": "pilot:" + s["group"], "image": s["original_image"]}
        for k in ["dev", "test"]
        for s in prior[k]
    ]
    near(train, prior_held, "train_vs_pilot_holdout")
    br = [{"source": "benchmark:" + s, "image": p} for s, p in bench_paths.items()]
    near(sources, br, "selection_vs_benchmark")
    summary = {
        "source_images": len(sources),
        "hashed_paths": len(paths),
        "pool_aliases_checked": sum(bool(s.get("pool_image")) for s in sources),
        "algorithm": "PIL grayscale resize 9x8 LANCZOS, adjacent horizontal >, Hamming <=6; conservative near-duplicate quarantine",
        "issues": issues,
        "quarantine_sources": sorted(quarantine),
        "quarantine_count": len(quarantine),
        "elapsed_s": time.time() - start,
        "benchmark_sha256": hashlib.sha256(bench.read_bytes()).hexdigest(),
        "prior_split_sha256": hashlib.sha256(pilot.read_bytes()).hexdigest(),
    }
    (a.output / "source_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in summary.items()
                if k not in ["issues", "quarantine_sources"]
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
