#!/usr/bin/env python3
"""Adapt the existing training data to instruction protocol 2.0 (docs/07).  Writes new files only.

  python scripts/data_cleaning/protocol_v2_adapt.py v1     # cleaned v1 pairs -> protocol_v2_blind/
  python scripts/data_cleaning/protocol_v2_adapt.py sa1b   # SA-1B 50k release -> train_protocol_v2.jsonl

Each command refuses to overwrite an existing output and writes a summary next to it.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from multiprocessing import Pool
from pathlib import Path
import sys
import time

from samtok_edit21.preparation import protocol_v2_adapt as pv

USER = Path("/mnt/bn/strategy-mllm-train/user/tanyue")
V1 = USER / "datasets/SAMTokEdit_Cleaned_20261009/protocol_adapt_qwen38_27b_v1"
SA1B = USER / "datasets/sa1b_samtok_edit_50k_20261014/release"


def read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_new(path, rows):
    if path.exists():
        sys.exit(f"{path} exists; refusing to overwrite")
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.rename(path)


def summarize(rows, started, extra):
    def op(r):
        return r["instruction_spec"]["operation"]
    return {"adapter_version": pv.ADAPTER_VERSION, "protocol_sha256": pv.protocol_hash(),
            "protocol_source": pv.PROTOCOL_SOURCE, "rows": len(rows),
            "with_noref": sum(r["instruction_noref"] is not None for r in rows),
            "noref_ineligible": dict(Counter(r["noref_ineligible"] for r in rows if r["noref_ineligible"])),
            "changes": dict(Counter(r["protocol_v2_change"] for r in rows if r["protocol_v2_change"])),
            "by_operation": dict(Counter(op(r) for r in rows)),
            "noref_ineligible_by_operation": dict(Counter(op(r) for r in rows if r["noref_ineligible"])),
            "seconds": round(time.time() - started, 1), **extra}


def run_v1(args):
    out_dir = V1 / "protocol_v2_blind"
    out_dir.mkdir(exist_ok=True)
    started = time.time()
    source = V1 / "manifest_protocol.jsonl"
    with Pool(args.workers) as pool:
        rows = pool.map(pv.adapt_v1, read(source), chunksize=64)
    write_new(out_dir / "manifest_protocol_v2.jsonl", rows)
    summary = summarize(rows, started, {"input": str(source)})
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


def _sa1b(row):
    try:
        return pv.adapt_sa1b(row, SA1B)
    except Exception as error:  # surface the failing record, never drop it silently
        raise RuntimeError(f"{row['sample_id']}: {error}") from error


def run_sa1b(args):
    target = SA1B / "annotations/train_protocol_v2.jsonl"
    if target.exists():
        sys.exit(f"{target} exists; refusing to overwrite")
    started = time.time()
    source = SA1B / "annotations/train.jsonl"
    rows = read(source)
    with Pool(args.workers) as pool:
        out = []
        for i, row in enumerate(pool.imap(_sa1b, rows, chunksize=8)):
            out.append(row)
            if i % 5000 == 0:
                print(f"{i}/{len(rows)} {time.time() - started:.0f}s", flush=True)
    write_new(target, out)
    summary = summarize(out, started, {"input": str(source), "new_masks": "masks/region_v2/"})
    (SA1B / "protocol_v2_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", choices=["v1", "sa1b"])
    p.add_argument("--workers", type=int, default=48)
    args = p.parse_args()
    (run_v1 if args.dataset == "v1" else run_sa1b)(args)


if __name__ == "__main__":
    main()
