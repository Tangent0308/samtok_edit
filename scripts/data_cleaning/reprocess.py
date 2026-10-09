#!/usr/bin/env python3
"""Re-run chosen pairs of an existing cleaned dataset after a rule change and patch decisions.jsonl.

Pairs that were kept before and are dropped now leave image files behind; they are listed in
<out>/logs/orphans_<tag>.txt for removal.  Run `clean_pairs.py finalize` afterwards.

usage: reprocess.py --out DIR --tag NAME [--workers N] (--target-grounded | --ids FILE)
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clean_pairs as cp  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--sources", default=cp.SOURCES)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--png-level", type=int, default=3)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--target-grounded", action="store_true", help="pairs with an instance grounded on the target")
    group.add_argument("--ids")
    args = parser.parse_args()
    out = Path(args.out)
    wanted = set(Path(args.ids).read_text().split()) if args.ids else None
    lines = []
    with open(args.sources) as stream:
        for line in stream:
            if wanted is not None:
                if cp.line_id(line) in wanted:
                    lines.append(line)
            elif '"mapped_from_target": true' in line:
                lines.append(line)
    print(f"[{time.strftime('%H:%M:%S')}] re-processing {len(lines)} pairs with {args.workers} workers", flush=True)
    new = {}
    with Pool(args.workers, initializer=cp._init, initargs=(str(out), args.png_level)) as pool:
        for k, res in enumerate(pool.imap_unordered(cp.process, lines, chunksize=4), 1):
            new[res["id"]] = res
            if k % 2000 == 0 or k == len(lines):
                print(f"[{time.strftime('%H:%M:%S')}] {k}/{len(lines)}", flush=True)
    changes, orphans = Counter(), []
    patched = out / f"decisions.jsonl.{args.tag}.tmp"
    with (out / "decisions.jsonl").open() as old_stream, patched.open("w") as stream:
        for line in old_stream:
            old = json.loads(line)
            res = new.get(old["id"])
            if res is None:
                stream.write(line)
                continue
            if res["decision"] != old["decision"]:
                changes[(old["decision"], res["decision"])] += 1
            for key in ("source_image", "target_image"):
                if old.get(key) and old.get(key) != res.get(key):
                    orphans.append(old[key])
            stream.write(json.dumps(res, ensure_ascii=False) + "\n")
    os.replace(out / "decisions.jsonl", out / f"decisions.before_{args.tag}.jsonl")
    os.replace(patched, out / "decisions.jsonl")
    (out / "logs" / f"orphans_{args.tag}.txt").write_text("".join(f"{p}\n" for p in orphans))
    print(f"[{time.strftime('%H:%M:%S')}] patched {len(new)} records; decision changes: "
          + (", ".join(f"{a}->{b}: {n}" for (a, b), n in sorted(changes.items())) or "none")
          + f"; orphan files listed: {len(orphans)}", flush=True)


if __name__ == "__main__":
    main()
