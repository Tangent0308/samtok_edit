"""Launch independent GPU workers, preserving every exit code and raw shard."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["infer", "score"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--predictions")
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    gpus = args.gpus.split(",")
    processes = []
    files = []
    for i, gpu in enumerate(gpus):
        env = dict(
            os.environ, CUDA_VISIBLE_DEVICES=gpu, HF_HUB_DISABLE_PROGRESS_BARS="1"
        )
        cmd = [
            sys.executable,
            "-m",
            "samtok_edit21.pass1." + args.mode,
            "--data",
            args.data,
            "--output",
            str(out / f"shard_{i}.jsonl"),
            "--shard",
            str(i),
            "--shards",
            str(len(gpus)),
        ]
        if args.adapter:
            cmd += ["--adapter", args.adapter]
        if args.predictions:
            cmd += ["--predictions", args.predictions]
        f = (out / f"worker_{i}.log").open("w")
        files.append(f)
        processes.append(
            subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
        )
    while any(p.poll() is None for p in processes):
        failed = [
            (i, p.returncode)
            for i, p in enumerate(processes)
            if p.poll() not in (None, 0)
        ]
        if failed:
            for p in processes:
                if p.poll() is None:
                    p.terminate()
            for p in processes:
                p.wait()
            raise RuntimeError(f"Worker failures: {failed}; inspect {out}")
        time.sleep(1)
    for f in files:
        f.close()
    if any(p.returncode for p in processes):
        raise RuntimeError([p.returncode for p in processes])
    rows = []
    for i in range(len(gpus)):
        rows.extend(
            json.loads(x) for x in (out / f"shard_{i}.jsonl").read_text().splitlines()
        )
    expected = [json.loads(x)["id"] for x in Path(args.data).read_text().splitlines()]
    if len(rows) != len(expected) or {x["id"] for x in rows} != set(expected):
        raise RuntimeError("Missing/duplicate request")
    rows.sort(key=lambda x: x["id"])
    (out / "combined.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows)
    )
    print(
        json.dumps(
            {
                "mode": args.mode,
                "requests": len(rows),
                "output": str(out / "combined.jsonl"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
