"""Queue safely for idle GPUs, run fixed experiment, and preserve all receipts.

Never terminates unrelated processes. A worker failure stops this run; no
quality-dependent retries, altered prompts, or alternative checkpoints.
"""

import datetime
import fcntl
import os
import subprocess
import sys
import time
import traceback

from common import (
    ADAPTER,
    PREVIEW,
    ROOT,
    Path,
    json,
    read_jsonl,
    write_json,
    write_jsonl,
)


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def status(phase, **extra):
    value = {"at": stamp(), "pid": os.getpid(), "phase": phase, **extra}
    write_json(ROOT / "STATUS.json", value)
    with (ROOT / "logs/history.jsonl").open("a") as f:
        f.write(json.dumps(value) + "\n")
    print(json.dumps(value), flush=True)


def gpu_table():
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return [
        tuple(int(v.strip()) for v in line.split(","))
        for line in raw.strip().splitlines()
    ]


def wait_gpus():
    previous = []
    while True:
        table = gpu_table()
        # The host has a persistent ~2 GiB GPU heartbeat with nonzero utilization.
        # Require ample memory twice; don't kill it or wait forever for 0% util.
        available = [str(i) for i, free, util in table if free >= 70000]
        common = [i for i in available if i in previous]
        if common:
            selected = common[:4]
            status("gpus_acquired", gpus=selected, telemetry=table)
            return selected
        status(
            "waiting_for_free_gpu",
            telemetry=table,
            policy=">=70000 MiB free, two observations 30s apart; coexist with small host heartbeat",
        )
        previous = available
        time.sleep(30)


def parallel(phase, gpus, commands, outputs):
    status(phase, gpus=gpus, workers=len(commands))
    processes, handles = [], []
    try:
        for i, (gpu, cmd) in enumerate(zip(gpus, commands)):
            log = (ROOT / "logs" / f"{phase}_{i}.log").open("x")
            handles.append(log)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            processes.append(
                subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
            )
        while any(p.poll() is None for p in processes):
            failures = [
                (i, p.returncode)
                for i, p in enumerate(processes)
                if p.poll() not in (None, 0)
            ]
            if failures:
                raise RuntimeError(f"{phase}: worker failures {failures}")
            status(
                phase,
                gpus=gpus,
                counts=[sum(1 for _ in p.open()) if p.exists() else 0 for p in outputs],
            )
            time.sleep(30)
        if any(p.returncode for p in processes):
            raise RuntimeError(f"{phase}: {[p.returncode for p in processes]}")
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()  # own workers only
        for p in processes:
            p.wait()
        for f in handles:
            f.close()


def merge(outputs, destination, expected):
    rows = [x for p in outputs for x in read_jsonl(p)]
    assert len(rows) == len(expected) and {x["id"] for x in rows} == set(expected)
    write_jsonl(destination, sorted(rows, key=lambda x: x["id"]))


def main():
    lock = (ROOT / "logs/run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert (ROOT / "provenance/preflight.json").exists()
    assert not (ROOT / "DONE.json").exists()
    (ROOT / "logs/runner.pid").write_text(str(os.getpid()) + "\n")
    status("queued", checkpoint=str(ADAPTER))
    started = stamp()
    gpus = wait_gpus()
    codec_outputs = [ROOT / f"data/codec_{i}.jsonl" for i in range(len(gpus))]
    parallel(
        "codec",
        gpus,
        [
            [
                sys.executable,
                "-B",
                str(Path(__file__).resolve().parent / "codec_prepare.py"),
                "--shard",
                str(i),
                "--shards",
                str(len(gpus)),
            ]
            for i in range(len(gpus))
        ],
        codec_outputs,
    )
    subprocess.run(
        [
            sys.executable,
            "-B",
            str(Path(__file__).resolve().parent / "build_inputs.py"),
        ],
        check=True,
    )
    expected = [x["id"] for x in read_jsonl(ROOT / "data/inference_public.jsonl")]
    for name, adapter in [("step2000", ADAPTER), ("base", None)]:
        # Recheck after releasing our preceding models; never evict another task.
        gpus = wait_gpus()
        directory = ROOT / "eval" / name / "inference"
        directory.mkdir(parents=True, exist_ok=False)
        outputs = [directory / f"shard_{i}.jsonl" for i in range(len(gpus))]
        commands = [
            [
                sys.executable,
                "-B",
                str(Path(__file__).resolve().parent / "infer_seeded.py"),
                "--data",
                str(ROOT / "data/inference_public.jsonl"),
                "--output",
                str(outputs[i]),
                "--shard",
                str(i),
                "--shards",
                str(len(gpus)),
            ]
            + (["--adapter", str(adapter)] if adapter else [])
            for i in range(len(gpus))
        ]
        parallel(f"infer_{name}", gpus, commands, outputs)
        merge(outputs, directory / "combined.jsonl", expected)
        # Score only after generation, from a separate file containing hidden GT.
        gpus = wait_gpus()
        score_dir = ROOT / "eval" / name / "score"
        score_dir.mkdir(parents=True, exist_ok=False)
        outputs = [score_dir / f"shard_{i}.jsonl" for i in range(len(gpus))]
        commands = [
            [
                sys.executable,
                "-B",
                str(Path(__file__).resolve().parent / "score_checked.py"),
                "--data",
                str(ROOT / "data/scoring_truth.jsonl"),
                "--predictions",
                str(directory / "combined.jsonl"),
                "--output",
                str(outputs[i]),
                "--shard",
                str(i),
                "--shards",
                str(len(gpus)),
            ]
            for i in range(len(gpus))
        ]
        parallel(f"score_{name}", gpus, commands, outputs)
        merge(outputs, score_dir / "combined.jsonl", expected)
    status("reporting")
    subprocess.run(
        [sys.executable, "-B", str(Path(__file__).resolve().parent / "report.py")],
        check=True,
    )
    receipt = {
        "started_at": started,
        "completed_at": stamp(),
        "models": ["base", "step2000"],
        "requests_per_model": 200,
        "unique_images": 20,
        "unique_targets": 72,
        "scope": "Pass1 only; no edits or benchmark training/test",
        "report": str(ROOT / "REPORT.md"),
        "visualizations": str(PREVIEW),
    }
    write_json(ROOT / "DONE.json", receipt)
    status("complete", **receipt)


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        write_json(
            ROOT / "FAILED.json",
            {"at": stamp(), "error": repr(e), "traceback": traceback.format_exc()},
        )
        status("failed", error=repr(e))
        raise
