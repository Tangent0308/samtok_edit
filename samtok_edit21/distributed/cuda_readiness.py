"""Bounded CUDA startup probes, each in a fresh interpreter after a failed init."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from samtok_edit21.distributed.cluster import atomic_json


def probe(expected):
    import torch

    print(json.dumps({"hostname": socket.gethostname(), "torch": torch.__version__,
                      "cuda_runtime": torch.version.cuda,
                      "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}), flush=True)
    # Force runtime initialization; nvidia-smi/NVML enumeration alone is insufficient.
    torch.cuda.init()
    count = torch.cuda.device_count()
    if count != expected:
        raise RuntimeError(f"Expected {expected} visible GPUs, found {count}")
    for device in range(count):
        with torch.cuda.device(device):
            value = torch.ones(1, device=f"cuda:{device}")
            if value.sum().item() != 1:
                raise RuntimeError(f"CUDA computation failed on GPU {device}")
            torch.cuda.synchronize()
            del value
    print(f"CUDA_READY: allocation, computation and synchronization passed on {count} GPUs", flush=True)


def diagnostics(output, label):
    """Capture bounded, read-only GPU diagnostics without dumping environment secrets."""
    path = output / f"nvidia-smi-{label}.log"
    with path.open("w") as stream:
        try:
            subprocess.run(["nvidia-smi", "-q"], stdout=stream, stderr=subprocess.STDOUT,
                           timeout=15, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            stream.write(str(exc) + "\n")


def wait_for_cuda(output, expected=8, timeout=600, interval=15, probe_command=None):
    if timeout <= 0 or interval <= 0 or expected <= 0:
        raise ValueError("CUDA timeout, interval and GPU count must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    command = probe_command or [sys.executable, "-m", "samtok_edit21.cuda_readiness",
                                "--probe", "--expected", str(expected)]
    started = time.monotonic()
    report = {"hostname": socket.gethostname(), "node_rank": os.environ.get("ARNOLD_ID"),
              "expected_gpus": expected, "timeout_seconds": timeout, "attempts": [], "ready": False}
    while True:
        attempt = len(report["attempts"]) + 1
        path = output / f"attempt-{attempt:03d}.log"
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0:
            break
        with path.open("w") as stream:
            try:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                        timeout=min(60, remaining), check=False)
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = 124
                stream.write("CUDA probe timed out\n")
        detail = path.read_text(errors="replace")
        print(detail, end="", flush=True)
        report["attempts"].append({"attempt": attempt, "exit_code": code,
                                   "log": str(path), "time": time.time()})
        report["ready"] = code == 0
        atomic_json(output / "readiness.json", report)
        if code == 0:
            return report
        if attempt == 1:
            diagnostics(output, "first-failure")
        if code == 124 and time.monotonic() - started >= timeout:
            break
        # Retry the observed not-yet-initialized condition, not version/count/OOM errors.
        transient = "system not yet initialized" in detail.lower() or "Error 802" in detail
        if not transient:
            raise RuntimeError(f"CUDA preflight failed (exit {code}); see {path}")
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0:
            break
        print(f"CUDA is not ready; retrying in a fresh process in {min(interval, remaining):.1f}s "
              f"({remaining:.1f}s remaining).", flush=True)
        time.sleep(min(interval, remaining))
    diagnostics(output, "timeout")
    raise TimeoutError(f"CUDA did not become ready within {timeout}s; see {output / 'readiness.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--expected", type=int, default=8)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--interval", type=float, default=15)
    args = parser.parse_args()
    if args.probe:
        probe(args.expected)
    else:
        if args.output is None:
            parser.error("--output is required unless --probe is set")
        try:
            wait_for_cuda(args.output, args.expected, args.timeout, args.interval)
        except Exception as exc:
            failure = args.output / "failure.json"
            atomic_json(failure, {"error": str(exc), "type": type(exc).__name__,
                                  "hostname": socket.gethostname(), "time": time.time()})
            # Publish the detailed initial cause before the shell's fallback ERR trap.
            if os.environ.get("SAMTOK_RUN_ROOT") and os.environ.get("ARNOLD_ID"):
                target = Path(os.environ["SAMTOK_RUN_ROOT"]) / "nodes" / os.environ["ARNOLD_ID"] / "failure.json"
                target.parent.mkdir(parents=True, exist_ok=True)
                from samtok_edit21.distributed.cluster import record_failure
                record_failure(target, json.loads(failure.read_text()))
            raise


if __name__ == "__main__":
    main()
