"""Compare uninterrupted two-step training with saved/resumed training."""

import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file


def main():
    r = Path(os.environ["PASS1_FORMAL_ROOT"]).expanduser().resolve()
    s = json.loads((r / "status.json").read_text())
    if s["phase"] != "materialize_full_data":
        raise RuntimeError(
            "Run only while formal GPU training is waiting for CPU materialization"
        )
    data = r / "data/resume_smoke"
    out = r / "runs/resume_continuous_control"
    cmd = [
        sys.executable,
        "-B",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=8",
        "-m",
        "samtok_edit21.pass1.training.formal",
        "--data",
        str(data / "train.jsonl"),
        "--batch-plan",
        str(data / "batch_plan.json"),
        "--output",
        str(out),
        "--stop-after-step",
        "2",
        "--save-every",
        "1",
    ]
    (r / "logs/resume_continuous_control_command.json").write_text(
        json.dumps(cmd, indent=2) + "\n"
    )
    started = time.time()
    subprocess.run(cmd, check=True, cwd=r)
    a = load_file(str(r / "runs/resume_smoke/step_0002/adapter_model.safetensors"))
    b = load_file(str(out / "step_0002/adapter_model.safetensors"))
    assert set(a) == set(b)
    maximum = max(float((a[k] - b[k]).abs().max()) for k in a)
    error = sum(float((a[k].double() - b[k].double()).square().sum()) for k in a)
    norm = sum(float(b[k].double().square().sum()) for k in a)
    ea = [
        json.loads(x)
        for x in (r / "runs/resume_smoke/train.jsonl").read_text().splitlines()
    ]
    eb = [json.loads(x) for x in (out / "train.jsonl").read_text().splitlines()]
    result = {
        "passed": maximum < 1e-7,
        "bitwise_equal_tensors": sum(torch.equal(a[k], b[k]) for k in a),
        "tensor_count": len(a),
        "max_abs_weight_difference": maximum,
        "relative_l2_difference": (error / norm) ** 0.5,
        "resumed_loss": [x["loss"] for x in ea],
        "continuous_loss": [x["loss"] for x in eb],
        "tolerance": 1e-7,
        "elapsed_s": time.time() - started,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    (r / "reports/resume_equivalence.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result), flush=True)
    if not result["passed"]:
        raise RuntimeError("Resumed weights differ from uninterrupted control")


if __name__ == "__main__":
    main()
