"""Check a real two-update distributed save/resume, before formal initialization."""

import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file


def main():
    r = Path(os.environ["PASS1_FORMAL_ROOT"]).expanduser().resolve()
    run = r / "runs/resume_smoke"
    one = torch.load(
        run / "step_0001/training_state.pt", map_location="cpu", weights_only=False
    )
    two = torch.load(
        run / "step_0002/training_state.pt", map_location="cpu", weights_only=False
    )
    assert (one["step"], two["step"]) == (1, 2)
    assert len(one["rng_states"]) == len(two["rng_states"]) == 8
    assert len(one["optimizer"]["state"]) == len(two["optimizer"]["state"]) == 504
    assert all(int(x["step"]) == 1 for x in one["optimizer"]["state"].values())
    assert all(int(x["step"]) == 2 for x in two["optimizer"]["state"].values())
    assert one["frozen_sha256"] == two["frozen_sha256"]
    assert all(
        torch.equal(one["base_probe"][k], two["base_probe"][k])
        for k in one["base_probe"]
    )
    a = load_file(str(run / "step_0001/adapter_model.safetensors"))
    b = load_file(str(run / "step_0002/adapter_model.safetensors"))
    assert set(a) == set(b) and len(a) == 504
    assert all(v.dtype == torch.float32 and torch.isfinite(v).all() for v in b.values())
    assert any(not torch.equal(a[k], b[k]) for k in a)
    events = [json.loads(x) for x in (run / "train.jsonl").read_text().splitlines()]
    assert [x["step"] for x in events] == [1, 2]
    assert events[1]["attempt_start_step"] == 1 and events[1]["requests_seen"] == 128
    result = {
        "passed": True,
        "optimizer_steps": [1, 2],
        "optimizer_tensors": len(two["optimizer"]["state"]),
        "fp32_lora_tensors": len(b),
        "rng_ranks": 8,
        "frozen_base_and_adapter_off_probe_equal": True,
        "real_requests_used": 128,
        "formal_initialization": "fresh; smoke weights are never used",
    }
    (r / "reports/resume_smoke.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
