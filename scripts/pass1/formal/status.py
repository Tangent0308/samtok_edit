"""Read-only, compact status for this formal experiment."""

import json
import os
from pathlib import Path


def main():
    r = Path(os.environ["PASS1_FORMAL_ROOT"]).expanduser().resolve()
    s = json.loads((r / "status.json").read_text())
    print("Directory:", r)
    print(
        "State:", s["state"], "| Phase:", s["phase"], "| Updated UTC:", s["updated_utc"]
    )
    if s.get("active_log"):
        print("Active log:", s["active_log"])
    if s.get("error"):
        print("Error:", s["error"])
    progress = s.get("last_materialization_progress") or {}
    if "processed" in progress:
        print(
            "Materialized:", progress["processed"], "| failed:", progress.get("failed")
        )
    if s.get("last_training_update"):
        t = s["last_training_update"]
        print(
            "Training update:",
            t["step"],
            "/",
            t["total_steps"],
            "| stage:",
            t["stage"],
            "| loss:",
            t["loss"],
        )
        print(
            "Real requests seen:",
            t["requests_seen"],
            "| rank0 peak allocated GB:",
            t["peak_gpu_gb_rank0"],
        )
    complete = sorted((r / "runs/formal").glob("step_*/COMPLETE.json"))
    if complete:
        print("Latest complete checkpoint:", complete[-1].parent)
    if (r / "selection.json").exists():
        print(
            "Dev-selected checkpoint:",
            json.loads((r / "selection.json").read_text())["adapter"],
        )
    if (r / "REPORT.md").exists():
        print("Final report:", r / "REPORT.md")


if __name__ == "__main__":
    main()
