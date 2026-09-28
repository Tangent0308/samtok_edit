"""Eager rank-zero W&B tracking; failures are broadcast before the next collective."""
from __future__ import annotations

import os
from pathlib import Path

from accelerate.utils import broadcast_object_list

from .data import write_json


class TrainingTracker:
    def __init__(self, args, accelerator, plan):
        self.args, self.accelerator, self.run = args, accelerator, None
        if args.wandb_mode == "disabled":
            return

        def start():
            # Required by the internal byted-wandb client in ARNOLD containers.
            os.environ.setdefault("WANDB_DISABLE_SERVICE", "true")
            os.environ.setdefault("WANDB_START_METHOD", "thread")
            import wandb
            directory = Path(args.output) / "tracking"
            directory.mkdir(parents=True, exist_ok=True)
            self.run = wandb.init(
                project=args.wandb_project, entity=args.wandb_entity,
                name=args.wandb_name, id=args.wandb_id,
                mode=args.wandb_mode, dir=str(directory), resume="never",
                config={"training": vars(args), "plan": plan},
                settings=wandb.Settings(start_method="thread", init_timeout=180),
            )
            if self.run is None:
                raise RuntimeError("W&B initialization returned no run")
            write_json(Path(args.output) / "wandb.json", {
                "id": self.run.id, "name": self.run.name, "mode": args.wandb_mode,
                "project": args.wandb_project, "entity": args.wandb_entity,
                "url": self.run.url if args.wandb_mode == "online" else None,
                "status": "initialized",
            })
        self._collective(start)

    def _collective(self, function):
        error = [None]
        if self.accelerator.is_main_process:
            try:
                function()
            except Exception as exc:
                import traceback
                traceback.print_exc()
                # Don't include environment variables or credentials in diagnostics.
                error[0] = f"W&B operation failed ({type(exc).__name__}); see rank-zero W&B log"
        broadcast_object_list(error)
        if error[0]:
            raise RuntimeError(error[0])

    def log(self, entry, learning_rate):
        if self.args.wandb_mode == "disabled":
            return
        def log():
            metrics = {f"train/{k}": v for k, v in entry["metrics"].items()}
            metrics.update({f"count/{k}": v for k, v in entry["counts"].items()})
            metrics.update({f"branch/{k}": v for k, v in entry["branches"].items()})
            metrics.update({"train/lr": learning_rate, "train/skipped": int(entry["skipped"]),
                            "train/world_size": self.accelerator.num_processes,
                            "train/samples": entry["samples"]})
            self.run.log(metrics, step=entry["optimizer_step"])
        self._collective(log)

    def finish(self):
        if self.args.wandb_mode == "disabled":
            return
        def finish():
            import json
            self.run.finish()
            path = Path(self.args.output) / "wandb.json"
            value = json.loads(path.read_text())
            value["status"] = "finished"
            write_json(path, value)
        self._collective(finish)
