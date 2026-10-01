"""Eager rank-zero W&B tracking; failures are broadcast before the next collective."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from accelerate.utils import broadcast_object_list

from samtok_edit21.data.io import write_json


def wandb_config(training, plan):
    """Flatten config with bounded keys for ARNOLD's byted-wandb backend.

    The backend flattens nested mappings and rejects the entire config request
    if any resulting key exceeds 64 characters. Keep full paths in a local
    sidecar, and add a stable digest to shortened keys to avoid collisions.
    """
    values, paths = {}, {}

    def visit(value, path):
        if isinstance(value, dict):
            for name, item in value.items():
                visit(item, (*path, str(name)))
            return
        full = ".".join(path)
        key = full if len(full) <= 64 else full[:47] + "_" + hashlib.sha256(full.encode()).hexdigest()[:16]
        if key in paths:
            raise ValueError(f"Duplicate flattened W&B config key: {key}")
        values[key], paths[key] = value, list(path)

    visit(training, ("training",))
    visit(plan, ("plan",))
    return values, paths


def check_wandb_upload_errors(directory):
    """Catch HTTP errors swallowed by byted-wandb's background client.

    A clean local log is not proof of remote delivery, but a recorded rejection
    must never be reported as successful tracking. Do not echo request bodies.
    """
    for path in sorted(Path(directory).glob("wandb/run-*/logs/debug-internal.log")):
        with path.open(errors="replace") as stream:
            for line_number, line in enumerate(stream, 1):
                if "ERROR" in line and "Request failed!" in line:
                    raise RuntimeError(f"W&B upload was rejected; see {path}:{line_number}")


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
            config, config_paths = wandb_config(vars(args), plan)
            write_json(directory / "config-key-map.json", config_paths)
            self.run = wandb.init(
                project=args.wandb_project, entity=args.wandb_entity,
                name=args.wandb_name, id=args.wandb_id,
                mode=args.wandb_mode, dir=str(directory), resume="never",
                config=config,
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
            metrics.update({f"gradient_zero/{k}": v for k, v in entry.get("gradient_zero_reasons", {}).items()})
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
            check_wandb_upload_errors(Path(self.args.output) / "tracking")
            path = Path(self.args.output) / "wandb.json"
            value = json.loads(path.read_text())
            value["status"] = "finished"
            write_json(path, value)
        self._collective(finish)
