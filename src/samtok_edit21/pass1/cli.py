"""Lazy command dispatcher for Pass1 data, training and evaluation."""

from __future__ import annotations

import argparse
import importlib
import sys

COMMANDS = {
    "prepare-pilot": "data.pilot",
    "select-data": "data.selection",
    "materialize-data": "data.materialization",
    "prepare-curriculum": "data.curriculum",
    "train-pilot": "training.pilot",
    "train": "training.formal",
    "infer": "evaluation.inference",
    "score": "evaluation.scoring",
    "summarize": "evaluation.metrics",
    "shards": "evaluation.shards",
    "audit": "evaluation.audit",
    "report-pilot": "evaluation.pilot_report",
}


def main(argv=None):
    """Forward command arguments without importing models for top-level help."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=COMMANDS)
    if not argv or argv[0] in ("-h", "--help"):
        parser.print_help()
        return 0
    command = parser.parse_args(argv[:1]).command
    module = importlib.import_module(f"samtok_edit21.pass1.{COMMANDS[command]}")
    previous = sys.argv
    try:
        sys.argv = [f"samtok-pass1 {command}", *argv[1:]]
        return module.main()
    finally:
        sys.argv = previous


if __name__ == "__main__":
    raise SystemExit(main())
