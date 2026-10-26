"""``samtok-edit <command>`` / ``python -m samtok_edit21 <command>``: each command is implemented and documented by
its module (``samtok-edit <command> --help``).  Multi-GPU commands run under ``torch.distributed.run``."""
from __future__ import annotations

import argparse
import sys
from importlib import import_module

COMMANDS = {
    "round2-data": ("samtok_edit21.preparation.round2_data",
                    "Round-2 data: development split, mask pool and token caches, training rows, evaluation set"),
    "merge-data": ("samtok_edit21.preparation.merge_data", "Merge the protocol-2.0 edit data into one training set"),
    "round2-train": ("samtok_edit21.training.round2_engine", "Round-2 trainer (torchrun)"),
    "round2-eval": ("samtok_edit21.evaluation.round2_eval", "Development-set probes of an adapter (torchrun)"),
    "round2-report": ("samtok_edit21.evaluation.round2_report", "Paired report of a development-set evaluation"),
}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in COMMANDS:
        return import_module(COMMANDS[argv[0]][0]).main(argv[1:])
    parser = argparse.ArgumentParser(prog="samtok-edit", description=__doc__.split("\n\n")[0])
    subs = parser.add_subparsers(dest="command", required=True)
    for name, (_, summary) in COMMANDS.items():
        subs.add_parser(name, help=summary)
    parser.parse_args(argv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
