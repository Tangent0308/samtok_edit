"""Paths and serialization confined to this independent Pass1 diagnostic."""

import hashlib
import json
import os
from pathlib import Path

# These scripts reproduce one frozen dev20 study, not an arbitrary test split.
# Require explicit output/formal roots to avoid writing beside installed code.
ROOT = Path(os.environ["PASS1_BENCH_ROOT"]).expanduser().resolve()
FORMAL = Path(os.environ["PASS1_FORMAL_ROOT"]).expanduser().resolve()
REPO = Path(os.environ["PASS1_BENCH_REPO"]).expanduser().resolve()
DATASET = Path(os.environ["PASS1_BENCH_DATASET"]).expanduser().resolve()
BUNDLE = Path(os.environ["PASS1_DEV20_BUNDLE"]).expanduser().resolve()
ADAPTER = Path(os.environ.get("PASS1_ADAPTER", str(FORMAL / "runs/formal/step_2000")))
PREVIEW = Path(os.environ["PASS1_PREVIEW_ROOT"]).expanduser().resolve()


def read_jsonl(path):
    with Path(path).open() as f:
        return [json.loads(x) for x in f if x.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def rle(mask):
    import numpy as np
    from pycocotools import mask as mu

    value = mu.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"size": list(value["size"]), "counts": value["counts"].decode()}
