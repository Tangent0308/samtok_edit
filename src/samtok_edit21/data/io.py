"""Small I/O helpers: atomic JSON writes, a canonical JSON hash and the training canvas size."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def row_hash(row):
    return hashlib.sha256(
        json.dumps(row, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def dimensions(image, max_pixels):
    scale = min(1.0, math.sqrt(max_pixels / (image.width * image.height)))
    return max(32, round(image.width * scale / 32) * 32), max(
        32, round(image.height * scale / 32) * 32
    )
