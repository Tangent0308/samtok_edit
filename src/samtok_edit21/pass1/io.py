"""Small shared I/O helpers independent of model loading."""

import hashlib
from pathlib import Path


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()
