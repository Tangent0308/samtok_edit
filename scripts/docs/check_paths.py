#!/usr/bin/env python3
"""Check that the paths quoted in the current docs exist.

The docs quote paths in backticks using a few conventions; each is resolved and tested:

    /abs/path                     absolute
    datasets/… experiments/… experiments2/…     relative to the user root on the shared disk
    V3/…                          the v3 experiment tree          REVIEW/…   the v2 review folder
    src/… scripts/… tests/… docs/… third_party/…  relative to the repository

Paths containing a placeholder (``<…>``, ``…``, ``*``, ``{a,b}``, ``$VAR``) are skipped, and so are paths listed in a
table row or sentence that says the path does not exist (``**否**``, ``不存在``, ``已不存在``, ``本机没有``).
Run it after moving artifacts or editing the docs: ``python scripts/docs/check_paths.py``.  It needs the shared disk.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
USER = Path("/mnt/bn/strategy-mllm-train/user/tanyue")
PREFIXES = {
    "V3/": USER / "experiments2/SAMTokEdit/qwen21_v3",
    "REVIEW/": USER / "experiments2/SAMTokEdit/qwen21_v2/eval/protocol_v2_001/review_v2_20261011",
}
USER_RELATIVE = ("datasets/", "experiments/", "experiments2/")
REPO_RELATIVE = ("src/", "scripts/", "tests/", "docs/", "third_party/")
PLACEHOLDER = re.compile(r"[<>…*{}$]|xxxxx|\.\.\.")
ABSENT = ("**否**", "不存在", "本机没有", "已迁走", "旧路径", "失效")


def resolve(text):
    if text.startswith("/"):
        return Path(text)
    for prefix, base in PREFIXES.items():
        if text.startswith(prefix):
            return base / text[len(prefix):]
    if text.startswith(USER_RELATIVE):
        return USER / text
    if text.startswith(REPO_RELATIVE):
        return ROOT / text
    return None


def main():
    if not USER.is_dir():
        print(f"{USER} is not mounted; nothing checked")
        return 0
    missing, checked = [], 0
    for document in sorted((ROOT / "docs").glob("*.md")):
        for number, line in enumerate(document.read_text(encoding="utf-8").split("\n"), 1):
            if any(marker in line for marker in ABSENT):
                continue
            for quoted in re.findall(r"`([^`\n]+)`", line):
                candidate = quoted.strip().split("::")[0].rstrip("/")
                path = resolve(candidate)
                if path is None or PLACEHOLDER.search(candidate) or " " in candidate:
                    continue
                checked += 1
                if not path.exists():
                    missing.append((document.name, number, quoted))
    for name, number, quoted in missing:
        print(f"missing: {name}:{number}: {quoted}")
    print(f"{checked} paths checked, {len(missing)} missing")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
