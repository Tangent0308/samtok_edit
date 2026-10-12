#!/usr/bin/env python3
"""Drop some edit types from a pair manifest to make a training selection (docs/09, section 6.15).

Kept lines are copied byte for byte and in order, so the output is a subset of the input in the same format and
its image paths stay relative to the same dataset root.  A summary next to the output records the input and output
hashes, the dropped types, the code commit and per dataset x type counts.  Existing outputs are never overwritten.

usage: select_types.py --manifest IN.jsonl --drop-types text,action --output OUT.jsonl [--summary OUT.json]
       ("none" in --drop-types means pairs without a single edit type)
"""
import argparse
import hashlib
import json
import subprocess
import time
from collections import Counter
from pathlib import Path


def type_key(value):
    return "none" if value is None else str(value)


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_commit():
    """Commit of this script's checkout, with '+dirty' when the script itself is uncommitted or modified."""
    here = Path(__file__).resolve()
    quiet = {"cwd": here.parent, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=here.parent, text=True,
                                         stderr=subprocess.DEVNULL).strip()
        untracked = subprocess.run(["git", "ls-files", "--error-unmatch", here.name], **quiet).returncode != 0
        modified = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", here.name], **quiet).returncode != 0
    except (OSError, subprocess.CalledProcessError):
        return None
    return commit + ("+dirty" if untracked or modified else "")


def counts_table(counter):
    table = {}
    for (dataset, edit_type), n in sorted(counter.items()):
        table.setdefault(dataset, {})[edit_type] = n
    return table


def select(manifest, drop_types, output, summary_path):
    manifest, output, summary_path = Path(manifest), Path(output), Path(summary_path)
    drop = {t.strip().lower() for t in drop_types if t.strip()}
    if not drop:
        raise ValueError("No edit types to drop")
    for path in (output, summary_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    kept, dropped = Counter(), Counter()
    temporary = output.with_name(f".{output.name}.tmp")
    with manifest.open("rb") as source, temporary.open("wb") as target:
        for line in source:
            row = json.loads(line)
            key = (row["dataset"], type_key(row["type"]))
            if key[1].lower() in drop:
                dropped[key] += 1
            else:
                kept[key] += 1
                target.write(line)
    temporary.replace(output)
    by_type_kept, by_type_dropped = Counter(), Counter()
    for (_, edit_type), n in kept.items():
        by_type_kept[edit_type] += n
    for (_, edit_type), n in dropped.items():
        by_type_dropped[edit_type] += n
    summary = {
        "input": str(manifest.resolve()), "input_sha256": file_hash(manifest),
        "input_count": sum(kept.values()) + sum(dropped.values()),
        "dropped_types": sorted(drop), "output": str(output.resolve()), "output_sha256": file_hash(output),
        "kept_count": sum(kept.values()), "dropped_count": sum(dropped.values()),
        "kept_by_type": dict(sorted(by_type_kept.items())), "dropped_by_type": dict(sorted(by_type_dropped.items())),
        "kept_by_dataset_type": counts_table(kept), "dropped_by_dataset_type": counts_table(dropped),
        "code": "scripts/data_cleaning/select_types.py", "code_commit": code_commit(),
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    temporary = summary_path.with_name(f".{summary_path.name}.tmp")
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(summary_path)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, help="Input manifest, e.g. <check>/manifest_kept.jsonl")
    parser.add_argument("--drop-types", required=True, help="Comma-separated edit types to drop")
    parser.add_argument("--output", required=True, help="Output manifest (must not exist)")
    parser.add_argument("--summary", help="Summary JSON (default: OUTPUT with .summary.json)")
    args = parser.parse_args()
    output = Path(args.output)
    summary = select(args.manifest, args.drop_types.split(","), output,
                     args.summary or output.with_name(output.stem + ".summary.json"))
    print(json.dumps({k: summary[k] for k in ("input_count", "kept_count", "dropped_count", "dropped_by_type")}))


if __name__ == "__main__":
    main()
