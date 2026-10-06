"""Stratified smoke subset of v2 metadata.

Takes the first ``--per-group`` sources (by sha256 of their id) of every
(dataset, edit_type) group, keeping all of a source's rows, plus a matching
share of rec_ntp rows, so a smoke run covers every row kind, region kind and
dataset while staying small. Writes stage1.jsonl, stage2.jsonl and
metadata_report.json in the same contract as the full data.
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from samtok_edit21.data.io import file_hash, write_json
from samtok_edit21.data.protocol import validate_row


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", required=True, help="Full v2 metadata directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-group", type=int, default=4)
    args = parser.parse_args()
    data, output = Path(args.data), Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Use a fresh output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    files = {name: (data / f"{name}.jsonl").read_text().splitlines() for name in ("stage1", "stage2")}
    groups, rows_by_id, replay = defaultdict(set), defaultdict(list), []
    with (data / "provenance.jsonl").open() as stream:
        for line in stream:
            entry = json.loads(line)
            if entry["conversion"] == "rec_replay":
                replay.append(entry)
                continue
            groups[(entry["dataset"], entry["edit_type"])].add(entry["id"])
            rows_by_id[entry["id"]].append(entry)
    key = lambda uid: hashlib.sha256(uid.encode()).hexdigest()
    selected = {uid for ids in groups.values() for uid in sorted(ids, key=key)[:args.per_group]}
    chosen = [entry for uid in sorted(selected, key=key) for entry in rows_by_id[uid]]
    ntp = sum(entry["sample_type"] == "edit_ntp" for entry in chosen)
    chosen += sorted(replay, key=lambda e: key(e["id"]))[:max(1, round(ntp / 7))]
    counts = Counter()
    for name in ("stage1", "stage2"):
        with (output / f"{name}.jsonl").open("w") as stream:
            for entry in chosen:
                if entry["file"] == name:
                    row = json.loads(files[name][entry["line"]])
                    validate_row(row)
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    counts[f"{name}:{entry['sample_type']}:{entry['edit_type']}"] += 1
    write_json(output / "metadata_report.json", {
        "format": "samtok21-metadata-v2", "smoke_of": str(data.resolve()), "per_group": args.per_group,
        "sources": len(selected), "rows": dict(sorted(counts.items())),
        "stage1_rows": sum(n for k, n in counts.items() if k.startswith("stage1")),
        "stage2_rows": sum(n for k, n in counts.items() if k.startswith("stage2")),
        "stage1_sha256": file_hash(output / "stage1.jsonl"), "stage2_sha256": file_hash(output / "stage2.jsonl"),
        "training_ready": True})
    print(json.dumps({"sources": len(selected), "rows": dict(sorted(counts.items()))}))


if __name__ == "__main__":
    main()
