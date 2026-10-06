"""Aggregate pair_v2 judge records for v2 runs.

Groups by method x setting x compiled edit type x v2 split and reports n,
mean edit (E) / preservation (P) / quality (Q) and the strict-success rate.
``--compare A B`` adds a paired, case-level bootstrap of the strict-rate
difference per setting (the plan's selection rule needs gains beyond seed
noise, so compare seeds of B0 with the same tool).
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def load_records(directory, variant="pair_v2"):
    records = []
    for path in Path(directory).glob("*.json"):
        record = json.loads(path.read_text())
        if record.get("variant") == variant and record.get("status") == "ok" and record.get("scores"):
            records.append(record)
    return records


def summarize(rows):
    n = len(rows)
    if not n:
        return {"n": 0}
    mean = lambda key: sum(r[key] for r in rows if r[key] is not None) / max(1, sum(r[key] is not None for r in rows))
    return {"n": n, "E": round(mean("edit"), 3), "P": round(mean("preservation"), 3),
            "Q": round(mean("quality"), 3), "strict": round(sum(bool(r["strict_success"]) for r in rows) / n, 3)}


def paired_bootstrap(a, b, samples=2000, seed=0):
    """Strict-rate difference b - a over shared cases, 95% interval."""
    shared = sorted(set(a) & set(b))
    if not shared:
        return None
    diffs = [float(b[c]) - float(a[c]) for c in shared]
    rng = random.Random(seed)
    stats = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(samples))
    return {"cases": len(shared), "diff": round(sum(diffs) / len(diffs), 3),
            "ci95": [round(stats[int(0.025 * samples)], 3), round(stats[int(0.975 * samples) - 1], 3)]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--records", required=True, help="Judge run records directory")
    parser.add_argument("--compiled", required=True)
    parser.add_argument("--output", required=True, help="report.json (a .md table is written next to it)")
    parser.add_argument("--compare", nargs=2, action="append", default=[], metavar=("A", "B"))
    args = parser.parse_args(argv)
    types = {e["case_id"]: e["edit_type"] for e in map(json.loads, Path(args.compiled).read_text().splitlines()) if e}
    groups, strict = defaultdict(list), defaultdict(dict)
    for record in load_records(args.records):
        sample, scores = record["sample"], record["scores"]
        row = {**scores, "case_id": sample["case_id"]}
        keys = [(sample["method"], sample["setting"], "all", sample.get("v2_split", "all")),
                (sample["method"], sample["setting"], types.get(sample["case_id"], "?"), sample.get("v2_split", "all"))]
        for method, setting, edit_type, split in keys:
            groups[(method, setting, edit_type, split)].append(row)
            groups[(method, setting, edit_type, "all")].append(row)
        strict[(sample["method"], sample["setting"])][sample["case_id"]] = bool(scores["strict_success"])
    table = {"|".join(key): summarize(rows) for key, rows in sorted(groups.items())}
    report = {"groups": table, "comparisons": {}}
    for a, b in args.compare:
        for (method, setting), cases in strict.items():
            if method == a and (b, setting) in strict:
                report["comparisons"][f"{b} - {a} | {setting}"] = paired_bootstrap(cases, strict[(b, setting)])
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    lines = ["| method | setting | type | split | n | E | P | Q | strict |", "|---|---|---|---|---|---|---|---|---|"]
    for key, value in table.items():
        method, setting, edit_type, split = key.split("|")
        if value["n"]:
            lines.append(f"| {method} | {setting} | {edit_type} | {split} | {value['n']} | {value['E']} | "
                         f"{value['P']} | {value['Q']} | {value['strict']} |")
    if report["comparisons"]:
        lines += ["", "| comparison | cases | strict diff | 95% CI |", "|---|---|---|---|"]
        lines += [f"| {k} | {v['cases']} | {v['diff']} | {v['ci95']} |" for k, v in report["comparisons"].items() if v]
    Path(args.output).with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
