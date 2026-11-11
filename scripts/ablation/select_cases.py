"""The 200 IF-Bench source images of the ablation (CPU, run once): ``plan.CASES``.

Every image of release 2.6.0 except the 80 of ``ifbench_cases80.txt`` (the development cases of the R2 decisions:
nothing chosen on them is evaluated on them), stratified by the number of units (2-5) in proportion to the remaining
images, in a fixed hash order per stratum (largest-remainder rounding).  Frozen before any ablation output exists.

    python scripts/ablation/select_cases.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plan  # noqa: E402

SALT = "ablation-cases-20261112"
COUNT = 200


def main():
    manifest = plan.IFBENCH_BUNDLE / "repo/data/v2_2.6.0/cases.jsonl"
    cases = [json.loads(line) for line in open(manifest)]
    dev = set(plan.DEV_CASES.read_text().split())          # one line, ids separated by spaces
    if len(cases) != 1084 or len(dev) != 80 or not dev <= {c["id"] for c in cases}:
        raise ValueError("Unexpected manifest or development case list")
    strata = defaultdict(list)
    for case in cases:
        if case["id"] not in dev:
            strata[len(case["units"])].append(case["id"])
    pool = sum(len(v) for v in strata.values())
    quota = {k: COUNT * len(v) / pool for k, v in strata.items()}
    take = {k: int(q) for k, q in quota.items()}
    for k in sorted(quota, key=lambda k: (-(quota[k] - take[k]), k))[:COUNT - sum(take.values())]:
        take[k] += 1
    order = lambda case_id: hashlib.sha256(f"{SALT}:{case_id}".encode()).hexdigest()  # noqa: E731
    chosen = sorted(c for k, ids in strata.items() for c in sorted(ids, key=order)[:take[k]])
    if plan.CASES.exists():
        old = plan.CASES.read_text().split()
        if old != chosen:
            raise SystemExit(f"{plan.CASES} exists with a different selection; it is frozen")
    else:
        plan.CASES.parent.mkdir(parents=True, exist_ok=True)
        plan.CASES.write_text("\n".join(chosen) + "\n")
    units = Counter(len(c["units"]) for c in cases if c["id"] in set(chosen))
    info = {"count": len(chosen), "salt": SALT, "excluded_development_cases": len(dev), "pool": pool,
            "per_units": {k: [units[k], len(strata[k])] for k in sorted(strata)},
            "units_total": sum(len(c["units"]) for c in cases if c["id"] in set(chosen)),
            "manifest": str(manifest)}
    (plan.ROOT / "cases200_stats.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
