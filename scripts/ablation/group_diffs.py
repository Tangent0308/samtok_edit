"""Paired differences of the ablation on groups of IF-Bench settings (after ``report.py``): named-reference settings
(S1, S2-mask) and blind-reference settings (S3-mask, S4-noref) for every component arm and the final-model diagnostics,
on the same bootstrap draws as ``report.py`` (10,000 resamples of the source images, seed 20261027).

    PYTHONPATH=src python scripts/ablation/group_diffs.py   # writes <ROOT>/REPORT/ABLATION_groups.{json,md}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plan  # noqa: E402
import report  # noqa: E402

GROUPS = {"named (S1, S2-mask)": ["S1", "S2-mask"], "blind (S3-mask, S4-noref)": ["S3-mask", "S4-noref"],
          "all four": ["S1", "S2-mask", "S3-mask", "S4-noref"]}
PAIRS = [(f"abl_{arm}", "abl_ref") for arm in plan.ARMS if arm != "ref"] + [("final_g1", "final"),
                                                                             ("final_notoken", "final")]


def main():
    cases = plan.read_cases()
    units = report.unit_counts(cases)
    index = np.random.default_rng(report.SEED).integers(0, len(cases), size=(report.REPLICATES, len(cases)))
    result, lines = {}, ["| 系统 | 参照 | 设置组 | UP 参照 → 系统（Δ [95% 区间]） | JP 参照 → 系统（Δ [95% 区间]） |",
                         "|---|---|---|---|---|"]
    for name, reference in PAIRS:
        own_settings = plan.systems()[name][1]
        for group, settings in GROUPS.items():
            if not set(settings) <= set(own_settings):
                continue
            mine = report.arrays(json.load(open(plan.ROOT / "ifbench" / name / "REPORT.json")), cases, settings)
            theirs = report.arrays(json.load(open(plan.ROOT / "ifbench" / reference / "REPORT.json")), cases, settings)
            om, ot = report.rates(mine, units), report.rates(theirs, units)
            dm, dt = report.rates(mine, units, index), report.rates(theirs, units, index)
            entry, cells = {}, []
            for k, metric in enumerate(("UnitPass", "JointPass")):
                ref = float(np.mean([ot[s][k] for s in settings]))
                value = float(np.mean([om[s][k] - ot[s][k] for s in settings]))
                ci = report.interval(np.mean([dm[s][k] - dt[s][k] for s in settings], axis=0))
                entry[metric] = {"reference": ref, "system": ref + value, "difference": value, "CI95": ci}
                cells.append(f"{100 * ref:.1f} → {100 * (ref + value):.1f}（{100 * value:+.1f} [{100 * ci[0]:+.1f}, "
                             f"{100 * ci[1]:+.1f}]）")
            result.setdefault(name, {"reference": reference})[group] = entry
            lines.append(f"| {name} | {reference} | {group} | {cells[0]} | {cells[1]} |")
    out = plan.ROOT / "REPORT"
    (out / "ABLATION_groups.json").write_text(json.dumps({"seed": report.SEED, "replicates": report.REPLICATES,
                                                          "groups": GROUPS, "results": result}, indent=1))
    text = ("# 消融的分组配对差（与 ABLATION.md 同一组 bootstrap 抽样）\n\n" + "\n".join(lines) + "\n")
    (out / "ABLATION_groups.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
