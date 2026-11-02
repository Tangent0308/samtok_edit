"""Merge a SAMTokEdit IF-Bench run into the benchmark's full report: every method's UnitPass / JointPass, confidence
intervals and paired differences on one set of bootstrap draws.

The statistics are the benchmark's own (``fairness_dev_report.Bootstrap`` / ``rates`` / ``derived``): source-cluster
resampling with one seed and one index table for all methods and settings, and full denominators, so generation
failures count and a missing Judge answer leaves a row incomplete.  Every baseline row is recomputed from its per-case
records and must equal the published row, and the index table must have the published digest; no node percentages are
averaged.  ``--overlay`` runs (a repair run judged on its own, e.g. after the Pass1 format repair) replace the main
run's entries of their requests; the replaced requests are listed in the merged report.  Run with the benchmark
snapshot on PYTHONPATH:

    python -m samtok_edit21.evaluation.ifbench_report --benchmark-report <results root>/report/REPORT.json \\
        --run <run dir> [--overlay <repair run dir>] --output <dir>
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

METHOD = "samtokedit"
METRICS = ("UnitPass", "JointPass")


def baseline_arrays(row, case_ids):
    """Per-source passed units, joint pass and missing flag from a published row."""
    by_case = {c["case_id"]: c for c in row["per_case"]}
    up, jp, unknown = np.zeros(len(case_ids), int), np.zeros(len(case_ids), int), np.zeros(len(case_ids), bool)
    for i, case in enumerate(case_ids):
        entry = by_case[case]
        if entry["UnitPass_count"] is None:
            unknown[i] = True
        else:
            up[i], jp[i] = entry["UnitPass_count"], entry["JointPass"]
    return up, jp, unknown


def method_arrays(requests, setting, case_ids, failures):
    """The same arrays from our run's report; a generation failure is a zero, anything unscored is missing."""
    by_case = {r["case_id"]: r for r in requests if r["setting"] == setting}
    up, jp, unknown = np.zeros(len(case_ids), int), np.zeros(len(case_ids), int), np.zeros(len(case_ids), bool)
    for i, case in enumerate(case_ids):
        entry = by_case[case]
        if entry["status"] == "scored":
            up[i], jp[i] = sum(entry["unit_pass"]), entry["joint_pass"]
        elif entry["status"] not in failures:
            unknown[i] = True
    return up, jp, unknown


def check_published(row, computed):
    """A recomputed baseline row must reproduce the published numbers."""
    for metric in METRICS:
        published = row[metric]
        if (published is None) != (computed[metric] is None) or (
            published is not None and not math.isclose(published, computed[metric], abs_tol=1e-12)
        ):
            raise ValueError(f"{row['model']}/{row['setting']}: {metric} differs from the published report")
        if published is not None:
            pair = zip(row["bootstrap"][metric]["CI95"], computed["bootstrap"][metric]["CI95"])
            if not all(math.isclose(a, b, abs_tol=1e-9) for a, b in pair):
                raise ValueError(f"{row['model']}/{row['setting']}: {metric} interval differs from the published report")


def apply_overlays(ours, overlays):
    """Our report with the entries of the overlay reports' requests replaced, overlays in the given order (a later one
    can supersede an entry whose repair also failed); also the list of replacements.  A scored entry is never replaced
    by an unscored one."""
    if not overlays:
        return ours, []
    by_id = {r["request_id"]: r for r in ours["requests"]}
    replaced = []
    for overlay in overlays:
        if overlay["method"] != ours["method"]:
            raise ValueError("An overlay must come from the same method configuration")
        for entry in overlay["requests"]:
            if entry["request_id"] not in by_id:
                raise ValueError(f"Overlay request {entry['request_id']} is not in the main run")
            if entry["status"] != "scored" and by_id[entry["request_id"]]["status"] == "scored":
                raise ValueError(f"Overlay request {entry['request_id']} has no score to replace a scored entry")
            replaced.append({"request_id": entry["request_id"], "overlay_run": overlay["run"],
                             "replaced_status": by_id[entry["request_id"]]["status"],
                             "replaced_judge_key": by_id[entry["request_id"]]["judge_key"],
                             "judge_key": entry["judge_key"]})
            by_id[entry["request_id"]] = entry
    return dict(ours, requests=list(by_id.values())), replaced


def merge(benchmark, ours, overlays=()):
    from samtok_benchmark.io import digest
    from samtok_benchmark.ifbench.fairness_dev_report import FAILURES, Bootstrap, derived, quantiles, rates

    ours, replaced = apply_overlays(ours, overlays)
    case_ids, settings = benchmark["case_ids"], benchmark["settings"]
    if sorted(ours["case_ids"]) != sorted(case_ids):
        raise ValueError("Our run covers different sources than the benchmark report")
    k_of = {c["case_id"]: c["K"] for c in benchmark["rows"][0]["per_case"]}
    ks = [k_of[case] for case in case_ids]
    boot = Bootstrap(ks)
    if digest(boot.indices.tolist()) != benchmark["bootstrap"]["shared_indices_sha256"]:
        raise ValueError("Bootstrap indices differ from the published report")
    ks = np.asarray(ks)
    rows, draws, arrays = {}, {}, {}
    for row in benchmark["rows"]:
        arrays[row["model"], row["setting"]] = baseline_arrays(row, case_ids)
    for setting in settings:
        arrays[METHOD, setting] = method_arrays(ours["requests"], setting, case_ids, FAILURES)
    methods = [*benchmark["model_order"], METHOD]
    by_row = {(r["model"], r["setting"]): r for r in benchmark["rows"]}
    for method in methods:
        rows[method], draws[method] = {}, {}
        for setting in settings:
            up, jp, unknown = arrays[method, setting]
            computed, distributions = rates(up, jp, ks, unknown, boot)
            if method != METHOD:
                check_published(by_row[method, setting], computed)
            computed["by_K"] = {str(k): rates(up, jp, ks, unknown, boot, ix, selected)[0]
                                for k, (selected, ix) in boot.k_indices.items()}
            rows[method][setting], draws[method][setting] = computed, distributions
    means, paired = {}, {}
    for method in methods:
        means[method] = {}
        for metric in METRICS:
            complete = all(rows[method][s][metric] is not None for s in settings)
            means[method][metric] = None if not complete else {
                "value": float(np.mean([rows[method][s][metric] for s in settings])),
                "bootstrap": quantiles(np.mean([draws[method][s][metric] for s in settings], axis=0))}
    for baseline in benchmark["model_order"]:
        paired[baseline] = {}
        for metric in METRICS:
            usable = [s for s in settings if rows[METHOD][s][metric] is not None and rows[baseline][s][metric] is not None]
            entry = {s: {"value": rows[METHOD][s][metric] - rows[baseline][s][metric],
                         "bootstrap": quantiles(draws[METHOD][s][metric] - draws[baseline][s][metric])} for s in usable}
            if len(usable) == len(settings):
                entry["mean"] = {"value": float(np.mean([entry[s]["value"] for s in settings])),
                                 "bootstrap": quantiles(np.mean([draws[METHOD][s][metric] - draws[baseline][s][metric]
                                                                 for s in settings], axis=0))}
            paired[baseline][metric] = entry
    return {"schema": "samtokedit-ifbench-merged-report-v1", "method": ours["method"], "methods": methods,
            "settings": settings, "source_count": len(case_ids), "units": int(ks.sum()),
            "benchmark_plan_digest": benchmark.get("plan_digest"), "benchmark_source_commit": benchmark.get("source_commit"),
            "bootstrap": benchmark["bootstrap"], "rows": rows, "mean_over_settings": means,
            "derived": {m: derived(rows[m], draws[m]) for m in methods},
            f"paired_{METHOD}_minus_baseline": paired, "replaced_requests": replaced}


def cell(value, interval):
    if value is None:
        return "不完整"
    return f"{100 * value:.1f} [{100 * interval[0]:.1f}, {100 * interval[1]:.1f}]"


def markdown(merged):
    settings = merged["settings"]
    lines = ["# IF-Bench 全量结果：七模型 + SAMTokEdit", "",
             f"{merged['source_count']} 张源图、{merged['units']} 个单元。数字是百分比，方括号是 95% 区间"
             "（benchmark 的源图配对 bootstrap，所有方法共用同一组抽样）。mean 是十种 setting 的算术平均，只作概览。", ""]
    if merged["replaced_requests"]:
        lines += [f"{METHOD} 有 {len(merged['replaced_requests'])} 个请求用修复运行的结果替换："
                  + "、".join(r["request_id"] for r in merged["replaced_requests"]) + "。", ""]
    header = "| 方法 | " + " | ".join(settings) + " | mean |"
    rule = "|---|" + "---:|" * (len(settings) + 1)
    for metric in METRICS:
        lines += [f"## {metric}", "", header, rule]
        for method in merged["methods"]:
            row = merged["rows"][method]
            mean = merged["mean_over_settings"][method][metric]
            cells = [cell(row[s][metric], (row[s]["bootstrap"] or {}).get(metric, {}).get("CI95")) for s in settings]
            cells.append(cell(mean["value"], mean["bootstrap"]["CI95"]) if mean else "不完整")
            lines.append(f"| {method} | " + " | ".join(cells) + " |")
        lines.append("")
    paired = merged[f"paired_{METHOD}_minus_baseline"]
    for metric in METRICS:
        lines += [f"## {METHOD} 减基线：{metric}（百分点）", "", header.replace("方法", "基线"), rule]
        for baseline, entry in paired.items():
            cells = [cell(entry[metric][s]["value"], entry[metric][s]["bootstrap"]["CI95"]) if s in entry[metric] else "不完整"
                     for s in [*settings, "mean"]]
            lines.append(f"| {baseline} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--benchmark-report", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--overlay", type=Path, action="append", default=[], help="repair run whose entries replace the main run's")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    overlays = [json.loads((run / "REPORT.json").read_text()) for run in a.overlay]
    merged = merge(json.loads(a.benchmark_report.read_text()), json.loads((a.run / "REPORT.json").read_text()), overlays)
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "MERGED_REPORT.json").write_text(json.dumps(merged, ensure_ascii=False, indent=1))
    (a.output / "MERGED_REPORT.md").write_text(markdown(merged))
    print(a.output / "MERGED_REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
