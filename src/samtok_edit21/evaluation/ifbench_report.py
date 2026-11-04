"""Merge a SAMTokEdit IF-Bench run into the benchmark's full report: every method's UnitPass / JointPass, confidence
intervals and paired differences on one set of bootstrap draws.

The statistics are the benchmark's own (``fairness_dev_report.Bootstrap`` / ``rates`` / ``derived``): source-cluster
resampling with one seed and one index table for all methods and settings, and full denominators, so generation
failures count and a missing Judge answer leaves a row incomplete.  Every baseline row is recomputed from its per-case
records and must equal the published row, and the index table must have the published digest; no node percentages are
averaged.  ``--overlay`` runs (a repair run judged on its own, e.g. after the Pass1 format repair) replace the main
run's entries of their requests; ``--baseline-overlay`` directories (``scripts/ifbench/rejudge_baselines.py``: one more
Judge call for a baseline answer that failed) give the failed answers of the published rows their new scores, after the
published rows have been reproduced from the original records.  Every replacement is listed in the merged report.

Before the benchmark's global report exists, ``--bundle`` computes the baseline rows from the four machines' Judge records
with the benchmark's own ``make_row`` (a preliminary report); ``--expect`` checks that a later merge gives the same
numbers.  Run with the benchmark snapshot on PYTHONPATH:

    python -m samtok_edit21.evaluation.ifbench_report (--benchmark-report <results root>/report/REPORT.json | \\
        --bundle <bundle>) --run <run dir> [--overlay <repair run dir>] [--baseline-overlay <re-judge dir>] \\
        [--expect <earlier MERGED_REPORT.json>] --output <dir>
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


def node_report(bundle, nodes=4):
    """A stand-in for the benchmark's global report, computed before it: the benchmark's own ``make_row`` over the
    machines' Judge bindings and effective records, on the same index table.  Each machine verified its records when it
    completed (``COMPLETED.json``); the global report re-verifies every hash and must later give the same rows."""
    from concurrent.futures import ThreadPoolExecutor

    from samtok_benchmark.io import digest, read_jsonl, sha256_file
    from samtok_benchmark.ifbench.fairness_dev_report import REPLICATES, SEED, Bootstrap, make_row

    plan = json.loads((bundle / "plan.json").read_text())
    ready = json.loads((bundle / "READY.json").read_text())
    if sha256_file(plan["case_manifest"]) != plan["case_manifest_sha256"]:
        raise ValueError("Case manifest differs from the plan")
    cases = {c["id"]: c for c in read_jsonl(plan["case_manifest"])}
    results = Path(plan["results_root"])
    bound, paths = {}, {}
    for node in range(nodes):
        directory = results / "nodes" / f"node-{node}"
        done = json.loads((directory / "COMPLETED.json").read_text()) if (directory / "COMPLETED.json").exists() else {}
        if done.get("status") != "completed" or done.get("plan_digest") != plan["digest"]:
            raise ValueError(f"Node {node} has not completed this plan")
        for b in json.loads((directory / "judge" / "bindings.json").read_text()):
            if (b["model"], b["setting"], b["case_id"]) in bound:
                raise ValueError(f"{b['key']} is bound on two machines")
            bound[b["model"], b["setting"], b["case_id"]] = b
            if b["generation_status"] == "generated":
                paths[b["judge_key"]] = directory / "judge" / "effective_records" / f"{b['judge_key']}.json"
    if set(bound) != {(t["model"], t["setting"], t["case_id"]) for t in plan["tasks"]}:
        raise ValueError("The machines' bindings do not cover the plan")
    with ThreadPoolExecutor(max_workers=32) as pool:
        effective = dict(zip(paths, pool.map(lambda path: json.loads(path.read_text()), paths.values())))
    context = {"case_ids": plan["case_ids"], "cases": cases}
    boot = Bootstrap([len(cases[c]["units"]) for c in plan["case_ids"]])
    rows = [make_row(m, s, context, bound, effective, boot)[0] for m in plan["model_order"] for s in plan["settings"]]
    return {"preliminary": True, "case_ids": plan["case_ids"], "settings": plan["settings"],
            "model_order": plan["model_order"], "rows": rows, "judge_config": plan["judge_config"],
            "plan_digest": plan["digest"], "source_commit": ready["source_commit"],
            "bootstrap": {"seed": SEED, "replicates": REPLICATES, "shared_indices_sha256": digest(boot.indices.tolist()),
                          "cluster": "source image", "scope": "the four machines' records before the global report"}}


def differences(merged, expected):
    """Every rate, interval, mean and paired difference of ``merged`` that differs from ``expected``."""
    found = []

    def compare(path, a, b):
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                compare(f"{path}/{key}", a.get(key), b.get(key))
        elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
            for i, (x, y) in enumerate(zip(a, b)):
                compare(f"{path}/{i}", x, y)
        elif isinstance(a, float) and isinstance(b, float):
            if not math.isclose(a, b, abs_tol=1e-12):
                found.append(path)
        elif a != b:
            found.append(path)

    for key in ("rows", "mean_over_settings", f"paired_{METHOD}_minus_baseline"):
        compare(key, merged[key], expected[key])
    return found


def apply_baseline_overlays(rows, overlays, failed_twice_as_failure=False):
    """The published baseline rows with every failed Judge answer that an overlay judged once more and scored given its
    new pass counts (the benchmark's rule: E=4, P>=3, Q>=3 per unit; all units and global P, Q >= 3 for the source);
    also the list of replacements and the list of answers counted as failures.  A key is judged once more at most once,
    and each overlay entry must match the failed answers of exactly its logical requests.  An answer whose second call
    also failed stays missing, or, with ``failed_twice_as_failure`` (user decision 11-04), counts as a failed source:
    no unit passes and no joint pass."""
    from samtok_benchmark.ifbench.fairness_dev_judge import validate_scores

    answers = {}
    for overlay in overlays:
        for entry in overlay["entries"]:
            if entry["judge_key"] in answers:
                raise ValueError(f"Judge key {entry['judge_key']} is re-judged twice")
            answers[entry["judge_key"]] = (overlay, entry)
    replaced, counted, result, matched = [], [], [], {}
    for row in rows:
        cases = []
        for case in row["per_case"]:
            found = answers.get(case.get("judge_key"))
            if found is None:
                cases.append(case)
                continue
            overlay, entry = found
            logical = {(x["model"], x["setting"], x["case_id"]) for x in entry["logical"]}
            if (row["model"], row["setting"], case["case_id"]) not in logical or entry["original"]["status"] != case["status"] \
                    or case["status"] == "scored":
                raise ValueError(f"{row['model']}/{row['setting']}/{case['case_id']}: re-judge entry does not match")
            matched[entry["judge_key"]] = matched.get(entry["judge_key"], 0) + 1
            if entry["rejudge"]["status"] != "scored":
                if failed_twice_as_failure:
                    cases.append(dict(case, status="judge-failed-twice", UnitPass_count=0, JointPass=0))
                    counted.append({"model": row["model"], "setting": row["setting"], "case_id": case["case_id"],
                                    "judge_key": case["judge_key"], "overlay": overlay.get("source")})
                else:
                    cases.append(case)
                continue
            if len(entry["unit_ids"]) != case["K"]:
                raise ValueError(f"{case['case_id']}: re-judged unit count differs from K")
            score = entry["rejudge"]["scores"]
            validate_scores(score, entry["unit_ids"])
            bits = [u["E"]["score"] == 4 and u["P"]["score"] >= 3 and u["Q"]["score"] >= 3 for u in score["units"]]
            joint = all(bits) and score["global_P"]["score"] >= 3 and score["global_Q"]["score"] >= 3
            cases.append(dict(case, status="scored", UnitPass_count=sum(bits), JointPass=int(joint),
                              rejudge_effective_record=entry["rejudge"]["effective_record"]))
            replaced.append({"model": row["model"], "setting": row["setting"], "case_id": case["case_id"],
                             "judge_key": case["judge_key"], "replaced_status": case["status"],
                             "overlay": overlay.get("source"), "UnitPass_count": sum(bits), "JointPass": int(joint)})
        result.append(dict(row, per_case=cases))
    for key, (_, entry) in answers.items():
        if matched.get(key, 0) != len(entry["logical"]):
            raise ValueError(f"Re-judged key {key} does not match the published failed answers")
    return result, replaced, counted


def merge(benchmark, ours, overlays=(), baseline_overlays=(), failed_twice_as_failure=False):
    from samtok_benchmark.io import digest
    from samtok_benchmark.ifbench.fairness_dev_report import FAILURES, Bootstrap, derived, quantiles, rates

    ours, replaced = apply_overlays(ours, overlays)
    counted = []
    if failed_twice_as_failure:            # the same rule for our answers: re-judged by an overlay and failed again
        rejudged = {r["request_id"] for r in replaced}
        requests = []
        for entry in ours["requests"]:
            if entry.get("request_id") in rejudged and entry["status"] != "scored" \
                    and entry.get("generation_status") == "generated":
                entry = dict(entry, status="judge-failed-twice")
                counted.append({"model": METHOD, "setting": entry["setting"], "case_id": entry["case_id"],
                                "judge_key": entry["judge_key"], "overlay": None})
            requests.append(entry)
        ours = dict(ours, requests=requests)
        FAILURES = FAILURES | {"judge-failed-twice"}
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
    for row in benchmark["rows"]:          # the published rows first, from the benchmark's own records
        up, jp, unknown = baseline_arrays(row, case_ids)
        check_published(row, rates(up, jp, ks, unknown, boot)[0])
    if any(o["judge_run_id"] != digest(benchmark["judge_config"]) for o in baseline_overlays):
        raise ValueError("A baseline overlay was judged with another Judge configuration")
    published, baseline_replaced, baseline_counted = apply_baseline_overlays(benchmark["rows"], baseline_overlays,
                                                                             failed_twice_as_failure)
    for row in published:
        arrays[row["model"], row["setting"]] = baseline_arrays(row, case_ids)
    for setting in settings:
        arrays[METHOD, setting] = method_arrays(ours["requests"], setting, case_ids, FAILURES)
    methods = [*benchmark["model_order"], METHOD]
    for method in methods:
        rows[method], draws[method] = {}, {}
        for setting in settings:
            up, jp, unknown = arrays[method, setting]
            computed, distributions = rates(up, jp, ks, unknown, boot)
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
            f"paired_{METHOD}_minus_baseline": paired, "replaced_requests": replaced,
            "baseline_overlays": [o.get("source") for o in baseline_overlays],
            "replaced_baseline_answers": baseline_replaced, "preliminary": bool(benchmark.get("preliminary")),
            "failed_twice_as_failure": failed_twice_as_failure,
            "failed_twice_counted_as_failure": counted + baseline_counted}


def cell(value, interval):
    if value is None:
        return "不完整"
    return f"{100 * value:.1f} [{100 * interval[0]:.1f}, {100 * interval[1]:.1f}]"


def markdown(merged):
    settings = merged["settings"]
    lines = ["# IF-Bench 全量结果：七模型 + SAMTokEdit" + ("（初步）" if merged.get("preliminary") else ""), "",
             f"{merged['source_count']} 张源图、{merged['units']} 个单元。数字是百分比，方括号是 95% 区间"
             "（benchmark 的源图配对 bootstrap，所有方法共用同一组抽样）。mean 是十种 setting 的算术平均，只作概览。", ""]
    if merged.get("preliminary"):
        lines += ["基线各行由四台机器的判分记录直接计算（benchmark 自己的 `make_row`），benchmark 的全局报告尚未生成；"
                  "全局报告出来后重新合并，数字应完全相同。", ""]
    if merged["replaced_requests"]:
        lines += [f"{METHOD} 有 {len(merged['replaced_requests'])} 个请求用修复运行的结果替换："
                  + "、".join(r["request_id"] for r in merged["replaced_requests"]) + "。", ""]
    if merged.get("replaced_baseline_answers"):
        counts = {}
        for r in merged["replaced_baseline_answers"]:
            counts[r["model"]] = counts.get(r["model"], 0) + 1
        lines += ["基线 Judge 失败的答案按同一规则再判一次后计分（列表见 MERGED_REPORT.json）："
                  + "、".join(f"{m} {n} 个" for m, n in counts.items()) + "。", ""]
    if merged.get("failed_twice_counted_as_failure"):
        lines += ["两次判分都失败、按失败计（该图无单元通过、整图不通过）："
                  + "、".join(f"{r['model']}/{r['setting']}/{r['case_id']}" for r in merged["failed_twice_counted_as_failure"])
                  + "。", ""]
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
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--benchmark-report", type=Path)
    source.add_argument("--bundle", type=Path, help="compute the baseline rows from the machines' records (preliminary)")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--overlay", type=Path, action="append", default=[], help="repair run whose entries replace the main run's")
    p.add_argument("--baseline-overlay", type=Path, action="append", default=[],
                   help="rejudge_baselines.py output whose scored answers replace failed baseline answers")
    p.add_argument("--failed-twice-as-failure", action="store_true",
                   help="count an answer whose second Judge call also failed as a failed source (user decision 11-04)")
    p.add_argument("--expect", type=Path, help="an earlier MERGED_REPORT.json whose numbers this merge must reproduce")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    overlays = [json.loads((run / "REPORT.json").read_text()) for run in a.overlay]
    baseline_overlays = [dict(json.loads((d / "OVERLAY.json").read_text()), source=str(d)) for d in a.baseline_overlay]
    benchmark = json.loads(a.benchmark_report.read_text()) if a.benchmark_report else node_report(a.bundle)
    merged = merge(benchmark, json.loads((a.run / "REPORT.json").read_text()), overlays, baseline_overlays,
                   a.failed_twice_as_failure)
    if a.expect and (found := differences(merged, json.loads(a.expect.read_text()))):
        raise SystemExit(f"{len(found)} numbers differ from {a.expect}: {found[:10]}")
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "MERGED_REPORT.json").write_text(json.dumps(merged, ensure_ascii=False, indent=1))
    (a.output / "MERGED_REPORT.md").write_text(markdown(merged))
    print(a.output / "MERGED_REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
