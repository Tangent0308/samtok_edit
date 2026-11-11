"""Ablation tables (after ``judge_many.py --report-only`` wrote each run's REPORT.json): IF-Bench UnitPass / JointPass
per system with 95 % intervals and paired differences to the system's reference, and the development-set probes.

Statistics as in the paper's main table: per setting, UnitPass = passed units / units and JointPass = passed images /
images with full denominators (a generation failure or a missing Judge answer counts as not passed, and missing answers
are listed); the mean is the arithmetic mean over the system's settings.  Intervals: paired bootstrap over the source
images, 10,000 resamples with seed 20261027, one index table for every system.

    python scripts/ablation/report.py            # writes <ROOT>/REPORT/ABLATION.json and ABLATION.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plan  # noqa: E402

SEED, REPLICATES = 20261027, 10000
PROBE_KEYS = ["single_noref", "single_ref", "single_noref_small", "multi_region_iou", "multi_cross_talk",
              "pair_follow_instance", "pair_follow_part_host", "remove_pair_follow", "edit_out_change",
              "edit_in_target_err"]
TITLES = {
    "abl_ref": "完整配方（缩减协议）", "abl_no_identification": "无目标识别任务（只用编辑对）",
    "abl_no_attention": "无注意力监督（λ_route = λ_read = 0）", "abl_no_twins": "无孪生编辑对",
    "abl_no_region": "无区域均衡项（λ_region = 0）", "abl_no_injector": "无 CodeInjector",
    "abl_no_multi": "无多目标记录", "abl_no_neighbor": "无近邻保持项（λ_distractor = 0）",
    "final": "最终模型", "final_notoken": "最终模型，推理时去掉 Space Token", "final_g1": "最终模型，γ = 1（无 token 引导）",
    "s25": "25% 记录", "s50": "50% 记录", "s100": "100% 记录（S2@16000）", "base": "原版 Qwen-Image-2.1",
}


def arrays(report, cases, settings):
    """Per setting over ``cases`` (in order): passed units, joint pass and a missing-answer flag.  Denominators are the
    manifest's unit counts (``unit_counts``), so failures and missing answers count as not passed."""
    by_key = {(r["setting"], r["case_id"]): r for r in report["requests"]}
    out = {}
    for s in settings:
        passed, joint, missing = np.zeros(len(cases), int), np.zeros(len(cases), int), np.zeros(len(cases), bool)
        for i, case in enumerate(cases):
            r = by_key[(s, case)]
            if r["status"] == "scored":
                passed[i], joint[i] = sum(r["unit_pass"]), int(r["joint_pass"])
            else:
                missing[i] = r["status"] not in ("failed", "failed-planner")
        out[s] = (passed, joint, missing)
    return out


def unit_counts(cases):
    manifest = plan.IFBENCH_BUNDLE / "repo/data/v2_2.6.0/cases.jsonl"
    count = {}
    for line in open(manifest):
        c = json.loads(line)
        count[c["id"]] = len(c["units"])
    return np.array([count[c] for c in cases])


def rates(data, units, index=None):
    """UnitPass and JointPass per setting for one index draw (None: the observed sample)."""
    out = {}
    for s, (passed, joint, _) in data.items():
        if index is None:
            out[s] = (passed.sum() / units.sum(), joint.mean())
        else:
            out[s] = (passed[index].sum(axis=1) / units[index].sum(axis=1), joint[index].mean(axis=1))
    return out


def interval(values):
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def main():
    cases = plan.read_cases()
    units = unit_counts(cases)
    index = np.random.default_rng(SEED).integers(0, len(cases), size=(REPLICATES, len(cases)))
    systems = plan.systems()
    loaded, result = {}, {"cases": len(cases), "units": int(units.sum()), "seed": SEED, "replicates": REPLICATES,
                           "systems": {}, "differences": {}, "probes": {}}
    for name, (adapter, settings, overrides, reference) in systems.items():
        path = plan.ROOT / "ifbench" / name / "REPORT.json"
        if not path.exists():
            result["systems"][name] = {"missing_report": str(path)}
            continue
        report = json.loads(path.read_text())
        data = arrays(report, cases, settings)
        loaded[name] = data
        observed, draws = rates(data, units), rates(data, units, index)
        entry = {"title": TITLES.get(name, name), "adapter": str(adapter), "overrides": overrides, "settings": {}}
        for k, metric in enumerate(("UnitPass", "JointPass")):
            entry[metric] = {"value": float(np.mean([observed[s][k] for s in settings])),
                             "CI95": interval(np.mean([draws[s][k] for s in settings], axis=0))}
        for s in settings:
            entry["settings"][s] = {"UnitPass": float(observed[s][0]), "JointPass": float(observed[s][1]),
                                    "missing": int(data[s][2].sum())}
        entry["missing"] = int(sum(data[s][2].sum() for s in settings))
        result["systems"][name] = entry
    for name, (_, settings, _, reference) in systems.items():
        if reference is None or name not in loaded or reference not in loaded:
            continue
        mine = rates(loaded[name], units, index)
        theirs = rates({s: loaded[reference][s] for s in settings}, units, index)
        obs_mine = rates(loaded[name], units)
        obs_theirs = rates({s: loaded[reference][s] for s in settings}, units)
        diff = {"reference": reference, "settings": settings}
        for k, metric in enumerate(("UnitPass", "JointPass")):
            diff[metric] = {"value": float(np.mean([obs_mine[s][k] - obs_theirs[s][k] for s in settings])),
                            "CI95": interval(np.mean([mine[s][k] - theirs[s][k] for s in settings], axis=0)),
                            "reference_value": float(np.mean([obs_theirs[s][k] for s in settings]))}
        result["differences"][name] = diff
    for name in plan.probes():
        path = plan.ROOT / "probe" / name / "eval.json"
        if path.exists():
            summary = json.loads(path.read_text())["summary"]
            result["probes"][name] = {k: summary.get(k) for k in ["n"] + PROBE_KEYS}
    out = plan.ROOT / "REPORT"
    out.mkdir(parents=True, exist_ok=True)
    (out / "ABLATION.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
    (out / "ABLATION.md").write_text(markdown(result))
    print(markdown(result))


def pct(value):
    return "—" if value is None else f"{100 * value:.1f}"


def ci(entry):
    return f"{pct(entry['value'])} [{pct(entry['CI95'][0])}, {pct(entry['CI95'][1])}]"


def signed(entry):
    v, (lo, hi) = entry["value"], entry["CI95"]
    return f"{100 * v:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"


def markdown(result):
    lines = [f"# 消融结果（IF-Bench {result['cases']} 张图、{result['units']} 个单元；开发集探针）", "",
             "UnitPass / JointPass 为各系统 setting 的平均（%），区间为按源图配对的 bootstrap 95% 区间"
             f"（{result['replicates']} 次，种子 {result['seed']}，所有系统共用一组抽样）。", ""]
    blocks = [("## 1. 组件消融（缩减协议，参照：完整配方）", [f"abl_{a}" for a in plan.ARMS]),
              ("## 2. 最终模型的推理诊断（参照：最终模型）", ["final", "final_notoken", "final_g1"]),
              ("## 3. 数据规模（S2 配方，相同 epoch 数，十个 setting；参照：100%）", list(plan.SCALE) + ["s100"])]
    for title, names in blocks:
        lines += [title, "", "| 系统 | settings | UnitPass | JointPass | ΔUnitPass | ΔJointPass | 缺失 |",
                  "|---|---|---|---|---|---|---|"]
        for name in names:
            entry = result["systems"].get(name, {})
            if "UnitPass" not in entry:
                lines.append(f"| {TITLES.get(name, name)} | — | 无结果 | | | | |")
                continue
            diff = result["differences"].get(name)
            lines.append(f"| {entry['title']} | {', '.join(entry['settings'])} | {ci(entry['UnitPass'])} | "
                         f"{ci(entry['JointPass'])} | {signed(diff['UnitPass']) if diff else '参照'} | "
                         f"{signed(diff['JointPass']) if diff else '参照'} | {entry['missing']} |")
        lines.append("")
    lines += ["## 4. 开发集探针（1,320 项，20 步，不用判分模型）", "",
              "| 系统 | " + " | ".join(PROBE_KEYS) + " |", "|---|" + "---|" * len(PROBE_KEYS)]
    for name, values in result["probes"].items():
        lines.append(f"| {TITLES.get(name, name)} | " + " | ".join("—" if values.get(k) is None else str(values[k])
                                                                 for k in PROBE_KEYS) + " |")
    lines += ["", "注：final_notoken 只在 S1、S2-mask 上评测（盲指代 setting 去掉 token 后指令中没有目标），"
              "其差值只在这两个 setting 上与最终模型比较。"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
