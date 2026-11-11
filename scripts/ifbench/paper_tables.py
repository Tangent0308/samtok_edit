"""Tables of the benchmark repository's docs/PAPER_RESULTS.md, generated from the final merged report and the
benchmark's and our per-request records (read-only).  Writes ``tables.md`` (one marked block per table) and
``stats.json`` into the output directory.

    python scripts/ifbench/paper_tables.py --output <dir>
"""
import argparse
import collections
import json
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path("/mnt/bn/strategy-mllm-train/user/tanyue")
RES = ROOT / "experiments2/IF-Bench/one-mp-full4-20261031"
OURS = ROOT / "experiments2/SAMTokEdit/ifbench_one_mp_full4_final"
MERGED = ROOT / "experiments2/SAMTokEdit/ifbench_merged_final_20261104/MERGED_REPORT.json"
OVERLAY = ROOT / "experiments2/SAMTokEdit/ifbench_one_mp_full4_baseline_rejudge1/OVERLAY.json"
OUR_REJUDGE = ROOT / "experiments2/SAMTokEdit/ifbench_one_mp_full4_final_rejudge1/REPORT.json"
p_ = argparse.ArgumentParser(description=__doc__.split("\n")[0])
p_.add_argument("--output", type=Path, required=True)
HERE = p_.parse_args().output
HERE.mkdir(parents=True, exist_ok=True)

m = json.loads(MERGED.read_text())
rep = json.loads((RES / "report/REPORT.json").read_text())
S = m["settings"]
ORDER = ["samtokedit", "qwen21", "flux3", "mirage", "mice", "flux2_klein9b", "qwen2511", "replan"]
NAME = {"samtokedit": "**SAMTokEdit（我方）**", "qwen21": "Qwen-Image-2.1", "flux3": "FLUX.3 Klein distilled",
        "mirage": "MIRAGE（Qwen-2511）", "mice": "MICE（Klein-4B）", "flux2_klein9b": "FLUX.2 Klein-9B distilled",
        "qwen2511": "Qwen-Image-Edit-2511", "replan": "RePlan（Qwen-2511）"}
BASE = [x for x in ORDER if x != "samtokedit"]
out = []


def p(v, digits=1):
    return "—" if v is None else f"{100 * v:.{digits}f}"


def ci(entry, sign=False, br=False):
    """'v [lo, hi]' in percent (points for differences); entry has value/bootstrap.CI95."""
    if not entry or entry.get("value") is None:
        return "—"
    lo, hi = entry["bootstrap"]["CI95"]
    f = (lambda x: f"{100 * x:+.1f}") if sign else (lambda x: f"{100 * x:.1f}")
    sep = "<br>" if br else " "
    mark = "" if not sign or lo > 0 or hi < 0 else " ⁿˢ"
    return f"{f(entry['value'])}{mark}{sep}[{f(lo)}, {f(hi)}]"


def row_entry(method, setting, metric):
    r = m["rows"][method][setting]
    return {"value": r[metric], "bootstrap": {"CI95": r["bootstrap"][metric]["CI95"]}} if r[metric] is not None else None


# 1. Means over the ten settings with paired differences.
mean = m["mean_over_settings"]
paired = m["paired_samtokedit_minus_baseline"]
out.append("<!-- T-mean -->\n| 方法 | UnitPass % | JointPass % | 我方 − 该方法 ΔUP（百分点） | 我方 − 该方法 ΔJP（百分点） |\n|---|---:|---:|---:|---:|")
for x in ORDER:
    du = ci(paired[x]["UnitPass"].get("mean"), sign=True) if x != "samtokedit" else "—"
    dj = ci(paired[x]["JointPass"].get("mean"), sign=True) if x != "samtokedit" else "—"
    out.append(f"| {NAME[x]} | {ci(mean[x]['UnitPass'])} | {ci(mean[x]['JointPass'])} | {du} | {dj} |")

# 2./3. Per-setting rates with intervals.
for metric in ("UnitPass", "JointPass"):
    out.append(f"\n<!-- T-{metric} -->\n| 方法 | " + " | ".join(S) + " |\n|---|" + "---:|" * len(S))
    for x in ORDER:
        out.append(f"| {NAME[x]} | " + " | ".join(ci(row_entry(x, s, metric), br=True) for s in S) + " |")

# 4. Per-setting paired differences.
for metric in ("UnitPass", "JointPass"):
    out.append(f"\n<!-- T-paired-{metric} -->\n| 基线 | " + " | ".join(S) + " | 十种平均 |\n|---|" + "---:|" * (len(S) + 1))
    for b in BASE:
        cells = [ci(paired[b][metric].get(s), sign=True, br=True) for s in S] + [ci(paired[b][metric].get("mean"), sign=True, br=True)]
        out.append(f"| {NAME[b]} | " + " | ".join(cells) + " |")

# 5. Derived diagnostics.
LABEL = {"Delta_lang_point": "语言依赖 point", "Delta_lang_box": "语言依赖 box", "Delta_lang_mask": "语言依赖 mask",
         "G_region_point": "区域增益 point", "G_region_box": "区域增益 box", "G_region_mask": "区域增益 mask",
         "R_mod": "形式鲁棒性", "C_mix": "混合代价"}
for metric in ("UnitPass", "JointPass"):
    out.append(f"\n<!-- T-derived-{metric} -->\n| 方法 | " + " | ".join(LABEL.values()) + " |\n|---|" + "---:|" * len(LABEL))
    for x in ORDER:
        items = {d["diagnostic"]: d for d in m["derived"][x] if d["metric"] == metric}
        cells = []
        for key in LABEL:
            d = items[key]
            if d["value"] is None:
                cells.append("—")
            elif key == "R_mod":
                lo, hi = d["bootstrap"]["CI95"]
                cells.append(f"{d['value']:.2f}<br>[{lo:.2f}, {hi:.2f}]")
            else:
                cells.append(ci(d, sign=True, br=True).replace(" ⁿˢ", ""))
        out.append(f"| {NAME[x]} | " + " | ".join(cells) + " |")

# 6. JointPass and UnitPass by number of targets K (mean of the ten settings).
kc = rep["K_counts"]
out.append("\n<!-- T-byK -->\n| 方法 | " + " | ".join(f"K={k} JP（{kc[str(k)]} 图）" for k in range(2, 6))
           + " | " + " | ".join(f"K={k} UP" for k in range(2, 6)) + " |\n|---|" + "---:|" * 8)
for x in ORDER:
    jp = [sum(m["rows"][x][s]["by_K"][str(k)]["JointPass"] for s in S) / len(S) for k in range(2, 6)]
    up = [sum(m["rows"][x][s]["by_K"][str(k)]["UnitPass"] for s in S) / len(S) for k in range(2, 6)]
    out.append(f"| {NAME[x]} | " + " | ".join(p(v) for v in jp) + " | " + " | ".join(p(v) for v in up) + " |")

# 7. Judge dimension means over every final scored answer, plus Judge failure statistics.
overlay = json.loads(OVERLAY.read_text())
rejudged = {e["judge_key"]: e for e in overlay["entries"]}
bindings, logical = {}, []
for n in range(4):
    for b in json.loads((RES / f"nodes/node-{n}/judge/bindings.json").read_text()):
        if b["generation_status"] == "generated":
            bindings.setdefault(b["model"], {}).setdefault(b["judge_key"], n)
            logical.append((b["model"], b["judge_key"]))


def first_answer(item):
    model, key, node = item
    v = json.loads((RES / f"nodes/node-{node}/judge/effective_records/{key}.json").read_text())
    return model, key, v["status"], v.get("scores"), bool(v.get("deterministic_repair_used"))


items = [(model, key, node) for model, keys in bindings.items() for key, node in keys.items()]
with ThreadPoolExecutor(48) as pool:
    answers = list(pool.map(first_answer, items))
dims = {x: {d: [0, 0] for d in ("E", "P", "Q", "global_P", "global_Q")} for x in ORDER}
judge = {x: {"keys": 0, "first_failed": 0, "repaired": 0, "rejudged_scored": 0, "failed_twice": 0} for x in ORDER}


def add(model, scores):
    for u in scores["units"]:
        for d in "EPQ":
            dims[model][d][0] += u[d]["score"]
            dims[model][d][1] += 1
    for d in ("global_P", "global_Q"):
        dims[model][d][0] += scores[d]["score"]
        dims[model][d][1] += 1


final = {}                      # key -> final scores (first answer, or the re-judge), None if it failed twice
for model, key, status, scores, repaired in answers:
    judge[model]["keys"] += 1
    judge[model]["repaired"] += repaired
    if status == "scored":
        final[key] = scores
        continue
    judge[model]["first_failed"] += 1
    e = rejudged[key]["rejudge"]
    if e["status"] == "scored":
        judge[model]["rejudged_scored"] += 1
        final[key] = e["scores"]
    else:
        judge[model]["failed_twice"] += 1
        final[key] = None
for model, key in logical:         # the benchmark's E/P/Q means count logical requests, as its rows do
    if final[key] is not None:
        add(model, final[key])
# Our method: main run answers plus its one re-judge overlay.
run = json.loads((OURS / "RUN.json").read_text())
run_id = next(p.name for p in (OURS / "judge").iterdir())
ours_report = json.loads((OURS / "REPORT.json").read_text())
again = {r["request_id"]: r for r in json.loads(OUR_REJUDGE.read_text())["requests"]}
seen = set()


def our_answer(entry):
    path = OURS / "judge" / run_id / "effective" / f"{entry['judge_key']}.json"
    return entry, json.loads(path.read_text())


with ThreadPoolExecutor(48) as pool:
    ours = list(pool.map(our_answer, [r for r in ours_report["requests"] if r["judge_key"]]))
rejudge_run = json.loads(OUR_REJUDGE.read_text())
rj_root = Path(rejudge_run["run"]) / "judge" / rejudge_run["judge_run_id"] / "effective"
for entry, v in ours:
    key = entry["judge_key"]
    scores = v.get("scores") if v["status"] == "scored" else None
    if scores is None:
        r = again[entry["request_id"]]
        w = json.loads((rj_root / f"{r['judge_key']}.json").read_text())
        scores = w.get("scores") if w["status"] == "scored" else None
    if scores is not None:
        add("samtokedit", scores)
    if key in seen:
        continue
    seen.add(key)
    judge["samtokedit"]["keys"] += 1
    judge["samtokedit"]["repaired"] += bool(v.get("deterministic_repair_used"))
    if v["status"] != "scored":
        judge["samtokedit"]["first_failed"] += 1
        judge["samtokedit"]["rejudged_scored" if scores is not None else "failed_twice"] += 1
out.append("\n<!-- T-dims -->\n| 方法 | 单元 E | 单元 P | 单元 Q | 全局 P | 全局 Q | 计分单元数 | 计分请求数 |\n|---|---:|---:|---:|---:|---:|---:|---:|")
for x in ORDER:
    d = dims[x]
    out.append(f"| {NAME[x]} | " + " | ".join(f"{d[k][0] / d[k][1]:.2f}" for k in ("E", "P", "Q", "global_P", "global_Q"))
               + f" | {d['E'][1]:,} | {d['global_P'][1]:,} |")
out.append("\n<!-- T-judge -->\n| 方法 | 独立判分 key | 首次即计分 | 其中经确定性 JSON 修复 | 首次失败 | 再判一次后计分 | 两次都失败（按失败计） |\n|---|---:|---:|---:|---:|---:|---:|")
for x in ORDER:
    j = judge[x]
    out.append(f"| {NAME[x]} | {j['keys']:,} | {j['keys'] - j['first_failed']:,} | {j['repaired']} | {j['first_failed']} | "
               f"{j['rejudged_scored']} | {j['failed_twice']} |")

# 8. Generation status per method and setting.
out.append("\n<!-- T-generation -->\n| 方法 | " + " | ".join(S) + " | 合计 | 类型 |\n|---|" + "---:|" * (len(S) + 1) + "---|")
gen = {r["model"]: {} for r in rep["rows"]}
for r in rep["rows"]:
    gen[r["model"]][r["setting"]] = r["generation_failed_cases"]
kinds = {"replan": "规划给出非法 bbox（failed）", "mirage": "原生规划失败（failed-planner）",
         "mice": "原生 200-token 条件放不下（MiceTextOverflow）"}
for x in BASE:
    total = sum(gen[x].values())
    out.append(f"| {NAME[x]} | " + " | ".join(str(gen[x][s]) for s in S) + f" | {total} | {kinds.get(x, '—')} |")
our_fail = {s: ours_report["settings"][s]["generation_failed_cases"] for s in S}
out.append("| " + NAME["samtokedit"] + " | " + " | ".join(str(our_fail[s]) for s in S) + f" | {sum(our_fail.values())} | — |")

# 9. Inference cost.
calls = json.loads((RES / "report/CALLS.json").read_text())
edit, fwd, plan, plan_calls = (collections.defaultdict(list), collections.defaultdict(collections.Counter),
                               collections.defaultdict(list), collections.Counter())
for node in calls["nodes"]:
    for r in node["editor_records"]:
        model = r["task_key"].split("/")[0]
        if r["status"] == "generated" and r.get("wall_seconds") is not None:
            edit[model].append(r["wall_seconds"])
            fwd[model][str(r.get("transformer_forward_calls"))] += 1
    for r in node["planner_records"]:
        model = r["path"].split("/runs/")[1].split("/")[0]
        if r.get("wall_seconds") is not None:
            plan[model].append(r["wall_seconds"] / (r.get("batch_items") or 1))
        plan_calls[model] += r.get("new_api_calls") or 0
cost = {x: {"edit_requests_timed": len(edit[x]), "edit_wall_median_s": statistics.median(edit[x]),
            "transformer_forward_calls": dict(fwd[x])} for x in edit}
for x in plan:
    cost[x].update(planner_records=len(plan[x]), planner_wall_per_item_median_s=statistics.median(plan[x]),
                   planner_api_calls=plan_calls[x])


def our_records(r):
    return (json.loads((OURS / "pass1" / r["setting"] / r["case_id"] / "result.json").read_text()),
            json.loads((OURS / "outputs" / r["setting"] / r["case_id"] / "result.json").read_text()))


with ThreadPoolExecutor(48) as pool:
    recs = list(pool.map(our_records, run["requests"]))
ours_stats = {"requests": len(recs), "pass1_wall_median": statistics.median(a["wall_seconds"] for a, _ in recs),
              "pass2_generate_median": statistics.median(b["generate_seconds"] for _, b in recs),
              "pass2_wall_median": statistics.median(b["wall_seconds"] for _, b in recs),
              "fallback_units": sum(a.get("fallback_units", 0) for a, _ in recs),
              "unit_token_sources": dict(collections.Counter(u["source"] for a, _ in recs for u in a["units"])),
              "pass2_peak_gib": max(b["peak_gpu_allocated_bytes"] for _, b in recs) / 2 ** 30}
out.append("\n<!-- T-cost -->\n| 方法 | 编辑每请求中位耗时（秒） | DiT/Transformer 前向次数 | 原生规划 | 计时请求数 |\n|---|---:|---|---|---:|")
out.append(f"| {NAME['samtokedit']} | {ours_stats['pass2_generate_median']:.1f}（Pass2 生成；含编码 {ours_stats['pass2_wall_median']:.1f}） | 80 | "
           f"Pass1 定位每请求中位 {ours_stats['pass1_wall_median']:.1f} 秒 | {ours_stats['requests']:,} |")
for x in BASE:
    c = cost[x]
    fwd = "、".join(f"{k}×{v:,}" for k, v in sorted(c["transformer_forward_calls"].items(), key=lambda kv: int(kv[0])))
    planner = "—"
    if "planner_records" in c:
        planner = f"{c['planner_api_calls']:,} 次调用，每项中位 {c['planner_wall_per_item_median_s']:.1f} 秒"
    out.append(f"| {NAME[x]} | {c['edit_wall_median_s']:.1f} | {fwd} | {planner} | {c['edit_requests_timed']:,} |")

(HERE / "tables.md").write_text("\n".join(out) + "\n")
(HERE / "stats.json").write_text(json.dumps({"judge": judge, "dims": dims, "cost": cost, "ours": ours_stats}, indent=1))
print(HERE / "tables.md")
