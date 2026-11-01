"""Canonical IF-Bench Judge and report for a SAMTokEdit IF-Bench run (frozen Judge runtime).

The benchmark's canonical Judge, unchanged: ``dev_judge.evidence`` / ``binding``, its Backend with the bundle's fixed
``judge_config`` (the actual runtime fingerprint must equal it), batches of 16 with the benchmark's bounded prefetch,
``scored`` for the raw answer and ``effective`` for the deterministic JSON-recovery overlay.  A machine judges the
outputs of its own images (``case_ids[node::nodes]``, as in generation), so a Judge key has one owner.  Admissions, raw
answers and effective records are write-once; an admitted request without an answer is never resubmitted.  The report
counts UnitPass (E=4, P>=3, Q>=3) and JointPass (all units, global P and Q >= 3) per setting with full denominators:
generation failures and missing Judge answers stay in them.

Run with the runtime's wrapper (its PYTHONPATH is the benchmark snapshot), outside any repository directory:
    <judge runtime>/python <v4>/scripts/ifbench/judge.py --run <run dir> [--node 0 --nodes 4] [--report-only]
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

from samtok_benchmark.io import digest, read_jsonl, sha256_file, write_json
from samtok_benchmark.ifbench import MANIFEST_SHA256
from samtok_benchmark.ifbench import dev_judge as canonical
from samtok_benchmark.ifbench.dev_judge_stream import prefetched_batches, scored
from samtok_benchmark.ifbench.fairness_dev_judge import FAILURES, effective, immutable, validate_scores


def context(run):
    plan = json.loads((run / "RUN.json").read_text())
    bundle = Path(plan["bundle"])
    config = json.loads((bundle / "configuration.json").read_text())["judge_config"]
    manifest = Path(canonical.__file__).resolve().parents[3] / "data/v2_2.6.0/cases.jsonl"
    if sha256_file(manifest) != MANIFEST_SHA256:
        raise ValueError("Benchmark manifest differs from release 2.6.0")
    wanted = set(plan["case_ids"])
    cases = {c["id"]: c for c in read_jsonl(manifest) if c["id"] in wanted}
    if set(cases) != wanted:
        raise ValueError("Run cases missing from the manifest")
    return plan, config, cases


def jobs(run, plan, config, cases, selected=None):
    """One row per request (of the ``selected`` images); generated outputs carry their canonical Judge key."""

    def row(item):
        path = run / "outputs" / item["setting"] / item["case_id"] / "result.json"
        entry = {"request_id": item["request_id"], "setting": item["setting"], "case_id": item["case_id"],
                 "units": len(cases[item["case_id"]]["units"]), "result": str(path), "key": None}
        if not path.exists():
            entry["generation_status"] = "pending"
            return entry
        record = json.loads(path.read_text())
        entry["generation_status"] = record["status"]
        if record["status"] == "generated":
            if sha256_file(record["output_image"]) != record["output_sha256"]:
                raise ValueError(f"{record['output_image']} changed after generation")
            entry["key"] = canonical.binding(cases[item["case_id"]], record["output_sha256"], config)
            entry["record"] = record
        return entry

    items = [i for i in plan["requests"] if selected is None or i["case_id"] in selected]
    with ThreadPoolExecutor(max_workers=16) as pool:
        return list(pool.map(row, items))


def judge(run, config, cases, rows, node):
    run_id = digest(config)
    root = run / "judge" / run_id
    admitted, raws, effectives = root / "admitted", root / "raw", root / "effective"
    for d in (admitted, raws, effectives):
        d.mkdir(parents=True, exist_ok=True)
    immutable(root / "config.json", config)
    pending, seen = [], set()
    for row in rows:
        key = row["key"]
        if key is None or key in seen or (effectives / f"{key}.json").exists():
            continue
        seen.add(key)
        if (admitted / f"{key}.json").exists():
            print(json.dumps({"admitted_without_answer": row["request_id"], "key": key}), flush=True)
            continue                         # never resubmitted
        pending.append((key, (cases[row["case_id"]], row["record"])))
    if not pending:
        return run_id
    if canonical.fingerprint() != config:
        raise ValueError("Actual Judge runtime/checkpoint fingerprint differs from the bundle's judge_config")
    execution = root / "executions" / f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-node{node}-{os.getpid()}"
    execution.mkdir(parents=True)
    receipt = {"status": "loading", "started_at": time.time(), "run_id": run_id, "node": node, "pending": len(pending),
               "new_admitted_calls": 0, "new_raw_answers": 0,
               "canonical_sha256": sha256_file(Path(canonical.__file__))}
    write_json(execution / "receipt.json", receipt)
    try:
        args = SimpleNamespace(model=canonical.MODEL, **config["engine"], **config["decode"])
        backend = canonical.Backend(args, 22)
        receipt.update(status="running", engine_ready_at=time.time())
        for batch, conversations in prefetched_batches(
            pending, lambda item: canonical.evidence(item[1][0], item[1][1]["output_image"]), size=16, workers=4
        ):
            for key, (case, record) in batch:
                immutable(admitted / f"{key}.json", {"key": key, "case_id": case["id"],
                                                     "output_sha256": record["output_sha256"],
                                                     "canonical_prompt_digest": digest(canonical.prompt(case)),
                                                     "at": time.time(), "execution": str(execution), "attempt": 1})
                receipt["new_admitted_calls"] += 1
            write_json(execution / "receipt.json", receipt)
            tick = time.monotonic()
            answers = backend.generate(conversations)
            elapsed = time.monotonic() - tick
            if len(answers) != len(batch):
                raise RuntimeError("Judge returned a different answer count; admissions kept, no retry")
            for item, answer in zip(batch, answers, strict=True):
                key, (case, _) = item
                raw = scored(item, answer, run_id)
                raw.update(case_id=case["id"], completed_at=time.time(), batch_size=len(batch),
                           batch_wall_seconds=elapsed, execution_receipt=str(execution / "receipt.json"),
                           new_judge_calls=1)
                immutable(raws / f"{key}.json", raw)
                immutable(effectives / f"{key}.json",
                          effective(raw, [u["id"] for u in case["units"]], raws / f"{key}.json"))
                receipt["new_raw_answers"] += 1
            write_json(execution / "receipt.json", receipt)
            print(json.dumps({"node": node, "judged": receipt["new_raw_answers"], "of": len(pending),
                              "batch_seconds": round(elapsed, 1)}), flush=True)
        receipt.update(status="completed", ended_at=time.time())
    except BaseException as error:
        receipt.update(status="error", error=repr(error), ended_at=time.time())
        raise
    finally:
        write_json(execution / "receipt.json", receipt)
    return run_id


def report(run, plan, cases, rows, run_id):
    effectives = run / "judge" / run_id / "effective"
    table, per_request = {}, []
    for setting in plan["settings"]:
        stats = Counter()
        scores = {d: [] for d in ("E", "P", "Q", "global_P", "global_Q")}
        for row in (r for r in rows if r["setting"] == setting):
            stats["cases"] += 1
            stats["units"] += row["units"]
            status = row["generation_status"]
            entry = {"request_id": row["request_id"], "case_id": row["case_id"], "setting": setting,
                     "generation_status": status, "judge_key": row["key"]}
            if status == "generated":
                path = effectives / f"{row['key']}.json"
                value = json.loads(path.read_text()) if path.exists() else None
                status = value["status"] if value else "judge-pending"
                if status == "scored":
                    score = value["scores"]
                    validate_scores(score, [u["id"] for u in cases[row["case_id"]]["units"]])
                    bits = [u["E"]["score"] == 4 and u["P"]["score"] >= 3 and u["Q"]["score"] >= 3
                            for u in score["units"]]
                    joint = all(bits) and score["global_P"]["score"] >= 3 and score["global_Q"]["score"] >= 3
                    stats["unit_pass"] += sum(bits)
                    stats["joint_pass"] += joint
                    stats["scored"] += 1
                    for d in "EPQ":
                        scores[d] += [u[d]["score"] for u in score["units"]]
                    for d in ("global_P", "global_Q"):
                        scores[d].append(score[d]["score"])
                    entry.update(unit_pass=bits, joint_pass=joint,
                                 E=[u["E"]["score"] for u in score["units"]],
                                 P=[u["P"]["score"] for u in score["units"]],
                                 Q=[u["Q"]["score"] for u in score["units"]],
                                 global_P=score["global_P"]["score"], global_Q=score["global_Q"]["score"],
                                 repaired=value.get("deterministic_repair_used", False))
                else:
                    stats["missing"] += 1
            elif status in FAILURES:
                stats["generation_failed"] += 1
            else:
                stats["missing"] += 1
            entry["status"] = status
            per_request.append(entry)
        table[setting] = {
            "cases": stats["cases"], "units": stats["units"], "scored_cases": stats["scored"],
            "generation_failed_cases": stats["generation_failed"], "missing_cases": stats["missing"],
            "UnitPass": stats["unit_pass"] / stats["units"] if stats["units"] else None,
            "JointPass": stats["joint_pass"] / stats["cases"] if stats["cases"] else None,
            "unit_pass": stats["unit_pass"], "joint_pass": stats["joint_pass"],
            "means_among_scored": {d: (sum(v) / len(v) if v else None) for d, v in scores.items()},
        }
    result = {"run": str(run), "judge_run_id": run_id, "method": plan["method"], "case_ids": plan["case_ids"],
              "settings": table, "requests": per_request,
              "note": "our method alone; the merged report adds the source-paired bootstrap"}
    write_json(run / "REPORT.json", result)
    print(f"{'setting':10s} {'cases':>5s} {'units':>5s} {'UP':>9s} {'JP':>9s} {'E':>5s} {'P':>5s} {'Q':>5s} fail miss")
    for setting, r in table.items():
        m = r["means_among_scored"]
        fmt = lambda v: f"{v:5.2f}" if v is not None else "  n/a"  # noqa: E731
        print(f"{setting:10s} {r['cases']:5d} {r['units']:5d} {r['unit_pass']:4d}/{r['units']:<4d} "
              f"{r['joint_pass']:4d}/{r['cases']:<4d} {fmt(m['E'])} {fmt(m['P'])} {fmt(m['Q'])} "
              f"{r['generation_failed_cases']:4d} {r['missing_cases']:4d}")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--node", type=int, default=0)
    p.add_argument("--nodes", type=int, default=1)
    p.add_argument("--report-only", action="store_true")
    a = p.parse_args()
    run = a.run.resolve()
    plan, config, cases = context(run)
    if a.report_only:
        report(run, plan, cases, jobs(run, plan, config, cases), digest(config))
        return
    rows = jobs(run, plan, config, cases, set(plan["case_ids"][a.node::a.nodes]))
    open_rows = [r["request_id"] for r in rows if r["generation_status"] not in FAILURES | {"generated"}]
    if open_rows:
        raise SystemExit(f"Generation not terminal for {open_rows[:10]} ({len(open_rows)} in all); "
                         "the Judge starts after all generation")
    judge(run, config, cases, rows, a.node)


if __name__ == "__main__":
    main()
