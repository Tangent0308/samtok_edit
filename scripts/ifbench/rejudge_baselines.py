"""Judge once more every baseline answer that failed in the benchmark's full run, without touching its records.

The agreed rule for all eight methods (docs/03 §12.8): an answer that failed the canonical Judge gets one more call of
the same Judge.  The benchmark never retries, and one missing answer leaves its whole model/setting row incomplete.  For
each machine whose benchmark run has completed (``nodes/node-<n>/COMPLETED.json``), this reads the benchmark's Judge
bindings and effective records, takes every Judge key whose effective answer is not scored, checks the output against
its generation record and the canonical binding, and judges it with our ``judge.py`` loop (the benchmark's canonical
Judge, batches of 16, write-once admissions, raw answers and effective records under ``--output``).  An admitted key
without an answer is never resubmitted.  ``OVERLAY.json`` lists every failed key, the logical requests it serves and the
new answer; pass the directory to ``ifbench_report --baseline-overlay``.  Machines can be done one at a time.

Run with the Judge runtime's wrapper (its PYTHONPATH is the benchmark snapshot), outside any repository directory:
    <judge runtime>/python <v4>/scripts/ifbench/rejudge_baselines.py --bundle <bundle> --output <dir> \\
        [--node 1 --node 2 ...] [--collect-only]
"""
import argparse
import importlib.util
import json
from pathlib import Path

from samtok_benchmark.io import digest, read_jsonl, sha256_file, write_json
from samtok_benchmark.ifbench import MANIFEST_SHA256
from samtok_benchmark.ifbench import dev_judge as canonical
from samtok_benchmark.ifbench.fairness_dev_judge import validate_scores

NODES = 4
SCHEMA = "samtokedit-ifbench-baseline-rejudge-v1"
RULE = "one more call of the same Judge for answers that failed (agreed rule, docs/03 §12.8)"


def ours():
    spec = importlib.util.spec_from_file_location("ifbench_judge", Path(__file__).with_name("judge.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read(path):
    return json.loads(Path(path).read_text())


def bundle_context(bundle):
    ready = read(bundle / "READY.json")
    config = read(bundle / "configuration.json")["judge_config"]
    manifest = Path(canonical.__file__).resolve().parents[3] / "data/v2_2.6.0/cases.jsonl"
    if sha256_file(manifest) != MANIFEST_SHA256:
        raise ValueError("Benchmark manifest differs from release 2.6.0")
    cases = {c["id"]: c for c in read_jsonl(manifest)}
    return ready, config, cases


def completed_nodes(results, ready):
    done = {}
    for node in range(NODES):
        path = results / "nodes" / f"node-{node}" / "COMPLETED.json"
        if path.exists():
            value = read(path)
            if value.get("status") == "completed":
                if value["plan_digest"] != ready["preparation_plan_digest"]:
                    raise ValueError(f"Node {node} completed another plan")
                done[node] = value
    return done


def failed_answers(results, node, ready, config, cases):
    """Every Judge key of one machine whose benchmark answer is not scored, with the logical requests it serves; the
    output must still match its generation record and the canonical binding."""
    root = results / "nodes" / f"node-{node}" / "judge"
    if read(root / "config.json") != config:
        raise ValueError(f"Node {node} Judge config differs from the bundle's judge_config")
    if read(root / "plan_binding.json")["plan_sha256"] != ready["plan_sha256"]:
        raise ValueError(f"Node {node} Judge ran another plan")
    by_key = {}
    for b in read(root / "bindings.json"):
        if b["generation_status"] == "generated":
            by_key.setdefault(b["judge_key"], []).append(b)
    failed = []
    for key, bound in sorted(by_key.items()):
        path = root / "effective_records" / f"{key}.json"
        original = read(path)                     # a completed node has an effective record for every key
        if original["status"] == "scored":
            continue
        first = bound[0]
        case = cases[first["case_id"]]
        for b in bound:
            if sha256_file(b["result_path"]) != b["generation_record_sha256"]:
                raise ValueError(f"{b['key']}: generation record changed after the Judge")
            if b["output_sha256"] != first["output_sha256"] or b["case_id"] != first["case_id"]:
                raise ValueError(f"{key}: one Judge key binds different outputs")
        if sha256_file(first["output_image"]) != first["output_sha256"]:
            raise ValueError(f"{first['output_image']} changed after generation")
        if canonical.binding(case, first["output_sha256"], config) != key:
            raise ValueError(f"{key}: the canonical binding differs")
        failed.append({
            "judge_key": key, "node": node, "case_id": first["case_id"], "unit_ids": [u["id"] for u in case["units"]],
            "logical": [{k: b[k] for k in ("key", "model", "setting", "case_id")} for b in bound],
            "output_image": first["output_image"], "output_sha256": first["output_sha256"],
            "original": {"effective_record": str(path), "effective_sha256": sha256_file(path),
                         "status": original["status"], "reason": original.get("reason"),
                         "repair_reason": original.get("repair_reason"),
                         "finish_reason": (original.get("raw_response") or {}).get("finish_reason")}})
    return failed


def overlay(output, failed, run_id, nodes, results):
    """OVERLAY.json from the failed keys and whatever our re-judge has answered so far."""
    effectives = output / "judge" / run_id / "effective"
    entries = []
    for item in failed:
        path = effectives / f"{item['judge_key']}.json"
        value = read(path) if path.exists() else None
        answer = {"status": value["status"] if value else "judge-pending"}
        if value:
            answer.update(effective_record=str(path), effective_sha256=sha256_file(path),
                          deterministic_repair_used=value.get("deterministic_repair_used", False))
            if value["status"] == "scored":
                validate_scores(value["scores"], item["unit_ids"])
                answer["scores"] = value["scores"]
            else:
                answer["reason"] = value.get("reason")
        entries.append({k: v for k, v in item.items() if k != "output_image"} | {"rejudge": answer})
    result = {"schema": SCHEMA, "rule": RULE, "results_root": str(results), "judge_run_id": run_id,
              "nodes": sorted(nodes), "entries": entries,
              "statuses": {s: sum(e["rejudge"]["status"] == s for e in entries)
                           for s in sorted({e["rejudge"]["status"] for e in entries})}}
    write_json(output / "OVERLAY.json", result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--node", type=int, action="append", choices=range(NODES),
                   help="machines to take (default: every completed one)")
    p.add_argument("--collect-only", action="store_true", help="write OVERLAY.json without Judge calls")
    a = p.parse_args(argv)
    bundle, output = a.bundle.resolve(), a.output.resolve()
    ready, config, cases = bundle_context(bundle)
    results = Path(ready["results_root"])
    done = completed_nodes(results, ready)
    nodes = sorted(set(a.node or done))
    if missing := [n for n in nodes if n not in done]:
        raise SystemExit(f"Benchmark machines {missing} have not completed")
    # Keep the machines an earlier invocation took, so OVERLAY.json always covers all of them.
    previous = read(output / "OVERLAY.json")["nodes"] if (output / "OVERLAY.json").exists() else []
    nodes = sorted(set(nodes) | set(previous))
    failed = [item for node in nodes for item in failed_answers(results, node, ready, config, cases)]
    run_id = digest(config)
    output.mkdir(parents=True, exist_ok=True)
    if not a.collect_only:
        rows = [{"request_id": item["logical"][0]["key"], "key": item["judge_key"], "case_id": item["case_id"],
                 "record": {"output_image": item["output_image"], "output_sha256": item["output_sha256"]}}
                for item in failed]
        ours().judge(output, config, cases, rows, "".join(map(str, nodes)))
    result = overlay(output, failed, run_id, nodes, results)
    print(json.dumps({"nodes": result["nodes"], "failed_keys": len(failed), "statuses": result["statuses"]}))


if __name__ == "__main__":
    main()
