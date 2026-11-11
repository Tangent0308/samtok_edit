"""The ablation's single Judge session: the canonical IF-Bench Judge of ``scripts/ifbench/judge.py`` over every run, one
engine per machine for all of them.

``judge.py`` is used unchanged (same jobs, keys, admissions, write-once records and reports per run); only its
``Backend`` is created once per process and handed to every run, so all systems are scored by the same engine with the
same bundle configuration.  A machine judges the images ``case_ids[node::nodes]`` of every run (the runs share one sorted
case list, so an image of every system goes to the same engine).  Run with the frozen Judge runtime's wrapper, outside
any repository directory:

    <judge runtime>/python <repo>/scripts/ablation/judge_many.py --runs <run> ... [--node 0 --nodes 4] [--report-only]
"""
import argparse
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ifbench_judge", HERE.parent / "ifbench/judge.py")
judge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(judge)


def shared_backend():
    """Replace ``canonical.Backend`` by a factory that builds the engine on first use and then returns it."""
    original, cache = judge.canonical.Backend, {}

    def backend(args, *rest, **kwargs):
        if "engine" not in cache:
            cache["engine"] = original(args, *rest, **kwargs)
            cache["args"] = vars(args)
        elif vars(args) != cache["args"]:
            raise ValueError("Runs ask for different Judge engine arguments")
        return cache["engine"]

    judge.canonical.Backend = backend


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--node", type=int, default=0)
    p.add_argument("--nodes", type=int, default=1)
    p.add_argument("--report-only", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="List what would be judged; load no engine")
    a = p.parse_args()
    runs = [r.resolve() for r in a.runs]
    contexts = [judge.context(run) for run in runs]
    if len({json.dumps(c[1], sort_keys=True) for c in contexts}) != 1:
        raise SystemExit("The runs use different Judge configurations")
    case_lists = {tuple(plan["case_ids"]) for plan, _, _ in contexts}
    if len(case_lists) != 1:
        raise SystemExit("The runs must share one case list (the machines' shares of images)")
    if a.report_only:
        for run, (plan, config, cases) in zip(runs, contexts):
            print(f"== {run.name}")
            judge.report(run, plan, cases, judge.jobs(run, plan, config, cases), judge.digest(config))
        return
    selected, work = None, []
    for run, (plan, config, cases) in zip(runs, contexts):
        selected = set(plan["case_ids"][a.node::a.nodes])
        rows = judge.jobs(run, plan, config, cases, selected)
        open_rows = [r["request_id"] for r in rows if r["generation_status"] not in judge.FAILURES | {"generated"}]
        if open_rows:
            raise SystemExit(f"{run.name}: generation not terminal for {open_rows[:10]} ({len(open_rows)} in all); "
                             "the Judge starts after all generation")
        work.append((run, config, cases, rows))
        print(json.dumps({"run": run.name, "node": a.node, "requests": len(rows),
                          "generated": sum(r["key"] is not None for r in rows)}), flush=True)
    if a.dry_run:
        return
    shared_backend()
    for run, config, cases, rows in work:
        judge.judge(run, config, cases, rows, a.node)
        print(json.dumps({"run": run.name, "node": a.node, "judged": True}), flush=True)


if __name__ == "__main__":
    main()
