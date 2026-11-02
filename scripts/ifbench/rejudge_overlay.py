"""Build an overlay run that judges once more every request still without a valid Judge answer.

Reads the main run and its overlay runs in order (as ``ifbench_report --overlay`` applies them), takes each request whose
current entry is not scored, and writes a new run directory holding those requests with their current generation
records (the outputs are referenced, not copied).  Judge it with ``scripts/ifbench/judge.py --run <new run>`` and pass it
as the last ``--overlay`` of the merge.  The agreed rule (docs/03 §12.8): one more call of the same Judge per failed answer.

    python scripts/ifbench/rejudge_overlay.py --run <main run> [--overlay <run> ...] --output <new run dir>
"""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
p.add_argument("--run", type=Path, required=True)
p.add_argument("--overlay", type=Path, action="append", default=[])
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
plan = json.loads((a.run / "RUN.json").read_text())
current = {r["request_id"]: (a.run, r) for r in json.loads((a.run / "REPORT.json").read_text())["requests"]}
for overlay in a.overlay:
    for entry in json.loads((overlay / "REPORT.json").read_text())["requests"]:
        if entry["status"] != "scored" and current[entry["request_id"]][1]["status"] == "scored":
            raise SystemExit(f"{overlay}: {entry['request_id']} would replace a scored entry")
        current[entry["request_id"]] = (overlay, entry)
failed = [(run, entry) for run, entry in current.values() if entry["status"] != "scored"]
by_id = {r["request_id"]: r for r in plan["requests"]}
if a.output.exists():
    raise SystemExit(f"{a.output} exists")
requests = []
for run, entry in failed:
    if entry["generation_status"] != "generated":
        continue                      # generation failures stay failures; only Judge answers are redone
    item = by_id[entry["request_id"]]
    record = json.loads((run / "outputs" / item["setting"] / item["case_id"] / "result.json").read_text())
    target = a.output / "outputs" / item["setting"] / item["case_id"]
    target.mkdir(parents=True)
    (target / "result.json").write_text(json.dumps(dict(record, rejudge_of=str(run)), ensure_ascii=False, indent=1))
    requests.append(item)
new = dict(plan, requests=requests, case_ids=sorted({r["case_id"] for r in requests}),
           settings=[s for s in plan["settings"] if any(r["setting"] == s for r in requests)], repair_of=str(a.run),
           repair_reason="one more call of the same Judge for answers that failed (agreed rule, docs/03 §12.8)",
           rejudge_sources=[str(x) for x in [a.run, *a.overlay]])
(a.output / "RUN.json").write_text(json.dumps(new, ensure_ascii=False, indent=1))
print(json.dumps({"requests": len(requests), "by_source": {str(r): sum(1 for x, _ in failed if x == r) for r in [a.run, *a.overlay]}}))
