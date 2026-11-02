"""Assemble the final run of the method for one more, uniform Judge pass: every request with its final tokens and output.

The final output of a request is the main run's, unless an overlay run regenerated it (the overlay then has a Pass1
record for it: a format repair, or a rerun after a hardware fault); outputs are referenced, not copied.  RUN.json is the
main run's when the bundle is the same, else rebuilt from the given bundle with ``prepare`` (the public requests are the
same bytes).  The cluster entry then finds every request terminal and only runs the Judge.

    PYTHONPATH=<v4>/src:<bundle>/repo/src python scripts/ifbench/build_final_run.py --run <main run> \\
        --overlay <run> ... --bundle <bundle> --config configs/ifbench/main_row.json --output <final run>
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from samtok_edit21.evaluation import ifbench

p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
p.add_argument("--run", type=Path, required=True)
p.add_argument("--overlay", type=Path, action="append", default=[])
p.add_argument("--bundle", type=Path, required=True)
p.add_argument("--config", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
main = ifbench.load_run(a.run)
if a.output.exists():
    raise SystemExit(f"{a.output} exists")
if Path(main["bundle"]) == a.bundle.resolve():
    a.output.mkdir(parents=True)
    (a.output / "RUN.json").write_text((a.run / "RUN.json").read_text())
else:
    ifbench.prepare(a.bundle, a.output, a.config, expect=len(main["requests"]))
plan = ifbench.load_run(a.output)
if [r["request_id"] for r in plan["requests"]] != [r["request_id"] for r in main["requests"]]:
    raise SystemExit("The bundle gives other requests than the main run")
source, replaced = {}, []
for item in plan["requests"]:
    source[item["request_id"]] = a.run
    for overlay in a.overlay:
        if (ifbench.request_dir(overlay, "pass1", item) / "result.json").exists():
            source[item["request_id"]] = overlay
    if source[item["request_id"]] != a.run:
        replaced.append({"request_id": item["request_id"], "from": str(source[item["request_id"]])})
def copy(item):
    origin = source[item["request_id"]]
    for stage in ("pass1", "pass2"):
        record = json.loads((ifbench.request_dir(origin, stage, item) / "result.json").read_text())
        if stage == "pass2":     # judge.py verifies every output's SHA before scoring it
            if record["status"] != "generated" or not Path(record["output_image"]).is_file():
                raise SystemExit(f"{item['request_id']}: no generated output in {origin}")
        elif record["status"] != "tokens":
            raise SystemExit(f"{item['request_id']}: no tokens in {origin}")
        ifbench.create_json(ifbench.request_dir(a.output, stage, item) / "result.json", dict(record, final_from=str(origin)))


with ThreadPoolExecutor(max_workers=32) as pool:      # small files on the shared disk: latency, not bandwidth
    list(pool.map(copy, plan["requests"]))
ifbench.create_json(a.output / "FINAL_SOURCES.json", {"main": str(a.run), "overlays": [str(o) for o in a.overlay],
                                                       "replaced": replaced})
print(json.dumps({"requests": len(plan["requests"]), "replaced": replaced}, ensure_ascii=False))
