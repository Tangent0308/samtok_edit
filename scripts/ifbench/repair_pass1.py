"""Format repair of Pass1 answers in an IF-Bench repair run (run dir with RUN.json and the ``pass1`` stage done).

An answer that failed the strict parser only because a "mask_2d" key name is missing gets its spans back
(``recover_missing_key``: the model's own tokens, nothing changed or added); the strict record is kept as
``result.strict.json``.  Any other failure stays a text-only fallback.  Run before ``pass2``:

    PYTHONPATH=<v4>/src python scripts/ifbench/repair_pass1.py --run <repair run>
"""
import argparse
import json
import os
from pathlib import Path

from samtok_edit21.evaluation import ifbench

p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
p.add_argument("--run", type=Path, required=True)
a = p.parse_args()
plan = ifbench.load_run(a.run)
for item in plan["requests"]:
    directory = ifbench.request_dir(a.run, "pass1", item)
    record = json.loads((directory / "result.json").read_text())
    if record["status"] != "tokens" or not record.get("pass1") or record["pass1"]["complete"]:
        print(item["request_id"], "unchanged")
        continue
    labels = [t["label"] for t in record["pass1"]["request"]["targets"]]
    spans = ifbench.recover_missing_key(record["pass1"]["raw"], labels)
    if spans is None:
        print(item["request_id"], "not repairable: stays text-only for", record["fallback_units"], "units")
        continue
    os.replace(directory / "result.json", directory / "result.strict.json")
    repaired = []
    for unit in record["units"]:
        if unit["source"] == "fallback_text":
            unit.update(token=spans[unit["id"]], source="pass1_format_repair")
            repaired.append(unit["id"])
    record.update(fallback_units=0, format_repair={"rule": "missing mask_2d key name", "units": repaired,
                                                    "strict_record": str(directory / "result.strict.json")})
    ifbench.create_json(directory / "result.json", record)
    print(item["request_id"], "repaired units", repaired)
