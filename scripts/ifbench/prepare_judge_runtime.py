"""Unpack the frozen IF-Bench Judge runtime with the benchmark's own bootstrap from the bundle's code snapshot: archive
SHA check, relocation, the wrapper with the snapshot on its PYTHONPATH, CPU import preflight of the pinned versions.

    python prepare_judge_runtime.py --bundle <bundle> --work <node-local dir>     # the wrapper: <work>/judge/python
"""
import argparse
import importlib.util
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--bundle", type=Path, required=True)
parser.add_argument("--work", type=Path, required=True)
args = parser.parse_args()
repo = args.bundle / "repo"
spec = importlib.util.spec_from_file_location(
    "bootstrap_fairness_dev20", repo / "src/evaluation/protocol_audit/bootstrap_fairness_dev20.py")
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)
descriptor = json.loads((args.bundle / "bootstrap.json").read_text())["judge"]
args.work.mkdir(parents=True, exist_ok=True)
python, _, proof = bootstrap.prepare_runtime("judge", descriptor, args.work, repo)
print(json.dumps({"judge_python": str(python), "packages": proof["packages"]}))
