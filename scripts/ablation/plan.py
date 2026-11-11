"""Ablation plan of the paper (2026-11-12): the single source of truth of ``scripts/ablation/run_4n.sh``.

Three blocks, all judged on IF-Bench in one Judge session (``judge_many.py``) and probed on the development set:

* component arms (reduced protocol): from the pretrained DiT, Read 250 then Edit 750 updates x 32 rows, the first
  rows of the formal ``rows_s1`` / ``rows_s2`` (so ``ref`` is the formal recipe, shortened); every other arm changes one
  thing: its rows (``build_data.py``) or one option of the trainer;
* data scale: the Edit stage from S1@6000 on 25 % / 50 % of the merged records (whole source-image groups, nested), the
  same draw shares and epochs as S2 (4,000 / 8,000 updates); 100 % is the formal S2@16000;
* the final model (R2@1000) with its own diagnostics: Space Tokens removed from the prompt, token guidance off.

    python plan.py train-args <arm> <stage> --world W     # round2-train options (stage: read | edit | scale)
    python plan.py list <train-arms | scale-arms | probe | ifbench-first | ifbench-scale | ifbench-all>
    python plan.py probe-args <system>                     # round2-eval options of a probe system
    python plan.py prepare-run <system> --repo <checkout>  # node 0: RUN.json + shared Pass1 records of a system
    python plan.py check-inputs [--stage start|scale]      # every input of the run exists
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

NAS = Path("/mnt/bn/strategy-mllm-train/user/tanyue")
SE = NAS / "experiments2/SAMTokEdit"
# ABL_ROOT / ABL_DATA / ABL_CASES / ABL_SMOKE exist only for the one-machine rehearsal (scripts/ablation/rehearse.sh).
ROOT = Path(os.environ.get("ABL_ROOT", SE / "ablation_20261112"))
DATA = Path(os.environ.get("ABL_DATA", SE / "ablation_20261112/data"))   # build_data.py: links to ORIG_DATA + rows
ORIG_DATA = SE / "qwen21_v3/data/round2_scaleup_20261030"
S1_ADAPTER = SE / "qwen21_v3/runs/s1_4n_001/train/adapter_6000"
S2_ADAPTER = SE / "qwen21_v3/runs/s2_4n_001/train/adapter_16000"
FINAL_ADAPTER = SE / "quality_anchor_20261104/runs_v5/v6_full/r2/train/adapter_1000"
IFBENCH_BUNDLE = NAS / ".ifbench-tmp/one-mp-full4-20261103-judgeio"
R2_RUN = SE / "ifbench_r2_full_20261110"                  # Pass1 records reused: Pass1 does not depend on the editor
DEV_CASES = SE / "quality_anchor_20261104/ifbench_cases80.txt"
CASES = Path(os.environ.get("ABL_CASES", SE / "ablation_20261112/cases200.txt"))   # select_cases.py
SMOKE = os.environ.get("ABL_SMOKE") == "1"
SET4 = ["S1", "S2-mask", "S3-mask", "S4-noref"]
SET10 = ["S1", "S2-point", "S2-box", "S2-mask", "S3-point", "S3-box", "S3-mask", "S4-ref", "S4-noref", "S5"]

# ------------------------------------------------------------------------------------------------ training
# Losses of S1/S2 (code 47c0efd, whose defaults these are): region 1 on every row, neighbour preservation 1, write/read
# routing 0.05, hard negatives 8, one token of context around edit regions, overlay highlights.
LOSS = ["--heavy-weight", "8", "--lambda-region", "1", "--lambda-distractor", "1", "--lambda-route", "0.05",
        "--lambda-read", "0.05", "--edit-region-dilate", "1", "--highlight-style", "overlay", "--rank", "32"]
STAGES = {
    "read": {"rows": "rows_s1.jsonl", "updates": 250, "warmup": 25, "lr": "1e-4", "lr_min": "1e-5"},
    "edit": {"rows": "rows_s2.jsonl", "updates": 750, "warmup": 50, "lr": "5e-5", "lr_min": "5e-6"},
}
# In training order: the reference first, then the arms of the main ablation table, then the rest.
ARMS = {
    "ref": {},
    "no_identification": {"rows": {"read": "rows_abl_noid_s1.jsonl", "edit": "rows_abl_noid_s2.jsonl"}},
    "no_attention": {"args": ["--lambda-route", "0", "--lambda-read", "0"]},
    "no_twins": {"rows": {"read": "rows_abl_notwin_s1.jsonl", "edit": "rows_abl_notwin_s2.jsonl"}},
    "no_region": {"args": ["--lambda-region", "0", "--lambda-region-hl", "0"]},
    "no_injector": {"args": ["--no-code-injector"]},
    "no_multi": {"rows": {"edit": "rows_abl_nomulti_s2.jsonl"}},
    "no_neighbor": {"args": ["--lambda-distractor", "0"]},
}
SCALE = {
    "s25": {"fraction": 0.25, "updates": 4000, "rows": "rows_abl_s25.jsonl"},
    "s50": {"fraction": 0.50, "updates": 8000, "rows": "rows_abl_s50.jsonl"},
}
SCALE_SCHEDULE = {"warmup": 100, "lr": "5e-5", "lr_min": "5e-6"}          # as S2
CHECKPOINT_EVERY = {"read": 250, "edit": 250, "scale": 1000}             # resume points of an interrupted stage
if SMOKE:                                                                  # rehearsal: two updates per stage
    for _spec in list(STAGES.values()) + list(SCALE.values()):
        _spec.update(updates=2, warmup=1)
    SCALE_SCHEDULE = dict(SCALE_SCHEDULE, warmup=1)
    CHECKPOINT_EVERY = {"read": 1, "edit": 1, "scale": 1}


def train_dir(arm, stage):
    return ROOT / ("scale" if stage == "scale" else "train") / arm / ("train" if stage == "scale" else stage)


def final_update(arm, stage):
    return SCALE[arm]["updates"] if stage == "scale" else STAGES[stage]["updates"]


def adapter_of(arm, stage):
    return train_dir(arm, stage) / f"adapter_{final_update(arm, stage)}"


def train_args(arm, stage, world):
    if 32 % world:
        raise ValueError(f"32 rows per update do not split over {world} GPUs")
    if stage == "scale":
        spec = SCALE[arm]
        rows, updates, schedule, init, extra = spec["rows"], spec["updates"], SCALE_SCHEDULE, S1_ADAPTER, []
    else:
        config = ARMS[arm]
        rows = config.get("rows", {}).get(stage, STAGES[stage]["rows"])
        updates, schedule, extra = STAGES[stage]["updates"], STAGES[stage], config.get("args", [])
        init = adapter_of(arm, "read") if stage == "edit" else None
    every = CHECKPOINT_EVERY[stage]
    checkpoints = ",".join(str(u) for u in range(every, updates + 1, every))
    args = ["--data", str(DATA), "--output", str(train_dir(arm, stage)), "--rows", rows,
            "--quick-items", "eval_c2_quick_idx.json", "--updates", str(updates), "--accumulation", str(32 // world),
            "--warmup", str(schedule["warmup"]), "--lr", schedule["lr"], "--lr-schedule", "cosine",
            "--lr-min", schedule["lr_min"], "--checkpoints", checkpoints, "--quick-eval", str(updates),
            "--wandb-mode", "disabled", *LOSS, *extra]
    if SMOKE:
        args += ["--quick-limit", "1"]
    if init is not None:
        args += ["--init-adapter", str(init)]
    return args


# ---------------------------------------------------------------------------------------------- evaluation
def systems():
    """IF-Bench systems in generation order: ``name -> (adapter, settings, method overrides, reference)``."""
    out = {"final": (FINAL_ADAPTER, SET4, {}, None),
           "final_notoken": (FINAL_ADAPTER, ["S1", "S2-mask"], {"drop_tokens": True}, "final"),
           "final_g1": (FINAL_ADAPTER, SET4, {"token_guidance": 1.0}, "final")}
    for arm in ARMS:
        out[f"abl_{arm}"] = (adapter_of(arm, "edit"), SET4, {}, None if arm == "ref" else "abl_ref")
    for arm in SCALE:
        out[arm] = (adapter_of(arm, "scale"), SET10, {}, "s100")
    out["s100"] = (S2_ADAPTER, SET10, {}, None)
    return out


def probes():
    """Development-set probe systems: ``name -> adapter`` (None: the pretrained DiT)."""
    out = {"base": None, "final": FINAL_ADAPTER}
    out.update({f"abl_{arm}": adapter_of(arm, "edit") for arm in ARMS})
    out.update({arm: adapter_of(arm, "scale") for arm in SCALE})
    out["s100"] = S2_ADAPTER
    return out


FIRST = ["final", "final_notoken", "final_g1"] + [f"abl_{arm}" for arm in ARMS]
LATER = list(SCALE) + ["s100"]


def probe_args(name):
    adapter = probes()[name]
    args = ["--data", str(DATA), "--output", str(ROOT / "probe" / name), "--steps", "20"]
    if SMOKE:
        args += ["--limit", "1"]
    return args + (["--adapter", str(adapter)] if adapter is not None else [])


def read_cases():
    cases = [line.strip() for line in CASES.read_text().splitlines() if line.strip()]
    if (len(cases) != 200 and not SMOKE) or cases != sorted(set(cases)):
        raise ValueError(f"{CASES} must hold 200 sorted distinct case ids")
    return cases


def copy_pass1(item, shared):
    """Copy one request's Pass1 record (result.json and decoded unit masks, not the claim) into the shared tree."""
    src = R2_RUN / "pass1" / item["setting"] / item["case_id"]
    dst = shared / item["setting"] / item["case_id"]
    record = json.loads((src / "result.json").read_text())
    if record["request_id"] != item["request_id"] or record["public_sha256"] != item["public_sha256"]:
        raise ValueError(f"Pass1 record {src} is not the request {item['request_id']}")
    if (dst / "result.json").exists():
        return record["status"]
    dst.mkdir(parents=True, exist_ok=True)
    for path in sorted(src.iterdir()):
        if path.is_file() and path.name != "result.json":
            shutil.copy2(path, dst / path.name)
    shutil.copy2(src / "result.json", dst / "result.json.tmp")
    (dst / "result.json.tmp").replace(dst / "result.json")       # last: a partial copy is never terminal
    return record["status"]


def prepare_run(name, repo):
    """RUN.json of one system (``ifbench.prepare``) and its Pass1 records, shared by all systems."""
    sys.path.insert(0, str(Path(repo) / "src"))
    from samtok_edit21.evaluation import ifbench

    adapter, settings, overrides, _ = systems()[name]
    run = ROOT / "ifbench" / name
    method = json.loads((Path(repo) / "configs/ifbench/main_row.json").read_text())
    method.update(name=f"ablation_{name}", dit_adapter=str(adapter), **overrides)
    run.mkdir(parents=True, exist_ok=True)
    ifbench.create_json(run / "method.json", method)
    cases = read_cases()
    plan = ifbench.prepare(IFBENCH_BUNDLE, run, run / "method.json", cases, settings, expect=len(cases) * len(settings))
    shared = ROOT / "ifbench" / "_pass1"
    link = run / "pass1"
    if not link.exists():
        link.symlink_to(shared, target_is_directory=True)
    elif link.resolve() != shared.resolve():
        raise ValueError(f"{link} is not the shared Pass1 tree")
    with ThreadPoolExecutor(max_workers=16) as pool:
        states = list(pool.map(lambda item: copy_pass1(item, shared), plan["requests"]))
    if any(s != "tokens" for s in states):
        raise ValueError(f"{name}: {sum(s != 'tokens' for s in states)} Pass1 records without tokens")
    # Claims left by a stopped submission (no worker runs now): logged release, same input and seed.
    for stage, request in ifbench.open_claims(run):
        ifbench.release(run, stage, request, "ablation resubmission: the worker of the earlier submission stopped")
    print(json.dumps({"system": name, "requests": len(plan["requests"]), "settings": settings,
                      "adapter": str(adapter), "overrides": overrides}))


def check_inputs(stage):
    """Everything the run reads exists (and the rows hold enough lines) before any GPU work starts."""
    missing = []
    need = [DATA / name for name in ("eval_c1.json", "eval_c2_quick_idx.json", "edit_tokens.jsonl", "pool",
                                     "dev_split.json")]
    need += [S1_ADAPTER / "adapter.safetensors", S2_ADAPTER / "adapter.safetensors", FINAL_ADAPTER / "adapter.safetensors",
             IFBENCH_BUNDLE / "configuration.json", R2_RUN / "pass1", CASES]
    rows = {STAGES[s]["rows"]: STAGES[s]["updates"] * 32 for s in STAGES}
    for arm in ARMS.values():
        for s, name in arm.get("rows", {}).items():
            rows[name] = STAGES[s]["updates"] * 32
    if stage in ("start", "scale"):
        rows.update({spec["rows"]: spec["updates"] * 32 for spec in SCALE.values()})
    for path in need:
        if not path.exists():
            missing.append(str(path))
    for name, needed in rows.items():
        path = DATA / name
        if not path.exists():
            missing.append(str(path))
            continue
        with open(path) as f:
            count = sum(1 for _ in f)
        if count < needed:
            missing.append(f"{path} has {count} rows, {needed} needed")
    if CASES.exists():
        read_cases()
    if missing:
        raise SystemExit("Missing inputs:\n  " + "\n  ".join(missing))
    print(json.dumps({"inputs": "ok", "rows_files": len(rows)}))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("action", choices=["train-args", "train-dir", "list", "probe-args", "prepare-run", "check-inputs",
                                      "adapter", "runs"])
    p.add_argument("names", nargs="*")
    p.add_argument("--world", type=int, default=32)
    p.add_argument("--repo", type=Path)
    p.add_argument("--stage", default="start")
    a = p.parse_args(argv)
    if a.action == "train-args":
        print(" ".join(train_args(a.names[0], a.names[1], a.world)))
    elif a.action == "adapter":
        print(adapter_of(a.names[0], a.names[1]))
    elif a.action == "train-dir":
        print(train_dir(a.names[0], a.names[1]))
    elif a.action == "runs":
        print(" ".join(str(ROOT / "ifbench" / name) for name in FIRST + LATER))
    elif a.action == "list":
        what = a.names[0]
        values = {"train-arms": list(ARMS), "scale-arms": list(SCALE), "probe-first": ["base", "final"] +
                  [f"abl_{arm}" for arm in ARMS], "probe-scale": LATER, "ifbench-first": FIRST,
                  "ifbench-scale": LATER, "ifbench-all": FIRST + LATER}[what]
        print(" ".join(values))
    elif a.action == "probe-args":
        print(" ".join(probe_args(a.names[0])))
    elif a.action == "prepare-run":
        prepare_run(a.names[0], a.repo)
    else:
        check_inputs(a.stage)
    return 0


if __name__ == "__main__":
    sys.exit(main())
