"""The ablation plan (scripts/ablation/plan.py) and the in-place row replacement of build_data.py."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    sys.path.insert(0, str(ROOT / "scripts/ablation"))
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/ablation/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def value(args, flag):
    return args[len(args) - 1 - args[::-1].index(flag) + 1]   # the last occurrence wins, as in argparse


def test_every_arm_has_both_stages_with_32_rows_per_update():
    plan = load("plan")
    for world in (32, 8, 4):
        for arm in plan.ARMS:
            read, edit = plan.train_args(arm, "read", world), plan.train_args(arm, "edit", world)
            assert int(value(read, "--accumulation")) * world == 32
            assert "--init-adapter" not in read                           # Read starts from the pretrained DiT
            assert value(edit, "--init-adapter") == str(plan.adapter_of(arm, "read"))
            # Read: S1's first 1,000 updates (its 6,000-update schedule, stopped); Edit: 750 on its own schedule
            assert value(read, "--updates") == "6000" and value(read, "--stop-after") == "1000"
            assert value(read, "--warmup") == "50" and value(read, "--checkpoints").split(",")[-1] == "1000"
            assert value(edit, "--updates") == "750" and "--stop-after" not in edit
            assert value(edit, "--init-adapter").endswith("/read/adapter_1000")
    ref = plan.train_args("ref", "edit", 32)
    assert value(ref, "--rows") == "rows_s2.jsonl" and value(ref, "--lambda-region") == "1"
    assert value(plan.train_args("no_region", "edit", 32), "--lambda-region") == "0"
    assert value(plan.train_args("no_region", "read", 32), "--lambda-region-hl") == "0"
    assert value(plan.train_args("no_attention", "read", 32), "--lambda-read") == "0"
    assert value(plan.train_args("no_neighbor", "edit", 32), "--lambda-distractor") == "0"
    assert "--no-code-injector" in plan.train_args("no_injector", "read", 32)
    assert value(plan.train_args("no_multi", "read", 32), "--rows") == "rows_s1.jsonl"
    assert value(plan.train_args("no_identification", "read", 32), "--rows") == "rows_abl_noid_s1_r1000.jsonl"
    assert value(plan.train_args("no_twins", "read", 32), "--rows") == "rows_abl_notwin_s1_r1000.jsonl"
    assert str(plan.ROOT).endswith("ablation_20261112/v2") and plan.DATA.name == "data"
    assert value(plan.train_args("no_multi", "edit", 32), "--rows") == "rows_abl_nomulti_s2.jsonl"
    scale = plan.train_args("s50", "scale", 32)
    assert value(scale, "--init-adapter") == str(plan.S1_ADAPTER) and value(scale, "--updates") == "8000"
    assert value(scale, "--checkpoints").split(",")[-1] == "8000"


def test_systems_have_references_and_settings():
    plan = load("plan")
    systems = plan.systems()
    assert list(systems) == plan.FIRST + plan.LATER
    for name, (adapter, settings, overrides, reference) in systems.items():
        assert set(settings) <= set(plan.SET10)
        if reference is not None:
            assert set(settings) <= set(systems[reference][1]), name
    assert systems["final_notoken"][2] == {"drop_tokens": True}
    assert systems["final_g1"][2] == {"token_guidance": 1.0}
    assert systems["s100"][0] == plan.S2_ADAPTER and systems["s25"][1] == plan.SET10
    assert set(plan.probes()) == {"base", "final", *(f"abl_{a}" for a in plan.ARMS), *plan.SCALE, "s100"}


def test_replaced_keeps_positions_and_pairs():
    build = load("build_data")
    rows = ([{"kind": "hl", "pair_key": "p"}, {"kind": "hl", "pair_key": "p"},
             {"kind": "edit_merged", "id": "a", "pair_kind": "twin", "partners": ["b"]},
             {"kind": "edit_merged", "id": "b", "pair_kind": "twin", "partners": ["a"]},
             {"kind": "edit_merged", "id": "m", "order": [1, 0], "forms": ["noref_token", "ref_token"]},
             {"kind": "edit_merged", "id": "s", "form": "noref_token"}]
            + [{"kind": "edit_merged", "id": f"t{i}", "form": "ref_token"} for i in range(4)])
    try:
        build.replaced(rows, 6, lambda r: True, build.is_single)
        raise AssertionError("replaced must refuse when the pool is short and cycle is off")
    except RuntimeError:
        pass
    out, count = build.replaced(rows, 6, lambda r: True, build.is_single, cycle=True)
    assert count == 6 and [r["id"] for r in out] == ["t0", "t1", "t2", "t3", "t0", "t1"]
    out, count = build.replaced(rows, 6, build.is_twin, build.is_single)
    assert count == 2 and [r["id"] for r in out[2:4]] == ["t0", "t1"] and out[:2] == rows[:2] and out[4:] == rows[4:6]
    out, count = build.replaced(rows, 6, lambda r: r["kind"] in build.HIGHLIGHT, build.is_edit_not_twin)
    assert count == 2 and not any(r["kind"] == "hl" for r in out) and out[2:] == rows[2:6]
    out, _ = build.replaced(rows, 6, build.is_multi, build.is_single)
    assert out[4]["id"] == "t0" and not any(map(build.is_multi, out))
