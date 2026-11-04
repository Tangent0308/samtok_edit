"""IF-Bench 1MP evaluation of the full method: public-text token placement, Pass2 prompts, canvas coordinates."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from samtok_edit21.evaluation import ifbench
from samtok_edit21.evaluation.ifbench import pass2_prompt, place_token, unit_sentences
from samtok_edit21.pass1.protocol import box_normalized, point_normalized
from samtok_edit21.preparation import instruction_protocol_v2 as p2

ROOT = Path(__file__).resolve().parents[1]
T = "<|mt_start|><|mt_0001|><|mt_0257|><|mt_end|>"
U = "<|mt_start|><|mt_0002|><|mt_0258|><|mt_end|>"


def spec(operation, ref, payload="", target_part="", remove_mode="whole", attributes=(), placement=None):
    return {"version": "2.0", "operation": operation, "ref": ref, "scope": "object", "payload": payload,
            "target_part": target_part, "remove_mode": remove_mode, "attributes": list(attributes),
            "placement": placement or {}, "qualifier": ""}


SPECS = [
    spec("replace", "the outer sleeve of the woman on the left", "a flared bell sleeve"),
    spec("attribute", "the red coat of the woman on the right", "navy blue",
         attributes=[{"property": "color", "value": "navy blue"}]),
    spec("attribute", "the third bar chair from the front", "matte black", target_part="wooden frame",
         attributes=[{"property": "color", "value": "matte black"}]),
    spec("attribute", "the narrow package left of the blue bag", "amber",
         attributes=[{"property": "tint", "value": "amber"}]),
    spec("attribute", "the black board above the car", "RACE DAY",
         attributes=[{"property": "text", "value": "RACE DAY"}]),
    spec("remove", "the hood hanging down the person's back"),
    spec("remove", "the drummer's pink T-shirt", target_part="the white chest emblem", remove_mode="part"),
    spec("remove", "the woven basket in the foreground", remove_mode="contents"),
    spec("remove", "the floral skirt of the woman in the green top", remove_mode="hem"),
    spec("remove", "the jeans of the foremost man", remove_mode="lowerlegs"),
    spec("add", "the central girl's leather jacket", "one small round enamel pin",
         placement={"relation": "on", "site": "upper back", "anchor_mode": "support"}),
    spec("add", "the chair facing away", "one burgundy seat cushion",
         placement={"relation": "on", "anchor_mode": "support"}),
]


@pytest.mark.parametrize("s", SPECS, ids=lambda s: f"{s['operation']}-{s['remove_mode']}")
@pytest.mark.parametrize("has_ref", [True, False])
def test_public_text_placement_reproduces_the_renderer(s, has_ref):
    text = p2.render(s, has_ref)
    assert place_token(text, s["operation"], has_ref, T) == p2.render(s, has_ref, selector_token=T)


def test_last_with_rule_is_the_frozen_choice():
    # Known 3.7% case of the frozen rule: the payload's own " with " receives the token.
    s = spec("replace", "the old lamp", "a brass lamp with a green shade")
    text = p2.render(s, True)
    assert place_token(text, "replace", True, T) == f"Replace the old lamp with a brass lamp {T} with a green shade."


def test_noref_needs_exactly_one_selected_region():
    with pytest.raises(ValueError):
        place_token("Remove the red car.", "remove", False, T)


def public():
    return {"units": [
        {"id": "U1", "operation": "remove", "instruction": "Remove the red car on the left.", "has_ref": True,
         "locator": "none", "region": None},
        {"id": "U2", "operation": "replace", "instruction": "Replace the selected region with a stool.",
         "has_ref": False, "locator": "point", "region": {"coordinates": [10.5, 20.25]}},
    ]}


def test_units_without_a_token_stay_text_only():
    p = public()
    snapshot = copy.deepcopy(p)
    assert unit_sentences(p, {"U2": U}) == ["Remove the red car on the left.",
                                            f"Replace the selected region {U} with a stool."]
    assert p == snapshot
    assert pass2_prompt(p, {"U1": T, "U2": U}) == (
        "Execute all numbered instructions and keep everything else unchanged.\n"
        f"U1: Remove the red car on the left {T}.\nU2: Replace the selected region {U} with a stool.")


def test_canvas_coordinates_use_the_trained_formulas():
    assert point_normalized([10.5, 20.25], (896, 1120), continuous=True) == [12, 18]
    assert point_normalized([10, 20], (896, 1120), continuous=True) == point_normalized([10, 20], (896, 1120))
    assert box_normalized([1.2, 3.9, 500.01, 1119.5], (896, 1120), continuous=True) == [1, 3, 559, 1000]
    for bad in ([10.5, 20], [float("nan"), 3], [True, 3]):
        with pytest.raises(ValueError):
            point_normalized(bad, (896, 1120))
    with pytest.raises(ValueError):
        point_normalized([float("inf"), 3], (896, 1120), continuous=True)


def run_plan(tmp_path, cases=("a", "b", "c", "d", "e"), settings=("S1", "S5")):
    plan = {"schema": "samtok-ifbench-1mp-run-v1", "method": {}, "settings": list(settings), "case_ids": list(cases),
            "requests": [{"request_id": f"{c}--{s}", "setting": s, "case_id": c} for s in settings for c in cases]}
    (tmp_path / "RUN.json").write_text(json.dumps(plan))
    return plan


def test_each_machine_owns_all_settings_of_its_images(tmp_path):
    plan = run_plan(tmp_path)
    owners = {}
    for node in range(2):
        for worker in range(3):
            items = ifbench.worker_requests(plan, node, 2, worker, 3)
            assert sorted(i["request_id"] for i in items) == sorted(
                r["request_id"] for r in plan["requests"] if r["case_id"] in plan["case_ids"][node::2])
            for item in items:
                assert owners.setdefault(item["case_id"], node) == node
    assert set(owners) == set(plan["case_ids"])
    starts = [ifbench.worker_requests(plan, 0, 1, worker, 2)[0]["request_id"] for worker in range(2)]
    assert len(set(starts)) == 2


def test_claims_are_exclusive_and_a_stale_claim_needs_a_logged_release(tmp_path):
    plan = run_plan(tmp_path)
    item = plan["requests"][0]
    directory = ifbench.request_dir(tmp_path, "pass2", item)
    assert ifbench.request_state(tmp_path, "pass2", item) == "pending"
    assert ifbench.claim(directory, item, "pass2") is True
    assert ifbench.claim(directory, item, "pass2") is False
    assert ifbench.open_claims(tmp_path) == [("pass2", item["request_id"])]
    ifbench.release(tmp_path, "pass2", item["request_id"], "node lost")
    assert ifbench.open_claims(tmp_path) == []
    assert "node lost" in (tmp_path / "RELEASES.jsonl").read_text()
    assert ifbench.claim(directory, item, "pass2") is True
    ifbench.write_json(directory / "result.json", {"status": "generated"})
    assert ifbench.claim(directory, item, "pass2") is False
    with pytest.raises(ValueError):
        ifbench.release(tmp_path, "pass2", item["request_id"], "never")
    assert ifbench.status(tmp_path)["pass2"]["generated"] == 1


def test_failure_guard_stops_a_broken_worker():
    guard = ifbench.FailureGuard()
    guard.check({"request_id": "x", "status": "failed"})
    guard.check({"request_id": "x", "status": "generated"})
    for _ in range(ifbench.MAX_CONSECUTIVE_FAILURES - 1):
        guard.check({"request_id": "x", "status": "failed"})
    with pytest.raises(RuntimeError):
        guard.check({"request_id": "x", "status": "failed-planner"})
    with pytest.raises(RuntimeError):
        ifbench.FailureGuard().check({"request_id": "x", "status": "failed", "fatal_cuda_error": True})


def load_judge():
    pytest.importorskip("samtok_benchmark.ifbench.dev_judge")
    spec = importlib.util.spec_from_file_location("ifbench_judge", ROOT / "scripts/ifbench/judge.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def judge_answer(unit_ids, e):
    block = lambda score: {"score": score, "pass": score >= 3, "evidence": "seen"}  # noqa: E731
    units = [{"id": uid, "E": {"score": e, "pass": e == 4, "evidence": "seen"}, "P": block(4), "Q": block(3)}
             for uid in unit_ids]
    return {"text": json.dumps({"units": units, "global_P": block(4), "global_Q": block(4)}), "finish_reason": "stop",
            "prompt_tokens": 1, "output_tokens": 1}


def test_judge_scores_each_machines_outputs_once_and_keeps_full_denominators(tmp_path, monkeypatch):
    judge = load_judge()
    canonical = judge.canonical
    config = {"engine": {}, "decode": {}}
    calls = []

    class Backend:
        def __init__(self, args, images):
            calls.append("load")

        def generate(self, conversations):
            calls.extend(c["case"] for c in conversations)
            return [judge_answer(c["units"], 4) for c in conversations]

    monkeypatch.setattr(canonical, "Backend", Backend)
    monkeypatch.setattr(canonical, "fingerprint", lambda: config)
    monkeypatch.setattr(canonical, "binding", lambda case, sha, _: f"{case['id']}-{sha[:8]}")
    monkeypatch.setattr(canonical, "prompt", lambda case: case["id"])
    monkeypatch.setattr(canonical, "evidence", lambda case, _: {"case": case["id"],
                                                                "units": [u["id"] for u in case["units"]]})
    cases = {"a": {"id": "a", "units": [{"id": "U1"}, {"id": "U2"}]},
             "b": {"id": "b", "units": [{"id": "U1"}, {"id": "U2"}, {"id": "U3"}]}}
    plan = run_plan(tmp_path, cases=("a", "b"), settings=("S1", "S5"))
    outputs = {("a", "S1"): "1", ("a", "S5"): None, ("b", "S1"): "2", ("b", "S5"): "2"}   # b: one output for two settings
    for (case, setting), content in outputs.items():
        directory = tmp_path / "outputs" / setting / case
        directory.mkdir(parents=True)
        record = {"status": "failed"}
        if content:
            image = directory / "output.png"
            image.write_text(content)
            record = {"status": "generated", "output_image": str(image), "output_sha256": ifbench.sha256_file(image)}
        (directory / "result.json").write_text(json.dumps(record))
    for node in range(2):
        rows = judge.jobs(tmp_path, plan, config, cases, set(plan["case_ids"][node::2]))
        judge.judge(tmp_path, config, cases, rows, node)
    assert sorted(c for c in calls if c != "load") == ["a", "b"]    # one call per distinct output, each on its owner
    assert calls.count("load") == 2
    calls.clear()
    rows = judge.jobs(tmp_path, plan, config, cases)
    run_id = judge.digest(config)
    report = judge.report(tmp_path, plan, cases, rows, run_id)
    assert report["settings"]["S1"]["unit_pass"] == 5 and report["settings"]["S1"]["joint_pass"] == 2
    assert report["settings"]["S5"]["unit_pass"] == 3 and report["settings"]["S5"]["units"] == 5   # the failure stays in
    assert report["settings"]["S5"]["joint_pass"] == 1 and report["settings"]["S5"]["generation_failed_cases"] == 1
    # An admitted request without an answer is never resubmitted and stays missing, not zero.
    key = next(r["key"] for r in rows if r["case_id"] == "a" and r["key"])
    root = tmp_path / "judge" / run_id
    (root / "effective" / f"{key}.json").unlink()
    (root / "raw" / f"{key}.json").unlink()
    judge.judge(tmp_path, config, cases, judge.jobs(tmp_path, plan, config, cases), 0)
    assert calls == []
    report = judge.report(tmp_path, plan, cases, judge.jobs(tmp_path, plan, config, cases), run_id)
    assert report["settings"]["S1"]["missing_cases"] == 1


def test_merge_uses_the_benchmark_statistics_on_one_set_of_draws():
    stats = pytest.importorskip("samtok_benchmark.ifbench.fairness_dev_report")
    from samtok_benchmark.io import digest
    from samtok_edit21.evaluation import ifbench_report

    case_ids = [f"c{i}" for i in range(12)]
    ks = [2 + i % 4 for i in range(12)]
    boot = stats.Bootstrap(ks)
    rows, requests = [], []
    for setting in ifbench.SETTINGS:
        for model, shift in (("base", 0), ("other", 1)):
            up = np.array([max(0, k - 1 - shift) for k in ks])
            jp = np.array([int(i % 3 == 0) for i in range(12)])
            row, _ = stats.rates(up, jp, np.array(ks), np.zeros(12, bool), boot)
            row.update(model=model, setting=setting, per_case=[
                {"case_id": c, "K": k, "UnitPass_count": int(u), "JointPass": int(j)}
                for c, k, u, j in zip(case_ids, ks, up, jp)])
            rows.append(row)
        for i, (case, k) in enumerate(zip(case_ids, ks)):
            status = "failed" if i == 0 else "scored"
            requests.append({"case_id": case, "setting": setting, "status": status, "unit_pass": [True] * k,
                             "joint_pass": i % 2 == 0})
    benchmark = {"case_ids": case_ids, "settings": ifbench.SETTINGS, "model_order": ["base", "other"], "rows": rows,
                 "plan_digest": "p", "source_commit": "c",
                 "bootstrap": {"shared_indices_sha256": digest(boot.indices.tolist())}}
    ours = {"case_ids": case_ids, "method": {"name": "m"}, "requests": requests}
    merged = ifbench_report.merge(benchmark, ours)
    row = merged["rows"]["samtokedit"]["S1"]
    assert row["UnitPass"] == pytest.approx((sum(ks) - ks[0]) / sum(ks))        # the failed source counts as zero
    assert row["JointPass"] == pytest.approx(5 / 12)
    paired = merged["paired_samtokedit_minus_baseline"]["base"]["UnitPass"]["S1"]
    assert paired["value"] == pytest.approx(row["UnitPass"] - merged["rows"]["base"]["S1"]["UnitPass"])
    low, high = paired["bootstrap"]["CI95"]
    assert low <= paired["value"] <= high
    assert "samtokedit" in ifbench_report.markdown(merged)
    assert len(merged["derived"]["samtokedit"]) == 16
    # A baseline that does not reproduce its published row, or other draws, stops the merge.
    rows[0]["UnitPass"] += 0.01
    with pytest.raises(ValueError):
        ifbench_report.merge(benchmark, ours)
    rows[0]["UnitPass"] -= 0.01
    benchmark["bootstrap"]["shared_indices_sha256"] = "x"
    with pytest.raises(ValueError):
        ifbench_report.merge(benchmark, ours)
    # A request without a score leaves our row incomplete instead of counting as zero.
    benchmark["bootstrap"]["shared_indices_sha256"] = digest(boot.indices.tolist())
    requests[1]["status"] = "judge-pending"
    merged = ifbench_report.merge(benchmark, ours)
    assert merged["rows"]["samtokedit"]["S1"]["UnitPass"] is None
    assert "不完整" in ifbench_report.markdown(merged)


def test_generation_uses_the_exact_canvas_a_fresh_compile_state_and_token_guidance_only_with_tokens(tmp_path, monkeypatch):
    canvas = pytest.importorskip("samtok_benchmark.ifbench.canvas")
    import torch
    from PIL import Image
    import samtok_edit21.models.pipeline as pipeline

    geometry = canvas.geometry([1500, 1875])
    width, height = geometry["canvas_size"]
    source = tmp_path / "source.png"
    Image.new("RGB", (width, height), "gray").save(source)
    p = public()
    p["source"] = {"path": str(source), "sha256": "0" * 64, "size": [width, height]}
    monkeypatch.setattr(ifbench, "checked_request", lambda item: (p, geometry))
    calls = []
    monkeypatch.setattr(torch._dynamo, "reset", lambda: calls.append("reset"))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)

    def edit(pipe, prompt, images, **parameters):
        calls.append(("edit", prompt, parameters))
        assert images[0].size == (width, height) and images[0].mode == "RGBA"
        return Image.new("RGBA", (width, height), "white"), {}

    monkeypatch.setattr(pipeline, "edit", edit)
    method = {"steps": 40, "seed": 42, "token_guidance": 2.0, "dit_adapter": "adapter"}
    item = {"request_id": "c--S5"}
    tokens = {"units": [{"id": "U1", "token": None, "source": "fallback_text"}, {"id": "U2", "token": U, "source": "pass1"}],
              "fallback_units": 1}
    out = tmp_path / "guided"
    out.mkdir()
    result = ifbench.generate_request(None, method, item, tokens, out)
    assert calls[0] == "reset"
    parameters = calls[1][2]
    assert (parameters["width"], parameters["height"]) == (width, height)
    assert parameters["mode"] == "inline" and parameters["cfg_scale"] == 2.0 and parameters["pass2_te"] == "raw"
    assert "<|mt_" not in parameters["negative_prompt"] and U in calls[1][1]
    assert result["transformer_forward_calls"] == 80 and result["fallback_units"] == 1
    assert Image.open(result["output_image"]).size == (1500, 1875) and result["raw_size"] == [width, height]
    # Every unit fell back to text: plain editing without guidance.
    calls.clear()
    tokens = {"units": [{"id": "U1", "token": None, "source": "fallback_text"}, {"id": "U2", "token": None, "source": "fallback_text"}],
              "fallback_units": 2}
    out = tmp_path / "plain"
    out.mkdir()
    result = ifbench.generate_request(None, method, item, tokens, out)
    parameters = calls[1][2]
    assert parameters["mode"] == "direct" and parameters["cfg_scale"] == 1.0 and "negative_prompt" not in parameters
    assert result["transformer_forward_calls"] == 40


def test_format_repair_restores_only_a_missing_key_name():
    good = [f'{{"label":"U{i}","mask_2d":"{span}"}}' for i, span in ((1, T), (2, U))]
    labels = ["U1", "U2"]
    raw = "[" + good[0] + ',{"label":"U2":"' + U + '"}]<|im_end|>'
    assert ifbench.recover_missing_key(raw, labels) == {"U1": T, "U2": U}
    assert ifbench.recover_missing_key("[" + ",".join(good) + "]", labels) == {"U1": T, "U2": U}
    for bad in ("[" + good[0] + "]",                                        # a label is missing
                "[" + good[1] + "," + good[0] + "]",                        # wrong order
                "[" + good[0] + "," + good[1] + "," + good[1] + "]",         # a label twice
                "[" + good[0] + ',{"label":"U2","box":"' + U + '"}]',        # another key, not a missing one
                "[" + good[0] + "," + good[1] + "] extra",                  # trailing text
                "!!!!!!!!"):
        assert ifbench.recover_missing_key(bad, labels) is None
    broken = "[" + good[0] + ',{"label":"U2":"<|mt_start|><|mt_0300|><|mt_0257|><|mt_end|>"}]'   # invalid code order
    assert ifbench.recover_missing_key(broken, labels) is None


def test_overlay_replaces_the_entries_of_its_requests():
    pytest.importorskip("samtok_benchmark.ifbench.fairness_dev_report")
    from samtok_edit21.evaluation import ifbench_report

    ours = {"run": "main", "method": {"name": "m"}, "case_ids": ["a", "b"],
            "requests": [{"request_id": "a--S1", "case_id": "a", "setting": "S1", "status": "scored", "judge_key": "k1",
                          "unit_pass": [False, False], "joint_pass": False},
                         {"request_id": "b--S1", "case_id": "b", "setting": "S1", "status": "scored", "judge_key": "k2",
                          "unit_pass": [True, True], "joint_pass": True}]}
    fix = {"run": "repair", "method": {"name": "m"},
           "requests": [{"request_id": "a--S1", "case_id": "a", "setting": "S1", "status": "scored", "judge_key": "k3",
                         "unit_pass": [True, True], "joint_pass": True}]}
    merged, replaced = ifbench_report.apply_overlays(ours, [fix])
    assert [r["judge_key"] for r in merged["requests"]] == ["k3", "k2"]
    assert replaced == [{"request_id": "a--S1", "overlay_run": "repair", "replaced_status": "scored",
                         "replaced_judge_key": "k1", "judge_key": "k3"}]
    assert ours["requests"][0]["judge_key"] == "k1"                       # the main report is not modified
    with pytest.raises(ValueError):
        ifbench_report.apply_overlays(ours, [dict(fix, method={"name": "other"})])
    with pytest.raises(ValueError):          # an unscored overlay entry cannot replace a scored one
        ifbench_report.apply_overlays(ours, [dict(fix, requests=[dict(fix["requests"][1 - 1], request_id="b--S1",
                                                                        case_id="b", status="judge-failed")])])
    # Layers: a first repair whose own Judge answer failed, superseded by a second one.
    failed = dict(ours, requests=[dict(ours["requests"][0], status="judge-failed", judge_key="k0"), ours["requests"][1]])
    first = dict(fix, run="repair1", requests=[dict(fix["requests"][0], status="judge-failed", judge_key="k4")])
    merged, replaced = ifbench_report.apply_overlays(failed, [first, fix])
    assert merged["requests"][0]["judge_key"] == "k3" and [r["overlay_run"] for r in replaced] == ["repair1", "repair"]



def load_rejudge():
    pytest.importorskip("samtok_benchmark.ifbench.dev_judge")
    spec = importlib.util.spec_from_file_location("ifbench_rejudge", ROOT / "scripts/ifbench/rejudge_baselines.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_baseline_rejudge_judges_each_failed_key_once(tmp_path, monkeypatch):
    rejudge = load_rejudge()
    canonical = rejudge.canonical
    config = {"engine": {}, "decode": {}}
    calls = []

    class Backend:
        def __init__(self, args, images):
            calls.append("load")

        def generate(self, conversations):
            calls.extend(c["case"] for c in conversations)
            return [judge_answer(c["units"], 4) for c in conversations]

    monkeypatch.setattr(canonical, "Backend", Backend)
    monkeypatch.setattr(canonical, "fingerprint", lambda: config)
    monkeypatch.setattr(canonical, "binding", lambda case, sha, _: f"{case['id']}-{sha[:8]}")
    monkeypatch.setattr(canonical, "prompt", lambda case: case["id"])
    monkeypatch.setattr(canonical, "evidence", lambda case, _: {"case": case["id"],
                                                                "units": [u["id"] for u in case["units"]]})
    cases = {c: {"id": c, "units": [{"id": "U1"}, {"id": "U2"}]} for c in ("a", "b", "c")}
    results, bundle = tmp_path / "results", tmp_path / "bundle"
    bundle.mkdir()
    ready = {"results_root": str(results), "plan_sha256": "s", "preparation_plan_digest": "d"}
    (bundle / "READY.json").write_text(json.dumps(ready))
    monkeypatch.setattr(rejudge, "bundle_context", lambda _: (ready, config, cases))

    def node(n, outputs, statuses, completed=True):
        root = results / "nodes" / f"node-{n}"
        (root / "judge" / "effective_records").mkdir(parents=True)
        if completed:
            (root / "COMPLETED.json").write_text(json.dumps({"status": "completed", "plan_digest": "d"}))
        (root / "judge" / "config.json").write_text(json.dumps(config))
        (root / "judge" / "plan_binding.json").write_text(json.dumps({"plan_sha256": "s"}))
        bindings = []
        for (model, setting, case), content in outputs.items():
            directory = results / "runs" / model / setting / case
            directory.mkdir(parents=True)
            image = directory / "output.png"
            image.write_text(content)
            sha = ifbench.sha256_file(image)
            (directory / "result.json").write_text(json.dumps({"status": "generated"}))
            key = f"{case}-{sha[:8]}"
            bindings.append({"key": f"{model}/{setting}/{case}", "model": model, "setting": setting, "case_id": case,
                             "result_path": str(directory / "result.json"), "generation_status": "generated",
                             "judge_key": key, "generation_record_sha256": ifbench.sha256_file(directory / "result.json"),
                             "output_sha256": sha, "output_image": str(image)})
            (root / "judge" / "effective_records" / f"{key}.json").write_text(json.dumps(
                {"status": statuses[case], "raw_response": {"finish_reason": "length"}, "reason": "Truncated"}))
        (root / "judge" / "bindings.json").write_text(json.dumps(bindings))

    # Node 1: case a failed and its output serves two settings; case b is scored.  Node 0 is still running.
    node(1, {("m", "S1", "a"): "1", ("m", "S5", "a"): "1", ("m", "S1", "b"): "2"},
         {"a": "judge-failed", "b": "scored"})
    node(0, {("m", "S1", "c"): "3"}, {"c": "judge-failed"}, completed=False)
    output = tmp_path / "rejudge"
    with pytest.raises(SystemExit):
        rejudge.main(["--bundle", str(bundle), "--output", str(output), "--node", "0"])
    rejudge.main(["--bundle", str(bundle), "--output", str(output)])
    assert [c for c in calls if c != "load"] == ["a"]
    result = json.loads((output / "OVERLAY.json").read_text())
    assert result["nodes"] == [1] and result["statuses"] == {"scored": 1}
    entry = result["entries"][0]
    assert sorted(x["setting"] for x in entry["logical"]) == ["S1", "S5"]
    assert entry["original"]["status"] == "judge-failed" and entry["rejudge"]["scores"]["units"][0]["E"]["score"] == 4
    # Node 0 later: only its key is judged, node 1 stays in the overlay and is not judged again.
    calls.clear()
    (results / "nodes" / "node-0" / "COMPLETED.json").write_text(json.dumps({"status": "completed", "plan_digest": "d"}))
    rejudge.main(["--bundle", str(bundle), "--output", str(output), "--node", "0"])
    assert [c for c in calls if c != "load"] == ["c"]
    result = json.loads((output / "OVERLAY.json").read_text())
    assert result["nodes"] == [0, 1] and result["statuses"] == {"scored": 2}
    # A generation record that changed after the Judge stops the re-judge.
    (results / "runs" / "m" / "S1" / "a" / "result.json").write_text("{}")
    with pytest.raises(ValueError):
        rejudge.main(["--bundle", str(bundle), "--output", str(output), "--collect-only"])


def test_baseline_overlay_scores_only_failed_answers():
    stats = pytest.importorskip("samtok_benchmark.ifbench.fairness_dev_report")
    from samtok_benchmark.io import digest
    from samtok_edit21.evaluation import ifbench_report

    case_ids, ks = ["c0", "c1", "c2", "c3"], [2, 2, 3, 2]
    boot = stats.Bootstrap(ks)
    rows = []
    for setting in ifbench.SETTINGS:
        up, jp = np.array([2, 1, 3, 0]), np.array([1, 0, 1, 0])
        unknown = np.array([False, False, setting == "S1", False])
        row, _ = stats.rates(up * ~unknown, jp * ~unknown, np.array(ks), unknown, boot)
        row.update(model="base", setting=setting, per_case=[
            {"case_id": c, "K": k, "judge_key": f"{setting}-{c}", "status": "judge-failed" if miss else "scored",
             "UnitPass_count": None if miss else int(u), "JointPass": None if miss else int(j)}
            for c, k, u, j, miss in zip(case_ids, ks, up, jp, unknown)])
        rows.append(row)
    config = {"engine": {}}
    benchmark = {"case_ids": case_ids, "settings": ifbench.SETTINGS, "model_order": ["base"], "rows": rows,
                 "judge_config": config, "bootstrap": {"shared_indices_sha256": digest(boot.indices.tolist())}}
    requests = [{"case_id": c, "setting": s, "status": "scored", "unit_pass": [True] * k, "joint_pass": True}
                for s in ifbench.SETTINGS for c, k in zip(case_ids, ks)]
    ours = {"case_ids": case_ids, "method": {"name": "m"}, "requests": requests}
    assert ifbench_report.merge(benchmark, ours)["rows"]["base"]["S1"]["UnitPass"] is None   # incomplete as published
    block = lambda score: {"score": score, "pass": score >= 3, "evidence": "seen"}  # noqa: E731
    scores = {"units": [{"id": f"U{i}", "E": {"score": 4, "pass": True, "evidence": "seen"}, "P": block(4),
                         "Q": block(3 if i < 3 else 2)} for i in (1, 2, 3)], "global_P": block(4), "global_Q": block(4)}
    entry = {"judge_key": "S1-c2", "unit_ids": ["U1", "U2", "U3"],
             "logical": [{"model": "base", "setting": "S1", "case_id": "c2"}],
             "original": {"status": "judge-failed"}, "rejudge": {"status": "scored", "scores": scores,
                                                                 "effective_record": "e"}}
    overlay = {"source": "rejudge", "judge_run_id": digest(config), "entries": [entry]}
    merged = ifbench_report.merge(benchmark, ours, baseline_overlays=[overlay])
    assert merged["rows"]["base"]["S1"]["UnitPass"] == pytest.approx((2 + 1 + 2 + 0) / 9)   # two of three units pass
    assert merged["rows"]["base"]["S1"]["JointPass"] == pytest.approx(1 / 4)
    assert merged["rows"]["base"]["S5"] == ifbench_report.merge(benchmark, ours)["rows"]["base"]["S5"]
    assert merged["replaced_baseline_answers"] == [{"model": "base", "setting": "S1", "case_id": "c2",
                                                    "judge_key": "S1-c2", "replaced_status": "judge-failed",
                                                    "overlay": "rejudge", "UnitPass_count": 2, "JointPass": 0}]
    assert "base 1 个" in ifbench_report.markdown(merged)
    assert rows[0]["per_case"][2]["status"] == "judge-failed"              # the published rows are not modified
    # A failed re-judge leaves the row incomplete; mismatches and a second re-judge of one key stop the merge.
    failed = dict(overlay, entries=[dict(entry, rejudge={"status": "judge-failed"})])
    assert ifbench_report.merge(benchmark, ours, baseline_overlays=[failed])["rows"]["base"]["S1"]["UnitPass"] is None
    # With the user's 11-04 rule an answer that failed twice counts as a failed source instead.
    counted = ifbench_report.merge(benchmark, ours, baseline_overlays=[failed], failed_twice_as_failure=True)
    assert counted["rows"]["base"]["S1"]["UnitPass"] == pytest.approx((2 + 1 + 0 + 0) / 9)
    assert counted["rows"]["base"]["S1"]["JointPass"] == pytest.approx(1 / 4)
    assert counted["failed_twice_counted_as_failure"] == [{"model": "base", "setting": "S1", "case_id": "c2",
                                                           "judge_key": "S1-c2", "overlay": "rejudge"}]
    assert "按失败计" in ifbench_report.markdown(counted)
    for bad in (dict(overlay, judge_run_id="x"),
                dict(overlay, entries=[dict(entry, judge_key="S1-c1", logical=[dict(entry["logical"][0], case_id="c1")])]),
                dict(overlay, entries=[dict(entry, logical=[dict(entry["logical"][0], setting="S5")])]),
                dict(overlay, entries=[dict(entry, logical=[*entry["logical"], {"model": "base", "setting": "S5",
                                                                                 "case_id": "c2"}])])):
        with pytest.raises(ValueError):
            ifbench_report.merge(benchmark, ours, baseline_overlays=[bad])
    with pytest.raises(ValueError):
        ifbench_report.merge(benchmark, ours, baseline_overlays=[overlay, overlay])


def test_preliminary_rows_come_from_the_machines_records_with_the_benchmarks_make_row(tmp_path):
    stats = pytest.importorskip("samtok_benchmark.ifbench.fairness_dev_report")
    from samtok_benchmark.io import digest, sha256_file
    from samtok_edit21.evaluation import ifbench_report

    case_ids, ks, models = ["c0", "c1", "c2", "c3"], [2, 3, 2, 2], ["m1", "m2"]
    manifest = tmp_path / "cases.jsonl"
    manifest.write_text("".join(json.dumps({"id": c, "units": [{"id": f"U{i + 1}"} for i in range(k)]}) + "\n"
                                for c, k in zip(case_ids, ks)))
    results, bundle = tmp_path / "results", tmp_path / "bundle"
    bundle.mkdir()
    owner = {"c0": 0, "c2": 0, "c1": 1, "c3": 1}
    plan = {"case_manifest": str(manifest), "case_manifest_sha256": sha256_file(manifest), "results_root": str(results),
            "case_ids": case_ids, "settings": ifbench.SETTINGS, "model_order": models, "judge_config": {"engine": {}},
            "digest": "d", "tasks": [{"model": m, "setting": s, "case_id": c} for m in models for s in ifbench.SETTINGS
                                     for c in case_ids]}
    (bundle / "plan.json").write_text(json.dumps(plan))
    (bundle / "READY.json").write_text(json.dumps({"source_commit": "c"}))
    block = lambda score: {"score": score, "pass": score >= 3, "evidence": "seen"}  # noqa: E731
    bindings = {0: [], 1: []}
    for m in models:
        for s in ifbench.SETTINGS:
            for c, k in zip(case_ids, ks):
                key = f"{m}-{s}-{c}"
                status = "failed" if (m, s, c) == ("m2", "S1", "c0") else "generated"
                bindings[owner[c]].append({"key": key, "model": m, "setting": s, "case_id": c, "result_path": key,
                                           "generation_status": status, "judge_key": key if status == "generated" else None})
                if status != "generated":
                    continue
                answer = {"status": "scored", "scores": {
                    "units": [{"id": f"U{i + 1}", "E": {"score": 4, "pass": True, "evidence": "seen"}, "P": block(4),
                               "Q": block(4)} for i in range(k)], "global_P": block(4), "global_Q": block(4)}}
                if (m, s, c) == ("m2", "S5", "c1"):
                    answer = {"status": "judge-failed", "reason": "Truncated Judge response"}
                directory = results / "nodes" / f"node-{owner[c]}" / "judge" / "effective_records"
                directory.mkdir(parents=True, exist_ok=True)
                (directory / f"{key}.json").write_text(json.dumps(answer))
    for node, items in bindings.items():
        (results / "nodes" / f"node-{node}" / "judge" / "bindings.json").write_text(json.dumps(items))
    (results / "nodes" / "node-0" / "COMPLETED.json").write_text(json.dumps({"status": "completed", "plan_digest": "d"}))
    with pytest.raises(ValueError):                                    # node 1 has not completed yet
        ifbench_report.node_report(bundle, nodes=2)
    (results / "nodes" / "node-1" / "COMPLETED.json").write_text(json.dumps({"status": "completed", "plan_digest": "d"}))
    benchmark = ifbench_report.node_report(bundle, nodes=2)
    rows = {(r["model"], r["setting"]): r for r in benchmark["rows"]}
    assert rows["m1", "S1"]["UnitPass"] == 1.0 and rows["m1", "S1"]["JointPass"] == 1.0
    assert rows["m2", "S1"]["UnitPass"] == pytest.approx(7 / 9)      # the generation failure counts as zero
    assert rows["m2", "S5"]["UnitPass"] is None                       # the failed Judge answer leaves the row incomplete
    assert benchmark["bootstrap"]["shared_indices_sha256"] == digest(stats.Bootstrap(ks).indices.tolist())
    requests = [{"case_id": c, "setting": s, "status": "scored", "unit_pass": [True] * k, "joint_pass": True}
                for s in ifbench.SETTINGS for c, k in zip(case_ids, ks)]
    merged = ifbench_report.merge(benchmark, {"case_ids": case_ids, "method": {"name": "m"}, "requests": requests})
    assert merged["preliminary"] and "（初步）" in ifbench_report.markdown(merged)
    assert ifbench_report.differences(merged, merged) == []
    changed = copy.deepcopy(merged)
    changed["rows"]["m1"]["S1"]["UnitPass"] = 0.5
    assert ifbench_report.differences(merged, changed) == ["rows/m1/S1/UnitPass"]


def test_our_answer_that_failed_twice_follows_the_same_rule():
    pytest.importorskip("samtok_benchmark.ifbench.fairness_dev_report")
    from samtok_edit21.evaluation import ifbench_report
    import samtok_benchmark.ifbench.fairness_dev_report as stats
    from samtok_benchmark.io import digest

    case_ids, ks = ["c0", "c1"], [2, 2]
    boot = stats.Bootstrap(ks)
    rows = []
    for setting in ifbench.SETTINGS:
        row, _ = stats.rates(np.array([2, 2]), np.array([1, 1]), np.array(ks), np.zeros(2, bool), boot)
        row.update(model="base", setting=setting, per_case=[
            {"case_id": c, "K": k, "judge_key": f"{setting}-{c}", "status": "scored", "UnitPass_count": 2, "JointPass": 1}
            for c, k in zip(case_ids, ks)])
        rows.append(row)
    benchmark = {"case_ids": case_ids, "settings": ifbench.SETTINGS, "model_order": ["base"], "rows": rows,
                 "judge_config": {}, "bootstrap": {"shared_indices_sha256": digest(boot.indices.tolist())}}
    requests = [{"request_id": f"{c}--{s}", "case_id": c, "setting": s, "status": "scored", "generation_status": "generated",
                 "judge_key": f"k-{c}-{s}", "unit_pass": [True, True], "joint_pass": True}
                for s in ifbench.SETTINGS for c in case_ids]
    requests[0] = dict(requests[0], status="judge-failed")
    ours = {"run": "main", "method": {"name": "m"}, "case_ids": case_ids, "requests": requests}
    again = {"run": "rejudge", "method": {"name": "m"}, "requests": [dict(requests[0], judge_key="k-again")]}
    assert ifbench_report.merge(benchmark, ours, [again])["rows"]["samtokedit"]["S1"]["UnitPass"] is None
    merged = ifbench_report.merge(benchmark, ours, [again], failed_twice_as_failure=True)
    assert merged["rows"]["samtokedit"]["S1"]["UnitPass"] == pytest.approx(2 / 4)       # c0 counts as a failed source
    assert [r["case_id"] for r in merged["failed_twice_counted_as_failure"]] == ["c0"]
    # A failed answer that was never judged a second time stays missing even with the rule.
    assert ifbench_report.merge(benchmark, ours, failed_twice_as_failure=True)["rows"]["samtokedit"]["S1"]["UnitPass"] is None
