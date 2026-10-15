"""Multi-object pairs as several protocol 1.0 units (src/samtok_edit21/preparation/protocol_multi.py); no servers."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from samtok_edit21.preparation import instruction_protocol as ip
from samtok_edit21.preparation import protocol_adapt as pa
from samtok_edit21.preparation import protocol_multi as pm


def square(y1, y2, x1, x2, shape=(40, 60)):
    mask = np.zeros(shape, dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def instance(mask, ref="person", side="source"):
    return {"instance_id": ref, "ref": ref, "grounding_image": side, "mapped_from_target": side == "target",
            **pa.encode(mask)}


def unit(operation="remove", ref="the man on the left", scope="man", n=1, role="main", **values):
    return {"operation": operation, "instances": [{"n": n, "role": role}], "ref": ref, "scope": scope,
            "remove_mode": "whole", "s_location": "left", **values}


def test_the_prompt_reuses_the_single_unit_rules():
    row = {"instruction": "Remove the two men.", "type": "remove", "instances": [instance(square(1, 9, 1, 9))]}
    prompt = pm.multi_slot_prompt(row)
    assert pa.WRITING_RULES in prompt and "3. operation, judged by what AFTER shows" in prompt
    assert '"units": [' in prompt and "{{" not in prompt


def test_units_are_one_object_each_with_distinct_refs():
    two = [unit(), unit("attribute", "the chair on the right", "chair", n=2,
                        attributes=[{"property": "color", "value": "red"}])]
    decision, units = pm.normalize_units({"decision": "accept", "units": two}, 2)
    assert decision == "accept" and [u["unit"] for u in units] == ["U1", "U2"]
    assert units[1]["spec"]["attributes"] == [{"property": "color", "value": "red"}]
    bad = [
        ({"decision": "accept", "units": two[:1]}, "needs 2 to 5 units"),
        ({"decision": "accept", "units": [two[0], {**two[0], "instances": [{"n": 2, "role": "main"}]}]}, "share the ref"),
        ({"decision": "accept", "units": [two[0], {**two[1], "instances": [{"n": 1, "role": "main"}]}]}, "both U1 and U2"),
        ({"decision": "accept", "units": [unit(ref="the group of people", scope="group of people"), two[1]]},
         "names several separate objects"),
        ({"decision": "accept", "units": [unit(ref="the striped lounge chairs", scope="lounge chairs"), two[1]]},
         "names several separate objects"),
        ({"decision": "accept", "units": [two[0], unit("add", "the man on the right", n=2, payload="raised arms",
                                                       placement={"relation": "attached to"})]}, "re-posing"),
        ({"decision": "maybe"}, "unknown decision"),
    ]
    for raw, message in bad:
        with pytest.raises(ValueError, match=message):
            pm.normalize_units(raw, 2)
    man = unit("attribute", "the man on the right", n=2, attributes=[{"property": "color", "value": "blue"}])
    with pytest.raises(ValueError, match="not the whole person"):
        pm.normalize_units({"decision": "accept", "units": [two[0], man]}, 2)
    skin = [two[0], {**man, "attributes": [{"property": "color", "value": "darker skin tone"}]}]
    assert pm.normalize_units({"decision": "accept", "units": skin}, 2)[0] == "accept"
    mixed = [two[0], {**man, "attributes": [{"property": "color", "value": "darker skin and a blue uniform"}]}]
    with pytest.raises(ValueError, match="skin and clothing are two units"):
        pm.normalize_units({"decision": "accept", "units": mixed}, 2)
    pile = [unit(ref="the pile of pumpkins on the left", scope="pile of pumpkins"), two[1]]
    assert pm.normalize_units({"decision": "accept", "units": pile}, 2)[0] == "accept"
    assert pm.normalize_units({"decision": "reject", "reason": "six hats"}, 2) == ("reject", [])


def test_a_unit_row_keeps_its_own_outlines_or_looks_inside_a_shared_one():
    row = {"id": "p", "dataset": "crispedit", "type": "remove", "instruction": "x",
           "instances": [instance(square(1, 9, 1, 9)), instance(square(1, 9, 30, 40)), instance(square(0, 40, 0, 60))],
           "region": pa.union([instance(square(0, 40, 0, 60))])}
    _, units = pm.normalize_units({"decision": "accept", "units": [unit(), unit(ref="the man on the right", n=2)]}, 3)
    first = pm.unit_row(row, units[0], [3])
    assert first["id"] == "p~U1" and first["pair_id"] == "p" and len(first["instances"]) == 1
    assert first["region"]["box_1000"] == pa.union([row["instances"][0]])["box_1000"] and "ground_hint" not in first
    assert pm.unit_slots(units[1])["instances"] == [{"n": 1, "role": "main"}]
    alone = {**units[0], "slots": {**units[0]["slots"], "instances": []}}
    inside = pm.unit_row(row, alone, [3])
    assert inside["region"] is None and inside["instances"] == [] and "outlined area" in inside["ground_hint"]
    add = pm.normalize_units({"decision": "accept", "units": [
        unit("add", "the road next to the man", "road", n=1, role="new_item", payload="a small dog", granularity="surface",
             host_ref="the road", host_scope="road", placement={"relation": "on"}), unit(n=2)]}, 3)[1][0]
    with_item = pm.unit_row(row, add, [])
    assert with_item["region"]["box_1000"] == pa.union([row["instances"][0]])["box_1000"]


def test_pair_checks_catch_shared_objects_and_unexplained_changes():
    shape = (120, 160)
    rng = np.random.default_rng(0)
    base = rng.integers(60, 200, size=(shape[0] // 8, shape[1] // 8, 3), dtype=np.uint8)
    source = Image.fromarray(base).resize((shape[1], shape[0]), Image.Resampling.BILINEAR)
    target = np.asarray(source).copy()
    a, b, c = square(10, 40, 10, 40, shape), square(10, 40, 100, 130, shape), square(80, 110, 60, 90, shape)
    target[a] = (250, 10, 10)
    target[b] = (10, 250, 10)
    images = (source, Image.fromarray(target))
    row = {"instances": [instance(a), instance(b)]}

    def u(name, s):
        region, _ = pa.edited_region(*images, s)
        return {"id": name, "selection": pa.encode(s), "execution_region": None, "edited_region": pa.encode(region)}

    checks, hard, total, _ = pm.pair_checks(row, [u("U1", a), u("U2", b)], images)
    assert hard == [] and checks["unexplained_frac"] < 0.05 and total is not None
    part = square(12, 20, 12, 20, shape)  # a part of U1's object
    checks, hard, _, _ = pm.pair_checks(row, [u("U1", a), {**u("U2", b), "selection": pa.encode(part)}], images)
    assert "units_share_s" in hard and checks["same_s"][0][:2] == ["U1", "U2"]
    target[c] = (10, 10, 250)  # a third change that no unit describes
    images = (source, Image.fromarray(target))
    row = {"instances": [instance(a), instance(b), instance(c)]}
    checks, hard, _, _ = pm.pair_checks(row, [u("U1", a), u("U2", b)], images)
    assert "change_not_in_units" in hard and checks["unexplained_frac"] > 0.2


def test_assemble_keeps_a_pair_only_when_every_unit_is_accepted(tmp_path, monkeypatch):
    row = {"id": "p", "dataset": "crispedit", "type": "remove", "instruction": "x", "instances": []}
    record = {"slots": {"decision": "accept", "observed": "two men gone"},
              "units": [{"id": "p~U1", "unit": "U1"}, {"id": "p~U2", "unit": "U2"}]}
    spec = ip.make_spec("remove", "the man", "man")
    built = {"p~U1": {"status": "accepted", "instruction_spec": spec, "flags": ["fix:x", "sam3_sparse"]},
             "p~U2": {"status": "rejected", "reason": "edited_region_misses_s"}}
    out = pm.assemble(row, record, built, tmp_path)
    assert out["status"] == "rejected" and out["reason"] == "U2:edited_region_misses_s"
    monkeypatch.setattr(pm, "pair_checks", lambda *a: ({"same_s": []}, [], None, {}))
    monkeypatch.setattr(pa, "open_pair", lambda *a: (None, None))
    built["p~U2"] = {"status": "accepted", "instruction_spec": spec, "flags": []}
    out = pm.assemble(row, record, built, tmp_path)
    assert out["status"] == "accepted" and [u["id"] for u in out["units"]] == ["U1", "U2"]
    assert out["flags"] == ["U1:sam3_sparse"]
    assert pm.assemble(row, {"slots": {"decision": "reject"}, "units": []}, {}, tmp_path)["reason"] == "slots_reject"
    single = {"slots": {"decision": "single"}, "units": [{"id": "p~U1", "unit": "U1"}]}
    assert pm.assemble(row, single, built, tmp_path)["status"] == "single_unit"


def test_the_pair_audit_needs_every_unit_and_the_whole_to_pass():
    spec = ip.make_spec("remove", "the man", "man")
    pair = {"id": "p", "units": [{"id": "U1", "instruction_spec": spec, "instruction_ref": "Remove the man.",
                                  "instruction_noref": "Remove the selected man."}]}
    prompt = pm.multi_audit_prompt(pair)
    assert 'U1 (outline 1, red; remove)' in prompt and '{"id": "U1", "edit": "pass"' in prompt

    class Fake:
        def __init__(self, answers):
            self.answers = answers

        async def call(self, parts, prompt, call, seed=0):
            return {"text": json.dumps(self.answers[seed]), "tokens": 1, "finish": "stop"}

    ok = {"units": [{"id": "U1", **{k: "pass" for k in pm.UNIT_AUDIT_KEYS}}], "complete": "pass", "problems": ""}
    bad = {**ok, "complete": "fail", "problems": "a chair also changes"}
    import asyncio
    result = asyncio.run(pm.multi_audit(Fake([ok, bad]), pair, [], votes=2))
    assert result["passed"] is False and result["complete"] == "fail" and pm.audit_failures(result) == ["complete"]
    assert asyncio.run(pm.multi_audit(Fake([ok, ok]), pair, [], votes=2))["passed"] is True


def test_repair_and_reason_codes_of_the_cli(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "data_cleaning"))
    import protocol_multi as cli
    assembled = tmp_path / "assembled.jsonl"
    rows = [{"id": "a", "status": "rejected", "reason": "U2:support_not_touching_item"},
            {"id": "b", "status": "rejected", "reason": "slots_reject"},
            {"id": "c", "status": "accepted", "units": []},
            {"id": "d", "status": "rejected", "reason": "change_not_in_units"}]
    assembled.write_text("".join(json.dumps(r) + "\n" for r in rows))
    audit = tmp_path / "audit.jsonl"
    audit.write_text(json.dumps({"id": "c", "passed": False, "units": {"U1": {"edit": "fail"}}, "complete": "pass",
                                 "problems": "U1 not removed"}) + "\n")
    out = tmp_path / "repair.jsonl"
    cli.repair(SimpleNamespace(assembled=str(assembled), audit=[str(audit)], output=str(out)))
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert [x["id"] for x in lines] == ["a", "c", "d"]
    assert lines[0]["problems"].startswith("U2: the new item does not touch S") and "U1:edit" in lines[1]["problems"]
    assert cli.reason_key({"reason": "U2:edited_region_misses_s,selection_off_edit"}) == "unit:edited_region_misses_s"
    assert cli.reason_key({"reason": "audit:U1:edit"}) == "audit"
    assert cli.reason_key({"reason": "units_share_s"}) == "units_share_s"
