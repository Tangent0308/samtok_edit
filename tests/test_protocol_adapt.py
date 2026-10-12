"""Protocol 1.0 adaptation (src/samtok_edit21/preparation/protocol_adapt.py) on synthetic rows; no model calls."""
import hashlib
from pathlib import Path

import numpy as np
import pytest

from samtok_edit21.preparation import instruction_protocol as ip
from samtok_edit21.preparation import protocol_adapt as pa


def rle(mask):
    out = pa.encode(np.asarray(mask, dtype=bool))
    return {"rle_size": out["rle_size"], "rle_counts": out["rle_counts"], "box_1000": out["box_1000"]}


def square(y1, y2, x1, x2, shape=(40, 60)):
    mask = np.zeros(shape, dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def instance(mask, ref, side="source", number=0):
    return {"instance_id": f"i{number}", "ref": ref, "grounding_image": side, "mapped_from_target": side == "target",
            **rle(mask)}


def row(kind, instances, instruction="x", dataset="crispedit"):
    region = pa.union(instances)
    return {"id": f"{dataset}-{kind}", "dataset": dataset, "type": kind, "instruction": instruction,
            "instances": instances, "region": region}


def slots(**values):
    base = {"decision": "accept", "operation": "remove", "ref": "the red jar", "scope": "jar", "target_part": "",
            "payload": "", "remove_mode": "whole", "attributes": [], "placement": {}, "qualifier": "",
            "instances": [{"n": 1, "role": "main"}]}
    return {**base, **values}


def test_vendored_protocol_is_the_benchmark_module():
    assert hashlib.sha256(Path(ip.__file__).read_bytes()).hexdigest() == pa.PROTOCOL_SHA256


def test_add_slots_render_like_the_benchmark():
    s, fixes = pa.normalize_slots(slots(operation="add", ref="central girl's leather jacket", scope="the jacket",
                                        payload="one small round enamel pin.",
                                        placement={"relation": "on", "site": "the upper back", "anchor_mode": "support"}))
    spec = pa.to_spec(s)
    assert ip.render(spec, True) == "Add one small round enamel pin on the upper back of the central girl's leather jacket."
    assert ip.render(spec, False) == "Add one small round enamel pin on the upper back of the selected jacket."
    assert fixes == ["ref_article"]


def test_relation_repairs_and_reference_mode():
    s, fixes = pa.normalize_slots(slots(operation="add", ref="the stone bench", scope="bench", payload="a black dog",
                                        placement={"relation": "to the left of", "anchor_mode": "support"}))
    assert s["placement"] == {"relation": "beside", "site": "left side", "anchor_mode": "reference"}
    assert ip.render(pa.to_spec(s), False) == "Add a black dog beside the left side of the selected bench."
    assert fixes == ["side_relation", "anchor_mode"]
    s, fixes = pa.normalize_slots(slots(operation="add", ref="the desk", scope="desk", payload="a cup",
                                        placement={"relation": "to", "site": "top"}))
    assert s["placement"]["relation"] == "on" and "to_with_site" in fixes
    s, _ = pa.normalize_slots(slots(operation="add", ref="the bottle", scope="bottle", payload="a glass",
                                    placement={"relation": "in front of"}))
    with pytest.raises(ValueError):
        pa.to_spec(s)


def test_remove_part_text_and_operation_specific_slots():
    s, fixes = pa.normalize_slots(slots(target_part="the lid", payload="ignored"))
    assert s["remove_mode"] == "part" and s["payload"] == "" and fixes == ["remove_mode_part"]
    assert ip.render(pa.to_spec(s), False) == "Remove the lid from the selected jar."
    s, _ = pa.normalize_slots(slots(operation="attribute", ref="the canopy sign", scope="sign",
                                    attributes=[{"property": "Text", "value": "'MOBIL'"}],
                                    placement={"relation": "on"}))
    assert s["placement"] == {} and s["remove_mode"] == "whole"
    assert ip.render(pa.to_spec(s), True) == 'Change the text on the canopy sign to "MOBIL".'


def test_selection_plan_reuses_labels_only_when_they_mark_s():
    jar, lid, item = square(5, 30, 5, 30), square(2, 6, 8, 20), square(10, 20, 40, 55)
    added = row("add", [instance(item, "a cup", "target")])
    spec = pa.to_spec(pa.normalize_slots(slots(operation="add", payload="a cup", instances=[{"n": 1, "role": "anchor"}],
                                               placement={"relation": "on"}))[0])
    assert pa.selection_plan(added, slots(instances=[{"n": 1, "role": "anchor"}]), spec) == {"source": "sam3"}
    hosted = row("add", [instance(jar, "red jar")], dataset="derived")
    assert pa.selection_plan(hosted, slots(instances=[{"n": 1, "role": "anchor"}]), spec) == {"source": "reused",
                                                                                              "instances": [1]}
    removed = row("remove", [instance(jar, "jar", number=0), instance(lid, "lid", number=1)])
    plan_slots = slots(instances=[{"n": 1, "role": "main"}, {"n": 2, "role": "attached"}])
    remove_spec = pa.to_spec(pa.normalize_slots(plan_slots)[0])
    assert pa.selection_plan(removed, plan_slots, remove_spec) == {"source": "split", "instances": [1]}
    part_spec = pa.to_spec(pa.normalize_slots(slots(target_part="lid"))[0])
    assert pa.selection_plan(removed, plan_slots, part_spec) == {"source": "split", "instances": [1]}
    only_after = row("replace", [instance(jar, "new vase", "target")])
    assert pa.selection_plan(only_after, slots(instances=[{"n": 1, "role": "new_item"}]), remove_spec) == {"source": "sam3"}


def test_union_keeps_a_single_rle_byte_for_byte_and_merges_exactly():
    a, b = instance(square(0, 10, 0, 10), "a"), instance(square(5, 20, 5, 20), "b")
    assert pa.union([a])["rle_counts"] == a["rle_counts"]
    assert pa.decode(pa.union([a, b])).sum() == (square(0, 10, 0, 10) | square(5, 20, 5, 20)).sum()
    assert pa.instance_side({"ref": "additional changed surface adjoining the screen"}) == "extra"
    assert pa.instance_side({"ref": "hat", "mapped_from_target": True}) == "target"


def test_draft_routes_text_and_keeps_quoted_values():
    d = pa.draft({"type": "text", "instruction": "Change the text on the right canopy sign from 'EXXON' to 'MOBIL'."})
    assert d["route"] == "attribute" and d["quoted"] == ["EXXON", "MOBIL"]
    d = pa.draft({"type": "add", "instruction": "Add a small, round, red coffee cup on the desk next to the folder."})
    assert d["payload"] == "a small, round, red coffee cup" and d["relation"] == "on"
    assert pa.text_route("Remove the text from the red banner on the left.") == "remove"
    assert pa.draft({"type": "text", "instruction": "Add the text 'App' inside the circle."})["route"] == "add"


def test_flags_catch_lost_and_invented_values_and_unclean_scopes():
    original = pa.draft({"type": "replace", "instruction": "Replace the blue boat with a red sailboat."})
    spec = pa.to_spec(pa.normalize_slots(slots(operation="replace", ref="the blue boat", scope="blue boat",
                                               payload="a green sailboat"))[0])
    flags = pa.text_flags({"instruction": "Replace the blue boat with a red sailboat."}, spec, ip.render(spec),
                          ip.render(spec, False), original)
    assert "lost_values:red" in flags and "new_values:green" in flags and "scope_not_plain" in flags
    spec = pa.to_spec(pa.normalize_slots(slots(scope="chair back"))[0])
    assert "scope_not_plain" not in pa.text_flags({"instruction": "Remove the chair back."}, spec, "", "",
                                                  pa.draft({"type": "remove", "instruction": "Remove the chair back."}))


def test_local_execution_must_not_cover_the_host_and_reference_adds_need_one():
    spec = pa.to_spec(pa.normalize_slots(slots(target_part="lid"))[0])
    jar = square(5, 30, 5, 30)
    assert pa.geometry_flags(spec, pa.geometry(spec, jar, jar), "cleaned_region")[1] == ["execution_region_covers_host"]
    assert pa.geometry_flags(spec, pa.geometry(spec, jar, square(5, 8, 5, 30)), "cleaned_region")[1] == []
    add = pa.to_spec(pa.normalize_slots(slots(operation="add", payload="a dog", placement={"relation": "beside"}))[0])
    assert pa.geometry_flags(add, {}, "none")[1] == ["reference_add_without_execution_region"]
    geo = pa.geometry(add, square(10, 30, 30, 40), square(20, 30, 5, 15))
    assert geo["dx"] < 0 and geo["exec_touch_s"] == 0


def test_pilot_sample_is_stratified_and_deterministic():
    rows = [{"id": str(i), "dataset": "d" + str(i % 2), "type": t, "instruction": text}
            for i, (t, text) in enumerate([("add", "Add a")] * 6 + [("text", "Remove the word")] * 3
                                          + [("text", "Change 'a' to 'b'")] * 3)]
    first = pa.pilot_rows(rows, {"add": 2}, {"remove": 1, "attribute": 2})
    assert first == pa.pilot_rows(rows, {"add": 2}, {"remove": 1, "attribute": 2})
    assert sorted((r["type"], r["instruction"][:3]) for r in first) == sorted(
        [("add", "Add")] * 4 + [("text", "Rem"), ("text", "Cha"), ("text", "Cha")])


def test_scope_naming_the_site_is_sent_back():
    s, _ = pa.normalize_slots(slots(operation="add", ref="the middle man", scope="head", payload="a beanie",
                                    placement={"relation": "on", "site": "head"}))
    assert pa.slot_problems(s) and not pa.slot_problems({**s, "scope": "man"})


def test_labeled_part_named_as_target_part_becomes_s():
    harness = square(10, 20, 10, 30)
    r = row("attribute", [instance(harness, "leather harness on the second horse")], dataset="derived")
    s, _ = pa.normalize_slots(slots(operation="attribute", ref="the second horse", scope="horse",
                                    target_part="leather harness", attributes=[{"property": "color", "value": "yellow"}]))
    extracted = {"slots": {**s, "decision": "accept", "fixes": []}, "spec": pa.to_spec(s),
                 "plan": {"source": "reused", "instances": [1]}, "blind": {"bbox_2d": r["region"]["box_1000"]}}
    record = pa.build(r, extracted, None)
    assert record["status"] == "accepted" and "fix:part_is_s" in record["flags"]
    assert record["instruction_noref"] == "Change the color of the selected leather harness to yellow."
    assert record["instruction_ref"] == "Change the color of the leather harness of the second horse to yellow."


def test_reference_relation_contradicted_by_overlap_is_rejected():
    add = pa.to_spec(pa.normalize_slots(slots(operation="add", payload="a glass",
                                              placement={"relation": "beside", "site": "left side"}))[0])
    bottle, glass = square(5, 35, 20, 30), square(20, 34, 18, 30)
    assert pa.geometry_flags(add, pa.geometry(add, bottle, glass), "cleaned_region")[1] == ["reference_item_inside_anchor"]
    near = pa.to_spec(pa.normalize_slots(slots(operation="add", payload="a glass", placement={"relation": "near"}))[0])
    flags, hard = pa.geometry_flags(near, pa.geometry(near, bottle, glass), "cleaned_region")
    assert hard == [] and "reference_item_inside_anchor" in flags
