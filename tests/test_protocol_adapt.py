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


def test_named_part_becomes_s_and_a_host_mask_is_never_relabeled_as_the_part():
    s, _ = pa.normalize_slots(slots(operation="attribute", ref="the second horse", scope="horse",
                                    target_part="leather harness", attributes=[{"property": "color", "value": "yellow"}]))
    part = pa.part_as_s(pa.to_spec(s))
    assert ip.render(part) == "Change the color of the leather harness of the second horse to yellow."
    assert ip.render(part, False) == "Change the color of the selected leather harness to yellow."
    # Kept as S + target_part on a reused (host) mask, the record has no part execution region and is rejected.
    horse = square(5, 35, 5, 50)
    r = row("attribute", [instance(horse, "second horse")], dataset="derived")
    extracted = {"slots": {**s, "decision": "accept", "fixes": []}, "spec": pa.to_spec(s),
                 "plan": {"source": "reused", "instances": [1]}, "blind": {"bbox_2d": r["region"]["box_1000"]}}
    record = pa.build(r, extracted, None)
    assert record["status"] == "rejected" and record["reason"] == "execution_region_covers_host"


def test_site_with_a_relation_word_is_sent_back():
    s, _ = pa.normalize_slots(slots(operation="add", ref="the pole", scope="pole", payload="a sticker",
                                    placement={"relation": "on", "site": "below the sign"}))
    assert any("relation word" in p for p in pa.slot_problems(s))


def test_reference_relation_contradicted_by_overlap_is_rejected():
    add = pa.to_spec(pa.normalize_slots(slots(operation="add", payload="a glass",
                                              placement={"relation": "beside", "site": "left side"}))[0])
    bottle, glass = square(5, 35, 20, 30), square(20, 34, 18, 30)
    assert pa.geometry_flags(add, pa.geometry(add, bottle, glass), "cleaned_region")[1] == ["reference_item_inside_anchor"]
    near = pa.to_spec(pa.normalize_slots(slots(operation="add", payload="a glass", placement={"relation": "near"}))[0])
    flags, hard = pa.geometry_flags(near, pa.geometry(near, bottle, glass), "cleaned_region")
    assert hard == [] and "reference_item_inside_anchor" in flags


def test_audit_prompt_states_the_add_relation_and_the_outlines():
    spec = pa.to_spec(pa.normalize_slots(slots(operation="add", payload="a bench", placement={"relation": "under"}))[0])
    record = {"instruction_spec": spec, "instruction_ref": ip.render(spec), "instruction_noref": ip.render(spec, False),
              "execution_region": {"rle_size": [1, 1]}}
    prompt = pa.audit_prompt(record)
    assert "(under)" in prompt and "orange" in prompt and record["instruction_noref"] in prompt
    spec = pa.to_spec(pa.normalize_slots(slots())[0])
    prompt = pa.audit_prompt({"instruction_spec": spec, "instruction_ref": "r", "instruction_noref": "n"})
    assert 'relation: answer "pass"' in prompt and "orange" not in prompt and "visible extent" in prompt


def test_sam3_flags_catch_spill_and_sparse_masks():
    mask = square(10, 30, 10, 30, shape=(100, 100))
    assert pa.sam3_flags(mask, [100, 100, 300, 300]) == []
    assert pa.sam3_flags(mask, [100, 100, 150, 150]) == ["sam3_spill"]
    assert pa.sam3_flags(square(10, 12, 10, 12, shape=(100, 100)), [100, 100, 300, 300]) == ["sam3_sparse"]


def test_remasked_s_of_a_non_add_edit_replaces_a_labeled_region_that_misses_it():
    wrong, right = square(0, 10, 0, 10), square(20, 35, 30, 55)
    r = row("attribute", [instance(wrong, "back umbrella")])
    s, _ = pa.normalize_slots(slots(operation="attribute", ref="the front umbrella", scope="umbrella",
                                    attributes=[{"property": "color", "value": "blue"}]))
    extracted = {"slots": {**s, "decision": "accept", "fixes": []}, "spec": pa.to_spec(s), "plan": {"source": "sam3"},
                 "blind": {"bbox_2d": pa.box_1000(right)}, "hinted": {"bbox_2d": pa.box_1000(right)}}
    record = pa.build(r, extracted, {"mask": pa.encode(right)})
    assert record["status"] == "accepted" and record["execution_region"]["source"] == "selection"
    assert (pa.decode(record["execution_region"]) == right).all()


def test_far_near_and_wrong_side_are_hard_and_part_refs_do_not_repeat():
    near = pa.to_spec(pa.normalize_slots(slots(operation="add", payload="an umbrella", placement={"relation": "near"}))[0])
    person, far = square(30, 40, 50, 60, shape=(100, 100)), square(5, 10, 5, 10, shape=(100, 100))
    assert "reference_anchor_not_adjacent" in pa.geometry_flags(near, pa.geometry(near, person, far), "cleaned_region")[1]
    left = pa.to_spec(pa.normalize_slots(slots(operation="add", payload="a box",
                                               placement={"relation": "beside", "site": "left side"}))[0])
    right_of = square(30, 40, 61, 70, shape=(100, 100))
    assert "reference_side" in pa.geometry_flags(left, pa.geometry(left, person, right_of), "cleaned_region")[1]
    s, _ = pa.normalize_slots(slots(operation="attribute", ref="the pink tennis dress", scope="dress",
                                    target_part="dress", attributes=[{"property": "color", "value": "blue"}]))
    assert ip.render(pa.part_as_s(pa.to_spec(s))) == "Change the color of the pink tennis dress to blue."


def test_contents_removal_needs_the_container_inside_its_execution_region():
    vase = square(10, 30, 10, 20, shape=(50, 50))
    contents = pa.to_spec(pa.normalize_slots(slots(remove_mode="contents"))[0])
    assert pa.geometry_flags(contents, pa.geometry(contents, vase, vase | square(5, 10, 10, 20, shape=(50, 50))),
                             "cleaned_region")[1] == []
    assert pa.geometry_flags(contents, pa.geometry(contents, vase, square(5, 12, 10, 20, shape=(50, 50))),
                             "cleaned_region")[1] == ["execution_region_misses_container"]


def _scene(seed=0, size=(200, 160)):
    from PIL import Image
    rng = np.random.default_rng(seed)
    base = rng.integers(60, 200, size=(size[1] // 8, size[0] // 8, 3), dtype=np.uint8)
    return Image.fromarray(base).resize(size, Image.Resampling.BILINEAR)


def _with(image, box, color, noise=0.0, seed=1):
    from PIL import Image
    a = np.asarray(image).astype(np.float32).copy()
    if box:
        x1, y1, x2, y2 = box
        a[y1:y2, x1:x2] = color
    if noise:
        a += np.random.default_rng(seed).normal(0, noise, a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def test_edited_region_finds_the_change_and_ignores_noise():
    source = _scene()
    s_mask = square(40, 100, 50, 120, shape=(160, 200))
    target = _with(source, (60, 50, 110, 90), (250, 20, 20), noise=1.5)
    region, stats = pa.edited_region(source, target, s_mask)
    truth = square(50, 90, 60, 110, shape=(160, 200))
    iou = (region & truth).sum() / (region | truth).sum()
    assert iou > 0.85 and stats["components"] == 1 and stats["in_anchor"] > 0.9
    region, stats = pa.edited_region(source, _with(source, None, None, noise=1.5), s_mask)
    assert region is None


def test_edited_region_keeps_faint_shadows_joined_to_the_edit_and_drops_far_changes():
    source = _scene(3)
    s_mask = square(40, 100, 50, 120, shape=(160, 200))
    target = _with(source, (60, 50, 110, 90), (20, 220, 40))
    shadow = np.asarray(target).astype(np.float32)
    shadow[90:100, 60:110] *= 0.85  # Delta E about 9: between the thresholds, kept only because it joins the edit
    from PIL import Image
    target = Image.fromarray(shadow.astype(np.uint8))
    region, _ = pa.edited_region(source, target, s_mask)
    assert region[95, 85] and region[70, 85]
    far = _with(source, (0, 0, 20, 20), (255, 255, 255))
    region, stats = pa.edited_region(source, far, s_mask)
    assert region is None and stats["dropped_far_frac"] > 0


def test_records_without_an_edited_region_are_not_accepted():
    s, _ = pa.normalize_slots(slots(operation="attribute", attributes=[{"property": "color", "value": "red"}]))
    record = {"status": "accepted", "flags": [], "instruction_spec": pa.to_spec(s)}
    stats = {"area_frac": 0.1, "dropped_far_frac": 0.0, "s_covered": 0.05, "in_anchor": 0.9}
    assert pa.attach_edited_region(record, None, stats)["status"] == "rejected"
    kept = pa.attach_edited_region(record, square(1, 9, 1, 9, shape=(10, 10)), stats)
    assert kept["status"] == "accepted" and "edited_region_misses_s" in kept["flags"]
    assert kept["edited_region"]["method"] == pa.EDITED_METHOD and kept["edited_region"]["rle_counts"]


def add_spec(relation="on", **values):
    values = {"payload": "a small dog", "ref": "the road", "scope": "road", **values}
    return pa.to_spec(pa.normalize_slots(slots(operation="add", placement={"relation": relation}, **values))[0])


def test_a_large_anchor_is_cut_to_the_spot_around_the_new_item():
    road, dog = square(100, 200, 0, 300, shape=(200, 300)), square(120, 150, 140, 160, shape=(200, 300))
    local, info = pa.localize_anchor(road, dog)
    assert info["cut"] and local.sum() < 0.2 * road.sum() and (local <= road).all()
    assert local[dog].all()  # the road where the dog will stand is part of the spot
    assert local.sum() >= pa.ANCHOR_PARAMS["min_area_k"] * dog.sum()
    ys, xs = np.nonzero(local)
    assert 100 <= xs.min() and xs.max() < 200 and ys.max() < 190
    head, hat = square(20, 40, 50, 70, shape=(200, 300)), square(10, 25, 48, 72, shape=(200, 300))
    local, info = pa.localize_anchor(head, hat)  # about the item's size: it already says where the hat goes
    assert local is head and not info["cut"]
    assert pa.localize_anchor(square(0, 10, 0, 10, shape=(200, 300)), dog)[0] is None


def test_a_small_item_on_a_large_anchor_still_gets_a_usable_spot():
    tower, cross = square(40, 200, 100, 140, shape=(200, 300)), square(30, 40, 118, 122, shape=(200, 300))
    local, info = pa.localize_anchor(tower, cross)
    assert info["cut"] and local.sum() >= pa.ANCHOR_PARAMS["min_area_frac"] * 200 * 300
    assert np.nonzero(local)[0].min() == 40 and np.nonzero(local)[0].max() < 80  # the top of the tower only
    roof_edge = square(40, 44, 60, 200, shape=(200, 300))  # thin and small, but reaching far beyond the item
    local, info = pa.localize_anchor(roof_edge, cross)
    assert info["cut"] and np.nonzero(local)[1].min() > 60 and np.nonzero(local)[1].max() < 200


def test_anchors_across_open_space_are_rejected_and_touching_ones_pass():
    man, dog = square(50, 150, 200, 240, shape=(200, 300)), square(110, 150, 120, 160, shape=(200, 300))
    beside = add_spec("beside", ref="the man", scope="man")
    assert "reference_anchor_not_adjacent" in pa.geometry_flags(beside, pa.geometry(beside, man, dog), "cleaned_region")[1]
    touching = square(110, 150, 160, 200, shape=(200, 300))
    assert pa.geometry_flags(beside, pa.geometry(beside, man, touching), "cleaned_region")[1] == []
    table, cup = square(100, 120, 0, 100, shape=(200, 300)), square(60, 80, 200, 220, shape=(200, 300))
    on = add_spec("on", ref="the table", scope="table")
    assert pa.geometry_flags(on, pa.geometry(on, table, cup), "cleaned_region")[1] == ["support_not_touching_item"]
    assert pa.geometry_flags(on, pa.geometry(on, table, square(80, 100, 40, 60, shape=(200, 300))),
                             "cleaned_region")[1] == []


def test_build_cuts_a_remasked_anchor_and_keeps_the_full_mask():
    road, dog = square(100, 200, 0, 300, shape=(200, 300)), square(120, 150, 140, 160, shape=(200, 300))
    r = row("add", [instance(dog, "white dog", side="target")])
    s, _ = pa.normalize_slots(slots(operation="add", payload="a small white dog", ref="the road in the foreground",
                                    scope="road", placement={"relation": "on"}, instances=[{"n": 1, "role": "new_item"}]))
    extracted = {"slots": {**s, "decision": "accept", "fixes": []}, "spec": pa.to_spec(s), "plan": {"source": "sam3"},
                 "blind": {"bbox_2d": pa.box_1000(road)}, "hinted": {"bbox_2d": pa.box_1000(road)}}
    record = pa.build(r, extracted, {"mask": pa.encode(road)})
    assert record["status"] == "accepted" and record["instruction_noref"] == "Add a small white dog on the selected road."
    local, full = pa.decode(record["selection"]), pa.decode(record["selection"]["localized"]["full"])
    assert (full == road).all() and local.sum() < 0.2 * road.sum() and local[dog].all()
    assert record["selection"]["localized"]["footprint"] == "execution_region"


def test_a_derived_add_finds_the_new_item_in_the_pixels():
    from PIL import Image
    zebra, saddle = square(40, 160, 20, 280, shape=(200, 300)), square(50, 80, 120, 170, shape=(200, 300))
    before = np.full((200, 300, 3), 120, np.uint8)
    after = before.copy()
    after[saddle] = (150, 60, 20)
    r = row("add", [instance(zebra, "zebra")], dataset="derived")
    s, _ = pa.normalize_slots(slots(operation="add", payload="a brown saddle", ref="the back of the zebra",
                                    scope="zebra back", placement={"relation": "on"}, instances=[{"n": 1, "role": "anchor"}]))
    extracted = {"slots": {**s, "decision": "accept", "fixes": []}, "spec": pa.to_spec(s),
                 "plan": {"source": "reused", "instances": [1]}, "blind": {"bbox_2d": pa.box_1000(zebra)}}
    images = (Image.fromarray(before), Image.fromarray(after))
    record = pa.build(r, extracted, None, images=images)
    assert record["status"] == "accepted" and record["execution_region"] is None
    local = pa.decode(record["selection"])
    assert record["selection"]["localized"]["footprint"] == "edited_region" and local[saddle].all()
    assert local.sum() < 0.5 * zebra.sum()
    assert "anchor_not_localized" in pa.build(r, extracted, None)["flags"]


def test_an_add_anchor_is_segmented_whole_and_its_ref_may_name_the_spot():
    s, _ = pa.normalize_slots(slots(operation="add", payload="the text 'rockstar'", ref="the knuckles of the central fist",
                                    scope="knuckles", placement={"relation": "across"}, s_object="central raised fist"))
    assert s["s_object"] == "the central raised fist"
    assert pa.sam_prompt(pa.to_spec(s), s["s_object"]) == "central raised fist"
    assert pa.normalize_slots(slots(operation="add", payload="a cup", ref="the table", scope="table"))[0]["s_object"] == "the table"
    assert pa.normalize_slots(slots())[0]["s_object"] == ""
    assert pa.sam_prompt(pa.to_spec(pa.normalize_slots(slots())[0])) == "red jar"
    assert pa.box_holds_center([100, 100, 500, 500], [280, 300, 320, 340])
    assert not pa.box_holds_center([100, 100, 500, 500], [600, 600, 700, 700]) and not pa.box_holds_center(None, [1, 1, 2, 2])


def test_a_derived_add_keeps_the_changed_pixels_inside_the_new_items_box():
    from PIL import Image
    pizza, olive = square(40, 160, 20, 280, shape=(200, 300)), square(90, 100, 140, 150, shape=(200, 300))
    before = np.full((200, 300, 3), 120, np.uint8)
    after = before.copy()
    after[pizza] = (126, 126, 126)  # the generator redrew the whole pizza a little
    after[50:60, 30:60] = (160, 100, 100)  # and changed a spot far from the olive
    after[olive] = (10, 10, 10)
    images = (Image.fromarray(before), Image.fromarray(after))
    footprint, kind = pa.add_footprint(None, pizza, images, pa.box_1000(olive))
    assert kind == "edited_region_in_item_box" and footprint[olive].all() and footprint.sum() < 3 * olive.sum()
    assert pa.add_footprint(None, pizza, images)[1] == "edited_region"
    assert pa.add_footprint(square(1, 5, 1, 5, shape=(200, 300)), pizza, images)[1] == "execution_region"


def test_a_cut_anchor_keeps_one_compact_spot():
    floor, robot = square(100, 200, 0, 300, shape=(200, 300)), square(110, 160, 140, 170, shape=(200, 300))
    floor[100:200, 100:104] = False  # a pole splits off a thin strip on the left
    floor[100:103, 104:110] = False
    local, info = pa.localize_anchor(floor, robot)
    assert info["cut"] and np.nonzero(local)[1].min() >= 104
