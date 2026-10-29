"""The merge's new SA-1B families (preparation/sa1b_groups.py) and the edited-region port (regions/edited.py)."""
import numpy as np
from PIL import Image

from samtok_edit21.preparation import round2_data as rd
from samtok_edit21.preparation import sa1b_groups as g
from samtok_edit21.regions.edited import encode, sa1b_edited_region


def _pair(h=120, w=160):
    rng = np.random.default_rng(0)
    src = rng.integers(60, 200, size=(h, w, 3), dtype=np.uint8)
    tgt = src.copy()
    tgt[40:70, 50:90] = [250, 20, 20]                      # the edit
    support = np.zeros((h, w), bool)
    support[30:80, 40:100] = True
    s = np.zeros((h, w), bool)
    s[40:70, 50:90] = True
    return Image.fromarray(src), Image.fromarray(tgt), s, support


def test_edited_region_finds_the_edit_inside_the_support():
    src, tgt, s, support = _pair()
    field, region = sa1b_edited_region(src, tgt, s, support)
    assert field["method"] == "lab-diff-hysteresis-sa1b-1"
    assert region is not None and not (region & ~support).any()
    assert field["s_covered"] >= 0.95 and field["stats"]["s_covered"] >= 0.95
    # the blur and the closing keep the soft edge of the change: a little more than S, never less
    assert s.mean() <= field["area_frac"] <= 1.3 * s.mean()


def test_unchanged_target_has_no_region():
    src, _, s, support = _pair()
    field, region = sa1b_edited_region(src, src, s, support)
    assert region is None and field["rle_counts"] is None and field["area_frac"] == 0.0


def _unit(op, **extra):
    spec = {"operation": op, "ref": "the cup", "scope": "cup", "payload": "a red cup" if op in ("add", "replace") else "",
            "target_part": "", "remove_mode": "whole", "attributes": [], "placement": {}, "qualifier": "", "version": "2.0"}
    if op == "attribute":
        spec["attributes"] = [{"property": "color", "value": "red"}]
    unit = {"unit_id": "U1", "operation": op, "instruction": "x", "instruction_noref": "y", "object_id": 3}
    unit.update(extra)
    return unit, spec


def _tok(site_rt, host_rt=None):
    def entry(rt):
        mask = np.zeros((8, 8), bool)
        mask[2:5, 2:5] = True
        return {"span": "<|mt_start|><|mt_0001|><|mt_0300|><|mt_end|>", "dec": rd.encode_rle(mask),
                "roundtrip": rt, "area": 0.1, "rle": encode(mask)}
    return {"uid": "U1", "site": entry(site_rt), "host": entry(host_rt) if host_rt is not None else None}


def _reg():
    mask = np.zeros((8, 8), bool)
    mask[2:5, 2:5] = True
    field = {"method": "m", "stats": {"s_covered": 0.9}, **encode(mask), "area_frac": 0.14, "s_covered": 0.9}
    return {"selection": encode(mask), "support": encode(mask), "edited_region": field}


def test_add_takes_its_host_when_the_spot_token_fails():
    unit, spec = _unit("add")
    out = g.merged_unit("D12", unit, spec, _tok(0.3, 0.8), _reg(), {})
    assert out["selection"]["source"] == "host" and out["token"]["roundtrip"] == 0.8
    assert out["selection"]["site"]["roundtrip"] == 0.3 and g.bad_unit(out) == []
    out = g.merged_unit("D12", unit, spec, _tok(0.7, 0.9), _reg(), {})
    assert out["selection"]["source"] == "site" and out["token"]["roundtrip"] == 0.7
    out = g.merged_unit("D12", unit, spec, _tok(0.3, 0.4), _reg(), {})
    assert out["selection"]["source"] == "site" and g.bad_unit(out) == ["roundtrip_low"]


def test_half_done_and_text_tags():
    unit, spec = _unit("replace")
    reg = _reg()
    reg["edited_region"]["stats"]["s_covered"] = 0.3
    out = g.merged_unit("D13", unit, spec, _tok(0.9), reg, {})
    assert out["half_done"] and g.bad_unit(out) == ["half_done"]
    unit, spec = _unit("add", generation_context={"text_verification": {
        "requirement": {"mode": "exact", "text": "OPEN"}, "decision": "pass", "version": "text-edits-5"}})
    tag = g.merged_unit("D13", unit, spec, _tok(0.9), _reg(), {})["text"]
    assert tag["role"] == "render" and tag["requested"] == "OPEN" and tag["verdict"] == "pass"
    unit, spec = _unit("attribute", generation_context={"text_verification": {
        "requirement": {"mode": "exact", "text": "CITY BUS"}, "decision": "pass"}})
    spec["attributes"] = [{"property": "text", "value": "CITY BUS"}]
    out = g.merged_unit("D13", unit, spec, _tok(0.9), _reg(), {})
    assert out["op"] == "text" and out["text"]["role"] == "change"


def test_recompose_restores_only_the_removed_support(tmp_path):
    rng = np.random.default_rng(1)
    src = rng.integers(0, 255, size=(40, 50, 3), dtype=np.uint8)
    joint = src.copy()
    a = np.zeros((40, 50), bool)
    a[2:10, 2:10] = True
    b = np.zeros((40, 50), bool)
    b[20:30, 30:45] = True
    joint[a] = 255
    joint[b] = 0
    Image.fromarray(src).save(tmp_path / "s.png")
    Image.fromarray(joint).save(tmp_path / "j.png")
    restored = g.recompose(tmp_path / "s.png", tmp_path / "j.png", [encode(b)], tmp_path / "out" / "r.png")
    new = np.asarray(Image.open(tmp_path / "out" / "r.png"))
    assert restored == b.sum()
    assert (new[b] == src[b]).all() and (new[a] == joint[a]).all() and (new[~(a | b)] == src[~(a | b)]).all()
