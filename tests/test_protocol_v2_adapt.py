import hashlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from samtok_edit21.preparation import instruction_protocol_v2 as p2
from samtok_edit21.preparation import protocol_adapt as pa
from samtok_edit21.preparation import protocol_v2_adapt as pv


def spec(**kw):
    base = dict(version="1.0", operation="attribute", ref="the red coat of the woman on the right", scope="coat",
                payload="", target_part="", remove_mode="whole", attributes=[{"property": "color", "value": "navy blue"}],
                placement={}, qualifier="")
    base.update(kw)
    return base


def test_vendored_module_is_the_benchmark_file():
    assert hashlib.sha256(Path(p2.__file__).read_bytes()).hexdigest() == pv.PROTOCOL_SHA256


def test_noref_names_no_target_and_ref_is_unchanged():
    s, ref, noref, reason = pv.sentences(spec(), "Change the color of the red coat of the woman on the right to navy blue.")
    assert s["version"] == "2.0" and reason is None
    assert noref == "Change the color of the selected region to navy blue."
    add = spec(operation="add", attributes=[], payload="a red tag", ref="the left suitcase", scope="suitcase",
               placement={"relation": "clipped to", "site": "the handle", "anchor_mode": "support"})
    _, _, noref, _ = pv.sentences(add, "Add a red tag clipped to the handle of the left suitcase.")
    assert noref == "Add a red tag clipped to the selected region."
    with pytest.raises(ValueError):
        pv.sentences(spec(), "Change the color of the coat to navy blue.")


def test_qualifiers_follow_the_module():
    rem = spec(operation="remove", attributes=[], ref="the dot", scope="dot")
    keep = {**rem, "qualifier": "preserving the underlying surface"}
    _, _, noref, _ = pv.sentences(keep, "Remove the dot, preserving the underlying surface.")
    assert noref == "Remove the selected region, preserving the underlying surface."
    named = {**rem, "qualifier": "preserving the plant growing in it"}
    _, _, noref, _ = pv.sentences(named, "Remove the dot, preserving the plant growing in it.")
    assert noref == "Remove the selected region."
    scope = {**rem, "qualifier": "including its pedestal"}
    _, _, noref, reason = pv.sentences(scope, "Remove the dot, including its pedestal.")
    assert noref is None and reason == "unclassified_qualifier"  # the module rejects it; ref-only


def test_main_item_and_site_spot():
    changed = np.zeros((200, 200), bool)
    changed[20:40, 20:40] = True     # the new item
    changed[150:153, 150:153] = True  # a regenerated speck on the host
    item = pv.main_item(changed)
    assert item[30, 30] and not item[151, 151]
    host = np.zeros((200, 200), bool)
    host[10:190, 10:190] = True
    spot, info = pv.site_spot(host, item)
    assert spot is not None and spot.sum() < host.sum() and spot[30, 30] and not spot[180, 180]
    assert pv.site_spot(host, None) == (None, "no_footprint")


def test_box_and_point():
    m = np.zeros((50, 80), bool)
    m[10:20, 30:60] = True
    assert pv.tight_box(m) == [30, 10, 60, 20]
    x, y = pv.deepest_point(m)
    assert m[y, x] and 14 <= y <= 15 and 34 <= x <= 55


def test_adapt_v1_contents_and_site(tmp_path):
    h = w = 100
    s_mask = np.zeros((h, w), bool); s_mask[20:60, 20:60] = True
    contents = np.zeros((h, w), bool); contents[15:25, 30:50] = True
    rec = {"instruction_spec": spec(operation="remove", attributes=[], ref="the bowl", scope="bowl", remove_mode="contents"),
           "instruction_ref": "Remove the bowl, including its contents.", "instruction_noref": "x",
           "selection": {**pa.encode(s_mask), "source": "reused"}, "execution_region": pa.encode(s_mask | contents)}
    out = pv.adapt_v1(rec)
    assert out["instruction_noref"] == "Remove the selected region, including its contents."
    assert out["protocol_v2_change"] == "selection_s_with_contents" and pa.decode(out["selection"]).sum() == (s_mask | contents).sum()
    host = np.zeros((h, w), bool); host[5:95, 5:95] = True
    item = np.zeros((h, w), bool); item[8:14, 8:14] = True
    add = {"instruction_spec": spec(operation="add", attributes=[], payload="a lamp", ref="the wall", scope="wall",
                                    placement={"relation": "mounted on", "site": "the upper left corner", "anchor_mode": "support"}),
           "instruction_ref": "Add a lamp mounted on the upper left corner of the wall.", "instruction_noref": "x",
           "selection": {**pa.encode(host), "granularity": "object"}, "execution_region": pa.encode(item)}
    out = pv.adapt_v1(add)
    assert out["instruction_noref"] == "Add a lamp mounted on the selected region."
    new = pa.decode(out["selection"])
    assert out["selection_v1"] and new.sum() < host.sum() and new[10, 10]
