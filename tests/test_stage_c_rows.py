"""Stage-C row helpers (docs/03 section 6): pairs stay together, draws are exact, half-done targets are recognised."""
import random

import pytest

from samtok_edit21.preparation import round2_data as rd


def test_c2_units_keep_pairs_together():
    rows = [{"kind": "hl", "pair_key": "p1"}, {"kind": "edit"}, {"kind": "hl", "pair_key": "p1"}, {"kind": "edit_v1"}]
    units = rd.c2_units(rows)
    assert [len(u) for u in units] == [2, 1, 1] and units[0][1] is rows[2]
    with pytest.raises(ValueError):
        rd.c2_units(rows + [{"kind": "hl", "pair_key": "p1"}])


def test_draw_units_is_exact_and_skips_pairs_that_overshoot():
    units = [[{"k": i}, {"k": i}] for i in range(5)] + [[{"k": 10 + i}] for i in range(3)]
    for need in (1, 4, 7, 13):
        drawn = rd.draw_units(units, need, random.Random(need))
        assert sum(map(len, drawn)) == need
    with pytest.raises(ValueError):
        rd.draw_units(units, 14, random.Random(0))


def test_half_done_targets():
    assert rd.half_done("replace", 0.4) and rd.half_done("attribute", 0.49)
    assert not rd.half_done("replace", 0.5) and not rd.half_done("remove", 0.1)
    assert rd.half_done("add", 0.7) and not rd.half_done("add", 0.6) and not rd.half_done("add", None)
    text = {"operation": "attribute", "attributes": [{"property": "text"}]}
    assert rd.spec_op(text) == "text" and rd.spec_op({"operation": "remove", "attributes": []}) == "remove"


def test_stage_c_fills_the_slots():
    units = [[{"kind": "edit", "i": i}] for i in range(60)] + [[{"kind": "hl", "i": i}, {"kind": "hl", "i": i}] for i in range(2)]
    rows = rd.place_in_slots(units, random.Random(0), updates=2, world=8, accumulation=4)
    assert len(rows) == 64 and all(r is not None for r in rows)
