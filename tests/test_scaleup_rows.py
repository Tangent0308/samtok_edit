"""Row helpers of the two-stage retrain (docs/03): single/multi draws, the description cap, D12 twin pairs, the
same-category highlight pairs and the area buckets of the highlight regions."""
import random

import numpy as np

from samtok_edit21.preparation import round2_data as rd


def unit(op="remove", payload=None):
    return {"op": op, "instruction_spec": {"operation": op, "payload": payload}, "text": {"role": None},
            "edited_region": {"area_frac": 0.01}}


def test_merged_match_single_and_multi():
    single = {"family": "D13", "units": [unit()]}
    multi = {"family": "D13", "units": [unit(), unit("add")]}
    assert rd.merged_match(single, {"op": "single"}) and not rd.merged_match(multi, {"op": "single"})
    assert rd.merged_match(multi, {"op": "multi"}) and not rd.merged_match(single, {"op": "multi"})
    assert rd.merged_match(single, {"family": ["D1", "D13"], "op": "remove"})
    assert not rd.merged_match(single, {"family": "D15"})


def test_payload_cap_keeps_multi_records_and_caps_singles_of_the_listed_families():
    records = [{"id": f"D13:m{k}", "family": "D13", "units": [unit("add", "a small bird"), unit("remove")]} for k in range(3)]
    records += [{"id": f"D15:s{k}", "family": "D15", "units": [unit("add", "small bird")]} for k in range(5)]
    records += [{"id": f"D11:s{k}", "family": "D11", "units": [unit("add", "small bird")]} for k in range(5)]
    capped = rd.payload_capped(records, {"limit": 4, "families": ["D13", "D15"]})
    # the three multi records count first ("a small bird" and "small bird" are one description), one single is left
    assert len(capped) == 4 and all(i.startswith("D15:") for i in capped)


def test_twin_pairs_need_both_members():
    def member(i, partner):
        return {"id": i, "twin": {"partner": partner}}
    options = [member("D12:g1-U1", "D12:g1-U2"), member("D12:g1-U2", "D12:g1-U1"), member("D12:g2-U1", "D12:g2-U2")]
    pairs = rd.twin_pairs(options)
    assert [(a["id"], b["id"]) for a, b in pairs] == [("D12:g1-U1", "D12:g1-U2")]


def entry_of(boxes, categories, size=100):
    from pycocotools import mask as mu
    masks = []
    for box, category in zip(boxes, categories):
        m = np.zeros((size, size), np.uint8)
        m[box[1]:box[3], box[0]:box[2]] = 1
        rle = mu.encode(np.asfortranarray(m))
        rle = {"size": list(rle["size"]), "counts": rle["counts"].decode()}
        masks.append({"rle": rle, "dec": rle, "category": category, "area": float(m.mean()), "roundtrip": 0.9,
                      "dec_area": float(m.mean()), "span": "<|mt_start|><|mt_0001|><|mt_0300|><|mt_end|>"})
    for m, rel in zip(masks, rd.relations(masks)):
        m["rel"] = rel
    return {"source_id": "sa_1", "masks": masks, "size": [size, size]}


def test_same_category_partner_is_another_instance_of_similar_size():
    # 0 and 2 are chairs of similar size, 3 a much larger chair, 1 a table next to 0
    entry = entry_of([(10, 10, 20, 20), (21, 10, 31, 20), (60, 60, 72, 72), (40, 70, 99, 99)],
                     ["chair", "table", "chair", "chair"])
    rng = random.Random(0)
    assert {rd.pair_partner(entry, 0, rng, "same") for _ in range(10)} == {2}
    assert rd.pair_partner(entry, 0, rng, "adjacent") == 1
    assert rd.pair_partner(entry, 1, rng, "same") is None


def test_highlight_pairs_follow_the_kind_weights_and_share_prompt_and_colour():
    entry = entry_of([(10, 10, 20, 20), (21, 10, 31, 20), (60, 60, 72, 72)], ["chair", "table", "chair"])
    units = rd.highlight_pairs([entry], random.Random(1), 6, "t", shares=(0, 0, 0, 0, 1, 0, 0),
                               kinds={"same": 1.0})
    assert len(units) == 6
    for a, b in units:
        assert a["pair_kind"] == b["pair_kind"] == "same" and a["pair_key"] == b["pair_key"]
        assert a["template"] == b["template"] and a["colors"] == b["colors"] and a["masks"] != b["masks"]
        assert b["masks"][0] in a["negatives"][0] and a["masks"][0] in b["negatives"][0]


def test_pick_regions_follows_the_bucket_shares():
    sizes = [3, 4, 6, 9, 12, 18, 30]          # one region per bucket on a 100 x 100 image (0.09 % ... 9 %)
    boxes, x = [], 0
    for s in sizes:
        boxes.append((x, 0, x + s, s))
        x += s + 1
    entry = entry_of(boxes, [None] * len(sizes))
    assert [rd.bucket(m["area"]) for m in entry["masks"]] == list(range(7))
    shares = (0.15, 0.175, 0.175, 0.125, 0.125, 0.15, 0.10)
    draws = rd.pick_regions([entry], random.Random(0), 400, shares=shares)
    counts = np.bincount([m for _, m in draws], minlength=7)
    assert list(counts) == [round(400 * s) for s in shares]
