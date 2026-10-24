"""Merge of the edit data (docs/01 section 2.13): default-view rules, caps, text tags, image names."""
from samtok_edit21.preparation import merge_data as md


def unit(op="replace", payload="a red bicycle", rt=0.9, half=False, verdict=None, role=None):
    spec = {"operation": op, "payload": payload, "attributes": [], "scope": ""}
    return {"uid": "U1", "instruction_spec": spec, "op": op, "token": {"span": "s", "dec": {}, "roundtrip": rt},
            "half_done": half, "text": {"role": role, "requested": None, "verdict": verdict, "reason": None}}


def record(rid, family="D11", units=None, group="sha1:a", split="train", size_ok=True):
    return {"id": rid, "family": family, "group": group, "split": split, "size_ok": size_ok,
            "units": units or [unit()], "exclude": []}


def test_hard_exclusions():
    assert md.hard_exclusions(record("a")) == []
    assert md.hard_exclusions(record("b", units=[unit(rt=0.4)])) == ["roundtrip_low"]
    assert md.hard_exclusions(record("c", size_ok=False, units=[unit(half=True)])) == ["size_mismatch", "half_done"]
    assert md.hard_exclusions(record("d", units=[unit(), unit(verdict="fail", role="render")])) == ["text_fail"]
    no_token = record("e")
    no_token["units"][0]["token"] = None
    assert md.hard_exclusions(no_token) == ["no_token"]


def test_payload_key_and_cap():
    assert md.payload_key({"operation": "add", "payload": "A  wooden, bench"}) == "wooden bench"
    assert md.payload_key({"operation": "remove", "payload": ""}) is None
    recs = [record(f"s{k}", group=f"sha1:{k}") for k in range(md.PAYLOAD_CAP + 5)]
    recs.append(record("m", units=[unit(), unit()], group="sha1:m"))
    md.apply_caps(recs)
    capped = [r for r in recs if "payload_cap" in r["exclude"]]
    assert len(capped) == 7 and "payload_cap" not in recs[-1]["exclude"]   # the pair counts 2 and is never dropped


def test_per_image_cap_keeps_family_order_and_spares_d1():
    recs = [record(f"d3-{k}", family="D3", units=[unit(payload=f"p{k}")]) for k in range(3)]
    recs += [record(f"d11-{k}", family="D11", units=[unit(payload=f"q{k}")]) for k in range(3)]
    recs += [record(f"d1-{k}", family="D1", units=[unit(payload=f"r{k}")]) for k in range(5)]
    md.apply_caps(recs)
    out = {r["id"] for r in recs if "per_image_cap" in r["exclude"]}
    assert out == {"d3-0", "d3-1", "d3-2"}


def test_text_tag_and_half_done():
    verdicts = {("D3", "r1", 0): {"text_role": "change", "requested": "OPEN", "verdict": "pass", "reason": None},
                ("D3", "r2", 0): {"text_role": None, "requested": None, "verdict": None, "reason": None}}
    assert md.text_tag("D3", "r1", 0, {"operation": "attribute"}, verdicts)["role"] == "change"
    assert md.text_tag("D3", "r2", 0, {"operation": "add", "payload": "a child reading"}, verdicts)["role"] is None
    assert md.text_tag("D3", "r3", 0, {"operation": "remove", "scope": "logo"}, verdicts)["role"] == "remove"
    region = {"stats": {"s_covered": 0.8}}
    assert md.half_done({"operation": "add"}, region, "D1") and not md.half_done({"operation": "add"}, region, "D3")
    assert md.half_done({"operation": "replace"}, {"stats": {"s_covered": 0.3}}, "D11")


def test_image_names():
    d1 = {"family": "D1", "source_ref": "abcdef", "sa1b": {"source_id": "sa_1"}}
    assert md.image_name(d1, "source", "/x/sa_1.jpg") == "images/D1/source/sa_1.jpg"
    assert md.image_name(d1, "target", "/x/abcdef.png") == "images/D1/target/ab/abcdef.png"
    d3 = {"family": "D3", "source_ref": "refedit-12"}
    assert md.image_name(d3, "source", "/y/a.png") == "images/D3/12/refedit-12.source.png"


def test_merged_regions_single_multi_negatives_and_local_loss():
    import numpy as np
    from samtok_edit21.preparation import round2_data as rd
    from samtok_edit21.training.round2 import merged_regions
    shape = (40, 60)

    def rle(y0, y1, x0, x1):
        m = np.zeros(shape, bool)
        m[y0:y1, x0:x1] = True
        e = rd.encode_rle(m)
        return {"rle_size": e["size"], "rle_counts": e["counts"]}, e

    s1, d1 = rle(5, 15, 5, 15)
    s2, d2 = rle(20, 30, 30, 40)
    spec = {"version": "2.0", "operation": "remove", "ref": "the cup", "scope": "cup", "payload": "", "target_part": "",
            "remove_mode": "whole", "attributes": [], "placement": {}, "qualifier": ""}
    unit = lambda s, d: {"selection": s, "execution_region": s, "edited_region": s, "instruction_spec": spec,  # noqa: E731
                         "token": {"span": "<|mt_start|><|mt_0001|><|mt_0002|><|mt_end|>", "dec": d}}
    single = {"units": [unit(s1, d1)], "loss_region": None}
    loss, route, token, sentences, negative, local = merged_regions(
        single, {"form": "noref_token", "negatives": [0]}, shape, pool_mask=lambda j: rd.decode_rle(d2).astype(bool))
    assert len(loss) == 1 and loss[0].sum() == 100 and token[0].sum() == 100 and local is None
    assert negative.sum() == 100 and not (negative & loss[0]).any() and "selected region" in sentences[0]
    pair = {"units": [unit(s1, d1), unit(s2, d2)], "loss_region": "local"}
    loss, route, token, sentences, negative, local = merged_regions(pair, {"order": [1, 0], "forms": ["ref_token"] * 2}, shape)
    assert [m.sum() for m in loss] == [100, 100] and loss[0][25, 35] and local.sum() == 200 and not negative.any()
    assert "the cup" in sentences[0]


def test_merged_draw_matching():
    from samtok_edit21.preparation.round2_data import merged_match
    single = {"family": "D11", "units": [{"op": "replace", "text": {"role": "render"}}]}
    pair = {"family": "D4", "units": [{"op": "remove", "text": {"role": None}}, {"op": "add", "text": {"role": None}}]}
    assert merged_match(single, {"family": ["D10", "D11"], "op": "replace", "text_role": "render"})
    assert not merged_match(single, {"family": "D11", "op": "add"})
    assert not merged_match(single, {"op": "multi"}) and merged_match(pair, {"op": "multi"})
    assert not merged_match(pair, {"op": "remove"}) and merged_match(pair, {"family": "D4"})
