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
