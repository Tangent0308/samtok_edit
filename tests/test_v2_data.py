import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pycocotools import mask as coco_mask

from samtok_edit21.data.io import file_hash, row_hash
from samtok_edit21.data.protocol import (
    REC_TEMPLATE, box_of, parse_cot, regions_in, span_of, to_cot, validate_row,
)
from samtok_edit21.preparation.v2_data import convert

W, H = 40, 20
A, B, C = span_of([3, 300]), span_of([40, 450]), span_of([7, 260])


def rle(mask):
    encoded = coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"rle_size": list(encoded["size"]), "rle_counts": encoded["counts"].decode("ascii")}


def box_mask(x0, y0, x1, y1):
    mask = np.zeros((H, W), bool)
    mask[y0:y1, x0:x1] = True
    return mask


def v1_rows(record_id, edit_type, instruction, label, spans, noref, *, image=True):
    common = {"edit_image": f"/{record_id}/s.png", "edit_type": edit_type}
    ntp = {**common, "sample_type": "edit_ntp", "prompt": instruction,
           "mt_cot": to_cot([(span, label) for span in spans])}
    fm = {**common, "image": f"/{record_id}/t.png"}
    joined = "".join(spans)
    ref = instruction.rstrip(".") + " " + joined + "."
    return [ntp, {**fm, "sample_type": "edit", "prompt": instruction},
            {**fm, "sample_type": "edit_umt", "instr_variant": "ref", "prompt": ref},
            {**fm, "sample_type": "edit_umt", "instr_variant": "noref", "prompt": noref.format(joined)}]


def build_v1(root):
    union = root / "union.png"
    Image.fromarray(box_mask(4, 2, 12, 10).astype(np.uint8) * 255).save(union)
    left, right = box_mask(2, 5, 8, 15), box_mask(30, 1, 36, 9)
    host = box_mask(10, 10, 30, 20)
    records = [
        # Single-unit add with an aggregate dataset mask.
        ("refedit-add", "refedit", {"dataset_mask": str(union), "instances": []},
         {"units": [{"edit_type": "add", "ref_phrase": "red ball", "mask_ids": ["union"]}]},
         v1_rows("refedit-add", "add", "Add a red ball.", "red ball", [A], "Add a red ball in this region {}.")),
        # Two instances: v1 spans follow the codec's x-center order (right stored first).
        ("crisp-add2", "crispedit", {"dataset_mask": None, "instances": [
            {"instance_id": "r", **rle(right)}, {"instance_id": "l", **rle(left)}]},
         {"units": [{"edit_type": "add", "ref_phrase": "cups", "mask_ids": ["r", "l"]}]},
         v1_rows("crisp-add2", "add", "Add cups.", "one of the cups", [B, C], "Add cups in this region {}.")),
        ("derived-add", "derived", {"dataset_mask": None, "instances": [{"instance_id": "h", **rle(host)}]},
         {"units": [{"edit_type": "add", "ref_phrase": "sticker to the lamp", "mask_ids": ["union"]}]},
         v1_rows("derived-add", "add", "Add a sticker to the lamp.", "sticker to the lamp", [A],
                 "Add a sticker in this region {}.")),
        ("derived-remove", "derived", {"dataset_mask": None, "instances": [{"instance_id": "h", **rle(host)}]},
         {"units": [{"edit_type": "remove", "ref_phrase": "lamp", "mask_ids": ["union"]}]},
         v1_rows("derived-remove", "remove", "Remove the lamp.", "lamp", [C],
                 "Remove the object in this region {}.")),
        ("refedit-composite", "refedit", {"dataset_mask": str(union), "instances": []},
         {"units": []},
         [{**row, "edit_type": "composite"} for row in
          v1_rows("refedit-composite", "remove", "Remove the lamp.", "lamp", [C],
                  "Remove the object in this region {}.")]),
        ("scale-plain", "scaleedit", {"dataset_mask": None, "instances": []}, None,
         [{"edit_image": "/p/s.png", "image": "/p/t.png", "edit_type": "action",
           "sample_type": "edit", "prompt": "Make it rain."}]),
    ]
    inputs, encoded, provenance = [], [], []
    for index, (uid, dataset, source, annotation, rows) in enumerate(records):
        source = {"id": uid, "dataset": dataset, "source_size": [W, H], **source}
        inputs.append({"index": index, "source": source, "annotation": annotation,
                       "conversion_method": "llm" if annotation else "plain_only"})
        encoded.append({"index": index, "id": uid, "dataset": dataset,
                        "conversion_method": "llm" if annotation else "plain_only", "rows": rows})
        provenance += [{"id": uid, "row_hash": row_hash(row)} for row in rows]
    (root / "input-00.jsonl").write_text("".join(json.dumps(x) + "\n" for x in inputs))
    worker = root / "encoded" / "worker-00"
    worker.mkdir(parents=True)
    (worker / "chunk-00000.jsonl").write_text("".join(json.dumps(x) + "\n" for x in encoded))
    (worker / "chunk-00000.receipt.json").write_text(json.dumps({"sha256": file_hash(worker / "chunk-00000.jsonl")}))
    (worker / "complete.json").write_text(json.dumps({"chunks": 1}))
    (root / "inputs.json").write_text(json.dumps(
        {"workers": 1, "input_sha256": {"input-00": file_hash(root / "input-00.jsonl")}}))
    (root / "provenance.jsonl").write_text("".join(json.dumps(x) + "\n" for x in provenance))
    (root / "metadata_report.json").write_text(json.dumps(
        {"provenance_sha256": file_hash(root / "provenance.jsonl"), "stage1_sha256": "s1",
         "stage2_sha256": "s2", "codec_sha256": "codec"}))
    return left, right


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def test_v2_conversion_boxes_retypes_and_replay(tmp_path):
    v1 = tmp_path / "v1"
    v1.mkdir()
    left, right = build_v1(v1)
    report = convert(v1, tmp_path / "v2", rec_per_ntp=1, threads=2)
    stage1, stage2 = read(tmp_path / "v2/stage1.jsonl"), read(tmp_path / "v2/stage2.jsonl")
    assert all(validate_row(row) for row in stage1 + stage2)
    assert {row["sample_type"] for row in stage1} == {"edit_ntp", "rec_ntp"}
    assert {row["sample_type"] for row in stage2} == {"edit", "edit_umt"}
    assert report["records_dropped_composite"] == 1 and not any(
        "lamp" in row["prompt"] and row["edit_type"] == "composite" for row in stage1 + stage2)
    single = box_of((100, 100, 300, 500))  # x 4..12 of 40, y 2..10 of 20
    by_image = {}
    for row in stage1 + stage2:
        by_image.setdefault(row["edit_image"], []).append(row)
    add = by_image["/refedit-add/s.png"]
    assert parse_cot(add[0]["mt_cot"]) == [(single, "red ball")]
    assert add[-1]["prompt"] == "Add a red ball in this region " + single + "."
    # The codec order (x-center) is preserved: left mask replaces the first span.
    cups = by_image["/crisp-add2/s.png"]
    expected = [box_of((50, 250, 200, 750)), box_of((750, 50, 900, 450))]
    assert [r for r, _ in parse_cot(cups[0]["mt_cot"])] == expected
    assert regions_in(cups[-1]["prompt"]) == expected
    derived = by_image["/derived-add/s.png"]
    assert {row["edit_type"] for row in derived} == {"attribute"} and A in derived[-1]["prompt"]
    replay = [row for row in stage1 if row["sample_type"] == "rec_ntp"]
    assert replay == [{"sample_type": "rec_ntp", "edit_type": "remove", "edit_image": "/derived-remove/s.png",
                       "prompt": REC_TEMPLATE.format("lamp"),
                       "mt_cot": to_cot([(box_of((250, 500, 750, 1000)), "lamp")])}]
    provenance = read(tmp_path / "v2/provenance.jsonl")
    assert len(provenance) == len(stage1) + len(stage2)
    assert {p["conversion"] for p in provenance} == {"add_box", "derived_add_as_attribute",
                                                     "unchanged", "rec_replay"}
    assert all(p["row_hash"] == row_hash(row) for p, row in zip(provenance, stage1 + stage2))
    with pytest.raises(ValueError, match="fresh"):
        convert(v1, tmp_path / "v2")


def test_v2_conversion_rejects_v1_provenance_drift(tmp_path):
    v1 = tmp_path / "v1"
    v1.mkdir()
    build_v1(v1)
    lines = (v1 / "provenance.jsonl").read_text().splitlines()
    lines[1] = json.dumps({"id": "refedit-add", "row_hash": "0" * 64})
    (v1 / "provenance.jsonl").write_text("\n".join(lines) + "\n")
    report = json.loads((v1 / "metadata_report.json").read_text())
    report["provenance_sha256"] = file_hash(v1 / "provenance.jsonl")
    (v1 / "metadata_report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="provenance"):
        convert(v1, tmp_path / "v2", threads=1)
