import json

import numpy as np
import pytest
from PIL import Image

from samtok_edit21.data.protocol import box_coords, regions_in
from samtok_edit21.evaluation.cases import build_cases, split_of
from samtok_edit21.evaluation.compile import SENTINEL, compile_case, semantic_sources
from samtok_edit21.evaluation.run import DEFAULT_ADD_BOX, export, point_box


def write_benchmark(root):
    (root / "images").mkdir(parents=True)
    (root / "regions").mkdir()
    Image.new("RGB", (40, 20)).save(root / "images/a.png")
    Image.new("RGB", (40, 20)).save(root / "images/m.png")
    for name in ("r1", "r2", "r3"):
        Image.new("L", (40, 20), 255).save(root / f"regions/{name}.png")
    region = lambda name: {"mask": f"regions/{name}.png", "box": [0, 0, 10, 10], "point": [5, 5]}
    rows = [
        {"id": "cb_1", "source_dataset": "compbench", "edit_type": "remove", "source_image": "images/a.png",
         "instruction": {"with_location_reference": "remove the left cup", "region_only": "Remove {region_1}."},
         "regions": [region("r1")]},
        {"id": "cb_2", "source_dataset": "compbench", "edit_type": "add", "source_image": "images/a.png",
         "instruction": {"with_location_reference": "add a cup", "region_only": "x"},
         "regions": [region("r1"), region("r2")]},
        {"id": "mirage_000", "source_dataset": "mirage", "edit_type": "mixed", "source_image": "images/m.png",
         "instruction": {"with_location_reference": "Change the color of the shell of the left turtle to red, "
                                                    "and add some snow onto the shell of the right turtle.",
                         "region_only": "For {region_1}, change the color of the shell to red. "
                                        "For {region_2}, add some snow onto the shell."},
         "regions": [region("r2"), region("r3")]},
    ]
    (root / "benchmark.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    selection = {"candidate_id": "mirage_000", "selected_atomic_edits": [
        {"region": 1, "operation": "replace", "refer_object": "the shell of the left turtle"},
        {"region": 2, "operation": "add", "refer_object": "the shell of the right turtle"}]}
    (root / "selection.jsonl").write_text(json.dumps(selection) + "\n")


def test_cases_split_mirage_and_skip_two_region_rows(tmp_path):
    write_benchmark(tmp_path)
    cases = build_cases(tmp_path, tmp_path / "selection.jsonl")
    assert [c["case_id"] for c in cases] == ["cb_1", "mirage_000#r1", "mirage_000#r2"]
    first, second = cases[1], cases[2]
    assert first["instruction"] == "Change the color of the shell of the left turtle to red."
    assert second["instruction"] == "Add some snow onto the shell of the right turtle."
    assert second["region_instruction"] == "For {region_1}, add some snow onto the shell."
    assert first["region"]["mask"].endswith("r2.png") and second["region"]["mask"].endswith("r3.png")
    assert first["benchmark_type"] == "replace" and second["benchmark_type"] == "add"
    assert first["split"] == second["split"] == split_of("images/m.png")
    sources = semantic_sources(cases)
    assert sources[0]["provisional_type"] == "remove" and sources[1]["provisional_type"] is None


def test_compile_uses_converter_and_falls_back_to_templates():
    case = {"case_id": "c", "instruction": "Remove the left cup.", "benchmark_type": "remove"}
    accepted = {"status": "accepted", "conversion_method": "llm", "annotation": {
        "units": [{"edit_type": "remove", "ref_phrase": "left cup", "type_resolution": "dataset_mapping"}],
        "noref_instruction": "Remove the object in this region {mask_0}."}}
    entry = compile_case(case, accepted)
    assert entry["noref_template"] == "Remove the object in this region {region}." and entry["valid_noref"]
    assert entry["ref_template"] == "Remove the left cup {region}."
    attribute = {**accepted, "annotation": {"units": [{"edit_type": "attribute", "ref_phrase": "shell"}],
                                            "noref_instruction": "Change the color of this region {mask_0} to red."}}
    assert compile_case({**case, "benchmark_type": "replace"}, attribute)["region_kind"] == "mask"
    fallback = compile_case({"case_id": "a", "instruction": "Add a cup.", "benchmark_type": "add"},
                            {"status": "failed"})
    assert fallback["method"] == "template" and fallback["region_kind"] == "box"
    assert fallback["noref_template"] == "Add a cup in this region {region}." and fallback["valid_noref"]
    assert fallback["ref_template"] is None
    prompt = fallback["noref_template"].replace("{region}", SENTINEL["box"])
    assert regions_in(prompt) == [SENTINEL["box"]]


def test_point_default_box_and_export():
    box = box_coords(point_box((20, 10), 40, 20))
    assert (box[2] - box[0], box[3] - box[1]) == DEFAULT_ADD_BOX
    assert box_coords(point_box((0, 0), 40, 20))[:2] == (0, 0)
    assert box_coords(point_box((40, 20), 40, 20))[2:] == (1000, 1000)
    rgba = Image.new("RGBA", (64, 32), (255, 0, 0, 0))
    out = export(rgba, Image.new("RGB", (40, 20)))
    assert out.mode == "RGB" and out.size == (40, 20) and np.asarray(out).min() == 255


def test_judge_row_contract(tmp_path):
    from samtok_edit21.evaluation.manifest import judge_row

    write_benchmark(tmp_path)
    case = build_cases(tmp_path, tmp_path / "selection.jsonl")[1]
    Image.new("RGB", (40, 20)).save(tmp_path / "out.png")
    row = judge_row(case, "b0", "mask+blend", tmp_path / "out.png", lambda r: "digest", {})
    assert row["sample_id"] == "mirage_000#r1/b0/mask+blend" and row["delivery_status"] == "available"
    assert row["split"] == "unassigned" and row["v2_split"] == case["split"] and len(row["regions"]) == 1
    missing = judge_row(case, "b0", "mask", tmp_path / "none.png", lambda r: "digest", {})
    assert missing["delivery_status"] == "missing_output" and missing["output_sha256"] is None
