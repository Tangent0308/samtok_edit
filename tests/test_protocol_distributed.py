"""Four-node protocol adaptation (src/samtok_edit21/distributed/protocol_adapt.py): merge and helpers; no servers."""
import json

from samtok_edit21.distributed import protocol_adapt as dist
from samtok_edit21.preparation import instruction_protocol as ip


def record(i, status, operation="remove", **extra):
    out = {"id": i, "dataset": "crispedit", "type_before": operation, "instruction_before": "x", "status": status,
           "adapter_version": "a", "instruction_protocol_version": ip.VERSION, "flags": [], **extra}
    if status == "accepted":
        placement = {"relation": "on", "anchor_mode": "support"} if operation == "add" else None
        spec = ip.make_spec(operation, "the jar", "jar", "a lid" if operation == "add" else "", placement=placement)
        out.update(instruction_spec=spec, instruction_ref=ip.render(spec), instruction_noref=ip.render(spec, False),
                   mask_semantics=ip.mask_contract(spec), execution_region=None, edited_region={"rle_size": [1, 1]},
                   selection={"source": "reused", "rle_size": [1, 1], "rle_counts": "0",
                              **({"granularity": "object"} if operation == "add" else {})},
                   host_view=None, slots={"observed": "kept out of the manifest"})
    return out


def test_merge_keeps_input_order_and_splits_by_status(tmp_path):
    rows = [{"id": f"r{k}", "source_image": f"images/{k}_a.jpg", "target_image": f"images/{k}_b.jpg"} for k in range(4)]
    node0 = [record("r2", "rejected", reason="audit:outline"), record("r0", "accepted", "add")]
    node1 = [record("r3", "multi_object", reason="two cars"), record("r1", "accepted")]
    finals = []
    for k, part in enumerate((node0, node1)):
        path = tmp_path / f"final{k}.jsonl"
        dist.write_jsonl(path, part)
        finals.append(path)
    common = {"run_id": "run", "git_commit": "abc"}
    summary = dist.merge(rows, finals, tmp_path / "out", common, {"adapt_hours_max": 0.1})
    assert summary["status"] == {"accepted": 2, "rejected": 1, "multi_object": 1}
    assert summary["accepted_by_operation"] == {"add": 1, "remove": 1} and summary["add_granularity"] == {"object": 1}
    assert summary["rejected_by_reason"] == {"audit": 1}
    read = lambda name: [json.loads(line) for line in (tmp_path / "out" / name).read_text().splitlines()]  # noqa: E731
    assert [r["id"] for r in read("records.jsonl")] == ["r0", "r1", "r2", "r3"]
    manifest = read("manifest_protocol.jsonl")
    assert [r["id"] for r in manifest] == ["r0", "r1"] and manifest[0]["source_image"] == "images/0_a.jpg"
    assert set(manifest[0]) == {"source_image", "target_image", *dist.MANIFEST_FIELDS} and "slots" not in manifest[0]
    assert [r["id"] for r in read("multi_object.jsonl")] == ["r3"]
    assert "accepted 2, multi-object 1, rejected 1" in (tmp_path / "out" / "README.md").read_text()
    try:
        dist.merge(rows + [{"id": "r9", "source_image": "a", "target_image": "b"}], finals, tmp_path / "out2", common, {})
    except RuntimeError as error:
        assert "no final record" in str(error)
    else:
        raise AssertionError("a row without a final record must fail the merge")


def test_merged_records_let_later_files_win(tmp_path):
    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    dist.write_jsonl(first, [{"id": "x", "v": 1}, {"id": "y", "v": 1}])
    dist.write_jsonl(second, [{"id": "y", "v": 2}])
    assert dist.merged_records([first, second, tmp_path / "missing.jsonl"]) == {"x": {"id": "x", "v": 1}, "y": {"id": "y", "v": 2}}
    assert dist.count_lines(first) == 2 and dist.count_lines(tmp_path / "missing.jsonl") == 0
