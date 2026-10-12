"""Type selection of a pair manifest (scripts/data_cleaning/select_types.py) on a synthetic manifest."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "data_cleaning"))
import select_types as st  # noqa: E402


def write_manifest(path, types):
    rows = [{"id": f"pair-{i}", "dataset": "a" if i % 2 else "b", "type": t, "instruction": "x" * i}
            for i, t in enumerate(types)]
    # Odd spacing on purpose: kept lines must be copied byte for byte, not re-serialized.
    path.write_text("".join(json.dumps(r, separators=(", ", ":  ")) + "\n" for r in rows))
    return path.read_bytes().splitlines(keepends=True)


def test_drops_types_and_keeps_lines_byte_identical(tmp_path):
    lines = write_manifest(tmp_path / "in.jsonl", ["add", "text", "remove", "action", None, "attribute", "text"])
    summary = st.select(tmp_path / "in.jsonl", ["text", " Action "], tmp_path / "out.jsonl", tmp_path / "out.json")
    assert (tmp_path / "out.jsonl").read_bytes().splitlines(keepends=True) == [lines[i] for i in (0, 2, 4, 5)]
    assert (summary["input_count"], summary["kept_count"], summary["dropped_count"]) == (7, 4, 3)
    assert summary["dropped_types"] == ["action", "text"]
    assert summary["dropped_by_type"] == {"action": 1, "text": 2}
    assert summary["kept_by_type"] == {"add": 1, "attribute": 1, "none": 1, "remove": 1}
    assert summary["kept_by_dataset_type"] == {"a": {"attribute": 1}, "b": {"add": 1, "none": 1, "remove": 1}}
    assert json.loads((tmp_path / "out.json").read_text())["output_sha256"] == st.file_hash(tmp_path / "out.jsonl")


def test_none_means_untyped_and_outputs_are_never_overwritten(tmp_path):
    write_manifest(tmp_path / "in.jsonl", ["add", None, "remove"])
    summary = st.select(tmp_path / "in.jsonl", ["none"], tmp_path / "out.jsonl", tmp_path / "out.json")
    assert summary["dropped_by_type"] == {"none": 1} and summary["kept_count"] == 2
    with pytest.raises(FileExistsError):
        st.select(tmp_path / "in.jsonl", ["add"], tmp_path / "out.jsonl", tmp_path / "other.json")
    with pytest.raises(ValueError):
        st.select(tmp_path / "in.jsonl", [" "], tmp_path / "new.jsonl", tmp_path / "new.json")
