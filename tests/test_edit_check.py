"""MLLM edit check (samtok_edit21.preparation.edit_check and its ARNOLD merge) on synthetic pairs."""
import asyncio
import json

import numpy as np
import pytest

mask_utils = pytest.importorskip("pycocotools.mask")
from PIL import Image  # noqa: E402

from samtok_edit21.distributed import edit_check as dist  # noqa: E402
from samtok_edit21.preparation import edit_check as ec  # noqa: E402

# Prompts, views and sampling validated in the pilot (docs/09 section 6.11).  A change here
# invalidates the pilot's recall / false-alarm numbers: re-validate before updating the hash.
PROTOCOL_SHA256 = "24057dee1266da843f6a4074fd47f8ce1c4df977c27d8bdaf4932c059c2f511c"


def test_protocol_is_the_validated_one():
    assert ec.protocol()["sha256"] == PROTOCOL_SHA256


def test_prompt_edit_types():
    row = {"type": "text", "instruction": ' Make the sign say "OPEN" '}
    assert ec.op_of("action") == ec.op_of("text") == "attribute"
    assert ec.op_of(None) == ec.op_of("background") == "other"
    prompt = ec.q2_prompt(row)
    assert "Edit type: attribute\nInstruction: Make the sign say \"OPEN\"\n" in prompt
    assert ec.QUESTIONS2["attribute"] in prompt
    judge = ec.judge2_prompt(row, 'The word "OPEN" was painted.')
    assert "\"The word 'OPEN' was painted.\"" in judge


def write_pair(tmp_path, identifier="synthetic-00", size=(800, 600), box=(300, 200, 420, 330)):
    W, H = size
    rng = np.random.default_rng(0)
    source = rng.integers(0, 200, (H, W, 3), dtype=np.uint8)
    target = source.copy()
    x0, y0, x1, y1 = box
    target[y0:y1, x0:x1] = (20, 200, 40)
    mask = np.zeros((H, W), np.uint8)
    mask[y0:y1, x0:x1] = 1
    (tmp_path / "images").mkdir(exist_ok=True)
    Image.fromarray(source).save(tmp_path / "images" / f"{identifier}.source.png")
    Image.fromarray(target).save(tmp_path / "images" / f"{identifier}.target.png")
    rle = mask_utils.encode(np.asfortranarray(mask))
    box_1000 = [x0 * 1000 // W, y0 * 1000 // H, -(-x1 * 1000 // W), -(-y1 * 1000 // H)]
    return {"id": identifier, "dataset": "synthetic", "type": "replace", "instruction": "replace the box",
            "source_image": f"images/{identifier}.source.png", "target_image": f"images/{identifier}.target.png",
            "region": {"rle_size": [H, W], "rle_counts": rle["counts"].decode(), "box_1000": box_1000,
                       "area": float(mask.mean())}}


def test_views(tmp_path):
    row = write_pair(tmp_path)
    source = Image.open(tmp_path / row["source_image"])
    target = Image.open(tmp_path / row["target_image"])
    mask = ec.region_mask(row["region"])
    assert mask.size == source.size and mask.mode == "L"
    views = ec.views(source, target, mask, row["region"]["box_1000"])
    assert [label for label, _ in views] == list(ec.VIEW_LABELS)
    full, after, close = (image for _, image in views)
    assert full.width * full.height <= ec.FULL_PIXELS and after.size == full.size
    # The contour is drawn just outside the mask; the source has no pure red pixels of its own.
    inside = np.asarray(mask.resize(full.size, Image.Resampling.NEAREST)) > 0
    red = np.all(np.asarray(full) == ec.CONTOUR_COLOR, axis=2)
    assert red.sum() > 100 and not (red & inside).any()
    # Close-up: two equal panels with a 20-pixel gap, at most CLOSE_LONG_SIDE wide.
    assert close.width <= ec.CLOSE_LONG_SIDE and (close.width - 20) % 2 == 0
    x0, y0, x1, y1 = ec.crop_box(row["region"]["box_1000"], source.size)
    assert 0 <= x0 < 300 and 0 <= y0 < 200 and x1 > 420 and y1 > 330 and x1 <= 800 and y1 <= 600
    parts = ec.image_parts(row, tmp_path)
    assert [p["type"] for p in parts] == ["text", "image_url"] * 3
    assert all(p["image_url"]["url"].startswith("data:image/jpeg;base64,") for p in parts[1::2])


def test_crop_box_stays_inside_the_image():
    assert ec.crop_box([0, 0, 50, 50], (1000, 500)) == (0, 0, 200, 200)
    assert ec.crop_box([900, 900, 1000, 1000], (1000, 500)) == (800, 300, 1000, 500)
    assert ec.crop_box([0, 0, 1000, 1000], (640, 480)) == (0, 0, 640, 480)


def test_parse_verdict():
    text = 'Thinking...</think>\n```json\n{"verdict": "fail", "problem": "partial", "reason": "one left"}\n```'
    assert ec.parse_verdict(text)["problem"] == "partial"
    for bad in ('{"verdict": "unsure"}', "no json here", '{"verdict": "pass"'):
        with pytest.raises(Exception):
            ec.parse_verdict(bad)


class FakeServer:
    """Answers by call; `script` maps a call to a list of replies, one per attempt."""

    def __init__(self, script):
        self.script, self.calls = script, []

    async def call(self, parts, prompt, call, seed):
        self.calls.append((call, seed))
        replies = self.script[call]
        return {**replies[min(seed, len(replies) - 1)], "tokens": 10}


def reply(text, finish="stop"):
    return {"text": text, "finish": finish}


def test_check_pair_retries_and_decision():
    row = {"id": "x", "dataset": "d", "type": "remove", "instruction": "remove the cup"}
    server = FakeServer({
        "q2": [reply("not json", "length"), reply('{"answers": "gone", "verdict": "pass", "problem": "none", "reason": "ok"}')],
        "describe": [reply("The cup is gone; the table is clean.")],
        "judge2": [reply('{"verdict": "fail", "problem": "defect", "reason": "smudge"}')]})
    record = asyncio.run(ec.check_pair(server, row, []))
    assert record["q2"]["verdict"] == "pass" and record["q2"]["attempts"] == 2 and record["q2"]["answers"] == "gone"
    assert record["dn2"]["verdict"] == "fail" and record["dn2"]["description"].startswith("The cup")
    assert record["keep"] is False
    assert ("q2", 0) in server.calls and ("q2", 1) in server.calls
    # Three answers without a verdict leave the check unchecked, and unchecked pairs are not kept.
    server = FakeServer({"q2": [reply("?")], "describe": [reply("", "length")], "judge2": [reply("?")]})
    record = asyncio.run(ec.check_pair(server, row, []))
    assert record["q2"]["verdict"] is None and record["dn2"]["verdict"] is None and not record["keep"]
    assert sum(call == "describe" for call, _ in server.calls) == ec.VERDICT_ATTEMPTS


def test_completed_records_drops_a_torn_last_line(tmp_path):
    path = tmp_path / "node0.jsonl"
    path.write_text(json.dumps({"id": "a"}) + "\n" + json.dumps({"id": "b"}) + "\n" + '{"id": "c", "ke')
    assert list(ec.completed_records(path)) == ["a", "b"]
    assert path.read_text().count("\n") == 2
    path.write_text('{"id": "a"}\n{"id": \n{"id": "b"}\n')
    with pytest.raises(ValueError):
        ec.completed_records(path)
    path.write_text('{"id": "a"}\n{"id": "b"}')
    assert list(ec.completed_records(path)) == ["a", "b"] and path.read_text().endswith("}\n")


def record(row, q2, dn2):
    return {"id": row["id"], "dataset": row["dataset"], "type": row["type"], "keep": q2 == dn2 == "pass",
            "q2": {"verdict": q2, "problem": "none" if q2 == "pass" else "partial", "reason": "", "attempts": 1, "tokens": 5},
            "dn2": {"verdict": dn2, "problem": "none" if dn2 == "pass" else "scope", "reason": "", "description": "d",
                    "describe_attempts": 1, "attempts": 1, "tokens": 7}, "seconds": 1.0}


def test_merge(tmp_path):
    rows = [write_pair(tmp_path, f"synthetic-{i:02d}") for i in range(6)]
    rows[5]["type"] = None
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    verdicts = [("pass", "pass"), ("fail", "pass"), ("pass", "fail"), ("fail", "fail"), (None, "pass"), ("pass", "pass")]
    shards = tmp_path / "run" / "shards"
    shards.mkdir(parents=True)
    for rank in range(2):
        path = shards / f"node{rank}.jsonl"
        path.write_text("".join(json.dumps(record(r, *verdicts[i])) + "\n" for i, r in enumerate(rows) if i % 2 == rank))
        (shards / f"complete-node{rank}.json").write_text(json.dumps({"sha256": dist.file_hash(path)}))
    run = {"nodes": 2, "gpus_per_node": 8, "sample": None, "sample_seed": 0, "run_id": "test_001", "git_commit": "0" * 40,
           "protocol_sha256": PROTOCOL_SHA256, "manifest_sha256": dist.file_hash(manifest), "model": "Qwen3.8-27B"}
    timing = {"check_hours_max": 0.1, "pairs_per_second_mean": 1.0, "nodes": []}
    output = tmp_path / "check"
    summary = dist.merge(manifest, shards, output, run, timing)
    assert (summary["input_count"], summary["kept_count"], summary["unchecked_count"]) == (6, 2, 1)
    assert summary["dropped_by"] == {"q2_only": 1, "dn2_only": 1, "both": 1, "unchecked": 1}
    assert summary["by_type"]["None"]["kept"] == 1
    kept = (output / "manifest_kept.jsonl").read_text().splitlines()
    original = manifest.read_text().splitlines()
    assert kept == [original[0], original[5]]
    lines = [json.loads(line) for line in (output / "verdicts.jsonl").read_text().splitlines()]
    assert [r["id"] for r in lines] == [r["id"] for r in rows] and lines[1]["instruction"] == "replace the box"
    assert "保留 **2 对**" in (output / "README.md").read_text()
    assert json.loads((output / "config.json").read_text())["protocol"]["sha256"] == PROTOCOL_SHA256
    # A shard edited after completion, or missing a pair, is rejected.
    with (shards / "node1.jsonl").open("a") as stream:
        stream.write(json.dumps(record(rows[0], "pass", "pass")) + "\n")
    with pytest.raises(ValueError):
        dist.merge(manifest, shards, tmp_path / "check2", run, timing)


def test_resume_identity_ignores_speed_flags():
    command = ["<model>", "--served-model-name", "qwen38-27b", "--max-model-len", "20000",
               "--gpu-memory-utilization", "0.92", "--max-num-seqs", "96", "--enable-prefix-caching", "--port", "8012"]
    assert dist.answer_flags(command) == ["<model>", "--served-model-name", "qwen38-27b", "--max-model-len", "20000",
                                          "--enable-prefix-caching"]
