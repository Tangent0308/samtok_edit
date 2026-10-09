"""Coordinate handling of scripts/data_cleaning/clean_pairs.py on synthetic pairs."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("scipy")
mask_utils = pytest.importorskip("pycocotools.mask")
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "data_cleaning"))
import clean_pairs as cp  # noqa: E402

from samtok_edit21.data.protocol import pixel_box  # noqa: E402

W, H = 640, 512


def textured(seed):
    rng = np.random.default_rng(seed)
    base = cv2.GaussianBlur(rng.integers(0, 256, (H, W, 3), dtype=np.uint8), (0, 0), 3)
    base = cv2.normalize(base, None, 30, 225, cv2.NORM_MINMAX)
    for _ in range(60):  # corners for the feature matcher
        x, y = int(rng.integers(10, W - 40)), int(rng.integers(10, H - 40))
        color = tuple(int(v) for v in rng.integers(0, 256, 3))
        cv2.rectangle(base, (x, y), (x + int(rng.integers(8, 30)), y + int(rng.integers(8, 30))), color, -1)
    return base


def make_pair(tmp_path, scale, shift, box_target):
    """Source, and a target that views the same scene through x_t = scale * x_s + shift with one new object."""
    source = textured(0)
    T = np.float32([[scale, 0, shift[0]], [0, scale, shift[1]]])
    target = cv2.warpAffine(source, T, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    x0, y0, x1, y1 = box_target
    target[y0:y1, x0:x1] = (255, 0, 255)
    target[y0 + 4:y1 - 4, x0 + 4:x1 - 4] = (0, 255, 0)
    stored = np.zeros((H, W), np.uint8)   # grounded on the target, kept in the source frame by a plain resize
    stored[y0:y1, x0:x1] = 1
    truth = np.zeros((H, W), bool)        # where the object really is in the source frame
    sx0, sy0 = (x0 - shift[0]) / scale, (y0 - shift[1]) / scale
    sx1, sy1 = (x1 - shift[0]) / scale, (y1 - shift[1]) / scale
    truth[max(0, round(sy0)):max(0, round(sy1)), max(0, round(sx0)):max(0, round(sx1))] = True
    Image.fromarray(source).save(tmp_path / "source.png")
    Image.fromarray(target).save(tmp_path / "target.png")
    rle = mask_utils.encode(np.asfortranarray(stored))
    rec = {"id": "synthetic-00", "dataset": "synthetic", "provisional_type": "add", "instruction": "add a box",
           "edit_image": str(tmp_path / "source.png"), "image": str(tmp_path / "target.png"),
           "instances": [{"instance_id": "target_0", "ref": "box", "grounding_image": "target", "mapped_from_target": True,
                          "rle_size": [H, W], "rle_counts": rle["counts"].decode()}]}
    return rec, stored.astype(bool), truth


def run(tmp_path, rec):
    cp._init(str(tmp_path / "out"), 1)
    return cp.process(json.dumps(rec))


def test_box_matches_protocol():
    rng = np.random.default_rng(1)
    for _ in range(300):
        h, w = (int(v) for v in rng.integers(5, 400, 2))
        mask = np.zeros((h, w), bool)
        y, x = int(rng.integers(0, h)), int(rng.integers(0, w))
        mask[y:int(rng.integers(y, h)) + 1, x:int(rng.integers(x, w)) + 1] = True
        assert cp.pixel_box(mask) == list(pixel_box(mask))


def test_target_grounded_instance_moves_with_registration(tmp_path):
    rec, stored, truth = make_pair(tmp_path, 1.08, (-30.0, -18.0), (300, 220, 380, 300))
    res = run(tmp_path, rec)
    assert res["decision"] == "fix_register", res.get("reason")
    inst = res["instances"][0]
    moved = mask_utils.decode({"size": inst["rle_size"], "counts": inst["rle_counts"].encode()}).astype(bool)
    assert inst["moved"]
    iou = lambda a, b: (a & b).sum() / (a | b).sum()  # noqa: E731
    assert iou(moved, truth) > 0.9 > iou(stored, truth)
    assert inst["box_1000"] == list(pixel_box(moved)) == res["region"]["box_1000"]
    out = Path(tmp_path / "out")
    source = np.asarray(Image.open(out / res["source_image"]).convert("RGB"), dtype=np.int16)
    target = np.asarray(Image.open(out / res["target_image"]).convert("RGB"), dtype=np.int16)
    assert target.shape == source.shape and res["size"] == [W, H]
    changed = np.abs(target - source).mean(axis=2) > 25
    assert changed[truth].mean() > 0.9          # the object sits where the corrected region says
    far = ~cv2.dilate(truth.astype(np.uint8), np.ones((41, 41), np.uint8)).astype(bool)
    assert changed[far].mean() < 0.001          # and everything else is the source again


def test_registration_that_pushes_the_instance_out_of_frame_is_not_used(tmp_path):
    # the target shows more than the source frame; an object at its left edge lies mostly outside the source
    rec, _, _ = make_pair(tmp_path, 0.90, (32.0, 26.0), (0, 200, 52, 300))
    res = run(tmp_path, rec)
    assert res["register"]["instance_area_kept"] < cp.RULES["instance_area_min"]
    assert res["decision"].startswith("drop")
