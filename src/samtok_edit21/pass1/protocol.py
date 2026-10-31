"""Public-only Pass1 protocol. No segmentation model is used for point/box/text."""

from __future__ import annotations

import json
import math
import re

VERSION = "samtok_pass1_v2"
EMPTY_THINK = "<think>\n\n</think>\n\n"
SPAN = re.compile(r"<\|mt_start\|><\|mt_([0-9]{4})\|><\|mt_([0-9]{4})\|><\|mt_end\|>")
PROMPT = """Locate the source-image selections for all requested targets. Return one mask token sequence per target, in the input order, in one JSON array.
Read each complete instruction to identify the source object, part, or region. Locate what is selected in the source image, not the requested replacement or the outline of a new object. For an addition, locate its supplied source support or attachment selection.
Coordinates are integers normalized to 0-1000 on the source image; x increases to the right and y increases downward. A point is a positive click. Consider the intended local part, complete object, or coherent surface using the image and the instruction; the selected mask must contain the point. A box is [x_min, y_min, x_max, y_max] and indicates the target's extent. Follow its visible boundary, including visible thin parts, and exclude unrelated neighbors.
Use any source description in the instruction together with the locator. For a mask locator, copy its complete mask_2d sequence exactly to the same label. Labels are identifiers, not object categories. Different selections may overlap.
Return exactly the requested labels, each once. Each object must contain only label and mask_2d. Each mask_2d is one complete SAMTok mask span. Return no alternative candidates, explanations, or Markdown fences.
Request:
"""


def valid_span(value):
    m = SPAN.fullmatch(value) if isinstance(value, str) else None
    return bool(m and 0 <= int(m[1]) < 256 and 256 <= int(m[2]) < 512)


def point_normalized(point, size):
    if len(point) != 2 or any(type(x) is not int for x in point):
        raise ValueError("Point must be two source-pixel integers")
    if any(not 0 <= p < s for p, s in zip(point, size)):
        raise ValueError("Point outside source")
    return [min(1000, max(0, round(1000 * p / s))) for p, s in zip(point, size)]


def point_pixels(point, size):
    return [min(s - 1, max(0, round(p * s / 1000))) for p, s in zip(point, size)]


def box_normalized(box, size):
    w, h = size
    if len(box) != 4 or any(type(x) is not int for x in box):
        raise ValueError("Box must be four half-open source-pixel integers")
    a, b, c, d = box
    if not (0 <= a < c <= w and 0 <= b < d <= h):
        raise ValueError("Invalid half-open box")
    return [
        math.floor(1000 * a / w),
        math.floor(1000 * b / h),
        math.ceil(1000 * c / w),
        math.ceil(1000 * d / h),
    ]


def validate_request(request):
    if set(request) != {"protocol", "coordinate_system", "targets"}:
        raise ValueError("Unexpected public request fields")
    if (
        request["protocol"] != VERSION
        or request["coordinate_system"] != "normalized_0_1000"
    ):
        raise ValueError("Unknown protocol or coordinates")
    targets = request["targets"]
    if not isinstance(targets, list) or not 1 <= len(targets) <= 5:
        raise ValueError("Expected 1-5 targets")
    labels = []
    for t in targets:
        loc = t.get("locator")
        keys = {"label", "instruction", "has_ref", "locator"} | (
            {"region"} if loc != "none" else set()
        )
        if set(t) != keys or loc not in {"none", "point", "box", "mask"}:
            raise ValueError("Private or missing target field")
        if not isinstance(t["label"], str) or not re.fullmatch(r"U[1-5]", t["label"]):
            raise ValueError("Invalid label")
        if type(t["has_ref"]) is not bool or (loc == "none" and not t["has_ref"]):
            raise ValueError("No-ref requires a locator")
        if (
            not isinstance(t["instruction"], str)
            or not t["instruction"].strip()
            or "<|mt_" in t["instruction"]
        ):
            raise ValueError("Instruction is missing or leaks a token")
        labels.append(t["label"])
        if loc == "none":
            continue
        key = "mask_2d" if loc == "mask" else loc
        if not isinstance(t["region"], dict) or set(t["region"]) != {key}:
            raise ValueError("Only the supplied locator is public")
        v = t["region"][key]
        if loc == "mask":
            if not valid_span(v):
                raise ValueError("Invalid known mask")
        elif (
            not isinstance(v, list)
            or len(v) != (2 if loc == "point" else 4)
            or any(type(x) is not int or not 0 <= x <= 1000 for x in v)
        ):
            raise ValueError("Invalid normalized coordinates")
        elif loc == "box" and not (v[0] < v[2] and v[1] < v[3]):
            raise ValueError("Degenerate normalized box")
    if len(labels) != len(set(labels)):
        raise ValueError("Duplicate label")
    return request


def prompt(request):
    validate_request(request)
    return PROMPT + json.dumps(request, ensure_ascii=False, separators=(",", ":"))


def parse_answer(text, request, *, compatible=False):
    """No first-candidate choice, GT rematching, repair, or implicit known-mask replacement."""
    validate_request(request)
    raw = text.strip()
    if raw.endswith("<|im_end|>"):
        raw = raw[: -len("<|im_end|>")].rstrip()
    if compatible:
        if raw.startswith(EMPTY_THINK.strip()):
            raw = raw[len(EMPTY_THINK.strip()) :].strip()
        m = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.S)
        if m:
            raw = m[1]
    errors, items, spans = [], [], {}

    def unique_keys(pairs):
        result = {}
        for k, v in pairs:
            if k in result:
                raise ValueError("duplicate_json_key")
            result[k] = v
        return result

    try:
        items = json.loads(raw, object_pairs_hook=unique_keys)
        if not isinstance(items, list):
            raise ValueError("not_array")
    except (ValueError, TypeError):
        return {"complete": False, "errors": ["invalid_json"], "spans": {}, "items": []}
    expected = [t["label"] for t in request["targets"]]
    for uid in expected:
        hits = [x for x in items if isinstance(x, dict) and x.get("label") == uid]
        if len(hits) != 1:
            errors.append(f"{uid}:count={len(hits)}")
        elif set(hits[0]) != {"label", "mask_2d"} or not valid_span(
            hits[0].get("mask_2d")
        ):
            errors.append(f"{uid}:invalid_fields_or_span")
        else:
            spans[uid] = hits[0]["mask_2d"]
    labels = [x.get("label") if isinstance(x, dict) else None for x in items]
    if labels != expected:
        errors.append("label_order_or_extra")
    return {"complete": not errors, "errors": errors, "spans": spans, "items": items}


def finalize_spans(parsed, request):
    """Deployment policy, separate from raw model copy/format metrics."""
    if not parsed["complete"]:
        raise ValueError(f"Incomplete Pass1: {parsed['errors']}")
    spans = dict(parsed["spans"])
    for t in request["targets"]:
        if t["locator"] == "mask":
            spans[t["label"]] = t["region"]["mask_2d"]
    return spans


def from_native_job(job, image, codec=None):
    """Compile only public fields. A codec is required ONLY for supplied binary masks."""
    if (
        job["protocol"] != "native_regions_v2"
        or tuple(job["source_size"]) != image.size
    ):
        raise ValueError("Expected a native job at source dimensions")
    targets = []
    for u in job["units"]:
        t = {
            "label": u["id"],
            "instruction": u["instruction"],
            "has_ref": u["has_ref"],
            "locator": u["locator"],
        }
        if u["locator"] == "point":
            t["region"] = {"point": point_normalized(u["region"]["point"], image.size)}
        elif u["locator"] == "box":
            t["region"] = {"box": box_normalized(u["region"]["box"], image.size)}
        elif u["locator"] == "mask":
            if codec is None:
                raise ValueError("Supplied binary mask needs a codec")
            import numpy as np
            from PIL import Image

            with Image.open(u["region"]["mask"]) as im:
                if im.size != image.size:
                    raise ValueError("User mask dimensions differ")
                mask = np.asarray(im.convert("L")) > 0
            t["region"] = {"mask_2d": codec.encode(image.convert("RGBA"), [mask])[0][0]}
        targets.append(t)
    return validate_request(
        {
            "protocol": VERSION,
            "coordinate_system": "normalized_0_1000",
            "targets": targets,
        }
    )
