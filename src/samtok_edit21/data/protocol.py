"""Region-token text protocol.

A SAMTok mask token ⟨M⟩ is four atomic special tokens, ``<|mt_start|><|mt_XXXX|><|mt_YYYY|><|mt_end|>``: two codebook
levels of 256 codes each, the second offset by 256.  Qwen3-VL boxes ``<|box_start|>[x1, y1, x2, y2]<|box_end|>`` (0-1000
relative coordinates) are recognised too, so that a prompt is always validated as a whole.  ``token_layout`` finds the
region tokens of a prompt in the sequence the text encoder actually encoded.
"""
from __future__ import annotations

import re

import torch


CODEBOOK_SIZE, CODEBOOK_DEPTH = 256, 2
SPAN_RE = re.compile(r"<\|mt_start\|><\|mt_(\d{4})\|><\|mt_(\d{4})\|><\|mt_end\|>")
BOX_SCALE = 1000
BOX_RE = re.compile(r"<\|box_start\|>\[(\d{1,4}), (\d{1,4}), (\d{1,4}), (\d{1,4})\]<\|box_end\|>")
REGION_RE = re.compile(f"(?:{SPAN_RE.pattern})|(?:{BOX_RE.pattern})")
EMPTY_THINK = "<think>\n\n</think>\n\n"
GROUP_RE = re.compile(r"(?:" + REGION_RE.pattern + r")+")


def valid_span_codes(c0, c1):
    return 0 <= c0 < 256 and 256 <= c1 < 512


def is_valid_span(value):
    m = SPAN_RE.fullmatch(value) if isinstance(value, str) else None
    return bool(m and valid_span_codes(*map(int, m.groups())))


def span_of(codes):
    if len(codes) != 2 or not valid_span_codes(*codes):
        raise ValueError("SAMTok needs code0 in [0,255], offset code1 in [256,511]")
    return f"<|mt_start|><|mt_{codes[0]:04d}|><|mt_{codes[1]:04d}|><|mt_end|>"


def valid_box_coords(coords):
    return (len(coords) == 4 and all(type(v) is int for v in coords)
            and 0 <= coords[0] < coords[2] <= BOX_SCALE and 0 <= coords[1] < coords[3] <= BOX_SCALE)


def box_of(coords):
    coords = list(coords)
    if not valid_box_coords(coords):
        raise ValueError("A box needs integer x1<x2, y1<y2 in [0,1000]")
    return "<|box_start|>[{}, {}, {}, {}]<|box_end|>".format(*coords)


def box_coords(value):
    m = BOX_RE.fullmatch(value) if isinstance(value, str) else None
    if not m:
        raise ValueError("Malformed box token sequence")
    coords = [int(v) for v in m.groups()]
    if box_of(coords) != value:  # rejects leading zeros and invalid geometry
        raise ValueError("Box coordinates must be canonical integers in [0,1000]")
    return tuple(coords)


def is_valid_box(value):
    try:
        box_coords(value)
    except ValueError:
        return False
    return True


def spans_in(text):
    matches = list(SPAN_RE.finditer(text))
    if any(not is_valid_span(m.group()) for m in matches) or "<|mt_" in SPAN_RE.sub(
        "", text
    ):
        raise ValueError("Malformed or out-of-codebook mask token sequence")
    return [m.group() for m in matches]


def boxes_in(text):
    matches = list(BOX_RE.finditer(text))
    if any(not is_valid_box(m.group()) for m in matches) or "<|box_" in BOX_RE.sub("", text):
        raise ValueError("Malformed or noncanonical box token sequence")
    return [m.group() for m in matches]


def regions_in(text):
    """All mask spans and boxes in text order, after validating both kinds."""
    spans_in(text)
    boxes_in(text)
    return [m.group() for m in REGION_RE.finditer(text)]


def token_layout(tokenizer, input_ids, prompt):
    """Region units and the instruction range in the trimmed TE token sequence.

    ``input_ids`` is ``prompt_input_ids`` from the official prompt embedder
    (system tokens dropped, -1 padding). Every unit is checked against the
    region groups of ``prompt`` so a tokenization surprise cannot bind the
    wrong tokens.
    """
    ids = torch.as_tensor(input_ids)
    if ids.ndim == 2:
        if ids.shape[0] != 1:
            raise ValueError("Binding supports batch one")
        ids = ids[0]
    ids = ids[ids >= 0].tolist()
    convert = tokenizer.convert_tokens_to_ids
    mt_start, mt_end = convert("<|mt_start|>"), convert("<|mt_end|>")
    box_start, box_end = convert("<|box_start|>"), convert("<|box_end|>")
    regions, index = [], 0
    while index < len(ids):
        token = ids[index]
        if token == mt_start:
            if index + 3 >= len(ids) or ids[index + 3] != mt_end:
                raise ValueError("Truncated mask span in the encoded prompt")
            regions.append((index, index + 4))
            index += 4
        elif token == box_start:
            try:
                end = ids.index(box_end, index)
            except ValueError as exc:
                raise ValueError("Unterminated box in the encoded prompt") from exc
            regions.append((index, end + 1))
            index = end + 1
        elif token in (mt_end, box_end):
            raise ValueError("Unpaired region end token in the encoded prompt")
        else:
            index += 1
    units = []
    for start, end in regions:
        if units and units[-1][1] == start:
            units[-1][1] = end
        else:
            units.append([start, end])
    groups = [m.group() for m in GROUP_RE.finditer(prompt)]
    if len(groups) != len(units) or any(
            tokenizer.decode(ids[s:e], skip_special_tokens=False) != g for (s, e), g in zip(units, groups)):
        raise ValueError("Encoded region tokens differ from the prompt's region groups")
    vision_end = [i for i, token in enumerate(ids) if token == convert("<|vision_end|>")]
    if len(vision_end) != 1 or (units and units[0][0] <= vision_end[0]):
        raise ValueError("Binding requires one source image before the instruction")
    return {"units": [tuple(unit) for unit in units], "instruction": (vision_end[0] + 1, len(ids)),
            "length": len(ids)}
