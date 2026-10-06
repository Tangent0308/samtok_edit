"""Data contract: native localization and inline region editing.

Model inputs contain only the documented fields. Builders keep provenance in a
separate manifest: an image path is not a unique editing instruction identity.

A region token is either a SAMTok mask span ⟨M⟩ or a Qwen3-VL box ⟨B⟩.  Add
places a new object, so its region is a box in 0-1000 relative source-image
coordinates; every other atomic type keeps the SAMTok mask of an existing
region.  ``REGION_KIND`` is the single source of that rule.
"""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from dataclasses import dataclass

CODEBOOK_SIZE, CODEBOOK_DEPTH = 256, 2
SPAN_RE = re.compile(r"<\|mt_start\|><\|mt_(\d{4})\|><\|mt_(\d{4})\|><\|mt_end\|>")
BOX_SCALE = 1000
BOX_RE = re.compile(r"<\|box_start\|>\[(\d{1,4}), (\d{1,4}), (\d{1,4}), (\d{1,4})\]<\|box_end\|>")
REGION_RE = re.compile(f"(?:{SPAN_RE.pattern})|(?:{BOX_RE.pattern})")
EDIT_TYPES = (
    "add",
    "remove",
    "replace",
    "attribute",
    "action",
    "text",
    "background",
    "global",
    "composite",
)
LOC_REQUEST = "Please identify and segment the region to be edited in this image."
# Box-grounding replay (rec_ntp) uses Qwen3-VL's native request so that the
# edit request above keeps one deterministic output format per edit type.
REC_TEMPLATE = "Locate the {} in this image and output its bbox coordinates in JSON format."
EMPTY_THINK = "<think>\n\n</think>\n\n"
GLOBAL_REFS = (
    "this image",
    "the entire image",
    "the whole image",
    "the image",
    "the photo",
    "the picture",
    "the scene",
)
REGION_REFS = (
    "the selected region",
    "the selected object",
    "the selected area",
    "this region",
    "this object",
    "this image",
    "this area",
    "here",
    "this",
    "it",
)
NOREF = {
    "remove": "the object in this region",
    "replace": "the object in this region",
    "action": "the object in this region",
    "text": "the text in this region",
    "attribute": "this region",
    "background": "this region",
    "global": "this image",
}
TYPE_WEIGHTS = dict(zip(EDIT_TYPES, (14, 14, 14, 20, 10, 10, 6, 6, 6)))
REGION_KIND = {t: "box" if t == "add" else "mask" for t in EDIT_TYPES if t != "composite"}


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


def is_valid_region(value):
    return is_valid_span(value) or is_valid_box(value)


def region_kind(value):
    if is_valid_span(value):
        return "mask"
    if is_valid_box(value):
        return "box"
    raise ValueError("Not a valid mask span or box")


def pixel_box(mask):
    """Outward-rounded 0-1000 box of a nonempty HxW mask in its own frame."""
    import numpy as np

    mask = np.asarray(mask) > 0
    if mask.ndim != 2 or not mask.any():
        raise ValueError("A box needs one nonempty 2D mask")
    height, width = mask.shape
    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    return relative_box((cols[0], rows[0], cols[-1] + 1, rows[-1] + 1), width, height)


def relative_box(xyxy, width, height):
    """Pixel xyxy (exclusive end) -> outward-rounded 0-1000 box.

    floor/ceil of a nonempty interval inside [0, size] stays inside [0, 1000]
    and keeps x1 < x2, y1 < y2, so no clipping or repair is needed.
    """
    import math

    x0, y0, x1, y1 = (float(v) for v in xyxy)
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("Pixel box must lie inside the image")
    box = (math.floor(BOX_SCALE * x0 / width), math.floor(BOX_SCALE * y0 / height),
           math.ceil(BOX_SCALE * x1 / width), math.ceil(BOX_SCALE * y1 / height))
    box_of(box)
    return box


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


def to_cot(items):
    """Preserve exact labels (including quoted text), escaping with JSON itself."""
    rows = []
    for region, label in items:
        if not is_valid_region(region) or not isinstance(label, str) or not label.strip():
            raise ValueError("Each item needs a valid mask span or box and a nonempty label")
        if "<|" in label or "<think>" in label or "</think>" in label or any(ord(c) < 32 for c in label):
            raise ValueError("Control tokens/characters are not allowed in a label")
        if region_kind(region) == "mask":
            rows.append({"mask_2d": region, "label": label.strip()})
        else:
            rows.append({"bbox_2d": list(box_coords(region)), "label": label.strip()})
    return (
        "```json\n["
        + ",\n".join(json.dumps(x, ensure_ascii=False) for x in rows)
        + "]\n```"
    )


def parse_cot(text, *, nonempty=False):
    """Strict parse; never fabricate labels/codes or drop one composite unit."""
    if not isinstance(text, str):
        raise ValueError("Localization JSON must be stored as text")
    text = text.strip()
    if text.endswith("<|im_end|>"):
        text = text[: -len("<|im_end|>")].rstrip()
    fenced = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.I)
    rows = json.loads(fenced.group(1) if fenced else text)
    if not isinstance(rows, list) or (nonempty and not rows):
        raise ValueError(
            "Expected a nonempty mask JSON list"
            if nonempty
            else "Expected a mask JSON list"
        )
    pairs = []
    for item in rows:
        if isinstance(item, dict) and set(item) == {"mask_2d", "label"}:
            pairs.append((item["mask_2d"], item["label"]))
        elif isinstance(item, dict) and set(item) == {"bbox_2d", "label"}:
            coords = item["bbox_2d"]
            if not isinstance(coords, list) or not valid_box_coords(coords):
                raise ValueError("bbox_2d must be four integers x1<x2, y1<y2 in [0,1000]")
            pairs.append((box_of(coords), item["label"]))
        else:
            raise ValueError("Items must have exactly mask_2d/bbox_2d and label")
    to_cot(pairs)  # validate without silently fixing content
    return pairs


def parse_generated_cot(text):
    # The released Qwen3 SAMTok can emit a well-formed thinking preamble even
    # though localization supervision is canonical JSON only. It is never
    # diffusion conditioning. Reject arbitrary prose and truncated preambles.
    if not isinstance(text, str):
        raise ValueError("Generated localization must be text")
    text = re.sub(r"^\s*<think>\s*</think>\s*", "", text, count=1)
    if text.removesuffix("<|im_end|>").strip() == "No target.":
        raise ValueError("Localization returned No target.")
    return parse_cot(text, nonempty=True)


def phrase_span(text, phrase):
    matches = list(re.finditer(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text, re.I))
    if len(matches) != 1:
        raise ValueError(
            f"Reference must match exactly once: {phrase!r} ({len(matches)} matches)"
        )
    return matches[0].span()


def grouped_units(instruction, items):
    groups = OrderedDict()
    plural_groups = {}
    previous = None
    for code, label in items:
        plural = bool(re.match(r"^one of the ", label, re.I))
        phrase = re.sub(r"^one of the ", "", label, flags=re.I)
        if phrase != label:
            try:
                phrase_span(instruction, phrase)
            except ValueError:
                # Canonical plural labels add "one of the" even when the
                # original instruction has no article (e.g. "Remove cats").
                without_article = re.sub(r"^(?:the|a|an)\s+", "", phrase, flags=re.I)
                phrase_span(instruction, without_article)
                phrase = without_article
        phrase = next((p for p in groups if p.lower() == phrase.lower()), phrase)
        if phrase in groups and phrase != previous:
            raise ValueError("Masks for one phrase must be contiguous in the JSON list")
        if phrase in groups and (not plural or not plural_groups[phrase]):
            raise ValueError("Repeated label requires explicit 'one of' multi-instance semantics")
        if phrase.lower() != "this image":
            phrase_span(instruction, phrase)
        groups.setdefault(phrase, []).append(code)
        plural_groups[phrase] = plural
        previous = phrase
    for codes in groups.values():
        if len({region_kind(code) for code in codes}) != 1:
            raise ValueError("One unit cannot mix mask spans and boxes")
    return [Unit(phrase, tuple(codes)) for phrase, codes in groups.items()]


@dataclass(frozen=True)
class Unit:
    ref_phrase: str
    codes: tuple[str, ...]
    edit_type: str = "attribute"
    anchor_phrase: str | None = None


def bind_edit_units(instruction, items, reviewed=None):
    """Noref needs reviewed semantics; localization JSON has no edit types."""
    units = grouped_units(instruction, items)
    if reviewed is not None:
        if len(reviewed) != len(units):
            raise ValueError("Reviewed units must match localization groups in order")
        result = []
        for unit, spec in zip(units, reviewed):
            if spec["ref_phrase"] != unit.ref_phrase:
                raise ValueError("Reviewed ref_phrase must exactly match bound localization label")
            typ = spec["edit_type"]
            if typ not in NOREF and typ != "add":
                raise ValueError("Reviewed edit_type must be atomic")
            result.append(Unit(unit.ref_phrase, unit.codes, typ, spec.get("anchor_phrase")))
        return result
    raise ValueError("Noref ablation requires reviewed edit_type for every unit")


def condition_localization(instruction, items, *, variant="ref", reviewed=None, strict=False):
    if variant not in {"ref", "noref"}:
        raise ValueError("Expected requested variant ref/noref")
    if strict and variant != "noref":
        raise ValueError("Strict noref requires variant=noref")
    groups = grouped_units(instruction, items)
    ref = render_units(instruction, groups, variant="ref")
    if variant == "ref":
        return {"conditioning_prompt": ref, "requested_variant": variant,
                "actual_variant": "ref", "fallback_reason": None}
    try:
        units = bind_edit_units(instruction, items, reviewed)
        prompt = render_units(instruction, units, variant="noref")
        return {"conditioning_prompt": prompt, "requested_variant": variant,
                "actual_variant": "noref", "fallback_reason": None}
    except (ValueError, KeyError, TypeError) as exc:
        if strict:
            raise ValueError(f"Strict noref unsupported: {exc}") from exc
        return {"conditioning_prompt": ref, "requested_variant": variant,
                "actual_variant": "ref", "fallback_reason": str(exc)}


def _with_codes(phrase, codes):
    if not codes or any(not is_valid_region(s) for s in codes):
        raise ValueError("Unit needs one or more valid mask spans or boxes")
    if len({region_kind(s) for s in codes}) != 1:
        raise ValueError("One unit cannot mix mask spans and boxes")
    return phrase + " " + "".join(codes)


def render_units(instruction, units, *, variant="ref"):
    """Resolve all spans against the original text before editing right-to-left.

    Ambiguous/overlapping phrases are rejected, not assigned to an arbitrary object.
    An online caller can then explicitly fall back to the original instruction.
    """
    if variant not in {"ref", "noref"} or not units:
        raise ValueError("Expected ref/noref and at least one unit")
    regions_in(instruction)
    if "<|" in instruction or "<think>" in instruction or "</think>" in instruction:
        raise ValueError(
            "Input instruction must not already contain control/mask tokens"
        )
    replacements = []
    for unit in units:
        phrase = unit.ref_phrase
        if phrase.lower() == "this image" or unit.edit_type == "global":
            pattern = r"(?<!\w)(?:" + "|".join(map(re.escape, GLOBAL_REFS)) + r")(?!\w)"
            found = [m.span() for m in re.finditer(pattern, instruction, re.I)]
            if len(found) > 1:
                raise ValueError("Ambiguous whole-image references")
            if found:
                start, end = min(found, key=lambda p: (p[0], -p[1]))
                replacement = (
                    instruction[start:end] if variant == "ref" else "this image"
                )
            else:
                if len(units) != 1:
                    raise ValueError("Unanchored global unit in composite instruction")
                end = len(instruction.rstrip(".!? "))
                start, replacement = end, " to this image"
                if re.fullmatch(r"(?:colorize|restore|enhance|sharpen|unblur)", instruction[:end], re.I):
                    replacement = " this image"
        else:
            start, end = phrase_span(instruction, phrase)
            replacement = instruction[start:end]
            if variant == "noref":
                if unit.edit_type == "background" and re.match(r"(?:extract|cut out)\b", instruction, re.I):
                    raise ValueError("Extraction background needs a reviewed noref rewrite")
                if unit.edit_type == "add":
                    if unit.anchor_phrase:
                        a, b = phrase_span(replacement, unit.anchor_phrase)
                        # Require anchor at the end so token follows the actual region phrase.
                        if b != len(replacement):
                            raise ValueError(
                                "Nonterminal add anchor requires an explicit reviewed rewrite"
                            )
                        replacement = replacement[:a] + "in this region"
                    else:
                        replacement += " in this region"
                else:
                    replacement = NOREF[unit.edit_type]
                    # Include an article immediately before the annotated noun phrase.
                    article = re.search(r"\b(?:the|a|an) $", instruction[:start], re.I)
                    if article:
                        start = article.start()
                    if unit.edit_type == "text":
                        text_prefix = re.search(
                            r"\b(?:the )?text\s+$", instruction[:start], re.I
                        )
                        if text_prefix:
                            start = text_prefix.start()
                    if unit.edit_type == "remove":
                        suffix = re.match(
                            r"\s+from\s+(?:this|the)\s+(?:image|photo|picture|scene)(?=[.;!?]?$)",
                            instruction[end:], re.I
                        )
                        # Do not delete arbitrary 'from ...' tails: they may
                        # contain what/how, not just a redundant image reference.
                        if suffix:
                            end += suffix.end()
        replacements.append((start, end, _with_codes(replacement, unit.codes)))
    replacements.sort()
    if any(b[0] < a[1] or b[0] == a[0] for a, b in zip(replacements, replacements[1:])):
        raise ValueError("Overlapping references require explicit disambiguated spans")
    result = instruction
    for start, end, value in reversed(replacements):
        result = result[:start] + value + result[end:]
    return result


def interactive_prompt(instruction, code_groups, *, whole_image=False):
    """Bind selected regions to explicit deictic phrases, in text order."""
    instruction = instruction.strip()
    if not code_groups or not instruction or "<|" in instruction or "<think>" in instruction or "</think>" in instruction:
        raise ValueError("Supply clean instruction text and at least one region")
    for codes in code_groups:
        _with_codes("", codes)
    pattern = r"(?<!\w)(?:" + "|".join(map(re.escape, REGION_REFS)) + r")(?!\w)"
    # Words being edited (e.g. Replace 'it' with 'go') are literal text,
    # not references to a selected region. Leave them to the text-edit branch.
    quoted_ranges = [m.span() for m in re.finditer(
        r'''(?<!\w)(?:"[^"\n]*"|'[^'\n]*'|“[^”\n]*”|‘[^’\n]*’)''', instruction
    )]
    refs = [m for m in re.finditer(pattern, instruction, re.I)
            if not any(start <= m.start() < end for start, end in quoted_ranges)]
    if refs:
        if len(refs) != len(code_groups):
            raise ValueError(
                "The number of referring phrases must equal selected region groups"
            )
        result = instruction
        for ref, codes in reversed(list(zip(refs, code_groups))):
            phrase = "this image" if whole_image and ref.group().lower() == "this region" else ref.group()
            result = result[:ref.start()] + _with_codes(phrase, codes) + result[ref.end():]
        regions_in(result)
        return result
    if len(code_groups) != 1:
        raise ValueError("Multiple regions need explicit referring phrases")
    region = "this image" if whole_image else "this region"
    tokens = _with_codes(region, code_groups[0])
    punctuation = instruction[len(instruction.rstrip(".!? ")):]
    body = instruction.rstrip(".!? ")
    text = re.match(r"^(\w+)\s+(?:the\s+)?text\b\s*(.*)$", body, re.I)
    if text:
        return f"{text[1]} the text in {tokens}" + (" " + text[2] if text[2] else "") + punctuation
    quoted = re.match(r"^(change|replace|swap|paint|color|write)\s+(.+)$", body, re.I)
    if quoted and re.search(r"[\"'“‘]", quoted[2]):
        # If a source string is supplied, replace it rather than duplicate it.
        tail = re.sub(r"^([\"'“‘]).*?[\"'”’]\s+(?=to\b|with\b|for\b)", "", quoted[2])
        return f"{quoted[1]} the text in {tokens} {tail}" + punctuation
    if re.match(r"^(?:add|insert|place|put|draw)\b", instruction, re.I):
        return body + " in " + tokens + punctuation
    if re.match(r"^(?:apply)\b", instruction, re.I):
        return body + " to " + tokens + punctuation
    m = re.match(r"^(remove|delete|erase|get rid of|replace|swap)\b\s*(.*)$", instruction, re.I)
    if m:
        return f"{m[1]} the object in {tokens}" + (" " + m[2] if m[2] else "")
    m = re.match(r"^(make|turn|change|paint|color|transform|restore|enhance|sharpen|unblur)\b\s*(.*)$", instruction, re.I)
    if m:
        return f"{m[1]} {tokens}" + (" " + m[2] if m[2] else "")
    return tokens + " " + instruction


def validate_inline(prompt, variant, edit_type):
    """Check observable binding syntax; semantic/mask QC belongs to annotation."""
    groups = list(re.finditer(r"(?:" + REGION_RE.pattern + r")+", prompt))
    if (edit_type == "composite" and len(groups) < 2) or (edit_type != "composite" and len(groups) != 1):
        raise ValueError("Region group count must match atomic/composite edit semantics")
    if edit_type in {"background", "global"} and len(regions_in(prompt)) != 1:
        raise ValueError("Background/global require exactly one mask")
    for group in groups:
        kinds = {region_kind(m.group()) for m in REGION_RE.finditer(group.group())}
        if len(kinds) != 1:
            raise ValueError("One region group cannot mix mask spans and boxes")
        if edit_type != "composite" and kinds != {REGION_KIND[edit_type]}:
            raise ValueError(f"{edit_type} binds a {REGION_KIND[edit_type]} region")
        before, after = prompt[:group.start()], prompt[group.end():]
        if not before.endswith(" ") or before.endswith("  "):
            raise ValueError("Region group must follow a complete phrase and one space")
        if before.endswith("<|mt_end|> ") or before.endswith("<|box_end|> "):
            raise ValueError("Regions for one phrase must be directly concatenated")
        # A complete reference can end in a stranded preposition, e.g.
        # "the sofa that the cat is resting on". Its binding was checked by
        # exact source-span matching; a last-word heuristic cannot reject it.
        forbidden = (r"the|a|an" if variant == "ref" else
                     r"the|a|an|to|of|on|in|at|with|from|near|under|over|behind|beside")
        named_a = variant == 'ref' and re.search(r'\b(?:[Mm]odel|labeled|labelled|label) A $', before)
        if re.search(r"(?:^|\s)(?:" + forbidden + r") $", before, re.I) and not named_a:
            raise ValueError("Region group cannot directly follow an article/preposition")
        if after and (after.startswith("  ") or (after[0].isalnum()) or re.match(r"\s+[.,;:!?]", after)):
            raise ValueError("Invalid spacing after region group")
        if variant == "noref":
            choices = list(NOREF.values()) + ["in this region"] if edit_type == "composite" else ["in this region" if edit_type == "add" else NOREF[edit_type]]
            # A grammatical preposition before the phrase is not its type.
            # E.g. attribute "Fill in this region" still ends in "this region".
            phrases = sorted(set(NOREF.values()) | {"in this region"}, key=len, reverse=True)
            matched = next((p for p in phrases if re.search(r"(?<!\w)" + re.escape(p) + r" $", before, re.I)), None)
            grammatical_in = matched == "in this region" and "this region" in choices
            if matched not in choices and not grammatical_in:
                raise ValueError("Noref region must follow the type-specific region phrase")


def _validate_rec(row, regions):
    """Box-grounding replay: one referring phrase, native Qwen3-VL request."""
    if regions:
        raise ValueError("rec_ntp prompts are plain grounding requests")
    pairs = parse_cot(row["mt_cot"], nonempty=True)
    if to_cot(pairs) != row["mt_cot"]:
        raise ValueError("mt_cot must be canonical; canonicalize during data preparation")
    labels = {label for _, label in pairs}
    if len(labels) != 1 or any(region_kind(region) != "box" for region, _ in pairs):
        raise ValueError("rec_ntp answers one phrase with bbox_2d items")
    if row["prompt"] != REC_TEMPLATE.format(labels.pop()):
        raise ValueError("rec_ntp prompt must be the canonical grounding request")


def validate_row(row):
    """v2 metadata contract: atomic edits only, add bound by a box."""
    if not isinstance(row, dict):
        raise ValueError("Each metadata row must be a JSON object")
    kind = row.get("sample_type")
    if kind not in {"edit", "edit_ntp", "edit_umt", "rec_ntp"}:
        raise ValueError("Use edit/edit_ntp/edit_umt/rec_ntp; legacy edit_mt must be converted")
    allowed = {"sample_type", "edit_type", "edit_image", "prompt"}
    allowed |= {"mt_cot"} if kind in {"edit_ntp", "rec_ntp"} else {"image"}
    if kind == "edit_umt":
        allowed.add("instr_variant")
    if set(row) != allowed:
        raise ValueError(f"Unexpected/missing metadata fields: {set(row) ^ allowed}")
    if row.get("edit_type") not in EDIT_TYPES:
        raise ValueError("Missing or invalid edit_type")
    if row["edit_type"] == "composite":
        raise ValueError("v2 metadata excludes composite edits")
    if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
        raise ValueError("Missing prompt")
    sources = row.get("edit_image")
    sources = [sources] if isinstance(sources, str) else sources
    if (
        not isinstance(sources, list)
        or not sources
        or any(not isinstance(x, str) or not x for x in sources)
    ):
        raise ValueError("edit_image must be a path or nonempty list of paths")
    if kind != "edit" and len(sources) != 1:
        raise ValueError(
            "Region-conditioned rows require one source image; multi-image binding is unspecified"
        )
    regions = regions_in(row["prompt"])
    if "<|" in REGION_RE.sub("", row["prompt"]) or "<think>" in row["prompt"] or "</think>" in row["prompt"]:
        raise ValueError("Chat/vision control tokens may not occur in the instruction")
    if kind == "rec_ntp":
        _validate_rec(row, regions)
    elif kind == "edit_ntp":
        if regions:
            raise ValueError("edit_ntp cannot contain an inline region")
        pairs = parse_cot(row.get("mt_cot", ""), nonempty=True)
        if to_cot(pairs) != row["mt_cot"]:
            raise ValueError(
                "mt_cot must be canonical; canonicalize during data preparation"
            )
        expected = REGION_KIND[row["edit_type"]]
        if any(region_kind(region) != expected for region, _ in pairs):
            raise ValueError(f"{row['edit_type']} localization must output {expected} regions")
        for _, label in pairs:
            phrase = re.sub(r"^one of the ", "", label, flags=re.I)
            if re.match(r"^(?:the|a|an)\s+", phrase, re.I):
                raise ValueError("Localization label must omit its leading article")
        units = grouped_units(row["prompt"], pairs)
        if len(units) != 1:
            raise ValueError("Atomic edit requires one label group")
        if row["edit_type"] in {"background", "global"} and len(pairs) != 1:
            raise ValueError("Background/global require one mask")
        if row["edit_type"] == "global" and pairs[0][1] != "this image":
            raise ValueError("Global label must be this image")
        render_units(row["prompt"], units)
    else:
        if not isinstance(row.get("image"), str) or not row["image"] or "mt_cot" in row:
            raise ValueError("FM row needs target image and must omit mt_cot")
        if kind == "edit_umt":
            if not regions or row.get("instr_variant") not in {"ref", "noref"}:
                raise ValueError(
                    "edit_umt needs inline regions and ref/noref instr_variant"
                )
            validate_inline(row["prompt"], row["instr_variant"], row["edit_type"])
        elif regions or "instr_variant" in row:
            raise ValueError("Plain edit must omit regions/instr_variant")
    return row
