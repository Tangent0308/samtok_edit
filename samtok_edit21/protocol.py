"""Version 2 data contract: native localization and inline region editing.

Model inputs contain only the documented fields. Builders keep provenance in a
separate manifest: an image path is not a unique editing instruction identity.
"""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from dataclasses import dataclass

CODEBOOK_SIZE, CODEBOOK_DEPTH = 256, 2
SPAN_RE = re.compile(r"<\|mt_start\|><\|mt_(\d{4})\|><\|mt_(\d{4})\|><\|mt_end\|>")
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


def valid_span_codes(c0, c1):
    return 0 <= c0 < 256 and 256 <= c1 < 512


def is_valid_span(value):
    m = SPAN_RE.fullmatch(value) if isinstance(value, str) else None
    return bool(m and valid_span_codes(*map(int, m.groups())))


def span_of(codes):
    if len(codes) != 2 or not valid_span_codes(*codes):
        raise ValueError("SAMTok needs code0 in [0,255], offset code1 in [256,511]")
    return f"<|mt_start|><|mt_{codes[0]:04d}|><|mt_{codes[1]:04d}|><|mt_end|>"


def spans_in(text):
    matches = list(SPAN_RE.finditer(text))
    if any(not is_valid_span(m.group()) for m in matches) or "<|mt_" in SPAN_RE.sub(
        "", text
    ):
        raise ValueError("Malformed or out-of-codebook mask token sequence")
    return [m.group() for m in matches]


def to_cot(items):
    """Preserve exact labels (including quoted text), escaping with JSON itself."""
    rows = []
    for span, label in items:
        if not is_valid_span(span) or not isinstance(label, str) or not label.strip():
            raise ValueError("Each item needs a valid mask span and nonempty label")
        if "<|" in label or any(ord(c) < 32 for c in label):
            raise ValueError("Control tokens/characters are not allowed in a label")
        rows.append({"mask_2d": span, "label": label.strip()})
    return (
        "```json\n["
        + ",\n".join(json.dumps(x, ensure_ascii=False) for x in rows)
        + "]\n```"
    )


def parse_cot(text, *, nonempty=False):
    """Strict parse; never fabricate labels/codes or drop one composite unit."""
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
        if not isinstance(item, dict) or set(item) != {"mask_2d", "label"}:
            raise ValueError("Mask items must have exactly mask_2d and label")
        pairs.append((item["mask_2d"], item["label"]))
    to_cot(pairs)  # validate without silently fixing content
    return pairs


def parse_generated_cot(text):
    # The released Qwen3 SAMTok can emit a well-formed thinking preamble even
    # though localization supervision is canonical JSON only. It is never
    # diffusion conditioning. Reject arbitrary prose and truncated preambles.
    text = re.sub(r"^\s*<think>[\s\S]*?</think>\s*", "", text, count=1)
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
    for code, label in items:
        phrase = re.sub(r"^one of ", "", label, flags=re.I)
        if phrase != label:
            try:
                phrase_span(instruction, phrase)
            except ValueError:
                # Canonical plural labels add "one of the" even when the
                # original instruction has no article (e.g. "Remove cats").
                without_article = re.sub(r"^(?:the|a|an)\s+", "", phrase, flags=re.I)
                phrase_span(instruction, without_article)
                phrase = without_article
        groups.setdefault(phrase, []).append(code)
    return [Unit(phrase, tuple(codes)) for phrase, codes in groups.items()]


@dataclass(frozen=True)
class Unit:
    ref_phrase: str
    codes: tuple[str, ...]
    edit_type: str = "attribute"
    anchor_phrase: str | None = None


def _with_codes(phrase, codes):
    if not codes or any(not is_valid_span(s) for s in codes):
        raise ValueError("Unit needs one or more valid mask spans")
    return phrase + " " + "".join(codes)


def render_units(instruction, units, *, variant="ref"):
    """Resolve all spans against the original text before editing right-to-left.

    Ambiguous/overlapping phrases are rejected, not assigned to an arbitrary object.
    An online caller can then explicitly fall back to the original instruction.
    """
    if variant not in {"ref", "noref"} or not units:
        raise ValueError("Expected ref/noref and at least one unit")
    spans_in(instruction)
    if "<|" in instruction:
        raise ValueError(
            "Input instruction must not already contain control/mask tokens"
        )
    replacements = []
    for unit in units:
        phrase = unit.ref_phrase
        if phrase.lower() == "this image" or unit.edit_type == "global":
            found = []
            for ref in GLOBAL_REFS:
                try:
                    found.append(phrase_span(instruction, ref))
                except ValueError:
                    pass
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
        else:
            start, end = phrase_span(instruction, phrase)
            replacement = instruction[start:end]
            if variant == "noref":
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
                            r"\s+from\s+[^,.;!?]+(?=[.;!?]?$)", instruction[end:], re.I
                        )
                        if suffix and not re.search(r"\band\b", suffix.group(), re.I):
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
    if not code_groups or "<|" in instruction:
        raise ValueError("Supply clean instruction text and at least one region")
    pattern = r"(?<!\w)(?:" + "|".join(map(re.escape, REGION_REFS)) + r")(?!\w)"
    refs = list(re.finditer(pattern, instruction, re.I))
    if refs:
        if len(refs) != len(code_groups):
            raise ValueError(
                "The number of referring phrases must equal selected region groups"
            )
        result = instruction
        for ref, codes in reversed(list(zip(refs, code_groups))):
            result = result[: ref.end()] + " " + "".join(codes) + result[ref.end() :]
        spans_in(result)
        return result
    if len(code_groups) != 1:
        raise ValueError("Multiple regions need explicit referring phrases")
    region = "this image" if whole_image else "this region"
    tokens = _with_codes(region, code_groups[0])
    # Conservative grammar: do not inject an object in front of an existing object.
    if re.match(r"^(?:add|insert|place|put|draw)\b", instruction, re.I):
        return instruction.rstrip(".!? ") + " in " + tokens
    if re.match(r"^(?:apply)\b", instruction, re.I):
        return instruction.rstrip(".!? ") + " to " + tokens
    m = re.match(r"^(replace|swap)\s+(with\b.*)$", instruction, re.I)
    if m:
        return f"{m[1]} the object in {tokens} {m[2]}"
    m = re.match(r"^(turn|change)\s+(into\b.*|to\b.*)$", instruction, re.I)
    if m:
        return f"{m[1]} {tokens} {m[2]}"
    m = re.match(r"^(make|paint|color)\s+(.+)$", instruction, re.I)
    if m:
        return f"{m[1]} {tokens} {m[2]}"
    if re.fullmatch(r"remove|delete|erase|get rid of", instruction, re.I):
        return instruction + " the object in " + tokens
    raise ValueError(
        "Use an explicit 'this region' phrase for this interactive instruction"
    )


def validate_row(row):
    kind = row.get("sample_type")
    if kind not in {"edit", "edit_ntp", "edit_umt"}:
        raise ValueError("Use edit/edit_ntp/edit_umt; legacy edit_mt must be converted")
    if row.get("edit_type") not in EDIT_TYPES:
        raise ValueError("Missing or invalid edit_type")
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
            "Mask-conditioned rows require one source image; multi-image mask binding is unspecified"
        )
    spans = spans_in(row["prompt"])
    if "<|" in SPAN_RE.sub("", row["prompt"]):
        raise ValueError("Chat/vision control tokens may not occur in the instruction")
    if kind == "edit_ntp":
        if "image" in row or spans or "instr_variant" in row:
            raise ValueError(
                "edit_ntp cannot contain a target image, inline mask, or instr_variant"
            )
        pairs = parse_cot(row.get("mt_cot", ""), nonempty=True)
        if to_cot(pairs) != row["mt_cot"]:
            raise ValueError(
                "mt_cot must be canonical; canonicalize during data preparation"
            )
    else:
        if not isinstance(row.get("image"), str) or not row["image"] or "mt_cot" in row:
            raise ValueError("FM row needs target image and must omit mt_cot")
        if kind == "edit_umt":
            if not spans or row.get("instr_variant") not in {"ref", "noref"}:
                raise ValueError(
                    "edit_umt needs inline masks and ref/noref instr_variant"
                )
        elif spans or "instr_variant" in row:
            raise ValueError("Plain edit must omit masks/instr_variant")
    return row
