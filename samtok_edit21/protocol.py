"""Data contract: native localization and inline region editing.

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
        if "<|" in label or "<think>" in label or "</think>" in label or any(ord(c) < 32 for c in label):
            raise ValueError("Control tokens/characters are not allowed in a label")
        rows.append({"mask_2d": span, "label": label.strip()})
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
        if not isinstance(item, dict) or set(item) != {"mask_2d", "label"}:
            raise ValueError("Mask items must have exactly mask_2d and label")
        pairs.append((item["mask_2d"], item["label"]))
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
        spans_in(result)
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
    groups = list(re.finditer(r"(?:" + SPAN_RE.pattern + r")+", prompt))
    if (edit_type == "composite" and len(groups) < 2) or (edit_type != "composite" and len(groups) != 1):
        raise ValueError("Mask group count must match atomic/composite edit semantics")
    if edit_type in {"background", "global"} and len(spans_in(prompt)) != 1:
        raise ValueError("Background/global require exactly one mask")
    for group in groups:
        before, after = prompt[:group.start()], prompt[group.end():]
        if not before.endswith(" ") or before.endswith("  "):
            raise ValueError("Mask group must follow a complete phrase and one space")
        if before.endswith("<|mt_end|> "):
            raise ValueError("Masks for one phrase must be directly concatenated")
        # A complete reference can end in a stranded preposition, e.g.
        # "the sofa that the cat is resting on". Its binding was checked by
        # exact source-span matching; a last-word heuristic cannot reject it.
        forbidden = (r"the|a|an" if variant == "ref" else
                     r"the|a|an|to|of|on|in|at|with|from|near|under|over|behind|beside")
        if re.search(r"(?:^|\s)(?:" + forbidden + r") $", before, re.I):
            raise ValueError("Mask group cannot directly follow an article/preposition")
        if after and (after.startswith("  ") or (after[0].isalnum()) or re.match(r"\s+[.,;:!?]", after)):
            raise ValueError("Invalid spacing after mask group")
        if variant == "noref":
            choices = list(NOREF.values()) + ["in this region"] if edit_type == "composite" else ["in this region" if edit_type == "add" else NOREF[edit_type]]
            # A grammatical preposition before the phrase is not its type.
            # E.g. attribute "Fill in this region" still ends in "this region".
            phrases = sorted(set(NOREF.values()) | {"in this region"}, key=len, reverse=True)
            matched = next((p for p in phrases if re.search(r"(?<!\w)" + re.escape(p) + r" $", before, re.I)), None)
            grammatical_in = matched == "in this region" and "this region" in choices
            if matched not in choices and not grammatical_in:
                raise ValueError("Noref mask must follow the type-specific region phrase")


def validate_row(row):
    if not isinstance(row, dict):
        raise ValueError("Each metadata row must be a JSON object")
    kind = row.get("sample_type")
    if kind not in {"edit", "edit_ntp", "edit_umt"}:
        raise ValueError("Use edit/edit_ntp/edit_umt; legacy edit_mt must be converted")
    allowed = {"sample_type", "edit_type", "edit_image", "prompt"}
    allowed |= {"mt_cot"} if kind == "edit_ntp" else {"image"}
    if kind == "edit_umt":
        allowed.add("instr_variant")
    if set(row) != allowed:
        raise ValueError(f"Unexpected/missing metadata fields: {set(row) ^ allowed}")
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
    if "<|" in SPAN_RE.sub("", row["prompt"]) or "<think>" in row["prompt"] or "</think>" in row["prompt"]:
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
        for _, label in pairs:
            phrase = re.sub(r"^one of the ", "", label, flags=re.I)
            if re.match(r"^(?:the|a|an)\s+", phrase, re.I):
                raise ValueError("Localization label must omit its leading article")
        units = grouped_units(row["prompt"], pairs)
        positions = [phrase_span(row["prompt"], u.ref_phrase)[0] for u in units if u.ref_phrase.lower() != "this image"]
        if positions != sorted(positions):
            raise ValueError("Localization units must follow instruction phrase order")
        if row["edit_type"] == "composite":
            if len(units) < 2:
                raise ValueError("Composite requires multiple distinct units")
        elif len(units) != 1:
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
            if not spans or row.get("instr_variant") not in {"ref", "noref"}:
                raise ValueError(
                    "edit_umt needs inline masks and ref/noref instr_variant"
                )
            validate_inline(row["prompt"], row["instr_variant"], row["edit_type"])
        elif spans or "instr_variant" in row:
            raise ValueError("Plain edit must omit masks/instr_variant")
    return row
