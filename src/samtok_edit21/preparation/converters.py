"""Small real-data integration set and dataset-independent units converter."""

from __future__ import annotations

import io
import json
import re
from collections import Counter
from pathlib import Path

import ijson
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from samtok_edit21.data.io import write_json, write_rows, row_hash, file_hash
from samtok_edit21.data.protocol import (
    Unit,
    EDIT_TYPES,
    phrase_span,
    is_valid_span,
    parse_cot,
    grouped_units,
    render_units,
    to_cot,
    validate_row,
    parse_generated_cot,
    SPAN_RE,
    NOREF,
    spans_in,
)

TYPE_MAP = {
    "color": "attribute",
    "motion": "action",
    "motion change": "action",
    "background": "background",
    "background change": "background",
    "style": "global",
    "add": "add",
    "remove": "remove",
    "replace": "replace",
    "object_addition": "add",
    "object_removal": "remove",
    "object_replacement": "replace",
    "color_change": "attribute",
    "material_change": "attribute",
    "visual_beautification": "attribute",
    "size_change": "action",
    "action_editing": "action",
    "background_replacement": "background",
    "part_extraction": "background",
    "style_transfer": "global",
    "tone_adjustment": "global",
    "viewpoint_transformation": "global",
}


def native_edit_type(record):
    """Map known dataset labels, never guess RefEdit modify/reasoning semantics."""
    label = record.get("final_task", record.get("raw_type", record.get("type")))
    if not isinstance(label, str):
        return None
    label = re.sub(r"^\d+(?:\.\d+)*[_ -]", "", label.lower())
    if label.endswith("_text_editing") or label == "text_editing":
        return "text"
    return TYPE_MAP.get(label)


def canonical_reference(phrase, edit_type):
    if edit_type == "global":
        return "this image"
    if not isinstance(phrase, str) or not phrase.strip():
        raise ValueError("Missing unit reference")
    phrase = re.sub(r"^(?:the|a|an)\s+", "", phrase.strip(), flags=re.I)
    if edit_type == "text":
        # An already quoted reference can itself contain an apostrophe or inch
        # mark, e.g. '5\'6"'. Preserve its complete outer-delimited spelling.
        pairs = {'"': '"', "'": "'", '“': '”', '‘': '’'}
        if len(phrase) > 1 and phrase[0] in pairs and phrase[-1] == pairs[phrase[0]]:
            # Internal apostrophes/inch marks are literal; a closing quote
            # followed by whitespace denotes multiple separately quoted spans.
            if not re.search(re.escape(pairs[phrase[0]]) + r'\s', phrase[1:-1]):
                return phrase
        quoted = re.findall(r'''"[^"\n]+"|'[^'\n]+'|“[^”\n]+”|‘[^’\n]+’''', phrase)
        if len(quoted) > 1:
            raise ValueError("Text unit must identify one original quoted string")
        if quoted:
            phrase = quoted[0]
    return phrase


def validate_mask_geometry(mask, edit_type, stable_foreground=None):
    """Raw annotation QC, before lossy codec encoding; coordinates are source pixels."""
    binary = np.asarray(mask) > 0
    if binary.ndim != 2 or not binary.size:
        raise ValueError("Mask must be a nonempty 2D array")
    area = float(binary.mean())
    if edit_type == "global":
        if not binary.all():
            raise ValueError("Global mask must cover the entire source")
    elif edit_type == "background":
        if not 0.20 <= area <= 0.97:
            raise ValueError("Background mask area must be in [20%,97%]")
        if stable_foreground is None:
            raise ValueError("Background QC requires the dilated stable foreground")
        foreground = np.asarray(stable_foreground) > 0
        if foreground.shape != binary.shape or not np.array_equal(binary, ~foreground):
            raise ValueError("Background must equal the complement of the dilated stable foreground")
    elif edit_type in set(EDIT_TYPES) - {"composite"}:
        if not 0.0005 <= area <= 0.60:
            raise ValueError("Local mask area must be in [0.05%,60%]")
    else:
        raise ValueError("Mask QC requires an atomic edit type")
    return area


def annotation_codes(spec, source_size=None):
    """Validate optional raw masks and sort paired codes by bounding-box centers.

    Token-only imports rely on upstream annotation QC; code text cannot prove
    area, geometry, or semantic target correctness.
    """
    codes = tuple(spec["mask_codes"])
    if "mask_paths" not in spec:
        return codes
    if len(spec["mask_paths"]) != len(codes):
        raise ValueError("mask_paths must pair one-to-one with mask_codes")
    foreground = None
    if spec.get("stable_foreground_path"):
        with Image.open(spec["stable_foreground_path"]) as im:
            foreground = np.asarray(im.convert("L")) > 0
    centers = []
    for path in spec["mask_paths"]:
        with Image.open(path) as im:
            if source_size is not None and im.size != source_size:
                raise ValueError("Raw masks must use source image coordinates")
            binary = np.asarray(im.convert("L")) > 0
        validate_mask_geometry(binary, spec["edit_type"], foreground)
        y, x = np.nonzero(binary)
        centers.append(((int(x.min()) + int(x.max())) / 2, (int(y.min()) + int(y.max())) / 2))
    return tuple(codes[i] for i in sorted(range(len(codes)), key=lambda i: centers[i]))


def reviewed_noref(record, units):
    """Compile an upstream Qwen3/reviewer rewrite with explicit unit placeholders.

    noref_instruction contains {mask_0}, {mask_1}, ... after the corresponding
    region phrase. Placeholders refer to annotation storage order, not text order.
    No free-form generated code is trusted or converted into mask tokens.
    """
    prompt = record["noref_instruction"]
    if "<|" in prompt or "<think>" in prompt or "</think>" in prompt:
        raise ValueError("Reviewed rewrite must contain placeholders, not control tokens")
    for index, unit in enumerate(units):
        marker = "{mask_" + str(index) + "}"
        if prompt.count(marker) != 1:
            raise ValueError("Reviewed rewrite needs exactly one placeholder per unit")
        before = prompt.split(marker)[0]
        phrase = "in this region" if unit.edit_type == "add" else NOREF[unit.edit_type]
        if not re.search(r"(?<!\w)" + re.escape(phrase) + r" $", before, re.I):
            raise ValueError("Reviewed rewrite placeholder must follow the unit's noref phrase")
        prompt = prompt.replace(marker, "".join(unit.codes))
    if re.search(r"\{mask_", prompt):
        raise ValueError("Unknown reviewed rewrite placeholder")
    return prompt


def convert_record(record):
    """Common interchange format: images, instruction, edit_type and units.

    Units contain reviewed ref_phrase, edit_type, mask_codes and optional add
    anchor_phrase. No dataset-specific language guessing is done by this converter.
    Unsupported rewrites are reported independently; valid NTP/ref/plain rows survive.
    """
    if "sample_type" in record:
        return convert_sample(record)
    instruction = record["instruction"].strip()
    source_size = None
    if any("mask_paths" in x for x in record["units"]):
        source = record["edit_image"]
        source = source[0] if isinstance(source, list) and len(source) == 1 else source
        with Image.open(source) as im:
            source_size = im.size
    units = [
        Unit(
            canonical_reference(x["ref_phrase"], x["edit_type"]),
            annotation_codes(x, source_size),
            x["edit_type"],
            x.get("anchor_phrase"),
        )
        for x in record["units"]
    ]
    if not units:
        raise ValueError("A mask record must have at least one unit")
    for unit in units:
        if unit.edit_type not in set(EDIT_TYPES) - {"composite"}:
            raise ValueError("Each unit must have an atomic edit_type")
        if not isinstance(unit.ref_phrase, str) or not unit.ref_phrase.strip():
            raise ValueError("Missing unit reference")
        if not unit.codes or not all(is_valid_span(x) for x in unit.codes):
            raise ValueError("Unit needs valid mask_codes")
        if unit.edit_type in {"background", "global"} and len(unit.codes) != 1:
            raise ValueError("Background/global require one mask")
    annotation_order = list(units)
    # NTP and UMT use the same canonical instruction order, independent of
    # annotation storage order. Global operations without a phrase go last.
    errors = {}
    try:
        units.sort(key=lambda u: len(instruction) if u.edit_type == "global" else phrase_span(instruction, u.ref_phrase)[0])
    except ValueError as exc:
        errors["binding"] = str(exc)
    edit_type = "composite" if len(units) > 1 else units[0].edit_type
    native = native_edit_type(record)
    # Native labels constrain atomic operations; multiple independently bound
    # operations still make the derived sample composite (e.g. two recolors).
    if native is not None and any(u.edit_type != native for u in units):
        raise ValueError("Reviewed units disagree with the dataset's native type mapping")
    if "edit_type" in record and record["edit_type"] != edit_type:
        raise ValueError("Record edit_type disagrees with its atomic units")
    items = []
    for unit in units:
        # Ambiguity after removing the article is rejected, never guessed.
        label = unit.ref_phrase
        if unit.edit_type == "global":
            label = "this image"
        if len(unit.codes) > 1:
            label = "one of the " + label
        items.extend((s, label) for s in unit.codes)
    # NTP-only examples must also be consumable by pass 2.
    try:
        rebound = grouped_units(instruction, parse_cot(to_cot(items), nonempty=True))
        if len(rebound) != len(units) or [u.codes for u in rebound] != [u.codes for u in units]:
            raise ValueError("Localization round-trip changed unit grouping/order")
        if render_units(instruction, rebound) != render_units(instruction, units):
            raise ValueError("Localization round-trip changed reference placement")
    except ValueError as exc:
        errors["binding"] = str(exc)
    common = {"edit_image": record["edit_image"], "edit_type": edit_type}
    ntp = {
        **common,
        "sample_type": "edit_ntp",
        "prompt": instruction,
        "mt_cot": to_cot(items),
    }
    rows = []
    if "binding" not in errors:
        try:
            rows.append(validate_row(ntp))
        except ValueError as exc:
            errors["binding"] = str(exc)
    if "image" in record:
        fm = {**common, "image": record["image"]}
        rows.append({**fm, "sample_type": "edit", "prompt": instruction})
        for variant in ("ref", "noref"):
            if variant == "ref" and "binding" in errors:
                errors[variant] = errors["binding"]
                continue
            try:
                prompt = (reviewed_noref(record, annotation_order)
                          if variant == "noref" and "noref_instruction" in record
                          else render_units(instruction, units, variant=variant))
                rows.append(validate_row(
                    {
                        **fm,
                        "sample_type": "edit_umt",
                        "prompt": prompt,
                        "instr_variant": variant,
                    }
                ))
            except (ValueError, TypeError, KeyError) as exc:
                errors[variant] = str(exc)
        if len(rows) >= 2 and rows[-1].get("instr_variant") == "noref" and rows[-2].get("instr_variant") == "ref" and rows[-1]["prompt"] == rows[-2]["prompt"]:
            rows.pop(-2)  # Identical global variants count only as noref.
    for row in rows:
        validate_row(row)
    return rows, errors


def sample_edit_type(record):
    native = native_edit_type(record)
    if "edit_type" in record:
        mismatch = native is not None and record["edit_type"] != native
        units = record.get("units", [])
        if record["edit_type"] == "composite" and len(units) >= 2:
            mismatch = native is not None and any(u.get("edit_type") != native for u in units)
        if record["edit_type"] not in EDIT_TYPES or mismatch:
            raise ValueError("Invalid edit_type or conflict with native dataset type")
        return record["edit_type"]
    if native is not None:
        return native
    paths = record["edit_image"]
    paths = [paths] if isinstance(paths, str) else paths
    for path in paths:
        for part in Path(path).parts:
            for prefix, typ in TYPE_MAP.items():
                if part.startswith(prefix + "_"):
                    return typ
    raise ValueError("Supply reviewed edit_type or a recognized CrispEdit path prefix")


def convert_sample(record):
    """Convert stored training rows without inventing masks or target semantics."""
    typ, kind = sample_edit_type(record), record["sample_type"]
    row = {k: record[k] for k in ("edit_image", "image", "prompt", "mt_cot", "instr_variant") if k in record}
    row.update(edit_type=typ, sample_type=kind)
    if kind == "edit_mt":
        pairs = parse_cot(re.sub(r"^\s*<think>\s*</think>\s*", "", row["mt_cot"]))
        if not pairs:
            row.pop("mt_cot")
            row["sample_type"] = "edit"
            return [validate_row(row)], {}
        units = grouped_units(row["prompt"], pairs)
        if typ == "composite":
            if not record.get("units"):
                raise ValueError("Composite conversion requires reviewed atomic units")
            specs = record["units"]
        else:
            specs = []
            for u in units:
                anchor = None
                if typ == "add":
                    found = re.search(r"\b(?:near|next to|beside|behind|in front of|on top of|on|in|under|above|to|at)\s+.+$", u.ref_phrase, re.I)
                    anchor = found.group() if found else None
                specs.append(dict(ref_phrase=u.ref_phrase, edit_type=typ, mask_codes=list(u.codes), anchor_phrase=anchor))
        if typ == "composite":
            if len(specs) != len(units) or any(tuple(s["mask_codes"]) != u.codes or s["ref_phrase"] != u.ref_phrase for s, u in zip(specs, units)):
                raise ValueError("Reviewed units must preserve localization phrases and codes")
        converted, errors = convert_record(dict(instruction=row["prompt"], edit_image=row["edit_image"], image=row["image"], units=specs,
            **({"noref_instruction": record["noref_instruction"]} if "noref_instruction" in record else {})))
        return [r for r in converted if r["sample_type"] != "edit"], errors
    if kind == "edit_ntp":
        row.pop("image", None)
        row["mt_cot"] = to_cot(parse_generated_cot(row["mt_cot"]))
    if kind == "edit_umt" and "instr_variant" not in row:
        if typ == "composite":
            raise ValueError("Inline composite conversion needs reviewed per-unit types")
        spans_in(row["prompt"])
        groups = list(re.finditer(r"(?:" + SPAN_RE.pattern + r")+", row["prompt"]))
        for group in reversed(groups):
            before = row["prompt"][:group.start()].rstrip()
            before = re.sub(r"\b(?:the|a|an)$", "", before, flags=re.I).rstrip()
            if typ == "add":
                before = re.sub(r"\b(?:on top of|next to|in front of|near|beside|behind|on|in|under|above|to|at)$", "", before, flags=re.I).rstrip()
            if typ == "text":
                before = re.sub(r"\b(?:the\s+)?text$", "", before, flags=re.I).rstrip()
            phrase = "in this region" if typ == "add" else NOREF[typ]
            row["prompt"] = before + " " + phrase + " " + group.group() + row["prompt"][group.end():]
        row["instr_variant"] = "noref"
    return [validate_row(row)], {}
