"""Multi-object pairs as several protocol 1.0 units edited together (benchmark v2's mixed setting).

The single-unit adaptation (protocol_adapt) leaves a pair out of its manifest when the edit acts on two or more
separate objects (multi_object.jsonl). Here such a pair becomes 2-5 units, as benchmark v2 writes a case: each unit
is one source object, or one part of it, with one operation, its own ref / no-ref sentence and its own source mask
S; all units are carried out together in the pair's AFTER.

  slots     Qwen3.8-27B lists the units (MULTI_SLOT_PROMPT: the single-unit rules, per unit), then every unit is
            grounded like a single record (protocol_adapt.ground_record);
  units     every unit becomes a unit row (its own outlines; none when one outline covers several units) and a unit
            slots record, so scripts/data_cleaning/protocol_remask.py and protocol_adapt.py parts / build run on units
            unchanged: per-unit S, add anchors, geometry gates, edited region and edit gates;
  assemble  a pair is kept only when every unit is accepted, no two units select the same object and the units'
            edits explain the pair's change (pair_checks);
  audit     one Qwen3.8-27B check of the whole pair: each unit, and whether every change belongs to a unit.
"""
from __future__ import annotations

from collections import Counter
import json
import re

import numpy as np
from PIL import Image

from samtok_edit21.preparation import edit_check as ec
from samtok_edit21.preparation import instruction_protocol as ip
from samtok_edit21.preparation import protocol_adapt as pa

MULTI_VERSION = "protocol-multi-1"
# Pair gates, set on the pilot: two units whose S overlap this much select the same object; a pair whose change lies
# this much outside every unit's edit (grown by explain_grow of the diagonal) has an edit no unit describes.
MULTI_PARAMS = {"min_units": 2, "max_units": 5, "same_s_iou": 0.5, "contained": 0.8, "unexplained": 0.1,
                "explain_grow": 0.01}
UNIT_FIELDS = ("instruction_spec", "instruction_ref", "instruction_noref", "mask_semantics", "selection",
               "execution_region", "edited_region", "flags")


def _between(text, start, end):
    return text[text.index(start): text.index(end)]


# The single-unit rules, per unit: operations, the source object S and the slots (a template, as SLOT_PROMPT).
_OPERATIONS = _between(pa.SLOT_PROMPT, "3. operation, judged by", "\n\n4. The source object S")
_SOURCE = _between(pa.SLOT_PROMPT, "4. The source object S", "\n- instances: give every outlined number")
_SLOTS = _between(pa.SLOT_PROMPT, "5. Slots.", "\n\nReturn only JSON:")

MULTI_SLOT_PROMPT = """You convert one training example of an image-editing dataset into structured edit records.
Image 1 is BEFORE (the original photo). Image 2 is BEFORE with labeled areas outlined. Image 3 is AFTER (the edited photo). Image 4 is a close-up around the labeled areas, BEFORE on the left and AFTER on the right.

Original instruction: "{instruction}"
Original type label: {label}
Outlines in image 2: {legend}

The example already shows an edit: describe that edit as AFTER shows it; never choose a different or better edit. Read colors from image 1, never from the outline colors.

An earlier check found that this example edits several separate objects at once. Write it as a list of units, one unit per separate object: a unit is ONE source object, or one part of it, with ONE of the operations below, and becomes one instruction of its own; all units are carried out together. Counted as ONE object: a pile, heap, bunch, stack or cluster of same-kind items lying together (a pile of sweet potatoes, a bunch of bananas, a stack of plates, a bouquet), and a matching pair worn or used by one person (shoes, gloves, sleeves). Several new items of the same kind at one anchor are one add unit; give their number.

1. observed: say in one sentence what actually changed from BEFORE to AFTER.

2. decision:
- "accept": every change in AFTER belongs to exactly one of 2 to 5 units. Changes that only follow from an edit (shadows, revealed background, a carried bag removed with a person) belong to that unit.
- "single": after all, the change acts on ONE object with one operation; give that one unit.
- "reject": a change fits no operation below (moving, flipping, re-posing, style or whole-image change; a person's own arms, legs or head appearing or moving is re-posing); more than 5 units would be needed; an edited object cannot be picked out with a short phrase (one person in a crowd, one of many identical objects in a row); a unit would have to cover several people or animals because they cannot each be named; one new item takes the place of several objects (two icons replaced by one button), or the number of new items differs from the number of objects they replace (three cars become two taxis); a person's face or body becomes another person's (an identity change is not a color change); AFTER does not show the instructed edits; or an added item has nothing visible to stand, rest or hang on, be attached to or be held by, and is not something that flies or floats by itself.

""" + _OPERATIONS + "\n\n" + _SOURCE + """
- Each unit's S is ONE object or one part of it, never a group of people or animals ("the group of people", "the soldiers", "the children"): give each person or animal its own unit. Two units never select the same object, nor an object and a part of it.
- When the clothing or the hair of a person changes, S is that garment or the hair ("the shirt of the man on the left"), not the whole person; a person's skin color and their clothing are two units.
- One object seen in several pieces (cut by something in front of it) is one unit. A change that only follows from another unit's edit (the hand that held a removed child, background revealed behind a removed object) is part of that unit, never a unit of its own. The refs of units whose S is the same kind of thing must tell their objects apart, e.g. "the man on the left of the white truck" and "the man on the right of the white truck".
- instances: for each unit, list the outlined numbers that belong to it, each with a role: "main" (S itself or a visible piece of it; for an add, the outline marks exactly S), "anchor" (for an add, the outline marks the larger object or surface S belongs to), "attached" (belongs to S and changes with it, e.g. its contents or a carried bag) or "new_item" (an added or replacing item, drawn where it appears in AFTER). A number belongs to at most one unit. When one outline covers the objects of several units (a group of people in one outline), list it for no unit and put it in "shared". Numbers that do not change go in "unchanged".

""" + _SLOTS + """

Return only JSON:
{{"observed": "...", "decision": "accept", "reason": "<at most 15 words>", "units": [{{"operation": "remove", "instances": [{{"n": 1, "role": "main"}}], "ref": "the man on the left of the white truck", "scope": "man", "host_ref": "", "host_scope": "", "part": "", "granularity": "", "target_part": "", "payload": "", "remove_mode": "whole", "attributes": [], "placement": {{}}, "qualifier": "", "s_location": "left middle"}}, {{"operation": "attribute", "instances": [{{"n": 2, "role": "main"}}], "ref": "the ...", "scope": "...", "host_ref": "", "host_scope": "", "part": "", "granularity": "", "target_part": "", "payload": "", "remove_mode": "whole", "attributes": [{{"property": "color", "value": "..."}}], "placement": {{}}, "qualifier": "", "s_location": "..."}}], "shared": [], "unchanged": []}}"""


def multi_slot_prompt(row):
    return MULTI_SLOT_PROMPT.format(instruction=row["instruction"].strip(),
                                    label=pa.TYPE_LABELS.get(row["type"], row["type"]),
                                    legend=pa.legend(row), writing_rules=pa.WRITING_RULES)


def _numbers(value, count):
    out = []
    for item in value or []:
        try:
            n = int(item)
        except (TypeError, ValueError):
            continue
        if 1 <= n <= count and n not in out:
            out.append(n)
    return out


GROUP = re.compile(r"\b(group|groups|crowd|people|persons|figures|men|women|children|kids|boys|girls|soldiers|players|"
                   r"tourists|pedestrians|spectators|workers|riders|dancers|animals|monks|row|rows|line|lines)\b", re.I)
# A plural head names several objects unless it is one thing named in the plural (a pair, a garment, parts of a
# host) or a pile, bunch or stack of same-kind items lying together (the user's rule, 2026-10-12).
PLURAL_OK = {"pants", "trousers", "jeans", "shorts", "leggings", "tights", "glasses", "sunglasses", "goggles",
             "spectacles", "binoculars", "headphones", "earphones", "scissors", "pliers", "tongs", "shoes", "boots",
             "sneakers", "sandals", "slippers", "heels", "gloves", "mittens", "socks", "earrings", "cufflinks",
             "sleeves", "stairs", "steps", "curtains", "drapes", "shutters", "blinds", "handlebars", "stripes",
             "contents", "clothes", "lettering", "graphics", "letters", "numbers", "digits", "words", "petals",
             "whiskers", "lips", "teeth", "eyes", "ears", "eyebrows", "lashes", "eyelashes", "wings", "horns",
             "antlers", "legs", "arms", "hands", "feet", "paws", "braids", "pigtails", "bangs", "reins", "rails",
             "tracks", "spokes", "buttons", "laces", "straps", "bristles", "keys", "candies", "sprinkles",
             "seeds", "grapes", "berries", "nuts", "beans", "fries", "noodles", "leaves", "flowers", "lights",
             "decorations", "ornaments", "garlands", "balloons", "bubbles", "sparkles", "fireworks", "clouds"}
COLLECTIVE = re.compile(r"\b(pile|heap|bunch|stack|cluster|bouquet|pair|set|bowl|basket|plate|tray|box|bag|jar|"
                        r"bottle|crate|bundle|string|wreath|garland|arrangement|swarm|flock|herd|school)\s+of\b", re.I)
SINGULAR_S = re.compile(r"(ss|us|is|ys|ics|lens|canvas|gas|chassis|series|species|news|atlas|bias)$", re.I)
PERSON = re.compile(r"^(man|woman|person|child|boy|girl|kid|baby|player|rider|soldier|lady|guy|people|figure)$", re.I)
BODY = re.compile(r"\b(arm|arms|hand|hands|leg|legs|foot|feet|head|heads)\b", re.I)
GARMENT = re.compile(r"\b(uniform|shirt|t-shirt|jersey|jacket|coat|dress|clothes|clothing|outfit|pants|trousers|"
                     r"shorts|skirt|hat|cap|hoodie|sweater|vest|suit)\b", re.I)


def several(spec):
    """Whether S's kind names several separate objects ("lounge chairs", "the monks"), not one object or a pile."""
    scope = spec["scope"].lower().strip()
    if not scope or COLLECTIVE.search(scope) or COLLECTIVE.search(spec["ref"]):
        return False
    if GROUP.search(scope):
        return True
    head = re.split(r"\s+(?:and|with)\s+", scope)[-1].split()[-1]
    if head in PLURAL_OK or SINGULAR_S.search(head) or not head.endswith("s"):
        return False
    return " of " not in spec["ref"].lower()  # "the legs of the table", "the stripes of the vest": parts of one host


def unit_problems(spec):
    """A unit that is not one object with one operation: a group, a whole person recolored, a re-posed body."""
    problems = []
    scope, op = spec["scope"].lower(), spec["operation"]
    if several(spec):
        problems.append(f'S "{spec["ref"]}" names several separate objects; give each its own unit, name a pile, '
                        'bunch or stack of same-kind items lying together as such ("the pile of pumpkins"), or decide '
                        '"reject"')
    values = " ".join(a["value"].lower() for a in spec["attributes"])
    if op == "attribute" and PERSON.match(scope) and ("skin" not in values or GARMENT.search(values)):
        problems.append(f'an attribute of "{spec["ref"]}" changes a garment or the hair: S is that garment or the hair '
                        '(e.g. "the shirt of the man on the left"), not the whole person; skin and clothing are two '
                        'units')
    if op == "add" and BODY.search(spec["payload"]) and PERSON.match(scope.split()[-1] if scope else ""):
        problems.append("a person's own arms, legs or head appearing or moving is re-posing, which is no operation: "
                        'decide "reject"')
    return problems


def normalize_units(raw, count):
    """(decision, units) from the model's answer; any problem raises ValueError, which sends the call back.
    A unit: {"unit": "U1", "slots": normalized slots (instances as numbers of the pair's outlines), "spec": spec}."""
    decision = ip.clean(raw.get("decision")).lower()
    if decision not in ("accept", "single", "reject"):
        raise ValueError(f"unknown decision {decision!r}")
    if decision == "reject":
        return decision, []
    items = raw.get("units") or []
    p = MULTI_PARAMS
    if decision == "accept" and not p["min_units"] <= len(items) <= p["max_units"]:
        raise ValueError(f'"accept" needs {p["min_units"]} to {p["max_units"]} units, got {len(items)}')
    if decision == "single" and len(items) != 1:
        raise ValueError(f'"single" needs exactly one unit, got {len(items)}')
    units, problems, owner = [], [], {}
    for k, item in enumerate(items, 1):
        name = f"U{k}"
        if not isinstance(item, dict):
            raise ValueError(f"{name} is not an object")
        try:
            slots, fixes = pa.normalize_slots(item)
            problems += [f"{name}: {x}" for x in pa.slot_problems(slots)]
            spec = pa.to_spec(slots)
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ValueError(f"{name}: {exc}") from exc
        if spec["target_part"] and spec["operation"] != "add":
            spec = pa.part_as_s(spec)
            fixes = fixes + ["part_as_s"]
        roles = pa.roles_of(item, count)
        for n, role in roles.items():
            if role in ("separate", "unchanged"):
                problems.append(f'{name}: role "{role}" is not a unit role; put outline {n} in "shared" or "unchanged"')
            elif n in owner:
                problems.append(f"outline {n} is listed for both {owner[n]} and {name}")
            else:
                owner[n] = name
        slots.update(decision="accept", fixes=fixes,
                     instances=[{"n": n, "role": r} for n, r in roles.items() if r not in ("separate", "unchanged")])
        units.append({"unit": name, "slots": slots, "spec": spec})
    for u in units:
        problems += [f"{u['unit']}: {x}" for x in unit_problems(u["spec"])]
    refs = Counter(u["spec"]["ref"].lower() for u in units)
    same = sorted(r for r, c in refs.items() if c > 1)
    if same:
        problems.append(f"units share the ref {same}: refs of units with the same kind of S must tell their objects "
                        "apart")
    if problems:
        raise ValueError("; ".join(problems))
    return decision, units


async def extract_multi(client, row, parts, ground, ground_after=None, feedback=None):
    """The units of a multi-object pair, each grounded as a single record (protocol_adapt.ground_record)."""
    record = {"id": row["id"], "attempts": [], "tokens": 0, **({"feedback": feedback} if feedback else {})}
    base = multi_slot_prompt(row)
    if feedback:
        base += ("\n\nA reviewer checked an earlier answer for this example and found: "
                 f"{feedback['problems']}. Answer again and fix this; the units, their operations, S, relations and "
                 "wording must match what AFTER shows.")
    prompt, decision, units, raw = base, None, [], {}
    for seed in range(pa.SLOT_ATTEMPTS):
        response = await client.call(parts, prompt, "slots", seed=seed)
        record["tokens"] += response["tokens"]
        try:
            raw = pa.parse_json(response["text"])
            decision, units = normalize_units(raw, len(row["instances"]))
            record["attempts"].append({"seed": seed, "ok": True})
            break
        except Exception as exc:  # noqa: BLE001 - malformed or invalid: ask again, saying what was wrong
            error = f"{type(exc).__name__}: {exc}"[:600]
            record["attempts"].append({"seed": seed, "ok": False, "error": error, "finish": response["finish"],
                                       "text": response["text"][-1500:]})
            prompt = base + f"\n\nYour previous answer was not usable ({error}). Answer again, fixing it."
            decision = None
    if decision is None:
        record.update(slots=None, units=[])
        return record
    count = len(row["instances"])
    record["slots"] = {"observed": ip.clean(raw.get("observed")), "decision": decision,
                       "reason": ip.clean(raw.get("reason")), "shared": _numbers(raw.get("shared"), count),
                       "unchanged": _numbers(raw.get("unchanged"), count)}
    record["units"] = []
    for unit in units:
        sub = unit_row(row, unit, record["slots"]["shared"])
        unit_record = {"id": sub["id"], "unit": unit["unit"], "tokens": 0, "slots": unit_slots(unit),
                       "spec": unit["spec"], "pair_instances": unit["slots"]["instances"]}
        await pa.ground_record(client, sub, ground, unit_record, None, ground_after)
        record["tokens"] += unit_record["tokens"]
        record["units"].append(unit_record)
    return record


def unit_numbers(unit):
    return [i["n"] for i in unit["slots"]["instances"]]


def unit_slots(unit):
    """The unit's slots with its outlines numbered as in its unit row (1..m in the order listed)."""
    return {**unit["slots"], "instances": [{"n": k, "role": i["role"]}
                                           for k, i in enumerate(unit["slots"]["instances"], 1)]}


def unit_row(row, unit, shared=()):
    """A unit as a row of its own: the pair's images and instruction, only the unit's outlines, and as the cleaned
    region their union (for an add only the new item's outline: it marks where the item goes). A unit without an
    outline of its own (split out of an outline covering several units) has no region and a hint where to look."""
    numbers = unit_numbers(unit)
    instances = [row["instances"][n - 1] for n in numbers]
    roles = [i["role"] for i in unit["slots"]["instances"]]
    if unit["spec"]["operation"] == "add":
        marked = [i for i, r in zip(instances, roles) if r == "new_item"]
    else:
        marked = instances
    out = {**row, "id": f"{row['id']}~{unit['unit']}", "pair_id": row["id"], "unit": unit["unit"],
           "instances": instances, "region": pa.union(marked) if marked else None}
    if not marked and shared:
        box = pa.union([row["instances"][n - 1] for n in shared])["box_1000"]
        out["ground_hint"] = ("it is one of the objects inside the outlined area at about [{}, {}, {}, {}] "
                              "(0-1000 coordinates)".format(*box))
    return out


def unit_rows(row, record):
    """(unit row, unit slots record) of every unit of a pair record from extract_multi."""
    shared = (record.get("slots") or {}).get("shared") or []
    out = []
    for unit_record in record.get("units") or []:
        unit = {"unit": unit_record["unit"], "spec": unit_record["spec"],
                "slots": {**unit_record["slots"], "instances": unit_record["pair_instances"]}}
        out.append((unit_row(row, unit, shared), unit_record))
    return out

# ---------------------------------------------------------------- pair checks and records


def _mask(region):
    return pa.decode(region).astype(bool)


def pair_checks(row, units, images, params=MULTI_PARAMS):
    """Whether the accepted units make a sound pair: no two select the same object, and their edits explain the
    pair's change. Returns (checks, hard reasons, the pair's edited region or None)."""
    p = params
    s_masks = [_mask(u["selection"]) for u in units]
    same = []
    for a in range(len(units)):
        for b in range(a + 1, len(units)):
            inter = (s_masks[a] & s_masks[b]).sum()
            iou = inter / max(int((s_masks[a] | s_masks[b]).sum()), 1)
            inside = inter / max(min(int(s_masks[a].sum()), int(s_masks[b].sum())), 1)  # a part of the other unit's S
            if iou > p["same_s_iou"] or inside > p["contained"]:
                same.append([units[a]["id"], units[b]["id"], round(float(iou), 3), round(float(inside), 3)])
    anchor = np.zeros_like(s_masks[0])
    for u, s in zip(units, s_masks):
        anchor |= s
        if u.get("execution_region"):
            anchor |= _mask(u["execution_region"])
    for instance in row["instances"]:
        anchor |= pa.decode(instance).astype(bool)
    total, stats = pa.edited_region(*images, anchor, None)
    covered = np.zeros_like(anchor)
    for u in units:
        covered |= _mask(u["edited_region"])
    h, w = anchor.shape
    covered = pa.grow(covered, p["explain_grow"] * (h * h + w * w) ** 0.5)
    unexplained = (float((total & ~covered).sum()) / max(int(total.sum()), 1)) if total is not None else 0.0
    checks = {"same_s": same, "unexplained_frac": round(unexplained, 3),
              "pair_edited_area_frac": stats["area_frac"], "pair_noise": stats["noise"]}
    hard = (["units_share_s"] if same else []) + (["change_not_in_units"] if unexplained > p["unexplained"] else [])
    return checks, hard, (total if total is not None and total.any() else None), stats


def unit_fields(name, record):
    return {"id": name, **{k: record.get(k) for k in UNIT_FIELDS}}


def assemble(row, record, unit_records, root):
    """The pair record from its slots record and its units' built records (protocol_adapt.build + edit gates);
    root: the dataset root of the pair's images."""
    out = {"id": row["id"], "dataset": row["dataset"], "type_before": row["type"],
           "instruction_before": row["instruction"], "adapter_version": pa.ADAPTER_VERSION,
           "multi_version": MULTI_VERSION, "instruction_protocol_version": ip.VERSION,
           "slots": record.get("slots")}
    slots = record.get("slots")
    if slots is None:
        return {**out, "status": "rejected", "reason": "no_valid_units"}
    if slots["decision"] == "reject":
        return {**out, "status": "rejected", "reason": "slots_reject"}
    built = [unit_records.get(u["id"]) for u in record["units"]]
    units = [unit_fields(u["unit"], b) if b else {"id": u["unit"]} for u, b in zip(record["units"], built)]
    failed = [f"{u['unit']}:{(b or {}).get('reason') or 'missing'}" for u, b in zip(record["units"], built)
              if not b or b["status"] != "accepted"]
    out["units"] = units
    if slots["decision"] == "single":
        return {**out, "status": "single_unit", "reason": ",".join(failed) or None}
    if failed:
        return {**out, "status": "rejected", "reason": ";".join(failed)}
    checks, hard, total, stats = pair_checks(row, units, pa.open_pair(row, root))
    out["checks"] = checks
    if total is not None:
        out["edited_region"] = {"method": pa.EDITED_METHOD, **pa.encode(total), "area_frac": stats["area_frac"],
                                "stats": stats}
    out["flags"] = sorted({f"{u['id']}:{f}" for u in units for f in u.get("flags") or [] if not f.startswith("fix:")})
    return {**out, "status": "rejected" if hard else "accepted", "reason": ",".join(hard) or None}


# ---------------------------------------------------------------- audit of a pair

MULTI_AUDIT_PROMPT = """You check one rewritten training example of an image-editing dataset that edits several objects at once.
Image 1 is BEFORE with the source object S of every unit outlined and numbered: {legend}. Image 2 is AFTER. Image 3 shows each unit close up, one row per unit in order, BEFORE with that unit's outline on the left and AFTER on the right.

The units, given together as one request:
{units}

Judge every unit "pass" or "fail":
1. edit: the unit's instruction describes what actually changed at its object from BEFORE to AFTER: the right operation, the right object, the requested new item or value (kind, count, color, material or exact text), with nothing important wrong or missing. Changes at the other units' objects belong to those units. Changes that only follow from the edit (shadows, revealed background, a carried bag removed with a person) are fine.
2. relation: for an add, the stated relation of the new item to S is true in AFTER and the new item touches S: it stands, rests, hangs or is attached on it, holds it up, or flies or floats inside the outlined sky or water. For other operations answer "pass".
3. target: the unit's outline marks what its reference names: for an add what the new item directly stands, rests, hangs, is attached or printed on in AFTER (the sky or water around it for something that flies or floats); for other operations the object or part that changes. Not a different instance and not a much larger or smaller thing.
4. outline: the outline covers the visible extent of that object or part (for an add it may be just the part of a larger object or surface at the spot where the new item goes), with no large missing piece and no large area of something else.
5. noref: using the unit's outline as the selection, the instruction without the reference asks for the same edit, and its "the selected ..." noun fits the outlined object.
6. wording: both instructions of the unit are clear, natural, concise English.
Then judge the whole example:
7. complete: every visible change from BEFORE to AFTER belongs to one of the units, apart from effects that follow from them; "fail" when a changed object or area is covered by no unit, when two units select the same object (or an object and a part of it), or when two units describe the same new item (one new object in AFTER where two units' objects were).

Return only JSON: {{"units": [{units_json}], "complete": "pass", "problems": "<at most 40 words, naming the unit; empty if everything passes>"}}"""
UNIT_AUDIT_KEYS = ("edit", "relation", "target", "outline", "noref", "wording")


def multi_audit_prompt(pair):
    colors = [pa.PALETTE[k % len(pa.PALETTE)][0] for k in range(len(pair["units"]))]
    legend = "; ".join(f"[{k}] {c} = {u['id']}" for k, (u, c) in enumerate(zip(pair["units"], colors), 1))
    lines = "\n".join(f'{u["id"]} (outline {k}, {c}; {u["instruction_spec"]["operation"]}): with the reference: '
                      f'"{u["instruction_ref"]}"; without it: "{u["instruction_noref"]}"'
                      for k, (u, c) in enumerate(zip(pair["units"], colors), 1))
    example = ", ".join('{"id": "%s", "edit": "pass", "relation": "pass", "target": "pass", "outline": "pass", '
                        '"noref": "pass", "wording": "pass"}' % u["id"] for u in pair["units"])
    return MULTI_AUDIT_PROMPT.format(legend=legend, units=lines, units_json=example)


def _unit_box(unit):
    boxes = [r["box_1000"] for r in (unit.get("selection"), unit.get("execution_region"), unit.get("edited_region"))
             if r and r.get("box_1000")]
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def multi_audit_parts(row, pair, root, width=1100, row_height=260):
    source, target = pa.open_pair(row, root)
    masks = [(_mask(u["selection"]), pa.PALETTE[k % len(pa.PALETTE)][1]) for k, u in enumerate(pair["units"])]
    strips = []
    for k, (u, m) in enumerate(zip(pair["units"], masks)):
        close = pa.close_up(source, target, [m], _unit_box(u), max_long=width,
                            left=f"{u['id']} BEFORE", right=f"{u['id']} AFTER")
        scale = min(width / close.width, row_height / close.height)
        strips.append(close.resize((max(1, round(close.width * scale)), max(1, round(close.height * scale))),
                                   Image.Resampling.LANCZOS))
    sheet = Image.new("RGB", (max(s.width for s in strips), sum(s.height for s in strips) + 8 * (len(strips) - 1)),
                      "white")
    y = 0
    for s in strips:
        sheet.paste(s, (0, y))
        y += s.height + 8
    views = [("Image 1, BEFORE with the units' outlines:", pa.outlined(source, masks, pa.FULL_PIXELS)),
             ("Image 2, AFTER:", ec.resized(target, pa.FULL_PIXELS)),
             ("Image 3, each unit close up (BEFORE left, AFTER right):", sheet)]
    parts = []
    for label, image in views:
        parts += [{"type": "text", "text": label}, {"type": "image_url", "image_url": {"url": ec.data_url(image)}}]
    return parts


async def multi_audit(client, pair, parts, votes=2):
    """Independent pass/fail check of an assembled pair; every vote must pass every unit and the whole."""
    prompt = multi_audit_prompt(pair)
    ids = [u["id"] for u in pair["units"]]
    results = []
    for vote in range(votes):
        response = await client.call(parts, prompt, "audit", seed=vote)
        result = {"tokens": response["tokens"], "units": {}, "complete": None, "problems": ""}
        try:
            raw = pa.parse_json(response["text"])
            for item in raw.get("units") or []:
                if isinstance(item, dict) and item.get("id") in ids:
                    result["units"][item["id"]] = {k: ip.clean(item.get(k)).lower() or None for k in UNIT_AUDIT_KEYS}
            result["complete"] = ip.clean(raw.get("complete")).lower() or None
            result["problems"] = ip.clean(raw.get("problems"))
        except Exception as exc:  # noqa: BLE001 - recorded as an unknown verdict
            result["error"] = f"{type(exc).__name__}: {exc}"[:200]
            result["text"] = response["text"][-600:]
        checks = [result["units"].get(i, {}).get(k) for i in ids for k in UNIT_AUDIT_KEYS] + [result["complete"]]
        result["passed"] = (None if any(c not in ("pass", "fail") for c in checks)
                            else all(c == "pass" for c in checks))
        results.append(result)
    merged = {"id": pair["id"], "tokens": sum(r["tokens"] for r in results), "votes": results}
    merged["units"] = {i: {k: ("fail" if any(r["units"].get(i, {}).get(k) == "fail" for r in results) else
                               "pass" if all(r["units"].get(i, {}).get(k) == "pass" for r in results) else None)
                           for k in UNIT_AUDIT_KEYS} for i in ids}
    complete = [r["complete"] for r in results]
    merged["complete"] = "fail" if "fail" in complete else "pass" if all(c == "pass" for c in complete) else None
    merged["passed"] = (None if any(r["passed"] is None for r in results) else all(r["passed"] for r in results))
    merged["problems"] = " | ".join(r["problems"] for r in results if r["problems"])
    return merged


def audit_failures(audit):
    """The failed checks of a pair audit as "U2:target" and "complete"."""
    out = [f"{i}:{k}" for i, checks in (audit.get("units") or {}).items() for k, v in checks.items() if v == "fail"]
    return out + (["complete"] if audit.get("complete") == "fail" else [])

# ---------------------------------------------------------------- review sheet


def pair_sheet(row, pair, root, width=1400):
    """BEFORE with the units' S outlined and numbered, AFTER with the pair's edited region, and the sentences."""
    from PIL import ImageDraw, ImageFont

    source, target = pa.open_pair(row, root)
    units = pair.get("units") or []
    masks = [(_mask(u["selection"]), pa.PALETTE[k % len(pa.PALETTE)][1]) for k, u in enumerate(units)
             if u.get("selection")]
    panel = (width // 2 - 10) ** 2
    before = ec.labeled(pa.outlined(source, masks, panel) if masks else ec.resized(source, panel),
                        "BEFORE: unit S numbered")
    edited = [(_mask(pair["edited_region"]), pa.EDITED_COLOR)] if pair.get("edited_region") else None
    after = ec.labeled(pa.outlined(target, edited, panel, numbers=False) if edited else ec.resized(target, panel),
                       "AFTER: changed area magenta")
    font = ImageFont.truetype(ec.font_path().replace("-Bold", ""), 17)
    probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    texts = [f"{pair['id']} | {pair['status']}" + (f" ({pair['reason']})" if pair.get("reason") else ""),
             "OLD: " + row["instruction"]]
    for k, u in enumerate(units, 1):
        if u.get("instruction_ref"):
            texts.append(f"[{k}] {u['id']} {u['instruction_spec']['operation']}: {u['instruction_ref']}  |  "
                         f"{u['instruction_noref']}")
    lines = [line for text in texts for line in pa.wrap(probe, text, font, before.width + after.width)]
    top = max(before.height, after.height)
    sheet = Image.new("RGB", (before.width + after.width + 20, top + 10 + 22 * len(lines)), "white")
    sheet.paste(before, (0, 0))
    sheet.paste(after, (before.width + 20, 0))
    draw = ImageDraw.Draw(sheet)
    for n, line in enumerate(lines):
        draw.text((8, top + 6 + 22 * n), line, font=font, fill=(0, 0, 0))
    return sheet


def read_jsonl(path):
    with open(path) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def clean_unit_id(text):
    return re.sub(r"[^A-Za-z0-9_.~-]", "_", text)
