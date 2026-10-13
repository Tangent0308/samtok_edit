"""Adapt cleaned training pairs to instruction protocol 1.0 (benchmark release 2.4.0).

Protocol 1.0 is the shared instruction protocol of the benchmark and the training data
(Tangent0308/samtok_edit_benchmark, branch v2branch, docs/INSTRUCTION_PROTOCOL_ZH.md, vendored as
docs/INSTRUCTION_PROTOCOL_ZH.md).  Its canonical module (records, rendering, mask_contract) and its writing style
are vendored byte for byte as instruction_protocol.py and instruction_style.py; tests pin the hashes.  The
cleaning-side rules are in docs/10_SAMTokEdit_Qwen21_训练数据协议适配.md.

Per pair:
  draft   rule-based parse of the original instruction, used only for cross-checks;
  slots   Qwen3.8-27B reads BEFORE, AFTER and the labeled areas and fills the protocol slots;
  ground  the same model boxes S (and for an add its host, its part and, in a Derived pair, the new item);
  remask  SAM3 with the labeling repos' settings (scripts/data_cleaning/protocol_remask.py), and for a part-level
          add SAM's part candidates inside the host;
  parts   the model picks the part among the shortlisted candidates;
  build   spec -> ref / no-ref instructions, S and execution regions, checks and flags.

S follows the protocol: the region marks an existing visible source instance or part.  For add it is what the new
item directly stands, rests or hangs on, is attached to or is held by, at the finest level that has a name: the
object, its part at the new item, or a stretch of a surface (a training-side extension the shared selector text does
not name yet); a large one is cut to the spot (localize_anchor), and a part-level add also keeps the benchmark-style
host view (the whole host, the part as the site).  A reference anchor is used only when the new item holds S up.
Otherwise S is the edited object or part: a labeled source mask that marks S is reused byte for byte (the user's
rule, 2026-10-12), and only an S without one is re-masked.  The cleaned `region` keeps its meaning and becomes the
execution region.
"""
from __future__ import annotations

import asyncio
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

from samtok_edit21.preparation import edit_check as ec
from samtok_edit21.preparation import instruction_protocol as ip
from samtok_edit21.preparation import instruction_style as style

PROTOCOL_SHA256 = "ed8c5d7b932e9d2e3092ad907435a0e4ac3f7203b3503178165202173dac5fb5"
PROTOCOL_SOURCE = "Tangent0308/samtok_edit_benchmark@fef0c51:src/samtok_benchmark/v2/instruction_protocol.py"
ADAPTER_VERSION = "protocol-adapt-3"  # 3: add S at three granularities, host views, shared style
STYLE_SHA256 = "6546b3958e94372bfdefa9e2d4290500349dc801253ab7930697ebc2757e95e5"
# The shared writing rules without the benchmark's add authoring policy: that policy chooses a natural item for a
# host, while a training pair already shows its item; the cleaning-side add rules are in the slot prompt.
assert style.INSTRUCTION_STYLE.endswith("\n\n" + style.ADD_ANCHOR_POLICY)
WRITING_RULES = style.INSTRUCTION_STYLE[: -len("\n\n" + style.ADD_ANCHOR_POLICY)]

FULL_PIXELS = ec.FULL_PIXELS
GROUND_PIXELS = 1024 * 1024
# call: (thinking, reasoning effort, max output tokens)
CALLS = {"slots": (True, "low", 12000), "ground": (False, None, 400), "audit": (True, "low", 8000)}
GREEDY = {"temperature": 0.0}  # the labeling repos' grounding calls are greedy without thinking
SLOT_ATTEMPTS = 3

PALETTE = (("red", (235, 50, 45)), ("blue", (40, 110, 240)), ("yellow", (250, 210, 0)),
           ("magenta", (220, 40, 220)), ("cyan", (0, 200, 220)), ("orange", (255, 140, 0)),
           ("green", (40, 190, 60)), ("white", (250, 250, 250)))
S_COLOR, EXEC_COLOR, EDITED_COLOR, HOST_COLOR = (40, 220, 60), (255, 150, 0), (230, 40, 200), (250, 225, 110)

# ---------------------------------------------------------------- rows and regions

EXTRA_SURFACE = re.compile(r"^additional changed surface\b", re.I)
REMOVE_VERB = re.compile(r"(remove|erase|delete|get rid of|take away|take out|eliminate|wipe off)\b", re.I)


def read_jsonl(path):
    with open(path) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_rows(no_text_action, kept):
    """Typed pairs without text and action, plus the text pairs, which the model routes."""
    rows = [r for r in read_jsonl(no_text_action) if r["type"]]
    return rows + [r for r in read_jsonl(kept) if r["type"] == "text"]


def text_route(instruction):
    s = instruction.strip()
    if REMOVE_VERB.match(s):
        return "remove"
    return "add" if re.match(r"add\b", s, re.I) else "attribute"


def pilot_rows(rows, per_type, per_text_route, seed=20261012):
    """A stratified sample: per (dataset, type) and, for text pairs, per rule-based route."""
    cells = defaultdict(list)
    for row in rows:
        key = ("text", text_route(row["instruction"])) if row["type"] == "text" else (row["dataset"], row["type"])
        cells[key].append(row)
    rng, picked = random.Random(seed), []
    for (group, kind), members in sorted(cells.items()):
        count = per_text_route.get(kind, 0) if group == "text" else per_type.get(kind, 0)
        picked += rng.sample(members, min(count, len(members)))
    return picked


def instance_side(instance):
    if instance.get("mapped_from_target") is True or instance.get("grounding_image") == "target":
        return "target"
    return "extra" if EXTRA_SURFACE.match(instance.get("ref") or "") else "source"


def decode(rle):
    from pycocotools import mask as mask_utils
    return mask_utils.decode({"size": list(rle["rle_size"]), "counts": rle["rle_counts"].encode()}).astype(bool)


def box_1000(mask):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    h, w = mask.shape
    return [int(xs.min() * 1000 // w), int(ys.min() * 1000 // h),
            int(-(-(xs.max() + 1) * 1000 // w)), int(-(-(ys.max() + 1) * 1000 // h))]


def encode(mask):
    from pycocotools import mask as mask_utils
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"rle_size": [int(v) for v in rle["size"]], "rle_counts": rle["counts"].decode(), "box_1000": box_1000(mask)}


def union(instances):
    """Union of labeled instances; a single instance keeps its RLE byte for byte."""
    if len(instances) == 1:
        rle = {k: instances[0][k] for k in ("rle_size", "rle_counts")}
        return {**rle, "box_1000": box_1000(decode(rle))}
    from pycocotools import mask as mask_utils
    merged = mask_utils.merge([{"size": list(i["rle_size"]), "counts": i["rle_counts"].encode()} for i in instances],
                              intersect=False)
    rle = {"rle_size": [int(v) for v in merged["size"]], "rle_counts": merged["counts"].decode()}
    return {**rle, "box_1000": box_1000(decode(rle))}


def derived_execution_region(row):
    """Derived remove pairs record what was actually removed (attached items included)."""
    locator = row["provenance"]["locator"]
    with open(locator["file"]) as stream:
        for number, line in enumerate(stream):
            if number == locator["row"]:
                region = json.loads(line).get("execution_region") or {}
                break
        else:
            return None
    mask = region.get("mask")
    if not mask or not str(region.get("status", "")).startswith("resolved"):
        return None
    rle = {"rle_size": list(mask["size"]), "rle_counts": mask["counts"]}
    return {**rle, "box_1000": box_1000(decode(rle))}

# ---------------------------------------------------------------- rule-based draft

QUOTED = re.compile(r"(?<![A-Za-z0-9])['\"‘“]([^'\"‘’“”]+?)['\"’”](?![A-Za-z0-9])")
COUNT_WORDS = r"one|two|three|four|five|six|seven|eight|nine|ten|single|pair|few|several|\d+"
COLOR_WORDS = (r"red|blue|green|yellow|black|white|pink|purple|orange|brown|gr[ae]y|gold|golden|silver|beige|navy|"
               r"teal|violet|maroon|cyan|magenta|turquoise|crimson|ivory|cream|tan|olive|lavender|amber|bronze|"
               r"burgundy|khaki|coral|mint|lime|indigo|copper")
MATERIAL_WORDS = (r"wood|wooden|metal|metallic|steel|iron|copper|brass|leather|glass|marble|stone|plastic|fabric|denim|"
                  r"velvet|silk|wool|woolen|ceramic|porcelain|concrete|brick|rubber|paper|cardboard|bamboo|wicker|"
                  r"chrome|gold|silver|bronze|knitted|straw|linen|cotton|suede|granite|crystal")
VALUE_WORD = re.compile(rf"\b({COUNT_WORDS}|{COLOR_WORDS}|{MATERIAL_WORDS})\b", re.I)
POSITION_WORD = re.compile(r"\b(left|right|top|bottom|upper|lower|center|centre|central|middle|front|back|"
                           r"foreground|background|rear|leftmost|rightmost|topmost|first|second|third|last|nearest|"
                           r"farthest)\b", re.I)
# Scope keeps kind and granularity only (README 5.7 C); "top", "back" or "front" can be part names.
SCOPE_LOCATOR = re.compile(rf"\b(left|right|leftmost|rightmost|center|centre|central|middle|foreground|background|"
                           rf"upper|lower|nearest|farthest|first|second|third|last|{COUNT_WORDS}|{COLOR_WORDS})\b", re.I)
IMAGE_POSITION = re.compile(r"\b(positioned in|occupying|of the (image|photo|picture|frame|scene)|quadrant|"
                            r"portion of|section of the image|area of the image)\b", re.I)


def value_words(text):
    return {m.group(1).lower() for m in VALUE_WORD.finditer(text or "")}


def draft(row):
    """What the original instruction says, parsed by rules; only compared with the model's slots."""
    s = row["instruction"].strip().rstrip(".")
    route = text_route(s) if row["type"] == "text" else row["type"]
    out = {"route": route, "quoted": QUOTED.findall(s), "values": sorted(value_words(s))}
    patterns = {
        "add": r"(?i)^add\s+(?P<payload>.+?)(?:\s+(?P<relation>on top of|onto|on|inside|into|in|around|across|along|"
               r"over|under|underneath|beneath|above|below|beside|next to|near|at|to the left of|to the right of|"
               r"in front of|behind|between|to)\s+(?P<anchor>(?:the|a|an|his|her|its|their)\b.+))?$",
        "remove": r"(?i)^(?:remove|erase|delete|get rid of|take away|take out|eliminate|wipe off)\s+(?P<target>.+?)"
                  r"(?:\s+from\s+(?P<host>(?:the|his|her|its|their)\b.+))?$",
        "replace": r"(?i)^(?:replace\s+(?P<target>.+?)\s+with\s+(?P<payload>.+)|(?:change|turn|transform|convert)\s+"
                   r"(?P<target2>.+?)\s+(?:into|to)\s+(?P<payload2>.+))$",
        "attribute": r"(?i)^(?:change the (?P<property>colou?r|material|texture|pattern|finish) of\s+(?P<target>.+?)\s+"
                     r"to\s+(?P<value>.+)|(?:turn|make|change|transform|paint|recolou?r|convert|tint)\s+(?P<target2>.+?)"
                     r"\s+(?:into|to be|to|as)\s+(?P<value2>.+))$",
    }
    match = re.match(patterns[route], s) if route in patterns else None
    if match:
        found = {k.rstrip("2"): v for k, v in match.groupdict().items() if v}
        out.update(found)
    out["parsed"] = bool(match)
    return out

# ---------------------------------------------------------------- views

SLOT_PROMPT = """You convert one training example of an image-editing dataset into a structured edit record.
Image 1 is BEFORE (the original photo). Image 2 is BEFORE with labeled areas outlined. Image 3 is AFTER (the edited photo). Image 4 is a close-up around the labeled areas, BEFORE on the left and AFTER on the right.

Original instruction: "{instruction}"
Original type label: {label}
Outlines in image 2: {legend}

The example already shows an edit: describe that edit as AFTER shows it; never choose a different or better edit. Read colors from image 1, never from the outline colors.

1. observed: say in one sentence what actually changed from BEFORE to AFTER.

2. decision:
- "accept": the edit acts on ONE source object, or one part of it, with ONE of the operations below. Several new items of the same kind at one anchor are one edit; give their number.
- "multi_object": the edit acts on two or more separate objects, combines two operations, or adds new items of different kinds. Separate objects count even when one outline covers all of them: a group of people, a crowd, several animals, a cluster or row of buildings, cars, bottles or chairs, "the two flagpoles". Counted as ONE object: a pile, heap, bunch, stack or cluster of same-kind items lying together (a pile of sweet potatoes, a bunch of bananas, a stack of plates, a bouquet), and a matching pair worn or used by one person (shoes, gloves, sleeves).
- "reject": no operation below fits (moving, flipping, re-posing, style or whole-image change); AFTER does not show the instructed edit; or an added item has nothing visible to stand, rest or hang on, be attached to or be held by, and is not something that flies or floats by itself.

3. operation, judged by what AFTER shows, not by the instruction's verb:
- add: a new physical object or attached item appears; what was there stays.
- replace: an object or part becomes a different kind of object or part (its structure or category changes). A change of only color, pattern or material is not replace.
- remove: an object, a part of it, or its contents disappear.
- attribute: the object keeps its identity and shape and only an appearance property changes: color, material, finish (glossy or matte), pattern, texture, text (what written text says), tint (color of a transparent object) or opacity.
- Written text: deleting text or a logo is remove; changing what text says is attribute with property "text"; writing new text where there was none is add.

4. The source object S, which must be visible in BEFORE:
- add: S is the anchor, what the new item directly stands, sits, lies or rests on, is in, hangs from, is attached, mounted, sewn, pinned or printed on, or is worn or held by in AFTER. Name S at the finest level that has a name (granularity):
  "object": a whole object small enough to say where the item goes, e.g. a cup, a candle, a sign, a hat.
  "part": a named part of a larger object, e.g. the head or hand of a person, the seat of a sofa, the top of a cake, the top of a tower, the back of a zebra, the handle of a bag, a branch of a tree.
  "surface": a stretch of a large surface, e.g. a road, street, sidewalk, ground, floor, grass, sand, snow, water, the sea, a wall, the sky, a large table top, a counter, a shelf or a bed.
  A dog added on the road next to a man has S "the road next to the man", not the man; a boat on the sea has S "the sea ...", not a rock on the shore; a cross on a tower has S "the top of the tower", not the whole church.
  Water carries what floats on it. The sky is S only for something that flies or floats in the air by itself (a balloon, a kite, a bird in flight, an aircraft, fireworks, the moon, a rainbow); a bird on a wire has S the wire. Text or a graphic added to the picture is printed on what carries it (a wall, a sign, a poster, or the background of a poster or graphic).
  One exception: when the new item holds S up or holds it (a bench added under a sitting person, a tray under a fish, a bowl under fruit, a pedestal under a statue), S is the object held, granularity "object", relation "under", anchor_mode "reference". No other reference is used: S is never an object the new item is merely near, above, below or beside.
  S is never the new item or empty space.
- remove, replace, attribute: S is the object or part the edit acts on, normally exactly the outlined object (if the outline is a part, such as a lid or a sleeve, S is that part).
- written text: when the text itself is outlined, S is that text area; name it with a noun such as "lettering", "logo", "label", "number" or "banner", never with the word "text" (the sentence already says "the text on ..."); otherwise S is the object that carries it.
- instances: give every outlined number a role: "main" (S itself or a visible piece of it), "anchor" (for an add, the outline marks S or the object or surface S belongs to), "attached" (belongs to S and changes with it, e.g. its contents, pedestal or carried bag), "new_item" (an added or replacing item, drawn where it appears in AFTER), "separate" (another object that is edited too; then the decision is "multi_object") or "unchanged".

5. Slots. Reuse the instruction's wording where possible and never add details that neither the instruction nor AFTER shows. The sentences are rendered from these slots; follow the writing rules shared with the benchmark:
{writing_rules}

- ref: the shortest noun phrase starting with "the" that picks out S in BEFORE (see REFERENCE above), written from BEFORE only, never from the edit result. Say "on the left" or "in the center", not "of the image". For add: an object by itself ("the white candle"); a part as "the <part> of <host>" ("the top of the unicorn cake"); a stretch of a surface by the surface and one landmark ("the road next to the monk", "the sand to the right of the yellow surfboard").
- scope: the kind of S itself at the same granularity as S, with no article, color, position, owner or number, e.g. "jacket", "chair seat", "suitcase handle", "sign". For add: the object ("candle"), the part ("cake top", "sofa seat", "tower top", "hand") or the surface ("road", "sand", "sky").
- host_ref, host_scope: for add, the whole object or surface S belongs to, and its kind: "the unicorn cake" and "cake" for S "the top of the unicorn cake"; "the road" and "road" for S "the road next to the monk"; the same as ref and scope when S is a whole object. For other operations "".
- part: for an add with granularity "part", the part as it would follow "the" and precede "of": "top", "seat", "back", "handle", "left sleeve"; otherwise "".
- granularity: for add "object", "part" or "surface"; otherwise "".
- target_part: only when the outline covers a whole object but the edit acts on just a named part of it, e.g. "lid" when the whole jar is outlined and only its lid is removed; otherwise "".
- payload: for add the new item, for replace the replacement, in the instruction's own words (count, size, color, kind), e.g. "a small red identification tag"; do not add colors, materials or other details the instruction does not state. The payload must be true of AFTER: name only what is new in AFTER, and drop or correct parts of the instruction that AFTER contradicts or that already existed in BEFORE (if the oranges were already there and only a basket appears, the payload is "a basket", not "a basket of oranges"). Placement words of the instruction say where S is: they belong in the ref ("the road next to the monk"), not in the payload, and the payload never refers to S with "its" or "it". For replace, one complete noun phrase for the new object including its changed form, e.g. "a square four-hole button". For remove and attribute "".
- remove_mode: for remove one of "whole", "part" (needs target_part), "contents" (a container such as a basket, box or bag removed with what it holds), "contents_only" (only what a container holds), "lowerlegs" (trousers cut to shorts), "hem" (skirt or dress shortened); otherwise "whole". Vehicles, furniture and people are not containers.
- attributes: for attribute a list of {{"property": ..., "value": ...}} with property one of color, material, finish, pattern, texture, text, tint, opacity. For text the value is the exact new text without quotes. tint and opacity stand alone; opacity is transparent, translucent or opaque. Otherwise [].
- placement: for add {{"relation": ..., "site": ..., "anchor_mode": ...}}, otherwise {{}}. relation: how the new item meets S, true in AFTER, one of to, on, in, inside, around, across, along, over, at, attached to, clipped to, pinned to, hanging from, hooked over, resting on, mounted on, sewn onto, strapped over, tied around; "to" takes no site. Use "on" for a surface the item stands or lies on and "in" for the sky, water, sand, snow or a container it is in; "under" only for the exception above. anchor_mode: "reference" for that exception, otherwise "support". site: "" unless S is a whole object and the item goes on one spot of it (e.g. "side" for a sticker on the side of a candle); never repeat the part.
- qualifier: normally "". Only a short clause the result needs, e.g. "including its pedestal" when an attached item is removed together with S.
- s_location: where S is in BEFORE, in a few words, e.g. "left foreground".

Return only JSON:
{{"observed": "...", "decision": "accept", "reason": "<at most 15 words>", "operation": "add", "instances": [{{"n": 1, "role": "new_item"}}], "ref": "the ...", "scope": "...", "host_ref": "the ...", "host_scope": "...", "part": "", "granularity": "surface", "target_part": "", "payload": "...", "remove_mode": "whole", "attributes": [], "placement": {{"relation": "on", "site": "", "anchor_mode": "support"}}, "qualifier": "", "s_location": "..."}}"""

GROUND_PROMPT = """Locate this object in the image: {ref}{hint}.
Use a tight box enclosing its complete visible extent; for a person include head and limbs. Do not box only a part of it, and do not box a larger object it belongs to.
Coordinates are [x1,y1,x2,y2], normalized to [0,1000] in this image. Return only JSON: {{"bbox_2d": [x1, y1, x2, y2]}}
If it is not visible, return {{"bbox_2d": null, "reason": "<short reason>"}}."""

PART_GROUND_PROMPT = """Locate this part of an object in the image: {ref}{hint}.
Use a tight box around that part only, not around the whole object it belongs to.
Coordinates are [x1,y1,x2,y2], normalized to [0,1000] in this image. Return only JSON: {{"bbox_2d": [x1, y1, x2, y2]}}
If it is not visible, return {{"bbox_2d": null, "reason": "<short reason>"}}."""

TYPE_LABELS = {"text": "text (a change to written text)"}
WHERE = {"source": "labeled in BEFORE", "target": "new content, drawn where it appears in AFTER",
         "extra": "an extra area that changed"}


def legend(row):
    items = []
    for number, instance in enumerate(row["instances"], 1):
        color = PALETTE[(number - 1) % len(PALETTE)][0]
        items.append(f'[{number}] {color}: "{instance.get("ref") or "labeled region"}" '
                     f'({WHERE[instance_side(instance)]})')
    return "; ".join(items)


def slot_prompt(row):
    return SLOT_PROMPT.format(instruction=row["instruction"].strip(), label=TYPE_LABELS.get(row["type"], row["type"]),
                              legend=legend(row), writing_rules=WRITING_RULES)


def ground_prompt(ref, hint=None, part=False):
    return (PART_GROUND_PROMPT if part else GROUND_PROMPT).format(ref=ref, hint=f" ({hint})" if hint else "")


def outlined(image, masks, max_pixels, numbers=True):
    """Each mask as a colored contour with a black halo (as the edit check draws it), numbered."""
    image = ec.resized(image, max_pixels)
    width = max(2, round(max(image.size) / 400))
    labels = []
    for number, (mask, color) in enumerate(masks, 1):
        m = Image.fromarray(mask.astype(np.uint8) * 255).resize(image.size, Image.Resampling.NEAREST)
        outer, inner = m.filter(ImageFilter.MaxFilter(2 * (width + 1) + 1)), m.filter(ImageFilter.MaxFilter(2 * width + 1))
        image.paste((0, 0, 0), mask=ImageChops.subtract(outer, m))
        image.paste(color, mask=ImageChops.subtract(inner, m))
        box = m.getbbox()
        if numbers and box:
            labels.append((number, color, box))
    draw, font = ImageDraw.Draw(image), ImageFont.truetype(ec.font_path(), max(14, round(max(image.size) / 40)))
    for number, color, (x, y, _, _) in labels:
        text = str(number)
        w, h = draw.textbbox((0, 0), text, font=font)[2:]
        x, y = min(max(0, x), image.width - w - 6), max(0, y - h - 6)
        draw.rectangle((x, y, x + w + 6, y + h + 6), fill=color, outline=(0, 0, 0))
        draw.text((x + 3, y + 1), text, font=font, fill=(0, 0, 0))
    return image


def crop_masks(masks, box):
    x1, y1, x2, y2 = box
    return [(mask[y1:y2, x1:x2], color) for mask, color in masks]


def close_up(source, target, masks, box_1000, max_long=ec.CLOSE_LONG_SIDE, left="BEFORE (close-up)",
             right="AFTER (close-up)", after_masks=None):
    box = ec.crop_box(box_1000, source.size)
    before = outlined(source.crop(box), crop_masks(masks, box), 10 ** 9, numbers=False)
    after = target.convert("RGB").crop(box)
    if after_masks:
        after = outlined(after, crop_masks(after_masks, box), 10 ** 9, numbers=False)
    scale = min((max_long / 2 - 10) / max(before.width, before.height), ec.CLOSE_MAX_SCALE)
    size = (max(1, round(before.width * scale)), max(1, round(before.height * scale)))
    before = ec.labeled(before.resize(size, Image.Resampling.LANCZOS), left)
    after = ec.labeled(after.resize(size, Image.Resampling.LANCZOS), right)
    canvas = Image.new("RGB", (before.width * 2 + 20, before.height), "white")
    canvas.paste(before, (0, 0))
    canvas.paste(after, (before.width + 20, 0))
    return canvas


def open_pair(row, root):
    root = Path(root)
    source = Image.open(root / row["source_image"]).convert("RGB")
    target = Image.open(root / row["target_image"]).convert("RGB")
    if target.size != source.size:
        target = target.resize(source.size, Image.Resampling.LANCZOS)
    return source, target


def slot_parts(row, root):
    source, target = open_pair(row, root)
    masks = [(decode(i), PALETTE[k % len(PALETTE)][1]) for k, i in enumerate(row["instances"])]
    views = [("Image 1, BEFORE:", ec.resized(source, FULL_PIXELS)),
             ("Image 2, BEFORE with the labeled areas outlined:", outlined(source, masks, FULL_PIXELS)),
             ("Image 3, AFTER:", ec.resized(target, FULL_PIXELS)),
             ("Image 4, close-up, BEFORE left and AFTER right:", close_up(source, target, masks, row["region"]["box_1000"]))]
    parts = []
    for label, image in views:
        parts += [{"type": "text", "text": label}, {"type": "image_url", "image_url": {"url": ec.data_url(image)}}]
    return parts


def ground_parts(row, root, image="source_image"):
    picture = Image.open(Path(root) / row[image]).convert("RGB")
    return [{"type": "image_url", "image_url": {"url": ec.data_url(ec.resized(picture, GROUND_PIXELS))}}]

# ---------------------------------------------------------------- model calls


def parse_json(text):
    raw = text.rsplit("</think>", 1)[-1].strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    value = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
    if not isinstance(value, dict):
        raise ValueError(f"not a JSON object: {raw[:200]}")
    return value


class Client:
    """OpenAI-compatible chat endpoint; transport errors are retried, then raised."""

    def __init__(self, session, url, model, effort="low"):
        self.session, self.url, self.model, self.effort = session, url, model, effort

    async def call(self, parts, prompt, call, seed=0):
        thinking, _, max_tokens = CALLS[call]
        body = {"model": self.model, "max_tokens": max_tokens, "seed": seed,
                "messages": [{"role": "user", "content": parts + [{"type": "text", "text": prompt}]}],
                "chat_template_kwargs": {"enable_thinking": thinking,
                                         **({"reasoning_effort": self.effort} if thinking else {})},
                **(ec.THINKING if thinking else GREEDY)}
        error = None
        for attempt in range(ec.TRANSPORT_ATTEMPTS):
            try:
                async with self.session.post(self.url, json=body) as response:
                    data = await response.json(content_type=None)
                if "choices" not in data:
                    raise RuntimeError(str(data)[:500])
                choice = data["choices"][0]
                return {"text": choice["message"].get("content") or "", "finish": choice.get("finish_reason"),
                        "tokens": (data.get("usage") or {}).get("completion_tokens", 0)}
            except Exception as exc:  # noqa: BLE001 - retried, then raised below
                error = f"{type(exc).__name__}: {exc}"
                await asyncio.sleep(2 + 5 * attempt)
        raise RuntimeError(f"{call} request failed {ec.TRANSPORT_ATTEMPTS} times: {error}")

# ---------------------------------------------------------------- slots -> protocol spec

GRANULARITIES = ("object", "part", "surface")
RELATION_START = re.compile(r"(on|onto|in|into|inside|at|to|of|from|below|above|under|beside|near|next|around|across|"
                            r"along|over)\b")
HOST_PRONOUN = re.compile(r"\b(on|in|to|onto|into|around|from|at|of|over|under|across|along|inside)\s+(its|it)\b", re.I)
RELATION_ALIASES = {"onto": "on", "on top of": "on", "upon": "on", "into": "in", "within": "in", "next to": "beside",
                    "by": "beside", "alongside": "beside", "underneath": "under", "beneath": "under",
                    "attached": "attached to", "hanging on": "hanging from", "close to": "near", "nearby": "near"}
SIDE_RELATIONS = {"to the left of": "left side", "left of": "left side", "to the right of": "right side",
                  "right of": "right side"}
ROLES = {"main", "anchor", "attached", "new_item", "separate", "unchanged"}


def clean_phrase(value):
    return ip.clean(value).strip(" .,;:")


def normalize_slots(slots):
    """Deterministic repairs of near-miss slots; every repair is listed so reviews can count them."""
    fixes, s = [], dict(slots)
    op = s["operation"] = ip.clean(s.get("operation")).lower()
    s["ref"] = clean_phrase(s.get("ref"))
    if s["ref"] and not re.match(r"(?i)^(the|a|an)\b", s["ref"]):
        s["ref"] = "the " + s["ref"]
        fixes.append("ref_article")
    s["scope"] = re.sub(r"(?i)^(the|a|an|selected)\s+", "", clean_phrase(s.get("scope")))
    s["target_part"] = re.sub(r"(?i)^(the|its)\s+", "", clean_phrase(s.get("target_part")))
    s["payload"] = clean_phrase(s.get("payload")) if op in ("add", "replace") else ""
    s["qualifier"] = clean_phrase(s.get("qualifier"))
    s["remove_mode"] = (ip.clean(s.get("remove_mode")).lower() or "whole") if op == "remove" else "whole"
    if op == "remove" and s["target_part"] and s["remove_mode"] == "whole":
        s["remove_mode"] = "part"
        fixes.append("remove_mode_part")
    s["attributes"] = []
    for item in (slots.get("attributes") or []) if op == "attribute" else []:
        prop = ip.clean(item.get("property")).lower().replace("colour", "color")
        prop = {"surface texture": "texture", "transparency": "opacity"}.get(prop, prop)
        value = str(item.get("value") or "")
        value = value.strip().strip("\"'“”‘’") if prop == "text" else clean_phrase(value)
        s["attributes"].append({"property": prop, "value": value})
    s["placement"] = {}
    if op == "add":
        raw = slots.get("placement") or {}
        relation = ip.clean(raw.get("relation")).lower() or "to"
        site = re.sub(r"(?i)^(the|its)\s+", "", clean_phrase(raw.get("site")))
        if relation in SIDE_RELATIONS:
            relation, site = "beside", site or SIDE_RELATIONS[relation]
            fixes.append("side_relation")
        if relation in RELATION_ALIASES:
            relation = RELATION_ALIASES[relation]
            fixes.append("relation_alias")
        if relation == "to" and site:
            relation = "on"
            fixes.append("to_with_site")
        mode = "reference" if relation in ip.REFERENCE_RELATIONS else "support"
        if ip.clean(raw.get("anchor_mode")).lower() not in ("", mode):
            fixes.append("anchor_mode")
        s["placement"] = {"relation": relation, "anchor_mode": mode, **({"site": site} if site else {})}
        normalize_anchor(slots, s, fixes)
    else:
        s.update(granularity="", host_ref="", host_scope="", part="")
    return s, fixes


def normalize_anchor(raw, s, fixes):
    """An add's S at one of three granularities and the host it belongs to; a part's ref follows from its host."""
    gran = ip.clean(raw.get("granularity")).lower()
    host_ref = clean_phrase(raw.get("host_ref"))
    host_scope = re.sub(r"(?i)^(the|a|an)\s+", "", clean_phrase(raw.get("host_scope")))
    part = re.sub(r"(?i)^(the|its)\s+", "", clean_phrase(raw.get("part")))
    if s["placement"]["anchor_mode"] == "reference" and gran != "object":
        gran = "object"  # the exception: the object the new item holds up, as a whole
        fixes.append("reference_object")
    if gran == "object":
        host_ref, host_scope, part = s["ref"], s["scope"], ""
    elif gran in ("part", "surface"):
        host_ref = ip.noun(host_ref) if host_ref else ""
        if gran == "part" and part and host_ref:
            ref = f"{ip.noun(part)} of {host_ref}"
            if ref != s["ref"]:
                s["ref"] = ref
                fixes.append("part_ref")
            if not s["scope"] and host_scope:
                s["scope"] = f"{host_scope} {part.split()[-1]}"
        if gran == "surface":
            host_scope = host_scope or s["scope"]
            part = ""
            if s["placement"].pop("site", None):
                fixes.append("surface_site")
    s.update(granularity=gran, host_ref=host_ref, host_scope=host_scope, part=part)


def to_spec(slots):
    return ip.make_spec(slots["operation"], slots["ref"], slots["scope"], slots["payload"],
                        target_part=slots["target_part"], remove_mode=slots["remove_mode"],
                        attributes=slots["attributes"] or None, placement=slots["placement"] or None,
                        qualifier=slots["qualifier"])


def host_spec(slots, spec):
    """The host view of a part-level add: the whole host selected, the part named as the site (as the benchmark
    writes most adds: "on the top of the selected cake"). Its ref sentence equals the part view's."""
    p = spec["placement"]
    site = slots["part"] if not p.get("site") else f"{p['site']} of the {slots['part']}"
    return ip.make_spec("add", slots["host_ref"], slots["host_scope"], spec["payload"], qualifier=spec["qualifier"],
                        placement={"relation": "on" if p["relation"] == "to" else p["relation"], "site": site,
                                   "anchor_mode": p["anchor_mode"]})


def roles_of(slots, count):
    roles = {}
    for item in slots.get("instances") or []:
        try:
            number, role = int(item.get("n")), ip.clean(item.get("role")).lower()
        except (TypeError, ValueError, AttributeError):
            continue
        if 1 <= number <= count and role in ROLES:
            roles[number] = role
    return roles


def is_text_change(spec):
    return spec["operation"] == "attribute" and spec["attributes"][0]["property"] == "text"


def selection_plan(row, slots, spec):
    """Where S comes from: labeled instances reused as they are, or a new SAM3 mask."""
    instances = row["instances"]
    roles = roles_of(slots, len(instances))
    numbered = list(enumerate(instances, 1))
    if spec["operation"] == "add":
        # A labeled anchor is the host (or S itself); a part is found inside the host by SAM later.
        anchors = [n for n, i in numbered if roles.get(n) in ("anchor", "main") and instance_side(i) == "source"]
        plan = {"source": "reused", "instances": anchors} if anchors else {"source": "sam3"}
        granularity = slots.get("granularity", "")
        return {**plan, "granularity": granularity, "part": granularity == "part"}
    # A labeled source instance that marks S is kept as it is (no re-mask by the user's rule, 2026-10-12);
    # only S that has no labeled source mask, e.g. a replace labeled in AFTER only, is re-masked.
    main = [n for n, i in numbered if roles.get(n) == "main" and instance_side(i) == "source"]
    if not main:
        return {"source": "sam3"}
    return {"source": "reused" if len(main) == len(instances) else "split", "instances": main}


def slot_problems(slots):
    """Slot errors the protocol module does not catch; any of them sends the call back with feedback."""
    problems = []
    site = (slots.get("placement") or {}).get("site", "").lower()
    scope = slots.get("scope", "").lower()
    if slots.get("operation") == "add" and site and (scope == site or scope.endswith(" " + site)):
        problems.append(f'scope "{slots["scope"]}" names the spot on S; scope must name the kind of S itself')
    if IMAGE_POSITION.search(slots.get("ref", "")):
        problems.append(f'ref "{slots["ref"]}" describes the image frame; say "on the left" or "in the center" instead')
    if RELATION_START.match(site):
        problems.append(f'placement.site "{site}" starts with a relation word; site names only the spot on S, '
                        "e.g. \"seat\", and the relation goes in placement.relation")
    if slots.get("operation") == "add":
        problems += anchor_problems(slots)
    return problems


def anchor_problems(slots):
    """An add's anchor must be what the new item touches, named at a granularity the build step can segment."""
    problems = []
    gran, relation = slots.get("granularity"), slots["placement"].get("relation", "to")
    if gran not in GRANULARITIES:
        problems.append('granularity must be "object", "part" or "surface"')
    if gran in ("part", "surface") and not slots.get("host_ref"):
        problems.append("host_ref must name the whole object or surface S belongs to")
    if gran == "part" and (not slots.get("part") or RELATION_START.match(slots["part"].lower())):
        problems.append('part must name the part of the host that S is, e.g. "top" or "seat", without a relation word')
    if relation in ("near", "above", "below", "beside"):
        problems.append(f'relation "{relation}" is not used: S is what the new item stands, rests or hangs on (an '
                        "object, a part of one, or a stretch of a surface such as the ground, floor, water or table "
                        'top); "under" only when the new item holds S up; if nothing visible carries the item, decide '
                        '"reject"')
    if re.search(r"\bsky\b", slots.get("host_scope", "") + " " + slots.get("scope", ""), re.I) and \
            relation not in ("in", "across"):
        problems.append('for the sky the relation is "in" (or "across" for a rainbow)')
    if HOST_PRONOUN.search(slots.get("payload", "")):
        problems.append(f'payload "{slots["payload"]}" refers to S with its/it; the payload names only the new item')
    return problems


def part_as_s(spec):
    """A named part of S becomes S itself: the labels mark whole objects, so the part is re-masked."""
    part = ip.noun(spec["target_part"])
    head = part.split()[-1].lower()
    ref = spec["ref"] if re.search(rf"\b{re.escape(head)}s?\b", spec["ref"], re.I) else f"{part} of {spec['ref']}"
    return ip.make_spec(spec["operation"], ref, re.sub(r"^the\s+", "", part), spec["payload"],
                        attributes=spec["attributes"] or None, qualifier=spec["qualifier"],
                        remove_mode="whole" if spec["remove_mode"] == "part" else spec["remove_mode"])


def sam_prompt(spec, host_ref=""):
    """SAM3's text prompt for S: the ref without its article. For add it is the whole object or surface S belongs
    to (host_ref, "left tower", "road"): SAM3 segments whole objects reliably; the part is then found with point and
    box prompts inside it (part_candidates), and a large anchor is cut to the spot (localize_anchor)."""
    text = (host_ref or spec["scope"]) if spec["operation"] == "add" else spec["ref"]
    return re.sub(r"(?i)^(the|a|an)\s+", "", text)


async def extract(client, row, parts, ground, feedback=None, ground_after=None):
    """Slots (re-asked with the problem stated until they form a valid spec), then the grounding calls.

    feedback: {"problems": reviewer text, "remask": bool} from a failed audit; the slots are asked
    again with the reviewer's words, and a remask forces a new SAM3 mask for S.
    ground_after: AFTER for the grounding call; for a Derived add (its label marks the host, not the new item)
    the new item is boxed there, so that the pixels the generator also changed elsewhere do not count as the item.
    """
    record = {"id": row["id"], "attempts": [], "tokens": 0, **({"feedback": feedback} if feedback else {})}
    slots = spec = None
    base = slot_prompt(row)
    if feedback:
        base += ("\n\nA reviewer checked an earlier answer for this example and found: "
                 f"{feedback['problems']}. Answer again and fix this; the operation, S, relation and wording must "
                 "match what AFTER shows.")
    prompt = base
    for seed in range(SLOT_ATTEMPTS):
        response = await client.call(parts, prompt, "slots", seed=seed)
        record["tokens"] += response["tokens"]
        try:
            raw = parse_json(response["text"])
            decision = ip.clean(raw.get("decision")).lower()
            if decision not in ("accept", "multi_object", "reject"):
                raise ValueError(f"unknown decision {decision!r}")
            normalized, fixes = normalize_slots(raw)
            if decision == "accept":
                problems = slot_problems(normalized)
                if problems:
                    raise ValueError("; ".join(problems))
                spec = to_spec(normalized)
            slots = {**normalized, "decision": decision, "fixes": fixes}
            record["attempts"].append({"seed": seed, "ok": True})
            break
        except Exception as exc:  # noqa: BLE001 - malformed or invalid slots: ask again, saying what was wrong
            error = f"{type(exc).__name__}: {exc}"[:300]
            record["attempts"].append({"seed": seed, "ok": False, "error": error,
                                       "finish": response["finish"], "text": response["text"][-1500:]})
            prompt = base + f"\n\nYour previous answer was not usable ({error}). Answer again, fixing it."
    if spec is not None and spec["target_part"] and spec["operation"] != "add":
        spec = part_as_s(spec)
        slots["fixes"] = slots.get("fixes", []) + ["part_as_s"]
    record["slots"], record["spec"] = slots, spec
    if spec is None:
        return record
    plan = selection_plan(row, slots, spec)
    if "part_as_s" in slots["fixes"] or (feedback or {}).get("remask"):
        plan = {"source": "sam3", **{k: v for k, v in plan.items() if k in ("granularity", "part")}}
    record["plan"] = plan
    add = spec["operation"] == "add"
    item_hint = ""
    if add and row["dataset"] != "derived":
        # The cleaned region marks the new item: the host box must reach that spot, and the part is the one there.
        item_hint = "the new item will be at about [{}, {}, {}, {}] (0-1000 coordinates)".format(*row["region"]["box_1000"])
    calls = [("blind", ground_prompt(spec["ref"], part=plan.get("part", False)), ground)]
    if add and row["dataset"] == "derived" and ground_after:
        item = re.sub(r"(?i)^(the|a|an|one)\s+", "", spec["payload"])
        calls.append(("item", ground_prompt(f"the {item}", f"{spec['placement'].get('relation', 'to')} {spec['ref']}"),
                      ground_after))
    if plan["source"] == "sam3":
        hint = "; ".join(x for x in (slots.get("s_location"),
                                     item_hint and item_hint + ", so the box must include the part of it there") if x)
        if (feedback or {}).get("remask"):
            hint += f"; a previous selection was wrong: {feedback['problems']}"
        # An add anchor is segmented whole (object or surface): its part is found inside it, a large one cut later.
        calls.append(("hinted", ground_prompt(slots["host_ref"] if add else spec["ref"], hint.strip("; ")), ground))
        record["sam_prompt"] = sam_prompt(spec, slots.get("host_ref", ""))
    if plan.get("part"):
        calls.append(("part", None, ground))  # its hint needs the item box of a Derived add, asked above
    for name, prompt, image in calls:
        if name == "part":
            item_box = (record.get("item") or {}).get("bbox_2d")
            near = item_hint or ("the new item will be at about [{}, {}, {}, {}] (0-1000 coordinates)".format(*item_box)
                                 if item_box else "")
            prompt = ground_prompt(spec["ref"], near and near + "; box the part at that spot", part=True)
        response = await client.call(image, prompt, "ground")
        record["tokens"] += response["tokens"]
        try:
            box = parse_json(response["text"]).get("bbox_2d")
            box = [int(round(float(v))) for v in box] if box is not None else None
            if box is not None and (len(box) != 4 or not (0 <= box[0] < box[2] <= 1000 and 0 <= box[1] < box[3] <= 1000)):
                raise ValueError(f"bad box {box}")
            record[name] = {"bbox_2d": box}
        except Exception as exc:  # noqa: BLE001 - recorded; the build step decides
            record[name] = {"bbox_2d": None, "error": f"{type(exc).__name__}: {exc}"[:200], "text": response["text"][-300:]}
    return record

# ---------------------------------------------------------------- audit

AUDIT_PROMPT = """You check one rewritten training example of an image-editing dataset.
Image 1 is BEFORE with the selected source object S outlined in green{orange}. Image 2 is AFTER. Image 3 is a close-up, BEFORE with the outlines on the left and AFTER on the right.

Instruction with a reference: "{ref}"
The same instruction without the reference: "{noref}"
Edit type: {op}. {role}

Judge each point "pass" or "fail":
1. edit: the instruction describes what actually changed from BEFORE to AFTER: the right operation, the right object, the requested new item or value (its kind, count, color, material or exact text), with nothing important wrong or missing. Check every phrase: anything the instruction says to add must be new in AFTER (not already in BEFORE), and every placement phrase (on, next to, near, under, ...) must be true in AFTER. Changes that only follow from the edit (shadows, revealed background, a carried bag removed with a person) are fine.
2. relation: {relation_rule}
3. target: {target_rule}
4. outline: {outline_rule}
5. noref: using the green outline as the selection, the instruction without the reference asks for the same edit, and its "the selected ..." noun fits the outlined object.
6. wording: both instructions are clear, natural, concise English, with no word that neither identifies the target nor defines the result.

Return only JSON: {{"edit": "pass", "relation": "pass", "target": "pass", "outline": "pass", "noref": "pass", "wording": "pass", "problems": "<at most 25 words; empty if everything passes>"}}"""
AUDIT_KEYS = ("edit", "relation", "target", "outline", "noref", "wording")
HOST_SUFFIX = "#host"  # audit id of a record's host view
AUDIT_RULES = {
    "add": {
        "relation": "the stated relation of the new item to S ({relation}) is true in AFTER, and the new item touches "
                    "S: it stands, rests, hangs or is attached on it, holds it up, or flies or floats inside the "
                    "outlined sky or water. S across open space from the new item fails.",
        "target": "the green outline is on what the new item directly stands, rests, hangs, is attached or printed "
                  "on in AFTER (for something that flies or floats, the sky or water around it; for a new item that "
                  "holds S up, the object it holds), and it is what the instruction's target phrase names. A "
                  "different object, or one with open space between it and the new item, fails.",
        "outline": "the green outline is one compact area of that object, part or surface at the spot where the new "
                   "item goes; it may be only the part of a larger object or surface around that spot. It fails when "
                   "it covers a large whole object or surface of which the new item takes only a small spot, marks a "
                   "place away from the new item, takes in other objects, or traces marks, text or specks instead of "
                   "the object or surface."},
    "add_host": {
        "relation": "the stated relation of the new item to S ({relation}) is true in AFTER, at the spot of S the "
                    "instruction names, and the new item touches S there.",
        "target": "the green outline marks the whole object that the instruction's target phrase names (the "
                  "instruction then names the spot on it, e.g. \"the top of\"), and the new item is at that spot in "
                  "AFTER.",
        "outline": "the green outline covers the visible extent of that object, with no large missing piece and no "
                   "large area of something else."},
    "other": {
        "relation": 'answer "pass" (only for add).',
        "target": "the green outline marks the object or part that the instruction's target phrase names, not a "
                  "different instance and not a much larger or smaller thing.",
        "outline": "the green outline covers the visible extent of that object or part, with no large missing piece "
                   "and no large area of something else."},
}


def audit_view(record, view="canonical"):
    """The record as the audit sees one view; the host view of a part-level add has its own id, S and sentences."""
    if view == "canonical":
        return record
    hv = record["host_view"]
    return {**record, "id": record["id"] + HOST_SUFFIX, "view": "host", "selection": hv["selection"],
            **{k: hv[k] for k in ("instruction_spec", "instruction_ref", "instruction_noref")}}


def audit_prompt(record):
    spec = record["instruction_spec"]
    add = spec["operation"] == "add"
    host = record.get("view") == "host"
    if host:
        role = ("For add, S is the whole existing object the new item goes on, and the instruction names the spot on "
                "it. The new item does not exist in BEFORE.")
    elif add:
        role = ("For add, S is the existing object, part or surface at the spot where the new item goes: what it "
                "stands, rests or hangs on, is attached or printed on, or the sky or water it flies or floats in; the "
                "outline may be just the part of a larger object or surface around that spot. The new item does not "
                "exist in BEFORE.")
    else:
        role = "S is the object or part the edit acts on."
    rules = AUDIT_RULES["add_host" if host else "add" if add else "other"]
    orange = (", and, where it differs from S, the area that changes outlined in orange"
              if record.get("execution_region") else "")
    return AUDIT_PROMPT.format(orange=orange, ref=record["instruction_ref"], noref=record["instruction_noref"],
                               op=spec["operation"], role=role, target_rule=rules["target"],
                               outline_rule=rules["outline"],
                               relation_rule=rules["relation"].format(relation=spec["placement"].get("relation", "to"))
                               if add else rules["relation"])


def audit_parts(row, record, root):
    source, target = open_pair(row, root)
    s_mask, masks = decode(record["selection"]), []
    if record.get("execution_region"):
        exec_mask = decode(record["execution_region"])
        if (exec_mask & s_mask).sum() < 0.95 * (exec_mask | s_mask).sum():  # drawn only where it differs from S
            masks.append((exec_mask, EXEC_COLOR))
    masks.append((s_mask, S_COLOR))  # last, so the green outline is never hidden
    boxes = [b for b in (record["selection"].get("box_1000"), (record.get("execution_region") or {}).get("box_1000")) if b]
    box = [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]
    views = [("Image 1, BEFORE with outlines:", outlined(source, masks, FULL_PIXELS, numbers=False)),
             ("Image 2, AFTER:", ec.resized(target, FULL_PIXELS)),
             ("Image 3, close-up, BEFORE left and AFTER right:", close_up(source, target, masks, box))]
    parts = []
    for label, image in views:
        parts += [{"type": "text", "text": label}, {"type": "image_url", "image_url": {"url": ec.data_url(image)}}]
    return parts


async def audit(client, record, parts, votes=1):
    """Independent pass/fail check of one accepted record; with several votes every vote must pass."""
    results = [await audit_vote(client, record, parts, vote) for vote in range(votes)]
    if votes == 1:
        return results[0]
    merged = {"id": record["id"], "tokens": sum(r["tokens"] for r in results), "votes": results}
    for k in AUDIT_KEYS:
        values = [r.get(k) for r in results]
        merged[k] = "fail" if "fail" in values else "pass" if all(v == "pass" for v in values) else None
    merged["passed"] = None if any(r["passed"] is None for r in results) else all(r["passed"] for r in results)
    merged["problems"] = " | ".join(r.get("problems") or "" for r in results if r.get("problems"))
    return merged


async def audit_vote(client, record, parts, vote=0):
    """One audit call, retried with new seeds while unparseable; vote k uses its own seeds."""
    out = {"id": record["id"], "tokens": 0}
    for attempt in range(SLOT_ATTEMPTS):
        seed = 100 * vote + attempt
        response = await client.call(parts, audit_prompt(record), "audit", seed=seed)
        out["tokens"] += response["tokens"]
        try:
            value = parse_json(response["text"])
            verdicts = {k: ip.clean(value.get(k)).lower() for k in AUDIT_KEYS}
            if any(v not in ("pass", "fail") for v in verdicts.values()):
                raise ValueError(f"bad verdicts {verdicts}")
            return {**out, **verdicts, "problems": ip.clean(value.get("problems")),
                    "passed": all(v == "pass" for v in verdicts.values()), "attempts": attempt + 1}
        except Exception as exc:  # noqa: BLE001 - next seed
            out["error"] = f"{type(exc).__name__}: {exc}"[:300]
    return {**out, "passed": None, "attempts": SLOT_ATTEMPTS}

# ---------------------------------------------------------------- checks


def box_iou(a, b):
    if not a or not b:
        return 0.0
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area if area else 0.0


def box_holds_center(outer, inner, grow=0.1):
    """Whether the center of box `inner` lies in box `outer` grown by 10 % of its size on each side."""
    if not outer or not inner:
        return False
    mx, my = grow * (outer[2] - outer[0]), grow * (outer[3] - outer[1])
    cx, cy = (inner[0] + inner[2]) / 2, (inner[1] + inner[3]) / 2
    return outer[0] - mx <= cx <= outer[2] + mx and outer[1] - my <= cy <= outer[3] + my


def dilate(mask, radius):
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    return np.asarray(image.filter(ImageFilter.MaxFilter(2 * radius + 1))) > 0


def geometry(spec, s_mask, exec_mask, host_scope=""):
    """How the execution region (for add: the new item) sits relative to S; for an add resting on or hanging
    from S also whether S is under (above) the item's center (support_probe)."""
    if exec_mask is None or not exec_mask.any() or not s_mask.any():
        return {}
    out = geometry_core(s_mask, exec_mask)
    if spec["operation"] == "add":
        probe = support_probe(exec_mask, spec["placement"].get("relation", "to"), host_scope)
        if probe is not None:
            out["s_in_probe"] = bool((probe & s_mask).any())
    return out


def geometry_core(s_mask, exec_mask):
    """Overlap, contact, offset and distances between S and the execution region."""
    h, w = s_mask.shape
    exec_area, s_area = exec_mask.sum(), s_mask.sum()
    near = dilate(s_mask, max(3, round(0.01 * max(h, w))))
    sy, sx = np.nonzero(s_mask)
    ey, ex = np.nonzero(exec_mask)
    sb, eb = box_1000(s_mask), box_1000(exec_mask)
    gap = max(0, max(sb[0], eb[0]) - min(sb[2], eb[2])) + max(0, max(sb[1], eb[1]) - min(sb[3], eb[3]))
    return {"exec_in_s": round(float((exec_mask & s_mask).sum() / exec_area), 3),
            "s_in_exec": round(float((exec_mask & s_mask).sum() / s_area), 3),
            "exec_touch_s": round(float((exec_mask & near).sum() / exec_area), 3),
            "dx": round(float(ex.mean() - sx.mean()) / w, 3), "dy": round(float(ey.mean() - sy.mean()) / h, 3),
            "box_gap_1000": int(gap), "gap_frac": round(mask_gap(exec_mask, s_mask) / (h * h + w * w) ** 0.5, 4)}


def mask_gap(a, b):
    """Shortest distance in pixels between two non-empty masks; 0 when they touch or overlap."""
    import cv2
    return float(cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2, 5)[a].min())

# ---------------------------------------------------------------- add anchors

ANCHOR_METHOD = "anchor-window-1"
# An anchor already says where the item goes, and stays whole, when it reaches no further than 4x the item's extent
# (box diagonal) and is either at most 3x the item's area and 5 % of the image, or at most 1 % of the image (the
# side of a candle for a sticker). Otherwise (a road, a cake, a long roof edge) the spot is S inside an ellipse
# around the new item's box (its pieces of at least a quarter of the largest one), grown on every side by 0.35x the
# item's mean side (at least 1.5 % of the image diagonal), and widened 1.5x at a time until it holds half the item's
# area (at least 0.2 % of the image) so that a thin or small item still gets a usable spot. An anchor the window
# holds almost whole (85 %) stays whole; of a cut, pieces under a quarter of the main one are dropped. Reference
# anchors must lie within 1.5 % of the diagonal of the item (touching or right against it); a support with contact
# (on, in, attached to, ...) more than 3 % away is not one.
ANCHOR_PARAMS = {"local_extent_k": 4.0, "local_k": 3.0, "local_frac": 0.05, "small_frac": 0.01, "piece_frac": 0.25,
                 "margin_k": 0.35, "margin_min_frac": 0.015, "grow": 1.5, "steps": 6, "min_area_k": 0.5,
                 "min_area_frac": 0.002, "keep_whole": 0.85, "piece_keep": 0.25, "adjacent_frac": 0.015,
                 "apart_frac": 0.03}
CONTACT_RELATIONS = {"to", "on", "in", "inside", "at", "around", "attached to", "clipped to", "resting on",
                     "mounted on", "sewn onto", "tied around"}


def add_footprint(exec_mask, s_mask, images, item_box=None):
    """Where the new item appears: the cleaned region, else (Derived adds label the host) the pixels that changed,
    kept to the model's box around the new item in AFTER (grown by 10 %) when there is one."""
    if exec_mask is not None and exec_mask.any():
        return exec_mask, "execution_region"
    if images is None:
        return None, None
    region, _ = edited_region(*images, s_mask, None)
    if item_box:
        h, w = s_mask.shape
        mx, my = 0.1 * (item_box[2] - item_box[0]), 0.1 * (item_box[3] - item_box[1])
        box = np.zeros((h, w), bool)
        box[max(0, int((item_box[1] - my) * h / 1000)):int(np.ceil((item_box[3] + my) * h / 1000)),
            max(0, int((item_box[0] - mx) * w / 1000)):int(np.ceil((item_box[2] + mx) * w / 1000))] = True
        if region is not None and (region & box).sum() >= 0.05 * box.sum():
            return region & box, "edited_region_in_item_box"
        return box, "item_box"
    return (region, "edited_region") if region is not None else (None, None)


def box_extent(mask):
    """Diagonal of the mask's box in pixels."""
    ys, xs = np.nonzero(mask)
    return float(np.hypot(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))


def localize_anchor(s_mask, footprint, params=ANCHOR_PARAMS):
    """The part of the anchor S around the new item's footprint, so that S says where on it the item goes.

    Returns (mask, info). The mask is S itself when S is already local or the window holds nearly all of it, and
    None when no part of S is near the item at all.
    """
    import cv2
    p = params
    h, w = s_mask.shape
    # The item's main pieces only: a changed area elsewhere (a regenerated cushion) must not widen the window.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(footprint.astype(np.uint8), connectivity=8)
    area = stats[:, cv2.CC_STAT_AREA]
    main = area >= p["piece_frac"] * area[1:].max()
    main[0] = False
    footprint = main[labels]
    ys, xs = np.nonzero(footprint)
    cx, cy = (xs.min() + xs.max() + 1) / 2, (ys.min() + ys.max() + 1) / 2
    a0, b0 = (xs.max() + 1 - xs.min()) / 2, (ys.max() + 1 - ys.min()) / 2
    margin = max(p["margin_k"] * 2 * (a0 * b0) ** 0.5, p["margin_min_frac"] * (h * h + w * w) ** 0.5)
    s_area = int(s_mask.sum())
    target = min(max(p["min_area_k"] * footprint.sum(), p["min_area_frac"] * h * w), p["keep_whole"] * s_area)
    yy, xx = np.ogrid[:h, :w]
    for step in range(p["steps"]):
        local = s_mask & ((((xx - cx) / (a0 + margin)) ** 2 + ((yy - cy) / (b0 + margin)) ** 2) <= 1)
        if local.sum() >= target or step == p["steps"] - 1:
            break
        margin *= p["grow"]
    info = {"method": ANCHOR_METHOD, "margin_px": int(round(margin)), "full_area_frac": round(s_area / (h * w), 5)}
    if not local.any():
        return None, {**info, "cut": False}
    already_local = box_extent(s_mask) <= p["local_extent_k"] * box_extent(footprint) and (
        (s_area <= p["local_k"] * footprint.sum() and s_area <= p["local_frac"] * h * w)
        or s_area <= p["small_frac"] * h * w)
    if already_local or local.sum() >= p["keep_whole"] * s_area:
        return s_mask, {**info, "cut": False}
    local = main_pieces(local, p["piece_keep"])
    return local, {**info, "cut": True, "area_frac": round(float(local.sum()) / (h * w), 5)}


def main_pieces(mask, keep_frac=ANCHOR_PARAMS["piece_keep"]):
    """One compact region: the main piece, and pieces of at least `keep_frac` of it (a surface split by an object
    on it); stray bits cut off a ragged edge or another stretch of the surface are dropped."""
    import cv2
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 2:
        return mask
    area = stats[:, cv2.CC_STAT_AREA]
    keep = area >= keep_frac * area[1:].max()
    keep[0] = False
    return keep[labels]


# A part is one of SAM's masks for the contact point and the model's part box (protocol_remask.py): mostly inside
# the host (85 %), at most 70 % of it (else it is the host), at least 0.3x the new item and 0.1 % of the image (else a
# speck), within 3 % of the diagonal of the item, and not a near-duplicate (IoU 0.85) of a smaller one. The model
# picks among these outlined and numbered (pick_part); without its pick, the one holding the contact point (1 % of
# the diagonal) that best matches the part box wins. A label whose box matches the part box (IoU 0.5) is the part.
PART_PARAMS = {"inside": 0.85, "max_host_frac": 0.7, "min_item_k": 0.3, "min_area_frac": 0.001, "touch_frac": 0.03,
               "same_iou": 0.85, "contact_frac": 0.01, "label_part_iou": 0.5, "score_k": 0.2}


# An item that rests on S has S under its center; one that hangs from S has S above it. The probe is the middle 40 %
# of the item's width, from its center down to its bottom (up to its top) and 3 % of the diagonal beyond. Items held,
# contained, attached, sewn, wrapped or flying meet S anywhere ("in the hand" is not "on").
BASE_RELATIONS = {"on", "resting on"}
TOP_RELATIONS = {"hanging from", "hooked over"}
PROBE_PARAMS = {"column_frac": 0.4, "beyond_frac": 0.03}


def support_probe(footprint, relation="", host_scope="", params=PROBE_PARAMS):
    """The region where S must be for the relation (see BASE_RELATIONS), or None when it has no direction."""
    if relation not in BASE_RELATIONS | TOP_RELATIONS or re.search(r"\bsky\b", host_scope or ""):
        return None
    h, w = footprint.shape
    ys, xs = np.nonzero(footprint)
    cx, cy = int(round(xs.mean())), int(round(ys.mean()))
    half = max(1, round(params["column_frac"] * (xs.max() + 1 - xs.min()) / 2))
    beyond = round(params["beyond_frac"] * (h * h + w * w) ** 0.5)
    probe = np.zeros((h, w), bool)
    if relation in TOP_RELATIONS:
        probe[max(0, ys.min() - beyond):cy + 1, max(0, cx - half):cx + half + 1] = True
    else:
        probe[cy:ys.max() + 1 + beyond, max(0, cx - half):cx + half + 1] = True
    return probe


def contact_point(host, footprint, relation="", host_scope=""):
    """Pixel [x, y] of the host where the new item meets it: under (above) its center for an item resting on
    (hanging from) the host, the host pixel nearest the center and a little inside; otherwise the deepest host pixel
    inside the item, or inside a margin around it (1.5, 3, then 6 % of the diagonal) when it only touches the host."""
    import cv2
    h, w = host.shape
    diag = (h * h + w * w) ** 0.5
    depth = cv2.distanceTransform(host.astype(np.uint8), cv2.DIST_L2, 5)
    probe = support_probe(footprint, relation, host_scope)
    if probe is not None and (host & probe).any():
        zone = host & probe
        ys = np.nonzero(zone)[0]
        span = max(2, round(0.1 * (np.ptp(np.nonzero(footprint)[0]) + 1)))  # a tenth of the item's height
        lo, hi = (ys.max() - span, ys.max()) if relation in TOP_RELATIONS else (ys.min(), ys.min() + span)
        rows = np.zeros(h, bool)
        rows[max(0, lo):hi + 1] = True  # the host rows nearest the item's center
        zone &= rows[:, None]
    else:
        zone = host & footprint
        for frac in (0.015, 0.03, 0.06):
            if zone.any():
                break
            zone = host & grow(footprint, frac * diag)
    if not zone.any():
        return None
    depth[~zone] = -1
    y, x = np.unravel_index(int(np.argmax(depth)), depth.shape)
    return [int(x), int(y)]


def part_shortlist(candidates, host, footprint, params=PART_PARAMS):
    """SAM's candidates that can be the part (see PART_PARAMS), each inside the host and without near-duplicates
    (IoU 0.85), smallest first; [(index, mask)] and the reason each other candidate was dropped."""
    p = params
    h, w = host.shape
    diag = (h * h + w * w) ** 0.5
    host_area, floor = int(host.sum()), max(p["min_item_k"] * footprint.sum(), p["min_area_frac"] * h * w)
    kept, dropped = [], {}
    for index, c in enumerate(candidates):
        raw = decode(c)
        m = raw & host
        area = int(m.sum())
        why = ("empty" if not area else "outside_host" if area < p["inside"] * raw.sum()
               else "whole_host" if area > p["max_host_frac"] * host_area else "speck" if area < floor
               else "away_from_item" if mask_gap(footprint, m) > p["touch_frac"] * diag else None)
        if why is None and any((m & k).sum() >= p["same_iou"] * (m | k).sum() for _, k in kept):
            why = "duplicate"
        if why:
            dropped[index] = why
        else:
            kept.append((index, m))
    return sorted(kept, key=lambda item: int(item[1].sum())), dropped


def choose_part(candidates, host, footprint, part_box=None, point=None, pick=None, params=PART_PARAMS):
    """The part of the host at the new item; (mask or None, audit). The model's pick among the shortlist decides
    (pick_part); without one, the shortlisted candidate that holds the contact point and best matches the part box."""
    p = params
    h, w = host.shape
    shortlist, dropped = part_shortlist(candidates, host, footprint, p)
    audit = {"source": "sam_candidates", "shortlist": [i for i, _ in shortlist], "dropped": dropped}
    if pick is not None:
        choice = pick.get("choice")
        audit.update(source="model_pick", pick=pick)
        if not choice or choice > len(shortlist) or pick.get("shortlist") != audit["shortlist"]:
            return None, audit
        index, mask = shortlist[choice - 1]
        audit["chosen"] = {"index": index, "prompt": candidates[index].get("prompt"), "k": candidates[index].get("k")}
        return main_pieces(mask), audit
    probe = None
    if point is not None:
        probe = np.zeros((h, w), bool)
        probe[min(max(point[1], 0), h - 1), min(max(point[0], 0), w - 1)] = True
    best = None
    for index, mask in shortlist:
        if probe is not None and mask_gap(probe, mask) > p["contact_frac"] * (h * h + w * w) ** 0.5:
            dropped[index] = "not_at_contact"
            continue
        score = (box_iou(box_1000(mask), part_box) if part_box else 0.0) + p["score_k"] * float(
            candidates[index].get("score", 0))
        if best is None or score > best[0]:
            best = (score, index, mask)
    if best is None:
        return None, audit
    audit["chosen"] = {"index": best[1], "prompt": candidates[best[1]].get("prompt"), "k": candidates[best[1]].get("k"),
                       "score": round(best[0], 4)}
    return main_pieces(best[2]), audit


PART_PICK_PROMPT = """Image 1 is BEFORE with numbered outlines of candidate regions. Image 2 is AFTER. Image 3 is a close-up, BEFORE with the outlines on the left and AFTER on the right.
In AFTER, {payload} has been added {relation} {ref}.
Which numbered outline marks {ref}: the part of {host_ref} that the new item {relation} in AFTER, and only that part? An outline that marks a different part (for example a piece behind or beside the new item), the whole {host_scope}, or something else does not fit.
Return only JSON: {{"choice": <number>, "reason": "<at most 12 words>"}}; the choice is 0 when no outline fits."""


def part_pick_inputs(row, extracted, remasked, root):
    """The shortlist of a part-level add (candidate indices) and the images that show it numbered; ([], None)
    when nothing is left to pick from."""
    plan = extracted["plan"]
    if plan["source"] == "sam3":
        host = decode(remasked["mask"]) if remasked.get("mask") else None
    else:
        host = decode(union([row["instances"][n - 1] for n in plan["instances"]]))
    candidates = (remasked.get("part") or {}).get("candidates") or []
    if host is None or not candidates:
        return [], None
    images = open_pair(row, root)
    exec_mask = decode(row["region"]) if row["dataset"] != "derived" else None
    footprint, _ = add_footprint(exec_mask, host, images, (extracted.get("item") or {}).get("bbox_2d"))
    if footprint is None:
        return [], None
    shortlist, _ = part_shortlist(candidates, host, footprint)
    return [i for i, _ in shortlist], (pick_parts(images, shortlist) if shortlist else None)


def pick_parts(images, shortlist):
    """Images for pick_part: the shortlisted part candidates outlined and numbered."""
    source, target = images
    masks = [(m, PALETTE[k % len(PALETTE)][1]) for k, (_, m) in enumerate(shortlist)]
    box = box_1000(np.logical_or.reduce([m for _, m in shortlist]))
    views = [("Image 1, BEFORE with numbered outlines:", outlined(source, masks, FULL_PIXELS)),
             ("Image 2, AFTER:", ec.resized(target, FULL_PIXELS)),
             ("Image 3, close-up, BEFORE left and AFTER right:", close_up(source, target, masks, box))]
    parts = []
    for label, image in views:
        parts += [{"type": "text", "text": label}, {"type": "image_url", "image_url": {"url": ec.data_url(image)}}]
    return parts


async def pick_part(client, extracted, parts, count):
    """The model's choice of the part among `count` numbered outlines; retried with new seeds while unusable."""
    spec, slots = extracted["spec"], extracted["slots"]
    prompt = PART_PICK_PROMPT.format(payload=spec["payload"], relation=spec["placement"].get("relation", "on"),
                                     ref=spec["ref"], host_ref=slots["host_ref"], host_scope=slots["host_scope"])
    out = {"tokens": 0}
    for seed in range(SLOT_ATTEMPTS):
        response = await client.call(parts, prompt, "audit", seed=seed)
        out["tokens"] += response["tokens"]
        try:
            choice = int(parse_json(response["text"]).get("choice"))
            if not 0 <= choice <= count:
                raise ValueError(f"choice {choice} of {count}")
            return {**out, "choice": choice, "reason": ip.clean(parse_json(response["text"]).get("reason")),
                    "attempts": seed + 1}
        except Exception as exc:  # noqa: BLE001 - next seed
            out["error"] = f"{type(exc).__name__}: {exc}"[:200]
    return {**out, "choice": None, "attempts": SLOT_ATTEMPTS}


def add_anchor(slots, spec, plan, host_sel, footprint, kind, remasked, extracted):
    """S of an add at its granularity: the object, its part at the new item, or the stretch of a surface there; a
    large one is cut to the spot (localize_anchor). Returns (selection, host_view, flags, hard): a part-level add
    also gets the host view, the whole host selected and the part named as the site."""
    gran, host = slots["granularity"], decode(host_sel)
    full = {k: host_sel[k] for k in ("rle_size", "rle_counts", "box_1000")}
    meta = {k: v for k, v in host_sel.items() if k not in full}
    info, flags = {"footprint": kind}, []
    if footprint is None:
        return {**host_sel, "granularity": gran, "anchor": info}, None, ["anchor_not_localized"], []
    base, host_known = host, True
    if gran == "part":
        part_box = (extracted.get("part") or {}).get("bbox_2d")
        if plan["source"] != "sam3" and part_box and box_iou(full["box_1000"], part_box) >= PART_PARAMS["label_part_iou"]:
            host_known = False  # the labeled anchor is the part itself; its host is not segmented
            info["part"] = {"source": "label"}
            flags.append("label_is_part")
        else:
            found = (remasked or {}).get("part") or {}
            part, info["part"] = choose_part(found.get("candidates", []), host, footprint, part_box,
                                             found.get("point"), found.get("pick"))
            if part is None:
                flags.append("part_not_found")  # S is cut from the host instead
            else:
                base = part
    if spec["placement"]["anchor_mode"] == "reference":
        local, cut = base, {"cut": False}  # the object the new item holds up, as a whole
    else:
        local, cut = localize_anchor(base, footprint)
    if local is None:
        return {**host_sel, "granularity": gran, "anchor": info}, None, flags, ["anchor_not_at_item"]
    info.update(cut)
    selection = {**meta, **(encode(local) if local is not host else full), "granularity": gran, "anchor": info}
    if base is not host:
        selection["source"] = "sam3_part"
        selection["host_source"] = host_sel["source"]
    if local is not host:
        selection["host"] = full
    if gran == "object" and cut.get("cut"):
        flags.append("object_cut")
    host_view = None
    if gran == "part" and host_known:
        hs = host_spec(slots, spec)
        host_view = {"instruction_spec": hs, "instruction_ref": ip.render(hs, True),
                     "instruction_noref": ip.render(hs, False), "mask_semantics": ip.mask_contract(hs),
                     "selection": dict(host_sel)}
    return selection, host_view, flags, []


def text_flags(row, spec, ref, noref, original):
    """Values the original asked for must survive; values the original never mentioned are suspicious."""
    flags = []
    new_values = {spec["payload"].lower()} | {a["value"].lower() for a in spec["attributes"]}
    if is_text_change(spec):
        quoted = original["quoted"]
        if quoted and spec["attributes"][0]["value"] not in quoted:
            flags.append("text_value_not_quoted_in_original")
    asked = value_words(original.get("payload", "") + " " + original.get("value", ""))
    lost = sorted(asked - value_words(ref))
    if lost:
        flags.append("lost_values:" + ",".join(lost))
    invented = sorted(value_words(" ".join(new_values)) - value_words(row["instruction"]) - {"one", "single"})
    if invented:
        flags.append("new_values:" + ",".join(invented))
    if SCOPE_LOCATOR.search(spec["scope"]) or re.search(r"\d|'s\b", spec["scope"]):
        flags.append("scope_not_plain")
    if IMAGE_POSITION.search(spec["ref"]):
        flags.append("ref_image_position")
    if original["route"] in ("add", "remove", "replace", "attribute") and original["route"] != spec["operation"]:
        flags.append(f"retyped:{original['route']}->{spec['operation']}")
    return flags + ip.length_advice(ref, noref)


def sam3_flags(mask, box):
    """A re-masked S should fill its grounding box and stay inside it (box grown by 10 % per side)."""
    if not box or not mask.any():
        return []
    h, w = mask.shape
    x1, y1, x2, y2 = box[0] * w / 1000, box[1] * h / 1000, box[2] * w / 1000, box[3] * h / 1000
    mx, my = 0.1 * (x2 - x1), 0.1 * (y2 - y1)
    inside = np.zeros_like(mask)
    inside[max(0, int(y1 - my)):int(np.ceil(y2 + my)), max(0, int(x1 - mx)):int(np.ceil(x2 + mx))] = True
    flags = []
    if (mask & ~inside).sum() / mask.sum() > 0.15:
        flags.append("sam3_spill")
    if mask.sum() / max((x2 - x1) * (y2 - y1), 1) < 0.15:
        flags.append("sam3_sparse")
    return flags


def geometry_flags(spec, geo, exec_kind):
    flags, hard = [], []
    if ip.needs_local_execution(spec):
        if not geo:
            hard.append("execution_region_missing")
        elif spec["remove_mode"] == "contents":
            if geo["s_in_exec"] < 0.9:  # the container and what it holds are both removed
                hard.append("execution_region_misses_container")
        elif geo["s_in_exec"] >= 0.9:  # a part, the contents only or a cut: never the whole host
            hard.append("execution_region_covers_host")
    if spec["operation"] == "add":
        mode = spec["placement"].get("anchor_mode", "support")
        relation = spec["placement"].get("relation", "to")
        if mode == "reference" and not geo:
            hard.append("reference_add_without_execution_region")
        elif geo and mode == "support":
            # The anchor is what the item touches: one with open space between them is a reference in disguise.
            if relation in CONTACT_RELATIONS and geo["gap_frac"] > ANCHOR_PARAMS["apart_frac"]:
                hard.append("support_not_touching_item")
            elif geo.get("s_in_probe") is False:
                # It touches S, but S is not under the item it carries (above the one hanging from it): a neighbor.
                hard.append("support_not_under_item")
            elif geo["gap_frac"] > ANCHOR_PARAMS["adjacent_frac"]:
                flags.append("item_not_touching_anchor")
        elif geo and mode == "reference":
            if geo["gap_frac"] > ANCHOR_PARAMS["adjacent_frac"]:  # only an object the item touches or stands against
                hard.append("reference_anchor_not_adjacent")
            if geo["exec_in_s"] > 0.5:
                # Mostly inside S: true for "near" (in front of S), false for beside, under, above or below.
                (flags if relation == "near" else hard).append("reference_item_inside_anchor")
            if (relation in ("under", "below") and geo["dy"] < 0) or (relation == "above" and geo["dy"] > 0):
                hard.append("reference_direction")
            site = spec["placement"].get("site", "")
            if ("left" in site and geo["dx"] > 0) or ("right" in site and geo["dx"] < 0):
                hard.append("reference_side")
    if is_text_change(spec) and geo and geo["exec_in_s"] < 0.7:
        flags.append("text_outside_carrier")
    if exec_kind == "none" and spec["operation"] == "add":
        flags.append("no_execution_region")
    return flags, hard


def build(row, extracted, remasked, derived_exec=None, images=None):
    """One protocol 1.0 record (accepted, multi_object or rejected) from the slots and the masks.

    images: the (BEFORE, AFTER) pair, needed only to find the new item of a Derived add (its label is the host),
    around which the anchor is cut.
    """
    out = {"id": row["id"], "dataset": row["dataset"], "type_before": row["type"],
           "instruction_before": row["instruction"], "adapter_version": ADAPTER_VERSION,
           "instruction_protocol_version": ip.VERSION}
    original = draft(row)
    slots, spec = extracted.get("slots"), extracted.get("spec")
    out["draft"], out["slots"] = original, slots
    if slots is None:
        return {**out, "status": "rejected", "reason": "no_valid_slots"}
    if slots["decision"] != "accept":
        return {**out, "status": slots["decision"] if slots["decision"] == "multi_object" else "rejected",
                "reason": slots.get("reason") or slots["decision"]}
    plan = extracted["plan"]
    roles = roles_of(slots, len(row["instances"]))
    if "separate" in roles.values():
        return {**out, "status": "multi_object", "reason": "an outlined object is edited separately"}
    if plan["source"] == "sam3":
        if not remasked or remasked.get("mask") is None:
            return {**out, "status": "rejected", "reason": "sam3_failed",
                    "remask": {k: v for k, v in (remasked or {}).items() if k != "mask"}}
        selection = {"source": "sam3", **remasked["mask"], "sam3": remasked.get("meta", {})}
    else:
        chosen = [row["instances"][n - 1] for n in plan["instances"]]
        selection = {"source": plan["source"], "instances": [i["instance_id"] for i in chosen], **union(chosen)}
    if row["dataset"] == "derived" and spec["operation"] == "add":
        execution = None
    elif row["dataset"] == "derived" and spec["operation"] == "remove" and derived_exec:
        execution = {"source": "derived_execution_region", **derived_exec}
    else:
        execution = {"source": "cleaned_region", **{k: row["region"][k] for k in ("rle_size", "rle_counts", "box_1000")}}
    s_mask = decode(selection)
    exec_mask = decode(execution) if execution else None
    if selection["source"] == "sam3" and spec["operation"] != "add" and exec_mask is not None:
        # The labels missed S; a labeled region that mostly misses the new S is not where the edit happened.
        if (exec_mask & s_mask).sum() < 0.5 * s_mask.sum():
            execution = {"source": "selection", **encode(s_mask)}
        elif spec["operation"] == "replace":
            execution = {"source": "cleaned_region_or_selection", **encode(exec_mask | s_mask)}
        exec_mask = decode(execution)
    flags_sam3 = sam3_flags(s_mask, (extracted.get("hinted") or {}).get("bbox_2d")) if selection["source"] == "sam3" else []
    footprint = kind = host_view = None
    anchor_flags, hard = [], []
    if spec["operation"] == "add":
        footprint, kind = add_footprint(exec_mask, s_mask, images, (extracted.get("item") or {}).get("bbox_2d"))
        selection, host_view, anchor_flags, hard = add_anchor(slots, spec, plan, selection, footprint, kind,
                                                              remasked, extracted)
        s_mask = decode(selection)
    geo = geometry(spec, s_mask, footprint if spec["operation"] == "add" else exec_mask, slots.get("host_scope", ""))
    fixes = []
    ref, noref = ip.render(spec, True), ip.render(spec, False)
    ip.validate_pair(spec, ref, noref)
    flags = text_flags(row, spec, ref, noref, original) + anchor_flags
    geo_flags, geo_hard = geometry_flags(spec, geo, "none" if execution is None else execution["source"])
    if kind == "item_box" and "support_not_under_item" in geo_hard:
        # The new item is known only from the model's box, too rough for this check: noted, not decided.
        geo_hard.remove("support_not_under_item")
        geo_flags.append("support_not_under_item")
    hard += geo_hard
    blind = (extracted.get("blind") or {}).get("bbox_2d")
    blind_iou = round(box_iou(blind, selection["box_1000"]), 3)
    # An add ref names a part or a stretch of a surface, which the cut S only has to lie in.
    located = blind_iou >= 0.5 or (spec["operation"] == "add" and box_holds_center(blind, selection["box_1000"]))
    if not located:
        flags.append("ref_not_located" if blind is None else "ref_points_elsewhere")
    flags += geo_flags + flags_sam3 + [f"fix:{f}" for f in slots.get("fixes", []) + fixes]
    record = {**out, "status": "rejected" if hard else "accepted", "reason": ",".join(hard) or None,
              "instruction_spec": spec, "instruction_ref": ref, "instruction_noref": noref,
              "mask_semantics": ip.mask_contract(spec), "selection": selection, "execution_region": execution,
              "host_view": host_view,
              "checks": {"geometry": geo, "blind_box": blind, "blind_iou": blind_iou}, "flags": flags}
    return record


def summarize(records):
    count = Counter(r["status"] for r in records)
    by_type = defaultdict(Counter)
    for r in records:
        op = (r.get("instruction_spec") or {}).get("operation") or "-"
        by_type[f"{r['dataset']}/{r['type_before']}"][f"{r['status']}:{op}"] += 1
    flags = Counter(f.split(":")[0] if not f.startswith(("fix:", "retyped:")) else f
                    for r in records for f in r.get("flags", []))
    return {"rows": len(records), "status": dict(count), "by_source_type": {k: dict(v) for k, v in sorted(by_type.items())},
            "flags": dict(flags.most_common()),
            "selection_sources": dict(Counter((r.get("selection") or {}).get("source") for r in records
                                              if r["status"] == "accepted")),
            "add_anchors": dict(Counter(f'{r["selection"]["granularity"]}:{"cut" if r["selection"]["anchor"].get("cut") else "whole"}'
                                        for r in records if r["status"] == "accepted"
                                        and "granularity" in (r.get("selection") or {}))),
            "host_views": sum(1 for r in records if r["status"] == "accepted" and r.get("host_view"))}

# ---------------------------------------------------------------- edited region

EDITED_METHOD = "lab-diff-hysteresis-1"
# Calibrated on round 7: backgrounds of pasted and most linked pairs are identical (Delta E 0); resized pairs carry
# noise with a median of about 2 and a 99th percentile of about 6; real edits sit at 10 to 40 and more. The noise
# level is 3x the median far from S, which matches that 99th percentile but ignores real changes elsewhere.
EDITED_PARAMS = {"blur_sigma": 1.0, "t_low": 6.0, "t_high": 12.0, "low_k": 1.5, "high_k": 2.5,
                 "noise_far_frac": 0.08, "zone_frac": 0.06, "close_frac": 0.004, "hole_frac": 0.005,
                 "min_component_frac": 0.0002, "keep_in_zone": 0.3}


def grow(mask, radius):
    """Pixels within `radius` of the mask (Euclidean), in time independent of the radius."""
    import cv2
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5) <= radius


def color_difference(source, target, sigma=EDITED_PARAMS["blur_sigma"]):
    """CIE76 Delta E per pixel after a light blur (JPEG and resampling noise)."""
    import cv2
    labs = []
    for image in (source, target):
        lab = cv2.cvtColor(cv2.GaussianBlur(np.asarray(image.convert("RGB")), (0, 0), sigma),
                           cv2.COLOR_RGB2LAB).astype(np.float32)
        lab[..., 0] *= 100 / 255
        lab[..., 1:] -= 128
        labs.append(lab)
    return np.sqrt(((labs[0] - labs[1]) ** 2).sum(-1))


def edited_region(source, target, s_mask, exec_mask=None, params=EDITED_PARAMS):
    """Mask of the pixels that really changed between BEFORE and AFTER, and how it was found.

    Hysteresis on Delta E: weak changes join a region only when connected to a strong change, so faint
    regeneration noise is ignored while the full extent of an edit (soft shadows, anti-aliased edges) is kept.
    Thresholds rise with this pair's own noise, measured far from S and the execution region. Components mostly
    outside the neighborhood of S and the execution region are dropped and counted, never silently kept.
    """
    import cv2
    p = params
    d = color_difference(source, target, p["blur_sigma"])
    h, w = d.shape
    diag = (h * h + w * w) ** 0.5
    anchor = s_mask | exec_mask if exec_mask is not None else s_mask
    far = ~grow(anchor, p["noise_far_frac"] * diag)
    noise = 3 * float(np.median(d[far] if far.sum() > 0.05 * h * w else d))
    t_low, t_high = max(p["t_low"], p["low_k"] * noise), max(p["t_high"], p["high_k"] * noise)
    low = d > t_low
    count, labels = cv2.connectedComponents(low.astype(np.uint8), connectivity=8)
    seeds = np.zeros(count, bool)
    seeds[np.unique(labels[d > t_high])] = True
    seeds[0] = False
    mask = seeds[labels]
    radius = max(2, round(p["close_frac"] * diag))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
    # Fill small holes (unchanged pixels inside a changed object); keep large ones, which can be real.
    count, labels, stats, _ = cv2.connectedComponentsWithStats((~mask).astype(np.uint8), connectivity=4)
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    small = stats[:, cv2.CC_STAT_AREA] < p["hole_frac"] * h * w
    small[border] = False
    mask |= small[labels]
    zone = grow(anchor, p["zone_frac"] * diag)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    area = stats[:, cv2.CC_STAT_AREA].astype(np.float64)
    in_zone = np.bincount(labels[zone], minlength=count)
    on_anchor = np.bincount(labels[anchor], minlength=count)
    # Size: a small piece counts when it touches S or the execution region (a sticker, a removed letter).
    big = (area >= max(30, p["min_component_frac"] * h * w)) | ((area >= 12) & (on_anchor > 0))
    # Place: mostly near S, or covering much of S and the execution region (a large replacement spills far).
    near = (in_zone >= p["keep_in_zone"] * area) | (on_anchor >= p["keep_in_zone"] * anchor.sum())
    keep = big & near
    keep[0] = False
    region = keep[labels]
    total = float(h * w)
    stats_out = {"noise": round(noise, 2), "t_low": round(t_low, 2), "t_high": round(t_high, 2),
                 "area_frac": round(region.sum() / total, 5), "components": int(keep.sum()),
                 "dropped_far_frac": round(float(area[1:][big[1:] & ~near[1:]].sum()) / total, 5),
                 "dropped_small_frac": round(float(area[1:][~big[1:]].sum()) / total, 5),
                 "s_covered": round(float((region & s_mask).sum()) / max(int(s_mask.sum()), 1), 3),
                 "in_anchor": round(float((region & anchor).sum()) / max(int(region.sum()), 1), 3)}
    return (region if region.any() else None), stats_out


def attach_edited_region(record, region, stats):
    """Store the edited region on an accepted record; a record without one cannot stay accepted."""
    if record["status"] != "accepted":
        return record
    record = {**record, "flags": list(record["flags"])}
    if region is None:
        return {**record, "status": "rejected", "reason": "edited_region_empty", "edited_region": None,
                "edited_region_stats": stats}
    op = record["instruction_spec"]["operation"]
    if stats["area_frac"] < 0.0003:
        record["flags"].append("edited_region_tiny")
    if stats["dropped_far_frac"] > stats["area_frac"]:
        record["flags"].append("edited_region_far_changes")
    if op in ("attribute", "remove", "replace") and stats["s_covered"] < 0.2:
        record["flags"].append("edited_region_misses_s")
    if stats["in_anchor"] < 0.3:
        record["flags"].append("edited_region_off_anchor")
    record["edited_region"] = {"method": EDITED_METHOD, **encode(region), "area_frac": stats["area_frac"],
                               "stats": stats}
    return record

# ---------------------------------------------------------------- review sheets


def wrap(draw, text, font, width):
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=font) <= width or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    return lines + ([line] if line else [])


def review_sheet(row, record, root, images=None):
    """BEFORE with S (green), the host S was cut from (yellow) and the execution region (orange), AFTER with the
    edited region (magenta), a close-up, and the texts."""
    source, target = images or open_pair(row, root)
    masks = []
    if (record.get("selection") or {}).get("host"):
        masks.append((decode(record["selection"]["host"]), HOST_COLOR))
    if record.get("execution_region"):
        masks.append((decode(record["execution_region"]), EXEC_COLOR))
    if record.get("selection"):
        masks.append((decode(record["selection"]), S_COLOR))
    if not masks:
        masks = [(decode(i), PALETTE[k % len(PALETTE)][1]) for k, i in enumerate(row["instances"])]
    panel = 560 * 560
    before = ec.labeled(outlined(source, masks, panel, numbers=False), "BEFORE: S green, host yellow, exec orange")
    if record.get("edited_region"):
        after = ec.labeled(outlined(target, [(decode(record["edited_region"]), EDITED_COLOR)], panel, numbers=False),
                           "AFTER: edited region magenta")
    else:
        after = ec.labeled(ec.resized(target, panel), "AFTER")
    boxes = [b for b in ((record.get("selection") or {}).get("box_1000"),
                         (record.get("execution_region") or {}).get("box_1000"), row["region"]["box_1000"]) if b]
    box = [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]
    edited = [(decode(record["edited_region"]), EDITED_COLOR)] if record.get("edited_region") else None
    if edited:
        box = [min(box[0], record["edited_region"]["box_1000"][0]), min(box[1], record["edited_region"]["box_1000"][1]),
               max(box[2], record["edited_region"]["box_1000"][2]), max(box[3], record["edited_region"]["box_1000"][3])]
    close = close_up(source, target, masks, box, max_long=1100, after_masks=edited)
    width = before.width + after.width + 20
    if close.width > width:
        close = close.resize((width, round(close.height * width / close.width)), Image.Resampling.LANCZOS)
    font, bold = ImageFont.truetype(ec.font_path().replace("-Bold", ""), 19), ImageFont.truetype(ec.font_path(), 19)
    spec = record.get("instruction_spec") or {}
    head = (f"{row['id']} | {row['dataset']}/{row['type']} -> {spec.get('operation', '-')} | {record['status']}"
            + (f" ({record.get('reason')})" if record.get("reason") else ""))
    texts = [(head, bold), ("OLD: " + row["instruction"], font)]
    if spec:
        texts += [("REF: " + record["instruction_ref"], bold), ("NOREF: " + record["instruction_noref"], bold)]
        if record.get("host_view"):
            texts.append(("HOST VIEW NOREF: " + record["host_view"]["instruction_noref"], bold))
        selection = record["selection"]
        texts += [(f"S: {selection['source']} {selection.get('granularity', '')} "
                   f"cut={(selection.get('anchor') or {}).get('cut')}; scope={spec['scope']!r}; "
                   f"part={spec['target_part']!r}; "
                   f"remove_mode={spec['remove_mode']}; attributes={spec['attributes']}; placement={spec['placement']}; "
                   f"qualifier={spec['qualifier']!r}", font),
                  (f"blind IoU={record['checks']['blind_iou']}; geometry={record['checks']['geometry']}", font),
                  ("flags: " + (", ".join(record["flags"]) or "none"), font)]
    slots = record.get("slots") or {}
    if slots:
        texts.append(("observed: " + str(slots.get("observed", "")), font))
    probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    lines = [(line, f) for text, f in texts for line in wrap(probe, text, f, width - 20)]
    height = before.height + close.height + 30 + 26 * len(lines) + 20
    sheet = Image.new("RGB", (width, height), "white")
    sheet.paste(before, (0, 0))
    sheet.paste(after, (before.width + 20, 0))
    sheet.paste(close, (0, before.height + 15))
    draw, y = ImageDraw.Draw(sheet), before.height + close.height + 30
    for line, f in lines:
        draw.text((10, y), line, font=f, fill=(0, 0, 0))
        y += 26
    return sheet
