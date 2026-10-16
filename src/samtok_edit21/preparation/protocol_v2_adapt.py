"""Adapt existing training records to instruction protocol 2.0 (no-ref names no target).

Rules: docs/07_SAMTokEdit_Qwen21_指令协议2.0_noref改为盲指代.md.  The protocol module is the benchmark's
canonical 2.0 file, vendored byte for byte as instruction_protocol_v2.py (tests pin the hash).

Per record:
* the ref sentence is re-rendered and must equal the stored one (the spec is unchanged apart from its
  version);
* the no-ref sentence is the canonical blind rendering, checked by ``validate_noref``; a record whose
  qualifier the module cannot classify keeps its ref sentence only (``noref_ineligible``);
* the source mask becomes the region actually edited.  In the existing data that changes two kinds of
  record: an add whose site is written in the sentence while S is the whole host (S is cut to the spot
  where the new item sits, with the anchor localisation of protocol_adapt), and a remove with its contents
  (S is joined with the execution region).  When no spot can be found the record keeps its ref sentence
  only, so no no-ref row ever pairs "the selected region" with a whole host.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from samtok_edit21.preparation import instruction_protocol_v2 as p2
from samtok_edit21.preparation import protocol_adapt as pa

PROTOCOL_SHA256 = "fe192e8d955a36ca58fb4a383585bb72110076f834d14294dd35e4ae286ffa85"
PROTOCOL_SOURCE = "Tangent0308/samtok_edit_benchmark@9e63e4f:src/samtok_benchmark/v2/instruction_protocol.py"
ADAPTER_VERSION = "protocol-v2-blind-1"
# Under 1.0 a small anchor stayed whole because the sentence still named the site ("the top edge of the
# backrest"); under 2.0 the site leaves the sentence, so the mask must carry it.  The window of protocol_adapt is
# reused with two changes: an anchor stays whole only when it is at most 3x the new item's area (no exemption for
# anchors under 1 % of the image), and the window stops growing once it holds half the item's area or 0.02 % of
# the image (not 0.2 %, which swallowed most small hosts whole).
SITE_PARAMS = {**pa.ANCHOR_PARAMS, "small_frac": 0.0, "min_area_frac": 0.0002}
# A cut spot must still be a usable region: at least 0.01 % of the image and 64 pixels.
MIN_SPOT_FRAC, MIN_SPOT_PX = 0.0001, 64


def protocol_hash():
    return hashlib.sha256(Path(p2.__file__).read_bytes()).hexdigest()


def spec_v2(spec):
    return {**spec, "version": p2.VERSION}


def has_site(spec):
    return spec["operation"] == "add" and bool((spec.get("placement") or {}).get("site"))


def sentences(spec, stored_ref):
    """(spec 2.0, ref, no-ref or None, reason).  The ref must reproduce the stored sentence."""
    s = spec_v2(spec)
    ref = p2.render(s, True)
    if ref != stored_ref:
        raise ValueError(f"ref sentence changed: {stored_ref!r} -> {ref!r}")
    noref = p2.render(s, False)
    try:
        p2.validate_noref(s, noref)
    except ValueError as error:
        return s, ref, None, "unclassified_qualifier" if "qualifier" in str(error) else str(error)
    return s, ref, noref, None


def deepest_point(mask):
    """The pixel inside the mask farthest from its border, as [x, y] (the benchmark's point rule)."""
    import cv2
    dist = cv2.distanceTransform(np.pad(mask, 1).astype(np.uint8), cv2.DIST_L2, 5)[1:-1, 1:-1]
    y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return [int(x), int(y)]


def tight_box(mask):
    """[x0, y0, x1, y1] in pixels, right and bottom exclusive."""
    ys, xs = np.nonzero(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def main_item(changed, keep=0.5):
    """The new item among the changed pixels: the largest connected piece and pieces at least half its size.

    The 50k generator redraws its whole support crop, so small regenerated patches elsewhere on the host also
    change; they must not widen the window around the item.
    """
    import cv2
    if changed is None or not changed.any():
        return None
    count, labels, stats, _ = cv2.connectedComponentsWithStats(changed.astype(np.uint8), connectivity=8)
    area = stats[:, cv2.CC_STAT_AREA]
    big = area >= keep * area[1:].max()
    big[0] = False
    return big[labels]


def site_spot(s_mask, footprint):
    """S cut to where the new item sits, or (None, reason).  Uses the protocol-1.0 anchor window with SITE_PARAMS."""
    if footprint is None or not footprint.any():
        return None, "no_footprint"
    spot, info = pa.localize_anchor(s_mask, footprint, SITE_PARAMS)
    if spot is None:
        return None, "item_not_on_anchor"
    if spot.sum() < max(MIN_SPOT_PX, MIN_SPOT_FRAC * spot.size):
        return None, "spot_too_small"
    return spot, info


# ---------------------------------------------------------------- cleaned v1 pairs (protocol 1.0 manifest)

def adapt_v1(record):
    """One accepted record of protocol_adapt_qwen38_27b_v1/manifest_protocol.jsonl -> protocol 2.0."""
    spec = record["instruction_spec"]
    s, ref, noref, reason = sentences(spec, record["instruction_ref"])
    out = {k: v for k, v in record.items() if k not in ("instruction_noref", "mask_semantics", "adapter_version",
                                                      "instruction_protocol_version")}
    out.update(instruction_spec=s, instruction_ref=ref, mask_semantics=p2.mask_contract(s),
               instruction_protocol_version=p2.VERSION, adapter_version=ADAPTER_VERSION)
    change = None
    if has_site(spec) and record["selection"].get("granularity") == "object":
        s_mask = pa.decode(record["selection"])
        footprint, source = None, None
        for key in ("execution_region", "edited_region"):
            region = record.get(key)
            if region:
                footprint, source = main_item(pa.decode(region)), key
                break
        spot, info = site_spot(s_mask, footprint)
        if spot is None:
            reason = reason or "site_spot_missing:" + info
        elif spot.sum() < s_mask.sum():
            change = {"selection": {**pa.encode(spot), "source": "site_spot", "footprint": source, "localized": info}}
        else:
            change = {"note": "anchor_already_local"}
    elif spec["operation"] == "remove" and spec["remove_mode"] == "contents" and record.get("execution_region"):
        s_mask = pa.decode(record["selection"])
        joined = s_mask | pa.decode(record["execution_region"])
        if joined.sum() > s_mask.sum():
            change = {"selection": {**pa.encode(joined), "source": "s_with_contents"}}
    if change and "selection" in change:
        out["selection_v1"] = record["selection"]
        out["selection"] = change["selection"]
    if record.get("host_view"):
        out["host_view"] = {**record["host_view"], "use": "ref_only"}
    out["instruction_noref"] = noref if reason is None else None
    out["noref_ineligible"] = reason
    out["protocol_v2_change"] = ("selection_" + change["selection"]["source"]) if change and "selection" in change \
        else (change or {}).get("note")
    return out


# ---------------------------------------------------------------- SA-1B 50k release

def adapt_sa1b(row, root, mask_dir="masks/region_v2", out_root=None):
    """One record of release/annotations/train.jsonl -> protocol 2.0.  Writes a new mask PNG when S changes,
    under ``out_root`` (default: the release root); the stored path is relative to it."""
    from PIL import Image
    spec = row["instruction_spec"]
    s, ref, noref, reason = sentences(spec, row["instruction"])
    out = {k: v for k, v in row.items() if k not in ("instruction_noref",)}
    out.update(instruction_spec=s, instruction=ref, instruction_protocol_version=p2.VERSION,
               mask_semantics=p2.mask_contract(s), adapter_version=ADAPTER_VERSION)
    change = None
    if has_site(spec):
        root = Path(root)
        s_mask = np.asarray(Image.open(root / row["mask"]).convert("L")) > 0
        source = Image.open(root / row["source_image"]).convert("RGB")
        target = Image.open(root / row["edited_image"]).convert("RGB")
        support = np.asarray(Image.open(root / row["support_mask"]).convert("L")) > 0
        if target.size != source.size:
            target = target.resize(source.size, Image.LANCZOS)
        footprint, stats = pa.edited_region(source, target, s_mask, support)
        if footprint is not None:
            footprint = main_item(footprint & support)  # the compositor wrote nothing outside the support
        spot, info = site_spot(s_mask, footprint)
        if spot is None:
            reason = reason or "site_spot_missing:" + info
        elif spot.sum() < s_mask.sum():
            name = f"{mask_dir}/{row['sample_id'][:2]}/{row['sample_id']}.png"
            path = Path(out_root or root) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(spot.astype(np.uint8) * 255).save(path)
            change = {"mask": name, "localized": info, "edited_region": stats}
        else:
            change = {"note": "anchor_already_local"}
    if change and "mask" in change:
        mask = np.asarray(Image.open(Path(out_root or root) / change["mask"]).convert("L")) > 0
        out.update(mask_v1=row["mask"], bbox_v1=row["bbox"], point_v1=row["point"], mask=change["mask"],
                   bbox=tight_box(mask), point=deepest_point(mask), site_spot=change["localized"])
    out["instruction_noref"] = noref if reason is None else None
    out["noref_ineligible"] = reason
    out["protocol_v2_change"] = "mask_site_spot" if change and "mask" in change else (change or {}).get("note")
    return out


# ---------------------------------------------------------------- SA-1B 50k: the pixels that really changed

# The 50k compositor pastes the edit back inside the support mask only, so outside it the target equals the source
# pixel for pixel and the noise there is zero.  The hysteresis thresholds therefore stay at their floors (Delta E 6
# and 12) instead of scaling with a noise estimate, which for a support covering most of the image would otherwise
# be taken over the edit itself.
SA1B_EDITED_METHOD = "lab-diff-hysteresis-sa1b-1"
SA1B_EDITED_PARAMS = {**pa.EDITED_PARAMS, "low_k": 0.0, "high_k": 0.0}


def sa1b_edited_region(row, root, out_root=None, mask_dir="masks/edited"):
    """Changed pixels of one 50k record, clipped to its support: (field dict for the record, PNG path or None)."""
    from PIL import Image
    root = Path(root)
    source = Image.open(root / row["source_image"]).convert("RGB")
    target = Image.open(root / row["edited_image"]).convert("RGB")
    if target.size != source.size:
        raise ValueError(f"{row['sample_id']}: target {target.size} != source {source.size}")
    s_mask = np.asarray(Image.open(root / row["mask"]).convert("L")) > 0
    support = np.asarray(Image.open(root / row["support_mask"]).convert("L")) > 0
    region, stats = pa.edited_region(source, target, s_mask, support, SA1B_EDITED_PARAMS)
    strong_outside = float(((pa.color_difference(source, target) > SA1B_EDITED_PARAMS["t_high"]) & ~support).mean())
    if region is not None:
        region &= support
    field = {"method": SA1B_EDITED_METHOD, "stats": {**stats, "strong_change_outside_support_frac": round(strong_outside, 6)}}
    if region is None or not region.any():
        return {**field, "rle_size": None, "rle_counts": None, "box_1000": None, "area_frac": 0.0}, None
    name = f"{mask_dir}/{row['sample_id'][:2]}/{row['sample_id']}.png"
    path = Path(out_root or root) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(region.astype(np.uint8) * 255).save(path)
    h, w = region.shape
    field.update(pa.encode(region), area_frac=round(float(region.sum()) / (h * w), 6),
                 s_covered=round(float((region & s_mask).sum()) / max(int(s_mask.sum()), 1), 3),
                 in_support=1.0)
    return field, name
