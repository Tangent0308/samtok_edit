"""Changed pixels between a source image and its edit: the ``edited_region`` of the merged data (docs/01 §1).

Ported unchanged from v3 (``preparation/protocol_adapt.py``: ``color_difference``, ``grow``, ``edited_region``;
``preparation/protocol_v2_adapt.py``: ``sa1b_edited_region``), which computed the field for the SA-1B 50k release
(D1) on 10-15/16.  The merge's newer SA-1B families (D12-D14, ``preparation/sa1b_groups.py``) use the same
function, so their regions mean the same as D1's; ``tests/test_edited_region.py`` checks the port.
"""
from __future__ import annotations

import numpy as np

EDITED_PARAMS = {"blur_sigma": 1.0, "t_low": 6.0, "t_high": 12.0, "low_k": 1.5, "high_k": 2.5,
                 "noise_far_frac": 0.08, "zone_frac": 0.06, "close_frac": 0.004, "hole_frac": 0.005,
                 "min_component_frac": 0.0002, "keep_in_zone": 0.3}
# Pasted-back edits (outside the support equal to the source): fixed thresholds, no noise scaling.
SA1B_EDITED_METHOD = "lab-diff-hysteresis-sa1b-1"
SA1B_EDITED_PARAMS = {**EDITED_PARAMS, "low_k": 0.0, "high_k": 0.0}


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


def grow(mask, radius):
    """Pixels within `radius` of the mask (Euclidean), in time independent of the radius."""
    import cv2
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5) <= radius


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


def edited_region(source, target, s_mask, exec_mask=None, params=EDITED_PARAMS, d=None):
    """Mask of the pixels that really changed between BEFORE and AFTER, and how it was found.

    Hysteresis on Delta E: weak changes join a region only when connected to a strong change, so faint
    regeneration noise is ignored while the full extent of an edit (soft shadows, anti-aliased edges) is kept.
    Thresholds rise with this pair's own noise, measured far from S and the execution region. Components mostly
    outside the neighborhood of S and the execution region are dropped and counted, never silently kept.
    ``d``: the pair's ``color_difference`` when already computed (several units edited into one target).
    """
    import cv2
    p = params
    if d is None:
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


def sa1b_edited_region(source, target, s_mask, support, d=None):
    """``edited_region`` field of a pasted-back SA-1B edit: changed pixels clipped to the support (as D1)."""
    params = SA1B_EDITED_PARAMS
    if d is None:
        d = color_difference(source, target, params["blur_sigma"])
    region, stats = edited_region(source, target, s_mask, support, params, d=d)
    strong_outside = float(((d > params["t_high"]) & ~support).mean())
    if region is not None:
        region &= support
    field = {"method": SA1B_EDITED_METHOD, "stats": {**stats, "strong_change_outside_support_frac": round(strong_outside, 6)}}
    if region is None or not region.any():
        return {**field, "rle_size": None, "rle_counts": None, "box_1000": None, "area_frac": 0.0}, None
    h, w = region.shape
    field.update(encode(region), area_frac=round(float(region.sum()) / (h * w), 6),
                 s_covered=round(float((region & s_mask).sum()) / max(int(s_mask.sum()), 1), 3),
                 in_support=1.0)
    return field, region
