"""Pixel checks used by the Stage 2 pair cleaning (docs/09, section 6.7).

coherent_outside: is there a solid block of real change far outside the region?
alignment:        is an unregistered target geometrically aligned with its source?
"""
import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

WORK = 512
ALIGN_SIDE = 256


def _resized(src, tgt, z, valid, side, margin):
    h, w = z.shape
    s = side / max(w, h)
    size = (max(1, round(w * s)), max(1, round(h * s)))
    a = np.asarray(Image.fromarray(src).resize(size, Image.BOX))
    b = np.asarray(Image.fromarray(tgt.astype(np.uint8)).resize(size, Image.BOX))
    zs = np.asarray(Image.fromarray(z.astype(np.uint8) * 255).resize(size, Image.NEAREST)) > 0
    vs = np.ones(zs.shape, bool) if valid is None else \
        np.asarray(Image.fromarray(valid.astype(np.uint8) * 255).resize(size, Image.NEAREST)) > 0
    far = (ndimage.distance_transform_edt(~zs) > margin * max(size)) & vs
    return a, b, far


def coherent_outside(src, tgt, z, thr=30.0, open_radius=3, margin=0.06, valid=None):
    """Largest solid block of change far outside the region, as a fraction of the image (at WORK px).

    The target is first colour-matched to the source (per-channel affine fitted on pixels beyond `margin` of the
    long side from the region), so global tone and lighting shifts do not count; strong differences are then opened
    with a disk so thin edge and texture noise from re-rendering disappears.
    """
    a, b, far = _resized(src, tgt, z, valid, WORK, margin)
    if far.sum() < 100:
        return 0.0
    a, b = a.astype(np.float32), b.astype(np.float32)
    bn = b.copy()
    for c in range(3):
        x, y = b[..., c][far], a[..., c][far]
        var = x.var()
        gain = ((x - x.mean()) * (y - y.mean())).mean() / var if var > 1e-3 else 1.0
        gain = float(np.clip(gain, 0.5, 2.0))
        bn[..., c] = gain * (b[..., c] - x.mean()) + y.mean()
    strong = np.abs(bn - a).mean(axis=2) > thr
    yy, xx = np.mgrid[-open_radius:open_radius + 1, -open_radius:open_radius + 1]
    blobs = ndimage.binary_opening(strong, structure=(xx ** 2 + yy ** 2) <= open_radius ** 2) & far
    lab, n = ndimage.label(blobs)
    if n == 0:
        return 0.0
    areas = ndimage.sum(np.ones(lab.shape), lab, index=np.arange(1, n + 1))
    return float(areas.max() / lab.size)


def alignment(src, tgt, z, margin=0.06, valid=None, tile=32, search=8, min_std=6.0, min_ncc=0.7):
    """(share of confident tiles whose best shift is <= 1 px, number of confident tiles) at ALIGN_SIDE px.

    Textured tiles far outside the region are matched by normalised cross-correlation within +-`search` px in
    the target.  Re-rendered but aligned targets keep their structures in place; zoomed or shifted targets do not.
    """
    a, b, far = _resized(src, tgt, z, valid, ALIGN_SIDE, margin)
    a = cv2.GaussianBlur(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 1.0)
    b = cv2.GaussianBlur(cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 1.0)
    H, W = a.shape
    good = conf = 0
    for y in range(search, H - tile - search + 1, tile // 2):
        for x in range(search, W - tile - search + 1, tile // 2):
            if far[y:y + tile, x:x + tile].mean() < 0.9:
                continue
            t = a[y:y + tile, x:x + tile]
            if t.std() < min_std:
                continue
            res = cv2.matchTemplate(b[y - search:y + tile + search, x - search:x + tile + search], t, cv2.TM_CCOEFF_NORMED)
            _, peak, _, loc = cv2.minMaxLoc(res)
            if peak < min_ncc:
                continue
            conf += 1
            good += max(abs(loc[0] - search), abs(loc[1] - search)) <= 1
    return (good / conf if conf else float("nan")), conf
