"""The "point it out" task family used to teach the DiT what a mask token refers to (v3 Stage A).

Three kinds of training rows share one region reference ``<phrase> <M>`` (or ``this region <M>`` without a phrase):

* highlight  - "Apply a red semi-transparent mask to <X> ...": the target is the source with a 50% colour overlay on the
               token's region.  The model has to output the region itself, so the loss is directly about the token.
* procedural - a local edit with an exact target (grayscale / brighter / darker / blur / invert) inside the region.
* real       - a cleaned real edit pair with its own instruction.

The region is always what the SAMTok codec decodes from the token, never a separate annotation, so supervision and
token agree by construction.  The same definitions give the read-back used for evaluation: generate a highlight, then
recover the painted region from the output image.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageFilter

# Order matters: data builders draw colours, templates and operations with a seeded RNG.
PALETTE = {
    "red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255), "yellow": (255, 255, 0),
    "magenta": (255, 0, 255), "cyan": (0, 255, 255), "orange": (255, 128, 0), "purple": (128, 0, 255),
}
COLORS = tuple(PALETTE)
READBACK_COLORS = ("red", "green", "blue", "magenta", "yellow", "cyan")  # saturated colours a read-back can separate
HIGHLIGHT_TEMPLATES = (
    "Apply {a} {color} semi-transparent mask to {x} and keep everything else unchanged.",
    "Highlight {x} with {a} {color} semi-transparent mask; leave the rest of the image unchanged.",
    "Cover {x} with {a} {color} semi-transparent overlay and keep everything else unchanged.",
    "Mark the pixels of {x} with {a} {color} semi-transparent mask and leave the rest unchanged.",
)
EVAL_TEMPLATE = HIGHLIGHT_TEMPLATES[0]
PROCEDURAL_TEMPLATES = {
    "grayscale": "Convert {x} to grayscale and keep everything else unchanged.",
    "brighter": "Make {x} much brighter and keep everything else unchanged.",
    "darker": "Make {x} much darker and keep everything else unchanged.",
    "blur": "Blur {x} and keep everything else unchanged.",
    "invert": "Invert the colors of {x} and keep everything else unchanged.",
}
REMOVE_PROBE = "Remove the object in this region {span}."
OVERLAY_ALPHA = 0.5
_LEADING = {"the", "a", "an", "one", "two", "three", "all", "both", "this", "that", "these", "those"}


def article(color):
    return "an" if color[0] in "aeiou" else "a"


def region_reference(span, phrase=None):
    """``<phrase> <M>`` when the object is named, ``this region <M>`` otherwise."""
    return f"{phrase} {span}" if phrase else f"this region {span}"


def highlight_prompt(span, phrase, color, template=EVAL_TEMPLATE):
    return template.format(a=article(color), color=color, x=region_reference(span, phrase))


def procedural_prompt(operation, span, phrase=None):
    return PROCEDURAL_TEMPLATES[operation].format(x=region_reference(span, phrase))


def normalize_phrase(reference, *, legacy=False):
    """Annotation referring text -> a noun phrase that reads well before a mask token.

    ``legacy=True`` reproduces the first evaluation set (E-A1), which kept a leading article's case and stripped
    only a trailing period.
    """
    if not reference:
        return None
    text = " ".join(str(reference).split())
    text = text.rstrip(".") if legacy else text.strip(" .,:;")
    if not text or len(text) > 160:
        return None
    first = text.split()[0]
    if first.lower() in _LEADING:
        return text if legacy else text[0].lower() + text[1:]
    return "the " + (text[0].lower() + text[1:] if first[0].isupper() and not first.isupper() else text)


def resize_mask(mask, size, *, soft=True):
    """Boolean mask -> float mask in [0, 1] at ``size`` = (width, height)."""
    image = Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255)
    return np.asarray(image.resize(size, Image.BILINEAR if soft else Image.NEAREST), np.float32) / 255.0


def overlay(rgb, weight, color, alpha=OVERLAY_ALPHA):
    """Alpha-blend ``color`` over a float RGB image where ``weight`` (soft mask) is set."""
    paint = np.asarray(color, np.float32)[None, None]
    return rgb * (1 - alpha * weight[..., None]) + paint * (alpha * weight[..., None])


def procedural_edit(rgb, weight, operation):
    """Exact local edit of a float RGB image inside the soft mask ``weight``."""
    image = Image.fromarray(rgb.clip(0, 255).astype(np.uint8))
    if operation == "grayscale":
        edited = np.asarray(image.convert("L").convert("RGB"), np.float32)
    elif operation == "brighter":
        edited = (rgb * 1.6 + 25).clip(0, 255)
    elif operation == "darker":
        edited = rgb * 0.45
    elif operation == "blur":
        radius = max(4.0, 0.012 * math.hypot(*image.size))
        edited = np.asarray(image.filter(ImageFilter.GaussianBlur(radius)), np.float32)
    elif operation == "invert":
        edited = 255.0 - rgb
    else:
        raise ValueError(f"Unknown procedural operation: {operation}")
    return rgb * (1 - weight[..., None]) + edited * weight[..., None]


def render_target(prepared_rgb, mask, row):
    """Target image of a highlight or procedural row, built on the resized source the model is conditioned on."""
    rgb = np.asarray(prepared_rgb.convert("RGB"), np.float32)
    weight = resize_mask(mask, prepared_rgb.size)
    if row["kind"] == "hl":
        out = overlay(rgb, weight, PALETTE[row["color"]])
    elif row["kind"] == "proc":
        out = procedural_edit(rgb, weight, row["op"])
    else:
        raise ValueError(f"Row kind {row['kind']!r} has a real target image")
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def pick_color(source_rgb):
    """The read-back colour farthest from the image's mean colour."""
    mean = np.asarray(source_rgb.resize((64, 64)), np.float32).reshape(-1, 3).mean(0)
    return max(READBACK_COLORS, key=lambda name: float(np.linalg.norm(np.asarray(PALETTE[name], np.float32) - mean)))


def extract_mask(source_rgb, output_rgb, color, threshold=0.5):
    """Recover the highlighted region from a generated image.

    For an exact overlay ``out = src + alpha (color - src)``; projecting ``out - src`` on ``color - src`` gives alpha,
    scaled here so a 50% overlay reads 1.  Pixels already close to the colour carry no signal and read 0.
    """
    source = np.asarray(source_rgb, np.float32) / 255.0
    output = np.asarray(output_rgb.resize(source_rgb.size, Image.BILINEAR), np.float32) / 255.0
    direction = np.asarray(color, np.float32)[None, None] / 255.0 - source
    energy = (direction * direction).sum(-1)
    alpha = ((output - source) * direction).sum(-1) / np.maximum(energy, 1e-3) / OVERLAY_ALPHA
    alpha[energy < 0.02] = 0.0
    smooth = Image.fromarray((alpha.clip(0, 1) * 255).astype(np.uint8)).filter(ImageFilter.BoxBlur(2))
    return np.asarray(smooth, np.float32) / 255.0 > threshold


def iou(a, b):
    a, b = np.asarray(a) > 0, np.asarray(b) > 0
    return float((a & b).sum()) / max(float((a | b).sum()), 1.0)


# ---------------------------------------------------------------- texture-preserving tint (v5 highlights)
# The 50 % overlay above paints a flat, fully saturated colour over the region (half the luminance contrast is lost), and
# a model trained on it learns flat recolouring.  The tint keeps the region's OKLab lightness (contrast compressed to
# 75 % around mid-grey so dark and bright pixels can still carry a hue), sets the hue of the named colour with a moderate
# chroma that is highest in the mid-tones, and lowers the chroma where the result would leave the sRGB gamut.
HIGHLIGHT_STYLES = ("overlay", "tint")
_STYLE = {"name": "overlay"}
TINT_CHROMA = 0.13
TINT_L_CENTER, TINT_L_KEEP, TINT_L_RANGE = 0.6, 0.75, (0.15, 0.92)
_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]], np.float32)
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]], np.float32)
_M1_INV = np.linalg.inv(_M1).astype(np.float32)
_M2_INV = np.linalg.inv(_M2).astype(np.float32)


def set_style(name):
    """Highlight style of this process (trainer / evaluator): ``overlay`` (v4) or ``tint`` (v5)."""
    if name not in HIGHLIGHT_STYLES:
        raise ValueError(f"Unknown highlight style {name!r}")
    _STYLE["name"] = name


def style():
    return _STYLE["name"]


def rgb_to_oklab(rgb01):
    rgb01 = np.asarray(rgb01, np.float32)
    linear = np.where(rgb01 <= 0.04045, rgb01 / 12.92, ((rgb01 + 0.055) / 1.055) ** 2.4)
    return np.cbrt(linear @ _M1.T) @ _M2.T


def oklab_to_rgb(lab):
    linear = ((np.asarray(lab, np.float32) @ _M2_INV.T) ** 3) @ _M1_INV.T
    linear = np.clip(linear, 0.0, None)
    return np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055)


_HUES = {}
for _name, _rgb in PALETTE.items():
    _lab = rgb_to_oklab(np.asarray(_rgb, np.float32)[None] / 255.0)[0]
    _HUES[_name] = float(np.arctan2(_lab[2], _lab[1]))


def tint(rgb, color):
    """Texture-preserving tint of a float RGB array in [0, 255] towards the named palette colour."""
    lab = rgb_to_oklab(np.asarray(rgb, np.float32) / 255.0)
    lightness = np.clip(TINT_L_CENTER + TINT_L_KEEP * (lab[..., 0] - TINT_L_CENTER), *TINT_L_RANGE)
    chroma = TINT_CHROMA * np.clip(1.0 - ((lightness - 0.62) / 0.5) ** 2, 0.35, 1.0)
    hue = _HUES[color]
    out = np.zeros(lightness.shape + (3,), np.float32)
    todo = np.ones(lightness.shape, bool)
    for scale in (1.0, 0.8, 0.6, 0.45, 0.3, 0.15, 0.0):
        c = chroma * scale
        candidate = oklab_to_rgb(np.stack([lightness, c * np.cos(hue), c * np.sin(hue)], -1))
        fits = todo & np.all((candidate >= -0.002) & (candidate <= 1.002), -1)
        out[fits] = candidate[fits]
        todo &= ~fits
        if not todo.any():
            break
    return np.clip(out, 0.0, 1.0) * 255.0


def tint_regions_rgb(rgb, weights, colors):
    """Each soft region weight tinted with its own colour (regions are disjoint); only the region's box is converted."""
    out = np.asarray(rgb, np.float32).copy()
    for weight, color in zip(weights, colors):
        ys, xs = np.nonzero(weight > 0)
        if ys.size == 0:
            continue
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        crop = out[y0:y1, x0:x1]
        w = weight[y0:y1, x0:x1, None]
        out[y0:y1, x0:x1] = crop * (1 - w) + tint(crop, color) * w
    return out


def extract_mask_style(source_rgb, output_rgb, color, threshold=0.5):
    """Read-back of a single highlight in the process's style (``color`` is the palette name)."""
    if style() == "overlay":
        return extract_mask(source_rgb, output_rgb, PALETTE[color], threshold)
    source = np.asarray(source_rgb, np.float32) / 255.0
    output = np.asarray(output_rgb.resize(source_rgb.size, Image.BILINEAR), np.float32) / 255.0
    direction = tint(source * 255.0, color) / 255.0 - source
    energy = (direction * direction).sum(-1)
    alpha = ((output - source) * direction).sum(-1) / np.maximum(energy, 1e-4)
    alpha[energy < TINT_MIN_ENERGY] = 0.0
    smooth = Image.fromarray((alpha.clip(0, 1) * 255).astype(np.uint8)).filter(ImageFilter.BoxBlur(2))
    return np.asarray(smooth, np.float32) / 255.0 > threshold


TINT_MIN_ENERGY = 0.004


def changed_mask(source_rgb, output_rgb, threshold=0.04):
    """Style-agnostic read-back: pixels whose mean absolute RGB change (lightly smoothed) exceeds ``threshold``."""
    source = np.asarray(source_rgb, np.float32) / 255.0
    output = np.asarray(output_rgb.resize(source_rgb.size, Image.BILINEAR), np.float32) / 255.0
    diff = np.abs(output - source).mean(-1)
    smooth = Image.fromarray((np.clip(diff / (2 * threshold), 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.BoxBlur(2))
    return np.asarray(smooth, np.float32) / 255.0 > 0.5
