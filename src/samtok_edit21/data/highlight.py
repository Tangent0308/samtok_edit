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
