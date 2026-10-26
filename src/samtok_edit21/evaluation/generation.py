"""Token-only generation: the source image and a prompt that carries region tokens, nothing else (no codec-decoded
region, no attention bias; latent blending only when a caller asks for it, as the teacher test does).  The canvas is
about 1 MP at the source aspect ratio on a 32 px grid, and the output is exported back at the source size."""
from __future__ import annotations

import math

import torch
from PIL import Image


CANVAS_AREA = 1024 * 1024


def canvas(image, area=CANVAS_AREA):
    """Generation size (height, width): about ``area`` pixels at the source aspect ratio, multiples of 32."""
    ratio = image.width / image.height
    return (max(32, round(math.sqrt(area / ratio) / 32) * 32), max(32, round(math.sqrt(area * ratio) / 32) * 32))


def export(image, size):
    """Model output -> RGB at the source size (white under transparency), as the benchmark scores it."""
    rgba = image.convert("RGBA")
    flat = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB")
    return flat.resize(size, Image.Resampling.LANCZOS)


@torch.no_grad()
def generate(pipe, codec, prompt, source_rgb, *, steps=20, seed=0, area=CANVAS_AREA, cfg_scale=1.0, negative_prompt=None,
             blend_region=None, blend_dilate=2, blend_feather=1.0):
    """Token-only inference: inline region tokens, raw text encoder, CFG 1 (or token guidance: ``cfg_scale`` with a
    token-free ``negative_prompt``).  No blending, unless ``blend_region`` (a pixel mask, teacher test only) asks to
    keep the source outside it.  ``codec`` is unused, kept so that every caller keeps its signature."""
    from samtok_edit21.models.pipeline import edit

    height, width = canvas(source_rgb, area)
    extra = {} if negative_prompt is None else {"negative_prompt": negative_prompt}
    image, _ = edit(pipe, prompt, [source_rgb.convert("RGBA")], mode="inline", height=height, width=width,
                    num_inference_steps=steps, cfg_scale=cfg_scale, seed=seed, use_kv_cache=True, pass2_te="raw",
                    blend_region=blend_region, blend_dilate=blend_dilate, blend_feather=blend_feather, **extra)
    return export(image, source_rgb.size), (height, width)


def thumbnail(image, side=768):
    scale = side / max(image.size)
    if scale >= 1:
        return image
    return image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
