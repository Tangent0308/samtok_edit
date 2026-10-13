"""Stage A objective (v3): teach the DiT what a mask token in the prompt refers to.

One training row is one region token on one image (see ``data.highlight`` for the row kinds).  The region is whatever
the SAMTok codec decodes from the token; it is used here only to build targets and loss weights.  The model input is
the same as at inference: source image + prompt with the token.

    loss = w(t) * ( FM  +  lambda_region * FM_region  +  lambda_distractor * FM_distractor )  +  lambda_route * routing

* FM            - the official flow-matching loss (same timestep sampling, target and weight as ``objectives.flow_loss``);
* FM_region     - the same squared error averaged over the region only, so a small region counts as much as a large one;
* FM_distractor - the squared error on *other* instances of the image, which must come out unchanged;
* routing       - ``training.routing``: target tokens inside the region attend to the token, others do not.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from samtok_edit21.data.highlight import render_target, resize_mask
from samtok_edit21.data.io import dimensions
from samtok_edit21.models.binding import token_layout
from samtok_edit21.models.code_injector import (
    CodeInjector, attach_code_injector, has_code_injector, load_code_injector,
)
from samtok_edit21.models.pipeline import encode_edit, resize_sources
from samtok_edit21.training.objectives import load_adapter, validate_conditioning
from samtok_edit21.training.routing import route_targets

MAX_PIXELS = 1024 * 1024
MAX_DISTRACTORS = 3
LATENT_STRIDE = 16           # one target token covers 16 x 16 pixels
DISTRACTOR_COVERAGE = 0.25   # a token counts as "on another instance" above this coverage


@dataclass
class GroundingSample:
    inputs: dict                 # conditioning for pipe.model_fn: input_latents, edit_latents, prompt_embeds, ...
    region: torch.Tensor         # [h, w] coverage of the token's region on the target token grid, in [0, 1]
    distractors: torch.Tensor    # [h, w] coverage of the other instances (region excluded)
    span: list                   # joint-sequence positions of the token's four tokens


def load_grounding_adapter(pipe, codec, directory, *, trainable=False):
    """Load a v3 adapter directory: the DiT LoRA and, when the run trained one, the code injector.

    Returns ``(adapter config, injector or None)``.  After this call ``edit(pipe, prompt_with_tokens, ...,
    mode="inline")`` is the complete token-only inference path.
    """
    config = load_adapter(pipe.dit, directory, trainable=trainable)
    injector = None
    if has_code_injector(directory):
        injector = attach_code_injector(pipe, CodeInjector.from_codec(codec, pipe.processor.tokenizer))
        load_code_injector(injector, directory)
        injector.requires_grad_(trainable)
    return config, injector


def resolve_image(path, data_root=None):
    """Rows store absolute paths, or paths relative to the data directory for images the builder extracted."""
    path = Path(path)
    return path if path.is_absolute() or data_root is None else Path(data_root) / path


def grid_coverage(weight, grid):
    """Pixel soft mask -> fraction of each target token covered."""
    coverage = F.avg_pool2d(torch.as_tensor(weight)[None, None], LATENT_STRIDE, LATENT_STRIDE)[0, 0].clamp(0, 1)
    if tuple(coverage.shape) != tuple(grid):
        raise ValueError(f"Region grid {tuple(coverage.shape)} does not match the target latent {tuple(grid)}")
    return coverage


def span_token_positions(tokenizer, prompt, conditioning):
    """Positions of the single mask span in the DiT's joint sequence.

    ``prompt_input_ids`` index the trimmed text-encoder sequence; every image placeholder token there expands to four
    DiT tokens (``QwenImage21DiT.forward``), all of which precede the instruction.
    """
    ids = conditioning["prompt_input_ids"][0]
    layout = token_layout(tokenizer, ids[ids >= 0], prompt)
    if len(layout["units"]) != 1 or layout["units"][0][1] - layout["units"][0][0] != 4:
        raise ValueError("A grounding row carries exactly one mask-token span")
    start, end = layout["units"][0]
    repeats = torch.where(conditioning["edit_image_pad_mask"][0], 4, 1)
    offsets = (repeats.cumsum(0) - repeats).tolist()
    return [offsets[index] for index in range(start, end)]


def prepare_sample(pipe, codec, row, *, data_root=None, max_pixels=MAX_PIXELS):
    """Conditioning, target latent and region maps for one row.

    Everything runs without gradients except the prompt encoding, where the only trainable path is the code injector
    (the text encoder itself stays frozen).
    """
    with Image.open(resolve_image(row["edit_image"], data_root)) as image:
        source = image.convert("RGBA")
    with torch.no_grad():
        region_px = codec.decode_strict(source, row["span"])[0] > 0
        others = [codec.decode_strict(source, span)[0] > 0 for span in row.get("others", [])[:MAX_DISTRACTORS]]
    distractor_px = np.zeros_like(region_px)
    for other in others:
        distractor_px |= other
    distractor_px &= ~region_px
    with torch.no_grad():
        if row["kind"] == "real":
            with Image.open(resolve_image(row["image"], data_root)) as image:
                edited = image.convert("RGBA")
            width, height = dimensions(edited, max_pixels)
            target = edited.resize((width, height), Image.Resampling.LANCZOS)
            prepared = resize_sources(pipe, [source], height, width)
            size = (width, height)
        else:
            width, height = dimensions(source, max_pixels)
            prepared = resize_sources(pipe, [source], height, width)
            size = prepared[0].size
            target = render_target(prepared[0], region_px, row).convert("RGBA")
        target_latent = pipe.vae.encode(pipe.preprocess_image(target))
        source_latents = [pipe.vae.encode(pipe.preprocess_image(image)) for image in prepared]
    conditioning = encode_edit(pipe, row["prompt"], prepared, return_ids=True)
    span = span_token_positions(pipe.processor.tokenizer, row["prompt"], conditioning)
    conditioning.pop("prompt_input_ids")
    inputs = {"input_latents": target_latent, "edit_latents": source_latents, **conditioning}
    validate_conditioning({k: (v.detach() if torch.is_tensor(v) else v) for k, v in inputs.items()})
    grid = target_latent.shape[-2:]
    device = target_latent.device
    return GroundingSample(
        inputs=inputs,
        region=grid_coverage(resize_mask(region_px, size), grid).to(device),
        distractors=grid_coverage(resize_mask(distractor_px, size), grid).to(device),
        span=span,
    )


def region_terms(squared_error, region, distractors):
    """Region-normalised and distractor-normalised means of a per-token squared error map."""
    mean = squared_error.mean()
    weight = F.max_pool2d(region[None, None].float(), 3, 1, 1)[0, 0]   # one token of context around the region
    in_region = (squared_error * weight).sum() / weight.sum() if float(weight.sum()) > 0 else mean.detach() * 0
    other = (distractors > DISTRACTOR_COVERAGE).float()
    on_distractors = (squared_error * other).sum() / other.sum() if float(other.sum()) > 0 else mean.detach() * 0
    return mean, in_region, on_distractors


def grounding_loss(pipe, sample, router=None, *, lambda_region=1.0, lambda_distractor=1.0,
                   timestep_index=None, noise=None, checkpointing=True):
    """Returns ``(flow, routing, metrics)``; the caller forms ``flow + lambda_route * routing``.

    With a ``router`` the caller must call ``router.disarm()`` after ``backward`` (see ``AttentionRouter``).
    ``timestep_index`` and ``noise`` make the loss repeatable for tests; by default they are sampled exactly like
    the official ``FlowMatchSFTLoss``.
    """
    inputs = sample.inputs
    index = (torch.randint(0, len(pipe.scheduler.timesteps), (1,)) if timestep_index is None
             else torch.tensor([timestep_index]))
    timestep = pipe.scheduler.timesteps[index].to(device=pipe.device, dtype=pipe.torch_dtype)
    clean = inputs["input_latents"]
    noise = torch.randn_like(clean) if noise is None else noise
    noisy = pipe.scheduler.add_noise(clean, noise, timestep)
    target = pipe.scheduler.training_target(clean, noise, timestep)

    def predict():
        return pipe.model_fn(dit=pipe.dit, latents=noisy, timestep=timestep, **inputs, kv_cache=None,
                             use_gradient_checkpointing=checkpointing, region_binding=None)

    routing = None
    if router is None:
        prediction = predict()
    else:
        labels, weights = route_targets(sample.region, sample.distractors)
        router.arm(sample.region.numel(), sample.span, labels.to(clean.device), weights.to(clean.device))
        with router.recording():
            prediction = predict()
        routing = router.collect()
    if prediction.shape != target.shape:
        raise RuntimeError("FM prediction/target shape mismatch")
    squared_error = (prediction.float() - target.float()).pow(2).mean(1)[0]
    mean, in_region, on_distractors = region_terms(squared_error, sample.region, sample.distractors)
    weight = pipe.scheduler.training_weight(timestep).to(pipe.device)
    flow = (mean + lambda_region * in_region + lambda_distractor * on_distractors) * weight
    metrics = {"mse": float(mean.detach()), "reg": float(in_region.detach()), "dis": float(on_distractors.detach()),
               "t": float(timestep), "route": float(routing.detach()) if routing is not None else 0.0}
    return flow, routing, metrics
