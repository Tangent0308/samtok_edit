"""Round-2 objective (docs/08 section 5.3): one prompt may carry several mask tokens; supervision only in the loss.

A row is a highlight with one token (``hl``, ``hl_ref``), a highlight with several tokens (``hlm``, one colour each)
or an SA-1B 50k edit (``edit``: blind no-ref + token, ref + token, or ref without a token).  The model input is what
inference gives it: the source image and the prompt.  Regions enter only through targets and loss weights:

    loss = w(t) * (FM + lambda_region * FM_region + lambda_distractor * FM_distractor)
           + lambda_route * write + lambda_read * read

* FM_region      highlight: the error inside each token's region (one token of context), averaged per region and
                 then over the regions, so a small region counts as much as a large one; edit: inside the support.
* FM_distractor  the error on regions that must not change: other instances of the same category and the host of
                 a part (minus every highlighted region).
* write          each span: target tokens inside its region attend to it more than those outside; negatives are the
                 other spans' regions and the distractors (3x).  Edits use the source mask plus the changed pixels.
* read           each span: its attention to the source-image tokens is higher inside the region the token names.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from samtok_edit21.data.highlight import OVERLAY_ALPHA, PALETTE, resize_mask
from samtok_edit21.data.io import dimensions
from samtok_edit21.models.binding import token_layout
from samtok_edit21.models.pipeline import encode_edit, resize_sources
from samtok_edit21.preparation import instruction_protocol_v2 as p2
from samtok_edit21.preparation import round2_data as rd
from samtok_edit21.training.grounding import DISTRACTOR_COVERAGE, grid_coverage
from samtok_edit21.training.objectives import validate_conditioning
from samtok_edit21.training.routing import DISTRACTOR_WEIGHT

MAX_PIXELS = 1024 * 1024


@dataclass
class Round2Sample:
    inputs: dict
    kind: str
    regions: list                       # [h, w] coverage per span (target grid); for edits: the support
    distractors: torch.Tensor           # [h, w] coverage that must stay unchanged
    spans: list = field(default_factory=list)          # joint positions of each span
    route_regions: list = field(default_factory=list)  # [h, w] write-side positive region per span (target grid)
    source_positions: list = field(default_factory=list)
    source_regions: list = field(default_factory=list)  # [hs, ws] region per span on the source grid


class Context:
    """Lookups shared by all rows: the mask pool, the 50k records and their tokens (loaded once per rank)."""

    def __init__(self, data_dir):
        self.data = Path(data_dir)
        self.release = {r["sample_id"]: r for r in map(json.loads, open(rd.RELEASE / "annotations/train_protocol_v2.jsonl"))}
        self.tokens = {t["sample_id"]: t for t in map(json.loads, open(self.data / "edit_tokens.jsonl"))}
        self._pool = lru_cache(maxsize=256)(self._load)

    def _load(self, source_id):
        return json.load(open(self.data / "pool" / f"{source_id}.json"))

    def pool(self, source_id):
        return self._pool(source_id)


def span_positions_all(tokenizer, prompt, conditioning, count):
    """Joint-sequence positions of every mask span, in prompt order (each expands one TE position to one DiT token)."""
    ids = conditioning["prompt_input_ids"][0]
    layout = token_layout(tokenizer, ids[ids >= 0], prompt)
    units = layout["units"]
    if len(units) != count or any(end - start != 4 for start, end in units):
        raise ValueError(f"expected {count} separate four-token spans, found {units}")
    repeats = torch.where(conditioning["edit_image_pad_mask"][0], 4, 1)
    offsets = (repeats.cumsum(0) - repeats).tolist()
    return [[offsets[index] for index in range(start, end)] for start, end in units]


def source_token_positions(conditioning, grid):
    """Joint positions of the (single) source image's tokens, raster order of its latent grid."""
    pad = conditioning["edit_image_pad_mask"][0]
    repeats = torch.where(pad, 4, 1)
    offsets = (repeats.cumsum(0) - repeats).tolist()
    positions = [offsets[i] + j for i in torch.nonzero(pad).flatten().tolist() for j in range(4)]
    if len(positions) != grid[0] * grid[1]:
        raise ValueError(f"{len(positions)} source tokens for a {tuple(grid)} source grid")
    return positions


def overlay_regions(prepared_rgb, regions_px, colors):
    """The highlight target: each region alpha-blended with its own colour (regions are disjoint)."""
    rgb = np.asarray(prepared_rgb.convert("RGB"), np.float32)
    out = rgb.copy()
    for region, color in zip(regions_px, colors):
        weight = resize_mask(region, prepared_rgb.size)[..., None] * OVERLAY_ALPHA
        out = out * (1 - weight) + np.asarray(PALETTE[color], np.float32)[None, None] * weight
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def highlight_prompt(row, spans, entry):
    if row["kind"] == "hlm":
        return rd.multi_prompt(spans, row["colors"], row["template"])
    return rd.highlight_prompt_r2(spans[0], row["colors"][0], row.get("phrase"), row["template"])


def edit_prompt(row, record, span):
    spec = record["instruction_spec"]
    if row["form"] == "ref_only":
        return record["instruction"]
    if row["form"] == "ref_token":
        return p2.render(spec, True, selector_token=span)
    return p2.render(spec, False, selector_token=span)


def load_mask(path, size=None):
    with Image.open(path) as im:
        mask = np.asarray(im.convert("L")) > 0
    return mask


def prepare_round2(pipe, ctx, row, *, max_pixels=MAX_PIXELS):
    """Conditioning, target latent and every region map of one row (no gradients outside the code injector)."""
    if row["kind"] == "edit":
        return _prepare_edit(pipe, ctx, row, max_pixels)
    entry = ctx.pool(row["source_id"])
    with Image.open(entry["image"]) as im:
        source = im.convert("RGBA")
    masks = [entry["masks"][i] for i in row["masks"]]
    spans = [m["span"] for m in masks]
    regions_px = [rd.decode_rle(m["dec"]) for m in masks]
    union = np.zeros_like(regions_px[0])
    for region in regions_px:
        union |= region
    negative_px = np.zeros_like(union)
    for group in row["negatives"]:
        for j in group:
            negative_px |= rd.decode_rle(entry["masks"][j]["dec"])
    negative_px &= ~union
    prompt = highlight_prompt(row, spans, entry)
    with torch.no_grad():
        width, height = dimensions(source, max_pixels)
        prepared = resize_sources(pipe, [source], height, width)
        size = prepared[0].size
        target = overlay_regions(prepared[0], regions_px, row["colors"]).convert("RGBA")
        target_latent = pipe.vae.encode(pipe.preprocess_image(target))
        source_latents = [pipe.vae.encode(pipe.preprocess_image(image)) for image in prepared]
    conditioning = encode_edit(pipe, prompt, prepared, return_ids=True)
    span_pos = span_positions_all(pipe.processor.tokenizer, prompt, conditioning, len(spans))
    grid, source_grid = target_latent.shape[-2:], source_latents[0].shape[-2:]
    source_pos = source_token_positions(conditioning, source_grid)
    conditioning.pop("prompt_input_ids")
    inputs = {"input_latents": target_latent, "edit_latents": source_latents, **conditioning}
    validate_conditioning({k: (v.detach() if torch.is_tensor(v) else v) for k, v in inputs.items()})
    device = target_latent.device
    regions = [grid_coverage(resize_mask(r, size), grid).to(device) for r in regions_px]
    source_size = prepared[0].size
    return Round2Sample(
        inputs=inputs, kind=row["kind"], regions=regions,
        distractors=grid_coverage(resize_mask(negative_px, size), grid).to(device),
        spans=span_pos, route_regions=regions, source_positions=source_pos,
        source_regions=[grid_coverage(resize_mask(r, source_size), source_grid).to(device) for r in regions_px])


def _prepare_edit(pipe, ctx, row, max_pixels):
    record = ctx.release[row["sample_id"]]
    token = ctx.tokens[row["sample_id"]]
    root = rd.RELEASE
    with Image.open(root / record["source_image"]) as im:
        source = im.convert("RGBA")
    with Image.open(root / record["edited_image"]) as im:
        edited = im.convert("RGBA")
    support_px = load_mask(root / record["support_mask"])
    s_px = load_mask(root / record["mask"])
    changed = record["edited_region"]
    changed_px = rd.decode_rle({"size": changed["rle_size"], "counts": changed["rle_counts"]}) if changed.get("rle_counts") \
        else np.zeros_like(s_px)
    span = None if row["form"] == "ref_only" else token["span"]
    prompt = edit_prompt(row, record, span)
    with torch.no_grad():
        width, height = dimensions(edited, max_pixels)
        target = edited.resize((width, height), Image.Resampling.LANCZOS)
        prepared = resize_sources(pipe, [source], height, width)
        target_latent = pipe.vae.encode(pipe.preprocess_image(target))
        source_latents = [pipe.vae.encode(pipe.preprocess_image(image)) for image in prepared]
    conditioning = encode_edit(pipe, prompt, prepared, return_ids=True)
    grid, source_grid = target_latent.shape[-2:], source_latents[0].shape[-2:]
    device = target_latent.device
    size = (width, height)
    sample = Round2Sample(inputs={}, kind="edit", regions=[grid_coverage(resize_mask(support_px, size), grid).to(device)],
                          distractors=torch.zeros(tuple(grid), device=device))
    if span is not None:
        sample.spans = span_positions_all(pipe.processor.tokenizer, prompt, conditioning, 1)
        sample.source_positions = source_token_positions(conditioning, source_grid)
        sample.route_regions = [grid_coverage(resize_mask(s_px | changed_px, size), grid).to(device)]
        token_px = rd.decode_rle(token["dec"])
        sample.source_regions = [grid_coverage(resize_mask(token_px, prepared[0].size), source_grid).to(device)]
    conditioning.pop("prompt_input_ids")
    sample.inputs = {"input_latents": target_latent, "edit_latents": source_latents, **conditioning}
    validate_conditioning({k: (v.detach() if torch.is_tensor(v) else v) for k, v in sample.inputs.items()})
    return sample


def balanced(labels, heavy):
    """Positive and negative halves of the weight; negatives on ``heavy`` count DISTRACTOR_WEIGHT times."""
    negative = 1 - labels
    weight_neg = negative * (1 + (DISTRACTOR_WEIGHT - 1) * heavy)
    weights = torch.zeros_like(labels)
    if labels.sum() > 0:
        weights = weights + labels * (0.5 / labels.sum())
    if weight_neg.sum() > 0:
        weights = weights + weight_neg * (0.5 / weight_neg.sum())
    return weights


def token_labels(coverage):
    """Tokens more than half covered; a region smaller than any token is represented by its best-covered token."""
    flat = coverage.flatten()
    labels = (flat > 0.5).float()
    if labels.sum() == 0 and float(flat.max()) > 0:
        labels[int(torch.argmax(flat))] = 1.0
    return labels


def route_entries(sample):
    """Router entries: per span, write-side labels on the target grid and read-side labels on the source grid.

    Negatives that count DISTRACTOR_WEIGHT times: the distractors and, with several spans, the other spans' regions.
    """
    target = [token_labels(r) for r in sample.route_regions]
    source = [token_labels(r) for r in sample.source_regions]
    distractor = (sample.distractors > 0.5).float().flatten()
    entries = []
    for k, span in enumerate(sample.spans):
        heavy = distractor.clone()
        for j, other in enumerate(target):
            if j != k:
                heavy = torch.maximum(heavy, other)
        entry = {"span": span, "labels": target[k], "weights": balanced(target[k], heavy)}
        if sample.source_positions:
            src_heavy = torch.zeros_like(source[k])
            for j, other in enumerate(source):
                if j != k:
                    src_heavy = torch.maximum(src_heavy, other)
            entry.update(source_positions=sample.source_positions, source_labels=source[k],
                         source_weights=balanced(source[k], src_heavy))
        entries.append(entry)
    return entries


def region_error(squared_error, regions, dilate=True):
    """Mean of the per-region normalised errors (each region with one token of context when ``dilate``)."""
    terms = []
    for region in regions:
        weight = F.max_pool2d(region[None, None].float(), 3, 1, 1)[0, 0] if dilate else region.float()
        if float(weight.sum()) > 0:
            terms.append((squared_error * weight).sum() / weight.sum())
    return torch.stack(terms).mean() if terms else squared_error.mean().detach() * 0


def round2_loss(pipe, sample, router=None, *, lambda_region=1.0, lambda_distractor=1.0, timestep_index=None,
                noise=None, checkpointing=True):
    """Returns ``(flow, write, read, metrics)``; the caller adds ``lambda_route * write + lambda_read * read``.
    With a router, call ``router.disarm()`` after backward (gradient checkpointing re-runs the hooks)."""
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

    write = read = None
    if router is None or not sample.spans:
        prediction = predict()
    else:
        router.arm_entries(sample.regions[0].numel(), route_entries(sample))
        with router.recording():
            prediction = predict()
        write, read = router.collect(), router.collect_read()
    if prediction.shape != target.shape:
        raise RuntimeError("FM prediction/target shape mismatch")
    squared_error = (prediction.float() - target.float()).pow(2).mean(1)[0]
    mean = squared_error.mean()
    in_region = region_error(squared_error, sample.regions, dilate=sample.kind != "edit")
    other = (sample.distractors > DISTRACTOR_COVERAGE).float()
    on_distractors = (squared_error * other).sum() / other.sum() if float(other.sum()) > 0 else mean.detach() * 0
    weight = pipe.scheduler.training_weight(timestep).to(pipe.device)
    flow = (mean + lambda_region * in_region + lambda_distractor * on_distractors) * weight
    metrics = {"mse": float(mean.detach()), "reg": float(in_region.detach()), "dis": float(on_distractors.detach()),
               "t": float(timestep), "write": float(write.detach()) if write is not None else None,
               "read": float(read.detach()) if read is not None else None}
    return flow, write, read, metrics
