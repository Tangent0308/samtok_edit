"""Round-2 objective: one prompt may carry several mask tokens; regions supervise only through the loss.

A row is a highlight with one token (``hl``, ``hl_ref``), a highlight with several tokens (``hlm``, one colour each)
or an edit: an SA-1B 50k edit (``edit``: blind no-ref + token, ref + token, or ref without a token), a cleaned v1 edit
(``edit_v1``), a cleaned v1 multi-object pair (``edit_multi_v1``, its real joint target) or 2-4 non-overlapping 50k
edits of one image joined into one multi-target edit (``edit_multi``); stage C adds the extedit batch-1 records and
pairs (``edit_ext``, ``edit_multi_ext``: same format as the cleaned v1 ones); ``edit_merged`` is any record of the
merged dataset (docs/01: single or multi-unit, token inline; SA-1B records may carry near negatives and twins).  A multi-target prompt has one numbered line and one token per unit.  The model input is what inference
gives it: the source image and the prompt.
Regions enter only through targets and loss weights:

    loss = w(t) * (FM + lambda_region * FM_region + lambda_distractor * FM_distractor)
           + lambda_route * write + lambda_read * read

* FM_region      highlight: the error inside each token's region (one token of context), averaged per region and
                 then over the regions, so a small region counts as much as a large one; edit: inside the support,
                 or (stage-B rows, ``region: changed``) inside the pixels that really changed, one token of context.
                 ``edit_region_dilate=False`` (stage C, method variant) drops that context for every edit row.
  Rows whose target also changes the image outside the edit (``loss_region: local``) take FM only near the edit.
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
from samtok_edit21.data.protocol import token_layout
from samtok_edit21.models.pipeline import encode_edit, resize_sources
from samtok_edit21.preparation import instruction_protocol_v2 as p2
from samtok_edit21.preparation import round2_data as rd
from samtok_edit21.training.routing import DISTRACTOR_WEIGHT

MAX_PIXELS = 1024 * 1024
LATENT_STRIDE = 16           # one target token covers 16 x 16 pixels
DISTRACTOR_COVERAGE = 0.25   # a token counts as "on another instance" above this coverage
EDIT_KINDS = ("edit", "edit_v1", "edit_multi", "edit_multi_v1", "edit_ext", "edit_multi_ext", "edit_merged")


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
    source_distractors: torch.Tensor | None = None      # [hs, ws] negatives on the source grid (C1b rows; read side)
    region_dilate: bool | None = None                   # FM_region context; None: highlights yes, edits no
    loss_mask: torch.Tensor | None = None                # [h, w] where FM is taken (rows with a drifting target)


def grid_coverage(weight, grid):
    """Pixel soft mask -> fraction of each target token covered."""
    coverage = F.avg_pool2d(torch.as_tensor(weight)[None, None], LATENT_STRIDE, LATENT_STRIDE)[0, 0].clamp(0, 1)
    if tuple(coverage.shape) != tuple(grid):
        raise ValueError(f"Region grid {tuple(coverage.shape)} does not match the target latent {tuple(grid)}")
    return coverage


def validate_conditioning(inputs):
    target, embeds, mask = (
        inputs["input_latents"],
        inputs["prompt_embeds"],
        inputs["edit_image_pad_mask"],
    )
    if target.ndim != 4 or target.shape[:2] != (1, 64):
        raise ValueError("Expected target latent [1,64,H/16,W/16]")
    if not target.is_floating_point() or not embeds.is_floating_point():
        raise ValueError("Latents and text features must be floating point")
    if embeds.ndim != 3 or embeds.shape[0] != 1 or embeds.shape[-1] != 4096:
        raise ValueError("Expected 4096-dim text features")
    if mask.dtype != torch.bool or mask.shape != embeds.shape[:2]:
        raise ValueError("Invalid Qwen3 image-pad mask")
    attention = inputs.get("prompt_embeds_mask")
    if attention is not None and (
        attention.shape != embeds.shape[:2]
        or not torch.all((attention == 0) | (attention == 1))
        or torch.any(mask & ~attention.bool())
    ):
        raise ValueError("Invalid text attention mask")
    if not isinstance(inputs["edit_latents"], (list, tuple)) or not inputs["edit_latents"]:
        raise ValueError("Missing source latents")
    for source in inputs["edit_latents"]:
        if not source.is_floating_point():
            raise ValueError("Source latents must be floating point")
        if source.ndim != 4 or source.shape[:2] != (1, 64) or any(
            size < 2 or size % 2 for size in source.shape[2:]
        ):
            raise ValueError("Expected source latent [1,64,even H,even W]")
    if any(size < 2 or size % 2 for size in target.shape[2:]):
        raise ValueError("Target latent grid must be positive and even")
    source_tokens = sum(x.shape[2] * x.shape[3] for x in inputs["edit_latents"])
    if int(mask.sum()) * 4 != source_tokens:
        raise ValueError(
            "Qwen3 visual grid and VAE grid disagree; do not independently resize TE images"
        )
    padded = torch.nn.functional.pad(mask[0].to(torch.int8), (1, 1))
    boundaries = padded[1:] - padded[:-1]
    starts = (boundaries == 1).nonzero().flatten()
    ends = (boundaries == -1).nonzero().flatten()
    grids = [x.shape[2] * x.shape[3] for x in inputs["edit_latents"]]
    if ((ends - starts) * 4).tolist() != grids:
        raise ValueError("Per-image Qwen3 visual blocks and VAE grids disagree")

    def finite(x):
        if (
            isinstance(x, torch.Tensor)
            and x.is_floating_point()
            and not torch.isfinite(x).all()
        ):
            raise ValueError("Nonfinite conditioning")
        if isinstance(x, dict):
            for v in x.values():
                finite(v)
        if isinstance(x, (list, tuple)):
            for v in x:
                finite(v)

    finite(inputs)


class Context:
    """Lookups shared by all rows: the mask pool, the 50k records and their tokens (loaded once per rank)."""

    def __init__(self, data_dir):
        self.data = Path(data_dir)
        self.release = {r["sample_id"]: r for r in map(json.loads, open(rd.RELEASE / "annotations/train_protocol_v2.jsonl"))}
        self.tokens = {t["sample_id"]: t for t in map(json.loads, open(self.data / "edit_tokens.jsonl"))}
        self._pool = lru_cache(maxsize=256)(self._load)
        self._v1 = self._d4 = self._ext = self._ext_multi = self._merged = None
        self.merged_ids = None          # set by the trainer: the merged records its rows use (others are not loaded)

    def d4(self, pair_id):
        """A cleaned v1 multi-object pair and its units' tokens (loaded on first use)."""
        if self._d4 is None:
            tokens = {t["id"]: t for t in map(json.loads, open(self.data / "d4_tokens.jsonl"))}
            self._d4 = {p["id"]: (p, tokens[p["id"]]) for p in map(json.loads, open(rd.D4_MANIFEST)) if p["id"] in tokens}
        return self._d4[pair_id]

    def v1(self, record_id):
        """A cleaned v1 record and its token (loaded on first use)."""
        if self._v1 is None:
            tokens = {t["id"]: t for t in map(json.loads, open(self.data / "v1_tokens.jsonl"))}
            self._v1 = {r["id"]: (r, tokens[r["id"]]) for r in map(json.loads, open(rd.V1_MANIFEST)) if r["id"] in tokens}
        return self._v1[record_id]

    def ext(self, record_id):
        """An extedit batch-1 record and its token (the cache ships with the dataset; loaded on first use)."""
        if self._ext is None:
            tokens = {t["id"]: t for t in map(json.loads, open(rd.EXT_ROOT / "tokens.jsonl"))}
            self._ext = {r["id"]: (r, tokens[r["id"]]) for r in map(json.loads, open(rd.EXT_MANIFEST)) if r["id"] in tokens}
        return self._ext[record_id]

    def ext_multi(self, pair_id):
        """An extedit batch-1 multi-unit pair and its units' tokens (loaded on first use)."""
        if self._ext_multi is None:
            tokens = {t["id"]: t for t in map(json.loads, open(rd.EXT_ROOT / "multi_tokens.jsonl"))}
            self._ext_multi = {p["id"]: (p, tokens[p["id"]]) for p in map(json.loads, open(rd.EXT_MULTI_MANIFEST))
                               if p["id"] in tokens}
        return self._ext_multi[pair_id]

    def merged(self, record_id):
        """A record of the merged dataset (``records.jsonl``, loaded on first use: only ``merged_ids`` when set; the
        id is read from the start of each line, so the other records are not parsed)."""
        if self._merged is None:
            self._merged = {}
            for line in open(rd.MERGED_ROOT / "records.jsonl"):
                rid = line[8:line.index('"', 8)]          # every record starts with {"id": "...
                if self.merged_ids is None or rid in self.merged_ids:
                    self._merged[rid] = json.loads(line)
        return self._merged[record_id]

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


def rle_region(field, shape):
    """A v1 RLE region (``rle_size``/``rle_counts``) at ``shape``; None when the record has none."""
    if not field or not field.get("rle_counts"):
        return None
    mask = rd.decode_rle({"size": field["rle_size"], "counts": field["rle_counts"]}).astype(bool)
    if mask.shape != shape:
        mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((shape[1], shape[0]), Image.NEAREST)) > 0
    return mask


def prepare_round2(pipe, ctx, row, *, max_pixels=MAX_PIXELS):
    """Conditioning, target latent and every region map of one row (no gradients outside the code injector)."""
    if row["kind"] == "edit":
        return _prepare_edit(pipe, ctx, row, max_pixels)
    if row["kind"] == "edit_v1":
        return _prepare_record(pipe, "edit_v1", rd.V1_ROOT, *ctx.v1(row["id"]), row, max_pixels)
    if row["kind"] == "edit_ext":
        return _prepare_record(pipe, "edit_ext", rd.EXT_ROOT, *ctx.ext(row["id"]), row, max_pixels)
    if row["kind"] == "edit_multi":
        return _prepare_edit_multi(pipe, ctx, row, max_pixels)
    if row["kind"] == "edit_multi_v1":
        return _prepare_multi_record(pipe, "edit_multi_v1", rd.V1_ROOT, *ctx.d4(row["id"]), row, max_pixels)
    if row["kind"] == "edit_multi_ext":
        return _prepare_multi_record(pipe, "edit_multi_ext", rd.EXT_ROOT, *ctx.ext_multi(row["id"]), row, max_pixels)
    if row["kind"] == "edit_merged":
        return _prepare_merged(pipe, ctx, row, max_pixels)
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
    key = "rle" if row.get("neg_ann") else "dec"            # C1b: the annotated objects themselves
    for group in row["negatives"]:
        for j in group:
            negative_px |= rd.decode_rle(entry["masks"][j][key])
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
    source_distractors = (grid_coverage(resize_mask(negative_px, source_size), source_grid).to(device)
                          if row.get("neg_ann") else None)
    return Round2Sample(
        inputs=inputs, kind=row["kind"], regions=regions,
        distractors=grid_coverage(resize_mask(negative_px, size), grid).to(device),
        spans=span_pos, route_regions=regions, source_positions=source_pos,
        source_regions=[grid_coverage(resize_mask(r, source_size), source_grid).to(device) for r in regions_px],
        source_distractors=source_distractors)


LOCAL_CONTEXT = 2             # tokens around the edit that a local loss still covers


def _edit_sample(pipe, kind, source, target, prompt, *, loss_px, route_px, token_px, negative_px, read_negatives,
                 max_pixels, dilate=None, local_px=None):
    """Conditioning, latents and region maps of an edit whose prompt carries ``len(token_px)`` spans in order.
    ``local_px``: take FM only within LOCAL_CONTEXT tokens of these pixels."""
    with torch.no_grad():
        width, height = dimensions(target, max_pixels)
        target = target.resize((width, height), Image.Resampling.LANCZOS)
        prepared = resize_sources(pipe, [source], height, width)
        target_latent = pipe.vae.encode(pipe.preprocess_image(target))
        source_latents = [pipe.vae.encode(pipe.preprocess_image(image)) for image in prepared]
    conditioning = encode_edit(pipe, prompt, prepared, return_ids=True)
    grid, source_grid = target_latent.shape[-2:], source_latents[0].shape[-2:]
    device = target_latent.device
    size = (width, height)
    sample = Round2Sample(inputs={}, kind=kind, regions=[grid_coverage(resize_mask(px, size), grid).to(device) for px in loss_px],
                          distractors=grid_coverage(resize_mask(negative_px, size), grid).to(device), region_dilate=dilate)
    if local_px is not None:
        near = (grid_coverage(resize_mask(local_px, size), grid) > 0).float()[None, None]
        sample.loss_mask = (F.max_pool2d(near, 2 * LOCAL_CONTEXT + 1, 1, LOCAL_CONTEXT)[0, 0] > 0).float().to(device)
    if token_px:
        sample.spans = span_positions_all(pipe.processor.tokenizer, prompt, conditioning, len(token_px))
        sample.source_positions = source_token_positions(conditioning, source_grid)
        sample.route_regions = [grid_coverage(resize_mask(px, size), grid).to(device) for px in route_px]
        sample.source_regions = [grid_coverage(resize_mask(px, prepared[0].size), source_grid).to(device) for px in token_px]
        if read_negatives:
            sample.source_distractors = grid_coverage(resize_mask(negative_px, prepared[0].size), source_grid).to(device)
    conditioning.pop("prompt_input_ids")
    sample.inputs = {"input_latents": target_latent, "edit_latents": source_latents, **conditioning}
    validate_conditioning({k: (v.detach() if torch.is_tensor(v) else v) for k, v in sample.inputs.items()})
    return sample


def _release_regions(root, record):
    """Execution region, source mask and changed pixels of a 50k record."""
    support_px = load_mask(root / record["support_mask"])
    s_px = load_mask(root / record["mask"])
    changed = record["edited_region"]
    changed_px = rd.decode_rle({"size": changed["rle_size"], "counts": changed["rle_counts"]}) if changed.get("rle_counts") \
        else np.zeros_like(s_px)
    return support_px, s_px, changed_px.astype(bool)


def _prepare_edit(pipe, ctx, row, max_pixels):
    record = ctx.release[row["sample_id"]]
    token = ctx.tokens[row["sample_id"]]
    root = rd.RELEASE
    with Image.open(root / record["source_image"]) as im:
        source = im.convert("RGBA")
    with Image.open(root / record["edited_image"]) as im:
        edited = im.convert("RGBA")
    support_px, s_px, changed_px = _release_regions(root, record)
    span = None if row["form"] == "ref_only" else token["span"]
    prompt = edit_prompt(row, record, span)
    negative_px = np.zeros_like(s_px)
    if row.get("negatives"):
        entry = ctx.pool(record["source_id"])
        for j in row["negatives"]:
            negative_px |= rd.decode_rle(entry["masks"][j]["rle"])
    for partner in row.get("partners", ()):           # stage-B twins: the region the other token names
        negative_px |= load_mask(root / ctx.release[partner]["mask"])
    if row.get("negatives") or row.get("partners"):
        negative_px &= ~(support_px | s_px | changed_px)
    changed_only = row.get("region") == "changed" and changed_px.any()
    return _edit_sample(pipe, "edit", source, edited, prompt, loss_px=[changed_px if changed_only else support_px],
                        route_px=[s_px | changed_px], token_px=[] if span is None else [rd.decode_rle(token["dec"])],
                        negative_px=negative_px, read_negatives=bool(row.get("negatives") or row.get("partners")),
                        max_pixels=max_pixels, dilate=True if row.get("region") == "changed" else None)


def _prepare_record(pipe, kind, root, record, token, row, max_pixels):
    """A single-unit record in the D3 format (cleaned v1 ``edit_v1``, extedit ``edit_ext``); image paths relative to
    ``root``."""
    with Image.open(root / record["source_image"]) as im:
        source = im.convert("RGBA")
    with Image.open(root / record["target_image"]) as im:
        target = im.convert("RGBA")
    shape = (source.height, source.width)
    s_px = rle_region(record["selection"], shape)
    changed_px = rle_region(record.get("edited_region"), shape)
    execution_px = rle_region(record.get("execution_region"), shape)
    loss_px = next(m for m in (changed_px, execution_px, s_px) if m is not None and m.any())
    route_px = s_px | changed_px if changed_px is not None else s_px
    prompt = p2.render(record["instruction_spec"], row["form"] == "ref_token", selector_token=token["span"])
    local_px = None
    if row.get("loss_region") == "local":
        local_px = s_px.copy()
        for m in (changed_px, execution_px):
            if m is not None:
                local_px |= m
    return _edit_sample(pipe, kind, source, target, prompt, loss_px=[loss_px], route_px=[route_px],
                        token_px=[rd.decode_rle(token["dec"])], negative_px=np.zeros(shape, bool), read_negatives=False,
                        max_pixels=max_pixels, dilate=True, local_px=local_px)


def unit_change(changed, own):
    """The changed pixels that belong to one unit of a multi-object pair.  In 1,598 of the 4,066 pairs every unit
    carries the pair's whole changed region, so it is cut to the unit's own region grown by 1 % of the image
    diagonal; when nothing is left, the unit's own region stands in."""
    from scipy.ndimage import distance_transform_edt   # not cv2: it needs libxcb, which the cluster images lack
    radius = max(4, round(0.01 * float(np.hypot(*own.shape))))
    grown = distance_transform_edt(~own) <= radius
    part = changed & grown if changed is not None else None
    return part if part is not None and part.any() else own


def _prepare_multi_record(pipe, kind, root, pair, tokens, row, max_pixels):
    """A multi-unit pair in the D4 format (cleaned v1 ``edit_multi_v1``, extedit ``edit_multi_ext``): the target
    already holds every unit's edit."""
    with Image.open(root / pair["source_image"]) as im:
        source = im.convert("RGBA")
    with Image.open(root / pair["target_image"]) as im:
        target = im.convert("RGBA")
    shape = (source.height, source.width)
    sentences, loss_px, route_px, token_px = [], [], [], []
    near = np.zeros(shape, bool)
    for index, form in zip(row["order"], row["forms"]):
        unit, token = pair["units"][index], tokens["units"][index]
        if token["uid"] != unit["id"].rsplit("#", 1)[-1]:    # D4 units are "U1"..., extedit units "<pair>#U1"...
            raise ValueError("multi-unit token file does not match the manifest")
        s_px = rle_region(unit["selection"], shape)
        execution_px = rle_region(unit.get("execution_region"), shape)
        own = s_px | execution_px if execution_px is not None else s_px
        changed_px = unit_change(rle_region(unit.get("edited_region"), shape), own)
        loss_px.append(changed_px)
        route_px.append(s_px | changed_px)
        token_px.append(rd.decode_rle(token["dec"]))
        sentences.append(p2.render(unit["instruction_spec"], form == "ref_token", selector_token=token["span"]))
        for m in (s_px, changed_px, execution_px):
            if m is not None:
                near |= m
    return _edit_sample(pipe, kind, source, target, rd.multi_edit_prompt(sentences), loss_px=loss_px,
                        route_px=route_px, token_px=token_px, negative_px=np.zeros(shape, bool), read_negatives=False,
                        max_pixels=max_pixels, dilate=True, local_px=near if row.get("loss_region") == "local" else None)


def merged_regions(record, row, shape, partner_selection=None, pool_mask=None):
    """The region maps of a merged record (no model needed): per unit in ``row``'s order the loss region (the changed
    pixels; a multi-unit pair's shared change cut to the unit), the write-side route region (S + change), the token's
    decoded region and the rendered sentence; the negatives (near pool regions, twins' S) and the local-loss area."""
    order = row.get("order") or list(range(len(record["units"])))
    forms = row.get("forms") or [row.get("form", "noref_token")] * len(order)
    multi = len(record["units"]) > 1
    loss_px, route_px, token_px, sentences = [], [], [], []
    near = np.zeros(shape, bool)
    for index, form in zip(order, forms):
        unit = record["units"][index]
        s_px = rle_region(unit["selection"], shape)
        execution_px = rle_region(unit.get("execution_region"), shape)
        changed_px = rle_region(unit.get("edited_region"), shape)
        if multi:
            changed_px = unit_change(changed_px, s_px | execution_px if execution_px is not None else s_px)
        loss_px.append(next(m for m in (changed_px, execution_px, s_px) if m is not None and m.any()))
        route_px.append(s_px | changed_px if changed_px is not None else s_px)
        token_px.append(rd.decode_rle(unit["token"]["dec"]).astype(bool))
        sentences.append(p2.render(unit["instruction_spec"], form == "ref_token", selector_token=unit["token"]["span"]))
        for m in (s_px, changed_px, execution_px):
            if m is not None:
                near |= m
    negative_px = np.zeros(shape, bool)
    for j in row.get("negatives", ()):
        negative_px |= pool_mask(j)
    for partner in row.get("partners", ()):
        negative_px |= partner_selection(partner)
    negative_px &= ~near
    local_px = near if record.get("loss_region") == "local" else None
    return loss_px, route_px, token_px, sentences, negative_px, local_px


def _prepare_merged(pipe, ctx, row, max_pixels):
    """A record of the merged dataset (docs/01 section 2.13)."""
    record = ctx.merged(row["id"])
    with Image.open(rd.MERGED_ROOT / record["source_image"]) as im:
        source = im.convert("RGBA")
    with Image.open(rd.MERGED_ROOT / record["target_image"]) as im:
        target = im.convert("RGBA")
    shape = (source.height, source.width)
    entry = ctx.pool(record["sa1b"]["source_id"]) if row.get("negatives") else None
    loss_px, route_px, token_px, sentences, negative_px, local_px = merged_regions(
        record, row, shape, partner_selection=lambda p: rle_region(ctx.merged(p)["units"][0]["selection"], shape),
        pool_mask=lambda j: rd.decode_rle(entry["masks"][j]["rle"]).astype(bool))
    prompt = rd.multi_edit_prompt(sentences) if len(sentences) > 1 else sentences[0]
    return _edit_sample(pipe, "edit_merged", source, target, prompt, loss_px=loss_px, route_px=route_px,
                        token_px=token_px, negative_px=negative_px,
                        read_negatives=bool(row.get("negatives") or row.get("partners")), max_pixels=max_pixels,
                        dilate=True, local_px=local_px)


def _prepare_edit_multi(pipe, ctx, row, max_pixels):
    """2-4 50k edits of one image: each target equals the source outside its own execution region and the regions are
    disjoint, so the joint target is the source plus every edit's difference."""
    root = rd.RELEASE
    records = [ctx.release[sample_id] for sample_id in row["sample_ids"]]
    with Image.open(root / records[0]["source_image"]) as im:
        source = im.convert("RGBA")
    base = np.asarray(source.convert("RGB"), np.int16)
    joint = base.copy()
    sentences, loss_px, route_px, token_px = [], [], [], []
    covered = np.zeros(base.shape[:2], bool)
    for record, form in zip(records, row["forms"]):
        if record["source_id"] != records[0]["source_id"]:
            raise ValueError("multi-target units must share their source image")
        token = ctx.tokens[record["sample_id"]]
        with Image.open(root / record["edited_image"]) as im:
            joint += np.asarray(im.convert("RGB"), np.int16) - base
        support_px, s_px, changed_px = _release_regions(root, record)
        if (covered & support_px).any():
            raise ValueError("multi-target execution regions overlap")
        covered |= support_px | s_px | changed_px
        sentences.append(p2.render(record["instruction_spec"], form == "ref_token", selector_token=token["span"]))
        loss_px.append(changed_px if changed_px.any() else support_px)
        route_px.append(s_px | changed_px)
        token_px.append(rd.decode_rle(token["dec"]))
    target = Image.fromarray(joint.clip(0, 255).astype(np.uint8)).convert("RGBA")
    negative_px = np.zeros(base.shape[:2], bool)
    if row.get("negatives"):
        entry = ctx.pool(row["source_id"])
        for j in row["negatives"]:
            negative_px |= rd.decode_rle(entry["masks"][j]["rle"])
        negative_px &= ~covered
    return _edit_sample(pipe, "edit_multi", source, target, rd.multi_edit_prompt(sentences), loss_px=loss_px,
                        route_px=route_px, token_px=token_px, negative_px=negative_px,
                        read_negatives=bool(row.get("negatives")), max_pixels=max_pixels, dilate=True)


def balanced(labels, heavy, heavy_weight=DISTRACTOR_WEIGHT):
    """Positive and negative halves of the weight; negatives on ``heavy`` count ``heavy_weight`` times."""
    negative = 1 - labels
    weight_neg = negative * (1 + (heavy_weight - 1) * heavy)
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


def route_entries(sample, heavy_weight=DISTRACTOR_WEIGHT):
    """Router entries: per span, write-side labels on the target grid and read-side labels on the source grid.

    Negatives that count ``heavy_weight`` times: the distractors (on the source grid too, when the sample has
    them) and, with several spans, the other spans' regions.
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
        entry = {"span": span, "labels": target[k], "weights": balanced(target[k], heavy, heavy_weight)}
        if sample.source_positions:
            src_heavy = (torch.zeros_like(source[k]) if sample.source_distractors is None
                         else (sample.source_distractors > 0.5).float().flatten())
            for j, other in enumerate(source):
                if j != k:
                    src_heavy = torch.maximum(src_heavy, other)
            entry.update(source_positions=sample.source_positions, source_labels=source[k],
                         source_weights=balanced(source[k], src_heavy, heavy_weight))
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
                noise=None, checkpointing=True, heavy_weight=DISTRACTOR_WEIGHT, edit_region_dilate=True):
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
        router.arm_entries(sample.regions[0].numel(), route_entries(sample, heavy_weight))
        with router.recording():
            prediction = predict()
        write, read = router.collect(), router.collect_read()
    if prediction.shape != target.shape:
        raise RuntimeError("FM prediction/target shape mismatch")
    squared_error = (prediction.float() - target.float()).pow(2).mean(1)[0]
    mean = squared_error.mean() if sample.loss_mask is None else \
        (squared_error * sample.loss_mask).sum() / sample.loss_mask.sum().clamp(min=1.0)
    dilate = sample.region_dilate if sample.region_dilate is not None else sample.kind != "edit"
    if not edit_region_dilate and sample.kind in EDIT_KINDS:   # stage C: only the tokens that really change
        dilate = False
    in_region = region_error(squared_error, sample.regions, dilate=dilate)
    other = (sample.distractors > DISTRACTOR_COVERAGE).float()
    on_distractors = (squared_error * other).sum() / other.sum() if float(other.sum()) > 0 else mean.detach() * 0
    weight = pipe.scheduler.training_weight(timestep).to(pipe.device)
    flow = (mean + lambda_region * in_region + lambda_distractor * on_distractors) * weight
    metrics = {"mse": float(mean.detach()), "reg": float(in_region.detach()), "dis": float(on_distractors.detach()),
               "t": float(timestep), "write": float(write.detach()) if write is not None else None,
               "read": float(read.detach()) if read is not None else None}
    return flow, write, read, metrics
