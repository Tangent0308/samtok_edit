"""Structural region binding between region tokens and the target latent grid.

v2 replaces the attention-ratio loss (A) with structure: the decoded region of
each region unit is handed to the DiT explicitly, identically in training (from
the conditioning cache) and in inference (from the prompt actually encoded).

A *unit* is a maximal run of directly concatenated region tokens in the edit
prompt (one or more SAMTok spans, or one or more boxes) bound to one phrase.
Its region map is the union of the unit's regions, as fractional coverage on a
latent grid (one value per 16x16-pixel latent token).

Modes (``--binding``):

``none``          B0: no binding.
``bias_span``     for target query q and key k among the unit's region tokens,
                  logit += beta * log(eps + (1 - eps) * m_u(q)).
``bias_clause``   the same bias on every instruction token after the source
                  image, including the trailing chat template: for a
                  single-unit prompt this is a regional prompt.
``region_embed``  target token q += m_u(q) * W(h_u), where h_u is the DiT text
                  feature (after ``txt_in``) of the unit's last region token
                  and W is a zero-initialized low-rank map.
``region_rope``   the unit's region tokens take the (h, w) RoPE index of the
                  centroid of m_u on the target grid.

Only target queries are biased and only prefix keys move under region_rope,
so a KV cache built on the first denoising step stays valid.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from samtok_edit21.data.protocol import REGION_RE, box_coords, region_kind, spans_in

BINDING_MODES = ("none", "bias_span", "bias_clause", "region_embed", "region_rope")
SCHEMA = "samtok-region-binding-v1"
GEOMETRY = "mask:codec-raw-gt0.5|box:outward-pixel-raster;bilinear-aa-avg16-maxpool3-fp32"
GROUP_RE = re.compile(r"(?:" + REGION_RE.pattern + r")+")
LOG_FLOOR = math.log(1e-6)  # eps = 0 is a hard mask up to this floor


def require_flex_attention():
    """Attention-bias binding trains through FlexAttention's score_mod."""
    from packaging.version import Version
    from diffsynth.core.attention import FLEX_ATTN_AVAILABLE
    if Version(torch.__version__.split("+")[0]) < Version("2.8") or not FLEX_ATTN_AVAILABLE:
        raise RuntimeError("Region binding requires PyTorch >= 2.8 with FlexAttention")


@dataclass(frozen=True)
class BindingConfig:
    mode: str = "none"
    beta: float = 1.0
    eps: float = 0.05
    rank: int = 64  # region_embed only

    def __post_init__(self):
        if self.mode not in BINDING_MODES:
            raise ValueError(f"Unknown binding mode {self.mode!r}")
        if not math.isfinite(self.beta) or self.beta < 0 or not 0 <= self.eps < 1 or self.rank < 1:
            raise ValueError("Binding needs finite beta >= 0, eps in [0, 1) and rank >= 1")

    def as_dict(self):
        return {"mode": self.mode, "beta": self.beta, "eps": self.eps, "rank": self.rank}


def coverage_grid(mask, height, width):
    """Pixel mask -> dilated fractional coverage on a (height/16, width/16) grid."""
    if height % 16 or width % 16:
        raise ValueError("Region canvas must align with VAE tokens")
    pixels = torch.as_tensor(np.asarray(mask), dtype=torch.float32)[None, None]
    pixels = F.interpolate(pixels, (height, width), mode="bilinear", align_corners=False, antialias=True)
    coverage = F.max_pool2d(F.avg_pool2d(pixels, 16, 16), 3, 1, 1)[0, 0]
    # Antialiasing can overshoot a constant-one input by roundoff only.
    return coverage.clamp(0, 1).contiguous()


def box_pixels(box, width, height, *, expand=0.0):
    """0-1000 box -> outward-rounded binary mask at the given pixel size.

    ``expand`` grows the box by that fraction of its width/height on each side
    (clipped to the image), e.g. 0.1 for the blending region of an add box.
    """
    x1, y1, x2, y2 = box_coords(box)
    dx, dy = expand * (x2 - x1), expand * (y2 - y1)
    x1, y1, x2, y2 = max(0, x1 - dx), max(0, y1 - dy), min(1000, x2 + dx), min(1000, y2 + dy)
    mask = np.zeros((height, width), dtype=bool)
    mask[math.floor(y1 * height / 1000):math.ceil(y2 * height / 1000),
         math.floor(x1 * width / 1000):math.ceil(x2 * width / 1000)] = True
    return mask


def blend_region(prompt, source_image, codec=None, *, box_expand=0.1):
    """Union of the prompt's regions in source pixels, add boxes expanded.

    Used when no user region exists (text-only setting): the region decoded
    from pass 1's tokens bounds the latent blending.
    """
    width, height = source_image.size
    spans = spans_in(prompt)
    decoded = iter(codec.decode_strict(source_image, prompt) if spans else [])
    union = np.zeros((height, width), dtype=bool)
    for region in (m.group() for m in REGION_RE.finditer(prompt)):
        if region_kind(region) == "mask":
            union |= np.asarray(next(decoded)) > 0
        else:
            union |= box_pixels(region, width, height, expand=box_expand)
    if not union.any():
        raise ValueError("The prompt has no nonempty region to blend around")
    return union


def unit_masks(prompt, source_image, codec=None):
    """Pixel union mask of every region unit, in prompt order, at source size."""
    groups = [m.group() for m in GROUP_RE.finditer(prompt)]
    spans = spans_in(prompt)
    if spans and codec is None:
        raise ValueError("Mask spans need the SAMTok codec to decode their regions")
    decoded = iter(codec.decode_strict(source_image, prompt) if spans else [])
    width, height = source_image.size
    masks = []
    for group in groups:
        union = np.zeros((height, width), dtype=bool)
        for region in (m.group() for m in REGION_RE.finditer(group)):
            if region_kind(region) == "mask":
                union |= np.asarray(next(decoded)) > 0
            else:
                union |= box_pixels(region, width, height)
        masks.append(union)
    if next(decoded, None) is not None:
        raise ValueError("Decoded span count differs from the prompt")
    return groups, masks


def token_layout(tokenizer, input_ids, prompt):
    """Region units and the instruction range in the trimmed TE token sequence.

    ``input_ids`` is ``prompt_input_ids`` from the official prompt embedder
    (system tokens dropped, -1 padding). Every unit is checked against the
    region groups of ``prompt`` so a tokenization surprise cannot bind the
    wrong tokens.
    """
    ids = torch.as_tensor(input_ids)
    if ids.ndim == 2:
        if ids.shape[0] != 1:
            raise ValueError("Binding supports batch one")
        ids = ids[0]
    ids = ids[ids >= 0].tolist()
    convert = tokenizer.convert_tokens_to_ids
    mt_start, mt_end = convert("<|mt_start|>"), convert("<|mt_end|>")
    box_start, box_end = convert("<|box_start|>"), convert("<|box_end|>")
    regions, index = [], 0
    while index < len(ids):
        token = ids[index]
        if token == mt_start:
            if index + 3 >= len(ids) or ids[index + 3] != mt_end:
                raise ValueError("Truncated mask span in the encoded prompt")
            regions.append((index, index + 4))
            index += 4
        elif token == box_start:
            try:
                end = ids.index(box_end, index)
            except ValueError as exc:
                raise ValueError("Unterminated box in the encoded prompt") from exc
            regions.append((index, end + 1))
            index = end + 1
        elif token in (mt_end, box_end):
            raise ValueError("Unpaired region end token in the encoded prompt")
        else:
            index += 1
    units = []
    for start, end in regions:
        if units and units[-1][1] == start:
            units[-1][1] = end
        else:
            units.append([start, end])
    groups = [m.group() for m in GROUP_RE.finditer(prompt)]
    if len(groups) != len(units) or any(
            tokenizer.decode(ids[s:e], skip_special_tokens=False) != g for (s, e), g in zip(units, groups)):
        raise ValueError("Encoded region tokens differ from the prompt's region groups")
    vision_end = [i for i, token in enumerate(ids) if token == convert("<|vision_end|>")]
    if len(vision_end) != 1 or (units and units[0][0] <= vision_end[0]):
        raise ValueError("Binding requires one source image before the instruction")
    return {"units": [tuple(unit) for unit in units], "instruction": (vision_end[0] + 1, len(ids)),
            "length": len(ids)}


def binding_payload(tokenizer, input_ids, prompt, source_image, canvases, codec=None):
    """Cache/inference record: token layout plus per-canvas unit coverage.

    ``canvases`` maps a name (``target``, ``source``) to the pixel (height,
    width) of that latent block. Units whose region decodes empty are kept but
    flagged; ``RegionBinding`` leaves them unbound.
    """
    layout = token_layout(tokenizer, input_ids, prompt)
    groups, masks = unit_masks(prompt, source_image, codec)
    if not groups:
        return None
    payload = {"schema": SCHEMA, "geometry": GEOMETRY,
               "units": [list(unit) for unit in layout["units"]],
               "instruction": list(layout["instruction"]), "length": layout["length"],
               "empty": [not mask.any() for mask in masks]}
    for name, (height, width) in canvases.items():
        payload[name] = torch.stack([coverage_grid(mask, height, width) for mask in masks])
    return payload


def validate_payload(payload, inputs):
    """Structural checks against the conditioning tensors of the same row."""
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA or payload.get("geometry") != GEOMETRY:
        raise ValueError("Missing or unknown region binding payload")
    units, (start, end) = payload["units"], payload["instruction"]
    text = ~inputs["edit_image_pad_mask"][0]
    if payload["length"] != text.numel() or not units or len(payload["empty"]) != len(units):
        raise ValueError("Binding layout disagrees with the encoded prompt")
    if not 0 < start <= units[0][0] or end != payload["length"]:
        raise ValueError("Instruction range must cover all region units")
    for unit_start, unit_end in units:
        if not unit_start < unit_end <= end or not bool(text[unit_start:unit_end].all()):
            raise ValueError("Region tokens must be text tokens of the instruction")
    if any(b[0] < a[1] for a, b in zip(units, units[1:])):
        raise ValueError("Region units overlap or are reordered")
    for name, latent in (("target", inputs["input_latents"]), ("source", inputs["edit_latents"][0])):
        maps = payload[name]
        if (not isinstance(maps, torch.Tensor) or maps.dtype != torch.float32
                or maps.shape != (len(units), *latent.shape[-2:])):
            raise ValueError(f"Invalid {name} region maps")
        if not torch.isfinite(maps).all() or (maps < 0).any() or (maps > 1).any():
            raise ValueError(f"Out-of-range {name} region maps")
        if [bool(m.max() <= 0) for m in maps] != payload["empty"]:
            raise ValueError("Empty-region flags disagree with the region maps")


class RegionEmbed(torch.nn.Module):
    """Zero-initialized low-rank map from a unit's text feature to an offset."""

    def __init__(self, dim=4096, rank=64):
        super().__init__()
        self.down = torch.nn.Linear(dim, rank, bias=False)
        self.up = torch.nn.Linear(rank, dim, bias=False)
        torch.nn.init.zeros_(self.up.weight)

    def forward(self, features):
        return self.up(self.down(features.to(self.down.weight.dtype)))


def attach_region_embed(dit, rank):
    """Create the trainable region_embed module (fp32, like LoRA weights)."""
    if hasattr(dit, "region_embed"):
        raise ValueError("DiT already has a region_embed module")
    dit.region_embed = RegionEmbed(dit.inner_dim, rank).to(device=next(dit.parameters()).device,
                                                           dtype=torch.float32)
    return dit.region_embed


class RegionBinding:
    """Per-sample binding passed to ``QwenImage21DiT.forward``.

    ``payload`` is a cache/inference record from ``binding_payload``; it may be
    None for prompts without regions, in which case only region_embed still
    touches its parameters (with zero effect) so DDP sees every parameter used.
    """

    def __init__(self, config, payload=None, embed=None):
        if config.mode == "none":
            raise ValueError("Use no RegionBinding for mode none")
        if (config.mode == "region_embed") != (embed is not None):
            raise ValueError("region_embed needs exactly the DiT region_embed module")
        self.config, self.embed = config, embed
        keep = [] if payload is None else [i for i, empty in enumerate(payload["empty"]) if not empty]
        self.units = [tuple(payload["units"][i]) for i in keep]
        self.instruction = None if payload is None else tuple(payload["instruction"])
        self.target = None if not keep else payload["target"][keep]
        if config.mode == "bias_clause" and len(self.units) > 1:
            raise ValueError("bias_clause is defined for single-unit prompts")

    def bind(self, *, repeats, target_token_mask, target_shape, encoder_hidden_states, pos_embed):
        return BoundBinding(self, repeats, target_token_mask, target_shape, encoder_hidden_states, pos_embed)


class BoundBinding:
    """A RegionBinding resolved to joint DiT positions for one forward pass."""

    def __init__(self, binding, repeats, target_token_mask, target_shape, encoder_hidden_states, pos_embed):
        config = binding.config
        self.mode = config.mode
        self.binding = binding
        device = repeats.device
        self.length = int(repeats.sum())
        self.target_index = target_token_mask.nonzero().flatten()
        height, width = target_shape
        if self.target_index.numel() != height * width:
            raise ValueError("Target block and region grid disagree")
        offsets = repeats.cumsum(0) - repeats
        self.text_length = int((~target_token_mask).sum())
        self.maps = None
        if binding.units:
            if binding.target.shape[1:] != (height, width):
                raise ValueError("Region maps do not match the target latent grid")
            self.maps = binding.target.to(device=device, dtype=torch.float32).flatten(1)
        self.groups = []  # joint positions bound to each unit
        for start, end in binding.units:
            if self.mode == "bias_clause":
                start, end = binding.instruction
            self.groups.append(offsets[torch.arange(start, end, device=device)])
        self.anchors = [end - 1 for _, end in binding.units]  # TE coordinates
        self.encoder_hidden_states = encoder_hidden_states
        self.pos_embed = pos_embed
        self._score_mod = None

    # ----- region_embed -------------------------------------------------
    def apply_embedding(self, joint_hidden_states):
        if self.mode != "region_embed":
            return joint_hidden_states
        embed = self.binding.embed
        if self.maps is None:  # keep every parameter in the graph for DDP
            dummy = embed(self.encoder_hidden_states.new_zeros(1, embed.down.in_features))
            return joint_hidden_states + 0 * dummy.sum().to(joint_hidden_states.dtype)
        vectors = embed(self.encoder_hidden_states[0, self.anchors])  # [U, D] fp32
        offset = (self.maps.t() @ vectors).to(joint_hidden_states.dtype)  # [T, D]
        joint_hidden_states = joint_hidden_states.clone()
        joint_hidden_states[:, self.target_index] = joint_hidden_states[:, self.target_index] + offset[None]
        return joint_hidden_states

    # ----- region_rope --------------------------------------------------
    def apply_rope(self, rotary_emb, target_shape):
        if self.mode != "region_rope" or self.maps is None:
            return rotary_emb
        height, width = target_shape
        rows = torch.arange(height, device=rotary_emb.device, dtype=torch.float32)
        cols = torch.arange(width, device=rotary_emb.device, dtype=torch.float32)
        axes = self.pos_embed.axes_dim
        h0, w0 = axes[0] // 2, axes[0] // 2 + axes[1] // 2
        rotary_emb = rotary_emb.clone()
        for maps, group in zip(self.maps, self.groups):
            grid = maps.view(height, width)
            total = grid.sum()
            # Centered grid indices, exactly as QwenImage21Rope numbers the target block.
            h = int(torch.round((grid.sum(1) * rows).sum() / total)) - (height - height // 2)
            w = int(torch.round((grid.sum(0) * cols).sum() / total)) - (width - width // 2)
            rotary_emb[group, h0:w0] = self.pos_embed.freqs[1][h].to(rotary_emb.device)
            rotary_emb[group, w0:] = self.pos_embed.freqs[2][w].to(rotary_emb.device)
        return rotary_emb

    # ----- attention bias -----------------------------------------------
    @property
    def biased(self):
        return self.mode in ("bias_span", "bias_clause") and self.maps is not None

    def _query_bias(self):
        config = self.binding.config
        values = torch.log(config.eps + (1 - config.eps) * self.maps).clamp_min(LOG_FLOOR)
        return config.beta * values  # [U, T]

    def score_mod(self, padded_length):
        """FlexAttention score_mod for the uncached (training / first-step) path."""
        if not self.biased:
            return None
        device = self.maps.device
        unit_of_key = torch.full((padded_length,), -1, dtype=torch.long, device=device)
        for unit, group in enumerate(self.groups):
            unit_of_key[group] = unit
        query_bias = torch.zeros(len(self.groups), padded_length, dtype=torch.float32, device=device)
        query_bias[:, self.target_index] = self._query_bias()

        def score_mod(score, batch, head, q_idx, kv_idx):
            unit = unit_of_key[kv_idx]
            return score + torch.where(unit >= 0, query_bias[unit.clamp(min=0), q_idx], 0.0)

        return score_mod

    def decode_mask(self, key_valid, dtype):
        """Additive mask for cached decoding: target queries x all keys."""
        if not self.biased:
            return None if key_valid is None else key_valid[:, None, None, :]
        bias = torch.zeros(self.target_index.numel(), self.length, dtype=torch.float32, device=self.maps.device)
        for query_bias, group in zip(self._query_bias(), self.groups):
            bias[:, group] = query_bias[:, None]
        if key_valid is not None:
            bias = bias.masked_fill(~key_valid[0][None, :], float("-inf"))
        return bias[None, None].to(dtype)
