"""Attention-routing loss (v3): tell the DiT *which* target tokens a mask token is about.

Flow matching on edit pairs only rewards a good picture; it never says that a particular ``<M>`` span stands for a
particular place, so the model can (and did, in v2) ignore the span.  The routing loss supervises exactly that link on
a few DiT blocks: a target token inside the token's region must attend to the span more strongly than a target token
outside it, and target tokens on *other* instances of the same image are the most important negatives.

Nothing here is used at inference.  The quantity supervised is the model's own pre-softmax attention logit, read with
a forward pre-hook, so no pixel mask or attention bias ever enters the forward pass.
"""

from __future__ import annotations

import math
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import nn

from diffsynth.models.qwen_image_21_dit import apply_rotary_emb_qwen

DEFAULT_LAYERS = (6, 10, 14, 18, 22, 26)
LOGIT_SCALE = 4.0
DISTRACTOR_WEIGHT = 3.0


def route_targets(region, distractors, distractor_weight=DISTRACTOR_WEIGHT):
    """Labels and balanced weights on the target latent grid.

    ``region`` / ``distractors`` are coverage maps in [0, 1].  Positives (inside the region) and negatives (outside)
    each carry half of the total weight, so a small region is not drowned out; negatives lying on other instances of
    the image count ``distractor_weight`` times.
    """
    labels = (region > 0.5).float().flatten()
    negative = 1 - labels
    weight_neg = negative + (distractor_weight - 1) * (distractors > 0.5).float().flatten() * negative
    weights = torch.zeros_like(labels)
    if labels.sum() > 0:
        weights = weights + labels * (0.5 / labels.sum())
    if weight_neg.sum() > 0:
        weights = weights + weight_neg * (0.5 / weight_neg.sum())
    return labels, weights


def span_attention_logits(block, hidden_states, modulation, target_token_mask, rotary_emb, n_target, span_positions):
    """Mean attention logit from each target token to the span tokens, as this block computes it.

    Mirrors ``QwenImage21TransformerBlock.forward`` up to the query/key that enter attention (norm1, timestep
    modulation, to_q/to_k with their LoRA, RMS norm, RoPE).  The target tokens are the last ``n_target`` positions of
    the joint sequence.  Returns ``[n_target]`` for batch element 0: the scaled dot product averaged over heads and
    over the span's tokens.
    """
    attention = block.attn
    scale_shift = modulation.chunk(2, dim=-1)[0]
    states, _ = block._modulate(block.img_norm1(hidden_states), scale_shift, target_token_mask)
    start = states.shape[1] - n_target
    query = attention.norm_q(attention.to_q(states[:, start:]).unflatten(-1, (attention.heads, -1))).to(states.dtype)
    query = apply_rotary_emb_qwen(query, rotary_emb[start:], use_real=False)
    key = attention.norm_k(attention.to_k(states[:, span_positions]).unflatten(-1, (attention.heads, -1))).to(states.dtype)
    key = apply_rotary_emb_qwen(key, rotary_emb[span_positions], use_real=False)
    logits = torch.einsum("bnhd,bshd->bn", query.float(), key.float())[0]
    return logits / (math.sqrt(query.shape[-1]) * query.shape[2] * key.shape[1])


def span_read_logits(block, hidden_states, modulation, target_token_mask, rotary_emb, span_positions, source_positions):
    """Mean attention logit from the span tokens to each source-image token, as this block computes it.

    The read side of a mask token: in the block-causal prefix the instruction (and its span) attends to the source
    image that precedes it, so this logit is part of the model's own attention at every denoising step and does not
    depend on the noisy target.  Returns ``[n_source]`` for batch element 0, averaged over heads and span tokens.
    """
    attention = block.attn
    scale_shift = modulation.chunk(2, dim=-1)[0]
    states, _ = block._modulate(block.img_norm1(hidden_states), scale_shift, target_token_mask)
    query = attention.norm_q(attention.to_q(states[:, span_positions]).unflatten(-1, (attention.heads, -1))).to(states.dtype)
    query = apply_rotary_emb_qwen(query, rotary_emb[span_positions], use_real=False)
    key = attention.norm_k(attention.to_k(states[:, source_positions]).unflatten(-1, (attention.heads, -1))).to(states.dtype)
    key = apply_rotary_emb_qwen(key, rotary_emb[source_positions], use_real=False)
    logits = torch.einsum("bshd,bnhd->bn", query.float(), key.float())[0]
    return logits / (math.sqrt(query.shape[-1]) * query.shape[2] * query.shape[1])


class AttentionRouter(nn.Module):
    """Collects the routing loss of selected DiT blocks through forward pre-hooks.

    Usage per sample::

        router.arm(n_target, span_positions, labels, weights)
        with router.recording():
            prediction = dit_forward(...)          # gradient checkpointing allowed
        loss = flow + lam * router.collect()
        loss.backward()
        router.disarm()                            # only after backward, see below

    The DiT runs its blocks under non-reentrant gradient checkpointing, which re-executes each block during
    ``backward``.  The hook must perform the same operations in that second pass or autograd finds a different number
    of saved tensors, so the router stays armed until ``disarm`` and only *appends* a term while ``recording``.

    The per-layer affine ``(scale, bias)`` lets the loss calibrate each layer's logit range instead of forcing the
    attention itself to absolute values.
    """

    def __init__(self, dit, layers=DEFAULT_LAYERS, logit_scale=LOGIT_SCALE, read=False):
        super().__init__()
        self.layers = tuple(int(layer) for layer in layers)
        if not self.layers or len(set(self.layers)) != len(self.layers):
            raise ValueError("Routing needs distinct DiT block indices")
        if min(self.layers) < 0 or max(self.layers) >= len(dit.transformer_blocks):
            raise ValueError("Routing layer outside the DiT")
        self.logit_scale = float(logit_scale)
        self.affine = nn.Parameter(torch.tensor([[1.0, 0.0]] * len(self.layers)))
        # The read side (span -> source tokens) has its own per-layer calibration, created only when it is used so
        # that round-1 adapters and their tests are unchanged.
        self.read_affine = nn.Parameter(torch.tensor([[1.0, 0.0]] * len(self.layers))) if read else None
        self._context, self._recording, self._terms, self._read_terms = None, False, [], []
        self._handles = [
            dit.transformer_blocks[layer].register_forward_pre_hook(self._hook(index), with_kwargs=True)
            for index, layer in enumerate(self.layers)
        ]

    def _hook(self, index):
        def before_block(block, args, kwargs):
            context = self._context
            if context is None:
                return
            for entry in context["entries"]:
                if entry.get("labels") is not None:
                    logits = span_attention_logits(
                        block, kwargs["hidden_states"], kwargs["modulation"], kwargs["target_token_mask"],
                        kwargs["rotary_emb"], context["n_target"], entry["span"])
                    scale, bias = self.affine[index]
                    calibrated = scale * (logits - logits.mean()) * self.logit_scale + bias
                    term = F.binary_cross_entropy_with_logits(
                        calibrated, entry["labels"], weight=entry["weights"], reduction="sum")
                    if self._recording:
                        self._terms.append(term)
                if entry.get("source_labels") is not None and self.read_affine is not None:
                    logits = span_read_logits(
                        block, kwargs["hidden_states"], kwargs["modulation"], kwargs["target_token_mask"],
                        kwargs["rotary_emb"], entry["span"], entry["source_positions"])
                    scale, bias = self.read_affine[index]
                    calibrated = scale * (logits - logits.mean()) * self.logit_scale + bias
                    term = F.binary_cross_entropy_with_logits(
                        calibrated, entry["source_labels"], weight=entry["source_weights"], reduction="sum")
                    if self._recording:
                        self._read_terms.append(term)

        return before_block

    def arm(self, n_target, span_positions, labels, weights):
        """One span, write side only (round 1)."""
        self.arm_entries(n_target, [{"span": list(span_positions), "labels": labels, "weights": weights}])

    def arm_entries(self, n_target, entries):
        """Several spans of one prompt.  Each entry: ``span`` (joint positions), optionally ``labels``/``weights`` on
        the target grid (write side) and ``source_positions``/``source_labels``/``source_weights`` (read side)."""
        self._context = {"n_target": int(n_target), "entries": [dict(entry, span=list(entry["span"])) for entry in entries]}
        self._terms, self._read_terms = [], []

    @contextmanager
    def recording(self):
        self._recording = True
        try:
            yield
        finally:
            self._recording = False

    def collect(self):
        """Mean write-side term over the hooked layers and spans of the last recorded forward (``None`` if none)."""
        terms, self._terms = self._terms, []
        return torch.stack(terms).mean() if terms else None

    def collect_read(self):
        """Mean read-side term over the hooked layers and spans of the last recorded forward (``None`` if none)."""
        terms, self._read_terms = self._read_terms, []
        return torch.stack(terms).mean() if terms else None

    def disarm(self):
        self._context, self._recording, self._terms, self._read_terms = None, False, [], []

    def remove(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self.disarm()
