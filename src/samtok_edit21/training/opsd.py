"""On-policy self-distillation (OPSD) with a mask-privileged teacher.

The student is the model being trained, conditioned exactly as at inference: the source image and the instruction with
mask tokens (the row's form, blind or named).  It samples its own trajectory with the inference schedule (no
gradients), and at a few states of that trajectory it learns to predict what the teacher predicts there.  The teacher
is the same DiT and text encoder with frozen weights (the warm-start adapter; optionally an EMA of the student) and
three privileged inputs from the training record, the configuration T3 of the teacher test
(experiments: analysis/opsd_teacher_20261025):

* every unit named: the instruction rendered with the object's name next to the token, the blind units too;
* token guidance: ``v = v_neg + w * (v_pos - v_neg)``, the negative branch being the named instruction without tokens;
* the source outside the allowed region A, the units' selection, execution region and changed pixels, dilated and
  feathered on the token grid: outside A the teacher's clean estimate is the source latent.

Flow matching here is ``x_t = (1 - sigma) * x0 + sigma * eps`` with the velocity ``v = eps - x0``, so a prediction
gives the clean estimate ``x0 = x_t - sigma * v``.  The teacher's estimate is
``x0_T = m * (x_t - sigma * v_T) + (1 - m) * x_src`` (m: the feathered allowed region), and the loss of a state is
``mean((x0_S - x0_T)^2)``: the velocity error weighted by ``sigma^2``, which stays bounded where the student has
drifted outside A at a small sigma (the velocity target ``(x_t - x_src) / sigma`` would not).

Two refinements after the first pilot (docs/02 §5.11), both off by default so that earlier runs stay reproducible:
``loss="split"`` normalises the error inside and outside A apart, ``sum(m e) / sum(m) + w * sum((1 - m) e) / sum(1 - m)``
(over the whole latent the outside, whose target is just the source, outweighs a small A); ``high_sigma`` and
``high_share`` take a set share of the supervised states from the noisy start of the trajectory, where whether and
where to edit is decided.

Order of operations, so that no gradient ever sees the teacher's weights: the rollout with the student's weights (no
gradients); then, for every supervised state, the teacher target (teacher weights swapped in, no gradients, swapped
back), the student's forward with gradients and its backward at once.  Backward therefore runs while the student's
weights are in place, and non-reentrant gradient checkpointing recomputes the student's forward, not the teacher's.
The rollout and the teacher's target run with the DiT in eval mode, as at inference (no LoRA dropout).
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from samtok_edit21.data.protocol import strip_tokens
from samtok_edit21.evaluation.generation import canvas
from samtok_edit21.models.pipeline import ATTENTION, blend_inputs, encode_edit, resize_sources
from samtok_edit21.preparation import instruction_protocol_v2 as p2
from samtok_edit21.preparation import round2_data as rd
from samtok_edit21.training.round2 import rle_region, validate_conditioning


@dataclass(frozen=True)
class OpsdConfig:
    steps: int = 16                 # student rollout: the inference schedule at this length
    states: int = 2                 # supervised states per rollout
    min_sigma: float = 0.1          # states below this noise level are not supervised
    guidance: float = 2.0           # teacher token guidance (1: off)
    student_guidance: float = 1.0   # token guidance of the rollout policy (1: the plain conditional prediction)
    dilate: int = 2                 # allowed region: dilation, in tokens
    feather: float = 1.0            # allowed region: Gaussian feather, in tokens
    loss: str = "mean"              # "mean": over the whole latent; "split": inside and outside A normalised apart
    outside_weight: float = 0.1     # split loss: weight of the outside part
    high_sigma: float = 0.0         # states at sigma >= this form the high band (0: no band, uniform picking)
    high_share: float = 0.5         # expected share of the supervised states taken from the high band

    def __post_init__(self):
        if self.steps < 1 or self.states < 1 or not 0 <= self.min_sigma < 1 or self.guidance < 1 \
                or self.student_guidance < 1 or self.dilate < 0 or self.feather < 0 or self.loss not in ("mean", "split") \
                or self.outside_weight < 0 or not 0 <= self.high_sigma <= 1 or not 0 <= self.high_share <= 1:
            raise ValueError(f"Invalid OPSD configuration {self}")


class TeacherWeights:
    """The teacher's copy of the trainable tensors that act in the forward pass (DiT LoRA and code injector).

    ``active()`` puts the copy into the parameters and restores the student's values on exit; ``update_ema`` moves the
    copy towards the student (momentum 0 is never used: a frozen teacher simply never updates)."""

    def __init__(self, parameters, weights=None):
        self.parameters = list(parameters)
        source = self.parameters if weights is None else list(weights)
        if len(source) != len(self.parameters) or any(w.shape != p.shape for w, p in zip(source, self.parameters)):
            raise ValueError("Teacher weights do not match the trainable parameters")
        self.weights = [w.detach().to(device=p.device, dtype=p.dtype).clone() for w, p in zip(source, self.parameters)]

    @contextmanager
    def active(self):
        backup = [p.detach().clone() for p in self.parameters]
        with torch.no_grad():
            for parameter, weight in zip(self.parameters, self.weights):
                parameter.copy_(weight)
        try:
            yield
        finally:
            with torch.no_grad():
                for parameter, value in zip(self.parameters, backup):
                    parameter.copy_(value)

    @torch.no_grad()
    def update_ema(self, momentum):
        if not 0 < momentum < 1:
            raise ValueError("EMA momentum must be in (0, 1)")
        for weight, parameter in zip(self.weights, self.parameters):
            weight.mul_(momentum).add_(parameter.detach(), alpha=1 - momentum)

    def state(self):
        return [w.detach().cpu() for w in self.weights]


@dataclass
class OpsdSample:
    prepared: list                  # the source image as the model sees it (resized to the canvas area)
    edit_latents: list              # its VAE latents (the source condition)
    source_latents: torch.Tensor    # x_src: the source on the target latent grid [1, 64, h, w]
    mask: torch.Tensor              # [1, 1, h, w] allowed region on the target grid, in [0, 1] (float32)
    student_prompt: str
    teacher_prompt: str
    teacher_negative: str


def prompts(record, row):
    """(student prompt, teacher prompt, teacher negative) of a merged record: the row's forms for the student, every
    unit named for the teacher, and the teacher prompt without its tokens for the guidance's negative branch."""
    order = row.get("order") or list(range(len(record["units"])))
    forms = row.get("forms") or [row.get("form", "noref_token")] * len(order)
    if sorted(order) != list(range(len(record["units"]))) or len(forms) != len(order):
        raise ValueError(f"{record['id']}: the row's unit order does not cover the record")
    student, teacher = [], []
    for index, form in zip(order, forms):
        unit = record["units"][index]
        span = unit["token"]["span"]
        student.append(p2.render(unit["instruction_spec"], form == "ref_token", selector_token=span))
        teacher.append(p2.render(unit["instruction_spec"], True, selector_token=span))
    if len(order) > 1:
        student, teacher = rd.multi_edit_prompt(student), rd.multi_edit_prompt(teacher)
    else:
        student, teacher = student[0], teacher[0]
    return student, teacher, strip_tokens(teacher)


def allowed_region(record, shape):
    """Pixels the teacher may change: the union of every unit's selection, execution region and changed pixels."""
    region = np.zeros(shape, bool)
    for unit in record["units"]:
        for key in ("selection", "execution_region", "edited_region"):
            mask = rle_region(unit.get(key), shape)
            if mask is not None:
                region |= mask
    if not region.any():
        raise ValueError(f"{record['id']}: empty allowed region")
    return region


def prepare_opsd(pipe, ctx, row, config):
    """An OPSD row of the merged dataset at its inference canvas (``generation.canvas``)."""
    record = ctx.merged(row["id"])
    with Image.open(rd.MERGED_ROOT / record["source_image"]) as im:
        source = im.convert("RGBA")
    height, width = canvas(source)
    student, teacher, negative = prompts(record, row)
    region = allowed_region(record, (source.height, source.width))
    with torch.no_grad():
        prepared = resize_sources(pipe, [source], height, width)
        edit_latents = [pipe.vae.encode(pipe.preprocess_image(image)) for image in prepared]
        mask, source_latents = blend_inputs(pipe, source, region, height, width,
                                            dilate=config.dilate, feather=config.feather)
    grid = (height // 16, width // 16)
    if tuple(source_latents.shape) != (1, 64, *grid) or tuple(mask.shape) != (1, 1, *grid):
        raise ValueError(f"{record['id']}: the source latent or the allowed region is off the target grid {grid}")
    return OpsdSample(prepared=prepared, edit_latents=edit_latents, source_latents=source_latents,
                      mask=mask.to(device=source_latents.device, dtype=torch.float32),
                      student_prompt=student, teacher_prompt=teacher, teacher_negative=negative)


@contextmanager
def evaluating(module):
    """``module`` in eval mode for the duration; its previous mode is restored on exit."""
    training = module.training
    module.eval()
    try:
        yield
    finally:
        module.train(training)


def encode(pipe, prompt, sample):
    """Conditioning of the DiT for ``prompt`` on the sample's source (gradients reach the code injector unless the
    caller disables them)."""
    conditioning = encode_edit(pipe, prompt, sample.prepared)
    conditioning["edit_latents"] = sample.edit_latents
    return conditioning


def predict(pipe, conditioning, latents, timestep, *, kv_cache=None, checkpointing=False):
    return pipe.model_fn(dit=pipe.dit, latents=latents, timestep=timestep, **conditioning, kv_cache=kv_cache,
                         use_gradient_checkpointing=checkpointing, **ATTENTION)


def guided(pipe, positive, negative, latents, timestep, scale, *, caches=(None, None)):
    """Token guidance as DiffSynth's CFG: ``v_neg + scale * (v_pos - v_neg)``; ``scale`` 1 is the positive branch."""
    v_pos = predict(pipe, positive, latents, timestep, kv_cache=caches[0])
    if scale == 1.0:
        return v_pos
    v_neg = predict(pipe, negative, latents, timestep, kv_cache=caches[1])
    return v_neg + scale * (v_pos - v_neg)


def model_timestep(pipe, timestep):
    return timestep.reshape(1).to(device=pipe.device, dtype=pipe.torch_dtype)


@torch.no_grad()
def rollout(pipe, scheduler, positive, negative, noise, guidance):
    """The student's own trajectory: the latent at the input of every step of ``scheduler`` (Euler steps of the
    pipeline, with the prefix KV cache the pipeline uses at inference)."""
    blocks = len(pipe.dit.transformer_blocks)
    caches = ([{} for _ in range(blocks)], [{} for _ in range(blocks)] if guidance != 1.0 else None)
    latents, states = noise, []
    for timestep in scheduler.timesteps:
        states.append(latents)
        velocity = guided(pipe, positive, negative, latents, model_timestep(pipe, timestep), guidance, caches=caches)
        latents = scheduler.step(velocity, timestep, latents)
    return states


def pick_states(sigmas, count, min_sigma, generator=None, high_sigma=0.0, high_share=0.5):
    """``count`` distinct rollout states with sigma >= ``min_sigma``, in order.  Without a band (``high_sigma`` at or
    below ``min_sigma``) they are drawn uniformly without replacement.  With one, ``count * high_share`` of them on
    average (stochastic rounding) come from the states at sigma >= ``high_sigma`` and the rest from those below; a band
    that runs short is filled from the other."""
    values = torch.as_tensor(sigmas).tolist()
    eligible = [index for index, sigma in enumerate(values) if sigma >= min_sigma]
    if not eligible:
        raise ValueError(f"no rollout state reaches sigma {min_sigma}")
    if high_sigma <= min_sigma:
        order = torch.randperm(len(eligible), generator=generator).tolist()
        return sorted(eligible[k] for k in order[:count])
    high = [index for index in eligible if values[index] >= high_sigma]
    low = [index for index in eligible if values[index] < high_sigma]
    total = min(count, len(eligible))
    wanted = total * high_share
    n_high = int(wanted) + int(float(torch.rand((), generator=generator)) < wanted - int(wanted))
    n_low = min(total - min(n_high, len(high)), len(low))
    picks = [high[k] for k in torch.randperm(len(high), generator=generator).tolist()[:total - n_low]]
    return sorted(picks + [low[k] for k in torch.randperm(len(low), generator=generator).tolist()[:n_low]])


def teacher_clean(latents, sigma, velocity, mask, source_latents):
    """The teacher's clean estimate: its own inside the allowed region, the source outside (float32)."""
    estimate = latents.float() - sigma * velocity.float()
    return mask * estimate + (1 - mask) * source_latents.float()


def student_loss(latents, sigma, velocity, target):
    """``mean((x0_S - x0_T)^2)`` with ``x0_S = x_t - sigma * v_S``."""
    return (latents.float() - sigma * velocity.float() - target).pow(2).mean()


def split_loss(latents, sigma, velocity, target, mask, outside_weight):
    """The squared error ``e`` of the clean estimates (mean over channels), inside the allowed region and outside it,
    each normalised by its own area: ``sum(m e) / sum(m) + outside_weight * sum((1 - m) e) / sum(1 - m)``.  Returns
    the loss and its two parts."""
    error = (latents.float() - sigma * velocity.float() - target).pow(2).mean(1)
    inside, outside = mask[:, 0].float(), 1 - mask[:, 0].float()
    inner = (inside * error).sum() / inside.sum().clamp_min(1e-6)
    outer = (outside * error).sum() / outside.sum().clamp_min(1e-6)
    return inner + outside_weight * outer, inner, outer


def opsd_backward(pipe, ctx, row, teacher, config, *, scale, checkpointing=True):
    """One OPSD row: the student's rollout, then for every picked state the teacher target and the student's
    backward of ``scale * loss / states``.  Returns the mean loss, its error inside and outside the allowed region
    (each normalised by its own area, whichever loss is trained) and the picked sigmas."""
    sample = prepare_opsd(pipe, ctx, row, config)
    scheduler = deepcopy(pipe.scheduler)            # the trainer's own schedule (1000 training steps) stays untouched
    scheduler.set_timesteps(config.steps, dynamic_shift_len=sample.source_latents.shape[2] * sample.source_latents.shape[3])
    noise = torch.randn(sample.source_latents.shape, device=pipe.device, dtype=pipe.torch_dtype)
    with torch.no_grad(), evaluating(pipe.dit):
        positive = encode(pipe, sample.student_prompt, sample)
        validate_conditioning({**positive, "input_latents": noise})
        negative = None if config.student_guidance == 1.0 else encode(pipe, strip_tokens(sample.student_prompt), sample)
        states = rollout(pipe, scheduler, positive, negative, noise, config.student_guidance)
        with teacher.active():
            teacher_positive = encode(pipe, sample.teacher_prompt, sample)
            teacher_negative = None if config.guidance == 1.0 else encode(pipe, sample.teacher_negative, sample)
    picked = pick_states(scheduler.sigmas, config.states, config.min_sigma, high_sigma=config.high_sigma,
                         high_share=config.high_share)
    losses, parts = [], []
    for index in picked:
        latents, timestep, sigma = states[index], scheduler.timesteps[index], float(scheduler.sigmas[index])
        with torch.no_grad(), evaluating(pipe.dit), teacher.active():
            velocity = guided(pipe, teacher_positive, teacher_negative, latents, model_timestep(pipe, timestep),
                              config.guidance)
            target = teacher_clean(latents, sigma, velocity, sample.mask, sample.source_latents)
        conditioning = encode(pipe, sample.student_prompt, sample)
        velocity = predict(pipe, conditioning, latents, model_timestep(pipe, timestep), checkpointing=checkpointing)
        if config.loss == "split":
            loss, inner, outer = split_loss(latents, sigma, velocity, target, sample.mask, config.outside_weight)
        else:
            loss = student_loss(latents, sigma, velocity, target)
            with torch.no_grad():
                _, inner, outer = split_loss(latents, sigma, velocity, target, sample.mask, 0.0)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite OPSD loss")
        (scale * loss / len(picked)).backward()
        losses.append(float(loss.detach()))
        parts.append((float(inner.detach()), float(outer.detach())))
    return {"opsd": sum(losses) / len(losses), "inside": sum(p[0] for p in parts) / len(parts),
            "outside": sum(p[1] for p in parts) / len(parts),
            "sigmas": [round(float(scheduler.sigmas[i]), 4) for i in picked]}
