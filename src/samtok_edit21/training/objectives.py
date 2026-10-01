"""Two-stage LoRA training, with official Qwen-Image-2.1 flow matching."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from diffsynth.diffusion.training_module import DiffusionTrainingModule
from peft import LoraConfig, inject_adapter_in_model
from safetensors.torch import load_file, save_file


def _disable_broken_bnb_backend():
    """Keep PEFT on its dense LoRA path when an unrelated bnb install is broken.

    The training recipe never quantizes the base models.  Some shared
    environments nevertheless expose a user-site bitsandbytes package that
    cannot import against the installed Triton/CUDA version.  PEFT detects the
    package by spec alone and then imports it while replacing every Linear,
    which would abort otherwise.  Disable only the optional dispatchers when
    that import fails; dense LoRA remains unchanged.
    """

    try:
        import bitsandbytes  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on host packages
        import peft.tuners.lora.model as lora_model

        lora_model.is_bnb_available = lambda: False
        lora_model.is_bnb_4bit_available = lambda: False
        print(f"[SAMTokEdit] disabling unusable bitsandbytes backend: {exc}")


_disable_broken_bnb_backend()

from samtok_edit21.data.io import (
    file_hash,
    load_images,
    row_hash,
    write_json,
)
from samtok_edit21.models.pipeline import encode_edit, resize_sources

TE_TARGETS = r"model\.model\.language_model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))"


def stage2_target_modules(model):
    """Match DiffSynth's empty ``lora_target_modules`` auto-detection."""
    targets = DiffusionTrainingModule().auto_detect_lora_target_modules(model)
    if not targets:
        raise ValueError("DiffSynth found no Stage 2 DiT LoRA target modules")
    return targets


def add_adapter(model, stage, rank, dropout=0.0, *, alpha=None, targets=None):
    if stage not in {"stage1", "stage2"} or rank < 1 or not 0 <= dropout < 1:
        raise ValueError("Invalid adapter recipe")
    if targets is None:
        targets = TE_TARGETS if stage == "stage1" else stage2_target_modules(model)
    config = LoraConfig(
        r=rank, lora_alpha=rank if alpha is None else alpha, lora_dropout=dropout,
        target_modules=targets,
    )
    inject_adapter_in_model(config, model)
    for name, p in model.named_parameters():
        if p.requires_grad:
            if "lora_" not in name:
                raise RuntimeError(f"Unexpected trainable parameter {name}")
            p.data = p.data.float()  # fp32 optimizer states/updates in both stages
    if not any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("No LoRA parameters")
    return model


def load_adapter(model, directory, *, trainable=False):
    directory = Path(directory)
    config = json.loads((directory / "adapter.json").read_text())
    if config.get("schema_version", 1) not in {1, 2}:
        raise ValueError("Unknown adapter schema version")
    recipe = {k: config[k] for k in ("stage", "rank", "alpha", "dropout", "target_modules") if k in config}
    if config.get("recipe_sha256") and config["recipe_sha256"] != row_hash(recipe):
        raise ValueError("Adapter recipe fingerprint mismatch")
    add_adapter(model, config["stage"], config["rank"], config["dropout"],
                alpha=config.get("alpha"), targets=config.get("target_modules"))
    state = load_file(str(directory / "adapter.safetensors"))
    expected = {k for k, p in model.named_parameters() if p.requires_grad}
    if set(state) != expected:
        raise ValueError("Adapter schema does not match model and LoRA recipe")
    parameters = dict(model.named_parameters())
    if any(state[k].shape != parameters[k].shape for k in state):
        raise ValueError("Adapter tensor shapes disagree with adapter.json; recover into a new directory")
    if not all(torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Nonfinite adapter weights")
    model.load_state_dict(state, strict=False)
    if not trainable:
        model.requires_grad_(False)
    return config


def save_adapter(model, directory, config):
    # PEFT is the source of truth after warm-start, not CLI defaults.
    peft_config = model.peft_config["default"]
    config = {**config, "schema_version": 2, "rank": peft_config.r,
              "alpha": peft_config.lora_alpha, "dropout": peft_config.lora_dropout,
              "target_modules": sorted(peft_config.target_modules)
              if isinstance(peft_config.target_modules, set) else peft_config.target_modules}
    config["recipe_sha256"] = row_hash({k: config[k] for k in
                                       ("stage", "rank", "alpha", "dropout", "target_modules")})
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state = {
        k: p.detach().float().cpu().contiguous()
        for k, p in model.named_parameters()
        if p.requires_grad
    }
    if not state or not all(torch.isfinite(x).all() for x in state.values()):
        raise RuntimeError("Invalid adapter parameters")
    for name, tensor in state.items():
        if (".lora_A." in name and tensor.shape[0] != config["rank"]) or (
            ".lora_B." in name and tensor.shape[1] != config["rank"]
        ):
            raise ValueError("PEFT rank/config disagree with actual LoRA tensors")
    tmp = directory / "adapter.safetensors.tmp"
    save_file(state, str(tmp))
    tmp.replace(directory / "adapter.safetensors")
    write_json(directory / "adapter.json", config)


def adapter_identity(path):
    return (
        None
        if not path
        else {
            "path": str(Path(path).absolute()),
            "sha256": file_hash(Path(path) / "adapter.safetensors"),
            "config_sha256": file_hash(Path(path) / "adapter.json"),
        }
    )


def verify_cache(directory, manifest):
    from samtok_edit21.data.protocol import validate_row

    directory = Path(directory)
    if manifest.get("format") != "samtok21-cache-v1":
        raise ValueError("Incompatible cache format")
    for row in manifest["rows"]:
        original = {k: v for k, v in row.items() if not k.startswith("_cache")}
        validate_row(original)
        name = row["_cache_file"]
        if Path(name).name != name:
            raise ValueError("Cache shard must be a local filename")
        side = json.loads(directory.joinpath(name).with_suffix(".json").read_text())
        if (
            side["row_hash"] != row_hash(original)
            or side["identity"] != manifest["identity"]
        ):
            raise ValueError("Mixed or stale cache metadata")
        if side["sha256"] != file_hash(directory / name):
            raise ValueError("Cache checksum mismatch")


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


def prepare_fm(pipe, row, base_path, max_pixels, *, te_grad=False, supervision=None):
    images, target, height, width = load_images(row, base_path, max_pixels)
    if target is None:
        raise ValueError("FM requires a target image")
    images = resize_sources(pipe, images, height, width)
    with torch.no_grad():
        target_latent = pipe.vae.encode(pipe.preprocess_image(target))
        source_latents = [pipe.vae.encode(pipe.preprocess_image(im)) for im in images]
    with torch.enable_grad() if te_grad else torch.no_grad():
        cond = encode_edit(pipe, row["prompt"], images, return_positions=True) if supervision is not None and supervision["eligible"] else encode_edit(pipe, row["prompt"], images)
    if supervision is not None:
        supervision = dict(supervision)
        if supervision["eligible"]:
            supervision["span_positions"] = cond.pop("span_positions")
    inputs = {"input_latents": target_latent, "edit_latents": source_latents, **cond}
    if supervision is not None:
        inputs["region_supervision"] = supervision
        from samtok_edit21.regions.supervision import validate_supervision
        validate_supervision(supervision, inputs, row, require_positions=supervision["eligible"])
    validate_conditioning(inputs)
    return inputs


def flow_loss(pipe, inputs, *, timestep_index=None, noise=None, checkpointing=True,
              region_weight=0.0, region_n_min=16.0, attention_weight=0.0,
              attention_layers=(), attention_read_weight=0.5, return_components=False):
    if not math.isfinite(attention_weight) or attention_weight < 0 or not math.isfinite(region_weight) or region_weight < 0:
        raise ValueError("Loss weights must be finite and nonnegative")
    if attention_weight and not attention_layers:
        raise ValueError("Nonzero attention weight requires selected layers")
    # Exactly the official FlowMatchSFTLoss sampling, target and weighting;
    # expose timestep/weight for audit and repeatable gradient checks.
    i = (
        torch.randint(0, len(pipe.scheduler.timesteps), (1,))
        if timestep_index is None
        else torch.tensor([timestep_index])
    )
    original_t = pipe.scheduler.timesteps[i]
    t = original_t.to(device=pipe.device, dtype=pipe.torch_dtype)
    x = inputs["input_latents"]
    noise = torch.randn_like(x) if noise is None else noise
    noisy = pipe.scheduler.add_noise(x, noise, t)
    target = pipe.scheduler.training_target(x, noise, t)
    supervision = inputs.get("region_supervision")
    enabled = region_weight > 0 or bool(attention_layers)
    eligible = False
    probe = None
    if enabled:
        from samtok_edit21.regions.supervision import validate_supervision
        validate_supervision(supervision, inputs, require_positions=bool(attention_layers))
        eligible = supervision["eligible"]
        if eligible and attention_layers:
            from samtok_edit21.training.attention import AttentionSupervision
            probe = AttentionSupervision(supervision["span_positions"], supervision["coverage_source"],
                                         supervision["coverage_target"], tuple(attention_layers))
    model_inputs = {k: v for k, v in inputs.items() if k != "region_supervision"}
    if probe is not None:
        model_inputs["attention_probe"] = probe
    pred = pipe.model_fn(
        dit=pipe.dit,
        latents=noisy,
        timestep=t,
        **model_inputs,
        kv_cache=None,
        use_gradient_checkpointing=checkpointing,
    )
    if probe is not None:
        pred, statistics = pred
    if pred.shape != target.shape:
        raise RuntimeError("FM prediction/target shape mismatch")
    mse = torch.nn.functional.mse_loss(pred.float(), target.float())
    weight = pipe.scheduler.training_weight(t).to(pipe.device)
    basic_fm = mse * weight
    fm, aux = basic_fm, mse.new_zeros(())
    metrics = {}
    if eligible:
        from samtok_edit21.regions.supervision import attention_regions, region_fm_loss
        coverage = supervision["coverage_target"].to(device=pred.device, dtype=torch.float32)
        if region_weight > 0:
            regional, values = region_fm_loss((pred.float() - target.float()).square().mean(1),
                                             coverage.amax(0)[None], region_weight, region_n_min)
            fm = regional * weight
            metrics.update(values)
        metrics.update(mask_count=len(coverage), mask_max_target=coverage.flatten(1).amax(1).mean(),
                       mask_max_source=supervision["coverage_source"].flatten(1).amax(1).mean(),
                       attn_uniform_target=attention_regions(coverage).mean(),
                       attn_uniform_source=attention_regions(supervision["coverage_source"]).mean())
        if probe is not None:
            from samtok_edit21.training.attention import attention_loss
            aux, values = attention_loss(statistics, read_weight=attention_read_weight,
                                        heads=pipe.dit.transformer_blocks[0].attn.heads,
                                        target_tokens=target.shape[-2] * target.shape[-1], layers=attention_layers)
            metrics.update(values)
    loss = fm + attention_weight * aux
    metrics.update({
        "loss_fm": fm.detach().item(),
        "loss_fm_basic": basic_fm.detach().item(),
        "loss_total": loss.detach().item(),
        "attention_weight": attention_weight,
        "region_weight": region_weight,
        "region_eligible": int(eligible),
        "fm_mse": mse.detach().item(),
        "timestep": t.item(),
        "timestep_index": i.item(),
        "timestep_before_cast": original_t.item(),
        "training_weight": weight.item(),
        "target_shape": list(target.shape),
    })
    if enabled and not eligible:
        metrics["region_skip_reason"] = supervision["reason"]
    metrics = {k: v.detach().item() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}
    if return_components:
        return loss, metrics, {"fm": fm, "attention": aux, "basic_fm": basic_fm}
    return loss, metrics


# Backward-compatible Python entry points; there is only one training loop.
def train(args):
    from samtok_edit21.training.engine import normalize_args, run_train
    args.command = "train"
    return run_train(normalize_args(args))


def cache(args):
    from samtok_edit21.training.engine import normalize_args, run_cache
    args.command = "cache"
    return run_cache(normalize_args(args))
