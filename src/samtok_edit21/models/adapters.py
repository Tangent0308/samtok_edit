"""DiT LoRA adapters (PEFT) and the method's adapter directories.

An adapter directory holds ``adapter.json`` (recipe and fingerprint) and ``adapter.safetensors`` (the trainable LoRA
tensors, fp32).  ``stage`` stays ``"stage2"``, the DiT LoRA, as in every adapter trained since v2, so that older
directories load unchanged; the v2 text-encoder LoRA and structural region binding are gone.  A method directory may
also hold the code injector (``models.code_injector``); ``load_grounding_adapter`` loads both.
"""
from __future__ import annotations

import json
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

from samtok_edit21.data.io import row_hash, write_json  # noqa: E402
from samtok_edit21.models.code_injector import (  # noqa: E402
    CodeInjector,
    attach_code_injector,
    has_code_injector,
    load_code_injector,
)


def stage2_target_modules(model):
    """Match DiffSynth's empty ``lora_target_modules`` auto-detection."""
    targets = DiffusionTrainingModule().auto_detect_lora_target_modules(model)
    if not targets:
        raise ValueError("DiffSynth found no Stage 2 DiT LoRA target modules")
    return targets


def add_adapter(model, stage, rank, dropout=0.0, *, alpha=None, targets=None):
    """Inject the DiT LoRA; ``stage`` must be ``"stage2"`` (the DiT), the name every adapter.json records."""
    if stage != "stage2" or rank < 1 or not 0 <= dropout < 1:
        raise ValueError("Invalid adapter recipe")
    if targets is None:
        targets = stage2_target_modules(model)
    config = LoraConfig(
        r=rank, lora_alpha=rank if alpha is None else alpha, lora_dropout=dropout,
        target_modules=targets,
    )
    inject_adapter_in_model(config, model)
    for name, p in model.named_parameters():
        if p.requires_grad:
            if "lora_" not in name:
                raise RuntimeError(f"Unexpected trainable parameter {name}")
            p.data = p.data.float()  # fp32 optimizer states/updates
    if not any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("No LoRA parameters")
    return model


def load_adapter(model, directory, *, trainable=False):
    directory = Path(directory)
    config = json.loads((directory / "adapter.json").read_text())
    if config.get("schema_version", 1) not in {1, 2, 3}:
        raise ValueError("Unknown adapter schema version")
    recipe = {k: config[k] for k in ("stage", "rank", "alpha", "dropout", "target_modules") if k in config}
    if config.get("recipe_sha256") and config["recipe_sha256"] != row_hash(recipe):
        raise ValueError("Adapter recipe fingerprint mismatch")
    if config.get("binding", {"mode": "none"}).get("mode", "none") != "none":
        raise ValueError("Structural region binding (v2) is not supported")
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
    config = {**config, "schema_version": 3, "rank": peft_config.r,
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
