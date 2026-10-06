"""Use native Qwen3-VL weights and official DiffSynth 2.1 conditioning.

v2 two-pass inference: pass 1 generates the region JSON with the localization
LoRA (Stage 1); pass 2 encodes the region-bound prompt with the raw SAMTok TE
(adapters disabled), exactly as the Stage 2 cache did.  Optional structural
region binding and latent blending act only inside the DiT pipeline call.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration
from diffsynth.core import ModelConfig
from diffsynth.pipelines.qwen_image_21 import (
    QwenImage21Pipeline,
    QwenImage21Unit_EditImageEmbedder,
    QwenImage21Unit_PromptEmbedder,
)

from samtok_edit21.data.protocol import (
    LOC_REQUEST,
    EMPTY_THINK,
    boxes_in,
    condition_localization,
    parse_cot,
    parse_generated_cot,
    regions_in,
    spans_in,
)

MODEL_ROOT = Path("/mnt/bn/strategy-mllm-train/user/tanyue/models")
DEFAULT_QWEN = str(MODEL_ROOT / "pretrained_models/Qwen-Image-2.1")
DEFAULT_SAMTOK = str(MODEL_ROOT / "SAMTok/Qwen3-VL-8B-SAMTok")


class SamtokTextEncoder(torch.nn.Module):
    """FM consumes pre-final-RMSNorm features; NTP consumes normalized features.

    The upstream wrapper also taps the norm input, but calls the complete LM head
    and leaves hooks attached. Calling the native backbone avoids full-vocabulary
    logits and scoped hooks avoid accumulating closures across training steps.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = model.config

    def encode(self, **inputs):
        captured = []
        norm = self.model.model.language_model.norm
        handle = norm.register_forward_pre_hook(
            lambda module, args: captured.append(args[0])
        )
        try:
            self.model.model.rope_deltas = None
            output = self.model.model(**inputs, use_cache=False, return_dict=True)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError(
                f"Expected one final norm invocation, got {len(captured)}"
            )
        return captured[0], output.last_hidden_state

    def forward(self, **inputs):
        return self.encode(**inputs)[0]

    def generate(self, **inputs):
        self.model.model.rope_deltas = None
        return self.model.generate(**inputs)


def build_processor(qwen_dir, samtok_dir=None):
    processor = AutoProcessor.from_pretrained(
        str(Path(qwen_dir) / "processor"), local_files_only=True
    )
    if samtok_dir:
        processor.tokenizer = AutoTokenizer.from_pretrained(
            samtok_dir, local_files_only=True
        )
        processor.chat_template = Path(samtok_dir, "chat_template.jinja").read_text()
        vocab_size = json.loads(Path(samtok_dir, "config.json").read_text())[
            "text_config"
        ]["vocab_size"]
        names = ["<|mt_start|>", *[f"<|mt_{i:04d}|>" for i in range(512)], "<|mt_end|>"]
        ids = [processor.tokenizer.encode(x, add_special_tokens=False) for x in names]
        if (
            any(len(x) != 1 or x[0] >= vocab_size for x in ids)
            or len({x[0] for x in ids}) != 514
        ):
            raise ValueError(
                "SAMTok tokens are not 514 distinct atomic vocabulary entries"
            )
    if processor.tokenizer.convert_tokens_to_ids("<|image_pad|>") != 151655:
        raise ValueError("Qwen3-VL image token ID mismatch")
    return processor


def load_pipeline(
    qwen_dir=DEFAULT_QWEN,
    samtok_dir=DEFAULT_SAMTOK,
    *,
    device="cuda",
    components=("text_encoder", "dit", "vae"),
):
    paths = {
        "dit": "transformer/diffusion_pytorch_model*.safetensors",
        "vae": "vae/diffusion_pytorch_model*.safetensors",
    }
    configs = []
    for name, pattern in paths.items():
        if name in components:
            files = sorted(str(p) for p in Path(qwen_dir).glob(pattern))
            if not files:
                raise FileNotFoundError(f"Missing {qwen_dir}/{pattern}")
            configs.append(ModelConfig(path=files))
    pipe = QwenImage21Pipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device=device, model_configs=configs
    )
    if "text_encoder" in components:
        model_path = samtok_dir or str(Path(qwen_dir) / "text_encoder")
        hf = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
            device_map={"": str(device)},
        )
        if (
            hf.config.text_config.hidden_size != 4096
            or hf.config.vision_config.patch_size != 16
        ):
            raise ValueError(
                "The text encoder is not architecture-compatible with Qwen-Image-2.1"
            )
        pipe.text_encoder = SamtokTextEncoder(hf)
        pipe.processor = build_processor(qwen_dir, samtok_dir)
    pipe.requires_grad_(False)
    pipe.eval()
    return pipe


def resize_sources(pipe, images, height, width):
    return QwenImage21Unit_EditImageEmbedder().resize_edit_image(
        pipe,
        [im.convert("RGBA") for im in images],
        height * width,
    )


@contextmanager
def te_adapters(pipe, enabled):
    """Temporarily enable/disable every PEFT layer of the TE.

    A disabled LoRA layer returns exactly its base layer's output, so pass 2
    with ``enabled=False`` is the raw SAMTok TE.  PEFT toggling also flips
    requires_grad; the previous flags are restored on exit.
    """
    from peft.tuners.tuners_utils import BaseTunerLayer

    layers = [m for m in pipe.text_encoder.modules() if isinstance(m, BaseTunerLayer)]
    states = [(m.disable_adapters, [(q, q.requires_grad) for q in m.parameters()]) for m in layers]
    for layer in layers:
        layer.enable_adapters(enabled)
    try:
        yield bool(layers)
    finally:
        for layer, (disabled, grads) in zip(layers, states):
            layer.enable_adapters(not disabled)
            for parameter, flag in grads:
                parameter.requires_grad_(flag)


def localization_inputs(pipe, instruction, images, *, cot=None, request=LOC_REQUEST):
    """Chat input for pass 1 / NTP; ``request=None`` uses the text as the whole request."""
    if len(images) != 1:
        raise ValueError("Localization binds regions to exactly one source image")
    if not instruction.strip() or any(t in instruction for t in ("<|", "<think>", "</think>")):
        raise ValueError("Localization needs clean, nonempty instruction text")
    text = instruction.strip() if request is None else instruction.strip() + "\n" + request
    content = [
        {"type": "image"},
        # Qwen3 SAMTok _build_messages strips whitespace around the segment
        # following <image>; retain the internal instruction/request newline.
        {"type": "text", "text": text},
    ]
    # Official SAMTok Qwen3 dataset passes only user/assistant messages. Its
    # released native template does not synthesize a helpful-assistant system.
    text = pipe.processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    text += EMPTY_THINK
    inputs = pipe.processor(
        text=[text],
        images=[
            QwenImage21Unit_PromptEmbedder.composite_over_white(im) for im in images
        ],
        return_tensors="pt",
        padding=True,
    ).to(pipe.device)
    prefix_len = inputs.input_ids.shape[1]
    labels = None
    if cot is not None:
        parse_cot(cot, nonempty=True)
        labels = pipe.processor.tokenizer(
            cot + "<|im_end|>", add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(pipe.device)
        for key in ("input_ids", "attention_mask", "mm_token_type_ids"):
            if key in inputs:
                tail = (
                    labels
                    if key == "input_ids"
                    else (
                        torch.ones_like(labels)
                        if key == "attention_mask"
                        else torch.zeros_like(labels)
                    )
                )
                inputs[key] = torch.cat([inputs[key], tail], dim=1)
    return inputs, prefix_len, labels


def ntp_loss(pipe, instruction, images, cot, *, request=LOC_REQUEST):
    inputs, prefix, labels = localization_inputs(pipe, instruction, images, cot=cot, request=request)
    _, normalized = pipe.text_encoder.encode(**inputs)
    supervised = normalized[:, prefix - 1 : prefix - 1 + labels.shape[1]]
    logits = pipe.text_encoder.model.lm_head(supervised)
    loss = torch.nn.functional.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1)
    )
    return loss, {
        "prefix_tokens": prefix,
        "label_tokens": labels.numel(),
        "hidden_start": prefix - 1,
        "last_label": int(labels[0, -1]),
        "loss_ntp": loss.detach().item(),
    }


def encode_edit(pipe, prompt, images, *, return_ids=False):
    """Official prompt embedding; region tokens must stay atomic special tokens."""
    tokenizer = pipe.processor.tokenizer
    for span in spans_in(prompt):
        if len(tokenizer.encode(span, add_special_tokens=False)) != 4:
            raise RuntimeError("Mask span is not four atomic tokens")
    for box in boxes_in(prompt):
        ids = tokenizer.encode(box, add_special_tokens=False)
        if ids[0] != tokenizer.convert_tokens_to_ids("<|box_start|>") or ids[-1] != tokenizer.convert_tokens_to_ids("<|box_end|>"):
            raise RuntimeError("Box delimiters are not atomic tokens")
    return QwenImage21Unit_PromptEmbedder().process(pipe, prompt, images, return_token_ids=return_ids)


def region_binding_for(pipe, prompt, images, height, width, config, codec=None):
    """Inference binding from the prompt actually encoded in pass 2 (D8).

    Same layout and region maps as the Stage 2 cache: mask spans are decoded by
    the codec on the source image, boxes are rasterized.
    """
    from samtok_edit21.models.binding import RegionBinding, binding_payload

    if config is None or config.mode == "none":
        return None
    embed = getattr(pipe.dit, "region_embed", None) if config.mode == "region_embed" else None
    if not regions_in(prompt):
        return RegionBinding(config, None, embed) if config.mode == "region_embed" else None
    prepared = resize_sources(pipe, images, height, width)
    ids = encode_edit(pipe, prompt, prepared, return_ids=True)["prompt_input_ids"]
    payload = binding_payload(pipe.processor.tokenizer, ids, prompt, images[0],
                              {"target": (height, width), "source": (prepared[0].height, prepared[0].width)},
                              codec)
    return RegionBinding(config, payload, embed)


def blend_inputs(pipe, image, region, height, width, *, dilate=2, feather=1.0):
    """Latent-grid blend mask and source latents for one target canvas.

    ``region`` is a boolean mask in full-frame source pixels.  A latent token is
    inside if any of its pixels is; the mask is dilated by ``dilate`` tokens and
    feathered by a Gaussian of std ``feather`` tokens.
    """
    mask = torch.as_tensor(np.asarray(region) > 0, dtype=torch.float32)[None, None]
    if mask.shape[-2:] != (image.height, image.width):
        raise ValueError("Blend region must use source-image pixels")
    mask = F.adaptive_max_pool2d(mask, (height // 16, width // 16))
    if dilate:
        mask = F.max_pool2d(mask, 2 * dilate + 1, 1, dilate)
    if feather:
        offsets = torch.arange(-3, 4, dtype=torch.float32)
        kernel = torch.exp(-offsets ** 2 / (2 * feather ** 2))
        kernel = kernel / kernel.sum()
        mask = F.conv2d(F.pad(mask, (3, 3, 0, 0), mode="replicate"), kernel.view(1, 1, 1, 7))
        mask = F.conv2d(F.pad(mask, (0, 0, 3, 3), mode="replicate"), kernel.view(1, 1, 7, 1))
    source = image.convert("RGBA").resize((width, height), resample=Image.Resampling.LANCZOS)
    with torch.no_grad():
        latents = pipe.vae.encode(pipe.preprocess_image(source))
    return mask.clamp(0, 1), latents


@torch.no_grad()
def localize(
    pipe,
    instruction,
    images,
    *,
    height=1024,
    width=1024,
    max_new_tokens=256,
    do_sample=False,
    temperature=0.8,
    variant="ref",
    reviewed_units=None,
    strict_noref=False,
):
    if strict_noref and variant != "noref":
        raise ValueError("strict_noref requires variant=noref")
    if len(images) != 1:
        raise ValueError("Localization requires exactly one source image")
    # Pass 2's ShapeChecker rounds the requested canvas up before resizing
    # source images. Localize on that same canvas even for nonaligned requests.
    height, width = pipe.check_resize_height_width(height, width)
    prepared = resize_sources(pipe, images, height, width)
    inputs, prefix, _ = localization_inputs(pipe, instruction, prepared)
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "eos_token_id": pipe.processor.tokenizer.convert_tokens_to_ids("<|im_end|>"),
        "pad_token_id": pipe.processor.tokenizer.pad_token_id,
        "use_cache": True,
    }
    if do_sample:
        kwargs["temperature"] = temperature
    with te_adapters(pipe, True):  # pass 1 uses the localization adapter
        ids = pipe.text_encoder.generate(**inputs, **kwargs)
    raw = pipe.processor.tokenizer.decode(ids[0, prefix:], skip_special_tokens=False)
    try:
        items = parse_generated_cot(raw)
        result = condition_localization(instruction, items, variant=variant,
                                        reviewed=reviewed_units, strict=strict_noref)
    except (ValueError, TypeError, KeyError) as exc:
        if strict_noref:
            raise ValueError(f"Strict localization failed: {exc}") from exc
        items = []
        result = {"conditioning_prompt": instruction, "requested_variant": variant,
                  "actual_variant": "plain", "fallback_reason": str(exc)}
    return {"raw": raw, "items": items, **result}


@torch.no_grad()
def edit(
    pipe, instruction, images, *, mode="online", cot=None, max_new_tokens=256,
    variant="ref", reviewed_units=None, strict_noref=False, pass2_te="raw",
    binding=None, codec=None, blend_region=None, blend_dilate=2, blend_feather=1.0, **kwargs
):
    """Direct, inline, explicit oracle, or online two-pass inference.

    ``pass2_te``: ``raw`` (v2) disables TE adapters while encoding the edit
    prompt; ``adapter`` keeps them (v1 checkpoints). ``binding`` is the DiT
    adapter's BindingConfig; ``blend_region`` a source-pixel mask for latent
    blending, ``"prompt"`` for the regions of the final edit prompt, or None.
    """
    regions = regions_in(instruction)
    if (regions or mode in {"online", "oracle"}) and len(images) != 1:
        raise ValueError("Region-conditioned editing requires exactly one source image")
    if mode == "direct" and regions:
        raise ValueError("direct is plain editing; use inline for region tokens")
    if mode == "inline" and not regions:
        raise ValueError("inline requires region tokens")
    if strict_noref and (variant != "noref" or mode not in {"online", "oracle"}):
        raise ValueError("strict_noref requires online/oracle with variant=noref")
    if pass2_te not in {"raw", "adapter"}:
        raise ValueError("pass2_te must be raw or adapter")
    result = {
        "requested_variant": variant if mode in {"online", "oracle"} else None,
        "actual_variant": "inline" if regions else "plain",
        "raw": None,
        "items": [],
        "conditioning_prompt": instruction,
        "fallback_reason": None,
    }
    if mode == "online":
        result = localize(
            pipe,
            instruction,
            images,
            height=kwargs.get("height", 1024),
            width=kwargs.get("width", 1024),
            max_new_tokens=max_new_tokens,
            variant=variant, reviewed_units=reviewed_units, strict_noref=strict_noref,
        )
    elif mode == "oracle":
        items = parse_generated_cot(cot)
        result.update(items=items, **condition_localization(
            instruction, items, variant=variant, reviewed=reviewed_units, strict=strict_noref))
    elif mode not in {"direct", "inline"}:
        raise ValueError(f"Unknown mode: {mode}")
    prompt = result["conditioning_prompt"]
    regions_in(prompt)
    if regions_in(kwargs.get("negative_prompt", "")):
        raise ValueError("CFG negative branch must not contain region tokens")
    height, width = pipe.check_resize_height_width(kwargs.get("height", 1024), kwargs.get("width", 1024), verbose=0)
    pipe.scheduler.training = False
    with te_adapters(pipe, pass2_te == "adapter"):
        region_binding = region_binding_for(pipe, prompt, images, height, width, binding, codec)
        blend = {}
        if isinstance(blend_region, str):
            if blend_region != "prompt":
                raise ValueError("blend_region must be a mask, 'prompt' or None")
            from samtok_edit21.models.binding import blend_region as prompt_region
            blend_region = prompt_region(prompt, images[0], codec) if regions_in(prompt) else None
        if blend_region is not None:
            mask, latents = blend_inputs(pipe, images[0], blend_region, height, width,
                                         dilate=blend_dilate, feather=blend_feather)
            blend = {"blend_mask": mask, "blend_latents": latents}
        image = pipe(prompt, edit_image=images, region_binding=region_binding, **blend, **kwargs)
    result["pass2_te"] = pass2_te
    result["binding"] = None if binding is None else {
        **binding.as_dict(), "bound_units": 0 if region_binding is None else len(region_binding.units)}
    result["blend"] = None if not blend else {"dilate": blend_dilate, "feather": blend_feather,
                                              "area": float((blend["blend_mask"] > 0.5).float().mean())}
    return image, result
