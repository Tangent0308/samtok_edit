"""The Qwen-Image-2.1 editing pipeline with the SAMTok text encoder.

``load_pipeline`` builds DiffSynth's ``QwenImage21Pipeline`` (DiT and VAE of Qwen-Image-2.1) and swaps its text
encoder for Qwen3-VL-8B-SAMTok, whose vocabulary holds the 514 mask-token entries.  A region reaches the model only as
mask tokens written into the instruction (``edit(..., mode="inline")``); the optional ``CodeInjector`` attached to the
text encoder adds the SAMTok code vectors at the mask-token positions.  ``blend_inputs`` prepares latent blending,
which only the teacher test of on-policy self-distillation uses.
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

from samtok_edit21.data.protocol import boxes_in, regions_in, spans_in


MODEL_ROOT = Path("/mnt/bn/strategy-mllm-train/user/tanyue/models")
DEFAULT_QWEN = str(MODEL_ROOT / "pretrained_models/Qwen-Image-2.1")
DEFAULT_SAMTOK = str(MODEL_ROOT / "SAMTok/Qwen3-VL-8B-SAMTok")
# The DiT attention path the method was trained and evaluated with: the block-causal FlexAttention mask (DiffSynth's
# default up to 2.1.8).  Since 2.1.9 DiffSynth defaults to the segmented SDPA path, which computes the same attention
# with different kernels; every model call of this package passes this flag to stay on the original path.
ATTENTION = {"use_flex_attention": True}


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
        # Optional v3 side channel (models.code_injector); None keeps the v2 behaviour bit for bit.
        self.code_injector = None

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
        injector = self.code_injector
        if injector is None or not injector.enabled:
            return self.encode(**inputs)[0]
        # The side channel is trained against a frozen backbone: no graph through the text encoder,
        # gradients reach only the injector's projections.
        with torch.no_grad():
            hidden = self.encode(**inputs)[0]
        return hidden + injector(inputs["input_ids"]).to(hidden.dtype)

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


def edit(pipe, instruction, images, *, mode="inline", pass2_te="raw", blend_region=None, blend_dilate=2,
         blend_feather=1.0, **kwargs):
    """One edit through the DiffSynth pipeline call; returns ``(image, report)``.

    ``mode="inline"``: the instruction carries mask tokens (exactly one source image); ``mode="direct"``: plain text
    editing without region tokens.  ``pass2_te="raw"`` encodes with every PEFT layer of the text encoder disabled (the
    SAMTok TE as released), ``"adapter"`` keeps them.  ``blend_region`` (a boolean mask in source pixels) keeps the
    source outside it by latent blending; the method itself never blends.  Other keyword arguments (size, steps,
    seed, ``cfg_scale``, ``negative_prompt``, ...) go to the pipeline call unchanged.
    """
    regions = regions_in(instruction)
    if regions and len(images) != 1:
        raise ValueError("Region-conditioned editing requires exactly one source image")
    if mode == "direct" and regions:
        raise ValueError("direct is plain editing; use inline for region tokens")
    if mode == "inline" and not regions:
        raise ValueError("inline requires region tokens")
    if mode not in {"direct", "inline"}:
        raise ValueError(f"Unknown mode: {mode}")
    if pass2_te not in {"raw", "adapter"}:
        raise ValueError("pass2_te must be raw or adapter")
    if regions_in(kwargs.get("negative_prompt", "")):
        raise ValueError("CFG negative branch must not contain region tokens")
    height, width = pipe.check_resize_height_width(kwargs.get("height", 1024), kwargs.get("width", 1024), verbose=0)
    pipe.scheduler.training = False
    blend = {}
    with te_adapters(pipe, pass2_te == "adapter"):
        if blend_region is not None:
            mask, latents = blend_inputs(pipe, images[0], blend_region, height, width,
                                         dilate=blend_dilate, feather=blend_feather)
            blend = {"blend_mask": mask, "blend_latents": latents}
        image = pipe(instruction, edit_image=images, **blend, **{**ATTENTION, **kwargs})
    report = {"conditioning_prompt": instruction, "actual_variant": "inline" if regions else "plain",
              "pass2_te": pass2_te,
              "blend": None if not blend else {"dilate": blend_dilate, "feather": blend_feather,
                                               "area": float((blend["blend_mask"] > 0.5).float().mean())}}
    return image, report
