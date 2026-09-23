"""Use native Qwen3-VL weights and official DiffSynth 2.1 conditioning."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration
from diffsynth.core import ModelConfig
from diffsynth.pipelines.qwen_image_21 import (
    QwenImage21Pipeline,
    QwenImage21Unit_EditImageEmbedder,
    QwenImage21Unit_PromptEmbedder,
)

from .protocol import (
    LOC_REQUEST,
    grouped_units,
    parse_cot,
    parse_generated_cot,
    render_units,
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


def localization_inputs(pipe, instruction, images, *, cot=None):
    if len(images) != 1:
        raise ValueError("Localization binds regions to exactly one source image")
    content = [
        {"type": "image"},
        # Qwen3 SAMTok _build_messages strips whitespace around the segment
        # following <image>; retain the internal instruction/request newline.
        {"type": "text", "text": instruction + "\n" + LOC_REQUEST},
    ]
    # Official SAMTok Qwen3 dataset passes only user/assistant messages. Its
    # released native template does not synthesize a helpful-assistant system.
    text = pipe.processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )
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


def ntp_loss(pipe, instruction, images, cot):
    inputs, prefix, labels = localization_inputs(pipe, instruction, images, cot=cot)
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


def encode_edit(pipe, prompt, images):
    spans = spans_in(prompt)
    unit = QwenImage21Unit_PromptEmbedder()
    result = unit.process(pipe, prompt, images)
    if spans:
        # Added SAMTok tokens are atomic special tokens, so the official
        # processor retains boundaries without a second manual BPE encoding.
        for span in spans:
            if (
                len(pipe.processor.tokenizer.encode(span, add_special_tokens=False))
                != 4
            ):
                raise RuntimeError("Mask span is not four atomic tokens")
    return result


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
):
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
    ids = pipe.text_encoder.generate(**inputs, **kwargs)
    raw = pipe.processor.tokenizer.decode(ids[0, prefix:], skip_special_tokens=False)
    try:
        items = parse_generated_cot(raw)
        conditioned = render_units(instruction, grouped_units(instruction, items))
        reason = None
    except (ValueError, TypeError, KeyError) as exc:
        items, conditioned, reason = [], instruction, str(exc)
    return {
        "raw": raw,
        "items": items,
        "conditioning_prompt": conditioned,
        "fallback_reason": reason,
    }


@torch.no_grad()
def edit(
    pipe, instruction, images, *, mode="online", cot=None, max_new_tokens=256, **kwargs
):
    """Direct, inline, explicit oracle, or online two-pass inference."""
    result = {
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
        )
    elif mode == "oracle":
        items = parse_generated_cot(cot)
        result.update(
            items=items,
            conditioning_prompt=render_units(
                instruction, grouped_units(instruction, items)
            ),
        )
    elif mode not in {"direct", "inline"}:
        raise ValueError(f"Unknown mode: {mode}")
    spans_in(result["conditioning_prompt"])
    if "<|mt_" in kwargs.get("negative_prompt", ""):
        raise ValueError("CFG negative branch must not contain mask tokens")
    pipe.scheduler.training = False
    image = pipe(result["conditioning_prompt"], edit_image=images, **kwargs)
    return image, result
