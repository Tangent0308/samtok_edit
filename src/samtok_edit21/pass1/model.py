"""Native Qwen3-VL processing, assistant-only labels, and pure LM LoRA."""

from __future__ import annotations

import hashlib
import json
import os
import re
from contextlib import contextmanager

import torch
import torch.nn.functional as F

from samtok_edit21.pass1.protocol import EMPTY_THINK, prompt

MODEL = os.environ.get("SAMTOK_PASS1_MODEL", "models/Qwen3-VL-8B-SAMTok")
TARGET_RE = r".*language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"


def processor_for(model=MODEL, max_pixels=2097152):
    from transformers import AutoProcessor

    p = AutoProcessor.from_pretrained(model, local_files_only=True)
    p.image_processor.size = {"shortest_edge": 65536, "longest_edge": max_pixels}
    names = ["<|mt_start|>", *[f"<|mt_{i:04d}|>" for i in range(512)], "<|mt_end|>"]
    ids = [p.tokenizer.encode(x, add_special_tokens=False) for x in names]
    if any(len(x) != 1 for x in ids) or len({x[0] for x in ids}) != 514:
        raise ValueError("Mask vocabulary is not atomic")
    return p


def prefix_text(processor, request):
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt(request)}],
        }
    ]
    return (
        processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        + EMPTY_THINK
    )


def encode(processor, image, request, answer=None):
    prefix = prefix_text(processor, request)
    if answer is None:
        return processor(
            text=[prefix], images=[image.convert("RGB")], return_tensors="pt"
        )
    from samtok_edit21.pass1.protocol import parse_answer

    result = parse_answer(json.dumps(answer), request)
    if not result["complete"]:
        raise ValueError(result["errors"])
    suffix = (
        json.dumps(answer, ensure_ascii=False, separators=(",", ":")) + "<|im_end|>"
    )
    inputs = processor(
        text=[prefix + suffix], images=[image.convert("RGB")], return_tensors="pt"
    )
    # Locate the exact separately tokenized suffix in the real expanded visual sequence.
    suffix_ids = processor.tokenizer.encode(suffix, add_special_tokens=False)
    ids = inputs.input_ids[0]
    start = len(ids) - len(suffix_ids)
    if ids[start:].tolist() != suffix_ids or start <= 0:
        raise ValueError("Assistant suffix boundary changed under tokenization")
    positions = torch.arange(start, len(ids))
    groups = torch.full((len(suffix_ids),), 2, dtype=torch.long)  # format
    code_start = processor.tokenizer.convert_tokens_to_ids("<|mt_start|>")
    starts = [i for i, x in enumerate(suffix_ids) if x == code_start]
    if len(starts) != len(answer):
        raise ValueError("Missing supervised mask span")
    for i, t in zip(starts, request["targets"]):
        groups[i + 1 : i + 3] = 1 if t["locator"] == "mask" else 0
    expected = [
        processor.tokenizer.encode(x["mask_2d"], add_special_tokens=False)
        for x in answer
    ]
    if any(suffix_ids[i : i + 4] != e for i, e in zip(starts, expected)):
        raise ValueError("Supervised codes changed")
    return inputs, positions - 1, ids[positions].clone(), groups


def grouped_loss(logits, labels, groups, mode="grouped"):
    ce = F.cross_entropy(logits.float(), labels, reduction="none")
    if mode == "ce":
        return ce.mean(), {"ce": float(ce.detach().mean())}
    weight = (1.0, 0.25, 0.25)
    pieces, stats, total = [], {}, 0.0
    for g, w in enumerate(weight):
        active = groups == g
        if active.any():
            v = ce[active].mean()
            pieces.append(w * v)
            stats[("prediction", "copy", "format")[g]] = float(v.detach())
            total += w
    return sum(pieces) / total, stats


def load_model(device, model=MODEL, adapter=None, *, train=False, rank=64):
    from transformers import Qwen3VLForConditionalGeneration

    m = Qwen3VLForConditionalGeneration.from_pretrained(
        model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
        device_map={"": device},
    )
    m.requires_grad_(False)
    if train or adapter:
        m = attach_lora(m, adapter=adapter, train=train, rank=rank)
    return m.train() if train else m.eval()


def attach_lora(m, adapter=None, *, train=False, rank=64):
    if getattr(m, "peft_config", None):
        raise ValueError("Attach Pass1 only to a model without existing TE adapters")
    m.requires_grad_(False)
    # Match only LM layers; keep native model/wrapper paths intact for Pass2.
    from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict
    from safetensors.torch import load_file

    cfg = LoraConfig(
        r=rank,
        lora_alpha=2 * rank,
        lora_dropout=0.05,
        target_modules=TARGET_RE,
        bias="none",
    )
    if adapter:
        cfg = LoraConfig.from_pretrained(adapter)
    m = inject_adapter_in_model(cfg, m, adapter_name="pass1_grounding")
    for name, p in m.named_parameters():
        p.requires_grad_(train and "lora_" in name)
        if "lora_" in name:
            p.data = p.data.float()
    # Injection follows the BF16 base dtype. Promote BEFORE copy_ from the
    # checkpoint; casting after loading would silently round FP32 LoRA weights.
    if adapter:
        result = set_peft_model_state_dict(
            m,
            load_file(str(adapter) + "/adapter_model.safetensors"),
            adapter_name="pass1_grounding",
        )
        if result.unexpected_keys or any("lora_" in k for k in result.missing_keys):
            raise ValueError(f"Adapter mismatch: {result}")
    matched = [n for n, mod in m.named_modules() if hasattr(mod, "lora_A")]
    if len(matched) != 252 or any(not re.fullmatch(TARGET_RE, n) for n in matched):
        raise ValueError(f"Unexpected LoRA targets: {matched}")
    return m


@contextmanager
def adapter_enabled(model, enabled):
    from peft.tuners.tuners_utils import BaseTunerLayer

    layers = [x for x in model.modules() if isinstance(x, BaseTunerLayer)]
    states = [
        (x.disable_adapters, [(p, p.requires_grad) for p in x.parameters()])
        for x in layers
    ]
    for x in layers:
        x.enable_adapters(enabled)
    try:
        yield
    finally:
        for x, (disabled, grads) in zip(layers, states):
            x.enable_adapters(not disabled)
            for p, flag in grads:
                p.requires_grad_(flag)


def frozen_fingerprint(model):
    """Hash frozen tensors in bounded CPU chunks; LoRA tensor names excluded."""
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        if "lora_" in name:
            continue
        h.update(name.encode())
        flat = p.detach().reshape(-1)
        for s in range(0, flat.numel(), 8_000_000):
            h.update(
                flat[s : s + 8_000_000]
                .contiguous()
                .view(torch.uint8)
                .cpu()
                .numpy()
                .tobytes()
            )
    return h.hexdigest()
