#!/usr/bin/env python3
"""Run one two-pass SAMTok-guided Qwen-Image-Edit-2511 edit."""

from __future__ import annotations

import argparse
import json
import os
import sys
from glob import glob
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[2]
for path in [_REPO_ROOT / "DiffSynth-Studio", _REPO_ROOT / "scripts" / "data"]:
    sys.path.insert(0, str(path))

from diffsynth.pipelines.qwen_image_samtok import (  # noqa: E402
    ModelConfig,
    QwenImageSamtokPipeline,
)


DEFAULT_QWEN_2511 = Path(
    "/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/"
    "Qwen-Image-Edit-2511"
)
DEFAULT_SAMTOK_TE = Path(
    "/mnt/bn/strategy-mllm-train/user/tanyue/models/SAMTok/"
    "Qwen2.5-VL-7B-SAMTok-gres-ft"
)


def build_pipeline(
    qwen_2511_dir: Path,
    samtok_te_dir: Path,
    merged_te_dir: Path,
    te_lora: Path | None = None,
    dit_lora: Path | None = None,
    device: str = "cuda",
):
    dit_shards = sorted(
        glob(
            str(
                qwen_2511_dir
                / "transformer"
                / "diffusion_pytorch_model*.safetensors"
            )
        )
    )
    te_shards = sorted(glob(str(samtok_te_dir / "model*.safetensors")))
    vae = qwen_2511_dir / "vae" / "diffusion_pytorch_model.safetensors"
    if not dit_shards or not te_shards or not vae.is_file():
        raise FileNotFoundError("Missing Qwen-Image-Edit-2511 or SAMTok model shards")
    pipe = QwenImageSamtokPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(path=dit_shards),
            ModelConfig(path=te_shards),
            ModelConfig(path=str(vae)),
        ],
        tokenizer_config=ModelConfig(path=str(merged_te_dir)),
        processor_config=ModelConfig(path=str(merged_te_dir)),
    )
    if te_lora:
        pipe.load_lora(pipe.text_encoder, str(te_lora))
    if dit_lora:
        pipe.load_lora(pipe.dit, str(dit_lora))
    return pipe


def run_edit(
    pipe,
    image,
    prompt,
    seed=0,
    num_inference_steps=40,
    cfg_scale=4.0,
    mt_cot=None,
    enable_samtok_cot=True,
):
    return pipe(
        prompt,
        edit_image=[image],
        seed=seed,
        num_inference_steps=num_inference_steps,
        cfg_scale=cfg_scale,
        height=image.size[1],
        width=image.size[0],
        edit_image_auto_resize=True,
        zero_cond_t=True,
        mt_cot=mt_cot,
        enable_samtok_cot=enable_samtok_cot,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image_path", type=Path, required=True)
    parser.add_argument("--save_path", type=Path, required=True)
    parser.add_argument("--qwen_2511_dir", type=Path, default=DEFAULT_QWEN_2511)
    parser.add_argument("--samtok_te_dir", type=Path, default=DEFAULT_SAMTOK_TE)
    parser.add_argument(
        "--merged_te_dir",
        type=Path,
        default=_REPO_ROOT / "models" / "merged_samtok_te",
    )
    parser.add_argument("--te_lora", type=Path, default=None)
    parser.add_argument("--dit_lora", type=Path, default=None)
    parser.add_argument("--mt_cot", default=None, help="Explicit GT CoT ablation")
    parser.add_argument("--disable_cot", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sam2_ckpt", type=Path, default=None)
    parser.add_argument("--mask_tokenizer_ckpt", type=Path, default=None)
    args = parser.parse_args()

    pipe = build_pipeline(
        args.qwen_2511_dir,
        args.samtok_te_dir,
        args.merged_te_dir,
        te_lora=args.te_lora,
        dit_lora=args.dit_lora,
        device=args.device,
    )
    image = Image.open(args.image_path).convert("RGB")
    output = run_edit(
        pipe,
        image,
        args.prompt,
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
        cfg_scale=args.cfg_scale,
        mt_cot=args.mt_cot,
        enable_samtok_cot=not args.disable_cot,
    )
    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(args.save_path)
    print(
        json.dumps(
            {
                "mt_cot": pipe.last_mt_cot,
                "parse_layer": pipe.last_parse_layer,
                "pass1_raw": pipe.last_pass1_raw,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if pipe.last_mt_cot and args.sam2_ckpt and args.mask_tokenizer_ckpt:
        from samtok_codec import SamtokCodec

        codec = SamtokCodec(
            args.sam2_ckpt, args.mask_tokenizer_ckpt, device=args.device
        )
        masks = codec.decode(image, pipe.last_mt_cot)
        if masks:
            visualization = np.asarray(image).astype(np.float32).copy()
            for mask in masks:
                visualization[mask > 0] = (
                    visualization[mask > 0] * 0.5
                    + np.asarray([255, 0, 0], dtype=np.float32) * 0.5
                )
            mask_path = args.save_path.with_name(
                args.save_path.stem + "_pass1_mask" + args.save_path.suffix
            )
            Image.fromarray(visualization.astype(np.uint8)).save(mask_path)


if __name__ == "__main__":
    main()
