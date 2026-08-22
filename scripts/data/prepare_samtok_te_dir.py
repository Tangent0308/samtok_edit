#!/usr/bin/env python3
"""Merge the SAMTok tokenizer with Qwen-Image-Edit-2511 image preprocessing."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from glob import glob
from pathlib import Path


DEFAULT_SAMTOK_TE = Path(
    "/mnt/bn/strategy-mllm-train/user/tanyue/models/SAMTok/"
    "Qwen2.5-VL-7B-SAMTok-gres-ft"
)
DEFAULT_QWEN_2511 = Path(
    "/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/"
    "Qwen-Image-Edit-2511"
)


def copy_if_present(source_dir: Path, output_dir: Path, name: str) -> bool:
    source = source_dir / name
    if not source.exists():
        return False
    shutil.copy2(source, output_dir / name)
    return True


def prepare(samtok_dir: Path, qwen_2511_dir: Path, output_dir: Path):
    if not (samtok_dir / "config.json").is_file():
        raise FileNotFoundError(f"Invalid SAMTok checkpoint directory: {samtok_dir}")
    processor_dir = (
        qwen_2511_dir / "processor"
        if (qwen_2511_dir / "processor").is_dir()
        else qwen_2511_dir
    )
    if not (processor_dir / "preprocessor_config.json").is_file():
        raise FileNotFoundError(f"Invalid Qwen-Image-Edit-2511 processor: {processor_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "special_tokens_map.json",
        "chat_template.json",
        "chat_template.jinja",
    ]
    for name in tokenizer_files:
        copy_if_present(samtok_dir, output_dir, name)
    for name in ["preprocessor_config.json", "video_preprocessor_config.json"]:
        copy_if_present(processor_dir, output_dir, name)
    for name in ["config.json", "generation_config.json"]:
        copy_if_present(samtok_dir, output_dir, name)

    from transformers import Qwen2VLProcessor

    processor = Qwen2VLProcessor.from_pretrained(output_dir)
    tokenizer = processor.tokenizer
    required_tokens = [
        "<|mt_start|>",
        "<|mt_0000|>",
        "<|mt_0511|>",
        "<|mt_end|>",
    ]
    token_ids = {}
    for token in required_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id == tokenizer.unk_token_id:
            raise ValueError(f"SAMTok tokenizer is missing required token {token}")
        token_ids[token] = token_id

    config = json.loads((samtok_dir / "config.json").read_text(encoding="utf-8"))
    vocab_size = config.get("vocab_size") or config.get("text_config", {}).get(
        "vocab_size"
    )
    if len(tokenizer) != vocab_size:
        raise ValueError(
            f"Tokenizer/model vocabulary mismatch: {len(tokenizer)} != {vocab_size}"
        )

    diffsynth_root = Path(__file__).resolve().parents[2] / "DiffSynth-Studio"
    sys.path.insert(0, str(diffsynth_root))
    from diffsynth.core.loader import hash_model_file

    shards = sorted(glob(str(samtok_dir / "model*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"No model*.safetensors shards under {samtok_dir}")
    model_hash = hash_model_file(shards)
    report = {
        "samtok_checkpoint": str(samtok_dir),
        "qwen_image_edit_checkpoint": str(qwen_2511_dir),
        "processor_class": type(processor).__name__,
        "tokenizer_length": len(tokenizer),
        "model_vocab_size": vocab_size,
        "tie_word_embeddings": config.get("tie_word_embeddings"),
        "required_token_ids": token_ids,
        "te_model_hash": model_hash,
    }
    (output_dir / "samtok_edit_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samtok_dir", type=Path, default=DEFAULT_SAMTOK_TE)
    parser.add_argument("--qwen_2511_dir", type=Path, default=DEFAULT_QWEN_2511)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "models" / "merged_samtok_te",
    )
    args = parser.parse_args()
    prepare(args.samtok_dir.resolve(), args.qwen_2511_dir.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
