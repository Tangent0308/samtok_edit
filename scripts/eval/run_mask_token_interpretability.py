#!/usr/bin/env python3
"""Run paired edit_umt inference while recording mask-token-to-source attention."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from einops import rearrange
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in [
    REPO_ROOT / "DiffSynth-Studio",
    REPO_ROOT / "scripts" / "inference",
    REPO_ROOT / "scripts" / "eval",
]:
    sys.path.insert(0, str(path))

from diffsynth.models.qwen_image_dit import apply_rotary_emb_qwen  # noqa: E402
from diffsynth.pipelines.qwen_image_samtok import (  # noqa: E402
    EDIT_DROP_IDX,
    build_edit_model_inputs,
)
from infer_samtok_edit import (  # noqa: E402
    DEFAULT_QWEN_2511,
    DEFAULT_SAMTOK_TE,
    build_pipeline,
    run_edit,
)
from run_eval import _atomic_write_json, sha256_file  # noqa: E402
from run_stage2_eval import DEFAULT_MERGED_TE, official_output_size  # noqa: E402


DEFAULT_EXPERIMENT_ROOT = Path(
    "/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/"
    "crispedit_refined/interpretability/dit_mask_token_counterfactual"
)
DEFAULT_STAGE1_TE_LORA = Path(
    "/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/"
    "crispedit_refined_4node/crispedit-refined-4node-20260910-run2/"
    "stage1_te_lora/step-2648.safetensors"
)
DEFAULT_STAGE2_DIT_LORA = Path(
    "/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/"
    "crispedit_refined_4node/crispedit-refined-4node-20260910-run2/"
    "stage2_dit_lora/step-5296.safetensors"
)


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def locate_mask_token_positions(pipe, prompt: str, span: str, source: Image.Image) -> list[int]:
    model_inputs = build_edit_model_inputs(pipe, prompt, [source])
    input_ids = model_inputs.input_ids[0].tolist()
    span_ids = pipe.processor.tokenizer(
        span, add_special_tokens=False, return_tensors="pt"
    ).input_ids[0].tolist()
    if len(span_ids) != 4:
        raise ValueError(f"SAMTok span is not four atomic tokens: {span_ids}")
    starts = [
        index
        for index in range(len(input_ids) - len(span_ids) + 1)
        if input_ids[index : index + len(span_ids)] == span_ids
    ]
    if len(starts) != 1:
        raise ValueError(f"Expected one mask span in templated input, found {starts}")
    positions = [starts[0] + offset - EDIT_DROP_IDX for offset in range(len(span_ids))]
    if min(positions) < 0:
        raise ValueError(f"Mask span fell inside dropped template prefix: {positions}")
    return positions


class MaskTokenAttentionProbe:
    """Capture exact joint-attention probabilities without changing DiT outputs."""

    def __init__(self, layer_ids: list[int], step_ids: list[int]):
        self.layer_ids = set(layer_ids)
        self.step_ids = set(step_ids)
        self.mask_positions: list[int] = []
        self.current: dict | None = None
        self.records: list[dict] = []
        self.handles = []

    def install(self, pipe) -> None:
        for layer_id in sorted(self.layer_ids):
            attention = pipe.dit.transformer_blocks[layer_id].attn
            handle = attention.register_forward_pre_hook(
                lambda module, args, kwargs, layer_id=layer_id: self._pre_hook(
                    layer_id, module, args, kwargs
                ),
                with_kwargs=True,
            )
            self.handles.append(handle)

        original = pipe.cfg_guided_model_fn

        def wrapped_cfg(pipe_self, model_fn, cfg_scale, inputs_shared, inputs_posi, inputs_nega, **other):
            call_index = 0

            def observed_model_fn(**kwargs):
                nonlocal call_index
                branch = "positive" if call_index == 0 else "negative"
                call_index += 1
                self._begin_forward(branch, kwargs)
                try:
                    return model_fn(**kwargs)
                finally:
                    self.current = None

            return original(
                observed_model_fn,
                cfg_scale,
                inputs_shared,
                inputs_posi,
                inputs_nega,
                **other,
            )

        pipe.cfg_guided_model_fn = MethodType(wrapped_cfg, pipe)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def reset(self, mask_positions: list[int]) -> None:
        self.mask_positions = list(mask_positions)
        self.records = []
        self.current = None

    def _begin_forward(self, branch: str, kwargs: dict) -> None:
        progress_id = int(kwargs["progress_id"])
        if branch != "positive" or progress_id not in self.step_ids:
            self.current = None
            return
        latents = kwargs["latents"]
        output_shape = (latents.shape[2] // 2, latents.shape[3] // 2)
        edit_latents = kwargs.get("edit_latents")
        if isinstance(edit_latents, list):
            if len(edit_latents) != 1:
                raise ValueError("Interpretability probe requires exactly one source image")
            source_latents = edit_latents[0]
        else:
            source_latents = edit_latents
        if source_latents is None:
            raise ValueError("Interpretability probe requires source edit latents")
        source_shape = (source_latents.shape[2] // 2, source_latents.shape[3] // 2)
        self.current = {
            "progress_id": progress_id,
            "timestep": float(kwargs["timestep"].flatten()[0].item()),
            "source_offset": int(output_shape[0] * output_shape[1]),
            "source_shape": source_shape,
        }

    @torch.no_grad()
    def _pre_hook(self, layer_id: int, module, args, kwargs):
        if self.current is None:
            return None
        image = kwargs["image"]
        text = kwargs["text"]
        rotary = kwargs.get("image_rotary_emb")
        attention_mask = kwargs.get("attention_mask")
        if max(self.mask_positions) >= text.shape[1]:
            raise ValueError(
                f"Mask-token position {max(self.mask_positions)} exceeds text length {text.shape[1]}"
            )

        img_q = rearrange(
            module.to_q(image), "b s (h d) -> b h s d", h=module.num_heads
        )
        img_k = rearrange(
            module.to_k(image), "b s (h d) -> b h s d", h=module.num_heads
        )
        txt_q = rearrange(
            module.add_q_proj(text), "b s (h d) -> b h s d", h=module.num_heads
        )
        txt_k = rearrange(
            module.add_k_proj(text), "b s (h d) -> b h s d", h=module.num_heads
        )
        img_q = module.norm_q(img_q)
        img_k = module.norm_k(img_k)
        txt_q = module.norm_added_q(txt_q)
        txt_k = module.norm_added_k(txt_k)
        if rotary is not None:
            image_frequencies, text_frequencies = rotary
            img_q = apply_rotary_emb_qwen(img_q, image_frequencies)
            img_k = apply_rotary_emb_qwen(img_k, image_frequencies)
            txt_q = apply_rotary_emb_qwen(txt_q, text_frequencies)
            txt_k = apply_rotary_emb_qwen(txt_k, text_frequencies)

        indices = torch.as_tensor(self.mask_positions, device=text.device)
        mask_queries = txt_q.index_select(2, indices)
        joint_keys = torch.cat([txt_k, img_k], dim=2)
        logits = torch.matmul(
            mask_queries.float(), joint_keys.float().transpose(-2, -1)
        ) / math.sqrt(module.head_dim)
        if attention_mask is not None:
            query_rows = attention_mask[:, :, indices, :]
            logits = logits + query_rows.float()
        probabilities = torch.softmax(logits, dim=-1)

        text_length = text.shape[1]
        source_start = text_length + self.current["source_offset"]
        source_height, source_width = self.current["source_shape"]
        source_stop = source_start + source_height * source_width
        source_probabilities = probabilities[..., source_start:source_stop]
        raw_source_mass = source_probabilities.sum(dim=-1).mean().item()
        mask_to_source = source_probabilities.mean(dim=(0, 1, 2))
        mask_to_source = mask_to_source / mask_to_source.sum().clamp_min(1e-12)

        source_queries = img_q[
            :, :, self.current["source_offset"] : self.current["source_offset"]
            + source_height * source_width
        ]
        source_logits = torch.matmul(
            source_queries.float(), joint_keys.float().transpose(-2, -1)
        ) / math.sqrt(module.head_dim)
        if attention_mask is not None:
            source_query_start = text_length + self.current["source_offset"]
            source_query_stop = source_query_start + source_height * source_width
            source_logits = source_logits + attention_mask[
                :, :, source_query_start:source_query_stop, :
            ].float()
        source_attention = torch.softmax(source_logits, dim=-1)
        source_to_mask = source_attention.index_select(-1, indices).sum(dim=-1)
        raw_mask_key_mass = source_to_mask.mean().item()
        source_to_mask = source_to_mask.mean(dim=(0, 1))
        source_to_mask = source_to_mask / source_to_mask.sum().clamp_min(1e-12)
        self.records.append(
            {
                "layer_id": layer_id,
                "progress_id": self.current["progress_id"],
                "timestep": self.current["timestep"],
                "raw_source_attention_mass": raw_source_mass,
                "raw_mask_key_attention_mass": raw_mask_key_mass,
                "source_height": source_height,
                "source_width": source_width,
                "mask_query_to_source_heatmap": mask_to_source.reshape(
                    source_height, source_width
                ).cpu().numpy(),
                "source_query_to_mask_heatmap": source_to_mask.reshape(
                    source_height, source_width
                ).cpu().numpy(),
            }
        )
        return None


def save_attention(path: Path, records: list[dict], expected: int) -> dict:
    if len(records) != expected:
        raise RuntimeError(f"Expected {expected} attention maps, captured {len(records)}")
    identities = [(row["progress_id"], row["layer_id"]) for row in records]
    if len(set(identities)) != expected:
        raise RuntimeError(f"Duplicate/missing attention layer-step identities: {identities}")
    records = sorted(records, key=lambda row: (row["progress_id"], row["layer_id"]))
    mask_to_source_maps = np.stack(
        [row.pop("mask_query_to_source_heatmap") for row in records]
    )
    source_to_mask_maps = np.stack(
        [row.pop("source_query_to_mask_heatmap") for row in records]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        mask_query_to_source_heatmaps=mask_to_source_maps,
        source_query_to_mask_heatmaps=source_to_mask_maps,
        layer_ids=np.asarray([row["layer_id"] for row in records]),
        progress_ids=np.asarray([row["progress_id"] for row in records]),
        timesteps=np.asarray([row["timestep"] for row in records]),
        raw_source_attention_mass=np.asarray(
            [row["raw_source_attention_mass"] for row in records]
        ),
        raw_mask_key_attention_mass=np.asarray(
            [row["raw_mask_key_attention_mass"] for row in records]
        ),
    )
    return {
        "num_maps": len(records),
        "source_grid": [records[0]["source_width"], records[0]["source_height"]],
        "mean_raw_source_attention_mass": float(
            np.mean([row["raw_source_attention_mass"] for row in records])
        ),
        "mean_raw_mask_key_attention_mass": float(
            np.mean([row["raw_mask_key_attention_mass"] for row in records])
        ),
        "layer_step_pairs": [list(identity) for identity in identities],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOT / "data" / "interventions.jsonl",
    )
    parser.add_argument("--output_root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--qwen_2511_dir", type=Path, default=DEFAULT_QWEN_2511)
    parser.add_argument("--samtok_te_dir", type=Path, default=DEFAULT_SAMTOK_TE)
    parser.add_argument("--merged_te_dir", type=Path, default=DEFAULT_MERGED_TE)
    parser.add_argument("--stage1_te_lora", type=Path, default=DEFAULT_STAGE1_TE_LORA)
    parser.add_argument("--stage2_dit_lora", type=Path, default=DEFAULT_STAGE2_DIT_LORA)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--layers", default="5,15,30,45,59")
    parser.add_argument("--steps", default="0,10,20,30,39")
    args = parser.parse_args()

    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    world_size = int(os.environ.get("LOCAL_WORLD_SIZE", os.environ.get("WORLD_SIZE", "1")))
    device = f"cuda:{rank}"
    layers = [int(value) for value in args.layers.split(",")]
    steps = [int(value) for value in args.steps.split(",")]
    if max(steps) >= args.num_inference_steps:
        raise ValueError("Attention step IDs must be smaller than num_inference_steps")

    rows = read_rows(args.manifest)
    local_rows = [row for index, row in enumerate(rows) if index % world_size == rank]
    if not local_rows:
        print(f"[rank={rank}] no assigned cases", flush=True)
        return
    pipe = build_pipeline(
        args.qwen_2511_dir,
        args.samtok_te_dir,
        args.merged_te_dir,
        te_lora=args.stage1_te_lora,
        dit_lora=args.stage2_dit_lora,
        device=device,
    )
    probe = MaskTokenAttentionProbe(layers, steps)
    probe.install(pipe)

    for row in local_rows:
        with Image.open(row["source_image"]) as handle:
            source = handle.convert("RGB")
        output_width, output_height = official_output_size(source)
        case_seed = args.seed + int(row["case_index"])
        for condition in ("original", "alternate"):
            prompt = row[f"{condition}_prompt"]
            span = row[f"{condition}_mask_span"]
            token_positions = locate_mask_token_positions(pipe, prompt, span, source)
            probe.reset(token_positions)
            started = time.perf_counter()
            output = run_edit(
                pipe,
                source,
                prompt,
                seed=case_seed,
                num_inference_steps=args.num_inference_steps,
                cfg_scale=args.cfg_scale,
                mt_cot=None,
                enable_samtok_cot=False,
                output_height=output_height,
                output_width=output_width,
            )
            elapsed = time.perf_counter() - started
            case_dir = args.output_root / "runs" / f"case_{int(row['case_index']):02d}_{row['benchmark_id']}"
            output_path = case_dir / f"{condition}_output.png"
            attention_path = case_dir / f"{condition}_attention.npz"
            record_path = case_dir / f"{condition}_record.json"
            case_dir.mkdir(parents=True, exist_ok=True)
            output.save(output_path)
            attention_audit = save_attention(
                attention_path, probe.records, len(layers) * len(steps)
            )
            user_audit = getattr(pipe, "last_user_mask_audit", None) or {}
            if not (
                user_audit.get("user_mask_span_count") == 1
                and user_audit.get("user_mask_spans_atomic") is True
                and user_audit.get("user_mask_spans_in_template") is True
            ):
                raise RuntimeError(f"Prompt mask-token audit failed: {user_audit}")
            record = {
                "case_index": row["case_index"],
                "benchmark_id": row["benchmark_id"],
                "condition": condition,
                "prompt": prompt,
                "mask_span": span,
                "mask_token_prompt_positions_after_drop": token_positions,
                "output": str(output_path.resolve()),
                "attention": str(attention_path.resolve()),
                "elapsed_seconds": elapsed,
                "seed": case_seed,
                "num_inference_steps": args.num_inference_steps,
                "cfg_scale": args.cfg_scale,
                "output_size": list(output.size),
                "user_mask_audit": user_audit,
                "attention_audit": attention_audit,
                "models": {
                    "qwen_2511": str(args.qwen_2511_dir.resolve()),
                    "samtok_te": str(args.samtok_te_dir.resolve()),
                    "merged_te": str(args.merged_te_dir.resolve()),
                    "stage1_te_lora": str(args.stage1_te_lora.resolve()),
                    "stage1_te_lora_sha256": sha256_file(args.stage1_te_lora),
                    "stage2_dit_lora": str(args.stage2_dit_lora.resolve()),
                    "stage2_dit_lora_sha256": sha256_file(args.stage2_dit_lora),
                },
                "worker_rank": rank,
                "world_size": world_size,
            }
            _atomic_write_json(record_path, record)
            print(
                f"[rank={rank}] case={row['case_index']} condition={condition} "
                f"seconds={elapsed:.2f} attention_maps={len(probe.records)}",
                flush=True,
            )
    probe.close()


if __name__ == "__main__":
    main()
