#!/usr/bin/env python3
"""Unified Stage-1/Stage-2 trainer for SAMTok-guided Qwen-Image-Edit."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_DIFFSYNTH_ROOT = os.path.join(_REPO_ROOT, "DiffSynth-Studio")
if _DIFFSYNTH_ROOT not in sys.path:
    sys.path.insert(0, _DIFFSYNTH_ROOT)

import accelerate
import torch
from tqdm import tqdm

from diffsynth.core.data.operators import (
    ImageCropAndResize,
    LoadImage,
    RouteByType,
    SequencialProcess,
    ToAbsolutePath,
    ToList,
)
from diffsynth.core.data.samtok_dataset import SamtokEditingDataset
from diffsynth.diffusion import DiffusionTrainingModule, ModelLogger
from diffsynth.diffusion.loss import FlowMatchSFTLoss, SamtokEditingLoss
from diffsynth.diffusion.parsers import add_general_config, add_image_size_config
from diffsynth.diffusion.runner import (
    get_optimizer_class,
    initialize_deepspeed_gradient_checkpointing,
    launch_data_process_task,
    launch_training_task,
    save_training_args,
)
from diffsynth.pipelines.qwen_image_samtok import QwenImageSamtokPipeline, ModelConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def trainable_parameter_audit(model):
    """Describe and validate the Stage-1 trainable parameter boundary."""

    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    invalid = [
        name
        for name, _ in trainable
        if not name.startswith("pipe.text_encoder.")
        or not (".lora_A." in name or ".lora_B." in name)
    ]
    dtypes = {}
    frozen_dtypes = {}
    for _, parameter in trainable:
        dtype = str(parameter.dtype).removeprefix("torch.")
        dtypes[dtype] = dtypes.get(dtype, 0) + parameter.numel()
    for parameter in model.parameters():
        if not parameter.requires_grad:
            dtype = str(parameter.dtype).removeprefix("torch.")
            frozen_dtypes[dtype] = frozen_dtypes.get(dtype, 0) + parameter.numel()
    report = {
        "trainable_tensors": len(trainable),
        "trainable_parameters": sum(parameter.numel() for _, parameter in trainable),
        "trainable_parameter_dtypes": dtypes,
        "frozen_parameter_dtypes": frozen_dtypes,
        "text_encoder_trainable_parameters": sum(
            parameter.numel()
            for name, parameter in trainable
            if name.startswith("pipe.text_encoder.")
        ),
        "dit_trainable_parameters": sum(
            parameter.numel()
            for name, parameter in trainable
            if name.startswith("pipe.dit.")
        ),
        "vae_trainable_parameters": sum(
            parameter.numel()
            for name, parameter in trainable
            if name.startswith("pipe.vae.")
        ),
        "invalid_trainable_names": invalid[:10],
    }
    if not trainable:
        raise RuntimeError("No trainable parameters were found")
    if invalid:
        raise RuntimeError(
            "Stage 1 must train only text-encoder LoRA parameters; found "
            f"{invalid[:10]}"
        )
    if any(parameter.dtype != torch.float32 for _, parameter in trainable):
        raise RuntimeError("Stage-1 text-encoder LoRA parameters must remain fp32")
    return report


def stage2_trainable_parameter_audit(model):
    """Validate that Stage 2 exposes only the official DiT LoRA tensors."""

    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    invalid = [
        name
        for name, _ in trainable
        if not name.startswith("pipe.dit.")
        or not (".lora_A." in name or ".lora_B." in name)
    ]
    dtype_counts = Counter()
    frozen_dtype_counts = Counter()
    for _, parameter in trainable:
        dtype_counts[str(parameter.dtype).removeprefix("torch.")] += parameter.numel()
    for parameter in model.parameters():
        if not parameter.requires_grad:
            frozen_dtype_counts[
                str(parameter.dtype).removeprefix("torch.")
            ] += parameter.numel()
    target_families = Counter()
    for name, _ in trainable:
        for family in [
            "to_q",
            "to_k",
            "to_v",
            "add_q_proj",
            "add_k_proj",
            "add_v_proj",
            "to_out.0",
            "to_add_out",
            "img_mlp.net.2",
            "img_mod.1",
            "txt_mlp.net.2",
            "txt_mod.1",
        ]:
            if f".{family}." in name:
                target_families[family] += 1
                break
    report = {
        "trainable_tensors": len(trainable),
        "trainable_parameters": sum(parameter.numel() for _, parameter in trainable),
        "trainable_parameter_dtypes": dict(dtype_counts),
        "frozen_parameter_dtypes": dict(frozen_dtype_counts),
        "dit_trainable_parameters": sum(
            parameter.numel()
            for name, parameter in trainable
            if name.startswith("pipe.dit.")
        ),
        "text_encoder_trainable_parameters": sum(
            parameter.numel()
            for name, parameter in trainable
            if name.startswith("pipe.text_encoder.")
        ),
        "vae_trainable_parameters": sum(
            parameter.numel()
            for name, parameter in trainable
            if name.startswith("pipe.vae.")
        ),
        "target_family_trainable_tensors": dict(sorted(target_families.items())),
        "invalid_trainable_names": invalid[:10],
    }
    if not trainable:
        raise RuntimeError("Stage 2 found no trainable DiT LoRA parameters")
    if invalid:
        raise RuntimeError(f"Stage 2 trainable boundary is invalid: {invalid[:10]}")
    if any(parameter.dtype != torch.bfloat16 for _, parameter in trainable):
        raise RuntimeError("Official Stage-2 DiT LoRA parameters must be bfloat16")
    if not target_families:
        raise RuntimeError("Stage 2 did not match any official DiT LoRA target family")
    return report


def gradient_audit(model):
    """Return finite-gradient statistics without retaining gradient tensors."""

    gradient_norms = []
    frozen_gradient_tensors = 0
    trainable_gradient_tensors = 0
    for parameter in model.parameters():
        if parameter.requires_grad:
            if parameter.grad is not None:
                trainable_gradient_tensors += 1
                gradient_norms.append(parameter.grad.detach().float().norm())
        elif parameter.grad is not None:
            frozen_gradient_tensors += 1
    if gradient_norms:
        norms = torch.stack(gradient_norms)
        total_norm = float(norms.norm().item())
        nonzero_tensors = int((norms > 0).sum().item())
        finite = bool(torch.isfinite(norms).all().item()) and math.isfinite(total_norm)
    else:
        total_norm = 0.0
        nonzero_tensors = 0
        finite = True
    return {
        "grad_norm_before_clip": total_norm,
        "trainable_grad_tensors": trainable_gradient_tensors,
        "nonzero_grad_tensors": nonzero_tensors,
        "frozen_grad_tensors": frozen_gradient_tensors,
        "gradients_finite": finite,
    }


def gather_scalar(accelerator, value, dtype=torch.float32):
    """Gather one scalar from every rank as a rank-ordered Python list."""

    tensor = torch.tensor([value], dtype=dtype, device=accelerator.device)
    return accelerator.gather(tensor).detach().cpu().tolist()


def scalar_range(values):
    values = [float(value) for value in values]
    return {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def tensor_tree_summary(value):
    """Return JSON-safe cache structure/dtype/shape metadata."""

    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype).removeprefix("torch."),
            "finite": bool(torch.isfinite(value).all().item())
            if value.is_floating_point()
            else True,
        }
    if isinstance(value, dict):
        return {key: tensor_tree_summary(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [tensor_tree_summary(item) for item in value]
    return type(value).__name__


class IndexedCacheDataset(torch.utils.data.Dataset):
    """Expose the physical cache file alongside each repeated cache item."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        cache_index = index % len(self.dataset.cached_data)
        return {
            "inputs": self.dataset[index],
            "cache_path": self.dataset.cached_data[cache_index],
            "cache_index": cache_index,
        }


def launch_data_process_task_samtok(
    accelerator,
    dataset,
    model,
    model_logger,
    args=None,
    **kwargs,
):
    """Stage 2a cache runner with row/type/tensor provenance sidecars."""

    if args is None:
        raise ValueError("SAMTok Stage 2a requires parsed arguments")
    if args.enable_model_cpu_offload or args.enable_optimizer_cpu_offload:
        raise ValueError("Stage 2a smoke audit does not support CPU offload")
    if accelerator.is_main_process:
        save_training_args(args)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        collate_fn=lambda batch: batch[0],
        num_workers=args.dataset_num_workers,
    )
    model.to(device=accelerator.device)
    model, dataloader = accelerator.prepare(model, dataloader)
    if len(dataset) != len(dataloader) * accelerator.num_processes:
        raise RuntimeError(
            "Stage 2a requires exact cache sharding: "
            f"global={len(dataset)}, local={len(dataloader)}, "
            f"world={accelerator.num_processes}"
        )

    global_type_counts = Counter()
    for data_id, data in enumerate(
        tqdm(dataloader, disable=not accelerator.is_local_main_process)
    ):
        source_row_id = int(data.get("_samtok_source_row_id", -1))
        sample_type = data.get("sample_type", "edit")
        type_to_id = {"edit_mt": 0, "edit": 1, "edit_umt": 2}
        sample_type_id = type_to_id.get(sample_type, -1)
        if source_row_id < 0 or sample_type_id < 0:
            raise RuntimeError(
                f"Invalid Stage 2a provenance: row={source_row_id}, type={sample_type}"
            )
        rank_source_rows = gather_scalar(
            accelerator, source_row_id, dtype=torch.int64
        )
        expected_rows = list(
            range(
                data_id * accelerator.num_processes,
                (data_id + 1) * accelerator.num_processes,
            )
        )
        if rank_source_rows != expected_rows:
            raise RuntimeError(
                f"Stage 2a DDP row sharding mismatch: {rank_source_rows} != {expected_rows}"
            )
        rank_type_ids = gather_scalar(
            accelerator, sample_type_id, dtype=torch.int64
        )
        id_to_type = {value: key for key, value in type_to_id.items()}
        global_type_counts.update(id_to_type[int(type_id)] for type_id in rank_type_ids)

        with torch.no_grad():
            cached_inputs = model(data)
        if not isinstance(cached_inputs, tuple) or len(cached_inputs) != 3:
            raise RuntimeError("Stage 2a cache must be a (shared, positive, negative) tuple")
        shared, positive, negative = cached_inputs
        required_shared = {"input_latents", "edit_latents"}
        required_positive = {"prompt_emb"}
        if not required_shared.issubset(shared) or not required_positive.issubset(positive):
            raise RuntimeError(
                "Stage 2a cache is missing required tensors: "
                f"shared={sorted(shared)}, positive={sorted(positive)}"
            )
        forbidden = {"samtok_cot_hidden", "samtok_cot_labels"}
        if forbidden.intersection(shared) or forbidden.intersection(positive):
            raise RuntimeError("Stage 2a unexpectedly cached NTP-only tensors")
        cache_summary = tensor_tree_summary(cached_inputs)

        rank_dir = Path(model_logger.output_path) / str(accelerator.process_index)
        rank_dir.mkdir(parents=True, exist_ok=True)
        cache_path = rank_dir / f"{data_id}.pth"
        torch.save(cached_inputs, cache_path)
        sidecar = {
            "cache_path": str(cache_path),
            "worker_rank": accelerator.process_index,
            "world_size": accelerator.num_processes,
            "local_cache_index": data_id,
            "metadata_index": source_row_id,
            "sample_type": sample_type,
            "prompt": data.get("prompt"),
            "has_mt_cot": bool(data.get("mt_cot")),
            "mt_cot_is_empty": data.get("mt_cot") == "```json\n[]\n```",
            "provenance": data.get("provenance"),
            "target_image_size": list(data["image"].size),
            "source_image_sizes": [list(image.size) for image in data["edit_image"]],
            "preset_te_lora_path": args.preset_lora_path,
            "cache_summary": cache_summary,
        }
        temporary = cache_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
        temporary.replace(cache_path.with_suffix(".json"))
        if args.debug_train_metrics:
            accelerator.print(
                "[SamtokDebug][stage2_cache] "
                + json.dumps(
                    {
                        "local_cache_index": data_id,
                        "rank_source_rows": rank_source_rows,
                        "rank_sample_type_ids": rank_type_ids,
                        "global_type_counts_so_far": dict(global_type_counts),
                        "local_cache": sidecar,
                    },
                    sort_keys=True,
                )
            )
    expected_counts = Counter(
        row.get("sample_type", "edit") for row in dataset.data
    )
    if global_type_counts != expected_counts:
        raise RuntimeError(
            f"Stage 2a type counts changed under DDP: {global_type_counts} != {expected_counts}"
        )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(
            "[SamtokDebug][stage2_cache_summary] "
            + json.dumps(
                {
                    "world_size": accelerator.num_processes,
                    "metadata_rows": len(dataset),
                    "cache_rows": sum(global_type_counts.values()),
                    "sample_type_counts": dict(global_type_counts),
                    "preset_te_lora_path": args.preset_lora_path,
                },
                sort_keys=True,
            )
        )
    accelerator.end_training()


def launch_training_task_stage2_debug(
    accelerator,
    dataset,
    model,
    model_logger,
    args=None,
    **kwargs,
):
    """Mirror the official Stage 2 runner while auditing every DDP step."""

    if args is None:
        raise ValueError("SAMTok Stage 2b debug runner requires parsed arguments")
    if not dataset.load_from_cache:
        raise ValueError("Stage 2b must read Stage 2a .pth cache files")
    if args.enable_model_cpu_offload or args.enable_optimizer_cpu_offload:
        raise ValueError("Stage 2b smoke audit does not support CPU offload")
    if accelerator.is_main_process:
        save_training_args(args)

    cache_manifests = {}
    physical_type_counts = Counter()
    physical_source_rows = set()
    for cache_file in dataset.cached_data:
        cache_path = Path(cache_file)
        sidecar_path = cache_path.with_suffix(".json")
        if not sidecar_path.is_file():
            raise FileNotFoundError(f"Missing Stage 2a cache sidecar: {sidecar_path}")
        record = json.loads(sidecar_path.read_text(encoding="utf-8"))
        cache_manifests[str(cache_path)] = record
        physical_type_counts[record["sample_type"]] += 1
        physical_source_rows.add(int(record["metadata_index"]))
    if len(physical_source_rows) != len(dataset.cached_data):
        raise RuntimeError("Stage 2a cache metadata indices are not one-to-one")

    optimizer = get_optimizer_class(args.customized_optimizer)(
        model.trainable_modules(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    indexed_dataset = IndexedCacheDataset(dataset)
    dataloader = torch.utils.data.DataLoader(
        indexed_dataset,
        shuffle=True,
        collate_fn=lambda batch: batch[0],
        num_workers=args.dataset_num_workers,
    )
    model.to(device=accelerator.device)
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )
    initialize_deepspeed_gradient_checkpointing(accelerator)

    audit = stage2_trainable_parameter_audit(accelerator.unwrap_model(model))
    optimizer_group = optimizer.param_groups[0]
    runtime_audit = {
        "world_size": accelerator.num_processes,
        "distributed_type": str(accelerator.distributed_type),
        "accelerate_mixed_precision": accelerator.mixed_precision,
        "pipeline_dtype": str(
            accelerator.unwrap_model(model).pipe.torch_dtype
        ).removeprefix("torch."),
        "physical_cache_rows": len(dataset.cached_data),
        "physical_cache_type_counts": dict(physical_type_counts),
        "dataset_repeat": args.dataset_repeat,
        "dataset_rows_per_epoch": len(dataset),
        "micro_steps_per_rank_per_epoch": len(dataloader),
        "num_epochs": args.num_epochs,
        "total_optimizer_steps": len(dataloader) * args.num_epochs,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_global_batch_size": (
            accelerator.num_processes * args.gradient_accumulation_steps
        ),
        "optimizer": type(optimizer).__name__,
        "optimizer_betas": list(optimizer_group["betas"]),
        "weight_decay": optimizer_group["weight_decay"],
        "base_learning_rate": args.learning_rate,
        "scheduler": type(scheduler.scheduler).__name__
        if hasattr(scheduler, "scheduler")
        else type(scheduler).__name__,
        "scheduler_note": "official ConstantLR defaults: factor=1/3, total_iters=5",
        "gradient_clipping_enabled": False,
        "zero_cond_t": args.zero_cond_t,
        "gradient_checkpointing": args.use_gradient_checkpointing,
        "find_unused_parameters": args.find_unused_parameters,
        "seed": args.seed,
    }
    if accelerator.is_main_process:
        print("[SamtokDebug][stage2_parameter_audit] " + json.dumps(audit, sort_keys=True))
        print("[SamtokDebug][stage2_runtime_audit] " + json.dumps(runtime_audit, sort_keys=True))

    probe_name, probe_parameter = None, None
    for name, parameter in accelerator.unwrap_model(model).named_parameters():
        if parameter.requires_grad and ".lora_B." in name:
            probe_name, probe_parameter = name, parameter
            break
    if probe_parameter is None:
        raise RuntimeError("No Stage 2 LoRA B tensor found for update audit")

    optimizer_step = 0
    for epoch_id in range(args.num_epochs):
        epoch_type_counts = Counter()
        epoch_source_counts = Counter()
        for micro_step_in_epoch, batch in enumerate(
            tqdm(dataloader, disable=not accelerator.is_local_main_process), 1
        ):
            with accelerator.accumulate(model):
                cache_path = Path(batch["cache_path"])
                cache_record = cache_manifests[str(cache_path)]
                type_to_id = {"edit_mt": 0, "edit": 1, "edit_umt": 2}
                sample_type_id = type_to_id.get(
                    cache_record["sample_type"], -1
                )
                if sample_type_id < 0:
                    raise RuntimeError(f"Bad cached sample type: {cache_record}")
                rank_type_ids = gather_scalar(
                    accelerator, sample_type_id, dtype=torch.int64
                )
                rank_source_rows = gather_scalar(
                    accelerator,
                    cache_record["metadata_index"],
                    dtype=torch.int64,
                )
                id_to_type = {value: key for key, value in type_to_id.items()}
                epoch_type_counts.update(
                    id_to_type[int(type_id)] for type_id in rank_type_ids
                )
                epoch_source_counts.update(int(row_id) for row_id in rank_source_rows)

                loss = model({}, inputs=batch["inputs"])
                if loss.dtype != torch.float32 or not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Stage 2 FM loss must be finite fp32, got {loss} ({loss.dtype})"
                    )
                flow_debug = (
                    getattr(accelerator.unwrap_model(model).pipe, "last_flow_match_debug", None)
                    or {}
                )
                if (
                    flow_debug.get("input_latents_shape")
                    != flow_debug.get("noise_pred_shape")
                    or flow_debug.get("input_latents_shape")
                    != flow_debug.get("training_target_shape")
                ):
                    raise RuntimeError(f"Stage 2 FM tensor alignment failed: {flow_debug}")
                if flow_debug.get("loss_fm_dtype") != "float32":
                    raise RuntimeError(f"Stage 2 FM loss dtype audit failed: {flow_debug}")
                rank_losses = gather_scalar(accelerator, float(loss.detach().item()))
                rank_timesteps = gather_scalar(
                    accelerator, flow_debug.get("timestep", -1.0)
                )

                accelerator.backward(loss)
                local_gradient_audit = gradient_audit(
                    accelerator.unwrap_model(model)
                )
                if not local_gradient_audit["gradients_finite"]:
                    raise FloatingPointError("Non-finite Stage 2 gradients")
                if local_gradient_audit["nonzero_grad_tensors"] == 0:
                    raise RuntimeError("No non-zero Stage 2 LoRA gradients")
                if local_gradient_audit["frozen_grad_tensors"]:
                    raise RuntimeError("A frozen Stage 2 parameter received a gradient")
                rank_grad_norms = gather_scalar(
                    accelerator, local_gradient_audit["grad_norm_before_clip"]
                )

                probe_before = probe_parameter.detach().float().clone()
                learning_rate_used = float(optimizer.param_groups[0]["lr"])
                optimizer.step()
                scheduler.step()
                optimizer_step += int(accelerator.sync_gradients)
                probe_after = probe_parameter.detach().float()
                probe_update = float((probe_after - probe_before).norm().item())
                rank_probe_updates = gather_scalar(accelerator, probe_update)
                rank_probe_norms = gather_scalar(
                    accelerator, float(probe_after.norm().item())
                )
                if max(rank_probe_norms) - min(rank_probe_norms) > 1e-5 * max(
                    1.0, max(rank_probe_norms)
                ):
                    raise RuntimeError(
                        f"Stage 2 LoRA parameters diverged across ranks: {rank_probe_norms}"
                    )
                optimizer.zero_grad(set_to_none=True)

                debug_metrics = {
                    **local_gradient_audit,
                    "optimizer_step": optimizer_step,
                    "learning_rate_used": learning_rate_used,
                    "learning_rate_next": float(optimizer.param_groups[0]["lr"]),
                    "probe_update_l2_norm": probe_update,
                    "sync_gradients": int(accelerator.sync_gradients),
                }
                if micro_step_in_epoch % args.debug_log_steps == 0:
                    record = {
                        "epoch": epoch_id,
                        "micro_step_in_epoch": micro_step_in_epoch,
                        "optimizer_step": optimizer_step,
                        "sample_type": cache_record["sample_type"],
                        "metadata_index": cache_record["metadata_index"],
                        "cache_path": str(cache_path),
                        "rank_sample_type_ids": rank_type_ids,
                        "rank_source_row_ids": rank_source_rows,
                        "rank_loss_fm": rank_losses,
                        "rank_loss_summary": scalar_range(rank_losses),
                        "rank_timesteps": rank_timesteps,
                        "rank_grad_norm_before_clip": rank_grad_norms,
                        "rank_grad_norm_summary": scalar_range(rank_grad_norms),
                        "rank_probe_update_l2_norm": rank_probe_updates,
                        "rank_probe_parameter_l2_norm": rank_probe_norms,
                        "probe_parameter": probe_name,
                        **flow_debug,
                        **debug_metrics,
                    }
                    accelerator.print(
                        "[SamtokDebug][stage2_step] "
                        + json.dumps(record, sort_keys=True)
                    )
                model_logger.on_step_end(
                    accelerator,
                    model,
                    args.save_steps,
                    loss=loss,
                    debug_metrics=debug_metrics,
                )

        expected_type_counts = Counter(
            {
                sample_type: count * args.dataset_repeat
                for sample_type, count in physical_type_counts.items()
            }
        )
        expected_source_counts = Counter(
            {index: args.dataset_repeat for index in physical_source_rows}
        )
        if epoch_type_counts != expected_type_counts:
            raise RuntimeError(
                f"Stage 2 epoch type ratio mismatch: {epoch_type_counts} != {expected_type_counts}"
            )
        if epoch_source_counts != expected_source_counts:
            raise RuntimeError("Stage 2 did not consume every cache row exactly repeat times")
        if accelerator.is_main_process:
            print(
                "[SamtokDebug][stage2_epoch_audit] "
                + json.dumps(
                    {
                        "epoch": epoch_id,
                        "sample_type_counts": dict(epoch_type_counts),
                        "unique_metadata_rows": len(epoch_source_counts),
                        "uses_per_metadata_row": sorted(set(epoch_source_counts.values())),
                        "optimizer_step": optimizer_step,
                    },
                    sort_keys=True,
                )
            )
        if args.save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)

    model_logger.on_training_end(accelerator, model, args.save_steps)
    accelerator.end_training()


def launch_training_task_ordered(
    accelerator,
    dataset,
    model,
    model_logger,
    args=None,
    **kwargs,
):
    """Stage-1 runner preserving the dataset's DDP-aware micro-step order."""

    if args is None:
        raise ValueError("The ordered SAMTok runner requires parsed training arguments")
    if args.enable_model_cpu_offload or args.enable_optimizer_cpu_offload:
        raise ValueError("Stage 1 does not support model/optimizer CPU offload")
    if accelerator.is_main_process:
        save_training_args(args)

    optimizer = get_optimizer_class(args.customized_optimizer)(
        model.trainable_modules(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    total_steps = (
        len(dataset)
        // (accelerator.num_processes * args.gradient_accumulation_steps)
        * args.num_epochs
    )
    if total_steps < 1:
        raise ValueError("Dataset is too small to form one distributed optimizer step")
    from transformers import get_cosine_schedule_with_warmup

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda batch: batch[0],
        num_workers=args.dataset_num_workers,
    )
    model.to(device=accelerator.device)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    initialize_deepspeed_gradient_checkpointing(accelerator)

    if args.debug_train_metrics:
        audit = trainable_parameter_audit(accelerator.unwrap_model(model))
        runtime_audit = {
            "world_size": accelerator.num_processes,
            "distributed_type": str(accelerator.distributed_type),
            "accelerate_mixed_precision": accelerator.mixed_precision,
            "pipeline_dtype": str(
                accelerator.unwrap_model(model).pipe.torch_dtype
            ).removeprefix("torch."),
            "dataset_schedule_rows_global": len(dataset),
            "micro_steps_per_rank": len(dataloader) * args.num_epochs,
            "optimizer_steps": total_steps,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "sample_type_ratio": args.sample_type_ratio,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "warmup_ratio": args.warmup_ratio,
            "warmup_steps": int(args.warmup_ratio * total_steps),
            "ntp_loss_weight": args.ntp_loss_weight,
            "fm_loss_weight": args.fm_loss_weight,
            "seed": args.seed,
            "rank_seed": args.seed + accelerator.process_index,
        }
        if len(dataset) != len(dataloader) * accelerator.num_processes:
            raise RuntimeError(
                "Prepared DDP dataloader did not partition the schedule exactly: "
                f"global={len(dataset)}, local={len(dataloader)}, "
                f"world={accelerator.num_processes}"
            )
        if accelerator.is_main_process:
            print(
                "[SamtokDebug][parameter_audit] "
                + json.dumps(audit, sort_keys=True)
            )
            print(
                "[SamtokDebug][runtime_audit] "
                + json.dumps(runtime_audit, sort_keys=True)
            )

    probe_name, probe_parameter = None, None
    if args.debug_train_metrics:
        for name, parameter in accelerator.unwrap_model(model).named_parameters():
            if parameter.requires_grad and ".lora_B." in name:
                probe_name, probe_parameter = name, parameter
                break

    micro_step = 0
    optimizer_step = 0

    for epoch_id in range(args.num_epochs):
        for data in tqdm(dataloader, disable=not accelerator.is_local_main_process):
            with accelerator.accumulate(model):
                micro_step += 1
                accumulation_slot = (
                    (micro_step - 1) % args.gradient_accumulation_steps
                ) + 1
                sample_type = data.get("sample_type", "edit")
                debug_metrics = {}
                debug_record_fields = {}
                if args.debug_train_metrics:
                    sample_type_id = {
                        "edit_mt": 0,
                        "edit_ntp": 1,
                        "edit": 2,
                        "edit_umt": 3,
                    }.get(sample_type, -1)
                    rank_type_ids = gather_scalar(
                        accelerator, sample_type_id, dtype=torch.int64
                    )
                    if sample_type_id < 0 or len(set(rank_type_ids)) != 1:
                        raise RuntimeError(
                            f"DDP sample types diverged at micro-step {micro_step}: "
                            f"{rank_type_ids}"
                        )
                    schedule_position = data.get("_samtok_schedule_position")
                    if schedule_position is None:
                        raise RuntimeError("Missing runtime schedule provenance")
                    rank_schedule_positions = gather_scalar(
                        accelerator, schedule_position, dtype=torch.int64
                    )
                    first_position = min(rank_schedule_positions)
                    expected_positions = list(
                        range(first_position, first_position + accelerator.num_processes)
                    )
                    if (
                        rank_schedule_positions != expected_positions
                        or first_position % accelerator.num_processes
                    ):
                        raise RuntimeError(
                            "Accelerate did not shard one homogeneous schedule group "
                            f"across ranks: {rank_schedule_positions}"
                        )
                    rank_source_row_ids = gather_scalar(
                        accelerator,
                        data.get("_samtok_source_row_id", -1),
                        dtype=torch.int64,
                    )
                    debug_metrics.update(
                        {
                            "ddp_sample_type_consistent": 1,
                            "ddp_schedule_group_consecutive": 1,
                            "world_size": accelerator.num_processes,
                            "fresh_accumulation_gradients": int(
                                accumulation_slot == 1
                            ),
                        }
                    )
                    debug_record_fields.update(
                        {
                            "rank_sample_type_ids": rank_type_ids,
                            "rank_schedule_positions": rank_schedule_positions,
                            "rank_source_row_ids": rank_source_row_ids,
                        }
                    )
                loss = model(data)
                components = {}
                if args.debug_train_metrics:
                    pipe = accelerator.unwrap_model(model).pipe
                    components = getattr(pipe, "last_loss_log", None) or {}
                    expected_components = {
                        "edit_mt": {"loss_ntp", "loss_fm"},
                        "edit_ntp": {"loss_ntp"},
                        "edit": {"loss_fm"},
                        "edit_umt": {"loss_fm"},
                    }[sample_type]
                    if set(components) != expected_components:
                        raise RuntimeError(
                            f"Loss dispatch mismatch for {sample_type}: {components}"
                        )
                    if loss.dtype != torch.float32:
                        raise RuntimeError(
                            f"Stage-1 loss must be fp32, got {loss.dtype}"
                        )
                    expected_total = (
                        args.ntp_loss_weight * components.get("loss_ntp", 0.0)
                        + args.fm_loss_weight * components.get("loss_fm", 0.0)
                    )
                    local_loss = float(loss.detach().item())
                    identity_error = abs(local_loss - expected_total)
                    if identity_error > 1e-5 * max(1.0, abs(expected_total)):
                        raise RuntimeError(
                            f"Weighted loss identity failed for {sample_type}: "
                            f"loss={local_loss}, expected={expected_total}"
                        )
                    rank_loss_total = gather_scalar(accelerator, local_loss)
                    if not all(math.isfinite(value) for value in rank_loss_total):
                        raise FloatingPointError(
                            f"Non-finite rank loss at micro-step {micro_step}: "
                            f"{rank_loss_total}"
                        )
                    rank_component_losses = {
                        name: gather_scalar(accelerator, components[name])
                        for name in sorted(expected_components)
                    }
                    rank_identity_errors = gather_scalar(
                        accelerator, identity_error
                    )
                    training_debug = getattr(pipe, "last_training_debug", None) or {}
                    flow_debug = getattr(pipe, "last_flow_match_debug", None) or {}
                    if "loss_fm" in expected_components:
                        if (
                            flow_debug.get("input_latents_shape")
                            != flow_debug.get("noise_pred_shape")
                            or flow_debug.get("input_latents_shape")
                            != flow_debug.get("training_target_shape")
                        ):
                            raise RuntimeError(
                                f"Stage-1 FM tensor alignment failed: {flow_debug}"
                            )
                        if flow_debug.get("loss_fm_dtype") != "float32":
                            raise RuntimeError(
                                f"Stage-1 FM loss dtype audit failed: {flow_debug}"
                            )
                    if sample_type in {"edit_mt", "edit_ntp"}:
                        if not training_debug.get("ntp_shift_alignment_ok"):
                            raise RuntimeError(
                                f"Missing valid shifted NTP alignment for {sample_type}"
                            )
                    elif training_debug.get("cot_tokens") != 0:
                        raise RuntimeError(
                            f"{sample_type} sample unexpectedly has NTP labels"
                        )
                    if sample_type == "edit_umt":
                        if (
                            training_debug.get("user_mask_span_count") != 1
                            or not training_debug.get("user_mask_spans_atomic")
                            or not training_debug.get("user_mask_spans_in_template")
                        ):
                            raise RuntimeError(
                                "edit_umt mask span was not preserved as four atomic "
                                f"user-prompt tokens: {training_debug}"
                            )
                    elif training_debug.get("user_mask_span_count") != 0:
                        raise RuntimeError(
                            f"{sample_type} unexpectedly contains a user-prompt mask span"
                        )
                    debug_metrics.update(
                        {
                            "loss_identity_error": identity_error,
                            "rank_loss_total_min": min(rank_loss_total),
                            "rank_loss_total_max": max(rank_loss_total),
                        }
                    )
                    debug_record_fields.update(
                        {
                            "rank_loss_total": rank_loss_total,
                            "rank_component_losses": rank_component_losses,
                            "rank_loss_identity_error": rank_identity_errors,
                            "rank_loss_total_summary": scalar_range(rank_loss_total),
                        }
                    )
                accelerator.backward(loss)
                if args.debug_train_metrics:
                    local_gradient_audit = gradient_audit(
                        accelerator.unwrap_model(model)
                    )
                    debug_metrics.update(local_gradient_audit)
                    if not debug_metrics["gradients_finite"]:
                        raise FloatingPointError(
                            f"Non-finite Stage-1 gradients at micro-step {micro_step}"
                        )
                    if debug_metrics["nonzero_grad_tensors"] == 0:
                        raise RuntimeError(
                            f"No non-zero LoRA gradients at micro-step {micro_step}"
                        )
                    if debug_metrics["frozen_grad_tensors"]:
                        raise RuntimeError(
                            "A frozen parameter unexpectedly received a .grad tensor"
                        )
                    rank_grad_norms = gather_scalar(
                        accelerator, debug_metrics["grad_norm_before_clip"]
                    )
                    rank_nonzero_grad_tensors = gather_scalar(
                        accelerator,
                        debug_metrics["nonzero_grad_tensors"],
                        dtype=torch.int64,
                    )
                    debug_record_fields.update(
                        {
                            "rank_grad_norm_before_clip": rank_grad_norms,
                            "rank_nonzero_grad_tensors": rank_nonzero_grad_tensors,
                            "rank_grad_norm_summary": scalar_range(rank_grad_norms),
                        }
                    )

                clip_norm = None
                if accelerator.sync_gradients:
                    clip_norm = accelerator.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    )
                probe_before = (
                    probe_parameter.detach().float().clone()
                    if accelerator.sync_gradients and probe_parameter is not None
                    else None
                )
                optimizer.step()
                if accelerator.sync_gradients:
                    scheduler.step()
                    optimizer_step += 1
                probe_after = (
                    probe_parameter.detach().float()
                    if accelerator.sync_gradients and probe_parameter is not None
                    else None
                )
                optimizer.zero_grad(set_to_none=True)

                if args.debug_train_metrics:
                    pipe = accelerator.unwrap_model(model).pipe
                    if probe_after is not None:
                        probe_update = float(
                            (probe_after - probe_before).norm().item()
                        )
                        rank_probe_updates = gather_scalar(
                            accelerator, probe_update
                        )
                        rank_probe_norms = gather_scalar(
                            accelerator, float(probe_after.norm().item())
                        )
                        if max(rank_probe_norms) - min(rank_probe_norms) > 1e-5 * max(
                            1.0, max(rank_probe_norms)
                        ):
                            raise RuntimeError(
                                "LoRA parameters diverged across DDP ranks after an "
                                f"optimizer step: {rank_probe_norms}"
                            )
                        debug_record_fields.update(
                            {
                                "rank_probe_update_l2_norm": rank_probe_updates,
                                "rank_probe_parameter_l2_norm": rank_probe_norms,
                            }
                        )
                    else:
                        probe_update = None
                    debug_metrics.update(
                        {
                            "sync_gradients": int(accelerator.sync_gradients),
                            "optimizer_step": optimizer_step,
                            "learning_rate": float(optimizer.param_groups[0]["lr"]),
                            "clip_return_norm": (
                                float(clip_norm.detach().float().item())
                                if isinstance(clip_norm, torch.Tensor)
                                else float(clip_norm) if clip_norm is not None else None
                            ),
                            "probe_update_l2_norm": (
                                probe_update
                            ),
                        }
                    )
                    if micro_step % args.debug_log_steps == 0:
                        record = {
                            "epoch": epoch_id,
                            "micro_step": micro_step,
                            "accumulation_slot": accumulation_slot,
                            "sample_type": sample_type,
                            "loss_total": float(loss.detach().float().item()),
                            **components,
                            **(getattr(pipe, "last_training_debug", None) or {}),
                            **(
                                flow_debug
                                if "loss_fm" in expected_components
                                else {}
                            ),
                            **(getattr(pipe, "last_loss_debug", None) or {}),
                            **debug_metrics,
                            **debug_record_fields,
                            "probe_parameter": probe_name,
                        }
                        accelerator.print(
                            "[SamtokDebug][micro_step] "
                            + json.dumps(record, sort_keys=True)
                        )
                model_logger.on_step_end(
                    accelerator,
                    model,
                    args.save_steps,
                    loss=loss,
                    debug_metrics=debug_metrics,
                )
        if args.save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, args.save_steps)
    accelerator.end_training()


class SamtokModelLogger(ModelLogger):
    """Standard model logger plus separate NTP and flow-matching curves."""

    def __init__(self, *args, run_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_config = run_config

    def init_loggers(self):
        super().init_loggers()
        if self.enable_wandb_log and self.run_config is not None:
            import wandb

            wandb.config.update(self.run_config, allow_val_change=True)

    def on_step_end(self, accelerator, model, save_steps=None, **kwargs):
        super().on_step_end(accelerator, model, save_steps, **kwargs)
        if not accelerator.is_main_process or not self.loggers_initialized:
            return
        components = (
            getattr(accelerator.unwrap_model(model).pipe, "last_loss_log", None) or {}
        )
        for name, value in components.items():
            for logger in self.loggers:
                logger.log(name, float(value), self.num_steps)
        for name, value in (kwargs.get("debug_metrics") or {}).items():
            if value is None:
                continue
            for logger in self.loggers:
                logger.log("debug/" + name, float(value), self.num_steps)


class QwenImageSamtokTrainingModule(DiffusionTrainingModule):
    @staticmethod
    def _model_path_lookup_key(path):
        if isinstance(path, str):
            return path
        return json.dumps(path, separators=(",", ":"))

    def parse_model_configs(
        self,
        model_paths,
        model_id_with_origin_paths,
        fp8_models=None,
        offload_models=None,
        quant_options=None,
        device="cpu",
    ):
        """Keep DiffSynth's list-of-shards model path support hash-safe."""

        if model_paths is None:
            return super().parse_model_configs(
                model_paths,
                model_id_with_origin_paths,
                fp8_models=fp8_models,
                offload_models=offload_models,
                quant_options=quant_options,
                device=device,
            )

        decoded_paths = json.loads(model_paths)
        if all(isinstance(path, str) for path in decoded_paths):
            return super().parse_model_configs(
                model_paths,
                model_id_with_origin_paths,
                fp8_models=fp8_models,
                offload_models=offload_models,
                quant_options=quant_options,
                device=device,
            )

        fp8_models_arg = fp8_models
        offload_models_arg = offload_models
        fp8_model_names = [] if fp8_models is None else fp8_models.split(",")
        offload_model_names = (
            [] if offload_models is None else offload_models.split(",")
        )
        quant_map = self.parse_quant_options(quant_options)
        model_configs = []
        for path in decoded_paths:
            path_key = self._model_path_lookup_key(path)
            vram_config = self.parse_vram_config(
                fp8=path_key in fp8_model_names,
                offload=path_key in offload_model_names,
                device=device,
            )
            model_configs.append(
                ModelConfig(
                    path=path,
                    quantize=quant_map.get(path_key),
                    **vram_config,
                )
            )
        model_configs.extend(
            super().parse_model_configs(
                None,
                model_id_with_origin_paths,
                fp8_models=fp8_models_arg,
                offload_models=offload_models_arg,
                quant_options=quant_options,
                device=device,
            )
        )
        return model_configs

    def __init__(
        self,
        model_paths=None,
        model_id_with_origin_paths=None,
        tokenizer_path=None,
        processor_path=None,
        trainable_models=None,
        lora_base_model=None,
        lora_target_modules="",
        lora_rank=32,
        lora_checkpoint=None,
        preset_lora_path=None,
        preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        quant_options=None,
        resume_from_checkpoint=None,
        remove_prefix_in_ckpt=None,
        device="cpu",
        task="sft",
        zero_cond_t=True,
        ntp_loss_weight=1.0,
        fm_loss_weight=1.0,
        lora_dropout=0.05,
    ):
        super().__init__()
        self._te_lora_recipe = lora_base_model == "text_encoder"
        self.lora_dropout = lora_dropout
        model_configs = self.parse_model_configs(
            model_paths,
            model_id_with_origin_paths,
            fp8_models=fp8_models,
            offload_models=offload_models,
            quant_options=quant_options,
            device=device,
        )
        tokenizer_config = ModelConfig(tokenizer_path) if tokenizer_path else None
        processor_config = ModelConfig(processor_path) if processor_path else None
        self.pipe = QwenImageSamtokPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            processor_config=processor_config,
        )
        self.pipe = self.split_pipeline_units(
            task,
            self.pipe,
            trainable_models,
            lora_base_model,
            remove_unnecessary_params=True,
        )
        self.resume_from_checkpoint(resume_from_checkpoint, remove_prefix_in_ckpt)
        self.switch_pipe_to_training_mode(
            self.pipe,
            trainable_models,
            lora_base_model,
            lora_target_modules,
            lora_rank,
            lora_checkpoint,
            preset_lora_path,
            preset_lora_model,
            task=task,
        )

        if (
            self._te_lora_recipe
            and self.pipe.text_encoder is not None
            and use_gradient_checkpointing
        ):
            self.pipe.text_encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            # freeze_except() puts the entire pipe in eval mode; Transformers
            # activates its internal checkpointing only while the module trains.
            self.pipe.text_encoder.train()

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs else []
        self.task = task
        self.zero_cond_t = zero_cond_t
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *inputs: inputs,
            "sft": lambda pipe, shared, positive, negative: SamtokEditingLoss(
                pipe,
                ntp_weight=ntp_loss_weight,
                fm_weight=fm_loss_weight,
                **shared,
                **positive,
            ),
            "sft:train": lambda pipe, shared, positive, negative: FlowMatchSFTLoss(
                pipe, **shared, **positive
            ),
        }
        if task not in self.task_to_loss:
            raise ValueError(f"Unsupported SAMTok training task: {task}")

    def add_lora_to_model(
        self,
        model,
        target_modules,
        lora_rank,
        lora_alpha=None,
        upcast_dtype=None,
    ):
        if not self._te_lora_recipe:
            return super().add_lora_to_model(
                model,
                target_modules,
                lora_rank,
                lora_alpha=lora_alpha,
                upcast_dtype=upcast_dtype,
            )
        from peft import LoraConfig, inject_adapter_in_model

        if isinstance(target_modules, list) and len(target_modules) == 1:
            target_modules = target_modules[0]
        model = inject_adapter_in_model(
            LoraConfig(
                r=lora_rank,
                lora_alpha=lora_rank,
                lora_dropout=self.lora_dropout,
                target_modules=target_modules,
            ),
            model,
        )
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        return model

    def get_pipeline_inputs(self, data):
        sample_type = data.get("sample_type", "edit")
        edit_image = data["edit_image"]
        if not isinstance(edit_image, list) or not edit_image:
            raise ValueError("edit_image operator must produce a non-empty PIL image list")
        inputs_posi = {"prompt": data["prompt"], "mt_cot": data.get("mt_cot")}
        inputs_nega = {"negative_prompt": ""}
        inputs_shared = {
            "cfg_scale": 1,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "edit_image_auto_resize": True,
            "zero_cond_t": self.zero_cond_t,
            "edit_image": edit_image,
            "sample_type": sample_type,
            "samtok_online_cot": False,
            "samtok_need_ntp": sample_type in {"edit_ntp", "edit_mt"}
            and not self.task.endswith(":data_process"),
        }
        if sample_type == "edit_ntp":
            width, height = edit_image[0].size
            inputs_shared.update(
                input_image=None,
                height=max(16, height // 16 * 16),
                width=max(16, width // 16 * 16),
            )
        else:
            image = data["image"]
            inputs_shared.update(
                input_image=image,
                height=image.size[1],
                width=image.size[0],
            )
        return self.parse_extra_inputs(data, self.extra_inputs, inputs_shared), inputs_posi, inputs_nega

    def forward(self, data, inputs=None):
        if inputs is None:
            inputs = self.get_pipeline_inputs(data)
        else:
            inputs[0]["use_gradient_checkpointing"] = self.use_gradient_checkpointing
            inputs[0][
                "use_gradient_checkpointing_offload"
            ] = self.use_gradient_checkpointing_offload
        inputs = self.transfer_data_to_device(
            inputs, self.pipe.device, self.pipe.torch_dtype
        )
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        if self.task == "sft":
            shared, positive, _ = inputs
            cot_labels = positive.get("samtok_cot_labels")
            cot_hidden = positive.get("samtok_cot_hidden")
            prompt_mask = positive.get("prompt_emb_mask")
            prompt_emb = positive.get("prompt_emb")
            input_latents = shared.get("input_latents")
            edit_latents = shared.get("edit_latents")
            if edit_latents is not None and not isinstance(edit_latents, list):
                edit_latents = [edit_latents]
            alignment = getattr(self.pipe, "last_ntp_alignment", None) or {}
            if cot_labels is not None:
                if cot_hidden is None or cot_hidden.shape[:2] != cot_labels.shape:
                    raise RuntimeError(
                        "NTP hidden positions do not align one-to-one with labels"
                    )
                if not alignment.get("ntp_shift_alignment_ok"):
                    raise RuntimeError("NTP supervision boundary audit is missing")
                if int(cot_labels[0, -1].item()) != self.pipe.im_end_id:
                    raise RuntimeError("NTP supervision does not end on <|im_end|>")
            elif cot_hidden is not None:
                raise RuntimeError("NTP hidden exists without NTP labels")
            expected_fm = data.get("sample_type", "edit") in {
                "edit_mt",
                "edit",
                "edit_umt",
            }
            if (input_latents is not None) != expected_fm:
                raise RuntimeError(
                    "FM target dispatch mismatch: target latents must come from the "
                    "metadata image field only for edit_mt/edit/edit_umt"
                )
            self.pipe.last_training_debug = {
                "cot_tokens": int(cot_labels.numel()) if cot_labels is not None else 0,
                "cot_hidden_shape": (
                    list(cot_hidden.shape) if cot_hidden is not None else None
                ),
                "cot_label_shape": (
                    list(cot_labels.shape) if cot_labels is not None else None
                ),
                "cot_hidden_dtype": (
                    str(cot_hidden.dtype).removeprefix("torch.")
                    if cot_hidden is not None
                    else None
                ),
                "cot_label_dtype": (
                    str(cot_labels.dtype).removeprefix("torch.")
                    if cot_labels is not None
                    else None
                ),
                "prompt_tokens": (
                    int(prompt_mask.sum().item()) if prompt_mask is not None else 0
                ),
                "prompt_emb_dtype": (
                    str(prompt_emb.dtype).removeprefix("torch.")
                    if prompt_emb is not None
                    else None
                ),
                "has_ntp_loss": cot_labels is not None,
                "has_fm_loss": input_latents is not None,
                "fm_target_is_metadata_image": input_latents is not None,
                "conditioning_is_metadata_edit_image": bool(edit_latents),
                "target_latent_dtype": (
                    str(input_latents.dtype).removeprefix("torch.")
                    if input_latents is not None
                    else None
                ),
                "target_latent_shape": (
                    list(input_latents.shape) if input_latents is not None else None
                ),
                "edit_latent_shapes": (
                    [list(latent.shape) for latent in edit_latents]
                    if edit_latents is not None
                    else []
                ),
                **(getattr(self.pipe, "last_user_mask_audit", None) or {}),
                **alignment,
            }
        return self.task_to_loss[self.task](self.pipe, *inputs)


def samtok_parser():
    parser = argparse.ArgumentParser(
        description="SAMTokEdit Stage-1/Stage-2 training"
    )
    parser = add_general_config(parser)
    # SAMTok experiments use WandB by default. The launcher validates the
    # account environment, while direct Python invocation is also checked
    # before any model is loaded. Offline runs must opt out explicitly.
    parser.set_defaults(
        enable_wandb_log=True,
        wandb_project=os.environ.get("WANDB_PROJECT"),
    )
    parser.add_argument(
        "--disable_wandb_log",
        dest="enable_wandb_log",
        action="store_false",
        help="Explicitly disable the default WandB logger for an offline run.",
    )
    parser = add_image_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--processor_path", type=str, default=None)
    parser.add_argument("--zero_cond_t", default=False, action="store_true")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true")
    parser.add_argument("--ntp_loss_weight", type=float, default=0.05)
    parser.add_argument("--fm_loss_weight", type=float, default=1.0)
    parser.add_argument(
        "--sample_type_ratio",
        type=str,
        default="edit_mt:4,edit_ntp:2,edit:1,edit_umt:1",
    )
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--debug_train_metrics", action="store_true")
    parser.add_argument("--debug_log_steps", type=int, default=1)
    return parser


def validate_wandb_credentials(args):
    """Fail before model loading when the default WandB logger lacks account data."""

    # Data processing only writes latent/cache shards and never initializes a
    # training logger; it does not require an experiment account.
    if args.task.endswith(":data_process") or not args.enable_wandb_log:
        return
    missing = [
        name
        for name in ("WANDB_API_KEY", "WANDB_ENTITY")
        if not os.environ.get(name)
    ]
    if not args.wandb_project:
        missing.append("WANDB_PROJECT or --wandb_project")
    if missing:
        raise RuntimeError(
            "WandB logging is enabled by default for SAMTokEdit. Set the account "
            "environment before training; missing: "
            + ", ".join(missing)
            + ". WANDB_API_KEY is never persisted to training_args.json. "
            "Use --disable_wandb_log only for an explicit offline run."
        )


def main():
    args = samtok_parser().parse_args()
    if args.debug_log_steps < 1:
        raise ValueError("debug_log_steps must be positive")
    validate_wandb_credentials(args)
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[
            accelerate.DistributedDataParallelKwargs(
                find_unused_parameters=args.find_unused_parameters
            )
        ],
    )
    # Keep the metadata schedule identical on all ranks while giving each rank
    # independent timestep/noise RNG streams.
    accelerate.utils.set_seed(args.seed, device_specific=True)

    edit_image_operator = RouteByType(
        operator_map=[
            (
                str,
                ToAbsolutePath(args.dataset_base_path)
                >> LoadImage()
                >> ImageCropAndResize(
                    args.height, args.width, args.max_pixels, 16, 16
                )
                >> ToList(),
            ),
            (
                list,
                SequencialProcess(
                    ToAbsolutePath(args.dataset_base_path)
                    >> LoadImage()
                    >> ImageCropAndResize(
                        args.height, args.width, args.max_pixels, 16, 16
                    )
                ),
            ),
        ]
    )
    dataset = SamtokEditingDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=[key.strip() for key in args.data_file_keys.split(",") if key.strip()],
        main_data_operator=SamtokEditingDataset.default_image_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
        ),
        special_operator_map={"edit_image": edit_image_operator},
        type_ratio=args.sample_type_ratio,
        num_processes=accelerator.num_processes,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        seed=args.seed,
    )
    model = QwenImageSamtokTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        processor_path=args.processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        quant_options=args.quant_options,
        resume_from_checkpoint=args.resume_from_checkpoint,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        task=args.task,
        device=(
            "cpu"
            if args.initialize_model_on_cpu or args.enable_model_cpu_offload
            else accelerator.device
        ),
        zero_cond_t=args.zero_cond_t,
        ntp_loss_weight=args.ntp_loss_weight,
        fm_loss_weight=args.fm_loss_weight,
        lora_dropout=args.lora_dropout,
    )
    model_logger = SamtokModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        enable_tensorboard_log=args.enable_tensorboard_log,
        enable_swanlab_log=args.enable_swanlab_log,
        swanlab_project=args.swanlab_project,
        enable_wandb_log=args.enable_wandb_log,
        wandb_project=args.wandb_project,
        enable_csv_log=args.enable_csv_log,
        run_config=vars(args),
    )
    launcher = {
        "sft:data_process": launch_data_process_task_samtok,
        "sft": launch_training_task_ordered,
        "sft:train": (
            launch_training_task_stage2_debug
            if args.debug_train_metrics
            else launch_training_task
        ),
    }[args.task]
    launcher(accelerator, dataset, model, model_logger, args=args)


if __name__ == "__main__":
    main()
