#!/usr/bin/env python3
"""Unified Stage-1/Stage-2 trainer for SAMTok-guided Qwen-Image-Edit."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

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
    for _, parameter in trainable:
        dtype = str(parameter.dtype).removeprefix("torch.")
        dtypes[dtype] = dtypes.get(dtype, 0) + parameter.numel()
    report = {
        "trainable_tensors": len(trainable),
        "trainable_parameters": sum(parameter.numel() for _, parameter in trainable),
        "trainable_parameter_dtypes": dtypes,
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

    if args.debug_train_metrics and accelerator.is_main_process:
        audit = trainable_parameter_audit(accelerator.unwrap_model(model))
        print("[SamtokDebug][parameter_audit] " + json.dumps(audit, sort_keys=True))

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
                sample_type = data.get("sample_type", "edit")
                loss = model(data)
                accelerator.backward(loss)
                debug_metrics = {}
                if args.debug_train_metrics:
                    debug_metrics.update(gradient_audit(accelerator.unwrap_model(model)))
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
                    components = getattr(pipe, "last_loss_log", None) or {}
                    expected_components = {
                        "edit_mt": {"loss_ntp", "loss_fm"},
                        "edit_ntp": {"loss_ntp"},
                        "edit": {"loss_fm"},
                    }[sample_type]
                    if set(components) != expected_components:
                        raise RuntimeError(
                            f"Loss dispatch mismatch for {sample_type}: {components}"
                        )
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
                                float((probe_after - probe_before).norm().item())
                                if probe_before is not None and probe_after is not None
                                else None
                            ),
                        }
                    )
                    if micro_step % args.debug_log_steps == 0:
                        record = {
                            "epoch": epoch_id,
                            "micro_step": micro_step,
                            "accumulation_slot": (
                                (micro_step - 1) % args.gradient_accumulation_steps
                            )
                            + 1,
                            "sample_type": sample_type,
                            "loss_total": float(loss.detach().float().item()),
                            **components,
                            **(getattr(pipe, "last_training_debug", None) or {}),
                            **debug_metrics,
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
            prompt_mask = positive.get("prompt_emb_mask")
            input_latents = shared.get("input_latents")
            edit_latents = shared.get("edit_latents")
            if edit_latents is not None and not isinstance(edit_latents, list):
                edit_latents = [edit_latents]
            self.pipe.last_training_debug = {
                "cot_tokens": int(cot_labels.numel()) if cot_labels is not None else 0,
                "prompt_tokens": (
                    int(prompt_mask.sum().item()) if prompt_mask is not None else 0
                ),
                "has_ntp_loss": cot_labels is not None,
                "has_fm_loss": input_latents is not None,
                "target_latent_shape": (
                    list(input_latents.shape) if input_latents is not None else None
                ),
                "edit_latent_shapes": (
                    [list(latent.shape) for latent in edit_latents]
                    if edit_latents is not None
                    else []
                ),
            }
        return self.task_to_loss[self.task](self.pipe, *inputs)


def samtok_parser():
    parser = argparse.ArgumentParser(
        description="SAMTokEdit Stage-1/Stage-2 training"
    )
    parser = add_general_config(parser)
    parser = add_image_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--processor_path", type=str, default=None)
    parser.add_argument("--zero_cond_t", default=False, action="store_true")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true")
    parser.add_argument("--ntp_loss_weight", type=float, default=1.0)
    parser.add_argument("--fm_loss_weight", type=float, default=1.0)
    parser.add_argument(
        "--sample_type_ratio", type=str, default="edit_mt:2,edit_ntp:1,edit:1"
    )
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--debug_train_metrics", action="store_true")
    parser.add_argument("--debug_log_steps", type=int, default=1)
    return parser


def main():
    args = samtok_parser().parse_args()
    if args.debug_log_steps < 1:
        raise ValueError("debug_log_steps must be positive")
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[
            accelerate.DistributedDataParallelKwargs(
                find_unused_parameters=args.find_unused_parameters
            )
        ],
    )

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
        "sft:data_process": launch_data_process_task,
        "sft": launch_training_task_ordered,
        "sft:train": launch_training_task,
    }[args.task]
    launcher(accelerator, dataset, model, model_logger, args=args)


if __name__ == "__main__":
    main()
