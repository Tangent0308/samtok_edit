#!/usr/bin/env python3
"""Unified Stage-1/Stage-2 trainer for SAMTok-guided Qwen-Image-Edit."""

from __future__ import annotations

import argparse
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

    for epoch_id in range(args.num_epochs):
        for data in tqdm(dataloader, disable=not accelerator.is_local_main_process):
            with accelerator.accumulate(model):
                loss = model(data)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                if accelerator.sync_gradients:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                model_logger.on_step_end(
                    accelerator, model, args.save_steps, loss=loss
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


class QwenImageSamtokTrainingModule(DiffusionTrainingModule):
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
    return parser


def main():
    args = samtok_parser().parse_args()
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
