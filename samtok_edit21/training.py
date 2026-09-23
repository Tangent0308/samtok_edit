"""Two-stage LoRA training, with official Qwen-Image-2.1 flow matching."""

from __future__ import annotations

import contextlib
import json
import math
import time
from pathlib import Path

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from peft import LoraConfig, inject_adapter_in_model
from safetensors.torch import load_file, save_file

from .data import (
    file_hash,
    load_images,
    make_schedule,
    read_rows,
    row_hash,
    row_kind,
    write_json,
)
from .model import encode_edit, load_pipeline, ntp_loss, resize_sources

TE_TARGETS = r"model\.model\.language_model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))"


def add_adapter(model, stage, rank, dropout=0.0):
    if stage == "stage1":
        targets = TE_TARGETS
    else:
        # Mirrors upstream empty target_modules: every nn.Linear in 2.1 DiT.
        # 2511's add_q_proj/txt_mlp/txt_mod branches do not exist here.
        targets = [
            name for name, m in model.named_modules() if isinstance(m, torch.nn.Linear)
        ]
    config = LoraConfig(
        r=rank, lora_alpha=rank, lora_dropout=dropout, target_modules=targets
    )
    inject_adapter_in_model(config, model)
    for name, p in model.named_parameters():
        if p.requires_grad:
            if "lora_" not in name:
                raise RuntimeError(f"Unexpected trainable parameter {name}")
            p.data = p.data.float()  # fp32 optimizer states/updates in both stages
    if not any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("No LoRA parameters")
    return model


def load_adapter(model, directory, *, trainable=False):
    directory = Path(directory)
    config = json.loads((directory / "adapter.json").read_text())
    add_adapter(model, config["stage"], config["rank"], config["dropout"])
    state = load_file(str(directory / "adapter.safetensors"))
    expected = {k for k, p in model.named_parameters() if p.requires_grad}
    if set(state) != expected:
        raise ValueError("Adapter schema does not match model and LoRA recipe")
    model.load_state_dict(state, strict=False)
    if not trainable:
        model.requires_grad_(False)
    return config


def save_adapter(model, directory, config):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state = {
        k: p.detach().float().cpu().contiguous()
        for k, p in model.named_parameters()
        if p.requires_grad
    }
    if not state or not all(torch.isfinite(x).all() for x in state.values()):
        raise RuntimeError("Invalid adapter parameters")
    tmp = directory / "adapter.safetensors.tmp"
    save_file(state, str(tmp))
    tmp.replace(directory / "adapter.safetensors")
    write_json(directory / "adapter.json", config)


def adapter_identity(path):
    return (
        None
        if not path
        else {
            "path": str(Path(path).absolute()),
            "sha256": file_hash(Path(path) / "adapter.safetensors"),
        }
    )


def verify_cache(directory, manifest):
    from .protocol import validate_row

    directory = Path(directory)
    if manifest.get("format") != "samtok21-cache-v1":
        raise ValueError("Incompatible cache format")
    for row in manifest["rows"]:
        original = {k: v for k, v in row.items() if not k.startswith("_cache")}
        validate_row(original)
        name = row["_cache_file"]
        if Path(name).name != name:
            raise ValueError("Cache shard must be a local filename")
        side = json.loads(directory.joinpath(name).with_suffix(".json").read_text())
        if (
            side["row_hash"] != row_hash(original)
            or side["identity"] != manifest["identity"]
        ):
            raise ValueError("Mixed or stale cache metadata")
        if side["sha256"] != file_hash(directory / name):
            raise ValueError("Cache checksum mismatch")


def validate_conditioning(inputs):
    target, embeds, mask = (
        inputs["input_latents"],
        inputs["prompt_embeds"],
        inputs["edit_image_pad_mask"],
    )
    if target.ndim != 4 or target.shape[:2] != (1, 64):
        raise ValueError("Expected target latent [1,64,H/16,W/16]")
    if embeds.ndim != 3 or embeds.shape[0] != 1 or embeds.shape[-1] != 4096:
        raise ValueError("Expected 4096-dim text features")
    if mask.dtype != torch.bool or mask.shape != embeds.shape[:2]:
        raise ValueError("Invalid Qwen3 image-pad mask")
    source_tokens = sum(x.shape[2] * x.shape[3] for x in inputs["edit_latents"])
    if int(mask.sum()) * 4 != source_tokens:
        raise ValueError(
            "Qwen3 visual grid and VAE grid disagree; do not independently resize TE images"
        )

    def finite(x):
        if (
            isinstance(x, torch.Tensor)
            and x.is_floating_point()
            and not torch.isfinite(x).all()
        ):
            raise ValueError("Nonfinite conditioning")
        if isinstance(x, dict):
            for v in x.values():
                finite(v)
        if isinstance(x, (list, tuple)):
            for v in x:
                finite(v)

    finite(inputs)


def prepare_fm(pipe, row, base_path, max_pixels, *, te_grad=False):
    images, target, height, width = load_images(row, base_path, max_pixels)
    if target is None:
        raise ValueError("FM requires a target image")
    images = resize_sources(pipe, images, height, width)
    with torch.no_grad():
        target_latent = pipe.vae.encode(pipe.preprocess_image(target))
        source_latents = [pipe.vae.encode(pipe.preprocess_image(im)) for im in images]
    with torch.enable_grad() if te_grad else torch.no_grad():
        cond = encode_edit(pipe, row["prompt"], images)
    inputs = {"input_latents": target_latent, "edit_latents": source_latents, **cond}
    validate_conditioning(inputs)
    return inputs


def flow_loss(pipe, inputs, *, timestep_index=None, noise=None, checkpointing=True):
    # Exactly the official FlowMatchSFTLoss sampling, target and weighting;
    # expose timestep/weight for audit and repeatable gradient checks.
    i = (
        torch.randint(0, len(pipe.scheduler.timesteps), (1,))
        if timestep_index is None
        else torch.tensor([timestep_index])
    )
    t = pipe.scheduler.timesteps[i].to(device=pipe.device, dtype=pipe.torch_dtype)
    x = inputs["input_latents"]
    noise = torch.randn_like(x) if noise is None else noise
    noisy = pipe.scheduler.add_noise(x, noise, t)
    target = pipe.scheduler.training_target(x, noise, t)
    pred = pipe.model_fn(
        dit=pipe.dit,
        latents=noisy,
        timestep=t,
        **inputs,
        kv_cache=None,
        use_gradient_checkpointing=checkpointing,
    )
    if pred.shape != target.shape:
        raise RuntimeError("FM prediction/target shape mismatch")
    mse = torch.nn.functional.mse_loss(pred.float(), target.float())
    weight = pipe.scheduler.training_weight(t).to(pipe.device)
    loss = mse * weight
    return loss, {
        "loss_fm": loss.detach().item(),
        "fm_mse": mse.detach().item(),
        "timestep": t.item(),
        "training_weight": weight.item(),
        "target_shape": list(target.shape),
    }


class TrainingModel(torch.nn.Module):
    def __init__(self, pipe, args):
        super().__init__()
        self.pipe, self.args, self.last_metrics = pipe, args, {}

    def forward(self, row):
        if self.args.stage == "stage2":
            inputs = torch.load(
                row["_cache_path"], map_location=self.pipe.device, weights_only=True
            )
            validate_conditioning(inputs)
            loss, metrics = flow_loss(self.pipe, inputs)
        elif row["sample_type"] == "edit_ntp":
            images, _, h, w = load_images(
                row, self.args.base_path, self.args.max_pixels
            )
            images = resize_sources(self.pipe, images, h, w)
            raw, metrics = ntp_loss(self.pipe, row["prompt"], images, row["mt_cot"])
            loss = raw * self.args.ntp_weight
        else:
            inputs = prepare_fm(
                self.pipe, row, self.args.base_path, self.args.max_pixels, te_grad=True
            )
            if not inputs["prompt_embeds"].requires_grad:
                raise RuntimeError("FM lost its gradient connection to TE")
            raw, metrics = flow_loss(self.pipe, inputs)
            loss = raw * self.args.fm_weight
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite loss")
        self.last_metrics = {
            **metrics,
            "loss": loss.detach().item(),
            "sample_type": row["sample_type"],
            "kind": row_kind(row),
        }
        return loss


def train(args):
    accelerator = Accelerator(
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)]
    )
    if Path(args.output, "run.json").exists():
        raise ValueError(
            "Use a fresh training output directory; --init-adapter warm starts weights only"
        )
    torch.manual_seed(args.seed)
    if args.stage == "stage2":
        manifest = json.loads(Path(args.cache, "manifest.json").read_text())
        rows = manifest["rows"]
        if accelerator.is_main_process:
            verify_cache(args.cache, manifest)
        accelerator.wait_for_everyone()
        if (
            manifest["identity"]["qwen"] != args.qwen
            or manifest["identity"]["samtok"] != args.samtok
        ):
            raise ValueError("Cache base models disagree with training configuration")
        for row in rows:
            row["_cache_path"] = str(Path(args.cache) / row["_cache_file"])
        pipe = load_pipeline(
            args.qwen, args.samtok, device=accelerator.device, components=("dit",)
        )
        trainable = pipe.dit
    else:
        rows = read_rows(args.metadata)
        pipe = load_pipeline(args.qwen, args.samtok, device=accelerator.device)
        trainable = pipe.text_encoder
    if args.resume_adapter:
        loaded = load_adapter(trainable, args.resume_adapter, trainable=True)
        if loaded["stage"] != args.stage:
            raise ValueError("Wrong-stage adapter")
        args.rank, args.dropout = loaded["rank"], loaded["dropout"]
    else:
        add_adapter(trainable, args.stage, args.rank, args.dropout)
    if args.stage == "stage1":
        pipe.text_encoder.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        pipe.text_encoder.train()
        pipe.text_encoder.model.model.visual.eval()
    else:
        pipe.dit.train()
    pipe.scheduler.set_timesteps(1000, training=True)
    model = TrainingModel(pipe, args)
    params = [p for p in model.parameters() if p.requires_grad]
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    expected_prefix = "pipe.text_encoder." if args.stage == "stage1" else "pipe.dit."
    if any(not n.startswith(expected_prefix) or "lora_" not in n for n in names):
        raise RuntimeError("Trainable boundary violation")
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    model, optim = accelerator.prepare(model, optim)
    schedule, report = make_schedule(
        rows,
        args.stage,
        accelerator.num_processes,
        args.accumulation,
        steps=args.steps,
        seed=args.seed,
    )
    total_steps = report["steps"]
    if not 0 <= args.warmup_steps < total_steps:
        raise ValueError(
            "warmup_steps must be nonnegative and smaller than total steps"
        )
    local = schedule[accelerator.process_index :: accelerator.num_processes]
    out = Path(args.output)
    if accelerator.is_main_process:
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "schedule.json", report)
        write_json(
            out / "run.json",
            {
                **vars(args),
                "world_size": accelerator.num_processes,
                "trainable_parameters": sum(p.numel() for p in params),
                "trainable_tensors": len(params),
                "trainable_names": names,
            },
        )
    accelerator.wait_for_everyone()
    torch.manual_seed(args.seed + accelerator.process_index)
    logfile = (out / f"metrics.rank{accelerator.process_index}.jsonl").open("w")
    config = {
        "stage": args.stage,
        "rank": args.rank,
        "dropout": args.dropout,
        "qwen": args.qwen,
        "samtok": args.samtok,
    }
    if args.stage == "stage2":
        config["conditioning_identity"] = manifest["identity"]
    optim.zero_grad(set_to_none=True)
    probe = next(p for n, p in trainable.named_parameters() if "lora_B" in n)
    micro_gradient = {}
    probe_hook = probe.register_hook(
        lambda grad: micro_gradient.update(
            probe_micro_grad_norm=float(grad.float().norm())
        )
    )
    previous_probe = probe.detach().clone()
    start = time.monotonic()
    for micro, index in enumerate(local):
        step_index = micro // args.accumulation
        if step_index < args.warmup_steps:
            factor = (step_index + 1) / args.warmup_steps
        elif args.lr_schedule == "cosine":
            factor = 0.5 * (
                1
                + math.cos(
                    math.pi
                    * (step_index - args.warmup_steps)
                    / max(1, total_steps - args.warmup_steps)
                )
            )
        else:
            factor = 1.0
        for group in optim.param_groups:
            group["lr"] = args.lr * factor
        micro_gradient.clear()
        sync = (micro + 1) % args.accumulation == 0
        with contextlib.nullcontext() if sync else accelerator.no_sync(model):
            loss = model(rows[index])
            accelerator.backward(loss / args.accumulation)
        record = dict(accelerator.unwrap_model(model).last_metrics)
        record.update(
            micro_step=micro + 1,
            row_index=index,
            rank=accelerator.process_index,
            lr=args.lr * factor,
            **micro_gradient,
        )
        if sync:
            step = (micro + 1) // args.accumulation
            # Do not let Accelerate advance a scheduler once per process.
            grad_norm = torch.nn.utils.clip_grad_norm_(
                params, args.max_grad_norm, error_if_nonfinite=True
            )
            if any(
                p.grad is not None for p in pipe.parameters() if not p.requires_grad
            ):
                raise RuntimeError("Frozen parameter received gradients")
            optim.step()
            update = (probe.detach() - previous_probe).float().norm().item()
            previous_probe.copy_(probe.detach())
            signatures = accelerator.gather(
                probe.detach()
                .float()
                .reshape(-1)[:: max(1, probe.numel() // 64)]
                .unsqueeze(0)
            )
            if not torch.equal(signatures, signatures[0:1].expand_as(signatures)):
                raise RuntimeError("DDP parameter signature divergence")
            norms = accelerator.gather(probe.detach().float().norm().reshape(1))
            if not torch.allclose(
                norms, norms[0].expand_as(norms), atol=1e-6, rtol=1e-5
            ):
                raise RuntimeError("DDP parameter divergence")
            record.update(
                optimizer_step=step,
                grad_norm=float(grad_norm),
                probe_update=update,
                ddp_probe_norms=norms.tolist(),
                ddp_probe_equal=True,
                peak_memory_gib=torch.cuda.max_memory_allocated() / 2**30,
            )
            optim.zero_grad(set_to_none=True)
            if accelerator.is_main_process:
                print(
                    json.dumps({**record, "elapsed": time.monotonic() - start}),
                    flush=True,
                )
                if step % args.save_every == 0 or micro + 1 == len(local):
                    save_adapter(
                        trainable,
                        out / f"step-{step:06d}",
                        {**config, "optimizer_step": step},
                    )
        logfile.write(json.dumps(record) + "\n")
        logfile.flush()
    logfile.close()
    probe_hook.remove()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        write_json(
            out / "complete.json",
            {
                "steps": len(local) // args.accumulation,
                "seconds": time.monotonic() - start,
                "schedule": report,
            },
        )
    accelerator.end_training()


@torch.no_grad()
def cache(args):
    accelerator = Accelerator()
    if Path(args.output, "manifest.json").exists():
        raise ValueError("Completed cache exists; use a fresh output directory")
    rows = read_rows(args.metadata)
    if any(r["sample_type"] == "edit_ntp" for r in rows):
        raise ValueError("Cache metadata must contain only FM rows")
    pipe = load_pipeline(
        args.qwen,
        args.samtok,
        device=accelerator.device,
        components=("text_encoder", "vae"),
    )
    if args.te_adapter:
        load_adapter(pipe.text_encoder, args.te_adapter)
    pipe.eval()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    identity = {
        "metadata_sha256": file_hash(args.metadata),
        "te_adapter": adapter_identity(args.te_adapter),
        "qwen": args.qwen,
        "samtok": args.samtok,
        "max_pixels": args.max_pixels,
    }
    for i in range(accelerator.process_index, len(rows), accelerator.num_processes):
        row = rows[i]
        inputs = prepare_fm(pipe, row, args.base_path, args.max_pixels)

        def cpu(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu()
            if isinstance(x, list):
                return [cpu(v) for v in x]
            if isinstance(x, dict):
                return {k: cpu(v) for k, v in x.items()}
            return x

        name = f"{i:08d}.pt"
        torch.save(cpu(inputs), out / (name + ".tmp"))
        (out / (name + ".tmp")).replace(out / name)
        write_json(
            out / f"{i:08d}.json",
            {
                "row_hash": row_hash(row),
                "identity": identity,
                "sha256": file_hash(out / name),
            },
        )
        print(f"cache {i + 1}/{len(rows)}", flush=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        for i, row in enumerate(rows):
            side = json.loads((out / f"{i:08d}.json").read_text())
            if side["row_hash"] != row_hash(row) or side["identity"] != identity:
                raise ValueError("Mixed or stale cache")
            if side["sha256"] != file_hash(out / f"{i:08d}.pt"):
                raise ValueError("Cache checksum mismatch")
            validate_conditioning(torch.load(out / f"{i:08d}.pt", weights_only=True))
        write_json(
            out / "manifest.json",
            {
                "format": "samtok21-cache-v1",
                "identity": identity,
                "rows": [
                    {**r, "_cache_file": f"{i:08d}.pt"} for i, r in enumerate(rows)
                ],
            },
        )
    accelerator.end_training()
