import os, json, time, zipfile, torch, importlib
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger
from diffsynth.core import OffloadTrainingManager


def get_optimizer_class(customized_optimizer=None):
    if customized_optimizer is None:
        return torch.optim.AdamW
    else:
        module_name, class_name = customized_optimizer.rsplit(".", 1)
        module = importlib.import_module(module_name)
        print(f"Customized opimizer `{customized_optimizer}` imported.")
        return getattr(module, class_name)


def save_training_args(args):
    output_path = getattr(args, "output_path", None) if args is not None else None
    if output_path is None:
        return
    try:
        os.makedirs(args.output_path, exist_ok=True)
        save_path = os.path.join(args.output_path, "training_args.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=4, ensure_ascii=False, default=str)
        print(f"Training arguments saved to `{save_path}`.")
    except Exception as e:
        print(f"Warning: failed to save training arguments: {e}")


def exclude_quantized_params_from_ddp_sync(accelerator: Accelerator, model: DiffusionTrainingModule):
    """DDP broadcasts every parameter when it is constructed, but a quantized weight backed by a
    tensor subclass cannot be flattened into a broadcast bucket. Such weights are frozen and every
    rank loads them from the same checkpoint, so let DDP skip them."""
    try:
        from torch.utils._python_dispatch import is_traceable_wrapper_subclass
        quant_configs = [module.quantize_config for module in model.modules() if getattr(module, "quantize_config", None) is not None]
        ignored = [
            f"{name}.weight" for name, module in model.named_modules()
            if any(quantize.is_quantized_linear(module) for quantize in quant_configs)
            and not module.weight.requires_grad and is_traceable_wrapper_subclass(module.weight)
        ]
        if len(ignored) > 0:
            model._ddp_params_and_buffers_to_ignore = ignored
            if accelerator.is_main_process:
                print(f"{len(ignored)} quantized weights are excluded from DDP state synchronization.")
    except Exception as e:
        print(f"Warning: failed to exclude quantized weights from DDP state synchronization: {e}")


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    max_grad_norm: float = None,
    enable_model_cpu_offload: bool = False,
    enable_optimizer_cpu_offload: bool = False,
    cpu_offload_split_threshold: int = None,
    customized_optimizer: str = None,
    scheduler_factory = None,
    training_seed: int = None,
    args = None,
    **kwargs,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        enable_model_cpu_offload = args.enable_model_cpu_offload
        enable_optimizer_cpu_offload = args.enable_optimizer_cpu_offload
        cpu_offload_split_threshold = args.cpu_offload_split_threshold
        customized_optimizer = args.customized_optimizer

    if accelerator.is_main_process:
        save_training_args(args)

    optimizer_class = get_optimizer_class(customized_optimizer)
    optimizer = optimizer_class(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = (torch.optim.lr_scheduler.ConstantLR(optimizer) if scheduler_factory is None
                 else scheduler_factory(optimizer))
    manual_scheduler = scheduler_factory is not None
    # Research datasets may provide a deterministic sampler.  This keeps the
    # official optimizer/DDP/checkpoint path while allowing a project to encode
    # a global sample schedule (for example NTP:ref:noref:plain).  Ordinary
    # datasets retain the upstream shuffle=True behavior.
    sampler = getattr(dataset, "schedule_sampler", None)
    if sampler is None:
        # Compatibility for older project datasets that used the temporary
        # attribute name before the schedule API was made neutral.
        sampler = getattr(dataset, "official_sampler", None)
    if sampler is None:
        dataloader = torch.utils.data.DataLoader(
            dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset, sampler=sampler, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers
        )

    if enable_model_cpu_offload:
        if manual_scheduler:
            optimizer, dataloader = accelerator.prepare(optimizer, dataloader)
        else:
            optimizer, dataloader, scheduler = accelerator.prepare(optimizer, dataloader, scheduler)
        model.pipe.device = accelerator.device
        offload_manager = OffloadTrainingManager(model, accelerator.device, enable_optimizer_cpu_offload, cpu_offload_split_threshold)
    else:
        model.to(device=accelerator.device)
        exclude_quantized_params_from_ddp_sync(accelerator, model)
        if manual_scheduler:
            model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
        else:
            model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    initialize_deepspeed_gradient_checkpointing(accelerator)
    if training_seed is not None:
        from accelerate.utils import set_seed
        set_seed(training_seed + accelerator.process_index)
    optimizer_step = 0
    for epoch_id in range(num_epochs):
        for data in tqdm(dataloader):
            with accelerator.accumulate(model):
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                accelerator.backward(loss)
                if enable_model_cpu_offload:
                    offload_manager.after_backward()
                audit = getattr(accelerator.unwrap_model(model), "after_backward_audit", None)
                if audit is not None:
                    audit_result = audit()
                    if accelerator.is_main_process:
                        print(f"project_gradient_audit={audit_result}", flush=True)
                # Accelerate accumulates gradients across micro-steps.  Clip
                # only on the synchronized optimizer step so accumulation has
                # the same semantics as the upstream runner.
                if max_grad_norm is not None and accelerator.sync_gradients:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_grad_norm, error_if_nonfinite=True
                    )
                effective_lr = optimizer.param_groups[0]["lr"]
                optimizer.step()
                if not manual_scheduler or (accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped):
                    scheduler.step()
                if accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped:
                    optimizer_step += 1
                    if manual_scheduler and accelerator.is_main_process:
                        record = {"optimizer_step": optimizer_step, "lr": effective_lr,
                                  "loss_last_microstep": float(loss.detach())}
                        with open(os.path.join(model_logger.output_path, "optimizer_steps.jsonl"), "a") as f:
                            f.write(json.dumps(record) + "\n")
                if accelerator.sync_gradients:
                    update_hook = getattr(accelerator.unwrap_model(model), "on_optimizer_step", None)
                    if update_hook is not None:
                        update_hook(optimizer_step, accelerator, skipped=accelerator.optimizer_step_was_skipped, learning_rate=effective_lr)
                optimizer.zero_grad()
                model_logger.on_step_end(accelerator, model, save_steps, loss=loss)
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)

    model_logger.on_training_end(accelerator, model, save_steps)


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
    resume: bool = False,
    save_retries: int = 8,
    save_retry_backoff: float = 2.0,
    **kwargs,
):
    # Keep the public function usable without the argparse namespace used by
    # the example scripts.  The SAMTok adapter passes explicit defaults and
    # still uses this official entry point.
    enable_model_cpu_offload = False
    enable_optimizer_cpu_offload = False
    cpu_offload_split_threshold = None
    if args is not None:
        num_workers = args.dataset_num_workers
        enable_model_cpu_offload = args.enable_model_cpu_offload
        enable_optimizer_cpu_offload = args.enable_optimizer_cpu_offload
        cpu_offload_split_threshold = args.cpu_offload_split_threshold
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    if enable_model_cpu_offload:
        dataloader = accelerator.prepare(dataloader)
        offload_manager = OffloadTrainingManager(model, accelerator.device, enable_optimizer_cpu_offload, cpu_offload_split_threshold)
        model.pipe.device = accelerator.device
    else:
        model.to(device=accelerator.device)
        exclude_quantized_params_from_ddp_sync(accelerator, model)
        model, dataloader = accelerator.prepare(model, dataloader)
    
    reused = written = retries = 0
    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                # A cache run may be restarted after a shared-filesystem
                # interruption.  Reuse only a readable payload whose row
                # identity still points at this scheduled item; malformed or
                # stale files are regenerated below.
                if resume and _reusable_cache_file(save_path, data.get("_row_index")):
                    reused += 1
                    continue
                data = model(data)
                retries += _save_cache_file(
                    data, save_path, max_retries=save_retries,
                    retry_backoff=save_retry_backoff,
                )
                written += 1
                if enable_model_cpu_offload:
                    offload_manager.after_backward()
    print(json.dumps({
        "cache_rank": accelerator.process_index,
        "cache_reused": reused,
        "cache_written": written,
        "cache_save_retries": retries,
    }), flush=True)


def _reusable_cache_file(path, expected_row_index):
    """Return whether an existing cache payload can be reused safely.

    ``torch.save`` uses a zip container.  Loading the payload verifies both
    the container and the row index, while the final manifest pass verifies
    the full row hash, model identity, tensors, and checksum.
    """
    if expected_row_index is None or not os.path.isfile(path):
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                return False
        payload = torch.load(path, map_location="cpu", weights_only=True)
        return payload.get("row_index") == expected_row_index
    except Exception:
        return False


def _save_cache_file(data, path, *, max_retries=8, retry_backoff=2.0):
    """Write one cache payload atomically and retry transient shared-FS I/O.

    The temporary name prevents a failed ``torch.save`` from looking like a
    complete shard to a resumed run.  ``os.replace`` publishes the payload
    only after serialization has finished.
    """
    if not isinstance(max_retries, int) or max_retries < 0:
        raise ValueError("cache save retries must be a nonnegative integer")
    if not isinstance(retry_backoff, (int, float)) or retry_backoff < 0:
        raise ValueError("cache retry backoff must be nonnegative")
    path = os.fspath(path)
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)
    for attempt in range(max_retries + 1):
        temporary = f"{path}.{os.getpid()}.{attempt}.tmp"
        try:
            torch.save(data, temporary)
            # Force the completed zip to the filesystem before publishing its
            # final name.  This is cheap for local storage and avoids exposing
            # a partially flushed shard on the shared mount.
            with open(temporary, "rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            return attempt
        except (OSError, RuntimeError):
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            if attempt >= max_retries:
                raise
            # A small rank-dependent jitter prevents all 32 writers from
            # retrying the same metadata operation at exactly the same time.
            rank_jitter = 1.0 + 0.03 * (int(os.environ.get("RANK", "0")) % 16)
            time.sleep(float(retry_backoff) * (2 ** attempt) * rank_jitter)

def initialize_deepspeed_gradient_checkpointing(accelerator: Accelerator):
    if getattr(accelerator.state, "deepspeed_plugin", None) is not None:
        ds_config = accelerator.state.deepspeed_plugin.deepspeed_config
        if "activation_checkpointing" in ds_config:
            import deepspeed
            act_config = ds_config["activation_checkpointing"]
            deepspeed.checkpointing.configure(
                mpu_=None, 
                partition_activations=act_config.get("partition_activations", False),
                checkpoint_in_cpu=act_config.get("cpu_checkpointing", False),
                contiguous_checkpointing=act_config.get("contiguous_memory_optimization", False)
            )
        else:
            print("Do not find activation_checkpointing config in deepspeed config, skip initializing deepspeed gradient checkpointing.")
