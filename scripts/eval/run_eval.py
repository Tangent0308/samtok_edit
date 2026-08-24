#!/usr/bin/env python3
"""Run the unified eight-setting SAMTokEdit image-editing evaluation.

The script deliberately separates the stock Qwen-Image-Edit-2511 text encoder,
the initial SAMTok gres-ft text encoder, the Stage-1 TE-LoRA checkpoint, and an
optional Stage-2 DiT-LoRA checkpoint.  A ``--dry_run`` performs all metadata/
model-artifact checks without loading a model or producing an image.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from glob import glob
from math import prod
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
from safetensors import safe_open

_REPO_ROOT = Path(__file__).resolve().parents[2]
for path in [
    _REPO_ROOT / "DiffSynth-Studio",
    _REPO_ROOT / "scripts" / "inference",
    _REPO_ROOT / "scripts" / "data",
]:
    sys.path.insert(0, str(path))

from diffsynth.core.data.samtok_dataset import (  # noqa: E402
    parse_and_canonicalize_mt_cot,
)
from diffsynth.pipelines.qwen_image import (  # noqa: E402
    ModelConfig,
    QwenImagePipeline,
)
from infer_samtok_edit import (  # noqa: E402
    DEFAULT_QWEN_2511,
    DEFAULT_SAMTOK_TE,
    build_pipeline,
    run_edit,
)


DEFAULT_EXPERIMENT_ROOT = Path(
    "/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit"
)
DEFAULT_DATASET_BASE = (
    DEFAULT_EXPERIMENT_ROOT / "validation_edit_mt_64/data/crispedit_samtok"
)
DEFAULT_VALSET = DEFAULT_DATASET_BASE / "validation_edit_mt.jsonl"
DEFAULT_MERGED_TE = DEFAULT_EXPERIMENT_ROOT / "artifacts/merged_samtok_te"
DEFAULT_STAGE1_TE_LORA = (
    DEFAULT_EXPERIMENT_ROOT
    / "stage1_20k_mt/train_8gpu_lambda_0.05_1/step-5000.safetensors"
)
DEFAULT_OUTPUT_DIR = DEFAULT_EXPERIMENT_ROOT / "stage1_evaluation/five_settings"


@dataclass(frozen=True)
class SettingSpec:
    key: str
    number: int
    text_encoder: str
    stage1_te_lora: bool
    cot_mode: str
    description: str


@dataclass(frozen=True)
class DistributedContext:
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


SETTING_SPECS = (
    SettingSpec(
        key="s1_qwen2511_stock",
        number=1,
        text_encoder="Qwen-Image-Edit-2511 stock text encoder",
        stage1_te_lora=False,
        cot_mode="disabled",
        description="Stock Qwen-Image-Edit-2511 official direct edit",
    ),
    SettingSpec(
        key="s2_samtok_initial_direct",
        number=2,
        text_encoder="Qwen2.5-VL-7B-SAMTok-gres-ft",
        stage1_te_lora=False,
        cot_mode="disabled",
        description="Initial SAMTok TE, direct edit, no CoT generation",
    ),
    SettingSpec(
        key="s3_stage1_te_direct",
        number=3,
        text_encoder="Qwen2.5-VL-7B-SAMTok-gres-ft + Stage-1 TE LoRA",
        stage1_te_lora=True,
        cot_mode="disabled",
        description="Stage-1 TE, direct edit, no CoT generation",
    ),
    SettingSpec(
        key="s4_stage1_te_online_cot",
        number=4,
        text_encoder="Qwen2.5-VL-7B-SAMTok-gres-ft + Stage-1 TE LoRA",
        stage1_te_lora=True,
        cot_mode="online",
        description="Stage-1 TE autoregressively predicts CoT, then edits",
    ),
    SettingSpec(
        key="s5_stage1_te_gt_cot",
        number=5,
        text_encoder="Qwen2.5-VL-7B-SAMTok-gres-ft + Stage-1 TE LoRA",
        stage1_te_lora=True,
        cot_mode="ground_truth",
        description="Stage-1 TE edits with validation-row ground-truth CoT",
    ),
    SettingSpec(
        key="s6_stage2_direct",
        number=6,
        text_encoder="Qwen2.5-VL-7B-SAMTok-gres-ft + Stage-1 TE LoRA",
        stage1_te_lora=True,
        cot_mode="disabled",
        description="Stage-2 DiT LoRA direct edit, no CoT generation",
    ),
    SettingSpec(
        key="s7_stage2_online_cot",
        number=7,
        text_encoder="Qwen2.5-VL-7B-SAMTok-gres-ft + Stage-1 TE LoRA",
        stage1_te_lora=True,
        cot_mode="online",
        description="Stage-2 DiT LoRA edit with online mask-token CoT",
    ),
    SettingSpec(
        key="s8_stage2_gt_cot",
        number=8,
        text_encoder="Qwen2.5-VL-7B-SAMTok-gres-ft + Stage-1 TE LoRA",
        stage1_te_lora=True,
        cot_mode="ground_truth",
        description="Stage-2 DiT LoRA edit with validation ground-truth CoT",
    ),
)
SETTING_BY_KEY = {spec.key: spec for spec in SETTING_SPECS}
SETTING_ALIASES = {
    str(spec.number): spec.key for spec in SETTING_SPECS
} | {f"s{spec.number}": spec.key for spec in SETTING_SPECS}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_data_path(path: str | Path, base: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else base / path


def parse_settings(values: list[str]) -> list[SettingSpec]:
    if not values or values == ["all"]:
        return list(SETTING_SPECS)
    keys: list[str] = []
    for value in values:
        for token in value.split(","):
            token = token.strip()
            key = SETTING_ALIASES.get(token, token)
            if key not in SETTING_BY_KEY:
                allowed = ", ".join(spec.key for spec in SETTING_SPECS)
                raise ValueError(f"Unknown setting {token!r}; expected 1-8 or one of: {allowed}")
            if key not in keys:
                keys.append(key)
    return [SETTING_BY_KEY[key] for key in keys]


def initialize_distributed(device: str) -> tuple[DistributedContext, str]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    context = DistributedContext(world_size, rank, local_rank)
    if not context.enabled:
        return context, device
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed SAMTokEdit evaluation requires CUDA")
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} exceeds visible CUDA device count "
            f"{torch.cuda.device_count()}"
        )
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="gloo")
    return context, f"cuda:{local_rank}"


def distributed_preflight(
    context: DistributedContext,
    callback,
):
    """Run expensive validation on rank 0 and broadcast either data or failure."""

    if not context.enabled:
        return callback()
    payload = [None]
    if context.is_main:
        try:
            payload[0] = {"ok": True, "value": callback()}
        except Exception as error:  # Broadcast rank-0 validation failures to every rank.
            payload[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    torch.distributed.broadcast_object_list(payload, src=0)
    if not payload[0]["ok"]:
        raise RuntimeError(payload[0]["error"])
    return payload[0]["value"]


def _verify_image(path: Path, field: str, line_number: int) -> tuple[int, int]:
    if not path.is_file():
        raise FileNotFoundError(f"line {line_number}: missing {field}: {path}")
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return image.size
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(
            f"line {line_number}: {field} is not a decodable image: {path}"
        ) from error


def load_and_validate_rows(
    metadata_path: Path,
    dataset_base: Path,
    start_index: int = 0,
    max_samples: int | None = None,
) -> tuple[list[dict], dict]:
    """Validate the complete metadata file, then select a stable contiguous slice."""

    if not metadata_path.is_file():
        raise FileNotFoundError(f"Validation metadata does not exist: {metadata_path}")
    rows: list[dict] = []
    source_refs: set[str] = set()
    target_refs: set[str] = set()
    provenance_ids: set[tuple[str, int]] = set()
    edit_types: Counter[str] = Counter()
    source_sizes: Counter[str] = Counter()
    empty_cot = 0

    with metadata_path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            line_number = index + 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"line {line_number}: invalid JSON") from error
            required = {"image", "edit_image", "prompt", "sample_type", "mt_cot"}
            missing = sorted(required - row.keys())
            if missing:
                raise ValueError(f"line {line_number}: missing fields {missing}")
            if row["sample_type"] != "edit_mt":
                raise ValueError(
                    f"line {line_number}: expected sample_type=edit_mt, "
                    f"got {row['sample_type']!r}"
                )
            if not isinstance(row["prompt"], str) or not row["prompt"].strip():
                raise ValueError(f"line {line_number}: prompt must be a non-empty string")
            if any(ord(character) > 127 for character in row["prompt"]):
                raise ValueError(f"line {line_number}: prompt is not English/ASCII-only")
            if any(ord(character) > 127 for character in str(row["mt_cot"])):
                raise ValueError(f"line {line_number}: mt_cot is not English/ASCII-only")
            canonical = parse_and_canonicalize_mt_cot(row["mt_cot"])
            if canonical is None or canonical != row["mt_cot"]:
                raise ValueError(f"line {line_number}: mt_cot is not canonical")
            if canonical == "```json\n[]\n```":
                empty_cot += 1

            source_ref = str(row["edit_image"])
            target_ref = str(row["image"])
            if source_ref in source_refs:
                raise ValueError(f"line {line_number}: duplicate source reference {source_ref}")
            if target_ref in target_refs:
                raise ValueError(f"line {line_number}: duplicate target reference {target_ref}")
            source_refs.add(source_ref)
            target_refs.add(target_ref)

            provenance = row.get("provenance")
            if not isinstance(provenance, dict):
                raise ValueError(f"line {line_number}: missing provenance object")
            try:
                provenance_id = (
                    str(provenance["source_parquet"]),
                    int(provenance["row_idx"]),
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"line {line_number}: invalid provenance source_parquet/row_idx"
                ) from error
            if provenance_id in provenance_ids:
                raise ValueError(
                    f"line {line_number}: duplicate provenance identity {provenance_id}"
                )
            provenance_ids.add(provenance_id)
            edit_types[str(provenance.get("edit_type", "unknown"))] += 1

            source_size = _verify_image(
                resolve_data_path(source_ref, dataset_base), "edit_image", line_number
            )
            _verify_image(resolve_data_path(target_ref, dataset_base), "image", line_number)
            source_sizes[f"{source_size[0]}x{source_size[1]}"] += 1
            row = dict(row)
            row["_metadata_index"] = len(rows)
            rows.append(row)

    if not rows:
        raise ValueError(f"Validation metadata is empty: {metadata_path}")
    if start_index < 0 or start_index >= len(rows):
        raise ValueError(
            f"start_index must be in [0, {len(rows) - 1}], got {start_index}"
        )
    stop = None if max_samples is None else start_index + max_samples
    selected = rows[start_index:stop]
    if not selected:
        raise ValueError("No rows selected")

    report = {
        "metadata_path": str(metadata_path.resolve()),
        "metadata_sha256": sha256_file(metadata_path),
        "dataset_base": str(dataset_base.resolve()),
        "total_rows_validated": len(rows),
        "selected_rows": len(selected),
        "selected_index_start": selected[0]["_metadata_index"],
        "selected_index_stop_exclusive": selected[-1]["_metadata_index"] + 1,
        "sample_types": {"edit_mt": len(rows)},
        "canonical_nonempty_cot": len(rows) - empty_cot,
        "canonical_empty_cot": empty_cot,
        "edit_types": dict(sorted(edit_types.items())),
        "source_sizes": dict(sorted(source_sizes.items())),
        "unique_source_refs": len(source_refs),
        "unique_target_refs": len(target_refs),
        "unique_provenance_ids": len(provenance_ids),
    }
    return selected, report


def _require_files(label: str, paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {label}: {missing[:5]}")


DIT_TARGET_FAMILIES = (
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
)
CHECKPOINT_PATTERN = re.compile(r"^step-(\d+)\.safetensors$")


def _lora_family(key: str) -> str | None:
    for family in DIT_TARGET_FAMILIES:
        if f".{family}.lora_" in key:
            return family
    return None


def validate_stage2_checkpoint(
    checkpoint: Path,
    *,
    training_world_size: int,
) -> dict:
    """Strongly validate that ``checkpoint`` is the formal DiT-only LoRA."""

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Stage-2 DiT LoRA does not exist: {checkpoint}")
    match = CHECKPOINT_PATTERN.fullmatch(checkpoint.name)
    if match is None:
        raise ValueError(
            "Stage-2 checkpoint filename must be step-<N>.safetensors, got "
            f"{checkpoint.name!r}"
        )
    checkpoint_step = int(match.group(1))
    if checkpoint_step <= 0 or training_world_size <= 0:
        raise ValueError("checkpoint step and training_world_size must be positive")

    dtype_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    parameter_count = 0
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        key_set = set(keys)
        for key in keys:
            tensor_slice = handle.get_slice(key)
            dtype_counts[str(tensor_slice.get_dtype())] += 1
            parameter_count += prod(tensor_slice.get_shape())
            family = _lora_family(key)
            if family is None:
                raise ValueError(
                    f"Stage-2 checkpoint contains a non-target DiT LoRA key: {key}"
                )
            family_counts[family] += 1

    lora_a = [key for key in keys if ".lora_A." in key]
    lora_b = [key for key in keys if ".lora_B." in key]
    non_lora = [
        key for key in keys if ".lora_A." not in key and ".lora_B." not in key
    ]
    missing_pairs = []
    for key in lora_a:
        partner = key.replace(".lora_A.", ".lora_B.", 1)
        if partner not in key_set:
            missing_pairs.append((key, partner))
    for key in lora_b:
        partner = key.replace(".lora_B.", ".lora_A.", 1)
        if partner not in key_set:
            missing_pairs.append((key, partner))
    if non_lora or missing_pairs or len(lora_a) != len(lora_b):
        raise ValueError(
            "Stage-2 checkpoint must contain paired DiT LoRA A/B tensors only; "
            f"keys={len(keys)}, A={len(lora_a)}, B={len(lora_b)}, "
            f"non_lora={len(non_lora)}, missing_pairs={len(missing_pairs)}"
        )
    if len(keys) != 1440 or len(lora_a) != 720:
        raise ValueError(
            "Stage-2 checkpoint does not match the official 2511 rank-32 target "
            f"layout: keys={len(keys)}, pairs={len(lora_a)}"
        )
    if set(dtype_counts) != {"BF16"}:
        raise ValueError(
            f"Stage-2 checkpoint must be entirely BF16, got {dict(dtype_counts)}"
        )
    expected_family_counts = {family: 120 for family in DIT_TARGET_FAMILIES}
    if dict(family_counts) != expected_family_counts:
        raise ValueError(
            "Unexpected Stage-2 target-family counts: "
            f"{dict(family_counts)} != {expected_family_counts}"
        )

    training_args_path = checkpoint.parent / "training_args.json"
    training_args = None
    if training_args_path.is_file():
        training_args = json.loads(training_args_path.read_text(encoding="utf-8"))
        expected = {
            "lora_base_model": "dit",
            "lora_rank": 32,
            "dataset_repeat": 2,
            "num_epochs": 1,
            "gradient_accumulation_steps": 1,
        }
        mismatches = {
            key: {"expected": value, "actual": training_args.get(key)}
            for key, value in expected.items()
            if training_args.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"Stage-2 training_args do not match the formal run: {mismatches}"
            )

    gradient_accumulation_steps = (
        int(training_args["gradient_accumulation_steps"]) if training_args else 1
    )
    samples_per_optimizer_step = training_world_size * gradient_accumulation_steps
    return {
        "dit_lora": str(checkpoint.resolve()),
        "dit_lora_sha256": sha256_file(checkpoint),
        "dit_lora_size_bytes": checkpoint.stat().st_size,
        "dit_lora_tensor_keys": len(keys),
        "dit_lora_pairs": len(lora_a),
        "dit_lora_parameter_count": parameter_count,
        "dit_lora_dtypes": dict(dtype_counts),
        "dit_lora_target_family_counts": dict(sorted(family_counts.items())),
        "checkpoint_step": checkpoint_step,
        "training_args": str(training_args_path.resolve())
        if training_args_path.is_file()
        else None,
        "training_world_size": training_world_size,
        "samples_per_optimizer_step": samples_per_optimizer_step,
        "samples_consumed_with_repeat": checkpoint_step
        * samples_per_optimizer_step,
        "consumption_note": (
            "Counts repeated sample presentations consumed by all training ranks "
            "through this checkpoint; it is not a unique-physical-row count."
        ),
    }


def validate_model_artifacts(
    qwen_2511_dir: Path,
    samtok_te_dir: Path,
    merged_te_dir: Path,
    stage1_te_lora: Path,
    stage2_dit_lora: Path | None,
    training_world_size: int,
    settings: list[SettingSpec],
) -> dict:
    """Check exact base/LoRA model roles without loading full model tensors."""

    model_index_path = qwen_2511_dir / "model_index.json"
    _require_files("Qwen-Image-Edit-2511 model index", [model_index_path])
    model_index = json.loads(model_index_path.read_text(encoding="utf-8"))
    if model_index.get("_class_name") != "QwenImageEditPlusPipeline":
        raise ValueError(
            "--qwen_2511_dir is not a Qwen-Image-Edit-2511/EditPlus checkpoint: "
            f"_class_name={model_index.get('_class_name')!r}"
        )
    qwen_dit = sorted(
        Path(path)
        for path in glob(
            str(qwen_2511_dir / "transformer/diffusion_pytorch_model*.safetensors")
        )
    )
    qwen_te = sorted(
        Path(path)
        for path in glob(str(qwen_2511_dir / "text_encoder/model*.safetensors"))
    )
    qwen_vae = qwen_2511_dir / "vae/diffusion_pytorch_model.safetensors"
    _require_files("Qwen-Image-Edit-2511 artifacts", qwen_dit + qwen_te + [qwen_vae])
    if len(qwen_dit) != 5 or len(qwen_te) != 4:
        raise ValueError(
            "Expected Qwen-Image-Edit-2511 to have 5 DiT and 4 TE shards; "
            f"found {len(qwen_dit)} and {len(qwen_te)}"
        )
    _require_files(
        "Qwen-Image-Edit-2511 processor/tokenizer",
        [
            qwen_2511_dir / "processor/preprocessor_config.json",
            qwen_2511_dir / "processor/tokenizer_config.json",
            qwen_2511_dir / "tokenizer/tokenizer_config.json",
        ],
    )

    report: dict = {
        "qwen_2511_dir": str(qwen_2511_dir.resolve()),
        "qwen_2511_dit_shards": len(qwen_dit),
        "qwen_2511_stock_te_shards": len(qwen_te),
        "qwen_2511_vae": str(qwen_vae.resolve()),
        "dit_lora": None,
    }
    needs_samtok = any(spec.number >= 2 for spec in settings)
    if needs_samtok:
        samtok_te = sorted(
            Path(path) for path in glob(str(samtok_te_dir / "model*.safetensors"))
        )
        _require_files("SAMTok gres-ft TE shards", samtok_te)
        if len(samtok_te) != 4:
            raise ValueError(
                f"Expected 4 SAMTok gres-ft TE shards, found {len(samtok_te)}"
            )
        _require_files(
            "merged SAMTok processor",
            [
                merged_te_dir / "samtok_edit_manifest.json",
                merged_te_dir / "preprocessor_config.json",
                merged_te_dir / "tokenizer_config.json",
            ],
        )
        manifest = json.loads(
            (merged_te_dir / "samtok_edit_manifest.json").read_text(encoding="utf-8")
        )
        expected_samtok = str(samtok_te_dir.resolve())
        if str(Path(manifest.get("samtok_checkpoint", "")).resolve()) != expected_samtok:
            raise ValueError(
                "Merged processor manifest does not match --samtok_te_dir: "
                f"{manifest.get('samtok_checkpoint')!r} != {expected_samtok!r}"
            )
        report.update(
            {
                "samtok_te_dir": expected_samtok,
                "samtok_te_shards": len(samtok_te),
                "merged_te_dir": str(merged_te_dir.resolve()),
                "merged_te_model_hash": manifest.get("te_model_hash"),
                "merged_te_tokenizer_length": manifest.get("tokenizer_length"),
            }
        )

    needs_stage1 = any(spec.stage1_te_lora for spec in settings)
    if needs_stage1:
        if not stage1_te_lora.is_file():
            raise FileNotFoundError(f"Stage-1 TE LoRA does not exist: {stage1_te_lora}")
        with safe_open(stage1_te_lora, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
        lora_a = sum(".lora_A." in key for key in keys)
        lora_b = sum(".lora_B." in key for key in keys)
        if not keys or lora_a != lora_b or lora_a * 2 != len(keys):
            raise ValueError(
                "Stage-1 checkpoint must contain paired TE LoRA A/B tensors only; "
                f"keys={len(keys)}, A={lora_a}, B={lora_b}"
            )
        if not all(key.startswith("model.language_model.") for key in keys):
            raise ValueError("Stage-1 checkpoint contains non-TE LoRA keys")
        report.update(
            {
                "stage1_te_lora": str(stage1_te_lora.resolve()),
                "stage1_te_lora_sha256": sha256_file(stage1_te_lora),
                "stage1_te_lora_size_bytes": stage1_te_lora.stat().st_size,
                "stage1_te_lora_tensor_keys": len(keys),
                "stage1_te_lora_pairs": lora_a,
            }
        )
    needs_stage2 = any(spec.number >= 6 for spec in settings)
    if needs_stage2:
        if stage2_dit_lora is None:
            raise ValueError("--dit_lora is required for settings 6-8")
        report.update(
            validate_stage2_checkpoint(
                stage2_dit_lora,
                training_world_size=training_world_size,
            )
        )
    return report


def load_stock_pipeline(qwen_2511_dir: Path, device: str):
    """Load the unmodified 2511 components through DiffSynth's official pipeline."""

    return QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(
                path=sorted(
                    glob(
                        str(
                            qwen_2511_dir
                            / "transformer/diffusion_pytorch_model*.safetensors"
                        )
                    )
                )
            ),
            ModelConfig(
                path=sorted(
                    glob(str(qwen_2511_dir / "text_encoder/model*.safetensors"))
                )
            ),
            ModelConfig(
                path=str(qwen_2511_dir / "vae/diffusion_pytorch_model.safetensors")
            ),
        ],
        tokenizer_config=ModelConfig(path=str(qwen_2511_dir / "tokenizer")),
        processor_config=ModelConfig(path=str(qwen_2511_dir / "processor")),
    )


def stock_edit(pipe, source, prompt, seed, steps, cfg_scale):
    """Use the official Qwen-Image-Edit-2511 generation arguments."""

    return pipe(
        prompt,
        edit_image=[source],
        seed=seed,
        num_inference_steps=steps,
        cfg_scale=cfg_scale,
        height=source.size[1],
        width=source.size[0],
        edit_image_auto_resize=True,
        zero_cond_t=True,
    )


def run_samtok_setting(pipe, spec: SettingSpec, row: dict, source, common: dict):
    if spec.cot_mode == "disabled":
        return run_edit(
            pipe,
            source,
            row["prompt"],
            mt_cot=None,
            enable_samtok_cot=False,
            **common,
        )
    if spec.cot_mode == "online":
        return run_edit(
            pipe,
            source,
            row["prompt"],
            mt_cot=None,
            enable_samtok_cot=True,
            **common,
        )
    if spec.cot_mode == "ground_truth":
        return run_edit(
            pipe,
            source,
            row["prompt"],
            mt_cot=row["mt_cot"],
            enable_samtok_cot=False,
            **common,
        )
    raise ValueError(f"Unsupported CoT mode: {spec.cot_mode}")


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    temporary.replace(path)


def _telemetry(pipe, spec: SettingSpec) -> dict:
    if spec.number == 1:
        return {"mt_cot": None, "parse_layer": None, "pass1_raw": None}
    return {
        "mt_cot": getattr(pipe, "last_mt_cot", None),
        "parse_layer": getattr(pipe, "last_parse_layer", None),
        "pass1_raw": getattr(pipe, "last_pass1_raw", None),
    }


def _record_paths(output_dir: Path, spec: SettingSpec, index: int) -> tuple[Path, Path]:
    setting_dir = output_dir / spec.key
    return setting_dir / f"{index:04d}.png", setting_dir / f"{index:04d}.json"


def _completed_record(output_dir: Path, spec: SettingSpec, index: int) -> dict | None:
    image_path, record_path = _record_paths(output_dir, spec, index)
    if not image_path.is_file() or not record_path.is_file():
        return None
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if record.get("setting") != spec.key or record.get("metadata_index") != index:
        return None
    return record


def run_one_setting(
    pipe,
    spec: SettingSpec,
    rows: list[dict],
    dataset_base: Path,
    output_dir: Path,
    seed: int,
    steps: int,
    cfg_scale: float,
    samtok_max_new_tokens: int,
    resume: bool,
    worker_rank: int = 0,
    world_size: int = 1,
    write_results_jsonl: bool = True,
) -> list[dict]:
    records: list[dict] = []
    for selected_index, row in enumerate(rows, start=1):
        index = row["_metadata_index"]
        if resume:
            completed = _completed_record(output_dir, spec, index)
            if completed is not None:
                records.append(completed)
                print(
                    f"[{spec.key} rank={worker_rank}] {selected_index}/{len(rows)} "
                    f"metadata_index={index} resumed",
                    flush=True,
                )
                continue

        source_path = resolve_data_path(row["edit_image"], dataset_base)
        target_path = resolve_data_path(row["image"], dataset_base)
        with Image.open(source_path) as image:
            source = image.convert("RGB")
        sample_seed = seed + index
        started = time.perf_counter()
        if spec.number == 1:
            output = stock_edit(
                pipe, source, row["prompt"], sample_seed, steps, cfg_scale
            )
        else:
            output = run_samtok_setting(
                pipe,
                spec,
                row,
                source,
                {
                    "seed": sample_seed,
                    "num_inference_steps": steps,
                    "cfg_scale": cfg_scale,
                    "samtok_max_new_tokens": samtok_max_new_tokens,
                },
            )
        elapsed = time.perf_counter() - started
        image_path, record_path = _record_paths(output_dir, spec, index)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        output.save(image_path)
        telemetry = _telemetry(pipe, spec)
        record = {
            "metadata_index": index,
            "setting": spec.key,
            "setting_number": spec.number,
            "description": spec.description,
            "source": str(source_path.resolve()),
            "target": str(target_path.resolve()),
            "output": str(image_path.resolve()),
            "prompt": row["prompt"],
            "seed": sample_seed,
            "num_inference_steps": steps,
            "cfg_scale": cfg_scale,
            "samtok_max_new_tokens": (
                samtok_max_new_tokens if spec.cot_mode == "online" else None
            ),
            "output_size": list(output.size),
            "elapsed_seconds": elapsed,
            "gt_mt_cot": row["mt_cot"],
            "conditioned_mt_cot": telemetry["mt_cot"],
            "parse_layer": telemetry["parse_layer"],
            "pass1_raw": telemetry["pass1_raw"],
            "provenance": row.get("provenance"),
            "worker_rank": worker_rank,
            "world_size": world_size,
        }
        _atomic_write_json(record_path, record)
        records.append(record)
        print(
            f"[{spec.key} rank={worker_rank}] {selected_index}/{len(rows)} "
            f"metadata_index={index} seed={sample_seed} seconds={elapsed:.2f} "
            f"parse={telemetry['parse_layer']}",
            flush=True,
        )

    records.sort(key=lambda record: record["metadata_index"])
    if write_results_jsonl:
        _atomic_write_jsonl(output_dir / spec.key / "results.jsonl", records)
    return records


def collect_setting_records(
    output_dir: Path,
    spec: SettingSpec,
    rows: list[dict],
) -> list[dict]:
    records = []
    missing = []
    for row in rows:
        index = row["_metadata_index"]
        record = _completed_record(output_dir, spec, index)
        if record is None:
            missing.append(index)
        else:
            records.append(record)
    if missing:
        raise RuntimeError(
            f"{spec.key} is missing {len(missing)} completed outputs: {missing[:16]}"
        )
    records.sort(key=lambda record: record["metadata_index"])
    _atomic_write_jsonl(output_dir / spec.key / "results.jsonl", records)
    return records


def release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _fit_panel_cell(image: Image.Image, cell_size: tuple[int, int]) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail(cell_size, Image.Resampling.LANCZOS)
    cell = Image.new("RGB", cell_size, "white")
    left = (cell_size[0] - image.size[0]) // 2
    top = (cell_size[1] - image.size[1]) // 2
    cell.paste(image, (left, top))
    return cell


def _panel_font(size: int, *, bold: bool = False):
    """Use a readable system font while retaining a Pillow-only fallback."""

    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    candidates = [
        Path("/usr/share/fonts/truetype/dejavu") / name,
        Path("/usr/share/fonts/dejavu") / name,
    ]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _wrap_panel_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font,
    max_width: int,
) -> list[str]:
    """Wrap on words according to rendered pixel width."""

    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and draw.textbbox((0, 0), candidate, font=font)[2] > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or [""]


def make_panels(
    rows: list[dict],
    settings: list[SettingSpec],
    dataset_base: Path,
    output_dir: Path,
) -> None:
    cell_size = (320, 320)
    label_height = 38
    labels = ["Source", "Target"] + [
        {
            1: "S1 Stock 2511",
            2: "S2 Initial direct",
            3: "S3 Stage-1 direct",
            4: "S4 Online CoT",
            5: "S5 GT CoT",
            6: "S6 Stage-2 direct",
            7: "S7 Stage-2 online CoT",
            8: "S8 Stage-2 GT CoT",
        }.get(spec.number, f"S{spec.number}")
        for spec in settings
    ]
    panel_dir = output_dir / "panels_with_instruction"
    panel_dir.mkdir(parents=True, exist_ok=True)
    instruction_font = _panel_font(22, bold=True)
    label_font = _panel_font(18, bold=True)
    panel_manifest = []
    for row in rows:
        index = row["_metadata_index"]
        paths = [
            resolve_data_path(row["edit_image"], dataset_base),
            resolve_data_path(row["image"], dataset_base),
        ] + [_record_paths(output_dir, spec, index)[0] for spec in settings]
        if not all(path.is_file() for path in paths):
            continue
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(_fit_panel_cell(image, cell_size))
        panel_width = cell_size[0] * len(images)
        edit_type = row.get("provenance", {}).get("edit_type", "unknown")
        instruction = row["prompt"].strip()
        heading = f"#{index:04d} | {edit_type} | Instruction: {instruction}"
        scratch = Image.new("RGB", (panel_width, 1), "white")
        scratch_draw = ImageDraw.Draw(scratch)
        heading_lines = _wrap_panel_text(
            scratch_draw, heading, instruction_font, panel_width - 32
        )
        line_height = 30
        header_height = max(58, 18 + line_height * len(heading_lines))
        panel = Image.new(
            "RGB",
            (panel_width, header_height + label_height + cell_size[1]),
            "white",
        )
        draw = ImageDraw.Draw(panel)
        draw.rectangle((0, 0, panel_width, header_height), fill=(22, 34, 52))
        for line_index, line in enumerate(heading_lines):
            draw.text(
                (16, 10 + line_index * line_height),
                line,
                font=instruction_font,
                fill="white",
            )
        for column, (image, label) in enumerate(zip(images, labels)):
            left = column * cell_size[0]
            label_bbox = draw.textbbox((0, 0), label, font=label_font)
            label_width = label_bbox[2] - label_bbox[0]
            label_left = left + max(6, (cell_size[0] - label_width) // 2)
            draw.text(
                (label_left, header_height + 8),
                label,
                font=label_font,
                fill="black",
            )
            panel.paste(image, (left, header_height + label_height))
        panel_path = panel_dir / f"{index:04d}.jpg"
        panel.save(panel_path, quality=92)
        panel_manifest.append(
            {
                "metadata_index": index,
                "edit_type": edit_type,
                "instruction": instruction,
                "panel": str(panel_path),
            }
        )
    _atomic_write_jsonl(panel_dir / "manifest.jsonl", panel_manifest)
    representative_paths = []
    seen_edit_types = set()
    for record in panel_manifest:
        if record["edit_type"] in seen_edit_types:
            continue
        seen_edit_types.add(record["edit_type"])
        representative_paths.append(Path(record["panel"]))
    if representative_paths:
        representative_images = []
        for path in representative_paths:
            with Image.open(path) as image:
                representative_images.append(image.convert("RGB"))
        overview = Image.new(
            "RGB",
            (
                max(image.width for image in representative_images),
                sum(image.height for image in representative_images),
            ),
            "white",
        )
        top = 0
        for image in representative_images:
            overview.paste(image, (0, top))
            top += image.height
        overview.save(
            panel_dir / "overview_representative_7types.jpg",
            quality=92,
        )


def build_run_config(
    args,
    settings: list[SettingSpec],
    data_report: dict,
    model_report: dict,
    context: DistributedContext | None = None,
):
    context = context or DistributedContext()
    has_stage2 = any(spec.number >= 6 for spec in settings)
    return {
        "protocol": (
            "samtok_edit_unified_evaluation_v2"
            if has_stage2
            else "samtok_edit_stage1_evaluation_v1"
        ),
        "settings": [asdict(spec) for spec in settings],
        "generation": {
            "seed_rule": f"{args.seed} + metadata_index",
            "base_seed": args.seed,
            "num_inference_steps": args.num_inference_steps,
            "cfg_scale": args.cfg_scale,
            "samtok_max_new_tokens": args.samtok_max_new_tokens,
            "edit_image_is_list": True,
            "edit_image_auto_resize": True,
            "zero_cond_t": True,
            "height_width": "source image size; pipeline rounds each dimension up to /16",
            "torch_dtype": "bfloat16",
            "dit_weights": (
                "Qwen-Image-Edit-2511 + explicit Stage-2 DiT LoRA for S6-S8"
                if has_stage2
                else "Qwen-Image-Edit-2511 stock for all settings"
            ),
        },
        "data": data_report,
        "models": model_report,
        "parallelism": {
            "world_size": context.world_size,
            "partition": "selected_rows[rank::world_size]",
            "one_setting_per_torchrun": context.enabled,
        },
        "output_dir": str(args.output_dir.resolve()),
    }


def summarize_records(records_by_setting: dict[str, list[dict]]) -> dict:
    report = {"settings": {}}
    for setting, records in records_by_setting.items():
        parse_layers = Counter(record.get("parse_layer") for record in records)
        elapsed = [float(record["elapsed_seconds"]) for record in records]
        report["settings"][setting] = {
            "count": len(records),
            "parse_layers": {
                "null" if key is None else str(key): value
                for key, value in sorted(
                    parse_layers.items(), key=lambda item: str(item[0])
                )
            },
            "elapsed_seconds_total": sum(elapsed),
            "elapsed_seconds_mean": sum(elapsed) / len(elapsed) if elapsed else None,
        }
    return report


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--valset", type=Path, default=DEFAULT_VALSET)
    parser.add_argument("--dataset_base", type=Path, default=DEFAULT_DATASET_BASE)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--qwen_2511_dir", type=Path, default=DEFAULT_QWEN_2511)
    parser.add_argument("--samtok_te_dir", type=Path, default=DEFAULT_SAMTOK_TE)
    parser.add_argument("--merged_te_dir", type=Path, default=DEFAULT_MERGED_TE)
    parser.add_argument("--stage1_te_lora", type=Path, default=DEFAULT_STAGE1_TE_LORA)
    parser.add_argument(
        "--dit_lora",
        type=Path,
        default=None,
        help="required for Stage-2 settings S6-S8; ignored by S1-S5",
    )
    parser.add_argument("--training_world_size", type=int, default=8)
    parser.add_argument(
        "--settings",
        nargs="+",
        default=["1", "2", "3", "4", "5"],
        help=(
            "default: 1-5 for backward compatibility; use all, numbers 1-8, "
            "or full setting keys (space/comma separated)"
        ),
    )
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--samtok_max_new_tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--make_panels", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="validate data/artifacts and print the exact plan; do not load models",
    )
    parser.add_argument(
        "--finalize_only",
        action="store_true",
        help="aggregate the selected completed per-setting runs and build panels; load no model",
    )
    args = parser.parse_args(argv)

    if args.num_inference_steps <= 0:
        parser.error("--num_inference_steps must be positive")
    if args.cfg_scale <= 0:
        parser.error("--cfg_scale must be positive")
    if args.samtok_max_new_tokens <= 0:
        parser.error("--samtok_max_new_tokens must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max_samples must be positive")
    try:
        settings = parse_settings(args.settings)
        context, args.device = initialize_distributed(args.device)
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    if context.enabled and len(settings) != 1:
        parser.error(
            "A distributed invocation must run exactly one setting; use the 8-GPU "
            "controller to run the selected settings sequentially"
        )
    if context.enabled and args.finalize_only:
        parser.error("--finalize_only must run in one process")

    def preflight():
        selected_rows, selected_data_report = load_and_validate_rows(
            args.valset, args.dataset_base, args.start_index, args.max_samples
        )
        selected_model_report = validate_model_artifacts(
            args.qwen_2511_dir,
            args.samtok_te_dir,
            args.merged_te_dir,
            args.stage1_te_lora,
            args.dit_lora,
            args.training_world_size,
            settings,
        )
        return selected_rows, selected_data_report, selected_model_report

    try:
        rows, data_report, model_report = distributed_preflight(context, preflight)
    except (FileNotFoundError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        if context.enabled:
            torch.distributed.destroy_process_group()
        parser.error(str(error))

    run_config = build_run_config(args, settings, data_report, model_report, context)
    if args.dry_run:
        if context.is_main:
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "dry_run": True,
                        "models_loaded": False,
                        "images_generated": 0,
                        "planned_generations": len(rows) * len(settings),
                        "planned_rows_per_rank": [
                            len(rows[rank :: context.world_size])
                            for rank in range(context.world_size)
                        ],
                        **run_config,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        if context.enabled:
            torch.distributed.destroy_process_group()
        return

    if args.finalize_only:
        if not args.output_dir.is_dir():
            parser.error(f"Output directory does not exist: {args.output_dir}")
        setting_configs = []
        records_by_setting = {}
        try:
            for spec in settings:
                setting_config_path = args.output_dir / spec.key / "run_config.json"
                if not setting_config_path.is_file():
                    raise RuntimeError(f"Missing setting config: {setting_config_path}")
                setting_config = json.loads(
                    setting_config_path.read_text(encoding="utf-8")
                )
                setting_configs.append(setting_config)
                records_by_setting[spec.key] = collect_setting_records(
                    args.output_dir, spec, rows
                )
            world_sizes = {
                int(config["parallelism"]["world_size"])
                for config in setting_configs
            }
            if len(world_sizes) != 1:
                raise RuntimeError(
                    f"Per-setting runs used inconsistent world sizes: {world_sizes}"
                )
            metadata_hashes = {
                config["data"]["metadata_sha256"] for config in setting_configs
            }
            if metadata_hashes != {data_report["metadata_sha256"]}:
                raise RuntimeError(
                    "Per-setting metadata hashes do not match finalization metadata"
                )
            if any(spec.number >= 6 for spec in settings):
                dit_hashes = {
                    config["models"]["dit_lora_sha256"]
                    for config in setting_configs
                }
                if dit_hashes != {model_report["dit_lora_sha256"]}:
                    raise RuntimeError(
                        "Per-setting Stage-2 DiT LoRA hashes do not match finalization"
                    )
            final_context = DistributedContext(world_size=world_sizes.pop())
            run_config = build_run_config(
                args, settings, data_report, model_report, final_context
            )
            _atomic_write_json(args.output_dir / "run_config.json", run_config)
            if args.make_panels:
                make_panels(rows, settings, args.dataset_base, args.output_dir)
            report = {
                "status": "complete",
                "protocol": run_config["protocol"],
                "data": data_report,
                "parallelism": run_config["parallelism"],
                **summarize_records(records_by_setting),
            }
            _atomic_write_json(args.output_dir / "report.json", report)
            print(json.dumps(report, ensure_ascii=False, indent=2))
        except (KeyError, OSError, RuntimeError, ValueError) as error:
            parser.error(str(error))
        return

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error(f"CUDA is unavailable, cannot use --device {args.device!r}")

    isolated_setting_run = len(settings) == 1

    def prepare_output():
        args.output_dir.mkdir(parents=True, exist_ok=True)
        config_dir = (
            args.output_dir / settings[0].key
            if isolated_setting_run
            else args.output_dir
        )
        if config_dir.exists() and any(config_dir.iterdir()) and not args.resume:
            raise RuntimeError(
                f"Setting output directory is not empty: {config_dir}; "
                "pass --resume to continue it"
            )
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "run_config.json"
        if args.resume and config_path.is_file():
            old_config = json.loads(config_path.read_text(encoding="utf-8"))
            if old_config != run_config:
                raise RuntimeError(
                    f"Existing {config_path} does not match this resumed invocation"
                )
        else:
            _atomic_write_json(config_path, run_config)
        return str(config_path)

    try:
        config_path = distributed_preflight(context, prepare_output)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        if context.enabled:
            torch.distributed.destroy_process_group()
        parser.error(str(error))

    worker_rows = rows[context.rank :: context.world_size]
    print(
        f"[worker] rank={context.rank}/{context.world_size} device={args.device} "
        f"setting={settings[0].key if isolated_setting_run else 'all'} "
        f"rows={[row['_metadata_index'] for row in worker_rows]} "
        f"config={config_path}",
        flush=True,
    )

    selected = {spec.number: spec for spec in settings}
    records_by_setting: dict[str, list[dict]] = {}

    if 1 in selected:
        pipe = load_stock_pipeline(args.qwen_2511_dir, args.device)
        spec = selected[1]
        records_by_setting[spec.key] = run_one_setting(
            pipe,
            spec,
            worker_rows,
            args.dataset_base,
            args.output_dir,
            args.seed,
            args.num_inference_steps,
            args.cfg_scale,
            args.samtok_max_new_tokens,
            args.resume,
            context.rank,
            context.world_size,
            not context.enabled,
        )
        del pipe
        release_cuda_memory()

    if 2 in selected:
        pipe = build_pipeline(
            args.qwen_2511_dir,
            args.samtok_te_dir,
            args.merged_te_dir,
            te_lora=None,
            dit_lora=None,
            device=args.device,
        )
        spec = selected[2]
        records_by_setting[spec.key] = run_one_setting(
            pipe,
            spec,
            worker_rows,
            args.dataset_base,
            args.output_dir,
            args.seed,
            args.num_inference_steps,
            args.cfg_scale,
            args.samtok_max_new_tokens,
            args.resume,
            context.rank,
            context.world_size,
            not context.enabled,
        )
        del pipe
        release_cuda_memory()

    stage1_specs = [spec for spec in settings if spec.number in {3, 4, 5}]
    if stage1_specs:
        pipe = build_pipeline(
            args.qwen_2511_dir,
            args.samtok_te_dir,
            args.merged_te_dir,
            te_lora=args.stage1_te_lora,
            dit_lora=None,
            device=args.device,
        )
        for spec in stage1_specs:
            records_by_setting[spec.key] = run_one_setting(
                pipe,
                spec,
                worker_rows,
                args.dataset_base,
                args.output_dir,
                args.seed,
                args.num_inference_steps,
                args.cfg_scale,
                args.samtok_max_new_tokens,
                args.resume,
                context.rank,
                context.world_size,
                not context.enabled,
            )
        del pipe
        release_cuda_memory()

    stage2_specs = [spec for spec in settings if spec.number in {6, 7, 8}]
    if stage2_specs:
        pipe = build_pipeline(
            args.qwen_2511_dir,
            args.samtok_te_dir,
            args.merged_te_dir,
            te_lora=args.stage1_te_lora,
            dit_lora=args.dit_lora,
            device=args.device,
        )
        for spec in stage2_specs:
            records_by_setting[spec.key] = run_one_setting(
                pipe,
                spec,
                worker_rows,
                args.dataset_base,
                args.output_dir,
                args.seed,
                args.num_inference_steps,
                args.cfg_scale,
                args.samtok_max_new_tokens,
                args.resume,
                context.rank,
                context.world_size,
                not context.enabled,
            )
        del pipe
        release_cuda_memory()

    if context.enabled:
        torch.distributed.barrier()

        def aggregate_setting():
            spec = settings[0]
            complete_records = collect_setting_records(args.output_dir, spec, rows)
            setting_report = {
                "status": "complete",
                "protocol": run_config["protocol"],
                "data": data_report,
                "parallelism": run_config["parallelism"],
                **summarize_records({spec.key: complete_records}),
            }
            _atomic_write_json(
                args.output_dir / spec.key / "report.json", setting_report
            )
            return setting_report

        try:
            setting_report = distributed_preflight(context, aggregate_setting)
        except RuntimeError as error:
            torch.distributed.destroy_process_group()
            parser.error(str(error))
        if context.is_main:
            print(json.dumps(setting_report, ensure_ascii=False, indent=2))
        torch.distributed.destroy_process_group()
        return

    if args.make_panels:
        make_panels(rows, settings, args.dataset_base, args.output_dir)
    report = {
        "status": "complete",
        "protocol": run_config["protocol"],
        "data": data_report,
        **summarize_records(records_by_setting),
    }
    report_path = (
        args.output_dir / settings[0].key / "report.json"
        if isolated_setting_run
        else args.output_dir / "report.json"
    )
    _atomic_write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
