#!/usr/bin/env python3
"""Run the three-setting refined Stage 2 image-editing evaluation."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image, ImageDraw
from safetensors import safe_open


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in [
    REPO_ROOT / "DiffSynth-Studio",
    REPO_ROOT / "scripts" / "inference",
    REPO_ROOT / "scripts" / "data",
    Path(__file__).resolve().parent,
]:
    sys.path.insert(0, str(path))

from diffsynth.core.data.samtok_dataset import (  # noqa: E402
    SPAN_RE,
    parse_and_canonicalize_mt_cot,
)
from infer_samtok_edit import (  # noqa: E402
    DEFAULT_QWEN_2511,
    DEFAULT_SAMTOK_TE,
    build_pipeline,
    run_edit,
)
from diffsynth.pipelines.qwen_image_samtok import build_edit_model_inputs  # noqa: E402
from run_stage1_eval import (  # noqa: E402
    DistributedContext,
    _atomic_write_json,
    _atomic_write_jsonl,
    _fit_panel_cell,
    _panel_font,
    _verify_image,
    _wrap_panel_text,
    distributed_preflight,
    initialize_distributed,
    load_stock_pipeline,
    resolve_data_path,
    sha256_file,
    validate_model_artifacts,
)


DEFAULT_EXPERIMENT_ROOT = Path(
    "/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/"
    "crispedit_refined/stage2_evaluation/scaleedit_precision_32"
)
DEFAULT_DATASET_BASE = DEFAULT_EXPERIMENT_ROOT / "data/scaleedit_samtok"
DEFAULT_VALSET = DEFAULT_DATASET_BASE / "validation.jsonl"
DEFAULT_OUTPUT_DIR = DEFAULT_EXPERIMENT_ROOT / "three_settings"
DEFAULT_MERGED_TE = Path(
    "/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/"
    "artifacts/merged_samtok_te"
)
DEFAULT_STAGE1_TE_LORA = Path(
    "/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/"
    "crispedit_refined/stage1_full/train_8gpu_1ep/step-10584.safetensors"
)
DEFAULT_STAGE2_DIT_LORA = Path(
    "/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/"
    "crispedit_refined/stage2_full/stage2_dit_lora/step-21160.safetensors"
)


@dataclass(frozen=True)
class SettingSpec:
    key: str
    number: int
    stage1_te_lora: bool
    stage2_dit_lora: bool
    cot_mode: str
    prompt_mode: str
    description: str


SETTINGS = (
    SettingSpec(
        "s1_qwen2511_stock",
        1,
        False,
        False,
        "disabled",
        "original_instruction",
        "Stock Qwen-Image-Edit-2511 official direct edit",
    ),
    SettingSpec(
        "s2_stage2_online_cot",
        2,
        True,
        True,
        "online",
        "original_instruction",
        "Stage-1 TE LoRA + Stage-2 DiT LoRA with online mask-token CoT",
    ),
    SettingSpec(
        "s3_stage2_edit_umt",
        3,
        True,
        True,
        "disabled",
        "edit_umt",
        "Stage-1 TE LoRA + Stage-2 DiT LoRA with a GT mask span in the user prompt",
    ),
)
SETTING_BY_KEY = {setting.key: setting for setting in SETTINGS}
SETTING_ALIASES = {str(setting.number): setting.key for setting in SETTINGS} | {
    f"s{setting.number}": setting.key for setting in SETTINGS
}


def parse_settings(values: list[str]) -> list[SettingSpec]:
    if not values or values == ["all"]:
        return list(SETTINGS)
    keys = []
    for value in values:
        for token in value.split(","):
            token = token.strip()
            key = SETTING_ALIASES.get(token, token)
            if key not in SETTING_BY_KEY:
                raise ValueError(f"Unknown Stage 2 evaluation setting: {token!r}")
            if key not in keys:
                keys.append(key)
    return [SETTING_BY_KEY[key] for key in keys]


def load_and_validate_rows(
    metadata_path: Path,
    dataset_base: Path,
    start_index: int = 0,
    max_samples: int | None = None,
) -> tuple[list[dict], dict]:
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Validation metadata does not exist: {metadata_path}")
    rows, ids, sample_ids = [], set(), set()
    categories, tasks, source_sizes = Counter(), Counter(), Counter()
    with metadata_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {
                "eval_index",
                "eval_id",
                "image",
                "edit_image",
                "prompt",
                "sample_type",
                "mt_cot",
                "gt_mask_span",
                "edit_umt_prompt",
                "gt_mask",
                "gt_decoded_mask",
                "primary_category",
                "selection_tags",
                "provenance",
            }
            missing = sorted(required - row.keys())
            if missing:
                raise ValueError(f"line {line_number}: missing fields {missing}")
            index = int(row["eval_index"])
            if index != len(rows) or row["eval_id"] != f"scaleedit_{index:04d}":
                raise ValueError(f"line {line_number}: eval index/id is not contiguous")
            if row["eval_id"] in ids:
                raise ValueError(f"line {line_number}: duplicate eval_id")
            ids.add(row["eval_id"])
            provenance = row["provenance"]
            sample_id = str(provenance.get("sample_id", ""))
            if not sample_id or sample_id in sample_ids:
                raise ValueError(f"line {line_number}: invalid/duplicate ScaleEdit sample_id")
            sample_ids.add(sample_id)
            if row["sample_type"] != "edit_mt":
                raise ValueError(f"line {line_number}: canonical eval view must be edit_mt")
            if not row["prompt"].strip() or not row["prompt"].isascii():
                raise ValueError(f"line {line_number}: prompt must be ASCII English")
            canonical, layer = parse_and_canonicalize_mt_cot(
                row["mt_cot"], return_layer=True
            )
            if canonical != row["mt_cot"] or layer != "strict":
                raise ValueError(f"line {line_number}: GT CoT is not non-empty canonical strict")
            cot_spans = [match.group(0) for match in SPAN_RE.finditer(row["mt_cot"])]
            if cot_spans != [row["gt_mask_span"]]:
                raise ValueError(f"line {line_number}: GT mask span and CoT diverge")
            umt_spans = [
                match.group(0) for match in SPAN_RE.finditer(row["edit_umt_prompt"])
            ]
            if umt_spans != [row["gt_mask_span"]]:
                raise ValueError(f"line {line_number}: edit_umt must contain the one GT span")
            source_size = _verify_image(
                resolve_data_path(row["edit_image"], dataset_base),
                "edit_image",
                line_number,
            )
            _verify_image(resolve_data_path(row["image"], dataset_base), "image", line_number)
            mask_size = _verify_image(
                resolve_data_path(row["gt_mask"], dataset_base), "gt_mask", line_number
            )
            decoded_size = _verify_image(
                resolve_data_path(row["gt_decoded_mask"], dataset_base),
                "gt_decoded_mask",
                line_number,
            )
            if mask_size != source_size or decoded_size != source_size:
                raise ValueError(f"line {line_number}: source/mask geometry mismatch")
            categories[row["primary_category"]] += 1
            tasks[provenance.get("final_task", "unknown")] += 1
            source_sizes[f"{source_size[0]}x{source_size[1]}"] += 1
            rows.append(dict(row))
    if len(rows) != 32:
        raise ValueError(f"Expected the curated 32-row validation set, found {len(rows)}")
    expected_categories = {
        "fine_grained": 8,
        "multi_instance": 8,
        "precise_edit": 8,
        "small_object": 8,
    }
    if dict(categories) != expected_categories:
        raise ValueError(f"Unexpected primary-category balance: {dict(categories)}")
    if start_index < 0 or start_index >= len(rows):
        raise ValueError(f"start_index must be in [0, {len(rows) - 1}]")
    stop = None if max_samples is None else start_index + max_samples
    selected = rows[start_index:stop]
    if not selected:
        raise ValueError("No evaluation rows selected")
    return selected, {
        "metadata_path": str(metadata_path.resolve()),
        "metadata_sha256": sha256_file(metadata_path),
        "dataset_base": str(dataset_base.resolve()),
        "total_rows_validated": len(rows),
        "selected_rows": len(selected),
        "selected_index_start": selected[0]["eval_index"],
        "selected_index_stop_exclusive": selected[-1]["eval_index"] + 1,
        "primary_category_counts": dict(sorted(categories.items())),
        "final_task_counts": dict(sorted(tasks.items())),
        "source_sizes": dict(sorted(source_sizes.items())),
        "unique_eval_ids": len(ids),
        "unique_scaleedit_sample_ids": len(sample_ids),
        "canonical_nonempty_gt_cot": len(rows),
        "valid_single_span_edit_umt": len(rows),
    }


def validate_model_artifacts_stage2(args, settings: list[SettingSpec]) -> dict:
    report = validate_model_artifacts(
        args.qwen_2511_dir,
        args.samtok_te_dir,
        args.merged_te_dir,
        args.stage1_te_lora,
        settings,
    )
    if any(setting.stage2_dit_lora for setting in settings):
        path = args.stage2_dit_lora
        if not path.is_file():
            raise FileNotFoundError(f"Stage-2 DiT LoRA does not exist: {path}")
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
        count_a = sum(".lora_A." in key for key in keys)
        count_b = sum(".lora_B." in key for key in keys)
        if not keys or count_a != count_b or count_a * 2 != len(keys):
            raise ValueError(
                "Stage-2 checkpoint must contain paired DiT LoRA tensors only; "
                f"keys={len(keys)}, A={count_a}, B={count_b}"
            )
        if any(key.startswith("model.language_model.") for key in keys):
            raise ValueError("Stage-2 checkpoint unexpectedly contains TE LoRA keys")
        report.update(
            {
                "dit_lora": str(path.resolve()),
                "dit_lora_sha256": sha256_file(path),
                "dit_lora_size_bytes": path.stat().st_size,
                "dit_lora_tensor_keys": len(keys),
                "dit_lora_pairs": count_a,
            }
        )
    return report


def validate_umt_tokenization(
    rows: list[dict], dataset_base: Path, merged_te_dir: Path
) -> dict:
    """Exercise the real merged processor and 2511 template without model weights."""

    from transformers import Qwen2VLProcessor

    processor = Qwen2VLProcessor.from_pretrained(merged_te_dir)
    pipe = SimpleNamespace(processor=processor, device="cpu")
    atomic, embedded = 0, 0
    for row in rows:
        span = row["gt_mask_span"]
        span_ids = processor.tokenizer(
            span, add_special_tokens=False, return_tensors="pt"
        ).input_ids[0].tolist()
        if len(span_ids) != 4:
            raise ValueError(
                f"{row['eval_id']}: mask span is not four atomic tokenizer ids: {span_ids}"
            )
        atomic += 1
        source_path = resolve_data_path(row["edit_image"], dataset_base)
        with Image.open(source_path) as image:
            source = image.convert("RGB")
        model_inputs = build_edit_model_inputs(
            pipe, row["edit_umt_prompt"], [source]
        )
        ids = model_inputs.input_ids[0].tolist()
        if not any(
            ids[start : start + len(span_ids)] == span_ids
            for start in range(len(ids) - len(span_ids) + 1)
        ):
            raise ValueError(
                f"{row['eval_id']}: mask span was not preserved inside the 2511 template"
            )
        embedded += 1
    return {
        "processor": str(merged_te_dir.resolve()),
        "rows": len(rows),
        "four_atomic_token_spans": atomic,
        "spans_preserved_in_2511_user_template": embedded,
        "passed": atomic == embedded == len(rows),
    }


def record_paths(output_dir: Path, setting: SettingSpec, index: int) -> tuple[Path, Path]:
    root = output_dir / setting.key
    return root / f"{index:04d}.png", root / f"{index:04d}.json"


def completed_record(output_dir: Path, setting: SettingSpec, index: int) -> dict | None:
    image_path, json_path = record_paths(output_dir, setting, index)
    if not image_path.is_file() or not json_path.is_file():
        return None
    try:
        record = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if record.get("setting") != setting.key or record.get("eval_index") != index:
        return None
    return record


def run_method(pipe, setting: SettingSpec, row: dict, source: Image.Image, common: dict):
    if setting.cot_mode == "online":
        prompt = row["prompt"]
        output = run_edit(
            pipe,
            source,
            prompt,
            mt_cot=None,
            enable_samtok_cot=True,
            **common,
        )
    elif setting.prompt_mode == "edit_umt":
        prompt = row["edit_umt_prompt"]
        output = run_edit(
            pipe,
            source,
            prompt,
            mt_cot=None,
            enable_samtok_cot=False,
            **common,
        )
        audit = getattr(pipe, "last_user_mask_audit", None) or {}
        if not (
            audit.get("user_mask_span_count") == 1
            and audit.get("user_mask_spans_atomic") is True
            and audit.get("user_mask_spans_in_template") is True
        ):
            raise RuntimeError(f"edit_umt token/template audit failed: {audit}")
    else:
        raise ValueError(f"Unsupported Stage 2 setting: {setting}")
    return output, prompt


def official_output_size(
    source: Image.Image, target_area: int = 1024 * 1024
) -> tuple[int, int]:
    """Return width/height at the official ~1 MP scale and source aspect ratio."""

    ratio = source.width / source.height
    width = round(math.sqrt(target_area * ratio) / 32) * 32
    height = round(math.sqrt(target_area / ratio) / 32) * 32
    return max(32, width), max(32, height)


def run_one_setting(
    pipe,
    setting: SettingSpec,
    rows: list[dict],
    dataset_base: Path,
    output_dir: Path,
    seed: int,
    steps: int,
    cfg_scale: float,
    max_new_tokens: int,
    resume: bool,
    rank: int,
    world_size: int,
) -> list[dict]:
    records = []
    for position, row in enumerate(rows, 1):
        index = int(row["eval_index"])
        if resume:
            old = completed_record(output_dir, setting, index)
            if old is not None:
                records.append(old)
                print(
                    f"[{setting.key} rank={rank}] {position}/{len(rows)} "
                    f"eval_index={index} resumed",
                    flush=True,
                )
                continue
        source_path = resolve_data_path(row["edit_image"], dataset_base)
        target_path = resolve_data_path(row["image"], dataset_base)
        with Image.open(source_path) as image:
            source = image.convert("RGB")
        output_width, output_height = official_output_size(source)
        sample_seed = seed + index
        started = time.perf_counter()
        if setting.number == 1:
            conditioned_prompt = row["prompt"]
            output = pipe(
                conditioned_prompt,
                edit_image=[source],
                seed=sample_seed,
                num_inference_steps=steps,
                cfg_scale=cfg_scale,
                height=output_height,
                width=output_width,
                edit_image_auto_resize=True,
                zero_cond_t=True,
            )
        else:
            output, conditioned_prompt = run_method(
                pipe,
                setting,
                row,
                source,
                {
                    "seed": sample_seed,
                    "num_inference_steps": steps,
                    "cfg_scale": cfg_scale,
                    "samtok_max_new_tokens": max_new_tokens,
                    "output_height": output_height,
                    "output_width": output_width,
                },
            )
        elapsed = time.perf_counter() - started
        image_path, json_path = record_paths(output_dir, setting, index)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        output.save(image_path)
        telemetry = (
            {"mt_cot": None, "parse_layer": None, "pass1_raw": None, "user_mask_audit": None}
            if setting.number == 1
            else {
                "mt_cot": getattr(pipe, "last_mt_cot", None),
                "parse_layer": getattr(pipe, "last_parse_layer", None),
                "pass1_raw": getattr(pipe, "last_pass1_raw", None),
                "user_mask_audit": getattr(pipe, "last_user_mask_audit", None),
            }
        )
        record = {
            "eval_index": index,
            "eval_id": row["eval_id"],
            "setting": setting.key,
            "setting_number": setting.number,
            "description": setting.description,
            "source": str(source_path.resolve()),
            "target": str(target_path.resolve()),
            "output": str(image_path.resolve()),
            "original_prompt": row["prompt"],
            "conditioned_prompt": conditioned_prompt,
            "seed": sample_seed,
            "num_inference_steps": steps,
            "cfg_scale": cfg_scale,
            "samtok_max_new_tokens": max_new_tokens if setting.cot_mode == "online" else None,
            "elapsed_seconds": elapsed,
            "output_size": list(output.size),
            "requested_output_size": [output_width, output_height],
            "gt_mt_cot": row["mt_cot"],
            "gt_mask_span": row["gt_mask_span"],
            "conditioned_mt_cot": telemetry["mt_cot"],
            "parse_layer": telemetry["parse_layer"],
            "pass1_raw": telemetry["pass1_raw"],
            "user_mask_audit": telemetry["user_mask_audit"],
            "primary_category": row["primary_category"],
            "selection_tags": row["selection_tags"],
            "provenance": row["provenance"],
            "worker_rank": rank,
            "world_size": world_size,
        }
        _atomic_write_json(json_path, record)
        records.append(record)
        print(
            f"[{setting.key} rank={rank}] {position}/{len(rows)} eval_index={index} "
            f"seconds={elapsed:.2f} parse={telemetry['parse_layer']}",
            flush=True,
        )
    records.sort(key=lambda record: record["eval_index"])
    if world_size == 1:
        _atomic_write_jsonl(output_dir / setting.key / "results.jsonl", records)
    return records


def collect_setting_records(
    output_dir: Path, setting: SettingSpec, rows: list[dict]
) -> list[dict]:
    records, missing = [], []
    for row in rows:
        record = completed_record(output_dir, setting, int(row["eval_index"]))
        if record is None:
            missing.append(row["eval_index"])
        else:
            records.append(record)
    if missing:
        raise RuntimeError(f"{setting.key} is missing outputs: {missing}")
    records.sort(key=lambda record: record["eval_index"])
    _atomic_write_jsonl(output_dir / setting.key / "results.jsonl", records)
    return records


def release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_edit_panels(rows: list[dict], dataset_base: Path, output_dir: Path) -> dict:
    cell_size, label_height = (320, 320), 38
    labels = ["Source", "GT edited", "Stock 2511", "Stage 2 online CoT", "Stage 2 edit_umt"]
    settings = list(SETTINGS)
    panel_root = output_dir / "visualizations/edit_comparisons"
    heading_font, label_font = _panel_font(21, bold=True), _panel_font(17, bold=True)
    by_category: defaultdict[str, list[Path]] = defaultdict(list)
    manifest = []
    for row in rows:
        index = int(row["eval_index"])
        paths = [
            resolve_data_path(row["edit_image"], dataset_base),
            resolve_data_path(row["image"], dataset_base),
            *[record_paths(output_dir, setting, index)[0] for setting in settings],
        ]
        if not all(path.is_file() for path in paths):
            raise FileNotFoundError(f"Missing edit comparison input for eval_index={index}")
        cells = []
        for path in paths:
            with Image.open(path) as image:
                cells.append(_fit_panel_cell(image, cell_size))
        width = cell_size[0] * len(cells)
        scratch = ImageDraw.Draw(Image.new("RGB", (width, 1), "white"))
        heading = (
            f"#{index:04d} | {row['primary_category']} | "
            f"Instruction: {row['prompt']}"
        )
        lines = _wrap_panel_text(scratch, heading, heading_font, width - 32)
        header_height = max(58, 18 + 29 * len(lines))
        panel = Image.new("RGB", (width, header_height + label_height + 320), "white")
        draw = ImageDraw.Draw(panel)
        draw.rectangle((0, 0, width, header_height), fill=(22, 34, 52))
        for line_index, line in enumerate(lines):
            draw.text((16, 10 + line_index * 29), line, font=heading_font, fill="white")
        for column, (cell, label) in enumerate(zip(cells, labels)):
            left = column * 320
            label_width = draw.textbbox((0, 0), label, font=label_font)[2]
            draw.text(
                (left + max(5, (320 - label_width) // 2), header_height + 8),
                label,
                font=label_font,
                fill="black",
            )
            panel.paste(cell, (left, header_height + label_height))
        category = row["primary_category"]
        path = panel_root / category / f"{index:04d}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        panel.save(path, quality=92)
        by_category[category].append(path)
        manifest.append(
            {
                "eval_index": index,
                "category": category,
                "instruction": row["prompt"],
                "panel": str(path.resolve()),
            }
        )
    overviews = {}
    for category, paths in sorted(by_category.items()):
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        overview = Image.new(
            "RGB", (max(image.width for image in images), sum(image.height for image in images)), "white"
        )
        top = 0
        for image in images:
            overview.paste(image, (0, top))
            top += image.height
        path = panel_root / f"overview_{category}.jpg"
        overview.save(path, quality=92)
        overviews[category] = str(path.resolve())
    _atomic_write_jsonl(panel_root / "manifest.jsonl", manifest)
    return {"per_case_panels": len(manifest), "category_overviews": overviews}


def summarize(records_by_setting: dict[str, list[dict]]) -> dict:
    return {
        setting: {
            "count": len(records),
            "parse_layers": dict(
                sorted(Counter(str(record.get("parse_layer")) for record in records).items())
            ),
            "elapsed_seconds_total": sum(float(record["elapsed_seconds"]) for record in records),
        }
        for setting, records in records_by_setting.items()
    }


def run_config(args, settings, data_report, model_report, context):
    return {
        "protocol": "samtok_edit_refined_stage2_scaleedit_evaluation_v1",
        "settings": [asdict(setting) for setting in settings],
        "generation": {
            "seed_rule": f"{args.seed} + eval_index",
            "num_inference_steps": args.num_inference_steps,
            "cfg_scale": args.cfg_scale,
            "samtok_max_new_tokens": args.samtok_max_new_tokens,
            "dtype": "bfloat16",
            "edit_image": "one-element source-image list",
            "edit_image_auto_resize": True,
            "zero_cond_t": True,
            "height_width": "source aspect ratio at target area 1024*1024, rounded to /32",
            "stock_setting": "QwenImagePipeline with stock 2511 TE/DiT/VAE/processor",
            "method_settings": "QwenImageSamtokPipeline with refined Stage-1 TE LoRA and Stage-2 DiT LoRA",
            "online_cot": "greedy decoding (do_sample=False) ending at <|im_end|>",
            "edit_umt": "one canonical GT mask span embedded in the user instruction; online CoT disabled",
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--valset", type=Path, default=DEFAULT_VALSET)
    parser.add_argument("--dataset_base", type=Path, default=DEFAULT_DATASET_BASE)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--qwen_2511_dir", type=Path, default=DEFAULT_QWEN_2511)
    parser.add_argument("--samtok_te_dir", type=Path, default=DEFAULT_SAMTOK_TE)
    parser.add_argument("--merged_te_dir", type=Path, default=DEFAULT_MERGED_TE)
    parser.add_argument("--stage1_te_lora", type=Path, default=DEFAULT_STAGE1_TE_LORA)
    parser.add_argument("--stage2_dit_lora", type=Path, default=DEFAULT_STAGE2_DIT_LORA)
    parser.add_argument("--settings", nargs="+", default=["all"])
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--samtok_max_new_tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--make_panels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--finalize_only", action="store_true")
    args = parser.parse_args(argv)
    if args.num_inference_steps <= 0 or args.cfg_scale <= 0 or args.samtok_max_new_tokens <= 0:
        parser.error("generation step/token counts and cfg_scale must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max_samples must be positive")
    try:
        settings = parse_settings(args.settings)
        context, args.device = initialize_distributed(args.device)
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    if context.enabled and len(settings) != 1:
        parser.error("Distributed evaluation must run exactly one setting per torchrun")
    if context.enabled and args.finalize_only:
        parser.error("--finalize_only is single-process only")

    def preflight():
        rows, data_report = load_and_validate_rows(
            args.valset, args.dataset_base, args.start_index, args.max_samples
        )
        model_report = validate_model_artifacts_stage2(args, settings)
        if any(setting.prompt_mode == "edit_umt" for setting in settings):
            data_report["edit_umt_tokenizer_audit"] = validate_umt_tokenization(
                rows, args.dataset_base, args.merged_te_dir
            )
        return rows, data_report, model_report

    try:
        rows, data_report, model_report = distributed_preflight(context, preflight)
    except (FileNotFoundError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        if context.enabled:
            torch.distributed.destroy_process_group()
        parser.error(str(error))
    config = run_config(args, settings, data_report, model_report, context)
    if args.dry_run:
        if context.is_main:
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "dry_run": True,
                        "models_loaded": False,
                        "planned_generations": len(rows) * len(settings),
                        "planned_rows_per_rank": [
                            len(rows[rank :: context.world_size]) for rank in range(context.world_size)
                        ],
                        **config,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        if context.enabled:
            torch.distributed.destroy_process_group()
        return

    if args.finalize_only:
        if settings != list(SETTINGS):
            parser.error("--finalize_only requires --settings all")
        setting_configs = []
        for setting in settings:
            path = args.output_dir / setting.key / "run_config.json"
            if not path.is_file():
                parser.error(f"Missing per-setting run config: {path}")
            setting_configs.append(json.loads(path.read_text(encoding="utf-8")))
        world_sizes = {
            int(setting_config["parallelism"]["world_size"])
            for setting_config in setting_configs
        }
        if len(world_sizes) != 1:
            parser.error(f"Per-setting world sizes differ: {world_sizes}")
        metadata_hashes = {
            setting_config["data"]["metadata_sha256"]
            for setting_config in setting_configs
        }
        if metadata_hashes != {data_report["metadata_sha256"]}:
            parser.error("Per-setting metadata hashes do not match finalization input")
        generations = [setting_config["generation"] for setting_config in setting_configs]
        shared_generation_fields = [
            "seed_rule",
            "num_inference_steps",
            "cfg_scale",
            "samtok_max_new_tokens",
            "dtype",
            "edit_image_auto_resize",
            "zero_cond_t",
        ]
        if any(
            generation[field] != generations[0][field]
            for generation in generations[1:]
            for field in shared_generation_fields
        ):
            parser.error("Per-setting generation configurations are inconsistent")
        config = run_config(
            args,
            settings,
            data_report,
            model_report,
            DistributedContext(world_size=world_sizes.pop()),
        )
        records = {
            setting.key: collect_setting_records(args.output_dir, setting, rows)
            for setting in settings
        }
        panel_report = build_edit_panels(rows, args.dataset_base, args.output_dir) if args.make_panels else None
        report = {
            "status": "complete",
            "protocol": config["protocol"],
            "data": data_report,
            "settings": summarize(records),
            "edit_visualizations": panel_report,
        }
        _atomic_write_json(args.output_dir / "run_config.json", config)
        _atomic_write_json(args.output_dir / "report.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error(f"CUDA is unavailable, cannot use {args.device}")
    isolated = len(settings) == 1

    def prepare_output():
        args.output_dir.mkdir(parents=True, exist_ok=True)
        root = args.output_dir / settings[0].key if isolated else args.output_dir
        if root.exists() and any(root.iterdir()) and not args.resume:
            raise RuntimeError(f"Output setting directory is not empty: {root}")
        root.mkdir(parents=True, exist_ok=True)
        path = root / "run_config.json"
        if args.resume and path.is_file():
            old = json.loads(path.read_text(encoding="utf-8"))
            if old != config:
                raise RuntimeError(f"Resume config mismatch: {path}")
        else:
            _atomic_write_json(path, config)
        return str(path)

    try:
        config_path = distributed_preflight(context, prepare_output)
    except (OSError, RuntimeError, ValueError) as error:
        if context.enabled:
            torch.distributed.destroy_process_group()
        parser.error(str(error))
    worker_rows = rows[context.rank :: context.world_size]
    print(
        f"[worker] rank={context.rank}/{context.world_size} device={args.device} "
        f"settings={[setting.key for setting in settings]} "
        f"rows={[row['eval_index'] for row in worker_rows]} config={config_path}",
        flush=True,
    )
    records_by_setting = {}
    if any(setting.number == 1 for setting in settings):
        setting = SETTING_BY_KEY["s1_qwen2511_stock"]
        pipe = load_stock_pipeline(args.qwen_2511_dir, args.device)
        records_by_setting[setting.key] = run_one_setting(
            pipe, setting, worker_rows, args.dataset_base, args.output_dir, args.seed,
            args.num_inference_steps, args.cfg_scale, args.samtok_max_new_tokens,
            args.resume, context.rank, context.world_size,
        )
        del pipe
        release_cuda_memory()
    method_settings = [setting for setting in settings if setting.number in {2, 3}]
    if method_settings:
        pipe = build_pipeline(
            args.qwen_2511_dir,
            args.samtok_te_dir,
            args.merged_te_dir,
            te_lora=args.stage1_te_lora,
            dit_lora=args.stage2_dit_lora,
            device=args.device,
        )
        for setting in method_settings:
            records_by_setting[setting.key] = run_one_setting(
                pipe, setting, worker_rows, args.dataset_base, args.output_dir, args.seed,
                args.num_inference_steps, args.cfg_scale, args.samtok_max_new_tokens,
                args.resume, context.rank, context.world_size,
            )
        del pipe
        release_cuda_memory()

    if context.enabled:
        torch.distributed.barrier()

        def aggregate():
            setting = settings[0]
            records = collect_setting_records(args.output_dir, setting, rows)
            report = {
                "status": "complete",
                "protocol": config["protocol"],
                "data": data_report,
                "settings": summarize({setting.key: records}),
            }
            _atomic_write_json(args.output_dir / setting.key / "report.json", report)
            return report

        try:
            report = distributed_preflight(context, aggregate)
        except RuntimeError as error:
            torch.distributed.destroy_process_group()
            parser.error(str(error))
        if context.is_main:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        torch.distributed.destroy_process_group()
        return

    if args.make_panels and settings == list(SETTINGS):
        build_edit_panels(rows, args.dataset_base, args.output_dir)
    report = {
        "status": "complete",
        "protocol": config["protocol"],
        "data": data_report,
        "settings": summarize(records_by_setting),
    }
    _atomic_write_json(args.output_dir / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
