#!/usr/bin/env python3
"""Run the A/B/C/D SAMTokEdit ablation grid and optional pass-1 mask IoU."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import Counter, defaultdict
from glob import glob
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

_REPO_ROOT = Path(__file__).resolve().parents[2]
for path in [
    _REPO_ROOT / "DiffSynth-Studio",
    _REPO_ROOT / "scripts" / "inference",
    _REPO_ROOT / "scripts" / "data",
]:
    sys.path.insert(0, str(path))

from diffsynth.pipelines.qwen_image import ModelConfig, QwenImagePipeline  # noqa: E402
from infer_samtok_edit import (  # noqa: E402
    DEFAULT_QWEN_2511,
    DEFAULT_SAMTOK_TE,
    build_pipeline,
    run_edit,
)


def resolve(path, base):
    path = Path(path)
    return path if path.is_absolute() else base / path


def load_baseline(qwen_dir: Path, device: str):
    return QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(
                path=sorted(
                    glob(
                        str(
                            qwen_dir
                            / "transformer"
                            / "diffusion_pytorch_model*.safetensors"
                        )
                    )
                )
            ),
            ModelConfig(path=sorted(glob(str(qwen_dir / "text_encoder" / "model*.safetensors")))),
            ModelConfig(path=str(qwen_dir / "vae" / "diffusion_pytorch_model.safetensors")),
        ],
        tokenizer_config=ModelConfig(path=str(qwen_dir / "tokenizer")),
        processor_config=ModelConfig(path=str(qwen_dir / "processor")),
    )


def baseline_edit(pipe, source, prompt, seed, steps, cfg_scale):
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


def fit(image, size):
    return image.convert("RGB").resize(size, Image.Resampling.LANCZOS)


def make_panel(columns, labels, size):
    label_height = 28
    canvas = Image.new("RGB", (size[0] * len(columns), size[1] + label_height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (image, label) in enumerate(zip(columns, labels)):
        canvas.paste(fit(image, size), (index * size[0], label_height))
        draw.text((index * size[0] + 6, 6), label, fill="black")
    return canvas


def overlay_masks(source, masks):
    visualization = np.asarray(source.convert("RGB")).astype(np.float32).copy()
    union = np.zeros((source.size[1], source.size[0]), dtype=bool)
    for mask in masks:
        union |= mask.astype(bool)
    visualization[union] = visualization[union] * 0.5 + np.array([255, 0, 0]) * 0.5
    return Image.fromarray(visualization.astype(np.uint8)), union


def load_gt_mask(path, base, size):
    with Image.open(resolve(path, base)) as image:
        return np.asarray(image.convert("L").resize(size, Image.Resampling.NEAREST)) > 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--valset", type=Path, required=True)
    parser.add_argument("--dataset_base", type=Path, default=Path("."))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--qwen_2511_dir", type=Path, default=DEFAULT_QWEN_2511)
    parser.add_argument("--samtok_te_dir", type=Path, default=DEFAULT_SAMTOK_TE)
    parser.add_argument("--merged_te_dir", type=Path, default=_REPO_ROOT / "models" / "merged_samtok_te")
    parser.add_argument("--te_lora", type=Path, required=True)
    parser.add_argument("--dit_lora", type=Path, required=True)
    parser.add_argument("--sam2_ckpt", type=Path, default=None)
    parser.add_argument("--mask_tokenizer_ckpt", type=Path, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip_baseline", action="store_true")
    args = parser.parse_args()

    with args.valset.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    baseline_paths = {}
    if not args.skip_baseline:
        baseline = load_baseline(args.qwen_2511_dir, args.device)
        for index, row in enumerate(rows):
            source = Image.open(resolve(row["edit_image"], args.dataset_base)).convert("RGB")
            output = baseline_edit(
                baseline,
                source,
                row["prompt"],
                args.seed + index,
                args.num_inference_steps,
                args.cfg_scale,
            )
            path = args.output_dir / f"{index:04d}_A.png"
            output.save(path)
            baseline_paths[index] = path
        del baseline
        gc.collect()
        torch.cuda.empty_cache()

    pipe = build_pipeline(
        args.qwen_2511_dir,
        args.samtok_te_dir,
        args.merged_te_dir,
        args.te_lora,
        args.dit_lora,
        args.device,
    )
    codec = None
    if args.sam2_ckpt and args.mask_tokenizer_ckpt:
        from samtok_codec import SamtokCodec

        codec = SamtokCodec(args.sam2_ckpt, args.mask_tokenizer_ckpt, args.device)

    parse_layers = Counter()
    tag_metrics = defaultdict(lambda: {"intersection": 0, "union": 0, "ious": []})
    records = []
    for index, row in enumerate(rows):
        source = Image.open(resolve(row["edit_image"], args.dataset_base)).convert("RGB")
        common = dict(
            seed=args.seed + index,
            num_inference_steps=args.num_inference_steps,
            cfg_scale=args.cfg_scale,
        )
        output_b = run_edit(pipe, source, row["prompt"], **common)
        predicted_cot = pipe.last_mt_cot
        predicted_layer = pipe.last_parse_layer or "invalid"
        parse_layers[predicted_layer] += 1
        output_c = (
            run_edit(
                pipe,
                source,
                row["prompt"],
                mt_cot=row["mt_cot_gt"],
                enable_samtok_cot=False,
                **common,
            )
            if row.get("mt_cot_gt")
            else Image.new("RGB", source.size, "white")
        )
        output_d = run_edit(
            pipe,
            source,
            row["prompt"],
            enable_samtok_cot=False,
            **common,
        )
        output_a = (
            Image.open(baseline_paths[index]).convert("RGB")
            if index in baseline_paths
            else Image.new("RGB", source.size, "white")
        )
        mask_visualization = Image.new("RGB", source.size, "white")
        iou = None
        if codec is not None:
            decoded = codec.decode(source, predicted_cot) if predicted_cot else []
            predicted_union = np.zeros(
                (source.size[1], source.size[0]), dtype=bool
            )
            if decoded:
                mask_visualization, predicted_union = overlay_masks(source, decoded)
            if row.get("gt_mask"):
                target = load_gt_mask(row["gt_mask"], args.dataset_base, source.size)
                intersection = int(np.logical_and(predicted_union, target).sum())
                union = int(np.logical_or(predicted_union, target).sum())
                iou = intersection / union if union else 1.0
                metrics = tag_metrics[row.get("tag", "all")]
                metrics["intersection"] += intersection
                metrics["union"] += union
                metrics["ious"].append(iou)

        panel = make_panel(
            [source, output_a, output_b, output_c, output_d, mask_visualization],
            ["source", "A stock", "B predicted", "C GT CoT", "D no CoT", "B pass-1"],
            source.size,
        )
        panel_path = args.output_dir / f"{index:04d}_panel.jpg"
        panel.save(panel_path, quality=92)
        records.append(
            {
                "index": index,
                "tag": row.get("tag"),
                "prompt": row["prompt"],
                "panel": str(panel_path),
                "predicted_mt_cot": predicted_cot,
                "parse_layer": predicted_layer,
                "mask_iou": iou,
            }
        )

    metrics_report = {}
    for tag, values in tag_metrics.items():
        metrics_report[tag] = {
            "count": len(values["ious"]),
            "cIoU": values["intersection"] / values["union"] if values["union"] else None,
            "gIoU": float(np.mean(values["ious"])) if values["ious"] else None,
        }
    report = {
        "num_samples": len(rows),
        "parse_layers": dict(parse_layers),
        "mask_metrics_by_tag": metrics_report,
        "records": records,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
