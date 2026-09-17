#!/usr/bin/env python3
"""Build paired original/alternate-mask interventions from the fine-grained benchmark."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in [REPO_ROOT, REPO_ROOT / "DiffSynth-Studio", REPO_ROOT / "scripts" / "data"]:
    sys.path.insert(0, str(path))

from samtok_codec import SamtokCodec  # noqa: E402


DEFAULT_BENCHMARK_ROOT = Path(
    "/mnt/bn/strategy-mllm-train/user/tanyue/datasets/samtok_edit_benchmark"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/"
    "crispedit_refined/interpretability/dit_mask_token_counterfactual"
)
DEFAULT_SAMTOK_ROOT = Path(
    "/mnt/bn/strategy-mllm-train/user/tanyue/models/SAMTok/"
    "Qwen2.5-VL-7B-SAMTok-gres-ft"
)


@dataclass(frozen=True)
class InterventionSpec:
    benchmark_index: int
    alternate_point: tuple[int, int]
    semantic_category: str


# Reviewed same-class multi-instance cases.  The alternate point lies on a
# different instance of the same semantic class; the benchmark target point is
# also supplied to SAM2 as a negative click.
CORE_SELECTIONS = (
    InterventionSpec(240, (267, 410), "zebra"),  # right zebra -> left zebra
    InterventionSpec(245, (380, 390), "elephant"),  # left elephant -> middle elephant
    InterventionSpec(488, (245, 330), "cat"),  # rightmost cat -> middle cat
)

# Category-balanced extension reviewed on the source images.  Every alternate
# click targets a different instance of the same class as the benchmark mask.
ADDITIONAL_SELECTIONS = (
    InterventionSpec(260, (350, 460), "fish"),
    InterventionSpec(255, (250, 525), "turtle"),
    InterventionSpec(274, (450, 280), "monkey"),
    InterventionSpec(259, (210, 220), "duck"),
    InterventionSpec(269, (520, 225), "chicken"),
    InterventionSpec(277, (40, 250), "horse"),
    InterventionSpec(490, (340, 150), "dog"),
    InterventionSpec(475, (560, 330), "giraffe"),
    InterventionSpec(480, (430, 350), "rabbit"),
)

SELECTION_SETS = {
    "core": CORE_SELECTIONS,
    "additional": ADDITIONAL_SELECTIONS,
    "all": CORE_SELECTIONS + ADDITIONAL_SELECTIONS,
}
# Backward-compatible import used by the original regression test and any
# external analysis code.
SELECTIONS = CORE_SELECTIONS


LOCATION_WORDS = re.compile(
    r"\b(left|right|upper|lower|middle|center|first|second|third|last|most|corner|"
    r"top|bottom|near|behind|front|between)\b",
    re.IGNORECASE,
)


def atomic_write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def read_binary_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        if image.size != size:
            raise ValueError(f"Mask geometry mismatch: {path}: {image.size} != {size}")
        return np.asarray(image.convert("L")) > 0


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=bool)
    second = np.asarray(second, dtype=bool)
    union = np.logical_or(first, second).sum()
    return float(np.logical_and(first, second).sum() / union) if union else 1.0


@torch.no_grad()
def sam2_point_mask(
    codec: SamtokCodec,
    image: Image.Image,
    positive_point: tuple[int, int],
    negative_point: tuple[int, int],
) -> tuple[np.ndarray, dict]:
    """Run the SAM2.1 image head already loaded inside the released codec."""

    array = codec.resize.apply_image(np.asarray(image.convert("RGB")))
    pixel = (
        torch.from_numpy(array)
        .permute(2, 0, 1)
        .contiguous()
        .to(codec.device, dtype=codec.vq.dtype)
    )
    wrapper = codec.vq.model
    states = wrapper.get_sam2_embeddings(wrapper.preprocess_image(pixel).unsqueeze(0))
    vision_features = states["current_vision_feats"]
    feature_sizes = states["feat_sizes"]
    high_resolution = [
        tensor.permute(1, 2, 0).view(tensor.size(1), tensor.size(2), *size)
        for tensor, size in zip(vision_features[:-1], feature_sizes[:-1])
    ]
    batch = vision_features[-1].size(1)
    channels = wrapper.hidden_dim
    grid_height, grid_width = feature_sizes[-1]
    backbone = (
        vision_features[-1] + wrapper.sam2_model.no_mem_embed
    ).permute(1, 2, 0).view(batch, channels, grid_height, grid_width)

    scale_x, scale_y = 1024 / image.width, 1024 / image.height
    points = torch.tensor(
        [[
            [positive_point[0] * scale_x, positive_point[1] * scale_y],
            [negative_point[0] * scale_x, negative_point[1] * scale_y],
        ]],
        dtype=torch.float32,
        device=codec.device,
    )
    labels = torch.tensor([[1, 0]], dtype=torch.int32, device=codec.device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = wrapper.sam2_model._forward_sam_heads(
            backbone_features=backbone,
            point_inputs={"point_coords": points, "point_labels": labels},
            mask_inputs=None,
            high_res_features=high_resolution,
            multimask_output=True,
        )
    score_tensor = outputs[2][0].float()
    selected_index = int(torch.argmax(score_tensor).item())
    # outputs[4] is already the one-channel best-IoU mask selected internally
    # by SAM2 when multimask_output=True; outputs[1] contains all candidates.
    high_resolution_mask = outputs[4][0, 0:1]
    mask = torch.nn.functional.interpolate(
        high_resolution_mask.unsqueeze(0),
        size=(image.height, image.width),
        mode="bilinear",
        align_corners=False,
    )[0, 0] > 0
    scores = score_tensor.cpu().tolist()
    return mask.cpu().numpy(), {
        "positive_point_xy": list(positive_point),
        "negative_point_xy": list(negative_point),
        "candidate_iou_scores": scores,
        "selected_candidate_index": selected_index,
        "selected_score": scores[selected_index],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark_root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--samtok_root", type=Path, default=DEFAULT_SAMTOK_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--selection_set",
        choices=tuple(SELECTION_SETS),
        default="core",
        help="Use the original three cases, the category-balanced extension, or both.",
    )
    args = parser.parse_args()
    selections = SELECTION_SETS[args.selection_set]
    if len({spec.benchmark_index for spec in selections}) != len(selections):
        raise ValueError(f"Duplicate benchmark index in selection set {args.selection_set}")
    if len({spec.semantic_category for spec in selections}) != len(selections):
        raise ValueError(f"Duplicate semantic category in selection set {args.selection_set}")

    benchmark_path = args.benchmark_root / "benchmark.jsonl"
    benchmark_rows = [json.loads(line) for line in benchmark_path.open(encoding="utf-8")]
    codec = SamtokCodec(
        args.samtok_root / "sam2.1_hiera_large.pt",
        args.samtok_root / "mask_tokenizer_256x2.pth",
        device=args.device,
    )

    prepared = []
    for case_index, spec in enumerate(selections):
        row = benchmark_rows[spec.benchmark_index]
        if not row["difficulty"].get("same_class_multi_instance"):
            raise ValueError(f"{row['id']} is not marked same_class_multi_instance")
        if len(row["regions"]) != 1 or row["edit_type"] not in {"remove", "replace"}:
            raise ValueError(f"{row['id']} is not a single-region remove/replace case")

        source_path = args.benchmark_root / row["source_image"]
        original_mask_path = args.benchmark_root / row["regions"][0]["mask"]
        with Image.open(source_path) as handle:
            source = handle.convert("RGB")
        original_mask = read_binary_mask(original_mask_path, source.size)
        alternate_mask, sam_audit = sam2_point_mask(
            codec,
            source,
            spec.alternate_point,
            tuple(row["regions"][0]["point"]),
        )
        alternate_area_fraction = float(alternate_mask.mean())
        raw_mask_iou = mask_iou(original_mask, alternate_mask)
        if not 0.001 <= alternate_area_fraction <= 0.35:
            raise ValueError(
                f"{row['id']}: implausible alternate mask area {alternate_area_fraction:.6f}"
            )
        if raw_mask_iou >= 0.20:
            raise ValueError(f"{row['id']}: original/alternate masks overlap too much: {raw_mask_iou}")

        spans = codec.encode_single_batch(
            [(source, original_mask), (source, alternate_mask)]
        )
        decoded = codec.decode_single_batch(
            [(source, spans[0]), (source, spans[1])]
        )
        if spans[0] == spans[1]:
            raise ValueError(f"{row['id']}: intervention did not change the mask tokens")
        original_decode_iou = mask_iou(original_mask, decoded[0])
        alternate_decode_iou = mask_iou(alternate_mask, decoded[1])
        decoded_pair_iou = mask_iou(decoded[0], decoded[1])
        if min(original_decode_iou, alternate_decode_iou) < 0.50:
            raise ValueError(
                f"{row['id']}: token decode fidelity is too low: "
                f"A={original_decode_iou:.6f}, B={alternate_decode_iou:.6f}"
            )
        if decoded_pair_iou >= 0.20:
            raise ValueError(
                f"{row['id']}: decoded original/alternate masks overlap too much: "
                f"{decoded_pair_iou:.6f}"
            )

        case_dir = args.output_root / "data" / f"case_{case_index:02d}_{row['id']}"
        case_dir.mkdir(parents=True, exist_ok=True)
        local_source = case_dir / "source.png"
        local_original = case_dir / "original_mask.png"
        local_alternate = case_dir / "alternate_mask.png"
        local_original_decoded = case_dir / "original_token_decode.png"
        local_alternate_decoded = case_dir / "alternate_token_decode.png"
        source.save(local_source)
        Image.fromarray(original_mask.astype(np.uint8) * 255).save(local_original)
        Image.fromarray(alternate_mask.astype(np.uint8) * 255).save(local_alternate)
        Image.fromarray(np.asarray(decoded[0], dtype=np.uint8) * 255).save(local_original_decoded)
        Image.fromarray(np.asarray(decoded[1], dtype=np.uint8) * 255).save(local_alternate_decoded)

        template = row["instruction"]["region_only"]
        if template.count("{region_1}") != 1:
            raise ValueError(f"{row['id']}: invalid region-only template: {template!r}")
        original_prompt = template.replace("{region_1}", spans[0])
        alternate_prompt = template.replace("{region_1}", spans[1])
        prompt_without_span = template.replace("{region_1}", "the selected region")
        if LOCATION_WORDS.search(prompt_without_span):
            raise ValueError(f"{row['id']}: location leakage in region-only prompt: {template!r}")

        prepared.append(
            {
                "case_index": case_index,
                "benchmark_index": spec.benchmark_index,
                "benchmark_id": row["id"],
                "source_dataset": row["source_dataset"],
                "edit_type": row["edit_type"],
                "semantic_category": spec.semantic_category,
                "source_image": str(local_source.resolve()),
                "benchmark_source_image": str(source_path.resolve()),
                "benchmark_original_mask": str(original_mask_path.resolve()),
                "original_mask": str(local_original.resolve()),
                "alternate_mask": str(local_alternate.resolve()),
                "original_token_decode": str(local_original_decoded.resolve()),
                "alternate_token_decode": str(local_alternate_decoded.resolve()),
                "location_free_template": template,
                "original_prompt": original_prompt,
                "alternate_prompt": alternate_prompt,
                "original_mask_span": spans[0],
                "alternate_mask_span": spans[1],
                "benchmark_location_instruction_provenance_only": row["instruction"][
                    "with_location_reference"
                ],
                "sam2_alternate_mask": sam_audit,
                "audit": {
                    "source_size": list(source.size),
                    "original_area_fraction": float(original_mask.mean()),
                    "alternate_area_fraction": alternate_area_fraction,
                    "original_vs_alternate_raw_iou": raw_mask_iou,
                    "original_raw_vs_token_decode_iou": original_decode_iou,
                    "alternate_raw_vs_token_decode_iou": alternate_decode_iou,
                    "original_vs_alternate_token_decode_iou": decoded_pair_iou,
                    "mask_tokens_changed": spans[0] != spans[1],
                    "location_words_absent": True,
                },
            }
        )
        print(
            f"[prepare] {case_index + 1}/{len(selections)} {row['id']} "
            f"raw_iou={raw_mask_iou:.4f} tokens_changed={spans[0] != spans[1]}",
            flush=True,
        )

    manifest = args.output_root / "data" / "interventions.jsonl"
    atomic_write_jsonl(manifest, prepared)
    report = {
        "status": "passed",
        "benchmark_manifest": str(benchmark_path.resolve()),
        "output_manifest": str(manifest.resolve()),
        "num_cases": len(prepared),
        "num_conditions": 2 * len(prepared),
        "selection_set": args.selection_set,
        "case_ids": [row["benchmark_id"] for row in prepared],
        "semantic_categories": [row["semantic_category"] for row in prepared],
        "semantic_category_counts": dict(
            sorted(Counter(row["semantic_category"] for row in prepared).items())
        ),
        "edit_type_counts": dict(sorted(Counter(row["edit_type"] for row in prepared).items())),
        "all_same_class_multi_instance": True,
        "all_location_free_prompts": all(
            row["audit"]["location_words_absent"] for row in prepared
        ),
        "all_mask_tokens_changed": all(
            row["audit"]["mask_tokens_changed"] for row in prepared
        ),
        "maximum_original_alternate_raw_iou": max(
            row["audit"]["original_vs_alternate_raw_iou"] for row in prepared
        ),
        "minimum_original_raw_token_decode_iou": min(
            row["audit"]["original_raw_vs_token_decode_iou"] for row in prepared
        ),
        "minimum_alternate_raw_token_decode_iou": min(
            row["audit"]["alternate_raw_vs_token_decode_iou"] for row in prepared
        ),
        "maximum_original_alternate_token_decode_iou": max(
            row["audit"]["original_vs_alternate_token_decode_iou"] for row in prepared
        ),
        "samtok_root": str(args.samtok_root.resolve()),
        "alternate_mask_method": (
            "SAM2.1 Hiera-L positive click on alternate same-class instance plus negative "
            "click at the released benchmark target point; every result is then encoded and "
            "decoded by the released gres-ft SAMTok codec"
        ),
    }
    atomic_write_json(args.output_root / "data" / "preparation_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
