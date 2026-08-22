#!/usr/bin/env python3
"""Convert released GRES SAMTok conversations into edit-template NTP rows."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "DiffSynth-Studio"))

from diffsynth.core.data.samtok_dataset import (  # noqa: E402
    FALLBACK_LABEL,
    parse_and_canonicalize_mt_cot,
    sanitize_label,
    to_cot,
)


DEFAULT_GRES_JSON = Path(
    "/mnt/bn/strategy-mllm-train/user/tanyue/datasets/"
    "SAMTok_Training_Data/mask_generation_gres209k.json"
)
DEFAULT_IMAGE_ROOT = Path(
    "/mnt/bn/strategy-mllm-train/intern/common_datasets/"
    "Sa2VA-Training/osprey-724k"
)

SEGMENTATION_QUESTIONS = [
    "Can you segment the {class_name} in this image?",
    "Please segment {class_name} in this image.",
    "What is {class_name} in this image? Please respond with segmentation mask.",
    "What is {class_name} in this image? Please output segmentation mask.",
    "Can you segment the {class_name} in this image",
    "Please segment {class_name} in this image",
    "What is {class_name} in this image? Please respond with segmentation mask",
    "What is {class_name} in this image? Please output segmentation mask",
    "Could you provide a segmentation mask for the {class_name} in this image?",
    "Please identify and segment the {class_name} in this image.",
    "Where is the {class_name} in this picture? Please respond with a segmentation mask.",
    "Can you highlight the {class_name} in this image with a segmentation mask?",
    "Could you provide a segmentation mask for the {class_name} in this image",
    "Please identify and segment the {class_name} in this image",
    "Where is the {class_name} in this picture? Please respond with a segmentation mask",
    "Can you highlight the {class_name} in this image with a segmentation mask",
]
EDIT_VERB_TEMPLATES = [
    "change {expr} to blue",
    "remove {expr}",
    "delete {expr}",
    "replace {expr} with a corgi",
    "make {expr} glow",
    "recolor {expr} to green",
]
GLOBAL_TEMPLATES = [
    "apply a vintage film look to the whole image",
    "turn the entire image into a watercolor painting",
    "make the whole scene look like it was photographed at dusk",
]

_QUESTION_PATTERNS = []
for _template in SEGMENTATION_QUESTIONS:
    pattern = re.escape(_template).replace(
        re.escape("{class_name}"), r"(?P<expression>.+?)"
    )
    _QUESTION_PATTERNS.append(re.compile(r"^" + pattern + r"$", re.S))


def expression_from_question(question: str) -> str:
    question = question.removeprefix("<image>\n").strip()
    for pattern in _QUESTION_PATTERNS:
        match = pattern.fullmatch(question)
        if match:
            return sanitize_label(match.group("expression"))
    raise ValueError(f"Unrecognized released GRES question template: {question!r}")


def expression_from_cot(cot: str) -> str | None:
    match = re.search(r"```json\s*(.*?)\s*```", cot, re.S | re.I)
    if not match:
        return None
    try:
        items = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        label = sanitize_label(item.get("label", ""))
        if label.startswith("one of the "):
            label = label[len("one of the ") :]
        if label and label != FALLBACK_LABEL:
            return label
    return None


def build_rows(
    source_rows,
    rng: random.Random,
    global_ratio: float,
    image_root: Path | None,
    check_images: bool,
    absolute_image_paths: bool,
):
    output = []
    layers = Counter()
    for row_id, source in enumerate(source_rows):
        conversations = source.get("conversations")
        if not isinstance(conversations, list) or len(conversations) < 2:
            raise ValueError(f"Row {row_id} has malformed conversations")
        question = conversations[0].get("value", "")
        raw_cot = conversations[1].get("value", "")
        mt_cot, layer = parse_and_canonicalize_mt_cot(raw_cot, return_layer=True)
        if mt_cot is None:
            raise ValueError(f"Row {row_id} has no recoverable SAMTok target")
        layers[layer] += 1
        expression = expression_from_cot(mt_cot) or expression_from_question(question)
        image = source["image"]
        resolved_image = image_root / image if image_root is not None else Path(image)
        if check_images and not resolved_image.is_file():
            raise FileNotFoundError(resolved_image)
        metadata_image = str(resolved_image.resolve()) if absolute_image_paths else image
        output.append(
            {
                "edit_image": metadata_image,
                "prompt": rng.choice(EDIT_VERB_TEMPLATES).format(expr=expression),
                "mt_cot": mt_cot,
                "sample_type": "edit_ntp",
            }
        )

    if not 0 <= global_ratio < 1:
        raise ValueError("global_ratio must be in [0, 1)")
    global_count = (
        round(len(output) * global_ratio / (1 - global_ratio)) if output else 0
    )
    base_rows = output[:]
    for _ in range(global_count):
        source = rng.choice(base_rows)
        output.append(
            {
                "edit_image": source["edit_image"],
                "prompt": rng.choice(GLOBAL_TEMPLATES),
                "mt_cot": to_cot([]),
                "sample_type": "edit_ntp",
            }
        )
    rng.shuffle(output)
    return output, layers, global_count


def write_jsonl_atomic(rows, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, output_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_json", type=Path, default=DEFAULT_GRES_JSON)
    parser.add_argument("--image_root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument(
        "--output_jsonl",
        type=Path,
        default=(
            _REPO_ROOT
            / "data"
            / "crispedit_samtok"
            / "edit_ntp_gres.jsonl"
        ),
    )
    parser.add_argument("--global_ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--check_images", action="store_true")
    parser.add_argument(
        "--relative_image_paths",
        action="store_true",
        help="Keep GRES paths relative to --image_root (absolute paths are the default)",
    )
    args = parser.parse_args()

    with args.input_json.open(encoding="utf-8") as handle:
        source_rows = json.load(handle)
    if args.max_rows is not None:
        source_rows = source_rows[: args.max_rows]
    rows, layers, global_count = build_rows(
        source_rows,
        random.Random(args.seed),
        args.global_ratio,
        args.image_root,
        args.check_images,
        not args.relative_image_paths,
    )
    write_jsonl_atomic(rows, args.output_jsonl)
    print(
        json.dumps(
            {
                "output": str(args.output_jsonl),
                "gres_rows": len(source_rows),
                "global_rows": global_count,
                "total_rows": len(rows),
                "parse_layers": dict(layers),
                "dataset_base_path": str(args.image_root),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
