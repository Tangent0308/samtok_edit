#!/usr/bin/env python3
"""Convert non-empty released GRES SAMTok targets into edit-template NTP rows."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

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
    image_root: Path | None,
    check_images: bool,
    absolute_image_paths: bool,
    source_dataset: Path = DEFAULT_GRES_JSON,
):
    output = []
    layers = Counter()
    for row_id, source in enumerate(source_rows):
        conversations = source.get("conversations")
        if not isinstance(conversations, list) or len(conversations) < 2:
            raise ValueError(f"Row {row_id} has malformed conversations")
        question = conversations[0].get("value", "")
        raw_cot = conversations[1].get("value", "")
        mt_cot = source.get("_samtok_canonical_cot")
        layer = source.get("_samtok_parse_layer")
        if mt_cot is None:
            mt_cot, layer = parse_and_canonicalize_mt_cot(
                raw_cot, return_layer=True
            )
        if mt_cot is None:
            raise ValueError(f"Row {row_id} has no recoverable SAMTok target")
        if layer == "empty" or mt_cot == to_cot([]):
            raise ValueError(
                f"Row {row_id} has an empty SAMTok target; refined edit_ntp forbids []"
            )
        layers[layer] += 1
        expression = source.get("_samtok_expression") or expression_from_cot(
            mt_cot
        ) or expression_from_question(question)
        image = source["image"]
        resolved_image = image_root / image if image_root is not None else Path(image)
        if check_images and not resolved_image.is_file():
            raise FileNotFoundError(resolved_image)
        # Both configured roots are already absolute.  Path.resolve() performs an
        # additional remote-filesystem lookup for every row and is unnecessary here.
        metadata_image = str(resolved_image.absolute()) if absolute_image_paths else image
        output.append(
            {
                "edit_image": metadata_image,
                "prompt": rng.choice(EDIT_VERB_TEMPLATES).format(expr=expression),
                "mt_cot": mt_cot,
                "sample_type": "edit_ntp",
                "provenance": {
                    "source_dataset": str(source_dataset),
                    "source_row_idx": int(source.get("_samtok_source_row_idx", row_id)),
                    "source_image": image,
                },
            }
        )
    rng.shuffle(output)
    return output, layers


def validate_source_images(
    source_rows: list[dict], image_root: Path | None, io_workers: int
) -> None:
    """Check remote image paths concurrently before metadata materialization."""

    if io_workers < 1:
        raise ValueError("io_workers must be positive")
    paths = [
        image_root / row["image"] if image_root is not None else Path(row["image"])
        for row in source_rows
    ]

    def missing(path: Path) -> Path | None:
        return None if path.is_file() else path

    with ThreadPoolExecutor(max_workers=io_workers) as executor:
        results = executor.map(missing, paths)
        for result in tqdm(
            results,
            total=len(paths),
            desc="Checking selected GRES images",
        ):
            if result is not None:
                raise FileNotFoundError(result)


def select_source_rows(
    source_rows: list[dict],
    max_rows: int | None,
    sample_rows: int | None,
    seed: int,
) -> list[dict]:
    if max_rows is not None and sample_rows is not None:
        raise ValueError("max_rows and sample_rows cannot be combined")
    if sample_rows is not None:
        if sample_rows < 1:
            raise ValueError("sample_rows must be positive")
        if sample_rows > len(source_rows):
            raise ValueError(
                f"Requested {sample_rows} rows from only {len(source_rows)} source rows"
            )
        return random.Random(seed).sample(source_rows, sample_rows)
    if max_rows is not None:
        return source_rows[:max_rows]
    return source_rows[:]


def filter_source_rows(source_rows: list[dict], ascii_only: bool):
    """Remove empty/invalid/non-English targets before exact-count sampling."""

    eligible = []
    stats = Counter()
    for row_id, source in enumerate(source_rows):
        conversations = source.get("conversations")
        if not isinstance(conversations, list) or len(conversations) < 2:
            stats["malformed_drop"] += 1
            continue
        raw_cot = conversations[1].get("value", "")
        mt_cot, layer = parse_and_canonicalize_mt_cot(raw_cot, return_layer=True)
        if mt_cot is None:
            stats["invalid_cot_drop"] += 1
            continue
        if layer == "empty" or mt_cot == to_cot([]):
            stats["empty_cot_drop"] += 1
            continue
        try:
            expression = expression_from_cot(mt_cot) or expression_from_question(
                conversations[0].get("value", "")
            )
        except ValueError:
            stats["expression_drop"] += 1
            continue
        if ascii_only and (not mt_cot.isascii() or not expression.isascii()):
            stats["non_ascii_drop"] += 1
            continue
        eligible.append(
            {
                **source,
                "_samtok_source_row_idx": row_id,
                "_samtok_canonical_cot": mt_cot,
                "_samtok_parse_layer": layer,
                "_samtok_expression": expression,
            }
        )
        stats["eligible"] += 1
        stats[f"parse_layer:{layer}"] += 1
    return eligible, stats


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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument(
        "--sample_rows",
        type=int,
        default=None,
        help="Randomly sample this many released GRES rows with --seed",
    )
    parser.add_argument(
        "--ascii_only",
        action="store_true",
        help="Require every generated prompt and canonical CoT to contain ASCII text only",
    )
    parser.add_argument("--check_images", action="store_true")
    parser.add_argument(
        "--io_workers",
        type=int,
        default=32,
        help="Threads used by --check_images for remote filesystem checks",
    )
    parser.add_argument(
        "--relative_image_paths",
        action="store_true",
        help="Keep GRES paths relative to --image_root (absolute paths are the default)",
    )
    args = parser.parse_args()

    with args.input_json.open(encoding="utf-8") as handle:
        source_rows = json.load(handle)
    source_rows, filter_stats = filter_source_rows(source_rows, args.ascii_only)
    source_rows = select_source_rows(
        source_rows, args.max_rows, args.sample_rows, args.seed
    )
    if args.check_images:
        validate_source_images(source_rows, args.image_root, args.io_workers)
    rows, layers = build_rows(
        source_rows,
        random.Random(args.seed),
        args.image_root,
        False,
        not args.relative_image_paths,
        args.input_json,
    )
    if args.ascii_only:
        for row_id, row in enumerate(rows):
            if not row["prompt"].isascii() or not row["mt_cot"].isascii():
                raise ValueError(f"Generated row {row_id} contains non-ASCII text")
    write_jsonl_atomic(rows, args.output_jsonl)
    print(
        json.dumps(
            {
                "output": str(args.output_jsonl),
                "gres_rows": len(source_rows),
                "selection_mode": "global_random" if args.sample_rows is not None else "prefix",
                "seed": args.seed,
                "ascii_only": args.ascii_only,
                "empty_rows": 0,
                "total_rows": len(rows),
                "source_filter_stats": dict(filter_stats),
                "parse_layers": dict(layers),
                "dataset_base_path": str(args.image_root),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
