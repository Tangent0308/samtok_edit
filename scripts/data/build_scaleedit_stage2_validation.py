#!/usr/bin/env python3
"""Build the curated ScaleEdit validation set used by Stage 2 evaluation.

The source release contains 200 embedded source/target pairs and audited masks.
This builder materializes a fixed, reviewable 32-case subset, encodes every raw
mask with the released SAMTok codec, verifies that the tokens decode, and emits
both standard ``edit_mt`` and ``edit_umt`` views of the same examples.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import os
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image, UnidentifiedImageError


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "DiffSynth-Studio"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from diffsynth.core.data.samtok_dataset import (  # noqa: E402
    SPAN_RE,
    make_labels,
    parse_and_canonicalize_mt_cot,
    sanitize_label,
    to_cot,
)
from samtok_codec import SamtokCodec  # noqa: E402


DEFAULT_SOURCE = Path(
    "/mnt/bn/strategy-mllm-train/user/tanyue/datasets/ScaleEdit-200-samples"
)
DEFAULT_EXPERIMENT_ROOT = Path(
    "/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/"
    "crispedit_refined/stage2_evaluation/scaleedit_precision_32"
)
DEFAULT_SAMTOK = Path(
    "/mnt/bn/strategy-mllm-train/user/tanyue/models/SAMTok/"
    "Qwen2.5-VL-7B-SAMTok-gres-ft"
)
DEFAULT_TRAINING_DATASETS = (
    (
        Path(
            "/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/"
            "crispedit_refined/stage1_full/data/crispedit_samtok/stage1.jsonl"
        ),
        Path(
            "/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/"
            "crispedit_refined/stage1_full/data/crispedit_samtok"
        ),
    ),
    (
        Path(
            "/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/"
            "crispedit_refined/stage2_full/data/crispedit_samtok/stage2.jsonl"
        ),
        Path(
            "/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/"
            "crispedit_refined/stage2_full/data/crispedit_samtok"
        ),
    ),
)


@dataclass(frozen=True)
class Selection:
    sample_id: str
    primary_category: str
    umt_replaced_text: str
    rationale: str


# The cases were reviewed as source / GT target / raw-mask overlay.  Stable
# sample_id values are used instead of parquet row positions.
SELECTIONS = (
    # Small objects (raw edited area <= 3%).
    Selection("3.4_count_change/count_change_0000.parquet#59", "small_object", "the side of the white charging device, aligned with the current four USB ports", "Two tiny aligned USB ports."),
    Selection("3.4_count_change/count_change_0000.parquet#434", "small_object", "the existing one on the court", "A small added ball beside an instance."),
    Selection("5.3_social_reasoning/social_reasoning_0000.parquet#1717", "small_object", "the sun icon on the button to the left of the dial", "A tiny illuminated control icon."),
    Selection("5.4_scientific_reasoning/scientific_reasoning_0000.parquet#4292", "small_object", "the upper arm", "Three very small marks on an arm."),
    Selection("4.4_building_surface_text_editing/building_surface_text_editing_0002.parquet#17120", "small_object", "the text '53'", "A small house-number replacement."),
    Selection("2.1_object_addition/object_addition_0009.parquet#29747", "small_object", "the second-story window on the beige building", "A small awning at an exact facade location."),
    Selection("2.3_object_replacement/object_replacement_0007.parquet#22365", "small_object", "the red heart", "Removal of a tiny chest emblem."),
    Selection("2.3_object_replacement/object_replacement_0007.parquet#15862", "small_object", "the flag", "Removal of a distant flag."),
    # Fine-grained local attributes, parts, symbols, or text.
    Selection("5.1_perceptual_reasoning/perceptual_reasoning_0000.parquet#4090", "fine_grained", "the overly sharp tip of the cat's tail", "Local shape repair at the tail tip."),
    Selection("5.1_perceptual_reasoning/perceptual_reasoning_0000.parquet#3567", "fine_grained", "the perforations on the handle of the pitcher", "Removal of small perforations while preserving the handle."),
    Selection("5.2_symbolic_reasoning/symbolic_reasoning_0000.parquet#1087", "fine_grained", "the black question mark", "Precise symbolic deletion in one cell."),
    Selection("4.1_movie_poster_text_editing/movie_poster_text_editing_0007.parquet#9177", "fine_grained", "the text 'HERCEGNO'", "Localized poster-title text replacement."),
    Selection("4.4_building_surface_text_editing/building_surface_text_editing_0002.parquet#26219", "fine_grained", "the text 'betway.ug'", "Localized advertising-board text replacement."),
    Selection("3.2_material_change/material_change_0002.parquet#11134", "fine_grained", "the straw hat's material", "Material change confined to one object."),
    Selection("2.3_object_replacement/object_replacement_0005.parquet#16243", "fine_grained", "the raised fist", "Finger-level hand gesture replacement."),
    Selection("2.3_object_replacement/object_replacement_0005.parquet#22042", "fine_grained", "the woman's smile", "Localized facial-expression edit."),
    # Multiple source/target instances or disjoint edit regions.
    Selection("3.4_count_change/count_change_0000.parquet#385", "multi_instance", "two geese", "Two added animal instances."),
    Selection("3.4_count_change/count_change_0000.parquet#139", "multi_instance", "two more identical mobile phones", "Two added repeated objects with spacing constraints."),
    Selection("3.4_count_change/count_change_0000.parquet#182", "multi_instance", "two oranges", "Two additions on opposite sides of an existing object."),
    Selection("5.1_perceptual_reasoning/perceptual_reasoning_0000.parquet#3830", "multi_instance", "the holes from the spout and handle of the teapot", "Two disjoint part regions on one object."),
    Selection("5.3_social_reasoning/social_reasoning_0000.parquet#1844", "multi_instance", "the amount of sand in the top half of the hourglass", "Coordinated change across both hourglass bulbs."),
    Selection("2.4_action_editing/action_editing_0000.parquet#31622", "multi_instance", "each pair of shoes", "Repeated object motion on three shelves."),
    Selection("2.3_object_replacement/object_replacement_0007.parquet#26138", "multi_instance", "the purple flowers", "Color change across several flower clusters."),
    Selection("2.3_object_replacement/object_replacement_0007.parquet#38105", "multi_instance", "the red lanterns hanging from the temple", "Color change across many small lanterns."),
    # Exact structural, semantic, and spatial edits.
    Selection("3.3_visual_beautification/visual_beautification_0000.parquet#20081", "precise_edit", "the woman sitting on the bench in the foreground", "Edit one small foreground person without changing bystanders."),
    Selection("2.3_object_replacement/object_replacement_0007.parquet#11853", "precise_edit", "the bottom drawer", "Close one specified drawer and preserve the others."),
    Selection("2.3_object_replacement/object_replacement_0007.parquet#25856", "precise_edit", "the thumbs-up gesture", "Replace only the specified gesture."),
    Selection("2.3_object_replacement/object_replacement_0007.parquet#13908", "precise_edit", "the neon \"1st\" sign", "Replace a distant neon sign at the same location."),
    Selection("2.3_object_replacement/object_replacement_0007.parquet#19851", "precise_edit", "the roof of the building on the left", "Change one roof while preserving neighboring roofs."),
    Selection("2.3_object_replacement/object_replacement_0007.parquet#29514", "precise_edit", "the metal handrail", "Material replacement with shape preservation."),
    Selection("2.3_object_replacement/object_replacement_0007.parquet#14845", "precise_edit", "the cluster of seat memory and tailgate control buttons", "Replace one compact control cluster."),
    Selection("2.3_object_replacement/object_replacement_0002.parquet#36987", "precise_edit", "the two people on the mountain path", "Replace two distant people while preserving the landscape."),
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_bytes(path: Path, value: bytes, *, replace_derived: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if path.read_bytes() == value:
            return
        if not replace_derived:
            raise ValueError(f"Existing artifact differs from source bytes: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(value)
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def image_extension(data: bytes) -> str:
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
            name = (image.format or "png").lower()
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError("Embedded source/target bytes are not a valid image") from error
    return "jpg" if name == "jpeg" else name


def load_source_rows(source_dir: Path) -> tuple[dict[str, dict], list[dict]]:
    shards = sorted(source_dir.glob("part-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No ScaleEdit parquet shards under {source_dir}")
    rows, by_id = [], {}
    for shard in shards:
        for shard_index, row in enumerate(pq.read_table(shard).to_pylist()):
            row = dict(row)
            row["_source_shard"] = shard.name
            row["_source_shard_row"] = shard_index
            row["_source_global_index"] = len(rows)
            sample_id = str(row["sample_id"])
            if sample_id in by_id:
                raise ValueError(f"Duplicate ScaleEdit sample_id: {sample_id}")
            by_id[sample_id] = row
            rows.append(row)
    if len(rows) != 200:
        raise ValueError(f"Expected the audited 200-row release, found {len(rows)}")
    return by_id, rows


def mask_label(row: dict, selection: Selection) -> str:
    instances = [
        item
        for item in (row.get("instance_masks") or [])
        if (item or {}).get("role") == "edit_region"
    ]
    prompt = str(row["final_instruction"])
    addition = prompt.lower().startswith(("add ", "draw ", "fill "))
    preferred_side = "target" if addition else "source"
    refs = [
        sanitize_label(str(item.get("ref") or ""))
        for item in instances
        if item.get("grounding_image") == preferred_side
    ]
    if not refs:
        refs = [sanitize_label(str(item.get("ref") or "")) for item in instances]
    unique = []
    for ref in refs:
        if ref and ref.casefold() not in {item.casefold() for item in unique}:
            unique.append(ref)
    # A union of multiple verbose grounding refs can otherwise be cut at the
    # canonical 80-character limit mid-phrase.  The curated reference is the
    # shorter human-reviewed description of that same edit region.
    label = (
        selection.umt_replaced_text
        if len(unique) > 1
        else (unique[0] if unique else selection.umt_replaced_text or prompt)
    )
    return make_labels(label, 1)[0]


def logical_instance_count(row: dict) -> int:
    counts = Counter(
        str(item.get("grounding_image") or "unknown")
        for item in (row.get("instance_masks") or [])
        if (item or {}).get("role") == "edit_region"
    )
    return max(counts.values(), default=0)


def selection_tags(selection: Selection, row: dict) -> list[str]:
    tags = {selection.primary_category, "precise_edit"}
    if float(row["area_frac"]) <= 0.03:
        tags.add("small_object")
    if logical_instance_count(row) >= 2:
        tags.add("multi_instance")
    if float(row["area_frac"]) <= 0.08:
        tags.add("fine_grained")
    order = ["small_object", "fine_grained", "multi_instance", "precise_edit"]
    return [tag for tag in order if tag in tags]


def parse_training_dataset(value: str) -> tuple[Path, Path]:
    metadata, separator, base = value.partition("::")
    if not separator:
        raise argparse.ArgumentTypeError("training datasets use METADATA_JSONL::DATASET_BASE")
    return Path(metadata), Path(base)


def audit_exact_training_overlap(
    validation_hashes: set[str],
    validation_sizes: set[int],
    training_datasets: list[tuple[Path, Path]],
    workers: int,
) -> dict:
    references: set[Path] = set()
    dataset_rows = []
    for metadata, base in training_datasets:
        if not metadata.is_file():
            raise FileNotFoundError(f"Training metadata is missing: {metadata}")
        count = 0
        with metadata.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                count += 1
                row = json.loads(line)
                for field in ("edit_image", "image"):
                    value = row.get(field)
                    if not value:
                        continue
                    path = Path(value)
                    references.add(path if path.is_absolute() else base / path)
        dataset_rows.append({"metadata": str(metadata.resolve()), "rows": count})

    def inspect(path: Path):
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            return "missing", str(path), None
        if size not in validation_sizes:
            return "different_size", str(path), None
        return "hashed", str(path), sha256_file(path)

    missing, size_candidates, collisions = [], 0, []
    paths = sorted(references)
    chunk_size = 20000
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for start in range(0, len(paths), chunk_size):
            chunk = paths[start : start + chunk_size]
            for status, path, digest in executor.map(inspect, chunk):
                if status == "missing":
                    missing.append(path)
                elif status == "hashed":
                    size_candidates += 1
                    if digest in validation_hashes:
                        collisions.append(
                            {"path": str(Path(path).resolve()), "sha256": digest}
                        )
            print(
                f"[disjointness] checked={min(start + len(chunk), len(paths))}/"
                f"{len(paths)} size_candidates={size_candidates} "
                f"collisions={len(collisions)} workers={workers}",
                flush=True,
            )
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} training image references are missing; first={missing[0]}"
        )
    if collisions:
        raise ValueError(
            f"Validation/training exact image-content overlap found: {collisions[:3]}"
        )
    return {
        "method": "parallel stat plus SHA256 over every size-compatible unique training image",
        "workers": workers,
        "training_datasets": dataset_rows,
        "training_unique_image_references": len(references),
        "size_compatible_images_hashed": size_candidates,
        "validation_unique_image_hashes": len(validation_hashes),
        "exact_hash_collisions": collisions,
        "passed": True,
    }


def build(args) -> dict:
    by_id, all_rows = load_source_rows(args.source_dir)
    if len(SELECTIONS) != 32 or len({item.sample_id for item in SELECTIONS}) != 32:
        raise RuntimeError("The curated selection must contain exactly 32 unique cases")
    missing = [item.sample_id for item in SELECTIONS if item.sample_id not in by_id]
    if missing:
        raise ValueError(f"Curated ScaleEdit ids are absent from the release: {missing}")

    output_root = args.output_root.resolve()
    selected = []
    validation_hashes, validation_sizes = set(), set()
    for eval_index, selection in enumerate(SELECTIONS):
        row = by_id[selection.sample_id]
        if row.get("qc_flag") != "OK" or row.get("grounding_status") not in {
            "OK",
            "PROTECT_FOREGROUND",
        }:
            raise ValueError(f"Selected row is not an audited local mask: {selection.sample_id}")
        if row.get("mask_source") == "full_image":
            raise ValueError(f"Full-image mask is forbidden here: {selection.sample_id}")
        prompt = str(row["final_instruction"]).strip()
        if not prompt or not prompt.isascii():
            raise ValueError(f"Prompt must be non-empty ASCII English: {selection.sample_id}")
        if prompt.count(selection.umt_replaced_text) != 1:
            raise ValueError(
                f"UMT reference is not unique in {selection.sample_id}: "
                f"{selection.umt_replaced_text!r}"
            )

        source_bytes = bytes(row["source_image"])
        target_bytes = bytes(row["edited_image"])
        mask_bytes = bytes(row["mask_png"])
        source_ext, target_ext = image_extension(source_bytes), image_extension(target_bytes)
        stem = f"{eval_index:04d}"
        source_rel = Path("images") / f"{stem}_source.{source_ext}"
        target_rel = Path("images") / f"{stem}_target.{target_ext}"
        mask_rel = Path("masks") / f"{stem}_raw.png"
        atomic_bytes(output_root / source_rel, source_bytes)
        atomic_bytes(output_root / target_rel, target_bytes)
        atomic_bytes(output_root / mask_rel, mask_bytes)
        for value in (source_bytes, target_bytes):
            validation_hashes.add(sha256_bytes(value))
            validation_sizes.add(len(value))

        with Image.open(io.BytesIO(source_bytes)) as image:
            source = image.convert("RGB")
        with Image.open(io.BytesIO(target_bytes)) as image:
            target_size = image.size
        with Image.open(io.BytesIO(mask_bytes)) as image:
            raw_mask = np.asarray(image.convert("L")) > 0
        if raw_mask.shape != (source.height, source.width) or not raw_mask.any():
            raise ValueError(f"Invalid mask geometry/content: {selection.sample_id}")
        if source.size != (int(row["source_image_width"]), int(row["source_image_height"])):
            raise ValueError(f"Source size metadata mismatch: {selection.sample_id}")
        if target_size != (int(row["edited_image_width"]), int(row["edited_image_height"])):
            raise ValueError(f"Target size metadata mismatch: {selection.sample_id}")

        selected.append(
            {
                "selection": selection,
                "source_row": row,
                "source": source,
                "raw_mask": raw_mask,
                "source_rel": source_rel,
                "target_rel": target_rel,
                "mask_rel": mask_rel,
                "label": mask_label(row, selection),
                "tags": selection_tags(selection, row),
            }
        )

    codec = SamtokCodec(
        args.sam2_ckpt,
        args.mask_tokenizer_ckpt,
        device=args.device,
        dtype=torch.float32,
    )
    spans = []
    for start in range(0, len(selected), args.codec_batch_size):
        batch = selected[start : start + args.codec_batch_size]
        spans.extend(
            codec.encode_single_batch((item["source"], item["raw_mask"]) for item in batch)
        )
        print(f"[codec] encoded={min(start + len(batch), len(selected))}/{len(selected)}", flush=True)

    decoded_masks = []
    for start in range(0, len(selected), args.codec_batch_size):
        batch = selected[start : start + args.codec_batch_size]
        batch_spans = spans[start : start + len(batch)]
        decoded_masks.extend(
            codec.decode_single_batch(
                (item["source"], span) for item, span in zip(batch, batch_spans)
            )
        )
        print(
            f"[codec] decoded={min(start + len(batch), len(selected))}/{len(selected)}",
            flush=True,
        )

    eval_rows, mt_rows, umt_rows = [], [], []
    decoded_nonempty = 0
    for eval_index, (item, span, decoded_mask) in enumerate(
        zip(selected, spans, decoded_masks)
    ):
        selection, row = item["selection"], item["source_row"]
        cot = to_cot([(span, item["label"])])
        canonical, layer = parse_and_canonicalize_mt_cot(cot, return_layer=True)
        if canonical != cot or layer != "strict" or len(SPAN_RE.findall(span)) != 1:
            raise ValueError(f"Codec emitted a non-canonical span for {selection.sample_id}")
        if decoded_mask.shape != item["raw_mask"].shape or not decoded_mask.any():
            raise ValueError(f"GT mask token decode failed for {selection.sample_id}")
        decoded_nonempty += 1
        decoded_rel = Path("masks") / f"{eval_index:04d}_gt_token_decode.png"
        buffer = io.BytesIO()
        Image.fromarray(np.asarray(decoded_mask, dtype=np.uint8) * 255).save(buffer, format="PNG")
        atomic_bytes(
            output_root / decoded_rel,
            buffer.getvalue(),
            replace_derived=True,
        )

        prompt = str(row["final_instruction"]).strip()
        umt_prompt = prompt.replace(selection.umt_replaced_text, span, 1)
        if len(SPAN_RE.findall(umt_prompt)) != 1 or umt_prompt.count("<|mt_start|>") != 1:
            raise ValueError(f"Invalid edit_umt prompt for {selection.sample_id}")
        provenance = {
            "source_dataset": str(args.source_dir.resolve()),
            "sample_id": selection.sample_id,
            "source_shard": row["_source_shard"],
            "source_shard_row": row["_source_shard_row"],
            "source_global_index": row["_source_global_index"],
            "edit_task": row["edit_task"],
            "final_task": row["final_task"],
            "mask_source": row["mask_source"],
            "qc_flag": row["qc_flag"],
            "grounding_status": row["grounding_status"],
            "area_frac": float(row["area_frac"]),
            "logical_instance_count": logical_instance_count(row),
            "original_prompt": prompt,
            "umt_replaced_text": selection.umt_replaced_text,
            "umt_rewrite_method": "curated_exact_reference",
        }
        common = {
            "image": item["target_rel"].as_posix(),
            "edit_image": item["source_rel"].as_posix(),
            "prompt": prompt,
            "provenance": provenance,
        }
        mt_rows.append({**common, "sample_type": "edit_mt", "mt_cot": cot})
        umt_rows.append(
            {
                **common,
                "prompt": umt_prompt,
                "sample_type": "edit_umt",
                "provenance": provenance,
            }
        )
        eval_rows.append(
            {
                "eval_index": eval_index,
                "eval_id": f"scaleedit_{eval_index:04d}",
                **common,
                "sample_type": "edit_mt",
                "mt_cot": cot,
                "gt_mask_span": span,
                "edit_umt_prompt": umt_prompt,
                "gt_mask": item["mask_rel"].as_posix(),
                "gt_decoded_mask": decoded_rel.as_posix(),
                "mask_label": item["label"],
                "primary_category": selection.primary_category,
                "selection_tags": item["tags"],
                "selection_rationale": selection.rationale,
            }
        )

    atomic_jsonl(output_root / "validation.jsonl", eval_rows)
    atomic_jsonl(output_root / "validation_edit_mt.jsonl", mt_rows)
    atomic_jsonl(output_root / "validation_edit_umt.jsonl", umt_rows)

    disjointness = audit_exact_training_overlap(
        validation_hashes,
        validation_sizes,
        list(args.training_dataset),
        args.disjointness_workers,
    )
    category_counts = Counter(row["primary_category"] for row in eval_rows)
    tag_counts = Counter(tag for row in eval_rows for tag in row["selection_tags"])
    areas = [float(row["provenance"]["area_frac"]) for row in eval_rows]
    report = {
        "status": "complete",
        "source_dataset": str(args.source_dir.resolve()),
        "source_release_rows": len(all_rows),
        "selection_method": "fixed manual review of source, GT edit, and raw-mask overlay",
        "rows": len(eval_rows),
        "primary_category_counts": dict(sorted(category_counts.items())),
        "selection_tag_counts": dict(sorted(tag_counts.items())),
        "final_task_counts": dict(sorted(Counter(row["provenance"]["final_task"] for row in eval_rows).items())),
        "mask_source_counts": dict(sorted(Counter(row["provenance"]["mask_source"] for row in eval_rows).items())),
        "area_frac": {
            "min": min(areas),
            "median": statistics.median(areas),
            "mean": statistics.mean(areas),
            "max": max(areas),
        },
        "english_ascii_prompts": sum(row["prompt"].isascii() for row in eval_rows),
        "canonical_nonempty_gt_cot": len(eval_rows),
        "valid_single_span_edit_umt": len(umt_rows),
        "nonempty_gt_token_decodes": decoded_nonempty,
        "unique_scaleedit_sample_ids": len({row["provenance"]["sample_id"] for row in eval_rows}),
        "disjointness": disjointness,
        "artifacts": {
            name: {
                "path": str((output_root / name).resolve()),
                "sha256": sha256_file(output_root / name),
            }
            for name in (
                "validation.jsonl",
                "validation_edit_mt.jsonl",
                "validation_edit_umt.jsonl",
            )
        },
    }
    report_path = args.report or output_root.parent.parent / "reports" / "data_build_report.json"
    atomic_json(report_path, report)
    return {**report, "report_path": str(report_path.resolve()), "output_root": str(output_root)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--output_root", type=Path, default=DEFAULT_EXPERIMENT_ROOT / "data/scaleedit_samtok"
    )
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--sam2_ckpt", type=Path, default=DEFAULT_SAMTOK / "sam2.1_hiera_large.pt")
    parser.add_argument(
        "--mask_tokenizer_ckpt",
        type=Path,
        default=DEFAULT_SAMTOK / "mask_tokenizer_256x2.pth",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--codec_batch_size", type=int, default=8)
    parser.add_argument("--disjointness_workers", type=int, default=32)
    parser.add_argument(
        "--training_dataset",
        type=parse_training_dataset,
        action="append",
        default=None,
        metavar="METADATA_JSONL::DATASET_BASE",
        help="Training split used for exact image-content disjointness; repeatable.",
    )
    args = parser.parse_args()
    if args.codec_batch_size < 1 or args.disjointness_workers < 1:
        parser.error("--codec_batch_size and --disjointness_workers must be positive")
    if args.training_dataset is None:
        args.training_dataset = list(DEFAULT_TRAINING_DATASETS)
    report = build(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
