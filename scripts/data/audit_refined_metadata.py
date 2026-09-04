#!/usr/bin/env python3
"""Audit refined Stage-1 metadata against every authoritative source row."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "DiffSynth-Studio"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from diffsynth.core.data.samtok_dataset import (  # noqa: E402
    SPAN_RE,
    make_labels,
    parse_and_canonicalize_mt_cot,
    to_cot,
)
from build_edit_mt_metadata import (  # noqa: E402
    DEFAULT_CRISPEDIT,
    DEFAULT_MASKS,
    DEFAULT_SAMTOK_DIR,
    _decode_image,
    _decode_mask,
    _image_bytes,
    _label_for_row,
    build_umt_prompt,
)
from build_edit_metadata import DEFAULT_FILTER_MANIFEST  # noqa: E402
from build_edit_ntp_metadata import (  # noqa: E402
    DEFAULT_GRES_JSON,
    DEFAULT_IMAGE_ROOT,
    EDIT_VERB_TEMPLATES,
    expression_from_cot,
    expression_from_question,
)
from samtok_codec import SamtokCodec  # noqa: E402


def _read_jsonl(path: Path):
    digest = hashlib.sha256()
    rows = []
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            if line.strip():
                rows.append(json.loads(line))
    return rows, digest.hexdigest()


def _resolve(path: str, base_path: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else base_path / path


def _cot_span(cot: str) -> str:
    canonical, layer = parse_and_canonicalize_mt_cot(cot, return_layer=True)
    if canonical != cot or layer == "empty":
        raise ValueError("Expected a non-empty canonical CoT")
    matches = list(SPAN_RE.finditer(cot))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one CoT mask span, got {len(matches)}")
    return matches[0].group(0)


def _umt_span(prompt: str) -> str:
    matches = list(SPAN_RE.finditer(prompt))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one UMT prompt span, got {len(matches)}")
    return matches[0].group(0)


def _write_json_atomic(payload: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata_jsonl", type=Path, required=True)
    parser.add_argument("--base_path", type=Path, required=True)
    parser.add_argument("--crispedit_dir", type=Path, default=DEFAULT_CRISPEDIT)
    parser.add_argument("--mask_dir", type=Path, default=DEFAULT_MASKS)
    parser.add_argument(
        "--filter_manifest_dir", type=Path, default=DEFAULT_FILTER_MANIFEST
    )
    parser.add_argument("--gres_json", type=Path, default=DEFAULT_GRES_JSON)
    parser.add_argument("--gres_image_root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument(
        "--image_byte_sample",
        type=int,
        default=0,
        help="CrispEdit identities whose source/target bytes are checked; -1 means all",
    )
    parser.add_argument(
        "--codec_sample",
        type=int,
        default=0,
        help="Random edit_mt identities whose mask spans are re-encoded and compared",
    )
    parser.add_argument(
        "--sam2_ckpt", type=Path, default=DEFAULT_SAMTOK_DIR / "sam2.1_hiera_large.pt"
    )
    parser.add_argument(
        "--mask_tokenizer_ckpt",
        type=Path,
        default=DEFAULT_SAMTOK_DIR / "mask_tokenizer_256x2.pth",
    )
    parser.add_argument("--codec_device", default="cuda:0")
    parser.add_argument("--codec_batch_size", type=int, default=8)
    parser.add_argument(
        "--io_workers",
        type=int,
        default=8,
        help="Concurrent parquet shards used by the exact source/image audit",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report_json", type=Path, default=None)
    args = parser.parse_args()

    rows, metadata_sha256 = _read_jsonl(args.metadata_jsonl)
    counts = Counter(row.get("sample_type") for row in rows)
    expected_types = {"edit_mt", "edit_ntp", "edit", "edit_umt"}
    if set(counts) != expected_types:
        raise ValueError(f"Expected all refined Stage-1 types, got {dict(counts)}")
    if not (
        counts["edit_mt"] == 2 * counts["edit_ntp"]
        and counts["edit_mt"] == 4 * counts["edit"]
        and counts["edit_mt"] == 4 * counts["edit_umt"]
    ):
        raise ValueError(f"Bad refined Stage-1 ratio: {dict(counts)}")

    crisp_rows = defaultdict(dict)
    gres_rows = []
    gres_seen = {}
    identities_by_type = defaultdict(set)
    spans_by_identity = defaultdict(set)
    schedule_padding = Counter()
    edit_types_by_sample_type = defaultdict(Counter)
    qc_flags_by_sample_type = defaultdict(Counter)
    umt_rewrite_methods = Counter()
    for metadata_index, row in enumerate(rows):
        sample_type = row["sample_type"]
        padding = row.get("schedule_padding")
        if padding is not None:
            expected_reasons = {
                "exact_4_to_2_to_1_to_1_ratio",
                "exact_4_to_2_to_1_to_1_ratio_and_distributed_step_divisibility",
            }
            if padding.get("reason") not in expected_reasons:
                raise ValueError(
                    f"Bad schedule padding at metadata row {metadata_index}: {padding}"
                )
            schedule_padding[sample_type] += 1
        if not row["prompt"].isascii():
            raise ValueError(f"Non-ASCII prompt at metadata row {metadata_index}")
        provenance = row.get("provenance") or {}
        if provenance.get("edit_type") is not None:
            edit_types_by_sample_type[sample_type][provenance["edit_type"]] += 1
        if provenance.get("qc_flag") is not None:
            qc_flags_by_sample_type[sample_type][provenance["qc_flag"]] += 1
        if sample_type == "edit_umt":
            umt_rewrite_methods[provenance.get("umt_rewrite_method", "")] += 1
        if sample_type == "edit_ntp":
            source_index = int(provenance["source_row_idx"])
            if source_index in gres_seen:
                previous_index, previous = gres_seen[source_index]
                previous_core = {
                    key: value
                    for key, value in previous.items()
                    if key != "schedule_padding"
                }
                current_core = {
                    key: value
                    for key, value in row.items()
                    if key != "schedule_padding"
                }
                if previous_core != current_core or (
                    "schedule_padding" not in previous and padding is None
                ):
                    raise ValueError(
                        f"Unmarked or divergent duplicate edit_ntp source: {source_index}"
                    )
                if padding is None:
                    gres_seen[source_index] = (metadata_index, row)
            else:
                gres_seen[source_index] = (metadata_index, row)
            gres_rows.append((metadata_index, row))
            continue
        provenance = row.get("provenance") or {}
        identity = (str(provenance.get("source_parquet")), int(provenance["row_idx"]))
        if identity in crisp_rows[sample_type]:
            previous_index, previous = crisp_rows[sample_type][identity]
            previous_core = {
                key: value
                for key, value in previous.items()
                if key != "schedule_padding"
            }
            current_core = {key: value for key, value in row.items() if key != "schedule_padding"}
            if previous_core != current_core or (
                "schedule_padding" not in previous and padding is None
            ):
                raise ValueError(f"Unmarked or divergent duplicate {sample_type}: {identity}")
            if padding is None:
                crisp_rows[sample_type][identity] = (metadata_index, row)
        else:
            crisp_rows[sample_type][identity] = (metadata_index, row)
        identities_by_type[sample_type].add(identity)
        if sample_type == "edit_mt":
            spans_by_identity[identity].add(_cot_span(row["mt_cot"]))
        elif sample_type == "edit_umt":
            spans_by_identity[identity].add(_umt_span(row["prompt"]))
        elif "mt_cot" in row:
            raise ValueError(f"Plain edit row {metadata_index} unexpectedly has mt_cot")

    if not identities_by_type["edit_umt"].issubset(identities_by_type["edit_mt"]):
        raise ValueError("Every edit_umt source must also exist in the full edit_mt pool")
    divergent = [identity for identity, spans in spans_by_identity.items() if len(spans) > 1]
    if divergent:
        raise ValueError(f"edit_mt/edit_umt mask spans diverge; first={divergent[0]}")

    all_crisp_ids = set().union(*identities_by_type.values())
    if args.image_byte_sample < -1:
        raise ValueError("image_byte_sample must be -1 or non-negative")
    if args.image_byte_sample == -1:
        image_audit_ids = all_crisp_ids
    else:
        sample_size = min(args.image_byte_sample, len(all_crisp_ids))
        image_audit_ids = set(random.Random(args.seed).sample(sorted(all_crisp_ids), sample_size))
    if args.codec_sample < 0:
        raise ValueError("codec_sample must be non-negative")
    codec_audit_ids = set(
        random.Random(args.seed + 1).sample(
            sorted(identities_by_type["edit_mt"]),
            min(args.codec_sample, len(identities_by_type["edit_mt"])),
        )
    )
    codec_inputs = []

    ids_by_file = defaultdict(set)
    for filename, row_idx in all_crisp_ids:
        ids_by_file[filename].add(row_idx)
    if args.io_workers < 1:
        raise ValueError("io_workers must be positive")
    byte_checked = 0
    mask_checked = 0
    manifest_checked = 0
    edit_types = Counter()

    def audit_crisp_shard(filename: str, wanted_indices: set[int]):
        local_byte_checked = 0
        local_mask_checked = 0
        local_manifest_checked = 0
        local_edit_types = Counter()
        local_codec_inputs = []
        raw_path = args.crispedit_dir / filename
        mask_path = args.mask_dir / filename
        manifest_path = args.filter_manifest_dir / filename
        if not raw_path.is_file() or not mask_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"Missing aligned source shard for {filename}")
        need_raw_images = bool(
            wanted_indices & (image_audit_ids | codec_audit_ids)
        )
        raw_columns = ["instruction", "type"]
        if need_raw_images:
            raw_columns.extend(["input_img", "output_img"])
        raw = pq.read_table(
            raw_path, columns=raw_columns
        ).to_pylist()
        mask = pq.read_table(
            mask_path,
            columns=[
                "row_idx",
                "instruction",
                "filter_decision",
                "mask_png",
                "mask_sum",
                "canonical_type",
                "qc_flag",
                "instance_masks",
                "ground_json",
            ],
        ).to_pylist()
        manifest = pq.read_table(
            manifest_path, columns=["row_idx", "filter_decision"]
        ).to_pylist()
        if not (len(raw) == len(mask) == len(manifest)):
            raise ValueError(f"Aligned shard lengths changed for {filename}")
        for row_idx in wanted_indices:
            source = raw[row_idx]
            mask_source = mask[row_idx]
            manifest_source = manifest[row_idx]
            if (
                int(mask_source["row_idx"]) != row_idx
                or int(manifest_source["row_idx"]) != row_idx
            ):
                raise ValueError(f"row_idx alignment changed for {filename}:{row_idx}")
            local_edit_types[str(source.get("type") or "")] += 1
            identity = (filename, row_idx)
            identity_references = {
                (record[1]["edit_image"], record[1]["image"])
                for sample_type in ("edit_mt", "edit", "edit_umt")
                if (record := crisp_rows[sample_type].get(identity)) is not None
            }
            if len(identity_references) != 1:
                raise ValueError(
                    f"CrispEdit branches disagree on materialized images: {identity}"
                )
            for sample_type in ("edit_mt", "edit", "edit_umt"):
                record = crisp_rows[sample_type].get(identity)
                if record is None:
                    continue
                metadata_index, row = record
                expected_prompt = (
                    (row.get("provenance") or {}).get("original_prompt")
                    if sample_type == "edit_umt"
                    else row["prompt"]
                )
                if expected_prompt != source["instruction"]:
                    raise ValueError(
                        f"Prompt/source mismatch at metadata row {metadata_index}"
                    )
                provenance = row.get("provenance") or {}
                expected_edit_type = (
                    mask_source["canonical_type"]
                    if sample_type in {"edit_mt", "edit_umt"}
                    else source.get("type")
                )
                if provenance.get("edit_type") != expected_edit_type:
                    raise ValueError(
                        f"Edit-type provenance changed at metadata row {metadata_index}"
                    )
                if sample_type in {"edit_mt", "edit_umt"} and (
                    provenance.get("qc_flag") != mask_source.get("qc_flag")
                    or int(provenance.get("mask_sum") or 0)
                    != int(mask_source.get("mask_sum") or 0)
                ):
                    raise ValueError(
                        f"Mask provenance changed at metadata row {metadata_index}"
                    )
                if sample_type == "edit_mt":
                    span = _cot_span(row["mt_cot"])
                    expected_label = _label_for_row(
                        mask_source,
                        mask_source["canonical_type"],
                        source["instruction"],
                    )
                    expected_cot = to_cot(
                        [(span, make_labels(expected_label, 1)[0])]
                    )
                    if row["mt_cot"] != expected_cot:
                        raise ValueError(
                            f"MT CoT label/source mismatch at metadata row {metadata_index}"
                        )
                if sample_type == "edit_umt":
                    span = _umt_span(row["prompt"])
                    transformed = build_umt_prompt(
                        expected_prompt, mask_source["canonical_type"], span
                    )
                    provenance = row["provenance"]
                    if transformed is None or transformed != (
                        row["prompt"],
                        provenance["umt_replaced_text"],
                        provenance["umt_rewrite_method"],
                    ):
                        raise ValueError(
                            f"UMT derivation mismatch at metadata row {metadata_index}"
                        )
            if identity in identities_by_type["edit"]:
                if manifest_source["filter_decision"] != "keep":
                    raise ValueError(f"Plain edit bypassed fact prefilter: {identity}")
                local_manifest_checked += 1
            if identity in identities_by_type["edit_mt"]:
                if (
                    mask_source["filter_decision"] != "keep"
                    or not mask_source["mask_png"]
                    or int(mask_source["mask_sum"] or 0) <= 0
                    or mask_source["instruction"] != source["instruction"]
                ):
                    raise ValueError(f"Invalid refined mask source: {identity}")
                local_mask_checked += 1
                if identity in codec_audit_ids:
                    source_bytes = _image_bytes(source["input_img"], raw_path.parent)
                    source_image, _ = _decode_image(source_bytes)
                    source_mask = _decode_mask(mask_source["mask_png"], source_image.size)
                    expected_span = _cot_span(
                        crisp_rows["edit_mt"][identity][1]["mt_cot"]
                    )
                    local_codec_inputs.append(
                        (identity, source_image, source_mask, expected_span)
                    )
            if identity in image_audit_ids:
                representative = next(
                    crisp_rows[sample_type][identity][1]
                    for sample_type in ("edit_mt", "edit", "edit_umt")
                    if identity in crisp_rows[sample_type]
                )
                expected_source = _image_bytes(source["input_img"], raw_path.parent)
                expected_target = _image_bytes(source["output_img"], raw_path.parent)
                actual_source = _resolve(representative["edit_image"], args.base_path).read_bytes()
                actual_target = _resolve(representative["image"], args.base_path).read_bytes()
                if actual_source != expected_source or actual_target != expected_target:
                    raise ValueError(f"Materialized image bytes differ for {identity}")
                local_byte_checked += 1
        return (
            local_byte_checked,
            local_mask_checked,
            local_manifest_checked,
            local_edit_types,
            local_codec_inputs,
        )

    shard_items = sorted(ids_by_file.items())
    with ThreadPoolExecutor(max_workers=args.io_workers) as executor:
        futures = {
            executor.submit(audit_crisp_shard, filename, wanted_indices): filename
            for filename, wanted_indices in shard_items
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            (
                shard_bytes,
                shard_masks,
                shard_manifests,
                shard_types,
                shard_codec_inputs,
            ) = future.result()
            byte_checked += shard_bytes
            mask_checked += shard_masks
            manifest_checked += shard_manifests
            edit_types.update(shard_types)
            codec_inputs.extend(shard_codec_inputs)
            if completed % 25 == 0 or completed == len(futures):
                print(
                    json.dumps(
                        {
                            "audit_progress": "crispedit_source_integrity",
                            "completed_shards": completed,
                            "total_shards": len(futures),
                            "exact_image_byte_identities_checked": byte_checked,
                        }
                    ),
                    flush=True,
                )

    codec_checked = 0
    if codec_inputs:
        codec = SamtokCodec(
            args.sam2_ckpt,
            args.mask_tokenizer_ckpt,
            device=args.codec_device,
        )
        for start in range(0, len(codec_inputs), args.codec_batch_size):
            batch = codec_inputs[start : start + args.codec_batch_size]
            actual_spans = codec.encode_single_batch(
                (source_image, source_mask)
                for _, source_image, source_mask, _ in batch
            )
            for (identity, _, _, expected_span), actual_span in zip(batch, actual_spans):
                if actual_span != expected_span:
                    raise ValueError(
                        f"Re-encoded SAMTok span changed for {identity}: "
                        f"{actual_span} != {expected_span}"
                    )
                codec_checked += 1

    with args.gres_json.open(encoding="utf-8") as handle:
        gres_source = json.load(handle)
    expected_gres_dataset = args.gres_json.absolute()
    for completed, (metadata_index, row) in enumerate(gres_rows, start=1):
        provenance = row.get("provenance") or {}
        source_index = int(provenance["source_row_idx"])
        source = gres_source[source_index]
        if Path(provenance.get("source_dataset", "")).absolute() != expected_gres_dataset:
            raise ValueError(f"GRES source dataset changed at row {metadata_index}")
        if provenance.get("source_image") != source["image"]:
            raise ValueError(f"GRES source image provenance changed at row {metadata_index}")
        conversations = source["conversations"]
        source_cot = parse_and_canonicalize_mt_cot(conversations[1]["value"])
        if source_cot != row["mt_cot"]:
            raise ValueError(f"GRES CoT changed at metadata row {metadata_index}")
        expression = expression_from_cot(source_cot) or expression_from_question(
            conversations[0]["value"]
        )
        valid_prompts = {
            template.format(expr=expression) for template in EDIT_VERB_TEMPLATES
        }
        if row["prompt"] not in valid_prompts:
            raise ValueError(f"GRES template derivation changed at row {metadata_index}")
        expected_image = (args.gres_image_root / source["image"]).absolute()
        if _resolve(row["edit_image"], args.base_path).absolute() != expected_image:
            raise ValueError(f"GRES image path changed at row {metadata_index}")
        if completed % 5000 == 0 or completed == len(gres_rows):
            print(
                json.dumps(
                    {
                        "audit_progress": "gres_source_integrity",
                        "completed_rows": completed,
                        "total_rows": len(gres_rows),
                    }
                ),
                flush=True,
            )

    overlap = {}
    crisp_types = ("edit_mt", "edit", "edit_umt")
    for left_index, left in enumerate(crisp_types):
        for right in crisp_types[left_index + 1 :]:
            overlap[f"{left}&{right}"] = len(
                identities_by_type[left] & identities_by_type[right]
            )
    report = {
        "metadata": str(args.metadata_jsonl.resolve()),
        "metadata_sha256": metadata_sha256,
        "rows": len(rows),
        "counts": dict(counts),
        "schedule_padding": dict(schedule_padding),
        "ratio": "edit_mt:edit_ntp:edit:edit_umt=4:2:1:1",
        "unique_crispedit_identities": len(all_crisp_ids),
        "unique_crispedit_identities_by_type": {
            sample_type: len(identities)
            for sample_type, identities in sorted(identities_by_type.items())
        },
        "unique_gres_source_rows": len(gres_seen),
        "source_identity_overlap_allowed": overlap,
        "edit_types_by_sample_type": {
            sample_type: dict(values)
            for sample_type, values in sorted(edit_types_by_sample_type.items())
        },
        "qc_flags_by_sample_type": {
            sample_type: dict(values)
            for sample_type, values in sorted(qc_flags_by_sample_type.items())
        },
        "umt_rewrite_methods": dict(umt_rewrite_methods),
        "mask_source_rows_checked": mask_checked,
        "mt_cot_labels_rederived_and_matched": mask_checked,
        "manifest_source_rows_checked": manifest_checked,
        "gres_source_rows_checked": len(gres_rows),
        "exact_image_byte_identities_checked": byte_checked,
        "image_byte_check_scope": "all" if args.image_byte_sample == -1 else "sample",
        "codec_spans_reencoded_and_matched": codec_checked,
        "edit_types_over_unique_identities": dict(edit_types),
        "empty_cot": 0,
        "non_ascii_text": 0,
        "umt_is_subset_of_edit_mt": True,
        "mt_umt_span_consistent": True,
    }
    if args.report_json is not None:
        _write_json_atomic(report, args.report_json)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
