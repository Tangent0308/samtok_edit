"""Small real-data integration set and dataset-independent units converter."""

from __future__ import annotations

import io
import json
import re
from collections import Counter
from pathlib import Path

import ijson
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from .data import write_json, write_rows, row_hash, file_hash
from .protocol import (
    Unit,
    EDIT_TYPES,
    phrase_span,
    is_valid_span,
    parse_cot,
    render_units,
    to_cot,
    validate_row,
)

TYPE_MAP = {
    "color": "attribute",
    "motion": "action",
    "motion change": "action",
    "background": "background",
    "background change": "background",
    "style": "global",
    "add": "add",
    "remove": "remove",
    "replace": "replace",
}


def convert_record(record):
    """Common interchange format: images, instruction, edit_type and units.

    Units contain reviewed ref_phrase, edit_type, mask_codes and optional add
    anchor_phrase. No dataset-specific language guessing is done by this converter.
    Unsupported rewrites are reported independently; valid NTP/ref/plain rows survive.
    """
    instruction = record["instruction"]
    units = [
        Unit(
            x["ref_phrase"],
            tuple(x["mask_codes"]),
            x["edit_type"],
            x.get("anchor_phrase"),
        )
        for x in record["units"]
    ]
    if not units:
        raise ValueError("A mask record must have at least one unit")
    for unit in units:
        if unit.edit_type not in set(EDIT_TYPES) - {"composite"}:
            raise ValueError("Each unit must have an atomic edit_type")
        if not isinstance(unit.ref_phrase, str) or not unit.ref_phrase.strip():
            raise ValueError("Missing unit reference")
        if not unit.codes or not all(is_valid_span(x) for x in unit.codes):
            raise ValueError("Unit needs valid mask_codes")
    # NTP and UMT use the same canonical instruction order, independent of
    # annotation storage order. Global operations without a phrase go last.
    units.sort(
        key=lambda u: len(instruction)
        if u.edit_type == "global"
        else phrase_span(instruction, u.ref_phrase)[0]
    )
    edit_type = "composite" if len(units) > 1 else units[0].edit_type
    items = []
    for unit in units:
        label = re.sub(r"^(?:the|a|an)\s+", "", unit.ref_phrase, flags=re.I)
        if unit.edit_type == "global":
            label = "this image"
        if len(unit.codes) > 1:
            label = "one of the " + label
        items.extend((s, label) for s in unit.codes)
    common = {"edit_image": record["edit_image"], "edit_type": edit_type}
    ntp = {
        **common,
        "sample_type": "edit_ntp",
        "prompt": instruction,
        "mt_cot": to_cot(items),
    }
    rows = [ntp]
    errors = {}
    if "image" in record:
        fm = {**common, "image": record["image"]}
        rows.append({**fm, "sample_type": "edit", "prompt": instruction})
        for variant in ("ref", "noref"):
            try:
                prompt = render_units(instruction, units, variant=variant)
                rows.append(
                    {
                        **fm,
                        "sample_type": "edit_umt",
                        "prompt": prompt,
                        "instr_variant": variant,
                    }
                )
            except ValueError as exc:
                errors[variant] = str(exc)
        if len(rows) == 4 and rows[-1]["prompt"] == rows[-2]["prompt"]:
            rows.pop(-2)  # Identical global variants count only as noref.
    for row in rows:
        validate_row(row)
    return rows, errors


def debug_reference(prompt, typ):
    """Conservative, explicitly debug-only templates; ambiguity means skip."""
    patterns = {
        "attribute": [
            r"^(?:Turn|Transform) (.+?) into .+",
            r"^(?:Change|Modify) (?:the colou?r of )?(.+?) to .+",
        ],
        "remove": [r"^(?:Remove|Delete|Erase|Get rid of) (.+?)(?:[.!]?$)"],
        "replace": [r"^(?:Replace|Swap) (.+?) (?:with|for) .+", r"^Change (.+?) to .+"],
        "add": [r"^(?:Add|Insert|Place|Put|Draw) (.+?)[.!]?$"],
        "action": [
            r"^(?:Make|Have) (.+?) (?:stand|sit|move|walk|run|jump|spread|raise|lower|turn|open|close|look|smile)\b.+"
        ],
    }
    if typ == "global":
        return "this image", None
    if typ == "background":
        for phrase in ("background", "scene", "backdrop"):
            if len(re.findall(r"\b" + phrase + r"\b", prompt, re.I)) == 1:
                return phrase, None
        raise ValueError("No unique background reference")
    for pattern in patterns.get(typ, []):
        m = re.match(pattern, prompt, re.I)
        if m:
            phrase = m[1].strip()
            anchor = None
            if typ == "add":
                a = re.search(
                    r"\b(?:near|next to|beside|behind|in front of|on top of|on|under|above)\s+.+$",
                    phrase,
                    re.I,
                )
                if a:
                    anchor = a.group()
            return phrase, anchor
    raise ValueError("No reliable debug reference rule")


def build_debug(args):
    from .codec import SamtokCodec
    import torch

    out = Path(args.output)
    if (out / "build_report.json").exists():
        raise ValueError("Use a fresh debug output directory")
    out.mkdir(parents=True, exist_ok=True)
    codec = SamtokCodec(
        str(Path(args.samtok) / "sam2.1_hiera_large.pt"),
        str(Path(args.samtok) / "mask_tokenizer_256x2.pth"),
        device=args.device,
        dtype=torch.float32,
    )
    rows, manifest, skips = [], [], Counter()
    rawroot, maskroot = Path(args.crisp), Path(args.masks)
    # Bounded per-type prefix is intentional for integration debugging, not a
    # statistically representative training/evaluation sample.
    for prefix in (
        "color",
        "remove",
        "replace",
        "add",
        "background",
        "style",
        "motion",
    ):
        accepted = 0
        for maskfile in sorted(maskroot.glob(prefix + "*.parquet"))[:3]:
            rawfile = rawroot / maskfile.name
            if not rawfile.exists():
                skips["missing_raw_shard"] += 1
                continue
            maskrows = pq.read_table(maskfile).to_pylist()
            candidate = []
            for maskrow in maskrows:
                if (
                    maskrow["filter_decision"] != "keep"
                    or not maskrow["mask_png"]
                    or not maskrow["mask_sum"]
                ):
                    continue
                typ = TYPE_MAP.get(maskrow["raw_type"])
                if not typ:
                    continue
                try:
                    ref, anchor = debug_reference(maskrow["instruction"], typ)
                except ValueError:
                    skips["ambiguous_reference"] += 1
                    continue
                candidate.append((int(maskrow["row_idx"]), maskrow, typ, ref, anchor))
                if len(candidate) >= args.per_type * 3:
                    break
            if not candidate:
                continue
            needed = {x[0]: x[1:] for x in candidate}
            offset = 0
            for batch in pq.ParquetFile(rawfile).iter_batches(batch_size=8):
                for j, raw in enumerate(batch.to_pylist()):
                    idx = offset + j
                    if idx not in needed:
                        continue
                    maskrow, typ, ref, anchor = needed[idx]
                    if raw["instruction"] != maskrow["instruction"]:
                        raise ValueError("Raw/mask instruction join mismatch")
                    source = Image.open(io.BytesIO(raw["input_img"]["bytes"])).convert(
                        "RGB"
                    )
                    target = Image.open(io.BytesIO(raw["output_img"]["bytes"])).convert(
                        "RGBA"
                    )
                    mask = Image.open(io.BytesIO(maskrow["mask_png"])).convert("L")
                    if mask.size != source.size:
                        skips["mask_geometry_mismatch"] += 1
                        continue
                    binary = np.asarray(mask) > 0
                    if typ == "global" and not binary.all():
                        skips["nonfull_global_mask"] += 1
                        continue
                    codes = codec.encode_single_batch([(source, binary)])
                    stem = f"{rawfile.stem.replace(' ', '_')}_{idx:06d}"
                    folder = out / "images"
                    folder.mkdir(exist_ok=True)
                    srcpath = folder / f"{stem}_source.png"
                    tarpath = folder / f"{stem}_target.png"
                    source.save(srcpath)
                    target.save(tarpath)
                    masks = out / "masks"
                    masks.mkdir(exist_ok=True)
                    mask.save(masks / f"{stem}_raw.png")
                    decoded = codec.decode(source, codes[0])[0]
                    Image.fromarray(decoded * 255).save(masks / f"{stem}_decoded.png")
                    iou = float(
                        (decoded.astype(bool) & binary).sum()
                        / max(1, (decoded.astype(bool) | binary).sum())
                    )
                    record = {
                        "edit_image": str(srcpath.relative_to(out)),
                        "image": str(tarpath.relative_to(out)),
                        "instruction": raw["instruction"],
                        "units": [
                            {
                                "ref_phrase": ref,
                                "mask_codes": codes,
                                "edit_type": typ,
                                "anchor_phrase": anchor,
                            }
                        ],
                    }
                    converted, errors = convert_record(record)
                    rows.extend(converted)
                    manifest.append(
                        {
                            "source_dataset": "CrispEdit-2M",
                            "parquet": str(rawfile),
                            "row_idx": idx,
                            "record": record,
                            "derived_row_hashes": [row_hash(r) for r in converted],
                            "rewrite_errors": errors,
                            "codec_iou": iou,
                            "mask_path": str(masks / f"{stem}_raw.png"),
                            "decoded_path": str(masks / f"{stem}_decoded.png"),
                            "debug_only": True,
                            "qc_flag": maskrow["qc_flag"],
                        }
                    )
                    accepted += 1
                    print(
                        f"{prefix}: {accepted}/{args.per_type}; codec IoU={iou:.3f}",
                        flush=True,
                    )
                    if accepted >= args.per_type:
                        break
                offset += len(batch)
                if accepted >= args.per_type or offset > max(needed):
                    break
            if accepted >= args.per_type:
                break
    # Positive SAMTok/GRES replay, preserving released mask codes and labels.
    seen_sources = set()
    ntp_count = 0
    with Path(args.gres).open("rb") as f:
        for source_idx, raw in enumerate(ijson.items(f, "item")):
            if source_idx > 20000:
                break
            image = Path(args.gres_images) / raw["image"]
            if str(image) in seen_sources or not image.exists():
                continue
            try:
                answer = next(
                    x["value"] for x in raw["conversations"] if x["from"] == "gpt"
                )
                pairs = parse_cot(answer, nonempty=True)
                expressions = {
                    re.sub(r"^one of (?:the )?", "", label, flags=re.I)
                    for _, label in pairs
                }
                if len(expressions) != 1:
                    continue
                expression = next(iter(expressions))
                prompt = f"Change the color of {expression} to blue."
                row = {
                    "edit_image": str(image),
                    "prompt": prompt,
                    "sample_type": "edit_ntp",
                    "edit_type": "attribute",
                    "mt_cot": to_cot(pairs),
                }
                validate_row(row)
            except (ValueError, TypeError, StopIteration):
                continue
            rows.append(row)
            seen_sources.add(str(image))
            ntp_count += 1
            manifest.append(
                {
                    "source_dataset": "SAMTok/GRES209k",
                    "source_index": source_idx,
                    "image": str(image),
                    "derived_row_hashes": [row_hash(row)],
                    "debug_only": True,
                }
            )
            if ntp_count >= args.gres_count:
                break
    if ntp_count < args.gres_count:
        raise RuntimeError(
            "Insufficient GRES images found; check --gres-images mapping"
        )
    # Additional genuinely plain raw edits, without requiring a mask.
    plainfile = next(iter(sorted(rawroot.glob("color*.parquet"))))
    for idx, raw in enumerate(
        next(pq.ParquetFile(plainfile).iter_batches(batch_size=2)).to_pylist()
    ):
        paths = []
        for key, suffix in [("input_img", "source"), ("output_img", "target")]:
            path = out / "images" / f"plain_{idx}_{suffix}.png"
            Image.open(io.BytesIO(raw[key]["bytes"])).convert("RGBA").save(path)
            paths.append(str(path.relative_to(out)))
        row = {
            "edit_image": paths[0],
            "image": paths[1],
            "prompt": raw["instruction"],
            "sample_type": "edit",
            "edit_type": TYPE_MAP[raw["type"]],
        }
        rows.append(row)
        manifest.append(
            {
                "source_dataset": "CrispEdit-2M/plain",
                "parquet": str(plainfile),
                "row_idx": idx,
                "derived_row_hashes": [row_hash(row)],
                "debug_only": True,
            }
        )
    write_rows(out / "stage1.jsonl", rows)
    write_rows(
        out / "stage2.jsonl", [r for r in rows if r["sample_type"] != "edit_ntp"]
    )
    write_json(out / "provenance.json", manifest)
    report = {
        "debug_only": True,
        "rows": len(rows),
        "sample_types": dict(Counter(r["sample_type"] for r in rows)),
        "edit_types": dict(Counter(r["edit_type"] for r in rows)),
        "skips": dict(skips),
        "gres_count": ntp_count,
        "sha256": {n: file_hash(out / n) for n in ["stage1.jsonl", "stage2.jsonl"]},
    }
    write_json(out / "build_report.json", report)
    print(json.dumps(report), flush=True)
