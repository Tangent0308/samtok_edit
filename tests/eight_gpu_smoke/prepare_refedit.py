"""Build a tiny protocol-valid smoke set from self-contained RefEdit parquet rows.

The source dataset is read only. Images and metadata are materialized under the
experiment output directory so the distributed train/cache commands have a
stable base path and do not depend on parquet readers inside worker processes.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from samtok_edit21.codec import SamtokCodec
from samtok_edit21.data import write_json, write_rows
from samtok_edit21.protocol import to_cot, validate_row


def _first_source_ref(record):
    try:
        ground = json.loads(record["ground_json"])
        source = ground.get("source") or []
        if source and source[0].get("ref"):
            return source[0]["ref"]
    except (TypeError, ValueError, KeyError):
        pass
    return "the selected region"


def _read_candidates(source):
    for shard in sorted((source / "data").glob("*.parquet")):
        for row in pq.ParquetFile(shard).iter_batches(
            batch_size=64,
            columns=[
                "sample_id",
                "instruction",
                "final_task",
                "source_img",
                "target_img",
                "mask_png",
                "mask_sum",
                "ground_json",
                "qc_flag",
            ],
        ):
            for record in row.to_pylist():
                if (
                    record["mask_png"]
                    and int(record["mask_sum"]) > 0
                    and record["source_img"].get("bytes")
                    and record["target_img"].get("bytes")
                    and record["qc_flag"] == "OK"
                ):
                    yield record, shard


def build(args):
    source = Path(args.source)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    image_dir = output / "images"
    image_dir.mkdir(exist_ok=True)

    codec = SamtokCodec(
        args.samtok + "/sam2.1_hiera_large.pt",
        args.samtok + "/mask_tokenizer_256x2.pth",
        device=args.device,
    )
    selected = []
    for record, shard in _read_candidates(source):
        selected.append((record, shard))
        if len(selected) >= args.unique_rows:
            break
    if len(selected) < args.unique_rows:
        raise RuntimeError(f"only found {len(selected)} usable rows")

    base_rows = []
    provenance = []
    for index, (record, shard) in enumerate(selected):
        source_image = Image.open(io.BytesIO(record["source_img"]["bytes"])).convert("RGBA")
        target_image = Image.open(io.BytesIO(record["target_img"]["bytes"])).convert("RGBA")
        mask = Image.open(io.BytesIO(record["mask_png"])).convert("L")
        if mask.size != source_image.size:
            raise ValueError(f"{record['sample_id']}: mask/source geometry mismatch")
        span = codec.encode_single_batch([(source_image, np.asarray(mask) > 0)])[0]
        stem = f"refedit_{index:02d}"
        source_path = image_dir / f"{stem}_source.png"
        target_path = image_dir / f"{stem}_target.png"
        source_image.save(source_path)
        target_image.save(target_path)
        label = _first_source_ref(record)
        base_rows.append(
            {
                "edit_image": str(source_path.relative_to(output)),
                "image": str(target_path.relative_to(output)),
                "prompt": record["instruction"],
                "edit_type": "attribute",
                "span": span,
                "label": label,
                "sample_id": record["sample_id"],
            }
        )
        provenance.append(
            {
                "sample_id": record["sample_id"],
                "source_parquet": str(shard),
                "source_mask_sum": int(record["mask_sum"]),
                "instruction": record["instruction"],
                "encoded_span": span,
                "label": label,
            }
        )

    def make_ntp(base):
        row = {
            "edit_image": base["edit_image"],
            "prompt": base["prompt"],
            "sample_type": "edit_ntp",
            "edit_type": base["edit_type"],
            "mt_cot": to_cot([(base["span"], base["label"])]),
        }
        return validate_row(row)

    def make_umt(base, variant):
        # Inline placement is deliberately explicit for this smoke set. The
        # production converter should use the reviewed phrase-level rewrite.
        row = {
            "edit_image": base["edit_image"],
            "image": base["image"],
            "prompt": base["prompt"].rstrip(".!? ") + " in this region " + base["span"] + ".",
            "sample_type": "edit_umt",
            "instr_variant": variant,
            "edit_type": base["edit_type"],
        }
        return validate_row(row)

    def make_plain(base):
        return validate_row(
            {
                "edit_image": base["edit_image"],
                "image": base["image"],
                "prompt": base["prompt"],
                "sample_type": "edit",
                "edit_type": base["edit_type"],
            }
        )

    stage1 = [
        make_ntp(base_rows[0]),
        make_ntp(base_rows[1]),
        make_ntp(base_rows[2]),
        make_umt(base_rows[0], "ref"),
        make_umt(base_rows[3], "ref"),
        make_umt(base_rows[4], "noref"),
        make_umt(base_rows[5], "noref"),
        make_plain(base_rows[6]),
    ]
    stage2 = [
        make_umt(base_rows[0], "ref"),
        make_umt(base_rows[1], "ref"),
        make_umt(base_rows[2], "noref"),
        make_umt(base_rows[3], "noref"),
        make_umt(base_rows[4], "noref"),
        make_umt(base_rows[5], "noref"),
        make_plain(base_rows[6]),
        make_plain(base_rows[7]),
    ]
    write_rows(output / "stage1.jsonl", stage1)
    write_rows(output / "stage2.jsonl", stage2)
    write_json(output / "provenance.json", provenance)
    write_json(
        output / "build_report.json",
        {
            "source": str(source),
            "unique_rows": len(selected),
            "stage1_rows": len(stage1),
            "stage2_rows": len(stage2),
            "debug_only": True,
        },
    )
    print(json.dumps({"output": str(output), "stage1": len(stage1), "stage2": len(stage2)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samtok", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--unique-rows", type=int, default=8)
    build(parser.parse_args())
