"""Convert the v1 training corpus into the v2 metadata contract.

v1 built every row from ``input-XX.jsonl`` (source + reviewed annotation) and
encoded the dataset masks with the released SAMTok codec into
``encoded/worker-XX``.  v2 reuses those encoded rows verbatim and changes only
what the v2 plan specifies:

* composite records are dropped (all of their rows);
* add units are re-expressed as boxes.  Each box is the outward-rounded
  0-1000 bounding box of exactly the instance mask that produced the v1 span,
  in the codec's own multi-instance order, so the box replaces its span in
  place in the NTP answer and in both UMT prompts;
* Derived add rows (a small detail painted on an existing host instance whose
  mask is the host) become attribute rows and keep their mask spans;
* rec_ntp rows (Qwen3-VL grounding request -> bbox_2d) are added for Stage 1
  box-grounding replay, built from single-instance non-add units of the
  instance-accurate RefEdit and Derived sources.

Stage 1 is NTP only (edit_ntp + rec_ntp); Stage 2 is edit + edit_umt.  Every
written row passes ``validate_row``; every v1 row is checked against the v1
provenance hash before it is used.
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path

import numpy as np

from samtok_edit21.data.io import file_hash, row_hash, write_json
from samtok_edit21.data.protocol import (
    REC_TEMPLATE, box_coords, box_of, parse_cot, pixel_box, spans_in, to_cot, validate_row,
)
from samtok_edit21.preparation.corpus import masks_for_record, read_jsonl, worker_rows

FORMAT = "samtok21-metadata-v2"
V1_DATA = ("/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/"
           "qwen21_full4_20260928/data/train_full_9b_rules_003")
REC_DATASETS = ("derived", "refedit")
REC_TYPES = ("remove", "replace", "attribute", "action")


def code_commit():
    import subprocess

    from samtok_edit21.paths import repository_root
    try:
        root = repository_root()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src"], cwd=root,
                               capture_output=True, text=True, check=True).stdout.strip()
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                              text=True, check=True).stdout.strip()
        return head + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError, RuntimeError):
        return None


def kind_of(row):
    return row["sample_type"] + (":" + row["instr_variant"] if row["sample_type"] == "edit_umt" else "")


def v1_records(v1):
    """(input record, encoded record) pairs in global source-index order."""
    identity = json.loads((v1 / "inputs.json").read_text())

    def worker(rank):
        path = v1 / f"input-{rank:02d}.jsonl"
        if file_hash(path) != identity["input_sha256"][f"input-{rank:02d}"]:
            raise ValueError(f"v1 worker input changed: {path}")
        for item, encoded in zip_longest(read_jsonl(path), worker_rows(v1, rank)):
            if item is None or encoded is None or item["index"] != encoded["index"]:
                raise ValueError(f"v1 input/encoded records disagree for worker {rank}")
            yield item, encoded

    return heapq.merge(*(worker(rank) for rank in range(identity["workers"])),
                       key=lambda pair: pair[0]["index"])


def ordered_unit_masks(source, units):
    """The masks behind each unit's spans, in the order v1 encoded them."""
    from samtok_edit21.models.codec import SamtokCodec

    groups = []
    width, height = source["source_size"]
    for group in masks_for_record(source, units):
        if any(np.asarray(mask).shape != (height, width) for mask in group):
            raise ValueError(f"{source['id']}: mask is not in source-image coordinates")
        if len(group) > 1:
            _, order = SamtokCodec._ordered_masks(group)
            group = [group[index] for index in order]
        groups.append(group)
    return groups


def add_boxes(item, rows):
    """Replace each add span by the box of the mask it encodes, in place."""
    source, annotation = item["source"], item["annotation"]
    units = annotation["units"]
    if len(units) != 1 or units[0]["edit_type"] != "add":
        raise ValueError(f"{source['id']}: an atomic add record has one add unit")
    masks = ordered_unit_masks(source, units)[0]
    ntp = [row for row in rows if row["sample_type"] == "edit_ntp"]
    if len(ntp) != 1:
        raise ValueError(f"{source['id']}: add record needs one NTP row")
    pairs = parse_cot(ntp[0]["mt_cot"], nonempty=True)
    spans = [span for span, _ in pairs]
    if len(spans) != len(masks):
        raise ValueError(f"{source['id']}: {len(spans)} spans for {len(masks)} masks")
    boxes = [box_of(pixel_box(mask)) for mask in masks]
    converted = []
    for row in rows:
        row = dict(row)
        if row["sample_type"] == "edit_ntp":
            row["mt_cot"] = to_cot([(box, label) for (_, label), box in zip(pairs, boxes)])
        elif row["sample_type"] == "edit_umt":
            joined = "".join(spans)
            if spans_in(row["prompt"]) != spans or row["prompt"].count(joined) != 1:
                raise ValueError(f"{source['id']}: UMT spans differ from the NTP answer")
            row["prompt"] = row["prompt"].replace(joined, "".join(boxes))
        converted.append(row)
    geometry = []
    for mask, box in zip(masks, boxes):
        x1, y1, x2, y2 = box_coords(box)
        box_area = (x2 - x1) * (y2 - y1) / 1e6
        geometry.append({"box_area": box_area, "fill": float(np.asarray(mask).mean()) / box_area})
    return converted, geometry


def rec_eligible(item, rows, v1_type):
    """Cheap pre-check: single-instance, non-add unit of an instance-accurate source."""
    source, annotation = item["source"], item["annotation"]
    if (annotation is None or source["dataset"] not in REC_DATASETS or v1_type not in REC_TYPES
            or len(annotation["units"]) != 1 or len(annotation["units"][0]["mask_ids"]) != 1):
        return False
    ntp = [row for row in rows if row["sample_type"] == "edit_ntp"]
    return len(ntp) == 1 and len(parse_cot(ntp[0]["mt_cot"], nonempty=True)) == 1


def rec_row(item, rows, v1_type):
    """Grounding replay row: the unit's referring label and its mask's box."""
    ntp = next(row for row in rows if row["sample_type"] == "edit_ntp")
    (_, label), = parse_cot(ntp["mt_cot"], nonempty=True)
    masks = ordered_unit_masks(item["source"], item["annotation"]["units"])[0]
    if len(masks) != 1:
        raise ValueError(f"{item['source']['id']}: replay unit must have one mask")
    return validate_row({"sample_type": "rec_ntp", "edit_type": v1_type,
                         "edit_image": ntp["edit_image"], "prompt": REC_TEMPLATE.format(label),
                         "mt_cot": to_cot([(box_of(pixel_box(masks[0])), label)])})


def quantiles(values):
    if not values:
        return None
    q = np.quantile(np.asarray(values, dtype=np.float64), [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    return dict(zip(("min", "p10", "p25", "p50", "p75", "p90", "max"), (round(float(x), 5) for x in q)))


def _entry(encoded, row, old, conversion):
    return {"id": encoded["id"], "dataset": encoded["dataset"],
            "conversion_method": encoded["conversion_method"], "sample_type": kind_of(row),
            "edit_type": row["edit_type"], "v1_edit_type": old["edit_type"] if old else row["edit_type"],
            "conversion": conversion, "row_hash": row_hash(row),
            "v1_row_hash": row_hash(old) if old else None}


def convert(v1, output, *, rec_per_ntp=7, limit=0, threads=32):
    """Pass 1 reads and classifies every v1 record; pass 2 decodes only the
    masks it needs (add boxes, selected replay rows) on a thread pool, because
    mask PNGs on the shared filesystem dominate the run time."""
    from concurrent.futures import ThreadPoolExecutor

    v1, output = Path(v1).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Use a fresh output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    v1_report = json.loads((v1 / "metadata_report.json").read_text())
    if file_hash(v1 / "provenance.jsonl") != v1_report["provenance_sha256"]:
        raise ValueError("v1 provenance differs from its metadata report")
    v1_provenance = read_jsonl(v1 / "provenance.jsonl")
    drops, records, rec_pool = Counter(), [], []
    ntp_rows = dropped_records = 0
    for item, encoded in v1_records(v1):
        if limit and len(records) + dropped_records >= limit:
            break
        rows, dataset = encoded["rows"], encoded["dataset"]
        for row in rows:  # the exact v1 rows, in v1 merge order
            expected = next(v1_provenance)
            if expected["id"] != encoded["id"] or expected["row_hash"] != row_hash(row):
                raise ValueError(f"v1 encoded rows differ from v1 provenance at {encoded['id']}")
        types = {row["edit_type"] for row in rows}
        if len(types) != 1:
            raise ValueError(f"{encoded['id']}: rows disagree on edit_type")
        v1_type = types.pop()
        if v1_type == "composite":
            drops.update(f"{dataset}:{kind_of(row)}" for row in rows)
            dropped_records += 1
            continue
        if v1_type == "add" and dataset == "derived":
            conversion = "derived_add_as_attribute"
        elif v1_type == "add" and item["annotation"] is not None:
            conversion = "add_box"
        else:
            conversion = "unchanged"
        needs_item = conversion == "add_box" or rec_eligible(item, rows, v1_type)
        record = {"encoded": encoded, "v1_type": v1_type, "conversion": conversion,
                  "item": item if needs_item else None}
        if conversion != "add_box" and needs_item:
            rec_pool.append((hashlib.sha256(f"rec:{encoded['id']}".encode()).hexdigest(), len(records)))
        ntp_rows += sum(row["sample_type"] == "edit_ntp" for row in rows)
        records.append(record)
    if not limit and next(v1_provenance, None) is not None:
        raise ValueError("v1 provenance has rows beyond the encoded records")
    rec_count = min(len(rec_pool), round(ntp_rows / rec_per_ntp))
    selected = sorted(rec_pool)[:rec_count]
    for _, index in selected:
        records[index]["rec"] = True

    def process(record):
        rows, item = record["encoded"]["rows"], record["item"]
        geometry = None
        if record["conversion"] == "derived_add_as_attribute":
            new_rows = [{**row, "edit_type": "attribute"} for row in rows]
        elif record["conversion"] == "add_box":
            new_rows, geometry = add_boxes(item, rows)
        else:
            new_rows = rows
        rec = rec_row(item, rows, record["v1_type"]) if record.get("rec") else None
        return new_rows, geometry, rec

    with ThreadPoolExecutor(threads) as pool:
        results = list(pool.map(process, records, chunksize=64))
    counts, conversions = Counter(), Counter()
    add_geometry, instances_per_add = defaultdict(list), Counter()
    stage1, stage2, replay = [], [], {}
    for record, (new_rows, geometry, rec) in zip(records, results):
        encoded = record["encoded"]
        conversions[f"{encoded['dataset']}:{record['conversion']}"] += 1
        if geometry is not None:
            add_geometry[encoded["dataset"]].extend(geometry)
            instances_per_add[len(geometry)] += 1
        for old, row in zip(encoded["rows"], new_rows):
            validate_row(row)
            (stage1 if row["sample_type"] == "edit_ntp" else stage2).append(
                (row, _entry(encoded, row, old, record["conversion"])))
            counts[(encoded["dataset"], kind_of(row), row["edit_type"])] += 1
        if rec is not None:
            replay[encoded["id"]] = (rec, _entry(encoded, rec, None, "rec_replay"))
    for _, index in selected:  # deterministic hash order, after all edit_ntp rows
        rec, entry = replay[records[index]["encoded"]["id"]]
        stage1.append((rec, entry))
        counts[(entry["dataset"], "rec_ntp", rec["edit_type"])] += 1
    provenance = []
    for name, rows in (("stage1", stage1), ("stage2", stage2)):
        path = output / f"{name}.jsonl"
        with path.with_suffix(".tmp").open("w") as stream:
            for line, (row, entry) in enumerate(rows):
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                provenance.append({"file": name, "line": line, **entry})
        path.with_suffix(".tmp").replace(path)
    with (output / "provenance.tmp").open("w") as stream:
        for entry in provenance:
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
    (output / "provenance.tmp").replace(output / "provenance.jsonl")
    by_kind = Counter()
    for (dataset, kind, edit_type), n in counts.items():
        by_kind[kind] += n
    report = {
        "format": FORMAT, "code_commit": code_commit(), "v1_data": str(v1), "records_kept": len(records),
        "records_dropped_composite": dropped_records, "limit": limit,
        "v1_identity": {k: v1_report[k] for k in ("stage1_sha256", "stage2_sha256",
                                                  "provenance_sha256", "codec_sha256")},
        "stage1_rows": len(stage1), "stage2_rows": len(stage2), "rows_by_kind": dict(by_kind),
        "rows_by_dataset_kind_type": {f"{d}:{k}:{t}": n for (d, k, t), n in sorted(counts.items())},
        "records_by_conversion": dict(sorted(conversions.items())),
        "dropped_composite_rows": dict(sorted(drops.items())),
        "add_instances_per_unit": {str(k): v for k, v in sorted(instances_per_add.items())},
        "add_box_area_fraction": {d: quantiles([g["box_area"] for g in v]) for d, v in add_geometry.items()},
        "add_mask_fill_of_box": {d: quantiles([g["fill"] for g in v]) for d, v in add_geometry.items()},
        "rec_replay": {"candidates": len(rec_pool), "selected": rec_count, "rec_per_ntp": rec_per_ntp,
                       "datasets": list(REC_DATASETS), "edit_types": list(REC_TYPES),
                       "selection": "lowest sha256('rec:'+id)"},
        "stage1_sha256": file_hash(output / "stage1.jsonl"),
        "stage2_sha256": file_hash(output / "stage2.jsonl"),
        "provenance_sha256": file_hash(output / "provenance.jsonl"),
        "codec_sha256": v1_report["codec_sha256"],
    }
    write_json(output / "conversion_report.json", report)
    write_json(output / "metadata_report.json", {
        "format": FORMAT, "stage1_rows": len(stage1), "stage2_rows": len(stage2),
        "rows_by_kind": dict(by_kind), "stage1_sha256": report["stage1_sha256"],
        "stage2_sha256": report["stage2_sha256"], "provenance_sha256": report["provenance_sha256"],
        "codec_sha256": report["codec_sha256"], "training_ready": True})
    print(json.dumps({k: report[k] for k in ("stage1_rows", "stage2_rows", "rows_by_kind",
                                             "records_by_conversion", "rec_replay")}), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--v1-data", default=V1_DATA)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rec-per-ntp", type=int, default=7,
                        help="One rec_ntp row per this many edit_ntp rows (Stage 1 ratio 7:1)")
    parser.add_argument("--limit", type=int, default=0, help="Convert only the first N sources (debug)")
    args = parser.parse_args(argv)
    convert(args.v1_data, args.output, rec_per_ntp=args.rec_per_ntp, limit=args.limit)


if __name__ == "__main__":
    main()
