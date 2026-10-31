"""Materialize frozen region selections; raw data and images remain read-only.

Use --limit-per-split for a bounded CPU preflight. No segmentation model, token
encoder, LLM, or GPU is called. Cached canonical labels are reused verbatim.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from pycocotools import mask as mu
from scipy.ndimage import distance_transform_edt

from samtok_edit21.pass1.protocol import (
    VERSION,
    box_normalized,
    parse_answer,
    point_normalized,
    point_pixels,
    validate_request,
)


def decode(rle):
    counts = rle["counts"].encode() if isinstance(rle["counts"], str) else rle["counts"]
    return mu.decode({"size": rle["size"], "counts": counts}).astype(bool)


def sample_point(mask, decoded, size, *, policy, seed, training):
    """Evaluation never conditions clicks on the cached teacher decode."""
    ys, xs = np.where(mask)
    if not len(xs):
        raise ValueError("empty_selection")
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    crop = mask[y0:y1, x0:x1]
    dist = distance_transform_edt(np.pad(crop, 1))[1:-1, 1:-1]
    values = dist[ys - y0, xs - x0]
    allowed = np.ones(len(xs), dtype=bool)
    if training:
        allowed &= decoded[ys, xs]
    indices = np.where(allowed)[0]
    if not len(indices):
        raise ValueError("no_teacher_consistent_training_point")
    rng = np.random.default_rng(seed)
    if policy == "deep":
        # Stable deepest candidates; do not turn an invalid deep click into an
        # unreported uniformly sampled easier example.
        indices = indices[np.argsort(-values[indices], kind="stable")][:256]
    elif policy == "inner_boundary":
        cutoff = np.quantile(values, 0.20)
        indices = indices[values[indices] <= cutoff]
        if not len(indices):
            raise ValueError("no_teacher_consistent_boundary_point")
        indices = rng.permutation(indices)[:512]
    elif policy == "uniform":
        indices = rng.permutation(indices)[:512]
    else:
        raise ValueError("unknown_point_policy")
    for i in indices:
        point = point_normalized([int(xs[i]), int(ys[i])], size)
        x, y = point_pixels(point, size)
        if mask[y, x] and (not training or decoded[y, x]):
            return point, {
                "pixel_before_quantization": [int(xs[i]), int(ys[i])],
                "pixel_after_quantization": [x, y],
                "teacher_contains_point": bool(decoded[y, x]),
                "policy": policy,
                "training_teacher_filter": training,
            }
    raise ValueError("no_quantization_safe_" + policy + "_point")


def sample_box(bbox, size, *, policy, seed, training):
    box = list(map(int, bbox))
    if policy == "outward_jitter_5pct":
        if not training:
            raise ValueError("evaluation_box_jitter_forbidden")
        rng = random.Random(seed)
        w, h = size
        dx, dy = (box[2] - box[0]) * 0.05, (box[3] - box[1]) * 0.05
        box = [
            max(0, box[0] - round(rng.random() * dx)),
            max(0, box[1] - round(rng.random() * dy)),
            min(w, box[2] + round(rng.random() * dx)),
            min(h, box[3] + round(rng.random() * dy)),
        ]
    elif policy != "tight":
        raise ValueError("unknown_box_policy")
    return box_normalized(box, size), {"pixel_box": box, "policy": policy}


def load_supervision(region_map, source):
    selected = collections.defaultdict(dict)
    for key, row in region_map.items():
        if key.startswith("M:"):
            selected[row["record"]][row["uid"]] = key
    result = {}
    if selected:
        with (source / "records.jsonl").open() as f:
            for line in f:
                head = line.split('"', 4)
                if len(head) < 4 or head[3] not in selected:
                    continue
                record = json.loads(line)
                for unit in record["units"]:
                    key = selected[record["id"]].get(unit["uid"])
                    if key:
                        sel = unit["selection"]
                        result[key] = {
                            "selection": {
                                "size": sel["rle_size"],
                                "counts": sel["rle_counts"],
                            },
                            "canonical_dec": unit["token"]["dec"],
                            "span": unit["token"]["span"],
                        }
    pools = collections.defaultdict(list)
    for key, row in region_map.items():
        if key.startswith("P:"):
            pools[row["pool_file"]].append((key, row["index"]))
    for path, keys in pools.items():
        data = json.loads(Path(path).read_text())
        for key, index in keys:
            unit = data["masks"][index]
            result[key] = {
                "selection": unit["rle"],
                "canonical_dec": unit["dec"],
                "span": unit["span"],
            }
    if set(result) != set(region_map):
        raise ValueError("Missing selected supervision rows")
    for key, data in result.items():
        counts = data["selection"]["counts"]
        digest = hashlib.sha1(
            counts.encode() if isinstance(counts, str) else counts
        ).hexdigest()
        if (
            digest != region_map[key]["s_digest"]
            or data["span"] != region_map[key]["span"]
        ):
            raise ValueError("Source annotation changed after selection: " + key)
    return result


def materialize(plan, region_map, supervision, sources, hashes):
    source = sources[plan["source"]]
    size = [source["w"], source["h"]]
    training = plan["split"] == "train"
    targets, answer, truth, diagnostics = [], [], [], []
    for target in plan["targets"]:
        key = target["region_key"]
        r = region_map[key]
        teacher_key = r.get("canonical_label_key") or key
        teacher = region_map[teacher_key]
        if teacher["source"] != r["source"] or teacher["s_digest"] != r["s_digest"]:
            raise ValueError("canonical_teacher_does_not_select_identical_S")
        g = supervision[teacher_key]
        mode = target["mode"]
        has_ref, locator = mode.split("_", 1)
        if mode not in r["modes"]:
            raise ValueError("mode_not_eligible")
        mask, dec = decode(g["selection"]), decode(g["canonical_dec"])
        if mask.shape != tuple(size[::-1]) or dec.shape != mask.shape:
            raise ValueError("mask_image_size_mismatch")
        actual_rt = float(
            np.logical_and(mask, dec).sum() / max(1, np.logical_or(mask, dec).sum())
        )
        if abs(actual_rt - teacher["roundtrip"]) > 0.002:
            raise ValueError("cached_roundtrip_mismatch")
        if key.startswith("P:"):
            instruction = "Return the segmentation of the selected region."
        else:
            instruction = (
                r["ref_instruction"] if has_ref == "ref" else r["noref_instruction"]
            )
        public = {
            "label": target["label"],
            "instruction": instruction,
            "has_ref": has_ref == "ref",
            "locator": locator,
        }
        diag = {
            "label": target["label"],
            "region_key": key,
            "area": float(mask.mean()),
            "roundtrip": actual_rt,
            "mode": mode,
            "canonical_label_key": teacher_key,
        }
        if locator == "point":
            point, info = sample_point(
                mask,
                dec,
                size,
                policy=target["locator_policy"],
                seed=target["geometry_seed"],
                training=training,
            )
            public["region"] = {"point": point}
            diag.update(info)
        elif locator == "box":
            box, info = sample_box(
                r["bbox"],
                size,
                policy=target["locator_policy"],
                seed=target["geometry_seed"],
                training=training,
            )
            public["region"] = {"box": box}
            diag.update(info)
        elif locator == "mask":
            public["region"] = {"mask_2d": g["span"]}
        targets.append(public)
        answer.append({"label": target["label"], "mask_2d": g["span"]})
        truth.append(
            {
                **g,
                "area": float(mask.mean()),
                "roundtrip": actual_rt,
                "label": target["label"],
                "family": r.get("family") or "dense",
                "record": r.get("record") or key,
                "original_uid": r.get("uid") or r.get("oid"),
                "operation": r.get("operation") or "segmentation",
            }
        )
        diagnostics.append(diag)
    request = validate_request(
        {
            "protocol": VERSION,
            "coordinate_system": "normalized_0_1000",
            "targets": targets,
        }
    )
    if not parse_answer(json.dumps(answer), request)["complete"]:
        raise ValueError("invalid_supervised_answer")
    result = {
        "id": plan["id"],
        "split": plan["split"],
        "group": plan["source"],
        "image": source["image"],
        "image_sha256": hashes[source["image"]]["sha256"],
        "size": size,
        "request": request,
        "answer": answer,
        "truth": truth,
        "diagnostics": diagnostics,
    }
    for field in ("pair_id", "pair_role", "contrast_label"):
        if field in plan:
            result[field] = plan[field]
    return result


def bounded_subset(rows, n, rng):
    """Keep both members of selected counterfactual pairs in a CPU preflight."""
    if not n or len(rows) <= n:
        return rows
    pairs = collections.defaultdict(list)
    singles = []
    for row in rows:
        if row.get("pair_id"):
            pairs[row["pair_id"]].append(row)
        else:
            singles.append(row)
    chosen = []
    pair_keys = sorted(pairs)
    rng.shuffle(pair_keys)
    for key in pair_keys[: min(n // 8, len(pair_keys))]:
        if len(pairs[key]) == 2:
            chosen.extend(pairs[key])
    rng.shuffle(singles)
    chosen.extend(singles[: n - len(chosen)])
    return chosen


def conflicting_requests(rows):
    """Return ALL sides of contradictory public inputs, independent of ID/context."""
    requests = collections.defaultdict(lambda: collections.defaultdict(set))
    targets = collections.defaultdict(lambda: collections.defaultdict(set))
    for row in rows:
        signature = hashlib.sha256(
            (row["image_sha256"] + json.dumps(row["request"], sort_keys=True)).encode()
        ).hexdigest()
        requests[signature][json.dumps(row["answer"], sort_keys=True)].add(row["id"])
        for target, answer in zip(row["request"]["targets"], row["answer"]):
            public = {k: v for k, v in target.items() if k != "label"}
            signature = hashlib.sha256(
                (row["image_sha256"] + json.dumps(public, sort_keys=True)).encode()
            ).hexdigest()
            targets[signature][answer["mask_2d"]].add(row["id"])
    bad, events = set(), []
    for kind, mapping in [("request", requests), ("target", targets)]:
        for signature, answers in mapping.items():
            if len(answers) > 1:
                ids = sorted(set().union(*answers.values()))
                bad.update(ids)
                events.append(
                    {
                        "kind": kind,
                        "public_sha256": signature,
                        "request_ids": ids,
                        "answer_variants": len(answers),
                    }
                )
    return bad, events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--merged-source", type=Path, required=True)
    ap.add_argument("--image-hashes", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--limit-per-split", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--quarantine-conflicts", action="store_true")
    args = ap.parse_args()
    started = time.time()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Materialization output must be empty")
    args.output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(2026103004)
    plans = []
    for name in ("warmup", "main", "dev", "test"):
        rows = [
            json.loads(x)
            for x in (args.selection / (name + ".selection.jsonl"))
            .read_text()
            .splitlines()
        ]
        plans.extend(bounded_subset(rows, args.limit_per_split, rng))
    keys = {t["region_key"] for r in plans for t in r["targets"]}
    all_regions = {
        r["key"]: r
        for r in pq.read_table(args.selection / "selected_regions.parquet").to_pylist()
    }
    keys |= {all_regions[k].get("canonical_label_key") or k for k in list(keys)}
    region_map = {k: all_regions[k] for k in keys}
    sources = {
        r["source"]: r
        for r in pq.read_table(args.selection / "selected_sources.parquet").to_pylist()
    }
    hashes = {r["path"]: r for r in pq.read_table(args.image_hashes).to_pylist()}
    supervision = load_supervision(region_map, args.merged_source)
    print(
        json.dumps(
            {
                "planned_requests": len(plans),
                "selected_annotation_rows": len(region_map),
            }
        ),
        flush=True,
    )

    def one(plan):
        try:
            return materialize(plan, region_map, supervision, sources, hashes), None
        except ValueError as error:
            return None, {
                "id": plan["id"],
                "split": plan["split"],
                "stage": plan["stage"],
                "pair_id": plan.get("pair_id"),
                "reason": str(error),
            }

    outputs, failures = [], []
    with concurrent.futures.ThreadPoolExecutor(args.workers) as executor:
        for i, (row, error) in enumerate(executor.map(one, plans), 1):
            if error:
                failures.append(error)
            else:
                outputs.append(row)
            if i % 100 == 0:
                print(
                    json.dumps(
                        {
                            "processed": i,
                            "failed": len(failures),
                            "elapsed_s": time.time() - started,
                        }
                    ),
                    flush=True,
                )
    # A failed partner invalidates the entire contrastive pair. Never leave a
    # nominal paired dataset with one-sided or identical-coordinate examples.
    pair_failures = {x["pair_id"] for x in failures if x.get("pair_id")}
    pairs = collections.defaultdict(list)
    for r in outputs:
        if r.get("pair_id"):
            pairs[r["pair_id"]].append(r)
    for pair, rs in pairs.items():
        if len(rs) != 2:
            pair_failures.add(pair)
            continue
        a, b = rs
        label = a["contrast_label"]
        aa, bb = a["request"]["targets"], b["request"]["targets"]
        changed = [i for i, (x, y) in enumerate(zip(aa, bb)) if x != y]
        if len(changed) != 1 or aa[changed[0]]["label"] != label:
            pair_failures.add(pair)
            failures.append(
                {
                    "id": pair,
                    "reason": "counterfactual_public_inputs_do_not_differ_at_exactly_one_target",
                }
            )
    outputs = [r for r in outputs if r.get("pair_id") not in pair_failures]
    conflict_ids, conflict_events = conflicting_requests(outputs)
    (args.output / "public_conflict_audit.json").write_text(
        json.dumps(conflict_events, indent=2) + "\n"
    )
    if conflict_ids and not args.quarantine_conflicts:
        raise ValueError("Contradictory public inputs; see public_conflict_audit.json")
    if conflict_ids:
        conflict_pairs = {
            r["pair_id"]
            for r in outputs
            if r["id"] in conflict_ids and r.get("pair_id")
        }
        pair_failures |= conflict_pairs
        for r in outputs:
            if r["id"] in conflict_ids or r.get("pair_id") in conflict_pairs:
                failures.append(
                    {
                        "id": r["id"],
                        "split": r["split"],
                        "pair_id": r.get("pair_id"),
                        "reason": "public_input_conflict"
                        if r["id"] in conflict_ids
                        else "conflicting_pair_partner",
                    }
                )
        outputs = [
            r
            for r in outputs
            if r["id"] not in conflict_ids and r.get("pair_id") not in conflict_pairs
        ]
    by_id = {r["id"]: r for r in outputs}
    stats = {}
    for name in ("warmup", "main", "dev", "test"):
        rows = (
            [by_id[p["id"]] for p in plans if p["stage"] == name and p["id"] in by_id]
            if name in ("warmup", "main")
            else [r for r in outputs if r["split"] == name]
        )
        path = args.output / (name + ".jsonl")
        with path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
        ds = [d for r in rows for d in r["diagnostics"]]
        stats[name] = {
            "requests": len(rows),
            "targets": len(ds),
            "point_targets": sum(d["mode"].endswith("_point") for d in ds),
            "teacher_misses_eval_point": sum(
                d.get("teacher_contains_point") is False for d in ds
            ),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    manifest = {
        "status": "bounded_geometry_preflight"
        if args.limit_per_split
        else "materialized",
        "planned_requests": len(plans),
        "accepted_requests": len(outputs),
        "stats": stats,
        "failures": failures,
        "invalidated_pairs": sorted(pair_failures),
        "valid_counterfactual_pairs": len(
            {r["pair_id"] for r in outputs if r.get("pair_id")}
        ),
        "quarantined_conflict_requests": len(conflict_ids),
        "conflict_groups_before_quarantine": len(conflict_events),
        "contradictory_public_inputs": 0,
        "contradictory_public_targets": 0,
        "elapsed_s": time.time() - started,
        "no_model_or_gpu_used": True,
        "evaluation_points": "Sampled from S only; teacher misses retained and reported, never repaired using decoded mask",
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
