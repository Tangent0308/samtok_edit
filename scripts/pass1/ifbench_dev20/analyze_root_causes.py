"""Read-only audit of frozen Pass1 results and training exposure; no inference."""

import csv
import json
from collections import Counter

import numpy as np
import pyarrow.parquet as pq
from common import FORMAL, ROOT, read_jsonl, sha256, write_json

OUT = ROOT / "reports/root_causes_20261031"
AREA_EDGES = [0.0005, 0.001, 0.005, 0.01, 0.05]
AREA_LABELS = ["<0.05%", "0.05–0.1%", "0.1–0.5%", "0.5–1%", "1–5%", ">=5%"]


def area_bin(value):
    return sum(value >= x for x in AREA_EDGES)


def mean(values):
    return float(np.mean(values)) if len(values) else None


def cluster_interval(rows, key, seed=20261031):
    groups = sorted(set(x["group"] for x in rows))
    sums = np.array([sum(x[key] for x in rows if x["group"] == g) for g in groups])
    counts = np.array([sum(x["group"] == g for x in rows) for g in groups])
    rng = np.random.default_rng(seed)
    picks = rng.integers(len(groups), size=(5000, len(groups)))
    samples = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    return [float(x) for x in np.quantile(samples, [0.025, 0.975])]


def benchmark_audit():
    inputs = {r["id"]: r for r in read_jsonl(ROOT / "data/inference_public.jsonl")}
    results, flat = {}, {}
    pairs = []
    for model in ("base", "step2000"):
        requests = read_jsonl(ROOT / f"eval/{model}/score/combined.jsonl")
        targets = [t for r in requests for t in r["targets"]]
        flat[model] = targets
        scores = {(t["id"], t["label"]): t for t in targets}
        for t in targets:
            case, setting = t["id"].split("--")
            if setting not in ("S4-ref", "S4-noref", "S5") or t["locator"] == "mask":
                continue
            counterpart = (
                "S1"
                if t["locator"] == "none"
                else (("S2-" if t["has_ref"] else "S3-") + t["locator"])
            )
            other_id = case + "--" + counterpart
            own = next(
                x
                for x in inputs[t["id"]]["request"]["targets"]
                if x["label"] == t["label"]
            )
            other = next(
                x
                for x in inputs[other_id]["request"]["targets"]
                if x["label"] == t["label"]
            )
            assert own == other, (t["id"], t["label"], other_id)
            ref = scores[other_id, t["label"]]
            pairs.append(
                dict(
                    model=model,
                    group=t["group"],
                    request_id=t["id"],
                    label=t["label"],
                    setting=setting,
                    counterpart=other_id,
                    mode=t["mode"],
                    own_input_identical=True,
                    mixed_iou=t["iou"],
                    homogeneous_iou=ref["iou"],
                    delta=t["iou"] - ref["iou"],
                )
            )
        unknown = [t for t in targets if t["locator"] != "mask"]
        known = [t for t in targets if t["locator"] == "mask"]
        results[model] = dict(
            requests=len(requests),
            targets=len(targets),
            unknown_outputs=len(unknown),
            strict_requests=sum(r["strict"] for r in requests),
            unknown_raw_iou=mean([t["iou"] for t in unknown]),
            unknown_compatible_iou=mean([t["compatible_iou"] for t in unknown]),
            known_exact_copy=sum(t["exact_code"] for t in known),
            known_outputs=len(known),
            point_hits=sum(t["point_hit"] for t in unknown if t["locator"] == "point"),
            point_outputs=sum(t["locator"] == "point" for t in unknown),
            possible_requested_target_swaps=sum(t["possible_id_swap"] for t in targets),
        )
        results[model]["matched_mixing"] = {}
        for setting in ("S4-ref", "S4-noref", "S5"):
            subset = [
                x for x in pairs if x["model"] == model and x["setting"] == setting
            ]
            results[model]["matched_mixing"][setting] = dict(
                outputs=len(subset),
                sources=len(set(x["group"] for x in subset)),
                mixed_iou=mean([x["mixed_iou"] for x in subset]),
                matched_homogeneous_iou=mean([x["homogeneous_iou"] for x in subset]),
                mean_delta=mean([x["delta"] for x in subset]),
                delta_source_bootstrap95=cluster_interval(subset, "delta"),
                changes_over_005=sum(abs(x["delta"]) > 0.05 for x in subset),
            )
    trained = flat["step2000"]
    unknown = [t for t in trained if t["locator"] != "mask"]
    unique = {(t["group"], t["label"]): t for t in trained}
    failures = [t for t in unknown if t["iou"] < 0.5]
    breakdown = dict(
        unique_targets=len(unique),
        sources=len(set(t["group"] for t in trained)),
        canonical_reference_iou=mean([t["canonical_iou"] for t in unique.values()]),
        canonical_below05=sum(t["canonical_iou"] < 0.5 for t in unique.values()),
        canonical_below07=sum(t["canonical_iou"] < 0.7 for t in unique.values()),
        unknown_below05_outputs=len(failures),
        unknown_below05_unique_targets=len(
            set((t["group"], t["label"]) for t in failures)
        ),
        low_outputs_and_canonical_below05=sum(
            t["canonical_iou"] < 0.5 for t in failures
        ),
        low_outputs_and_canonical_atleast07=sum(
            t["canonical_iou"] >= 0.7 for t in failures
        ),
        predictions_above_canonical_by005=sum(
            t["iou"] > t["canonical_iou"] + 0.05 for t in unknown
        ),
        unknown_iou50_rate=mean([t["iou"] >= 0.5 for t in unknown]),
        unknown_iou75_rate=mean([t["iou"] >= 0.75 for t in unknown]),
        benchmark_unique_area_bins=dict(
            Counter(AREA_LABELS[area_bin(t["area"])] for t in unique.values())
        ),
        unknown_by_area={},
        unknown_by_operation={},
        unknown_by_locator={},
    )
    for kind, labels, field in [
        ("area", AREA_LABELS, lambda t: AREA_LABELS[area_bin(t["area"])]),
        (
            "operation",
            sorted(set(t["operation"] for t in unknown)),
            lambda t: t["operation"],
        ),
        ("locator", ["none", "point", "box"], lambda t: t["locator"]),
    ]:
        for label in labels:
            subset = [t for t in unknown if field(t) == label]
            breakdown["unknown_by_" + kind][label] = dict(
                outputs=len(subset),
                unique_targets=len(set((t["group"], t["label"]) for t in subset)),
                iou=mean([t["iou"] for t in subset]),
                canonical=mean([t["canonical_iou"] for t in subset]),
                precision=mean([t["precision"] for t in subset]),
                recall=mean([t["recall"] for t in subset]),
                below05=sum(t["iou"] < 0.5 for t in subset),
            )
    examples = []
    wanted = [
        ("v2-sa_455310--S1", "U2"),
        ("v2-sa_8415518--S1", "U4"),
        ("v2-sa_5447626--S2-point", "U4"),
        ("v2-sa_5447626--S3-point", "U4"),
        ("v2-sa_10831959--S1", "U5"),
        ("v2-sa_5831338--S2-mask", "U4"),
    ]
    for uid, label in wanted:
        before = next(t for t in flat["base"] if t["id"] == uid and t["label"] == label)
        after = next(t for t in trained if t["id"] == uid and t["label"] == label)
        prompt = next(
            t for t in inputs[uid]["request"]["targets"] if t["label"] == label
        )
        examples.append(
            dict(
                request_id=uid,
                label=label,
                prompt=prompt,
                before=before["iou"],
                after=after["iou"],
                reference=after["canonical_iou"],
                area=after["area"],
                precision=after["precision"],
                recall=after["recall"],
                point_hit=after.get("point_hit"),
            )
        )
    with (OUT / "matched_mixing.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    return dict(
        models=results,
        failure_breakdown=breakdown,
        examples=examples,
        interpretation="Both sides of mixing comparison are joint multi-target requests. Only own target public input is identical; other targets may differ. Raw IoU avoids request-format gating. Repeated setting outputs are not independent targets. Canonical codec roundtrip is a reference, not an upper bound.",
    )


def training_audit():
    plan = json.loads((FORMAL / "data/prepared/batch_plan.json").read_text())
    seen_indices = {i for b in plan["batches"][:2000] for i in b["indices"] if i >= 0}
    all_ids, seen_ids, seen_pair_rows = set(), set(), Counter()
    for i, line in enumerate((FORMAL / "data/prepared/train.jsonl").open()):
        r = json.loads(line)
        all_ids.add(r["id"])
        if i in seen_indices:
            seen_ids.add(r["id"])
            if r.get("pair_id"):
                seen_pair_rows[r["pair_id"]] += 1
    assert len(seen_ids) == 127829
    cols = [
        "key",
        "area",
        "roundtrip",
        "operation",
        "has_site",
        "parts",
        "family",
        "origin",
        "scope",
        "target_part",
    ]
    table = pq.read_table(
        FORMAL / "data/selection/selected_regions.parquet", columns=cols
    )
    regions = {r["key"]: r for r in table.to_pylist()}
    stats = {
        name: dict(
            requests=0,
            targets=0,
            area_bins=Counter(),
            quality_bins=Counter(),
            modes=Counter(),
            point_policies=Counter(),
            origins=Counter(),
            operation_modes=Counter(),
            host_fallback_modes=Counter(),
            dense_parent_modes=Counter(),
            dense_targets=0,
            counterfactual_requests=0,
            K=Counter(),
        )
        for name in ("seen_step2000", "seen_step2000_main", "full_warmup", "full_main")
    }
    for stage in ("warmup", "main"):
        with (FORMAL / f"data/selection/{stage}.selection.jsonl").open() as f:
            for line in f:
                row = json.loads(line)
                if row["id"] not in all_ids:
                    continue
                names = ["full_" + stage]
                if row["id"] in seen_ids:
                    names.append("seen_step2000")
                    if stage == "main":
                        names.append("seen_step2000_main")
                for name in names:
                    s = stats[name]
                    s["requests"] += 1
                    s["K"][str(row["K"])] += 1
                    s["counterfactual_requests"] += bool(row.get("pair_id"))
                    for t in row["targets"]:
                        r = regions[t["region_key"]]
                        mode = t["mode"]
                        s["targets"] += 1
                        s["area_bins"][AREA_LABELS[area_bin(r["area"])]] += 1
                        q = r["roundtrip"]
                        s["quality_bins"][
                            "<.7" if q < 0.7 else ".7–.8" if q < 0.8 else ">=.8"
                        ] += 1
                        s["modes"][mode] += 1
                        s["origins"][r["family"] or ("pool_" + str(r["origin"]))] += 1
                        s["operation_modes"][str(r["operation"]) + "|" + mode] += 1
                        if mode.endswith("_point"):
                            s["point_policies"][t["locator_policy"]] += 1
                        dense = r["key"].startswith("P:")
                        s["dense_targets"] += dense
                        if dense and r["parts"]:
                            s["dense_parent_modes"][mode] += 1
                        if r["operation"] == "add" and r["has_site"]:
                            s["host_fallback_modes"][mode] += 1
    for s in stats.values():
        s["dense_target_fraction"] = s["dense_targets"] / s["targets"]
        s["area_below1pct_fraction"] = (
            sum(s["area_bins"][x] for x in AREA_LABELS[:4]) / s["targets"]
        )
        s["area_below005pct_fraction"] = s["area_bins"][AREA_LABELS[0]] / s["targets"]
    return dict(
        stats=stats,
        seen_step2000_requests=len(seen_ids),
        seen_counterfactual_pairs=len(seen_pair_rows),
        pair_row_counts=dict(Counter(seen_pair_rows.values())),
        all_training_requests=len(all_ids),
        unused_small_low_quality_reserve=pq.ParquetFile(
            FORMAL / "data/selection/small_lowq_reserve.parquet"
        ).metadata.num_rows,
        exposure_definition="Observed train.jsonl IDs intersected with first 2000 frozen batch-plan indices; target metadata comes from original selection and selected_regions. No RLE, image, labels, model or prior results changed.",
    )


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    benchmark = benchmark_audit()
    print("Benchmark audit complete", flush=True)
    training = training_audit()
    write_json(OUT / "analysis.json", dict(benchmark=benchmark, training=training))
    write_json(
        OUT / "provenance.json",
        dict(
            script=__file__,
            script_sha256=sha256(__file__),
            inputs={
                str(p): sha256(p)
                for p in [
                    ROOT / "data/inference_public.jsonl",
                    ROOT / "eval/base/score/combined.jsonl",
                    ROOT / "eval/step2000/score/combined.jsonl",
                    FORMAL / "data/prepared/batch_plan.json",
                ]
            },
            inference_run=False,
            training_run=False,
            original_artifacts_modified=False,
        ),
    )
    print(
        json.dumps(
            dict(output=str(OUT), seen_requests=training["seen_step2000_requests"]),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
