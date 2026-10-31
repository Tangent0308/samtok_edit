"""Freeze official dev20, project public inputs, audit without generating masks."""

import copy
import json
import subprocess
from collections import Counter

import numpy as np
from adapter import adapt
from common import (
    ADAPTER,
    BUNDLE,
    DATASET,
    FORMAL,
    REPO,
    ROOT,
    read_jsonl,
    rle,
    sha256,
    write_json,
    write_jsonl,
)
from PIL import Image
from samtok_benchmark.ifbench import MANIFEST_SHA256, SETTINGS
from samtok_benchmark.ifbench.plan import assignment, split_cases
from samtok_benchmark.ifbench.public import project, verify
from samtok_benchmark.io import digest

from samtok_edit21.pass1.protocol import parse_answer, point_normalized, point_pixels


def main():
    manifest = REPO / "data/v2_2.6.0/cases.jsonl"
    assert sha256(manifest) == MANIFEST_SHA256
    cases = read_jsonl(manifest)
    by_id = {x["id"]: x for x in cases}
    selection = json.loads((BUNDLE / "selection.json").read_text())
    split = split_cases(cases)
    split["manifest_sha256"] = MANIFEST_SHA256
    assert digest(split) == selection["parent_split_digest"]
    ids = selection["case_ids"]
    assert len(ids) == len(set(ids)) == 20
    assert set(ids) <= set(split["dev"]) and not set(ids) & set(split["test"])
    selected = [by_id[x] for x in ids]
    assert sum(len(c["units"]) for c in selected) == 72
    assert (
        dict(Counter(str(len(c["units"])) for c in selected))
        == selection["selected_K_counts"]
    )
    write_json(ROOT / "provenance/selection.json", selection)
    write_json(
        ROOT / "provenance/assignments.json", {c["id"]: assignment(c) for c in selected}
    )
    publics, truth, geometry = [], [], []
    for case in selected:
        assigned = assignment(case)
        for setting in SETTINGS:
            public = project(case, setting, assigned, DATASET)
            verify(public, assets=True)
            publics.append(public)
        units = []
        for u in case["units"]:
            target = u["target"]
            with Image.open(DATASET / target["mask"]) as im:
                mask = np.asarray(im.convert("L")) > 0
            p = target["point"]
            pn = point_normalized(p, case["source"]["size"])
            pr = point_pixels(pn, case["source"]["size"])
            geometry.append(
                {
                    "case_id": case["id"],
                    "label": u["id"],
                    "source_point": p,
                    "normalized_point": pn,
                    "roundtrip_point": pr,
                    "original_in_gt": bool(mask[p[1], p[0]]),
                    "quantized_in_gt": bool(mask[pr[1], pr[0]]),
                    "area": float(mask.mean()),
                }
            )
            units.append(
                {
                    "label": u["id"],
                    "mask_path": str(DATASET / target["mask"]),
                    "mask_sha256": target["mask_sha256"],
                    "operation": u["operation"],
                    "selection": rle(mask),
                    "area": float(mask.mean()),
                    "scope": target.get("scope"),
                    "ref": target["ref"],
                    "instruction_ref": u["instruction_ref"],
                    "instruction_noref": u["instruction_noref"],
                }
            )
        truth.append(
            {
                "id": case["id"],
                "image": str(DATASET / case["source"]["image"]),
                "image_sha256": case["source"]["sha256"],
                "size": case["source"]["size"],
                "group": case["source"]["family_id"],
                "units": units,
            }
        )
    write_jsonl(ROOT / "data/benchmark_public.jsonl", publics)
    write_jsonl(ROOT / "data/offline_gt.jsonl", truth)
    write_jsonl(ROOT / "provenance/coordinate_audit.jsonl", geometry)

    # Verify adapter information boundary and ID integrity for every setting.
    dummy = "<|mt_start|><|mt_0001|><|mt_0257|><|mt_end|>"

    class TrackingCache(dict):
        def __init__(self):
            super().__init__()
            self.reads = []

        def __getitem__(self, key):
            self.reads.append(key)
            return dummy

    for public in publics:
        cache = TrackingCache()
        request = adapt(public, cache)
        assert len(cache.reads) == sum(u["locator"] == "mask" for u in public["units"])
        for u, t in zip(public["units"], request["targets"]):
            assert u["instruction"] == t["instruction"] and u["id"] == t["label"]
            assert u["locator"] == t["locator"] and u["has_ref"] == t["has_ref"]
            if u["locator"] == "none":
                assert "region" not in t
        answer = [{"label": t["label"], "mask_2d": dummy} for t in request["targets"]]
        assert parse_answer(json.dumps(answer), request)["complete"]
        assert not parse_answer(json.dumps(answer[::-1]), request)["complete"]
        for field, value in [
            ("target", {"mask": "/private/gt.png"}),
            ("ref", "hidden object"),
        ]:
            poisoned = copy.deepcopy(public)
            poisoned["units"][0][field] = value
            try:
                adapt(poisoned, cache)
            except ValueError:
                pass
            else:
                raise AssertionError("Private field accepted")
    for case_id in ids:
        ref = next(
            p for p in publics if p["case_id"] == case_id and p["setting"] == "S4-ref"
        )
        nr = next(
            p for p in publics if p["case_id"] == case_id and p["setting"] == "S4-noref"
        )
        assert [(u["locator"], u["region"]) for u in ref["units"]] == [
            (u["locator"], u["region"]) for u in nr["units"]
        ]

    # Check the actual training inputs, not merely a source-pool exclusion claim.
    benchmark_hashes = {c["source"]["sha256"] for c in selected}
    train_hashes, overlaps, n = set(), [], 0
    with (FORMAL / "data/prepared/train.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            n += 1
            train_hashes.add(row["image_sha256"])
            if row["image_sha256"] in benchmark_hashes:
                overlaps.append(row["id"])
    assert not overlaps, overlaps
    audit = {
        "scope": "Pass1 localization diagnostic, fixed official dev20; not editing or test",
        "benchmark_commit": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        "manifest_sha256": MANIFEST_SHA256,
        "selection_sha256": sha256(BUNDLE / "selection.json"),
        "cases": 20,
        "unique_targets": 72,
        "requests": len(publics),
        "settings": list(SETTINGS),
        "K": selection["selected_K_counts"],
        "per_setting": {
            s: dict(
                Counter(
                    ("ref" if u["has_ref"] else "noref") + "_" + u["locator"]
                    for p in publics
                    if p["setting"] == s
                    for u in p["units"]
                )
            )
            for s in SETTINGS
        },
        "train_requests": n,
        "train_unique_images": len(train_hashes),
        "train_overlap": overlaps,
        "point_original_outside_gt": sum(not x["original_in_gt"] for x in geometry),
        "point_quantized_outside_gt": sum(not x["quantized_in_gt"] for x in geometry),
        "adapter_tests": "200 requests, private fields rejected, same S4 geometry, ordered IDs enforced",
        "adapter": str(ADAPTER),
        "generation_seed": 42,
        "do_sample": False,
        "max_new_tokens": 512,
        "max_pixels": 2097152,
        "mask_policy": "encode only supplied mask; no point/box SAM frontend; score raw output before known-mask override",
        "benchmark_public_sha256": sha256(ROOT / "data/benchmark_public.jsonl"),
        "test_assets_opened": 0,
    }
    write_json(ROOT / "provenance/preflight.json", audit)
    print(json.dumps(audit, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
