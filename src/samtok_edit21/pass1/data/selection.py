"""Freeze Pass1 region/request selections, without changing source data or running models.

This is a selection manifest, NOT a trainer-ready JSONL. Coordinate materialization
must check quantized clicks against S (and teacher decode for training only).
"""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

MODES = (
    "ref_none",
    "ref_point",
    "noref_point",
    "ref_box",
    "noref_box",
    "ref_mask",
    "noref_mask",
)


def stable(value, seed):
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def area_bin(area):
    for i, threshold in enumerate((0.0005, 0.001, 0.005, 0.01, 0.05)):
        if area < threshold:
            return i
    return 5


def eligible_modes(row, config):
    if row["key"].startswith("P:"):
        return list(
            config["dense_parent_modes"]
            if row.get("parts")
            else ("noref_point", "noref_box", "noref_mask")
        )
    if row.get("operation") == "add" and row.get("has_site"):
        allowed = config["known_host_fallback_modes"]
    else:
        allowed = MODES
    return [
        m
        for m in allowed
        if (
            row.get("ref_instruction")
            if m.startswith("ref_")
            else row.get("noref_eligible")
        )
    ]


def region_key(row):
    return row["source"], row["s_digest"]


def distinct(rows):
    """One geometric region/code per request; preserve overlap with different codes."""
    out, spans, digests = [], set(), set()
    for r in rows:
        if r["span"] not in spans and r["s_digest"] not in digests:
            out.append(r)
            spans.add(r["span"])
            digests.add(r["s_digest"])
    return out


def canonicalize_teachers(rows):
    """Resolve multiple cached codes for identical S without altering raw caches.

    Public instruction provenance stays on the original row; only the offline
    teacher reference is shared. This is not model-output candidate selection.
    """
    groups = collections.defaultdict(list)
    for row in rows:
        groups[region_key(row)].append(row)
    conflicts = []
    for (source, digest), group in groups.items():
        best = min(group, key=lambda r: (-r["roundtrip"], r["key"]))
        different = len({r["span"] for r in group}) > 1
        for row in group:
            row["canonical_label_key"] = best["key"] if different else row["key"]
        if different:
            conflicts.append(
                {
                    "source": source,
                    "s_digest": digest,
                    "canonical_label_key": best["key"],
                    "alternatives": [
                        {
                            "key": r["key"],
                            "roundtrip": r["roundtrip"],
                            "span": r["span"],
                        }
                        for r in group
                    ],
                }
            )
    return conflicts


def build_bank(inventory, config):
    images = pq.read_table(inventory / "merged_images.parquet").to_pylist()
    sid_to_source = {}
    for x in images:
        if x["sid"]:
            if x["sid"] in sid_to_source and sid_to_source[x["sid"]] != x["group"]:
                raise ValueError("An SA-1B ID maps to multiple image hashes")
            sid_to_source[x["sid"]] = x["group"]
    sources = {x["group"]: dict(x, source=x["group"], has_pool=False) for x in images}
    for x in pq.read_table(inventory / "pool_images.parquet").to_pylist():
        source = sid_to_source.get(x["sid"], "sa:" + x["sid"])
        if source not in sources:
            sources[source] = dict(x, group=source, source=source, family="D2")
        s = sources[source]
        if (s["w"], s["h"]) != (x["w"], x["h"]):
            raise ValueError("Pool/merged image dimensions disagree")
        s["has_pool"] = True
        s["pool_image"] = x["image"]
        s["pool_origin"] = x["origin"]
        for flag in ("old_dev", "benchmark", "pilot_holdout", "pilot_train"):
            s[flag] = s.get(flag, False) or x.get(flag, False)
    core, evaluation, reserve = (
        collections.defaultdict(list),
        collections.defaultdict(list),
        [],
    )
    rejected = collections.Counter()
    q = config["quality"]
    for kind in ("merged", "pool"):
        for batch in pq.ParquetFile(
            inventory / (kind + "_regions.parquet")
        ).iter_batches(batch_size=10000):
            for r in batch.to_pylist():
                source = (
                    r.get("group")
                    if kind == "merged"
                    else sid_to_source.get(r["sid"], "sa:" + r["sid"])
                )
                r["source"] = source
                s = sources[source]
                if any(
                    s.get(f, False) for f in ("old_dev", "benchmark", "pilot_holdout")
                ):
                    rejected["excluded_" + kind] += 1
                    continue
                if not r["valid"] or r["dec_area"] <= 0 or r["area"] <= 0:
                    rejected["invalid_" + kind] += 1
                    continue
                if "<|mt_" in r.get("ref_instruction", "") or "<|mt_" in r.get(
                    "noref_instruction", ""
                ):
                    rejected["instruction_leaks_span"] += 1
                    continue
                r["modes"] = eligible_modes(r, config)
                r["area_bin"] = area_bin(r["area"])
                if not r["modes"]:
                    rejected["no_modes"] += 1
                    continue
                # Holdouts retain .5-.7 teachers and large selections; do not make
                # the evaluation set artificially easy with the core train gate.
                if r["roundtrip"] >= 0.5 and not s.get("pilot_train", False):
                    evaluation[source].append(r)
                if q["min_area"] <= r["area"] <= q["max_area"]:
                    if r["roundtrip"] >= q["core_min_roundtrip"]:
                        core[source].append(r)
                    elif (
                        r["roundtrip"] >= q["reserve_min_roundtrip"]
                        and r["area"] < q["reserve_max_area"]
                    ):
                        reserve.append(r)
                    else:
                        rejected["quality_" + kind] += 1
                else:
                    rejected["area_" + kind] += 1
    return sources, core, evaluation, reserve, rejected


def split_sources(sources, evaluation, config):
    """Fresh source holdouts: fixed K and domain cells, all pilot-seen images excluded."""
    seed = config["seed"]
    available = {"external": [], "sa_real": [], "pool": []}
    caps = {}
    for s, rows in evaluation.items():
        info = sources[s]
        real = distinct(
            sorted(
                (r for r in rows if r["key"].startswith("M:")), key=lambda r: r["key"]
            )
        )
        all_rows = distinct(
            real
            + sorted(
                (r for r in rows if r["key"].startswith("P:")), key=lambda r: r["key"]
            )
        )
        caps[s] = (len(real), len(all_rows))
        kind = (
            "pool"
            if info["has_pool"]
            else ("sa_real" if info.get("sid") else "external")
        )
        available[kind].append(s)
    for v in available.values():
        v.sort(key=lambda s: stable(s, seed))
    # Per split: 200 external + 410 real SA + 390 pool images.
    cells = {
        "external": {2: 100, 3: 80, 4: 15, 5: 5},
        "sa_real": {2: 50, 3: 170, 4: 190},
        "pool": {4: 295, 5: 95},
    }
    if config["holdout_per_split"] != 1000:
        raise ValueError("This frozen domain/K design expects 1000 sources per holdout")
    chosen, used = {}, set()
    # Reserve scarce high-K cells before low-K cells; dev/test get symmetric rules.
    for split in ("dev", "test"):
        chosen[split] = []
        for kind, ks in cells.items():
            for k, n in sorted(ks.items(), reverse=True):
                candidates = [
                    s
                    for s in available[kind]
                    if s not in used and caps[s][kind == "pool"] >= k
                ]
                if len(candidates) < n:
                    raise ValueError(
                        f"Holdout cell shortage {split}/{kind}/K{k}: {len(candidates)} < {n}"
                    )
                for source in candidates[:n]:
                    chosen[split].append((source, k, kind))
                    used.add(source)
    return chosen, used


class Selector:
    def __init__(self, sources, bank, config):
        self.sources, self.bank, self.config = sources, bank, config
        self.rng = random.Random(config["seed"])
        self.source_uses = collections.Counter()
        self.region_uses = collections.Counter()
        self.family_requests = collections.Counter()
        self.pool_requests = collections.Counter()
        self.stage_modes = collections.defaultdict(collections.Counter)
        self.stage_areas = collections.defaultdict(collections.Counter)
        self.stage_total = collections.Counter()
        self.seen = set()
        self.bank_real, self.bank_dense = {}, {}
        for source, rows in bank.items():
            # Prefer less ambiguous ref supervision, then quality; do not let
            # repeated edit instructions inflate geometric target capacity.
            real = sorted(
                (r for r in rows if r["key"].startswith("M:")),
                key=lambda r: (-len(r["modes"]), -r["roundtrip"], r["key"]),
            )
            self.bank_real[source] = distinct(real)
            self.bank_dense[source] = distinct(
                sorted(
                    (r for r in rows if r["key"].startswith("P:")),
                    key=lambda r: (-r["roundtrip"], r["key"]),
                )
            )
        self.candidates = {}
        for kind in ("real", "hybrid", "dense"):
            for k in range(1, 6):
                ok = []
                for source in bank:
                    nr, np = len(self.bank_real[source]), len(self.bank_dense[source])
                    if (
                        (kind == "real" and nr >= k)
                        or (kind == "dense" and np >= k)
                        or (
                            kind == "hybrid"
                            and k >= 2
                            and nr >= 1
                            and np >= k - min(nr, max(1, k - 2))
                        )
                    ):
                        ok.append(source)
                self.rng.shuffle(ok)
                self.candidates[kind, k] = ok
        self.cursor = collections.Counter()

    def choose_rows(self, source, kind, k, stage):
        real, dense = self.bank_real[source], self.bank_dense[source]
        # A hybrid contains at least one real instruction and at least one dense
        # region. Prefer two real instructions at K>=4 to retain ref coverage.
        nr = (
            k
            if kind == "real"
            else (0 if kind == "dense" else min(len(real), max(1, k - 2)))
        )
        selected = []
        area_weights = list(self.config["area_target"].values())
        for collection, n in ((real, nr), (dense, k - nr)):
            for _ in range(n):
                candidates = [
                    r
                    for r in collection
                    if self.region_uses[region_key(r)]
                    < self.config["region_exposure_cap"]
                    and all(
                        r["span"] != a["span"] and r["s_digest"] != a["s_digest"]
                        for a in selected
                    )
                ]
                if not candidates:
                    return None
                # Prioritize under-covered area buckets, then source diversity.
                weights = []
                total = self.stage_total[stage]
                for r in candidates:
                    b = r["area_bin"]
                    deficit = max(
                        0.15,
                        (area_weights[b] * (total + 200) + 1)
                        / (self.stage_areas[stage][b] + 1),
                    )
                    weights.append(
                        min(8.0, deficit) / (1 + self.region_uses[region_key(r)])
                    )
                selected.append(self.rng.choices(candidates, weights=weights, k=1)[0])
        return selected

    def allocate_modes(self, rows, stage):
        weights = self.config["main_modes" if stage == "main" else "warmup_modes"]
        counts = self.stage_modes[stage].copy()
        total = self.stage_total[stage]
        assigned = [None] * len(rows)
        # Constrained dense/host rows first; real rows compensate remaining modes.
        order = sorted(range(len(rows)), key=lambda i: len(rows[i]["modes"]))
        for i in order:
            modes = rows[i]["modes"]
            mode = max(
                modes, key=lambda m: weights[m] * (total + len(rows) + 30) - counts[m]
            )
            assigned[i] = mode
            counts[mode] += 1
        # A regular multi-target training row has >=2 locator types and >=1
        # unknown target. Copy-only and homogeneous modes belong in diagnostics.
        if all(m.endswith("_mask") for m in assigned):
            for i, r in enumerate(rows):
                alternatives = [m for m in r["modes"] if not m.endswith("_mask")]
                if alternatives:
                    assigned[i] = alternatives[0]
                    break
            else:
                return None
        if len(rows) > 1 and len({m.split("_", 1)[1] for m in assigned}) == 1:
            loc = assigned[0].split("_", 1)[1]
            for i, r in enumerate(rows):
                alternatives = [m for m in r["modes"] if m.split("_", 1)[1] != loc]
                if alternatives:
                    assigned[i] = alternatives[0]
                    break
        return assigned

    def one(self, kind, k, stage, number):
        options = self.candidates[kind, k]
        if not options:
            raise ValueError(f"Empty cell {kind}/{k}")
        budget = self.config["main_requests"] + self.config["warmup_requests"]
        for _ in range(len(options) * 2):
            cursor = self.cursor[kind, k]
            self.cursor[kind, k] += 1
            source = options[cursor % len(options)]
            if self.source_uses[source] >= self.config["source_request_cap"]:
                continue
            info = self.sources[source]
            has_d1 = any(r.get("family") == "D1" for r in self.bank_real[source])
            if has_d1 and self.family_requests["D1"] >= int(
                budget * self.config["d1_request_fraction_cap"]
            ):
                continue
            is_old = info.get("pool_origin") == "base"
            if is_old and self.pool_requests["base"] >= int(
                budget * self.config["old_pool_request_fraction_cap"]
            ):
                continue
            rows = self.choose_rows(source, kind, k, stage)
            if rows is None:
                continue
            self.rng.shuffle(rows)
            modes = self.allocate_modes(rows, stage)
            if modes is None:
                continue
            labels = [f"U{i + 1}" for i in range(k)]
            if self.rng.random() < self.config["nonsequential_label_fraction"]:
                labels = self.rng.sample(["U1", "U2", "U3", "U4", "U5"], k)
            targets = []
            for label, row, mode in zip(labels, rows, modes):
                locator = mode.split("_", 1)[1]
                policy = (
                    "copy"
                    if locator == "mask"
                    else (
                        "text"
                        if locator == "none"
                        else self.rng.choices(
                            list(self.config["point_train"]),
                            weights=list(self.config["point_train"].values()),
                        )[0]
                        if locator == "point"
                        else self.rng.choices(
                            list(self.config["box_train"]),
                            weights=list(self.config["box_train"].values()),
                        )[0]
                    )
                )
                targets.append(
                    {
                        "label": label,
                        "region_key": row["key"],
                        "mode": mode,
                        "locator_policy": policy,
                        "geometry_seed": self.rng.randrange(2**31),
                    }
                )
            # Exact duplicate planned requests are not useful repeat supervision.
            signature = (
                source,
                tuple(
                    (
                        t["region_key"],
                        t["mode"],
                        t["label"],
                        t["locator_policy"],
                        t["geometry_seed"]
                        if "point" in t["mode"] or "box" in t["mode"]
                        else 0,
                    )
                    for t in targets
                ),
            )
            if signature in self.seen:
                continue
            self.seen.add(signature)
            self.source_uses[source] += 1
            if has_d1:
                self.family_requests["D1"] += 1
            if is_old:
                self.pool_requests["base"] += 1
            for row, mode in zip(rows, modes):
                self.region_uses[region_key(row)] += 1
                self.stage_modes[stage][mode] += 1
                self.stage_areas[stage][row["area_bin"]] += 1
                self.stage_total[stage] += 1
            return {
                "id": f"{stage}-{number:06d}",
                "split": "train",
                "stage": stage,
                "source": source,
                "kind": kind,
                "K": k,
                "targets": targets,
            }
        raise ValueError(
            f"Cell exhausted under source/region caps: {stage}/{kind}/K{k}, row {number}"
        )


def holdout_requests(chosen, sources, evaluation, config):
    rng = random.Random(config["seed"] + 2)
    result = {}
    for split, groups in chosen.items():
        rows_out = []
        for i, (source, k, kind) in enumerate(groups):
            rows = evaluation[source]
            real = distinct(
                sorted(
                    (r for r in rows if r["key"].startswith("M:")),
                    key=lambda r: stable(r["key"], config["seed"]),
                )
            )
            dense = distinct(
                sorted(
                    (r for r in rows if r["key"].startswith("P:")),
                    key=lambda r: stable(r["key"], config["seed"]),
                )
            )
            candidates = (
                distinct(real[: min(len(real), k)] + dense) if kind == "pool" else real
            )
            selected = candidates[:k]
            if len(selected) != k:
                raise ValueError("Holdout capacity changed")
            rng.shuffle(selected)
            labels = rng.sample(["U1", "U2", "U3", "U4", "U5"], k)
            targets = []
            for j, r in enumerate(selected):
                mode = MODES[(i + j) % len(MODES)]
                if mode not in r["modes"]:
                    mode = r["modes"][(i + j) % len(r["modes"])]
                loc = mode.split("_", 1)[1]
                policy = (
                    rng.choices(
                        list(config["point_eval"]),
                        weights=list(config["point_eval"].values()),
                    )[0]
                    if loc == "point"
                    else ("tight" if loc == "box" else loc)
                )
                targets.append(
                    {
                        "label": labels[j],
                        "region_key": r["key"],
                        "mode": mode,
                        "locator_policy": policy,
                        "geometry_seed": rng.randrange(2**31),
                    }
                )
            rows_out.append(
                {
                    "id": f"{split}-{i:05d}",
                    "split": split,
                    "stage": "eval",
                    "source": source,
                    "kind": kind,
                    "K": k,
                    "targets": targets,
                }
            )
        result[split] = rows_out
    return result


def describe(rows, region_map, sources):
    targets = [t for x in rows for t in x["targets"]]
    rs = [region_map[t["region_key"]] for t in targets]
    return {
        "requests": len(rows),
        "targets": len(targets),
        "source_images": len({x["source"] for x in rows}),
        "unique_regions": len({region_key(r) for r in rs}),
        "K": dict(collections.Counter(x["K"] for x in rows)),
        "kinds": dict(collections.Counter(x["kind"] for x in rows)),
        "modes": dict(collections.Counter(t["mode"] for t in targets)),
        "area_bins": dict(collections.Counter(r["area_bin"] for r in rs)),
        "target_origins": dict(
            collections.Counter(r.get("family") or ("pool_" + r["origin"]) for r in rs)
        ),
        "quality_below_0.7": sum(r["roundtrip"] < 0.7 for r in rs),
        "restricted_host_fallback_targets": sum(
            r.get("operation") == "add" and r.get("has_site", False) for r in rs
        ),
        "pool_parent_targets": sum(bool(r.get("parts")) for r in rs),
        "nonsequential_labels": sum(
            [t["label"] for t in x["targets"]] != [f"U{j + 1}" for j in range(x["K"])]
            for x in rows
        ),
    }


def rebalance_modes(rows, region_map, config, stage):
    """Allocate constrained targets first against final, not running, margins.

    Assigning real rows first with running percentages leaves dense rows unable
    to compensate the missing ref modes. Keep selection and mode allocation
    separate so source-selection order does not bias the public inputs.
    """
    rng = random.Random(config["seed"] + (12 if stage == "main" else 13))
    weights = config[stage + "_modes"]
    targets = [t for row in rows for t in row["targets"]]
    total = len(targets)
    counts = collections.Counter()
    order = list(range(total))
    rng.shuffle(order)
    order.sort(key=lambda i: len(region_map[targets[i]["region_key"]]["modes"]))
    for i in order:
        target = targets[i]
        allowed = region_map[target["region_key"]]["modes"]
        mode = max(
            allowed,
            key=lambda m: (weights[m] * total - counts[m])
            / max(1.0, weights[m] * total),
        )
        target["mode"] = mode
        counts[mode] += 1
    for row in rows:
        ts = row["targets"]
        if len(ts) <= 1:
            continue  # A small single-target mask-copy control is valid.
        if len({t["mode"].split("_", 1)[1] for t in ts}) == 1:
            loc = ts[0]["mode"].split("_", 1)[1]
            candidates = [
                (i, m)
                for i, t in enumerate(ts)
                for m in region_map[t["region_key"]]["modes"]
                if m.split("_", 1)[1] != loc
            ]
            if not candidates:
                raise ValueError("Cannot construct a mixed-input request")
            i, mode = max(
                candidates, key=lambda im: weights[im[1]] * total - counts[im[1]]
            )
            counts[ts[i]["mode"]] -= 1
            ts[i]["mode"] = mode
            counts[mode] += 1
    for t in targets:
        loc = t["mode"].split("_", 1)[1]
        if loc == "point":
            t["locator_policy"] = rng.choices(
                list(config["point_train"]),
                weights=list(config["point_train"].values()),
            )[0]
        elif loc == "box":
            t["locator_policy"] = (
                "tight"
                if stage == "eval"
                else rng.choices(
                    list(config["box_train"]),
                    weights=list(config["box_train"].values()),
                )[0]
            )
        else:
            t["locator_policy"] = "copy" if loc == "mask" else "text"


def add_counterfactual_pairs(rows, region_map, bank, config):
    """Reuse two existing same-source/K slots; never exceed source exposure caps.

    The pair has identical instructions, IDs and all other target inputs. Only
    one neutral no-ref point/box changes to a distinct same-category region.
    Actual public-coordinate inequality is checked during materialization.
    """
    rng = random.Random(config["seed"] + 14)
    groups = collections.defaultdict(list)
    for i, row in enumerate(rows):
        if row["kind"] != "real":
            groups[row["source"], row["kind"], row["K"]].append(i)
    pool_indices = {
        s: {r["index"]: r for r in rs if r["key"].startswith("P:")}
        for s, rs in bank.items()
    }
    keys = list(groups)
    rng.shuffle(keys)
    wanted = int(len(rows) * config["counterfactual_main_row_fraction"] / 2)
    done = 0
    pair_modes = collections.Counter()
    for key in keys:
        slots = groups[key]
        rng.shuffle(slots)
        for left, right in zip(slots[::2], slots[1::2]):
            if done >= wanted:
                break
            first = rows[left]
            candidates = []
            current = [region_map[t["region_key"]] for t in first["targets"]]
            for ti, target in enumerate(first["targets"]):
                a = region_map[target["region_key"]]
                if not a["key"].startswith("P:"):
                    continue
                for bi in a.get("same", []):
                    b = pool_indices[key[0]].get(bi)
                    if (
                        b is None
                        or b["span"] == a["span"]
                        or b["s_digest"] == a["s_digest"]
                    ):
                        continue
                    if any(
                        b["span"] == x["span"] or b["s_digest"] == x["s_digest"]
                        for x in current
                    ):
                        continue
                    modes = [
                        m
                        for m in ("noref_point", "noref_box")
                        if m in a["modes"] and m in b["modes"]
                    ]
                    if a["bbox"] == b["bbox"]:
                        modes = [m for m in modes if m != "noref_box"]
                    if modes:
                        candidates.append((ti, b, modes))
            if not candidates:
                continue
            ti, b, modes = rng.choice(candidates)
            preferred = (
                "noref_point"
                if pair_modes["noref_point"] <= pair_modes["noref_box"]
                else "noref_box"
            )
            mode = preferred if preferred in modes else modes[0]
            first["targets"][ti]["mode"] = mode
            first["targets"][ti]["locator_policy"] = (
                "deep" if mode == "noref_point" else "tight"
            )
            # Preserve the second slot's identity, while copying all public-input
            # plans and seeds from the first slot before changing the one region.
            second = copy.deepcopy(first)
            second["id"] = rows[right]["id"]
            second["targets"][ti]["region_key"] = b["key"]
            pair = f"cf-{done:06d}"
            for role, row in (("a", first), ("b", second)):
                row["pair_id"] = pair
                row["pair_role"] = role
                row["contrast_label"] = first["targets"][ti]["label"]
            # Avoid making a homogeneous multi-target row after the intervention.
            if len({t["mode"].split("_", 1)[1] for t in first["targets"]}) == 1:
                for j, target in enumerate(first["targets"]):
                    if j == ti:
                        continue
                    alternatives = [
                        m
                        for m in region_map[target["region_key"]]["modes"]
                        if m.split("_", 1)[1] != mode.split("_", 1)[1]
                    ]
                    if alternatives:
                        target["mode"] = alternatives[0]
                        loc = target["mode"].split("_", 1)[1]
                        target["locator_policy"] = {
                            "none": "text",
                            "mask": "copy",
                            "box": "tight",
                            "point": "deep",
                        }[loc]
                        second["targets"][j] = copy.deepcopy(target)
                        break
            rows[right] = second
            done += 1
            pair_modes[mode] += 1
        if done >= wanted:
            break
    return {
        "requested_pairs": wanted,
        "selected_pairs": done,
        "rows": 2 * done,
        "modes": dict(pair_modes),
        "shortfall_pairs": wanted - done,
        "definition": "same source, K, IDs and neutral instructions; one no-ref locator changes between distinct same-category regions; coordinate inequality pending geometry audit",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(
            "Selection output must be empty; never overwrite a frozen selection"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())
    sources, core, evaluation, reserve, rejected = build_bank(args.inventory, config)
    print(
        json.dumps(
            {
                "core_sources": len(core),
                "core_rows": sum(map(len, core.values())),
                "reserve": len(reserve),
            }
        ),
        flush=True,
    )
    chosen, heldout = split_sources(sources, evaluation, config)
    train = {s: rs for s, rs in core.items() if s not in heldout}
    selector = Selector(sources, train, config)
    selections = holdout_requests(chosen, sources, evaluation, config)
    # Reserve the scarce K=5/4 real-source cells first. Final files are shuffled.
    main_rows = []
    number = 0
    for kind in ("real", "hybrid", "dense"):
        for kstr, n in sorted(
            config["main_cells"][kind].items(), key=lambda item: -int(item[0])
        ):
            k = int(kstr)
            for _ in range(n):
                main_rows.append(selector.one(kind, k, "main", number))
                number += 1
            print(
                json.dumps(
                    {"stage": "main", "kind": kind, "K": k, "completed": number}
                ),
                flush=True,
            )
    if len(main_rows) != config["main_requests"]:
        raise ValueError("Main quota mismatch")
    selections["main"] = main_rows
    warm = []
    for kstr, n in sorted(config["warmup_k"].items(), key=lambda item: -int(item[0])):
        k = int(kstr)
        for i in range(n):
            # Mix editing instructions and geometry; warm-up is still Pass1 SFT.
            kind = (
                "dense"
                if i % 4 == 0
                else ("hybrid" if k > 1 and i % 4 == 1 else "real")
            )
            warm.append(selector.one(kind, k, "warmup", len(warm)))
        print(
            json.dumps({"stage": "warmup", "K": k, "completed": len(warm)}), flush=True
        )
    if len(warm) != config["warmup_requests"]:
        raise ValueError("Warmup quota mismatch")
    selections["warmup"] = warm
    region_map = {r["key"]: r for rs in core.values() for r in rs}
    region_map.update({r["key"]: r for source in heldout for r in evaluation[source]})
    rebalance_modes(selections["main"], region_map, config, "main")
    rebalance_modes(selections["warmup"], region_map, config, "warmup")
    rebalance_modes(selections["dev"], region_map, config, "eval")
    rebalance_modes(selections["test"], region_map, config, "eval")
    counterfactual = add_counterfactual_pairs(
        selections["main"], region_map, train, config
    )
    all_keys = {
        t["region_key"]
        for rows in selections.values()
        for x in rows
        for t in x["targets"]
    }
    selected_regions = [region_map[k] for k in sorted(all_keys)]
    teacher_conflicts = canonicalize_teachers(selected_regions)
    # Union schemas explicitly: Arrow's inferred schema otherwise depends on
    # whether the first row happens to be a pool or a merged region.
    fields = sorted({k for r in selected_regions for k in r})
    for r in selected_regions:
        for k in fields:
            r.setdefault(k, None)
    pq.write_table(
        pa.Table.from_pylist(selected_regions),
        args.output / "selected_regions.parquet",
        compression="zstd",
    )
    reserve = [r for r in reserve if r["source"] not in heldout]
    if reserve:
        fields_r = sorted({k for r in reserve for k in r})
        for r in reserve:
            for k in fields_r:
                r.setdefault(k, None)
        pq.write_table(
            pa.Table.from_pylist(reserve),
            args.output / "small_lowq_reserve.parquet",
            compression="zstd",
        )
    used_sources = {x["source"] for rows in selections.values() for x in rows}
    source_rows = []
    for s in sorted(used_sources):
        split = next(
            (name for name in ("dev", "test") if any(g[0] == s for g in chosen[name])),
            "train",
        )
        source_rows.append(
            {**sources[s], "split": split, "train_requests": selector.source_uses[s]}
        )
    source_fields = sorted({k for r in source_rows for k in r})
    for r in source_rows:
        for k in source_fields:
            r.setdefault(k, None)
    pq.write_table(
        pa.Table.from_pylist(source_rows),
        args.output / "selected_sources.parquet",
        compression="zstd",
    )
    summaries, hashes = {}, {}
    for name, rows in selections.items():
        if name in ("main", "warmup"):
            selector.rng.shuffle(rows)
        path = args.output / (name + ".selection.jsonl")
        with path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row, separators=(",", ":")) + "\n")
        summaries[name] = describe(rows, region_map, sources)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    sets = {name: {x["source"] for x in rows} for name, rows in selections.items()}
    train_sources = sets["main"] | sets["warmup"]
    if train_sources & (sets["dev"] | sets["test"]) or sets["dev"] & sets["test"]:
        raise AssertionError("Cross-split source leakage")
    if any(
        sources[s].get("pilot_train") or sources[s].get("pilot_holdout")
        for s in heldout
    ):
        raise AssertionError("A fresh holdout source was used in the pilot")
    if max(selector.source_uses.values()) > config["source_request_cap"]:
        raise AssertionError("Source cap violated")
    manifest = {
        "version": config["version"],
        "status": "selection_frozen_pending_geometry_materialization",
        "config": config,
        "summary": summaries,
        "rejected_inventory_rows": dict(rejected),
        "counterfactual": counterfactual,
        "canonical_teacher_groups_resolved": len(teacher_conflicts),
        "selected_unique_regions": len({region_key(r) for r in selected_regions}),
        "selected_annotation_rows": len(selected_regions),
        "train_sources": len(train_sources),
        "max_train_requests_per_source": max(selector.source_uses.values()),
        "max_train_region_exposures": max(selector.region_uses.values()),
        "d1_source_requests": selector.family_requests["D1"],
        "base_pool_source_requests": selector.pool_requests["base"],
        "reserved_lowq_rows": len(reserve),
        "reserved_lowq_sources": len({r["source"] for r in reserve}),
        "source_id_or_existing_sha1_overlap": 0,
        "prior_pilot_overlap_in_new_dev_test": 0,
        "selection_sha256": hashes,
        "pending_checks": [
            "Exact byte identity of pool/merged aliases and selected sources",
            "Fresh split near-duplicate image audit (upstream benchmark exclusion is inherited)",
            "Point/box geometry materialization and public-input conflict audit",
            "Payload/template repetition audit after final materialization",
        ],
        "no_model_training_or_inference_performed": True,
    }
    dump(args.output / "manifest.json", manifest)
    dump(args.output / "canonical_teacher_resolution.json", teacher_conflicts)
    dump(args.output / "selection_config.json", config)
    print(json.dumps(summaries, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
