"""Training data of the ablation (CPU, run once before the four-machine run): ``plan.DATA`` and its rows files.

``plan.DATA`` links every entry of the formal S1/S2 data directory (pool, token caches, development items, rows_s1 /
rows_s2) and adds:

* component rows, made from the prefix a reduced stage trains on (Read: rows_s1[:32000], the first 1,000 updates of S1;
  Edit: rows_s2[:24000]) by replacing rows in place, so every update keeps its other rows and every pair stays within
  its update.  A Read file keeps the untouched rest of rows_s1 behind the prefix (the trainer checks that the rows
  cover the whole 6,000-update schedule; training stops at 1,000).  v1's 8,000-row Read files stay as they were:
  - ``rows_abl_noid_s{1,2}``: every target-identification row (hl, hl_ref, hlm, highlight pairs) -> an edit row
    (not a twin) from the unused rest of the same file: editing only, no target identification;
  - ``rows_abl_notwin_s{1,2}``: both members of every twin edit pair -> single-target edit rows (not twins);
  - ``rows_abl_nomulti_s2``: every multi-target edit row -> a single-target edit row (rows_s1 has none);
* data-scale rows ``rows_abl_s{25,50}``: the S2 spec (scripts/round2/specs/rows_s2.json, same seed, shares and
  epochs) at 4,000 / 8,000 updates, drawn only from the merged records of 25 % / 50 % of the source-image groups
  (``group``; nested: one hash decides both), so twins and multi-target records stay whole.

    PYTHONPATH=<repo>/src:<repo>/third_party/diffsynth:<repo>/third_party python scripts/ablation/build_data.py \\
        [--component] [--scale s25 s50]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plan  # noqa: E402

HIGHLIGHT = ("hl", "hl_ref", "hlm")
SCALE_SALT = "ablation-scale-20261112"


def is_twin(row):
    return row["kind"] == "edit_merged" and row.get("pair_kind") == "twin"


def is_multi(row):
    return row["kind"] == "edit_multi" or (row["kind"] == "edit_merged" and "order" in row)


def is_single(row):
    return row["kind"] == "edit_merged" and not is_twin(row) and not is_multi(row)


def is_edit_not_twin(row):
    return row["kind"].startswith("edit") and row["kind"] != "edit_plain" and not is_twin(row)


def link_data():
    plan.DATA.mkdir(parents=True, exist_ok=True)
    for entry in sorted(plan.ORIG_DATA.iterdir()):
        target = plan.DATA / entry.name
        if not target.exists() and not target.is_symlink():
            target.symlink_to(entry, target_is_directory=entry.is_dir())
        elif target.resolve() != entry.resolve():
            raise ValueError(f"{target} exists and is not a link to {entry}")


def replaced(rows, prefix, drop, keep, cycle=False):
    """``rows[:prefix]`` with every row matching ``drop`` replaced, in place, by the next row of ``rows[prefix:]``
    matching ``keep``.  ``cycle``: when those run out, start again from the first of them (rows seen twice), so the
    replacements stay of the same kind instead of coming from another file."""
    out, pool, used = list(rows[:prefix]), [r for r in rows[prefix:] if keep(r)], 0
    need = sum(1 for r in out if drop(r))
    if need > len(pool) and not cycle:
        raise RuntimeError(f"{need} replacements needed, {len(pool)} replacement rows")
    for i, row in enumerate(out):
        if drop(row):
            out[i] = pool[used % len(pool)]
            used += 1
    return out, used


def stats(name, rows, replaced_rows, source, prefix):
    kinds = Counter(r["kind"] for r in rows)
    return {"name": name, "rows": len(rows), "updates": len(rows) // 32, "source": source, "prefix": prefix,
            "replaced": replaced_rows, "kinds": dict(kinds),
            "twin_rows": sum(map(is_twin, rows)), "multi_rows": sum(map(is_multi, rows)),
            "single_rows": sum(map(is_single, rows)), "highlight_rows": sum(r["kind"] in HIGHLIGHT for r in rows),
            "highlight_pair_rows": sum(r["kind"] == "hl" and bool(r.get("pair_key")) for r in rows)}


def write_rows(name, rows, info):
    """Write a rows file and its statistics; an existing file with the same content is left untouched (a running job may
    be reading it)."""
    path = plan.DATA / f"{name}.jsonl"
    text = "".join(json.dumps(row) + "\n" for row in rows)
    if path.exists():
        if path.read_text() != text:
            raise SystemExit(f"{path} exists with different rows; use a new name")
        print(json.dumps({"name": name, "unchanged": True}))
        return
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)
    (plan.DATA / f"{name}_stats.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: info[k] for k in ("name", "rows", "replaced", "twin_rows", "multi_rows", "highlight_rows")}))


def component_rows():
    files = {"read": [json.loads(line) for line in open(plan.ORIG_DATA / plan.STAGES["read"]["rows"])],
             "edit": [json.loads(line) for line in open(plan.ORIG_DATA / plan.STAGES["edit"]["rows"])]}
    recipes = {"noid": (lambda r: r["kind"] in HIGHLIGHT, is_edit_not_twin),
               "notwin": (is_twin, is_single),
               "nomulti": (is_multi, is_single)}
    for arm in plan.ARMS.values():
        for stage, name in arm.get("rows", {}).items():
            recipe = name.split("_")[2]                      # rows_abl_<recipe>_<stage>[_<variant>].jsonl
            prefix = plan.STAGES[stage]["updates"] * 32
            rows, count = replaced(files[stage], prefix, *recipes[recipe], cycle=recipe == "noid")
            trained = rows
            rows = rows + files[stage][prefix:plan.STAGES[stage]["schedule"] * 32]   # never trained (stop-after)
            if recipe == "noid":
                assert not any(r["kind"] in HIGHLIGHT for r in trained)
            elif recipe == "notwin":
                assert not any(map(is_twin, trained))
            else:
                assert not any(map(is_multi, trained))
            info = stats(name, trained, count, plan.STAGES[stage]["rows"], prefix)
            info["file_rows"] = len(rows)
            ids = Counter(json.dumps(r, sort_keys=True) for r in trained)
            info["rows_seen_twice"] = sum(n - 1 for n in ids.values() if n > 1)
            write_rows(name[:-len(".jsonl")], rows, info)
    for stage, spec in plan.STAGES.items():   # reference statistics of the unchanged prefixes
        prefix = spec["updates"] * 32
        info = stats(spec["rows"], files[stage][:prefix], 0, spec["rows"], prefix)
        (plan.DATA / f"prefix_{stage}_{prefix // 32}_stats.json").write_text(json.dumps(info, indent=1))


def group_value(group):
    return int(hashlib.sha256(f"{SCALE_SALT}:{group}".encode()).hexdigest()[:13], 16) / 16 ** 13


def scale_rows(names):
    from samtok_edit21.preparation import round2_data as rd

    records = [json.loads(line) for line in open(rd.MERGED_ROOT / "records.jsonl")]
    base = json.loads((Path(__file__).resolve().parents[1] / "round2/specs/rows_s2.json").read_text())
    for name in names:
        fraction, updates, rows_name = plan.SCALE[name]["fraction"], plan.SCALE[name]["updates"], plan.SCALE[name]["rows"]
        kept = Counter()
        total = Counter()
        exclude = []
        for r in records:
            if not r["default"]:
                continue
            total[r["family"]] += 1
            if group_value(r["group"]) < fraction:
                kept[r["family"]] += 1
            else:
                exclude.append(r["id"])
        exclude_path = plan.DATA / f"scale_exclude_{name}.txt"
        exclude_path.write_text("\n".join(exclude) + "\n")
        spec = dict(base, name=rows_name[:-len(".jsonl")], updates=updates, exclude=str(exclude_path))
        spec_path = plan.DATA / f"spec_{name}.json"
        spec_path.write_text(json.dumps(spec, indent=1))
        result = rd.build_rows_merged(plan.DATA, spec_path)
        info = {"name": name, "fraction": fraction, "salt": SCALE_SALT, "updates": updates,
                "default_records": sum(total.values()), "kept_records": sum(kept.values()),
                "kept_by_family": {f: [kept[f], total[f], round(kept[f] / total[f], 4)] for f in sorted(total)},
                "rows": result["rows"], "merged_records_distinct": result["merged_records_distinct"],
                "twin_pairs": result["twin_pairs"], "merged_rows_multi": result["merged_rows_multi"]}
        (plan.DATA / f"scale_{name}_subset.json").write_text(json.dumps(info, indent=1))
        print(json.dumps(info))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--component", action="store_true")
    p.add_argument("--scale", nargs="*", default=[])
    a = p.parse_args(argv)
    link_data()
    if a.component:
        component_rows()
    if a.scale:
        scale_rows(a.scale)


if __name__ == "__main__":
    main()
