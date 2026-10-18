"""Round-2 paired report (docs/08 sections 8.4 and 9): one evaluation against reference evaluations of the same items.

    python -m samtok_edit21 round2-report --eval <run>/eval --name C2 \\
        --reference stock=<stock eval> --reference C1b=<C1b eval> [--train <run>/train] --output <run>/analysis.txt

Every comparison uses the items scored by every evaluation, with bootstrap 95 % intervals over items (pairs for the
token-swap and remove probes).  Edits are read with the background drift removed: an edit counts as not done when the
change inside the really changed pixels exceeds the change outside the execution region by less than NOOP_MARGIN
(the stock model shifts the whole image by about 5-6 grey levels, so a fixed threshold on the inside change alone would
almost never call a stock edit undone).  ``analysis.json`` next to the text holds the same numbers.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

AREA_EDGES = (0.0005, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 1.01)
OPERATIONS = ("add", "remove", "replace", "attribute")
NOOP_MARGIN = 5.0


def load(path):
    data = json.load(open(Path(path) / "eval.json"))
    return {r["idx"]: r for r in data["records"]}


def boot(values, n=2000, seed=0):
    values = np.asarray(values, float)
    if len(values) == 0:
        return None
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (n, len(values)))].mean(1)
    return [float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def fmt(m):
    return "n/a" if m is None else f"{m[0]:.3f} [{m[1]:.3f}, {m[2]:.3f}]"


def follow(records, own, other):
    pairs = defaultdict(list)
    for r in records:
        if r.get("valid"):
            pairs[r["pair"]].append(r[own] > r[other])
    return [all(v) for v in pairs.values() if len(v) == 2]


def noop(r):
    return r["in_change"] - r["out_change"] < NOOP_MARGIN


def summarize(runs):
    """``runs``: {label: {idx: record}} with the evaluated model last.  Returns (text lines, numbers)."""
    names = list(runs)
    model = names[-1]
    common = set.intersection(*[set(r) for r in runs.values()])
    first = runs[model]
    out, numbers = [f"items scored by every evaluation: {len(common)}"], {"items": len(common), "runs": names}

    def section(title):
        out.append("")
        out.append(f"== {title}")

    section("single-token highlight IoU (against the token's decoded region)")
    for variant in ("noref", "ref"):
        idx = sorted(i for i in common if first[i]["set"] == "single" and first[i]["variant"] == variant)
        values = {n: boot([runs[n][i]["iou"] for i in idx]) for n in names}
        numbers[f"single_{variant}"] = values
        out.append(f"  {variant} (n={len(idx)}): " + " | ".join(f"{n} {fmt(values[n])}" for n in names))
        for n in names[:-1]:
            d = boot([runs[model][i]["iou"] - runs[n][i]["iou"] for i in idx])
            numbers[f"single_{variant}_{model}_minus_{n}"] = d
            out.append(f"     {model} - {n}: {fmt(d)}")
    idx = sorted(i for i in common if first[i]["set"] == "single" and first[i]["variant"] == "noref")
    out.append("  noref by region area (share of the image):")
    for lo, hi in zip(AREA_EDGES[:-1], AREA_EDGES[1:]):
        sel = [i for i in idx if lo <= first[i]["area"] < hi]
        if sel:
            out.append(f"    {lo * 100:5.2f}-{min(hi, 1) * 100:6.2f}% n={len(sel):3d}: "
                       + " | ".join(f"{n} {np.mean([runs[n][i]['iou'] for i in sel]):.3f}" for n in names))

    section("multi-token highlight")
    idx = sorted(i for i in common if first[i]["set"] == "multi")
    for n in names:
        r = [runs[n][i] for i in idx]
        if r:
            numbers[f"multi_{n}"] = {"all_correct": float(np.mean([x["all_correct"] for x in r])),
                                     "region_iou": float(np.mean([v for x in r for v in x["ious"]])),
                                     "cross_talk": float(np.mean([x["cross_talk"] for x in r]))}
            m = numbers[f"multi_{n}"]
            out.append(f"  {n}: all regions IoU >= 0.5 {m['all_correct']:.3f} | region IoU {m['region_iou']:.3f}"
                       f" | colour in another token's region {m['cross_talk']:.3f}")

    section("token-swap pairs (same blind prompt, token A vs token B): both outputs follow their own token")
    for kind in ("instance", "part_host"):
        idx = sorted(i for i in common if first[i]["set"] == "pair" and first[i]["pair_kind"] == kind)
        values = {n: boot(follow([runs[n][i] for i in idx], "iou_self", "iou_other")) for n in names}
        numbers[f"pair_{kind}"] = values
        out.append(f"  {kind}: " + " | ".join(f"{n} {fmt(values[n])}" for n in names))

    section('remove probe ("Remove the selected region <M>.")')
    idx = sorted(i for i in common if first[i]["set"] == "remove")
    for n in names:
        r = [runs[n][i] for i in idx if runs[n][i].get("valid")]
        values = boot(follow(r, "change_own", "change_other"))
        numbers[f"remove_{n}"] = values
        if r:
            out.append(f"  {n}: pair-follow {fmt(values)} | own - other change "
                       f"{np.mean([x['change_own'] - x['change_other'] for x in r]):.1f} | far change {np.mean([x['change_far'] for x in r]):.1f}")

    section("SA-1B dev edits (blind sentence + token): drift-corrected; 'not done' = inside - outside change < "
            f"{NOOP_MARGIN:g}")
    idx = sorted(i for i in common if first[i]["set"] == "edit")
    for op in OPERATIONS + (None,):
        sel = [i for i in idx if op is None or first[i]["operation"] == op]
        if not sel:
            continue
        row = []
        for n in names:
            r = [runs[n][i] for i in sel]
            stats = {"not_done": float(np.mean([noop(x) for x in r])),
                     "strength": float(np.mean([x["in_change"] - x["out_change"] for x in r])),
                     "outside": float(np.mean([x["out_change"] for x in r])),
                     "error_vs_teacher": float(np.mean([x["in_target_err"] for x in r]))}
            numbers[f"edit_{op or 'all'}_{n}"] = stats
            row.append(f"{n} not done {stats['not_done']:.0%} / strength {stats['strength']:.1f} / outside {stats['outside']:.1f}"
                       f" / error {stats['error_vs_teacher']:.1f}")
        teacher = float(np.mean([first[i]["teacher_in_change"] for i in sel]))
        numbers[f"edit_{op or 'all'}_teacher_strength"] = teacher
        out.append(f"  {op or 'all'} (n={len(sel)}): " + " | ".join(row) + f" | teacher strength {teacher:.1f}")
    for n in names[:-1]:
        d = boot([float(noop(runs[model][i])) - float(noop(runs[n][i])) for i in idx])
        numbers[f"edit_not_done_{model}_minus_{n}"] = d
        out.append(f"  not done, {model} - {n}: {fmt(d)}")
    return out, numbers


def training_lines(train_dir):
    path = Path(train_dir) / "train_metrics.jsonl"
    if not path.exists():
        return [], {}
    rows = [json.loads(line) for line in open(path)]
    updates = [r for r in rows if "update" in r]
    if not updates:
        return [], {}
    last = updates[-1]
    numbers = {"updates": last["update"], "skipped": sum(r["skipped"] for r in updates), "seconds": last["sec"],
               "last": {k: last[k] for k in ("loss", "mse", "reg", "dis", "write", "read", "lr")}}
    lines = ["", "== training",
             f"  {last['update']} updates in {last['sec'] / 3600:.1f} h, skipped samples {numbers['skipped']}",
             "  last: " + ", ".join(f"{k} {v:.4g}" for k, v in numbers["last"].items() if v is not None)]
    quick = Path(train_dir) / "quick_summary.jsonl"
    if quick.exists():
        lines.append("  quick evaluations:")
        for r in map(json.loads, open(quick)):
            lines.append("    " + ", ".join(f"{k} {v}" for k, v in r.items() if k not in ("sec",)))
    return lines, numbers


def parser():
    p = argparse.ArgumentParser(prog="samtok-edit round2-report", description=__doc__.split("\n\n")[0])
    p.add_argument("--eval", required=True, help="Evaluation directory of the model (holds eval.json)")
    p.add_argument("--name", default="model")
    p.add_argument("--reference", action="append", default=[], help="LABEL=EVAL_DIR, base model first; repeatable")
    p.add_argument("--train", help="Training directory (train_metrics.jsonl, quick_summary.jsonl)")
    p.add_argument("--output", required=True, help="Text report; the numbers go to the same name with .json")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    runs = {}
    for reference in args.reference:
        label, path = reference.split("=", 1)
        if (Path(path) / "eval.json").exists():
            runs[label] = load(path)
        else:
            print(f"reference {label} skipped: no eval.json in {path}", flush=True)
    runs[args.name] = load(args.eval)
    lines, numbers = summarize(runs)
    extra, train = training_lines(args.train) if args.train else ([], {})
    text = "\n".join([f"Round-2 report: {args.name}", *lines, *extra]) + "\n"
    output = Path(args.output)
    output.write_text(text)
    output.with_suffix(".json").write_text(json.dumps({"name": args.name, **numbers, "training": train}, indent=1))
    print(text, flush=True)


if __name__ == "__main__":
    main()
