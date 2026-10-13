"""Text report of a token-grounding evaluation (v3), with paired comparisons against reference runs.

    samtok-edit ground-report --eval <eval_dir> --data <data_dir> --train <train_dir> --name E-A2 \\
        --reference stock=<stock_eval_dir> --reference E-A1=<ea1_eval_dir> --output <run_dir>/analysis.txt

Every comparison is paired: only items present in this run and in all references are used, and differences come with
a bootstrap 95% interval.  Held-out numbers are also given on the leak-free subset (items not flagged ``leaky_*``).
The first reference is the base model: the gate "dev with phrase must not fall below the base" refers to it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from samtok_edit21.evaluation.grounding import metric_key

MODEL_ALIASES = {"ea2": "trained", "stock": "base"}   # labels written by the E-A2 scripts
AREA_BUCKETS = (("<0.5%", 0, 0.005), ("0.5-5%", 0.005, 0.05), (">=5%", 0.05, 2))
BOOTSTRAP = 3000


class Bootstrap:
    """One generator for the whole report, so a report is reproducible line by line."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def interval(self, values):
        values = np.asarray(values, float)
        means = [values[self.rng.integers(0, len(values), len(values))].mean() for _ in range(BOOTSTRAP)]
        return values.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)

    def value(self, values):
        if len(values) == 0:
            return "n/a"
        mean, low, high = self.interval(values)
        return f"{mean:.3f} [{low:.3f},{high:.3f}]"

    def difference(self, values):
        mean, low, high = self.interval(values)
        return f"{mean:+.3f} [{low:+.3f},{high:+.3f}]"


def load_eval(path):
    path = Path(path)
    return json.load(open(path / "eval.json" if path.is_dir() else path))


def by_index(evaluation):
    return {record["idx"]: record for record in evaluation["recs"]}


def is_leaky(item):
    return any(value for key, value in item.items() if key.startswith("leaky"))


def pairs_of(items, records):
    """pair id -> {token: record} for pairs whose two probes are both valid."""
    pairs = {}
    for index, item in enumerate(items):
        if item["set"] == "pair" and index in records and records[index].get("valid"):
            pairs.setdefault(item["pair"], {})[item["tok"]] = records[index]
    return {pair: tokens for pair, tokens in pairs.items() if len(tokens) == 2}


def follows(tokens, own="iou_self", other="iou_other"):
    return float(all(token[own] > token[other] for token in tokens.values()))


def build_report(items, evaluation, references, *, name, train_dir=None, gates=None):
    """Returns ``(lines, summary)``; ``references`` is an ordered ``{label: evaluation}`` with the base first."""
    gates = {"heldout": 0.60, "dev": 0.50, "pair": 0.70, **(gates or {})}
    boot = Bootstrap()
    lines, summary = [], {"name": name, "highlight": {}, "gate": {}}
    mine = by_index(evaluation)
    refs = {label: by_index(reference) for label, reference in references.items()}
    key = lambda index: metric_key(mine[index])  # noqa: E731

    if train_dir is not None:
        train_dir = Path(train_dir)
        lines.append("== training (every 100 updates)")
        for text in open(train_dir / "train_metrics.jsonl"):
            row = json.loads(text)
            if "update" not in row or row["update"] % 100 == 0 or row["update"] in (10, 50):
                lines.append(f"   {row}")
        lines.append("== quick evals (small subset, trend only)")
        lines += [f"   {text.strip()}" for text in open(train_dir / "quick_summary.jsonl")]

    def indices(name_, variant, clean=None):
        return [index for index, item in enumerate(items)
                if item["set"] == name_ and item["variant"] == variant and index in mine
                and all(index in ref for ref in refs.values()) and (clean is None or (not is_leaky(item)) == clean)]

    header = " | ".join([*refs, name])
    lines.append(f"== token-only highlight IoU ({header}) and paired differences")
    for set_name, variant in (("ho", "noref"), ("ho", "ref"), ("dev", "noref"), ("dev", "ref")):
        for clean in ((None, True) if set_name == "ho" else (None,)):
            chosen = indices(set_name, variant, clean)
            tag = f"{set_name}-{variant}" + ("" if clean is None else " (leak-free)")
            if not chosen:
                continue
            ours = [mine[i][key(i)] for i in chosen]
            theirs = {label: [ref[i][key(i)] for i in chosen] for label, ref in refs.items()}
            parts = [f"{label} {np.mean(values):.3f}" for label, values in theirs.items()]
            parts.append(f"{name} {boot.value(ours)}")
            parts += [f"{name}-{label} {boot.difference(np.subtract(ours, values))}" for label, values in theirs.items()]
            lines.append(f"  {tag:22s} n={len(chosen):4d} " + " | ".join(parts))
            summary["highlight"][tag] = {"n": len(chosen), name: float(np.mean(ours)),
                                         **{label: float(np.mean(values)) for label, values in theirs.items()}}
    for label, low, high in AREA_BUCKETS:
        chosen = [i for i in indices("ho", "noref") if low <= mine[i]["area"] < high]
        if chosen:
            parts = [f"{ref_label} {np.mean([ref[i]['iou_codec'] for i in chosen]):.3f}" for ref_label, ref in refs.items()]
            parts.append(f"{name} {np.mean([mine[i]['iou_codec'] for i in chosen]):.3f}")
            lines.append(f"     held-out noref area {label:7s} n={len(chosen):4d} " + " ".join(parts))

    my_pairs = pairs_of(items, mine)
    ref_pairs = {label: pairs_of(items, ref) for label, ref in refs.items()}
    leaky_pair = {item["pair"]: is_leaky(item) for item in items if item["set"] == "pair"}
    for clean in (None, True):
        common = set(my_pairs)
        for pairs in ref_pairs.values():
            common &= set(pairs)
        chosen = sorted(pair for pair in common if clean is None or not leaky_pair[pair])
        if not chosen:
            continue
        tag = "pairs" + ("" if clean is None else " (leak-free)")
        ours = [follows(my_pairs[pair]) for pair in chosen]
        parts = [f"{label} {np.mean([follows(pairs[pair]) for pair in chosen]):.3f}" for label, pairs in ref_pairs.items()]
        parts.append(f"{name} {boot.value(ours)}")
        if ref_pairs:
            base_label, base_pairs = next(iter(ref_pairs.items()))
            parts.append(f"{name}-{base_label} "
                         + boot.difference([follows(my_pairs[pair]) - follows(base_pairs[pair]) for pair in chosen]))
        per_token = np.mean([t["iou_self"] > t["iou_other"] for pair in chosen for t in my_pairs[pair].values()])
        lines.append(f"  {tag:22s} n={len(chosen):4d} pair-follow " + " | ".join(parts) + f" | per-token {name} {per_token:.3f}")
        summary["highlight"][tag] = {"n": len(chosen), name: float(np.mean(ours)), "per_token": float(per_token)}

    got = summary["highlight"]
    lines.append(f"== gate (held-out noref >= {gates['heldout']:.2f}, dev noref >= {gates['dev']:.2f}, "
                 f"pair-follow >= {gates['pair']:.2f}, dev ref >= base)")
    checks = []
    if "ho-noref" in got:
        checks.append(("held-out noref", got["ho-noref"][name] >= gates["heldout"]))
    if "dev-noref" in got:
        checks.append(("dev noref", got["dev-noref"][name] >= gates["dev"]))
    if "pairs" in got:
        checks.append(("pair-follow", got["pairs"][name] >= gates["pair"]))
    if "dev-ref" in got and refs:
        checks.append(("dev ref vs base", got["dev-ref"][name] >= got["dev-ref"][next(iter(refs))]))
    for label, passed in checks:
        lines.append(f"   {label:18s} {'PASS' if passed else 'fail'}")
        summary["gate"][label] = bool(passed)

    probes = {}
    for record in evaluation.get("remove_pairs", []):
        if record.get("valid"):
            model = MODEL_ALIASES.get(record["model"], record["model"])
            probes.setdefault((model, record["pair"]), {})[record["tok"]] = record
    if probes:
        lines.append("== remove probe ('Remove the object in this region <M>.', token only)")
        summary["remove"] = {}
        for model in ("base", "trained"):
            chosen = sorted(pair for (label, pair), tokens in probes.items() if label == model and len(tokens) == 2)
            if not chosen:
                continue
            tokens = [probes[(model, pair)] for pair in chosen]
            follow = [follows(t, "change_own", "change_other") for t in tokens]
            per_token = np.mean([r["change_own"] > r["change_other"] for t in tokens for r in t.values()])
            selectivity = np.mean([np.mean([r["change_own"] - r["change_other"] for r in t.values()]) for t in tokens])
            far = np.mean([np.mean([r["change_far"] for r in t.values()]) for t in tokens])
            shown = name if model == "trained" else "base"
            lines.append(f"   {shown:8s} pairs {len(chosen)}: pair-follow {boot.value(follow)} per-token {per_token:.3f} "
                         f"selectivity {selectivity:.1f} far-change {far:.1f}")
            summary["remove"][model] = {"pairs": len(chosen), "pair_follow": float(np.mean(follow)),
                                        "per_token": float(per_token), "selectivity": float(selectivity),
                                        "far_change": float(far)}
    edits = evaluation.get("edits", [])
    if edits:
        lines.append("== edit locality (benchmark instructions with phrase + token): mean change inside / far outside")
        summary["edits"] = {}
        for model in ("base", "trained"):
            chosen = [record for record in edits if MODEL_ALIASES.get(record.get("model", "trained"), record.get("model", "trained")) == model]
            if chosen:
                inside, outside = np.mean([r["diff_in"] for r in chosen]), np.mean([r["diff_out"] for r in chosen])
                shown = name if model == "trained" else "base"
                lines.append(f"   {shown:8s} cases {len(chosen)}: {inside:.1f} / {outside:.2f}")
                summary["edits"][model] = {"cases": len(chosen), "inside": float(inside), "outside": float(outside)}
    return lines, summary


def parser():
    p = argparse.ArgumentParser(prog="samtok-edit ground-report", description=__doc__.split("\n\n")[0])
    p.add_argument("--eval", required=True, help="Evaluation directory (or eval.json) of this run")
    p.add_argument("--data", required=True, help="Data directory holding the eval.json item list")
    p.add_argument("--train", help="Training directory, to include the loss curve and quick evals")
    p.add_argument("--name", default="trained", help="Label of this run in the report")
    p.add_argument("--reference", action="append", default=[], metavar="LABEL=PATH",
                   help="Reference evaluation (repeatable); give the base model first")
    p.add_argument("--gate-heldout", type=float, default=0.60)
    p.add_argument("--gate-dev", type=float, default=0.50)
    p.add_argument("--gate-pair", type=float, default=0.70)
    p.add_argument("--output", help="Write the report here (and a .json summary next to it); default: print only")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    items = json.load(open(Path(args.data) / "eval.json"))
    references = {}
    for entry in args.reference:
        label, path = entry.split("=", 1)
        references[label] = load_eval(path)
    lines, summary = build_report(
        items, load_eval(args.eval), references, name=args.name, train_dir=args.train,
        gates={"heldout": args.gate_heldout, "dev": args.gate_dev, "pair": args.gate_pair})
    text = "\n".join(lines) + "\n"
    print(text, end="")
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)
        output.with_suffix(".json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


if __name__ == "__main__":
    main()
