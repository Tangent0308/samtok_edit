"""Build Stage A data (v3): token-grounding training rows and the evaluation set.

Two commands::

    samtok-edit ground-data eval-set --output <dir>/eval.json ...     # the fixed evaluation items
    samtok-edit ground-data train --eval-set <eval.json> --output <data_dir> ...

Training rows are *units* - one region token on one image - turned into highlight, procedural and cleaned real-edit
rows (``data.highlight``).  Units come from the stage-2 manifest (single mask-token rows; ``add`` rows use a box token
and are left out) and from GRES referring data, which contributes images with several named instances.  Each row
lists the other tokens on its image (``others``): they are the distractors of the routing loss.

Leakage guard.  The held-out split is by source *path* (``sha256(path) % 10 == 0``), but the same picture is stored
under several paths (several instructions per image, duplicated folders), so a path split alone leaks.  Every training
source whose 16x16 average hash equals that of an evaluation source is therefore dropped as well.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import random
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from samtok_edit21.data.highlight import (
    COLORS, HIGHLIGHT_TEMPLATES, PROCEDURAL_TEMPLATES, article, normalize_phrase, region_reference,
)

ROOT = "/mnt/bn/strategy-mllm-train/user/tanyue"
STAGE2_DATA = f"{ROOT}/experiments2/SAMTokEdit/qwen21_v2/data/train_v2_box_001"
SOURCES = f"{ROOT}/experiments/SAMTokEdit/qwen21_full4_20260928/data/sources.jsonl"
DECISIONS = f"{ROOT}/datasets/SAMTokEdit_Cleaned_20261009/decisions.jsonl"
GRES = f"{ROOT}/datasets/SAMTok_Training_Data/mix_gres8k_ver4k_train.parquet"
EVAL_PROTOCOL = f"{ROOT}/experiments2/SAMTokEdit/qwen21_v2/eval/protocol_v2_001"
SPAN = re.compile(r"<\|mt_start\|><\|mt_\d{4}\|><\|mt_\d{4}\|><\|mt_end\|>")
BOX = re.compile(r"<\|box_start\|>")
CLEAN_REAL_DATASETS = {"refedit", "derived"}   # sources whose kept pairs are pixel-aligned
MAX_OTHERS = 3


def heldout(path):
    """Path-level held-out split shared by every probe since the v2 review."""
    return int(hashlib.sha256(path.encode()).hexdigest(), 16) % 10 == 0


def average_hash(image):
    gray = np.asarray(image.convert("L").resize((16, 16), Image.BILINEAR), np.float32)
    return np.packbits((gray > gray.mean()).flatten()).tobytes().hex()


def average_hash_of(path):
    try:
        with Image.open(path) as image:
            return path, average_hash(image)
    except Exception:
        return path, None


def hash_images(paths, workers=48):
    with ProcessPoolExecutor(workers) as pool:
        return dict(pool.map(average_hash_of, paths, chunksize=64))


def single_token_rows(stage2_data, *, include_add=False):
    """``(line, row, span)`` of stage-2 rows that carry exactly one mask token and no box.

    ``add`` rows select where a new object goes, not an existing instance, so grounding data leaves them out.
    """
    with open(Path(stage2_data) / "stage2.jsonl") as handle:
        for line, text in enumerate(handle):
            row = json.loads(text)
            if row["sample_type"] != "edit_umt" or (row["edit_type"] == "add" and not include_add):
                continue
            spans = SPAN.findall(row["prompt"])
            if len(spans) == 1 and not BOX.search(row["prompt"]):
                yield line, row, spans[0]


def provenance_ids(stage2_data):
    """stage-2 line -> source pair id."""
    records = map(json.loads, open(Path(stage2_data) / "provenance.jsonl"))
    return {record["line"]: record["id"] for record in records if record["file"] == "stage2"}


def reference_phrases(sources):
    """pair id -> annotated referring text, for pairs with exactly one annotated instance."""
    phrases = {}
    for text in open(sources):
        record = json.loads(text)
        instances = record.get("instances") or []
        if len(instances) == 1 and instances[0].get("ref"):
            phrases[record["id"]] = instances[0]["ref"]
    return phrases


def gres_phrases(answer):
    """``[(span, phrase)]`` of a GRES answer: a JSON list with labels, or a sentence with ``phrase (<span>)``."""
    found = []
    block = re.search(r"\[\s*\{.*\}\s*\]", answer or "", re.S)
    if block:
        try:
            for entry in json.loads(block.group(0)):
                spans = SPAN.findall(entry.get("mask_2d", ""))
                if spans:
                    found.append((spans[0], normalize_phrase(entry.get("label"))))
            return found
        except Exception:
            pass
    pattern = r"([A-Za-z][^().;:\n]{1,90}?)\s*\(\s*(" + SPAN.pattern + r")\s*\)"
    for match in re.finditer(pattern, answer or ""):
        found.append((match.group(2), normalize_phrase(re.split(r"\b(?:and|or|while|with)\b", match.group(1))[-1])))
    return found


def build_eval_set(args):
    """Held-out rows + benchmark dev cases (with and without the phrase) + counterfactual pairs."""
    provenance, phrases = provenance_ids(args.stage2_data), reference_phrases(args.sources)
    phrase = lambda text: normalize_phrase(text, legacy=args.legacy_phrases)  # noqa: E731
    pool = [(line, row, span) for line, row, span in single_token_rows(args.stage2_data) if heldout(row["edit_image"])]
    random.Random(1).shuffle(pool)
    items = []
    for line, row, span in pool[:args.heldout]:
        named = row["instr_variant"] == "ref" and provenance.get(line) in phrases
        text = phrase(phrases[provenance[line]]) if named else None
        items.append({"set": "ho", "source": row["edit_image"], "span": span, "variant": "ref" if text else "noref",
                      "phrase": text, "edit_type": row["edit_type"]})
    protocol = Path(args.eval_protocol)
    compiled = {record["case_id"]: record for record in map(json.loads, open(protocol / "compiled.jsonl"))}
    for case in map(json.loads, open(protocol / "cases.jsonl")):
        if case["split"] != "dev" or compiled[case["case_id"]]["region_kind"] != "mask":
            continue
        for variant in ("ref", "noref"):
            text = phrase(compiled[case["case_id"]].get("ref_phrase")) if variant == "ref" else None
            items.append({"set": "dev", "case_id": case["case_id"], "source": case["source_image"],
                          "gt_mask": case["region"]["mask"], "variant": "ref" if text else "noref", "phrase": text,
                          "edit_type": compiled[case["case_id"]]["edit_type"],
                          "instruction": compiled[case["case_id"]]["instruction"]})
    for index, pair in enumerate(json.load(open(args.pairs))):
        for token in ("a", "b"):
            items.append({"set": "pair", "pair": index, "tok": token, "source": pair["source"], "span": pair[token],
                          "other": pair["b" if token == "a" else "a"], "variant": "noref", "phrase": None,
                          "unseen": pair["unseen"]})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    json.dump(items, open(output, "w"), ensure_ascii=False)
    print(json.dumps(dict(collections.Counter(f"{item['set']}/{item['variant']}" for item in items))))
    return items


def quick_indices(items):
    """In-training check: items not flagged as leaky, 40 + 24 held-out, 32 + 32 dev, 16 pairs."""
    clean = lambda item: not any(value for key, value in item.items() if key.startswith("leaky"))  # noqa: E731
    pick = lambda name, variant, count, flag: [  # noqa: E731
        index for index, item in enumerate(items)
        if item["set"] == name and item["variant"] == variant and (not flag or clean(item))][:count]
    chosen = pick("ho", "noref", 40, True) + pick("ho", "ref", 24, True) \
        + pick("dev", "noref", 32, False) + pick("dev", "ref", 32, False)
    pairs = set(sorted({item["pair"] for item in items if item["set"] == "pair" and clean(item)})[:16])
    return chosen + [index for index, item in enumerate(items) if item["set"] == "pair" and item["pair"] in pairs]


def build_training_data(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    items = json.load(open(args.eval_set))
    eval_sources = sorted({item["source"] for item in items})

    # ---- stage-2 units and cleaned real-edit rows (training side of the path split)
    provenance, phrases = provenance_ids(args.stage2_data), reference_phrases(args.sources)
    decisions = {}
    for text in open(args.decisions):
        record = json.loads(text)
        decisions[record["id"]] = (record["decision"], record["dataset"])
    units, real_rows = {}, []
    for line, row, span in single_token_rows(args.stage2_data):
        if heldout(row["edit_image"]):
            continue
        pair_id = provenance.get(line)
        unit = units.setdefault((row["edit_image"], span), {"edit_image": row["edit_image"], "span": span, "phrase": None})
        if unit["phrase"] is None and pair_id in phrases:
            unit["phrase"] = normalize_phrase(phrases[pair_id])
        if pair_id in decisions and decisions[pair_id][0] == "keep" and decisions[pair_id][1] in CLEAN_REAL_DATASETS:
            real_rows.append({"kind": "real", "edit_image": row["edit_image"], "image": row["image"], "span": span,
                              "prompt": row["prompt"], "variant": row["instr_variant"], "edit_type": row["edit_type"]})

    # ---- content-level leakage guard
    candidates = sorted({unit["edit_image"] for unit in units.values()} | set(eval_sources))
    hashes = hash_images(candidates, args.workers)
    eval_hashes = {hashes[path] for path in eval_sources if hashes.get(path)}
    eval_source_set = set(eval_sources)
    leaked = {path for path in candidates if path not in eval_source_set and hashes.get(path) in eval_hashes}
    spans_of = collections.defaultdict(list)
    for unit in units.values():
        spans_of[unit["edit_image"]].append(unit["span"])
    units = [unit for unit in units.values() if unit["edit_image"] not in leaked and hashes.get(unit["edit_image"])]
    others = lambda record: [s for s in spans_of[record["edit_image"]] if s != record["span"]][:MAX_OTHERS]  # noqa: E731
    for unit in units:
        unit["others"] = others(unit)
    real_rows = [row for row in real_rows if row["edit_image"] not in leaked]
    for row in real_rows:
        row["others"] = others(row)
    multi = [unit for unit in units if unit["others"]]
    single = [unit for unit in units if not unit["others"]]
    rng.shuffle(multi)
    rng.shuffle(single)
    rng.shuffle(real_rows)

    # ---- GRES units; their images live inside the parquet and are extracted next to the rows
    import pyarrow.parquet as pq

    table = pq.read_table(args.gres, columns=["answer", "images"])
    image_dir = out / "gres_images"
    image_dir.mkdir(exist_ok=True)
    gres_units, gres_leaked = [], 0
    for index, (answer, images) in enumerate(zip(table.column("answer").to_pylist(), table.column("images").to_pylist())):
        named = gres_phrases(answer)
        if not named:
            continue
        first = images[0] if isinstance(images, list) else images
        data = first.get("bytes") if isinstance(first, dict) else None
        if data is None:
            continue
        try:
            digest = average_hash(Image.open(io.BytesIO(data)))
        except Exception:
            continue
        if digest in eval_hashes:
            gres_leaked += 1
            continue
        path = image_dir / f"{index:05d}.jpg"
        if not path.exists():
            path.write_bytes(data)
        spans = [span for span, _ in named]
        for span, text in named:
            gres_units.append({"edit_image": f"gres_images/{index:05d}.jpg", "span": span, "phrase": text,
                               "others": [s for s in spans if s != span][:MAX_OTHERS], "gres_row": index})
    rng.shuffle(gres_units)

    # ---- rows (the RNG order below is part of the dataset definition)
    def highlight(unit, named):
        text = unit["phrase"] if named else None
        color = rng.choice(COLORS)
        prompt = rng.choice(HIGHLIGHT_TEMPLATES).format(a=article(color), color=color, x=region_reference(unit["span"], text))
        return {"kind": "hl", "edit_image": unit["edit_image"], "span": unit["span"], "others": unit["others"],
                "color": color, "variant": "ref" if text else "noref", "prompt": prompt}

    def procedural(unit, named):
        text = unit["phrase"] if named else None
        operation = rng.choice(list(PROCEDURAL_TEMPLATES))
        return {"kind": "proc", "edit_image": unit["edit_image"], "span": unit["span"], "others": unit["others"],
                "op": operation, "variant": "ref" if text else "noref",
                "prompt": PROCEDURAL_TEMPLATES[operation].format(x=region_reference(unit["span"], text))}

    # every multi-instance unit without the phrase, again with it where one exists; single-instance units alternate
    stage2_rows = [highlight(unit, False) for unit in multi] + [highlight(unit, True) for unit in multi if unit["phrase"]]
    stage2_rows += [highlight(unit, k % 2 == 1) for k, unit in enumerate(single[:max(0, args.highlight - len(stage2_rows))])]
    gres_multi = [unit for unit in gres_units if unit["others"]]
    gres_single = [unit for unit in gres_units if not unit["others"]]
    gres_rows = [highlight(unit, False) for unit in gres_multi] + [highlight(unit, True) for unit in gres_multi if unit["phrase"]]
    gres_rows += [highlight(unit, k % 2 == 1) for k, unit in enumerate(gres_single[:max(0, args.gres_highlight - len(gres_rows))])]
    gres_rows = gres_rows[:args.gres_highlight]
    pool = multi + gres_multi + single[-20000:]
    rng.shuffle(pool)
    procedural_rows = [procedural(unit, k % 5 >= 3) for k, unit in enumerate(pool[:args.procedural])]
    epoch = stage2_rows[:args.highlight] + gres_rows + procedural_rows + real_rows[:args.real]
    with open(out / "epoch_rows.jsonl", "w") as handle:
        for row in epoch:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    schedule = []
    for number in range(args.epochs):
        order = list(range(len(epoch)))
        random.Random(1000 + number).shuffle(order)
        schedule += order
    json.dump(schedule, open(out / "schedule.json", "w"))

    # ---- evaluation items travel with the data; optional flags mark overlap with an earlier training set
    for name, rows_file in (flag.split("=", 1) for flag in args.flag_overlap):
        seen = {json.loads(text)["edit_image"] for text in open(rows_file)}
        seen_hashes = {hashes[path] for path in seen if hashes.get(path)}
        for item in items:
            item[f"leaky_{name}"] = bool(hashes.get(item["source"]) and hashes[item["source"]] in seen_hashes)
    json.dump(items, open(out / "eval.json", "w"), ensure_ascii=False)
    quick = quick_indices(items)
    json.dump(quick, open(out / "eval_quick_idx.json", "w"))

    train_hashes = {hashes[unit["edit_image"]] for unit in units if hashes.get(unit["edit_image"])}
    still_leaking = {path for path in eval_sources if hashes.get(path) in train_hashes}
    stats = {
        "stage2_units": len(units), "multi_units": len(multi), "single_units": len(single),
        "dropped_sources_matching_eval": len(leaked), "real_keep_rows": len(real_rows),
        "gres_units": len(gres_units), "gres_multi_units": len(gres_multi),
        "gres_rows_dropped_matching_eval": gres_leaked, "epoch_rows": len(epoch),
        "kinds": dict(collections.Counter(row["kind"] for row in epoch)),
        "hl_variants": dict(collections.Counter(row["variant"] for row in epoch if row["kind"] == "hl")),
        "rows_with_distractors": sum(bool(row.get("others")) for row in epoch), "schedule_len": len(schedule),
        "eval_items": len(items), "quick_items": len(quick),
        "eval_flags": {key: sum(bool(item.get(key)) for item in items)
                       for key in sorted({k for item in items for k in item if k.startswith("leaky")})},
        "train_sources_still_matching_eval_after_filter": len(still_leaking),
        "inputs": {"stage2_data": str(args.stage2_data), "sources": str(args.sources), "decisions": str(args.decisions),
                   "gres": str(args.gres), "eval_set": str(args.eval_set), "seed": args.seed},
    }
    if still_leaking:
        raise RuntimeError(f"{len(still_leaking)} evaluation pictures are still in the training rows")
    json.dump(stats, open(out / "data_stats.json", "w"), indent=1)
    print(json.dumps(stats, indent=1))
    return stats


def parser():
    p = argparse.ArgumentParser(prog="samtok-edit ground-data", description=__doc__.split("\n\n")[0])
    commands = p.add_subparsers(dest="action", required=True)

    def shared(sub):
        sub.add_argument("--stage2-data", default=STAGE2_DATA, help="Directory with stage2.jsonl and provenance.jsonl")
        sub.add_argument("--sources", default=SOURCES, help="sources.jsonl with annotated instances (referring text)")
        sub.add_argument("--output", required=True)

    train = commands.add_parser("train", help="Training rows, schedule and quick-eval indices")
    shared(train)
    train.add_argument("--eval-set", required=True, help="eval.json; its pictures never enter training")
    train.add_argument("--decisions", default=DECISIONS, help="Per-pair cleaning decisions (keep/drop)")
    train.add_argument("--gres", default=GRES, help="GRES referring parquet (answer, images)")
    train.add_argument("--highlight", type=int, default=50000, help="Highlight rows from stage-2 units")
    train.add_argument("--gres-highlight", type=int, default=12000, help="Highlight rows from GRES units")
    train.add_argument("--procedural", type=int, default=12000)
    train.add_argument("--real", type=int, default=8000, help="Cleaned real-edit rows")
    train.add_argument("--epochs", type=int, default=2)
    train.add_argument("--seed", type=int, default=20261012)
    train.add_argument("--workers", type=int, default=48)
    train.add_argument("--flag-overlap", action="append", default=[], metavar="NAME=ROWS.jsonl",
                       help="Mark eval items whose picture is in that row file as leaky_NAME (repeatable)")

    evalset = commands.add_parser("eval-set", help="Held-out rows, benchmark dev cases and counterfactual pairs")
    shared(evalset)
    evalset.add_argument("--eval-protocol", default=EVAL_PROTOCOL, help="Directory with cases.jsonl / compiled.jsonl")
    evalset.add_argument("--pairs", required=True, help="JSON list of {source, a, b, unseen}: two tokens on one image")
    evalset.add_argument("--heldout", type=int, default=1000, help="Held-out rows to keep")
    evalset.add_argument("--legacy-phrases", action="store_true", help="Phrase normalisation of the first eval set")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    return build_training_data(args) if args.action == "train" else build_eval_set(args)


if __name__ == "__main__":
    main()
