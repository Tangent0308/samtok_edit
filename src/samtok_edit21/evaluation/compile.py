"""Compile v2 benchmark prompts with the training noref converter (D2).

``sources`` writes the cases as semantic-converter source records; the
converter itself (``samtok_edit21.preparation.semantic``, 9B model + rule
fallback, the same code and prompt that built the training noref rows) runs
on them in the annotation environment.  ``compile`` turns its annotations
into one region-bound noref template per case, with ``{region}`` where the
setting's region tokens go, and falls back to the deterministic interactive
template when the converter fails.  Templates are validated with the training
noref syntax; ``review.md`` samples them per type for a manual check.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from samtok_edit21.data.io import write_json
from samtok_edit21.data.protocol import (
    REGION_KIND, Unit, box_of, interactive_prompt, render_units, span_of, validate_inline,
)

SENTINEL = {"mask": span_of([0, 256]), "box": box_of((0, 0, 1000, 1000))}
PINNED_TYPES = {"add": "add", "remove": "remove"}  # like dataset-mapped training types
TEMPLATE_TYPES = {"add": "add", "remove": "remove", "replace": "replace"}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def semantic_sources(cases):
    """Atomic cases as converter inputs: one existing region ('union')."""
    return [{"id": case["case_id"], "dataset": case["source_dataset"], "instruction": case["instruction"],
             "native_type": case["benchmark_type"], "provisional_type": PINNED_TYPES.get(case["benchmark_type"]),
             "instances": [{"instance_id": "region", "ref": "", "grounding_image": "source"}],
             "edit_image": case["source_image"], "image": case["source_image"], "grounding": {}}
            for case in cases]


def compile_case(case, result):
    """One noref template with ``{region}``; converter output first, template fallback second."""
    entry = {"case_id": case["case_id"], "instruction": case["instruction"]}
    units = (result or {}).get("annotation", {}).get("units", [])
    noref = (result or {}).get("annotation", {}).get("noref_instruction", "")
    if result and result["status"] == "accepted" and len(units) == 1 and noref.count("{mask_0}") == 1:
        edit_type = units[0]["edit_type"]
        template = noref.replace("{mask_0}", "{region}")
        entry.update(method=result["conversion_method"], ref_phrase=units[0]["ref_phrase"],
                     type_resolution=units[0].get("type_resolution"))
    else:
        edit_type = TEMPLATE_TYPES[case["benchmark_type"]]
        sentinel = SENTINEL[REGION_KIND[edit_type]]
        template = interactive_prompt(case["instruction"], [[sentinel]]).replace(sentinel, "{region}")
        entry.update(method="template", ref_phrase=None, type_resolution="benchmark_type",
                     converter_status=(result or {}).get("status", "missing"))
    entry.update(edit_type=edit_type, region_kind=REGION_KIND[edit_type], noref_template=template)
    sentinel = SENTINEL[entry["region_kind"]]
    try:
        validate_inline(template.replace("{region}", sentinel), "noref", edit_type)
        entry["valid_noref"], entry["invalid_reason"] = True, None
    except ValueError as exc:
        entry["valid_noref"], entry["invalid_reason"] = False, str(exc)
    # The ref variant (original instruction + region tokens) is the other
    # trained format; it keeps every attribute the noref rewrite may drop.
    entry["ref_template"] = None
    if entry["ref_phrase"]:
        try:
            unit = Unit(entry["ref_phrase"], (sentinel,), edit_type, units[0].get("anchor_phrase"))
            ref = render_units(case["instruction"], [unit], variant="ref")
            validate_inline(ref, "ref", edit_type)
            entry["ref_template"] = ref.replace(sentinel, "{region}")
        except ValueError:
            pass
    return entry


def review_markdown(compiled, cases, per_type=50, seed=0):
    by_case = {case["case_id"]: case for case in cases}
    groups = defaultdict(list)
    for entry in compiled:
        groups[entry["edit_type"]].append(entry)
    lines = ["# v2 benchmark noref prompts: manual review sample", "",
             f"Up to {per_type} cases per compiled edit type (seed {seed}). `{{region}}` marks the region tokens.", ""]
    for edit_type in sorted(groups):
        rows = sorted(groups[edit_type], key=lambda e: e["case_id"])
        random.Random(seed).shuffle(rows)
        lines += [f"## {edit_type} ({len(groups[edit_type])} cases, showing {min(per_type, len(rows))})", "",
                  "| case | benchmark type | instruction | noref template | ref template | method | valid |",
                  "|---|---|---|---|---|---|---|"]
        for entry in rows[:per_type]:
            cells = [entry["case_id"], by_case[entry["case_id"]]["benchmark_type"], entry["instruction"],
                     entry["noref_template"], entry.get("ref_template") or "—", entry["method"],
                     "yes" if entry["valid_noref"] else entry["invalid_reason"]]
            lines.append("| " + " | ".join(str(c).replace("|", "\\|") for c in cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sources = sub.add_parser("sources")
    sources.add_argument("--cases", required=True)
    sources.add_argument("--output", required=True)
    compile_parser = sub.add_parser("compile")
    compile_parser.add_argument("--cases", required=True)
    compile_parser.add_argument("--annotations", required=True, help="Converter output directory")
    compile_parser.add_argument("--output", required=True, help="compiled.jsonl")
    args = parser.parse_args(argv)
    cases = read_jsonl(args.cases)
    if args.command == "sources":
        Path(args.output).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in semantic_sources(cases)))
        print(json.dumps({"sources": len(cases)}))
        return
    results = {}
    for path in sorted(Path(args.annotations).glob("annotations-*.jsonl")):
        for result in read_jsonl(path):
            results[result["id"]] = result
    compiled = [compile_case(case, results.get(case["case_id"])) for case in cases]
    output = Path(args.output)
    output.write_text("".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in compiled))
    output.with_name("review.md").write_text(review_markdown(compiled, cases))
    summary = {"cases": len(compiled), "methods": dict(Counter(e["method"] for e in compiled)),
               "edit_types": dict(Counter(e["edit_type"] for e in compiled)),
               "benchmark_to_compiled_type": dict(Counter(
                   f"{c['benchmark_type']}->{e['edit_type']}" for c, e in zip(cases, compiled))),
               "invalid": [e["case_id"] for e in compiled if not e["valid_noref"]],
               "without_ref_template": [e["case_id"] for e in compiled if not e["ref_template"]]}
    write_json(output.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
