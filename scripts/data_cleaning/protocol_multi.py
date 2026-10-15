"""Multi-object pairs as several protocol 1.0 units edited together (src/samtok_edit21/preparation/protocol_multi.py).

  rows      the pairs the single-unit adaptation left out as multi-object, from its output folder (all or a sample);
  slots     Qwen3.8-27B units of every pair, each grounded like a single record (resumable; --feedback re-asks);
  units     unit rows and unit slots records for protocol_remask.py and protocol_adapt.py parts / build;
  assemble  pair records from the units' built records, with the pair checks;
  audit     an independent Qwen3.8-27B check of every assembled pair (resumable);
  repair    feedback for the pairs that failed (slots --feedback re-asks them);
  final     the gate: a pair stays accepted only with a passing audit;
  write     the dataset folder: manifest_multi.jsonl, records.jsonl, summary.json, config.json, README.md.

The round, as run on one 8-GPU machine, is scripts/data_cleaning/protocol_multi_round.sh.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import datetime
import json
import multiprocessing
from pathlib import Path
import random
import re
import sys

from samtok_edit21.data.io import file_hash
from samtok_edit21.preparation import edit_check as ec
from samtok_edit21.preparation import instruction_protocol as ip
from samtok_edit21.preparation import protocol_adapt as pa
from samtok_edit21.preparation import protocol_multi as pm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import protocol_adapt as single  # noqa: E402  (the single-unit CLI: write_jsonl, code_commit, repair messages)

MANIFEST_FIELDS = ("id", "dataset", "type_before", "instruction_before", "units", "edited_region", "flags",
                   "adapter_version", "multi_version", "instruction_protocol_version")


def rows(args):
    """The multi-object pairs (input rows, in the adaptation's order), all of them or a stratified sample."""
    multi = pa.read_jsonl(Path(args.adaptation) / "multi_object.jsonl")
    wanted = {r["id"]: r["type_before"] for r in multi}
    found = {r["id"]: r for r in pa.load_rows(args.no_text_action, args.kept) if r["id"] in wanted}
    picked = [found[r["id"]] for r in multi]
    if args.sample:
        cells = {}
        for r in picked:
            n = len(r["instances"])
            cells.setdefault((r["type"], "1" if n == 1 else "2-5" if n <= 5 else "6+"), []).append(r)
        rng, chosen = random.Random(args.seed), []
        total = len(picked)
        for key in sorted(cells):
            take = max(args.min_per_cell, round(args.sample * len(cells[key]) / total))
            chosen += rng.sample(cells[key], min(take, len(cells[key])))
        if args.exclude:
            skip = {r["id"] for path in args.exclude for r in pa.read_jsonl(path)}
            chosen = [r for r in chosen if r["id"] not in skip]
        order = {r["id"]: k for k, r in enumerate(picked)}
        picked = sorted(chosen, key=lambda r: order[r["id"]])
    single.write_jsonl(args.output, picked)
    print(json.dumps({"multi_object": len(multi), "rows": len(picked),
                      "by_type": dict(Counter(r["type"] for r in picked))}, indent=1))


async def run_slots(args):
    import aiohttp

    pairs = pa.read_jsonl(args.rows)
    feedback = {f["id"]: f for f in pa.read_jsonl(args.feedback)} if args.feedback else {}
    if args.feedback:
        pairs = [r for r in pairs if r["id"] in feedback]
    done = ec.completed_records(args.output)
    todo = [r for r in pairs if r["id"] not in done]
    loop = asyncio.get_running_loop()
    gate = asyncio.Semaphore(args.in_flight)
    pool = ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn"))
    with pool, open(args.output, "a") as stream:
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0),
                                         timeout=aiohttp.ClientTimeout(total=3600)) as session:
            client = pa.Client(session, args.url, args.model, args.effort)

            async def one(row):
                async with gate:
                    parts = await loop.run_in_executor(pool, pa.slot_parts, row, args.root)
                    ground = await loop.run_in_executor(pool, pa.ground_parts, row, args.root)
                    after = await loop.run_in_executor(pool, pa.ground_parts, row, args.root, "target_image")
                    record = await pm.extract_multi(client, row, parts, ground, after, feedback.get(row["id"]))
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                    print(f"{record['id']}: {(record.get('slots') or {}).get('decision')} "
                          f"{[u['spec']['operation'] for u in record['units']]} tokens={record['tokens']}", flush=True)

            await asyncio.gather(*(one(r) for r in todo))


def units(args):
    """Unit rows and unit slots records of every pair record, for protocol_remask.py, parts and build."""
    pairs = {r["id"]: r for r in pa.read_jsonl(args.rows)}
    out_rows, out_slots = [], []
    for record in ec.completed_records(args.pairs).values():
        for unit_row, unit_record in pm.unit_rows(pairs[record["id"]], record):
            out_rows.append(unit_row)
            out_slots.append({k: v for k, v in unit_record.items() if k != "pair_instances"})
    single.write_jsonl(args.unit_rows, out_rows)
    single.write_jsonl(args.unit_slots, out_slots)
    print(f"{len(out_rows)} units of {len(pairs)} pairs -> {args.unit_rows}")


def assemble_one(job):
    row, record, built, root = job
    return pm.assemble(row, record, built, root)


def assemble(args):
    pairs = {r["id"]: r for r in pa.read_jsonl(args.rows)}
    records = ec.completed_records(args.pairs)
    built = {}
    for r in pa.read_jsonl(args.unit_records):
        built[r["id"]] = r
    jobs = []
    for pid, record in records.items():
        mine = {u["id"]: built.get(u["id"]) for u in record.get("units") or []}
        jobs.append((pairs[pid], record, mine, args.root))
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        out = list(pool.map(assemble_one, jobs, chunksize=2))
    order = {r: k for k, r in enumerate(pairs)}
    out.sort(key=lambda r: order[r["id"]])
    single.write_jsonl(args.output, out)
    print(json.dumps({"pairs": len(out), "status": dict(Counter(r["status"] for r in out)),
                      "reasons": dict(Counter((r.get("reason") or "-").split(";")[0].split(":")[-1].split(",")[0]
                                              for r in out if r["status"] != "accepted").most_common(12))},
                     indent=1))


async def run_audit(args):
    import aiohttp

    pairs = {r["id"]: r for r in pa.read_jsonl(args.rows)}
    records = [r for r in pa.read_jsonl(args.assembled) if r["status"] == "accepted"]
    if args.ids:
        wanted = {line.split()[0] for line in open(args.ids) if line.strip()}
        records = [r for r in records if r["id"] in wanted]
    done = ec.completed_records(args.output)
    todo = [r for r in records if r["id"] not in done]
    loop = asyncio.get_running_loop()
    gate = asyncio.Semaphore(args.in_flight)
    pool = ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn"))
    with pool, open(args.output, "a") as stream:
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0),
                                         timeout=aiohttp.ClientTimeout(total=3600)) as session:
            client = pa.Client(session, args.url, args.model, args.effort)

            async def one(record):
                async with gate:
                    parts = await loop.run_in_executor(pool, pm.multi_audit_parts, pairs[record["id"]], record,
                                                       args.root)
                    result = await pm.multi_audit(client, record, parts, args.votes)
                    stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                    stream.flush()
                    print(f"{record['id']}: passed={result['passed']} {result.get('problems', '')[:160]}", flush=True)

            await asyncio.gather(*(one(r) for r in todo))


GEOMETRY = {"support_not_touching_item": "the new item does not touch S",
            "support_not_under_item": "S is not under the new item it is said to carry",
            "over_without_overlap": "the new item is above or next to S, not draped over or covering it",
            "anchor_not_at_item": "S is nowhere near where the new item appears",
            "reference_direction": "the new item is not above/below S as stated",
            "reference_item_inside_anchor": "the new item overlaps S, so the relation is false",
            "edited_region_misses_s": "the outlined S stays the same in AFTER; another object or part changes",
            "selection_off_edit": "the outlined S is not where the edit happened",
            "edited_region_empty": "nothing changes at S",
            "sam3_failed": "S could not be found from its ref",
            "units_share_s": "two units select the same object",
            "change_not_in_units": "a changed object or area in AFTER belongs to no unit"}


def repair(args):
    """Feedback for the pairs that failed a pair check, a unit's build gate or the audit."""
    audits = {}
    for path in args.audit or []:
        audits.update(ec.completed_records(path))
    out = []
    for r in pa.read_jsonl(args.assembled):
        problems = []
        if r["status"] == "rejected" and r.get("reason") not in ("slots_reject", "no_valid_units"):
            for part in (r.get("reason") or "").split(";"):
                unit, _, reasons = part.rpartition(":")
                for reason in reasons.split(","):
                    message = GEOMETRY.get(reason)
                    if message:
                        problems.append(f"{unit + ': ' if unit else ''}{message}")
        elif r["status"] == "accepted":
            a = audits.get(r["id"])
            if a and a.get("passed") is False:
                problems.append("failed " + ", ".join(pm.audit_failures(a)) + ": " + (a.get("problems") or ""))
            # Units checked again with the current unit rules (a pair asked before a rule was added).
            problems += [f"{u['id']}: {x}" for u in r["units"] for x in pm.unit_problems(u["instruction_spec"])]
        if problems:
            out.append({"id": r["id"], "problems": "; ".join(problems)})
    single.write_jsonl(args.output, out)
    print(f"{len(out)} pairs to repair -> {args.output}")


def final(args):
    """Per pair the latest assembled record (a repair replaces the first), accepted only with a passing audit."""
    audits = {}
    for path in args.audit or []:
        audits.update(ec.completed_records(path))
    latest = {}
    for path in args.assembled:
        for r in pa.read_jsonl(path):
            latest[r["id"]] = r
    out = []
    for r in latest.values():
        if r["status"] == "accepted":
            a = audits.get(r["id"])
            if not a or not a.get("passed"):
                r = {**r, "status": "rejected", "reason": "audit:" + (",".join(pm.audit_failures(a or {})) or "missing"),
                     "audit": a}
            else:
                r = {**r, "audit": a}
        out.append(r)
    single.write_jsonl(args.output, out)
    print(json.dumps({"pairs": len(out), "status": dict(Counter(r["status"] for r in out))}, indent=1))


def write(args):
    """The dataset folder of the recalled pairs, in the adaptation's multi-object order."""
    pairs = pa.read_jsonl(args.rows)
    final_records = {r["id"]: r for r in pa.read_jsonl(args.final)}
    missing = [r["id"] for r in pairs if r["id"] not in final_records]
    if missing:
        raise SystemExit(f"{len(missing)} pairs have no final record, e.g. {missing[:3]}")
    out = Path(args.output)
    if (out / "summary.json").exists():
        raise SystemExit(f"{out} already holds a finished recall; choose a new folder")
    out.mkdir(parents=True, exist_ok=True)
    records = [final_records[r["id"]] for r in pairs]
    images = {r["id"]: {k: r[k] for k in ("source_image", "target_image")} for r in pairs}
    accepted = [{**images[r["id"]], **{k: r.get(k) for k in MANIFEST_FIELDS}} for r in records
                if r["status"] == "accepted"]
    single.write_jsonl(out / "records.jsonl", records)
    single.write_jsonl(out / "manifest_multi.jsonl", accepted)
    status = Counter(r["status"] for r in records)
    unit_ops = Counter(u["instruction_spec"]["operation"] for r in accepted for u in r["units"])
    summary = {
        "pairs": len(records), "status": dict(status), "units": sum(len(r["units"]) for r in accepted),
        "units_per_pair": dict(sorted(Counter(len(r["units"]) for r in accepted).items())),
        "unit_operations": dict(unit_ops.most_common()),
        "pair_operation_mix": dict(Counter("+".join(sorted({u["instruction_spec"]["operation"] for u in r["units"]}))
                                           for r in accepted).most_common()),
        "by_dataset": {d: dict(Counter(r["status"] for r in records if r["dataset"] == d))
                       for d in sorted({r["dataset"] for r in records})},
        "rejected_by_reason": dict(Counter(reason_key(r) for r in records if r["status"] == "rejected").most_common()),
        "code_commit": single.code_commit(), "multi_version": pm.MULTI_VERSION, "adapter_version": pa.ADAPTER_VERSION,
        "time_utc": datetime.datetime.utcnow().isoformat(timespec="seconds"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    config = {"source": str(Path(args.adaptation).resolve()), "rows": str(Path(args.rows).resolve()),
              "rows_sha256": file_hash(args.rows), "multi_params": pm.MULTI_PARAMS,
              "edited_region": {"method": pa.EDITED_METHOD, "params": pa.EDITED_PARAMS}, "edit_gates": pa.EDIT_GATES,
              "anchor": {"method": pa.ANCHOR_METHOD, "params": pa.ANCHOR_PARAMS, "part_params": pa.PART_PARAMS},
              "prompts_sha256": {name: file_hash_text(text) for name, text in
                                 (("MULTI_SLOT_PROMPT", pm.MULTI_SLOT_PROMPT), ("MULTI_AUDIT_PROMPT", pm.MULTI_AUDIT_PROMPT),
                                  ("GROUND_PROMPT", pa.GROUND_PROMPT), ("PART_GROUND_PROMPT", pa.PART_GROUND_PROMPT),
                                  ("PART_PICK_PROMPT", pa.PART_PICK_PROMPT))},
              "protocol_sha256": file_hash(Path(ip.__file__)), "code_commit": summary["code_commit"],
              "model": args.model_note}
    (out / "config.json").write_text(json.dumps(config, indent=1, ensure_ascii=False) + "\n")
    (out / "README.md").write_text(readme(summary))
    print(json.dumps({k: summary[k] for k in ("pairs", "status", "units", "units_per_pair", "unit_operations")},
                     indent=1))


def reason_key(record):
    """A rejected pair's first reason as a code: "audit", a pair check, or "unit:<the unit's build reason>"."""
    first = (record.get("reason") or "-").split(";")[0]
    if first.startswith("audit:"):
        return "audit"
    unit, _, code = first.rpartition(":")
    code = code.split(",")[0]
    return ("unit:" if unit else "") + (code if re.fullmatch(r"[a-z0-9_]+|-", code) else "slots_reject")


def file_hash_text(text):
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()


def readme(summary):
    status = summary["status"]
    return "\n".join([
        "# Multi-object pairs as multi-unit protocol 1.0 records (Qwen3.8-27B, SAM3)",
        "",
        "The pairs the single-unit adaptation (`../protocol_adapt_qwen38_27b_v1/`) left out as multi-object, rewritten "
        "as 2-5 units edited together, as benchmark v2 writes a case. Each unit has its own protocol 1.0 spec, ref and "
        "no-ref sentences, mask semantics, source mask S, execution and edited regions. Rules: "
        "`docs/10_SAMTokEdit_Qwen21_训练数据协议适配.md` in the samtok_edit repository (branch qwen-image-2.1-v2, "
        f"commit {summary['code_commit']}). Image paths are relative to the parent folder.",
        "",
        "| File | Content |",
        "|---|---|",
        "| `manifest_multi.jsonl` | accepted pairs for training: image paths and `units` (U1..Uk) |",
        "| `records.jsonl` | every recalled pair with its full record (units, checks, audit, reasons) |",
        "| `summary.json`, `config.json` | counts; prompts, parameters, hashes and code |",
        "| `work/` | the round's step files and logs |",
        "",
        f"Pairs: {summary['pairs']}; accepted {status.get('accepted', 0)}, single-unit {status.get('single_unit', 0)}, "
        f"rejected {status.get('rejected', 0)}. Units in accepted pairs: {summary['units']}, per pair "
        + ", ".join(f"{k}: {v}" for k, v in summary["units_per_pair"].items()) + ".",
        "",
        "Unit operations: " + ", ".join(f"{k} {v}" for k, v in summary["unit_operations"].items()) + ".",
        "",
        "Rejected by reason: " + ", ".join(f"{k} {v}" for k, v in summary["rejected_by_reason"].items()) + ".",
        "",
    ])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("rows")
    s.add_argument("--adaptation", required=True, help="the single-unit output folder (multi_object.jsonl)")
    s.add_argument("--no-text-action", required=True)
    s.add_argument("--kept", required=True)
    s.add_argument("--sample", type=int, help="about this many pairs, stratified by type and outline count")
    s.add_argument("--min-per-cell", type=int, default=2)
    s.add_argument("--seed", type=int, default=20261015)
    s.add_argument("--exclude", action="append", default=[], help="rows files whose pairs are skipped")
    s.add_argument("--output", required=True)
    for name in ("slots", "audit"):
        s = sub.add_parser(name)
        s.add_argument("--rows", required=True)
        s.add_argument("--root", required=True)
        s.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
        s.add_argument("--model", default="qwen38-27b")
        s.add_argument("--effort", default="low")
        s.add_argument("--in-flight", type=int, default=64)
        s.add_argument("--workers", type=int, default=8)
        s.add_argument("--output", required=True)
        if name == "slots":
            s.add_argument("--feedback", help="repair lines: only these pairs are asked again")
        else:
            s.add_argument("--assembled", required=True)
            s.add_argument("--ids", help="only these pair ids (first column)")
            s.add_argument("--votes", type=int, default=2)
    s = sub.add_parser("units")
    s.add_argument("--rows", required=True)
    s.add_argument("--pairs", required=True)
    s.add_argument("--unit-rows", required=True)
    s.add_argument("--unit-slots", required=True)
    s = sub.add_parser("assemble")
    s.add_argument("--rows", required=True)
    s.add_argument("--root", required=True)
    s.add_argument("--pairs", required=True)
    s.add_argument("--unit-records", required=True, help="records.jsonl of protocol_adapt.py build on the unit rows")
    s.add_argument("--workers", type=int, default=32)
    s.add_argument("--output", required=True)
    s = sub.add_parser("repair")
    s.add_argument("--assembled", required=True)
    s.add_argument("--audit", action="append", default=[])
    s.add_argument("--output", required=True)
    s = sub.add_parser("final")
    s.add_argument("--assembled", action="append", required=True, help="assembled files; later ones win")
    s.add_argument("--audit", action="append", default=[], help="audit files; later ones win")
    s.add_argument("--output", required=True)
    s = sub.add_parser("write")
    s.add_argument("--adaptation", required=True)
    s.add_argument("--rows", required=True)
    s.add_argument("--final", required=True)
    s.add_argument("--model-note", default="Qwen3.8-27B (vLLM 0.28, DP 8), SAM3")
    s.add_argument("--output", required=True)
    args = p.parse_args()
    if args.command == "rows":
        rows(args)
    elif args.command == "slots":
        asyncio.run(run_slots(args))
    elif args.command == "units":
        units(args)
    elif args.command == "assemble":
        assemble(args)
    elif args.command == "audit":
        asyncio.run(run_audit(args))
    elif args.command == "repair":
        repair(args)
    elif args.command == "final":
        final(args)
    else:
        write(args)


if __name__ == "__main__":
    sys.exit(main())
