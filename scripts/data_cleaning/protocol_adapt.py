"""Adapt cleaned pairs to instruction protocol 1.0 (src/samtok_edit21/preparation/protocol_adapt.py).

  pilot   stratified rows from the cleaned manifests;
  slots   Qwen3.8-27B slots and grounding boxes from an OpenAI-compatible server (resumable);
  build   records, summary, review sheets and an HTML index, after protocol_remask.py;
  audit   an independent Qwen3.8-27B pass/fail check of every accepted record (resumable);
  report  records + audit: rates per source and type, and the rows to look at;
  repair  feedback for the rows that failed the audit (slots --feedback re-asks them, re-masking S when
          the audit faults the outline);
  final   the gate: only records that passed the audit stay accepted.
"""
from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ProcessPoolExecutor
import datetime
import html
import json
import multiprocessing
from pathlib import Path
import random
import subprocess
import sys

from samtok_edit21.data.io import file_hash
from samtok_edit21.preparation import edit_check as ec
from samtok_edit21.preparation import protocol_adapt as pa

REPO = Path(__file__).resolve().parents[2]


def code_commit():
    run = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True)  # noqa: E731
    commit = run("rev-parse", "HEAD").stdout.strip()
    dirty = run("status", "--porcelain", "--", "src", "scripts").stdout.strip()
    return commit + ("+dirty" if dirty else "")


def counts(spec):
    return {k: int(v) for k, v in (item.split("=") for item in spec.split(",") if item)}


def write_jsonl(path, rows):
    with open(path, "w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def pilot(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "rows.jsonl").exists():
        raise FileExistsError(out / "rows.jsonl")
    rows = pa.load_rows(args.no_text_action, args.kept)
    excluded = {r["id"] for path in args.exclude for r in pa.read_jsonl(path)}
    rows = [r for r in rows if r["id"] not in excluded]
    picked = pa.pilot_rows(rows, counts(args.per_type), counts(args.per_text_route), args.seed)
    if args.extra_multi:
        # Extra non-add rows whose label has several source instances: the multi-object / pile policy.
        chosen = {r["id"] for r in picked}
        multi = [r for r in rows if r["id"] not in chosen and r["type"] not in ("add", "text")
                 and sum(pa.instance_side(i) == "source" for i in r["instances"]) >= 2]
        picked += random.Random(args.seed).sample(multi, min(args.extra_multi, len(multi)))
    write_jsonl(out / "rows.jsonl", picked)
    meta = {"no_text_action": args.no_text_action, "no_text_action_sha256": file_hash(args.no_text_action),
            "kept": args.kept, "kept_sha256": file_hash(args.kept), "candidates": len(rows), "picked": len(picked),
            "per_type": counts(args.per_type), "per_text_route": counts(args.per_text_route), "seed": args.seed,
            "exclude": args.exclude, "extra_multi": args.extra_multi,
            "code_commit": code_commit(), "time_utc": datetime.datetime.utcnow().isoformat(timespec="seconds")}
    (out / "pilot.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta, indent=1))


async def run_slots(args):
    import aiohttp

    rows = pa.read_jsonl(args.rows)
    feedback = {f["id"]: f for f in pa.read_jsonl(args.feedback)} if args.feedback else {}
    if args.feedback:
        rows = [r for r in rows if r["id"] in feedback]
    done = ec.completed_records(args.output)
    todo = [r for r in rows if r["id"] not in done]
    loop = asyncio.get_running_loop()
    slots = asyncio.Semaphore(args.in_flight)
    pool = ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn"))
    with pool, open(args.output, "a") as stream:
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0),
                                         timeout=aiohttp.ClientTimeout(total=3600)) as session:
            client = pa.Client(session, args.url, args.model, args.effort)

            async def one(row):
                async with slots:
                    parts = await loop.run_in_executor(pool, pa.slot_parts, row, args.root)
                    ground = await loop.run_in_executor(pool, pa.ground_parts, row, args.root)
                    record = await pa.extract(client, row, parts, ground, feedback.get(row["id"]))
                    record["effort"] = args.effort
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                    print(f"{record['id']}: {(record.get('slots') or {}).get('decision')} "
                          f"{(record.get('spec') or {}).get('operation')} tokens={record['tokens']}", flush=True)

            await asyncio.gather(*(one(r) for r in todo))


async def run_audit(args):
    import aiohttp

    rows = {r["id"]: r for r in pa.read_jsonl(args.rows)}
    records = [r for r in pa.read_jsonl(args.records) if r["status"] == "accepted"]
    if args.ids:
        wanted = {line.split()[0] for line in open(args.ids) if line.strip()}
        records = [r for r in records if r["id"] in wanted]
    done = ec.completed_records(args.output)
    todo = [r for r in records if r["id"] not in done]
    loop = asyncio.get_running_loop()
    slots = asyncio.Semaphore(args.in_flight)
    pool = ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn"))
    with pool, open(args.output, "a") as stream:
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0),
                                         timeout=aiohttp.ClientTimeout(total=3600)) as session:
            client = pa.Client(session, args.url, args.model, args.effort)

            async def one(record):
                async with slots:
                    parts = await loop.run_in_executor(pool, pa.audit_parts, rows[record["id"]], record, args.root)
                    result = await pa.audit(client, record, parts, args.votes)
                    stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                    stream.flush()
                    print(f"{record['id']}: passed={result['passed']} {result.get('problems', '')}", flush=True)

            await asyncio.gather(*(one(r) for r in todo))


def read_audits(paths):
    audits = {}
    for path in paths or []:
        audits.update(ec.completed_records(path))  # later files (re-checks) win
    return audits


def repair(args):
    """Feedback lines for every accepted record whose audit failed; outline or target faults re-mask S."""
    audits = read_audits(args.audit)
    out = []
    geometry = {"reference_item_far": "the new item is far from S, so near/beside is false",
                "reference_side": "the new item is not on the stated side of S",
                "reference_direction": "the new item is not above/below S as stated",
                "reference_item_inside_anchor": "the new item overlaps S, so beside/under/above/below is false"}
    for r in pa.read_jsonl(args.records):
        reasons = (r.get("reason") or "").split(",")
        if r["status"] == "rejected" and set(reasons) <= set(geometry):
            # Geometry contradicts the stated relation: ask again for a true relation or a closer reference object.
            out.append({"id": r["id"], "failed": ["relation"], "remask": True,
                        "problems": "relation failed: " + "; ".join(geometry[x] for x in reasons)
                                    + ". Use a relation that is true in AFTER, or an object the new item touches or stands next to"})
            continue
        a = audits.get(r["id"])
        if r["status"] != "accepted" or not a or a.get("passed") is not False:
            continue
        failed = [k for k in pa.AUDIT_KEYS if a.get(k) == "fail"]
        out.append({"id": r["id"], "failed": failed, "remask": bool({"target", "outline"} & set(failed)),
                    "problems": f"{', '.join(failed)} failed: {a.get('problems') or 'no details'}"})
    write_jsonl(args.output, out)
    print(f"{len(out)} rows to repair ({sum(x['remask'] for x in out)} with a new mask) -> {args.output}")


def final(args):
    """Accepted records keep their status only with a passing audit; the rest are rejected with the reason."""
    from collections import Counter

    audits = read_audits(args.audit)
    out, count = [], Counter()
    for r in pa.read_jsonl(args.records):
        if r["status"] == "accepted":
            a = audits.get(r["id"])
            if not a or not a.get("passed"):
                failed = [k for k in pa.AUDIT_KEYS if (a or {}).get(k) == "fail"]
                r = {**r, "status": "rejected", "reason": "audit:" + (",".join(failed) or "missing"),
                     "audit": a}
            else:
                r = {**r, "audit": a}
        count[(r["type_before"], r["status"])] += 1
        out.append(r)
    write_jsonl(args.output, out)
    summary = {"rows": len(out), "status": dict(Counter(r["status"] for r in out)),
               "by_type": {f"{t}:{s}": n for (t, s), n in sorted(count.items())}}
    Path(args.output).with_suffix(".summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


def report(args):
    from collections import Counter, defaultdict

    records = pa.read_jsonl(args.records)
    audits = read_audits(args.audit)
    cells = defaultdict(Counter)
    look = []
    for r in records:
        op = (r.get("instruction_spec") or {}).get("operation") or "-"
        cell = cells[f"{r['dataset']}/{r['type_before']}"]
        cell[r["status"]] += 1
        if r["status"] != "accepted":
            continue
        a = audits.get(r["id"])
        failed = [k for k in pa.AUDIT_KEYS if a and a.get(k) == "fail"]
        cell["audit_pass" if a and a.get("passed") else "audit_fail" if a else "audit_missing"] += 1
        soft = [f for f in r["flags"]
                if not f.startswith(("fix:", "noref_length", "new_values", "retyped", "no_execution_region"))]
        if failed or soft:
            look.append({"id": r["id"], "op": op, "audit_failed": failed, "problems": (a or {}).get("problems", ""),
                         "flags": soft, "ref": r["instruction_ref"]})
    summary = {"cells": {k: dict(v) for k, v in sorted(cells.items())},
               "accepted": sum(r["status"] == "accepted" for r in records),
               "audit_pass": sum(1 for r in records if audits.get(r["id"], {}).get("passed")),
               "audit_failed_by_point": dict(Counter(k for x in look for k in x["audit_failed"])),
               "to_look_at": len(look)}
    Path(args.output).write_text(json.dumps({"summary": summary, "look": look}, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=1, ensure_ascii=False))


def save_sheet(job):
    row, record, root, path = job
    pa.review_sheet(row, record, root).save(path, quality=88)


def build(args):
    out = Path(args.output)
    (out / "sheets").mkdir(parents=True, exist_ok=True)
    rows = pa.read_jsonl(args.rows)
    extracted = ec.completed_records(args.slots)
    remasked = ec.completed_records(args.remask) if args.remask and Path(args.remask).exists() else {}
    records, items = [], []
    for row in rows:
        derived_exec = pa.derived_execution_region(row) if row["dataset"] == "derived" and row["type"] == "remove" else None
        record = pa.build(row, extracted.get(row["id"], {}), remasked.get(row["id"]), derived_exec)
        records.append(record)
        items.append((record, f"sheets/{row['id']}.jpg"))
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        list(pool.map(save_sheet, [(row, record, args.root, str(out / name))
                                   for row, (record, name) in zip(rows, items)], chunksize=4))
    write_jsonl(out / "records.jsonl", records)
    summary = {**pa.summarize(records), "adapter_version": pa.ADAPTER_VERSION, "protocol_source": pa.PROTOCOL_SOURCE,
               "protocol_sha256": file_hash(Path(pa.ip.__file__)), "code_commit": code_commit(),
               "time_utc": datetime.datetime.utcnow().isoformat(timespec="seconds")}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    cells = []
    for record, name in items:
        spec = record.get("instruction_spec") or {}
        text = [f"<b>{html.escape(record['id'])}</b> {record['dataset']}/{record['type_before']} &rarr; "
                f"{spec.get('operation', '-')} <b>{record['status']}</b> {html.escape(str(record.get('reason') or ''))}",
                "OLD: " + html.escape(record["instruction_before"])]
        if spec:
            text += ["REF: " + html.escape(record["instruction_ref"]), "NOREF: " + html.escape(record["instruction_noref"]),
                     "flags: " + html.escape(", ".join(record["flags"]) or "none")]
        cells.append(f'<div class="c"><p>{"<br>".join(text)}</p><a href="{name}"><img src="{name}"></a></div>')
    (out / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Protocol 1.0 pilot</title><style>body{font:14px sans-serif;"
        "margin:16px}.c{margin:0 0 28px}img{max-width:100%}</style>" + "".join(cells))
    print(json.dumps(summary, indent=1, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("pilot")
    s.add_argument("--no-text-action", required=True)
    s.add_argument("--kept", required=True, help="manifest_kept.jsonl, for the text pairs")
    s.add_argument("--per-type", default="add=3,remove=2,replace=2,attribute=2")
    s.add_argument("--per-text-route", default="attribute=2,remove=1,add=1")
    s.add_argument("--seed", type=int, default=20261012)
    s.add_argument("--exclude", action="append", default=[], help="rows files whose ids are skipped")
    s.add_argument("--extra-multi", type=int, default=0, help="extra non-add rows with several source instances")
    s.add_argument("--output", required=True)
    s = sub.add_parser("slots")
    s.add_argument("--rows", required=True)
    s.add_argument("--root", required=True, help="dataset root the manifest image paths are relative to")
    s.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    s.add_argument("--model", default="qwen38-27b")
    s.add_argument("--effort", default="low")
    s.add_argument("--in-flight", type=int, default=32)
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--feedback", help="repair lines: only these rows are asked again, with the reviewer's words")
    s.add_argument("--output", required=True)
    s = sub.add_parser("build")
    s.add_argument("--rows", required=True)
    s.add_argument("--root", required=True)
    s.add_argument("--slots", required=True)
    s.add_argument("--remask")
    s.add_argument("--workers", type=int, default=16)
    s.add_argument("--output", required=True)
    s = sub.add_parser("audit")
    s.add_argument("--rows", required=True)
    s.add_argument("--root", required=True)
    s.add_argument("--records", required=True, help="records.jsonl from build")
    s.add_argument("--ids", help="only these ids (first column)")
    s.add_argument("--votes", type=int, default=1, help="independent audit calls; every one must pass")
    s.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    s.add_argument("--model", default="qwen38-27b")
    s.add_argument("--effort", default="low")
    s.add_argument("--in-flight", type=int, default=32)
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--output", required=True)
    for name in ("report", "repair", "final"):
        s = sub.add_parser(name)
        s.add_argument("--records", required=True)
        s.add_argument("--audit", action="append", default=[], help="audit files; later ones win")
        s.add_argument("--output", required=True)
    args = p.parse_args()
    if args.command == "pilot":
        pilot(args)
    elif args.command == "slots":
        asyncio.run(run_slots(args))
    elif args.command == "audit":
        asyncio.run(run_audit(args))
    elif args.command == "report":
        report(args)
    elif args.command == "repair":
        repair(args)
    elif args.command == "final":
        final(args)
    else:
        build(args)


if __name__ == "__main__":
    sys.exit(main())
