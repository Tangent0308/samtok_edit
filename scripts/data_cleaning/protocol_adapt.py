"""Adapt cleaned pairs to instruction protocol 1.0 (src/samtok_edit21/preparation/protocol_adapt.py).

  pilot  stratified rows from the cleaned manifests;
  slots  Qwen3.8-27B slots and grounding boxes from an OpenAI-compatible server (resumable);
  build  records, summary, review sheets and an HTML index, after protocol_remask.py.
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
    picked = pa.pilot_rows(rows, counts(args.per_type), counts(args.per_text_route), args.seed)
    write_jsonl(out / "rows.jsonl", picked)
    meta = {"no_text_action": args.no_text_action, "no_text_action_sha256": file_hash(args.no_text_action),
            "kept": args.kept, "kept_sha256": file_hash(args.kept), "candidates": len(rows), "picked": len(picked),
            "per_type": counts(args.per_type), "per_text_route": counts(args.per_text_route), "seed": args.seed,
            "code_commit": code_commit(), "time_utc": datetime.datetime.utcnow().isoformat(timespec="seconds")}
    (out / "pilot.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta, indent=1))


async def run_slots(args):
    import aiohttp

    rows = pa.read_jsonl(args.rows)
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
                    record = await pa.extract(client, row, parts, ground)
                    record["effort"] = args.effort
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                    print(f"{record['id']}: {(record.get('slots') or {}).get('decision')} "
                          f"{(record.get('spec') or {}).get('operation')} tokens={record['tokens']}", flush=True)

            await asyncio.gather(*(one(r) for r in todo))


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
        name = f"sheets/{row['id']}.jpg"
        pa.review_sheet(row, record, args.root).save(out / name, quality=88)
        items.append((record, name))
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
    s.add_argument("--output", required=True)
    s = sub.add_parser("slots")
    s.add_argument("--rows", required=True)
    s.add_argument("--root", required=True, help="dataset root the manifest image paths are relative to")
    s.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    s.add_argument("--model", default="qwen38-27b")
    s.add_argument("--effort", default="low")
    s.add_argument("--in-flight", type=int, default=32)
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--output", required=True)
    s = sub.add_parser("build")
    s.add_argument("--rows", required=True)
    s.add_argument("--root", required=True)
    s.add_argument("--slots", required=True)
    s.add_argument("--remask")
    s.add_argument("--output", required=True)
    args = p.parse_args()
    if args.command == "pilot":
        pilot(args)
    elif args.command == "slots":
        asyncio.run(run_slots(args))
    else:
        build(args)


if __name__ == "__main__":
    sys.exit(main())
