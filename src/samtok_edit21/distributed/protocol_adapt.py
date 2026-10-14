"""ARNOLD protocol 1.0 adaptation of the cleaned pairs: 4 nodes x 8 GPUs, one data-parallel vLLM server per node.

Every node copies Qwen3.8-27B to local disk and serves it over its eight GPUs (the server of
samtok_edit21.distributed.edit_check), takes every 4th row of the input (the kept pairs without text and
action, and the text pairs, as protocol_adapt.load_rows) and runs the validated round of
scripts/data_cleaning on it: slots, SAM3 re-mask on all eight GPUs, part picks, build, two-vote audit, one
repair pass (slots, re-mask, part picks, build, audit of the repaired rows) and the final gate.  Every step
appends per row, so a failed run resumes with --resume-from.  Node 0 merges the nodes' records into the output
folder inside the cleaned dataset.  The rules are docs/10_SAMTokEdit_Qwen21_训练数据协议适配.md.  No DDP, no W&B.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import re
import shutil
import signal
import subprocess
import sys
import time

from samtok_edit21.data.io import file_hash
from samtok_edit21.distributed.edit_check import MODEL, EditCheckPipeline, copy_model
from samtok_edit21.distributed.training import atomic_json, record_failure
from samtok_edit21.preparation import edit_check as ec
from samtok_edit21.preparation import protocol_adapt as pa

SAM3_CHECKPOINT = "/mnt/bn/strategy-mllm-train/common/models/sam3/sam3.pt"
CLI = "scripts/data_cleaning/protocol_adapt.py"
REMASK = "scripts/data_cleaning/protocol_remask.py"
# Fields of an accepted record that the training manifest keeps (the full record stays in records.jsonl).
MANIFEST_FIELDS = ("id", "dataset", "type_before", "instruction_before", "instruction_spec", "instruction_ref",
                   "instruction_noref", "mask_semantics", "selection", "host_view", "execution_region",
                   "edited_region", "flags", "adapter_version", "instruction_protocol_version")
# Work files a resumed run copies; every step skips the rows they already hold.
RESUMABLE = ("slots.jsonl", "parts.jsonl", "audit.jsonl", "slots_fix.jsonl", "parts_fix.jsonl", "audit_fix.jsonl")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def count_lines(path):
    try:
        with open(path, "rb") as stream:
            return sum(1 for _ in stream)
    except FileNotFoundError:
        return 0


def merged_records(paths):
    """Records of several append-only files by id; later files win."""
    out = {}
    for path in paths:
        out.update(ec.completed_records(path))
    return out


def write_jsonl(path, rows):
    tmp = Path(path).with_suffix(".tmp")
    with open(tmp, "w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


class ProtocolPipeline(EditCheckPipeline):
    """The edit check's server, model copy and node bookkeeping; a protocol round instead of the check."""

    def __init__(self, args):
        super().__init__(args)
        self.work = self.root / "work" / f"node{self.rank}"

    def cli(self, *args):
        return [sys.executable, "-B", CLI, *args]

    def run_step(self, phase, command, progress=None):
        """One step as a subprocess with its log; `progress` (paths, total) adds a row count to the minute line."""
        atomic_json(self.node / f"{phase}.command.json", command)
        print(f"[node {self.rank}] {phase} started", flush=True)
        started = time.monotonic()
        progress = progress and (*progress, sum(count_lines(p) for p in progress[0]))  # rows a resumed run starts with
        with (self.logdir / f"{phase}.log").open("w") as stream:
            process = subprocess.Popen(command, cwd=self.repo, env=self.env, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            self.watch(phase, [process], started, progress)
        print(f"[node {self.rank}] {phase} finished in {time.monotonic() - started:.0f} s", flush=True)

    def run_parallel(self, phase, commands, progress=None):
        """Commands at once, one per GPU (CUDA_VISIBLE_DEVICES); fails when any of them fails."""
        atomic_json(self.node / f"{phase}.command.json", commands)
        print(f"[node {self.rank}] {phase} started on {len(commands)} GPUs", flush=True)
        started, processes, streams = time.monotonic(), [], []
        progress = progress and (*progress, sum(count_lines(p) for p in progress[0]))
        try:
            for gpu, command in enumerate(commands):
                stream = (self.logdir / f"{phase}.gpu{gpu}.log").open("w")
                streams.append(stream)
                env = {**self.env, "CUDA_VISIBLE_DEVICES": str(gpu)}
                processes.append(subprocess.Popen(command, cwd=self.repo, env=env, stdout=stream,
                                                  stderr=subprocess.STDOUT, start_new_session=True))
            self.watch(phase, processes, started, progress)
        finally:
            for stream in streams:
                stream.close()
        print(f"[node {self.rank}] {phase} finished in {time.monotonic() - started:.0f} s", flush=True)

    def watch(self, phase, processes, started, progress):
        last, deadline = time.monotonic(), started + self.args.timeout
        try:
            while any(p.poll() is None for p in processes):
                self.check_failures()
                if self.server is not None and self.server.poll() is not None:
                    raise RuntimeError(f"vLLM server exited with {self.server.returncode}; see {self.server_log}")
                now = time.monotonic()
                if now - last >= self.args.report_every:
                    status = {"phase": phase, "elapsed_seconds": int(now - started)}
                    if progress:
                        paths, total, base = progress
                        done = sum(count_lines(p) for p in paths)
                        rate = (done - base) / max(now - started, 1)
                        status.update(done=done, total=total, rows_per_second=round(rate, 2),
                                      eta_hours=round(max(total - done, 0) / rate / 3600, 2) if rate > 0 else None)
                    atomic_json(self.node / "progress.json", status)
                    print(f"[node {self.rank}] {json.dumps(status)}", flush=True)
                    last = now
                if now > deadline:
                    raise TimeoutError(f"Phase timed out: {phase}")
                time.sleep(2)
            failed = [p.returncode for p in processes if p.returncode]
            if failed:
                raise RuntimeError(f"{phase} exited {failed}; see {self.logdir}/{phase}*.log")
        finally:
            for p in processes:
                if p.poll() is None:
                    try:
                        os.killpg(p.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass

    def sam3(self, phase, rows, slots, output):
        """SAM3 re-mask of the rows in `slots` that need it, split over the node's GPUs, merged into `output`."""
        gpus, w = self.topo["gpus"], self.work
        outputs = [w / f"{output.stem}.gpu{g}.jsonl" for g in range(gpus)]
        jobs = sum(1 for e in ec.completed_records(slots).values()
                   if (e.get("plan") or {}).get("source") == "sam3" or (e.get("plan") or {}).get("part"))
        commands = [[sys.executable, "-B", REMASK, "--rows", str(rows), "--slots", str(slots),
                     "--root", str(self.args.root), "--labeling-repo", str(self.args.labeling_repo),
                     "--checkpoint", str(self.sam3_checkpoint), "--part", f"{g}/{gpus}", "--output", str(o)]
                    for g, o in enumerate(outputs)]
        self.run_parallel(phase, commands, (outputs, jobs))
        write_jsonl(output, merged_records(outputs).values())

    def round(self, rows_path, url):
        """The validated round of scripts/data_cleaning (run_round_full.sh) on this node's rows."""
        a, w = self.args, self.work
        total = count_lines(rows_path)
        server = ["--url", url, "--model", a.served_model_name, "--in-flight", str(a.rows_in_flight),
                  "--workers", str(a.workers)]
        common = ["--rows", str(rows_path), "--root", str(a.root)]
        self.run_step("slots", self.cli("slots", *common, *server, "--output", str(w / "slots.jsonl")),
                      ([w / "slots.jsonl"], total))
        self.sam3("remask", rows_path, w / "slots.jsonl", w / "remask.jsonl")
        self.run_step("parts", self.cli("parts", *common, "--slots", str(w / "slots.jsonl"), "--remask",
                                        str(w / "remask.jsonl"), *server, "--output", str(w / "parts.jsonl")))
        self.run_step("build", self.cli("build", *common, "--slots", str(w / "slots.jsonl"), "--remask",
                                        str(w / "parts.jsonl"), "--workers", str(a.build_workers), "--no-sheets",
                                        "--output", str(w / "build")))
        views = sum(1 + bool(r.get("host_view")) for r in pa.read_jsonl(w / "build/records.jsonl")
                    if r["status"] == "accepted")  # every record, and the host view of those that have one
        self.run_step("audit", self.cli("audit", *common, "--records", str(w / "build/records.jsonl"), *server,
                                        "--votes", str(a.votes), "--output", str(w / "audit.jsonl")),
                      ([w / "audit.jsonl"], views))
        self.run_step("repair", self.cli("repair", "--records", str(w / "build/records.jsonl"), "--audit",
                                         str(w / "audit.jsonl"), "--output", str(w / "repair.jsonl")))
        repaired = count_lines(w / "repair.jsonl")
        if repaired:
            self.run_step("slots_fix", self.cli("slots", *common, *server, "--feedback", str(w / "repair.jsonl"),
                                                "--output", str(w / "slots_fix.jsonl")),
                          ([w / "slots_fix.jsonl"], repaired))
            self.sam3("remask_fix", rows_path, w / "slots_fix.jsonl", w / "remask_fix.jsonl")
            self.run_step("parts_fix", self.cli("parts", *common, "--slots", str(w / "slots_fix.jsonl"), "--remask",
                                                str(w / "remask_fix.jsonl"), *server, "--output",
                                                str(w / "parts_fix.jsonl")))
        fixed = ec.completed_records(w / "slots_fix.jsonl")
        write_jsonl(w / "slots_final.jsonl", {**ec.completed_records(w / "slots.jsonl"), **fixed}.values())
        masks = {k: v for k, v in ec.completed_records(w / "parts.jsonl").items() if k not in fixed}
        write_jsonl(w / "remask_final.jsonl", {**masks, **ec.completed_records(w / "parts_fix.jsonl")}.values())
        (w / "repaired_ids.txt").write_text("".join(k + "\n" for k in fixed))
        self.run_step("build_final", self.cli("build", *common, "--slots", str(w / "slots_final.jsonl"), "--remask",
                                              str(w / "remask_final.jsonl"), "--workers", str(a.build_workers),
                                              "--no-sheets", "--output", str(w / "build_final")))
        if fixed:
            self.run_step("audit_fix", self.cli("audit", *common, "--records", str(w / "build_final/records.jsonl"),
                                                "--ids", str(w / "repaired_ids.txt"), *server, "--votes",
                                                str(a.votes), "--output", str(w / "audit_fix.jsonl")))
        audits = ["--audit", str(w / "audit.jsonl"), "--audit", str(w / "audit_fix.jsonl")]
        self.run_step("final", self.cli("final", "--records", str(w / "build_final/records.jsonl"), *audits,
                                        "--output", str(w / "final.jsonl")))
        self.run_step("report", self.cli("report", "--records", str(w / "build_final/records.jsonl"), *audits,
                                         "--output", str(w / "report.json")))

    def merge(self, rows, output, common, timing):
        """Node 0: the nodes' final records into the output folder, in input order."""
        finals = [self.root / "work" / f"node{rank}" / "final.jsonl" for rank in range(self.topo["nodes"])]
        return merge(rows, finals, output, common, timing)

    def run(self):
        args = self.args
        output = Path(args.output).resolve()
        if (output / "summary.json").exists():
            raise RuntimeError(f"{output} already holds a finished adaptation; choose a new output folder")
        font = ec.font_path()
        if not Path(font).is_file() or file_hash(font) != ec.FONT_SHA256:
            raise RuntimeError(f"Label font {font} missing or not the validated DejaVuSans-Bold; set SAMTOK_CHECK_FONT")
        labeling = subprocess.run(["git", "-C", str(args.labeling_repo), "rev-parse", "HEAD"], capture_output=True,
                                  text=True).stdout.strip()
        if args.labeling_commit and labeling != args.labeling_commit:
            raise RuntimeError(f"Labeling repo at {labeling}, expected {args.labeling_commit}")
        rows = pa.load_rows(args.no_text_action, args.kept)
        if args.sample:
            rows = sorted(random.Random(args.sample_seed).sample(rows, args.sample), key=lambda r: r["id"])
        own = rows[self.rank::self.topo["nodes"]]
        self.work.mkdir(parents=True, exist_ok=True)
        if args.resume_from:
            previous = Path(args.resume_from) / "work" / f"node{self.rank}"
            for name in RESUMABLE + tuple(f"remask.gpu{g}.jsonl" for g in range(self.topo["gpus"])) + tuple(
                    f"remask_fix.gpu{g}.jsonl" for g in range(self.topo["gpus"])):
                if (previous / name).exists():
                    shutil.copyfile(previous / name, self.work / name)
        rows_path = self.work / "rows.jsonl"
        write_jsonl(rows_path, own)
        local = Path(args.local_root) / self.root.name / f"node{self.rank}"
        local.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        source_model = Path(args.model).resolve()
        if args.external_server:
            model, model_files = source_model, copy_model(source_model, None)
        else:
            size = sum(p.stat().st_size for p in source_model.iterdir() if p.is_file())
            if shutil.disk_usage(local).free > size + (20 << 30):
                model, model_files = local / "model", copy_model(source_model, local / "model")
            else:
                print(f"[node {self.rank}] not enough local disk for the model; serving it from {source_model}", flush=True)
                model, model_files = source_model, copy_model(source_model, None)
        # SAM3 runs as eight processes per node: read its checkpoint from local disk, not from the shared one.
        self.sam3_checkpoint = local / "sam3.pt"
        shutil.copyfile(args.sam3_checkpoint, self.sam3_checkpoint)
        print(f"[node {self.rank}] model and SAM3 checkpoint ready in {time.monotonic() - start:.0f} s", flush=True)
        identity = {
            "nodes": self.topo["nodes"], "gpus_per_node": self.topo["gpus"],
            "no_text_action": str(Path(args.no_text_action).resolve()),
            "no_text_action_sha256": file_hash(args.no_text_action),
            "kept": str(Path(args.kept).resolve()), "kept_sha256": file_hash(args.kept), "rows": len(rows),
            "sample": args.sample, "sample_seed": args.sample_seed, "root": str(Path(args.root).resolve()),
            "model": str(source_model), "model_files": model_files, "served_model_name": args.served_model_name,
            "server": ["external", args.external_server] if args.external_server else self.server_command("<model>")[2:],
            "sam3_checkpoint": str(Path(args.sam3_checkpoint).resolve()), "sam3_sha256": file_hash(self.sam3_checkpoint),
            "labeling_commit": labeling, "protocol_sha256": file_hash(Path(pa.ip.__file__)),
            "style_sha256": file_hash(Path(pa.style.__file__)), "adapter_version": pa.ADAPTER_VERSION,
            "font_sha256": file_hash(font), "votes": args.votes,
            "packages": {k: importlib.metadata.version(k)
                         for k in ("vllm", "torch", "transformers", "pillow", "pycocotools", "opencv-python-headless",
                                   "timm")},
            # Prompts, rules and steps: a resumed run must not mix records of two versions.
            "code": {name: file_hash(self.repo / name) for name in (
                "src/samtok_edit21/preparation/protocol_adapt.py", "src/samtok_edit21/preparation/edit_check.py",
                "scripts/data_cleaning/protocol_adapt.py", "scripts/data_cleaning/protocol_remask.py")},
        }
        common = {**identity, "run_id": self.root.name, "output": str(output), "git_commit": self.git_commit,
                  "runner_sha256": file_hash(self.repo / "src/samtok_edit21/distributed/protocol_adapt.py"),
                  "client": {"rows_in_flight": args.rows_in_flight, "workers": args.workers,
                             "build_workers": args.build_workers},
                  "resume_from": str(Path(args.resume_from).resolve()) if args.resume_from else None}
        atomic_json(self.node / "topology.json", common)
        self.barrier("topology")
        for rank in range(self.topo["nodes"]):
            if json.loads((self.root / "nodes" / str(rank) / "topology.json").read_text()) != common:
                raise ValueError(f"Input/code/model/environment mismatch on node {rank}")
        if self.rank == 0:
            atomic_json(self.root / "run_manifest.json", common)
        if args.resume_from:
            old = json.loads((Path(args.resume_from) / "run_manifest.json").read_text())
            changed = sorted(k for k in identity if k not in ("server",) and old.get(k) != identity[k])
            if changed:
                raise ValueError(f"Cannot resume {args.resume_from}: different {', '.join(changed)}")
        print(f"[node {self.rank}] {len(own)} rows assigned", flush=True)
        started = time.monotonic()
        try:
            if args.external_server:
                url = args.external_server.rstrip("/") + "/v1/chat/completions"
            else:
                self.start_server(model, local)
                url = f"http://127.0.0.1:{args.port}/v1/chat/completions"
            self.round(rows_path, url)
        finally:
            self.stop_server()
        final = self.work / "final.jsonl"
        if {r["id"] for r in pa.read_jsonl(final)} != {r["id"] for r in own}:
            raise RuntimeError("final.jsonl does not cover every assigned row")
        statuses = Counter(r["status"] for r in pa.read_jsonl(final))
        atomic_json(self.root / "work" / f"complete-node{self.rank}.json",
                    {"rows": len(own), "status": dict(statuses), "seconds": round(time.monotonic() - started, 1),
                     "sha256": file_hash(final)})
        print(f"[node {self.rank}] done: {dict(statuses)} in {(time.monotonic() - started) / 3600:.2f} h", flush=True)
        self.barrier("adapt")
        if self.rank == 0:
            completes = [json.loads((self.root / "work" / f"complete-node{r}.json").read_text())
                         for r in range(self.topo["nodes"])]
            timing = {"adapt_hours_max": round(max(c["seconds"] for c in completes) / 3600, 2), "nodes": completes}
            summary = self.merge(rows, output, common, timing)
            atomic_json(self.root / "SUCCESS.json", {"time": time.time(), "output": str(output),
                                                     "rows": summary["rows"], "status": summary["status"]})
            print(json.dumps({k: summary[k] for k in ("rows", "status", "accepted_by_operation")}), flush=True)
        self.barrier("merge")
        if args.remove_local_model and model != source_model:
            shutil.rmtree(model, ignore_errors=True)


def merge(rows, finals, output, common, timing):
    """The nodes' final records (`finals`: their final.jsonl files) into the output folder, in input order."""
    records_by_id = {}
    for path in finals:
        records_by_id.update({r["id"]: r for r in pa.read_jsonl(path)})
    missing = [r["id"] for r in rows if r["id"] not in records_by_id]
    if missing:
        raise RuntimeError(f"{len(missing)} rows have no final record, e.g. {missing[:3]}")
    summary = {**write_dataset(rows, [records_by_id[r["id"]] for r in rows], output),
               "timing": timing, "run_id": common["run_id"], "git_commit": common["git_commit"],
               "time_utc": datetime.datetime.utcnow().isoformat(timespec="seconds")}
    output = Path(output)
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "config.json", {**common,
                                          "anchor": {"method": pa.ANCHOR_METHOD, "params": pa.ANCHOR_PARAMS,
                                                     "part_params": pa.PART_PARAMS},
                                          "edited_region": {"method": pa.EDITED_METHOD, "params": pa.EDITED_PARAMS},
                                          "edit_gates": pa.EDIT_GATES,
                                          "prompts_sha256": {name: sha256_bytes(getattr(pa, name).encode())
                                                             for name in ("SLOT_PROMPT", "AUDIT_PROMPT",
                                                                          "PART_PICK_PROMPT", "GROUND_PROMPT",
                                                                          "PART_GROUND_PROMPT")}})
    (output / "README.md").write_text(readme(summary, common))
    return summary


def write_dataset(rows, records, output):
    """records.jsonl (every row), manifest_protocol.jsonl (accepted, with image paths) and multi_object.jsonl of
    `records`, which follow the order of `rows`; returns their counts."""
    if [r["id"] for r in records] != [r["id"] for r in rows]:
        raise ValueError("records are not in input order")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "records.jsonl", records)
    images = {r["id"]: {k: r[k] for k in ("source_image", "target_image")} for r in rows}
    accepted = [{**images[r["id"]], **{k: r.get(k) for k in MANIFEST_FIELDS}} for r in records
                if r["status"] == "accepted"]
    write_jsonl(output / "manifest_protocol.jsonl", accepted)
    write_jsonl(output / "multi_object.jsonl", [r for r in records if r["status"] == "multi_object"])
    status = Counter(r["status"] for r in records)
    by = Counter((r["type_before"], r["status"], (r.get("instruction_spec") or {}).get("operation", "-"))
                 for r in records)
    adds = [r for r in accepted if r["instruction_spec"]["operation"] == "add"]
    return {
        "rows": len(records), "status": dict(status),
        "accepted_by_operation": dict(Counter(r["instruction_spec"]["operation"] for r in accepted)),
        "by_type_status_operation": {f"{t}:{s}:{o}": n for (t, s, o), n in sorted(by.items())},
        "rejected_by_reason": dict(Counter(reason_key(r) for r in records if r["status"] == "rejected").most_common()),
        "add_granularity": dict(Counter(r["selection"].get("granularity") for r in adds)),
        "add_anchor_mode": dict(Counter(r["instruction_spec"]["placement"]["anchor_mode"] for r in adds)),
        "host_views": sum(1 for r in accepted if r.get("host_view")),
        "selection_sources": dict(Counter(r["selection"]["source"] for r in accepted)),
    }


def reason_key(record):
    """A rejection's first reason code; the slot model's own words (its reject decisions) count as slots_reject."""
    key = (record.get("reason") or "-").split(":")[0].split(",")[0]
    return key if re.fullmatch(r"[a-z0-9_]+|-", key) else "slots_reject"


def regate(output, rows, records, before, apply_note):
    """A finished output rewritten in place with re-gated records (`records`, in input order): the earlier files go to
    `before`, the counts are recomputed, and config.json, summary.json and README.md record the pass (apply_note)."""
    output, before = Path(output), Path(before)
    if [r["id"] for r in records] != [r["id"] for r in rows]:
        raise ValueError("records are not in input order")  # checked before any file moves
    old_summary = json.loads((output / "summary.json").read_text())
    old_config = json.loads((output / "config.json").read_text())
    before.mkdir(parents=True, exist_ok=False)
    for name in ("records.jsonl", "manifest_protocol.jsonl", "multi_object.jsonl", "summary.json", "config.json",
                 "README.md"):
        (output / name).rename(before / name)
    passes = old_summary.get("post_fix", []) + [apply_note]
    summary = {**write_dataset(rows, records, output),
               **{k: old_summary[k] for k in ("timing", "run_id", "git_commit")},
               "time_utc": old_summary["time_utc"], "post_fix": passes}
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "config.json", {**old_config, "post_fix": old_config.get("post_fix", []) + [apply_note]})
    (output / "README.md").write_text(readme(summary, old_config))
    return summary


def readme(summary, common):
    status = summary["status"]
    lines = [
        "# Protocol 1.0 adaptation of the cleaned pairs (Qwen3.8-27B, SAM3)",
        "",
        "Written by `samtok_edit21.distributed.protocol_adapt` at the end of the run; the rules are",
        "`docs/10_SAMTokEdit_Qwen21_训练数据协议适配.md` in the samtok_edit repository (branch qwen-image-2.1-v2,",
        f"commit {common['git_commit']}). Image paths are relative to the parent folder, as in its manifests.",
        "",
        "| File | Content |",
        "|---|---|",
        "| `manifest_protocol.jsonl` | accepted records for training: image paths, instruction spec, ref and no-ref "
        "sentences, mask semantics, S (`selection`), host view, execution and edited regions |",
        "| `records.jsonl` | every input row with its full record (slots, checks, audit), in input order |",
        "| `multi_object.jsonl` | rows left out as multi-object, for later recall |",
        "| `summary.json`, `config.json` | counts; model, prompts, parameters, hashes and code |",
        "| `runs/<run ID>/` | the run: per-node work files, logs, consistency checks |",
        "",
        f"Rows: {summary['rows']}; accepted {status.get('accepted', 0)}, multi-object {status.get('multi_object', 0)}, "
        f"rejected {status.get('rejected', 0)}.",
        "",
        "Accepted by operation: " + ", ".join(f"{k} {v}" for k, v in sorted(summary["accepted_by_operation"].items())) + ".",
        "",
        "Add anchors by granularity: " + ", ".join(f"{k} {v}" for k, v in sorted(summary["add_granularity"].items()))
        + f"; host views {summary['host_views']}.",
        "",
        "Rejected by reason: " + ", ".join(f"{k} {v}" for k, v in summary["rejected_by_reason"].items()) + ".",
        "",
    ]
    for note in summary.get("post_fix", []):
        lines += [f"Post-run fix `{note['version']}` ({note['time_utc']}, commit {note['code_commit']}): "
                  f"{note['description']} Accepted {note['accepted_before']} -> {note['accepted_after']}; "
                  "changes: " + ", ".join(f"{k} {v}" for k, v in note["changes"].items())
                  + f". The files it replaced are in `{note['previous_files']}`.", ""]
    return "\n".join(lines)


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-text-action", required=True, help="manifest_kept_no_text_action.jsonl of the edit check")
    p.add_argument("--kept", required=True, help="manifest_kept.jsonl of the edit check (for the text pairs)")
    p.add_argument("--root", required=True, help="Folder the manifests' image paths are relative to")
    p.add_argument("--output", required=True, help="Output folder, e.g. <cleaned dataset>/protocol_adapt_qwen38_27b_v1")
    p.add_argument("--run-root", required=True, help="This run's folder, e.g. <output>/runs/<run ID>")
    p.add_argument("--labeling-repo", required=True, help="Checkout of github.com/Tangent0308/sam3-crispedit")
    p.add_argument("--labeling-commit", help="Refuse a labeling checkout at another commit")
    p.add_argument("--sam3-checkpoint", default=SAM3_CHECKPOINT)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--served-model-name", default="qwen38-27b")
    p.add_argument("--local-root", default="/tmp/samtok-protocol-adapt")
    p.add_argument("--remove-local-model", action="store_true", help="Delete the node-local model copy at the end")
    p.add_argument("--port", type=int, default=8012)
    p.add_argument("--external-server", help="Use this running OpenAI-compatible server (rehearsals) instead of starting vLLM")
    p.add_argument("--max-model-len", type=int, default=20000)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85,
                   help="Leave room for SAM3 on every GPU; 0.85 was validated next to it")
    p.add_argument("--max-num-seqs", type=int, default=96)
    p.add_argument("--rows-in-flight", type=int, default=256, help="Rows asked concurrently per node (speed only)")
    p.add_argument("--workers", type=int, default=16, help="Processes drawing the model's views per node")
    p.add_argument("--build-workers", type=int, default=48, help="Processes building records (CPU) per node")
    p.add_argument("--votes", type=int, default=2, help="Independent audit calls; every one must pass")
    p.add_argument("--sample", type=int, help="Only this many random rows (rehearsals)")
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--resume-from", help="Earlier run folder with the same input, model, code and nodes")
    p.add_argument("--report-every", type=int, default=60)
    p.add_argument("--server-timeout", type=int, default=3600)
    p.add_argument("--timeout", type=int, default=259200, help="Barrier and step timeout in seconds")
    p.add_argument("--local", action="store_true", help="One node (this machine)")
    args = p.parse_args()
    pipeline = ProtocolPipeline(args)
    try:
        pipeline.run()
    except BaseException as exc:
        record_failure(pipeline.node / "failure.json", {"error": str(exc), "type": type(exc).__name__, "time": time.time()})
        raise
    finally:
        pipeline.stop_server()


if __name__ == "__main__":
    main()
