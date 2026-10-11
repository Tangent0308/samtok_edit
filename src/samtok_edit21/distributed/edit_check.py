"""ARNOLD MLLM edit check: 4 nodes x 8 GPUs, one data-parallel vLLM server per node.

Every node copies Qwen3.8-27B to local disk, serves it data-parallel over its
eight GPUs and checks every 4th pair of the cleaned manifest (scheme 2 of
docs/09 section 6.11, see samtok_edit21.preparation.edit_check).  Results are
appended per pair, so a failed run resumes with --resume-from.  Node 0 merges
the shards into the output folder next to the dataset.  No DDP and no W&B.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

from samtok_edit21.data.io import file_hash
from samtok_edit21.distributed.training import Pipeline, atomic_json, record_failure
from samtok_edit21.preparation import edit_check as ec

MODEL = "/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen3.8-27B"
ROW_FIELDS = ("id", "dataset", "type", "instruction", "source_image", "target_image", "region")
# Not passed to the server: torch launcher variables would make the vLLM engines join a foreign
# process group; it needs none of the repository's import paths; and without OMP_NUM_THREADS vLLM
# runs one torch thread per engine, as in the pilot.
SERVER_UNSET = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK", "MASTER_ADDR",
                "MASTER_PORT", "NODE_RANK", "NNODES", "GPUS_PER_NODE", "PYTHONPATH", "OMP_NUM_THREADS")


# Server flags that only change speed; a resumed run may use other values.
SPEED_FLAGS = ("--gpu-memory-utilization", "--max-num-seqs", "--port")


def answer_flags(command):
    """Server arguments without SPEED_FLAGS and their values."""
    kept, skip = [], False
    for item in command:
        if skip:
            skip = False
        elif item in SPEED_FLAGS:
            skip = True
        else:
            kept.append(item)
    return kept


def read_rows(manifest, sample=None, seed=0):
    """Manifest rows (only the fields the check reads), optionally a fixed random subset in manifest order."""
    with Path(manifest).open() as stream:
        rows = [{k: value[k] for k in ROW_FIELDS} for value in map(json.loads, stream)]
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate IDs in the manifest")
    if sample:
        keep = set(random.Random(seed).sample(range(len(rows)), min(sample, len(rows))))
        rows = [r for i, r in enumerate(rows) if i in keep]
    return rows


def copy_model(source, target, threads=8):
    """Copy the model's top-level files to `target` (or only read them when None); returns {file: sha256}."""
    source = Path(source)
    files = sorted(p for p in source.iterdir() if p.is_file())
    if target is not None:
        target.mkdir(parents=True)

    def one(path):
        digest = hashlib.sha256()
        destination = target / path.name if target is not None else None
        with path.open("rb") as reader, (destination.open("wb") if destination else nullcontext()) as writer:
            while chunk := reader.read(16 << 20):
                digest.update(chunk)
                if writer:
                    writer.write(chunk)
        if destination is not None and destination.stat().st_size != path.stat().st_size:
            raise OSError(f"Incomplete model copy: {destination}")
        return path.name, digest.hexdigest()

    with ThreadPoolExecutor(threads) as pool:
        return dict(pool.map(one, files))


def read_shard(path):
    """Strict reader for merging: every line must be a complete record and every ID unique."""
    records = {}
    with Path(path).open() as stream:
        for number, line in enumerate(stream, 1):
            value = json.loads(line)
            if value["id"] in records:
                raise ValueError(f"Duplicate record {value['id']} in {path} line {number}")
            records[value["id"]] = value
    return records


def share(part, whole):
    return round(part / whole, 4) if whole else None


def tally(records, key):
    table = defaultdict(Counter)
    for record in records:
        group = table[key(record)]
        group["pairs"] += 1
        group["kept"] += record["keep"]
        group["q2_fail"] += record["q2"]["verdict"] == "fail"
        group["dn2_fail"] += record["dn2"]["verdict"] == "fail"
    return {name: {**counts, "kept_share": share(counts["kept"], counts["pairs"])}
            for name, counts in sorted(table.items(), key=lambda item: str(item[0]))}


def summarize(records, run_manifest, timing):
    unchecked = [r for r in records if r["q2"]["verdict"] is None or r["dn2"]["verdict"] is None]
    q2_fail = {r["id"] for r in records if r["q2"]["verdict"] == "fail"}
    dn2_fail = {r["id"] for r in records if r["dn2"]["verdict"] == "fail"}
    kept = sum(r["keep"] for r in records)
    return {
        "input_count": len(records), "kept_count": kept, "dropped_count": len(records) - kept,
        "kept_share": share(kept, len(records)),
        "dropped_by": {"q2_only": len(q2_fail - dn2_fail), "dn2_only": len(dn2_fail - q2_fail),
                       "both": len(q2_fail & dn2_fail),
                       "unchecked": sum(r["id"] not in q2_fail | dn2_fail for r in unchecked)},
        "unchecked_count": len(unchecked),
        "retried": {"q2": sum(r["q2"]["attempts"] > 1 for r in records),
                    "describe": sum(r["dn2"].get("describe_attempts", 1) > 1 for r in records),
                    "judge2": sum(r["dn2"]["attempts"] > 1 for r in records)},
        "problems": {"q2": dict(Counter(r["q2"].get("problem") for r in records if r["id"] in q2_fail).most_common()),
                     "dn2": dict(Counter(r["dn2"].get("problem") for r in records if r["id"] in dn2_fail).most_common())},
        "by_dataset": tally(records, lambda r: r["dataset"]),
        "by_type": tally(records, lambda r: str(r["type"])),
        "by_dataset_type": tally(records, lambda r: f"{r['dataset']}/{r['type']}"),
        "output_tokens": {"q2": sum(r["q2"]["tokens"] for r in records),
                          "dn2": sum(r["dn2"]["tokens"] for r in records)},
        "timing": timing,
        "run_id": run_manifest["run_id"], "git_commit": run_manifest["git_commit"],
        "protocol_sha256": run_manifest["protocol_sha256"], "manifest_sha256": run_manifest["manifest_sha256"],
    }


def readme(summary, run_manifest):
    """Chinese README of the output folder; numbers come from summary.json."""
    s = summary
    lines = [
        f"# SAMTokEdit 清洗后编辑对的 MLLM 编辑核验（{run_manifest['run_id']}）",
        "",
        f"对上一级目录 `manifest.jsonl` 的 {s['input_count']:,} 个编辑对，用 Qwen3.8-27B 逐对核验编辑是否正确完成"
        f"（方案 2：问题清单核验 + 盲描述核验，任一不通过即丢弃）。保留 **{s['kept_count']:,} 对**"
        f"（{s['kept_share']:.1%}），丢弃 {s['dropped_count']:,} 对。",
        "",
        "- 方法与小样本验证：仓库 `samtok_edit`（分支 `qwen-image-2.1-v2`）`docs/09` 第 6.11 节。"
        "在 325 对人工复核样本上，方案 2 抓到 68/79 个坏样本（86%），误判 15/246 个好样本（6.1%）；"
        "清洗后约 24% 的坏样本比例预计降到约 4.5%。",
        "- 代码：同一仓库 `src/samtok_edit21/preparation/edit_check.py`（视图、提示词、采样）和 "
        "`src/samtok_edit21/distributed/edit_check.py`（四机运行与合并），"
        f"提交 `{run_manifest['git_commit']}`；提示词与采样设置的完整文本见 `config.json`。",
        "",
        "## 方法",
        "",
        "每对图生成三张视图：原图（红色轮廓标出编辑区域）、编辑后的图、编辑区域的放大对照（左原图右结果）。同一组视图上做两项独立核验：",
        "",
        "1. **问题清单核验（q2）**：按编辑类型提几个具体问题（对象是否完全删除、新物体是否是要求的种类、颜色是否真的变了、"
        "有没有改到别的物体、有没有残影或悬空等），开启思考模式（低强度），给出通过/不通过。",
        "2. **盲描述核验（dn2）**：先不给指令，让模型只描述看到的改动（不开思考模式）；再把指令和这段描述一起交给模型逐项比对"
        "（开启思考模式），给出通过/不通过。",
        "",
        "两项都通过才保留。某项连续 3 次没有给出可解析的结论时记为未核验，也丢弃（见 `summary.json` 的 `unchecked_count`）。",
        "",
        "## 文件",
        "",
        "| 文件 | 内容 |",
        "|---|---|",
        "| `manifest_kept.jsonl` | 通过核验的编辑对，每行与上一级 `manifest.jsonl` 中对应行**逐字节相同**，顺序不变；图片路径相对**上一级目录** |",
        "| `verdicts.jsonl` | 全部编辑对的核验结果，顺序与 `manifest.jsonl` 相同（字段见下） |",
        "| `summary.json` | 保留/丢弃计数、按来源和类型的统计、丢弃原因、用时 |",
        "| `config.json` | 模型、服务参数、提示词全文、视图与采样设置、各文件 hash、代码提交 |",
        f"| `runs/<run ID>/` | 运行过程：`bootstrap/`（启动日志）、`logs/`（各节点与 vLLM 日志）、`shards/`（各节点逐对结果）、`nodes/` |",
        "",
        "`verdicts.jsonl` 字段：",
        "",
        "| 字段 | 含义 |",
        "|---|---|",
        "| `id`、`dataset`、`type`、`instruction` | 与 `manifest.jsonl` 相同 |",
        "| `keep` | 两项核验都通过为 `true` |",
        "| `q2` | `verdict`（`pass` / `fail` / `null` 未核验）、`problem`、`reason`、`answers`（对各问题的简答）、`attempts`、`tokens` |",
        "| `dn2` | `verdict`、`problem`、`reason`、`description`（盲描述）、`describe_attempts`、`attempts`、`tokens` |",
        "| `seconds` | 这一对的核验用时 |",
        "",
        "`problem` 取值：`not_done`（没做）、`too_weak`（改动太弱）、`partial`（只做了一部分）、`wrong_target`（改错对象）、"
        "`scope`（改到了别的物体）、`defect`（明显瑕疵）。",
        "",
        "## 统计",
        "",
        "| 丢弃原因 | 数量 |",
        "|---|---:|",
        f"| 只有问题清单核验不通过 | {s['dropped_by']['q2_only']:,} |",
        f"| 只有盲描述核验不通过 | {s['dropped_by']['dn2_only']:,} |",
        f"| 两项都不通过 | {s['dropped_by']['both']:,} |",
        f"| 未核验（无法得到结论） | {s['dropped_by']['unchecked']:,} |",
        "",
        "| 来源 | 类型 | 总数 | 保留 | 保留比例 | q2 不通过 | dn2 不通过 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, c in s["by_dataset_type"].items():
        dataset, edit_type = name.split("/", 1)
        lines.append(f"| {dataset} | {edit_type} | {c['pairs']:,} | {c['kept']:,} | {c['kept_share']:.0%} | "
                     f"{c['q2_fail']:,} | {c['dn2_fail']:,} |")
    lines += ["", "| 来源 | 总数 | 保留 | 保留比例 |", "|---|---:|---:|---:|"]
    for name, c in s["by_dataset"].items():
        lines.append(f"| {name} | {c['pairs']:,} | {c['kept']:,} | {c['kept_share']:.0%} |")
    timing = s["timing"]
    lines += ["", "## 运行", "",
              f"- run ID：`{run_manifest['run_id']}`；{run_manifest['nodes']} 机 × {run_manifest['gpus_per_node']} 卡，"
              f"每机一个 vLLM 服务（数据并行 {run_manifest['gpus_per_node']}）。",
              f"- 核验阶段用时（各节点最长）：{timing['check_hours_max']:.2f} 小时；平均每机 {timing['pairs_per_second_mean']:.2f} 对/秒。",
              f"- 模型：`{run_manifest['model']}`；输入 `manifest.jsonl` sha256 `{run_manifest['manifest_sha256'][:16]}…`。",
              ""]
    return "\n".join(lines)


def merge(manifest, shards, output, run_manifest, timing):
    """Join the node shards in manifest order and publish the output folder (summary.json last)."""
    nodes, output = run_manifest["nodes"], Path(output)
    rows = read_rows(manifest, run_manifest["sample"], run_manifest["sample_seed"])
    records = {}
    for rank in range(nodes):
        path = shards / f"node{rank}.jsonl"
        complete = json.loads((shards / f"complete-node{rank}.json").read_text())
        if complete["sha256"] != file_hash(path):
            raise ValueError(f"Shard {rank} changed after completion")
        shard = read_shard(path)
        if set(shard) != {r["id"] for r in rows[rank::nodes]}:
            raise ValueError(f"Shard {rank} does not cover exactly its assigned pairs")
        records.update(shard)
    ordered = []
    for row in rows:
        record = records[row["id"]]
        if (record["dataset"], record["type"]) != (row["dataset"], row["type"]):
            raise ValueError(f"Record {row['id']} does not match the manifest")
        ordered.append({"id": row["id"], "dataset": row["dataset"], "type": row["type"],
                        "instruction": row["instruction"], **{k: v for k, v in record.items()
                                                               if k not in ("id", "dataset", "type")}})
    output.mkdir(parents=True, exist_ok=True)
    kept = {r["id"] for r in ordered if r["keep"]}

    def publish(name, write):
        temporary = output / f".{name}.tmp"
        with temporary.open("w") as stream:
            write(stream)
        temporary.replace(output / name)

    publish("verdicts.jsonl", lambda f: f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in ordered))
    selected = {r["id"] for r in rows}

    def kept_lines(f):
        with Path(manifest).open() as stream:
            for line in stream:
                identifier = json.loads(line)["id"]
                if identifier in kept and identifier in selected:
                    f.write(line)

    publish("manifest_kept.jsonl", kept_lines)
    summary = summarize(ordered, run_manifest, timing)
    summary["files"] = {name: file_hash(output / name) for name in ("verdicts.jsonl", "manifest_kept.jsonl")}
    atomic_json(output / "config.json", {"protocol": ec.protocol(), "run": run_manifest})
    publish("README.md", lambda f: f.write(readme(summary, run_manifest)))
    atomic_json(output / "summary.json", summary)
    return summary


def port_free(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


class EditCheckPipeline(Pipeline):
    def __init__(self, args):
        super().__init__(args)
        self.shards = self.root / "shards"
        self.server = None
        self.next_total = 0.0

    def server_command(self, model):
        a = self.args
        return [str(Path(sys.executable).parent / "vllm"), "serve", str(model),
                "--served-model-name", a.served_model_name,
                "--data-parallel-size", str(self.topo["gpus"]),
                "--max-model-len", str(a.max_model_len),
                "--gpu-memory-utilization", str(a.gpu_memory_utilization),
                "--max-num-seqs", str(a.max_num_seqs),
                "--enable-prefix-caching", "--reasoning-parser", "qwen3",
                "--limit-mm-per-prompt", json.dumps({"image": 6}),
                "--host", "127.0.0.1", "--port", str(a.port)]

    def start_server(self, model, local):
        if not port_free(self.args.port):
            raise RuntimeError(f"Port {self.args.port} is in use; pass another --port")
        env = {k: v for k, v in self.env.items() if k not in SERVER_UNSET}
        env.update(VLLM_CACHE_ROOT=str(local / "vllm-cache"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   VLLM_WORKER_MULTIPROC_METHOD="spawn", VLLM_NO_USAGE_STATS="1", DO_NOT_TRACK="1",
                   PATH=str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", ""))
        command = self.server_command(model)
        atomic_json(self.node / "server.command.json", command)
        self.server_log = self.logdir / "vllm.log"
        stream = self.server_log.open("w")
        self.server = subprocess.Popen(command, env=env, cwd=local, stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        stream.close()
        print(f"[node {self.rank}] vLLM server starting (log: {self.server_log})", flush=True)
        start, deadline = time.monotonic(), time.monotonic() + self.args.server_timeout
        while True:
            if self.server.poll() is not None:
                tail = self.server_log.read_text(errors="replace")[-3000:]
                raise RuntimeError(f"vLLM server exited with {self.server.returncode}:\n{tail}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.args.port}/v1/models", timeout=10) as response:
                    served = [m["id"] for m in json.load(response).get("data", [])]
                if self.args.served_model_name in served:
                    break
            except (urllib.error.URLError, OSError, ValueError):
                pass
            if time.monotonic() > deadline:
                raise TimeoutError(f"vLLM server not ready after {self.args.server_timeout} s; see {self.server_log}")
            time.sleep(10)
        print(f"[node {self.rank}] vLLM server ready after {time.monotonic() - start:.0f} s", flush=True)

    def stop_server(self):
        if self.server is None or self.server.poll() is not None:
            return
        try:
            os.killpg(self.server.pid, signal.SIGTERM)
            self.server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(self.server.pid, signal.SIGKILL)
            self.server.wait()
        except ProcessLookupError:
            pass

    def report(self, status):
        atomic_json(self.shards / f"progress-node{self.rank}.json", {**status, "node": self.rank, "time": time.time()})
        eta = f", ETA {status['eta_hours']:.2f} h" if status.get("eta_hours") is not None else ""
        print(f"[node {self.rank}] {status['done']}/{status['total']} pairs, kept {status['kept']}, "
              f"unchecked {status['unchecked']}, {status['pairs_per_second']:.2f} pairs/s{eta}", flush=True)
        if self.rank == 0 and self.topo["nodes"] > 1 and time.monotonic() >= self.next_total:
            progress = []
            for path in self.shards.glob("progress-node*.json"):
                try:
                    progress.append(json.loads(path.read_text()))
                except (OSError, ValueError):
                    pass
            done, total = sum(p["done"] for p in progress), len(self.rows)
            rate = sum(p["pairs_per_second"] for p in progress)
            print(f"[all nodes] {done}/{total} pairs ({len(progress)} nodes reporting), kept {sum(p['kept'] for p in progress)}, "
                  f"{rate:.2f} pairs/s" + (f", ETA {(total - done) / rate / 3600:.2f} h" if rate else ""), flush=True)
            self.next_total = time.monotonic() + 120

    def run(self):
        args = self.args
        manifest = Path(args.manifest).resolve()
        images_root = Path(args.images_root or manifest.parent).resolve()
        output = Path(args.output).resolve()
        if (output / "summary.json").exists():
            raise RuntimeError(f"{output} already holds a finished check; choose a new output folder")
        if not Path(ec.FONT).is_file():
            raise RuntimeError(f"Missing label font {ec.FONT} (Debian package fonts-dejavu-core)")
        self.rows = read_rows(manifest, args.sample, args.sample_seed)
        own = self.rows[self.rank::self.topo["nodes"]]
        local = Path(args.local_root) / self.root.name / f"node{self.rank}"
        local.mkdir(parents=True, exist_ok=False)
        source_model = Path(args.model).resolve()
        size = sum(p.stat().st_size for p in source_model.iterdir() if p.is_file())
        start = time.monotonic()
        if not args.external_server and shutil.disk_usage(local).free > size + (20 << 30):
            model = local / "model"
            model_files = copy_model(source_model, model)
        elif args.external_server:
            model, model_files = source_model, copy_model(source_model, None)
        else:
            print(f"[node {self.rank}] not enough local disk for the model; serving it from {source_model}", flush=True)
            model, model_files = source_model, copy_model(source_model, None)
        print(f"[node {self.rank}] model {'copied and ' if model != source_model else ''}hashed in "
              f"{time.monotonic() - start:.0f} s", flush=True)
        identity = {
            "nodes": self.topo["nodes"], "gpus_per_node": self.topo["gpus"],
            "manifest": str(manifest), "manifest_sha256": file_hash(manifest), "rows": len(self.rows),
            "sample": args.sample, "sample_seed": args.sample_seed, "images_root": str(images_root),
            "model": str(source_model), "model_files": model_files, "served_model_name": args.served_model_name,
            "server": (["external", args.external_server] if args.external_server
                       else answer_flags(self.server_command("<model>")[2:])),
            "protocol_sha256": ec.protocol()["sha256"],
            "font_sha256": file_hash(ec.FONT),
            "packages": {k: importlib.metadata.version(k) for k in ("vllm", "torch", "transformers", "pillow", "pycocotools")},
        }
        common = {**identity, "run_id": self.root.name, "output": str(output), "git_commit": self.git_commit,
                  "server_command": self.server_command("<model>")[2:],
                  "code": {name: file_hash(self.repo / "src/samtok_edit21" / name)
                           for name in ("preparation/edit_check.py", "distributed/edit_check.py")},
                  "client": {"pairs_in_flight": args.pairs_in_flight, "workers": args.workers},
                  "resume_from": str(Path(args.resume_from).resolve()) if args.resume_from else None}
        atomic_json(self.node / "topology.json", common)
        self.barrier("topology")
        for rank in range(self.topo["nodes"]):
            if json.loads((self.root / "nodes" / str(rank) / "topology.json").read_text()) != common:
                raise ValueError(f"Input/code/model/environment mismatch on node {rank}")
        if self.rank == 0:
            atomic_json(self.root / "run_manifest.json", common)
        self.shards.mkdir(exist_ok=True)
        shard = self.shards / f"node{self.rank}.jsonl"
        if args.resume_from:
            previous = Path(args.resume_from)
            old = json.loads((previous / "run_manifest.json").read_text())
            changed = sorted(k for k in identity if old.get(k) != identity[k])
            if changed:
                raise ValueError(f"Cannot resume {previous}: different {', '.join(changed)}")
            if (previous / "shards" / shard.name).exists():
                temporary = shard.with_suffix(".resume.tmp")
                shutil.copyfile(previous / "shards" / shard.name, temporary)
                temporary.replace(shard)
        done = ec.completed_records(shard)
        remaining = sum(r["id"] not in done for r in own)
        print(f"[node {self.rank}] {len(own)} pairs assigned, {len(own) - remaining} already checked", flush=True)
        summary = {"done": len(own), "total": len(own), "checked_this_run": 0, "seconds": 0.0, "pairs_per_second": 0.0}
        if remaining:
            base = args.external_server.rstrip("/") if args.external_server else f"http://127.0.0.1:{args.port}"
            try:
                if not args.external_server:
                    self.start_server(model, local)
                summary = asyncio.run(ec.check_rows(
                    own, images_root, shard, f"{base}/v1/chat/completions",
                    args.served_model_name, pairs_in_flight=args.pairs_in_flight, workers=args.workers,
                    report=self.report, report_every=args.report_every))
                summary["seconds"] = round(summary.pop("elapsed_hours") * 3600, 1)
            finally:
                self.stop_server()
        if set(ec.completed_records(shard)) != {r["id"] for r in own}:
            raise RuntimeError("Shard does not cover every assigned pair")
        atomic_json(self.shards / f"complete-node{self.rank}.json", {**summary, "sha256": file_hash(shard)})
        self.barrier("check")
        if self.rank == 0:
            completes = [json.loads((self.shards / f"complete-node{r}.json").read_text()) for r in range(self.topo["nodes"])]
            timing = {"check_hours_max": max(c["seconds"] for c in completes) / 3600,
                      "pairs_per_second_mean": sum(c["pairs_per_second"] for c in completes) / len(completes),
                      "nodes": completes}
            result = merge(manifest, self.shards, output, common, timing)
            atomic_json(self.root / "SUCCESS.json", {
                "time": time.time(), "output": str(output),
                **{k: result[k] for k in ("input_count", "kept_count", "dropped_count", "kept_share", "unchecked_count")}})
            print(json.dumps({k: result[k] for k in ("input_count", "kept_count", "dropped_count", "dropped_by")}), flush=True)
        self.barrier("merge")
        if args.remove_local_model and model != source_model:
            shutil.rmtree(model, ignore_errors=True)


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, help="Cleaned manifest.jsonl")
    p.add_argument("--images-root", help="Folder the manifest's image paths are relative to (default: its folder)")
    p.add_argument("--output", required=True, help="Output folder of the check, e.g. <dataset>/edit_check_qwen38_27b_v1")
    p.add_argument("--run-root", required=True, help="This run's folder, e.g. <output>/runs/<run ID>")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--served-model-name", default="qwen38-27b")
    p.add_argument("--local-root", default="/tmp/samtok-edit-check")
    p.add_argument("--remove-local-model", action="store_true", help="Delete the node-local model copy at the end")
    p.add_argument("--port", type=int, default=8012)
    p.add_argument("--external-server", help="Use this running OpenAI-compatible server (rehearsals) instead of starting vLLM")
    p.add_argument("--max-model-len", type=int, default=20000)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-num-seqs", type=int, default=96)
    p.add_argument("--pairs-in-flight", type=int, default=256, help="Pairs checked concurrently per node")
    p.add_argument("--workers", type=int, default=16, help="Processes drawing the views per node")
    p.add_argument("--sample", type=int, help="Check only this many random pairs (smoke tests)")
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--resume-from", help="Earlier run folder with the same input, model, prompts and nodes")
    p.add_argument("--report-every", type=int, default=60)
    p.add_argument("--server-timeout", type=int, default=3600)
    p.add_argument("--timeout", type=int, default=259200, help="Barrier timeout in seconds")
    p.add_argument("--local", action="store_true", help="One node (this machine)")
    args = p.parse_args()
    pipeline = EditCheckPipeline(args)
    try:
        pipeline.run()
    except BaseException as exc:
        record_failure(pipeline.node / "failure.json", {"error": str(exc), "type": type(exc).__name__, "time": time.time()})
        raise
    finally:
        pipeline.stop_server()


if __name__ == "__main__":
    main()
