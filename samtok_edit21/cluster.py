"""Shared-filesystem ARNOLD pipeline: one invocation on each worker."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def topology(env, local=False):
    if local:
        return dict(nodes=1, node_rank=0, gpus=8, world_size=8,
                    master_addr="127.0.0.1", master_port=int(env.get("SAMTOK_LOCAL_PORT", "29541")))
    nodes = int(env.get("NNODES", env.get("ARNOLD_WORKER_NUM", "4")))
    rank = int(env.get("NODE_RANK", env.get("ARNOLD_ID", "-1")))
    gpus = int(env.get("GPUS_PER_NODE", env.get("ARNOLD_WORKER_GPU", "8")))
    host, port = None, None
    first = env.get("ARNOLD_WORKER_HOSTS", "").split(",")[0].strip()
    if first:
        if first.startswith("["):
            match = re.fullmatch(r"\[([^]]+)\]:(\d+)", first)
            if not match:
                raise ValueError("Expected [IPv6]:port in ARNOLD_WORKER_HOSTS")
            host, port = match.groups()
        else:
            host, separator, port = first.rpartition(":")
            if not separator or not host or ":" in host:
                raise ValueError("Expected host:port (bracket IPv6 addresses)")
    host = env.get("MASTER_ADDR") or host or env.get("ARNOLD_WORKER_0_HOST")
    port = env.get("MASTER_PORT") or port
    if not host or not port:
        raise ValueError("Supply ARNOLD_WORKER_HOSTS or common MASTER_ADDR/MASTER_PORT; PORT is intentionally ignored")
    if nodes != 4 or gpus != 8 or not 0 <= rank < nodes:
        raise ValueError(f"Expected 4 nodes x 8 GPUs; got {nodes=}, {gpus=}, {rank=}")
    if not 1024 <= int(port) <= 65535:
        raise ValueError("MASTER_PORT must be between 1024 and 65535")
    return dict(nodes=nodes, node_rank=rank, gpus=gpus, world_size=nodes*gpus,
                master_addr=host.strip("[]"), master_port=int(port))


def source_digest(root):
    digest = hashlib.sha256()
    for directory in ("samtok_edit21", "DiffSynth-Studio/diffsynth", "samtok", "scripts"):
        for path in sorted((root / directory).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".sh", ".yaml", ".json"}:
                digest.update(str(path.relative_to(root)).encode())
                digest.update(path.read_bytes())
    for name in ("requirements.txt", "constraints-tested.txt", "requirements-cluster.txt", "constraints-cluster.txt"):
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


class Pipeline:
    def __init__(self, args):
        self.args = args
        self.topo = topology(os.environ, args.local)
        self.rank = self.topo["node_rank"]
        self.root = Path(args.run_root).resolve()
        self.repo = Path(__file__).resolve().parents[1]
        self.node = self.root / "nodes" / str(self.rank)
        # Never consume stale success markers from a previous attempt.
        self.node.mkdir(parents=True, exist_ok=False)
        self.logdir = self.root / "logs" / f"node{self.rank}"
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.env = dict(os.environ, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
        self.env.setdefault("OMP_NUM_THREADS", "4")
        self.env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
        self.env.setdefault("WANDB_DISABLE_SERVICE", "true")
        self.env.setdefault("WANDB_START_METHOD", "thread")
        self.env["PYTHONPATH"] = f"{self.repo}:{self.repo / 'DiffSynth-Studio'}" + (
            ":" + self.env["PYTHONPATH"] if self.env.get("PYTHONPATH") else "")
        # Compilers must not contend on a network filesystem across machines.
        self.env.setdefault("TORCHINDUCTOR_CACHE_DIR", f"/tmp/samtok21-inductor-{os.getuid()}")

    def check_failures(self):
        failures = list((self.root / "nodes").glob("*/failure.json"))
        if failures:
            raise RuntimeError("Worker failure: " + ", ".join(str(p) for p in failures))

    def barrier(self, phase):
        atomic_json(self.node / f"{phase}.ok.json", {"time": time.time()})
        deadline = time.monotonic() + self.args.timeout
        while True:
            self.check_failures()
            if all((self.root / "nodes" / str(i) / f"{phase}.ok.json").is_file()
                   for i in range(self.topo["nodes"])):
                return
            if time.monotonic() > deadline:
                raise TimeoutError(f"Timed out waiting for all nodes: {phase}")
            time.sleep(2)

    def command(self, phase, command):
        atomic_json(self.node / f"{phase}.command.json", command)
        print(f"[node {self.rank}] {phase}: {json.dumps(command)}", flush=True)
        with (self.logdir / f"{phase}.log").open("w") as stream:
            process = subprocess.Popen(command, cwd=self.repo, env=self.env, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + self.args.timeout
            try:
                while process.poll() is None:
                    self.check_failures()
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"Phase timed out: {phase}")
                    time.sleep(2)
                if process.returncode:
                    raise RuntimeError(f"{phase} exited {process.returncode}; see {stream.name}")
            except BaseException:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                raise

    def distributed(self, phase, arguments):
        t = self.topo
        command = [sys.executable, "-m", "torch.distributed.run", "--nnodes", str(t["nodes"]),
                   "--nproc_per_node", str(t["gpus"]), "--node_rank", str(self.rank),
                   "--master_addr", t["master_addr"], "--master_port", str(t["master_port"]),
                   "--max_restarts", "0", "--log_dir", str(self.logdir / f"{phase}-ranks"),
                   "--tee", "3", *arguments]
        self.command(phase, command)
        self.barrier(phase)

    def run(self):
        a = self.args
        if a.wandb_mode == "online" and not os.environ.get("WANDB_API_KEY"):
            raise ValueError("Inject WANDB_API_KEY into every worker environment; do not put it in command logs")
        data = Path(a.data).resolve()
        for name in ("stage1.jsonl", "stage2.jsonl", "regions/manifest.json"):
            if not (data / name).is_file():
                raise FileNotFoundError(data / name)
        import importlib.metadata
        packages = {name: importlib.metadata.version(name) for name in
                    ("torch", "transformers", "accelerate", "peft", "byted-wandb", "setuptools")}
        common = {**{k:v for k,v in self.topo.items() if k != "node_rank"},
                  "args": vars(a), "source_sha256": source_digest(self.repo), "packages": packages,
                  "debug_scripts": {str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (a.inference_script, a.audit_script) if p},
                  "data": {n:hashlib.sha256((data/n).read_bytes()).hexdigest() for n in
                           ("stage1.jsonl", "stage2.jsonl", "regions/manifest.json")}}
        atomic_json(self.node / "topology.json", {"common": common, "hostname": socket.gethostname(),
                    "node_rank": self.rank, "python": sys.executable,
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "wandb_key_present": bool(os.environ.get("WANDB_API_KEY"))})
        self.barrier("topology")
        for i in range(self.topo["nodes"]):
            peer = json.loads((self.root/"nodes"/str(i)/"topology.json").read_text())
            if peer["common"] != common:
                raise ValueError(f"Topology/source/data/arguments/environment mismatch on node {i}")
        if self.rank == 0:
            atomic_json(self.root / "manifest.json", common)
        self.barrier("preflight")
        self.distributed("collectives", ["-m", "samtok_edit21.distributed_probe", "--output", str(self.root/"collectives")])
        shared = ["--max-pixels", str(a.max_pixels), "--qwen", a.qwen, "--samtok", a.samtok,
                  "--num-workers", "0", "--seed", str(a.seed)]
        def tracking(stage):
            name = self.root.name + "-" + stage
            return ["--wandb-mode", a.wandb_mode, "--wandb-project", a.wandb_project,
                    "--wandb-entity", a.wandb_entity, "--wandb-name", name,
                    "--wandb-id", hashlib.sha256(str(self.root).encode()).hexdigest()[:16] + "-" + stage]
        self.distributed("stage1", ["-m", "samtok_edit21.train", "train", "--stage", "stage1",
            "--metadata", str(data/"stage1.jsonl"), "--region-cache", str(data/"regions"),
            "--region-weight", str(a.region_weight), "--steps", str(a.stage1_steps),
            "--save-steps", "8", "--accumulation", "8", "--rank", str(a.stage1_rank),
            "--output", str(self.root/"stage1"), *shared, *tracking("stage1")])
        self.distributed("cache", ["-m", "samtok_edit21.train", "cache",
            "--metadata", str(data/"stage2.jsonl"), "--region-cache", str(data/"regions"),
            "--te-adapter", str(self.root/"stage1/adapter"), "--output", str(self.root/"cache"), *shared])
        self.distributed("stage2", ["-m", "samtok_edit21.train", "train", "--stage", "stage2",
            "--cache", str(self.root/"cache"), "--region-weight", str(a.region_weight),
            "--attention-weight", str(a.attention_weight), "--attention-warmup-steps", "1",
            "--steps", str(a.stage2_steps), "--save-steps", "4", "--accumulation", "4",
            "--rank", str(a.stage2_rank), "--output", str(self.root/"stage2"), *shared, *tracking("stage2")])
        if a.inference_script and self.rank == 0:
            self.command("inference", [sys.executable, "-m", "torch.distributed.run", "--standalone",
                         "--nproc_per_node", "8", "--max_restarts", "0", a.inference_script,
                         "--run-root", str(self.root), "--data", str(data), "--qwen", a.qwen, "--samtok", a.samtok])
        self.barrier("inference")
        if a.audit_script and self.rank == 0:
            self.command("audit", [sys.executable, a.audit_script, "--run-root", str(self.root)])
        self.barrier("audit")
        if self.rank == 0:
            atomic_json(self.root / "SUCCESS.json", {"time": time.time(), "world_size": self.topo["world_size"]})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", required=True, help="Fresh shared directory; identical on all nodes")
    p.add_argument("--data", required=True, help="Prepared stage1/stage2 JSONL and regions directory")
    p.add_argument("--local", action="store_true", help="One node x eight GPUs validation")
    p.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    p.add_argument("--wandb-project", default="samtok-edit")
    p.add_argument("--wandb-entity", default="2200012743-peking-university")
    p.add_argument("--qwen", default="/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen-Image-2.1")
    p.add_argument("--samtok", default="/mnt/bn/strategy-mllm-train/user/tanyue/models/SAMTok/Qwen3-VL-8B-SAMTok")
    p.add_argument("--stage1-steps", type=int, default=2)
    p.add_argument("--stage2-steps", type=int, default=3)
    p.add_argument("--stage1-rank", type=int, default=64)
    p.add_argument("--stage2-rank", type=int, default=32)
    p.add_argument("--max-pixels", type=int, default=65536)
    p.add_argument("--region-weight", type=float, default=0.5)
    p.add_argument("--attention-weight", type=float, default=0.1, help="Smoke coefficient only; calibrate for real training")
    p.add_argument("--seed", type=int, default=20260928)
    p.add_argument("--timeout", type=int, default=7200, help="Per-phase and barrier timeout in seconds")
    p.add_argument("--inference-script")
    p.add_argument("--audit-script")
    args = p.parse_args()
    pipeline = Pipeline(args)
    try:
        pipeline.run()
    except BaseException as exc:
        atomic_json(pipeline.node/"failure.json", {"error": str(exc), "type": type(exc).__name__, "time": time.time()})
        raise


if __name__ == "__main__":
    main()
