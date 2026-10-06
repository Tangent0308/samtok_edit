"""Shared-filesystem ARNOLD pipeline: one invocation on each worker.

v2 phases are independent: Stage 1 (localization LoRA), the raw-TE
conditioning cache and Stage 2 (DiT LoRA + optional binding). A run executes
any subset; Stage 2 ablation arms reuse one complete cache via --cache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from samtok_edit21.paths import repository_root
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


def record_failure(path, value):
    """Publish complete JSON without replacing an earlier, more useful failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.failure.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    try:
        try:
            os.link(tmp, path)
        except FileExistsError:
            pass
    finally:
        tmp.unlink()


def link_alias(target, source):
    """Atomically expose a reused artifact under the new run root."""
    target, source = Path(target), Path(source).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists():
        if target.is_symlink() and target.resolve() == source:
            return
        raise RuntimeError(f"Refusing to replace existing run artifact: {target}")
    try:
        # Symlink creation itself is atomic.  Avoid a rename-overwrite race
        # because all ranks publish the same alias concurrently.
        os.symlink(source, target, target_is_directory=True)
    except FileExistsError:
        # Another rank may have published the same alias between our
        # existence check and symlink call.  Accept that exact result;
        # reject any conflicting artifact.
        if not (target.is_symlink() and target.resolve() == source):
            raise


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
    for directory in ("src/samtok_edit21", "third_party/diffsynth/diffsynth", "third_party/samtok", "scripts"):
        for path in sorted((root / directory).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".sh", ".yaml", ".json"}:
                digest.update(str(path.relative_to(root)).encode())
                digest.update(path.read_bytes())
    for name in ("requirements.txt", "constraints-tested.txt", "requirements-cluster.txt", "constraints-cluster.txt", "pyproject.toml", "upstream_versions.json"):
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


class Pipeline:
    def __init__(self, args):
        self.args = args
        self.topo = topology(os.environ, args.local)
        self.rank = self.topo["node_rank"]
        self.root = Path(args.run_root).resolve()
        self.repo = repository_root()
        self.git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=self.repo, text=True
        ).strip()
        self.inference_script = self.repo / "scripts/diagnostics/smoke_inference8.py"
        self.audit_script = self.repo / "scripts/diagnostics/audit_run.py"
        self.node = self.root / "nodes" / str(self.rank)
        # Never consume stale success markers from a previous attempt.
        try:
            self.node.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise RuntimeError(f"Run directory already used: {self.node}. "
                               "Set the same NEW SAMTOK_RUN_ID on every worker; "
                               "old failure/success markers must not be reused.") from exc
        self.logdir = self.root / "logs" / f"node{self.rank}"
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.env = dict(os.environ, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
        self.env.setdefault("OMP_NUM_THREADS", "4")
        self.env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
        self.env.setdefault("WANDB_DISABLE_SERVICE", "true")
        self.env.setdefault("WANDB_START_METHOD", "thread")
        self.env["PYTHONPATH"] = f"{self.repo / 'src'}:{self.repo / 'third_party/diffsynth'}" + (
            ":" + self.env["PYTHONPATH"] if self.env.get("PYTHONPATH") else "")
        # Compilers must not contend on a network filesystem across machines.
        self.env.setdefault("TORCHINDUCTOR_CACHE_DIR", f"/tmp/samtok21-inductor-{os.getuid()}")

    def check_failures(self):
        failures = list((self.root / "nodes").glob("*/failure.json"))
        if failures:
            details = []
            for path in sorted(failures):
                try:
                    error = json.loads(path.read_text()).get("error", "unknown failure")
                except (OSError, ValueError):
                    error = "failure details unavailable"
                details.append(f"{path}: {str(error)[:1000]}")
            raise RuntimeError("Worker failure: " + "; ".join(details))

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
            started = last_progress = time.monotonic()
            deadline = started + self.args.timeout
            try:
                while process.poll() is None:
                    self.check_failures()
                    now = time.monotonic()
                    if now - last_progress >= 60:
                        stat = Path(stream.name).stat()
                        progress = {"phase": phase, "elapsed_seconds": int(now - started),
                                    "log": stream.name, "log_bytes": stat.st_size,
                                    "log_age_seconds": round(time.time() - stat.st_mtime, 1)}
                        atomic_json(self.node / f"{phase}.progress.json", progress)
                        print(f"[node {self.rank}] running: {json.dumps(progress)}", flush=True)
                        last_progress = now
                    if now > deadline:
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

    def localization_adapter(self):
        """Pass-1 adapter for the smoke inference: this run's Stage 1, or a given one."""
        if "stage1" in self.args.phases:
            return self.root / "stage1" / "adapter"
        return Path(self.args.stage1_adapter).resolve() if self.args.stage1_adapter else None

    def run(self):
        a = self.args
        if a.wandb_mode == "online" and not os.environ.get("WANDB_API_KEY"):
            raise ValueError("Inject WANDB_API_KEY into every worker environment; do not put it in command logs")
        phases = a.phases
        data = Path(a.data).resolve()
        data_files = ["metadata_report.json"]
        if "stage1" in phases:
            data_files.append("stage1.jsonl")
        if "cache" in phases:
            data_files.append("stage2.jsonl")
        for name in data_files:
            if not (data / name).is_file():
                raise FileNotFoundError(data / name)
        reused_cache = None
        if "cache" not in phases and "stage2" in phases:
            reused_cache = Path(a.cache).resolve()
            if not (reused_cache / "manifest.json").is_file():
                raise FileNotFoundError(f"Reused cache is incomplete: {reused_cache / 'manifest.json'}")
        import importlib.metadata
        packages = {name: importlib.metadata.version(name) for name in
                    ("torch", "transformers", "accelerate", "peft", "byted-wandb", "setuptools")}
        common = {**{k:v for k,v in self.topo.items() if k != "node_rank"},
                  "args": vars(a), "git_commit": self.git_commit,
                  "source_sha256": source_digest(self.repo), "packages": packages,
                  "scripts": {str(p.relative_to(self.repo)):
                      hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (self.inference_script, self.audit_script)},
                  "data": {n:hashlib.sha256((data/n).read_bytes()).hexdigest() for n in
                           data_files},
                  "reused_cache_manifest": None if reused_cache is None else
                      hashlib.sha256((reused_cache / "manifest.json").read_bytes()).hexdigest()}
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
        self.distributed("collectives", ["-m", "samtok_edit21.distributed.probe", "--output", str(self.root/"collectives")])
        shared = ["--max-pixels", str(a.max_pixels), "--qwen", a.qwen, "--samtok", a.samtok,
                  "--num-workers", "0", "--seed", str(a.seed)]
        def tracking(stage):
            name = self.root.name + "-" + stage
            return ["--wandb-mode", a.wandb_mode, "--wandb-project", a.wandb_project,
                    "--wandb-entity", a.wandb_entity, "--wandb-name", name,
                    "--wandb-id", hashlib.sha256(str(self.root).encode()).hexdigest()[:16] + "-" + stage]
        if "stage1" in phases:
            self.distributed("stage1", ["-m", "samtok_edit21.training.engine", "train", "--stage", "stage1",
                "--metadata", str(data/"stage1.jsonl"), "--steps", str(a.stage1_steps),
                "--save-steps", str(a.stage1_save_steps), "--accumulation", "8", "--rank", str(a.stage1_rank),
                *(["--type-weights", a.stage1_type_weights] if a.stage1_type_weights else []),
                "--output", str(self.root/"stage1"), *shared, *tracking("stage1")])
        cache_output = self.root / "cache"
        if "cache" in phases:
            if a.cache:
                cache_output = Path(a.cache).resolve()
                link_alias(self.root / "cache", cache_output)
            self.distributed("cache", ["-m", "samtok_edit21.training.engine", "cache",
                "--metadata", str(data/"stage2.jsonl"), "--output", str(cache_output),
                "--cache-save-retries", str(a.cache_save_retries),
                "--cache-save-retry-backoff", str(a.cache_save_retry_backoff),
                *(["--resume-cache"] if a.resume_cache else []), *shared])
        elif reused_cache is not None:
            cache_output = reused_cache
            link_alias(self.root / "cache", cache_output)
        if "stage2" in phases:
            self.distributed("stage2", ["-m", "samtok_edit21.training.engine", "train", "--stage", "stage2",
                "--cache", str(cache_output), "--binding", a.binding,
                "--binding-beta", str(a.binding_beta), "--binding-eps", str(a.binding_eps),
                "--binding-rank", str(a.binding_rank),
                "--steps", str(a.stage2_steps), "--save-steps", str(a.stage2_save_steps), "--accumulation", "4",
                "--rank", str(a.stage2_rank),
                *(["--type-weights", a.stage2_type_weights] if a.stage2_type_weights else []),
                "--output", str(self.root/"stage2"), *shared, *tracking("stage2")])
        if self.rank == 0:
            self.command("audit", [sys.executable, str(self.audit_script), "--run-root", str(self.root)])
        self.barrier("audit")
        localization = self.localization_adapter()
        if not a.full_training and "stage2" in phases and localization is not None:
            # Smoke only: every inference mode/binding/blend path on eight GPUs.
            if self.rank == 0:
                self.command("inference", [sys.executable, "-m", "torch.distributed.run", "--standalone",
                             "--nproc_per_node", "8", "--max_restarts", "0", str(self.inference_script),
                             "--run-root", str(self.root), "--data", str(data),
                             "--te-adapter", str(localization),
                             "--qwen", a.qwen, "--samtok", a.samtok])
            self.barrier("inference")
        if self.rank == 0:
            atomic_json(self.root / "TRAINING_COMPLETE.json",
                        {"time": time.time(), "world_size": self.topo["world_size"], "phases": phases,
                         "data": str(data), "cache": str(cache_output),
                         "stage1_steps": a.stage1_steps if "stage1" in phases else None,
                         "stage2_steps": a.stage2_steps if "stage2" in phases else None,
                         "binding": a.binding if "stage2" in phases else None})
            atomic_json(self.root / "SUCCESS.json", {"time": time.time(), "world_size": self.topo["world_size"]})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", required=True, help="Fresh shared directory; identical on all nodes")
    p.add_argument("--data", required=True, help="v2 metadata directory (stage1.jsonl, stage2.jsonl, metadata_report.json)")
    p.add_argument("--local", action="store_true", help="One node x eight GPUs validation")
    p.add_argument("--phases", default="stage1,cache,stage2",
                   help="Comma-separated subset of stage1,cache,stage2 (Stage 2 needs a cache: built here or --cache)")
    p.add_argument("--cache", help="Cache directory: output of the cache phase, or a complete cache to reuse")
    p.add_argument("--resume-cache", action="store_true", help="Resume an incomplete --cache and reuse its payloads")
    p.add_argument("--stage1-adapter", help="Localization adapter for the smoke inference when Stage 1 is not run")
    p.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    p.add_argument("--wandb-project", default="samtok-edit")
    p.add_argument("--wandb-entity", default="2200012743-peking-university")
    p.add_argument("--qwen", default="/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen-Image-2.1")
    p.add_argument("--samtok", default="/mnt/bn/strategy-mllm-train/user/tanyue/models/SAMTok/Qwen3-VL-8B-SAMTok")
    p.add_argument("--stage1-steps", type=int, default=2)
    p.add_argument("--stage2-steps", type=int, default=3)
    p.add_argument("--stage1-rank", type=int, default=64)
    p.add_argument("--stage2-rank", type=int, default=32)
    p.add_argument("--stage1-save-steps", type=int, default=8, help="Per-rank microsteps, multiple of 8")
    p.add_argument("--stage2-save-steps", type=int, default=4, help="Per-rank microsteps, multiple of 4")
    p.add_argument("--binding", default="none",
                   choices=("none", "bias_span", "bias_clause", "region_embed", "region_rope"))
    p.add_argument("--stage1-type-weights", choices=("v1", "natural", "main4"),
                   help="Default natural (engine): every NTP row equally often")
    p.add_argument("--stage2-type-weights", choices=("v1", "natural", "main4"),
                   help="Default main4 (engine): add/remove/replace/attribute dominate")
    p.add_argument("--binding-beta", type=float, default=1.0)
    p.add_argument("--binding-eps", type=float, default=0.05)
    p.add_argument("--binding-rank", type=int, default=64)
    p.add_argument("--cache-save-retries", type=int, default=8)
    p.add_argument("--cache-save-retry-backoff", type=float, default=2.0)
    p.add_argument("--max-pixels", type=int, default=65536)
    p.add_argument("--full-training", action="store_true", help="Formal run: audit only, no smoke inference")
    p.add_argument("--seed", type=int, default=20261006)
    p.add_argument("--timeout", type=int, default=7200, help="Per-phase and barrier timeout in seconds")
    args = p.parse_args()
    phases = [x for x in args.phases.split(",") if x]
    if not phases or set(phases) - {"stage1", "cache", "stage2"} or len(set(phases)) != len(phases):
        raise SystemExit("--phases must be a subset of stage1,cache,stage2")
    args.phases = [x for x in ("stage1", "cache", "stage2") if x in phases]
    if "stage2" in args.phases and "cache" not in args.phases and not args.cache:
        raise SystemExit("Stage 2 without the cache phase needs --cache pointing at a complete cache")
    if args.resume_cache and ("cache" not in args.phases or not args.cache):
        raise SystemExit("--resume-cache needs the cache phase and an explicit --cache directory")
    if args.cache_save_retries < 0 or args.cache_save_retry_backoff < 0:
        raise SystemExit("cache-save-retries/backoff must be nonnegative")
    pipeline = Pipeline(args)
    try:
        pipeline.run()
    except BaseException as exc:
        record_failure(pipeline.node/"failure.json", {"error": str(exc), "type": type(exc).__name__, "time": time.time()})
        raise

if __name__ == "__main__":
    main()
