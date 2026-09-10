#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/arnold_4node_env.sh"
samtok_init_arnold_topology

: "${RUN_ROOT:?Set RUN_ROOT to a new shared four-node experiment directory}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in the Arnold job environment}"
: "${WANDB_ENTITY:?Set WANDB_ENTITY in the Arnold job environment}"
: "${WANDB_PROJECT:?Set WANDB_PROJECT in the Arnold job environment}"

SAMTOK_EDIT_VENV="${SAMTOK_EDIT_VENV:-$REPO_ROOT/.venv}"
STAGE1_SOURCE_BASE="${STAGE1_SOURCE_BASE:-/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/crispedit_refined/stage1_full/data/crispedit_samtok}"
STAGE2_SOURCE_BASE="${STAGE2_SOURCE_BASE:-/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/crispedit_refined/stage2_full/data/crispedit_samtok}"
MERGED_TE_DIR="${MERGED_TE_DIR:-/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/artifacts/merged_samtok_te}"
STAGE1_METADATA="$RUN_ROOT/data/stage1_ws32.jsonl"
STAGE2_METADATA="$RUN_ROOT/data/stage2_ws32.jsonl"
STAGE1_OUTPUT="$RUN_ROOT/stage1_te_lora"
STAGE2_CACHE="$RUN_ROOT/stage2_cache"
STAGE2_OUTPUT="$RUN_ROOT/stage2_dit_lora"
LOG_DIR="$RUN_ROOT/logs"
REPORT_DIR="$RUN_ROOT/reports"
CONTROL_DIR="$RUN_ROOT/control"
RUN_ID="${SAMTOK_RUN_ID:-$(basename "$RUN_ROOT")}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-21600}"
TOPOLOGY_TIMEOUT_SECONDS="${TOPOLOGY_TIMEOUT_SECONDS:-300}"
CACHE_AUDIT_WORKERS="${CACHE_AUDIT_WORKERS:-32}"

[[ -x "$SAMTOK_EDIT_VENV/bin/python" ]] || { echo "uv environment is missing: $SAMTOK_EDIT_VENV" >&2; exit 1; }
[[ -d "$STAGE1_SOURCE_BASE" ]] || { echo "Stage-1 refined source base is missing: $STAGE1_SOURCE_BASE" >&2; exit 1; }
[[ -d "$STAGE2_SOURCE_BASE" ]] || { echo "Stage-2 refined source base is missing: $STAGE2_SOURCE_BASE" >&2; exit 1; }
[[ -d "$MERGED_TE_DIR" ]] || { echo "Merged SAMTok TE directory is missing: $MERGED_TE_DIR" >&2; exit 1; }

export SAMTOK_EDIT_VENV
export PATH="$SAMTOK_EDIT_VENV/bin:$PATH"
export PYTHONUNBUFFERED=1
export DIFFSYNTH_SKIP_DOWNLOAD=True
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

timestamp() {
  date -u +'%Y-%m-%dT%H:%M:%SZ'
}

log() {
  printf '[%s][node=%s] %s\n' "$(timestamp)" "$NODE_RANK" "$*"
}

atomic_marker() {
  local path="$1"
  local value="${2:-ok}"
  local temporary="${path}.tmp.${NODE_RANK}.$$"
  printf '%s\n' "$value" >"$temporary"
  mv "$temporary" "$path"
}

wait_for_stage() {
  local stage="$1"
  local started=$SECONDS
  local failed
  while true; do
    failed="$(find "$CONTROL_DIR" -maxdepth 1 -name "$stage.node*.failed" -print -quit 2>/dev/null || true)"
    if [[ -n "$failed" ]]; then
      echo "Stage $stage failed; marker=$failed; status=$(<"$failed")" >&2
      return 1
    fi
    if [[ -f "$CONTROL_DIR/$stage.ok" ]]; then
      return 0
    fi
    if (( SECONDS - started >= WAIT_TIMEOUT_SECONDS )); then
      echo "Timed out waiting for stage $stage after ${WAIT_TIMEOUT_SECONDS}s" >&2
      return 1
    fi
    sleep 2
  done
}

wait_for_all_nodes() {
  local stage="$1"
  local started=$SECONDS
  local completed
  local failed
  while true; do
    completed="$(find "$CONTROL_DIR" -maxdepth 1 -name "$stage.node*.done" -print 2>/dev/null | wc -l)"
    failed="$(find "$CONTROL_DIR" -maxdepth 1 -name "$stage.node*.failed" -print -quit 2>/dev/null || true)"
    if [[ -n "$failed" ]]; then
      echo "Stage $stage failed; marker=$failed; status=$(<"$failed")" >&2
      return 1
    fi
    if (( completed == NNODES )); then
      atomic_marker "$CONTROL_DIR/$stage.ok"
      return 0
    fi
    if (( SECONDS - started >= WAIT_TIMEOUT_SECONDS )); then
      echo "Timed out waiting for all nodes in stage $stage; completed=$completed/$NNODES" >&2
      return 1
    fi
    sleep 2
  done
}

run_rank0_stage() {
  local stage="$1"
  shift
  if (( NODE_RANK == 0 )); then
    log "$stage started on controller"
    set +e
    "$@"
    local status=$?
    set -e
    if (( status != 0 )); then
      atomic_marker "$CONTROL_DIR/$stage.node0.failed" "$status"
      return "$status"
    fi
    atomic_marker "$CONTROL_DIR/$stage.ok"
    log "$stage completed"
  else
    log "waiting for controller stage $stage"
    wait_for_stage "$stage"
  fi
}

run_distributed_stage() {
  local stage="$1"
  shift
  local node_log="$LOG_DIR/${stage}.node${NODE_RANK}.log"
  log "$stage started; log=$node_log"
  set +e
  "$@" 2>&1 | tee "$node_log"
  local status=${PIPESTATUS[0]}
  set -e
  if (( status != 0 )); then
    atomic_marker "$CONTROL_DIR/$stage.node${NODE_RANK}.failed" "$status"
    log "$stage failed with exit code $status"
    return "$status"
  fi
  atomic_marker "$CONTROL_DIR/$stage.node${NODE_RANK}.done"
  if (( NODE_RANK == 0 )); then
    wait_for_all_nodes "$stage"
  else
    wait_for_stage "$stage"
  fi
  log "$stage completed on all nodes"
}

initialize_run() {
  if [[ -e "$RUN_ROOT" ]]; then
    if [[ "${SAMTOK_ALLOW_BOOTSTRAP_RUN_ROOT:-0}" != "1" ]]; then
      echo "Refusing to reuse an existing RUN_ROOT: $RUN_ROOT" >&2
      return 1
    fi
    local unexpected_root
    local unexpected_log
    local unexpected_control
    unexpected_root="$(
      find "$RUN_ROOT" -mindepth 1 -maxdepth 1 \
        ! -name logs ! -name bootstrap_control -print -quit
    )"
    unexpected_log="$(
      find "$RUN_ROOT/logs" -mindepth 1 -maxdepth 1 \
        ! -name 'bootstrap.node*.log' -print -quit 2>/dev/null || true
    )"
    unexpected_control="$(
      find "$BOOTSTRAP_CONTROL" -mindepth 1 -maxdepth 1 \
        ! -name environment.ok ! -name git_commit.txt ! -name session.id -print -quit 2>/dev/null || true
    )"
    if [[ -n "$unexpected_root" || -n "$unexpected_log" || -n "$unexpected_control" ]]; then
      echo "Refusing to reuse a non-empty RUN_ROOT: $RUN_ROOT" >&2
      echo "unexpected_root=$unexpected_root" >&2
      echo "unexpected_log=$unexpected_log" >&2
      echo "unexpected_bootstrap_control=$unexpected_control" >&2
      return 1
    fi
  fi
  mkdir -p "$LOG_DIR" "$REPORT_DIR" "$CONTROL_DIR"
}

write_topology_report() {
  local report="$REPORT_DIR/topology.node${NODE_RANK}.json"
  local temporary="${report}.tmp.$$"
  python - "$temporary" "$RUN_ID" <<'PY'
import json
import os
import sys

payload = {
    "protocol": "samtok_edit_crispedit_refined_4node",
    "node_rank": int(os.environ["NODE_RANK"]),
    "nnodes": int(os.environ["NNODES"]),
    "gpus_per_node": int(os.environ["GPUS_PER_NODE"]),
    "world_size": int(os.environ["WORLD_SIZE"]),
    "master_addr": os.environ["MASTER_ADDR"],
    "master_port": int(os.environ["MASTER_PORT"]),
    "run_id": sys.argv[2],
}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")
print(json.dumps(payload, indent=2))
PY
  mv "$temporary" "$report"
}

validate_topology_consensus() {
  python - "$REPORT_DIR" "$NNODES" "$REPORT_DIR/topology.json" <<'PY'
import json
import sys
from pathlib import Path

report_dir = Path(sys.argv[1])
nnodes = int(sys.argv[2])
output_path = Path(sys.argv[3])
reports = [
    json.loads((report_dir / f"topology.node{rank}.json").read_text(encoding="utf-8"))
    for rank in range(nnodes)
]

expected_ranks = list(range(nnodes))
actual_ranks = sorted(report["node_rank"] for report in reports)
if actual_ranks != expected_ranks:
    raise SystemExit(f"node ranks differ: expected={expected_ranks}, actual={actual_ranks}")

shared_fields = ("protocol", "nnodes", "gpus_per_node", "world_size", "master_addr", "master_port", "run_id")
reference = {field: reports[0][field] for field in shared_fields}
mismatches = []
for report in reports[1:]:
    current = {field: report[field] for field in shared_fields}
    if current != reference:
        mismatches.append({"node_rank": report["node_rank"], "values": current})
if mismatches:
    raise SystemExit(
        "four-node topology mismatch; "
        + json.dumps({"node0": reference, "mismatches": mismatches}, sort_keys=True)
    )
if reference["nnodes"] != nnodes or reference["world_size"] != nnodes * reference["gpus_per_node"]:
    raise SystemExit(f"invalid world-size topology: {reference}")

payload = dict(reference)
payload["nodes"] = reports
temporary = output_path.with_name(output_path.name + ".tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
temporary.replace(output_path)
print(json.dumps(payload, indent=2))
PY
}

run_topology_consensus() {
  local stage="topology_consensus"
  local status
  local started
  local completed
  local failed

  set +e
  write_topology_report
  status=$?
  set -e
  if (( status != 0 )); then
    atomic_marker "$CONTROL_DIR/$stage.node${NODE_RANK}.failed" "$status"
    return "$status"
  fi
  atomic_marker "$CONTROL_DIR/$stage.node${NODE_RANK}.done"

  if (( NODE_RANK != 0 )); then
    log "waiting for four-node topology consensus"
    wait_for_stage "$stage"
    return
  fi

  started=$SECONDS
  while true; do
    completed="$(find "$CONTROL_DIR" -maxdepth 1 -name "$stage.node*.done" -print 2>/dev/null | wc -l)"
    failed="$(find "$CONTROL_DIR" -maxdepth 1 -name "$stage.node*.failed" -print -quit 2>/dev/null || true)"
    if [[ -n "$failed" ]]; then
      echo "Topology report failed; marker=$failed; status=$(<"$failed")" >&2
      return 1
    fi
    if (( completed == NNODES )); then
      break
    fi
    if (( SECONDS - started >= TOPOLOGY_TIMEOUT_SECONDS )); then
      atomic_marker "$CONTROL_DIR/$stage.node0.failed" "timed out: $completed/$NNODES reports"
      echo "Timed out waiting for topology reports after ${TOPOLOGY_TIMEOUT_SECONDS}s; completed=$completed/$NNODES" >&2
      return 1
    fi
    sleep 2
  done

  set +e
  validate_topology_consensus
  status=$?
  set -e
  if (( status != 0 )); then
    atomic_marker "$CONTROL_DIR/$stage.node0.failed" "$status"
    return "$status"
  fi
  atomic_marker "$CONTROL_DIR/$stage.ok"
  log "four-node topology consensus passed"
}

select_stage1_checkpoint() {
  local expected="$STAGE1_OUTPUT/step-2648.safetensors"
  [[ -s "$expected" ]] || { echo "Expected Stage-1 final checkpoint is missing: $expected" >&2; return 1; }
  printf '%s\n' "$expected" >"$CONTROL_DIR/stage1_checkpoint.txt.tmp"
  mv "$CONTROL_DIR/stage1_checkpoint.txt.tmp" "$CONTROL_DIR/stage1_checkpoint.txt"
  sha256sum "$expected" >"$REPORT_DIR/stage1_checkpoint_sha256.txt"
}

audit_stage2_cache() {
  local te_lora_path
  te_lora_path="$(<"$CONTROL_DIR/stage1_checkpoint.txt")"
  python "$SCRIPT_DIR/audit_stage2_cache.py" \
    --cache_root "$STAGE2_CACHE" \
    --expected_counts edit_mt:42368,edit:21184,edit_umt:21184 \
    --world_size 32 \
    --expected_te_lora "$te_lora_path" \
    --report_json "$REPORT_DIR/stage2_cache_audit.json" \
    --workers "$CACHE_AUDIT_WORKERS" \
    --torch_threads_per_worker 1 \
    --chunksize 4 \
    --log_every 500 \
    2>&1 | tee "$LOG_DIR/stage2_cache_audit.node0.log"
}

finalize_run() {
  local stage1_checkpoint
  local stage2_checkpoint="$STAGE2_OUTPUT/step-5296.safetensors"
  stage1_checkpoint="$(<"$CONTROL_DIR/stage1_checkpoint.txt")"
  [[ -s "$stage2_checkpoint" ]] || { echo "Expected Stage-2 final checkpoint is missing: $stage2_checkpoint" >&2; return 1; }
  sha256sum "$stage2_checkpoint" >"$REPORT_DIR/stage2_checkpoint_sha256.txt"
  python - \
    "$REPORT_DIR/run_manifest.json" \
    "$STAGE1_METADATA" \
    "$stage1_checkpoint" \
    "$STAGE2_METADATA" \
    "$REPORT_DIR/stage2_cache_audit.json" \
    "$stage2_checkpoint" <<'PY'
import hashlib
import json
import sys
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


output, stage1_metadata, stage1_checkpoint, stage2_metadata, cache_audit, stage2_checkpoint = map(Path, sys.argv[1:])
audit = json.loads(cache_audit.read_text(encoding="utf-8"))
if not audit.get("passed"):
    raise SystemExit("Stage-2 cache audit was not successful")
payload = {
    "protocol": "samtok_edit_crispedit_refined_4node",
    "world_size": 32,
    "stage1": {
        "metadata": str(stage1_metadata.resolve()),
        "metadata_sha256": sha256(stage1_metadata),
        "checkpoint": str(stage1_checkpoint.resolve()),
        "checkpoint_sha256": sha256(stage1_checkpoint),
    },
    "stage2": {
        "metadata": str(stage2_metadata.resolve()),
        "metadata_sha256": sha256(stage2_metadata),
        "cache_audit": str(cache_audit.resolve()),
        "cache_rows": audit["cache_files"],
        "checkpoint": str(stage2_checkpoint.resolve()),
        "checkpoint_sha256": sha256(stage2_checkpoint),
    },
}
output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY
}

INITIALIZE_FAILURE_MARKER="${RUN_ROOT}.initialize.failed"
if (( NODE_RANK == 0 )); then
  if ! initialize_run; then
    mkdir -p "$(dirname "$INITIALIZE_FAILURE_MARKER")"
    printf 'initialize_run failed\n' >"$INITIALIZE_FAILURE_MARKER"
    exit 1
  fi
  atomic_marker "$CONTROL_DIR/initialize.ok"
else
  while [[ ! -d "$CONTROL_DIR" ]]; do
    if [[ -f "$INITIALIZE_FAILURE_MARKER" ]]; then
      echo "Controller failed to initialize RUN_ROOT: $RUN_ROOT" >&2
      exit 1
    fi
    sleep 2
  done
  wait_for_stage initialize
fi

run_topology_consensus
log "topology nnodes=$NNODES gpus_per_node=$GPUS_PER_NODE world_size=$WORLD_SIZE master=$MASTER_ADDR:$MASTER_PORT"

run_rank0_stage prepare_metadata \
  env \
    RUN_ROOT="$RUN_ROOT" \
    WORLD_SIZE=32 \
    STAGE1_SOURCE_BASE="$STAGE1_SOURCE_BASE" \
    STAGE2_SOURCE_BASE="$STAGE2_SOURCE_BASE" \
    bash "$REPO_ROOT/scripts/data/prepare_4node_metadata.sh"

run_distributed_stage stage1_train \
  env \
    DATASET_BASE="$STAGE1_SOURCE_BASE" \
    STAGE1_METADATA="$STAGE1_METADATA" \
    OUTPUT_PATH="$STAGE1_OUTPUT" \
    MERGED_TE_DIR="$MERGED_TE_DIR" \
    DATASET_WORKERS="${STAGE1_DATASET_WORKERS:-2}" \
    DATASET_REPEAT=1 \
    NUM_EPOCHS=1 \
    GRADIENT_ACCUMULATION_STEPS=8 \
    SAVE_STEPS="${STAGE1_SAVE_STEPS:-2000}" \
    LEARNING_RATE="${STAGE1_LEARNING_RATE:-4e-5}" \
    WEIGHT_DECAY="${STAGE1_WEIGHT_DECAY:-0.05}" \
    NTP_LOSS_WEIGHT="${NTP_LOSS_WEIGHT:-0.05}" \
    FM_LOSS_WEIGHT="${FM_LOSS_WEIGHT:-1.0}" \
    DEBUG_TRAIN_METRICS="${STAGE1_DEBUG_TRAIN_METRICS:-0}" \
    WANDB_RUN_NAME="${RUN_ID}-stage1" \
    bash "$SCRIPT_DIR/launch_4node.sh" stage1

run_rank0_stage select_stage1_checkpoint select_stage1_checkpoint
wait_for_stage select_stage1_checkpoint
TE_LORA_PATH="$(<"$CONTROL_DIR/stage1_checkpoint.txt")"

run_distributed_stage stage2_cache \
  env \
    DATASET_BASE="$STAGE2_SOURCE_BASE" \
    STAGE2_METADATA="$STAGE2_METADATA" \
    OUTPUT_PATH="$STAGE2_CACHE" \
    TE_LORA_PATH="$TE_LORA_PATH" \
    MERGED_TE_DIR="$MERGED_TE_DIR" \
    DATASET_WORKERS="${STAGE2_CACHE_DATASET_WORKERS:-2}" \
    DEBUG_TRAIN_METRICS="${STAGE2_CACHE_DEBUG_TRAIN_METRICS:-0}" \
    bash "$SCRIPT_DIR/launch_4node.sh" stage2_cache

run_rank0_stage audit_stage2_cache audit_stage2_cache
wait_for_stage audit_stage2_cache

run_distributed_stage stage2_train \
  env \
    CACHE_ROOT="$STAGE2_CACHE" \
    OUTPUT_PATH="$STAGE2_OUTPUT" \
    MERGED_TE_DIR="$MERGED_TE_DIR" \
    DATASET_WORKERS="${STAGE2_TRAIN_DATASET_WORKERS:-2}" \
    DATASET_REPEAT=2 \
    NUM_EPOCHS=1 \
    GRADIENT_ACCUMULATION_STEPS=1 \
    SAVE_STEPS="${STAGE2_SAVE_STEPS:-4000}" \
    LEARNING_RATE="${STAGE2_LEARNING_RATE:-1e-4}" \
    WEIGHT_DECAY="${STAGE2_WEIGHT_DECAY:-0.01}" \
    DEBUG_TRAIN_METRICS="${STAGE2_DEBUG_TRAIN_METRICS:-0}" \
    WANDB_RUN_NAME="${RUN_ID}-stage2" \
    bash "$SCRIPT_DIR/launch_4node.sh" stage2_train

run_rank0_stage finalize finalize_run
wait_for_stage finalize
log "four-node Stage-1 and Stage-2 pipeline completed: $RUN_ROOT"
