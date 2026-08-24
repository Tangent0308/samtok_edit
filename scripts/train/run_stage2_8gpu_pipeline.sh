#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

: "${RUN_ROOT:?Set RUN_ROOT to the Stage-2 experiment directory}"
: "${DATASET_BASE:?Set DATASET_BASE to the materialized Stage-2 dataset directory}"
: "${STAGE2_METADATA:?Set STAGE2_METADATA to stage2.jsonl}"
: "${TE_LORA_PATH:?Set TE_LORA_PATH to the completed Stage-1 LoRA checkpoint}"
: "${MERGED_TE_DIR:?Set MERGED_TE_DIR to the merged SAMTok tokenizer/processor directory}"
: "${WANDB_ENV_FILE:?Set WANDB_ENV_FILE to the protected WandB environment file}"

CACHE_ROOT="${CACHE_ROOT:-$RUN_ROOT/stage2_cache}"
TRAIN_OUTPUT_PATH="${TRAIN_OUTPUT_PATH:-$RUN_ROOT/stage2_dit_lora}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
REPORT_DIR="${REPORT_DIR:-$RUN_ROOT/reports}"
EXPECTED_COUNTS="${EXPECTED_COUNTS:-edit_mt:110640,edit:55320}"
EXPECTED_METADATA_SHA256="${EXPECTED_METADATA_SHA256:-}"
EXPECTED_TE_LORA_SHA256="${EXPECTED_TE_LORA_SHA256:-}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
CACHE_PORT="${CACHE_PORT:-50851}"
TRAIN_PORT="${TRAIN_PORT:-50852}"
DATASET_WORKERS="${DATASET_WORKERS:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

timestamp() {
  date -u +'%Y-%m-%dT%H:%M:%SZ'
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

fail() {
  log "ERROR: $*" >&2
  exit 1
}

check_sha256() {
  local path="$1"
  local expected="$2"
  local actual
  [[ -z "$expected" ]] && return
  actual="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || fail "SHA256 mismatch for $path: $actual != $expected"
  log "SHA256 verified: $path ($actual)"
}

check_port() {
  python - "$1" <<'PY'
import socket
import sys

port = int(sys.argv[1])
sock = socket.socket()
try:
    sock.bind(("127.0.0.1", port))
finally:
    sock.close()
PY
}

run_logged() {
  local phase="$1"
  local log_path="$2"
  shift 2
  log "$phase started; detailed log: $log_path"
  if "$@" >"$log_path" 2>&1; then
    log "$phase completed successfully"
  else
    local status=$?
    log "$phase failed with exit code $status; see $log_path" >&2
    return "$status"
  fi
}

[[ -f "$STAGE2_METADATA" ]] || fail "Missing Stage-2 metadata: $STAGE2_METADATA"
[[ -f "$TE_LORA_PATH" ]] || fail "Missing Stage-1 TE LoRA: $TE_LORA_PATH"
[[ -d "$MERGED_TE_DIR" ]] || fail "Missing merged TE directory: $MERGED_TE_DIR"
[[ -r "$WANDB_ENV_FILE" ]] || fail "Cannot read WandB environment file: $WANDB_ENV_FILE"
[[ "$(stat -c '%a' "$WANDB_ENV_FILE")" == "600" ]] || fail "WandB environment file must have mode 600"
[[ ! -e "$CACHE_ROOT" ]] || fail "Cache output already exists: $CACHE_ROOT"
[[ ! -e "$TRAIN_OUTPUT_PATH" ]] || fail "Training output already exists: $TRAIN_OUTPUT_PATH"

mkdir -p "$LOG_DIR" "$REPORT_DIR"
check_sha256 "$STAGE2_METADATA" "$EXPECTED_METADATA_SHA256"
check_sha256 "$TE_LORA_PATH" "$EXPECTED_TE_LORA_SHA256"
check_port "$CACHE_PORT" || fail "Cache rendezvous port is unavailable: $CACHE_PORT"
check_port "$TRAIN_PORT" || fail "Training rendezvous port is unavailable: $TRAIN_PORT"

set -a
# shellcheck disable=SC1090
source "$WANDB_ENV_FILE"
set +a
for wandb_var in WANDB_API_KEY WANDB_ENTITY WANDB_PROJECT; do
  [[ -n "${!wandb_var:-}" ]] || fail "Missing $wandb_var after sourcing $WANDB_ENV_FILE"
done

export PYTHONUNBUFFERED=1
export DIFFSYNTH_SKIP_DOWNLOAD=True
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-stage2-full-8gpu-$(date -u +'%Y%m%d-%H%M%S')}"

log "Stage-2 pipeline preflight passed"
log "metadata=$STAGE2_METADATA"
log "cache=$CACHE_ROOT"
log "train_output=$TRAIN_OUTPUT_PATH"
log "world_size=$NUM_PROCESSES, dataset_workers_per_rank=$DATASET_WORKERS"
log "WandB entity/project/run=$WANDB_ENTITY/$WANDB_PROJECT/$WANDB_RUN_NAME"

run_logged "Stage 2a cache generation" "$LOG_DIR/stage2a_cache.log" \
  env \
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    NUM_PROCESSES="$NUM_PROCESSES" \
    MAIN_PROCESS_PORT="$CACHE_PORT" \
    DATASET_WORKERS="$DATASET_WORKERS" \
    DEBUG_TRAIN_METRICS=0 \
    DATASET_BASE="$DATASET_BASE" \
    STAGE2_METADATA="$STAGE2_METADATA" \
    OUTPUT_PATH="$CACHE_ROOT" \
    TE_LORA_PATH="$TE_LORA_PATH" \
    MERGED_TE_DIR="$MERGED_TE_DIR" \
    bash "$SCRIPT_DIR/stage2_data_process.sh"

run_logged "Stage 2 cache audit" "$LOG_DIR/stage2_cache_audit.log" \
  python "$SCRIPT_DIR/audit_stage2_cache.py" \
    --cache_root "$CACHE_ROOT" \
    --expected_counts "$EXPECTED_COUNTS" \
    --world_size "$NUM_PROCESSES" \
    --expected_te_lora "$TE_LORA_PATH" \
    --report_json "$REPORT_DIR/stage2_cache_audit.json"

check_port "$TRAIN_PORT" || fail "Training rendezvous port became unavailable: $TRAIN_PORT"
run_logged "Stage 2b DiT LoRA training" "$LOG_DIR/stage2b_train.log" \
  env \
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    NUM_PROCESSES="$NUM_PROCESSES" \
    MAIN_PROCESS_PORT="$TRAIN_PORT" \
    DATASET_WORKERS="$DATASET_WORKERS" \
    DEBUG_TRAIN_METRICS=0 \
    ENABLE_WANDB_LOG=1 \
    CACHE_ROOT="$CACHE_ROOT" \
    OUTPUT_PATH="$TRAIN_OUTPUT_PATH" \
    MERGED_TE_DIR="$MERGED_TE_DIR" \
    DATASET_REPEAT="${DATASET_REPEAT:-2}" \
    NUM_EPOCHS="${NUM_EPOCHS:-1}" \
    GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}" \
    LEARNING_RATE="${LEARNING_RATE:-1e-4}" \
    WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}" \
    SAVE_STEPS="${SAVE_STEPS:-4000}" \
    bash "$SCRIPT_DIR/stage2_dit_lora.sh"

log "Stage-2 pipeline completed successfully"
