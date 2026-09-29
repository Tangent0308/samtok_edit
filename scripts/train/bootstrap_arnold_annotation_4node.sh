#!/usr/bin/env bash
# Submit this complete script on all 4 ARNOLD workers (8 GPUs each).
set -Eeuo pipefail

export SAMTOK_DATA_EXPERIMENT="${SAMTOK_DATA_EXPERIMENT:-/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/qwen21_full4_20260928}"
export SAMTOK_ANNOTATION_RUN_ID="${SAMTOK_ANNOTATION_RUN_ID:-qwen21_noref4n_full_003}"
export SAMTOK_ANNOTATION_SOURCES="${SAMTOK_ANNOTATION_SOURCES:-$SAMTOK_DATA_EXPERIMENT/data/semantic_sources.jsonl}"
export SAMTOK_ANNOTATION_MODEL="${SAMTOK_ANNOTATION_MODEL:-/mnt/bn/strategy-mllm-train/common/models/Qwen3-4B-Instruct-2507}"
export SAMTOK_ANNOTATION_BATCH_SIZE="${SAMTOK_ANNOTATION_BATCH_SIZE:-64}"
export SAMTOK_EDIT_REPO_URL="https://github.com/Tangent0308/samtok_edit.git"
export SAMTOK_EDIT_BRANCH="qwen-image-2.1-dev"
# Optional: pin a pushed commit. The revised prompt/protocol needs a fresh run.
# Resume only a run made with identical code, model, input and sharding.
export SAMTOK_EDIT_COMMIT="${SAMTOK_EDIT_COMMIT:-}"
export SAMTOK_ANNOTATION_RESUME_FROM="${SAMTOK_ANNOTATION_RESUME_FROM:-}"

: "${ARNOLD_WORKER_HOSTS:?ARNOLD must inject the common worker host list}"
: "${ARNOLD_WORKER_NUM:?ARNOLD must inject ARNOLD_WORKER_NUM=4}"
: "${ARNOLD_WORKER_GPU:?ARNOLD must inject ARNOLD_WORKER_GPU=8}"
: "${ARNOLD_ID:?ARNOLD must inject ARNOLD_ID=0..3}"
[[ "$ARNOLD_WORKER_NUM" == 4 && "$ARNOLD_WORKER_GPU" == 8 && "$ARNOLD_ID" =~ ^[0-3]$ ]] || {
  echo 'Expected ARNOLD 4 workers x 8 GPUs' >&2; exit 2;
}
[[ "$SAMTOK_ANNOTATION_RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid run ID' >&2; exit 2; }
[[ -f "$SAMTOK_ANNOTATION_SOURCES" ]] || { echo 'Semantic source manifest is missing' >&2; exit 2; }
if [[ -n "$SAMTOK_ANNOTATION_RESUME_FROM" ]]; then
  [[ -d "$SAMTOK_ANNOTATION_RESUME_FROM/shards" ]] || {
    echo 'Resume run has no shards directory' >&2; exit 2;
  }
fi
[[ -f "$SAMTOK_DATA_EXPERIMENT/data/sources.jsonl" && -f "$SAMTOK_DATA_EXPERIMENT/data/source_inventory.json" && -f "$SAMTOK_DATA_EXPERIMENT/data/semantic_inventory.json" ]] || {
  echo 'Prepared image/mask manifest or inventory is missing' >&2; exit 2;
}
export SAMTOK_ANNOTATION_RUN_ROOT="$SAMTOK_DATA_EXPERIMENT/data/semantic_runs/$SAMTOK_ANNOTATION_RUN_ID"
RUN="$SAMTOK_ANNOTATION_RUN_ROOT"
NODE="$ARNOLD_ID"
REPO="/tmp/samtok-noref-code-${SAMTOK_ANNOTATION_RUN_ID}-node${NODE}"
[[ ! -e "$RUN/nodes/$NODE" && ! -e "$RUN/SUCCESS.json" && ! -e "$REPO" ]] || {
  echo 'Run already used. Set a NEW common SAMTOK_ANNOTATION_RUN_ID.' >&2; exit 2;
}
mkdir -p "$RUN/bootstrap"
mkdir "$RUN/bootstrap/node${NODE}.claimed" || { echo 'Worker already claimed this run' >&2; exit 2; }
exec > >(tee -a "$RUN/bootstrap/node${NODE}.log") 2>&1
PHASE=checkout
failed() {
  local result="${1:-$?}"
  trap - ERR TERM INT
  mkdir -p "$RUN/nodes/$NODE"
  local temporary="$RUN/nodes/$NODE/bootstrap-failure.$$.tmp"
  printf '{"error":"annotation bootstrap failed in %s; see bootstrap/node%s.log","exit_code":%d}\n' \
    "$PHASE" "$NODE" "$result" > "$temporary"
  ln "$temporary" "$RUN/nodes/$NODE/failure.json" 2>/dev/null || true
  unlink "$temporary"
  exit "$result"
}
trap failed ERR
trap 'failed 143' TERM
trap 'failed 130' INT
export GIT_TERMINAL_PROMPT=0
git clone --branch "$SAMTOK_EDIT_BRANCH" --single-branch "$SAMTOK_EDIT_REPO_URL" "$REPO"
cd "$REPO"
if [[ -n "$SAMTOK_EDIT_COMMIT" ]]; then
  [[ "$SAMTOK_EDIT_COMMIT" =~ ^[a-fA-F0-9]{40}$ ]] || { echo 'Use a full commit SHA' >&2; false; }
  git checkout --detach "$SAMTOK_EDIT_COMMIT"
fi
git rev-parse HEAD > "$RUN/bootstrap/node${NODE}.commit.txt"
export SAMTOK_ENV="/tmp/samtok-noref-${SAMTOK_ANNOTATION_RUN_ID}-node${NODE}-env"
export SAMTOK_PYTHON="${SAMTOK_PYTHON:-/usr/bin/python3.11}"
PHASE=environment-or-annotation
bash scripts/train/run_arnold_annotation_4node.sh "$@"
