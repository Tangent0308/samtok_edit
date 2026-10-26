#!/usr/bin/env bash
# Round-2 run (stage B, C2 by default): pre-clone ARNOLD entry.  Paste this whole file as the start command of every
# worker (4 workers x 8 GPUs), after editing the "run settings" block.  Each worker clones the repository at the pinned
# commit into node-local /tmp, builds the environment, then trains, evaluates and (worker 0) writes the report.
set -Eeuo pipefail
exec 2>&1  # ARNOLD's log page shows stdout; route every shell error there as well.

# ===== run settings (edit these) =====
# A fresh ID per attempt, identical on all workers.  A used ID is refused so a scheduler retry cannot overwrite logs.
export SAMTOK_RUN_ID="${SAMTOK_RUN_ID:?Set one fresh common SAMTOK_RUN_ID on all workers}"
# Label of this run in the report.
export SAMTOK_RUN_NAME="${SAMTOK_RUN_NAME:-$SAMTOK_RUN_ID}"
# Shared experiment root: runs go to $SAMTOK_EXPERIMENT/runs/<run id>.
export SAMTOK_EXPERIMENT="${SAMTOK_EXPERIMENT:-/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v3}"
# Data directory written by `samtok-edit round2-data` (rows, evaluation items, tokens, mask pool).
export SAMTOK_ROUND2_DATA="${SAMTOK_ROUND2_DATA:-$SAMTOK_EXPERIMENT/data/round2_20261016}"
# Trainer / evaluator arguments (see `samtok-edit round2-train --help`, `round2-eval --help`).  Default: C2 (docs/03),
# every row once: 3,522 updates x (32 GPUs x accumulation 1) = the 112,704 rows of rows_c2.jsonl.
export SAMTOK_TRAIN_ARGS="${SAMTOK_TRAIN_ARGS:---rows rows_c2.jsonl --quick-items eval_c2_quick_idx.json --init-adapter $SAMTOK_EXPERIMENT/runs/c1b_local8_001/train/adapter_800 --updates 3522 --accumulation 1 --warmup 20 --lr 1e-4 --lr-schedule cosine --lr-min 1e-5 --heavy-weight 8 --checkpoints 500,1000,1500,2000,2500,3000,3500,3522 --quick-eval 0,500,1000,1500,2000,2500,3000,3500,3522}"
export SAMTOK_EVAL_ARGS="${SAMTOK_EVAL_ARGS:-}"
# Reference evaluations for the paired report, "LABEL=PATH ..." with the base model first.
export SAMTOK_REPORT_REFERENCES="${SAMTOK_REPORT_REFERENCES:-stock=$SAMTOK_EXPERIMENT/baselines/round2_stock_eval C1=$SAMTOK_EXPERIMENT/runs/c1_local8_001/eval_1500 C1b=$SAMTOK_EXPERIMENT/runs/c1b_local8_001/eval_800}"
# W&B, as in the v2 runs: rank 0 logs the training metrics and quick evaluations; the report adds the dev results.
export SAMTOK_WANDB_MODE="${SAMTOK_WANDB_MODE:-online}"   # online | offline | disabled
export WANDB_ENTITY="${WANDB_ENTITY:-2200012743-peking-university}"
export WANDB_PROJECT="${WANDB_PROJECT:-samtok-edit}"
# W&B key: an ARNOLD secret if injected, else the private key file on the shared disk (export WANDB_API_KEY=..., mode
# 600).  The repository is public: never write the key into a script or commit it.
export SAMTOK_WANDB_KEY_FILE="${SAMTOK_WANDB_KEY_FILE:-/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/.secrets/wandb.env}"
# Phases to run: train,eval,report.  Without "train", set SAMTOK_ADAPTER to the adapter directory to evaluate.
export SAMTOK_PHASES="${SAMTOK_PHASES:-train,eval,report}"
export SAMTOK_ADAPTER="${SAMTOK_ADAPTER:-}"
# Code: branch and (recommended) the exact commit, so all workers and any later rerun use the same code.
export SAMTOK_EDIT_REPO_URL="${SAMTOK_EDIT_REPO_URL:-https://github.com/Tangent0308/samtok_edit.git}"
export SAMTOK_EDIT_BRANCH="${SAMTOK_EDIT_BRANCH:-qwen-image-2.1-v4}"
export SAMTOK_EDIT_COMMIT="${SAMTOK_EDIT_COMMIT:-}"
# Number of workers; 1 runs the same pipeline on a single 8-GPU machine (then use --accumulation 4).
export SAMTOK_NNODES="${SAMTOK_NNODES:-4}"

# ===== ARNOLD and input checks (before anything is cloned or installed) =====
: "${ARNOLD_WORKER_HOSTS:?ARNOLD must inject the worker host list}"
: "${ARNOLD_WORKER_NUM:?ARNOLD must inject ARNOLD_WORKER_NUM}"
: "${ARNOLD_WORKER_GPU:?ARNOLD must inject ARNOLD_WORKER_GPU}"
: "${ARNOLD_ID:?ARNOLD must inject ARNOLD_ID for this worker}"
[[ "$ARNOLD_WORKER_NUM" == "$SAMTOK_NNODES" ]] || { echo "Expected $SAMTOK_NNODES workers, ARNOLD reports $ARNOLD_WORKER_NUM" >&2; exit 2; }
[[ "$ARNOLD_WORKER_GPU" =~ ^[1-9][0-9]*$ ]] || { echo 'ARNOLD_WORKER_GPU must be a positive integer' >&2; exit 2; }
[[ "$ARNOLD_ID" =~ ^[0-9]+$ && "$ARNOLD_ID" -lt "$SAMTOK_NNODES" ]] || { echo "ARNOLD_ID must be 0..$((SAMTOK_NNODES - 1))" >&2; exit 2; }
[[ "$SAMTOK_RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid SAMTOK_RUN_ID' >&2; exit 2; }
[[ "$SAMTOK_WANDB_MODE" =~ ^(online|offline|disabled)$ ]] || { echo 'SAMTOK_WANDB_MODE must be online, offline or disabled' >&2; exit 2; }
if [[ "$SAMTOK_WANDB_MODE" == online ]]; then
  # shellcheck disable=SC1090  # the private key file
  [[ -n "${WANDB_API_KEY:-}" || ! -f "$SAMTOK_WANDB_KEY_FILE" ]] || source "$SAMTOK_WANDB_KEY_FILE"
  [[ -n "${WANDB_API_KEY:-}" ]] || { echo "No W&B key: inject WANDB_API_KEY or create $SAMTOK_WANDB_KEY_FILE" >&2; exit 2; }
  export WANDB_API_KEY
fi
option() {  # option() NAME DEFAULT ARGS...: the value given to --NAME in ARGS, else DEFAULT
  local name="$1" default="$2" previous="" word
  shift 2
  for word in "$@"; do
    [[ "$previous" == "--$name" ]] && { echo "$word"; return; }
    previous="$word"
  done
  echo "$default"
}
# shellcheck disable=SC2086  # the argument strings are lists of options
ITEMS="$(option items eval_c1.json $SAMTOK_EVAL_ARGS)"
[[ -f "$SAMTOK_ROUND2_DATA/$ITEMS" && -f "$SAMTOK_ROUND2_DATA/edit_tokens.jsonl" && -d "$SAMTOK_ROUND2_DATA/pool" ]] || {
  echo "Missing $ITEMS, edit_tokens.jsonl or pool/ in $SAMTOK_ROUND2_DATA" >&2; exit 2; }
if [[ ",$SAMTOK_PHASES," == *,train,* ]]; then
  # shellcheck disable=SC2086
  for needed in "$(option rows rows_c1.jsonl $SAMTOK_TRAIN_ARGS)" "$(option quick-items eval_c1_quick_idx.json $SAMTOK_TRAIN_ARGS)"; do
    [[ -f "$SAMTOK_ROUND2_DATA/$needed" ]] || { echo "No $needed in $SAMTOK_ROUND2_DATA" >&2; exit 2; }
  done
  # shellcheck disable=SC2086
  INIT="$(option init-adapter "" $SAMTOK_TRAIN_ARGS)"
  [[ -z "$INIT" || -f "$INIT/adapter.safetensors" ]] || { echo "No adapter.safetensors in --init-adapter $INIT" >&2; exit 2; }
elif [[ ",$SAMTOK_PHASES," == *,eval,* && ! -f "$SAMTOK_ADAPTER/adapter.safetensors" ]]; then
  echo 'Evaluation without training needs SAMTOK_ADAPTER=<adapter directory>' >&2; exit 2
fi

# ARNOLD_WORKER_HOSTS carries the common rendezvous port; generic PORT varies by worker.
unset PORT MASTER_ADDR MASTER_PORT NODE_RANK NNODES GPUS_PER_NODE
NODE="$ARNOLD_ID"
RUN="$SAMTOK_EXPERIMENT/runs/$SAMTOK_RUN_ID"
BOOTSTRAP="$RUN/bootstrap"
REPO="/tmp/samtok-edit-${SAMTOK_RUN_ID}-node${NODE}"
if [[ -e "$RUN/nodes/$NODE" || -e "$RUN/SUCCESS.json" ]]; then
  echo "Run already used: $RUN. Set a NEW common SAMTOK_RUN_ID; old logs are preserved." >&2
  exit 2
fi
# A full NAS directory quota rejects every new file while df still reports free space.
PROBE="$SAMTOK_EXPERIMENT/runs/.write_probe_${SAMTOK_RUN_ID}_node${NODE}"
if ! (mkdir -p "$SAMTOK_EXPERIMENT/runs" && printf 'probe\n' > "$PROBE" && rm -f "$PROBE"); then
  echo "Cannot write under $SAMTOK_EXPERIMENT/runs (NAS quota exceeded?). Nothing was started." >&2
  exit 3
fi
mkdir -p "$BOOTSTRAP"
# Atomic per-node claim: reject scheduler retries BEFORE cloning/installing or appending logs.
if ! mkdir "$BOOTSTRAP/node${NODE}.claimed"; then
  echo "Worker $NODE already started this run. Use a NEW common SAMTOK_RUN_ID." >&2
  exit 2
fi
mkdir -p "$RUN/nodes/$NODE"
exec > >(tee -a "$BOOTSTRAP/node${NODE}.log") 2>&1
export SAMTOK_PHASE_FILE="$RUN/nodes/$NODE/phase"
echo checkout > "$SAMTOK_PHASE_FILE"
bootstrap_failed() {
  local result="${1:-$?}"
  trap - ERR TERM INT
  printf '{"error":"failed during %s; see bootstrap/node%s.log","exit_code":%d}\n' \
    "$(cat "$SAMTOK_PHASE_FILE" 2>/dev/null || echo unknown)" "$NODE" "$result" > "$RUN/nodes/$NODE/failure.json"
  exit "$result"
}
trap bootstrap_failed ERR
trap 'bootstrap_failed 143' TERM
trap 'bootstrap_failed 130' INT
echo "[$(date -u +%FT%TZ)] node $NODE of $SAMTOK_NNODES: start, run $SAMTOK_RUN_ID"

if [[ -e "$REPO" ]]; then
  echo "Node-local checkout already exists: $REPO (choose a fresh SAMTOK_RUN_ID)" >&2
  false
fi
export GIT_TERMINAL_PROMPT=0
# Clone the exact requested branch into node-local /tmp; never execute an MNT source snapshot.
git clone --branch "$SAMTOK_EDIT_BRANCH" --single-branch "$SAMTOK_EDIT_REPO_URL" "$REPO"
if [[ -n "$SAMTOK_EDIT_COMMIT" ]]; then
  [[ "$SAMTOK_EDIT_COMMIT" =~ ^[a-fA-F0-9]{40}$ ]] || { echo 'Use a full commit SHA' >&2; false; }
  git -C "$REPO" checkout --detach "$SAMTOK_EDIT_COMMIT"
fi
git -C "$REPO" rev-parse HEAD > "$BOOTSTRAP/node${NODE}.commit.txt"
export SAMTOK_RUN_ROOT="$RUN" SAMTOK_NODE="$NODE"
bash "$REPO/scripts/round2/run_arnold.sh"
echo "[$(date -u +%FT%TZ)] node $NODE: done"
