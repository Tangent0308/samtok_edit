#!/usr/bin/env bash
# IF-Bench 全量评测（eval-only，不训练）——R2@1000 交付检查点
# 整段原样粘贴成 ARNOLD 四个 worker（4 × 8 卡）的启动命令，四个 worker 相同。
# 重新提交时只改 SAMTOK_RUN_ID。
set -Eeuo pipefail
exec 2>&1

# ===== 运行设置 =====
export SAMTOK_RUN_ID=r2_ifbench_001
export SAMTOK_RUN_NAME=R2-IFBench
export SAMTOK_EXPERIMENT=/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v3
export SAMTOK_ROUND2_DATA=$SAMTOK_EXPERIMENT/data_round2_v6p
export SAMTOK_PHASES=eval
export SAMTOK_ADAPTER=/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/quality_anchor_20261104/runs_v5/v6_full/r2/train/adapter_1000
export SAMTOK_EVAL_ARGS="--items eval_c2.json --lora-scale 1.0 --guidance 2.0"
export SAMTOK_TRAIN_ARGS=
export SAMTOK_REPORT_REFERENCES=
export SAMTOK_WANDB_MODE=disabled
export WANDB_ENTITY=2200012743-peking-university
export WANDB_PROJECT=samtok-edit
export SAMTOK_WANDB_KEY_FILE=$SAMTOK_EXPERIMENT/../../../experiments2/SAMTokEdit/.secrets/wandb.env
export SAMTOK_EDIT_REPO_URL=https://github.com/Tangent0308/samtok_edit.git
export SAMTOK_EDIT_BRANCH=qwen-image-2.1-v5
export SAMTOK_EDIT_COMMIT=22650e4b30830f95a11a3af8f4558326d98edb2f
export SAMTOK_NNODES=4

# ===== 以下为 scripts/round2/bootstrap_arnold.sh 全文（不用改）=====
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
  [[ -n "${WANDB_API_KEY:-}" || ! -f "$SAMTOK_WANDB_KEY_FILE" ]] || source "$SAMTOK_WANDB_KEY_FILE"
  [[ -n "${WANDB_API_KEY:-}" ]] || { echo "No W&B key: inject WANDB_API_KEY or create $SAMTOK_WANDB_KEY_FILE" >&2; exit 2; }
  export WANDB_API_KEY
fi
option() {
  local name="$1" default="$2" previous="" word
  shift 2
  for word in "$@"; do
    [[ "$previous" == "--$name" ]] && { echo "$word"; return; }
    previous="$word"
  done
  echo "$default"
}
# shellcheck disable=SC2086
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

unset PORT MASTER_ADDR MASTER_PORT NODE_RANK NNODES GPUS_PER_NODE
NODE="$ARNOLD_ID"
RUN="$SAMTOK_EXPERIMENT/runs/$SAMTOK_RUN_ID"
BOOTSTRAP="$RUN/bootstrap"
REPO="/tmp/samtok-edit-${SAMTOK_RUN_ID}-node${NODE}"
if [[ -e "$RUN/nodes/$NODE" || -e "$RUN/SUCCESS.json" ]]; then
  echo "Run already used: $RUN. Set a NEW common SAMTOK_RUN_ID; old logs are preserved." >&2
  exit 2
fi
PROBE="$SAMTOK_EXPERIMENT/runs/.write_probe_${SAMTOK_RUN_ID}_node${NODE}"
if ! (mkdir -p "$SAMTOK_EXPERIMENT/runs" && printf 'probe\n' > "$PROBE" && rm -f "$PROBE"); then
  echo "Cannot write under $SAMTOK_EXPERIMENT/runs (NAS quota exceeded?). Nothing was started." >&2
  exit 3
fi
mkdir -p "$BOOTSTRAP"
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
git clone --branch "$SAMTOK_EDIT_BRANCH" --single-branch "$SAMTOK_EDIT_REPO_URL" "$REPO"
if [[ -n "$SAMTOK_EDIT_COMMIT" ]]; then
  [[ "$SAMTOK_EDIT_COMMIT" =~ ^[a-fA-F0-9]{40}$ ]] || { echo 'Use a full commit SHA' >&2; false; }
  git -C "$REPO" checkout --detach "$SAMTOK_EDIT_COMMIT"
fi
git -C "$REPO" rev-parse HEAD > "$BOOTSTRAP/node${NODE}.commit.txt"
export SAMTOK_RUN_ROOT="$RUN" SAMTOK_NODE="$NODE"
bash "$REPO/scripts/round2/run_arnold.sh"
echo "[$(date -u +%FT%TZ)] node $NODE: done"
