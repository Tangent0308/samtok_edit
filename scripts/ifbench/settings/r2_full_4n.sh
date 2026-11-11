#!/usr/bin/env bash
# IF-Bench 全量 R2@1000（我方方法仅，七基线已完成）——整段原样粘贴成四个 worker（4 × 8 卡）的启动命令。
# 重新提交时只改 SAMTOK_ATTEMPT。七基线不重跑（SAMTOK_IFBENCH_CHAIN=0）。
# 结果写入 SAMTOK_IFBENCH_RUN；与 v4 正式结果目录独立。
set -Eeuo pipefail
exec 2>&1

# ===== 运行设置（重新提交时只改 SAMTOK_ATTEMPT）=====
export SAMTOK_ATTEMPT=r2_run001       # 每次提交换新的，四个 worker 相同
export SAMTOK_EDIT_COMMIT=93053f1b4455c4c8c1d31f33c65e31a1ffadd0da   # v5 HEAD，四个 worker 必须相同
export SAMTOK_EDIT_BRANCH=qwen-image-2.1-v5
export IFBENCH_BUNDLE=/mnt/bn/strategy-mllm-train/user/tanyue/.ifbench-tmp/one-mp-full4-20261103-judgeio
export SAMTOK_IFBENCH_RUN=/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/ifbench_r2_full_20261110
export SAMTOK_IFBENCH_CHAIN=0         # 只跑我方，七基线已完成
export SAMTOK_IFBENCH_JUDGE=1         # 本机同时跑 Judge
export SAMTOK_IFBENCH_EXPECT=10840    # 1,084 张图 × 10 个 setting
export SAMTOK_NNODES=4
export SAMTOK_EDIT_REPO_URL=https://github.com/Tangent0308/samtok_edit.git

# ===== 以下为 scripts/ifbench/bootstrap_arnold.sh 全文（不用改）=====
: "${ARNOLD_WORKER_NUM:?ARNOLD must inject ARNOLD_WORKER_NUM}"
: "${ARNOLD_WORKER_GPU:?ARNOLD must inject ARNOLD_WORKER_GPU}"
: "${ARNOLD_ID:?ARNOLD must inject ARNOLD_ID for this worker}"
[[ "$ARNOLD_WORKER_NUM" == "$SAMTOK_NNODES" ]] || { echo "Expected $SAMTOK_NNODES workers, ARNOLD reports $ARNOLD_WORKER_NUM" >&2; exit 2; }
[[ "$ARNOLD_ID" =~ ^[0-9]+$ && "$ARNOLD_ID" -lt "$SAMTOK_NNODES" ]] || { echo "ARNOLD_ID must be 0..$((SAMTOK_NNODES - 1))" >&2; exit 2; }
[[ "$SAMTOK_ATTEMPT" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,50}$ ]] || { echo 'Invalid SAMTOK_ATTEMPT' >&2; exit 2; }
[[ "$SAMTOK_EDIT_COMMIT" =~ ^[a-fA-F0-9]{40}$ ]] || { echo 'SAMTOK_EDIT_COMMIT must be a full commit SHA' >&2; exit 2; }
if [[ "$SAMTOK_IFBENCH_CHAIN" == 1 ]]; then
  [[ "$ARNOLD_WORKER_NUM" == 4 && "$ARNOLD_WORKER_GPU" == 8 ]] || { echo 'The benchmark run needs 4 workers x 8 GPUs' >&2; exit 2; }
  [[ -f "$IFBENCH_BUNDLE/READY.json" ]] || { echo "Bundle is not published: $IFBENCH_BUNDLE" >&2; exit 2; }
fi
for needed in configuration.json bootstrap.json repo/src assets/shared_inputs/requests; do
  [[ -e "$IFBENCH_BUNDLE/$needed" ]] || { echo "Missing $needed in $IFBENCH_BUNDLE" >&2; exit 2; }
done

NODE="$ARNOLD_ID"
export SAMTOK_NODE="$NODE"
ATTEMPT="$SAMTOK_IFBENCH_RUN/attempts/$SAMTOK_ATTEMPT"
REPO="/tmp/samtok-edit-ifbench-$SAMTOK_ATTEMPT-node$NODE"
mkdir -p "$ATTEMPT/failures"
if ! mkdir "$ATTEMPT/node$NODE.claimed"; then
  echo "Worker $NODE already started attempt $SAMTOK_ATTEMPT. Use a NEW common SAMTOK_ATTEMPT." >&2
  exit 2
fi
exec > >(tee -a "$ATTEMPT/node$NODE.log") 2>&1
bootstrap_failed() {
  local result="${1:-$?}"
  trap - ERR TERM INT
  printf '{"error":"failed during %s; see node%s.log","exit_code":%d}\n' \
    "$(cat "$ATTEMPT/node$NODE.phase" 2>/dev/null || echo unknown)" "$NODE" "$result" > "$ATTEMPT/failures/node$NODE.json"
  exit "$result"
}
trap bootstrap_failed ERR
trap 'bootstrap_failed 143' TERM
trap 'bootstrap_failed 130' INT
echo "[$(date -u +%FT%TZ)] node $NODE of $SAMTOK_NNODES: start, attempt $SAMTOK_ATTEMPT"

if [[ "$SAMTOK_IFBENCH_CHAIN" == 1 ]]; then
  echo benchmark-preflight > "$ATTEMPT/node$NODE.phase"
  bash "$IFBENCH_BUNDLE/start.sh" --attempt "$SAMTOK_ATTEMPT-validate" --validate-only
fi
if [[ -f "$SAMTOK_IFBENCH_RUN/SUCCESS.json" ]]; then
  echo "Our method already finished: $SAMTOK_IFBENCH_RUN/SUCCESS.json"
elif [[ "$SAMTOK_IFBENCH_JUDGE" == 0 && -f "$SAMTOK_IFBENCH_RUN/GENERATED.json" ]]; then
  echo "Our generation already finished: $SAMTOK_IFBENCH_RUN/GENERATED.json"
else
  echo checkout > "$ATTEMPT/node$NODE.phase"
  export GIT_TERMINAL_PROMPT=0
  git clone --branch "$SAMTOK_EDIT_BRANCH" --single-branch "$SAMTOK_EDIT_REPO_URL" "$REPO"
  git -C "$REPO" checkout --detach "$SAMTOK_EDIT_COMMIT"
  git -C "$REPO" rev-parse HEAD > "$ATTEMPT/node$NODE.commit.txt"
  bash "$REPO/scripts/ifbench/run_arnold.sh"
  rm -rf "$REPO"
fi
if [[ "$SAMTOK_IFBENCH_CHAIN" == 1 ]]; then
  echo benchmark-seven-models > "$ATTEMPT/node$NODE.phase"
  trap - ERR TERM INT
  exec bash "$IFBENCH_BUNDLE/start.sh" --attempt "$SAMTOK_ATTEMPT"
fi
echo "[$(date -u +%FT%TZ)] node $NODE: done"
