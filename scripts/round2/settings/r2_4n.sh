#!/usr/bin/env bash
# v6 R2：从正式 v4 续训，用高合成目标教材把画质拉回来（只做这一个实验，跑完即结束）。
#
# 为什么走这条路（2026-11-09 17:1x 的结果促成）：
#  - v4 正是我们缺的那一半：我们自己 bench 上 UnitPass 0.835 / JointPass 0.569（从头训的最好只有
#    0.770 / 0.456），但 MIRA 画质 PQ 6.23，比原版 Qwen-Image-2.1 低 0.95。
#  - 从头训的方向今天被否了：s2b（全程高合成目标 + 全规模）@8000 画质与原版完全持平（+0.00），
#    但 bench 只有 0.736 / 0.331，比小批量的 s2hq@2000（0.770 / 0.456）还低——规模堆不出 bench。
#    观察到的规律是 bench 与贴回目标占比正相关（v4 79% → 0.569；31% → 0.400；9.7% → 0.33–0.46），
#    画质与之相反。所以改从 bench 最强的 v4 出发补画质。
#  - 11-06 的 R1 试过从 v4 续训但判定无效（PQ 6.04，比 v4 的 6.23 还低 0.19）。那次配方不同：
#    教材里仍有 15% 贴回目标（当时还不知道贴回是画质杀手，那是 11-08 才测出来的），
#    而且只训了 2,000 步 × 8 行 = 16,000 行，远不足以改变画质（主线用 8,000 步 × 32 行才把画质压下 0.9）。
#
# 本次设置：
#  - 初始化：正式 v4（qwen21_v3/runs/s2_4n_001/train/adapter_16000）
#  - 教材：rows_v6_s2hqfull.jsonl（与 s2b 同一份）——贴回目标只剩孪生行 9.7%，
#    10% 教读 token 行（覆盖高亮）、80% token 编辑、10% 普通编辑；7,000 步 × 32 行只用前 224,000 行
#  - 7,000 步 × 32 行（四机，约 7.8 小时）；lr 8e-6 余弦降到 8e-7、热身 100
#    学习率取 8e-6：11-10 的三档扫描（各 2,000 步 × 8 行）里 8e-6 的 JointPass 守得最好（0.519，仅差 v4 的
#    0.569 0.05；2e-5 和 5e-5 都掉到 0.481），同时画质提升最大（PQ +0.47 [+0.03,+0.92]，统计显著）。
#  - 损失与第二阶段完全相同：整图 FM + 编辑行区域项 1 + 教读 token 行区域项 0 + 干扰物项 1 + 路由写/读 0.05
#  - 检查点 1000/2000/3000/5000/7000，每个都评 MIRA 与 IF-Bench 子集——这一臂的风险正是
#    "画质上来了但 bench 掉下去"，两条曲线都要盯
#
# 判定：MIRA 的 PQ 不低于原版超过 0.3（即 ≥ 约 6.9），且 IF-Bench 的 UnitPass 不低于 v4 的 0.835 超过 0.03。
#   任一检查点同时满足，即为本轮最佳模型。
# 代码快照 22650e4；输出 runs_v5/v6_full/r2/train/adapter_{1000,2000,3000,5000,7000}。
# 用法（四机）：bash v6_r2.sh ；中断后重新提交只需换新的 V5_RUN_ID（如 V5_RUN_ID=r2_002 bash v6_r2.sh），
# 会从 resume.pt 继续，不会重头训。
set -Eeuo pipefail
exec 2>&1

# ===== 运行设置 =====
export V5_EXP=v6_full
export V5_RUN_ID="${V5_RUN_ID:-r2_001}"
export V5_STAGES=r2                           # this script runs R2 only
export V5_NNODES=4
export V5_ROOT=/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/quality_anchor_20261104
export V5_BUNDLE="${V5_BUNDLE:-/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/quality_anchor_20261104/code/quality-anchor-snapshot-22650e4.bundle}"
export V5_COMMIT="${V5_COMMIT:-22650e4b30830f95a11a3af8f4558326d98edb2f}"
ORIG_DATA=/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v3/data/round2_scaleup_20261030
V5_DATA="${V5_DATA:-$V5_ROOT/data_round2_v6p}"

stage_config() {  # stage -> DATA ROWS INIT ARGS
  local exp_dir=$1 stage=$2
  case "$stage" in
    r2)    # continue the formal v4 on the high-composite rows, at a lower learning rate
      DATA=$V5_DATA; ROWS=rows_v6_s2hqfull.jsonl
      INIT="/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v3/runs/s2_4n_001/train/adapter_16000"
      ARGS="--updates 7000 --warmup 100 --lr 8e-6 --lr-schedule cosine --lr-min 8e-7 --heavy-weight 8 --checkpoints 1000,2000,3000,5000,7000 --quick-eval 99999 --lambda-region 1 --lambda-region-hl 0 --lambda-distractor 1 --lambda-route 0.05 --lambda-read 0.05" ;;
    *) echo "Unknown stage $stage" >&2; return 2 ;;
  esac
  [[ -n "${V5_ARGS_OVERRIDE:-}" ]] && ARGS="$V5_ARGS_OVERRIDE"   # smoke tests only
  return 0
}

# ===== ARNOLD 与输入检查 =====
: "${ARNOLD_WORKER_HOSTS:?ARNOLD must inject the worker host list}"
: "${ARNOLD_WORKER_NUM:?ARNOLD must inject ARNOLD_WORKER_NUM}"
: "${ARNOLD_WORKER_GPU:?ARNOLD must inject ARNOLD_WORKER_GPU}"
: "${ARNOLD_ID:?ARNOLD must inject ARNOLD_ID for this worker}"
[[ "$ARNOLD_WORKER_NUM" == "$V5_NNODES" ]] || { echo "Expected $V5_NNODES workers, ARNOLD reports $ARNOLD_WORKER_NUM" >&2; exit 2; }
[[ "$ARNOLD_WORKER_GPU" =~ ^[1-9][0-9]*$ ]] || { echo 'ARNOLD_WORKER_GPU must be a positive integer' >&2; exit 2; }
[[ "$ARNOLD_ID" =~ ^[0-9]+$ && "$ARNOLD_ID" -lt "$V5_NNODES" ]] || { echo "ARNOLD_ID must be 0..$((V5_NNODES - 1))" >&2; exit 2; }
[[ "$V5_RUN_ID" =~ ^[a-zA-Z0-9_-]+$ && "$V5_EXP" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid V5_RUN_ID or V5_EXP' >&2; exit 2; }
[[ "$V5_COMMIT" =~ ^[a-f0-9]{40}$ ]] || { echo 'V5_COMMIT must be a full commit SHA' >&2; exit 2; }
[[ -f "$V5_BUNDLE" ]] || { echo "No code bundle $V5_BUNDLE" >&2; exit 2; }
WORLD=$((V5_NNODES * ARNOLD_WORKER_GPU))
ACC="${V5_ACC:-$((32 / WORLD))}"
[[ "${V5_ACC:-}" != "" || $((ACC * WORLD)) == 32 ]] || { echo "32 rows per update do not split over $WORLD GPUs" >&2; exit 2; }
EXP_DIR="$V5_ROOT/runs_v5/$V5_EXP"
for stage in $V5_STAGES; do
  stage_config "$EXP_DIR" "$stage"
  for needed in "$ROWS" eval_c1.json eval_c2_quick_idx.json edit_tokens.jsonl pool; do
    [[ -e "$DATA/$needed" ]] || { echo "Missing $DATA/$needed for stage $stage" >&2; exit 2; }
  done
done

unset PORT MASTER_ADDR MASTER_PORT NODE_RANK NNODES GPUS_PER_NODE
NODE="$ARNOLD_ID"
RUN="$EXP_DIR/submissions/$V5_RUN_ID"
BOOT="$RUN/bootstrap"
if [[ -e "$RUN/nodes/$NODE" ]]; then
  echo "Submission already used: $RUN. Set a NEW common V5_RUN_ID; old logs are preserved." >&2; exit 2
fi
PROBE="$EXP_DIR/.write_probe_${V5_RUN_ID}_node${NODE}"
if ! (mkdir -p "$EXP_DIR" && printf 'probe\n' > "$PROBE" && rm -f "$PROBE"); then
  echo "Cannot write under $EXP_DIR (NAS quota exceeded?). Nothing was started." >&2; exit 3
fi
mkdir -p "$BOOT"
if ! mkdir "$BOOT/node${NODE}.claimed"; then
  echo "Worker $NODE already started this submission. Use a NEW common V5_RUN_ID." >&2; exit 2
fi
mkdir -p "$RUN/nodes/$NODE" "$RUN/logs/node${NODE}"
exec > >(tee -a "$BOOT/node${NODE}.log") 2>&1
PHASE_FILE="$RUN/nodes/$NODE/phase"
phase() { echo "$1" > "$PHASE_FILE"; echo "[$(date -u +%FT%TZ)] node $NODE: $1"; }
failed() {
  local result="${1:-$?}"
  trap - ERR TERM INT
  printf '{"error":"failed during %s; see bootstrap/node%s.log","exit_code":%d}\n' \
    "$(cat "$PHASE_FILE" 2>/dev/null || echo unknown)" "$NODE" "$result" > "$RUN/nodes/$NODE/failure.json"
  exit "$result"
}
trap failed ERR
trap 'failed 143' TERM
trap 'failed 130' INT
phase checkout
echo "node $NODE of $V5_NNODES: experiment $V5_EXP, submission $V5_RUN_ID, stages: $V5_STAGES, world $WORLD, accumulation $ACC"

REPO="/tmp/v5-${V5_RUN_ID}-node${NODE}"
[[ ! -e "$REPO" ]] || { echo "Node-local checkout already exists: $REPO" >&2; false; }
git clone --branch quality-anchor --single-branch "$V5_BUNDLE" "$REPO"
git -C "$REPO" checkout --detach "$V5_COMMIT"
[[ "$(git -C "$REPO" rev-parse HEAD)" == "$V5_COMMIT" ]] || { echo 'checkout is not V5_COMMIT' >&2; false; }
git -C "$REPO" rev-parse HEAD > "$BOOT/node${NODE}.commit.txt"

phase environment
export SAMTOK_ENV="/tmp/samtok21-${V5_RUN_ID}-node${NODE}-env"
export SAMTOK_PYTHON="${SAMTOK_PYTHON:-/usr/bin/python3.11}"
export SAMTOK_CUDA_DIAGNOSTICS="$BOOT/node${NODE}-cuda"
export SAMTOK_EXPECTED_GPUS="$ARNOLD_WORKER_GPU"
export SAMTOK_CUDA_READY_TIMEOUT="${SAMTOK_CUDA_READY_TIMEOUT:-600}" SAMTOK_CUDA_READY_INTERVAL="${SAMTOK_CUDA_READY_INTERVAL:-15}"
bash "$REPO/scripts/round2/setup_env.sh"
export PYTHONPATH="$REPO/src:$REPO/third_party/diffsynth:$REPO/third_party${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" NCCL_DEBUG="${NCCL_DEBUG:-WARN}" TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/samtok21-inductor-$(id -u)}"
export WANDB_DISABLE_SERVICE=true WANDB_START_METHOD=thread WANDB_MODE=disabled
PYTHON="$SAMTOK_ENV/bin/python"
cd "$REPO"
first="${ARNOLD_WORKER_HOSTS%%,*}"
first="${first// /}"
if [[ "$first" == \[* ]]; then
  MASTER="${first#[}"; MASTER="${MASTER%%]*}"; MASTER_PORT_="${first##*]:}"
else
  MASTER="${first%:*}"; MASTER_PORT_="${first##*:}"
fi
[[ -n "$MASTER" && "$MASTER_PORT_" =~ ^[0-9]+$ ]] || { echo "Cannot parse ARNOLD_WORKER_HOSTS=$ARNOLD_WORKER_HOSTS" >&2; false; }
TORCHRUN=("$PYTHON" -m torch.distributed.run --nnodes "$V5_NNODES" --nproc_per_node "$ARNOLD_WORKER_GPU"
          --node_rank "$NODE" --master_addr "$MASTER" --master_port "$MASTER_PORT_" --max_restarts 0)

wait_for() {  # wait_for FILE SECONDS
  local deadline=$(( $(date +%s) + $2 ))
  while [[ ! -f "$1" ]]; do
    [[ $(date +%s) -lt $deadline ]] || { echo "Missing $1" >&2; return 1; }
    sleep 5
  done
}
run_watched() {  # stop as soon as another worker of this submission reports a failure
  "$@" &
  local pid=$! other
  while kill -0 "$pid" 2>/dev/null; do
    for other in "$RUN"/nodes/*/failure.json; do
      if [[ -e "$other" && "$other" != "$RUN/nodes/$NODE/failure.json" ]]; then
        echo "Another worker failed ($other); stopping this one." >&2
        kill -TERM "$pid" 2>/dev/null || true; sleep 10; kill -KILL "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
        return 1
      fi
    done
    sleep 15
  done
  wait "$pid"
}

for stage in $V5_STAGES; do
  stage_config "$EXP_DIR" "$stage"
  OUT="$EXP_DIR/$stage/train"
  if [[ -f "$OUT/TRAIN_DONE" ]]; then
    echo "[$(date -u +%FT%TZ)] node $NODE: stage $stage already finished, skipped"
    continue
  fi
  RESUME=()
  [[ -f "$OUT/resume.pt" ]] && RESUME=(--resume)
  INIT_ARGS=()
  if [[ -n "$INIT" ]]; then
    wait_for "$INIT/adapter.safetensors" 600
    INIT_ARGS=(--init-adapter "$INIT")
  fi
  mkdir -p "$OUT"
  phase "train $stage"
  # shellcheck disable=SC2086  # ARGS is a list of options
  run_watched "${TORCHRUN[@]}" --log_dir "$RUN/logs/node${NODE}/${stage}-ranks" --tee 3 -m samtok_edit21 round2-train \
    --data "$DATA" --output "$OUT" --rows "$ROWS" --quick-items eval_c2_quick_idx.json --accumulation "$ACC" \
    --wandb-mode disabled "${INIT_ARGS[@]}" "${RESUME[@]}" $ARGS
  wait_for "$OUT/TRAIN_DONE" 600
  echo "[$(date -u +%FT%TZ)] node $NODE: stage $stage done"
  sleep 20
done
if [[ "$NODE" == 0 ]]; then
  printf '{"time":"%s","experiment":"%s","submission":"%s","commit":"%s","stages":"%s"}\n' "$(date -u +%FT%TZ)" \
    "$V5_EXP" "$V5_RUN_ID" "$V5_COMMIT" "$V5_STAGES" > "$RUN/SUCCESS.json"
fi
phase done
