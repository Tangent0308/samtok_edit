#!/usr/bin/env bash
# 论文消融实验的四机入口（2026-11-12，v2）：四个 worker（4 × 8 卡）都 bash 这一个文件，一次串行跑完。
# v2 与 v1（abl_001，11-11 停止）的区别只有组件消融的 Read 阶段：250 次更新 -> 正式 S1 的前 1,000 次更新
# （同样的行、同样按 6,000 次的学习率日程，在第 1,000 次停下）。v1 的参照臂读 token 太弱（快速探针 0.155，S1@1000 为 0.31）。
# 输出写在 ablation_20261112/v2/（v1 的输出留在 ablation_20261112/，不覆盖）；数据与 200 张评测图共用。
#
# 内容（全部定义在 scripts/ablation/plan.py；数据与评测图在本地已备好：build_data.py、select_cases.py）：
#  1. 组件消融 8 臂（缩减协议：从原版起训，Read 1,000 + Edit 750 次更新 × 32 行，取正式 rows_s1 / rows_s2 的前面部分）：
#     完整配方、无目标识别、无注意力监督、无孪生、无区域均衡项、无 CodeInjector、无多目标、无近邻保持项
#  2. 开发集探针（1,320 项，判分无关）：原版、最终模型、8 臂
#  3. IF-Bench 生成（200 张图；Pass1 复用 R2 全量评测的记录）：最终模型、最终模型去掉 token（S1、S2-mask）、
#     最终模型 γ = 1、8 臂（S1、S2-mask、S3-mask、S4-noref）
#  4. 数据规模 2 臂：从 S1@6000 起，25% / 50% 记录，S2 配方、相同 epoch 数（4,000 / 8,000 次更新）
#  5. 规模三点（含 100% = S2@16000）的探针与 IF-Bench 生成（十个 setting）
#  6. 所有系统一次判分（scripts/ablation/judge_many.py，每台机器一个 Judge 引擎），然后出表 REPORT/ABLATION.md
# 预计（按 S1/S2 与 v1 的实测速度，4.24 s 一次更新）：组件训练约 18.5 h，规模训练约 14.6 h，探针约 1.5 h，生成约 4.5 h，
# 判分约 3.5 h，共约 43 h。
#
# 中断后重新提交：只把 ABL_RUN_ID 换成新的值（四个 worker 相同）。已完成的步骤按结果文件跳过
# （TRAIN_DONE、EVAL_DONE、state/*.done），训练从最近的检查点续训，生成从未完成的请求继续。
set -Eeuo pipefail
exec 2>&1

# ===== 运行设置（重新提交时只改 ABL_RUN_ID）=====
export ABL_RUN_ID="${ABL_RUN_ID:-abl_v2_001}"
export ABL_ROOT_DEFAULT=/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/ablation_20261112
export ABL_COMMIT="${ABL_COMMIT:-__ABL_COMMIT__}"
export ABL_BUNDLE="${ABL_BUNDLE:-$ABL_ROOT_DEFAULT/code/ablation-__ABL_COMMIT__.bundle}"
export ABL_BRANCH=qwen-image-2.1-r2-ablation
export ABL_NNODES="${ABL_NNODES:-4}"
export IFBENCH_BUNDLE=/mnt/bn/strategy-mllm-train/user/tanyue/.ifbench-tmp/one-mp-full4-20261103-judgeio
# 只用于本机预演（scripts/ablation/rehearse.sh），ARNOLD 上不要设置：
#   ABL_ROOT ABL_DATA ABL_CASES ABL_SMOKE=1 ABL_ENV（已有环境，跳过建环境） ABL_JUDGE_PY（已有 Judge 运行时）
#   ABL_JUDGE_DRYRUN=1（不加载 Judge 引擎） ABL_REPO（直接用本地 checkout）
export ABL_ROOT="${ABL_ROOT:-$ABL_ROOT_DEFAULT/v2}"     # outputs of v2; code bundles stay in $ABL_ROOT_DEFAULT/code
# 两个四机作业并行（11-12）：component = 8 个组件臂（训练、探针、生成、判分）；scale = 数据规模 2 臂 + 最终模型诊断 +
# 原版 / 最终模型探针。各自判自己的系统，块内比较不跨判分会话。all = 一个作业跑全部。
export ABL_BLOCK="${ABL_BLOCK:-all}"

# ===== ARNOLD 与输入检查 =====
: "${ARNOLD_WORKER_HOSTS:?ARNOLD must inject the worker host list}"
: "${ARNOLD_WORKER_NUM:?ARNOLD must inject ARNOLD_WORKER_NUM}"
: "${ARNOLD_WORKER_GPU:?ARNOLD must inject ARNOLD_WORKER_GPU}"
: "${ARNOLD_ID:?ARNOLD must inject ARNOLD_ID for this worker}"
[[ "$ARNOLD_WORKER_NUM" == "$ABL_NNODES" ]] || { echo "Expected $ABL_NNODES workers, ARNOLD reports $ARNOLD_WORKER_NUM" >&2; exit 2; }
[[ "$ARNOLD_WORKER_GPU" =~ ^[1-9][0-9]*$ ]] || { echo 'ARNOLD_WORKER_GPU must be a positive integer' >&2; exit 2; }
[[ "$ARNOLD_ID" =~ ^[0-9]+$ && "$ARNOLD_ID" -lt "$ABL_NNODES" ]] || { echo "ARNOLD_ID must be 0..$((ABL_NNODES - 1))" >&2; exit 2; }
[[ "$ABL_RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,40}$ ]] || { echo 'Invalid ABL_RUN_ID' >&2; exit 2; }
[[ -n "${ABL_REPO:-}" || "$ABL_COMMIT" =~ ^[a-f0-9]{40}$ ]] || { echo 'ABL_COMMIT must be a full commit SHA' >&2; exit 2; }
[[ -n "${ABL_REPO:-}" || -f "$ABL_BUNDLE" ]] || { echo "No code bundle $ABL_BUNDLE" >&2; exit 2; }
NODE=$ARNOLD_ID
NODES=$ABL_NNODES
WORLD=$((NODES * ARNOLD_WORKER_GPU))
((32 % WORLD == 0)) || { echo "32 rows per update do not split over $WORLD GPUs" >&2; exit 2; }
unset PORT MASTER_ADDR MASTER_PORT RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE \
      NODE_RANK NNODES GPUS_PER_NODE

SUB=$ABL_ROOT/submissions/$ABL_RUN_ID
STATE=$ABL_ROOT/state                 # finished steps, shared by all submissions
LOGS=$SUB/logs/node$NODE
PROBE_FILE=$ABL_ROOT/.write_probe_${ABL_RUN_ID}_node$NODE
if ! (mkdir -p "$ABL_ROOT" && printf 'probe\n' > "$PROBE_FILE" && rm -f "$PROBE_FILE"); then
  echo "Cannot write under $ABL_ROOT (NAS quota exceeded?). Nothing was started." >&2; exit 3
fi
mkdir -p "$SUB/failures" "$SUB/markers" "$LOGS" "$STATE"
if ! mkdir "$SUB/node$NODE.claimed"; then
  echo "Worker $NODE already started submission $ABL_RUN_ID. Use a NEW common ABL_RUN_ID; old logs are kept." >&2; exit 2
fi
exec > >(tee -a "$SUB/node$NODE.log") 2>&1
PHASE_FILE=$SUB/node$NODE.phase
phase() { echo "$1" > "$PHASE_FILE"; echo "[$(date -u +%FT%TZ)] node $NODE: $1"; }
failed() {
  local result="${1:-$?}"
  trap - ERR TERM INT
  printf '{"error":"failed during %s; see node%s.log","exit_code":%d,"time":"%s"}\n' \
    "$(cat "$PHASE_FILE" 2>/dev/null || echo unknown)" "$NODE" "$result" "$(date -u +%FT%TZ)" > "$SUB/failures/node$NODE.json"
  exit "$result"
}
trap failed ERR
trap 'failed 143' TERM
trap 'failed 130' INT
echo "[$(date -u +%FT%TZ)] node $NODE of $NODES: submission $ABL_RUN_ID, block $ABL_BLOCK, world $WORLD, root $ABL_ROOT"

# ===== 代码与环境 =====
phase checkout
WORK=/tmp/abl-$ABL_RUN_ID-node$NODE
if [[ -n "${ABL_REPO:-}" ]]; then
  REPO=$ABL_REPO
else
  REPO=$WORK/repo
  [[ ! -e "$REPO" ]] || { echo "Node-local checkout already exists: $REPO" >&2; false; }
  mkdir -p "$WORK"
  git clone --branch "$ABL_BRANCH" --single-branch "$ABL_BUNDLE" "$REPO"
  git -C "$REPO" checkout --detach "$ABL_COMMIT"
  [[ "$(git -C "$REPO" rev-parse HEAD)" == "$ABL_COMMIT" ]] || { echo 'checkout is not ABL_COMMIT' >&2; false; }
fi
git -C "$REPO" rev-parse HEAD > "$SUB/node$NODE.commit.txt"
mkdir -p "$WORK"

phase environment
export SAMTOK_PYTHON="${SAMTOK_PYTHON:-/usr/bin/python3.11}"
if [[ -n "${ABL_ENV:-}" ]]; then
  export SAMTOK_ENV=$ABL_ENV
else
  export SAMTOK_ENV=$WORK/env_v4
  export SAMTOK_CUDA_DIAGNOSTICS=$SUB/node$NODE-cuda SAMTOK_EXPECTED_GPUS=$ARNOLD_WORKER_GPU
  export SAMTOK_CUDA_READY_TIMEOUT="${SAMTOK_CUDA_READY_TIMEOUT:-600}" SAMTOK_CUDA_READY_INTERVAL="${SAMTOK_CUDA_READY_INTERVAL:-15}"
  env -u CUDA_VISIBLE_DEVICES bash "$REPO/scripts/round2/setup_env.sh"
fi
PY=$SAMTOK_ENV/bin/python
if [[ -n "${ABL_JUDGE_PY:-}" ]]; then
  JUDGE_PY=$ABL_JUDGE_PY
else
  "$SAMTOK_PYTHON" -B "$REPO/scripts/ifbench/prepare_judge_runtime.py" --bundle "$IFBENCH_BUNDLE" --work "$WORK/runtime"
  JUDGE_PY=$WORK/runtime/judge/python
fi
# Training, probes and the plan see the method only; IF-Bench generation also the benchmark snapshot; the Judge neither.
export PYTHONPATH="$REPO/src:$REPO/third_party/diffsynth:$REPO/third_party"
BENCH_PATH="$PYTHONPATH:$IFBENCH_BUNDLE/repo/src"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" NCCL_DEBUG="${NCCL_DEBUG:-WARN}" TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export TRITON_CACHE_DIR=$WORK/kernels/triton TORCHINDUCTOR_CACHE_DIR=$WORK/kernels/inductor
export TORCH_EXTENSIONS_DIR=$WORK/kernels/extensions VLLM_CACHE_ROOT=$WORK/kernels/vllm
export WANDB_DISABLE_SERVICE=true WANDB_MODE=disabled
PLAN=("$PY" -B "$REPO/scripts/ablation/plan.py")
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=, read -ra CARDS <<< "$CUDA_VISIBLE_DEVICES"
else
  mapfile -t CARDS < <(seq 0 $((ARNOLD_WORKER_GPU - 1)))
fi
[[ ${#CARDS[@]} == "$ARNOLD_WORKER_GPU" ]] || { echo "Visible cards ${CARDS[*]} != ARNOLD_WORKER_GPU" >&2; false; }

phase inputs
"${PLAN[@]}" check-inputs

first="${ARNOLD_WORKER_HOSTS%%,*}"
first="${first// /}"
if [[ "$first" == \[* ]]; then
  MASTER="${first#[}"; MASTER="${MASTER%%]*}"; MASTER_PORT_="${first##*]:}"
else
  MASTER="${first%:*}"; MASTER_PORT_="${first##*:}"
fi
[[ -n "$MASTER" && "$MASTER_PORT_" =~ ^[0-9]+$ ]] || { echo "Cannot parse ARNOLD_WORKER_HOSTS=$ARNOLD_WORKER_HOSTS" >&2; false; }
TORCHRUN=("$PY" -m torch.distributed.run --nnodes "$NODES" --nproc_per_node "$ARNOLD_WORKER_GPU"
          --node_rank "$NODE" --master_addr "$MASTER" --master_port "$MASTER_PORT_" --max_restarts 0)

# ===== 同步与监控 =====
another_failed() { compgen -G "$SUB/failures/*.json" > /dev/null; }
mark() { mkdir -p "$SUB/markers/$1"; date -u +%FT%TZ > "$SUB/markers/$1/node$NODE"; }
barrier() {  # barrier NAME [COUNT]: wait for the markers of the first COUNT machines; stop on a failure elsewhere
  local name=$1 count=${2:-$NODES} deadline=$((SECONDS + 259200)) n missing
  while :; do
    if another_failed; then echo "Another worker failed; stopping at $name." >&2; return 1; fi
    missing=0
    for ((n = 0; n < count; n++)); do [[ -e "$SUB/markers/$name/node$n" ]] || missing=1; done
    ((missing == 0)) && return 0
    ((SECONDS < deadline)) || { echo "Timed out waiting for all workers at $name." >&2; return 1; }
    sleep 10
  done
}
wait_for() {  # wait_for FILE SECONDS
  local deadline=$(( $(date +%s) + $2 ))
  while [[ ! -e "$1" ]]; do
    [[ $(date +%s) -lt $deadline ]] || { echo "Missing $1" >&2; return 1; }
    sleep 5
  done
}
stop_processes() {
  kill -TERM "$@" 2>/dev/null || true
  sleep 20
  kill -KILL "$@" 2>/dev/null || true
  wait "$@" 2>/dev/null || true
}
watch_processes() {  # wait for background processes; stop them all when one fails or another machine reports a failure
  local name=$1 pid
  shift
  local remaining=("$@") next
  while ((${#remaining[@]})); do
    next=()
    for pid in "${remaining[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        next+=("$pid")
      elif ! wait "$pid"; then
        echo "$name: a process failed, see $LOGS" >&2
        stop_processes "$@"
        return 1
      fi
    done
    remaining=("${next[@]}")
    if another_failed; then
      echo "$name: another worker failed; stopping this one." >&2
      stop_processes "$@"
      return 1
    fi
    if ((${#remaining[@]})); then sleep 15; fi
  done
}
cards_free() {  # every card of this machine has at least 70 GiB free (our earlier processes have exited)
  local deadline=$((SECONDS + 900)) free
  while :; do
    free=$(nvidia-smi --id="$(IFS=,; echo "${CARDS[*]}")" --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
    ((free >= 70 * 1024)) && return 0
    ((SECONDS < deadline)) || { echo "Cards still busy (lowest free ${free} MiB)." >&2; return 1; }
    sleep 10
  done
}

# ===== 步骤 =====
torchrun_step() {  # torchrun_step NAME DONE_FILE MODULE_COMMAND ARGS...: one torchrun over all machines
  local name=$1 done_file=$2
  shift 2
  phase "$name"
  cards_free
  "${TORCHRUN[@]}" --log_dir "$LOGS/$name-ranks" --tee 3 -m samtok_edit21 "$@" &
  watch_processes "$name" $!
  wait_for "$done_file" 600
  echo "[$(date -u +%FT%TZ)] node $NODE: $name done"
  sleep 20                            # every machine leaves the rendezvous before the next one starts
}

train_step() {  # train_step ARM STAGE (stage: read | edit | scale)
  local arm=$1 stage=$2 out line args resume=()
  out=$("${PLAN[@]}" train-dir "$arm" "$stage")
  if [[ -f "$out/TRAIN_DONE" ]]; then echo "[$(date -u +%FT%TZ)] node $NODE: train $arm $stage already done"; return 0; fi
  line=$("${PLAN[@]}" train-args "$arm" "$stage" --world "$WORLD")
  read -ra args <<< "$line"
  [[ -f "$out/resume.pt" ]] && resume=(--resume)
  mkdir -p "$out"
  torchrun_step "train-$arm-$stage" "$out/TRAIN_DONE" round2-train "${args[@]}" "${resume[@]}"
}

probe_step() {  # probe_step SYSTEM: the development-set probes of one system
  local name=$1 line args
  if [[ -f "$ABL_ROOT/probe/$name/EVAL_DONE" ]]; then echo "[$(date -u +%FT%TZ)] node $NODE: probe $name already done"; return 0; fi
  line=$("${PLAN[@]}" probe-args "$name")
  read -ra args <<< "$line"
  torchrun_step "probe-$name" "$ABL_ROOT/probe/$name/EVAL_DONE" round2-eval "${args[@]}"
}

generate_step() {  # generate_step SYSTEM: IF-Bench Pass2 of one system on every card of every machine
  local name=$1 run=$ABL_ROOT/ifbench/$1 pids=() i
  if [[ -f "$STATE/generated_$name.done" ]]; then echo "[$(date -u +%FT%TZ)] node $NODE: generation $name already done"; return 0; fi
  phase "generate $name"
  if [[ "$NODE" == 0 ]]; then
    PYTHONPATH=$BENCH_PATH "${PLAN[@]}" prepare-run "$name" --repo "$REPO" | tee -a "$LOGS/prepare.log"
    mark "prepared-$name"
  fi
  barrier "prepared-$name" 1
  cards_free
  for i in "${!CARDS[@]}"; do
    CUDA_VISIBLE_DEVICES=${CARDS[$i]} PYTHONPATH=$BENCH_PATH "$PY" -B -m samtok_edit21.evaluation.ifbench pass2 --run "$run" --node "$NODE" \
      --nodes "$NODES" --worker "$i" --workers "${#CARDS[@]}" >> "$LOGS/pass2-$name.$i.log" 2>&1 &
    pids+=($!)
  done
  watch_processes "generate $name" "${pids[@]}"
  mark "generated-$name"
  barrier "generated-$name"
  if [[ "$NODE" == 0 ]]; then
    PYTHONPATH=$BENCH_PATH "$PY" -B - "$run" "$STATE/generated_$name.done" <<'PY'
import json, sys, time
from pathlib import Path
from samtok_edit21.evaluation import ifbench
run, done = Path(sys.argv[1]), Path(sys.argv[2])
state = ifbench.status(run)
open_requests = ifbench.open_claims(run)
if open_requests or sum(state["pass2"].values()) != state["requests"] or state["pass2"].get("pending"):
    raise SystemExit(f"Generation is not terminal everywhere: {state}, open {open_requests[:10]}")
done.write_text(json.dumps({"time": time.time(), "status": state}))
print(json.dumps({"generated": run.name, **state}))
PY
    mark "checked-$name"
  fi
  barrier "checked-$name" 1
}

judge_step() {  # every system in one Judge session: one engine per machine for all runs
  local line runs
  if [[ -f "$STATE/judged_$ABL_BLOCK.done" ]]; then echo "[$(date -u +%FT%TZ)] node $NODE: Judge already done"; return 0; fi
  line=$("${PLAN[@]}" runs)
  [[ -n "$line" ]] || { echo "No IF-Bench system in block $ABL_BLOCK" >&2; return 1; }
  read -ra runs <<< "$line"
  phase judge
  cards_free
  if [[ "${ABL_JUDGE_DRYRUN:-0}" == 1 ]]; then
    (cd / && env -u PYTHONPATH "$JUDGE_PY" "$REPO/scripts/ablation/judge_many.py" --runs "${runs[@]}" --node "$NODE" \
      --nodes "$NODES" --dry-run >> "$LOGS/judge.log" 2>&1) &
  else
    (cd / && CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${CARDS[*]}") env -u PYTHONPATH "$JUDGE_PY" \
      "$REPO/scripts/ablation/judge_many.py" --runs "${runs[@]}" --node "$NODE" --nodes "$NODES" >> "$LOGS/judge.log" 2>&1) &
  fi
  watch_processes judge $!
  mark "judged-$ABL_BLOCK"
  barrier "judged-$ABL_BLOCK"
  if [[ "$NODE" == 0 ]]; then
    phase report
    sleep 30                          # the other machines' last records on the shared disk
    (cd / && env -u PYTHONPATH "$JUDGE_PY" "$REPO/scripts/ablation/judge_many.py" --runs "${runs[@]}" --report-only) \
      | tee -a "$LOGS/report.log"
    "$PY" -B "$REPO/scripts/ablation/report.py" | tee -a "$LOGS/report.log"
    date -u +%FT%TZ > "$STATE/judged_$ABL_BLOCK.done"
    mark "reported-$ABL_BLOCK"
  fi
  barrier "reported-$ABL_BLOCK" 1
}

# ===== 串行执行 =====
listed() { local line; line=$("${PLAN[@]}" list "$1"); LISTED=(); [[ -z "$line" ]] || read -ra LISTED <<< "$line"; }
listed train-arms; ARMS=("${LISTED[@]}")
listed scale-arms; SCALES=("${LISTED[@]}")
listed probe-first; PROBES=("${LISTED[@]}")
listed ifbench-first; FIRST=("${LISTED[@]}")
listed probe-scale; LATER=("${LISTED[@]}")
for arm in "${ARMS[@]}"; do
  train_step "$arm" read
  train_step "$arm" edit
done
for name in "${PROBES[@]}"; do probe_step "$name"; done
for name in "${FIRST[@]}"; do generate_step "$name"; done
for arm in "${SCALES[@]}"; do train_step "$arm" scale; done
for name in "${LATER[@]}"; do probe_step "$name"; done
for name in "${LATER[@]}"; do generate_step "$name"; done
judge_step

if [[ "$NODE" == 0 ]]; then
  printf '{"time":"%s","submission":"%s","block":"%s","commit":"%s","report":"%s"}\n' "$(date -u +%FT%TZ)" "$ABL_RUN_ID" \
    "$ABL_BLOCK" "$(git -C "$REPO" rev-parse HEAD)" "$ABL_ROOT/REPORT/ABLATION.md" > "$SUB/SUCCESS.json"
fi
phase done
[[ -n "${ABL_REPO:-}" ]] || rm -rf "$WORK"
