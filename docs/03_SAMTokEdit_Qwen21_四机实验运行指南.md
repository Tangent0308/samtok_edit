# SAMTokEdit Qwen-Image-2.1 四机实验运行指南

此页是当前唯一的四机启动入口。下面每一节的 bash 块均可完整粘贴到 ARNOLD worker 启动命令；无需预先 clone，不依赖共享盘上的代码。四个 worker 执行同一段脚本。共享盘只保存数据、日志和产物；代码通过 GitHub 分支 clone 到各机 `/tmp`。

## 1. ARNOLD、环境与参数约定

ARNOLD 作业配置为 4 workers × 8 GPUs。平台向每个 worker 注入 `ARNOLD_WORKER_HOSTS`、`ARNOLD_WORKER_NUM=4`、`ARNOLD_WORKER_GPU=8`、`ARNOLD_ID=0..3`；不要把四台机器的 ARNOLD_ID 手填成同一个值。共同 rendezvous 端口来自 `ARNOLD_WORKER_HOSTS`，清除各 worker 可能不同的通用 PORT。训练需要 W&B key，推荐注入 secret，也可在命令占位处填写；noref 转换不使用 W&B。

训练环境按 `requirements.txt`、`requirements-cluster.txt` 锁定 torch 2.8.0；9B 转换按 `requirements-annotation-qwen35-lock.txt` 锁定 torch 2.10.0/vLLM 0.17.1。两者使用各自的节点本地 venv。真实 key 不写入 Git 或实验 manifest。节点执行一致性检查，commit、源码、数据和参数不一致时退出；需要固定版本时可在四台机器统一设置 `SAMTOK_EDIT_COMMIT` 为已推送的完整 40 位 SHA。

每次提交使用新的共同 run ID。节点启动 claim 防止调度器重试覆盖旧日志。脚本中的 bootstrap 与仓库 `scripts/train/bootstrap_arnold_*.sh` 保持一致，clone 后调用 `run_arnold_*.sh` 安装环境并执行管线。

## 2. 正式全量训练入口

已准备数据：`/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/qwen21_full4_20260928/data/train_full_9b_rules_003`。98,574 个源编辑对，Stage 1 为 390,657 行，Stage 2 为 293,296 行，详见[数据盘点](04_SAMTokEdit_Qwen21_训练数据盘点.md)。下面使用修复后新 run `_002`；旧 `_001` 的启动停滞记录见[实验记录第 17 节](02_SAMTokEdit_Qwen21_实验记录.md#17-2026-09-30正式全量-_001-启动停滞与预检修复)。新命令尚未完成正式四机运行。

| 参数 | Stage 1 | Stage 2 |
|---|---|---|
| 更新对象 | TE language LoRA | DiT LoRA |
| rank/dropout | 64 / 0.05 | 32 / 0 |
| optimizer updates | 3081 | 3081 |
| accumulation / global batch | 8 / 256 | 4 / 128 |
| LR / weight decay | 4e-5 / 0.05 | 1e-4 / 0.01 |
| LR schedule / warmup ratio | cosine / 0.04 | constant / 0.025 |
| NTP/FM weight | 0.05 / 1 | FM=1 |
| C / A | 0.5 / 0 | 0.5 / 0.1 |
| A warmup | 不启用 | 500 updates |
| 保存间隔 | 2000 microsteps（250 updates） | 2000 microsteps（500 updates） |

`3081` 是近似一轮调度长度；按类型池加权有放回采样，不保证每行逐次覆盖。分支比例、曝光率保存在 schedule/schedule_report 中。A=0.1 是当前可运行配置，不是全量效果最优系数，后续可做校准/消融。正式入口将单阶段及 barrier 超时设为 604800 秒（7 天），避免全量 cache/训练被 debug 的 7200 秒截断；GPU/NCCL 错误仍通过 worker 失败及时退出。


`--full-training` 自动让 Stage 1 复用同目录 `metadata_report.json`：全局 rank 0 核对报告就绪状态、metadata/region manifest hash、分辨率与计数，再广播结果。远程启动不再对全量图片和 coverage 逐条预扫描；正常读取 metadata、构造 schedule 和训练消费时的文件验证保留。Stage 1 之后新生成的 conditioning cache 仍需验证。详细实现见[代码说明第 13 节](01_SAMTokEdit_Qwen21_代码实现说明.md#13-正式训练复用已准备数据的验收报告)。无需重新打标或重建当前数据。

已运行的 `_001` 不会自动加载新代码。停止旧作业后，四个 worker 使用下面同一个新 ID 提交，确保 clone 到本次修复后的分支；如果设置了旧 `SAMTOK_EDIT_COMMIT`，须清除或改为此次修复的已推送 SHA。请保留 `_001` 的共享日志，不复用旧节点目录。

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
export SAMTOK_EDIT_REPO_URL=https://github.com/Tangent0308/samtok_edit.git
export SAMTOK_EDIT_BRANCH=qwen-image-2.1-dev
# 如需固定版本，可在四个 worker 上设置同一个已推送的完整 40 位 SAMTOK_EDIT_COMMIT。
export WANDB_ENTITY=2200012743-peking-university
export WANDB_PROJECT=samtok-edit
# 可直接替换下面的占位值；若 ARNOLD 已注入 secret，则沿用 secret。
export WANDB_API_KEY="${WANDB_API_KEY:-FILL_IN_WANDB_API_KEY}"
export SAMTOK_EXPERIMENT=/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/qwen21_full4_20260928
export SAMTOK_TRAIN_DATA="$SAMTOK_EXPERIMENT/data/train_full_9b_rules_003"
export SAMTOK_RUN_ID=qwen21_full_4n_formal_002


# ----- User settings -----
export SAMTOK_EXPERIMENT="${SAMTOK_EXPERIMENT:?Set the shared experiment directory}"
# Set this explicitly and identically on all workers; use a fresh ID per attempt.
: "${SAMTOK_RUN_ID:?Set one fresh common SAMTOK_RUN_ID on all workers}"
# Inject WANDB_API_KEY as an ARNOLD secret on every worker.
: "${WANDB_API_KEY:?Inject WANDB_API_KEY through ARNOLD secrets}"
export WANDB_ENTITY="${WANDB_ENTITY:-2200012743-peking-university}"
export WANDB_PROJECT="${WANDB_PROJECT:-samtok-edit}"
export SAMTOK_EDIT_REPO_URL="${SAMTOK_EDIT_REPO_URL:-https://github.com/Tangent0308/samtok_edit.git}"
export SAMTOK_EDIT_BRANCH="${SAMTOK_EDIT_BRANCH:-qwen-image-2.1-dev}"

# ----- ARNOLD checks -----
: "${ARNOLD_WORKER_HOSTS:?ARNOLD must inject the four-worker host list}"
: "${ARNOLD_WORKER_NUM:?ARNOLD must inject ARNOLD_WORKER_NUM=4}"
: "${ARNOLD_WORKER_GPU:?ARNOLD must inject ARNOLD_WORKER_GPU=8}"
: "${ARNOLD_ID:?ARNOLD must inject ARNOLD_ID for this worker (0..3)}"
[[ "$ARNOLD_WORKER_NUM" == 4 && "$ARNOLD_WORKER_GPU" == 8 ]] || { echo 'Expected 4 workers x 8 GPUs' >&2; exit 2; }
[[ "$ARNOLD_ID" =~ ^[0-3]$ ]] || { echo 'ARNOLD_ID must be 0, 1, 2, or 3' >&2; exit 2; }
[[ "$SAMTOK_RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid SAMTOK_RUN_ID' >&2; exit 2; }
[[ -n "$WANDB_API_KEY" && "$WANDB_API_KEY" != FILL_IN* ]] || { echo 'Set WANDB_API_KEY as an ARNOLD secret' >&2; exit 2; }

# ARNOLD_WORKER_HOSTS carries the common rendezvous port; generic PORT varies by worker.
unset PORT MASTER_ADDR MASTER_PORT NODE_RANK NNODES GPUS_PER_NODE
export NODE_RANK="$ARNOLD_ID"
export ARNOLD_WORKER_NUM=4 ARNOLD_WORKER_GPU=8

RUN="$SAMTOK_EXPERIMENT/runs/$SAMTOK_RUN_ID"
BOOTSTRAP="$RUN/bootstrap"
NODE="$ARNOLD_ID"
REPO="/tmp/samtok-edit-${SAMTOK_RUN_ID}-node${NODE}"
if [[ -e "$RUN/nodes/$NODE" || -e "$RUN/SUCCESS.json" ]]; then
  echo "Run already used: $RUN. Set a NEW common SAMTOK_RUN_ID; old logs are preserved." >&2
  exit 2
fi
mkdir -p "$BOOTSTRAP"
# Atomic per-node claim: reject scheduler retries BEFORE cloning/installing or appending logs.
if ! mkdir "$BOOTSTRAP/node${NODE}.claimed"; then
  echo "Worker $NODE already started this run. Use a NEW common SAMTOK_RUN_ID." >&2
  exit 2
fi
exec > >(tee -a "$BOOTSTRAP/node${NODE}.log") 2>&1
BOOTSTRAP_PHASE=checkout
bootstrap_failed() {
  local result="${1:-$?}"
  trap - ERR TERM INT
  mkdir -p "$RUN/nodes/$NODE"
  local failure_tmp="$RUN/nodes/$NODE/bootstrap-failure.$$.tmp"
  printf '{"error":"bootstrap failed during %s; see bootstrap/node%s.log","exit_code":%d}\n' \
    "$BOOTSTRAP_PHASE" "$NODE" "$result" > "$failure_tmp"
  # Publish complete JSON only if no more specific Python failure exists.
  ln "$failure_tmp" "$RUN/nodes/$NODE/failure.json" 2>/dev/null || true
  unlink "$failure_tmp"
  exit "$result"
}
trap bootstrap_failed ERR
trap 'bootstrap_failed 143' TERM
trap 'bootstrap_failed 130' INT

export WANDB_DISABLE_SERVICE=true WANDB_START_METHOD=thread
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export SAMTOK_ENV="/tmp/samtok21-${SAMTOK_RUN_ID}-node${NODE}-env"
export SAMTOK_PYTHON="${SAMTOK_PYTHON:-/usr/bin/python3.11}"
export SAMTOK_CUDA_READY_TIMEOUT="${SAMTOK_CUDA_READY_TIMEOUT:-600}"
export SAMTOK_CUDA_READY_INTERVAL="${SAMTOK_CUDA_READY_INTERVAL:-15}"

if [[ -e "$REPO" ]]; then
  echo "Node-local checkout already exists: $REPO (choose a fresh SAMTOK_RUN_ID)" >&2
  false
fi
export GIT_TERMINAL_PROMPT=0
# Clone the exact requested branch into node-local /tmp; never execute an MNT source snapshot.
git clone --branch "$SAMTOK_EDIT_BRANCH" --single-branch "$SAMTOK_EDIT_REPO_URL" "$REPO"
cd "$REPO"
if [[ -n "${SAMTOK_EDIT_COMMIT:-}" ]]; then
  [[ "$SAMTOK_EDIT_COMMIT" =~ ^[a-fA-F0-9]{40}$ ]] || { echo 'Use a full commit SHA' >&2; false; }
  git checkout --detach "$SAMTOK_EDIT_COMMIT"
fi
git rev-parse HEAD > "$BOOTSTRAP/node${NODE}.commit.txt"
BOOTSTRAP_PHASE=environment-or-pipeline
bash scripts/train/run_arnold_4node.sh \
  --full-training --stage1-steps 3081 --stage2-steps 3081 \
  --stage1-save-steps 2000 --stage2-save-steps 2000 \
  --max-pixels 1048576 --stage1-rank 64 --stage2-rank 32 \
  --region-weight 0.5 --attention-weight 0.1 \
  --attention-warmup-steps 500 --timeout 604800 --wandb-mode online
```

输出：`$SAMTOK_EXPERIMENT/runs/$SAMTOK_RUN_ID/`。顺序为 topology → 32-rank NCCL → Stage 1 → 全量 conditioning cache → Stage 2 → `audit_full.json` → `TRAINING_COMPLETE.json` → `SUCCESS.json`。正式模式不调用仅适用于 18 对 debug 样本的推理脚本。权重保存是 adapter 快照，不包含 optimizer-state resume。

## 3. 四机 debug 训练入口

18 对样本（前三个数据集各 6 对）、72/54 条 metadata。此入口复现历史 debug_002 的训练规模，同时使用当前修复后的代码和新 ID；旧 run 的 W&B 上传缺陷及验收边界见[实验记录第 13 节](02_SAMTokEdit_Qwen21_实验记录.md#13-2026-09-28四机-32-卡-debug_002-完整结果复核)。完整流程增加 node 0 八卡推理与 debug audit。

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
export SAMTOK_EDIT_REPO_URL=https://github.com/Tangent0308/samtok_edit.git
export SAMTOK_EDIT_BRANCH=qwen-image-2.1-dev
# 如需固定版本，可在四个 worker 上设置同一个已推送的完整 40 位 SAMTOK_EDIT_COMMIT。
export WANDB_ENTITY=2200012743-peking-university
export WANDB_PROJECT=samtok-edit
# 可直接替换下面的占位值；若 ARNOLD 已注入 secret，则沿用 secret。
export WANDB_API_KEY="${WANDB_API_KEY:-FILL_IN_WANDB_API_KEY}"
export SAMTOK_EXPERIMENT=/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/qwen21_4node_debug_20260928
export SAMTOK_TRAIN_DATA="$SAMTOK_EXPERIMENT/data"
export SAMTOK_RUN_ID=qwen21_4n_debug_003


# ----- User settings -----
export SAMTOK_EXPERIMENT="${SAMTOK_EXPERIMENT:?Set the shared experiment directory}"
# Set this explicitly and identically on all workers; use a fresh ID per attempt.
: "${SAMTOK_RUN_ID:?Set one fresh common SAMTOK_RUN_ID on all workers}"
# Inject WANDB_API_KEY as an ARNOLD secret on every worker.
: "${WANDB_API_KEY:?Inject WANDB_API_KEY through ARNOLD secrets}"
export WANDB_ENTITY="${WANDB_ENTITY:-2200012743-peking-university}"
export WANDB_PROJECT="${WANDB_PROJECT:-samtok-edit}"
export SAMTOK_EDIT_REPO_URL="${SAMTOK_EDIT_REPO_URL:-https://github.com/Tangent0308/samtok_edit.git}"
export SAMTOK_EDIT_BRANCH="${SAMTOK_EDIT_BRANCH:-qwen-image-2.1-dev}"

# ----- ARNOLD checks -----
: "${ARNOLD_WORKER_HOSTS:?ARNOLD must inject the four-worker host list}"
: "${ARNOLD_WORKER_NUM:?ARNOLD must inject ARNOLD_WORKER_NUM=4}"
: "${ARNOLD_WORKER_GPU:?ARNOLD must inject ARNOLD_WORKER_GPU=8}"
: "${ARNOLD_ID:?ARNOLD must inject ARNOLD_ID for this worker (0..3)}"
[[ "$ARNOLD_WORKER_NUM" == 4 && "$ARNOLD_WORKER_GPU" == 8 ]] || { echo 'Expected 4 workers x 8 GPUs' >&2; exit 2; }
[[ "$ARNOLD_ID" =~ ^[0-3]$ ]] || { echo 'ARNOLD_ID must be 0, 1, 2, or 3' >&2; exit 2; }
[[ "$SAMTOK_RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid SAMTOK_RUN_ID' >&2; exit 2; }
[[ -n "$WANDB_API_KEY" && "$WANDB_API_KEY" != FILL_IN* ]] || { echo 'Set WANDB_API_KEY as an ARNOLD secret' >&2; exit 2; }

# ARNOLD_WORKER_HOSTS carries the common rendezvous port; generic PORT varies by worker.
unset PORT MASTER_ADDR MASTER_PORT NODE_RANK NNODES GPUS_PER_NODE
export NODE_RANK="$ARNOLD_ID"
export ARNOLD_WORKER_NUM=4 ARNOLD_WORKER_GPU=8

RUN="$SAMTOK_EXPERIMENT/runs/$SAMTOK_RUN_ID"
BOOTSTRAP="$RUN/bootstrap"
NODE="$ARNOLD_ID"
REPO="/tmp/samtok-edit-${SAMTOK_RUN_ID}-node${NODE}"
if [[ -e "$RUN/nodes/$NODE" || -e "$RUN/SUCCESS.json" ]]; then
  echo "Run already used: $RUN. Set a NEW common SAMTOK_RUN_ID; old logs are preserved." >&2
  exit 2
fi
mkdir -p "$BOOTSTRAP"
# Atomic per-node claim: reject scheduler retries BEFORE cloning/installing or appending logs.
if ! mkdir "$BOOTSTRAP/node${NODE}.claimed"; then
  echo "Worker $NODE already started this run. Use a NEW common SAMTOK_RUN_ID." >&2
  exit 2
fi
exec > >(tee -a "$BOOTSTRAP/node${NODE}.log") 2>&1
BOOTSTRAP_PHASE=checkout
bootstrap_failed() {
  local result="${1:-$?}"
  trap - ERR TERM INT
  mkdir -p "$RUN/nodes/$NODE"
  local failure_tmp="$RUN/nodes/$NODE/bootstrap-failure.$$.tmp"
  printf '{"error":"bootstrap failed during %s; see bootstrap/node%s.log","exit_code":%d}\n' \
    "$BOOTSTRAP_PHASE" "$NODE" "$result" > "$failure_tmp"
  # Publish complete JSON only if no more specific Python failure exists.
  ln "$failure_tmp" "$RUN/nodes/$NODE/failure.json" 2>/dev/null || true
  unlink "$failure_tmp"
  exit "$result"
}
trap bootstrap_failed ERR
trap 'bootstrap_failed 143' TERM
trap 'bootstrap_failed 130' INT

export WANDB_DISABLE_SERVICE=true WANDB_START_METHOD=thread
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export SAMTOK_ENV="/tmp/samtok21-${SAMTOK_RUN_ID}-node${NODE}-env"
export SAMTOK_PYTHON="${SAMTOK_PYTHON:-/usr/bin/python3.11}"
export SAMTOK_CUDA_READY_TIMEOUT="${SAMTOK_CUDA_READY_TIMEOUT:-600}"
export SAMTOK_CUDA_READY_INTERVAL="${SAMTOK_CUDA_READY_INTERVAL:-15}"

if [[ -e "$REPO" ]]; then
  echo "Node-local checkout already exists: $REPO (choose a fresh SAMTOK_RUN_ID)" >&2
  false
fi
export GIT_TERMINAL_PROMPT=0
# Clone the exact requested branch into node-local /tmp; never execute an MNT source snapshot.
git clone --branch "$SAMTOK_EDIT_BRANCH" --single-branch "$SAMTOK_EDIT_REPO_URL" "$REPO"
cd "$REPO"
if [[ -n "${SAMTOK_EDIT_COMMIT:-}" ]]; then
  [[ "$SAMTOK_EDIT_COMMIT" =~ ^[a-fA-F0-9]{40}$ ]] || { echo 'Use a full commit SHA' >&2; false; }
  git checkout --detach "$SAMTOK_EDIT_COMMIT"
fi
git rev-parse HEAD > "$BOOTSTRAP/node${NODE}.commit.txt"
BOOTSTRAP_PHASE=environment-or-pipeline
bash scripts/train/run_arnold_4node.sh \
  --stage1-steps 2 --stage2-steps 3 \
  --stage1-save-steps 8 --stage2-save-steps 4 \
  --max-pixels 65536 --stage1-rank 64 --stage2-rank 32 \
  --region-weight 0.5 --attention-weight 0.1 \
  --attention-warmup-steps 1 --timeout 7200 --wandb-mode online
```

## 4. noref 四机转换入口（不使用 W&B）

数据已完成的生产版本是 `_003`，不需要为启动训练再次打标。下面 `_004` 是需要复跑时使用的新实验 ID，四机共 32 个独立 TP=1 vLLM 副本；使用 9B、thinking 关闭和规则回退。只有复跑产物重新验收/构建 metadata 后，才可更换正式训练数据目录；改 annotation ID 不会自动覆盖 train_full_9b_rules_003。

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
export SAMTOK_DATA_EXPERIMENT=/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/qwen21_full4_20260928
export SAMTOK_ANNOTATION_RUN_ID=qwen21_noref9b_rules_4n_full_004
export SAMTOK_ANNOTATION_MODEL=/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen3.5-9B
export SAMTOK_ANNOTATION_BATCH_SIZE=64
export SAMTOK_ANNOTATION_ATTEMPTS=3


export SAMTOK_DATA_EXPERIMENT="${SAMTOK_DATA_EXPERIMENT:-/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/qwen21_full4_20260928}"
: "${SAMTOK_ANNOTATION_RUN_ID:?Set a fresh common annotation run ID}"
export SAMTOK_ANNOTATION_SOURCES="${SAMTOK_ANNOTATION_SOURCES:-$SAMTOK_DATA_EXPERIMENT/data/semantic_sources.jsonl}"
export SAMTOK_ANNOTATION_MODEL="${SAMTOK_ANNOTATION_MODEL:-/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen3.5-9B}"
export SAMTOK_ANNOTATION_BATCH_SIZE="${SAMTOK_ANNOTATION_BATCH_SIZE:-64}"
export SAMTOK_ANNOTATION_ATTEMPTS="${SAMTOK_ANNOTATION_ATTEMPTS:-3}"
export SAMTOK_EDIT_REPO_URL="${SAMTOK_EDIT_REPO_URL:-https://github.com/Tangent0308/samtok_edit.git}"
export SAMTOK_EDIT_BRANCH="${SAMTOK_EDIT_BRANCH:-qwen-image-2.1-dev}"
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
```

输出：`$SAMTOK_DATA_EXPERIMENT/data/semantic_runs/$SAMTOK_ANNOTATION_RUN_ID/` 下的 `shards/`、`annotations.jsonl`、`rule_based.jsonl`、`failed.jsonl` 和 `SUCCESS.json`。`processed_complete` 表示全量已处理，不表示全部样本语义通过。32 个 shard 的 ID 归属、完整性和 hash 经过 merge 验证，failed 行单独保留。后续构建正式数据的步骤见[数据盘点第 9 节](04_SAMTokEdit_Qwen21_训练数据盘点.md#9-验证和重建命令)。

## 5. 训练前区域 cache（本地准备，不是四机训练）

正式版本的 regions 已完成，无需再运行。本节用于新的 metadata 版本，在已经安装训练环境的仓库根目录执行。16 个预处理 shards 分两批串行复用 8 张 GPU，每 GPU 同时仅一个 worker；解码 batch=16。必须逐个检查进程退出码后才能 merge。

```bash
set -Eeuo pipefail
export TRAIN_DATA=/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/qwen21_full4_20260928/data/train_full_rebuild_NEW_VERSION
export SAMTOK_CODEC=/mnt/bn/strategy-mllm-train/user/tanyue/models/SAMTok/Qwen3-VL-8B-SAMTok
export PYTHONPATH="$PWD:$PWD/DiffSynth-Studio${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$TRAIN_DATA/regions_logs"
for offset in 0 8; do
  pids=()
  for gpu in $(seq 0 7); do
    rank=$((offset + gpu))
    CUDA_VISIBLE_DEVICES="$gpu" python -m samtok_edit21.full_regions worker \
      --metadata "$TRAIN_DATA/stage1.jsonl" --output "$TRAIN_DATA/regions" \
      --qwen /mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen-Image-2.1 \
      --samtok "$SAMTOK_CODEC" --rank "$rank" --shards 16 --device cuda:0 \
      --max-pixels 1048576 --decode-batch-size 16 \
      > "$TRAIN_DATA/regions_logs/worker-$rank.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
done
python -m samtok_edit21.full_regions merge \
  --metadata "$TRAIN_DATA/stage1.jsonl" --output "$TRAIN_DATA/regions" --shards 16
```

## 6. 检查进度和结果

训练的每个节点日志在 `logs/node<rank>/`，环境/checkout/CUDA 诊断在 `bootstrap/`。ARNOLD 控制台每 60 秒输出 `running` 心跳和实际子进程日志路径；Stage 1/2 的 `startup.jsonl` 记录进入训练前的各阶段。控制台停留在 stage1 命令行本身不足以判断是否完成更新，须查看下面的文件。具体失败先看 `nodes/*/failure.json` 指向的原始节点日志；其他节点常只是连带退出。以下命令在可访问共享盘的机器执行，不含 W&B key：

```bash
RUN=/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/qwen21_full4_20260928/runs/qwen21_full_4n_formal_002
find "$RUN/nodes" -name failure.json -print -exec cat {} \;
tail -n 8 "$RUN/stage1/startup.jsonl"
tail -n 20 "$RUN/logs/node0/stage1.log"
cat "$RUN/nodes/0/stage1.progress.json"
tail -n 2 "$RUN/stage1/training_metrics.jsonl"
tail -n 2 "$RUN/stage2/training_metrics.jsonl"
cat "$RUN/audit_full.json"
cat "$RUN/TRAINING_COMPLETE.json" "$RUN/SUCCESS.json"
```

最终审计检查 optimizer updates、全局样本数和各分支比例、各 rank 参数 hash、有效 adapter、W&B 完成状态与 cache checksum；训练时另逐 backward 检查梯度。debug 使用 `audit.json` 且没有 `TRAINING_COMPLETE.json`。W&B 本地 finished 加 SDK 错误检查不等于独立读取远端所有点位的验证，最终可结合项目页面复核上传。文档与代码完成本地检查后提交；完整的新四机运行仍需由上述作业实际执行验证。
