# SAMTokEdit（Qwen-Image-2.1）v2 四机实验运行指南

本页是 v2 的四机启动入口。ARNOLD 作业配置为 4 workers × 8 GPUs，四个 worker 运行同一段脚本；代码由各节点从 GitHub clone 到 `/tmp`，共享盘只存数据、日志和产物。启动脚本与 v1 调通的入口相同（节点 claim、失败记录、环境安装、CUDA/NCCL 检查），只更新了分支、数据和阶段参数。v1 的入口和排错记录见 [archive/v1](archive/v1/03_SAMTokEdit_Qwen21_四机实验运行指南.md)。

## 1. 约定

- **代码**：分支 `qwen-image-2.1-v2`，固定 `SAMTOK_EDIT_COMMIT=c5535133fcad14303360d406e753f14af9313d41`（已推送；四台机器必须相同）。
- **实验根目录**：`SAMTOK_EXPERIMENT=/mnt/bn/strategy-mllm-train/intern/users/tanyue/experiments/SAMTokEdit/qwen21_v2`；每次运行写入 `$SAMTOK_EXPERIMENT/runs/$SAMTOK_RUN_ID/`。
- **数据**：`$SAMTOK_EXPERIMENT/data/train_v2_box_001`（[数据盘点](04_SAMTokEdit_Qwen21_训练数据盘点.md)）。
- **阶段**：`--phases` 取 `stage1,cache,stage2` 的子集。Stage 2 不依赖 Stage 1（缓存用 raw TE），所以第一次运行建缓存，之后所有 Stage 2 臂用 `--cache` 复用。
- **run ID**：每次提交用新的共同 `SAMTOK_RUN_ID`；已用过的 ID 会被拒绝（防止调度器重试覆盖日志）。
- **W&B**：通过 ARNOLD secret 注入 `WANDB_API_KEY`；run 名为 `<RUN_ID>-stage1/-stage2`。
- **存储**：全量缓存约 2.8 TB（每行约 9.6 MB）。启动前确认 intern 目录配额。
- **耗时估计**（本地八卡、GPU 与其他任务共享时的实测；集群独占时应更短）：Stage 1 约 10 s/update，860 update 约 2.5 小时；缓存约 1.7 s/行/卡，32 卡约 4–5 小时；Stage 2 约 28 s/update（bias 类慢约 10–15%），1,000 update 约 8 小时。峰值显存约 21 GiB/卡。

## 2. 运行 A：Stage 1（E1）+ 缓存（E2）+ Stage 2 B0 seed 1（E3）

下面是完整脚本，直接粘贴到 ARNOLD worker 启动命令。其他运行只需替换开头的 "运行设置" 块（第 3、4 节）。

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
# ===== 运行设置（各运行只改这一块） =====
export SAMTOK_RUN_ID=qwen21_v2_4n_A_s1_b0_001
export SAMTOK_EXPERIMENT=/mnt/bn/strategy-mllm-train/intern/users/tanyue/experiments/SAMTokEdit/qwen21_v2
export SAMTOK_TRAIN_DATA="$SAMTOK_EXPERIMENT/data/train_v2_box_001"
ARGS=(--full-training --phases stage1,cache,stage2
      --stage1-steps 860 --stage1-save-steps 1600 --stage1-rank 64
      --stage2-steps 1000 --stage2-save-steps 1000 --stage2-rank 32
      --binding none --seed 20261006
      --max-pixels 1048576 --timeout 604800 --wandb-mode online)
# ===== 固定设置 =====
export SAMTOK_EDIT_REPO_URL=https://github.com/Tangent0308/samtok_edit.git
export SAMTOK_EDIT_BRANCH=qwen-image-2.1-v2
export SAMTOK_EDIT_COMMIT=c5535133fcad14303360d406e753f14af9313d41
export SAMTOK_DISTRIBUTED_TIMEOUT_SECONDS="${SAMTOK_DISTRIBUTED_TIMEOUT_SECONDS:-86400}"
export WANDB_ENTITY=2200012743-peking-university
export WANDB_PROJECT=samtok-edit
export WANDB_API_KEY="${WANDB_API_KEY:-FILL_IN_WANDB_API_KEY}"   # 推荐用 ARNOLD secret 注入

: "${ARNOLD_WORKER_HOSTS:?ARNOLD must inject the four-worker host list}"
: "${ARNOLD_WORKER_NUM:?ARNOLD must inject ARNOLD_WORKER_NUM=4}"
: "${ARNOLD_WORKER_GPU:?ARNOLD must inject ARNOLD_WORKER_GPU=8}"
: "${ARNOLD_ID:?ARNOLD must inject ARNOLD_ID for this worker (0..3)}"
[[ "$ARNOLD_WORKER_NUM" == 4 && "$ARNOLD_WORKER_GPU" == 8 ]] || { echo 'Expected 4 workers x 8 GPUs' >&2; exit 2; }
[[ "$ARNOLD_ID" =~ ^[0-3]$ ]] || { echo 'ARNOLD_ID must be 0, 1, 2, or 3' >&2; exit 2; }
[[ "$SAMTOK_RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid SAMTOK_RUN_ID' >&2; exit 2; }
[[ -n "$WANDB_API_KEY" && "$WANDB_API_KEY" != FILL_IN* ]] || { echo 'Set WANDB_API_KEY as an ARNOLD secret' >&2; exit 2; }
[[ -f "$SAMTOK_TRAIN_DATA/metadata_report.json" ]] || { echo "Missing data: $SAMTOK_TRAIN_DATA" >&2; exit 2; }

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
git clone --branch "$SAMTOK_EDIT_BRANCH" --single-branch "$SAMTOK_EDIT_REPO_URL" "$REPO"
cd "$REPO"
git checkout --detach "$SAMTOK_EDIT_COMMIT"
git rev-parse HEAD > "$BOOTSTRAP/node${NODE}.commit.txt"
BOOTSTRAP_PHASE=environment-or-pipeline
bash scripts/training/run_arnold.sh "${ARGS[@]}"
```

执行顺序：拓扑与源码一致性检查 → 32 卡 NCCL 探针 → Stage 1（`stage1/`）→ 缓存（`cache/`）→ Stage 2（`stage2/`）→ rank 0 审计（`audit.json`）→ `TRAINING_COMPLETE.json`、`SUCCESS.json`。

- Stage 1：860 update（约 2 个 epoch），每 200 update（1,600 microsteps）存一次 `stage1/step-*.safetensors`，最终 adapter 在 `stage1/adapter/`。
- Stage 2：1,000 update（缩减日程 R，D7），每 250 update 存一次，最终 adapter 在 `stage2/adapter/`（adapter.json 记录 conditioning identity 和 binding）。

## 3. Stage 2 消融臂（复用运行 A 的缓存）

只替换运行设置块。`--cache` 指向运行 A 的缓存目录；Stage 2 启动时会并行校验全部缓存行。

```bash
# ===== 运行设置：B0 seed 2（E3 的第二个 seed） =====
export SAMTOK_RUN_ID=qwen21_v2_4n_S2_b0_s2_001
export SAMTOK_EXPERIMENT=/mnt/bn/strategy-mllm-train/intern/users/tanyue/experiments/SAMTokEdit/qwen21_v2
export SAMTOK_TRAIN_DATA="$SAMTOK_EXPERIMENT/data/train_v2_box_001"
CACHE="$SAMTOK_EXPERIMENT/runs/qwen21_v2_4n_A_s1_b0_001/cache"
ARGS=(--full-training --phases stage2 --cache "$CACHE"
      --stage2-steps 1000 --stage2-save-steps 1000 --stage2-rank 32
      --binding none --seed 20261007
      --max-pixels 1048576 --timeout 604800 --wandb-mode online)
```

| 臂（计划编号） | `SAMTOK_RUN_ID` 建议 | `--binding` 及参数 |
|---|---|---|
| B0 seed 2（E3） | `qwen21_v2_4n_S2_b0_s2_001` | `--binding none --seed 20261007` |
| 区域偏置（E5） | `qwen21_v2_4n_S2_bias_<span\|clause>_001` | `--binding bias_span` 或 `bias_clause`，`--binding-beta`/`--binding-eps` 取 E4 选出的值（默认 1.0 / 0.05） |
| 区域嵌入（E6） | `qwen21_v2_4n_S2_embed_001` | `--binding region_embed --binding-rank 64` |
| region-RoPE（E7） | `qwen21_v2_4n_S2_rope_001` | `--binding region_rope` |

E4（只在 B0 上做推理期偏置，不训练）不需要四机，见[代码实现说明第 7 节](01_SAMTokEdit_Qwen21_代码实现说明.md#7-推理)的 `--binding` 覆盖。

## 4. 四机 smoke（可选：正式提交前检查集群环境）

小 smoke 数据、少量步数，完整执行三个阶段，并在 node 0 上运行八卡推理 smoke（所有推理模式、融合和绑定覆盖）。本地八卡已用相同参数通过（[实验记录第 5 节](02_SAMTokEdit_Qwen21_实验记录.md#5-八卡-smoke)）。

```bash
# ===== 运行设置：四机 smoke =====
export SAMTOK_RUN_ID=qwen21_v2_4n_smoke_001
export SAMTOK_EXPERIMENT=/mnt/bn/strategy-mllm-train/intern/users/tanyue/experiments/SAMTokEdit/qwen21_v2
export SAMTOK_TRAIN_DATA="$SAMTOK_EXPERIMENT/smoke/data_smoke_001"
ARGS=(--phases stage1,cache,stage2
      --stage1-steps 2 --stage1-save-steps 8 --stage2-steps 3 --stage2-save-steps 4
      --binding none --max-pixels 65536 --timeout 7200 --wandb-mode online)
```

## 5. 产物、验收与失败处理

| 路径（`runs/<RUN_ID>/`） | 内容 |
|---|---|
| `manifest.json`、`nodes/<i>/topology.json` | 参数、commit、源码 hash、包版本、数据 hash；四节点一致才继续 |
| `logs/node<i>/<phase>.log`、`logs/node<i>/<phase>-ranks/` | 各阶段与各 rank 的日志 |
| `stage1/`、`stage2/` | `adapter/`、`step-*.safetensors`、`training_metrics.jsonl`（每个 update 的配比、类型、loss）、`gradients-rank*.jsonl`、`schedule.json`、`run.json`、`rank_parameters.json`、`wandb.json` |
| `cache/` | `<rank>/<index>.pth` + 同名 `.json` 校验文件、`manifest.json`（全部 payload 校验后才发布） |
| `audit.json` | 已运行阶段的审计：update 数、每 rank 精确配比、梯度、各 rank 权重一致、adapter 绑定配方、缓存抽查 |
| `SUCCESS.json`、`TRAINING_COMPLETE.json` | 全部阶段和审计通过 |

- **失败**：任一节点写 `nodes/<i>/failure.json`，其他节点检测到后退出；原因在该节点的 `bootstrap/node<i>.log` 和对应阶段日志中。修复后用**新的** run ID 重新提交。
- **缓存中途失败**（如共享盘瞬时写错误）：新 run ID，设置 `--phases cache,stage2 --cache <原 run>/cache --resume-cache`；已完成且可读的 payload 会复用，坏文件重算。
- **checkpoint**：`step-<microsteps>.safetensors` 只含可训练权重（不含 optimizer 状态），张量名和形状与最终 adapter 完全相同（含 region_embed）。评测中间 checkpoint 时，新建目录，把它复制为 `adapter.safetensors`，并复制同一 run 的 `adapter/adapter.json`。
