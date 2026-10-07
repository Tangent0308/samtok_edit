# SAMTokEdit（Qwen-Image-2.1）v2 四机实验运行指南

本页是 v2 的四机启动入口。ARNOLD 作业配置为 4 workers × 8 GPUs，四个 worker 运行同一段脚本；代码由各节点从 GitHub clone 到 `/tmp`，共享盘只存数据、日志和产物。启动脚本与 v1 调通的入口相同（节点 claim、失败记录、环境安装、CUDA/NCCL 检查），只更新了分支、数据和阶段参数；另外把所有报错并入 stdout，并在写共享盘之前做一次写探针。v1 的入口和排错记录见 [archive/v1](archive/v1/03_SAMTokEdit_Qwen21_四机实验运行指南.md)。

## 1. 约定

- **代码**：分支 `qwen-image-2.1-v2`，固定 `SAMTOK_EDIT_COMMIT=487a3e4282847b3af96bfcc324b0cfe5a0f44daa`（已推送；四台机器必须相同）。
- **实验根目录**：`SAMTOK_EXPERIMENT=/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v2`；每次运行写入 `$SAMTOK_EXPERIMENT/runs/$SAMTOK_RUN_ID/`。
- **数据**：`$SAMTOK_EXPERIMENT/data/train_v2_box_001`（[数据盘点](04_SAMTokEdit_Qwen21_训练数据盘点.md)）。
- **阶段**：`--phases` 取 `stage1,cache,stage2` 的子集，默认 `stage1,stage2`。Stage 2 默认在训练时即时计算条件，不建缓存（[代码实现说明 5.1 节](01_SAMTokEdit_Qwen21_代码实现说明.md#51-不建缓存stage-2-即时计算条件默认)），结果与读缓存训练逐位一致。Stage 2 不依赖 Stage 1，各臂只需 `--phases stage2`。
- **run ID**：每次提交用新的共同 `SAMTOK_RUN_ID`；已用过的 ID 会被拒绝（防止调度器重试覆盖日志）。
- **W&B**：入口优先用 ARNOLD 注入的 `WANDB_API_KEY`，没有时读取共享盘上的私有文件 `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/.secrets/wandb.env`（内容为 `export WANDB_API_KEY=...`，权限 600）。仓库是公开的，key 不要写进脚本或提交。run 名为 `<RUN_ID>-stage1/-stage2`。
- **存储**：不建缓存后，每个运行只写 adapter、checkpoint 和日志（Stage 1 约 6 GB，Stage 2 约 2 GB）。全量缓存需要约 2.8 TB，intern 和 user 的 NAS 配额都放不下（[实验记录第 7、9 节](02_SAMTokEdit_Qwen21_实验记录.md#9-运行-a-在缓存阶段因配额失败与-stage-2-改为即时计算2026-10-07)），所以不要再用 `cache` 阶段。NAS 配额从 `df` 看不出来，入口的写探针只能发现"已经写不进"。可写性检查（在开发机上）：
  ```bash
  P=/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v2/runs/.probe_$$; printf x > $P && rm $P && echo writable
  ```
- **耗时估计**：每个阶段开头先在 rank 0 计算模型文件 hash，集群上约 18 分钟。运行 A 的 Stage 1 在 32 卡上约 6 s/update，1,300 update 用了 2 小时 15 分钟。Stage 2（即时计算）在运行 A2 上实测约 15 s/update（32 卡），1,000 update 约 4 小时；第一个 update 含预热，约 10 分钟。峰值显存：Stage 1 约 21 GiB/卡，即时计算的 Stage 2 约 39 GiB/卡（多出来的是 TE 和 VAE）。

## 2. 运行 A2：Stage 2 B0 seed 1（E3），复用运行 A 的 Stage 1

运行 A（`qwen21_v2_4n_A_s1_b0_001`）已完成 Stage 1（E1），在缓存阶段因 user 配额写满失败（实验记录第 9 节）。A2 用同一个 seed 只跑 Stage 2，条件即时计算。

下面是完整脚本，直接粘贴到 ARNOLD worker 启动命令。其他运行只需替换开头的 "运行设置" 块（第 3、4 节）。

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
exec 2>&1  # ARNOLD's log page shows stdout; route every shell error there as well.
# ===== 运行设置（各运行只改这一块） =====
export SAMTOK_RUN_ID=qwen21_v2_4n_A2_s2_b0_001
export SAMTOK_EXPERIMENT=/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v2
export SAMTOK_TRAIN_DATA="$SAMTOK_EXPERIMENT/data/train_v2_box_001"
STAGE1="$SAMTOK_EXPERIMENT/runs/qwen21_v2_4n_A_s1_b0_001/stage1/adapter"
ARGS=(
  --full-training
  --phases stage2 --stage1-adapter "$STAGE1"
  --stage2-steps 1000 --stage2-save-steps 1000 --stage2-rank 32
  --binding none --seed 20261006
  --max-pixels 1048576 --timeout 604800 --wandb-mode online
)

# ===== 固定设置 =====
export SAMTOK_EDIT_REPO_URL=https://github.com/Tangent0308/samtok_edit.git
export SAMTOK_EDIT_BRANCH=qwen-image-2.1-v2
export SAMTOK_EDIT_COMMIT=487a3e4282847b3af96bfcc324b0cfe5a0f44daa
export SAMTOK_DISTRIBUTED_TIMEOUT_SECONDS="${SAMTOK_DISTRIBUTED_TIMEOUT_SECONDS:-86400}"
export WANDB_ENTITY=2200012743-peking-university
export WANDB_PROJECT=samtok-edit
# W&B key: an ARNOLD secret if injected, else the private key file on the shared disk. Never commit the key.
WANDB_KEY_FILE=/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/.secrets/wandb.env
[[ -n "${WANDB_API_KEY:-}" || ! -f "$WANDB_KEY_FILE" ]] || source "$WANDB_KEY_FILE"
export WANDB_API_KEY="${WANDB_API_KEY:-}"

: "${ARNOLD_WORKER_HOSTS:?ARNOLD must inject the four-worker host list}"
: "${ARNOLD_WORKER_NUM:?ARNOLD must inject ARNOLD_WORKER_NUM=4}"
: "${ARNOLD_WORKER_GPU:?ARNOLD must inject ARNOLD_WORKER_GPU=8}"
: "${ARNOLD_ID:?ARNOLD must inject ARNOLD_ID for this worker (0..3)}"
[[ "$ARNOLD_WORKER_NUM" == 4 && "$ARNOLD_WORKER_GPU" == 8 ]] || { echo 'Expected 4 workers x 8 GPUs' >&2; exit 2; }
[[ "$ARNOLD_ID" =~ ^[0-3]$ ]] || { echo 'ARNOLD_ID must be 0, 1, 2, or 3' >&2; exit 2; }
[[ "$SAMTOK_RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid SAMTOK_RUN_ID' >&2; exit 2; }
[[ -n "$WANDB_API_KEY" ]] || { echo "No W&B key: inject WANDB_API_KEY or create $WANDB_KEY_FILE" >&2; exit 2; }
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

执行顺序：拓扑与源码一致性检查 → 32 卡 NCCL 探针 → Stage 2（`stage2/`，每个样本即时计算条件）→ rank 0 审计（`audit.json`）→ `TRAINING_COMPLETE.json`、`SUCCESS.json`。`--stage1-adapter` 只记录与之配对的 Stage 1（写入 `manifest.json`），正式训练不加载它。

- Stage 1（运行 A 已完成）：1,300 update（约 3 个 epoch），每 200 update 存一次 `step-*.safetensors`，最终 adapter 在 `runs/qwen21_v2_4n_A_s1_b0_001/stage1/adapter/`。类型采样 `natural`：每行约见 3 次，add/remove/replace/attribute 占 95%。
- Stage 2：1,000 update（缩减日程 R，D7），每 250 update 存一次，最终 adapter 在 `stage2/adapter/`（adapter.json 记录 conditioning identity 和 binding）。类型采样默认 `main4`：add/remove/replace/attribute 按 v1 的 14:14:14:20 分配约 95%，其他类型保持自然占比。两阶段都可用 `--stage1-type-weights` / `--stage2-type-weights` 改为 `v1`、`natural` 或 `main4`。
- 先用 B0 的 250/500/750/1,000 update checkpoint 看学习曲线（反事实跟随率、漂移率）；若到 1,000 仍在明显上升，再把所有臂统一加长。

## 3. Stage 2 消融臂

每个臂是一个独立的 4 × 8 卡作业，可以与其他臂、与运行 A2 同时跑：
- 各臂只读同一份 `stage2.jsonl`，即时计算条件，各写自己的 `runs/<RUN_ID>/`（约 2 GB）。
- 提交方法：用第 2 节的完整脚本，只把开头的"运行设置"块（从 `export SAMTOK_RUN_ID` 到 `ARGS=( … )` 结束）换成下面对应的块，其余不动。
- 绑定臂（E5–E7）用与 B0 seed 1（运行 A2）相同的 seed 20261006，数据顺序完全相同，便于配对比较；B0 seed 2 换 seed，用来估计 seed 间方差。
- 耗时：每个臂约 18 分钟启动 + 1,000 update × 约 15 s ≈ 4.5 小时（bias 类约慢 10–15%）。

| 臂（计划编号） | 何时可跑 | `SAMTOK_RUN_ID` |
|---|---|---|
| B0 seed 2（E3） | 现在 | `qwen21_v2_4n_S2_b0_s2_001` |
| 区域嵌入（E6） | 现在 | `qwen21_v2_4n_S2_embed_001` |
| region-RoPE（E7） | 现在 | `qwen21_v2_4n_S2_rope_001` |
| 区域偏置（E5） | E4 选定 β/ε/作用范围之后 | `qwen21_v2_4n_S2_bias_<span\|clause>_001` |

```bash
# ===== 运行设置：B0 seed 2（E3） =====
export SAMTOK_RUN_ID=qwen21_v2_4n_S2_b0_s2_001
export SAMTOK_EXPERIMENT=/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v2
export SAMTOK_TRAIN_DATA="$SAMTOK_EXPERIMENT/data/train_v2_box_001"
STAGE1="$SAMTOK_EXPERIMENT/runs/qwen21_v2_4n_A_s1_b0_001/stage1/adapter"
ARGS=(
  --full-training
  --phases stage2 --stage1-adapter "$STAGE1"
  --stage2-steps 1000 --stage2-save-steps 1000 --stage2-rank 32
  --binding none --seed 20261007
  --max-pixels 1048576 --timeout 604800 --wandb-mode online
)
```

```bash
# ===== 运行设置：区域嵌入（E6） =====
export SAMTOK_RUN_ID=qwen21_v2_4n_S2_embed_001
export SAMTOK_EXPERIMENT=/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v2
export SAMTOK_TRAIN_DATA="$SAMTOK_EXPERIMENT/data/train_v2_box_001"
STAGE1="$SAMTOK_EXPERIMENT/runs/qwen21_v2_4n_A_s1_b0_001/stage1/adapter"
ARGS=(
  --full-training
  --phases stage2 --stage1-adapter "$STAGE1"
  --stage2-steps 1000 --stage2-save-steps 1000 --stage2-rank 32
  --binding region_embed --binding-rank 64 --seed 20261006
  --max-pixels 1048576 --timeout 604800 --wandb-mode online
)
```

```bash
# ===== 运行设置：region-RoPE（E7） =====
export SAMTOK_RUN_ID=qwen21_v2_4n_S2_rope_001
export SAMTOK_EXPERIMENT=/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v2
export SAMTOK_TRAIN_DATA="$SAMTOK_EXPERIMENT/data/train_v2_box_001"
STAGE1="$SAMTOK_EXPERIMENT/runs/qwen21_v2_4n_A_s1_b0_001/stage1/adapter"
ARGS=(
  --full-training
  --phases stage2 --stage1-adapter "$STAGE1"
  --stage2-steps 1000 --stage2-save-steps 1000 --stage2-rank 32
  --binding region_rope --seed 20261006
  --max-pixels 1048576 --timeout 604800 --wandb-mode online
)
```

E5 等 E4 的结果：把 `<SCOPE>` 换成 `span` 或 `clause`，β、ε 换成 E4 选出的值（默认 1.0 / 0.05）。

```bash
# ===== 运行设置：区域偏置（E5），E4 之后 =====
export SAMTOK_RUN_ID=qwen21_v2_4n_S2_bias_<SCOPE>_001
export SAMTOK_EXPERIMENT=/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v2
export SAMTOK_TRAIN_DATA="$SAMTOK_EXPERIMENT/data/train_v2_box_001"
STAGE1="$SAMTOK_EXPERIMENT/runs/qwen21_v2_4n_A_s1_b0_001/stage1/adapter"
ARGS=(
  --full-training
  --phases stage2 --stage1-adapter "$STAGE1"
  --stage2-steps 1000 --stage2-save-steps 1000 --stage2-rank 32
  --binding bias_<SCOPE> --binding-beta 1.0 --binding-eps 0.05 --seed 20261006
  --max-pixels 1048576 --timeout 604800 --wandb-mode online
)
```

E4（只在 B0 上做推理期偏置，不训练）在开发机上跑，不需要四机，见[代码实现说明第 7 节](01_SAMTokEdit_Qwen21_代码实现说明.md#7-推理)的 `--binding` 覆盖。

## 4. 四机 smoke（可选：正式提交前检查集群环境）

小 smoke 数据、少量步数，执行 Stage 1 和即时计算的 Stage 2，并在 node 0 上运行八卡推理 smoke（所有推理模式、融合和绑定覆盖）。本地八卡已用同样的阶段通过（[实验记录第 9 节](02_SAMTokEdit_Qwen21_实验记录.md#9-运行-a-在缓存阶段因配额失败与-stage-2-改为即时计算2026-10-07)）。

```bash
# ===== 运行设置：四机 smoke =====
export SAMTOK_RUN_ID=qwen21_v2_4n_smoke_001
export SAMTOK_EXPERIMENT=/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v2
export SAMTOK_TRAIN_DATA="$SAMTOK_EXPERIMENT/smoke/data_smoke_001"
ARGS=(
  --phases stage1,stage2
  --stage1-steps 2 --stage1-save-steps 8 --stage2-steps 3 --stage2-save-steps 4
  --binding none --max-pixels 65536 --timeout 7200 --wandb-mode online
)
```

## 5. 产物、验收与失败处理

| 路径（`runs/<RUN_ID>/`） | 内容 |
|---|---|
| `manifest.json`、`nodes/<i>/topology.json` | 参数、commit、源码 hash、包版本、数据 hash；四节点一致才继续 |
| `logs/node<i>/<phase>.log`、`logs/node<i>/<phase>-ranks/` | 各阶段与各 rank 的日志 |
| `stage1/`、`stage2/` | `adapter/`、`step-*.safetensors`、`training_metrics.jsonl`（每个 update 的配比、类型、loss）、`gradients-rank*.jsonl`、`schedule.json`、`run.json`、`rank_parameters.json`、`wandb.json` |
| `cache/` | 只在使用 `cache` 阶段时存在：`<rank>/<index>.pth` + 同名 `.json` 校验文件、`manifest.json` |
| `audit.json` | 已运行阶段的审计：update 数、每 rank 精确配比、梯度、各 rank 权重一致、adapter 绑定配方；有缓存时抽查缓存行，即时计算时核对 adapter 记录的条件身份与本次数据、分辨率一致 |
| `SUCCESS.json`、`TRAINING_COMPLETE.json` | 全部阶段和审计通过 |

- **失败**：任一节点写 `nodes/<i>/failure.json`，其他节点检测到后退出；原因在该节点的 `bootstrap/node<i>.log` 和对应阶段日志中。修复后用**新的** run ID 重新提交。
- **ARNOLD 报错退出、共享盘上却没有 run 目录**：入口在写共享盘之前就失败了，原因只在 ARNOLD 日志页（入口已把 stderr 并入 stdout）。退出码 3 表示共享盘写探针失败（通常是 NAS 配额已满）；这种情况下什么都没写，恢复后可以沿用同一个 run ID。
- **仍用缓存时**（不推荐，需约 2.8 TB）：缓存中途失败用新 run ID 和 `--phases cache,stage2 --cache <原 run>/cache --resume-cache` 续建，已完成且可读的 payload 会复用。
- **checkpoint**：`step-<microsteps>.safetensors` 只含可训练权重（不含 optimizer 状态），张量名和形状与最终 adapter 完全相同（含 region_embed）。评测中间 checkpoint 时，新建目录，把它复制为 `adapter.safetensors`，并复制同一 run 的 `adapter/adapter.json`。
