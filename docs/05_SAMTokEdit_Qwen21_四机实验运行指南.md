# SAMTokEdit（Qwen-Image-2.1）四机实验运行指南（v3）

本页是 v3 实验在 ARNOLD 上的启动入口，以一次完整运行（E-A2b）为样例。之后的实验只需要改开头的"运行设置"。第二轮（阶段 B，C2）用单独的入口，见第 7 节。v2 的入口（Stage 1 / Stage 2 / MLLM 编辑核验）没有变，见 [archive/v2/03_SAMTokEdit_Qwen21_四机实验运行指南.md](archive/v2/03_SAMTokEdit_Qwen21_四机实验运行指南.md)。

## 1. 约定

- **作业配置**：4 workers × 8 GPUs，四个 worker 运行同一段启动脚本。
- **代码**：每个节点从 GitHub 把分支 `qwen-image-2.1-v3` clone 到本机 `/tmp`，并切到 `SAMTOK_EDIT_COMMIT` 指定的 commit。共享盘只放数据、日志和产物，不从共享盘执行代码。本页示例里固定的是 `06e668d0`，即当前训练与评测代码所在的 commit（之后只有文档提交）；代码再有改动时，用 `git -C /opt/tiger/tanyue/samtok_edit_qwen-image-2.1-v3 rev-parse HEAD` 取新的 SHA 并更新这里，前提是它已经推送。
- **实验根目录**：`/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v3`，每次运行写入 `runs/<SAMTOK_RUN_ID>/`。
- **run ID**：每次提交都换一个新的，四个 worker 相同。用过的 ID 会被拒绝，这样调度器重试不会覆盖日志。建议写成 `<实验名>_<机器数>n_<序号>`，例如 `ea2b_4n_001`。
- **一次运行做三件事**：训练 → 在全部 32 张卡上评测 → 0 号 worker 出配对分析报告。可以用 `SAMTOK_PHASES` 只跑其中一部分。
- **有效 batch = 总卡数 × `--accumulation`。** 四机 32 卡时，`--accumulation 1` 就是 batch 32。数据顺序由数据目录里的调度文件决定，总卡数 × accumulation 相同，每次更新用到的样本就相同：32 卡 × 1 和 8 卡 × 4 是同一批样本。
- **不需要 W&B key。** 训练指标写在共享盘的 `train_metrics.jsonl`、`quick_summary.jsonl` 里。
- **存储**：一次运行约 2–3 GB（4 个保存点各 0.33 GB、评测图像约 0.3 GB、日志）。NAS 配额写满时 `df` 仍显示有空间但任何新文件都会失败，启动脚本一开始会做写入探测。

## 2. 样例：E-A2b（按 E-A2 配方、用修正了初始化的代码复跑）

把下面两段**依次粘贴**成 ARNOLD 的 worker 启动命令：先是"运行设置"，再是 2.2 的入口脚本全文。

### 2.1 运行设置（每次运行只改这一段）

```bash
# ===== 运行设置 =====
export SAMTOK_RUN_ID=ea2b_4n_001            # 每次提交换新的
export SAMTOK_RUN_NAME=E-A2b                # 报告里这次运行的名字
export SAMTOK_EDIT_COMMIT=06e668d0bcfc1a8d79f066e0d58806b9c3466342   # 代码 commit，四个 worker 必须相同
export SAMTOK_EXPERIMENT=/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v3
export SAMTOK_GROUND_DATA=$SAMTOK_EXPERIMENT/data/stageA_ea2_20261012
# 32 卡 × accumulation 1 = batch 32，与 E-A2（8 卡 × 4）每次更新用到的样本完全相同
export SAMTOK_TRAIN_ARGS="--updates 1300 --accumulation 1 --checkpoints 325,650,975,1300 --quick-eval 0,325,650,975,1300"
export SAMTOK_EVAL_ARGS=""
# 报告里的参考评测，原版放第一个
export SAMTOK_REPORT_REFERENCES="stock=$SAMTOK_EXPERIMENT/baselines/stock_highlight_eval E-A1=$SAMTOK_EXPERIMENT/runs/ea1_local8_001/eval E-A2=$SAMTOK_EXPERIMENT/runs/ea2_local8_001/eval"
```

这次运行的目的和验收：
- 配方、数据、每次更新的样本都与 E-A2 相同，差别只有各卡的 LoRA 初始化现在是一致的（文档 03 的 5.5 节），所以报告里 `E-A2b-E-A2` 那一列直接给出修正带来的变化。
- 预期不低于 E-A2（预留集不带物体名 0.614、dev 0.621、成对跟随 0.738、remove 探针 0.758）。四条放行线与 E-A2 相同，报告末尾逐条给出 PASS / fail。
- 预计耗时 3.5–4.5 小时（装环境 10–20 分钟，训练约 2.5–3.5 小时，评测约 20 分钟）。这是按单机实测速度推算的，四机上还没有实测过。

### 2.2 入口脚本（原样粘贴，不用改）

这段就是仓库里的 `scripts/grounding/bootstrap_arnold.sh`。它检查 ARNOLD 注入的变量、占用 run 目录、clone 代码，然后调用仓库里的 `scripts/grounding/run_arnold.sh` 完成装环境、训练、评测和出报告。

<!-- code: scripts/grounding/bootstrap_arnold.sh -->
```bash
#!/usr/bin/env bash
# v3 token-grounding run: pre-clone ARNOLD entry.  Paste this whole file as the start command of every worker
# (4 workers x 8 GPUs), after editing the "run settings" block.  Each worker clones the repository at the pinned
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
# Data directory written by `samtok-edit ground-data train`.
export SAMTOK_GROUND_DATA="${SAMTOK_GROUND_DATA:-$SAMTOK_EXPERIMENT/data/stageA_ea2_20261012}"
# Trainer / evaluator arguments (see `samtok-edit ground-train --help`, `ground-eval --help`).
export SAMTOK_TRAIN_ARGS="${SAMTOK_TRAIN_ARGS:---updates 1300 --accumulation 1 --checkpoints 325,650,975,1300 --quick-eval 0,325,650,975,1300}"
export SAMTOK_EVAL_ARGS="${SAMTOK_EVAL_ARGS:-}"
# Reference evaluations for the paired report, "LABEL=PATH ..." with the base model first.
export SAMTOK_REPORT_REFERENCES="${SAMTOK_REPORT_REFERENCES:-stock=$SAMTOK_EXPERIMENT/baselines/stock_highlight_eval E-A1=$SAMTOK_EXPERIMENT/runs/ea1_local8_001/eval E-A2=$SAMTOK_EXPERIMENT/runs/ea2_local8_001/eval}"
# Phases to run: train,eval,report.  Without "train", set SAMTOK_ADAPTER to the adapter directory to evaluate.
export SAMTOK_PHASES="${SAMTOK_PHASES:-train,eval,report}"
export SAMTOK_ADAPTER="${SAMTOK_ADAPTER:-}"
# Code: branch and (recommended) the exact commit, so all workers and any later rerun use the same code.
export SAMTOK_EDIT_REPO_URL="${SAMTOK_EDIT_REPO_URL:-https://github.com/Tangent0308/samtok_edit.git}"
export SAMTOK_EDIT_BRANCH="${SAMTOK_EDIT_BRANCH:-qwen-image-2.1-v3}"
export SAMTOK_EDIT_COMMIT="${SAMTOK_EDIT_COMMIT:-}"
# Number of workers; 1 runs the same pipeline on a single 8-GPU machine.
export SAMTOK_NNODES="${SAMTOK_NNODES:-4}"

# ===== ARNOLD checks =====
: "${ARNOLD_WORKER_HOSTS:?ARNOLD must inject the worker host list}"
: "${ARNOLD_WORKER_NUM:?ARNOLD must inject ARNOLD_WORKER_NUM}"
: "${ARNOLD_WORKER_GPU:?ARNOLD must inject ARNOLD_WORKER_GPU}"
: "${ARNOLD_ID:?ARNOLD must inject ARNOLD_ID for this worker}"
[[ "$ARNOLD_WORKER_NUM" == "$SAMTOK_NNODES" ]] || { echo "Expected $SAMTOK_NNODES workers, ARNOLD reports $ARNOLD_WORKER_NUM" >&2; exit 2; }
[[ "$ARNOLD_WORKER_GPU" =~ ^[1-9][0-9]*$ ]] || { echo 'ARNOLD_WORKER_GPU must be a positive integer' >&2; exit 2; }
[[ "$ARNOLD_ID" =~ ^[0-9]+$ && "$ARNOLD_ID" -lt "$SAMTOK_NNODES" ]] || { echo "ARNOLD_ID must be 0..$((SAMTOK_NNODES - 1))" >&2; exit 2; }
[[ "$SAMTOK_RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid SAMTOK_RUN_ID' >&2; exit 2; }
[[ -f "$SAMTOK_GROUND_DATA/eval.json" ]] || { echo "No eval.json in $SAMTOK_GROUND_DATA" >&2; exit 2; }
if [[ ",$SAMTOK_PHASES," == *,train,* ]]; then
  [[ -f "$SAMTOK_GROUND_DATA/epoch_rows.jsonl" && -f "$SAMTOK_GROUND_DATA/schedule.json" ]] || {
    echo "No training rows in $SAMTOK_GROUND_DATA" >&2; exit 2; }
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
bash "$REPO/scripts/grounding/run_arnold.sh"
echo "[$(date -u +%FT%TZ)] node $NODE: done"
```
<!-- /code -->

## 3. 改参数的参考

所有设置都是环境变量，写在 2.1 那一段里。

| 想做的事 | 怎么设 |
|---|---|
| 换训练规模 | 在 `SAMTOK_TRAIN_ARGS` 里改 `--updates`、`--accumulation`，同时改 `--checkpoints` 和 `--quick-eval`（最后一个保存点应等于 `--updates`）。batch = 32 × accumulation |
| 从已有权重继续训练（阶段 B） | `SAMTOK_TRAIN_ARGS` 里加 `--init-adapter <权重目录>`，并把 `SAMTOK_GROUND_DATA` 指到新的数据目录 |
| 做消融 | `SAMTOK_TRAIN_ARGS` 里加 `--lambda-route 0`（去掉注意力路由）、`--lambda-distractor 0`（去掉干扰实例项）、`--no-code-injector`（去掉码本向量注入）；三者都加就是 E-A1 的方法 |
| 换路由层或系数 | `--route-layers 6,10,14,18,22,26`、`--lambda-route 0.05` |
| 只评测一个已有权重 | `export SAMTOK_PHASES=eval,report`，`export SAMTOK_ADAPTER=<权重目录>` |
| 只训练不评测 | `export SAMTOK_PHASES=train` |
| 评测原版模型（生成新的基线） | 不走这个入口，直接在任意一台 8 卡机器上运行 `ground-eval` 且不给 `--adapter`，见文档 01 第 4 节 |
| 换评测规模 | `SAMTOK_EVAL_ARGS`，例如 `"--remove-pairs 120 --edit-cases 48 --steps 20"`（这些是默认值） |
| 换报告里的参考运行或放行线 | `SAMTOK_REPORT_REFERENCES="名字=评测目录 …"`；放行线改 `ground-report` 的 `--gate-*` 参数（事后在开发机上重出报告即可，不用重跑） |
| 用别的 commit | 改 `SAMTOK_EDIT_COMMIT`，写完整的 40 位 SHA。留空则用分支当前的头（不建议，四台机器 clone 的时间不同可能拿到不同的代码） |
| 机器数不是 4 | `export SAMTOK_NNODES=<机器数>`，并相应调整 `--accumulation` 以保持 batch |

全部训练参数见 `samtok-edit ground-train --help`，说明在文档 01 的 3.5 节。

**在单机 8 卡上用同一个入口跑**（开发机上验证流程，或四机排队时先跑）：

```bash
export SAMTOK_RUN_ID=<新的 run id> SAMTOK_NNODES=1
export ARNOLD_WORKER_HOSTS=127.0.0.1:39241 ARNOLD_WORKER_NUM=1 ARNOLD_WORKER_GPU=8 ARNOLD_ID=0
export SAMTOK_TRAIN_ARGS="--updates 1300 --accumulation 4 --checkpoints 325,650,975,1300 --quick-eval 0,325,650,975,1300"
bash /opt/tiger/tanyue/samtok_edit_qwen-image-2.1-v3/scripts/grounding/bootstrap_arnold.sh
```

端口换一个没被占用的即可。

## 4. 冒烟测试（正式提交前检查集群环境，约 30 分钟）

```bash
# ===== 运行设置：四机冒烟 =====
export SAMTOK_RUN_ID=smoke_4n_001
export SAMTOK_RUN_NAME=smoke
export SAMTOK_EDIT_COMMIT=06e668d0bcfc1a8d79f066e0d58806b9c3466342
export SAMTOK_TRAIN_ARGS="--updates 4 --accumulation 1 --checkpoints 4 --quick-eval 0,4 --quick-limit 1"
export SAMTOK_EVAL_ARGS="--limit 8 --remove-pairs 4 --edit-cases 4"
```

后面接 2.2 的入口脚本。跑通的标志是 `runs/smoke_4n_001/SUCCESS.json` 和 `analysis.txt` 都生成了。数值没有意义，只看流程。

## 5. 产物、进度与失败处理

**run 目录的内容**（`$SAMTOK_EXPERIMENT/runs/<run id>/`）：

| 路径 | 内容 |
|---|---|
| `bootstrap/node<i>.log` | 第 i 个 worker 的完整日志；`node<i>.commit.txt` 是它实际用的 commit |
| `nodes/<i>/phase` | 该 worker 当前所处的阶段；失败时同目录下有 `failure.json` |
| `logs/node<i>/train-ranks/`、`eval-ranks/` | 每张卡的单独日志 |
| `train/train_metrics.jsonl` | 每 10 次更新一行：各项损失、跳过的样本数、耗时 |
| `train/quick_summary.jsonl` | 训练中的快速评测（160 条，只看趋势） |
| `train/adapter_<次数>/` | 保存点：DiT LoRA 和码本向量注入 |
| `eval/eval.json`、`eval/images/` 等 | 完整评测的逐例结果和生成图 |
| `analysis.txt`、`analysis.json` | 配对分析报告和它的数值摘要 |
| `SUCCESS.json` | 全部完成的标记，记有 commit 和最终权重路径 |

**看进度**（开发机上）：

```bash
R=/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v3/runs/ea2b_4n_001
cat $R/nodes/*/phase; tail -n 2 $R/train/train_metrics.jsonl; cat $R/train/quick_summary.jsonl
ls $R/nodes/*/failure.json 2>/dev/null
```

`train_metrics.jsonl` 里要留意两项：`skipped`（读图失败等被跳过的样本数，应当是 0 或极少）和第一行的 `synced_init`（应当是 `true`）。

**常见失败**：

| 现象 | 原因与处理 |
|---|---|
| 启动即退出，提示 `Run already used` 或 `already started this run` | run ID 用过了，或调度器重试了同一个 worker。换一个新的 `SAMTOK_RUN_ID` 重新提交 |
| 提示 `Cannot write under …/runs` | NAS 配额满了。清理后重新提交 |
| 提示 `No training rows in …` / `No eval.json in …` | `SAMTOK_GROUND_DATA` 指错了 |
| 一个 worker 失败后其他 worker 也退出，日志里有 `Another worker failed` | 这是有意的：任一 worker 写了 `failure.json`，其余 worker 会在 15 秒内停下。看最先失败的那个 worker 的 `bootstrap/node<i>.log` |
| 训练报 `Ranks start from different weights` 或 `Ranks diverged` | 各卡权重一致性校验没过。不应出现；出现了请保留日志，不要用这次的权重 |
| `failure.json` 里写 `failed during environment` | 装环境或 CUDA 检查失败，看 `bootstrap/node<i>.log` 和 `bootstrap/node<i>-cuda/` |
| 卡在 rendezvous | 四个 worker 的 `ARNOLD_WORKER_HOSTS` 第一项应相同；确认作业确实是 4 个 worker 都启动了 |

失败的 run 目录保留着日志，不要手工往里补文件；修好后换新的 run ID 重跑。训练中途失败且已经有保存点时，可以用 `--init-adapter <该保存点>` 从那里起一个新的运行（优化器状态不继承）。

## 6. 这份入口验证到了什么程度

- 入口脚本以单机模式（`SAMTOK_NNODES=1`）从 GitHub 上的固定 commit `06e668d0` 完整跑通过（2026-10-13）：clone、装环境、训练、评测、出报告，脚本是按 ARNOLD 的方式从标准输入喂给 bash 的。见文档 03 的 5.4 节。
- 多机的 torchrun 参数与 v2 四机入口用的完全相同（v2 的四机运行都是这样启动的）。
- **v3 的训练和评测还没有在四机上实际跑过。** 第一次提交建议先跑第 4 节的冒烟测试。

## 7. 第二轮（阶段 B，C2）的四机运行（2026-10-18）

**与第一轮入口的区别**
- 入口脚本是 `scripts/round2/bootstrap_arnold.sh`，它调用 `scripts/round2/run_arnold.sh`。训练、评测、报告分别用 `round2-train`、`round2-eval`、`round2-report`。
- 数据目录变量是 `SAMTOK_ROUND2_DATA`，默认 `V3/data/round2_20261016`。clone 之前先检查这些都在：
  - 训练行（`--rows`）、快速评测项（`--quick-items`）、评测项（`--items`，默认 `eval_c1.json`）；
  - `edit_tokens.jsonl` 和 `pool/`；
  - `--init-adapter` 指向的权重。
- 报告 `analysis.txt` 由 `round2-report` 写出，与参考评测逐项配对比较，参考默认是原版、C1-1500、C1b-800。编辑一项看扣除背景漂移后"几乎没改"的比例、改动强度和区域外改动。
- 其余约定与第 1、5 节相同：run ID、clone 指定 commit、各节点自装环境、一个节点失败其余停下、产物目录。

### 7.1 运行设置（C2 正式运行）

```bash
# ===== 运行设置：C2 =====
export SAMTOK_RUN_ID=c2_4n_001             # 每次提交换新的
export SAMTOK_RUN_NAME=C2
export SAMTOK_EDIT_COMMIT=e681f238122ef1e3dcbd0d58c4f0a7cb7479688a   # 代码 commit，四个 worker 必须相同
export SAMTOK_EXPERIMENT=/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v3
export SAMTOK_ROUND2_DATA=$SAMTOK_EXPERIMENT/data/round2_20261016
# 32 卡 × accumulation 1 = batch 32，与 8 卡 × 4 每次更新用到的样本相同；3,522 次更新把 rows_c2.jsonl 的 112,704 行各用一次
export SAMTOK_TRAIN_ARGS="--rows rows_c2.jsonl --quick-items eval_c2_quick_idx.json --init-adapter $SAMTOK_EXPERIMENT/runs/c1b_local8_001/train/adapter_800 --updates 3522 --accumulation 1 --warmup 20 --lr 1e-4 --lr-schedule cosine --lr-min 1e-5 --heavy-weight 8 --checkpoints 500,1000,1500,2000,2500,3000,3500,3522 --quick-eval 0,500,1000,1500,2000,2500,3000,3500,3522"
export SAMTOK_EVAL_ARGS=""
export SAMTOK_REPORT_REFERENCES="stock=$SAMTOK_EXPERIMENT/baselines/round2_stock_eval C1=$SAMTOK_EXPERIMENT/runs/c1_local8_001/eval_1500 C1b=$SAMTOK_EXPERIMENT/runs/c1b_local8_001/eval_800"
```

实验设计见文档 08 第 9 节。

### 7.2 入口脚本（原样粘贴，不用改）

和 2.2 一样：把 7.1 和下面这段依次粘贴成 ARNOLD 的 worker 启动命令。

<!-- code: scripts/round2/bootstrap_arnold.sh -->
```bash
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
# Trainer / evaluator arguments (see `samtok-edit round2-train --help`, `round2-eval --help`).  Default: C2 (docs/08
# section 9), every row once: 3,522 updates x (32 GPUs x accumulation 1) = the 112,704 rows of rows_c2.jsonl.
export SAMTOK_TRAIN_ARGS="${SAMTOK_TRAIN_ARGS:---rows rows_c2.jsonl --quick-items eval_c2_quick_idx.json --init-adapter $SAMTOK_EXPERIMENT/runs/c1b_local8_001/train/adapter_800 --updates 3522 --accumulation 1 --warmup 20 --lr 1e-4 --lr-schedule cosine --lr-min 1e-5 --heavy-weight 8 --checkpoints 500,1000,1500,2000,2500,3000,3500,3522 --quick-eval 0,500,1000,1500,2000,2500,3000,3500,3522}"
export SAMTOK_EVAL_ARGS="${SAMTOK_EVAL_ARGS:-}"
# Reference evaluations for the paired report, "LABEL=PATH ..." with the base model first.
export SAMTOK_REPORT_REFERENCES="${SAMTOK_REPORT_REFERENCES:-stock=$SAMTOK_EXPERIMENT/baselines/round2_stock_eval C1=$SAMTOK_EXPERIMENT/runs/c1_local8_001/eval_1500 C1b=$SAMTOK_EXPERIMENT/runs/c1b_local8_001/eval_800}"
# Phases to run: train,eval,report.  Without "train", set SAMTOK_ADAPTER to the adapter directory to evaluate.
export SAMTOK_PHASES="${SAMTOK_PHASES:-train,eval,report}"
export SAMTOK_ADAPTER="${SAMTOK_ADAPTER:-}"
# Code: branch and (recommended) the exact commit, so all workers and any later rerun use the same code.
export SAMTOK_EDIT_REPO_URL="${SAMTOK_EDIT_REPO_URL:-https://github.com/Tangent0308/samtok_edit.git}"
export SAMTOK_EDIT_BRANCH="${SAMTOK_EDIT_BRANCH:-qwen-image-2.1-v3}"
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
```
<!-- /code -->

### 7.3 冒烟测试（正式提交前，约 30 分钟）

```bash
# ===== 运行设置：四机冒烟 =====
export SAMTOK_RUN_ID=c2_smoke_4n_001
export SAMTOK_RUN_NAME=smoke
export SAMTOK_EDIT_COMMIT=e681f238122ef1e3dcbd0d58c4f0a7cb7479688a
export SAMTOK_TRAIN_ARGS="--rows rows_c2.jsonl --quick-items eval_c2_quick_idx.json --init-adapter /mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v3/runs/c1b_local8_001/train/adapter_800 --updates 3 --accumulation 1 --warmup 1 --lr-schedule cosine --heavy-weight 8 --checkpoints 3 --quick-eval 3 --quick-limit 1"
export SAMTOK_EVAL_ARGS="--limit 2"
```

- 后面接 7.2 的入口脚本。
- 跑通的标志是 `runs/c2_smoke_4n_001/SUCCESS.json` 和 `analysis.txt` 都生成了。数值没有意义，只看流程。

### 7.4 预计耗时

| 环节 | 耗时 |
|---|---|
| 装环境、加载模型 | 约 10–20 分钟 |
| 训练 | 每次更新每卡 1 行，约 8–9 秒（本机每卡 4 行约 28 秒；四机之间同步梯度约 350 MB，可忽略）；3,522 次约 8–9 小时 |
| 快速评测 | 9 次，每次约 1 分钟 |
| 完整评测 | 1,320 项分到 32 卡，约 15 分钟 |
| 报告 | 1 分钟 |
| **合计** | **约 9–10 小时** |

本机 8 卡跑同样的配方约 30 小时。

### 7.5 看进度

```bash
R=/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v3/runs/c2_4n_001
cat $R/nodes/*/phase; tail -n 2 $R/train/train_metrics.jsonl; cat $R/train/quick_summary.jsonl
ls $R/nodes/*/failure.json 2>/dev/null
```

`train_metrics.jsonl` 每 10 次更新一行，要留意三项：
- `skipped`：应当是 0 或极少；
- `kinds`：七类训练行的数量；
- 第一行的 `synced_init`：应当是 `true`。

### 7.6 这份入口验证到了什么程度

- **单机完整跑通（2026-10-18）。** 入口以单机模式（`SAMTOK_NNODES=1`，8 卡 × accumulation 4）从 GitHub 上的固定 commit `e681f238` 跑完了全流程：
  - clone、装环境；
  - 从 C1b 接续训练 2 次更新：64 个样本无跳过，七类训练行中出现了五类，两个路由校准与 C1b 相同；
  - 快速评测、少量项的完整评测、对比报告、`SUCCESS.json` 都已生成；
  - run 目录 `V3/runs/c2_entry_check_1n_001/`，续训状态文件已删除。
- **多机。** torchrun 参数与第一轮入口相同。
- **四机实测：没有。** 第二轮的训练和评测还没有在四机上跑过，7.4 的耗时是按单机速度推算的。第一次提交建议先跑 7.3 的冒烟测试。
