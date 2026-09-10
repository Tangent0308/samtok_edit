# SAMTokEdit 四机 32 卡训练运行指南

本文说明如何在 Arnold 集群上从 Git clone 开始，建立 uv 环境并顺序完成 refined Stage 1 和
Stage 2 四机训练。当前四机拓扑固定为 4 台机器、每台 8 张 GPU，共 32 个训练进程。

## 1. 最短运行方式

推荐把仓库中的 `scripts/train/bootstrap_arnold_4node.sh` 作为 Arnold 四台 worker 的共同入口。
同一份入口必须在四台机器上同时执行，不能只在 worker 0 上执行。

在提交任务前，在 Arnold 入口脚本顶部的用户填写区填入：

```bash
export WANDB_API_KEY=""
export WANDB_ENTITY="2200012743-peking-university"
export WANDB_PROJECT="samtok-edit"
export SAMTOK_RUN_ID=""  # 必填：每次提交都使用全新的唯一名称
```

Arnold 会自动提供：

```text
ARNOLD_WORKER_HOSTS   四台 worker 的 host:port 列表
ARNOLD_WORKER_NUM     必须为 4
ARNOLD_WORKER_GPU     必须为 8
ARNOLD_ID             当前节点编号，取值 0/1/2/3
ARNOLD_WORKER_0_HOST  可选；worker 0 的地址
```

然后让每台 worker 执行：

```bash
bash /path/available-before-clone/bootstrap_arnold_4node.sh
```

这里的 `/path/available-before-clone/bootstrap_arnold_4node.sh` 必须在 Git clone 之前就可访问。
有两种常见做法：

1. 将 `scripts/train/bootstrap_arnold_4node.sh` 的完整内容直接粘贴到 Arnold 的入口脚本；
2. 提交任务前把该文件复制到四台机器都可见的共享路径，再从共享路径执行。

不要写成下面这样：

```bash
bash /尚未克隆的仓库/scripts/train/bootstrap_arnold_4node.sh
```

因为 bootstrap 本身负责 clone，执行它之前目标仓库还不存在。

## 2. 运行前必须确认

### 2.1 远端分支包含四机实现

bootstrap 默认 clone：

```text
repository: https://github.com/Tangent0308/samtok_edit.git
branch:     dev_crispedit_refined
```

因此必须先把以下文件提交并推送到 `origin/dev_crispedit_refined`：

```text
setup_env.sh
pyproject.toml
scripts/train/arnold_4node_env.sh
scripts/train/bootstrap_arnold_4node.sh
scripts/train/launch_4node.sh
scripts/train/run_arnold_4node_pipeline.sh
scripts/data/prepare_4node_metadata.sh
```

可以在提交任务前检查远端分支：

```bash
git fetch origin dev_crispedit_refined
git ls-tree -r --name-only origin/dev_crispedit_refined | \
  grep -E '(^setup_env.sh$|^scripts/(train|data)/.*4node.*\.sh$)'
```

如果远端没有这些文件，从 Git clone 启动的任务一定会失败；本机工作区里存在文件并不等于远端
分支已经包含它们。

### 2.2 共享文件系统

当前 bootstrap 假定 `/mnt/bn/strategy-mllm-train` 在四台机器上可见，并让 node 0 只执行一次
clone 和 uv 安装，其他节点等待共享 marker。以下内容必须对四台机器可见：

- clone 后的仓库和 `.venv`；
- refined Stage 1/2 component metadata 与图片；
- Qwen-Image-Edit-2511、SAMTok-gres-ft 和 merged TE；
- `RUN_ROOT` 下的 metadata、cache、checkpoint、日志和控制 marker。

如果集群的这些路径不是共享存储，不能直接使用当前 bootstrap。

### 2.3 默认输入和模型路径

```text
Stage 1 source data
/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/crispedit_refined/
stage1_full/data/crispedit_samtok

Stage 2 source data
/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/crispedit_refined/
stage2_full/data/crispedit_samtok

Qwen-Image-Edit-2511（注意是 2511）
/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen-Image-Edit-2511

SAMTok TE（注意是 gres-ft）
/mnt/bn/strategy-mllm-train/user/tanyue/models/SAMTok/Qwen2.5-VL-7B-SAMTok-gres-ft

Merged SAMTok TE processor/model
/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/artifacts/merged_samtok_te
```

这些路径在四台机器上都必须可读。脚本设置 `DIFFSYNTH_SKIP_DOWNLOAD=True`，训练过程中不会尝试
在线补下载缺失模型。

### 2.4 每次使用全新的 run ID

`SAMTOK_RUN_ID` 必须只包含字母、数字、点、下划线或连字符。默认输出路径为：

```text
/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/
crispedit_refined_4node/$SAMTOK_RUN_ID
```

bootstrap 的 clone 路径默认为：

```text
/mnt/bn/strategy-mllm-train/user/tanyue/workspaces/samtok_edit_$SAMTOK_RUN_ID
```

不显式设置时，脚本生成 `crispedit-refined-4node-<Arnold job id>`；没有 job id 时使用 master
port 作为后缀。脚本有防覆盖门禁：除本次 bootstrap 自己预建的日志和控制文件外，clone 目标
或 `RUN_ROOT` 已存在时会拒绝运行。重新提交实验时应换一个新的 `SAMTOK_RUN_ID`，不能把一个
旧实验目录直接当成新实验复用。

## 3. 可直接提交的 Arnold 完整入口

下面是从裸 worker 开始的完整执行顺序，不依赖事先存在的 SAMTokEdit 工作目录。需要把这段内容
作为四个 Arnold worker 的共同入口。它明确完成：系统依赖 → 直连 Git clone → 清除环境中可能残留的代理 →
安装 uv → `cd` 仓库 → `setup_env.sh` 安装环境 → 激活环境 → Stage 1/2 四机训练。

先在下面的“用户填写区”填入 W&B 账户信息和唯一实验名。这里不要保留空字符串，也不要使用
shell 尖括号占位符；应把本次实际值直接写在双引号内。

```bash
#!/usr/bin/env bash
set -euo pipefail

# ----- 0. 用户填写区 -----
# API key 和每次实验的唯一名称需要自己填写。
export WANDB_API_KEY=""
export SAMTOK_RUN_ID=""

# 与先前实验保持一致的 W&B 默认值，通常无需修改。
export WANDB_ENTITY="2200012743-peking-university"
export WANDB_PROJECT="samtok-edit"

# ----- 1. Arnold 和必填值检查 -----
: "${ARNOLD_WORKER_HOSTS:?Arnold must provide ARNOLD_WORKER_HOSTS}"
: "${ARNOLD_WORKER_NUM:?Arnold must provide ARNOLD_WORKER_NUM}"
: "${ARNOLD_WORKER_GPU:?Arnold must provide ARNOLD_WORKER_GPU}"
: "${ARNOLD_ID:?Arnold must provide ARNOLD_ID}"
: "${WANDB_API_KEY:?Fill WANDB_API_KEY in the user settings section}"
: "${SAMTOK_RUN_ID:?Fill SAMTOK_RUN_ID in the user settings section}"

# 必须是 4 机、每机 8 卡。
[[ "$ARNOLD_WORKER_NUM" == "4" ]] || { echo "ARNOLD_WORKER_NUM must be 4" >&2; exit 2; }
[[ "$ARNOLD_WORKER_GPU" == "8" ]] || { echo "ARNOLD_WORKER_GPU must be 8" >&2; exit 2; }
NODE_RANK="$ARNOLD_ID"

# Arnold 的通用 PORT 在不同 worker 上可能不同，它不是 torch rendezvous
# 端口。标准入口不接受外部残留端口，统一让四机代码解析
# ARNOLD_WORKER_HOSTS 第一项中的共享端口。
unset MASTER_PORT PORT

# ----- 2. 共享仓库、环境和实验路径 -----
export SAMTOK_EDIT_REPO_URL="https://github.com/Tangent0308/samtok_edit.git"
export SAMTOK_EDIT_BRANCH="dev_crispedit_refined"
export SAMTOK_EDIT_REPO_DIR="/mnt/bn/strategy-mllm-train/user/tanyue/workspaces/samtok_edit_${SAMTOK_RUN_ID}"
export SAMTOK_EDIT_VENV="${SAMTOK_EDIT_REPO_DIR}/.venv"
export RUN_ROOT="/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/crispedit_refined_4node/${SAMTOK_RUN_ID}"
export BOOTSTRAP_CONTROL="${RUN_ROOT}/bootstrap_control"

mkdir -p "${RUN_ROOT}/logs" "$BOOTSTRAP_CONTROL"
exec > >(tee -a "${RUN_ROOT}/logs/bootstrap.node${NODE_RANK}.log") 2>&1

# ----- 3. 每台 worker 安装系统依赖 -----
sudo apt-get install ffmpeg libsm6 libxext6 tmux htop -y

# ----- 4. 默认直连 GitHub -----
# 已在实际 Arnold worker 验证 git ls-remote 可直连成功。
# 不要设置 sys-proxy-rd-relay.byted.org:8118，该地址在本次 worker 不可达。
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

if [[ "$NODE_RANK" == "0" ]]; then
  # 共享文件系统上只能由 node 0 clone/安装，避免四台机器同时修改目录。
  if [[ -e "$SAMTOK_EDIT_REPO_DIR" ]]; then
    printf '%s\n' "repository already exists: $SAMTOK_EDIT_REPO_DIR" \
      >"$BOOTSTRAP_CONTROL/environment.failed"
    exit 1
  fi
  mkdir -p "$(dirname "$SAMTOK_EDIT_REPO_DIR")"
  if ! git clone --branch "$SAMTOK_EDIT_BRANCH" --single-branch \
    "$SAMTOK_EDIT_REPO_URL" "$SAMTOK_EDIT_REPO_DIR"; then
    printf '%s\n' "git clone failed while using direct GitHub access" \
      >"$BOOTSTRAP_CONTROL/environment.failed"
    exit 1
  fi

  # ----- 5. clone 完成后清除所有代理 -----
  unset http_proxy
  unset https_proxy
  unset HTTP_PROXY
  unset HTTPS_PROXY
  unset no_proxy
  unset NO_PROXY

  # ----- 6. 明确安装 uv -----
  python3.11 -m pip install --user \
    --index-url https://bytedpypi.byted.org/simple/ \
    'uv==0.11.32'
  UV_BIN="$(python3.11 -c 'import site; print(site.getuserbase())')/bin/uv"

  # ----- 7. 进入仓库，用 uv + pyproject.toml 安装完整训练环境 -----
  cd "$SAMTOK_EDIT_REPO_DIR"
  UV_BIN="$UV_BIN" \
  SAMTOK_EDIT_VENV="$SAMTOK_EDIT_VENV" \
  SAMTOK_EDIT_INDEX=https://bytedpypi.byted.org/simple/ \
  SAMTOK_EDIT_UV_VERSION=0.11.32 \
  SAMTOK_EDIT_REQUIRE_CUDA=1 \
  SAMTOK_EDIT_RUN_TESTS=1 \
    bash setup_env.sh

  git rev-parse HEAD >"$BOOTSTRAP_CONTROL/git_commit.txt"
  touch "$BOOTSTRAP_CONTROL/environment.ok"
else
  # 其他 worker 不使用网络安装，立即清除代理并等待 node 0。
  unset http_proxy
  unset https_proxy
  unset HTTP_PROXY
  unset HTTPS_PROXY
  unset no_proxy
  unset NO_PROXY
fi

# ----- 8. 四台 worker 同步 -----
START_SECONDS=$SECONDS
until [[ -f "$BOOTSTRAP_CONTROL/environment.ok" ]]; do
  if [[ -f "$BOOTSTRAP_CONTROL/environment.failed" ]]; then
    cat "$BOOTSTRAP_CONTROL/environment.failed" >&2
    exit 1
  fi
  if (( SECONDS - START_SECONDS >= 7200 )); then
    echo "Timed out waiting for node 0 environment setup" >&2
    exit 1
  fi
  sleep 2
done

# ----- 9. 四台 worker 都进入 clone 后的仓库，激活环境并启动训练 -----
cd "$SAMTOK_EDIT_REPO_DIR"
source "$SAMTOK_EDIT_VENV/bin/activate"
export SAMTOK_ALLOW_BOOTSTRAP_RUN_ROOT=1
export NCCL_DEBUG=INFO
bash scripts/train/run_arnold_4node_pipeline.sh
```

这段是为了完整展示入口步骤。仓库中实际维护的
`scripts/train/bootstrap_arnold_4node.sh` 还包含更严格的参数校验、IPv6 地址解析、原子 marker 和
失败状态记录。实际提交时，应将该 bootstrap 文件的完整内容粘贴为 Arnold entry；上面的代码
用于明确说明它不是一个只会调用未 clone 仓库的 wrapper。

必填的只有 `WANDB_API_KEY` 和每次实验唯一的 `SAMTOK_RUN_ID`。
`WANDB_ENTITY=2200012743-peking-university` 和 `WANDB_PROJECT=samtok-edit` 已默认填入。

## 4. bootstrap 实际执行了什么

`scripts/train/bootstrap_arnold_4node.sh` 的执行顺序如下：

1. 验证 Arnold 确实分配了 4 台机器和每台 8 卡；
2. 从 `ARNOLD_WORKER_HOSTS` 第一项解析 rendezvous address/port，并设置
   `MASTER_ADDR`、`MASTER_PORT`、`NNODES=4`、`NODE_RANK=$ARNOLD_ID` 和
   `GPUS_PER_NODE=8`；
3. 每台机器执行 `sudo apt-get install ffmpeg libsm6 libxext6 tmux htop -y`，可用
   `SAMTOK_SKIP_APT=1` 跳过；
4. 先清除可能残留的 proxy，只有 node 0 通过已实测可用的 GitHub 直连执行
   `git clone --branch dev_crispedit_refined --single-branch ...`；
5. clone 后再次明确 `unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY`；
6. node 0 执行 `python3.11 -m pip install --user ... uv==0.11.32`，然后 `cd` 进入 clone 后的仓库；
7. node 0 执行 `bash setup_env.sh`，使用 uv + `pyproject.toml` 创建共享 `.venv`，不生成
   `uv.lock`；
8. 环境安装后运行 CUDA、包版本、vendored DiffSynth 来源和单元测试检查；
9. node 1--3 等待 `environment.ok`，不会同时修改共享 Git 仓库或 `.venv`；
10. 四台机器都 `cd` 进入仓库、激活 `.venv`，共同执行 `run_arnold_4node_pipeline.sh`。

从确定 `RUN_ROOT` 开始，每个 worker 的 stdout/stderr 都由 `tee` 完整保存到：

```text
$RUN_ROOT/logs/bootstrap.node0.log
$RUN_ROOT/logs/bootstrap.node1.log
$RUN_ROOT/logs/bootstrap.node2.log
$RUN_ROOT/logs/bootstrap.node3.log
```

因此 apt、clone、uv 安装、环境检查、metadata、训练和后处理的终端输出都能在实验结果目录中
回看。bootstrap 的跨节点 marker 也存放在 `$RUN_ROOT/bootstrap_control/`，不再写到实验目录
之外的隐藏路径。

环境固定检查的核心版本包括 PyTorch 2.8.0 + CUDA 12.8、torchvision 0.23.0、Transformers
5.12.1、Accelerate 1.14.0 和 DiffSynth 2.1.2。`DiffSynth-Studio` 使用仓库内普通目录的
vendored 源码并以 editable 方式安装，不会作为嵌套 Git 仓库更新。

## 5. 四机拓扑如何映射到 Accelerate

`scripts/train/arnold_4node_env.sh` 最终为每个分布式阶段执行：

```text
accelerate launch
  --multi_gpu
  --mixed_precision no
  --dynamo_backend no
  --num_processes 32
  --num_machines 4
  --machine_rank $ARNOLD_ID
  --main_process_ip $MASTER_ADDR
  --main_process_port $MASTER_PORT
  --rdzv_backend static
  --same_network
```

其中 `--num_processes=32` 是四机全局进程数，不是每台机器的进程数。每台机器使用本地
`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`，因此每台启动 8 个 rank。

地址解析优先级为：

```text
MASTER_ADDR: 显式 MASTER_ADDR > ARNOLD_WORKER_0_HOST > ARNOLD_WORKER_HOSTS 第一项
MASTER_PORT: 显式且四机相同的 MASTER_PORT > ARNOLD_WORKER_HOSTS 第一项中的端口
```

同时支持 `host:port` 和 `[IPv6]:port`。脚本有意忽略 Arnold 的通用 `PORT`：该变量是
worker-local service port，四台机器可能得到四个不同值，不能用作 torch rendezvous port。

在构造 metadata 或加载模型之前，每台机器会分别写出：

```text
$RUN_ROOT/reports/topology.node0.json
$RUN_ROOT/reports/topology.node1.json
$RUN_ROOT/reports/topology.node2.json
$RUN_ROOT/reports/topology.node3.json
```

node 0 在 300 秒内收齐四份报告后，逐项核对 `run_id`、`nnodes`、`gpus_per_node`、
`world_size`、`master_addr` 和 `master_port`。只有完全一致时才生成
`$RUN_ROOT/reports/topology.json` 和 `control/topology_consensus.ok` 并继续训练；否则会写入
`control/topology_consensus.node0.failed` 后快速退出，不再等待 Accelerate 15 分钟超时。

## 6. 完整训练流水线

### 6.1 构造 ws32 metadata

node 0 复用已经构建并验收的 refined component JSONL 和图片，重新组织适合 world size 32 的
metadata；它不会重新运行 mask codec。构造和校验结果为：

```text
Stage 1: edit_mt=42,368, edit_ntp=21,184, edit=10,592, edit_umt=10,592
         total=84,736，比例 4:2:1:1

Stage 2: edit_mt=42,368, edit=21,184, edit_umt=21,184
         total=84,736，比例 2:1:1
```

输出：

```text
$RUN_ROOT/data/stage1_ws32.jsonl
$RUN_ROOT/data/stage2_ws32.jsonl
$RUN_ROOT/logs/metadata/
$RUN_ROOT/reports/metadata/
```

构建器会检查 ASCII English、图片路径/抽样解码、Stage 1 同步 schedule、Stage 2 各 rank 比例
以及 metadata SHA256。任一检查失败都不会开始训练。

### 6.2 Stage 1：训练 TE LoRA

四机 Stage 1 保持当前方案：

```text
model                 Qwen-Image-Edit-2511 + SAMTok-gres-ft TE
trainable             text encoder LoRA
LoRA rank/dropout     64 / 0.05
sample ratio          edit_mt:edit_ntp:edit:edit_umt = 4:2:1:1
epochs                1
gradient accumulation 8
learning rate         4e-5
weight decay          0.05
warmup ratio          0.05
lambda NTP / FM       0.05 / 1.0
precision             bf16
gradient checkpoint   enabled
zero_cond_t           enabled
effective global batch 32 ranks * 1 * 8 = 256
```

每 rank 消费 2,648 个 micro-batch，对应 331 次 optimizer update。trainer 的 checkpoint 名称
按每 rank dataloader step 计数，因此流水线选择的最终权重是：

```text
$RUN_ROOT/stage1_te_lora/step-2648.safetensors
```

不要把文件名 `step-2648` 误解为 2,648 次 optimizer update。

### 6.3 Stage 2a：生成 cache

四机共同加载 Stage 1 最终 TE LoRA，为 84,736 条 Stage 2 metadata 生成训练 cache。该阶段关闭
W&B，不训练参数，输出：

```text
$RUN_ROOT/stage2_cache
```

cache 由 32 个 rank 共同生成；不能复用之前单机 8 卡、sidecar 中记录 `world_size=8` 的 cache。

### 6.4 Stage 2 cache 强审计

Stage 2a 完成后只有 node 0 使用默认 32 个 CPU worker 执行全量审计，其他节点等待：

```text
expected total      84,736
expected counts     edit_mt=42,368, edit=21,184, edit_umt=21,184
expected world size 32
expected TE LoRA    本次 Stage 1 step-2648
```

审计报告必须为 `passed=true` 才允许进入 Stage 2b：

```text
$RUN_ROOT/reports/stage2_cache_audit.json
```

### 6.5 Stage 2b：训练 DiT LoRA

```text
trainable             DiT LoRA
LoRA rank             32
physical cache rows   84,736
dataset repeat        2
epochs                1
gradient accumulation 1
learning rate         1e-4
weight decay          0.01
precision             bf16
gradient checkpoint   enabled
zero_cond_t           enabled
effective global batch 32
```

总消费量为 `84,736 * 2 = 169,472`，每 rank 5,296 step，最终权重为：

```text
$RUN_ROOT/stage2_dit_lora/step-5296.safetensors
```

完成后 node 0 生成：

```text
$RUN_ROOT/reports/run_manifest.json
```

其中记录 world size、两阶段 metadata/checkpoint 路径和 SHA256、cache 审计路径与行数。

## 7. 日志和进度查看

定义本次输出根目录：

```bash
export SAMTOK_RUN_ID=<本次实际填写的唯一实验名>
export RUN_ROOT=/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/crispedit_refined_4node/$SAMTOK_RUN_ID
```

查看 node 0 当前训练日志：

```bash
tail -F "$RUN_ROOT/logs/bootstrap.node0.log"
tail -F "$RUN_ROOT/logs/stage1_train.node0.log"
tail -F "$RUN_ROOT/logs/stage2_cache.node0.log"
tail -F "$RUN_ROOT/logs/stage2_cache_audit.node0.log"
tail -F "$RUN_ROOT/logs/stage2_train.node0.log"
```

每个分布式阶段都会产生 `node0`--`node3` 四份日志。检查所有节点：

```bash
ls -lh "$RUN_ROOT/logs"/*.node*.log
tail -n 100 "$RUN_ROOT/logs"/stage1_train.node*.log
```

查看流水线 marker：

```bash
find "$RUN_ROOT/control" -maxdepth 1 -type f -printf '%f\n' | sort
```

成功阶段会出现 `.node0.done`--`.node3.done` 和对应 `.ok`；任一 `.failed` 都表示流水线已停止，
文件内容为退出码。查看拓扑和最终产物：

```bash
cat "$RUN_ROOT/reports/topology.json"
cat "$RUN_ROOT/reports/metadata/metadata_manifest.json"
cat "$RUN_ROOT/reports/stage2_cache_audit.json"
cat "$RUN_ROOT/reports/run_manifest.json"
```

W&B 会创建两个独立 run：

```text
$SAMTOK_RUN_ID-stage1
$SAMTOK_RUN_ID-stage2
```

Stage 2 cache 构建和 cache audit 不记录 W&B 曲线。

四机训练使用固定的 `byted-wandb==0.13.98`。该版本的 subprocess service 只有 30 秒固定启动
等待时间；在 32 rank 同时加载模型时可能因启动过慢而让 global rank 0 报
`assert ports_found`，随后其他 rank 才出现 TCPStore/NCCL watchdog 连锁错误。四机 launcher 因此
专门设置：

```text
WANDB_DISABLE_SERVICE=true
WANDB_START_METHOD=thread
```

并向两个训练阶段传入 `--eager_init_loggers`。global rank 0 会在读取 dataset 和模型权重之前初始化
W&B/CSV logger，再通过 distributed all-reduce 把初始化结果同步给全部 32 个 rank。成功日志必须
先出现：

```text
[SamtokLogger] eager initialization passed before model loading
```

如果 W&B 初始化失败，所有 rank 会在模型加载前一致退出；单机训练入口不传该开关，行为不变。

## 8. 常用覆盖项

以下变量可在 Arnold job environment 中覆盖：

```text
SAMTOK_EDIT_REPO_URL / SAMTOK_EDIT_BRANCH
SAMTOK_EDIT_REPO_DIR / SAMTOK_EDIT_VENV / RUN_ROOT
SAMTOK_SKIP_APT / SAMTOK_GIT_HTTP_PROXY
STAGE1_SOURCE_BASE / STAGE2_SOURCE_BASE / MERGED_TE_DIR
QWEN_2511 / SAMTOK_TE
STAGE1_DATASET_WORKERS / STAGE2_CACHE_DATASET_WORKERS / STAGE2_TRAIN_DATASET_WORKERS
CACHE_AUDIT_WORKERS
STAGE1_SAVE_STEPS / STAGE2_SAVE_STEPS
STAGE1_LEARNING_RATE / STAGE2_LEARNING_RATE
STAGE1_WEIGHT_DECAY / STAGE2_WEIGHT_DECAY
NTP_LOSS_WEIGHT / FM_LOSS_WEIGHT
NCCL_DEBUG / TORCH_NCCL_ASYNC_ERROR_HANDLING
BOOTSTRAP_TIMEOUT_SECONDS / WAIT_TIMEOUT_SECONDS / TOPOLOGY_TIMEOUT_SECONDS
```

除非明确设计新的实验，不建议改 world size、数据比例、epoch、repeat 或 gradient accumulation。
流水线的最终 checkpoint 门禁固定期待 `step-2648` 和 `step-5296`；改变这些训练语义后必须同步
修改 checkpoint 选择与最终验收，不能只覆盖环境变量。

## 9. 分阶段手工运行

完整流水线是推荐方式。如果仓库和 uv 环境已经位于四机共享目录，也可以让四台 worker 同时
执行某个 phase：

```bash
cd /path/to/samtok_edit
export SAMTOK_EDIT_VENV=/path/to/samtok_edit/.venv
export WANDB_API_KEY=""
export WANDB_ENTITY=2200012743-peking-university
export WANDB_PROJECT=samtok-edit

# Arnold 仍需在每台机器注入 ARNOLD_WORKER_HOSTS、ARNOLD_WORKER_NUM=4、
# ARNOLD_WORKER_GPU=8 和各自不同的 ARNOLD_ID。

DATASET_BASE=<stage1-base> \
STAGE1_METADATA=<stage1-ws32-jsonl> \
OUTPUT_PATH=<stage1-output> \
MERGED_TE_DIR=<merged-te> \
bash scripts/train/launch_4node.sh stage1
```

Stage 2 cache phase：

```bash
DATASET_BASE=<stage2-base> \
STAGE2_METADATA=<stage2-ws32-jsonl> \
OUTPUT_PATH=<stage2-cache-output> \
TE_LORA_PATH=<stage1-final-safetensors> \
MERGED_TE_DIR=<merged-te> \
bash scripts/train/launch_4node.sh stage2_cache
```

Stage 2 train phase：

```bash
CACHE_ROOT=<audited-stage2-cache> \
OUTPUT_PATH=<stage2-dit-lora-output> \
MERGED_TE_DIR=<merged-te> \
bash scripts/train/launch_4node.sh stage2_train
```

每条 phase 命令仍然必须由四台机器同时执行。手工模式不会替你完成阶段间 marker 协调、cache
审计或最终 manifest；除非在恢复故障，优先使用完整 pipeline。

## 10. 故障处理

- clone/setup 失败：检查 `$RUN_ROOT/bootstrap_control/environment.failed`、
  `$RUN_ROOT/logs/bootstrap.node<N>.log` 和 Arnold 四个 worker 的入口日志。
- rendezvous/NCCL 失败：确认四台机器得到完全相同的 `MASTER_ADDR:MASTER_PORT`、
  `ARNOLD_WORKER_NUM=4`、`ARNOLD_WORKER_GPU=8`，且 `ARNOLD_ID` 分别为 0--3。优先查看
  `reports/topology.node<N>.json`；如果只有 node 0 加入并报 `1/4 clients joined`，通常就是误把
  四台机器各自不同的通用 `PORT` 当成了 `MASTER_PORT`。当前实现已忽略通用 `PORT`。
- 某个 node 提前退出：查看 `$RUN_ROOT/control/*.failed`，再打开同名阶段对应的
  `$RUN_ROOT/logs/<stage>.node<N>.log`。
- cache audit 失败：不要启动 Stage 2b；以 `reports/stage2_cache_audit.json` 中第一批 error 为准
  排查，不要手工伪造 `.ok` marker。
- `RUN_ROOT already exists`：当前完整 pipeline 不提供透明断点续跑。保留旧目录用于诊断，并用
  新 `SAMTOK_RUN_ID` 重跑；如必须从某一阶段恢复，人工核验已有 checkpoint/cache 后再使用第
  9 节的 phase 入口。
- W&B 失败：确认入口顶部的 `WANDB_API_KEY`/`WANDB_ENTITY` 已填写，且
  `WANDB_ENTITY/WANDB_PROJECT` 在四节点一致。如果 global rank 0 出现 `assert ports_found` 或
  `/tmp/.../port-<pid>` 不存在，说明运行的 clone 尚未包含 thread-backend/eager-init 修复；检查
  `$RUN_ROOT/bootstrap_control/git_commit.txt`，并确认启动日志在模型加载前出现
  `[SamtokLogger] eager initialization passed before model loading`。不要通过增加 NCCL watchdog
  timeout 掩盖该错误。

## 11. 完成判据

只有以下条件同时成立，才算两阶段四机训练完成：

1. `$RUN_ROOT/control/finalize.ok` 存在；
2. `$RUN_ROOT/control/topology_consensus.ok` 存在，且 `reports/topology.json` 包含四台一致的
   rendezvous 拓扑；
3. Stage 1 最终权重 `$RUN_ROOT/stage1_te_lora/step-2648.safetensors` 存在且非空；
4. Stage 2 cache audit 报告 `passed=true`；
5. Stage 2 最终权重 `$RUN_ROOT/stage2_dit_lora/step-5296.safetensors` 存在且非空；
6. `$RUN_ROOT/reports/run_manifest.json` 存在并记录正确的 SHA256；
7. Stage 1 和 Stage 2 四个 node 日志均无 Traceback、OOM、NCCL error 或 non-finite loss；
8. 两个 W&B run 都正常结束并同步完整训练曲线；
9. `$RUN_ROOT/logs/bootstrap.node0.log`--`bootstrap.node3.log` 均存在，可完整回溯四个 worker 的
   环境构建与训练输出。

本流水线只完成 Stage 1/Stage 2 训练和产物审计，不会自动运行 ScaleEdit inference 评测。
