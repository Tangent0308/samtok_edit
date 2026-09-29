# Qwen-Image-2.1 全量训练数据与正式四机入口

本文件对应 `qwen-image-2.1-dev` 分支。数据根目录为：

```text
/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/qwen21_full4_20260928/data
```

源图像、目标图像和数据集原始 mask 已经按质量筛选后物化在 `assets/`，路径写在 `sources.jsonl`。训练 metadata 和 SAMTok 编码放在独立版本目录：

```text
data/train_full_9b_rules_003/
  inputs.json              # 输入 ID 覆盖、语义转换 hash、失败 plain-only 记录
  encoded/worker-00..07/  # 8 卡可断点复用的 SAMTok 编码分块和收据
  stage1.jsonl             # edit_ntp + plain + UMT-ref + UMT-noref
  stage2.jsonl             # plain + UMT-ref + UMT-noref
  provenance.jsonl         # 每行 metadata 到源 ID/数据集/转换方式的绑定
  metadata_report.json     # 行数、各来源统计、SHA256
  regions/                 # stage1 对应的冻结区域 coverage cache（按内容去重）
```

## 数据筛选与字段

最终质量通过的源编辑对共 98,574 条：RefEdit 7,804、CrispEdit 37,728、ScaleEdit 25,085、Derived 27,957。前三个数据集同时保留其已有 aggregate mask PNG；多编辑样本只按已有 `instance_id` 读取 RLE。Derived 使用其单个 `mask_rle`。代码不会预测、重算或合并 mask。

语义转换使用生产 run `_003`：

```text
data/semantic_runs/qwen21_noref9b_rules_4n_full_003/
```

其中 97,361 条通过（LLM 96,593、rule fallback 768），1,213 条失败。失败行沿用原 `instruction` 生成唯一 plain 编辑行；它们不会生成 NTP、ref 或 noref。通过行生成四行：

```text
edit_ntp     原始 instruction + 由官方 SAMTok codec 生成的 mt_cot
edit         原始 instruction + source/target 图像
edit_umt/ref 原始 instruction 中插入 mask span
edit_umt/noref 语义转换后的 noref instruction 中插入同一 mask span
```

因此最终预期行数为 Stage 1 `97,361×4 + 1,213 = 390,657`，Stage 2 `97,361×3 + 1,213 = 293,296`。`provenance.jsonl` 保存每条训练行的 source ID、dataset、edit type、sample kind 和转换方式；这些字段不进入模型协议。

## 生成与验证命令

以下命令在仓库环境中执行。小批量验证使用 `--sample-per-dataset 2` 写入 `/tmp`，已确认四个来源均生成四种结构合法行，并确认区域 cache 与旧预处理的 tensor 完全一致。

```bash
export DATA_ROOT=/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/qwen21_full4_20260928/data
export TRAIN_DATA=$DATA_ROOT/train_full_9b_rules_003
export SAMTOK_CODEC=/mnt/bn/strategy-mllm-train/user/tanyue/models/SAMTok/Qwen3-VL-8B-SAMTok

python -m samtok_edit21.full_training_data split \
  --source-root "$DATA_ROOT" \
  --semantic-run "$DATA_ROOT/semantic_runs/qwen21_noref9b_rules_4n_full_003" \
  --output "$TRAIN_DATA" --workers 8

# 需要在 8 张 GPU 上各执行一个 worker；CUDA_VISIBLE_DEVICES 使用 0..7。
CUDA_VISIBLE_DEVICES=0 python -m samtok_edit21.full_training_data encode-worker \
  --output "$TRAIN_DATA" --rank 0 --codec-root "$SAMTOK_CODEC" --device cuda:0 --batch-size 16

# rank 0..7 全部 complete.json 产生后：
python -m samtok_edit21.full_training_data merge --output "$TRAIN_DATA"
```

区域监督缓存使用 stage1 的 metadata 和同一 codec。这里使用 16 个本地预处理 shard，轮流复用 8 张 GPU（`rank % 8`）；它们只负责生成训练前的冻结 coverage，不改变正式训练的 4 机 × 8 卡拓扑：

```bash
for rank in $(seq 0 15); do
  gpu=$((rank % 8))
  CUDA_VISIBLE_DEVICES="$gpu" python -m samtok_edit21.full_regions worker \
    --metadata "$TRAIN_DATA/stage1.jsonl" --output "$TRAIN_DATA/regions" \
    --qwen /mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen-Image-2.1 \
    --samtok "$SAMTOK_CODEC" --rank "$rank" --shards 16 --device cuda:0 \
    --decode-batch-size 16 \
    > "$TRAIN_DATA/regions_logs/worker-$rank.log" 2>&1 &
done
wait
```

所有 worker 产生 `shard-00.json` 到 `shard-15.json` 后执行：

```bash
python -m samtok_edit21.full_regions merge \
  --metadata "$TRAIN_DATA/stage1.jsonl" --output "$TRAIN_DATA/regions" --shards 16
```

`full_training_data.py` 和 `full_regions.py` 都是可恢复的：编码 chunk、tensor 文件、sha256 收据和 row hash 不一致时会直接失败。合并前必须覆盖全部 source ID；区域 manifest 必须覆盖 stage1 的全部 row hash，任务不适用的 plain/NTP 行以 `reason=task` 显式记录。区域任务的 tensor 保存在 `regions/coverage/*.pt`，同一 source/target/span 组合的 ref 和 noref 行共享一份文件；manifest 对每行记录 coverage 相对路径和 SHA256，并在训练读取时重新校验源图哈希。

## 正式四机入口

四个 ARNOLD worker 都执行同一个远程 clone 入口；不要从共享目录执行本地源码。ARNOLD 必须注入 `ARNOLD_WORKER_HOSTS`、`ARNOLD_WORKER_NUM=4`、`ARNOLD_WORKER_GPU=8`、`ARNOLD_ID=0..3`，并把 `WANDB_API_KEY` 作为 secret 注入每个 worker。以下变量在四个 worker 上保持一致，`ARNOLD_ID` 除外：

```bash
export SAMTOK_EXPERIMENT=/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/qwen21_full4_20260928
export SAMTOK_TRAIN_DATA=$SAMTOK_EXPERIMENT/data/train_full_9b_rules_003
export SAMTOK_RUN_ID=qwen21_full_4n_formal_001
export WANDB_API_KEY='FILL_IN_WANDB_API_KEY'
export WANDB_ENTITY=2200012743-peking-university
export WANDB_PROJECT=samtok-edit
export SAMTOK_EDIT_REPO_URL=https://github.com/Tangent0308/samtok_edit.git
export SAMTOK_EDIT_BRANCH=qwen-image-2.1-dev

bash scripts/train/bootstrap_arnold_4node.sh \
  --full-training \
  --stage1-steps 3081 --stage2-steps 3081 \
  --max-pixels 1048576 \
  --stage1-rank 64 --stage2-rank 32 \
  --region-weight 0.5 --attention-weight 0.1 \
  --attention-warmup-steps 500 \
  --wandb-mode online
```

`3081` 是按全量 plain pool 和当前全局 batch 计算的一轮调度长度：Stage 1 的 global batch 是 256，比例为 NTP:ref:noref:plain = 3:2:2:1；Stage 2 的 global batch 是 128，比例为 ref:noref:plain = 1:2:1。训练会按 edit type 池有放回采样并严格满足每个 optimizer update 的比例；NTP/ref/noref 池在这一轮都会覆盖全部源行，plain 池是约一轮的随机采样（多出的 18 个位置仍按同一规则抽样，个别 plain 行可能留到下一轮）。`stage1_steps` 和 `stage2_steps` 仍是 optimizer updates，不是 microsteps。

训练输出位于：

```text
$SAMTOK_EXPERIMENT/runs/$SAMTOK_RUN_ID/
  stage1/adapter/       # TE LoRA
  cache/                # Stage 1 条件缓存及 manifest
  stage2/adapter/       # DiT LoRA
  stage1/training_metrics.jsonl
  stage2/training_metrics.jsonl
  manifest.json
  TRAINING_COMPLETE.json
```

正式模式会运行四机 NCCL、Stage 1、条件缓存和 Stage 2；不会调用只适用于 18 条调试数据的八卡 debug inference harness。每个训练阶段仍验证各 rank 梯度有限且非零、冻结参数无梯度、参数 hash 一致、缓存身份和 checksum 一致；Stage 1 和 Stage 2 各自创建 W&B online run。出现任一节点失败或数据 hash 不一致时，其他节点会在 barrier 检查中退出，且不会写 `TRAINING_COMPLETE.json`。

提交前检查：

```bash
python -m py_compile samtok_edit21/full_training_data.py samtok_edit21/full_regions.py \
  samtok_edit21/cluster.py samtok_edit21/region_supervision.py
git diff --check
```
