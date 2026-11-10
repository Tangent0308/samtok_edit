#!/usr/bin/env bash
# R2（Refine）阶段：从 v4 (S2@16000) 续训，使用高合成目标数据。四机正式运行的完整入口。
# 整段原样粘贴成 ARNOLD 四个 worker（4 × 8 卡）的启动命令：开头是运行设置，后面接
# 入口脚本 scripts/round2/bootstrap_arnold.sh 的全文。
#
# 交付检查点：adapter_1000（6.4 万行，lr 8e-6 余弦，高合成目标）。
# 方法说明见 docs/04_SITEdit_v5方法.md，实验结果见 docs/05_SITEdit_v5实验.md。

# ===== 运行设置（重新提交时只改 SAMTOK_RUN_ID）=====
export SAMTOK_RUN_ID=r2_4n_001       # 每次提交换新的，四个 worker 相同；用过的会被拒绝
export SAMTOK_RUN_NAME=R2
export SAMTOK_EDIT_COMMIT=22650e4b30830f95a11a3af8f4558326d98edb2f   # 代码快照，四个 worker 必须相同
export SAMTOK_NNODES=4
export SAMTOK_EXPERIMENT=/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v3
export SAMTOK_ROUND2_DATA=$SAMTOK_EXPERIMENT/data_round2_v6p

# 从 v4（S2@16000）续训。32 卡 × accumulation 1 = batch 32；
# rows_v6_s2hqfull.jsonl 共 512,000 行，7,000 次更新覆盖约 1.37 轮；
# 交付检查点为 adapter_1000（约 6.4 万行，~1.5 h on 4 × 8 GPUs）。
# 学习率 8e-6 预热 100，余弦降到 8e-7；损失：FM + 区域项 λ1（编辑行）+
# region_hl λ0（读 token 行）+ distractor λ1 + route λ0.05。
export SAMTOK_TRAIN_ARGS="\
--rows rows_v6_s2hqfull.jsonl \
--quick-items eval_c2_quick_idx.json \
--init-adapter $SAMTOK_EXPERIMENT/runs/s2_4n_001/train/adapter_16000 \
--updates 7000 \
--accumulation 1 \
--warmup 100 \
--lr 8e-6 \
--lr-schedule cosine \
--lr-min 8e-7 \
--heavy-weight 8 \
--checkpoints 1000,2000,3000,5000,7000 \
--quick-eval 0,1000,2000,3000,5000,7000"

# 四机只训练：开发集评测、报告和判分在开发机上做
export SAMTOK_PHASES=train
export SAMTOK_WANDB_MODE=online
export WANDB_ENTITY=2200012743-peking-university
export WANDB_PROJECT=samtok-edit
# ===== 以下接入口脚本 scripts/round2/bootstrap_arnold.sh（全文，不用改）=====
