#!/usr/bin/env bash
# C2（阶段 B，全部可用数据）四机正式运行的完整入口。整段原样粘贴成 ARNOLD 四个 worker（4 × 8 卡）的启动命令。
# 开头是运行设置，后面接入口脚本 scripts/round2/bootstrap_arnold.sh。方法见 docs/02 第 4.8–4.10 节，设置与结果见 docs/03 第 5 节，说明见 docs/05 第 7 节。
# ===== 运行设置（重新提交时只改 SAMTOK_RUN_ID）=====
export SAMTOK_RUN_ID=c2_4n_002       # 每次提交换新的，四个 worker 相同；用过的会被拒绝，调度器重试不会覆盖日志
export SAMTOK_RUN_NAME=C2            # 报告里这次运行的名字
export SAMTOK_EDIT_COMMIT=50ffa0ce3d1a1b9db88ffc4fb633b684f1e4df2a   # 代码 commit，四个 worker 必须相同
export SAMTOK_NNODES=4
export SAMTOK_EXPERIMENT=/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v3
export SAMTOK_ROUND2_DATA=$SAMTOK_EXPERIMENT/data/round2_20261016
# 32 卡 × accumulation 1 = batch 32，与 8 卡 × 4 每次更新用到的样本相同；3,522 次更新把 rows_c2.jsonl 的 112,704 行各用一次
export SAMTOK_TRAIN_ARGS="--rows rows_c2.jsonl --quick-items eval_c2_quick_idx.json --init-adapter $SAMTOK_EXPERIMENT/runs/c1b_local8_001/train/adapter_800 --updates 3522 --accumulation 1 --warmup 20 --lr 1e-4 --lr-schedule cosine --lr-min 1e-5 --heavy-weight 8 --checkpoints 500,1000,1500,2000,2500,3000,3500,3522 --quick-eval 0,500,1000,1500,2000,2500,3000,3500,3522"
# 四机只训练：评测、报告和分析在开发机上训练结束后做（scripts/round2/eval_local.sh，docs/05 第 7 节）
export SAMTOK_PHASES=train
# W&B 与 v2 相同：entity 2200012743-peking-university，project samtok-edit，run 名即 SAMTOK_RUN_ID。
# key 优先用 ARNOLD 注入的 WANDB_API_KEY，否则读共享盘上的私有文件（权限 600），不要写进脚本
export SAMTOK_WANDB_MODE=online
export WANDB_ENTITY=2200012743-peking-university
export WANDB_PROJECT=samtok-edit
# ===== 以下是入口脚本 scripts/round2/bootstrap_arnold.sh，不用改；其中的设置是默认值，已被上面的值覆盖 =====
