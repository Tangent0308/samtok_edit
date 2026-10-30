#!/usr/bin/env bash
# 两阶段重训的第二阶段（S2：在 S1 上加强编辑）四机正式运行的完整入口。整段原样粘贴成 ARNOLD 四个 worker（4 × 8 卡）的启动命令：
# 开头是运行设置，后面接入口脚本 scripts/round2/bootstrap_arnold.sh 的全文。计划见 qwen21_v3/analysis/plan_scaleup_20261030/README.md，
# 取行配置见 scripts/round2/specs/rows_s2.json，数据目录 data/round2_scaleup_20261030（扩池完成后生成 rows_s2.jsonl）。
# ===== 运行设置（重新提交时只改 SAMTOK_RUN_ID）=====
export SAMTOK_RUN_ID=s2_4n_001       # 每次提交换新的，四个 worker 相同；用过的会被拒绝
export SAMTOK_RUN_NAME=S2
export SAMTOK_EDIT_COMMIT=47c0efd16fe4bf458d98db41425b159bec444960   # 代码 commit，四个 worker 必须相同
export SAMTOK_NNODES=4
export SAMTOK_EXPERIMENT=/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v3
export SAMTOK_ROUND2_DATA=$SAMTOK_EXPERIMENT/data/round2_scaleup_20261030
# 从第一阶段挑出的点续训（挑点后改 S1_ADAPTER）。32 卡 × accumulation 1；16,000 次更新把 rows_s2.jsonl 的 512,000 行各用一次。
# 学习率 5e-5，预热 100，余弦降到 5e-6；近邻硬反例权重 8；每 2,000 次保存并快速评测，开发集看 4,000/8,000/12,000/16,000。
export S1_ADAPTER=$SAMTOK_EXPERIMENT/runs/s1_4n_001/train/adapter_6000
export SAMTOK_TRAIN_ARGS="--rows rows_s2.jsonl --quick-items eval_c2_quick_idx.json --init-adapter $S1_ADAPTER --updates 16000 --accumulation 1 --warmup 100 --lr 5e-5 --lr-schedule cosine --lr-min 5e-6 --heavy-weight 8 --checkpoints 2000,4000,6000,8000,10000,12000,14000,16000 --quick-eval 0,2000,4000,6000,8000,10000,12000,14000,16000"
# 四机只训练：开发集评测、报告和判分在开发机上做
export SAMTOK_PHASES=train
export SAMTOK_WANDB_MODE=online
export WANDB_ENTITY=2200012743-peking-university
export WANDB_PROJECT=samtok-edit
# ===== 以下接入口脚本 scripts/round2/bootstrap_arnold.sh（全文，不用改）=====
