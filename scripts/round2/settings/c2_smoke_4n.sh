#!/usr/bin/env bash
# C2 四机冒烟测试的完整入口（正式提交前检查集群环境，约 30 分钟）。整段原样粘贴成 ARNOLD 四个 worker（4 × 8 卡）的启动命令。
# 跑通的标志是 runs/<SAMTOK_RUN_ID>/SUCCESS.json 和 analysis.txt 都生成了；数值没有意义，只看流程。
# ===== 运行设置（重新提交时只改 SAMTOK_RUN_ID）=====
export SAMTOK_RUN_ID=c2_smoke_4n_001
export SAMTOK_RUN_NAME=smoke
export SAMTOK_EDIT_COMMIT=d485ddb2f69f8cf16e7333886f827a7849c31d44   # 与正式运行相同的代码 commit
export SAMTOK_NNODES=4
export SAMTOK_EXPERIMENT=/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v3
export SAMTOK_ROUND2_DATA=$SAMTOK_EXPERIMENT/data/round2_20261016
# 3 次更新、每卡 1 个快速评测项、每类 2 个评测项：只检查 clone、装环境、四机训练、评测和报告能否走通
export SAMTOK_TRAIN_ARGS="--rows rows_c2.jsonl --quick-items eval_c2_quick_idx.json --init-adapter $SAMTOK_EXPERIMENT/runs/c1b_local8_001/train/adapter_800 --updates 3 --accumulation 1 --warmup 1 --lr-schedule cosine --heavy-weight 8 --checkpoints 3 --quick-eval 3 --quick-limit 1"
export SAMTOK_EVAL_ARGS="--limit 2"
export SAMTOK_REPORT_REFERENCES="stock=$SAMTOK_EXPERIMENT/baselines/round2_stock_eval C1b=$SAMTOK_EXPERIMENT/runs/c1b_local8_001/eval_800"
export SAMTOK_PHASES=train,eval,report
# W&B 与 v2 相同：entity 2200012743-peking-university，project samtok-edit，run 名即 SAMTOK_RUN_ID。
# key 优先用 ARNOLD 注入的 WANDB_API_KEY，否则读共享盘上的私有文件（权限 600），不要写进脚本
export SAMTOK_WANDB_MODE=online
export WANDB_ENTITY=2200012743-peking-university
export WANDB_PROJECT=samtok-edit
# ===== 以下是入口脚本 scripts/round2/bootstrap_arnold.sh，不用改；其中的设置是默认值，已被上面的值覆盖 =====
