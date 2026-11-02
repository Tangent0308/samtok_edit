#!/usr/bin/env bash
# IF-Bench 正式全量（七模型 + SAMTokEdit）一个 ARNOLD 作业的完整入口。整段原样粘贴成四个 worker（4 × 8 卡）的启动命令：
# 开头是运行设置，后面接入口脚本 scripts/ifbench/bootstrap_arnold.sh 的全文。每台的顺序：七模型包的无推理预检 → 我方方法 → 七模型包的 start.sh。
# ===== 运行设置（重新提交时只改 SAMTOK_ATTEMPT）=====
export SAMTOK_ATTEMPT=run003          # 每次提交换新的，四个 worker 相同；benchmark 的 attempt 也用它命名（run001、run002 已用）
export SAMTOK_EDIT_COMMIT=REPLACE_WITH_FULL_COMMIT_SHA   # 代码 commit，四个 worker 必须相同
export IFBENCH_BUNDLE=/mnt/bn/strategy-mllm-train/user/tanyue/.ifbench-tmp/one-mp-full4-20261101-heartbeatfix
# 我方最终输出的统一重判（docs/03 §12.9）：出图已全部完成，入口只跑 Judge
export SAMTOK_IFBENCH_RUN=/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/ifbench_one_mp_full4_final
export SAMTOK_IFBENCH_EXPECT=10840    # 1,084 张图 × 10 个 setting；数量不符就停
export SAMTOK_NNODES=4
# ===== 以下接入口脚本 scripts/ifbench/bootstrap_arnold.sh（全文，不用改）=====
