#!/usr/bin/env bash
# 本机预演：与四机入口 run_4n.sh 同一份脚本，只缩小规模、换位置（1 台 × 4 卡，每个训练阶段 2 次更新，2 张评测图，
# 探针每类 1 项，Judge 只列出待判的请求而不加载引擎）。四机提交前跑通一遍，保证串行流程每一步都走到。
# 代码与四机相同：从 ABL_BUNDLE 按 ABL_COMMIT 检出（ABL_USE_LOCAL=1 时直接用本脚本所在的 checkout）；
# 只有环境用本机已有的（ABL_ENV），不重新建。
#   ABL_COMMIT=<sha> ABL_BUNDLE=<bundle> bash scripts/ablation/rehearse.sh [卡号，默认 0,1,2,3]
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CARDS=${1:-0,1,2,3}
STAMP=$(date -u +%m%d%H%M)
BASE=${ABL_REHEARSAL_DIR:-/tmp/abl_rehearsal_$STAMP}
mkdir -p "$BASE"
head -2 /mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/ablation_20261112/cases200.txt > "$BASE/cases2.txt"
export ABL_RUN_ID=rehearsal_$STAMP ABL_ROOT=$BASE/root ABL_CASES=$BASE/cases2.txt ABL_SMOKE=1 ABL_JUDGE_DRYRUN=1
export ABL_DATA=/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/ablation_20261112/data
export ABL_ENV=${ABL_ENV:-/tmp/ifbench/external_bench_20261102/env_v4} ABL_NNODES=1
if [[ "${ABL_USE_LOCAL:-0}" == 1 ]]; then
  export ABL_REPO=$REPO
else
  : "${ABL_COMMIT:?set ABL_COMMIT (and ABL_BUNDLE) to rehearse the bundled code}"
  export ABL_COMMIT ABL_BUNDLE
fi
[[ -n "${ABL_JUDGE_PY:-}" ]] && export ABL_JUDGE_PY
export ARNOLD_WORKER_HOSTS=127.0.0.1:${ABL_REHEARSAL_PORT:-39123} ARNOLD_WORKER_NUM=1 ARNOLD_ID=0
IFS=, read -ra list <<< "$CARDS"
export ARNOLD_WORKER_GPU=${#list[@]} CUDA_VISIBLE_DEVICES=$CARDS
echo "rehearsal root $ABL_ROOT, cards $CARDS"
bash "$REPO/scripts/ablation/run_4n.sh"
