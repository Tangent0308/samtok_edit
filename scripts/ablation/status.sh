#!/usr/bin/env bash
# 消融四机作业的进度（只读）：每台机器当前步骤、已完成的步骤、正在训练的臂的最新指标、失败记录。
#   bash status.sh [ABL_RUN_ID，默认最新的一次提交]
ROOT=${ABL_ROOT:-/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/ablation_20261112/v2}   # v1 (abl_001): ABL_ROOT=<上一级>
RUN=${1:-$(ls -t "$ROOT/submissions" 2>/dev/null | head -1)}
SUB=$ROOT/submissions/$RUN
[[ -d "$SUB" ]] || { echo "No submission $SUB"; exit 1; }
echo "submission $RUN"
for f in "$SUB"/node*.phase; do [[ -e "$f" ]] && echo "  $(basename "$f" .phase): $(cat "$f")"; done
echo "failures:"; ls "$SUB"/failures/*.json 2>/dev/null | while read -r f; do echo "  $(basename "$f"): $(cat "$f")"; done
echo "training (TRAIN_DONE):"
for d in "$ROOT"/train/*/{read,edit} "$ROOT"/scale/*/train; do
  [[ -d "$d" ]] || continue
  last=$(grep '"update"' "$d/train_metrics.jsonl" 2>/dev/null | tail -1 | cut -c1-120)
  echo "  ${d#$ROOT/}: $([[ -f $d/TRAIN_DONE ]] && echo done || echo "running  $last")"
done
echo "probes (EVAL_DONE): $(ls "$ROOT"/probe/*/EVAL_DONE 2>/dev/null | wc -l) / 13"
echo "generation (state/generated_*.done): $(ls "$ROOT"/state/generated_*.done 2>/dev/null | wc -l) / 14"
echo "judge: $([[ -f $ROOT/state/judged.done ]] && echo "done -> $ROOT/REPORT/ABLATION.md" || echo pending)"
