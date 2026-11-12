#!/usr/bin/env bash
# 消融四机作业的进度（只读）：每个提交（组件块、规模块各一个）各台机器的当前步骤与失败记录，各训练阶段、探针、生成、判分的完成情况。
#   bash status.sh            # v2；v1（abl_001）：ABL_ROOT=<上一级目录> bash status.sh
ROOT=${ABL_ROOT:-/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/ablation_20261112/v2}
[[ -d "$ROOT/submissions" ]] || { echo "No submission under $ROOT"; exit 1; }
for SUB in $(ls -td "$ROOT"/submissions/* 2>/dev/null); do
  echo "submission $(basename "$SUB")$( [[ -f $SUB/SUCCESS.json ]] && echo '  (SUCCESS)')"
  for f in "$SUB"/node*.phase; do [[ -e "$f" ]] && echo "  $(basename "$f" .phase): $(cat "$f")"; done
  ls "$SUB"/failures/*.json 2>/dev/null | while read -r f; do echo "  FAILURE $(basename "$f"): $(cat "$f")"; done
done
echo "training (TRAIN_DONE):"
for d in "$ROOT"/train/*/{read,edit} "$ROOT"/scale/*/train; do
  [[ -d "$d" ]] || continue
  last=$(grep '"update"' "$d/train_metrics.jsonl" 2>/dev/null | tail -1 | cut -c1-120)
  echo "  ${d#$ROOT/}: $([[ -f $d/TRAIN_DONE ]] && echo done || echo "running  $last")"
done
echo "probes (EVAL_DONE): $(ls "$ROOT"/probe/*/EVAL_DONE 2>/dev/null | wc -l) / 13  ($(ls "$ROOT"/probe/*/EVAL_DONE 2>/dev/null | xargs -r -n1 dirname | xargs -r -n1 basename | tr '\n' ' '))"
echo "generation (state/generated_*.done): $(ls "$ROOT"/state/generated_*.done 2>/dev/null | wc -l) / 14"
for b in component scale all; do [[ -f $ROOT/state/judged_$b.done ]] && echo "judge $b: done"; done
[[ -f $ROOT/REPORT/ABLATION.md ]] && echo "report: $ROOT/REPORT/ABLATION.md"
exit 0
