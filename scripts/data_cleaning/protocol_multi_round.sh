#!/usr/bin/env bash
# One round of the multi-object recall on one machine with GPUS GPUs and a running Qwen3.8-27B server (URL):
# slots -> units -> SAM3 on every GPU -> part picks -> build -> assemble -> audit, then the same for the pairs that
# failed (re-asked with the problems), then the final gate.
# usage: protocol_multi_round.sh ROWS OUT   (env: URL, ROOT, PY, GPUS, VOTES, IN_FLIGHT, LABELING_REPO)
set -euo pipefail
ROWS=$1 OUT=$2
ROOT=${ROOT:-/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTokEdit_Cleaned_20261009}
URL=${URL:-http://127.0.0.1:8000/v1/chat/completions}
PY=${PY:-python}
GPUS=${GPUS:-8}
FLIGHT=${IN_FLIGHT:-96}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
M="$PY -B scripts/data_cleaning/protocol_multi.py"
A="$PY -B scripts/data_cleaning/protocol_adapt.py"
step() { echo "[$(date -u +%T)] $*"; }
mkdir -p "$OUT"

remask() {  # unit rows, unit slots, merged output: one SAM3 process per GPU on interleaved parts
  local pids=() g
  for g in $(seq 0 $((GPUS - 1))); do
    CUDA_VISIBLE_DEVICES=$g $PY -B scripts/data_cleaning/protocol_remask.py --rows "$1" --slots "$2" --root "$ROOT" \
      --part "$g/$GPUS" --output "$3.part$g.jsonl" ${LABELING_REPO:+--labeling-repo "$LABELING_REPO"} \
      > "$3.part$g.log" 2>&1 &
    pids+=($!)
  done
  for g in "${pids[@]}"; do wait "$g"; done
  cat "$3".part*.jsonl > "$3" 2>/dev/null || : > "$3"
}

run() {  # tag ("" or "_fix"), extra slots arguments
  local t=$1
  step "slots$t"; $M slots --rows "$ROWS" --root "$ROOT" --url "$URL" --in-flight "$FLIGHT" --workers 16 $2 \
    --output "$OUT/pairs$t.jsonl" > "$OUT/slots$t.log" 2>&1
  step "units$t"; $M units --rows "$ROWS" --pairs "$OUT/pairs$t.jsonl" --unit-rows "$OUT/unit_rows$t.jsonl" \
    --unit-slots "$OUT/unit_slots$t.jsonl"
  step "remask$t"; remask "$OUT/unit_rows$t.jsonl" "$OUT/unit_slots$t.jsonl" "$OUT/remask$t.jsonl"
  step "parts$t"; $A parts --rows "$OUT/unit_rows$t.jsonl" --root "$ROOT" --slots "$OUT/unit_slots$t.jsonl" \
    --remask "$OUT/remask$t.jsonl" --url "$URL" --in-flight "$FLIGHT" --workers 16 \
    --output "$OUT/remask_picked$t.jsonl" > "$OUT/parts$t.log" 2>&1
  step "build$t"; $A build --rows "$OUT/unit_rows$t.jsonl" --root "$ROOT" --slots "$OUT/unit_slots$t.jsonl" \
    --remask "$OUT/remask_picked$t.jsonl" --workers 48 --no-sheets --output "$OUT/build$t" > "$OUT/build$t.log" 2>&1
  step "assemble$t"; $M assemble --rows "$ROWS" --root "$ROOT" --pairs "$OUT/pairs$t.jsonl" \
    --unit-records "$OUT/build$t/records.jsonl" --output "$OUT/assembled$t.jsonl" > "$OUT/assemble$t.log" 2>&1
  step "audit$t"; $M audit --rows "$ROWS" --root "$ROOT" --assembled "$OUT/assembled$t.jsonl" --url "$URL" \
    --in-flight "$FLIGHT" --workers 16 --votes "${VOTES:-2}" --output "$OUT/audit$t.jsonl" > "$OUT/audit$t.log" 2>&1
}

run "" ""
step repair; $M repair --assembled "$OUT/assembled.jsonl" --audit "$OUT/audit.jsonl" --output "$OUT/repair.jsonl"
fixes=()
if [[ -s "$OUT/repair.jsonl" ]]; then
  run _fix "--feedback $OUT/repair.jsonl"
  fixes=(--assembled "$OUT/assembled_fix.jsonl" --audit "$OUT/audit_fix.jsonl")
fi
step final; $M final --assembled "$OUT/assembled.jsonl" --audit "$OUT/audit.jsonl" "${fixes[@]}" --output "$OUT/final.jsonl"
step done
