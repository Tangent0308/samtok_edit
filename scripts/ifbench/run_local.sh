#!/bin/bash
# One-machine IF-Bench run of the full method (Pass1 + S2) on the shared 1MP public requests of a benchmark bundle:
# prepare (CPU) -> Pass1 (Pass1 environment, one worker per GPU) -> Pass2 (v4 environment, one worker per GPU) ->
# canonical Judge (frozen Judge runtime, TP8) -> report.  Re-running continues a run: finished requests are skipped,
# and a request claimed without a terminal record is left alone for inspection (never resampled).
#   RUN=<run dir> BUNDLE=<bundle> BENCH=<benchmark checkout> PASS1_PY=<python> V4_PY=<python> \
#   JUDGE_PY=<Judge runtime wrapper> bash scripts/ifbench/run_local.sh [--cases v2-sa_1 ...] [--settings S1 ...]
# STAGES (default "pass1 pass2 judge") selects stages; GPUS (default 0..7) the cards.
set -euo pipefail
V4=$(cd "$(dirname "$0")/../.." && pwd)
: "${RUN:?run directory}" "${BUNDLE:?benchmark bundle}" "${BENCH:?benchmark checkout}"
: "${PASS1_PY:?Pass1 environment python}" "${V4_PY:?v4 environment python}" "${JUDGE_PY:?Judge runtime wrapper}"
CONFIG=${CONFIG:-$V4/configs/ifbench/main_row.json}
STAGES=${STAGES:-pass1 pass2 judge}
IFS=, read -ra CARDS <<< "${GPUS:-0,1,2,3,4,5,6,7}"
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
SRC=$V4/src:$V4/third_party/diffsynth:$V4/third_party:$BENCH/src
mkdir -p "$RUN/logs"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$RUN/logs/run.log"; }

PYTHONPATH=$SRC "$V4_PY" -B -m samtok_edit21.evaluation.ifbench prepare --run "$RUN" --bundle "$BUNDLE" \
  --config "$CONFIG" "$@" | tee -a "$RUN/logs/run.log"
for repo in v4:$V4 benchmark:$BENCH; do
  { git -C "${repo#*:}" rev-parse HEAD; git -C "${repo#*:}" status --short; } > "$RUN/logs/${repo%%:*}_commit.txt"
done

stage() {   # stage name, python: one worker per card; the workers take requests by claim
  local name=$1 py=$2 pids=() failed=0
  log "$name: start on ${#CARDS[@]} cards"
  for i in "${!CARDS[@]}"; do
    CUDA_VISIBLE_DEVICES=${CARDS[$i]} PYTHONPATH=$SRC "$py" -B -m samtok_edit21.evaluation.ifbench "$name" \
      --run "$RUN" --worker "$i" --workers "${#CARDS[@]}" >> "$RUN/logs/$name.$i.log" 2>&1 &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  [[ $failed == 0 ]] || { log "$name: a worker failed, see $RUN/logs/$name.*.log"; exit 1; }
  log "$name: done"
}
for s in $STAGES; do
  case $s in
    pass1) stage pass1 "$PASS1_PY" ;;
    pass2) stage pass2 "$V4_PY" ;;
    judge) log "judge: start"
           (cd /tmp && "$JUDGE_PY" "$V4/scripts/ifbench/judge.py" --run "$RUN") 2>&1 | tee -a "$RUN/logs/judge.log"
           log "judge: done" ;;
    *) echo "unknown stage $s" >&2; exit 2 ;;
  esac
done
