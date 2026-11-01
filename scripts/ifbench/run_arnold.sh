#!/usr/bin/env bash
# SAMTokEdit on IF-Bench: post-clone worker entry (called by scripts/ifbench/bootstrap_arnold.sh on every worker).
# Environments -> Pass1 -> Pass2 -> canonical Judge -> report (worker 0).  Each machine handles all settings of its own
# images; machines meet only at marker files on the shared disk, so no ARNOLD port is used.  The benchmark code comes
# from the bundle's snapshot, everything else from the pinned clone, node-local environments and the shared disk.
set -Eeuo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${SAMTOK_ATTEMPT:?}" "${SAMTOK_IFBENCH_RUN:?}" "${IFBENCH_BUNDLE:?}" "${SAMTOK_NNODES:?}" "${SAMTOK_NODE:?}" "${ARNOLD_WORKER_GPU:?}"
RUN=$SAMTOK_IFBENCH_RUN
ATTEMPT=$RUN/attempts/$SAMTOK_ATTEMPT
NODE=$SAMTOK_NODE
NODES=$SAMTOK_NNODES
WORK=${SAMTOK_IFBENCH_WORK:-/tmp/samtok-ifbench-$SAMTOK_ATTEMPT-node$NODE}
CONFIG=${SAMTOK_IFBENCH_CONFIG:-$REPO/configs/ifbench/main_row.json}
SRC=$REPO/src:$REPO/third_party/diffsynth:$REPO/third_party:$IFBENCH_BUNDLE/repo/src
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=, read -ra CARDS <<< "$CUDA_VISIBLE_DEVICES"
else
  mapfile -t CARDS < <(seq 0 $((ARNOLD_WORKER_GPU - 1)))
fi
# The Judge is frozen at TP8: one engine per machine on all its cards (the overrides exist for one-machine rehearsals).
JUDGE_CARDS=${SAMTOK_IFBENCH_JUDGE_CARDS:-$(IFS=,; echo "${CARDS[*]}")}
JUDGE_NODES=${SAMTOK_IFBENCH_JUDGE_NODES:-$NODES}
LOGS=$ATTEMPT/logs/node$NODE
mkdir -p "$WORK" "$LOGS"
cd "$WORK"
# Native pipelines and vLLM must not inherit a platform-wide torchrun rank (the benchmark's child environment does the same).
unset PORT MASTER_ADDR MASTER_PORT RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE NODE_RANK NNODES GPUS_PER_NODE
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export TRITON_CACHE_DIR=$WORK/kernels/triton TORCHINDUCTOR_CACHE_DIR=$WORK/kernels/inductor
export TORCH_EXTENSIONS_DIR=$WORK/kernels/extensions VLLM_CACHE_ROOT=$WORK/kernels/vllm

phase() { echo "$1" > "$ATTEMPT/node$NODE.phase"; echo "[$(date -u +%FT%TZ)] node $NODE: $1"; }
another_failed() { compgen -G "$ATTEMPT/failures/*.json" > /dev/null; }
mark() { mkdir -p "$ATTEMPT/$1"; date -u +%FT%TZ > "$ATTEMPT/$1/node$NODE"; }

barrier() {  # barrier NAME [COUNT]: wait for the markers of the first COUNT machines; stop on a failure elsewhere
  local name=$1 count=${2:-$NODES} deadline=$((SECONDS + 259200)) n missing
  while :; do
    if another_failed; then echo "Another worker failed; stopping at $name." >&2; return 1; fi
    missing=0
    for ((n = 0; n < count; n++)); do [[ -e "$ATTEMPT/$name/node$n" ]] || missing=1; done
    if ((missing == 0)); then return 0; fi
    if ((SECONDS > deadline)); then echo "Timed out waiting for all workers at $name." >&2; return 1; fi
    sleep 15
  done
}

stop_processes() {
  kill -TERM "$@" 2>/dev/null || true
  sleep 20
  kill -KILL "$@" 2>/dev/null || true
  wait "$@" 2>/dev/null || true
}

watch_processes() {  # wait for background workers; stop them all when one fails or another machine reports a failure
  local name=$1 pid
  shift
  local remaining=("$@") next
  while ((${#remaining[@]})); do
    next=()
    for pid in "${remaining[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        next+=("$pid")
      elif ! wait "$pid"; then
        echo "$name: a worker failed, see $LOGS" >&2
        stop_processes "$@"
        return 1
      fi
    done
    remaining=("${next[@]}")
    if another_failed; then
      echo "$name: another worker failed; stopping this one." >&2
      stop_processes "$@"
      return 1
    fi
    if ((${#remaining[@]})); then sleep 15; fi
  done
}

run_workers() {  # run_workers pass1|pass2 PYTHON: one worker per card
  local stage=$1 python=$2 pids=() i
  for i in "${!CARDS[@]}"; do
    CUDA_VISIBLE_DEVICES=${CARDS[$i]} PYTHONPATH=$SRC "$python" -B -m samtok_edit21.evaluation.ifbench "$stage" \
      --run "$RUN" --node "$NODE" --nodes "$NODES" --worker "$i" --workers "${#CARDS[@]}" \
      >> "$LOGS/$stage.$i.log" 2>&1 &
    pids+=($!)
  done
  watch_processes "$stage" "${pids[@]}"
}

phase environment
export SAMTOK_ENV=$WORK/env_v4 SAMTOK_PYTHON=${SAMTOK_PYTHON:-/usr/bin/python3.11}
export SAMTOK_CUDA_DIAGNOSTICS=$ATTEMPT/node$NODE-cuda
env -u CUDA_VISIBLE_DEVICES bash "$REPO/scripts/round2/setup_env.sh"
PASS1_PY=$WORK/env_pass1/bin/python
V4_PY=$SAMTOK_ENV/bin/python
JUDGE_PY=$WORK/runtime/judge/python
bash "$REPO/scripts/ifbench/setup_pass1_env.sh" "$WORK/env_pass1"
"$SAMTOK_PYTHON" -B "$REPO/scripts/ifbench/prepare_judge_runtime.py" --bundle "$IFBENCH_BUNDLE" --work "$WORK/runtime"

phase start
if [[ "$NODE" == 0 ]]; then
  # shellcheck disable=SC2086  # lists of case IDs and settings; empty means every request of the bundle
  PYTHONPATH=$SRC "$V4_PY" -B -m samtok_edit21.evaluation.ifbench prepare --run "$RUN" --bundle "$IFBENCH_BUNDLE" \
    --config "$CONFIG" ${SAMTOK_IFBENCH_CASES:+--cases $SAMTOK_IFBENCH_CASES} \
    ${SAMTOK_IFBENCH_SETTINGS:+--settings $SAMTOK_IFBENCH_SETTINGS} ${SAMTOK_IFBENCH_EXPECT:+--expect $SAMTOK_IFBENCH_EXPECT}
  # A request claimed by an earlier attempt without a terminal record is never re-run silently.
  if ! PYTHONPATH=$SRC "$V4_PY" -B -m samtok_edit21.evaluation.ifbench open-claims --run "$RUN"; then
    echo "Earlier attempts left the requests above without a terminal record: inspect them, then use 'release'." >&2
    exit 1
  fi
  mark START
fi
barrier START 1

phase pass1
run_workers pass1 "$PASS1_PY"
mark PASS1
barrier PASS1

phase pass2
run_workers pass2 "$V4_PY"
mark PASS2
barrier PASS2

phase judge
if ((NODE < JUDGE_NODES)); then
  CUDA_VISIBLE_DEVICES=$JUDGE_CARDS "$JUDGE_PY" "$REPO/scripts/ifbench/judge.py" --run "$RUN" --node "$NODE" \
    --nodes "$JUDGE_NODES" >> "$LOGS/judge.log" 2>&1 &
  watch_processes judge $!
fi
mark JUDGE
barrier JUDGE

phase report
if [[ "$NODE" == 0 ]]; then
  sleep 30   # the other machines' last records on the shared disk
  "$JUDGE_PY" "$REPO/scripts/ifbench/judge.py" --run "$RUN" --report-only | tee -a "$LOGS/report.log"
  "$V4_PY" -B - "$RUN" "$SAMTOK_ATTEMPT" "$(git -C "$REPO" rev-parse HEAD)" <<'PY'
import json, sys, time
from pathlib import Path
run, attempt, commit = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
settings = json.loads((run / "REPORT.json").read_text())["settings"]
(run / "SUCCESS.json").write_text(json.dumps({
    "time": time.time(), "attempt": attempt, "commit": commit, "report": str(run / "REPORT.json"),
    "generation_failed_cases": sum(r["generation_failed_cases"] for r in settings.values()),
    "missing_cases": sum(r["missing_cases"] for r in settings.values())}, indent=1))
PY
  mark REPORT
fi
barrier REPORT 1

# Hand the cards over only after every process of ours has exited (the Judge engine's workers included; their command
# lines name $WORK), and the benchmark's entry needs at least 70 GiB free on every card.
deadline=$((SECONDS + 600))
while pgrep -f -- "$WORK/" > /dev/null; do
  if ((SECONDS > deadline)); then echo "Our processes are still running: $(pgrep -af -- "$WORK/" | cut -c1-160)" >&2; exit 1; fi
  sleep 10
done
while :; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
  if ((free >= 70 * 1024)); then break; fi
  if ((SECONDS > deadline)); then echo "Cards still busy after our processes ended (lowest free ${free} MiB)." >&2; exit 1; fi
  sleep 10
done
cd /
rm -rf "$WORK"
mark DONE
barrier DONE
phase done
