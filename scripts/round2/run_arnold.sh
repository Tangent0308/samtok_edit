#!/usr/bin/env bash
# Round-2 run: post-clone worker entry (called by scripts/round2/bootstrap_arnold.sh on every worker).
# environment -> train -> evaluate on all GPUs of all workers; worker 0 then writes the paired report.
set -Eeuo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${SAMTOK_RUN_ROOT:?Set by bootstrap_arnold.sh}"
: "${SAMTOK_NODE:?Set by bootstrap_arnold.sh}"
: "${SAMTOK_RUN_ID:?}" "${SAMTOK_ROUND2_DATA:?}" "${SAMTOK_NNODES:?}" "${ARNOLD_WORKER_HOSTS:?}" "${ARNOLD_WORKER_GPU:?}"
RUN="$SAMTOK_RUN_ROOT"
NODE="$SAMTOK_NODE"
PHASES=",${SAMTOK_PHASES:-train,eval,report},"
phase() { echo "$1" > "${SAMTOK_PHASE_FILE:-/dev/null}"; echo "[$(date -u +%FT%TZ)] node $NODE: $1"; }

phase environment
export SAMTOK_ENV="${SAMTOK_ENV:-/tmp/samtok21-${SAMTOK_RUN_ID}-node${NODE}-env}"
export SAMTOK_PYTHON="${SAMTOK_PYTHON:-/usr/bin/python3.11}"
export SAMTOK_CUDA_DIAGNOSTICS="$RUN/bootstrap/node${NODE}-cuda"
export SAMTOK_CUDA_READY_TIMEOUT="${SAMTOK_CUDA_READY_TIMEOUT:-600}" SAMTOK_CUDA_READY_INTERVAL="${SAMTOK_CUDA_READY_INTERVAL:-15}"
bash "$REPO/scripts/round2/setup_env.sh"
export PYTHONPATH="$REPO/src:$REPO/third_party/diffsynth:$REPO/third_party${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" NCCL_DEBUG="${NCCL_DEBUG:-WARN}" TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/samtok21-inductor-$(id -u)}"
export WANDB_DISABLE_SERVICE=true WANDB_START_METHOD=thread   # required by byted-wandb in ARNOLD containers
WANDB_ARGS=(--wandb-mode "${SAMTOK_WANDB_MODE:-disabled}" --wandb-project "${WANDB_PROJECT:-samtok-edit}"
            --wandb-entity "${WANDB_ENTITY:-2200012743-peking-university}" --wandb-name "$SAMTOK_RUN_ID")
PYTHON="$SAMTOK_ENV/bin/python"
cd "$REPO"

# Rendezvous: the first entry of ARNOLD_WORKER_HOSTS, written as [IPv6]:port or host:port.
first="${ARNOLD_WORKER_HOSTS%%,*}"
first="${first// /}"
if [[ "$first" == \[* ]]; then
  MASTER="${first#[}"; MASTER="${MASTER%%]*}"; MASTER_PORT_="${first##*]:}"
else
  MASTER="${first%:*}"; MASTER_PORT_="${first##*:}"
fi
[[ -n "$MASTER" && "$MASTER_PORT_" =~ ^[0-9]+$ ]] || { echo "Cannot parse ARNOLD_WORKER_HOSTS=$ARNOLD_WORKER_HOSTS" >&2; false; }
echo "rendezvous $MASTER:$MASTER_PORT_, node $NODE of $SAMTOK_NNODES, $ARNOLD_WORKER_GPU GPUs per node"
TORCHRUN=("$PYTHON" -m torch.distributed.run --nnodes "$SAMTOK_NNODES" --nproc_per_node "$ARNOLD_WORKER_GPU"
          --node_rank "$NODE" --master_addr "$MASTER" --master_port "$MASTER_PORT_" --max_restarts 0)
mkdir -p "$RUN/logs/node${NODE}"

wait_for() {  # rank 0 writes the marker last; allow for NAS latency on the other workers
  for _ in $(seq 1 120); do [[ -f "$1" ]] && return 0; sleep 5; done
  echo "Missing $1" >&2; return 1
}

run_watched() {  # run a command, but stop it as soon as another worker reports a failure
  "$@" &
  local pid=$! other
  while kill -0 "$pid" 2>/dev/null; do
    for other in "$RUN"/nodes/*/failure.json; do
      if [[ -e "$other" && "$other" != "$RUN/nodes/$NODE/failure.json" ]]; then
        echo "Another worker failed ($other); stopping this one." >&2
        kill -TERM "$pid" 2>/dev/null || true
        sleep "${SAMTOK_KILL_GRACE:-10}"
        kill -KILL "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
        return 1
      fi
    done
    sleep "${SAMTOK_WATCH_INTERVAL:-15}"
  done
  wait "$pid"
}

ADAPTER="${SAMTOK_ADAPTER:-}"
if [[ "$PHASES" == *,train,* ]]; then
  phase train
  # shellcheck disable=SC2086  # SAMTOK_TRAIN_ARGS is a list of options
  run_watched "${TORCHRUN[@]}" --log_dir "$RUN/logs/node${NODE}/train-ranks" --tee 3 -m samtok_edit21 round2-train \
    --data "$SAMTOK_ROUND2_DATA" --output "$RUN/train" "${WANDB_ARGS[@]}" ${SAMTOK_TRAIN_ARGS:-}
  wait_for "$RUN/train/TRAIN_DONE"
  ADAPTER="$RUN/train/$("$PYTHON" -c 'import json, sys; print(json.load(open(sys.argv[1]))["adapter"])' "$RUN/train/TRAIN_DONE")"
fi

if [[ "$PHASES" == *,eval,* ]]; then
  phase eval
  [[ -f "$ADAPTER/adapter.safetensors" ]] || { echo "No adapter to evaluate: $ADAPTER" >&2; false; }
  # shellcheck disable=SC2086
  run_watched "${TORCHRUN[@]}" --log_dir "$RUN/logs/node${NODE}/eval-ranks" --tee 3 -m samtok_edit21 round2-eval \
    --data "$SAMTOK_ROUND2_DATA" --adapter "$ADAPTER" --output "$RUN/eval" ${SAMTOK_EVAL_ARGS:-}
fi

if [[ "$NODE" == 0 ]]; then
  if [[ "$PHASES" == *,report,* ]]; then
    phase report
    wait_for "$RUN/eval/EVAL_DONE"
    REPORT=("$PYTHON" -m samtok_edit21 round2-report --eval "$RUN/eval" --name "${SAMTOK_RUN_NAME:-$SAMTOK_RUN_ID}"
            --output "$RUN/analysis.txt")
    [[ -f "$RUN/train/train_metrics.jsonl" ]] && REPORT+=(--train "$RUN/train")
    [[ -f "$RUN/train/wandb.json" ]] && REPORT+=(--wandb "$RUN/train/wandb.json")
    for reference in ${SAMTOK_REPORT_REFERENCES:-}; do REPORT+=(--reference "$reference"); done
    "${REPORT[@]}"
  fi
  printf '{"time":"%s","run":"%s","commit":"%s","adapter":"%s"}\n' "$(date -u +%FT%TZ)" "$SAMTOK_RUN_ID" \
    "$(git -C "$REPO" rev-parse HEAD)" "$ADAPTER" > "$RUN/SUCCESS.json"
  echo "finished; report: $RUN/analysis.txt"
fi
phase done
