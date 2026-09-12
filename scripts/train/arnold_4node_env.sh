#!/usr/bin/env bash
# Shared Arnold topology parsing and Accelerate launch helpers.
# This file is sourced by the dedicated four-node launchers.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Source this file from a four-node launcher; do not execute it directly." >&2
  exit 2
fi

samtok_require_uint() {
  local name="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    printf '%s must be a non-negative integer, got: %s\n' "$name" "$value" >&2
    return 2
  fi
}

samtok_init_arnold_topology() {
  local first_worker
  local inferred_addr
  local inferred_port
  local worker_zero

  first_worker="${ARNOLD_WORKER_HOSTS%%,*}"
  if [[ "$first_worker" =~ ^\[([^]]+)\]:([0-9]+)$ ]]; then
    inferred_addr="${BASH_REMATCH[1]}"
    inferred_port="${BASH_REMATCH[2]}"
  elif [[ "$first_worker" =~ ^([^:]+):([0-9]+)$ ]]; then
    inferred_addr="${BASH_REMATCH[1]}"
    inferred_port="${BASH_REMATCH[2]}"
  else
    inferred_addr="$first_worker"
    inferred_port=""
  fi

  worker_zero="${ARNOLD_WORKER_0_HOST:-}"
  if [[ "$worker_zero" =~ ^\[([^]]+)\]:([0-9]+)$ ]]; then
    worker_zero="${BASH_REMATCH[1]}"
  elif [[ "$worker_zero" =~ ^\[([^]]+)\]$ ]]; then
    worker_zero="${BASH_REMATCH[1]}"
  fi

  export NNODES="${NNODES:-${ARNOLD_WORKER_NUM:-}}"
  export NODE_RANK="${NODE_RANK:-${ARNOLD_ID:-}}"
  export GPUS_PER_NODE="${GPUS_PER_NODE:-${ARNOLD_WORKER_GPU:-}}"
  export MASTER_ADDR="${MASTER_ADDR:-${worker_zero:-$inferred_addr}}"
  # Arnold may expose a different generic PORT on every worker. It is a local
  # service port, not a distributed rendezvous port. Only an explicit
  # MASTER_PORT may override the port attached to worker 0 in the shared host
  # list.
  export MASTER_PORT="${MASTER_PORT:-$inferred_port}"
  if [[ "$MASTER_ADDR" =~ ^\[([^]]+)\]$ ]]; then
    export MASTER_ADDR="${BASH_REMATCH[1]}"
  fi

  : "${NNODES:?Set NNODES or ARNOLD_WORKER_NUM}"
  : "${NODE_RANK:?Set NODE_RANK or ARNOLD_ID}"
  : "${GPUS_PER_NODE:?Set GPUS_PER_NODE or ARNOLD_WORKER_GPU}"
  : "${MASTER_ADDR:?Set MASTER_ADDR, ARNOLD_WORKER_0_HOST, or ARNOLD_WORKER_HOSTS}"
  : "${MASTER_PORT:?Set MASTER_PORT or provide ports in ARNOLD_WORKER_HOSTS}"

  samtok_require_uint NNODES "$NNODES"
  samtok_require_uint NODE_RANK "$NODE_RANK"
  samtok_require_uint GPUS_PER_NODE "$GPUS_PER_NODE"
  samtok_require_uint MASTER_PORT "$MASTER_PORT"
  if (( NNODES != 4 )); then
    echo "The dedicated launcher requires exactly four nodes; got NNODES=$NNODES" >&2
    return 2
  fi
  if (( GPUS_PER_NODE != 8 )); then
    echo "The ws32 training contract requires eight GPUs per node; got GPUS_PER_NODE=$GPUS_PER_NODE" >&2
    return 2
  fi
  if (( NODE_RANK < 0 || NODE_RANK >= NNODES )); then
    echo "NODE_RANK=$NODE_RANK is outside [0, $((NNODES - 1))]" >&2
    return 2
  fi
  if (( MASTER_PORT < 1 || MASTER_PORT > 65535 )); then
    echo "MASTER_PORT must be in [1, 65535], got $MASTER_PORT" >&2
    return 2
  fi

  export NUM_MACHINES="$NNODES"
  export MACHINE_RANK="$NODE_RANK"
  export NUM_PROCESSES="$((NNODES * GPUS_PER_NODE))"
  export WORLD_SIZE="$NUM_PROCESSES"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
  export RDZV_BACKEND="${RDZV_BACKEND:-static}"

  if [[ "$WORLD_SIZE" != "32" ]]; then
    echo "Internal topology error: expected world size 32, got $WORLD_SIZE" >&2
    return 2
  fi
}

samtok_accelerate_launch() {
  local accelerate_bin="${ACCELERATE_BIN:-}"
  if [[ -z "$accelerate_bin" ]]; then
    if [[ -n "${SAMTOK_EDIT_VENV:-}" ]]; then
      accelerate_bin="$SAMTOK_EDIT_VENV/bin/accelerate"
    else
      accelerate_bin="$(command -v accelerate || true)"
    fi
  fi
  if [[ -z "$accelerate_bin" || ! -x "$accelerate_bin" ]]; then
    echo "Cannot find Accelerate. Activate the uv environment or set SAMTOK_EDIT_VENV." >&2
    return 1
  fi

  "$accelerate_bin" launch \
    --multi_gpu \
    --mixed_precision "${ACCELERATE_MIXED_PRECISION:-no}" \
    --dynamo_backend "${ACCELERATE_DYNAMO_BACKEND:-no}" \
    --num_processes "$NUM_PROCESSES" \
    --num_machines "$NUM_MACHINES" \
    --machine_rank "$MACHINE_RANK" \
    --main_process_ip "$MASTER_ADDR" \
    --main_process_port "$MASTER_PORT" \
    --rdzv_backend "$RDZV_BACKEND" \
    --same_network \
    "$@"
}
