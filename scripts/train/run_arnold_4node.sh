#!/usr/bin/env bash
# Post-clone worker entry: run the same command on all four ARNOLD workers.
set -Eeuo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${SAMTOK_EXPERIMENT:?Set shared experiment directory}"
: "${SAMTOK_RUN_ID:?Set one identical fresh run ID on all workers}"
: "${WANDB_API_KEY:?Inject WANDB_API_KEY through ARNOLD environment/secrets}"
: "${ARNOLD_WORKER_HOSTS:?ARNOLD_WORKER_HOSTS is required}"
: "${ARNOLD_WORKER_NUM:?ARNOLD_WORKER_NUM is required}"
: "${ARNOLD_WORKER_GPU:?ARNOLD_WORKER_GPU is required}"
: "${ARNOLD_ID:?ARNOLD_ID (0..3) is required}"
[[ "$ARNOLD_WORKER_NUM" == 4 && "$ARNOLD_WORKER_GPU" == 8 && "$ARNOLD_ID" =~ ^[0-3]$ ]] || {
  echo 'Expected ARNOLD 4 workers x 8 GPUs with ARNOLD_ID 0..3' >&2; exit 2;
}
[[ "$SAMTOK_RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid SAMTOK_RUN_ID' >&2; exit 2; }
export NODE_RANK="$ARNOLD_ID"
unset PORT MASTER_ADDR MASTER_PORT NNODES GPUS_PER_NODE
RUN="$SAMTOK_EXPERIMENT/runs/$SAMTOK_RUN_ID"
if [[ -e "$RUN/nodes/$ARNOLD_ID" ]]; then
  echo "Run already used: $RUN/nodes/$ARNOLD_ID; set a NEW common SAMTOK_RUN_ID." >&2
  exit 2
fi
mkdir -p "$RUN/bootstrap"
export SAMTOK_RUN_ROOT="$RUN"
export SAMTOK_CUDA_DIAGNOSTICS="$RUN/bootstrap/node${ARNOLD_ID}-cuda"
export SAMTOK_ENV="${SAMTOK_ENV:-/tmp/samtok21-${SAMTOK_RUN_ID}-node${ARNOLD_ID}-env}"
export WANDB_DISABLE_SERVICE=true WANDB_START_METHOD=thread
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTHONPATH="$REPO:$REPO/DiffSynth-Studio${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"
bash scripts/train/setup_cluster_env.sh
"$SAMTOK_ENV/bin/python" -m samtok_edit21.cluster \
  --run-root "$RUN" --data "$SAMTOK_EXPERIMENT/data" \
  --wandb-mode online --wandb-project "${WANDB_PROJECT:-samtok-edit}" \
  --wandb-entity "${WANDB_ENTITY:-2200012743-peking-university}" "$@"
