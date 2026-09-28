#!/usr/bin/env bash
# Execute the same command on all four ARNOLD workers.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${SAMTOK_EXPERIMENT:?Set shared experiment directory}"
: "${SAMTOK_RUN_ID:?Set one identical fresh run ID on all workers}"
: "${WANDB_API_KEY:?Inject WANDB_API_KEY through ARNOLD environment/secrets}"
[[ "$SAMTOK_RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid SAMTOK_RUN_ID' >&2; exit 2; }
NODE="${NODE_RANK:-${ARNOLD_ID:?ARNOLD_ID (0..3) is required}}"
RUN="$SAMTOK_EXPERIMENT/runs/$SAMTOK_RUN_ID"
mkdir -p "$RUN/bootstrap"
exec > >(tee -a "$RUN/bootstrap/node${NODE}.log") 2>&1
bootstrap_failed() {
  local result=$?
  mkdir -p "$RUN/nodes/$NODE"
  printf '{"error":"bootstrap failed; see bootstrap/node%s.log","exit_code":%d}\n' \
    "$NODE" "$result" > "$RUN/nodes/$NODE/failure.json"
  exit "$result"
}
trap bootstrap_failed ERR
export SAMTOK_ENV="${SAMTOK_ENV:-/tmp/samtok21-cluster-$SAMTOK_RUN_ID}"
export WANDB_DISABLE_SERVICE=true WANDB_START_METHOD=thread
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTHONPATH="$REPO:$REPO/DiffSynth-Studio${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"
bash scripts/train/setup_cluster_env.sh
exec "$SAMTOK_ENV/bin/python" -m samtok_edit21.cluster \
  --run-root "$RUN" --data "$SAMTOK_EXPERIMENT/data" \
  --wandb-mode online --wandb-project "${WANDB_PROJECT:-samtok-edit}" \
  --wandb-entity "${WANDB_ENTITY:-2200012743-peking-university}" \
  --inference-script "$SAMTOK_EXPERIMENT/tools/inference8.py" \
  --audit-script "$SAMTOK_EXPERIMENT/tools/audit.py" "$@"
