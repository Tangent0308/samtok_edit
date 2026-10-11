#!/usr/bin/env bash
# Post-clone worker entry of the MLLM edit check: the same command on all four ARNOLD workers.
set -Eeuo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${SAMTOK_CHECK_MANIFEST:?Set the cleaned manifest.jsonl}"
: "${SAMTOK_CHECK_OUTPUT:?Set the output folder of the check}"
: "${SAMTOK_RUN_ID:?Set one identical fresh run ID on all workers}"
: "${ARNOLD_ID:?ARNOLD_ID (0..3) is required}"
RUN="$SAMTOK_CHECK_OUTPUT/runs/$SAMTOK_RUN_ID"
export SAMTOK_ENV="${SAMTOK_ENV:-/tmp/samtok-edit-check-${SAMTOK_RUN_ID}-node${ARNOLD_ID}-env}"
export SAMTOK_CUDA_DIAGNOSTICS="$RUN/bootstrap/node${ARNOLD_ID}-cuda"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export WANDB_MODE=disabled
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
unset PORT MASTER_ADDR MASTER_PORT NNODES GPUS_PER_NODE
export NODE_RANK="$ARNOLD_ID"
cd "$REPO"
bash scripts/edit_check/setup_env.sh
export PATH="$SAMTOK_ENV/bin:$PATH"
args=()
if [[ -n "${SAMTOK_CHECK_RESUME_FROM:-}" ]]; then
  args+=(--resume-from "$SAMTOK_CHECK_RESUME_FROM")
fi
"$SAMTOK_ENV/bin/python" -m samtok_edit21.distributed.edit_check \
  --manifest "$SAMTOK_CHECK_MANIFEST" --output "$SAMTOK_CHECK_OUTPUT" --run-root "$RUN" "${args[@]}" "$@"
