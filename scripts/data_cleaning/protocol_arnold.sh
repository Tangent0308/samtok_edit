#!/usr/bin/env bash
# Post-clone worker entry of the protocol 1.0 adaptation: the same command on all four ARNOLD workers.
# The locked edit-check environment (scripts/edit_check/setup_env.sh) plus the SAM3 packages of
# requirements-protocol-sam3-lock.txt, then samtok_edit21.distributed.protocol_adapt.
set -Eeuo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${SAMTOK_DATASET:?Set the cleaned dataset root (image paths are relative to it)}"
: "${SAMTOK_PROTOCOL_INPUT:?Set the edit-check folder holding manifest_kept*.jsonl}"
: "${SAMTOK_PROTOCOL_OUTPUT:?Set the output folder of the adaptation}"
: "${SAMTOK_LABELING_REPO:?Set the node-local checkout of the CrispEdit labeling repo}"
: "${SAMTOK_RUN_ID:?Set one identical fresh run ID on all workers}"
: "${ARNOLD_ID:?ARNOLD_ID (0..3) is required}"
RUN="$SAMTOK_PROTOCOL_OUTPUT/runs/$SAMTOK_RUN_ID"
export SAMTOK_ENV="${SAMTOK_ENV:-/tmp/samtok-protocol-${SAMTOK_RUN_ID}-node${ARNOLD_ID}-env}"
export SAMTOK_CUDA_DIAGNOSTICS="$RUN/bootstrap/node${ARNOLD_ID}-cuda"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export WANDB_MODE=disabled
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
unset PORT MASTER_ADDR MASTER_PORT NNODES GPUS_PER_NODE
export NODE_RANK="$ARNOLD_ID"
cd "$REPO"
bash scripts/edit_check/setup_env.sh
# SAM3's packages on top of the lock, without touching it; their own dependencies come from the lock.
INDEX="${SAMTOK_INDEX:-https://bytedpypi.byted.org/simple/}"
UV="${UV_BIN:-$(command -v uv || true)}"
UV="${UV:-${SAMTOK_ENV}-uv/bin/uv}"
"$UV" pip install --python "$SAMTOK_ENV/bin/python" --index-url "$INDEX" --no-deps -r requirements-protocol-sam3-lock.txt
"$UV" pip check --python "$SAMTOK_ENV/bin/python"
"$SAMTOK_ENV/bin/python" - "$SAMTOK_LABELING_REPO" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
import cv2, ftfy, iopath, timm  # noqa: F401
from crispedit.mask.pipeline import segment_grounded_box  # noqa: F401
from sam3.model_builder import build_sam3_image_model  # noqa: F401
print("SAM3 imports ready")
PY
export PATH="$SAMTOK_ENV/bin:$PATH"
args=()
if [[ -n "${SAMTOK_PROTOCOL_RESUME_FROM:-}" ]]; then
  args+=(--resume-from "$SAMTOK_PROTOCOL_RESUME_FROM")
fi
if [[ -n "${SAMTOK_LABELING_COMMIT:-}" ]]; then
  args+=(--labeling-commit "$SAMTOK_LABELING_COMMIT")
fi
"$SAMTOK_ENV/bin/python" -B -m samtok_edit21.distributed.protocol_adapt \
  --no-text-action "$SAMTOK_PROTOCOL_INPUT/manifest_kept_no_text_action.jsonl" \
  --kept "$SAMTOK_PROTOCOL_INPUT/manifest_kept.jsonl" --root "$SAMTOK_DATASET" \
  --output "$SAMTOK_PROTOCOL_OUTPUT" --run-root "$RUN" --labeling-repo "$SAMTOK_LABELING_REPO" "${args[@]}" "$@"
