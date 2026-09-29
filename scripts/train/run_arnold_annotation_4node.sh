#!/usr/bin/env bash
set -Eeuo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${SAMTOK_ANNOTATION_RUN_ROOT:?Set shared annotation run directory}"
: "${SAMTOK_ANNOTATION_SOURCES:?Set prepared semantic_sources.jsonl}"
: "${ARNOLD_ID:?ARNOLD must provide the node rank}"
export SAMTOK_RUN_ROOT="$SAMTOK_ANNOTATION_RUN_ROOT"
export SAMTOK_ENV="${SAMTOK_ENV:-/tmp/samtok-noref-${SAMTOK_ANNOTATION_RUN_ID}-node${ARNOLD_ID}-env}"
export SAMTOK_CUDA_DIAGNOSTICS="$SAMTOK_RUN_ROOT/bootstrap/node${ARNOLD_ID}-cuda"
export VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
unset PORT MASTER_ADDR MASTER_PORT NODE_RANK NNODES GPUS_PER_NODE
cd "$REPO"
bash scripts/train/setup_annotation_env.sh
args=()
if [[ -n "${SAMTOK_ANNOTATION_RESUME_FROM:-}" ]]; then
  args+=(--resume-from "$SAMTOK_ANNOTATION_RESUME_FROM")
fi
"$SAMTOK_ENV/bin/python" -m samtok_edit21.annotation_cluster \
  --sources "$SAMTOK_ANNOTATION_SOURCES" --run-root "$SAMTOK_ANNOTATION_RUN_ROOT" \
  --prepared-data-root "$SAMTOK_DATA_EXPERIMENT/data" \
  --model "${SAMTOK_ANNOTATION_MODEL:-/mnt/bn/strategy-mllm-train/common/models/Qwen3-4B-Instruct-2507}" \
  --batch-size "${SAMTOK_ANNOTATION_BATCH_SIZE:-64}" "${args[@]}" "$@"
