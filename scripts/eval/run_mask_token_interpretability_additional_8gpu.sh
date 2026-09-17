#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/opt/tiger/tanyue/samtok_edit}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/crispedit_refined/interpretability/dit_mask_token_counterfactual_additional}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES PYTHONUNBUFFERED=1

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
NUM_WORKERS="${#GPU_IDS[@]}"
if [[ "${NUM_WORKERS}" -lt 1 ]]; then
  echo "CUDA_VISIBLE_DEVICES must contain at least one GPU" >&2
  exit 2
fi

mkdir -p "${EXPERIMENT_ROOT}/logs"
cd "${REPO_ROOT}"

STATUS_FILE="${EXPERIMENT_ROOT}/controller.status"
CURRENT_PHASE="prepare"
started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'status=running\nphase=%s\nstarted_at=%s\nworkers=%s\n' \
  "${CURRENT_PHASE}" "${started_at}" "${NUM_WORKERS}" > "${STATUS_FILE}"

write_final_status() {
  exit_code=$?
  finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if [[ ${exit_code} -eq 0 ]]; then
    final_status="complete"
    final_phase="complete"
  else
    final_status="failed"
    final_phase="${CURRENT_PHASE}"
  fi
  printf 'status=%s\nphase=%s\nstarted_at=%s\nfinished_at=%s\nexit_code=%s\nworkers=%s\n' \
    "${final_status}" "${final_phase}" "${started_at}" "${finished_at}" \
    "${exit_code}" "${NUM_WORKERS}" > "${STATUS_FILE}"
}
trap write_final_status EXIT

"${PYTHON_BIN}" scripts/eval/prepare_mask_token_interventions.py \
  --selection_set additional \
  --output_root "${EXPERIMENT_ROOT}" \
  2>&1 | tee "${EXPERIMENT_ROOT}/logs/prepare.log"

CURRENT_PHASE="inference"
printf 'status=running\nphase=%s\nstarted_at=%s\nworkers=%s\n' \
  "${CURRENT_PHASE}" "${started_at}" "${NUM_WORKERS}" > "${STATUS_FILE}"
"${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc-per-node="${NUM_WORKERS}" \
  --max-restarts=0 \
  --log-dir="${EXPERIMENT_ROOT}/logs/torchrun" \
  --tee=3 \
  scripts/eval/run_mask_token_interpretability.py \
  --manifest "${EXPERIMENT_ROOT}/data/interventions.jsonl" \
  --output_root "${EXPERIMENT_ROOT}" \
  2>&1 | tee "${EXPERIMENT_ROOT}/logs/inference.log"

CURRENT_PHASE="summarize"
printf 'status=running\nphase=%s\nstarted_at=%s\nworkers=%s\n' \
  "${CURRENT_PHASE}" "${started_at}" "${NUM_WORKERS}" > "${STATUS_FILE}"
"${PYTHON_BIN}" scripts/eval/summarize_mask_token_interpretability.py \
  --experiment_root "${EXPERIMENT_ROOT}" \
  2>&1 | tee "${EXPERIMENT_ROOT}/logs/summarize.log"

CURRENT_PHASE="visualize_clear"
printf 'status=running\nphase=%s\nstarted_at=%s\nworkers=%s\n' \
  "${CURRENT_PHASE}" "${started_at}" "${NUM_WORKERS}" > "${STATUS_FILE}"
"${PYTHON_BIN}" scripts/eval/visualize_mask_token_attention_clear.py \
  --experiment_root "${EXPERIMENT_ROOT}" \
  2>&1 | tee "${EXPERIMENT_ROOT}/logs/visualize_clear.log"
