#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit}"
STAGE2_TRAIN_DIR="${STAGE2_TRAIN_DIR:-${EXPERIMENT_ROOT}/stage2_full_edit_mt/stage2_dit_lora}"
STAGE1_OUTPUT_DIR="${STAGE1_OUTPUT_DIR:-${EXPERIMENT_ROOT}/stage1_evaluation/five_settings}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-${REPO_ROOT}/.venv/bin/torchrun}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

if [[ -z "${DIT_LORA:-}" ]]; then
  DIT_LORA="$(find "${STAGE2_TRAIN_DIR}" -maxdepth 1 -type f -name 'step-*.safetensors' -print | sort -V | tail -n 1)"
fi
if [[ -z "${DIT_LORA}" || ! -f "${DIT_LORA}" ]]; then
  echo "No complete Stage-2 step-*.safetensors checkpoint found in ${STAGE2_TRAIN_DIR}" >&2
  exit 2
fi
CHECKPOINT_STEM="$(basename "${DIT_LORA}" .safetensors)"
OUTPUT_BASE="${OUTPUT_BASE:-${EXPERIMENT_ROOT}/stage2_evaluation/${CHECKPOINT_STEM}}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_BASE}/three_settings}"
COMPARISON_DIR="${COMPARISON_DIR:-${OUTPUT_BASE}/eight_settings_comparison}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"
STATUS_FILE="${OUTPUT_DIR}/controller.status"
mkdir -p "${LOG_DIR}"

on_exit() {
  status=$?
  if [[ ${status} -eq 0 ]]; then
    printf 'status=complete\nfinished_at=%s\ndit_lora=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${DIT_LORA}" > "${STATUS_FILE}"
  else
    printf 'status=failed\nexit_code=%s\nfinished_at=%s\ndit_lora=%s\n' \
      "${status}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${DIT_LORA}" > "${STATUS_FILE}"
  fi
}
trap on_exit EXIT

printf 'status=running\nstarted_at=%s\ncontroller_pid=%s\ndit_lora=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" "${DIT_LORA}" > "${STATUS_FILE}"

resume_args=()
if [[ "${RESUME:-0}" == "1" ]]; then
  resume_args+=(--resume)
fi

cd "${REPO_ROOT}"
echo "[controller] Stage-2 evaluation started at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[controller] dit_lora=${DIT_LORA}"
echo "[controller] output_dir=${OUTPUT_DIR} nproc_per_node=${NPROC_PER_NODE}"

"${PYTHON_BIN}" scripts/eval/run_eval.py \
  --dit_lora "${DIT_LORA}" \
  --settings 6 7 8 \
  --output_dir "${OUTPUT_DIR}" \
  --dry_run > "${OUTPUT_DIR}/preflight.json"
echo "[controller] preflight complete"

for setting in 6 7 8; do
  echo "[controller] START setting=${setting} at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "${TORCHRUN_BIN}" \
    --standalone \
    --nnodes=1 \
    --nproc-per-node="${NPROC_PER_NODE}" \
    --max-restarts=0 \
    --log-dir="${LOG_DIR}/setting_${setting}" \
    --tee=3 \
    scripts/eval/run_eval.py \
    --dit_lora "${DIT_LORA}" \
    --settings "${setting}" \
    --output_dir "${OUTPUT_DIR}" \
    --no-make_panels \
    "${resume_args[@]}"
  echo "[controller] COMPLETE setting=${setting} at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
done

echo "[controller] Finalizing Stage-2 S6-S8 report"
"${PYTHON_BIN}" scripts/eval/run_eval.py \
  --dit_lora "${DIT_LORA}" \
  --settings 6 7 8 \
  --output_dir "${OUTPUT_DIR}" \
  --finalize_only \
  --no-make_panels

echo "[controller] Auditing and visualizing all eight settings"
"${PYTHON_BIN}" scripts/eval/analyze_eight_setting_eval.py \
  --stage1_root "${STAGE1_OUTPUT_DIR}" \
  --stage2_root "${OUTPUT_DIR}" \
  --output_dir "${COMPARISON_DIR}"
echo "[controller] Stage-2 evaluation complete at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
