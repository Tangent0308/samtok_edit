#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/opt/tiger/tanyue/samtok_edit}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/stage1_evaluation/five_settings}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

mkdir -p "${LOG_DIR}"
STATUS_FILE="${OUTPUT_DIR}/controller.status"

on_exit() {
  status=$?
  if [[ ${status} -eq 0 ]]; then
    printf 'status=complete\nfinished_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS_FILE}"
  else
    printf 'status=failed\nexit_code=%s\nfinished_at=%s\n' \
      "${status}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS_FILE}"
  fi
}
trap on_exit EXIT

printf 'status=running\nstarted_at=%s\ncontroller_pid=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" > "${STATUS_FILE}"

resume_args=()
if [[ "${RESUME:-0}" == "1" ]]; then
  resume_args+=(--resume)
fi

cd "${REPO_ROOT}"
echo "[controller] Stage 1 evaluation started at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[controller] output_dir=${OUTPUT_DIR} nproc_per_node=${NPROC_PER_NODE}"

for setting in 1 2 3 4 5; do
  echo "[controller] START setting=${setting} at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node="${NPROC_PER_NODE}" \
    --max-restarts=0 \
    --log-dir="${LOG_DIR}/setting_${setting}" \
    --tee=3 \
    scripts/eval/run_stage1_eval.py \
    --settings "${setting}" \
    --output_dir "${OUTPUT_DIR}" \
    --no-make_panels \
    "${resume_args[@]}"
  echo "[controller] COMPLETE setting=${setting} at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
done

echo "[controller] Finalizing aggregate report and S1-S5 panels"
"${PYTHON_BIN}" scripts/eval/run_stage1_eval.py \
  --settings all \
  --output_dir "${OUTPUT_DIR}" \
  --finalize_only
echo "[controller] Stage 1 evaluation complete at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
