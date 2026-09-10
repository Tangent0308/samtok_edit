#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/opt/tiger/tanyue/samtok_edit}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/crispedit_refined/stage2_evaluation/scaleedit_precision_32}"
DATASET_BASE="${DATASET_BASE:-${EXPERIMENT_ROOT}/data/scaleedit_samtok}"
VALSET="${VALSET:-${DATASET_BASE}/validation.jsonl}"
SINGLE_OUTPUT_DIR="${SINGLE_OUTPUT_DIR:-${EXPERIMENT_ROOT}/three_settings}"
OUTPUT_DIR="${OUTPUT_DIR:-${EXPERIMENT_ROOT}/four_node_settings}"
COMPARISON_DIR="${COMPARISON_DIR:-${EXPERIMENT_ROOT}/single_vs_four_node}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"
STAGE1_TE_LORA="${STAGE1_TE_LORA:-/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/crispedit_refined_4node/crispedit-refined-4node-20260910-run2/stage1_te_lora/step-2648.safetensors}"
STAGE2_DIT_LORA="${STAGE2_DIT_LORA:-/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/crispedit_refined_4node/crispedit-refined-4node-20260910-run2/stage2_dit_lora/step-5296.safetensors}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
SETTING_SEQUENCE="${SETTING_SEQUENCE:-2 3}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES PYTHONUNBUFFERED=1
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

common_args=(
  --valset "${VALSET}"
  --dataset_base "${DATASET_BASE}"
  --output_dir "${OUTPUT_DIR}"
  --stage1_te_lora "${STAGE1_TE_LORA}"
  --stage2_dit_lora "${STAGE2_DIT_LORA}"
  --num_inference_steps "${NUM_INFERENCE_STEPS:-40}"
  --cfg_scale "${CFG_SCALE:-4.0}"
  --samtok_max_new_tokens "${SAMTOK_MAX_NEW_TOKENS:-128}"
  --seed "${SEED:-0}"
)

cd "${REPO_ROOT}"
echo "[controller] Four-node-checkpoint Stage 2 evaluation started at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[controller] valset=${VALSET} output_dir=${OUTPUT_DIR} nproc=${NPROC_PER_NODE}"
echo "[controller] stage1_te_lora=${STAGE1_TE_LORA}"
echo "[controller] stage2_dit_lora=${STAGE2_DIT_LORA}"

for setting in ${SETTING_SEQUENCE}; do
  if [[ ! " ${setting} " =~ ^\ [23]\ $ ]]; then
    echo "Invalid setting in SETTING_SEQUENCE: ${setting}; expected 2 and/or 3" >&2
    exit 2
  fi
  echo "[controller] START setting=${setting} at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node="${NPROC_PER_NODE}" \
    --max-restarts=0 \
    --log-dir="${LOG_DIR}/setting_${setting}" \
    --tee=3 \
    scripts/eval/run_stage2_eval.py \
    "${common_args[@]}" \
    --settings "${setting}" \
    --no-make_panels \
    "${resume_args[@]}"
  echo "[controller] COMPLETE setting=${setting} at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
done

echo "[controller] Building single-node versus four-node visual comparisons"
"${PYTHON_BIN}" scripts/eval/compare_stage2_single_four_node.py \
  --valset "${VALSET}" \
  --dataset_base "${DATASET_BASE}" \
  --single_output_dir "${SINGLE_OUTPUT_DIR}" \
  --four_node_output_dir "${OUTPUT_DIR}" \
  --output_dir "${COMPARISON_DIR}" \
  --device cuda:0

echo "[controller] Four-node-checkpoint evaluation complete at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
