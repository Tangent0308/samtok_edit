#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RUN_ROOT="${RUN_ROOT:-/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/crispedit_refined/stage1_full}"
DATASET_BASE="${DATASET_BASE:-$RUN_ROOT/data/crispedit_samtok}"
REPORT_DIR="${REPORT_DIR:-$RUN_ROOT/reports}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
NUM_WORKERS="${NUM_WORKERS:-8}"
CODEC_BATCH_SIZE="${CODEC_BATCH_SIZE:-32}"
SEED="${SEED:-260911}"

# SAM2/PyTorch otherwise creates roughly two hundred host threads per worker
# on this 96-core machine, starving all eight GPU pipelines.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"

mkdir -p "$DATASET_BASE" "$REPORT_DIR" "$LOG_DIR"
cd "$REPO_ROOT"

worker_pids=()
for ((worker = 0; worker < NUM_WORKERS; worker++)); do
  python -u scripts/data/build_edit_mt_metadata.py \
    --output_root "$DATASET_BASE" \
    --all_eligible --ascii_only \
    --num_workers "$NUM_WORKERS" --worker_index "$worker" --skip_combine \
    --device "cuda:$worker" --dtype float32 \
    --codec_batch_size "$CODEC_BATCH_SIZE" --resume \
    >"$LOG_DIR/build_edit_mt_umt_worker_${worker}.log" 2>&1 &
  worker_pids+=("$!")
done

worker_failure=0
for ((worker = 0; worker < NUM_WORKERS; worker++)); do
  if ! wait "${worker_pids[$worker]}"; then
    printf 'edit_mt/edit_umt worker %d failed; log=%s\n' \
      "$worker" "$LOG_DIR/build_edit_mt_umt_worker_${worker}.log" >&2
    worker_failure=1
  fi
done
if (( worker_failure )); then
  exit 1
fi

python -u scripts/data/build_edit_mt_metadata.py \
  --output_root "$DATASET_BASE" --combine_only \
  | tee "$LOG_DIR/combine_edit_mt_umt.log"

edit_pids=()
for ((worker = 0; worker < NUM_WORKERS; worker++)); do
  python -u scripts/data/build_edit_metadata.py \
    --output_root "$DATASET_BASE" \
    --output_jsonl "$DATASET_BASE/edit.jsonl" \
    --sample_rows 10579 --seed "$SEED" --ascii_only \
    --num_workers "$NUM_WORKERS" --worker_index "$worker" --skip_combine \
    --resume \
    >"$LOG_DIR/build_edit_worker_${worker}.log" 2>&1 &
  edit_pids+=("$!")
done

edit_failure=0
for ((worker = 0; worker < NUM_WORKERS; worker++)); do
  if ! wait "${edit_pids[$worker]}"; then
    printf 'edit worker %d failed; log=%s\n' \
      "$worker" "$LOG_DIR/build_edit_worker_${worker}.log" >&2
    edit_failure=1
  fi
done
if (( edit_failure )); then
  exit 1
fi

python -u scripts/data/build_edit_metadata.py \
  --output_root "$DATASET_BASE" \
  --output_jsonl "$DATASET_BASE/edit.jsonl" \
  --combine_only \
  | tee "$LOG_DIR/combine_edit.log"

python -u scripts/data/build_edit_ntp_metadata.py \
  --output_jsonl "$DATASET_BASE/edit_ntp_gres.jsonl" \
  --sample_rows 21158 --seed "$((SEED + 1))" --ascii_only --check_images \
  | tee "$LOG_DIR/build_edit_ntp.log"

python -u scripts/data/compose_training_metadata.py \
  --edit_mt_jsonl "$DATASET_BASE/edit_mt.jsonl" \
  --edit_ntp_jsonl "$DATASET_BASE/edit_ntp_gres.jsonl" \
  --edit_jsonl "$DATASET_BASE/edit.jsonl" \
  --edit_umt_jsonl "$DATASET_BASE/edit_umt.jsonl" \
  --stage1_output "$DATASET_BASE/stage1.jsonl" \
  --max_edit_ntp 21158 --max_edit 10579 --max_edit_umt 10579 \
  --pad_stage1_to_ratio --stage1_num_processes 8 --seed "$((SEED + 2))" \
  | tee "$LOG_DIR/compose_stage1.log"

python -u scripts/data/validate_training_metadata.py \
  --metadata_jsonl "$DATASET_BASE/stage1.jsonl" \
  --base_path "$DATASET_BASE" \
  --expected_counts edit_mt:42336,edit_ntp:21168,edit:10584,edit_umt:10584 \
  --require_ascii --check_paths --decode_image_sample 1024 --io_workers 32 \
  --seed "$((SEED + 3))" --report_json "$REPORT_DIR/metadata_validation.json" \
  | tee "$LOG_DIR/validate_stage1.log"

python -u scripts/data/audit_stage1_schedule.py \
  --metadata_jsonl "$DATASET_BASE/stage1.jsonl" \
  --base_path "$DATASET_BASE" --world_size 8 \
  --gradient_accumulation_steps 8 --repeat 1 --seed "$((SEED + 4))" \
  --report_json "$REPORT_DIR/schedule_audit.json" \
  | tee "$LOG_DIR/audit_stage1_schedule.log"

python -u scripts/data/audit_refined_metadata.py \
  --metadata_jsonl "$DATASET_BASE/stage1.jsonl" \
  --base_path "$DATASET_BASE" \
  --image_byte_sample -1 --codec_sample 128 --codec_device cuda:0 \
  --codec_batch_size "$CODEC_BATCH_SIZE" --seed "$((SEED + 5))" \
  --report_json "$REPORT_DIR/source_integrity_audit.json" \
  | tee "$LOG_DIR/audit_source_integrity.log"

sha256sum \
  "$DATASET_BASE/edit_mt.jsonl" \
  "$DATASET_BASE/edit_ntp_gres.jsonl" \
  "$DATASET_BASE/edit.jsonl" \
  "$DATASET_BASE/edit_umt.jsonl" \
  "$DATASET_BASE/stage1.jsonl" \
  >"$REPORT_DIR/metadata_sha256.txt"

printf 'Refined Stage-1 full data build and audit passed: %s\n' "$RUN_ROOT"
