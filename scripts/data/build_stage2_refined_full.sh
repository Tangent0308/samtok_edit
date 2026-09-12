#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RUN_ROOT="${RUN_ROOT:-/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/crispedit_refined/stage2_full}"
DATASET_BASE="${DATASET_BASE:-$RUN_ROOT/data/crispedit_samtok}"
REPORT_DIR="${REPORT_DIR:-$RUN_ROOT/reports}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
MASK_POOL_BASE="${MASK_POOL_BASE:-/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/crispedit_refined/stage1_full/data/crispedit_samtok}"
NUM_WORKERS="${NUM_WORKERS:-8}"
CODEC_BATCH_SIZE="${CODEC_BATCH_SIZE:-32}"
SEED="${SEED:-260930}"
EDIT_ROWS="${EDIT_ROWS:-21157}"
UMT_ROWS="${UMT_ROWS:-21157}"

EXPECTED_EDIT_MT_SHA256="4365a3a17758bb96995ce41c795b0faf92434312ccc3484a59d27c55345a1e9b"
EXPECTED_EDIT_UMT_SHA256="76250185c8052fd4ce29a5f59fa2f40a3bdc74f2732effb6c53aa5315edd6966"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"

mkdir -p "$DATASET_BASE" "$REPORT_DIR" "$LOG_DIR"
cd "$REPO_ROOT"

verify_sha256() {
  local expected="$1"
  local path="$2"
  local actual
  actual="$(sha256sum "$path" | cut -d' ' -f1)"
  if [[ "$actual" != "$expected" ]]; then
    printf 'SHA256 mismatch for %s: expected=%s actual=%s\n' \
      "$path" "$expected" "$actual" >&2
    exit 1
  fi
}

# Stage 2 consumes the complete, already codec-encoded refined mask pool built
# through the real Stage-1 builder.  A read-only directory link avoids
# duplicating tens of thousands of immutable image files.  Plain-edit images
# use a separate namespace, so building them cannot mutate the linked pool.
# The final source audit independently checks every selected image byte and
# sampled codec span against authoritative sources.
verify_sha256 "$EXPECTED_EDIT_MT_SHA256" "$MASK_POOL_BASE/edit_mt.jsonl"
verify_sha256 "$EXPECTED_EDIT_UMT_SHA256" "$MASK_POOL_BASE/edit_umt.jsonl"
if [[ ! -e "$DATASET_BASE/images" ]]; then
  ln -s "$MASK_POOL_BASE/images" "$DATASET_BASE/images"
fi
if [[ "$(readlink -f "$DATASET_BASE/images")" != "$(readlink -f "$MASK_POOL_BASE/images")" ]]; then
  printf 'Unexpected mask image pool target: %s\n' "$DATASET_BASE/images" >&2
  exit 1
fi
if [[ ! -f "$DATASET_BASE/edit_mt.jsonl" ]]; then
  cp "$MASK_POOL_BASE/edit_mt.jsonl" "$DATASET_BASE/edit_mt.jsonl"
fi
if [[ ! -f "$DATASET_BASE/edit_umt.jsonl" ]]; then
  cp "$MASK_POOL_BASE/edit_umt.jsonl" "$DATASET_BASE/edit_umt.jsonl"
fi
verify_sha256 "$EXPECTED_EDIT_MT_SHA256" "$DATASET_BASE/edit_mt.jsonl"
verify_sha256 "$EXPECTED_EDIT_UMT_SHA256" "$DATASET_BASE/edit_umt.jsonl"

edit_pids=()
for ((worker = 0; worker < NUM_WORKERS; worker++)); do
  python -u scripts/data/build_edit_metadata.py \
    --output_root "$DATASET_BASE" \
    --image_subdir images_edit \
    --output_jsonl "$DATASET_BASE/edit.jsonl" \
    --sample_rows "$EDIT_ROWS" --seed "$SEED" --ascii_only \
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

python -u scripts/data/compose_training_metadata.py \
  --edit_mt_jsonl "$DATASET_BASE/edit_mt.jsonl" \
  --edit_jsonl "$DATASET_BASE/edit.jsonl" \
  --edit_umt_jsonl "$DATASET_BASE/edit_umt.jsonl" \
  --stage2_output "$DATASET_BASE/stage2.jsonl" \
  --max_edit "$EDIT_ROWS" --max_edit_umt "$UMT_ROWS" \
  --stage2_num_shards 8 --pad_stage2_to_shards --seed "$((SEED + 1))" \
  | tee "$LOG_DIR/compose_stage2.log"

python -u scripts/data/validate_training_metadata.py \
  --metadata_jsonl "$DATASET_BASE/stage2.jsonl" \
  --base_path "$DATASET_BASE" \
  --expected_counts edit_mt:42320,edit:21160,edit_umt:21160 \
  --require_ascii --check_paths --decode_image_sample 1024 --io_workers 32 \
  --seed "$((SEED + 2))" --report_json "$REPORT_DIR/metadata_validation.json" \
  | tee "$LOG_DIR/validate_stage2.log"

python -u scripts/data/audit_refined_metadata.py \
  --stage stage2 --world_size 8 \
  --metadata_jsonl "$DATASET_BASE/stage2.jsonl" \
  --base_path "$DATASET_BASE" \
  --image_byte_sample -1 --codec_sample 128 --codec_device cuda:0 \
  --codec_batch_size "$CODEC_BATCH_SIZE" --io_workers 32 \
  --seed "$((SEED + 3))" \
  --report_json "$REPORT_DIR/source_integrity_audit.json" \
  | tee "$LOG_DIR/audit_source_integrity.log"

sha256sum \
  "$DATASET_BASE/edit_mt.jsonl" \
  "$DATASET_BASE/edit.jsonl" \
  "$DATASET_BASE/edit_umt.jsonl" \
  "$DATASET_BASE/stage2.jsonl" \
  >"$REPORT_DIR/metadata_sha256.txt"

printf 'Refined Stage-2 full data build and audit passed: %s\n' "$RUN_ROOT"
