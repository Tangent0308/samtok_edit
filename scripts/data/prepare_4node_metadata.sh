#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

: "${RUN_ROOT:?Set RUN_ROOT to the new four-node experiment directory}"

WORLD_SIZE="${WORLD_SIZE:-32}"
SEED="${SEED:-261001}"
STAGE1_SOURCE_BASE="${STAGE1_SOURCE_BASE:-/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/crispedit_refined/stage1_full/data/crispedit_samtok}"
STAGE2_SOURCE_BASE="${STAGE2_SOURCE_BASE:-/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/crispedit_refined/stage2_full/data/crispedit_samtok}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/data}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs/metadata}"
REPORT_DIR="${REPORT_DIR:-$RUN_ROOT/reports/metadata}"
STAGE1_METADATA="${STAGE1_METADATA:-$OUTPUT_DIR/stage1_ws32.jsonl}"
STAGE2_METADATA="${STAGE2_METADATA:-$OUTPUT_DIR/stage2_ws32.jsonl}"

if [[ "$WORLD_SIZE" != "32" ]]; then
  echo "The four-node metadata contract requires WORLD_SIZE=32, got $WORLD_SIZE" >&2
  exit 2
fi
for path in \
  "$STAGE1_SOURCE_BASE/edit_mt.jsonl" \
  "$STAGE1_SOURCE_BASE/edit_ntp_gres.jsonl" \
  "$STAGE1_SOURCE_BASE/edit.jsonl" \
  "$STAGE1_SOURCE_BASE/edit_umt.jsonl" \
  "$STAGE2_SOURCE_BASE/edit_mt.jsonl" \
  "$STAGE2_SOURCE_BASE/edit.jsonl" \
  "$STAGE2_SOURCE_BASE/edit_umt.jsonl"
do
  [[ -f "$path" ]] || { echo "Required refined component metadata is missing: $path" >&2; exit 1; }
done
if [[ -e "$STAGE1_METADATA" || -e "$STAGE2_METADATA" ]]; then
  echo "Refusing to overwrite existing ws32 metadata under $OUTPUT_DIR" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "$REPORT_DIR"
cd "$REPO_ROOT"

python -u scripts/data/compose_training_metadata.py \
  --edit_mt_jsonl "$STAGE1_SOURCE_BASE/edit_mt.jsonl" \
  --edit_ntp_jsonl "$STAGE1_SOURCE_BASE/edit_ntp_gres.jsonl" \
  --edit_jsonl "$STAGE1_SOURCE_BASE/edit.jsonl" \
  --edit_umt_jsonl "$STAGE1_SOURCE_BASE/edit_umt.jsonl" \
  --stage1_output "$STAGE1_METADATA" \
  --max_edit_ntp 21158 \
  --max_edit 10579 \
  --max_edit_umt 10579 \
  --pad_stage1_to_ratio \
  --stage1_num_processes "$WORLD_SIZE" \
  --seed "$SEED" \
  | tee "$LOG_DIR/compose_stage1_ws32.log"

python -u scripts/data/validate_training_metadata.py \
  --metadata_jsonl "$STAGE1_METADATA" \
  --base_path "$STAGE1_SOURCE_BASE" \
  --expected_counts edit_mt:42368,edit_ntp:21184,edit:10592,edit_umt:10592 \
  --require_ascii \
  --check_paths \
  --decode_image_sample 1024 \
  --io_workers 32 \
  --seed "$((SEED + 1))" \
  --report_json "$REPORT_DIR/stage1_metadata_validation.json" \
  | tee "$LOG_DIR/validate_stage1_ws32.log"

python -u scripts/data/audit_stage1_schedule.py \
  --metadata_jsonl "$STAGE1_METADATA" \
  --base_path "$STAGE1_SOURCE_BASE" \
  --world_size "$WORLD_SIZE" \
  --gradient_accumulation_steps 8 \
  --repeat 1 \
  --seed "$((SEED + 2))" \
  --report_json "$REPORT_DIR/stage1_schedule_audit.json" \
  | tee "$LOG_DIR/audit_stage1_schedule_ws32.log"

python -u scripts/data/compose_training_metadata.py \
  --edit_mt_jsonl "$STAGE2_SOURCE_BASE/edit_mt.jsonl" \
  --edit_jsonl "$STAGE2_SOURCE_BASE/edit.jsonl" \
  --edit_umt_jsonl "$STAGE2_SOURCE_BASE/edit_umt.jsonl" \
  --stage2_output "$STAGE2_METADATA" \
  --max_edit 21157 \
  --max_edit_umt 21157 \
  --stage2_num_shards "$WORLD_SIZE" \
  --pad_stage2_to_shards \
  --seed "$((SEED + 3))" \
  | tee "$LOG_DIR/compose_stage2_ws32.log"

python -u scripts/data/validate_training_metadata.py \
  --metadata_jsonl "$STAGE2_METADATA" \
  --base_path "$STAGE2_SOURCE_BASE" \
  --expected_counts edit_mt:42368,edit:21184,edit_umt:21184 \
  --require_ascii \
  --check_paths \
  --decode_image_sample 1024 \
  --io_workers 32 \
  --seed "$((SEED + 4))" \
  --report_json "$REPORT_DIR/stage2_metadata_validation.json" \
  | tee "$LOG_DIR/validate_stage2_ws32.log"

python -u scripts/data/audit_refined_metadata.py \
  --stage stage2 \
  --world_size "$WORLD_SIZE" \
  --metadata_jsonl "$STAGE2_METADATA" \
  --base_path "$STAGE2_SOURCE_BASE" \
  --image_byte_sample 0 \
  --codec_sample 0 \
  --io_workers 32 \
  --seed "$((SEED + 5))" \
  --report_json "$REPORT_DIR/stage2_source_semantic_audit.json" \
  | tee "$LOG_DIR/audit_stage2_semantics_ws32.log"

sha256sum "$STAGE1_METADATA" "$STAGE2_METADATA" >"$REPORT_DIR/metadata_sha256.txt"

python - "$STAGE1_METADATA" "$STAGE2_METADATA" "$REPORT_DIR/metadata_manifest.json" <<'PY'
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

stage1_path, stage2_path, output_path = map(Path, sys.argv[1:])


def describe(path):
    digest = hashlib.sha256()
    counts = Counter()
    rows = 0
    with path.open("rb") as handle:
        for raw in handle:
            digest.update(raw)
            if raw.strip():
                rows += 1
                counts[json.loads(raw)["sample_type"]] += 1
    return {
        "path": str(path.resolve()),
        "rows": rows,
        "sample_type_counts": dict(counts),
        "sha256": digest.hexdigest(),
    }


payload = {
    "world_size": 32,
    "stage1": describe(stage1_path),
    "stage2": describe(stage2_path),
}
output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY

printf 'Four-node metadata preparation passed: %s\n' "$OUTPUT_DIR"
