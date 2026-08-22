#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DIFFSYNTH_DIR="$REPO_ROOT/DiffSynth-Studio"

QWEN_2511="${QWEN_2511:-/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen-Image-Edit-2511}"
SAMTOK_TE="${SAMTOK_TE:-/mnt/bn/strategy-mllm-train/user/tanyue/models/SAMTok/Qwen2.5-VL-7B-SAMTok-gres-ft}"
MERGED_TE_DIR="${MERGED_TE_DIR:-$REPO_ROOT/models/merged_samtok_te}"
DATASET_BASE="${DATASET_BASE:-$REPO_ROOT/data/crispedit_samtok}"
STAGE2_METADATA="${STAGE2_METADATA:-$DATASET_BASE/stage2.jsonl}"
OUTPUT_PATH="${OUTPUT_PATH:-$REPO_ROOT/models/stage2_cache}"
: "${TE_LORA_PATH:?Set TE_LORA_PATH to a completed Stage-1 LoRA safetensors file}"

MODEL_PATHS="$(python - "$SAMTOK_TE" "$QWEN_2511" <<'PY'
import glob, json, os, sys
te, qwen = sys.argv[1:]
paths = [
    sorted(glob.glob(os.path.join(te, "model*.safetensors"))),
    os.path.join(qwen, "vae", "diffusion_pytorch_model.safetensors"),
]
if not all(paths):
    raise SystemExit(f"Missing model files: {paths}")
print(json.dumps(paths))
PY
)"

cd "$DIFFSYNTH_DIR"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
if (( NUM_PROCESSES > 1 )); then
  LAUNCH_CMD=(accelerate launch --multi_gpu --num_processes "$NUM_PROCESSES")
else
  LAUNCH_CMD=(python)
fi
PYTHONPATH="$DIFFSYNTH_DIR:${PYTHONPATH:-}" "${LAUNCH_CMD[@]}" \
  "$SCRIPT_DIR/train_samtok_edit.py" \
  --dataset_base_path "$DATASET_BASE" \
  --dataset_metadata_path "$STAGE2_METADATA" \
  --data_file_keys "image,edit_image" \
  --sample_type_ratio none \
  --max_pixels 1048576 --dataset_num_workers "${DATASET_WORKERS:-8}" \
  --model_paths "$MODEL_PATHS" \
  --tokenizer_path "$MERGED_TE_DIR" --processor_path "$MERGED_TE_DIR" \
  --preset_lora_path "$TE_LORA_PATH" --preset_lora_model text_encoder \
  --lora_base_model dit \
  --remove_prefix_in_ckpt "pipe.dit." \
  --zero_cond_t \
  --output_path "$OUTPUT_PATH" \
  --task sft:data_process
