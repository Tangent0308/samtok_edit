#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DIFFSYNTH_DIR="$REPO_ROOT/DiffSynth-Studio"

QWEN_2511="${QWEN_2511:-/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen-Image-Edit-2511}"
SAMTOK_TE="${SAMTOK_TE:-/mnt/bn/strategy-mllm-train/user/tanyue/models/SAMTok/Qwen2.5-VL-7B-SAMTok-gres-ft}"
MERGED_TE_DIR="${MERGED_TE_DIR:-$REPO_ROOT/models/merged_samtok_te}"
DATASET_BASE="${DATASET_BASE:-$REPO_ROOT/data/crispedit_samtok}"
STAGE1_METADATA="${STAGE1_METADATA:-$DATASET_BASE/stage1.jsonl}"
OUTPUT_PATH="${OUTPUT_PATH:-$REPO_ROOT/models/stage1_te_lora}"
MAX_PIXELS="${MAX_PIXELS:-1048576}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
TRAIN_EXTRA_ARGS=()
if [[ "${DEBUG_TRAIN_METRICS:-0}" == "1" ]]; then
  TRAIN_EXTRA_ARGS+=(--debug_train_metrics --debug_log_steps "${DEBUG_LOG_STEPS:-1}")
fi

MODEL_PATHS="$(python - "$SAMTOK_TE" "$QWEN_2511" <<'PY'
import glob, json, os, sys
te, qwen = sys.argv[1:]
paths = [
    sorted(glob.glob(os.path.join(te, "model*.safetensors"))),
    sorted(glob.glob(os.path.join(qwen, "transformer", "diffusion_pytorch_model*.safetensors"))),
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
  --dataset_metadata_path "$STAGE1_METADATA" \
  --data_file_keys "image,edit_image" \
  --sample_type_ratio "edit_mt:2,edit_ntp:1,edit:1" \
  --max_pixels "$MAX_PIXELS" --dataset_repeat "${DATASET_REPEAT:-1}" --dataset_num_workers "${DATASET_WORKERS:-8}" \
  --model_paths "$MODEL_PATHS" \
  --tokenizer_path "$MERGED_TE_DIR" --processor_path "$MERGED_TE_DIR" \
  --lora_base_model text_encoder \
  --lora_target_modules '^model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)$' \
  --lora_rank 64 --lora_dropout 0.05 \
  --learning_rate 4e-5 --weight_decay 0.05 --max_grad_norm 1.0 --warmup_ratio 0.05 --num_epochs "$NUM_EPOCHS" \
  --ntp_loss_weight 1.0 --fm_loss_weight 1.0 \
  --remove_prefix_in_ckpt "pipe.text_encoder." \
  --use_gradient_checkpointing --zero_cond_t --find_unused_parameters \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --save_steps "${SAVE_STEPS:-2000}" --enable_csv_log \
  --output_path "$OUTPUT_PATH" \
  "${TRAIN_EXTRA_ARGS[@]}" \
  --task sft
