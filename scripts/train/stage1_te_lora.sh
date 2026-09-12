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
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
TRAIN_EXTRA_ARGS=()
ENABLE_WANDB_LOG="${ENABLE_WANDB_LOG:-1}"
case "$ENABLE_WANDB_LOG" in
  1)
    missing_wandb_vars=()
    for wandb_var in WANDB_API_KEY WANDB_ENTITY WANDB_PROJECT; do
      if [[ -z "${!wandb_var:-}" ]]; then
        missing_wandb_vars+=("$wandb_var")
      fi
    done
    if (( ${#missing_wandb_vars[@]} > 0 )); then
      echo "WandB logging is enabled by default. Set these variables before training: ${missing_wandb_vars[*]}" >&2
      echo "WANDB_API_KEY is read only from the environment and is never written to training_args.json." >&2
      exit 2
    fi
    TRAIN_EXTRA_ARGS+=(--enable_wandb_log)
    ;;
  0)
    TRAIN_EXTRA_ARGS+=(--disable_wandb_log)
    ;;
  *)
    echo "ENABLE_WANDB_LOG must be 0 or 1, got: $ENABLE_WANDB_LOG" >&2
    exit 2
    ;;
esac
if [[ "${DEBUG_TRAIN_METRICS:-0}" == "1" ]]; then
  TRAIN_EXTRA_ARGS+=(--debug_train_metrics --debug_log_steps "${DEBUG_LOG_STEPS:-1}")
fi
if [[ "${FIND_UNUSED_PARAMETERS:-0}" == "1" ]]; then
  TRAIN_EXTRA_ARGS+=(--find_unused_parameters)
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
  LAUNCH_CMD=(
    accelerate launch --multi_gpu --num_processes "$NUM_PROCESSES"
    --main_process_port "${MAIN_PROCESS_PORT:-29500}"
  )
else
  LAUNCH_CMD=(python)
fi
PYTHONPATH="$DIFFSYNTH_DIR:${PYTHONPATH:-}" "${LAUNCH_CMD[@]}" \
  "$SCRIPT_DIR/train_samtok_edit.py" \
  --dataset_base_path "$DATASET_BASE" \
  --dataset_metadata_path "$STAGE1_METADATA" \
  --data_file_keys "image,edit_image" \
  --sample_type_ratio "${SAMPLE_TYPE_RATIO:-edit_mt:4,edit_ntp:2,edit:1,edit_umt:1}" \
  --max_pixels "$MAX_PIXELS" --dataset_repeat "${DATASET_REPEAT:-1}" --dataset_num_workers "${DATASET_WORKERS:-8}" \
  --model_paths "$MODEL_PATHS" \
  --tokenizer_path "$MERGED_TE_DIR" --processor_path "$MERGED_TE_DIR" \
  --lora_base_model text_encoder \
  --lora_target_modules '^model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)$' \
  --lora_rank "${LORA_RANK:-64}" --lora_dropout "${LORA_DROPOUT:-0.05}" \
  --learning_rate "${LEARNING_RATE:-4e-5}" --weight_decay "${WEIGHT_DECAY:-0.05}" \
  --max_grad_norm "${MAX_GRAD_NORM:-1.0}" --warmup_ratio "${WARMUP_RATIO:-0.05}" --num_epochs "$NUM_EPOCHS" \
  --ntp_loss_weight "${NTP_LOSS_WEIGHT:-0.05}" --fm_loss_weight "${FM_LOSS_WEIGHT:-1.0}" \
  --remove_prefix_in_ckpt "pipe.text_encoder." \
  --use_gradient_checkpointing --zero_cond_t \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --seed "${SEED:-0}" \
  --save_steps "${SAVE_STEPS:-2000}" --enable_csv_log \
  --output_path "$OUTPUT_PATH" \
  "${TRAIN_EXTRA_ARGS[@]}" \
  --task sft
