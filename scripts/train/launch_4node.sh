#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DIFFSYNTH_DIR="$REPO_ROOT/DiffSynth-Studio"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/arnold_4node_env.sh"

show_help() {
  cat <<'EOF'
Launch one SAMTokEdit phase on four Arnold nodes (8 GPUs per node).

Usage:
  bash scripts/train/launch_4node.sh stage1
  bash scripts/train/launch_4node.sh stage2_cache
  bash scripts/train/launch_4node.sh stage2_train

Every Arnold worker must invoke the same command concurrently. The topology is
derived from ARNOLD_WORKER_HOSTS, ARNOLD_WORKER_NUM, ARNOLD_WORKER_GPU, and
ARNOLD_ID. MASTER_ADDR/MASTER_PORT/NNODES/NODE_RANK can override those values.

Required phase variables:
  stage1:      DATASET_BASE STAGE1_METADATA OUTPUT_PATH MERGED_TE_DIR
  stage2_cache: DATASET_BASE STAGE2_METADATA OUTPUT_PATH TE_LORA_PATH MERGED_TE_DIR
  stage2_train: CACHE_ROOT OUTPUT_PATH MERGED_TE_DIR

WANDB_API_KEY, WANDB_ENTITY, and WANDB_PROJECT are required for training phases.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  show_help
  exit 0
fi
if (( $# != 1 )); then
  show_help >&2
  exit 2
fi
PHASE="$1"
samtok_init_arnold_topology

QWEN_2511="${QWEN_2511:-/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen-Image-Edit-2511}"
SAMTOK_TE="${SAMTOK_TE:-/mnt/bn/strategy-mllm-train/user/tanyue/models/SAMTok/Qwen2.5-VL-7B-SAMTok-gres-ft}"
MAX_PIXELS="${MAX_PIXELS:-1048576}"
DATASET_WORKERS="${DATASET_WORKERS:-2}"
PYTHON_BIN="${PYTHON_BIN:-${SAMTOK_EDIT_VENV:-$REPO_ROOT/.venv}/bin/python}"

[[ -x "$PYTHON_BIN" ]] || { echo "Python environment is missing: $PYTHON_BIN" >&2; exit 1; }
[[ -d "$DIFFSYNTH_DIR" ]] || { echo "Vendored DiffSynth-Studio is missing: $DIFFSYNTH_DIR" >&2; exit 1; }
[[ -d "$QWEN_2511" ]] || { echo "Qwen-Image-Edit-2511 is missing: $QWEN_2511" >&2; exit 1; }

require_path() {
  local name="$1"
  local value="${!name:-}"
  [[ -n "$value" ]] || { echo "Set $name for phase $PHASE" >&2; exit 2; }
}

require_wandb() {
  local name
  for name in WANDB_API_KEY WANDB_ENTITY WANDB_PROJECT; do
    [[ -n "${!name:-}" ]] || { echo "Set $name before phase $PHASE" >&2; exit 2; }
  done
}

configure_four_node_wandb() {
  # byted-wandb 0.13.98 unconditionally enables a subprocess service whose
  # startup timeout is hard-coded to 30 seconds.  Under 32-rank model-loading
  # pressure that subprocess can miss the deadline and crash rank 0.  The
  # supported legacy thread backend has no service port-file handshake.
  export WANDB_DISABLE_SERVICE=true
  export WANDB_START_METHOD=thread
}

model_paths_stage1() {
  "$PYTHON_BIN" - "$SAMTOK_TE" "$QWEN_2511" <<'PY'
import glob
import json
import os
import sys

te, qwen = sys.argv[1:]
paths = [
    sorted(glob.glob(os.path.join(te, "model*.safetensors"))),
    sorted(glob.glob(os.path.join(qwen, "transformer", "diffusion_pytorch_model*.safetensors"))),
    os.path.join(qwen, "vae", "diffusion_pytorch_model.safetensors"),
]
if not all(paths):
    raise SystemExit(f"Missing Stage-1 model files: {paths}")
print(json.dumps(paths))
PY
}

model_paths_stage2_cache() {
  "$PYTHON_BIN" - "$SAMTOK_TE" "$QWEN_2511" <<'PY'
import glob
import json
import os
import sys

te, qwen = sys.argv[1:]
paths = [
    sorted(glob.glob(os.path.join(te, "model*.safetensors"))),
    os.path.join(qwen, "vae", "diffusion_pytorch_model.safetensors"),
]
if not all(paths):
    raise SystemExit(f"Missing Stage-2 cache model files: {paths}")
print(json.dumps(paths))
PY
}

model_paths_stage2_train() {
  "$PYTHON_BIN" - "$QWEN_2511" <<'PY'
import glob
import json
import os
import sys

qwen = sys.argv[1]
paths = [sorted(glob.glob(os.path.join(qwen, "transformer", "diffusion_pytorch_model*.safetensors")))]
if not paths[0]:
    raise SystemExit(f"Missing Qwen-Image-Edit-2511 transformer under {qwen}")
print(json.dumps(paths))
PY
}

TRAIN_EXTRA_ARGS=()
if [[ "${DEBUG_TRAIN_METRICS:-0}" == "1" ]]; then
  TRAIN_EXTRA_ARGS+=(--debug_train_metrics --debug_log_steps "${DEBUG_LOG_STEPS:-1}")
fi

export PYTHONPATH="$DIFFSYNTH_DIR:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export DIFFSYNTH_SKIP_DOWNLOAD=True
cd "$DIFFSYNTH_DIR"

case "$PHASE" in
  stage1)
    require_path DATASET_BASE
    require_path STAGE1_METADATA
    require_path OUTPUT_PATH
    require_path MERGED_TE_DIR
    require_wandb
    configure_four_node_wandb
    [[ -d "$SAMTOK_TE" ]] || { echo "SAMTok gres-ft TE is missing: $SAMTOK_TE" >&2; exit 1; }
    [[ -f "$STAGE1_METADATA" ]] || { echo "Stage-1 metadata is missing: $STAGE1_METADATA" >&2; exit 1; }
    [[ -d "$MERGED_TE_DIR" ]] || { echo "Merged TE directory is missing: $MERGED_TE_DIR" >&2; exit 1; }
    MODEL_PATHS="$(model_paths_stage1)"
    if [[ "${FIND_UNUSED_PARAMETERS:-0}" == "1" ]]; then
      TRAIN_EXTRA_ARGS+=(--find_unused_parameters)
    fi
    samtok_accelerate_launch \
      "$SCRIPT_DIR/train_samtok_edit.py" \
      --dataset_base_path "$DATASET_BASE" \
      --dataset_metadata_path "$STAGE1_METADATA" \
      --data_file_keys "image,edit_image" \
      --sample_type_ratio "${SAMPLE_TYPE_RATIO:-edit_mt:4,edit_ntp:2,edit:1,edit_umt:1}" \
      --max_pixels "$MAX_PIXELS" \
      --dataset_repeat "${DATASET_REPEAT:-1}" \
      --dataset_num_workers "$DATASET_WORKERS" \
      --model_paths "$MODEL_PATHS" \
      --tokenizer_path "$MERGED_TE_DIR" \
      --processor_path "$MERGED_TE_DIR" \
      --lora_base_model text_encoder \
      --lora_target_modules '^model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)$' \
      --lora_rank "${LORA_RANK:-64}" \
      --lora_dropout "${LORA_DROPOUT:-0.05}" \
      --learning_rate "${LEARNING_RATE:-4e-5}" \
      --weight_decay "${WEIGHT_DECAY:-0.05}" \
      --max_grad_norm "${MAX_GRAD_NORM:-1.0}" \
      --warmup_ratio "${WARMUP_RATIO:-0.05}" \
      --num_epochs "${NUM_EPOCHS:-1}" \
      --ntp_loss_weight "${NTP_LOSS_WEIGHT:-0.05}" \
      --fm_loss_weight "${FM_LOSS_WEIGHT:-1.0}" \
      --remove_prefix_in_ckpt "pipe.text_encoder." \
      --use_gradient_checkpointing \
      --zero_cond_t \
      --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-8}" \
      --seed "${SEED:-0}" \
      --save_steps "${SAVE_STEPS:-2000}" \
      --enable_csv_log \
      --enable_wandb_log \
      --eager_init_loggers \
      --output_path "$OUTPUT_PATH" \
      "${TRAIN_EXTRA_ARGS[@]}" \
      --task sft
    ;;

  stage2_cache)
    require_path DATASET_BASE
    require_path STAGE2_METADATA
    require_path OUTPUT_PATH
    require_path TE_LORA_PATH
    require_path MERGED_TE_DIR
    [[ -d "$SAMTOK_TE" ]] || { echo "SAMTok gres-ft TE is missing: $SAMTOK_TE" >&2; exit 1; }
    [[ -f "$STAGE2_METADATA" ]] || { echo "Stage-2 metadata is missing: $STAGE2_METADATA" >&2; exit 1; }
    [[ -f "$TE_LORA_PATH" ]] || { echo "Stage-1 TE LoRA is missing: $TE_LORA_PATH" >&2; exit 1; }
    [[ -d "$MERGED_TE_DIR" ]] || { echo "Merged TE directory is missing: $MERGED_TE_DIR" >&2; exit 1; }
    MODEL_PATHS="$(model_paths_stage2_cache)"
    samtok_accelerate_launch \
      "$SCRIPT_DIR/train_samtok_edit.py" \
      --dataset_base_path "$DATASET_BASE" \
      --dataset_metadata_path "$STAGE2_METADATA" \
      --data_file_keys "image,edit_image" \
      --sample_type_ratio none \
      --max_pixels "$MAX_PIXELS" \
      --dataset_num_workers "$DATASET_WORKERS" \
      --model_paths "$MODEL_PATHS" \
      --tokenizer_path "$MERGED_TE_DIR" \
      --processor_path "$MERGED_TE_DIR" \
      --preset_lora_path "$TE_LORA_PATH" \
      --preset_lora_model text_encoder \
      --lora_base_model dit \
      --remove_prefix_in_ckpt "pipe.dit." \
      --zero_cond_t \
      --output_path "$OUTPUT_PATH" \
      --disable_wandb_log \
      "${TRAIN_EXTRA_ARGS[@]}" \
      --task sft:data_process
    ;;

  stage2_train)
    require_path CACHE_ROOT
    require_path OUTPUT_PATH
    require_path MERGED_TE_DIR
    require_wandb
    configure_four_node_wandb
    [[ -d "$CACHE_ROOT" ]] || { echo "Stage-2 cache is missing: $CACHE_ROOT" >&2; exit 1; }
    [[ -d "$MERGED_TE_DIR" ]] || { echo "Merged TE directory is missing: $MERGED_TE_DIR" >&2; exit 1; }
    MODEL_PATHS="$(model_paths_stage2_train)"
    samtok_accelerate_launch \
      "$SCRIPT_DIR/train_samtok_edit.py" \
      --dataset_base_path "$CACHE_ROOT" \
      --sample_type_ratio none \
      --max_pixels "$MAX_PIXELS" \
      --dataset_repeat "${DATASET_REPEAT:-2}" \
      --dataset_num_workers "$DATASET_WORKERS" \
      --model_paths "$MODEL_PATHS" \
      --tokenizer_path "$MERGED_TE_DIR" \
      --processor_path "$MERGED_TE_DIR" \
      --learning_rate "${LEARNING_RATE:-1e-4}" \
      --weight_decay "${WEIGHT_DECAY:-0.01}" \
      --num_epochs "${NUM_EPOCHS:-1}" \
      --lora_base_model dit \
      --lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1" \
      --lora_rank 32 \
      --remove_prefix_in_ckpt "pipe.dit." \
      --use_gradient_checkpointing \
      --zero_cond_t \
      --find_unused_parameters \
      --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-1}" \
      --save_steps "${SAVE_STEPS:-4000}" \
      --enable_csv_log \
      --enable_wandb_log \
      --eager_init_loggers \
      --output_path "$OUTPUT_PATH" \
      "${TRAIN_EXTRA_ARGS[@]}" \
      --task sft:train
    ;;

  *)
    echo "Unknown phase: $PHASE" >&2
    show_help >&2
    exit 2
    ;;
esac
