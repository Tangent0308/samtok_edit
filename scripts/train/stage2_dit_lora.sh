#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DIFFSYNTH_DIR="$REPO_ROOT/DiffSynth-Studio"

QWEN_2511="${QWEN_2511:-/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen-Image-Edit-2511}"
MERGED_TE_DIR="${MERGED_TE_DIR:-$REPO_ROOT/models/merged_samtok_te}"
CACHE_ROOT="${CACHE_ROOT:-$REPO_ROOT/models/stage2_cache}"
OUTPUT_PATH="${OUTPUT_PATH:-$REPO_ROOT/models/stage2_dit_lora}"

MODEL_PATHS="$(python - "$QWEN_2511" <<'PY'
import glob, json, os, sys
qwen = sys.argv[1]
paths = [sorted(glob.glob(os.path.join(qwen, "transformer", "diffusion_pytorch_model*.safetensors")))]
if not paths[0]:
    raise SystemExit(f"Missing Qwen-Image-Edit-2511 transformer under {qwen}")
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
  --dataset_base_path "$CACHE_ROOT" \
  --sample_type_ratio none \
  --max_pixels 1048576 --dataset_repeat 2 --dataset_num_workers "${DATASET_WORKERS:-8}" \
  --model_paths "$MODEL_PATHS" \
  --tokenizer_path "$MERGED_TE_DIR" --processor_path "$MERGED_TE_DIR" \
  --learning_rate 1e-4 --num_epochs 5 \
  --lora_base_model dit \
  --lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1" \
  --lora_rank 32 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --use_gradient_checkpointing --zero_cond_t --find_unused_parameters \
  --save_steps "${SAVE_STEPS:-4000}" --enable_csv_log \
  --output_path "$OUTPUT_PATH" \
  --task sft:train
