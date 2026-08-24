# SAMTokEdit

This branch implements the two-stage training plan in `SamtokEdit 训练方案.md` for
SAMTok mask-token CoT conditioning of Qwen-Image-Edit-2511.

The checked model defaults intentionally point to:

- `/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen-Image-Edit-2511`
- `/mnt/bn/strategy-mllm-train/user/tanyue/models/SAMTok/Qwen2.5-VL-7B-SAMTok-gres-ft`

There is no 2509 or `-co` fallback in the provided scripts.

## Implemented invariants

- Pass 1 and pass 2 use one `build_edit_model_inputs` function and the exact 2511 edit prompt.
- Template and CoT are tokenized separately, then concatenated at the ID level.
- NTP labels cover canonical CoT plus `<|im_end|>` and use hidden positions starting at `L_T - 1`.
- Pass-1 recovery only removes invalid information; pass 2 only receives canonical CoT.
- Stage 1 trains fp32 TE LoRA parameters with online NTP + FM loss computation.
- Stage 2 fuses the Stage-1 TE LoRA into cached prompt embeddings, then trains only the 2511 DiT LoRA with `zero_cond_t`.
- The Stage-1 schedule gives every rank the same sample type per micro-step and realizes an exact 2:1:1 optimizer-step ratio.

## 1. Prepare the merged processor

```bash
python scripts/data/prepare_samtok_te_dir.py
```

The command verifies a 152179-token vocabulary, the four SAMTok boundary/code
tokens, and TE hash `7792f327a564edcc922f747808b18fb6`. The default output is
`models/merged_samtok_te`.

## 2. Build metadata

GRES already contains released SAMTok codes, so it does not need re-encoding:

```bash
python scripts/data/build_edit_ntp_metadata.py \
  --output_jsonl data/crispedit_samtok/edit_ntp_gres.jsonl \
  --check_images
```

Join the two aligned CrispEdit parquet directories, materialize the kept image
pairs, and encode mask annotations on the source image with VQ-SAM2:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/data/build_edit_mt_metadata.py \
  --output_root data/crispedit_samtok \
  --codec_batch_size 4 \
  --resume
```

This writes `edit_mt.jsonl` and a kept-subset `edit.jsonl`. The released VQ-SAM2
codec stays in fp32 because its prompt positional-encoding path is not safe under
a whole-module bf16 cast. This is independent of the bf16 Qwen/DiT training
precision.

To use every original CrispEdit row for the plain `edit` pool instead:

```bash
python scripts/data/build_edit_metadata.py \
  --output_root data/crispedit_samtok
```

Compose the two training files (choose either `edit.jsonl` or `edit_all.jsonl`):

```bash
python scripts/data/compose_training_metadata.py \
  --edit_mt_jsonl data/crispedit_samtok/edit_mt.jsonl \
  --edit_ntp_jsonl data/crispedit_samtok/edit_ntp_gres.jsonl \
  --edit_jsonl data/crispedit_samtok/edit.jsonl \
  --stage1_output data/crispedit_samtok/stage1.jsonl \
  --stage2_output data/crispedit_samtok/stage2.jsonl
```

GRES image paths are absolute by default, while materialized CrispEdit paths are
relative to `data/crispedit_samtok`; the combined dataset therefore works with
that directory as `dataset_base_path`.

## 3. Train

Stage 1 defaults to 8 processes and gradient accumulation 4. Accumulation must
remain a multiple of the 2:1:1 ratio block length (4).

```bash
bash scripts/train/stage1_te_lora.sh
```

Cache Stage-2 inputs using a completed Stage-1 checkpoint, then train the DiT:

```bash
TE_LORA_PATH=models/stage1_te_lora/step-XXXX.safetensors \
  bash scripts/train/stage2_data_process.sh

bash scripts/train/stage2_dit_lora.sh
```

All shell paths can be overridden with environment variables documented near the
top of each script. CSV loss logging is enabled by default; WandB can be enabled
by adding the standard DiffSynth flags.

## 4. Infer and run the Stage 1 evaluation

```bash
python scripts/inference/infer_samtok_edit.py \
  --image_path input.png \
  --prompt "change the left cat to blue" \
  --te_lora models/stage1_te_lora/step-XXXX.safetensors \
  --dit_lora models/stage2_dit_lora/step-XXXX.safetensors \
  --save_path output.png
```

`scripts/inference/validate.py` runs a small JSONL set with one loaded pipeline.
`scripts/eval/run_stage1_eval.py` is the dedicated Stage 1 evaluation entry point.
It compares stock Qwen-Image-Edit-2511, the initial SAMTok TE, direct editing
with the Stage 1 TE, online predicted-CoT editing, and GT-CoT editing. It writes
per-setting outputs/JSONL records and source/target/S1-S5 comparison panels.
`scripts/eval/run_stage1_eval_8gpu.sh` runs settings 1 through 5 sequentially;
each setting uses eight independent torchrun ranks and shards the 64 validation
rows evenly across the GPUs before rank 0 aggregates that setting.

## Tests

```bash
python -m unittest -v tests/test_samtok_edit.py
```

The tests cover canonical serialization/recovery, dual-codebook validation,
DDP schedule structure, and old/new HF checkpoint key conversion.
