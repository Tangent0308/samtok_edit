# SAMTokEdit

This repository implements two-stage SAMTok mask-token conditioning for
Qwen-Image-Edit-2511. Stage 1 trains the text-encoder LoRA with four data types:
`edit_mt` (NTP+FM), `edit_ntp` (NTP), `edit` (FM), and `edit_umt` (FM with the
mask span embedded in the user prompt). Stage 2 caches the trained text
conditioning and trains the DiT LoRA with `edit_mt:edit:edit_umt=2:1:1`.

The checked model defaults intentionally point to:

- `/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen-Image-Edit-2511`
- `/mnt/bn/strategy-mllm-train/user/tanyue/models/SAMTok/Qwen2.5-VL-7B-SAMTok-gres-ft`

There is no 2509 or `-co` fallback in the provided scripts.

## 0. Create the training environment

The repository defines the currently validated Python 3.11/CUDA 12.8 stack in
`pyproject.toml`. Create an isolated environment with:

```bash
bash setup_env.sh
source .venv/bin/activate
```

The setup uses the checked-in `DiffSynth-Studio` directory as an editable local
package and verifies that it wins over any separately installed DiffSynth. It
also checks the pinned package versions, CUDA availability, and the unit tests.
The default package index is the ByteDance internal PyPI and can be overridden
with `SAMTOK_EDIT_INDEX`; see `bash setup_env.sh --help` for all overrides.

No lockfile is used or generated. The script intentionally invokes the
pip-compatible `uv pip install` interface rather than `uv sync`, because the
latter creates `uv.lock`. Direct dependencies are exactly pinned in
`pyproject.toml`; without a lockfile, transitive dependency resolution is not
bit-for-bit frozen.

## Implemented invariants

- Pass 1 and pass 2 use one `build_edit_model_inputs` function and the exact 2511 edit prompt.
- Template and CoT are tokenized separately, then concatenated at the ID level.
- NTP labels cover canonical CoT plus `<|im_end|>` and use hidden positions starting at `L_T - 1`.
- Pass-1 recovery only removes invalid information; pass 2 only receives canonical CoT.
- Stage 1 trains fp32 TE LoRA parameters with online NTP + FM loss computation.
- Stage 2 fuses the Stage-1 TE LoRA into cached prompt embeddings, then trains only the 2511 DiT LoRA with `zero_cond_t`.
- Refined training data forbids empty masks, empty/global/no-op CoT targets, and non-English text.
- The Stage-1 schedule gives every rank the same sample type per micro-step and realizes an exact `edit_mt:edit_ntp:edit:edit_umt=4:2:1:1` optimizer-step ratio.

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
  --sample_rows 21158 --ascii_only --check_images
```

Join the two aligned CrispEdit parquet directories, materialize the kept image
pairs, and encode mask annotations on the source image with VQ-SAM2:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/data/build_edit_mt_metadata.py \
  --output_root data/crispedit_samtok \
  --all_eligible --ascii_only --codec_batch_size 32 \
  --resume
```

This writes `edit_mt.jsonl` and the user-mask-token counterpart
`edit_umt.jsonl`. The released VQ-SAM2
codec stays in fp32 because its prompt positional-encoding path is not safe under
a whole-module bf16 cast. This is independent of the bf16 Qwen/DiT training
precision.

Build the plain `edit` pool from the aligned fact-prefilter manifest:

```bash
python scripts/data/build_edit_metadata.py \
  --output_root data/crispedit_samtok \
  --output_jsonl data/crispedit_samtok/edit.jsonl \
  --sample_rows 10579 --ascii_only
```

Compose the current full Stage-1 metadata and explicitly mark the minimum rows
duplicated for an exact ratio and complete eight-GPU optimizer steps:

```bash
python scripts/data/compose_training_metadata.py \
  --edit_mt_jsonl data/crispedit_samtok/edit_mt.jsonl \
  --edit_ntp_jsonl data/crispedit_samtok/edit_ntp_gres.jsonl \
  --edit_jsonl data/crispedit_samtok/edit.jsonl \
  --edit_umt_jsonl data/crispedit_samtok/edit_umt.jsonl \
  --stage1_output data/crispedit_samtok/stage1.jsonl \
  --max_edit_ntp 21158 --max_edit 10579 --max_edit_umt 10579 \
  --pad_stage1_to_ratio --stage1_num_processes 8
```

The reproducible full build, validation, source-integrity audit, codec
re-encoding audit, and schedule audit are available as one fail-fast command:

```bash
NUM_WORKERS=8 CODEC_BATCH_SIZE=32 \
  bash scripts/data/build_stage1_refined_full.sh
```

Stage 2 is composed separately with `--stage2_output`,
`--stage2_num_shards 8`, and `--pad_stage2_to_shards` after selecting source
pools in the exact `edit_mt:edit:edit_umt=2:1:1` ratio.

GRES image paths are absolute by default, while materialized CrispEdit paths are
relative to `data/crispedit_samtok`; the combined dataset therefore works with
that directory as `dataset_base_path`.

## 3. Train

Stage 1 defaults to 8 processes and gradient accumulation 8. Accumulation must
remain a multiple of the `4:2:1:1` ratio block length (8).

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
top of each script. CSV and online WandB logging are enabled by default; the
launcher requires explicit `WANDB_API_KEY`, `WANDB_ENTITY`, and `WANDB_PROJECT`
values before loading the model. Set `ENABLE_WANDB_LOG=0` for an intentional
offline run.

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
