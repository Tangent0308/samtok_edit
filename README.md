# SAMTokEdit for Qwen-Image-2.1

This repository adapts the SAMTok region-localization method to Qwen-Image-2.1. It combines a Qwen3-VL-8B-SAMTok text encoder with the Qwen-Image-2.1 VAE/DiT, and provides strict data validation, two-stage LoRA training, deterministic multi-GPU sample mixing, TE conditioning cache, localization, and image editing inference.

The implementation is based on the DiffSynth version vendored in `DiffSynth-Studio/` and the SAMTok code under `samtok/`. Source datasets are read-only; generated debug data and experiment outputs belong under the experiment directory configured by the user.

## Environment

```bash
python3.11 -m venv /path/to/samtok21-venv
source /path/to/samtok21-venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$PWD:DiffSynth-Studio
```

Set local model paths with `--qwen` and `--samtok`, or use the defaults in `samtok_edit21/models/model.py`.

## Data validation

```bash
python -m samtok_edit21.cli validate \
  --metadata /path/stage1.jsonl --base-path /path/data
```

The protocol supports `edit`, `edit_ntp`, and `edit_umt` rows. Mask spans and localization JSON are validated strictly.
Localization uses the Qwen3-VL-8B-SAMTok chat template without a system message,
with a fixed empty thinking prefix excluded from NTP supervision. Metadata has
an exact field whitelist; annotation/provenance stays in a separate manifest.
The `convert` command accepts annotated units and stored sample rows, and requires
the input encoder's `--mask-tokenizer-sha256` to match the supplied SAMTok codec.

## Training

Stage 1 trains the Qwen3-VL LoRA with NTP plus flow matching:

Choose `--steps` explicitly. To inspect the exact sampling exposure before loading
the training model, run the same command with `--plan-only`; it does not write to
`--output`. Training saves weight snapshots every 2000 per-rank microsteps by
default, aligned to gradient accumulation, plus a final adapter. Snapshots are
not full optimizer-state resumes.

```bash
accelerate launch --num_processes 8 --mixed_precision bf16 \
  -m samtok_edit21.train train --stage stage1 \
  --metadata /path/stage1.jsonl --base-path /path/data \
  --output /path/stage1 --steps 1000 --accumulation 8
```

Build TE/VAE cache with the Stage 1 adapter:

```bash
accelerate launch --num_processes 8 --mixed_precision bf16 \
  -m samtok_edit21.train cache --metadata /path/stage2.jsonl \
  --base-path /path/data --te-adapter /path/stage1/adapter \
  --output /path/cache
```

Stage 2 trains the Qwen-Image-2.1 DiT LoRA from that cache:

```bash
accelerate launch --num_processes 8 --mixed_precision bf16 \
  -m samtok_edit21.train train --stage stage2 \
  --cache /path/cache --output /path/stage2 --steps 1000 --accumulation 4
```

## Inference

Pure-text localization/editing defaults to `--variant ref`. A `noref` ablation
requires reviewed per-unit types via `--units-file`; use `--strict-noref` to
reject fallback. Interactive masks bypass localization and bind to region phrases.

```bash
python -m samtok_edit21.cli localize --image /path/source.png \
  --prompt "Make the leftmost bird blue." \
  --te-adapter /path/stage1/adapter --output /path/localize.json
python -m samtok_edit21.cli infer --image /path/source.png \
  --prompt "..." --te-adapter /path/stage1/adapter \
  --dit-adapter /path/stage2/adapter --output /path/result.png
```

## Documentation

- [Implementation and code references](docs/01_SAMTokEdit_Qwen21_代码实现说明.md): method, official components, project changes, training, inference, and A/C supervision.
- [Experiment history](docs/02_SAMTokEdit_Qwen21_实验记录.md): debug results, model/prompt comparisons, failures, fixes, and validation limits.
- [Four-node run guide](docs/03_SAMTokEdit_Qwen21_四机实验运行指南.md): complete ARNOLD scripts for full training, debug training, and noref conversion.
- [Training data inventory](docs/04_SAMTokEdit_Qwen21_训练数据盘点.md): source filters, final paths, counts, fields, examples, and sampling ratios.

Implementation lives in `samtok_edit21/{schema,models,annotation,training_core,region,distributed}`. The original flat module names remain compatibility entry points. Noref conversion uses Qwen3.5-9B with vLLM, thinking disabled, and validated rule fallback; it does not use W&B. Two-stage training uses W&B online.
