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

Set local model paths with `--qwen` and `--samtok`, or use the defaults in `samtok_edit21/model.py`.

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

For the complete implementation notes and code references, read [`docs/SAMTokEdit_Qwen21_代码实现与使用.md`](docs/SAMTokEdit_Qwen21_代码实现与使用.md). Smoke commands and results are recorded in [`docs/SAMTokEdit_Qwen21_实验记录.md`](docs/SAMTokEdit_Qwen21_实验记录.md).

The attention supervision and regional flow-matching loss are implemented. See [the four-node training guide](docs/SAMTokEdit_Qwen21_四机训练运行指南.md) for the multi-node runner, W&B setup, and debug launch.

The [full dataset inventory and conversion audit](docs/SAMTokEdit_Qwen21_全量数据盘点与转换审计.md) documents final quality filters, all four datasets' type mappings, and the semantic review required before exporting no-reference instructions.

四数据集的 Qwen3-4B + vLLM 候选转换入口、八卡验证及质量限制见[四机指南第 7 节](docs/SAMTokEdit_Qwen21_四机训练运行指南.md#7-全量-noref-语义转换qwen3-4b--vllm)。该转换独立于训练的 W&B 记录，无需 W&B key。

当前精简版的 prompt、程序后处理、逐条审阅口径和速度对比见[noref 两字段转换与 4B/8B 对比](docs/SAMTokEdit_Qwen21_noref两字段转换与模型对比.md)。
