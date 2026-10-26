# SAMTokEdit for Qwen-Image-2.1

Region editing with SAMTok mask tokens on the Qwen-Image-2.1 editing pipeline. An edit region is encoded by SAMTok into a mask token (four special tokens) that is written into the instruction; the Qwen3-VL-8B-SAMTok text encoder replaces the pipeline's text encoder, and the Qwen-Image-2.1 DiT (with this project's LoRA and code injector) edits the image. At inference there is no pixel mask, no attention bias and no blending: the region reaches the model only as tokens.

This is the **v4** branch (`qwen-image-2.1-v4`): the round-2 method of v3 (token grounding, trained with flow matching, region terms and attention routing) on the current DiffSynth, plus on-policy self-distillation (OPSD) with a mask-privileged teacher. The v2 two-stage method and the v3 first-round tools are kept on the `qwen-image-2.1-v3` branch.

## Install

```bash
git clone --branch qwen-image-2.1-v4 --single-branch \
  https://github.com/Tangent0308/samtok_edit.git samtok_edit_qwen-image-2.1-v4
cd samtok_edit_qwen-image-2.1-v4
uv venv --python /usr/bin/python3.11 /path/to/samtok21-env
uv pip install --python /path/to/samtok21-env/bin/python -r requirements.txt
source /path/to/samtok21-env/bin/activate
```

`requirements.txt` installs the vendored DiffSynth in `third_party/diffsynth` (upstream plus two documented pipeline hooks, see [third_party/README.md](third_party/README.md)) and this package. From a checkout without installing, put `src:third_party/diffsynth:third_party` on `PYTHONPATH`. Model weights, data and experiment outputs live outside the repository.

## Python API

```python
from PIL import Image
from samtok_edit21 import generate, load_codec, load_grounding_adapter, load_pipeline

pipe = load_pipeline(device="cuda")                                    # Qwen-Image-2.1 DiT/VAE + SAMTok TE
codec = load_codec("/path/Qwen3-VL-8B-SAMTok", "cuda")
load_grounding_adapter(pipe, codec, "/path/run/train/adapter_1500")   # DiT LoRA + code injector
source = Image.open("/path/source.png").convert("RGB")
token = codec.encode(source.convert("RGBA"), [mask])[0][0]             # boolean mask -> mask token
image, _ = generate(pipe, codec, f"Replace the selected region {token} with a navy polo shirt.", source,
                    steps=40, seed=0)
```

`generate` runs about 1 MP at the source aspect ratio and returns the edit at the source size; `cfg_scale` with `negative_prompt` set to the prompt without its tokens gives token guidance. `load_pipeline`, `edit`, `generate`, `load_codec`, `SamtokCodec`, `load_adapter` and `load_grounding_adapter` are exposed lazily: importing the package does not load CUDA or models.

## Commands

`samtok-edit <command>` and `python -m samtok_edit21 <command>` are the same CLI; every command has `--help`.

| Command | What it does |
|---|---|
| `round2-data <step>` | Round-2 data: development split, SA-1B mask pool and token caches, training rows, evaluation set |
| `merge-data` | Merge the protocol-2.0 edit data into one training set |
| `round2-train` | The trainer (run under `torch.distributed.run`) |
| `round2-eval` | Development-set probes of an adapter (run under `torch.distributed.run`) |
| `round2-report` | Paired report of a development-set evaluation |

The benchmark v2 adapter is `samtok_edit21.evaluation.benchmark_token` (run by the benchmark's `run-editor`, or by its own multi-GPU `main`). The ARNOLD entry is `scripts/round2/bootstrap_arnold.sh`.

## Layout

- `src/samtok_edit21/`
  - `models/`: the pipeline with the SAMTok text encoder (`pipeline.py`), the SAMTok codec, the code injector, LoRA adapters;
  - `data/`: the region-token protocol, the highlight task, I/O helpers;
  - `preparation/`: instruction protocol 2.0, the round-2 data and rows, the merged dataset;
  - `training/`: the round-2 objective and trainer, attention routing, data-parallel helpers, W&B tracking;
  - `evaluation/`: token-only generation, the development-set probes and report, the benchmark adapter;
  - `regions/`: SAM2 point / box selection.
- `third_party/`: the vendored DiffSynth and SAMTok sources and their notices.
- `scripts/round2/`: the ARNOLD entry, environment setup and run settings; `scripts/docs/`: documentation checks.
- `tests/`: CPU tests (`pytest`).
- `docs/`: data, method and experiment documents.
