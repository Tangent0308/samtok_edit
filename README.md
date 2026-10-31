# SAMTokEdit: Pass1 mixed-target localization

The `pass1` branch trains a separate language-model LoRA on Qwen3-VL-8B-SAMTok. A single autoregressive call returns all targets' mask tokens from mixed text, point, box and supplied-mask inputs. The Pass1 LoRA is disabled when the original SAMTok model serves as the Pass2 text encoder.

Start with the [Pass1 method and protocol](docs/pass1_localization.md), [implementation](docs/pass1_implementation.md), [experiments with visual comparisons](docs/pass1_experiments.md), and [reproduction guide](docs/pass1_reproduction.md). The selected formal checkpoint is step 2000; IF-Bench dev20 measures localization only, not edited-image quality. Code, configuration and compact report assets are tracked here; datasets and weights remain in their experiment directories.

The underlying Pass2 edits images with SAMTok mask tokens on the Qwen-Image-2.1 pipeline. A region is represented by two code tokens enclosed in start/end markers; the Qwen3-VL-8B-SAMTok text encoder and this project's DiT LoRA/code injector execute the edit. Pass2 receives region tokens without pixel-mask blending or an inference attention bias. Pass1 may encode a user-supplied mask, while text/point/box inputs go directly to the MLLM.

This development branch is based on **v4** (`qwen-image-2.1-v4`): the round-2 method of v3 (token grounding, trained with flow matching, region terms and attention routing) on the current DiffSynth, plus on-policy self-distillation (OPSD) with a mask-privileged teacher. The v2 two-stage method and the v3 first-round tools are kept on the `qwen-image-2.1-v3` branch.

Project status, working conventions and next steps for a new session (in Chinese): [HANDOFF.md](HANDOFF.md). Data, method and experiment documents: [docs/](docs/README.md).

The Pass1 development base is v4 commit `6aa4e10`. [Frozen experiment identities](configs/pass1/artifacts.json) distinguish measured runs from this code organization; proposed geometric ranking, teacher-label repair and higher-resolution training have not been applied to the reported checkpoint.

## Install

```bash
git clone --branch pass1 --single-branch \
  https://github.com/Tangent0308/samtok_edit.git samtok_edit_pass1
cd samtok_edit_pass1
uv venv --python /usr/bin/python3.11 /path/to/samtok21-env
uv pip install --python /path/to/samtok21-env/bin/python \
  -c configs/pass1/requirements-tested.txt -e ./third_party/diffsynth \
  -e '.[training,data,pass1,test]'
source /path/to/samtok21-env/bin/activate
```

This installs vendored DiffSynth in `third_party/diffsynth` (upstream plus documented hooks, see [third_party/README.md](third_party/README.md)) and the Pass1 dependencies. The root `requirements.txt` retains the older v4 environment; use the Pass1 constraints above for the reported Pass1 implementation. From an uninstalled checkout, put `src:third_party/diffsynth:third_party` on `PYTHONPATH`. Set `SAMTOK_PASS1_MODEL` to the local SAMTok model directory before loading models. Weights, data and outputs live outside the repository.

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

`samtok-pass1 <command>` (or `python -m samtok_edit21.pass1`) provides data selection/materialization, curriculum preparation, pilot/formal training, inference, scoring and auditing. Run `samtok-pass1 --help`; the [reproduction guide](docs/pass1_reproduction.md) contains complete commands and explicit output paths.

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
  - `pass1/`: public protocol/model/grounder API, with separate `data/`, `training/` and `evaluation/` modules; historical module names are compatibility entries.
- `third_party/`: the vendored DiffSynth and SAMTok sources and their notices.
- `configs/pass1/`: sampling configuration, tested dependencies and frozen artifact identities.
- `scripts/pass1/`: data audits and formal/dev20 experiment report procedures.
- `scripts/round2/`: the ARNOLD entry, environment setup and run settings; `scripts/docs/`: documentation checks.
- `tests/`: CPU tests (`pytest`).
- `docs/`: data, method and experiment documents.
