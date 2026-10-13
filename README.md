# SAMTokEdit for Qwen-Image-2.1

An installable SAMTok extension for the fixed DiffSynth Qwen-Image-2.1 editing pipeline. An edit region is encoded by SAMTok into mask tokens that are written into the prompt; the Qwen3-VL-8B-SAMTok text encoder and the Qwen-Image-2.1 DiT/VAE do the rest, with no pixel mask at inference.

This is the **v3** branch (`qwen-image-2.1-v3`). v3 trains the DiT to understand what a mask token refers to (token grounding: a highlight task, an attention-routing loss, a code-vector side channel); masks are used only as training supervision. The v2 two-stage method (localization LoRA, structural region binding, conditioning caches) is kept for baselines.

## Install

```bash
git clone --branch qwen-image-2.1-v3 --single-branch \
  https://github.com/Tangent0308/samtok_edit.git samtok_edit_qwen-image-2.1-v3
cd samtok_edit_qwen-image-2.1-v3
uv venv --python /usr/bin/python3.11 /path/to/samtok21-env
uv pip install --python /path/to/samtok21-env/bin/python -r requirements.txt
source /path/to/samtok21-env/bin/activate
```

Install the **vendored project extension** in `third_party/diffsynth` as specified by `requirements.txt`. A stock DiffSynth installation does not include this project's optional training/attention hooks. Model weights, images and training artifacts are external to the package. Dependencies for four-node training, text-only annotation and the MLLM edit check of the cleaned data are installed by separate environment scripts; see the run guide.

## Python API

```python
from PIL import Image
from samtok_edit21 import load_pipeline, load_adapter, edit

pipe = load_pipeline(device="cuda")
load_adapter(pipe.text_encoder, "/path/stage1/adapter")
load_adapter(pipe.dit, "/path/stage2/adapter")
pipe.eval()
image, report = edit(
    pipe, "Make the leftmost bird blue.",
    [Image.open("/path/source.png").convert("RGBA")],
    mode="online", height=1024, width=1024, seed=0,
)
image.save("/path/result.png")
```

`load_pipeline`, `edit`, `localize`, `SamtokCodec` and `load_adapter` lazily expose the implementation objects. Importing the package itself does not load CUDA or models. Inference and the training engine work from an installed package outside the checkout; cluster orchestration additionally uses checkout scripts.

## Commands

`samtok-edit` and `python -m samtok_edit21` use the same CLI:

```bash
samtok-edit validate --metadata /path/stage1.jsonl --base-path /path/data
python -m torch.distributed.run --nproc_per_node 8 \
  -m samtok_edit21 train --stage stage1 --metadata /path/stage1.jsonl \
  --base-path /path/data --output /path/stage1 --steps 1000 --accumulation 8
python -m torch.distributed.run --nproc_per_node 8 \
  -m samtok_edit21 cache --metadata /path/stage2.jsonl \
  --base-path /path/data --te-adapter /path/stage1/adapter --output /path/cache
python -m torch.distributed.run --nproc_per_node 8 \
  -m samtok_edit21 train --stage stage2 --cache /path/cache \
  --output /path/stage2 --steps 1000 --accumulation 4
samtok-edit infer --image /path/source.png --prompt "Make the leftmost bird blue." \
  --te-adapter /path/stage1/adapter --dit-adapter /path/stage2/adapter \
  --output /path/result.png
```

These are minimal usage examples. The four-node guide contains the full production recipe with region/attention losses, prepared data, ARNOLD and W&B configuration. `--plan-only` inspects training exposure without loading models. Checkpoints are adapter weight snapshots, not complete optimizer-state resumes. Default inference canvas is 1024 × 1024, as before.

## v3 token grounding

```python
from PIL import Image
from samtok_edit21 import edit, load_codec, load_grounding_adapter, load_pipeline
from samtok_edit21.models.binding import BindingConfig

pipe = load_pipeline(device="cuda")
codec = load_codec("/path/Qwen3-VL-8B-SAMTok", "cuda")
load_grounding_adapter(pipe, codec, "/path/run/train/adapter_1300")   # DiT LoRA + code injector
source = Image.open("/path/source.png").convert("RGBA")
span = codec.encode(source, [mask])[0][0]                              # boolean mask -> mask-token span
image, _ = edit(pipe, f"Remove the object in this region {span}.", [source], mode="inline",
                height=1024, width=1024, seed=0, pass2_te="raw", binding=BindingConfig("none"),
                codec=codec, blend_region=None)
```

```bash
python -m torch.distributed.run --nproc_per_node 8 -m samtok_edit21 ground-train --data /path/data --output /path/run/train
python -m torch.distributed.run --nproc_per_node 8 -m samtok_edit21 ground-eval \
  --data /path/data --adapter /path/run/train/adapter_1300 --output /path/run/eval
samtok-edit ground-report --eval /path/run/eval --data /path/data --train /path/run/train --output /path/run/analysis.txt
```

`ground-data` builds the training rows and the evaluation set, `ground-gallery` builds a trained-vs-base page; every command has `--help`. The four-node entry is `scripts/grounding/bootstrap_arnold.sh`.

## Layout

- `src/samtok_edit21/`: project package, organized into `data`, `models`, `training`, `evaluation`, `regions`, `preparation`, and `distributed`. The v3 modules are `models/code_injector.py`, `data/highlight.py`, `training/{routing,grounding,grounding_engine}.py`, `preparation/grounding_data.py` and `evaluation/{grounding,grounding_report,gallery}.py`.
- `third_party/`: pinned DiffSynth and SAMTok sources; project-specific framework hooks are documented here.
- `scripts/{grounding,training,annotation,data_cleaning,edit_check,diagnostics,docs}/`: operational entry points, separate from reusable Python modules.
- `tests/`: tests against the installed package; `examples/metadata/`: historical small metadata fixtures.
- `docs/`: five current documents and an archive of historical records.

The previous flat compatibility modules have been removed. Current code paths and module entry points are listed in the implementation guide. The layout change preserves model computation, data/cache formats and training recipes.

## Documentation

1. [Implementation](docs/01_SAMTokEdit_Qwen21_代码实现说明.md): how the method is built on the official DiffSynth and SAMTok code, with code paths and excerpts kept in sync with the source.
2. [v2 analysis](docs/02_SAMTokEdit_Qwen21_v2分析.md): why v2 fell behind the stock model and whether its DiT used the mask tokens; evidence and artifact paths.
3. [v3 plan and progress](docs/03_SAMTokEdit_Qwen21_v3优化计划与进度.md): related work, the method, the experiment plan and every result so far.
4. [Data](docs/04_SAMTokEdit_Qwen21_数据工作.md): training data and benchmark inventory, data problems, cleaning and protocol adaptation, and which paths are in use.
5. [Four-node runs](docs/05_SAMTokEdit_Qwen21_四机实验运行指南.md): the ARNOLD entry for v3 runs with one worked example.

The v2-era documents, including the v2 four-node guide for Stage 1 / Stage 2 / the MLLM edit check, are in [docs/archive/v2/](docs/archive/v2/).
