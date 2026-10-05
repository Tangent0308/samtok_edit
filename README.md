# SAMTokEdit for Qwen-Image-2.1

An installable SAMTok extension for the fixed DiffSynth Qwen-Image-2.1 editing pipeline. It combines the Qwen3-VL-8B-SAMTok text encoder with the Qwen-Image-2.1 DiT/VAE, two-stage LoRA training, region supervision, conditioning caches and editing inference.

## Install

```bash
git clone --branch qwen-image-2.1-dev --single-branch \
  https://github.com/Tangent0308/samtok_edit.git samtok_edit_qwen-image-2.1-dev
cd samtok_edit_qwen-image-2.1-dev
uv venv --python /usr/bin/python3.11 /path/to/samtok21-env
uv pip install --python /path/to/samtok21-env/bin/python -r requirements.txt
source /path/to/samtok21-env/bin/activate
```

Install the **vendored project extension** in `third_party/diffsynth` as specified by `requirements.txt`. A stock DiffSynth installation does not include this project's optional training/attention hooks. Model weights, images and training artifacts are external to the package. Dependencies for four-node training and text-only annotation are installed by separate environment scripts; see the run guide.

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

## Layout

- `src/samtok_edit21/`: project package, organized into `data`, `models`, `training`, `regions`, `preparation`, and `distributed`.
- `third_party/`: pinned DiffSynth and SAMTok sources; project-specific framework hooks are documented here.
- `scripts/{training,annotation,diagnostics}/`: operational entry points, separate from reusable Python modules.
- `tests/`: tests against the installed package; `examples/metadata/`: historical small metadata fixtures.
- `docs/`: four current guides and an archive of historical records.

The previous flat compatibility modules have been removed. Current code paths and module entry points are listed in the implementation guide. The layout change preserves model computation, data/cache formats and training recipes.

## Documentation

1. [Implementation](docs/01_SAMTokEdit_Qwen21_代码实现说明.md): overall method, official starting points, project additions, code links and excerpts.
2. [Experiments](docs/02_SAMTokEdit_Qwen21_实验记录.md): debug and production history, validation results and limits.
3. [Four-node runs](docs/03_SAMTokEdit_Qwen21_四机实验运行指南.md): complete ARNOLD/git-clone/W&B entry scripts; noref annotation uses no W&B.
4. [Training data](docs/04_SAMTokEdit_Qwen21_训练数据盘点.md): sources, filtering, prepared paths, counts, protocol, examples and sampling.
