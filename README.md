# SITEdit (SAMTokEdit) for Qwen-Image-2.1

Interactive, fine-grained, multi-target image editing with region tokens. Every edit region is encoded by SAMTok into a discrete region token (four special tokens, "Space Token") that is written into its own sub-instruction; one joint generation of the Qwen-Image-2.1 DiT, with this project's LoRA and code injector, executes all sub-instructions. At inference there is no pixel mask, no attention bias, no latent blending and no paste-back: regions reach the editor only as tokens.

This is the **delivery branch** `qwen-image-2.1-r2-delivery`: the final method (R2@1000), with only the code it needs.

- **Pass1 (localization):** Qwen3-VL-8B-SAMTok with a language-model LoRA returns, in one autoregressive call, the region tokens of every target from text, points, boxes or supplied masks.
- **Pass2 (editing):** Qwen-Image-2.1 DiT/VAE with the DiT LoRA and code injector of R2@1000, the original SAMTok text encoder, 40 steps and token guidance 2.0.
- **Training:** three stages on the DiT side. S1 learns to read tokens. S2 learns to edit (S2 = the earlier v4 model). R2 continues v4 for 32,000 rows at lr 8e-6 on high-quality composite targets, with no region term on the token-reading rows. Pass1 is trained separately.

Documents (in Chinese):

- [docs/METHOD.md](docs/METHOD.md): method, code, weights and SHA256.
- [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md): every experiment from v4 to R2 with result paths.
- [docs/DATA.md](docs/DATA.md): data inventory.
- [docs/README.md](docs/README.md): index, including the Pass1 documents and the archived v4 documents.

## Install

```bash
git clone --branch qwen-image-2.1-r2-delivery --single-branch https://github.com/Tangent0308/samtok_edit.git
cd samtok_edit
uv venv --python /usr/bin/python3.11 /path/to/samtok21-env
uv pip install --python /path/to/samtok21-env/bin/python -r requirements.txt
source /path/to/samtok21-env/bin/activate
```

`requirements.txt` installs the vendored DiffSynth in `third_party/diffsynth` and this package; [third_party/README.md](third_party/README.md) documents the two pipeline hooks added on top of upstream. To run from a checkout without installing, put `src:third_party/diffsynth:third_party` on `PYTHONPATH`. Model weights, data and experiment outputs live outside the repository; see the weight table in [docs/METHOD.md](docs/METHOD.md) §7.

Pass1 uses newer Transformers (`transformers==5.19.0`) than the editing lock (`5.12.1` in `constraints-tested.txt`). Give it its own environment and do not combine the two constraint files:

```bash
uv venv --python /usr/bin/python3.11 /path/to/samtok-pass1-env
uv pip install --python /path/to/samtok-pass1-env/bin/python \
  -c configs/pass1/requirements-tested.txt -e ./third_party/diffsynth -e '.[training,data,pass1,test]'
```

## Python API

```python
from PIL import Image
from samtok_edit21 import generate, load_codec, load_grounding_adapter, load_pipeline

pipe = load_pipeline(device="cuda")                                    # Qwen-Image-2.1 DiT/VAE + SAMTok TE
codec = load_codec("/path/Qwen3-VL-8B-SAMTok", "cuda")
load_grounding_adapter(pipe, codec, "/path/r2/train/adapter_1000")    # DiT LoRA + code injector (R2@1000)
source = Image.open("/path/source.png").convert("RGB")
token = codec.encode(source.convert("RGBA"), [mask])[0][0]             # boolean mask -> region token
image, _ = generate(pipe, codec, f"Replace the selected region {token} with a navy polo shirt.", source,
                    steps=40, seed=42, cfg_scale=2.0,
                    negative_prompt="Replace the selected region with a navy polo shirt.")   # token guidance 2.0
```

`generate` edits at about 1 MP with the source aspect ratio and returns the result at the source size. The package exposes `load_pipeline`, `edit`, `generate`, `load_codec`, `SamtokCodec`, `load_adapter` and `load_grounding_adapter` lazily, so importing it loads neither CUDA nor any model.

For multi-target requests (Pass1 grounding followed by one joint Pass2 edit), use `samtok_edit21.evaluation.ifbench`. It runs IF-Bench public requests and is configured by `configs/ifbench/main_row.json`.

## Commands

`samtok-edit <command>` and `python -m samtok_edit21 <command>` are the same CLI, and every command has `--help`.

| Command | What it does |
|---|---|
| `merge-data` | Merge the protocol-2.0 edit data (D1, D3, D4, D10–D15) into one training set |
| `sa1b-groups` | Convert the SA-1B twin / multi-target / single-target productions into merge families D12–D15 |
| `round2-data <step>` | Development split, SA-1B region pool and token caches, training rows (`rows-merged --spec`), evaluation set |
| `round2-train` | The trainer (run under `torch.distributed.run`) |
| `round2-eval`, `round2-report` | Development-set token probes of an adapter and their paired report |

Pass1 has its own CLI, `samtok-pass1 <command>`; see [docs/pass1_reproduction.md](docs/pass1_reproduction.md).

Entry points:

| Task | Entry |
|---|---|
| S1 / S2 (four nodes) | `scripts/round2/settings/{s1,s2}_4n.sh` + `scripts/round2/bootstrap_arnold.sh` |
| R2 (four nodes; the exact script that produced the delivered checkpoint) | `scripts/round2/settings/r2_4n.sh` |
| R2 training rows from the self-distilled pools | `scripts/r2_data/` (see [docs/DATA.md](docs/DATA.md) §4) |
| IF-Bench full evaluation of R2 (four nodes) | `scripts/ifbench/settings/r2_full_4n.sh` |

## Layout

- `src/samtok_edit21/`
  - `models/`: pipeline with the SAMTok text encoder, SAMTok codec, code injector, LoRA adapters.
  - `data/`: region-token protocol, highlight task, I/O helpers.
  - `preparation/`: instruction protocol 2.0, merged dataset, SA-1B family conversion, round-2 data and rows.
  - `training/`: round-2 objective and trainer, attention routing, data-parallel helpers, W&B tracking.
  - `evaluation/`: token-only generation, IF-Bench two-pass runner and paired report, development-set probes.
  - `regions/`: changed-region extraction.
  - `pass1/`: Pass1 protocol, model and grounder, with `data/`, `training/` and `evaluation/` modules.
- `scripts/`: `round2/` (ARNOLD entry and settings), `r2_data/` (R2 data generation), `quality/` (self-distillation generator), `ifbench/`, `pass1/`, `docs/`.
- `configs/`: IF-Bench method configuration, Pass1 configurations and frozen artifact identities.
- `third_party/`: vendored DiffSynth and SAMTok sources and their notices.
- `tests/`: CPU tests (`pytest`).
- `docs/`: method, experiments and data documents; Pass1 documents; archived v4 documents.

## Removed from this branch

Other branches keep the following:

- on-policy self-distillation (OPSD);
- low-noise distillation (LND);
- the benchmark 2.x token adapter and the SAM2 point/box selection;
- the C-series run settings.

None of them is part of the final method.
