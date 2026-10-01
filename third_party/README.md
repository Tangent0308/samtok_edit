# Pinned upstream dependencies

`diffsynth/` is DiffSynth 2.1.8 at commit `7686e54d41d25c0e8ed5f1318acc23b6bb832654`, plus the existing SAMTokEdit opt-in hooks. The source files were moved without editing their contents. Install this checkout as a separate `diffsynth` distribution before using the training/inference APIs.

The retained extensions are:

- `diffsynth/diffusion/runner.py`: schedule sampler, gradient audit/clip, optimizer-update callback and scheduler clock.
- `diffsynth/core/attention/`: opt-in differentiable FlexAttention LSE.
- `diffsynth/models/qwen_image_21_dit.py`: selected-layer Q/K/LSE statistics through checkpointing.
- `diffsynth/pipelines/qwen_image_21.py`: actual trimmed token IDs and optional attention statistics forwarding.

`samtok/` retains the existing official SAMTok source namespace and its README/training references. The `samtok.models` codec dependencies ship in the project wheel; optional upstream training/demo packages remain available in this checkout and are not required for the project API. No SAMTok source or codebook was changed by the layout refactor.

[upstream_versions.json](../upstream_versions.json) records the fixed framework/model identities. Project adapters are implemented in `src/samtok_edit21/`; no generated data, model weights, or experimental outputs are stored here. Original license files and upstream notices remain in their respective directories.
