# Pinned upstream dependencies

`diffsynth/` is DiffSynth-Studio 2.1.12 at commit `60b709c4dab8eeabce06ee9bc168328ed4c780e8` (the upstream tree as released) plus two opt-in hooks in `diffsynth/pipelines/qwen_image_21.py`; nothing else differs from upstream. Install this checkout as the `diffsynth` distribution (`requirements.txt` does) before using the training or inference APIs.

The hooks, both inactive unless a caller asks for them:

- `QwenImage21Pipeline.__call__(..., blend_mask=None, blend_latents=None)` and `blend_outside_region`: latent blending outside a region (after every step the outside of the mask is reset to the source noised to the next sigma). Only the teacher test of on-policy self-distillation uses it; the method never blends.
- `QwenImage21Unit_PromptEmbedder.process(..., return_token_ids=False)`: with `True` it also returns `prompt_input_ids`, the token IDs of the encoded prompt after the system tokens are dropped (padding -1), so that the joint-sequence positions of the mask tokens can be found.

Since DiffSynth 2.1.9 the Qwen-Image-2.1 DiT runs the segmented SDPA attention path by default; this package passes `use_flex_attention=True` on every model call (`samtok_edit21.models.pipeline.ATTENTION`) to stay on the block-causal FlexAttention path its adapters were trained with.

`samtok/` retains the official SAMTok source namespace and its README/training references. The `samtok.models` codec dependencies ship in the project wheel; optional upstream training/demo packages remain available in this checkout and are not required for the project API. No SAMTok source or codebook was changed.

[upstream_versions.json](../upstream_versions.json) records the fixed framework/model identities. Project code is in `src/samtok_edit21/`; no generated data, model weights, or experimental outputs are stored here. Original license files and upstream notices remain in their respective directories.
