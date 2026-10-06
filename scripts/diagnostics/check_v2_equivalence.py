"""GPU equivalence checks for the v2 inference/training hooks (one GPU).

1. Pass 2 with TE adapters disabled == the raw SAMTok TE, bitwise; enabled
   adapters do change the encoding.
2. A zero-initialized region_embed binding == no binding, bitwise.
3. An all-ones latent-blend mask == no blending, bitwise.
4. Attention-bias binding: cached decoding (additive mask) agrees with the
   uncached FlexAttention score_mod path as closely as unbiased cached and
   uncached runs agree with each other.
5. beta = 0 bias == no binding (up to kernel differences).

Writes a JSON report; exits nonzero on any failure.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from samtok_edit21.data.protocol import box_of
from samtok_edit21.models.binding import BindingConfig, RegionBinding, attach_region_embed, binding_payload
from samtok_edit21.models.codec import SamtokCodec
from samtok_edit21.models.pipeline import (
    DEFAULT_QWEN, DEFAULT_SAMTOK, encode_edit, load_pipeline, resize_sources, te_adapters,
)
from samtok_edit21.training.objectives import load_adapter


def run(pipe, prompt, image, binding=None, codec=None, blend=None, kv=True, size=512, steps=4):
    prepared = resize_sources(pipe, [image], size, size)
    region = None
    if binding is not None:
        ids = encode_edit(pipe, prompt, prepared, return_ids=True)["prompt_input_ids"]
        payload = binding_payload(pipe.processor.tokenizer, ids, prompt, image,
                                  {"target": (size, size), "source": (prepared[0].height, prepared[0].width)}, codec)
        region = RegionBinding(binding, payload, getattr(pipe.dit, "region_embed", None)
                               if binding.mode == "region_embed" else None)
    pipe.scheduler.training = False
    return pipe(prompt, edit_image=[image], height=size, width=size, num_inference_steps=steps, seed=7,
                cfg_scale=1.0, use_kv_cache=kv, region_binding=region, **(blend or {}),
                progress_bar_cmd=lambda x: x)


def array(image):
    return np.asarray(image).astype(np.float64)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--te-adapter", required=True, help="A trained Stage 1 (localization) adapter")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    report, failures = {}, []
    image = Image.open(args.image).convert("RGBA")
    pipe = load_pipeline(DEFAULT_QWEN, DEFAULT_SAMTOK, device=args.device)
    codec = SamtokCodec(Path(DEFAULT_SAMTOK) / "sam2.1_hiera_large.pt",
                        Path(DEFAULT_SAMTOK) / "mask_tokenizer_256x2.pth", device=args.device)
    prompt = "Add a red ball in this region " + box_of((300, 300, 700, 700)) + "."
    prepared = resize_sources(pipe, [image], 512, 512)
    with torch.no_grad():
        raw = encode_edit(pipe, prompt, prepared)["prompt_embeds"]
        load_adapter(pipe.text_encoder, args.te_adapter)
        with te_adapters(pipe, False):
            disabled = encode_edit(pipe, prompt, prepared)["prompt_embeds"]
        enabled = encode_edit(pipe, prompt, prepared)["prompt_embeds"]
    report["te_disabled_equals_raw"] = bool(torch.equal(raw, disabled))
    report["te_enabled_max_abs_diff"] = float((enabled.float() - raw.float()).abs().max())
    if not report["te_disabled_equals_raw"] or report["te_enabled_max_abs_diff"] == 0:
        failures.append("pass-2 adapter toggle")

    with torch.no_grad(), te_adapters(pipe, False):
        base = run(pipe, prompt, image)
        attach_region_embed(pipe.dit, 64)
        embed = run(pipe, prompt, image, BindingConfig("region_embed"), codec)
        del pipe.dit.region_embed
        ones = torch.ones(1, 1, 32, 32)
        latents = torch.zeros(1, 64, 32, 32)
        blended = run(pipe, prompt, image, blend={"blend_mask": ones, "blend_latents": latents})
        report["region_embed_zero_init_equal"] = bool(np.array_equal(array(base), array(embed)))
        report["blend_all_ones_equal"] = bool(np.array_equal(array(base), array(blended)))
        if not report["region_embed_zero_init_equal"]:
            failures.append("region_embed zero init")
        if not report["blend_all_ones_equal"]:
            failures.append("blend no-op")
        reference = np.abs(array(base) - array(run(pipe, prompt, image, kv=False))).mean()
        for mode in ("bias_span", "bias_clause"):
            config = BindingConfig(mode, 2.0, 0.05)
            cached, uncached = run(pipe, prompt, image, config, codec), run(pipe, prompt, image, config, codec, kv=False)
            gap = float(np.abs(array(cached) - array(uncached)).mean())
            effect = float(np.abs(array(cached) - array(base)).mean())
            report[f"{mode}_cached_vs_uncached_mean_abs"] = gap
            report[f"{mode}_effect_mean_abs"] = effect
            if gap > max(2 * reference, 0.5) or effect == 0:
                failures.append(f"{mode} cached/uncached agreement")
        zero = run(pipe, prompt, image, BindingConfig("bias_clause", 0.0, 0.05), codec)
        report["unbiased_cached_vs_uncached_mean_abs"] = float(reference)
        report["beta0_vs_none_mean_abs"] = float(np.abs(array(zero) - array(base)).mean())
        if report["beta0_vs_none_mean_abs"] > max(2 * reference, 0.5):
            failures.append("beta 0 no-op")
        rope = run(pipe, prompt, image, BindingConfig("region_rope"), codec)
        report["region_rope_effect_mean_abs"] = float(np.abs(array(rope) - array(base)).mean())
    report["failures"] = failures
    report["passed"] = not failures
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
