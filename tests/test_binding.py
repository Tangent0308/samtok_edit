from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from samtok_edit21.data.protocol import box_of, span_of
from samtok_edit21.models.binding import (
    BindingConfig, RegionBinding, RegionEmbed, binding_payload, blend_region, box_pixels, coverage_grid,
    token_layout, unit_masks,
)
from samtok_edit21.models.pipeline import DEFAULT_QWEN, DEFAULT_SAMTOK

A, B = span_of([3, 300]), span_of([40, 450])
X, Y = box_of((0, 0, 500, 500)), box_of((500, 500, 1000, 1000))


class FakeCodec:
    """decode_strict returns one fixed mask per span, in prompt order."""

    def __init__(self, masks):
        self.masks = masks

    def decode_strict(self, image, prompt):
        return self.masks[:prompt.count("<|mt_start|>")]


def test_box_raster_and_coverage():
    mask = box_pixels(box_of((250, 0, 750, 500)), 64, 32)
    assert mask.shape == (32, 64) and mask[:16, 16:48].all() and mask.sum() == 16 * 32
    grown = box_pixels(box_of((250, 0, 750, 500)), 64, 32, expand=0.1)
    assert grown.sum() > mask.sum() and grown[:16, 16:48].all()
    grid = coverage_grid(mask, 32, 64)  # 2x4 latent tokens, 1-token dilation
    assert grid.shape == (2, 4) and torch.equal(grid, torch.ones(2, 4))
    small = np.zeros((64, 64), bool)
    small[:16, :16] = True
    grid = coverage_grid(small, 64, 64)
    assert grid[0, 0] == 1 and grid[1, 1] == 1 and grid[3, 3] == 0


def test_unit_masks_union_groups_and_kinds():
    image = Image.new("RGB", (40, 20))
    left = np.zeros((20, 40), bool)
    left[:, :10] = True
    groups, masks = unit_masks("Remove the cats " + A + B + " and add a dog " + Y + ".", image,
                               FakeCodec([left, left[:, ::-1]]))
    assert groups == [A + B, Y] and masks[0].sum() == 2 * 200 and masks[1][10:, 20:].all()
    with pytest.raises(ValueError):
        unit_masks("Remove the cat " + A + ".", image, None)
    region = blend_region("Add a dog in this region " + X + ".", image)
    assert region[:11, :21].all() and not region[-1, -1]


@pytest.fixture(scope="module")
def processor():
    if not (Path(DEFAULT_QWEN) / "processor").is_dir() or not Path(DEFAULT_SAMTOK).is_dir():
        pytest.skip("Qwen-Image-2.1 processor / SAMTok tokenizer not available")
    from samtok_edit21.models.pipeline import build_processor
    return build_processor(DEFAULT_QWEN, DEFAULT_SAMTOK)


def official_ids(processor, prompt):
    """prompt_input_ids from the official embedder, with a stub text encoder."""
    from types import SimpleNamespace
    from diffsynth.pipelines.qwen_image_21 import QwenImage21Unit_PromptEmbedder

    pipe = SimpleNamespace(processor=processor, device="cpu", torch_dtype=torch.float32,
                           load_models_to_device=lambda names: None,
                           text_encoder=lambda input_ids, **kw: torch.zeros(*input_ids.shape, 4096))
    image = Image.new("RGB", (64, 64), (128, 64, 32))
    return QwenImage21Unit_PromptEmbedder().process(pipe, prompt, [image], return_token_ids=True)


def test_token_layout_matches_official_embedder(processor):
    tokenizer = processor.tokenizer
    for prompt, sizes in (("Remove the object in this region " + A + ".", [4]),
                          ("Add cups in this region " + X + Y + ".", [None]),
                          ("Remove the cats " + A + B + " and add a dog " + X + ".", [8, None])):
        result = official_ids(processor, prompt)
        ids = result["prompt_input_ids"]
        layout = token_layout(tokenizer, ids, prompt)
        flat = ids[0].tolist()
        assert layout["length"] == len(flat) == result["edit_image_pad_mask"].shape[1]
        assert flat[layout["instruction"][0] - 1] == tokenizer.convert_tokens_to_ids("<|vision_end|>")
        for (start, end), size in zip(layout["units"], sizes):
            assert size is None or end - start == size
            assert not result["edit_image_pad_mask"][0, start:end].any()
    broken = official_ids(processor, "Add a dog in this region " + X + ".")["prompt_input_ids"].clone()
    end = tokenizer.convert_tokens_to_ids("<|box_end|>")
    broken[broken == end] = tokenizer.convert_tokens_to_ids(".")
    with pytest.raises(ValueError):
        token_layout(tokenizer, broken, "Add a dog in this region " + X + ".")
    payload = binding_payload(tokenizer, official_ids(processor, "Add a dog in this region " + X + ".")["prompt_input_ids"],
                              "Add a dog in this region " + X + ".", Image.new("RGB", (64, 64)),
                              {"target": (64, 64), "source": (32, 32)})
    assert payload["target"].shape == (1, 4, 4) and payload["source"].shape == (1, 2, 2)
    assert payload["target"][0, 0, 0] == 1 and payload["target"][0, 3, 3] == 0


def layout():
    # TE: 10 positions, position 2 is the source image pad (4 latent tokens);
    # one target pseudo-pad expands to a 2x2 target block -> joint length 17.
    repeats = torch.tensor([1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 4])
    target_token_mask = torch.zeros(17, dtype=torch.bool)
    target_token_mask[13:] = True
    payload = {"units": [[5, 7]], "instruction": [3, 10], "empty": [False],
               "target": torch.tensor([[[1.0, 0.0], [0.5, 0.0]]])}
    return repeats, target_token_mask, payload


def bound(mode, payload=None, embed=None, beta=2.0, eps=0.05):
    repeats, target_token_mask, default = layout()
    from diffsynth.models.qwen_image_21_dit import QwenImage21Rope
    binding = RegionBinding(BindingConfig(mode, beta, eps), default if payload is None else payload, embed)
    return binding.bind(repeats=repeats, target_token_mask=target_token_mask, target_shape=(2, 2),
                        encoder_hidden_states=torch.randn(1, 10, 8),
                        pos_embed=QwenImage21Rope(theta=10000, axes_dim=[16, 56, 56]))


@pytest.mark.parametrize("mode,columns", [("bias_span", [8, 9]), ("bias_clause", list(range(6, 13)))])
@pytest.mark.parametrize("eps", [0.05, 0.0])
def test_score_mod_and_decode_mask_agree(mode, columns, eps):
    b = bound(mode, eps=eps)
    score_mod = b.score_mod(128)
    q, kv = torch.arange(17)[:, None], torch.arange(17)[None, :]
    dense = score_mod(torch.zeros(17, 17), 0, 0, q, kv)
    decode = b.decode_mask(None, torch.float32)[0, 0]
    assert torch.equal(dense[13:], decode)
    assert not dense[:13].any() and not dense[:, [c for c in range(17) if c not in columns]].any()
    maps = torch.tensor([1.0, 0.0, 0.5, 0.0])
    expected = 2.0 * torch.log(eps + (1 - eps) * maps).clamp_min(np.log(1e-6))
    assert torch.allclose(dense[13:, columns[0]], expected)
    valid = torch.ones(1, 17, dtype=torch.bool)
    valid[0, 4] = False
    assert torch.isneginf(b.decode_mask(valid, torch.float32)[0, 0, :, 4]).all()


def test_region_embed_is_zero_at_init_and_keeps_parameters_in_graph():
    embed = RegionEmbed(dim=8, rank=2)
    joint = torch.randn(1, 17, 8)
    out = bound("region_embed", embed=embed).apply_embedding(joint)
    assert torch.equal(out, joint)
    empty = bound("region_embed", payload={"units": [], "instruction": [3, 10], "empty": [], "target": None},
                  embed=embed)
    out = empty.apply_embedding(joint.requires_grad_())
    out.sum().backward()
    assert embed.up.weight.grad is not None and not embed.up.weight.grad.any()
    torch.nn.init.ones_(embed.up.weight)
    moved = bound("region_embed", embed=embed).apply_embedding(joint.detach())
    assert torch.equal(moved[:, :13], joint[:, :13]) and not torch.equal(moved[:, 13:], joint[:, 13:])
    assert torch.equal(moved[:, 14], joint[0, 14][None])  # zero coverage token unchanged


def test_region_rope_moves_only_region_tokens_to_the_centroid():
    b = bound("region_rope")
    rotary = torch.polar(torch.ones(17, 64), torch.zeros(17, 64))
    moved = b.apply_rope(rotary, (2, 2))
    changed = (moved != rotary).any(1).nonzero().flatten().tolist()
    assert changed == [8, 9]  # TE positions 5, 6 after the 4-token source block
    # Centroid row 1/3 -> 0, col 0 -> 0; centered index 0 - (2 - 1) = -1.
    assert torch.equal(moved[8, 8:36], b.pos_embed.freqs[1][-1]) and torch.equal(moved[8, 36:], b.pos_embed.freqs[2][-1])
    assert torch.equal(moved[8, :8], rotary[8, :8])


def test_empty_regions_are_unbound_and_clause_needs_one_unit():
    payload = {"units": [[5, 7]], "instruction": [3, 10], "empty": [True], "target": torch.zeros(1, 2, 2)}
    b = bound("bias_span", payload=payload)
    assert not b.biased and b.score_mod(128) is None
    two = {"units": [[4, 5], [6, 7]], "instruction": [3, 10], "empty": [False, False],
           "target": torch.ones(2, 2, 2)}
    with pytest.raises(ValueError, match="single-unit"):
        RegionBinding(BindingConfig("bias_clause"), two)
    with pytest.raises(ValueError):
        BindingConfig("bias_span", eps=1.0)
