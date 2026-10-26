"""Shared fixtures.  Tests exercise the package from the checkout (PYTHONPATH=src:third_party/diffsynth:third_party);
install requirements.txt first."""
from types import SimpleNamespace

import pytest
import torch

import diffsynth.models.qwen_image_21_dit as qwen_dit


@pytest.fixture()
def tiny_dit(monkeypatch):
    """Real DiT class, four small blocks; prefix = 3 text + one 4x4 source image + 7 instruction tokens."""
    monkeypatch.setattr(qwen_dit, "FLEX_ATTN_AVAILABLE", False)        # the CPU path; hooks do not depend on it
    torch.manual_seed(0)
    model = qwen_dit.QwenImage21DiT(patch_size=1, in_channels=8, out_channels=8, num_layers=4, attention_head_dim=16,
                                    num_attention_heads=2, context_in_dim=12, axes_dims_rope=(4, 6, 6)).float()
    grid, text, pads, instruction = (4, 4), 3, 4, 7
    n_target = grid[0] * grid[1]
    image_mask = torch.tensor([[False] * text + [True] * pads + [False] * instruction + [True] * (n_target // 4)])
    encoder_states = torch.randn(1, text + pads + instruction, 12)
    source_tokens = torch.randn(1, pads * 4, 8)
    first_instruction = text + pads * 4
    span = [first_instruction + 2 + offset for offset in range(4)]

    def run(target_tokens, checkpointing=False):
        hidden = torch.cat([source_tokens, target_tokens], 1)
        out = model(hidden, encoder_states, torch.tensor([0.4]), [[(1, 4, 4), (1, *grid)]], image_mask,
                    use_gradient_checkpointing=checkpointing)
        return out[:, -n_target:]

    return SimpleNamespace(model=model, run=run, span=span, n_target=n_target, grid=grid)
