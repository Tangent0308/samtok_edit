"""Code-vector side channel for mask tokens (v3).

A region is written in the prompt as ``<|mt_start|><|mt_XXXX|><|mt_YYYY|><|mt_end|>``.  The text encoder sees the two
code tokens only as vocabulary entries, while the SAMTok codec owns what they stand for: a two-level residual codebook
whose two vectors sum to the quantised mask embedding.  The injector adds a learned projection of those vectors to the
text-encoder output at the span's own positions:

    position of the level-0 code : W0 e0 + b0
    position of the level-1 code : W1 e1 + b1
    position of ``<|mt_end|>``   : W2 (e0 + e1) + b2      (the quantised mask embedding)

so the DiT receives the geometry behind the codes instead of having to memorise 512 opaque ids.  Every projection is
zero-initialised: an untrained injector leaves the model exactly unchanged.  The inference input is still only the
prompt; nothing is decoded to pixels.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn

CODES_PER_LEVEL = 256
LEVELS = 2
WEIGHTS = "code_injector.safetensors"
CONFIG = "code_injector.json"
LEGACY_WEIGHTS = "ea2_extra.pt"  # E-A2 runs stored injector and router state in one torch file


class CodeInjector(nn.Module):
    """Maps mask-token ids to an additive correction of the text-encoder output."""

    def __init__(self, codebook, code_token_ids, end_token_id, hidden_size=4096):
        super().__init__()
        codes = CODES_PER_LEVEL * LEVELS
        codebook = torch.as_tensor(codebook).detach().float()
        ids = [int(i) for i in code_token_ids]
        end_token_id = int(end_token_id)
        if codebook.ndim != 2 or codebook.shape[0] != codes:
            raise ValueError(f"Expected a [{codes}, D] codebook (level 0 rows first)")
        if len(ids) != codes or len({*ids, end_token_id}) != codes + 1 or min(ids + [end_token_id]) < 0:
            raise ValueError("Mask code tokens and <|mt_end|> must be distinct vocabulary ids")
        lookup = torch.full((max(ids + [end_token_id]) + 1,), -1, dtype=torch.long)
        lookup[torch.tensor(ids)] = torch.arange(codes)
        self.register_buffer("codebook", codebook, persistent=False)
        self.register_buffer("lookup", lookup, persistent=False)
        self.end_token_id = end_token_id
        self.hidden_size = hidden_size
        self.proj = nn.ModuleList(nn.Linear(codebook.shape[1], hidden_size) for _ in range(LEVELS + 1))
        for layer in self.proj:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.enabled = True

    @classmethod
    def from_codec(cls, codec, tokenizer, hidden_size=4096):
        """Read the released codebooks; token ``<|mt_i|>`` is row ``i`` (level 0 first, then level 1)."""
        books = codec.vq.quantizer.codebooks
        codebook = torch.cat([books[level].weight[:CODES_PER_LEVEL] for level in range(LEVELS)], 0)
        ids = [tokenizer.convert_tokens_to_ids(f"<|mt_{i:04d}|>") for i in range(CODES_PER_LEVEL * LEVELS)]
        return cls(codebook, ids, tokenizer.convert_tokens_to_ids("<|mt_end|>"), hidden_size)

    def code_index(self, input_ids):
        """Codebook row of every token, ``-1`` where the token is not a mask code."""
        ids = input_ids.to(self.lookup.device)
        known = (ids >= 0) & (ids < self.lookup.numel())
        return torch.where(known, self.lookup[ids.clamp(0, self.lookup.numel() - 1)], torch.full_like(ids, -1))

    def forward(self, input_ids):
        """``[B, L]`` token ids -> ``[B, L, hidden]`` correction (zero away from mask spans)."""
        index = self.code_index(input_ids)
        length = index.shape[1]
        level0 = (index >= 0) & (index < CODES_PER_LEVEL)
        level1 = index >= CODES_PER_LEVEL
        book = self.codebook

        def rows(position, level):
            low = level * CODES_PER_LEVEL
            return book[position.clamp(low, low + CODES_PER_LEVEL - 1)]

        delta = self.proj[0](rows(index, 0)) * level0[..., None] + self.proj[1](rows(index, 1)) * level1[..., None]
        pad = index.new_full((index.shape[0], 1), -1)
        previous = torch.cat([pad, index], 1)[:, :length]
        before = torch.cat([pad, pad, index], 1)[:, :length]
        closes = ((input_ids.to(index.device) == self.end_token_id)
                  & (before >= 0) & (before < CODES_PER_LEVEL) & (previous >= CODES_PER_LEVEL))
        return delta + self.proj[2](rows(before, 0) + rows(previous, 1)) * closes[..., None]

    def fingerprint(self):
        """Identity of the codebook and token mapping the projections were trained against."""
        digest = hashlib.sha256(self.codebook.detach().float().cpu().contiguous().numpy().tobytes())
        digest.update(self.lookup.detach().cpu().contiguous().numpy().tobytes())
        digest.update(str(self.end_token_id).encode())
        return digest.hexdigest()


def attach_code_injector(pipe, injector):
    """Make ``pipe.text_encoder`` add the injector's correction to its output."""
    encoder = pipe.text_encoder
    if not hasattr(encoder, "code_injector"):
        raise TypeError("The text encoder does not support a code injector (expected SamtokTextEncoder)")
    encoder.code_injector = injector.to(pipe.device)
    return encoder.code_injector


def detach_code_injector(pipe):
    pipe.text_encoder.code_injector = None


def has_code_injector(directory):
    directory = Path(directory)
    return (directory / WEIGHTS).is_file() or (directory / LEGACY_WEIGHTS).is_file()


def save_code_injector(injector, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().float().cpu().contiguous() for k, v in injector.state_dict().items()}
    if not state or not all(torch.isfinite(v).all() for v in state.values()):
        raise RuntimeError("Invalid code injector parameters")
    tmp = directory / (WEIGHTS + ".tmp")
    save_file(state, str(tmp))
    tmp.replace(directory / WEIGHTS)
    (directory / CONFIG).write_text(json.dumps({
        "schema_version": 1, "hidden_size": injector.hidden_size, "code_dim": int(injector.codebook.shape[1]),
        "codes_per_level": CODES_PER_LEVEL, "levels": LEVELS, "codebook_sha256": injector.fingerprint(),
    }, indent=2) + "\n")


def load_code_injector(injector, directory):
    """Load projections saved by ``save_code_injector`` (or by an E-A2 run)."""
    directory = Path(directory)
    if (directory / WEIGHTS).is_file():
        config = json.loads((directory / CONFIG).read_text())
        if config.get("codebook_sha256") != injector.fingerprint():
            raise ValueError("Code injector was trained against a different SAMTok codebook or tokenizer")
        state = load_file(str(directory / WEIGHTS))
    elif (directory / LEGACY_WEIGHTS).is_file():
        legacy = torch.load(directory / LEGACY_WEIGHTS, map_location="cpu")
        state = {"proj." + k[len("inj.p."):]: v for k, v in legacy.items() if k.startswith("inj.p.")}
    else:
        raise FileNotFoundError(f"No code injector weights in {directory}")
    if set(state) != set(injector.state_dict()):
        raise ValueError("Code injector weights do not match the module")
    if not all(torch.isfinite(v).all() for v in state.values()):
        raise ValueError("Nonfinite code injector weights")
    injector.load_state_dict(state, strict=True)
    return injector


@contextmanager
def trained_weights(pipe, enabled):
    """Temporarily switch the DiT LoRA and the code injector on or off together.

    ``enabled=False`` is the stock model with the same text encoder, which is the base of every paired comparison.
    PEFT toggling also flips ``requires_grad``; the previous flags and states are restored on exit.
    """
    from peft.tuners.tuners_utils import BaseTunerLayer

    layers = [m for m in pipe.dit.modules() if isinstance(m, BaseTunerLayer)]
    states = [(m.disable_adapters, [(p, p.requires_grad) for p in m.parameters()]) for m in layers]
    injector = getattr(pipe.text_encoder, "code_injector", None)
    injector_state = None if injector is None else injector.enabled
    for layer in layers:
        layer.enable_adapters(enabled)
    if injector is not None:
        injector.enabled = enabled
    try:
        yield
    finally:
        for layer, (disabled, grads) in zip(layers, states):
            layer.enable_adapters(not disabled)
            for parameter, flag in grads:
                parameter.requires_grad_(flag)
        if injector is not None:
            injector.enabled = injector_state
