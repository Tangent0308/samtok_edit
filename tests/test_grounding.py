"""Token grounding components: the highlight task, the code injector, the routing loss and shared helpers.

Everything here runs on CPU.  The routing tests use a tiny instance of the real Qwen-Image-2.1 DiT class, so they
exercise the actual block code (norm, modulation, LoRA-able projections, RoPE, gradient checkpointing).
"""
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

import diffsynth.models.qwen_image_21_dit as qwen_dit
from samtok_edit21.data import highlight
from samtok_edit21.data.protocol import span_of
from samtok_edit21.evaluation import generation
from samtok_edit21.models.code_injector import (
    CodeInjector, has_code_injector, load_code_injector, save_code_injector,
)
from samtok_edit21.models.pipeline import DEFAULT_QWEN, DEFAULT_SAMTOK, SamtokTextEncoder
from samtok_edit21.training import round2
from samtok_edit21.training.routing import AttentionRouter, route_targets, span_attention_logits

A, B = span_of([12, 300]), span_of([40, 450])


# ------------------------------------------------------------------------------------------ highlight task

def test_overlay_is_read_back_exactly():
    rng = np.random.RandomState(0)
    rgb = rng.randint(40, 200, (96, 128, 3)).astype(np.float32)
    mask = np.zeros((96, 128), bool)
    mask[20:70, 30:90] = True
    source = Image.fromarray(rgb.astype(np.uint8))
    for name in highlight.READBACK_COLORS:
        painted = highlight.render_target(source, mask, {"kind": "hl", "color": name})
        recovered = highlight.extract_mask(source, painted, highlight.PALETTE[name])
        assert highlight.iou(recovered, mask) > 0.95
        assert not highlight.extract_mask(source, source, highlight.PALETTE[name]).any()
    with pytest.raises(ValueError):
        highlight.render_target(source, mask, {"kind": "real"})


def test_procedural_edits_touch_only_the_region():
    rng = np.random.RandomState(1)
    rgb = rng.randint(0, 256, (64, 64, 3)).astype(np.float32)
    weight = np.zeros((64, 64), np.float32)
    weight[:32] = 1.0
    for operation in highlight.PROCEDURAL_TEMPLATES:
        edited = highlight.procedural_edit(rgb, weight, operation)
        assert np.array_equal(edited[32:], rgb[32:]) and not np.array_equal(edited[:32], rgb[:32])
    with pytest.raises(ValueError):
        highlight.procedural_edit(rgb, weight, "sharpen")


def test_prompts_and_phrases():
    assert highlight.highlight_prompt(A, None, "orange") == (
        f"Apply an orange semi-transparent mask to this region {A} and keep everything else unchanged.")
    assert highlight.highlight_prompt(A, "the left cup", "red").startswith(f"Apply a red semi-transparent mask to the left cup {A}")
    assert highlight.procedural_prompt("blur", A, "the sky") == f"Blur the sky {A} and keep everything else unchanged."
    assert highlight.normalize_phrase("The Dog.") == "the Dog"
    assert highlight.normalize_phrase("The Dog.", legacy=True) == "The Dog"
    assert highlight.normalize_phrase("Leftmost bird,") == "the leftmost bird"
    assert highlight.normalize_phrase("USA flag") == "the USA flag"
    assert highlight.normalize_phrase("  ") is None and highlight.normalize_phrase("x" * 161) is None
    assert highlight.pick_color(Image.new("RGB", (8, 8), (250, 5, 5))) == "cyan"


# ------------------------------------------------------------------------------------------ code injector

class Tokens:
    """Mask-code ids scattered through a small vocabulary."""

    start, end = 4989, 4990

    def convert_tokens_to_ids(self, token):
        return {"<|mt_start|>": self.start, "<|mt_end|>": self.end}.get(token) or 5000 + 3 * int(token[5:9])


def fake_codec(seed=0):
    generator = torch.Generator().manual_seed(seed)
    books = [SimpleNamespace(weight=torch.randn(257, 32, generator=generator)) for _ in range(2)]
    return SimpleNamespace(vq=SimpleNamespace(quantizer=SimpleNamespace(codebooks=books)))


def span_ids(tokens, code0, code1):
    return [tokens.start, tokens.convert_tokens_to_ids(f"<|mt_{code0:04d}|>"),
            tokens.convert_tokens_to_ids(f"<|mt_{code1:04d}|>"), tokens.end]


def test_code_injector_positions_and_zero_init():
    tokens, codec = Tokens(), fake_codec()
    injector = CodeInjector.from_codec(codec, tokens, hidden_size=16)
    ids = torch.tensor([[7, 8, *span_ids(tokens, 3, 300), 9, -1], [tokens.end, 5000 + 3 * 300, 1, 2, 3, 4, 5, 6]])
    assert torch.equal(injector(ids), torch.zeros(2, 8, 16))          # untrained: the model is unchanged
    with torch.no_grad():
        for index, layer in enumerate(injector.proj):
            layer.weight.fill_(0.0)
            layer.bias.fill_(float(index + 1))                         # marks which projection fired
    delta = injector(ids)
    assert delta[0, :, 0].tolist() == [0, 0, 0, 1, 2, 3, 0, 0]         # start: none, code0: W0, code1: W1, end: W2
    assert delta[1, :, 0].tolist() == [0, 2, 0, 0, 0, 0, 0, 0]         # a stray end is ignored, a stray code is not
    with torch.no_grad():
        injector.proj[2].weight.normal_(generator=torch.Generator().manual_seed(0))
        injector.proj[2].bias.zero_()
    books = codec.vq.quantizer.codebooks
    expected = injector.proj[2](books[0].weight[3] + books[1].weight[300 - 256])
    # <|mt_end|> carries the quantised embedding (float32: the batched and the single-row linear round differently)
    assert torch.allclose(injector(ids)[0, 5], expected, atol=1e-5)
    with pytest.raises(ValueError):
        CodeInjector(torch.zeros(10, 4), range(512), 600)


def test_code_injector_save_load(tmp_path):
    tokens = Tokens()
    injector = CodeInjector.from_codec(fake_codec(), tokens, hidden_size=16)
    with torch.no_grad():
        for layer in injector.proj:
            layer.weight.normal_()
    ids = torch.tensor([span_ids(tokens, 9, 400)])
    save_code_injector(injector, tmp_path / "new")
    restored = load_code_injector(CodeInjector.from_codec(fake_codec(), tokens, hidden_size=16), tmp_path / "new")
    assert torch.equal(restored(ids), injector(ids)) and has_code_injector(tmp_path / "new")
    with pytest.raises(ValueError, match="different SAMTok codebook"):
        load_code_injector(CodeInjector.from_codec(fake_codec(seed=1), tokens, hidden_size=16), tmp_path / "new")
    (tmp_path / "legacy").mkdir()                                       # E-A2 runs: one torch file, keys inj.p.N.*
    torch.save({**{f"inj.p.{k[5:]}": v for k, v in injector.state_dict().items()}, "router.affine": torch.zeros(6, 2)},
               tmp_path / "legacy" / "ea2_extra.pt")
    legacy = load_code_injector(CodeInjector.from_codec(fake_codec(), tokens, hidden_size=16), tmp_path / "legacy")
    assert torch.equal(legacy(ids), injector(ids))
    assert not has_code_injector(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_code_injector(injector, tmp_path)


def test_text_encoder_adds_the_injection_with_a_frozen_backbone():
    class Encoder(SamtokTextEncoder):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.code_injector = None
            self.backbone = torch.nn.Linear(1, 16)

        def encode(self, **inputs):
            return self.backbone(inputs["input_ids"].float()[..., None]), None

    tokens, encoder = Tokens(), Encoder()
    ids = torch.tensor([[1, *span_ids(tokens, 5, 260), 2]])
    plain = encoder(input_ids=ids)
    injector = CodeInjector.from_codec(fake_codec(), tokens, hidden_size=16)
    encoder.code_injector = injector
    assert torch.equal(encoder(input_ids=ids), plain)                   # zero-initialised
    with torch.no_grad():
        injector.proj[0].bias.fill_(1.0)
    out = encoder(input_ids=ids)
    assert torch.equal(out[0, 2], plain[0, 2] + 1) and torch.equal(out[0, 0], plain[0, 0])
    out.sum().backward()
    assert injector.proj[0].bias.grad is not None and encoder.backbone.weight.grad is None
    injector.enabled = False
    assert torch.equal(encoder(input_ids=ids), plain)


# ------------------------------------------------------------------------------------------ routing

def test_route_targets_are_balanced_and_weight_distractors():
    region = torch.zeros(4, 6)
    region[1:3, 1:3] = 1.0
    distractors = torch.zeros(4, 6)
    distractors[0, 5] = 1.0
    labels, weights = route_targets(region, distractors)
    assert labels.sum() == 4 and weights[labels == 1].sum() == pytest.approx(0.5)
    assert weights[labels == 0].sum() == pytest.approx(0.5)
    negatives = weights.view(4, 6)
    assert negatives[0, 5] == pytest.approx(3 * negatives[0, 0])        # another instance counts three times
    labels, weights = route_targets(torch.zeros(4, 6), distractors)
    assert labels.sum() == 0 and weights.sum() == pytest.approx(0.5)   # no positive token: negatives only


def test_routing_logit_is_the_blocks_own_attention_logit(tiny_dit, monkeypatch):
    calls, seen = [], {}

    def spy(query, key, value, **kwargs):
        calls.append((query.detach(), key.detach()))
        return original(query, key, value, **kwargs)

    original = qwen_dit._attention
    monkeypatch.setattr(qwen_dit, "_attention", spy)
    block = tiny_dit.model.transformer_blocks[2]

    def before(module, args, kwargs):
        calls.clear()
        seen["logits"] = span_attention_logits(module, kwargs["hidden_states"], kwargs["modulation"],
                                               kwargs["target_token_mask"], kwargs["rotary_emb"],
                                               tiny_dit.n_target, tiny_dit.span)

    def after(module, args, kwargs, output):
        seen["query"], seen["key"] = calls[-1]                          # last call: target queries against all keys

    handles = [block.register_forward_pre_hook(before, with_kwargs=True), block.register_forward_hook(after, with_kwargs=True)]
    with torch.no_grad():
        tiny_dit.run(torch.randn(1, tiny_dit.n_target, 8))
    for handle in handles:
        handle.remove()
    query, key = seen["query"], seen["key"][:, tiny_dit.span]
    assert query.shape[1] == tiny_dit.n_target
    expected = torch.einsum("bnhd,bshd->bn", query, key)[0] / (math.sqrt(query.shape[-1]) * query.shape[2] * key.shape[1])
    assert torch.allclose(seen["logits"], expected, rtol=1e-5, atol=1e-6)


def test_router_gradients_survive_checkpointing_and_disarm_is_inert(tiny_dit):
    model = tiny_dit.model
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    region = torch.zeros(tiny_dit.grid)
    region[1:3, 1:4] = 1.0
    labels, weights = route_targets(region, torch.zeros(tiny_dit.grid))
    target = torch.randn(1, tiny_dit.n_target, 8)
    with torch.no_grad():
        plain = tiny_dit.run(target)
    router = AttentionRouter(model, layers=(1, 3))
    with torch.no_grad():
        assert torch.equal(tiny_dit.run(target), plain) and router.collect() is None     # not armed: nothing happens
    gradients = {}
    for checkpointing in (True, False):
        model.zero_grad(set_to_none=True)
        router.affine.grad = None
        router.arm(tiny_dit.n_target, tiny_dit.span, labels, weights)
        with router.recording():
            prediction = tiny_dit.run(target, checkpointing)
        term = router.collect()
        assert torch.equal(prediction, plain)                         # the hook reads, it never changes the forward
        (prediction.pow(2).mean() + term).backward()
        router.disarm()
        gradients[checkpointing] = (model.transformer_blocks[1].attn.to_q.weight.grad.clone(), router.affine.grad.clone())
    assert gradients[True][0].abs().sum() > 0 and gradients[True][1].abs().sum() > 0
    assert torch.allclose(gradients[True][0], gradients[False][0], rtol=1e-4, atol=1e-7)
    assert torch.allclose(gradients[True][1], gradients[False][1], rtol=1e-4, atol=1e-7)
    router.remove()
    with pytest.raises(ValueError):
        AttentionRouter(model, layers=(1, 9))


# ------------------------------------------------------------------------------------------ shared helpers

def test_grid_coverage():
    pixels = np.zeros((32, 48), np.float32)
    pixels[:16, :16] = 1.0
    coverage = round2.grid_coverage(pixels, (2, 3))
    assert coverage.tolist() == [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    with pytest.raises(ValueError):
        round2.grid_coverage(pixels, (3, 3))


@pytest.fixture(scope="module")
def processor():
    if not (Path(DEFAULT_QWEN) / "processor").is_dir() or not Path(DEFAULT_SAMTOK).is_dir():
        pytest.skip("Qwen-Image-2.1 processor / SAMTok tokenizer not available")
    from samtok_edit21.models.pipeline import build_processor
    return build_processor(DEFAULT_QWEN, DEFAULT_SAMTOK)


def test_span_positions_point_at_the_mask_tokens_in_the_joint_sequence(processor):
    from diffsynth.pipelines.qwen_image_21 import QwenImage21Unit_PromptEmbedder

    pipe = SimpleNamespace(processor=processor, device="cpu", torch_dtype=torch.float32,
                           load_models_to_device=lambda names: None,
                           text_encoder=lambda input_ids, **kw: torch.zeros(*input_ids.shape, 4096))
    prompt = highlight.highlight_prompt(A, "the left cup", "red")
    conditioning = QwenImage21Unit_PromptEmbedder().process(pipe, prompt, [Image.new("RGB", (64, 64))], return_token_ids=True)
    [positions] = round2.span_positions_all(processor.tokenizer, prompt, conditioning, 1)
    ids = conditioning["prompt_input_ids"][0]
    repeats = torch.where(conditioning["edit_image_pad_mask"][0], 4, 1)
    joint_ids = torch.repeat_interleave(ids, repeats)                   # what each joint position was expanded from
    assert processor.tokenizer.decode(joint_ids[positions], skip_special_tokens=False) == A
    assert positions == list(range(positions[0], positions[0] + 4))
    with pytest.raises(ValueError):              # two adjacent spans are one unit, not two four-token spans
        both = highlight.highlight_prompt(A + B, None, "red")
        round2.span_positions_all(processor.tokenizer, both, QwenImage21Unit_PromptEmbedder().process(
            pipe, both, [Image.new("RGB", (64, 64))], return_token_ids=True), 2)


# ------------------------------------------------------------------------------------------ generation

def test_canvas():
    assert generation.canvas(Image.new("RGB", (2000, 1000))) == (736, 1440)
    assert generation.canvas(Image.new("RGB", (10, 10))) == (1024, 1024)
