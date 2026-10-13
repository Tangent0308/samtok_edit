"""v3 token grounding: task definitions, code injector, routing loss, objective, data builder and report.

Everything here runs on CPU.  The routing tests use a tiny instance of the real Qwen-Image-2.1 DiT class, so they
exercise the actual block code (norm, modulation, LoRA-able projections, RoPE, gradient checkpointing).
"""
import json
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
from samtok_edit21.evaluation import grounding as evaluation
from samtok_edit21.evaluation.grounding_report import build_report
from samtok_edit21.models.code_injector import (
    CodeInjector, has_code_injector, load_code_injector, save_code_injector,
)
from samtok_edit21.models.pipeline import DEFAULT_QWEN, DEFAULT_SAMTOK, SamtokTextEncoder
from samtok_edit21.preparation import grounding_data
from samtok_edit21.training import grounding
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
        injector.proj[2].weight.normal_()
        injector.proj[2].bias.zero_()
    books = codec.vq.quantizer.codebooks
    expected = injector.proj[2](books[0].weight[3] + books[1].weight[300 - 256])
    assert torch.allclose(injector(ids)[0, 5], expected, atol=1e-6)    # <|mt_end|> carries the quantised embedding
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


def test_warm_start_restores_the_routing_calibration(tiny_dit, tmp_path):
    from samtok_edit21.training.grounding_engine import restore_route_affine

    router = AttentionRouter(tiny_dit.model, layers=(1, 3))
    recipe = {"grounding": {"route_layers": [1, 3], "route_affine": [[0.9, -0.2], [1.1, 0.3]]}}
    (tmp_path / "adapter.json").write_text(json.dumps(recipe))
    assert restore_route_affine(router, tmp_path)
    assert router.affine.flatten().tolist() == pytest.approx([0.9, -0.2, 1.1, 0.3])
    other = AttentionRouter(tiny_dit.model, layers=(0, 2))            # different layers: keep the default
    assert not restore_route_affine(other, tmp_path) and other.affine.tolist() == [[1.0, 0.0], [1.0, 0.0]]
    (tmp_path / "adapter.json").write_text(json.dumps({"stage": "stage2"}))
    assert not restore_route_affine(router, tmp_path)
    router.remove()
    other.remove()


# ------------------------------------------------------------------------------------------ objective

def test_region_terms_and_grid_coverage():
    error = torch.arange(16.0).view(4, 4)
    region = torch.zeros(4, 4)
    region[0, 0] = 1.0
    distractors = torch.zeros(4, 4)
    distractors[3, 3] = 0.3
    mean, in_region, on_distractors = grounding.region_terms(error, region, distractors)
    assert mean == pytest.approx(7.5)
    assert in_region == pytest.approx((0 + 1 + 4 + 5) / 4)             # the token and its one-token border
    assert on_distractors == pytest.approx(15.0)
    _, empty_region, empty_distractors = grounding.region_terms(error, torch.zeros(4, 4), torch.zeros(4, 4))
    assert empty_region == 0 and empty_distractors == 0
    pixels = np.zeros((32, 48), np.float32)
    pixels[:16, :16] = 1.0
    coverage = grounding.grid_coverage(pixels, (2, 3))
    assert coverage.tolist() == [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    with pytest.raises(ValueError):
        grounding.grid_coverage(pixels, (3, 3))


def test_grounding_loss_combines_terms_and_is_repeatable(tiny_dit):
    model = tiny_dit.model

    class Scheduler:
        timesteps = torch.linspace(1000.0, 1.0, 1000)

        def add_noise(self, clean, noise, timestep):
            sigma = (timestep / 1000).view(-1, 1, 1, 1)
            return (1 - sigma) * clean + sigma * noise

        def training_target(self, clean, noise, timestep):
            return noise - clean

        def training_weight(self, timestep):
            return torch.full_like(timestep, 2.0)

    def model_fn(dit, latents, timestep, **kwargs):
        prediction = tiny_dit.run(latents.flatten(2).transpose(1, 2), kwargs["use_gradient_checkpointing"])
        return prediction.transpose(1, 2).reshape(latents.shape)

    pipe = SimpleNamespace(scheduler=Scheduler(), model_fn=model_fn, dit=model, device="cpu", torch_dtype=torch.float32)
    region = torch.zeros(tiny_dit.grid)
    region[1:3, 1:3] = 1.0
    distractors = torch.zeros(tiny_dit.grid)
    distractors[0, 3] = 1.0
    sample = grounding.GroundingSample({"input_latents": torch.randn(1, 8, *tiny_dit.grid)}, region, distractors, tiny_dit.span)
    noise = torch.randn(1, 8, *tiny_dit.grid)
    router = AttentionRouter(model, layers=(1, 3))
    flow, routing, metrics = grounding.grounding_loss(pipe, sample, router, timestep_index=300, noise=noise)
    router.disarm()
    assert flow.item() == pytest.approx(2.0 * (metrics["mse"] + metrics["reg"] + metrics["dis"]), rel=1e-5)
    assert routing.item() == pytest.approx(metrics["route"], rel=1e-6) and metrics["dis"] > 0
    again, _, _ = grounding.grounding_loss(pipe, sample, None, timestep_index=300, noise=noise)
    assert torch.equal(again, flow)                                     # the router adds a term, never changes FM
    half, _, m = grounding.grounding_loss(pipe, sample, None, lambda_region=0.0, lambda_distractor=0.0,
                                          timestep_index=300, noise=noise)
    assert half.item() == pytest.approx(2.0 * m["mse"], rel=1e-6)
    router.remove()


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
    positions = grounding.span_token_positions(processor.tokenizer, prompt, conditioning)
    ids = conditioning["prompt_input_ids"][0]
    repeats = torch.where(conditioning["edit_image_pad_mask"][0], 4, 1)
    joint_ids = torch.repeat_interleave(ids, repeats)                   # what each joint position was expanded from
    assert processor.tokenizer.decode(joint_ids[positions], skip_special_tokens=False) == A
    assert positions == list(range(positions[0], positions[0] + 4))
    with pytest.raises(ValueError):
        both = highlight.highlight_prompt(A + B, None, "red")
        grounding.span_token_positions(processor.tokenizer, both, QwenImage21Unit_PromptEmbedder().process(
            pipe, both, [Image.new("RGB", (64, 64))], return_token_ids=True))


# ------------------------------------------------------------------------------------------ data builder

def picture(path, seed, size=(48, 40)):
    rng = np.random.RandomState(seed)
    blocks = rng.randint(0, 256, (5, 6, 3)).astype(np.uint8)
    Image.fromarray(blocks).resize(size, Image.NEAREST).save(path)
    return str(path)


def test_training_data_builder_guards_against_content_leaks(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    import io

    import pyarrow as pa

    images = tmp_path / "images"
    images.mkdir()
    train_paths, heldout_paths, seed = [], [], 0
    while len(train_paths) < 6 or len(heldout_paths) < 1:              # the split is by path hash
        path = str(images / f"img{seed}.png")
        (heldout_paths if grounding_data.heldout(path) else train_paths).append(picture(path, seed))
        seed += 1
    train_paths = train_paths[:6]
    eval_source = heldout_paths[0]
    duplicate = train_paths[5]                                          # same picture as the eval source, another path
    Image.open(eval_source).save(duplicate)
    spans = [span_of([i, 256 + i]) for i in range(8)]
    rows, provenance, sources, decisions = [], [], [], []

    def add(path, span, pair_id, edit_type="remove"):
        rows.append({"sample_type": "edit_umt", "edit_type": edit_type, "edit_image": path, "image": path,
                     "prompt": f"Remove the object in this region {span}.", "instr_variant": "noref"})
        provenance.append({"file": "stage2", "line": len(rows) - 1, "id": pair_id})
        sources.append({"id": pair_id, "instances": [{"ref": f"The thing {len(rows)}."}]})
        decisions.append({"id": pair_id, "decision": "keep", "dataset": "refedit"})

    add(train_paths[0], spans[0], "p0")
    add(train_paths[0], spans[1], "p1")                                 # two instances on one image: distractors
    add(train_paths[1], spans[2], "p2")
    add(train_paths[2], spans[3], "p3")
    add(duplicate, spans[4], "p4")                                      # must be dropped by content
    add(eval_source, spans[5], "p5")                                    # held-out by path
    add(train_paths[3], spans[6], "p6", edit_type="add")                # add rows are not grounding units
    rows.append({"sample_type": "edit", "edit_type": "remove", "edit_image": train_paths[4], "image": train_paths[4],
                 "prompt": "Remove the cup."})
    stage2 = tmp_path / "stage2"
    stage2.mkdir()
    (stage2 / "stage2.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (stage2 / "provenance.jsonl").write_text("".join(json.dumps(row) + "\n" for row in provenance))
    (tmp_path / "sources.jsonl").write_text("".join(json.dumps(row) + "\n" for row in sources))
    (tmp_path / "decisions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in decisions))
    buffer = io.BytesIO()
    Image.open(picture(tmp_path / "g.png", 99)).convert("RGB").save(buffer, "JPEG")
    answer = json.dumps([{"label": "The man", "mask_2d": spans[6]}, {"label": "a dog", "mask_2d": spans[7]}])
    pq.write_table(pa.table({"answer": [answer, "nothing"], "images": [[{"bytes": buffer.getvalue()}]] * 2}), tmp_path / "gres.parquet")
    items = [{"set": "ho", "source": eval_source, "span": spans[5], "variant": "noref", "phrase": None},
             {"set": "pair", "pair": 0, "tok": "a", "source": eval_source, "span": spans[5], "other": spans[4], "variant": "noref"},
             {"set": "pair", "pair": 0, "tok": "b", "source": eval_source, "span": spans[4], "other": spans[5], "variant": "noref"}]
    (tmp_path / "eval.json").write_text(json.dumps(items))
    (tmp_path / "earlier.jsonl").write_text(json.dumps({"edit_image": duplicate}) + "\n")

    def build(name):
        out = tmp_path / name
        stats = grounding_data.main([
            "train", "--stage2-data", str(stage2), "--sources", str(tmp_path / "sources.jsonl"),
            "--decisions", str(tmp_path / "decisions.jsonl"), "--gres", str(tmp_path / "gres.parquet"),
            "--eval-set", str(tmp_path / "eval.json"), "--output", str(out), "--workers", "2",
            "--flag-overlap", f"earlier={tmp_path / 'earlier.jsonl'}"])
        return out, stats

    out, stats = build("data")
    epoch = [json.loads(line) for line in open(out / "epoch_rows.jsonl")]
    sources_used = {row["edit_image"] for row in epoch}
    assert duplicate not in sources_used and eval_source not in sources_used and train_paths[3] not in sources_used
    assert stats["dropped_sources_matching_eval"] == 1 and stats["train_sources_still_matching_eval_after_filter"] == 0
    assert stats["stage2_units"] == 4 and stats["multi_units"] == 2 and stats["gres_multi_units"] == 2
    pair = [row for row in epoch if row["edit_image"] == train_paths[0] and row["kind"] == "hl"]
    assert {row["span"] for row in pair} == {spans[0], spans[1]} and all(len(row["others"]) == 1 for row in pair)
    named = [row for row in pair if row["variant"] == "ref"]
    assert named and all("the thing" in row["prompt"] for row in named)
    gres = [row for row in epoch if row["edit_image"].startswith("gres_images/")]
    assert gres and all((out / row["edit_image"]).is_file() for row in gres)   # relative: the data directory can move
    assert grounding.resolve_image(gres[0]["edit_image"], out) == out / gres[0]["edit_image"]
    assert {row["kind"] for row in epoch} == {"hl", "proc", "real"}
    schedule = json.load(open(out / "schedule.json"))
    assert len(schedule) == 2 * len(epoch) and sorted(schedule[:len(epoch)]) == list(range(len(epoch)))
    flagged = json.load(open(out / "eval.json"))
    assert all(item["leaky_earlier"] for item in flagged) and stats["eval_flags"] == {"leaky_earlier": 3}
    assert json.load(open(out / "eval_quick_idx.json")) == []           # every item is flagged, none is "clean"
    again, _ = build("data_again")
    assert (again / "epoch_rows.jsonl").read_text() == (out / "epoch_rows.jsonl").read_text()


def test_quick_indices_skip_flagged_items():
    items = [{"set": "ho", "variant": "noref", "leaky_ea1": index % 2 == 0} for index in range(100)]
    items += [{"set": "dev", "variant": "ref"} for _ in range(40)]
    items += [{"set": "pair", "pair": index // 2, "variant": "noref", "leaky_ea1": index < 4} for index in range(40)]
    chosen = grounding_data.quick_indices(items)
    assert len([i for i in chosen if items[i]["set"] == "ho"]) == 40 and all(not items[i].get("leaky_ea1") for i in chosen)
    assert len([i for i in chosen if items[i]["set"] == "dev"]) == 32
    assert sorted({items[i]["pair"] for i in chosen if items[i]["set"] == "pair"}) == list(range(2, 18))


# ------------------------------------------------------------------------------------------ evaluation / report

def test_canvas_and_pair_follow():
    assert evaluation.canvas(Image.new("RGB", (2000, 1000))) == (736, 1440)
    assert evaluation.canvas(Image.new("RGB", (10, 10))) == (1024, 1024)
    records = [{"pair": 0, "valid": True, "iou_self": 0.9, "iou_other": 0.1}, {"pair": 0, "valid": True, "iou_self": 0.2, "iou_other": 0.5},
               {"pair": 1, "valid": True, "iou_self": 0.8, "iou_other": 0.1}, {"pair": 1, "valid": True, "iou_self": 0.7, "iou_other": 0.3},
               {"pair": 2, "valid": True, "iou_self": 0.8, "iou_other": 0.1}, {"pair": 2, "valid": False}]
    assert evaluation.pair_follow(records) == {0: False, 1: True}
    summary = evaluation.quick_summary([{"set": "ho", "variant": "noref", "iou_codec": 0.5},
                                        {"set": "dev", "variant": "noref", "iou_codec": 0.1, "iou_gt": 0.7},
                                        *[{"set": "pair", "variant": "noref", "iou_codec": 0.0, **r} for r in records]])
    assert summary["ho_noref"] == 0.5 and summary["dev_noref"] == 0.7 and summary["pair_follow"] == 0.5 and summary["pairs"] == 2
    jobs = evaluation.build_jobs([{"set": "ho"}, {"set": "pair", "pair": 0}, {"set": "pair", "pair": 5}], [{}, {}],
                                 has_adapter=True, remove_pairs=3, limit=0)
    assert [j for j in jobs if j[0] == "remove"] == [("remove", 1, "trained"), ("remove", 1, "base")]
    assert len([j for j in jobs if j[0] == "edit"]) == 4 and len([j for j in jobs if j[0] == "highlight"]) == 3


def test_report_pairs_items_and_applies_gates():
    items = [{"set": "ho", "variant": "noref", "leaky_ea1": index >= 6} for index in range(8)]
    items += [{"set": "dev", "variant": "noref"}, {"set": "dev", "variant": "ref"}]
    items += [{"set": "pair", "pair": 0, "tok": "a", "variant": "noref"}, {"set": "pair", "pair": 0, "tok": "b", "variant": "noref"}]

    def evaluation_of(heldout, dev, follows):
        records = [{"idx": index, "iou_codec": heldout, "area": 0.1} for index in range(8)]
        records += [{"idx": 8, "iou_codec": 0.0, "iou_gt": dev, "area": 0.1}, {"idx": 9, "iou_codec": 0.0, "iou_gt": dev, "area": 0.1}]
        own, other = (0.9, 0.1) if follows else (0.1, 0.9)
        records += [{"idx": 10 + k, "iou_codec": 0.5, "area": 0.1, "valid": True, "iou_self": own, "iou_other": other} for k in range(2)]
        return {"recs": records}

    trained = evaluation_of(0.7, 0.6, True)
    trained["recs"].pop(7)                                              # an item missing here is left out everywhere
    trained["remove_pairs"] = [{"model": model, "pair": 0, "tok": tok, "valid": True, "change_own": own, "change_other": 5.0, "change_far": 1.0}
                               for model, own in (("trained", 40.0), ("base", 2.0)) for tok in "ab"]
    trained["edits"] = [{"case_id": "c", "model": "trained", "diff_in": 50.0, "diff_out": 2.0}]
    lines, summary = build_report(items, trained, {"stock": evaluation_of(0.4, 0.65, False)}, name="run")
    assert summary["highlight"]["ho-noref"] == {"n": 7, "run": pytest.approx(0.7), "stock": pytest.approx(0.4)}
    assert summary["highlight"]["ho-noref (leak-free)"]["n"] == 6
    assert summary["highlight"]["pairs"]["run"] == 1.0
    assert summary["gate"] == {"held-out noref": True, "dev noref": True, "pair-follow": True, "dev ref vs base": False}
    assert summary["remove"]["trained"]["pair_follow"] == 1.0 and summary["remove"]["base"]["pair_follow"] == 0.0
    assert summary["edits"]["trained"] == {"cases": 1, "inside": 50.0, "outside": 2.0}
    assert any("run-stock +0.300" in line for line in lines)
