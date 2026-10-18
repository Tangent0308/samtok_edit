"""Round 2 (docs/08 section 5): read-side routing, several spans per prompt, row building, loss and read-back.

CPU only.  The routing tests reuse the tiny instance of the real Qwen-Image-2.1 DiT class from test_grounding, so
they exercise the actual block code (modulation, projections, RoPE, block-causal prefix attention).
"""
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

import diffsynth.models.qwen_image_21_dit as qwen_dit
from samtok_edit21.evaluation import round2_eval as r2e
from samtok_edit21.preparation import round2_data as rd
from samtok_edit21.training import round2 as r2
from samtok_edit21.training.routing import AttentionRouter, span_read_logits

from test_grounding import tiny_dit  # noqa: F401  (pytest fixture)

SOURCE = list(range(3, 19))          # tiny_dit: 3 text tokens, then the 4x4 source image (16 joint positions)


def test_read_logit_is_the_blocks_own_span_to_source_attention(tiny_dit, monkeypatch):  # noqa: F811
    calls, seen = [], {}

    def spy(query, key, value, **kwargs):
        calls.append((query.detach(), key.detach()))
        return original(query, key, value, **kwargs)

    original = qwen_dit._attention
    monkeypatch.setattr(qwen_dit, "_attention", spy)
    block = tiny_dit.model.transformer_blocks[1]

    def before(module, args, kwargs):
        calls.clear()
        seen["logits"] = span_read_logits(module, kwargs["hidden_states"], kwargs["modulation"],
                                          kwargs["target_token_mask"], kwargs["rotary_emb"], tiny_dit.span, SOURCE)

    def after(module, args, kwargs, output):
        seen["calls"] = list(calls)

    handles = [block.register_forward_pre_hook(before, with_kwargs=True), block.register_forward_hook(after, with_kwargs=True)]
    with torch.no_grad():
        tiny_dit.run(torch.randn(1, tiny_dit.n_target, 8))
    for handle in handles:
        handle.remove()
    # the instruction segment (3 + 16 = 19 .. 26) attends causally to everything before it, the source included
    start = 19
    segment = [c for c in seen["calls"] if c[0].shape[1] == 7]
    assert len(segment) == 1
    query, key = segment[0]
    q = query[:, [p - start for p in tiny_dit.span]]
    k = key[:, SOURCE]
    expected = torch.einsum("bshd,bnhd->bn", q, k)[0] / (math.sqrt(q.shape[-1]) * q.shape[2] * q.shape[1])
    assert torch.allclose(seen["logits"], expected, rtol=1e-5, atol=1e-6)
    # independent of the noisy target: the prefix is block-causal
    with torch.no_grad():
        first = seen["logits"].clone()
        handles = [block.register_forward_pre_hook(before, with_kwargs=True)]
        tiny_dit.run(torch.randn(1, tiny_dit.n_target, 8) * 5)
        handles[0].remove()
    assert torch.allclose(seen["logits"], first)


def test_router_entries_write_and_read_with_checkpointing(tiny_dit):  # noqa: F811
    model = tiny_dit.model
    for p in model.parameters():
        p.requires_grad_(True)
    router = AttentionRouter(model, layers=(1, 3), read=True)
    region_a, region_b = torch.zeros(tiny_dit.grid), torch.zeros(tiny_dit.grid)
    region_a[0:2, 0:2], region_b[2:4, 2:4] = 1.0, 1.0
    sample = r2.Round2Sample(inputs={}, kind="hlm", regions=[region_a, region_b], distractors=torch.zeros(tiny_dit.grid),
                             spans=[tiny_dit.span, [p + 1 for p in tiny_dit.span]], route_regions=[region_a, region_b],
                             source_positions=SOURCE, source_regions=[region_a, region_b])
    entries = r2.route_entries(sample)
    assert len(entries) == 2
    # span 0: positives on A, B's tokens are the heavy negatives
    w = entries[0]["weights"].view(4, 4)
    assert entries[0]["labels"].view(4, 4)[0, 0] == 1 and entries[0]["labels"].view(4, 4)[3, 3] == 0
    assert w[3, 3] == pytest.approx(3 * w[0, 3])                         # B (heavy) vs plain background
    assert float(entries[0]["weights"][entries[0]["labels"] > 0].sum()) == pytest.approx(0.5)
    target = torch.randn(1, tiny_dit.n_target, 8)
    for checkpointing in (False, True):
        model.zero_grad()
        router.arm_entries(tiny_dit.n_target, entries)
        with router.recording():
            prediction = tiny_dit.run(target, checkpointing)
        write, read = router.collect(), router.collect_read()
        (prediction.pow(2).mean() + write + read).backward()
        router.disarm()
        grad = model.transformer_blocks[1].attn.to_q.weight.grad
        assert write is not None and read is not None and grad is not None and torch.isfinite(grad).all()
        assert router.read_affine.grad is not None
        router.read_affine.grad = None
        router.affine.grad = None
    router.remove()


def test_tiny_regions_get_their_best_token_and_labels_balance():
    coverage = torch.zeros(4, 4)
    coverage[2, 1] = 0.3
    labels = r2.token_labels(coverage)
    assert labels.sum() == 1 and labels.view(4, 4)[2, 1] == 1
    assert r2.token_labels(torch.zeros(4, 4)).sum() == 0
    weights = r2.balanced(labels, torch.zeros(16))
    assert float(weights[labels > 0].sum()) == pytest.approx(0.5) and float(weights[labels == 0].sum()) == pytest.approx(0.5)


def test_source_positions_follow_the_dit_expansion():
    pad = torch.tensor([[False, False, True, True, False, False]])     # 2 TE pads -> 8 DiT tokens
    positions = r2.source_token_positions({"edit_image_pad_mask": pad}, (2, 4))
    assert positions == [2, 3, 4, 5, 6, 7, 8, 9]
    with pytest.raises(ValueError):
        r2.source_token_positions({"edit_image_pad_mask": pad}, (3, 4))


def test_overlay_and_multi_readback_round_trip():
    rng = np.random.default_rng(0)
    source = Image.fromarray(rng.integers(60, 140, (64, 96, 3)).astype(np.uint8))
    a = np.zeros((64, 96), bool); a[5:25, 5:30] = True
    b = np.zeros((64, 96), bool); b[30:60, 50:90] = True
    out = r2.overlay_regions(source, [a, b], ["red", "cyan"])
    masks = r2e.extract_masks(source, out, ["red", "cyan"])
    assert np.mean(masks[0] == a) > 0.97 and np.mean(masks[1] == b) > 0.97
    assert not (masks[0] & b).any() and not (masks[1] & a).any()
    unchanged = r2e.extract_masks(source, source, ["red", "cyan"])
    assert not unchanged[0].any() and not unchanged[1].any()


def test_prompts_name_no_target():
    span = "<|mt_start|><|mt_0001|><|mt_0300|><|mt_end|>"
    single = rd.highlight_prompt_r2(span, "red")
    assert single == f"Apply a red semi-transparent mask to the selected region {span} and keep everything else unchanged."
    assert rd.highlight_prompt_r2(span, "orange", "chair").startswith("Apply an orange semi-transparent mask to the chair ")
    multi = rd.multi_prompt([span, span, span], ["red", "green", "blue"], 0)
    assert multi == (f"Apply semi-transparent masks: red to the selected region {span}, green to the selected region "
                     f"{span} and blue to the selected region {span}. Keep everything else unchanged.")
    for template in range(len(rd.HL_MULTI)):
        assert rd.multi_prompt([span, span], ["red", "green"], template).count(span) == 2


def test_relations_find_parts_hosts_and_same_category():
    from pycocotools import mask as mu

    def mask(box, shape=(50, 50)):
        m = np.zeros(shape, np.uint8)
        m[box[1]:box[3], box[0]:box[2]] = 1
        rle = mu.encode(np.asfortranarray(m))
        return {"rle": {"size": list(rle["size"]), "counts": rle["counts"].decode()}}

    masks = [dict(mask((0, 0, 20, 20)), category="chair"), dict(mask((2, 2, 8, 8)), category="seat"),
             dict(mask((30, 30, 48, 48)), category="chair"), dict(mask((0, 0, 20, 20)), category="chair")]
    rel = rd.relations(masks)
    assert rel[1]["hosts"] == [0, 3] and 1 in rel[0]["parts"]
    assert rel[0]["same"] == [2] and rel[0]["duplicate"] == [3]


def test_round2_loss_combines_terms(tiny_dit):  # noqa: F811
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
    a, b = torch.zeros(tiny_dit.grid), torch.zeros(tiny_dit.grid)
    a[0, 0], b[2:4, 2:4] = 1.0, 1.0
    distractors = torch.zeros(tiny_dit.grid)
    distractors[0, 3] = 1.0
    sample = r2.Round2Sample({"input_latents": torch.randn(1, 8, *tiny_dit.grid)}, "hlm", [a, b], distractors,
                             spans=[tiny_dit.span, [p + 1 for p in tiny_dit.span]], route_regions=[a, b],
                             source_positions=SOURCE, source_regions=[a, b])
    noise = torch.randn(1, 8, *tiny_dit.grid)
    router = AttentionRouter(model, layers=(1, 3), read=True)
    flow, write, read, metrics = r2.round2_loss(pipe, sample, router, timestep_index=300, noise=noise)
    router.disarm()
    assert flow.item() == pytest.approx(2.0 * (metrics["mse"] + metrics["reg"] + metrics["dis"]), rel=1e-5)
    assert write is not None and read is not None and metrics["dis"] > 0
    again, w2, r2_, _ = r2.round2_loss(pipe, sample, None, timestep_index=300, noise=noise)
    assert torch.equal(again, flow) and w2 is None and r2_ is None       # routing adds terms, never changes FM
    # the region term is the mean of per-region means (a small region counts as much as a large one)
    error = torch.arange(16.0).view(4, 4)
    per = [((error * torch.nn.functional.max_pool2d(r[None, None], 3, 1, 1)[0, 0]).sum()
            / torch.nn.functional.max_pool2d(r[None, None], 3, 1, 1)[0, 0].sum()) for r in (a, b)]
    assert r2.region_error(error, [a, b]).item() == pytest.approx(float(sum(per) / 2))
    router.remove()


def test_c1b_pairs_share_a_rank_and_an_update():
    import random
    units = [[{"id": f"p{k}a"}, {"id": f"p{k}b"}] for k in range(6)] + [[{"id": f"s{k}"}] for k in range(20)]
    rows = rd.place_in_slots(units, random.Random(0), updates=2, world=4, accumulation=4)
    assert sorted(r["id"] for r in rows) == sorted(x["id"] for u in units for x in u)
    for k in range(6):
        a, b = (rows.index(next(r for r in rows if r["id"] == f"p{k}{m}")) for m in "ab")
        (step_a, rank_a), (step_b, rank_b) = divmod(a, 4), divmod(b, 4)      # global step = update * 4 + step
        assert rank_a == rank_b and step_a // 4 == step_b // 4 and abs(step_a - step_b) == 1


def test_c1b_near_negatives_skip_parts_and_far_objects():
    from pycocotools import mask as mu

    def mask(box):
        m = np.zeros((100, 100), np.uint8)
        m[box[1]:box[3], box[0]:box[2]] = 1
        rle = mu.encode(np.asfortranarray(m))
        return {"size": list(rle["size"]), "counts": rle["counts"].decode()}

    boxes = [(10, 10, 40, 40), (15, 15, 25, 25), (42, 10, 60, 40), (90, 90, 99, 99), (0, 0, 60, 60)]
    masks = [{"rle": mask(b), "dec": mask(b), "category": None} for b in boxes]
    rel = rd.relations(masks)
    for m, r in zip(masks, rel):
        m["rel"] = r
    entry = {"masks": masks, "size": [100, 100]}
    # 1 is a part of 0, 2 a neighbour 2 px away, 3 far away, 4 the host of 0
    assert rd.near_negatives(entry, 0) == [2, 4]
    assert rd.near_negatives(entry, 0, exclude=[2]) == [4]


def test_c1b_learning_rate_warms_up_then_decays_to_the_floor():
    from samtok_edit21.training.round2_engine import learning_rate
    args = SimpleNamespace(lr=1e-4, lr_min=1e-5, warmup=10, updates=110, lr_schedule="cosine")
    assert learning_rate(args, 0) == pytest.approx(1e-5) and learning_rate(args, 9) == pytest.approx(1e-4)
    assert learning_rate(args, 10) == pytest.approx(1e-4) and learning_rate(args, 109) == pytest.approx(1e-5)
    assert learning_rate(args, 60) == pytest.approx(5.5e-5, rel=0.02)
    assert learning_rate(SimpleNamespace(**{**vars(args), "lr_schedule": "constant"}), 109) == pytest.approx(1e-4)


def test_c2_multi_edit_prompt_numbers_the_units_in_token_order():
    spans = [f"<|mt_start|><|mt_{k:04d}|><|mt_0300|><|mt_end|>" for k in range(3)]
    text = rd.multi_edit_prompt([f"Remove the selected region {spans[0]}.", f"Add a red hat to the selected region {spans[1]}.",
                                 f"Change the color of the selected region {spans[2]} to blue."])
    lines = text.split("\n")
    assert lines[0] == rd.MULTI_EDIT_INTRO and [line[:4] for line in lines[1:]] == ["U1: ", "U2: ", "U3: "]
    assert [text.index(s) for s in spans] == sorted(text.index(s) for s in spans)


def test_c2_composite_units_have_distinct_operations_and_separate_regions():
    import random
    grids = {}
    def grid(cells):
        g = np.zeros((8, 8), bool)
        for y, x in cells:
            g[y, x] = True
        return g
    records = [{"sample_id": "a", "operation": "add"}, {"sample_id": "b", "operation": "remove"},
               {"sample_id": "c", "operation": "remove"}, {"sample_id": "d", "operation": "replace"}]
    grids.update(a=grid([(0, 0), (0, 1), (1, 0), (1, 1)]), b=grid([(1, 1), (1, 2), (2, 1), (2, 2)]),   # b touches a
                 c=grid([(6, 6), (6, 7), (7, 6), (7, 7)]), d=grid([(0, 6), (0, 7), (1, 6), (1, 7)]))
    for seed in range(20):
        group = rd.composite_group(records, grids.__getitem__, random.Random(seed), 3)
        if group is None:
            continue
        ids = [r["sample_id"] for r in group]
        assert len({r["operation"] for r in group}) == 3 and not {"a", "b"} <= set(ids)
        assert not any((grids[x] & grids[y]).any() for i, x in enumerate(ids) for y in ids[i + 1:])
    assert rd.composite_group(records[:2], grids.__getitem__, random.Random(0), 3) is None


def test_c2_twins_and_v1_filter():
    import random
    group = [{"bbox": [0, 0, 10, 10]}, {"bbox": [12, 0, 20, 10]}, {"bbox": [90, 90, 99, 99]}]
    assert rd.twin_pair(group, random.Random(0), close=True) == (group[0], group[1])
    pairs, left = rd.pair_within(group + [{"bbox": [80, 90, 88, 99]}])
    assert [(a["bbox"][0], b["bbox"][0]) for a, b in pairs] == [(0, 12), (90, 80)] and left == []
    rows = [{"id": i, "instruction_noref": "Remove the selected region."} for i in "abcd"]
    tokens = {"a": {"roundtrip": 0.9, "size_ok": True, "out_diff": 0.2, "out_share8": 0.0},
              "b": {"roundtrip": 0.3, "size_ok": True, "out_diff": 0.2, "out_share8": 0.0},     # token misses the mask
              "c": {"roundtrip": 0.9, "size_ok": True, "out_diff": 4.0, "out_share8": 0.1},     # target drifts outside
              "d": {"roundtrip": 0.9, "size_ok": True, "out_diff": 0.1, "out_share8": 0.0}}
    assert [r["id"] for r in rd.v1_usable(rows, tokens, dev_ids={"d"})] == ["a", "c"]
    assert not rd.drifts(tokens["a"]) and rd.drifts(tokens["c"])


def test_c2_local_loss_takes_fm_only_near_the_edit(tiny_dit):  # noqa: F811
    class Scheduler:
        timesteps = torch.linspace(1000.0, 1.0, 1000)

        def add_noise(self, clean, noise, timestep):
            sigma = (timestep / 1000).view(-1, 1, 1, 1)
            return (1 - sigma) * clean + sigma * noise

        def training_target(self, clean, noise, timestep):
            return noise - clean

        def training_weight(self, timestep):
            return torch.ones_like(timestep)

    def model_fn(dit, latents, timestep, **kwargs):
        prediction = tiny_dit.run(latents.flatten(2).transpose(1, 2), kwargs["use_gradient_checkpointing"])
        return prediction.transpose(1, 2).reshape(latents.shape)

    pipe = SimpleNamespace(scheduler=Scheduler(), model_fn=model_fn, dit=tiny_dit.model, device="cpu", torch_dtype=torch.float32)
    region = torch.zeros(tiny_dit.grid)
    region[0, 0] = 1.0
    sample = r2.Round2Sample({"input_latents": torch.randn(1, 8, *tiny_dit.grid)}, "edit_v1", [region],
                             torch.zeros(tiny_dit.grid), region_dilate=True)
    noise = torch.randn(1, 8, *tiny_dit.grid)
    _, _, _, full = r2.round2_loss(pipe, sample, None, timestep_index=300, noise=noise, lambda_region=0.0)
    sample.loss_mask = torch.zeros(tiny_dit.grid)
    sample.loss_mask[:2, :2] = 1.0
    with torch.no_grad():
        latents = sample.inputs["input_latents"]
        t = Scheduler.timesteps[300].view(1)
        error = (model_fn(None, Scheduler().add_noise(latents, noise, t), t, use_gradient_checkpointing=False)
                 - (noise - latents)).pow(2).mean(1)[0]
    _, _, _, local = r2.round2_loss(pipe, sample, None, timestep_index=300, noise=noise, lambda_region=0.0)
    assert local["mse"] == pytest.approx(float(error[:2, :2].mean()), rel=1e-5)
    assert full["mse"] == pytest.approx(float(error.mean()), rel=1e-5)


def test_c2_unit_change_cuts_a_shared_changed_region_to_each_unit():
    changed = np.zeros((200, 200), bool)
    changed[10:60, 10:60] = True
    changed[120:180, 120:180] = True                  # the pair's whole change, given to every unit
    own_a = np.zeros_like(changed); own_a[15:55, 15:55] = True
    own_b = np.zeros_like(changed); own_b[125:175, 125:175] = True
    own_c = np.zeros_like(changed); own_c[80:90, 150:160] = True     # no change near it
    a, b, c = (r2.unit_change(changed, own) for own in (own_a, own_b, own_c))
    assert a[30, 30] and not a[150, 150] and b[150, 150] and not b[30, 30]
    assert (c == own_c).all() and r2.unit_change(None, own_c) is own_c
