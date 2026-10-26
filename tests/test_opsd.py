"""On-policy self-distillation: prompts, allowed region, teacher target and loss, state picking, teacher weights,
guidance and the rollout (CPU; the model is replaced by small stand-ins where only the arithmetic matters)."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from samtok_edit21.data.protocol import span_of
from samtok_edit21.preparation import round2_data as rd
from samtok_edit21.training import opsd

A, B = span_of([12, 300]), span_of([40, 450])
SPEC = {"version": "2.0", "operation": "replace", "ref": "the white wolf", "scope": "wolf", "payload": "a unicorn",
        "target_part": "", "remove_mode": "whole", "attributes": [], "placement": {}, "qualifier": ""}
COLOR = {"version": "2.0", "operation": "attribute", "ref": "the red car", "scope": "car", "payload": "",
         "target_part": "", "remove_mode": "whole", "attributes": [{"property": "color", "value": "blue"}],
         "placement": {}, "qualifier": ""}


def rle(mask):
    out = rd.encode_rle(mask)
    return {"rle_size": list(out["size"]), "rle_counts": out["counts"]}


def test_prompts_name_every_unit_for_the_teacher():
    record = {"id": "x", "units": [{"instruction_spec": SPEC, "token": {"span": A}}]}
    student, teacher, negative = opsd.prompts(record, {"form": "noref_token"})
    assert student == f"Replace the selected region {A} with a unicorn."
    assert teacher == f"Replace the white wolf {A} with a unicorn."
    assert negative == "Replace the white wolf with a unicorn."
    assert opsd.prompts(record, {"form": "ref_token"})[0] == teacher
    pair = {"id": "y", "units": [{"instruction_spec": SPEC, "token": {"span": A}},
                                 {"instruction_spec": COLOR, "token": {"span": B}}]}
    student, teacher, negative = opsd.prompts(pair, {"order": [1, 0], "forms": ["noref_token", "ref_token"]})
    assert student == rd.multi_edit_prompt([f"Change the color of the selected region {B} to blue.",
                                            f"Replace the white wolf {A} with a unicorn."])
    assert teacher == rd.multi_edit_prompt([f"Change the color of the red car {B} to blue.",
                                            f"Replace the white wolf {A} with a unicorn."])
    assert "<|mt_" not in negative and negative.count("U2:") == 1
    with pytest.raises(ValueError):
        opsd.prompts(pair, {"order": [0, 0], "forms": ["ref_token", "ref_token"]})


def test_allowed_region_is_the_union_of_every_units_regions():
    shape = (8, 10)
    s, e, c, other = (np.zeros(shape, bool) for _ in range(4))
    s[1:3, 1:3], e[2:4, 2:5], c[6:8, 8:10], other[0, 9] = True, True, True, True
    record = {"id": "x", "units": [{"selection": rle(s), "execution_region": rle(e), "edited_region": None},
                                   {"selection": rle(other), "edited_region": rle(c)}]}
    assert opsd.allowed_region(record, shape).tolist() == (s | e | c | other).tolist()
    with pytest.raises(ValueError):
        opsd.allowed_region({"id": "empty", "units": [{"selection": rle(np.zeros(shape, bool))}]}, shape)


def test_teacher_clean_estimate_and_loss():
    torch.manual_seed(0)
    latents, velocity, source = torch.randn(1, 4, 3, 3), torch.randn(1, 4, 3, 3), torch.randn(1, 4, 3, 3)
    sigma = 0.6
    inside = opsd.teacher_clean(latents, sigma, velocity, torch.ones(1, 1, 3, 3), source)
    outside = opsd.teacher_clean(latents, sigma, velocity, torch.zeros(1, 1, 3, 3), source)
    assert torch.allclose(inside, latents - sigma * velocity) and torch.equal(outside, source)
    half = opsd.teacher_clean(latents, sigma, velocity, torch.full((1, 1, 3, 3), 0.5), source)
    assert torch.allclose(half, 0.5 * (latents - sigma * velocity) + 0.5 * source)
    # the loss is the velocity error weighted by sigma^2, and zero at the teacher's own velocity
    assert float(opsd.student_loss(latents, sigma, velocity, inside)) == pytest.approx(0.0, abs=1e-12)
    other = torch.randn(1, 4, 3, 3)
    expected = (sigma ** 2) * (other - velocity).pow(2).mean()
    assert float(opsd.student_loss(latents, sigma, other, inside)) == pytest.approx(float(expected), rel=1e-5)


def test_opsd_rows_fill_whole_step_pairs_on_every_rank():
    import random

    updates, world, accumulation = 5, 4, 4
    units = ([[{"kind": "opsd_merged", "id": f"o{k}"}] for k in range(3 * 2 * world + 2)]
             + [[{"kind": "hl", "id": f"p{k}a"}, {"kind": "hl", "id": f"p{k}b"}] for k in range(6)])
    units += [[{"kind": "edit_merged", "id": f"s{k}"}] for k in range(updates * world * accumulation - 38)]
    rows = rd.place_in_slots(units, random.Random(0), updates, world, accumulation)
    assert sorted(r["id"] for r in rows) == sorted(x["id"] for u in units for x in u)
    heavy = [[rows[(2 * g + d) * world + r]["kind"] == "opsd_merged" for d in (0, 1) for r in range(world)]
             for g in range(updates * accumulation // 2)]
    assert [g for g, step in enumerate(heavy) if all(step)] == [3, 6, 9]      # three step pairs, spread evenly
    assert sum(any(step) and not all(step) for step in heavy) <= 2           # the two left over join the singles
    for k in range(6):                                                       # pairs keep one rank and one update
        a, b = (next(i for i, r in enumerate(rows) if r["id"] == f"p{k}{m}") for m in "ab")
        (step_a, rank_a), (step_b, rank_b) = divmod(a, world), divmod(b, world)
        assert rank_a == rank_b and step_a // accumulation == step_b // accumulation and abs(step_a - step_b) == 1


def test_pick_states_are_distinct_ordered_and_noisy_enough():
    sigmas = torch.tensor([1.0, 0.8, 0.5, 0.2, 0.05])
    for seed in range(20):
        picked = opsd.pick_states(sigmas, 3, 0.1, torch.Generator().manual_seed(seed))
        assert picked == sorted(set(picked)) and len(picked) == 3 and all(sigmas[i] >= 0.1 for i in picked)
    assert opsd.pick_states(sigmas, 10, 0.1) == [0, 1, 2, 3]
    with pytest.raises(ValueError):
        opsd.pick_states(sigmas, 1, 2.0)


def test_teacher_weights_swap_in_restore_and_ema():
    first, second = torch.nn.Parameter(torch.ones(3)), torch.nn.Parameter(torch.zeros(2, 2))
    teacher = opsd.TeacherWeights([first, second])
    with torch.no_grad():
        first.add_(1.0)
        second.add_(5.0)
    with teacher.active():
        assert first.tolist() == [1.0, 1.0, 1.0] and second.sum() == 0
    assert first.tolist() == [2.0, 2.0, 2.0] and float(second.detach().sum()) == 20.0
    with pytest.raises(RuntimeError):                    # the student's values come back after an error too
        with teacher.active():
            raise RuntimeError
    assert first.tolist() == [2.0, 2.0, 2.0]
    teacher.update_ema(0.75)
    assert teacher.weights[0].tolist() == [1.25, 1.25, 1.25]
    restored = opsd.TeacherWeights([first, second], teacher.state())
    assert restored.weights[0].tolist() == [1.25, 1.25, 1.25]
    with pytest.raises(ValueError):
        opsd.TeacherWeights([first], [torch.zeros(4)])
    with pytest.raises(ValueError):
        teacher.update_ema(1.0)


def test_evaluating_switches_to_eval_mode_and_back():
    module = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.5))
    with opsd.evaluating(module):
        assert not module.training and not module[1].training
    assert module.training and module[1].training
    with pytest.raises(RuntimeError):                    # restored after an error too
        with opsd.evaluating(module):
            raise RuntimeError
    assert module[1].training
    module.eval()
    with opsd.evaluating(module):
        pass
    assert not module.training


def fake_pipe(velocity_of):
    """A pipe whose model_fn returns ``velocity_of(latents, conditioning)``; records the KV caches it was given."""
    calls = []

    def model_fn(dit, latents, timestep, kv_cache=None, **kwargs):
        calls.append({"cache": kv_cache, "prompt": kwargs.get("prompt"), "flex": kwargs.get("use_flex_attention")})
        return velocity_of(latents, kwargs.get("prompt"))

    pipe = SimpleNamespace(dit=SimpleNamespace(transformer_blocks=[0, 1]), model_fn=model_fn, device="cpu",
                           torch_dtype=torch.float32)
    return pipe, calls


def test_guidance_combines_as_cfg_and_skips_the_negative_at_one():
    pipe, calls = fake_pipe(lambda latents, prompt: torch.full_like(latents, 3.0 if prompt == "pos" else 1.0))
    latents, t = torch.zeros(1, 2, 2, 2), torch.tensor([500.0])
    out = opsd.guided(pipe, {"prompt": "pos"}, {"prompt": "neg"}, latents, t, 2.0)
    assert torch.equal(out, torch.full_like(latents, 1.0 + 2.0 * (3.0 - 1.0)))
    calls.clear()
    assert torch.equal(opsd.guided(pipe, {"prompt": "pos"}, None, latents, t, 1.0), torch.full_like(latents, 3.0))
    assert [c["prompt"] for c in calls] == ["pos"] and calls[0]["flex"] is True


def test_rollout_takes_the_pipelines_euler_steps_with_a_kv_cache():
    from diffsynth.diffusion.flow_match import FlowMatchScheduler

    scheduler = FlowMatchScheduler("Qwen-Image")
    scheduler.set_timesteps(4, dynamic_shift_len=16)
    pipe, calls = fake_pipe(lambda latents, prompt: torch.ones_like(latents))
    noise = torch.randn(1, 2, 4, 4)
    states = opsd.rollout(pipe, scheduler, {"prompt": "pos"}, None, noise, 1.0)
    sigmas = scheduler.sigmas.tolist()
    assert len(states) == 4 and torch.equal(states[0], noise)
    for index in range(1, 4):                           # x_{i+1} = x_i + v * (sigma_{i+1} - sigma_i), v = 1
        assert torch.allclose(states[index], noise + (sigmas[index] - sigmas[0]), atol=1e-6)
    caches = [c["cache"] for c in calls]
    assert all(cache is caches[0] for cache in caches) and len(caches[0]) == 2   # one cache list, one dict per block
