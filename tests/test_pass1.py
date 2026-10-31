import json

import pytest
import torch
from PIL import Image

from samtok_edit21.pass1.model import adapter_enabled, grouped_loss
from samtok_edit21.pass1.protocol import (
    VERSION,
    box_normalized,
    finalize_spans,
    from_native_job,
    parse_answer,
    point_normalized,
    point_pixels,
    valid_span,
    validate_request,
)

S = "<|mt_start|><|mt_0001|><|mt_0257|><|mt_end|>"
T = "<|mt_start|><|mt_0002|><|mt_0258|><|mt_end|>"


def request():
    return {
        "protocol": VERSION,
        "coordinate_system": "normalized_0_1000",
        "targets": [
            {
                "label": "U1",
                "instruction": "Change the selected region to blue.",
                "has_ref": False,
                "locator": "point",
                "region": {"point": [400, 500]},
            },
            {
                "label": "U2",
                "instruction": "Remove the marked object.",
                "has_ref": False,
                "locator": "mask",
                "region": {"mask_2d": S},
            },
        ],
    }


def test_private_fields_and_wrong_locator_are_rejected():
    for key in ["category", "answer", "ref", "source_mask", "scope"]:
        r = request()
        r["targets"][0][key] = "hidden"
        with pytest.raises(ValueError):
            validate_request(r)
    r = request()
    r["targets"][0]["region"]["mask_2d"] = S
    with pytest.raises(ValueError):
        validate_request(r)
    r = request()
    r["targets"][0]["locator"] = "none"
    del r["targets"][0]["region"]
    with pytest.raises(ValueError):
        validate_request(r)


def test_binding_no_first_candidate_or_gt_repair():
    r = request()
    answer = [{"label": "U1", "mask_2d": T}, {"label": "U2", "mask_2d": T}]
    parsed = parse_answer(json.dumps(answer), r)
    assert parsed["complete"] and parsed["spans"]["U2"] == T
    assert finalize_spans(parsed, r) == {"U1": T, "U2": S}
    assert parsed["spans"]["U2"] == T  # raw metric remains wrong
    assert not parse_answer(json.dumps(answer[::-1]), r)["complete"]
    duplicate = parse_answer(json.dumps(answer + [answer[0]]), r)
    assert not duplicate["complete"] and "U1" not in duplicate["spans"]
    assert not parse_answer("```json\n" + json.dumps(answer) + "\n```", r)["complete"]
    assert parse_answer("```json\n" + json.dumps(answer) + "\n```", r, compatible=True)[
        "complete"
    ]
    assert not parse_answer('[{"label":"U1","label":"U2","mask_2d":"' + S + '"}]', r)[
        "complete"
    ]


def test_point_box_adapter_never_calls_codec_or_reads_private_mask():
    class ForbiddenCodec:
        def encode(self, *a, **kw):
            raise AssertionError("Unexpected codec call")

    job = {
        "protocol": "native_regions_v2",
        "source_size": [101, 99],
        "units": [
            {
                "id": "U1",
                "instruction": "Paint this red.",
                "has_ref": False,
                "locator": "point",
                "region": {"point": [100, 98]},
                "source_mask": "/nonexistent/private",
            },
            {
                "id": "U2",
                "instruction": "Remove this.",
                "has_ref": False,
                "locator": "box",
                "region": {"box": [0, 0, 101, 99]},
            },
        ],
    }
    r = from_native_job(job, Image.new("RGB", (101, 99)), ForbiddenCodec())
    assert r["targets"][1]["region"]["box"] == [0, 0, 1000, 1000]
    assert point_pixels(r["targets"][0]["region"]["point"], (101, 99)) == [100, 98]
    assert "source_mask" not in json.dumps(r)


def test_ranges_and_half_open_bounds():
    assert valid_span(S)
    assert not valid_span("<|mt_start|><|mt_0256|><|mt_0257|><|mt_end|>")
    assert not valid_span("<|mt_start|><|mt_0001|><|mt_0512|><|mt_end|>")
    assert not valid_span("<|mt_start|><|mt_０００１|><|mt_0257|><|mt_end|>")
    assert box_normalized([1, 2, 2, 3], (3000, 2000)) == [0, 1, 1, 2]
    with pytest.raises(ValueError):
        point_normalized([101, 2], (101, 99))


def test_group_loss_weights_and_gradient():
    logits = torch.tensor(
        [[0.0, 1.0], [1.0, 0.0], [0.0, 2.0], [2.0, 0.0]], requires_grad=True
    )
    labels = torch.tensor([0, 1, 0, 1])
    groups = torch.tensor([0, 0, 1, 2])
    loss, stats = grouped_loss(logits, labels, groups)
    ce = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    expected = (ce[:2].mean() + 0.25 * ce[2] + 0.25 * ce[3]) / 1.5
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert logits.grad.isfinite().all() and (logits.grad.abs().sum(-1) > 0).all()


def test_adapter_switch_restores_state_even_on_exception():
    from peft import LoraConfig, inject_adapter_in_model

    torch.manual_seed(7)
    base = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False))
    x = torch.ones(1, 4)
    raw = base(x).detach().clone()
    m = inject_adapter_in_model(
        LoraConfig(r=2, lora_alpha=2, target_modules=["0"]),
        base,
        adapter_name="pass1_grounding",
    )
    layer = m[0]
    with torch.no_grad():
        layer.lora_A["pass1_grounding"].weight.fill_(0.2)
        layer.lora_B["pass1_grounding"].weight.fill_(0.3)
    layer.enable_adapters(False)
    before = [p.requires_grad for p in m.parameters()]
    with pytest.raises(RuntimeError):
        with adapter_enabled(m, True):
            assert not torch.equal(m(x), raw)
            raise RuntimeError("simulated inference failure")
    assert layer.disable_adapters
    assert [p.requires_grad for p in m.parameters()] == before
    torch.testing.assert_close(m(x), raw, rtol=0, atol=0)


def test_fp32_checkpoint_reload_does_not_round_through_bf16(tmp_path):
    from peft import get_peft_model_state_dict

    from samtok_edit21.pass1.model import attach_lora
    from samtok_edit21.pass1.train import save_adapter

    def tiny_named_backbone():
        m = torch.nn.Module()
        m.language_model = torch.nn.Module()
        layers = []
        for _ in range(36):
            block = torch.nn.Module()
            block.self_attn = torch.nn.Module()
            block.mlp = torch.nn.Module()
            for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                setattr(block.self_attn, name, torch.nn.Linear(2, 2, bias=False))
            for name in ["gate_proj", "up_proj", "down_proj"]:
                setattr(block.mlp, name, torch.nn.Linear(2, 2, bias=False))
            layers.append(block)
        m.language_model.layers = torch.nn.ModuleList(layers)
        return m.bfloat16()

    trained = attach_lora(tiny_named_backbone(), train=True, rank=2)
    with torch.no_grad():
        for name, p in trained.named_parameters():
            if "lora_" in name:
                p.copy_(torch.arange(p.numel()).reshape(p.shape) / 12345 + 0.001234567)
    directory = tmp_path / "adapter"
    save_adapter(trained, directory, {"test": "roundtrip_precision"})
    loaded = attach_lora(tiny_named_backbone(), adapter=directory)
    expected = get_peft_model_state_dict(trained, adapter_name="pass1_grounding")
    actual = get_peft_model_state_dict(loaded, adapter_name="pass1_grounding")
    assert set(expected) == set(actual) and len(actual) == 504
    assert all(
        v.dtype == torch.float32 and torch.equal(v, actual[k])
        for k, v in expected.items()
    )
    assert all(not p.requires_grad for p in loaded.parameters())
