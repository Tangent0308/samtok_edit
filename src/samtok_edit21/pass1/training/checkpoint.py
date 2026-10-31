"""FP32 adapter serialization and adapter-off TE invariance probes."""

import json

import torch

from samtok_edit21.pass1.model import adapter_enabled


def singleton(items):
    if len(items) != 1:
        raise ValueError("This pilot uses microbatch=1")
    return items[0]


@torch.no_grad()
def pass2_probe(model, data, device):
    x, *_ = data[0]
    x = {k: v.to(device) for k, v in x.items()}
    captured = []

    def hook(_module, args):
        captured.append(args[0][:, -8:].detach().cpu().clone())

    handle = model.model.language_model.norm.register_forward_pre_hook(hook)
    was_training = model.training
    model.eval()
    try:
        with adapter_enabled(model, False):
            model.model.rope_deltas = None
            out = model(**x, logits_to_keep=1, use_cache=False)
        return {"logits": out.logits.cpu().clone(), "pre_norm": captured[0]}
    finally:
        handle.remove()
        model.train(was_training)


def save_adapter(model, path, metadata):
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file

    path.mkdir(parents=True, exist_ok=False)
    state = get_peft_model_state_dict(model, adapter_name="pass1_grounding")
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in state.items()},
        str(path / "adapter_model.safetensors"),
    )
    model.peft_config["pass1_grounding"].save_pretrained(path)
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
