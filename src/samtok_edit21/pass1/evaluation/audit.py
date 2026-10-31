"""Actual-checkpoint precision and shared-TE API audit, without loading a DiT."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from peft import get_peft_model_state_dict
from peft.tuners.tuners_utils import BaseTunerLayer
from PIL import Image
from safetensors.torch import load_file

from samtok_edit21.models.pipeline import SamtokTextEncoder
from samtok_edit21.pass1.grounder import Pass1Grounder
from samtok_edit21.pass1.model import encode, load_model, processor_for


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    data = Path(args.data)
    row = json.loads(data.read_text().splitlines()[0])
    with Image.open(data.parent / row["image"]) as im:
        image = im.convert("RGB")
    model = load_model("cuda:0")
    processor = processor_for()
    x = {k: v.cuda() for k, v in encode(processor, image, row["request"]).items()}
    pipe = SimpleNamespace(text_encoder=SamtokTextEncoder(model), processor=processor)

    @torch.inference_mode()
    def probe():
        pre, norm = pipe.text_encoder.encode(**x)
        return {
            "pre_norm": pre[:, -8:].cpu().clone(),
            "norm": norm[:, -8:].cpu().clone(),
        }

    raw = probe()
    grounder = Pass1Grounder.from_pipeline(pipe, args.adapter)
    state = get_peft_model_state_dict(model, adapter_name="pass1_grounding")
    saved = load_file(str(Path(args.adapter) / "adapter_model.safetensors"))
    assert set(state) == set(saved)
    exact = all(
        v.dtype == torch.float32 and torch.equal(v.detach().cpu(), saved[k])
        for k, v in state.items()
    )
    if not exact:
        raise RuntimeError("Loaded LoRA differs from FP32 checkpoint")
    result = grounder.ground(image, row["request"])
    off = all(
        m.disable_adapters for m in model.modules() if isinstance(m, BaseTunerLayer)
    )
    after = probe()
    features = {
        k: {
            "equal": torch.equal(v, after[k]),
            "max_abs_diff": float((v.float() - after[k].float()).abs().max()),
        }
        for k, v in raw.items()
    }
    output = {
        "adapter": args.adapter,
        "request_id": row["id"],
        "all_saved_tensors_reloaded_exactly": exact,
        "adapter_tensors": len(saved),
        "all_adapters_disabled_after_ground": off,
        "pass2_processor_identity_preserved": pipe.processor is processor,
        "pass2_features": features,
        "strict_output_complete": result["parsed"]["complete"],
        "raw": result["raw"],
        "returned_spans": result["spans"],
    }
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output), flush=True)
    if (
        not off
        or not result["parsed"]["complete"]
        or any(not v["equal"] for v in features.values())
    ):
        raise RuntimeError("Shared-TE integration audit failed")


if __name__ == "__main__":
    main()
