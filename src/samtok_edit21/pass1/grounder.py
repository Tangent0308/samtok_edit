"""Single-call grounding API; returns tokens ready for existing Pass2 rendering."""

from __future__ import annotations

import torch

from samtok_edit21.pass1.model import (
    MODEL,
    adapter_enabled,
    attach_lora,
    encode,
    load_model,
    processor_for,
)
from samtok_edit21.pass1.protocol import finalize_spans, from_native_job, parse_answer


class Pass1Grounder:
    def __init__(
        self,
        adapter=None,
        *,
        device="cuda:0",
        model=None,
        base_model=MODEL,
        max_pixels=2097152,
    ):
        self.model = (
            load_model(device, model=base_model, adapter=adapter)
            if model is None
            else model
        )
        self.processor = processor_for(model=base_model, max_pixels=max_pixels)
        self.device = next(self.model.parameters()).device
        # The shared TE stays raw between Pass1 calls, including after exceptions.
        from peft.tuners.tuners_utils import BaseTunerLayer

        for layer in self.model.modules():
            if isinstance(layer, BaseTunerLayer):
                layer.enable_adapters(False)
        self.model.eval()

    @classmethod
    def from_pipeline(cls, pipe, adapter, *, base_model=MODEL, max_pixels=2097152):
        """Borrow the pipeline's native TE; keep its Pass2 processor/CodeInjector intact."""
        hf = attach_lora(pipe.text_encoder.model, adapter=adapter)
        return cls(model=hf, base_model=base_model, max_pixels=max_pixels)

    @torch.inference_mode()
    def ground(self, image, request, *, max_new_tokens=512):
        inputs = {
            k: v.to(self.device)
            for k, v in encode(self.processor, image, request).items()
        }
        with adapter_enabled(self.model, True):
            self.model.model.rope_deltas = None
            ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True
            )
        generated = ids[0, inputs["input_ids"].shape[1] :]
        raw = self.processor.tokenizer.decode(generated, skip_special_tokens=False)
        parsed = parse_answer(raw, request)
        return {
            "raw": raw,
            "parsed": parsed,
            "request": request,
            "spans": finalize_spans(parsed, request) if parsed["complete"] else None,
        }

    def ground_native(self, job, image, codec=None, **kwargs):
        return self.ground(image, from_native_job(job, image, codec), **kwargs)
