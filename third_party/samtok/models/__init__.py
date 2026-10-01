# Keep the codec import independent from the optional XTuner training stack.
# The Qwen3/Qwen2.5 perception models import XTuner and its older Transformers
# compatibility layer, while the released VQ-SAM2 codec only needs the classes
# below. Optional names remain available through lazy imports for callers that
# use the original SAMTok training package.
from .sam2 import VQ_SAM2, VQ_SAM2Config, SAM2Config

_OPTIONAL_NAMES = {
    "PerceptionLM",
    "PerceptionLMConfig",
    "Qwen25VLForConditionalGeneration",
    "Qwen25VLProcessingInfo",
    "Qwen25VLProcessor",
    "Qwen3VLForConditionalGeneration",
    "Qwen3VLProcessingInfo",
    "Qwen3VLProcessor",
    "VQ_SAM2Model",
}


def __getattr__(name):
    if name in {"PerceptionLM", "PerceptionLMConfig"}:
        from .perceptionlm import PerceptionLM, PerceptionLMConfig

        return {"PerceptionLM": PerceptionLM, "PerceptionLMConfig": PerceptionLMConfig}[name]
    if name in {"Qwen25VLForConditionalGeneration", "Qwen25VLProcessingInfo", "Qwen25VLProcessor"}:
        from .qwen25vl import (
            Qwen25VLForConditionalGeneration,
            Qwen25VLProcessingInfo,
            Qwen25VLProcessor,
        )

        return {
            "Qwen25VLForConditionalGeneration": Qwen25VLForConditionalGeneration,
            "Qwen25VLProcessingInfo": Qwen25VLProcessingInfo,
            "Qwen25VLProcessor": Qwen25VLProcessor,
        }[name]
    if name in {"Qwen3VLForConditionalGeneration", "Qwen3VLProcessingInfo", "Qwen3VLProcessor"}:
        from .qwen3vl import Qwen3VLForConditionalGeneration, Qwen3VLProcessingInfo, Qwen3VLProcessor

        return {
            "Qwen3VLForConditionalGeneration": Qwen3VLForConditionalGeneration,
            "Qwen3VLProcessingInfo": Qwen3VLProcessingInfo,
            "Qwen3VLProcessor": Qwen3VLProcessor,
        }[name]
    if name == "VQ_SAM2Model":
        from .sam2 import VQ_SAM2Model

        return VQ_SAM2Model
    raise AttributeError(name)


__all__ = ["VQ_SAM2", "VQ_SAM2Config", "SAM2Config", *_OPTIONAL_NAMES]

import numpy as np
from torchvision.transforms.functional import resize, to_pil_image
class DirectResize:
    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        img = to_pil_image(image, mode='RGB')
        return np.array(img.resize((self.target_length, self.target_length)))
