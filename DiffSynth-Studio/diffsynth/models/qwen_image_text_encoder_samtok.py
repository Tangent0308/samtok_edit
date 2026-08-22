"""Qwen2.5-VL text encoder with the 514-token SAMTok vocabulary."""

from __future__ import annotations

import torch
from transformers import (
    GenerationConfig,
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
)


def _base_config(vocab_size: int) -> dict:
    """Return the Qwen-Image 7B text-encoder architecture configuration."""

    text_config = {
        "architectures": ["Qwen2_5_VLForConditionalGeneration"],
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "eos_token_id": 151645,
        "hidden_act": "silu",
        "hidden_size": 3584,
        "image_token_id": None,
        "initializer_range": 0.02,
        "intermediate_size": 18944,
        "layer_types": ["full_attention"] * 28,
        "max_position_embeddings": 128000,
        "max_window_layers": 28,
        "model_type": "qwen2_5_vl_text",
        "num_attention_heads": 28,
        "num_hidden_layers": 28,
        "num_key_value_heads": 4,
        "rms_norm_eps": 1e-6,
        "rope_scaling": {
            "mrope_section": [16, 24, 24],
            "rope_type": "default",
            "type": "default",
        },
        "rope_theta": 1_000_000.0,
        "sliding_window": None,
        "use_cache": True,
        "use_sliding_window": False,
        "video_token_id": None,
        "vision_end_token_id": 151653,
        "vision_start_token_id": 151652,
        "vision_token_id": 151654,
        "vocab_size": vocab_size,
    }
    return {
        "architectures": ["Qwen2_5_VLForConditionalGeneration"],
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "eos_token_id": 151645,
        "hidden_act": "silu",
        "hidden_size": 3584,
        "image_token_id": 151655,
        "initializer_range": 0.02,
        "intermediate_size": 18944,
        "max_position_embeddings": 128000,
        "max_window_layers": 28,
        "model_type": "qwen2_5_vl",
        "num_attention_heads": 28,
        "num_hidden_layers": 28,
        "num_key_value_heads": 4,
        "rms_norm_eps": 1e-6,
        "rope_scaling": {
            "mrope_section": [16, 24, 24],
            "rope_type": "default",
            "type": "default",
        },
        "rope_theta": 1_000_000.0,
        "sliding_window": 32768,
        "text_config": text_config,
        "tie_word_embeddings": False,
        "use_cache": True,
        "use_sliding_window": False,
        "video_token_id": 151656,
        "vision_config": {
            "depth": 32,
            "fullatt_block_indexes": [7, 15, 23, 31],
            "hidden_act": "silu",
            "hidden_size": 1280,
            "in_channels": 3,
            "in_chans": 3,
            "initializer_range": 0.02,
            "intermediate_size": 3420,
            "model_type": "qwen2_5_vl",
            "num_heads": 16,
            "out_hidden_size": 3584,
            "patch_size": 14,
            "spatial_merge_size": 2,
            "spatial_patch_size": 14,
            "temporal_patch_size": 2,
            "tokens_per_second": 2,
            "window_size": 112,
        },
        "vision_end_token_id": 151653,
        "vision_start_token_id": 151652,
        "vision_token_id": 151654,
        "vocab_size": vocab_size,
    }


class QwenImageSamtokTextEncoder(Qwen2_5_VLForConditionalGeneration):
    """Qwen-Image's Qwen2.5-VL encoder extended with SAMTok generation.

    ``forward`` intentionally remains the native Hugging Face implementation
    because ``generate`` depends on it.  DiffSynth conditioning uses ``encode``
    to obtain final-normalized hidden states without materializing full-vocab
    logits.  NTP applies ``lm_head`` only to the short CoT slice.
    """

    def __init__(
        self,
        vocab_size: int = 152179,
        im_end_token_id: int = 151645,
        pad_token_id: int = 151643,
    ):
        config = Qwen2_5_VLConfig(**_base_config(vocab_size))
        super().__init__(config)
        self.generation_config = GenerationConfig(
            eos_token_id=[im_end_token_id],
            pad_token_id=pad_token_id,
            do_sample=False,
            use_cache=True,
        )

    def encode(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
            **kwargs,
        )
        return outputs.hidden_states

    def ntp_logits(self, hidden_slice: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_slice)
