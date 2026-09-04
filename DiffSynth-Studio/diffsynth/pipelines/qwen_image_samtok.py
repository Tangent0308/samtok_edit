"""Two-pass SAMTok-conditioned Qwen-Image-Edit-2511 pipeline."""

from __future__ import annotations

import math
from typing import Union

import torch
from PIL import Image

from ..core import ModelConfig
from ..core.data.samtok_dataset import SPAN_RE, parse_and_canonicalize_mt_cot
from ..core.device.npu_compatible_device import get_device_type
from ..diffusion.base_pipeline import PipelineUnit
from .qwen_image import (
    QwenImageBlockwiseMultiControlNet,
    QwenImagePipeline,
    QwenImageUnit_BlockwiseControlNet,
    QwenImageUnit_ContextImageEmbedder,
    QwenImageUnit_EditImageEmbedder,
    QwenImageUnit_EntityControl,
    QwenImageUnit_Inpaint,
    QwenImageUnit_InputImageEmbedder,
    QwenImageUnit_LayerInputImageEmbedder,
    QwenImageUnit_NoiseInitializer,
    QwenImageUnit_PromptEmbedder,
    QwenImageUnit_ShapeChecker,
)


EDIT_TEMPLATE_2511 = (
    "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, "
    "texture, objects, background), then explain how the user's text instruction should alter "
    "or modify the image. Generate a new image that meets the user's requirements while "
    "maintaining consistency with the original input where appropriate.<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)
EDIT_DROP_IDX = 64
IMAGE_PROMPT_TEMPLATE = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"


def shifted_cot_supervision(
    hidden: torch.Tensor,
    cot_ids: torch.Tensor,
    template_length: int,
) -> torch.Tensor:
    """Select hidden positions that predict every CoT token, including im_end."""

    if hidden.ndim != 3 or cot_ids.ndim != 2:
        raise ValueError(
            f"Expected hidden [B,L,H] and cot_ids [B,C], got "
            f"{tuple(hidden.shape)} and {tuple(cot_ids.shape)}"
        )
    if hidden.shape[0] != cot_ids.shape[0]:
        raise ValueError("CoT hidden/label batch sizes do not match")
    cot_length = cot_ids.shape[1]
    start = int(template_length) - 1
    stop = start + cot_length
    if start < 0 or stop > hidden.shape[1]:
        raise ValueError(
            f"Invalid shifted CoT slice [{start}:{stop}] for hidden length "
            f"{hidden.shape[1]}"
        )
    shifted = hidden[:, start:stop]
    if shifted.shape[:2] != cot_ids.shape:
        raise RuntimeError(
            f"Shifted CoT hidden/label mismatch: {tuple(shifted.shape)} vs "
            f"{tuple(cot_ids.shape)}"
        )
    return shifted


def _calculate_dimensions(target_area: int, ratio: float) -> tuple[int, int]:
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    return round(width / 32) * 32, round(height / 32) * 32


def _resize_condition_image(image: Image.Image, target_area: int = 384 * 384) -> Image.Image:
    width, height = _calculate_dimensions(target_area, image.size[0] / image.size[1])
    return image.resize((width, height))


def build_edit_model_inputs(
    pipe: QwenImagePipeline,
    prompt: str,
    edit_image: list[Image.Image],
    condition_image_area: int = 384 * 384,
):
    """Build the shared pass-1/pass-2 template prefix (alignment rule R2)."""

    if not isinstance(edit_image, list) or not edit_image:
        raise ValueError("Qwen-Image-Edit-2511 conditioning expects a non-empty image list")
    images = [
        _resize_condition_image(image, condition_image_area) for image in edit_image
    ]
    picture_prefix = "".join(
        IMAGE_PROMPT_TEMPLATE.format(index + 1) for index in range(len(images))
    )
    text = [EDIT_TEMPLATE_2511.format(picture_prefix + prompt)]
    return pipe.processor(
        text=text,
        images=images,
        padding=True,
        return_tensors="pt",
    ).to(pipe.device)


class QwenImageUnit_SamtokEmbedder(PipelineUnit):
    """Generate or pass through the canonical mask-token CoT before encoding."""

    def __init__(self):
        super().__init__(
            take_over=True,
            input_params=("edit_image", "samtok_online_cot", "samtok_max_new_tokens"),
            input_params_posi={"prompt": "prompt", "mt_cot": "mt_cot"},
            output_params=("mt_cot",),
            onload_model_names=("text_encoder",),
        )

    def process(self, pipe, inputs_shared, inputs_posi, inputs_nega):
        mt_cot = inputs_posi.get("mt_cot")
        if mt_cot is None:
            mt_cot = getattr(pipe, "_samtok_requested_mt_cot", None)

        if mt_cot is not None:
            mt_cot, layer = parse_and_canonicalize_mt_cot(mt_cot, return_layer=True)
            if mt_cot is None:
                raise ValueError("Explicit mt_cot does not contain a valid SAMTok span or []")
            pipe.last_parse_layer = f"provided:{layer}"

        online = inputs_shared.get("samtok_online_cot")
        if online is None:
            online = getattr(pipe, "_samtok_online_cot", False)
        max_new_tokens = inputs_shared.get("samtok_max_new_tokens")
        if max_new_tokens is None:
            max_new_tokens = getattr(pipe, "_samtok_max_new_tokens", 128)
        edit_image = inputs_shared.get("edit_image")

        if mt_cot is None and online and pipe.text_encoder is not None and edit_image is not None:
            pipe.load_models_to_device(self.onload_model_names)
            model_inputs = build_edit_model_inputs(
                pipe, inputs_posi["prompt"], edit_image
            )
            with torch.no_grad():
                output_ids = pipe.text_encoder.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    eos_token_id=pipe.im_end_id,
                )
            generated = pipe.processor.tokenizer.decode(
                output_ids[0, model_inputs.input_ids.shape[1] :],
                skip_special_tokens=False,
            )
            mt_cot, pipe.last_parse_layer = parse_and_canonicalize_mt_cot(
                generated, return_layer=True
            )
            pipe.last_pass1_raw = generated

        inputs_posi["mt_cot"] = mt_cot
        inputs_nega["mt_cot"] = None
        pipe.last_mt_cot = mt_cot
        return inputs_shared, inputs_posi, inputs_nega


class QwenImageUnit_SamtokPromptEmbedder(QwenImageUnit_PromptEmbedder):
    """Encode template+CoT once and optionally expose the shifted NTP slice."""

    def __init__(self):
        PipelineUnit.__init__(
            self,
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt", "mt_cot": "mt_cot"},
            input_params_nega={"prompt": "negative_prompt"},
            input_params=("edit_image", "samtok_need_ntp"),
            output_params=(
                "prompt_emb",
                "prompt_emb_mask",
                "samtok_cot_hidden",
                "samtok_cot_labels",
            ),
            onload_model_names=("text_encoder",),
        )

    def encode_prompt(self, pipe: QwenImagePipeline, prompt):
        template = (
            "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
            "texture, quantity, text, spatial relationships of the objects and background:"
            "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        )
        drop_idx = 34
        text = [template.format(value) for value in prompt]
        tokens = pipe.tokenizer(
            text,
            max_length=4096 + drop_idx,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(pipe.device)
        if tokens.input_ids.shape[1] >= 1024:
            print(
                "Warning: Qwen-Image was trained on shorter prompts; current prompt uses "
                f"{tokens.input_ids.shape[1] - drop_idx} conditioning tokens."
            )
        hidden = pipe.text_encoder.encode(
            input_ids=tokens.input_ids,
            attention_mask=tokens.attention_mask,
        )[-1]
        split = self.extract_masked_hidden(hidden, tokens.attention_mask)
        return [item[drop_idx:] for item in split]

    def encode_prompt_edit_multi(
        self,
        pipe: QwenImagePipeline,
        prompt: str,
        edit_image: list[Image.Image],
        mt_cot: str | None = None,
        need_ntp: bool = False,
    ):
        pipe.last_ntp_alignment = None
        model_inputs = build_edit_model_inputs(pipe, prompt, edit_image)
        input_ids = model_inputs.input_ids
        attention_mask = model_inputs.attention_mask
        template_length = input_ids.shape[1]
        cot_ids = None

        prompt_spans = [match.group(0) for match in SPAN_RE.finditer(prompt)]
        prompt_span_ids = []
        for span in prompt_spans:
            ids = pipe.processor.tokenizer(
                span, add_special_tokens=False, return_tensors="pt"
            ).input_ids[0].tolist()
            prompt_span_ids.append(ids)
        pipe.last_user_mask_audit = {
            "user_mask_span_count": len(prompt_spans),
            "user_mask_span_token_ids": prompt_span_ids,
            "user_mask_spans_atomic": all(len(ids) == 4 for ids in prompt_span_ids),
            "user_mask_spans_in_template": all(
                any(
                    input_ids[0, start : start + len(ids)].tolist() == ids
                    for start in range(template_length - len(ids) + 1)
                )
                for ids in prompt_span_ids
            ),
        }

        if mt_cot is not None:
            cot_ids = pipe.processor.tokenizer(
                mt_cot + "<|im_end|>",
                add_special_tokens=False,
                return_tensors="pt",
            ).input_ids.to(input_ids.device)
            input_ids = torch.cat([input_ids, cot_ids], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(cot_ids)], dim=1
            )

        hidden = pipe.text_encoder.encode(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=model_inputs.pixel_values,
            image_grid_thw=model_inputs.image_grid_thw,
        )[-1]
        split = self.extract_masked_hidden(hidden, attention_mask)
        split = [item[EDIT_DROP_IDX:] for item in split]

        extra = {}
        if need_ntp:
            if cot_ids is None:
                raise ValueError("NTP supervision requested for a sample without mt_cot")
            cot_length = cot_ids.shape[1]
            shifted_hidden = shifted_cot_supervision(
                hidden, cot_ids, template_length
            )
            if pipe.im_end_id is None or int(cot_ids[0, -1].item()) != pipe.im_end_id:
                raise RuntimeError("The final NTP label must be the <|im_end|> token")
            extra["samtok_cot_hidden"] = shifted_hidden
            extra["samtok_cot_labels"] = cot_ids
            pipe.last_ntp_alignment = {
                "template_tokens": int(template_length),
                "full_sequence_tokens": int(hidden.shape[1]),
                "cot_label_tokens": int(cot_length),
                "cot_hidden_start": int(template_length - 1),
                "cot_hidden_stop": int(template_length - 1 + cot_length),
                "cot_hidden_tokens": int(shifted_hidden.shape[1]),
                "cot_first_label_id": int(cot_ids[0, 0].item()),
                "cot_last_label_id": int(cot_ids[0, -1].item()),
                "im_end_token_id": int(pipe.im_end_id),
                "ntp_shift_alignment_ok": True,
            }
        return split, extra

    def process(
        self,
        pipe: QwenImagePipeline,
        prompt,
        edit_image=None,
        mt_cot=None,
        samtok_need_ntp=False,
    ) -> dict:
        pipe.load_models_to_device(self.onload_model_names)
        if pipe.text_encoder is None:
            return {}
        extra = {}
        if edit_image is None:
            split = self.encode_prompt(pipe, [prompt])
        else:
            if isinstance(edit_image, Image.Image):
                edit_image = [edit_image]
            split, extra = self.encode_prompt_edit_multi(
                pipe,
                prompt,
                edit_image,
                mt_cot=mt_cot,
                need_ntp=samtok_need_ntp,
            )

        masks = [
            torch.ones(item.size(0), dtype=torch.long, device=item.device)
            for item in split
        ]
        max_length = max(item.size(0) for item in split)
        prompt_embeds = torch.stack(
            [
                torch.cat(
                    [item, item.new_zeros(max_length - item.size(0), item.size(1))]
                )
                for item in split
            ]
        )
        encoder_attention_mask = torch.stack(
            [
                torch.cat([mask, mask.new_zeros(max_length - mask.size(0))])
                for mask in masks
            ]
        )
        output = {
            "prompt_emb": prompt_embeds.to(
                dtype=pipe.torch_dtype, device=pipe.device
            ),
            "prompt_emb_mask": encoder_attention_mask,
        }
        output.update(extra)
        return output


class QwenImageUnit_SamtokEntityControl(QwenImageUnit_EntityControl):
    """Retain optional EliGen support with the native-HF SAMTok wrapper."""

    def get_prompt_emb(self, pipe: QwenImagePipeline, prompt) -> dict:
        if pipe.text_encoder is None:
            return {}
        embedder = QwenImageUnit_SamtokPromptEmbedder()
        split = embedder.encode_prompt(pipe, [prompt])
        masks = [
            torch.ones(item.size(0), dtype=torch.long, device=item.device)
            for item in split
        ]
        max_length = max(item.size(0) for item in split)
        embeddings = torch.stack(
            [
                torch.cat(
                    [item, item.new_zeros(max_length - item.size(0), item.size(1))]
                )
                for item in split
            ]
        ).to(dtype=pipe.torch_dtype, device=pipe.device)
        attention_mask = torch.stack(
            [
                torch.cat([mask, mask.new_zeros(max_length - mask.size(0))])
                for mask in masks
            ]
        )
        return {"prompt_emb": embeddings, "prompt_emb_mask": attention_mask}


class QwenImageSamtokPipeline(QwenImagePipeline):
    """Qwen-Image-Edit-2511 with online SAMTok CoT generation and conditioning."""

    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.units = [
            QwenImageUnit_ShapeChecker(),
            QwenImageUnit_NoiseInitializer(),
            QwenImageUnit_InputImageEmbedder(),
            QwenImageUnit_Inpaint(),
            QwenImageUnit_EditImageEmbedder(),
            QwenImageUnit_LayerInputImageEmbedder(),
            QwenImageUnit_ContextImageEmbedder(),
            QwenImageUnit_SamtokEmbedder(),
            QwenImageUnit_SamtokPromptEmbedder(),
            QwenImageUnit_SamtokEntityControl(),
            QwenImageUnit_BlockwiseControlNet(),
        ]
        self.im_end_id = None
        self.last_mt_cot = None
        self.last_pass1_raw = None
        self.last_parse_layer = None
        self.last_ntp_alignment = None

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] | None = None,
        tokenizer_config: ModelConfig | None = None,
        processor_config: ModelConfig | None = None,
        vram_limit: float | None = None,
    ):
        pipe = QwenImageSamtokPipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs or [], vram_limit)
        pipe.text_encoder = model_pool.fetch_model("qwen_image_text_encoder")
        pipe.dit = model_pool.fetch_model("qwen_image_dit")
        pipe.vae = model_pool.fetch_model("qwen_image_vae")
        blockwise_models = model_pool.fetch_model(
            "qwen_image_blockwise_controlnet", index="all"
        )
        pipe.blockwise_controlnet = (
            None
            if blockwise_models is None
            else QwenImageBlockwiseMultiControlNet(blockwise_models)
        )
        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary()
            from transformers import Qwen2Tokenizer

            pipe.tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_config.path)
        if processor_config is not None:
            processor_config.download_if_necessary()
            from transformers import Qwen2VLProcessor

            pipe.processor = Qwen2VLProcessor.from_pretrained(processor_config.path)
        pipe.siglip2_image_encoder = model_pool.fetch_model("siglip2_image_encoder")
        pipe.dinov3_image_encoder = model_pool.fetch_model("dinov3_image_encoder")
        pipe.image2lora_style = model_pool.fetch_model("qwen_image_image2lora_style")
        pipe.image2lora_coarse = model_pool.fetch_model("qwen_image_image2lora_coarse")
        pipe.image2lora_fine = model_pool.fetch_model("qwen_image_image2lora_fine")

        tokenizer = pipe.processor.tokenizer if pipe.processor is not None else pipe.tokenizer
        if pipe.text_encoder is not None:
            if tokenizer is None or pipe.processor is None:
                raise ValueError(
                    "SAMTok text encoding requires tokenizer_config and processor_config "
                    "pointing to the merged SAMTok processor directory"
                )
            mt_start_id = tokenizer.convert_tokens_to_ids("<|mt_start|>")
            if mt_start_id is None or mt_start_id == tokenizer.unk_token_id:
                raise ValueError(
                    "Tokenizer lacks SAMTok tokens; use prepare_samtok_te_dir.py first"
                )
            pipe.im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        *args,
        mt_cot: str | None = None,
        enable_samtok_cot: bool = True,
        samtok_max_new_tokens: int = 128,
        **kwargs,
    ):
        self.last_mt_cot = None
        self.last_pass1_raw = None
        self.last_parse_layer = None
        self._samtok_requested_mt_cot = mt_cot
        self._samtok_online_cot = bool(enable_samtok_cot)
        self._samtok_max_new_tokens = int(samtok_max_new_tokens)
        try:
            return super().__call__(*args, **kwargs)
        finally:
            self._samtok_requested_mt_cot = None
            self._samtok_online_cot = False
