#!/usr/bin/env python3
"""Thin, inference-only wrapper around the released SAMTok VQ-SAM2 codec."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DIFFSYNTH_ROOT = _REPO_ROOT / "DiffSynth-Studio"
for path in [str(_REPO_ROOT), str(_DIFFSYNTH_ROOT)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from diffsynth.core.data.samtok_dataset import (  # noqa: E402
    CODEBOOK_DEPTH,
    CODEBOOK_SIZE,
    SPAN_RE,
    span_of,
    valid_span_codes,
)


def fix_mt_format_comprehensive(text: str) -> str:
    """SAMTok's visualization-only malformed-span fixer.

    This function must never be imported by the conditioning pipeline: it can
    invent the out-of-vocabulary ``mt_9999`` placeholder and is only retained
    to mirror the released detokenizer's debugging behavior.
    """

    text = re.sub(
        r"(<\|mt_start\|>)(<\|mt_\d+\|>)(<\|mt_\d+\|>)"
        r"(?:<\|mt_\d+\|>)+<\|mt_end\|>",
        r"\1\2\3<|mt_end|>",
        text,
    )
    text = re.sub(
        r"(<\|mt_start\|>)(<\|mt_\d+\|>)(<\|mt_end\|>)",
        r"\1\2<|mt_9999|><|mt_end|>",
        text,
    )
    return re.sub(
        r"(<\|mt_start\|>)(<\|mt_\d+\|>)(?!<\|mt_)",
        r"\1\2<|mt_9999|><|mt_end|>",
        text,
    )


class SamtokCodec:
    def __init__(
        self,
        sam2_ckpt: str | os.PathLike,
        tokenizer_ckpt: str | os.PathLike,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        from samtok.models import DirectResize, SAM2Config, VQ_SAM2, VQ_SAM2Config

        sam2_ckpt = str(Path(sam2_ckpt).expanduser().resolve())
        tokenizer_ckpt = str(Path(tokenizer_ckpt).expanduser().resolve())
        config = VQ_SAM2Config(
            sam2_config=SAM2Config(ckpt_path=sam2_ckpt),
            codebook_size=CODEBOOK_SIZE,
            codebook_depth=CODEBOOK_DEPTH,
            shared_codebook=False,
            latent_dim=256,
        )
        # Keep the released codec in fp32.  Its positional-encoding buffers are
        # consumed by explicit fp32 coordinates; casting the entire module to
        # bf16 causes a float/bfloat16 matmul mismatch in SAM's prompt encoder.
        if dtype != torch.float32:
            print(
                f"[SamtokCodec] requested dtype={dtype}; using float32 for released VQ-SAM2 compatibility"
            )
        self.vq = VQ_SAM2(config).to(device=device).eval()
        state = torch.load(tokenizer_ckpt, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.vq.load_state_dict(state, strict=True)
        self.resize = DirectResize(1024)
        self.device = torch.device(device)

    def _pixel_values(self, images) -> torch.Tensor:
        tensors = []
        for image in images:
            array = self.resize.apply_image(np.asarray(image.convert("RGB")))
            tensors.append(
                torch.from_numpy(array).permute(2, 0, 1).contiguous()
            )
        return torch.stack(tensors).to(device=self.device, dtype=self.vq.dtype)

    @staticmethod
    def _ordered_masks(binary_masks) -> tuple[torch.Tensor, list[int]]:
        masks = [
            torch.from_numpy(np.ascontiguousarray(np.asarray(mask) > 0))
            for mask in binary_masks
        ]
        if not masks or any(mask.sum() == 0 for mask in masks):
            raise ValueError("SAMTok encode requires one or more non-empty binary masks")
        masks = torch.stack(masks)
        try:
            boxes = torchvision.ops.masks_to_boxes(masks)
            x_center = ((boxes[:, 0] + boxes[:, 2]) / 2).cpu().numpy()
            y_center = ((boxes[:, 1] + boxes[:, 3]) / 2).cpu().numpy()
            order = np.lexsort((y_center, x_center))
        except (RuntimeError, ValueError):
            order = np.arange(masks.shape[0])
        order_tensor = torch.as_tensor(order, dtype=torch.long)
        return masks[order_tensor], order.tolist()

    @torch.no_grad()
    def encode(self, pil_image, binary_masks) -> tuple[list[str], list[int]]:
        """Encode masks for one source image, sorted left-to-right/top-to-bottom."""

        masks, order = self._ordered_masks(binary_masks)
        width, height = pil_image.size
        for mask in masks:
            if tuple(mask.shape) != (height, width):
                raise ValueError(
                    f"Mask/image geometry mismatch: {tuple(mask.shape)} vs {(height, width)}"
                )
        boxes = torchvision.ops.masks_to_boxes(masks).to(torch.float32)
        boxes = boxes / torch.tensor([[width, height, width, height]])
        boxes = boxes.to(self.device)
        mask_list = [mask.unsqueeze(0).to(self.device) for mask in masks]
        output = self.vq(
            self._pixel_values([pil_image] * len(mask_list)),
            mask_list,
            boxes,
            reconstruct_mask=False,
        )
        codes = output.quant_codes.detach().cpu().reshape(len(mask_list), -1)
        if codes.shape[1] != CODEBOOK_DEPTH:
            raise RuntimeError(f"Unexpected quant code shape: {tuple(output.quant_codes.shape)}")
        spans = [
            span_of(
                [depth * CODEBOOK_SIZE + int(code) for depth, code in enumerate(row)]
            )
            for row in codes.tolist()
        ]
        return spans, order

    @torch.no_grad()
    def encode_single_batch(self, image_mask_pairs) -> list[str]:
        """Batch different source images when each row has exactly one mask."""

        pairs = list(image_mask_pairs)
        if not pairs:
            return []
        images, mask_list, boxes = [], [], []
        for image, binary_mask in pairs:
            mask = torch.from_numpy(
                np.ascontiguousarray(np.asarray(binary_mask) > 0)
            )
            width, height = image.size
            if tuple(mask.shape) != (height, width) or mask.sum() == 0:
                raise ValueError("Each batch mask must be non-empty and match its source image")
            box = torchvision.ops.masks_to_boxes(mask.unsqueeze(0)).to(torch.float32)
            box = box / torch.tensor([[width, height, width, height]])
            images.append(image)
            mask_list.append(mask.unsqueeze(0).to(self.device))
            boxes.append(box)
        output = self.vq(
            self._pixel_values(images),
            mask_list,
            torch.cat(boxes).to(self.device),
            reconstruct_mask=False,
        )
        codes = output.quant_codes.detach().cpu().reshape(len(pairs), -1)
        return [
            span_of(
                [depth * CODEBOOK_SIZE + int(code) for depth, code in enumerate(row)]
            )
            for row in codes.tolist()
        ]

    @torch.no_grad()
    def decode(self, pil_image, cot_text: str) -> list[np.ndarray]:
        """Decode valid spans for visualization or IoU evaluation."""

        pairs = SPAN_RE.findall(fix_mt_format_comprehensive(cot_text))
        pairs = [
            (int(first), int(second))
            for first, second in pairs
            if valid_span_codes(int(first), int(second))
        ]
        if not pairs:
            return []
        codes = [[first, second - CODEBOOK_SIZE] for first, second in pairs]
        masks = self.vq.forward_with_codes(
            self._pixel_values([pil_image] * len(codes)),
            torch.tensor(codes, dtype=torch.long, device=self.device),
        )
        width, height = pil_image.size
        masks = torch.nn.functional.interpolate(
            masks, size=(height, width), mode="bilinear", align_corners=False
        ) > 0.5
        return [mask for mask in masks[:, 0].cpu().numpy().astype(np.uint8)]
