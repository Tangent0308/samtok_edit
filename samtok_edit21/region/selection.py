"""SAM2 point/box candidates, decoded SAMTok previews and explicit selection."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..schema.data import write_json


@torch.no_grad()
def segment(codec, image, *, points=(), box=None):
    """Coordinates are source-image pixels; point triples are x,y,label (0/1)."""
    coords, labels = [], []
    if box is not None:
        x0, y0, x1, y1 = map(float, box)
        if not (0 <= x0 < x1 <= image.width and 0 <= y0 < y1 <= image.height):
            raise ValueError("Box must be xyxy inside the source image")
        coords.extend([[x0, y0], [x1, y1]])
        labels.extend([2, 3])
    for x, y, label in points:
        if label not in (0, 1) or not (0 <= x < image.width and 0 <= y < image.height):
            raise ValueError("Points must be x,y,0|1 inside the source image")
        coords.append([x, y])
        labels.append(label)
    if not coords:
        raise ValueError("Provide at least one point or one box")
    # VQ training modifies SAM2's mask decoder. Its point/box quality and IoU
    # scores need not remain calibrated. Use the original SAM2.1 weights for
    # proposals, and the released VQ checkpoint only for mask<->token conversion.
    if not hasattr(codec, "prompt_sam"):
        from samtok.models.sam2 import SAM2Model, SAM2Config

        codec.prompt_sam = (
            SAM2Model(SAM2Config(ckpt_path=codec.sam2_ckpt)).to(codec.device).eval()
        )
    wrapper = codec.prompt_sam
    pixels = codec._pixel_values([image])
    states = wrapper.get_sam2_embeddings(wrapper.preprocess_image(pixels))
    feats, sizes = states["current_vision_feats"], states["feat_sizes"]
    highres = [
        f.permute(1, 2, 0).view(f.size(1), f.size(2), *s)
        for f, s in zip(feats[:-1], sizes[:-1])
    ]
    backbone = (
        (feats[-1] + wrapper.sam2_model.no_mem_embed)
        .permute(1, 2, 0)
        .view(1, wrapper.hidden_dim, *sizes[-1])
    )
    xy = torch.tensor([coords], device=codec.device, dtype=torch.float32)
    xy *= torch.tensor([1024 / image.width, 1024 / image.height], device=codec.device)
    tags = torch.tensor([labels], device=codec.device, dtype=torch.int32)
    with torch.autocast(device_type=codec.device.type, dtype=torch.bfloat16):
        outputs = wrapper.sam2_model._forward_sam_heads(
            backbone_features=backbone,
            point_inputs={"point_coords": xy, "point_labels": tags},
            mask_inputs=None,
            high_res_features=highres,
            multimask_output=True,
        )
    masks = (
        torch.nn.functional.interpolate(
            outputs[1].float(),
            size=(image.height, image.width),
            mode="bilinear",
            align_corners=False,
        )[0]
        > 0
    )
    return masks.cpu().numpy(), outputs[2][0].float().cpu().tolist()


def save_candidates(codec, image, masks, scores, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for i, (mask, score) in enumerate(zip(masks, scores)):
        name = f"candidate-{i:02d}.png"
        Image.fromarray(np.uint8(mask) * 255).save(directory / name)
        codes = codec.encode(image, [mask])[0] if mask.any() else []
        records.append(
            {
                "index": i,
                "mask": name,
                "sam2_iou_score": score,
                "mask_codes": codes,
                "area_fraction": float(mask.mean()),
            }
        )
    valid = [r for r in records if r["mask_codes"]]
    if not valid:
        raise ValueError("SAM2 produced no nonempty candidate")
    selected = max(valid, key=lambda r: r["sam2_iou_score"])["index"]
    write_json(
        directory / "candidates.json",
        {"candidates": records, "selected_index": selected},
    )
    return records, selected


def decode_localizations(codec, image, results, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for i, result in enumerate(results):
        if result["fallback_reason"]:
            continue
        masks = codec.decode(image, result["raw"])
        files = []
        for j, mask in enumerate(masks):
            path = directory / f"hypothesis-{i:02d}-target-{j:02d}.png"
            Image.fromarray(np.uint8(mask) * 255).save(path)
            files.append(str(path))
        result["decoded_masks"] = files
    return results
