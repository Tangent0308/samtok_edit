"""Decode already-generated tokens for evaluation. GT never enters generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from samtok_edit21.pass1.data.pilot import decode_rle
from samtok_edit21.pass1.model import MODEL
from samtok_edit21.pass1.protocol import point_pixels


def iou(a, b):
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def decode_stable(codec, image, spans):
    """Native batch-one decoder, independent of how many different codes a model returns.

    Reuse only the same-image frozen features. This avoids batch-shape-dependent
    BF16 rounding differences between baseline and adapter evaluations.
    """
    backend = codec.vq.model
    original = backend.get_sam2_embeddings
    cache = {}

    def cached(images, expand_size=1):
        if images.shape[0] != 1 or expand_size != 1:
            raise ValueError("Expected one image")
        if not cache:
            cache["pixels"] = images.clone()
            cache["states"] = original(images, expand_size=1)
        elif not torch.equal(images, cache["pixels"]):
            raise ValueError("Feature cache crossed an image boundary")
        return cache["states"]

    backend.get_sam2_embeddings = cached
    try:
        return [codec.decode_strict(image, span)[0] for span in spans]
    finally:
        backend.get_sam2_embeddings = original


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    args = ap.parse_args()
    path = Path(args.data)
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    preds = {
        r["id"]: r
        for r in (
            json.loads(x) for x in Path(args.predictions).read_text().splitlines()
        )
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    masks_dir = out.parent / "masks"
    masks_dir.mkdir(exist_ok=True)
    from samtok_edit21.models.codec import load_codec

    codec = load_codec(MODEL, "cuda:0")
    with out.open("w") as f:
        for row in rows[args.shard :: args.shards]:
            pred = preds[row["id"]]
            # Separate syntax compliance from localization. Compatibility only
            # strips the fixed empty-think prefix / Markdown fence, never labels,
            # token values, ordering, multiplicity, or extra object fields.
            spans = pred["compatible"]["spans"]
            expected = {t["label"]: t["span"] for t in row["truth"]}
            all_spans = list(
                dict.fromkeys(list(spans.values()) + list(expected.values()))
            )
            with Image.open(path.parent / row["image"]) as im:
                # Strict native decoder, no malformed-span repair.
                masks = decode_stable(codec, im.convert("RGBA"), all_spans)
            decoded = dict(zip(all_spans, masks))
            gt = {t["label"]: decode_rle(t["selection"]) for t in row["truth"]}
            records = []
            for target, truth in zip(row["request"]["targets"], row["truth"]):
                uid = target["label"]
                token = spans.get(uid)
                mask = decoded.get(token, np.zeros_like(gt[uid]))
                ref = decoded[expected[uid]]
                raw_iou = iou(mask, gt[uid])
                reference_iou = iou(ref, gt[uid])
                overlap = float(np.logical_and(mask, gt[uid]).sum())
                pred_area = float(mask.sum())
                gt_area = float(gt[uid].sum())
                other = {k: iou(mask, m) for k, m in gt.items() if k != uid}
                mask_path = masks_dir / (row["id"] + "_" + uid + ".png")
                Image.fromarray((mask.astype(np.uint8) * 255)).save(mask_path)
                rec = {
                    "id": row["id"],
                    "group": row["group"],
                    "label": uid,
                    "K": len(row["truth"]),
                    "locator": target["locator"],
                    "has_ref": target["has_ref"],
                    "mode": ("ref" if target["has_ref"] else "noref")
                    + "_"
                    + target["locator"],
                    "family": truth["family"],
                    "operation": truth["operation"],
                    "area": truth["area"],
                    "valid_id_span": token is not None,
                    "request_strict": pred["strict"]["complete"],
                    "iou": raw_iou,
                    "strict_iou": raw_iou if pred["strict"]["complete"] else 0.0,
                    "compatible_iou": raw_iou
                    if pred["compatible"]["complete"]
                    else 0.0,
                    "precision": overlap / max(1.0, pred_area),
                    "recall": overlap / max(1.0, gt_area),
                    "predicted_area": float(mask.mean()),
                    "area_ratio": pred_area / max(1.0, gt_area),
                    "canonical_iou": reference_iou,
                    "canonical_cached_agreement": iou(
                        ref, decode_rle(truth["canonical_dec"])
                    ),
                    "canonical_cached_equal": bool(
                        np.array_equal(ref > 0, decode_rle(truth["canonical_dec"]))
                    ),
                    "exact_code": token == expected[uid],
                    "mask_path": str(mask_path),
                    "possible_id_swap": max(other.values(), default=0)
                    > max(0.5, raw_iou + 0.2),
                    "best_other_target_iou": max(other.values(), default=0),
                    "predicted_span": token,
                    "expected_span": expected[uid],
                }
                if target["locator"] == "point":
                    x, y = point_pixels(target["region"]["point"], row["size"])
                    rec["point_hit"] = bool(mask[y, x])
                records.append(rec)
            result = {
                "id": row["id"],
                "group": row["group"],
                "strict": pred["strict"]["complete"],
                "compatible": pred["compatible"]["complete"],
                "errors": pred["strict"]["errors"],
                "all_iou50": all(r["strict_iou"] >= 0.5 for r in records),
                "compatible_all_iou50": all(
                    r["compatible_iou"] >= 0.5 for r in records
                ),
                "targets": records,
            }
            f.write(json.dumps(result) + "\n")
            f.flush()
            print(
                json.dumps(
                    {
                        "id": row["id"],
                        "strict": result["strict"],
                        "mean_iou": np.mean([r["iou"] for r in records]),
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
