"""Encode the 72 publicly supplied S2/S3 masks and offline round-trip references."""

import argparse

import numpy as np
import torch
from common import ROOT, json, read_jsonl, rle, sha256
from PIL import Image

from samtok_edit21.models.codec import load_codec
from samtok_edit21.pass1.evaluation.scoring import decode_stable, iou
from samtok_edit21.pass1.model import MODEL
from samtok_edit21.pass1.protocol import valid_span


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    args = ap.parse_args()
    rows = read_jsonl(ROOT / "data/offline_gt.jsonl")
    public_keys = {
        (p["source"]["sha256"], u["region"]["sha256"])
        for p in read_jsonl(ROOT / "data/benchmark_public.jsonl")
        for u in p["units"]
        if u["locator"] == "mask"
    }
    codec = load_codec(MODEL, "cuda:0")
    out = ROOT / f"data/codec_{args.shard}.jsonl"
    with out.open("x") as f, torch.inference_mode():
        for row in rows[args.shard :: args.shards]:
            with Image.open(row["image"]) as im:
                image = im.convert("RGBA")
            masks = []
            for u in row["units"]:
                assert (row["image_sha256"], u["mask_sha256"]) in public_keys
                assert sha256(u["mask_path"]) == u["mask_sha256"]
                with Image.open(u["mask_path"]) as im:
                    masks.append(np.asarray(im.convert("L")) > 0)
            spans = [codec.encode(image, [mask])[0][0] for mask in masks]
            assert all(valid_span(s) for s in spans)
            decoded = decode_stable(codec, image, spans)
            units = [
                dict(u, span=s, canonical_dec=rle(d), roundtrip=iou(d, mask))
                for u, s, d, mask in zip(row["units"], spans, decoded, masks)
            ]
            result = dict(row, units=units)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            print(
                json.dumps(
                    {"case": row["id"], "roundtrip": [u["roundtrip"] for u in units]}
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
