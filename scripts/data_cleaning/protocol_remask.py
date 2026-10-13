"""Re-mask the source object S where the labels do not cover it (protocol 1.0 adaptation).

S is segmented exactly as the CrispEdit labeling repo segments a grounded source object:
the same SAM3 checkpoint and processor threshold, the same context crop for boxes below 65 %
of the image, and crispedit.mask.pipeline.segment_grounded_box with the box from the
model's hinted grounding call and pa.sam_prompt (the ref without its article; for add the whole object or
surface S belongs to, which protocol_adapt.localize_anchor cuts to the spot) as the text prompt.

Run it in an environment with torch and the SAM3 dependencies, with the labeling repo on
--labeling-repo (imported read-only; set PYTHONDONTWRITEBYTECODE=1).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image

from samtok_edit21.preparation import edit_check as ec
from samtok_edit21.preparation import protocol_adapt as pa

CHECKPOINT = "/mnt/bn/strategy-mllm-train/common/models/sam3/sam3.pt"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rows", required=True)
    p.add_argument("--slots", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--labeling-repo", default="/opt/tiger/tanyue/sam3-crispedit-labeling")
    p.add_argument("--checkpoint", default=CHECKPOINT)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    sys.path.insert(0, args.labeling_repo)
    from crispedit.mask.pipeline import _box_area, normalized_box_to_pixels, segment_grounded_box
    from crispedit.mask.regions import context_crop
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    labeling_commit = subprocess.run(["git", "-C", args.labeling_repo, "rev-parse", "HEAD"],
                                     capture_output=True, text=True).stdout.strip()
    rows = {r["id"]: r for r in pa.read_jsonl(args.rows)}
    jobs = [e for e in ec.completed_records(args.slots).values() if (e.get("plan") or {}).get("source") == "sam3"]
    done = ec.completed_records(args.output)
    jobs = [e for e in jobs if e["id"] not in done]
    if not jobs:
        print("nothing to re-mask")
        return
    model = build_sam3_image_model(device=args.device, checkpoint_path=args.checkpoint, load_from_HF=False,
                                   enable_inst_interactivity=True)
    processor = Sam3Processor(model, device=args.device, confidence_threshold=0.3)
    with open(args.output, "a") as stream:
        for extracted in jobs:
            row, start = rows[extracted["id"]], time.monotonic()
            box = (extracted.get("hinted") or {}).get("bbox_2d")
            prompt = extracted.get("sam_prompt") or pa.sam_prompt(extracted["spec"])
            record = {"id": row["id"], "box_2d": box, "sam_prompt": prompt, "labeling_commit": labeling_commit,
                      "checkpoint": args.checkpoint, "mask": None}
            if box is None:
                record["error"] = "no hinted grounding box"
            else:
                image = Image.open(Path(args.root) / row["source_image"]).convert("RGB")
                shape = (image.height, image.width)
                raw = normalized_box_to_pixels(box, shape)
                local_shape, local_box, crop = shape, box, None
                if _box_area(raw) / (shape[0] * shape[1]) < 0.65:  # as crispedit.mask.pipeline._segment_items
                    crop = context_crop(raw, shape)
                    x1, y1, x2, y2 = crop
                    local_shape = (y2 - y1, x2 - x1)
                    local_box = [(raw[0] - x1) * 1000 / (x2 - x1), (raw[1] - y1) * 1000 / (y2 - y1),
                                 (raw[2] - x1) * 1000 / (x2 - x1), (raw[3] - y1) * 1000 / (y2 - y1)]
                    state = processor.set_image(image.crop(crop))
                else:
                    state = processor.set_image(image)
                mask, meta = segment_grounded_box(processor, state, prompt, local_box, local_shape, edit_type=None)
                if crop is not None:
                    full = np.zeros(shape, dtype=np.uint8)
                    full[crop[1]:crop[3], crop[0]:crop[2]] = mask
                    mask = full
                keep = {k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool)) or v is None}
                record.update(meta=keep, crop_xyxy=list(crop) if crop else [])
                if mask.any():
                    record["mask"] = pa.encode(mask.astype(bool))
                else:
                    record["error"] = "empty mask"
            record["seconds"] = round(time.monotonic() - start, 2)
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"{row['id']}: {'ok' if record['mask'] else record.get('error')} {record['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
