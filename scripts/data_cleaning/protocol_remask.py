"""Re-mask the source object S where the labels do not cover it (protocol 1.0 adaptation).

S is segmented exactly as the CrispEdit labeling repo segments a grounded source object:
the same SAM3 checkpoint and processor threshold, the same context crop for boxes below 65 %
of the image, and crispedit.mask.pipeline.segment_grounded_box with the box from the
model's hinted grounding call and pa.sam_prompt (the ref without its article; for add the whole object or
surface S belongs to, which protocol_adapt.localize_anchor cuts to the spot) as the text prompt.

For a part-level add the host (re-masked as above, or the reused label) is then searched for the part at the
new item: SAM's interactive predictor with the contact point (protocol_adapt.contact_point), with the model's part
box, and with both, three masks each, on a context crop around them. The candidates are stored; the build step
chooses one (protocol_adapt.choose_part).

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


def part_candidates(processor, image, host, footprint, part_box, contact, helpers):
    """SAM's masks for the part of the host where the new item meets it, in full-image coordinates.
    contact: (relation, host scope), which say which band of the new item meets the host."""
    context_crop, normalized_box_to_pixels, resize_mask = helpers
    point = pa.contact_point(host, footprint, *contact)
    if point is None:
        return {"error": "new item not at the host"}
    shape = host.shape
    boxes = [normalized_box_to_pixels(pa.box_1000(footprint), shape), [point[0], point[1], point[0] + 1, point[1] + 1]]
    box = normalized_box_to_pixels(part_box, shape) if part_box else None
    if box is not None:
        boxes.append(box)
    union = [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]
    x1, y1, x2, y2 = context_crop(union, shape)
    state = processor.set_image(image.crop((x1, y1, x2, y2)))
    local_point = np.asarray([[point[0] - x1, point[1] - y1]], dtype=np.float32)
    prompts = [("point", {"point_coords": local_point, "point_labels": np.asarray([1], dtype=np.int32)})]
    if box is not None:
        local_box = np.asarray([box[0] - x1, box[1] - y1, box[2] - x1, box[3] - y1], dtype=np.float32)
        prompts += [("box", {"box": local_box}),
                    ("point+box", {"point_coords": local_point, "point_labels": np.asarray([1], dtype=np.int32),
                                   "box": local_box})]
    out = []
    for name, kwargs in prompts:
        masks, scores, _ = processor.model.predict_inst(inference_state=state, multimask_output=True, **kwargs)
        for k in range(len(masks)):
            mask = (np.asarray(masks[k]) > 0).astype(np.uint8)
            if mask.shape != (y2 - y1, x2 - x1):
                mask = resize_mask(mask, (y2 - y1, x2 - x1))
            full = np.zeros(shape, dtype=bool)
            full[y1:y2, x1:x2] = mask > 0
            if full.any():
                score = round(float(np.asarray(scores[k]).item()), 4)
                out.append({**pa.encode(full), "prompt": name, "k": k, "score": score})
    return {"point": point, "crop_xyxy": [x1, y1, x2, y2], "candidates": out}


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
    from crispedit.mask.pipeline import _box_area, normalized_box_to_pixels, resize_mask, segment_grounded_box
    from crispedit.mask.regions import context_crop
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    labeling_commit = subprocess.run(["git", "-C", args.labeling_repo, "rev-parse", "HEAD"],
                                     capture_output=True, text=True).stdout.strip()
    rows = {r["id"]: r for r in pa.read_jsonl(args.rows)}
    jobs = [e for e in ec.completed_records(args.slots).values()
            if (e.get("plan") or {}).get("source") == "sam3" or (e.get("plan") or {}).get("part")]
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
            plan = extracted["plan"]
            box = (extracted.get("hinted") or {}).get("bbox_2d")
            prompt = extracted.get("sam_prompt") or pa.sam_prompt(extracted["spec"])
            record = {"id": row["id"], "box_2d": box, "sam_prompt": prompt, "labeling_commit": labeling_commit,
                      "checkpoint": args.checkpoint, "mask": None}
            image = Image.open(Path(args.root) / row["source_image"]).convert("RGB")
            host = None
            if plan["source"] != "sam3":
                host = pa.decode(pa.union([row["instances"][n - 1] for n in plan["instances"]]))
            elif box is None:
                record["error"] = "no hinted grounding box"
            else:
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
                    host = mask.astype(bool)
                else:
                    record["error"] = "empty mask"
            if plan.get("part") and host is not None and host.any():
                exec_mask = pa.decode(row["region"]) if row["dataset"] != "derived" else None
                images = pa.open_pair(row, args.root)
                item_box = (extracted.get("item") or {}).get("bbox_2d")
                footprint, kind = pa.add_footprint(exec_mask, host, images, item_box)
                record["part"] = ({"error": "new item not found"} if footprint is None else
                                  {"footprint": kind, **part_candidates(processor, image, host, footprint,
                                                                        (extracted.get("part") or {}).get("bbox_2d"),
                                                                        (extracted["spec"]["placement"].get("relation", ""),
                                                                         extracted["slots"].get("host_scope", "")),
                                                                        (context_crop, normalized_box_to_pixels,
                                                                         resize_mask))})
            record["seconds"] = round(time.monotonic() - start, 2)
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            part = record.get("part") or {}
            host_state = "ok" if record["mask"] or plan["source"] != "sam3" else record.get("error")
            part_state = (f", part candidates {len(part.get('candidates', []))} {part.get('error', '')}"
                          if plan.get("part") else "")
            print(f"{row['id']}: host {host_state}{part_state} {record['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
