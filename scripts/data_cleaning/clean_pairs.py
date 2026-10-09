#!/usr/bin/env python3
"""Clean the Stage 2 source pairs and write a self-contained dataset (docs/09, sections 6.1 and 6.7).

Decision per pair of sources.jsonl, in this order:
  drop_D     the region barely changes (fill < 0.10): no-op or too weak an edit
  keep       the target already matches the source outside the region (far_out <= 0.02) and has no solid
             change far outside it
  otherwise  register the target onto the source (SIFT + RANSAC affine on features outside the region) and
             use the registered target when it is clearly better aligned; then
  drop_C     a solid block of change larger than 1% of the image lies far outside the region
  drop_M     registration was not used and the target is not aligned with the source
  fix_*      paste the source back outside the dilated region (fix_resize / fix_register), unless the result
  drop_gate  still changes outside, lost the edit, cuts the edit at the paste boundary, shows a seam, or the
             pasted area is too large

Coordinates.  Every kept target has the size of its source and every region is in source pixels:
  - a target of another size is resized to the source size (what training always did);
  - with registration the target is warped into the source frame.  Instances grounded on the target
    (`mapped_from_target`) were stored as a plain resize into the source frame, so they move with the same
    transform, instance by instance; instances grounded on the source stay where they are.  A registration
    that would push more than 10% of such an instance out of the frame is not used;
  - each instance and the region union carry a COCO RLE mask and the outward-rounded 0-1000 box that
    samtok_edit21.data.protocol.pixel_box gives for that mask.

Images are hard links to the original files when unchanged and PNG files when resized or repaired.

usage: clean_pairs.py run --out DIR [--workers N] [--ids FILE] [--sources FILE]
       clean_pairs.py finalize --out DIR [--sources FILE]
"""
import argparse
import errno
import json
import math
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checks import alignment, coherent_outside  # noqa: E402

BASE = "/mnt/bn/strategy-mllm-train/user/tanyue"
SOURCES = f"{BASE}/experiments/SAMTokEdit/qwen21_full4_20260928/data/sources.jsonl"
WORK = 512            # long side for the pixel statistics
REG_SIDE = 768        # long side for feature matching
PASTE_DILATE = 0.02   # paste-back region: region dilated by 2% of the short side
RULES = {"fill_min": 0.10, "far_out_keep": 0.02, "blob_max": 0.01, "align_min": 0.4, "align_tiles": 5,
         "register_gain": 0.02, "register_cover": 0.98, "instance_area_min": 0.90}
GATE = {"far_out_final": 0.02, "fill_final": 0.10, "ring_cut": 0.30, "ring_mad": 25.0, "area_kept": 0.60}
KEPT = ("keep", "fix_resize", "fix_register")
EXT = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}
BOX_SCALE = 1000
# the subset datasets moved on 2026-10-09; locators recorded in sources.jsonl still carry the old places
LOCATOR_MOVES = (
    (f"{BASE}/datasets/RefEdit-mask-prefiltered-qwen38-self-contained/",
     f"{BASE}/datasets/RefEdit_Labeling/final/RefEdit-mask-prefiltered-qwen38-self-contained/"),
    (f"{BASE}/CrispEdit-labeling/", f"{BASE}/datasets/CrispEdit_Labeling/"),
    (f"{BASE}/scaleedit_25k/", f"{BASE}/datasets/ScaleEdit_Labeling/final/scaleedit_25k/"),
)
OUT = None
PNG_LEVEL = 3


def small(x, size, nearest=False):
    img = Image.fromarray(x.astype(np.uint8) if x.dtype != bool else x.astype(np.uint8) * 255)
    out = np.asarray(img.resize(size, Image.NEAREST if nearest else Image.BOX))
    return out > 0 if x.dtype == bool else out.astype(np.float32)


def work_size(shape):
    h, w = shape
    s = WORK / max(w, h)
    return max(1, round(w * s)), max(1, round(h * s))


def stats(src, tgt, z, valid=None):
    """far_out: share of clearly changed pixels beyond 4% of the long side from the region;
    in_share: share of all changed pixels that lie within that distance; fill: changed share of the region."""
    size = work_size(z.shape)
    a, b, zs = small(src, size), small(tgt, size), small(z, size, nearest=True)
    vs = small(valid, size, nearest=True) if valid is not None else np.ones(zs.shape, bool)
    changed = (np.abs(b - a).mean(axis=2) > 25) & vs
    near = ndimage.distance_transform_edt(~zs) <= 0.04 * max(size)
    far = ~near & vs
    return {"far_out": float(changed[far].mean()) if far.any() else 0.0,
            "in_share": float((changed & near).sum() / changed.sum()) if changed.any() else 1.0,
            "fill": float(changed[zs].mean()) if zs.any() else 0.0}


def register(src, tgt, z):
    """Affine from target pixels to source pixels, estimated outside the region; None if it fails."""
    sa = REG_SIDE / max(src.shape[:2])
    ta = REG_SIDE / max(tgt.shape[:2])
    a = cv2.resize(cv2.cvtColor(src, cv2.COLOR_RGB2GRAY), None, fx=sa, fy=sa, interpolation=cv2.INTER_AREA)
    t = cv2.resize(cv2.cvtColor(tgt, cv2.COLOR_RGB2GRAY), None, fx=ta, fy=ta, interpolation=cv2.INTER_AREA)
    zs = cv2.resize(z.astype(np.uint8), (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    keep = (ndimage.distance_transform_edt(~zs) > 0.06 * REG_SIDE).astype(np.uint8) * 255
    sift = cv2.SIFT_create(4000)
    ka, da = sift.detectAndCompute(a, keep)
    kb, db = sift.detectAndCompute(t, None)
    if da is None or db is None or len(ka) < 20 or len(kb) < 20:
        return None, "few_features"
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 12:
        return None, "few_matches"
    pa = np.float32([ka[m.queryIdx].pt for m in good]) / sa
    pb = np.float32([kb[m.trainIdx].pt for m in good]) / ta
    M, inl = cv2.estimateAffine2D(pb, pa, method=cv2.RANSAC, ransacReprojThreshold=2.0 / sa, maxIters=4000)
    if M is None or inl is None or inl.sum() < 12:
        return None, "no_transform"
    return M, f"ok:{int(inl.sum())}/{len(good)}"


def paste(src, tgt, z):
    """Target inside the region dilated by 2% of the short side (Gaussian edge), source elsewhere."""
    r = max(2, int(PASTE_DILATE * min(z.shape)))
    grown = ndimage.binary_dilation(z, iterations=r)
    alpha = ndimage.gaussian_filter(grown.astype(np.float32), sigma=2)[..., None]
    out = tgt.astype(np.float32) * alpha + src.astype(np.float32) * (1 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8), grown


def ring_metrics(src, tgt, grown):
    """Change in the band just outside the paste region, where a paste would cut the edit or leave a seam."""
    size = work_size(grown.shape)
    g = small(grown, size, nearest=True)
    band = ndimage.binary_dilation(g, iterations=max(2, int(0.02 * max(size)))) & ~g
    if not band.any():
        return 0.0, 0.0
    d = np.abs(small(tgt, size) - small(src, size)).mean(axis=2)
    return float((d[band] > 25).mean()), float(d[band].mean())


def pixel_box(mask):
    """Outward-rounded 0-1000 box of a nonempty mask (same as samtok_edit21.data.protocol.pixel_box)."""
    height, width = mask.shape
    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    x0, y0, x1, y1 = float(cols[0]), float(rows[0]), float(cols[-1] + 1), float(rows[-1] + 1)
    return [math.floor(BOX_SCALE * x0 / width), math.floor(BOX_SCALE * y0 / height),
            math.ceil(BOX_SCALE * x1 / width), math.ceil(BOX_SCALE * y1 / height)]


def rle_of(mask):
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode()


def decode_instances(rec, width, height):
    """[(instance record, bool mask in source pixels)] for the instances that carry a mask."""
    out = []
    for inst in rec.get("instances") or []:
        if not inst.get("rle_counts"):
            continue
        m = mask_utils.decode({"size": inst["rle_size"], "counts": inst["rle_counts"].encode()}).astype(bool)
        if m.shape != (height, width):
            m = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((width, height), Image.NEAREST)) > 0
        out.append((inst, m))
    return out


def annotations(insts, masks, moved, shape):
    """Instance and union annotations in source pixels (masks after any move)."""
    h, w = shape
    items = []
    for (inst, original), mask, was_moved in zip(insts, masks, moved):
        item = {k: inst[k] for k in ("instance_id", "ref", "grounding_image", "mapped_from_target", "edit_id", "change_id")
                if k in inst}
        keep_rle = not was_moved and list(inst["rle_size"]) == [h, w]
        item.update(moved=bool(was_moved), rle_size=[h, w],
                    rle_counts=inst["rle_counts"] if keep_rle else rle_of(mask), box_1000=pixel_box(mask))
        items.append(item)
    union = np.logical_or.reduce(masks)
    return items, {"rle_size": [h, w], "rle_counts": rle_of(union), "box_1000": pixel_box(union),
                   "area": float(union.mean())}


def link_file(original, dst):
    """Hard link (no extra space); a re-run finds the same file already in place.  Copies across filesystems."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.link(original, dst)
    except FileExistsError:
        if not os.path.samefile(original, dst) and os.stat(original).st_dev == os.stat(dst).st_dev:
            raise
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        tmp = f"{dst}.tmp{os.getpid()}"
        shutil.copyfile(original, tmp)
        os.replace(tmp, dst)


def save_png(array, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = f"{dst}.tmp{os.getpid()}"
    Image.fromarray(array).save(tmp, format="PNG", compress_level=PNG_LEVEL)
    os.replace(tmp, dst)


def write_pair(rec, res, src_fmt, tgt_fmt, target_array, insts, masks, moved, shape):
    """Materialise a kept pair: images under OUT and the annotations in source pixels."""
    stem = f"images/{rec['dataset']}/{rec['id'][-2:]}/{rec['id']}"
    source_rel = f"{stem}.source{EXT.get(src_fmt, '.img')}"
    link_file(rec["edit_image"], f"{OUT}/{source_rel}")
    if target_array is None:
        target_rel = f"{stem}.target{EXT.get(tgt_fmt, '.img')}"
        link_file(rec["image"], f"{OUT}/{target_rel}")
    else:
        target_rel = f"{stem}.target.png"
        save_png(target_array, f"{OUT}/{target_rel}")
    items, union = annotations(insts, masks, moved, shape)
    res.update(source_image=source_rel, target_image=target_rel, size=[shape[1], shape[0]], instances=items, region=union)
    return res


def clean(rec):
    res = {"id": rec["id"], "dataset": rec["dataset"], "type": rec.get("provisional_type")}
    with Image.open(rec["edit_image"]) as im:
        src_fmt, src = im.format, np.asarray(im.convert("RGB"))
    with Image.open(rec["image"]) as im:
        tgt_fmt, tgt_orig = im.format, np.asarray(im.convert("RGB"))
    H, W = src.shape[:2]
    th, tw = tgt_orig.shape[:2]
    same_size = (th, tw) == (H, W)
    tgt_base = tgt_orig if same_size else np.asarray(Image.fromarray(tgt_orig).resize((W, H), Image.BICUBIC))
    insts = decode_instances(rec, W, H)
    originals = [m for _, m in insts]
    if not insts or not all(m.any() for m in originals):
        res.update(decision="skip_no_region", reason="no usable instance mask")
        return res
    z = np.logical_or.reduce(originals)
    still = [False] * len(insts)
    base = stats(src, tgt_base, z)
    res["before"] = base
    if base["fill"] < RULES["fill_min"]:
        res.update(decision="drop_D", reason="no-op or too weak (fill < 0.10)")
        return res
    if base["far_out"] <= RULES["far_out_keep"]:
        res["blob"] = coherent_outside(src, tgt_base, z)
        if res["blob"] > RULES["blob_max"]:
            res.update(decision="drop_C", reason=f"solid change far outside the region ({res['blob']:.1%} of the image)")
            return res
        res.update(decision="keep", treatment="link" if same_size else "resize", after=base)
        return write_pair(rec, res, src_fmt, tgt_fmt, None if same_size else tgt_base, insts, originals, still, (H, W))
    chosen, region, how, valid_reg, masks, moved = tgt_base, z, "resize", None, originals, still
    M, status = register(src, tgt_orig, z)
    res["register"] = {"status": status}
    if M is not None:
        warped = cv2.warpAffine(tgt_orig, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        valid = cv2.warpAffine(np.full((th, tw), 255, np.uint8), M, (W, H), flags=cv2.INTER_NEAREST) > 0
        # target-grounded instances sit in the source frame through a plain resize; undo it and apply M
        A = M.copy()
        A[:, 0] *= tw / W
        A[:, 1] *= th / H
        flags = [bool(inst.get("mapped_from_target")) for inst, _ in insts]
        shifted = [cv2.warpAffine(m.astype(np.uint8), A, (W, H), flags=cv2.INTER_NEAREST) > 0 if f else m
                   for m, f in zip(originals, flags)]
        zr = np.logical_or.reduce(shifted)
        reg = stats(src, warped, zr, valid)
        covered = float(valid[zr].mean()) if zr.any() else 0.0
        # a moved instance must stay inside the frame, or the edit it marks is cut by the image border
        area_scale = abs(float(np.linalg.det(A[:, :2])))
        inside = min((float(n.sum()) / (float(m.sum()) * area_scale) for m, n, f in zip(originals, shifted, flags) if f),
                     default=1.0)
        res["register"].update(reg, region_covered=covered, instance_area_kept=inside,
                               scale=float(np.sqrt(abs(np.linalg.det(M[:, :2])))),
                               shift=float(np.hypot(M[0, 2], M[1, 2])), matrix=[[float(v) for v in row] for row in M])
        if (reg["far_out"] < base["far_out"] - RULES["register_gain"] and covered >= RULES["register_cover"]
                and inside >= RULES["instance_area_min"] and all(m.any() for m in shifted)):
            chosen, region, how, valid_reg, masks, moved = warped, zr, "register", valid, shifted, flags
    res["blob"] = coherent_outside(src, chosen, region, valid=valid_reg)
    if res["blob"] > RULES["blob_max"]:
        res.update(decision="drop_C", reason=f"solid change far outside the region after {how} ({res['blob']:.1%} of the image)")
        return res
    if how == "resize":  # registration not used: pasting is only safe when the target is already aligned
        share, tiles = alignment(src, chosen, region)
        res.update(aligned=None if tiles == 0 else share, align_tiles=tiles)
        if tiles >= RULES["align_tiles"] and share < RULES["align_min"]:
            res.update(decision="drop_M", reason=f"target not aligned with the source and registration not usable "
                                                 f"({share:.0%} of {tiles} tiles in place)")
            return res
    pasted, grown = paste(src, chosen, region)
    cut, mad = ring_metrics(src, chosen, grown)
    final = stats(src, pasted, region)
    res.update(treatment=how + "+paste", after=final, ring_cut=cut, ring_mad=mad, area_kept=float(grown.mean()))
    fails = []
    if final["far_out"] > GATE["far_out_final"]:
        fails.append("outside still changed")
    if final["fill"] < GATE["fill_final"]:
        fails.append("edit lost after paste")
    if cut > GATE["ring_cut"]:
        fails.append("edit cut at the paste boundary")
    if mad > GATE["ring_mad"]:
        fails.append("seam at the paste boundary")
    if grown.mean() > GATE["area_kept"]:
        fails.append("region too large")
    if fails:
        res.update(decision="drop_gate", reason="; ".join(fails))
        return res
    res["decision"] = "fix_" + how
    return write_pair(rec, res, src_fmt, tgt_fmt, pasted, insts, masks, moved, (H, W))


def process(line):
    rec = json.loads(line)
    try:
        return clean(rec)
    except Exception as error:  # noqa: BLE001  one unreadable pair must not stop the run
        return {"id": rec["id"], "dataset": rec["dataset"], "type": rec.get("provisional_type"),
                "decision": "error", "reason": f"{type(error).__name__}: {error}"}


def _init(out, png_level):
    global OUT, PNG_LEVEL
    OUT, PNG_LEVEL = out, png_level
    cv2.setNumThreads(1)


def line_id(line):
    return line[8:80].split('"')[0]


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    decisions = out / "decisions.jsonl"
    done = set()
    if decisions.exists():
        with decisions.open() as stream:
            done = {json.loads(line)["id"] for line in stream}
    wanted = set(Path(args.ids).read_text().split()) if args.ids else None
    total, pending = 0, []
    with open(args.sources) as stream:
        for line in stream:
            rid = line_id(line)
            if wanted is not None and rid not in wanted:
                continue
            total += 1
            if rid not in done:
                pending.append(line)
    print(f"[{time.strftime('%H:%M:%S')}] {total} pairs, {len(done)} already done, {len(pending)} to process, "
          f"{args.workers} workers -> {out}", flush=True)
    counts, start = Counter(), time.time()
    with Pool(args.workers, initializer=_init, initargs=(str(out), args.png_level)) as pool, decisions.open("a") as stream:
        for k, res in enumerate(pool.imap_unordered(process, pending, chunksize=4), 1):
            stream.write(json.dumps(res, ensure_ascii=False) + "\n")
            counts[res["decision"]] += 1
            if k % 500 == 0 or k == len(pending):
                stream.flush()
            if k % args.log_every == 0 or k == len(pending):
                rate = k / (time.time() - start)
                print(f"[{time.strftime('%H:%M:%S')}] {len(done) + k}/{total} ({(len(done) + k) / total:.1%}) "
                      f"{rate:.1f} pairs/s, ETA {(len(pending) - k) / rate / 60:.1f} min | "
                      + " ".join(f"{d}={n}" for d, n in sorted(counts.items())), flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] processing finished in {(time.time() - start) / 60:.1f} min", flush=True)
    finalize(args)


def moved_locator(locator):
    locator = dict(locator or {})
    path = locator.get("file")
    for old, new in LOCATOR_MOVES:
        if path and path.startswith(old):
            locator["file"] = new + path[len(old):]
    return locator


def finalize(args):
    """manifest.jsonl (kept pairs, in sources order) and summary.json; checks that every listed file exists."""
    out = Path(args.out)
    results = {}
    with (out / "decisions.jsonl").open() as stream:
        for line in stream:
            res = json.loads(line)
            results[res["id"]] = res
    table = defaultdict(Counter)
    kept = 0
    files = []
    worst = {"far_out": 0.0, "fill": 1.0}
    locator_files = set()
    with open(args.sources) as stream, (out / "manifest.jsonl").open("w") as manifest:
        for line in stream:
            res = results.get(line_id(line))
            if res is None:
                continue
            table[f"{res['dataset']}/{res['type']}"][res["decision"]] += 1
            if res["decision"] not in KEPT:
                continue
            rec = json.loads(line)
            files.extend(str(out / res[key]) for key in ("source_image", "target_image"))
            worst["far_out"] = max(worst["far_out"], res["after"]["far_out"])
            worst["fill"] = min(worst["fill"], res["after"]["fill"])
            locator = moved_locator(rec.get("locator"))
            locator_files.add(locator.get("file"))
            row = {"id": rec["id"], "dataset": rec["dataset"], "type": rec.get("provisional_type"),
                   "native_type": rec.get("native_type"), "instruction": rec["instruction"],
                   "source_image": res["source_image"], "target_image": res["target_image"], "size": res["size"],
                   "decision": res["decision"], "treatment": res["treatment"],
                   "instances": res["instances"], "region": res["region"],
                   "metrics": {k: res[k] for k in ("before", "after", "blob", "aligned", "align_tiles", "ring_cut", "ring_mad",
                                                   "area_kept", "register") if k in res},
                   "provenance": {"locator": locator, "original_source": rec["edit_image"], "original_target": rec["image"],
                                  "original_source_size": rec.get("source_size"), "original_target_size": rec.get("target_size"),
                                  "split_group": rec.get("split_group"), "quality_accepted": rec.get("quality_accepted"),
                                  "training_ready": rec.get("training_ready")}}
            manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
    with ThreadPoolExecutor(64) as pool:  # one stat per file is slow on the shared filesystem
        missing = sum(not found for found in pool.map(os.path.isfile, files))
        locator_missing = sum(not found for found in pool.map(lambda f: bool(f) and os.path.isfile(f), locator_files))
    decisions = Counter()
    for counter in table.values():
        decisions.update(counter)
    summary = {"sources": args.sources, "pairs": len(results), "kept": kept, "decisions": dict(decisions),
               "by_dataset_type": {k: dict(v) for k, v in sorted(table.items())},
               "rules": RULES, "gate": GATE, "missing_files": missing, "worst_kept": worst,
               "locator_files": len(locator_files),
               "locator_files_missing": locator_missing}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n")
    print(f"[{time.strftime('%H:%M:%S')}] finalize: {len(results)} pairs, kept {kept}, missing files {missing}, "
          f"worst kept far_out {worst['far_out']:.4f} fill {worst['fill']:.4f}, "
          f"locator files {summary['locator_files']} (missing {summary['locator_files_missing']}) | "
          + " ".join(f"{d}={n}" for d, n in sorted(decisions.items())), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("run", "finalize"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--sources", default=SOURCES)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--ids", help="file with the pair ids to process (default: all)")
    parser.add_argument("--png-level", type=int, default=3)
    parser.add_argument("--log-every", type=int, default=500, help="pairs between progress lines")
    args = parser.parse_args()
    (run if args.command == "run" else finalize)(args)


if __name__ == "__main__":
    main()
