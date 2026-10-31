"""Render existing benchmark predictions; no inference, editing, or data mutation."""

import csv
import json
import shutil
import sys

import numpy as np
from common import PREVIEW, ROOT, read_jsonl, sha256, write_json
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_dilation, binary_erosion

from samtok_edit21.pass1.data.pilot import decode_rle

OUT = ROOT / "reports/before_after_20261031"
VIEW = PREVIEW / "before_after"
W, CW, RH, HEADER = 1680, 420, 528, 144
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
GT = (45, 199, 100)
BEFORE = (242, 150, 43)
AFTER = (0, 185, 230)
RED = (234, 66, 73)


def font(n, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, n)


def wrap_pixels(text, f, maxwidth):
    words = text.split()
    lines = []
    current = ""
    for word in words:
        candidate = (current + " " + word).strip()
        if current and f.getlength(candidate) > maxwidth:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def resized_mask(mask, size):
    return (
        np.asarray(
            Image.fromarray(mask.astype(np.uint8) * 255).resize(
                size, Image.Resampling.NEAREST
            )
        )
        > 0
    )


def panel(source, gt, pred, unit, kind, size, crop=None):
    """All geometry changes below are solely for plotting, never model inputs."""
    box = crop or (0, 0, source.width, source.height)
    source = source.crop(box)
    gm = gt[box[1] : box[3], box[0] : box[2]]
    pm = None if pred is None else pred[box[1] : box[3], box[0] : box[2]]
    scale = min(size[0] / source.width, size[1] / source.height)
    actual = (max(1, round(source.width * scale)), max(1, round(source.height * scale)))
    arr = np.asarray(source.resize(actual, Image.Resampling.LANCZOS)).copy()
    g = resized_mask(gm, actual)
    mask = (
        g
        if kind in {"gt", "input_mask"}
        else (resized_mask(pm, actual) if pm is not None else None)
    )
    color = {"gt": GT, "before": BEFORE, "after": AFTER, "input_mask": RED}.get(kind)
    if mask is not None:
        arr[mask] = (0.62 * arr[mask] + 0.38 * np.array(color)).astype(np.uint8)
        edge = binary_dilation(mask) & ~binary_erosion(mask)
        arr[edge] = color
    if kind in {"before", "after"}:
        # Shared green reference contour makes false inclusion/exclusion readable.
        edge = binary_dilation(g) & ~g
        arr[edge] = GT
    image = Image.fromarray(arr)
    d = ImageDraw.Draw(image)
    if kind.startswith("input"):
        loc = unit["locator"]
        if loc == "point":
            x, y = unit["region"]["coordinates"]
            x = (x - box[0]) * scale
            y = (y - box[1]) * scale
            rr = 5
            d.ellipse(
                (x - rr, y - rr, x + rr, y + rr), fill=RED, outline="white", width=2
            )
        elif loc == "box":
            x0, y0, x1, y1 = unit["region"]["coordinates"]
            coords = (
                (x0 - box[0]) * scale,
                (y0 - box[1]) * scale,
                (x1 - box[0]) * scale,
                (y1 - box[1]) * scale,
            )
            d.rectangle(coords, outline=RED, width=2)
    out = Image.new("RGB", size, (238, 242, 247))
    out.paste(image, ((size[0] - actual[0]) // 2, (size[1] - actual[1]) // 2))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok="--refresh-gallery" in sys.argv)
    VIEW.mkdir(parents=True, exist_ok=True)
    public = {
        r["request_id"]: r for r in read_jsonl(ROOT / "data/benchmark_public.jsonl")
    }
    cases = {r["id"]: r for r in read_jsonl(ROOT / "data/offline_gt.jsonl")}
    records = {
        name: read_jsonl(ROOT / "eval" / name / "score/combined.jsonl")
        for name in ["base", "step2000"]
    }
    score = {
        name: {(r["id"], t["label"]): t for r in rows for t in r["targets"]}
        for name, rows in records.items()
    }
    base_complete = {r["id"]: r["compatible"] for r in records["base"]}

    # Illustrative gains: select by recorded per-target improvement, avoiding repeated sources.
    gains = []
    seen = set()
    for setting in ["S1", "S3-point", "S3-box", "S4-noref"]:
        candidates = [
            (key, t)
            for key, t in score["step2000"].items()
            if key[0].endswith("--" + setting)
            and t["locator"] != "mask"
            and key[0].split("--")[0] not in seen
        ]
        key, t = max(
            candidates,
            key=lambda item: (item[1]["iou"] - score["base"][item[0]]["iou"], item[0]),
        )
        seen.add(key[0].split("--")[0])
        gains.append((*key, "large improvement"))
    mixed = [
        ("v2-sa_3042396--S5", f"U{i}", "same joint generation") for i in range(1, 5)
    ]
    cautions = [
        ("v2-sa_5447626--S2-point", "U4", "regression"),
        ("v2-sa_455310--S2-box", "U1", "small change"),
        ("v2-sa_6779549--S2-mask", "U2", "exact mask copy; codec loss"),
        ("v2-sa_455310--S1", "U2", "both still inaccurate"),
    ]
    sheets = [
        (
            "01_improvements",
            "Examples with clear localization gains",
            "Selected by large per-target IoU gain in four settings; illustrative, not a random sample.",
            gains,
        ),
        (
            "02_joint_s5",
            "S5: point + box + mask + text in ONE request",
            "All four rows belong to the same image and one generate() per model. No per-target reruns.",
            mixed,
        ),
        (
            "03_limits",
            "Regressions, small changes, and codec limitations",
            "Includes a regression, a near tie, exact mask-copy reconstruction loss, and a remaining failure.",
            cautions,
        ),
    ]
    selected = []
    pages = []
    for filename, title, subtitle, items in sheets:
        canvas = Image.new("RGB", (W, HEADER + RH * len(items) + 70), "white")
        d = ImageDraw.Draw(canvas)
        d.rectangle((0, 0, W, HEADER), fill=(20, 35, 54))
        d.text((20, 12), title, font=font(29, True), fill="white")
        d.text((20, 52), subtitle, font=font(19), fill=(207, 220, 235))
        headings = [
            ("PUBLIC INPUT", RED),
            ("GROUND TRUTH", GT),
            ("BEFORE: BASE SAMTok", BEFORE),
            ("AFTER: LoRA step2000", AFTER),
        ]
        for i, (label, color) in enumerate(headings):
            d.text((i * CW + 16, 98), label, font=font(22, True), fill=color)
        for j, (rid, uid, reason) in enumerate(items):
            key = (rid, uid)
            p = public[rid]
            unit = next(u for u in p["units"] if u["id"] == uid)
            case = cases[p["case_id"]]
            truth = next(u for u in case["units"] if u["label"] == uid)
            before, after = score["base"][key], score["step2000"][key]
            with Image.open(case["image"]) as im:
                source = im.convert("RGB")
            gt = decode_rle(truth["selection"])
            with Image.open(before["mask_path"]) as im:
                bm = np.asarray(im.convert("L")) > 0
            with Image.open(after["mask_path"]) as im:
                am = np.asarray(im.convert("L")) > 0
            assert gt.shape == bm.shape == am.shape == (source.height, source.width)

            # Confirm plotted masks reproduce the saved per-target IoUs.
            def iou(mask):
                return float((mask & gt).sum() / max(1, (mask | gt).sum()))

            assert (
                abs(iou(bm) - before["iou"]) < 1e-10
                and abs(iou(am) - after["iou"]) < 1e-10
            )
            yy, xx = np.where(gt)
            margin = max(
                40, round(max(xx.max() - xx.min(), yy.max() - yy.min()) * 0.65)
            )
            crop = (
                max(0, int(xx.min()) - margin),
                max(0, int(yy.min()) - margin),
                min(source.width, int(xx.max()) + margin + 1),
                min(source.height, int(yy.max()) + margin + 1),
            )
            y = HEADER + j * RH
            d.rectangle(
                (0, y, W, y + RH),
                fill=(255, 255, 255) if j % 2 == 0 else (248, 250, 253),
            )
            d.line((0, y, W, y), fill=(210, 219, 230), width=2)
            loc = "text only" if unit["locator"] == "none" else unit["locator"]
            delta = 100 * (after["iou"] - before["iou"])
            caption = f"{p['setting']}  |  {p['case_id']}  {uid}  |  {loc}  |  {reason}"
            d.text((16, y + 9), caption, font=font(21, True), fill=(27, 43, 63))
            d.text(
                (W - 230, y + 10),
                f"Delta: {delta:+.1f} pp",
                font=font(20, True),
                fill=(32, 120, 83) if delta >= 0 else (190, 56, 56),
            )
            lines = wrap_pixels(unit["instruction"], font(18), W - 32)
            assert len(lines) <= 2, lines
            for k, line in enumerate(lines):
                d.text((16, y + 40 + k * 21), line, font=font(18), fill=(61, 76, 95))
            kinds = [
                "input_mask" if unit["locator"] == "mask" else "input",
                "gt",
                "before",
                "after",
            ]
            masks = [None, None, bm, am]
            for col, (kind, pred) in enumerate(zip(kinds, masks)):
                x = col * CW + 10
                canvas.paste(
                    panel(source, gt, pred, unit, kind, (CW - 20, 154)), (x, y + 88)
                )
                canvas.paste(
                    panel(source, gt, pred, unit, kind, (CW - 20, 226), crop=crop),
                    (x, y + 255),
                )
            public_label = loc + (
                " + ref" if unit["has_ref"] and loc != "text only" else ""
            )
            labels = [
                public_label,
                f"GT area: {100 * truth['area']:.3f}%",
                f"IoU: {100 * before['iou']:.1f}%"
                + (" | MISSING" if not before["valid_id_span"] else ""),
                f"IoU: {100 * after['iou']:.1f}%",
            ]
            for col, label in enumerate(labels):
                d.text(
                    (col * CW + 14, y + 489),
                    label,
                    font=font(23, True),
                    fill=(30, 47, 67),
                )
            visible = []
            for col, mask in [(2, bm), (3, am)]:
                fraction = float(
                    mask[crop[1] : crop[3], crop[0] : crop[2]].sum()
                    / max(1, mask.sum())
                )
                visible.append(fraction)
                if mask.any() and fraction < 0.05:
                    d.text(
                        (col * CW + 20, y + 263),
                        "Prediction lies outside this zoom",
                        font=font(16, True),
                        fill=(186, 44, 49),
                        stroke_width=2,
                        stroke_fill="white",
                    )
            selected.append(
                {
                    "sheet": filename,
                    "request_id": rid,
                    "case_id": p["case_id"],
                    "label": uid,
                    "locator": unit["locator"],
                    "has_ref": unit["has_ref"],
                    "instruction": unit["instruction"],
                    "selection_reason": reason,
                    "before_iou": before["iou"],
                    "after_iou": after["iou"],
                    "delta": after["iou"] - before["iou"],
                    "before_valid_target": before["valid_id_span"],
                    "before_request_complete_compatible": base_complete[rid],
                    "before_span": before["predicted_span"],
                    "after_span": after["predicted_span"],
                    "before_mask_sha256": sha256(before["mask_path"]),
                    "after_mask_sha256": sha256(after["mask_path"]),
                    "crop_source_xyxy": list(crop),
                    "before_after_predicted_area_visible_in_crop": visible,
                }
            )
        fy = HEADER + RH * len(items) + 10
        d.text(
            (16, fy),
            "Top: full view. Bottom: shared detail crop. Marks and zooms are display-only; model inputs follow the public protocol.",
            font=font(17),
            fill=(63, 77, 94),
        )
        d.text(
            (16, fy + 25),
            "Green = GT contour; orange = before; cyan = after. IoU uses original masks per target, without whole-request format gating. Pass1 only.",
            font=font(17),
            fill=(63, 77, 94),
        )
        full = OUT / f"{filename}.png"
        canvas.save(full)
        thumb = canvas.copy()
        thumb.thumbnail((1260, 2400), Image.Resampling.LANCZOS)
        thumb.save(OUT / f"{filename}_preview.jpg", quality=94, subsampling=0)
        shutil.copy2(full, VIEW / full.name)
        shutil.copy2(OUT / f"{filename}_preview.jpg", VIEW / f"{filename}_preview.jpg")
        pages.append(canvas)
    pdf = OUT / "benchmark_before_after.pdf"
    pages[0].save(pdf, save_all=True, append_images=pages[1:], resolution=150.0)
    shutil.copy2(pdf, VIEW / pdf.name)
    write_json(
        OUT / "selection.json",
        {
            "policy": "Illustrations only: improvement examples are selected by gain; S5 selected by four-modality coverage; limitations chosen from prior reported errors. No inference reruns or metric changes.",
            "rows": selected,
        },
    )
    with (OUT / "comparison.csv").open("w") as f:
        keys = [
            "sheet",
            "request_id",
            "label",
            "locator",
            "has_ref",
            "before_iou",
            "after_iou",
            "delta",
            "selection_reason",
        ]
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)
    text = [
        "# Benchmark Pass1 训练前后对比缩略图",
        "",
        "来源：既有 IF-Bench dev20 的 base SAMTok 与 Pass1 LoRA step2000 预测。只读取原始预测 mask，不重新推理，不修改评测结果。",
        "",
        "图列依次为：公开输入展示、GT、训练前、训练后。每项上部展示完整源图，下部展示同一 GT 邻域的局部放大；这些放大和 GT 轮廓仅用于报告。原模型输入仍为干净完整源图与公开协议。",
        "",
        "图中 IoU 为逐目标原始解码 IoU，不因整条请求其他目标缺失而将其清零；原汇总报告中严格/兼容解析的整请求计分口径保持不变。",
        "",
        "第一组按增幅选择明显提升的例子，不代表随机样本；第二组完整展示一次 S5 四目标联合请求；第三组包含退步、变化较小、codec 误差及仍然失败的例子。",
        "",
        f"- [明显提升]({VIEW}/01_improvements.png)",
        f"- [S5 四种输入同图联合输出]({VIEW}/02_joint_s5.png)",
        f"- [退步与局限]({VIEW}/03_limits.png)",
        f"- [三页 PDF]({VIEW}/benchmark_before_after.pdf)",
        "",
        "| 组 | request / label | 输入 | 训练前 IoU | 训练后 IoU |",
        "|---|---|---|---:|---:|",
    ]
    for x in selected:
        text.append(
            f"| {x['sheet']} | {x['request_id']} / {x['label']} | {x['locator']} | {100 * x['before_iou']:.1f}% | {100 * x['after_iou']:.1f}% |"
        )
    (OUT / "README.md").write_text("\n".join(text) + "\n")
    shutil.copy2(OUT / "README.md", VIEW / "README.md")
    print(
        json.dumps(
            {
                "output": str(VIEW),
                "pages": len(pages),
                "target_comparisons": len(selected),
                "files": [p.name for p in VIEW.iterdir()],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
