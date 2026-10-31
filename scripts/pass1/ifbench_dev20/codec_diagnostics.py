"""Display codec-only losses, without changing masks, inputs, or model answers."""

import shutil

import numpy as np
from common import PREVIEW, ROOT, read_jsonl
from PIL import Image, ImageDraw
from report import contain, font, overlay

from samtok_edit21.pass1.data.pilot import decode_rle


def main():
    rows = [
        r for p in sorted((ROOT / "data").glob("codec_*.jsonl")) for r in read_jsonl(p)
    ]
    units = [(c, u) for c in rows for u in c["units"]]
    units.sort(key=lambda x: x[1]["roundtrip"])
    bad = [(c, u) for c, u in units if u["roundtrip"] < 0.5]
    out = ROOT / "reports/visuals"
    out.mkdir(parents=True, exist_ok=True)
    PREVIEW.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(bad), 4):
        items = bad[start : start + 4]
        page = Image.new("RGB", (2400, 100 + 410 * len(items)), "white")
        d = ImageDraw.Draw(page)
        d.text(
            (15, 12),
            "Codec-only diagnostic: GT mask -> two-code token -> decoded mask",
            font=font(28),
            fill="black",
        )
        d.text(
            (15, 52),
            "No MLLM prediction here. Right two panels zoom the same GT-centered crop; outside-crop predictions remain visible in full view.",
            font=font(20),
            fill="#444444",
        )
        for j, (case, u) in enumerate(items):
            y = 100 + 410 * j
            d.text(
                (15, y),
                f"{case['id']} {u['label']} | codec IoU {100 * u['roundtrip']:.1f}% | GT area {100 * u['area']:.3f}%",
                font=font(22),
                fill="black",
            )
            d.text((15, y + 28), u["ref"], font=font(19), fill="#3d4958")
            with Image.open(case["image"]) as im:
                src = im.convert("RGB")
            gt = decode_rle(u["selection"])
            rt = decode_rle(u["canonical_dec"])
            yy, xx = np.where(gt)
            margin = max(40, round(max(xx.max() - xx.min(), yy.max() - yy.min()) * 0.6))
            crop = (
                max(0, int(xx.min()) - margin),
                max(0, int(yy.min()) - margin),
                min(src.width, int(xx.max()) + margin + 1),
                min(src.height, int(yy.max()) + margin + 1),
            )
            gi = overlay(src, gt, (30, 220, 60))
            ri = overlay(src, rt, (210, 80, 240))
            panels = [src, gi, ri, gi.crop(crop), ri.crop(crop)]
            for k, (image, label) in enumerate(
                zip(
                    panels,
                    [
                        "Source image",
                        "GT source selector",
                        "Codec decoded mask",
                        "GT crop",
                        "Codec crop",
                    ],
                )
            ):
                page.paste(contain(image, 468, 300), (480 * k + 6, y + 62))
                d.text((480 * k + 10, y + 368), label, font=font(21), fill="black")
        p = out / f"codec_failures_{start // 4 + 1}.jpg"
        page.save(p, quality=92)
        shutil.copy2(p, PREVIEW / p.name)
    print(
        {
            "reference_mIoU": np.mean([u["roundtrip"] for _, u in units]),
            "IoU_below_50": len(bad),
            "files": str(PREVIEW),
        }
    )


if __name__ == "__main__":
    main()
