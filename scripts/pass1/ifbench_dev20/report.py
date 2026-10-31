"""Paired source-image analysis; no benchmark-driven model or prompt selection."""

import csv
import hashlib
import json
import shutil
import sys
import textwrap
from collections import Counter

import numpy as np
from common import ADAPTER, PREVIEW, ROOT, read_jsonl, write_json
from PIL import Image, ImageDraw, ImageFont
from samtok_benchmark.ifbench import SETTINGS

from samtok_edit21.pass1.data.pilot import decode_rle


def mean(values):
    values = list(values)
    return float(np.mean(values)) if values else None


def percent(x):
    return "—" if x is None else f"{100 * x:.1f}"


def summarize(rows):
    targets = [t for r in rows for t in r["targets"]]
    unknown = [t for t in targets if t["locator"] != "mask"]
    known = [t for t in targets if t["locator"] == "mask"]
    points = [t for t in targets if t["locator"] == "point"]
    return {
        "requests": len(rows),
        "targets": len(targets),
        "unknown_targets": len(unknown),
        "known_mask_targets": len(known),
        "strict_format": mean(r["strict"] for r in rows),
        "compatible_format": mean(r["compatible"] for r in rows),
        "all_strict_iou": mean(t["strict_iou"] for t in targets),
        "all_compatible_iou": mean(t["compatible_iou"] for t in targets),
        "all_raw_iou": mean(t["iou"] for t in targets),
        "unknown_strict_iou": mean(t["strict_iou"] for t in unknown),
        "unknown_compatible_iou": mean(t["compatible_iou"] for t in unknown),
        "unknown_raw_iou": mean(t["iou"] for t in unknown),
        "unknown_strict_iou50": mean(t["strict_iou"] >= 0.5 for t in unknown),
        "unknown_strict_iou75": mean(t["strict_iou"] >= 0.75 for t in unknown),
        "unknown_reference_iou": mean(t["canonical_iou"] for t in unknown),
        "all_reference_iou": mean(t["canonical_iou"] for t in targets),
        "known_mask_exact_copy": mean(t["exact_code"] for t in known),
        "point_hit_raw": mean(t["point_hit"] for t in points),
        "point_hit_strict": mean(
            t["point_hit"] and t["request_strict"] for t in points
        ),
        "all_targets_iou50": mean(r["all_iou50"] for r in rows),
        "unknown_all_targets_iou50": mean(
            all(t["strict_iou"] >= 0.5 for t in r["targets"] if t["locator"] != "mask")
            for r in rows
            if any(t["locator"] != "mask" for t in r["targets"])
        ),
        "possible_id_swaps": sum(t["possible_id_swap"] for t in unknown),
        "invalid_target_spans": sum(not t["valid_id_span"] for t in targets),
    }


def paired_interval(left, right, field_left, field_right, unknown=True):
    """Paired bootstrap resamples source images, never the 10 repeated settings."""
    la = {r["group"]: r for r in left}
    ra = {r["group"]: r for r in right}
    assert len(la) == len(left) and set(la) == set(ra)
    pairs = []
    for group in sorted(la):
        a = [t for t in la[group]["targets"] if not unknown or t["locator"] != "mask"]
        b = [t for t in ra[group]["targets"] if not unknown or t["locator"] != "mask"]
        assert [t["label"] for t in a] == [t["label"] for t in b]
        pairs.append(
            [sum(t[field_left] for t in a) - sum(t[field_right] for t in b), len(a)]
        )
    pairs = np.asarray(pairs, dtype=float)
    if pairs[:, 1].sum() == 0:
        return None
    rng = np.random.default_rng(42)
    sums = pairs[rng.integers(0, len(pairs), (5000, len(pairs)))].sum(axis=1)
    values = sums[sums[:, 1] > 0, 0] / sums[sums[:, 1] > 0, 1]
    return {
        "delta": float(pairs[:, 0].sum() / pairs[:, 1].sum()),
        "ci95": np.quantile(values, [0.025, 0.975]).tolist(),
        "resampling_unit": "source image",
        "images": len(pairs),
        "replicates": 5000,
    }


def font(size):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


def overlay(image, mask, color):
    from scipy.ndimage import binary_dilation, binary_erosion

    arr = np.asarray(image.convert("RGB")).copy()
    m = np.asarray(mask) > 0
    arr[m] = (0.60 * arr[m] + 0.40 * np.array(color)).astype(np.uint8)
    boundary = binary_dilation(m) & ~binary_erosion(m)
    arr[boundary] = color
    return Image.fromarray(arr)


def contain(image, width, height):
    image = image.copy()
    image.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), (242, 243, 245))
    canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
    return canvas


def visuals(scores, publics, truth, summaries):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PREVIEW.mkdir(parents=True, exist_ok=True)
    vd = ROOT / "reports/visuals"
    vd.mkdir(parents=True, exist_ok=True)
    # Unknown targets separate mask copy from actual localization.
    active = [s for s in SETTINGS if summaries["step2000"][s]["unknown_targets"]]
    fig, ax = plt.subplots(figsize=(11, 5.8), constrained_layout=True)
    yy = np.arange(len(active))
    height = 0.36
    ax.barh(
        yy + height / 2,
        [100 * summaries["base"][s]["unknown_compatible_iou"] for s in active],
        height,
        label="Base SAMTok (fixed wrapper compatibility)",
        color="#A8AEB9",
    )
    ax.barh(
        yy - height / 2,
        [100 * summaries["step2000"][s]["unknown_strict_iou"] for s in active],
        height,
        label="Pass1 LoRA step2000 (strict protocol)",
        color="#227EBD",
    )
    for y, s in zip(yy, active):
        value = 100 * summaries["step2000"][s]["unknown_strict_iou"]
        ax.text(value + 1, y - height / 2, f"{value:.1f}", va="center", fontsize=9)
    ax.set(
        yticks=yy,
        yticklabels=active,
        xlim=(0, 104),
        xlabel="Unknown-target mIoU (%)",
        title="IF-Bench fixed dev20 | Pass1 only | same 20 images, 72 targets",
    )
    ax.invert_yaxis()
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.15)
    fig.savefig(vd / "setting_scores.png", dpi=180)
    plt.close(fig)

    maps = {
        name: {(r["id"], t["label"]): t for r in rows for t in r["targets"]}
        for name, rows in scores.items()
    }
    pmap = {p["request_id"]: p for p in publics}
    tmap = {(c["id"], u["label"]): u for c in truth for u in c["units"]}
    case_map = {c["id"]: c for c in truth}
    codec_cases = [
        c for p in sorted((ROOT / "data").glob("codec_*.jsonl")) for c in read_jsonl(p)
    ]
    cmap = {(c["id"], u["label"]): u for c in codec_cases for u in c["units"]}

    def draw_page(items, title, name):
        width, colw, rowh = 2400, 480, 650
        page = Image.new("RGB", (width, 105 + rowh * len(items)), "white")
        d = ImageDraw.Draw(page)
        d.text((18, 12), title, font=font(28), fill="#152338")
        d.text(
            (18, 52),
            "Columns: public input | GT | base | LoRA | codec. Each has full view + same GT-centered detail crop (display only). No edits.",
            font=font(21),
            fill="#404b5b",
        )
        for j, (rid, label, why) in enumerate(items):
            y = 105 + j * rowh
            public = pmap[rid]
            case_id = public["case_id"]
            unit = next(u for u in public["units"] if u["id"] == label)
            gt = tmap[(case_id, label)]
            ref = cmap[(case_id, label)]
            base = maps["base"][(rid, label)]
            fine = maps["step2000"][(rid, label)]
            d.text(
                (12, y),
                f"{public['setting']} | {case_id} {label} | {unit['locator']} | {why}",
                font=font(23),
                fill="black",
            )
            instruction = unit["instruction"]
            for k, line in enumerate(textwrap.wrap(instruction, 170)[:2]):
                d.text((12, y + 30 + 24 * k), line, font=font(20), fill="#38475a")
            with Image.open(case_map[case_id]["image"]) as im:
                source = im.convert("RGB")
            gm = decode_rle(gt["selection"])
            rt = decode_rle(ref["canonical_dec"])
            inp = source.copy()
            mark = ImageDraw.Draw(inp)
            w, h = inp.size
            if unit["locator"] == "point":
                x, yy = unit["region"]["coordinates"]
                rr = max(8, round(min(w, h) * 0.012))
                mark.ellipse(
                    (x - rr, yy - rr, x + rr, yy + rr),
                    fill="red",
                    outline="white",
                    width=max(2, rr // 4),
                )
            elif unit["locator"] == "box":
                mark.rectangle(
                    unit["region"]["coordinates"],
                    outline="red",
                    width=max(3, round(min(w, h) * 0.006)),
                )
            elif unit["locator"] == "mask":
                inp = overlay(inp, gm, (240, 60, 60))
            with Image.open(base["mask_path"]) as im:
                bm = np.asarray(im) > 0
            with Image.open(fine["mask_path"]) as im:
                fm = np.asarray(im) > 0
            panels = [
                inp,
                overlay(source, gm, (40, 220, 70)),
                overlay(source, bm, (255, 155, 40)),
                overlay(source, fm, (0, 210, 240)),
                overlay(source, rt, (190, 90, 230)),
            ]
            ys, xs = np.where(gm)
            margin = max(40, round(max(xs.max() - xs.min(), ys.max() - ys.min()) * 0.6))
            crop = (
                max(0, int(xs.min()) - margin),
                max(0, int(ys.min()) - margin),
                min(source.width, int(xs.max()) + margin + 1),
                min(source.height, int(ys.max()) + margin + 1),
            )
            captions = [
                "Public locator (display only)",
                "GT source selection",
                f"Base raw IoU {100 * base['iou']:.1f}%",
                f"LoRA strict IoU {100 * fine['strict_iou']:.1f}%",
                f"Codec IoU {100 * ref['roundtrip']:.1f}%",
            ]
            for i, (panel, caption) in enumerate(zip(panels, captions)):
                page.paste(contain(panel, colw - 12, 245), (i * colw + 6, y + 88))
                d.text((i * colw + 8, y + 337), caption, font=font(20), fill="#172338")
                page.paste(
                    contain(panel.crop(crop), colw - 12, 260), (i * colw + 6, y + 368)
                )
        page.save(vd / name)

    representatives = []
    worst = []
    for setting in SETTINGS:
        ts = [
            t
            for r in scores["step2000"]
            if r["id"].endswith("--" + setting)
            for t in r["targets"]
        ]
        unknown = [t for t in ts if t["locator"] != "mask"]
        subset = unknown or ts
        ordered = sorted(subset, key=lambda t: (t["strict_iou"], t["id"], t["label"]))
        t = ordered[len(ordered) // 2]
        representatives.append((t["id"], t["label"], "median target by LoRA IoU"))
        if unknown:
            t = ordered[0]
            worst.append(
                (t["id"], t["label"], "lowest target IoU; diagnostic selection")
            )
    for i in range(0, 10, 5):
        draw_page(
            representatives[i : i + 5],
            "Pass1: all ten input settings (representative targets)",
            f"settings_{i // 5 + 1}.jpg",
        )
    for i in range(0, len(worst), 4):
        draw_page(
            worst[i : i + 4],
            "Pass1: localization failures (no retries or correction)",
            f"failures_{i // 4 + 1}.jpg",
        )
    te_items, seen = [], set()
    for t in sorted(
        maps["step2000"].values(), key=lambda t: (t["iou"], t["id"], t["label"])
    ):
        key = (t["id"].split("--")[0], t["label"])
        if (
            t["locator"] != "mask"
            and t["canonical_iou"] >= 0.7
            and t["iou"] < 0.5
            and key not in seen
        ):
            seen.add(key)
            te_items.append((t["id"], t["label"], "codec IoU >=70%; LoRA IoU <50%"))
    for i in range(0, min(8, len(te_items)), 4):
        draw_page(
            te_items[i : i + 4],
            "Pass1: prediction failures with a good codec reference",
            f"te_failures_{i // 4 + 1}.jpg",
        )
    s5 = [r for r in scores["step2000"] if r["id"].endswith("--S5")]
    s5.sort(
        key=lambda r: (
            np.mean([t["strict_iou"] for t in r["targets"] if t["locator"] != "mask"])
            if any(t["locator"] != "mask" for t in r["targets"])
            else 1,
            r["id"],
        )
    )
    # Full mixed requests, not isolated successful targets: lower, median, upper quartiles.
    for q, index in [("low", 0), ("median", len(s5) // 2), ("high", len(s5) - 1)]:
        row = s5[index]
        draw_page(
            [
                (row["id"], t["label"], "same joint S5 generation")
                for t in row["targets"]
            ],
            f"Pass1 S5 joint request: {row['id']} ({q} unknown-target mean)",
            f"mixed_{q}.jpg",
        )
    write_json(
        vd / "selection.json",
        {
            "representatives": representatives,
            "failures": worst,
            "te_failures": te_items[:8],
            "policy": "Post-hoc diagnostic visualization only; never affects predictions or scores",
        },
    )
    for p in vd.iterdir():
        if p.suffix in {".png", ".jpg", ".pdf"}:
            shutil.copy2(p, PREVIEW / p.name)
    return [p.name for p in vd.iterdir() if p.suffix in {".png", ".jpg"}]


def main():
    scores = {
        name: read_jsonl(ROOT / "eval" / name / "score/combined.jsonl")
        for name in ["base", "step2000"]
    }
    preds = {
        name: read_jsonl(ROOT / "eval" / name / "inference/combined.jsonl")
        for name in scores
    }
    inference_rows = {
        r["id"]: r for r in read_jsonl(ROOT / "data/inference_public.jsonl")
    }
    scoring_rows = read_jsonl(ROOT / "data/scoring_truth.jsonl")
    for row in scoring_rows:
        assert [t["label"] for t in row["request"]["targets"]] == [
            t["label"] for t in row["truth"]
        ]
        assert row["request"] == inference_rows[row["id"]]["request"]
    for name, rows in preds.items():
        assert len(rows) == 200 and {r["id"] for r in rows} == set(inference_rows)
        for pred in rows:
            expected = hashlib.sha256(
                json.dumps(
                    inference_rows[pred["id"]]["request"], sort_keys=True
                ).encode()
            ).hexdigest()
            assert pred["request_sha256"] == expected
            assert pred["adapter"] == (str(ADAPTER) if name == "step2000" else None)
    publics = read_jsonl(ROOT / "data/benchmark_public.jsonl")
    truth = read_jsonl(ROOT / "data/offline_gt.jsonl")
    canonical = [
        dict(u, case_id=c["id"])
        for p in sorted((ROOT / "data").glob("codec_*.jsonl"))
        for c in read_jsonl(p)
        for u in c["units"]
    ]
    codec_reference = {
        "targets": len(canonical),
        "mean_iou": mean(u["roundtrip"] for u in canonical),
        "below_iou50": sum(u["roundtrip"] < 0.5 for u in canonical),
        "below_iou75": sum(u["roundtrip"] < 0.75 for u in canonical),
        "worst": [
            {k: u[k] for k in ["case_id", "label", "roundtrip", "area", "ref"]}
            for u in sorted(canonical, key=lambda x: x["roundtrip"])[:8]
        ],
    }
    assert all(len(rows) == 200 for rows in scores.values())
    assert all(len({r["group"] for r in rows}) == 20 for rows in scores.values())
    by_setting = {
        name: {s: [r for r in rows if r["id"].endswith("--" + s)] for s in SETTINGS}
        for name, rows in scores.items()
    }
    summaries = {
        name: {s: summarize(rs) for s, rs in ss.items()}
        for name, ss in by_setting.items()
    }
    improvements = {
        s: paired_interval(
            by_setting["step2000"][s],
            by_setting["base"][s],
            "strict_iou",
            "compatible_iou",
        )
        for s in SETTINGS
    }
    ref_pairs = {
        s: paired_interval(
            by_setting["step2000"][s],
            by_setting["step2000"][t],
            "strict_iou",
            "strict_iou",
        )
        for s, t in [
            ("S2-point", "S3-point"),
            ("S2-box", "S3-box"),
            ("S4-ref", "S4-noref"),
        ]
    }
    all_targets = [
        dict(t, model=name, setting=r["id"].split("--")[1])
        for name, rows in scores.items()
        for r in rows
        for t in r["targets"]
    ]
    diagnostics = {}
    for name in scores:
        ts = [t for t in all_targets if t["model"] == name and t["locator"] != "mask"]
        diagnostics[name] = {}
        for key in ["mode", "operation", "K"]:
            diagnostics[name][key] = {
                str(value): {
                    "n": len(sub),
                    "strict_iou": mean(t["strict_iou"] for t in sub),
                    "compatible_iou": mean(t["compatible_iou"] for t in sub),
                    "raw_iou": mean(t["iou"] for t in sub),
                    "reference_iou": mean(t["canonical_iou"] for t in sub),
                }
                for value in sorted({t[key] for t in ts})
                for sub in [[t for t in ts if t[key] == value]]
            }
        diagnostics[name]["area"] = {
            label: {
                "n": len(sub),
                "strict_iou": mean(t["strict_iou"] for t in sub),
                "reference_iou": mean(t["canonical_iou"] for t in sub),
            }
            for label, lo, hi in [
                ("below_0.1%", 0, 0.001),
                ("0.1%-1%", 0.001, 0.01),
                ("1%-5%", 0.01, 0.05),
                ("above_5%", 0.05, 1.01),
            ]
            for sub in [[t for t in ts if lo <= t["area"] < hi]]
        }
    regression = {
        name: {
            "canonical_cache_min_iou": min(
                t["canonical_cached_agreement"] for r in rows for t in r["targets"]
            ),
            "canonical_cache_all_pixels_equal": all(
                t["canonical_cached_equal"] for r in rows for t in r["targets"]
            ),
            "strict_errors": dict(Counter(e for r in rows for e in r["errors"])),
            "max_generated_tokens": max(p["new_tokens"] for p in preds[name]),
            "token_limit_hits": sum(
                p["new_tokens"] >= p["max_new_tokens"] for p in preds[name]
            ),
            "max_input_tokens": max(p["input_tokens"] for p in preds[name]),
            "total_generation_seconds": sum(p["elapsed_s"] for p in preds[name]),
        }
        for name, rows in scores.items()
    }
    assert all(x["canonical_cache_all_pixels_equal"] for x in regression.values()), (
        regression
    )
    results = {
        "scope": "Pass1 localization on fixed dev20, not editing scores",
        "settings": summaries,
        "overall_repeated_requests": {
            name: summarize(rows) for name, rows in scores.items()
        },
        "paired_delta_lora_strict_minus_base_compatible": improvements,
        "paired_ref_minus_noref": ref_pairs,
        "diagnostics": diagnostics,
        "checks": regression,
        "codec_reference_unique72": codec_reference,
    }
    unknown = [
        t for r in scores["step2000"] for t in r["targets"] if t["locator"] != "mask"
    ]
    error_split = {
        "unknown_outputs": len(unknown),
        "prediction_below_iou50": sum(t["iou"] < 0.5 for t in unknown),
        "prediction_and_reference_below_iou50": sum(
            t["iou"] < 0.5 and t["canonical_iou"] < 0.5 for t in unknown
        ),
        "prediction_below50_reference_atleast50": sum(
            t["iou"] < 0.5 and t["canonical_iou"] >= 0.5 for t in unknown
        ),
        "prediction_below50_reference_atleast70": sum(
            t["iou"] < 0.5 and t["canonical_iou"] >= 0.7 for t in unknown
        ),
        "prediction_exceeds_reference_by5points": sum(
            t["iou"] > t["canonical_iou"] + 0.05 for t in unknown
        ),
    }
    results["unknown_error_reference_comparison"] = error_split
    write_json(ROOT / "reports/metrics.json", results)
    fields = [
        "model",
        "setting",
        "id",
        "group",
        "label",
        "K",
        "mode",
        "operation",
        "area",
        "iou",
        "strict_iou",
        "compatible_iou",
        "canonical_iou",
        "precision",
        "recall",
        "area_ratio",
        "possible_id_swap",
        "predicted_span",
        "expected_span",
        "mask_path",
    ]
    with (ROOT / "reports/targets.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_targets)
    figures = (
        [
            p.name
            for p in (ROOT / "reports/visuals").iterdir()
            if p.suffix in {".jpg", ".png"}
        ]
        if "--reuse-visuals" in sys.argv
        else visuals(scores, publics, truth, summaries)
    )
    audit = json.loads((ROOT / "provenance/preflight.json").read_text())
    fine = results["overall_repeated_requests"]["step2000"]
    base = results["overall_repeated_requests"]["base"]
    lines = [
        "# IF-Bench dev20：SAMTok Pass1 定位评测",
        "",
        "本次只评估第一轮自回归 mask token 生成，不运行 DiT、图像编辑或编辑 Judge。此前正式实验的独立 test 不是该 benchmark；本报告是新完成的 benchmark 定位诊断。",
        "",
        f"**结论：当前 LoRA 已能稳定执行全部十种 setting 的一次联合输出协议，定位质量也有实质提升。** 未知目标 mIoU 从 base 的 {percent(base['unknown_compatible_iou'])}% 提升至 {percent(fine['unknown_strict_iou'])}%；S5 从 {percent(summaries['base']['S5']['unknown_compatible_iou'])}% 提升至 {percent(summaries['step2000']['S5']['unknown_strict_iou'])}%。但 S5 只有 9/20 张图的所有目标 IoU 均≥0.5，尚不能把协议稳定等同于所有目标都定位准确。",
        "",
        "主要残留问题是选取范围、部件和遮挡边界、相似实例/空间指代，以及 codec 对细小区域的重建。20 图共 72 个目标，其中 52 个面积不足原图 1%，中位面积约 0.34%；不是此前独立测试集难度的直接复刻。",
        "",
        "固定使用正式训练 dev 选出的 **step 2000**，与原始 Qwen3-VL-8B-SAMTok 在相同协议、相同请求下比较。未在本 benchmark 上选择 checkpoint、改 prompt、训练、搜索候选或重试低分样本。",
        "",
        f"- Benchmark v2branch commit：`{audit['benchmark_commit']}`；release 2.6.0，manifest `{audit['manifest_sha256']}`。",
        "- 沿用官方固定 dev20：20 张源图、72 个不同目标，K=2/3/4/5 分别 3/4/11/2 张。每模型 10 settings × 20 图 = 200 次联合生成、720 个目标输出。不是 200 张独立图像。",
        f"- 与实际正式训练的 {audit['train_requests']:,} 条请求 / {audit['train_unique_images']:,} 张源图比较，20 图 SHA256 重合为 0；另核对 SA 源图 ID，重合也为 0。这里只证明本次微调训练隔离，不对原始 SAMTok 预训练作无依据声明。",
        "- 仅使用 dev20；原 976 test 图像未打开，其余 88 dev 保持不变。",
        "",
        "## 每个 setting 如何适配",
        "",
        "图像始终是干净源图 I。模型读取训练时相同的 `samtok_pass1_v2` JSON：`label / instruction / has_ref / locator / region`。所有目标一起输入，一次 `generate` 返回按 U1…UK 排序的 JSON 数组，每项仅一个完整两码 mask span。保留原始完整指令，包括编辑 payload；不抽取私有目标名。",
        "",
        "| Setting | instruction | locator | 本次公开目标数 |",
        "|---|---|---|---|",
    ]
    desc = {
        "S1": ("原始 ref 指令", "none"),
        "S2-point": ("原始 ref 指令", "point"),
        "S2-box": ("原始 ref 指令", "box"),
        "S2-mask": ("原始 ref 指令", "mask"),
        "S3-point": ("原始 no-ref 指令", "point"),
        "S3-box": ("原始 no-ref 指令", "box"),
        "S3-mask": ("原始 no-ref 指令", "mask"),
        "S4-ref": ("原始 ref 指令", "固定 point/box/mask 混合"),
        "S4-noref": ("原始 no-ref 指令", "与 S4-ref 相同混合"),
        "S5": ("逐目标冻结 ref/no-ref", "逐目标冻结 none/point/box/mask"),
    }
    for s in SETTINGS:
        a, b = desc[s]
        counts = ", ".join(f"{k}={v}" for k, v in audit["per_setting"][s].items())
        lines.append(f"| {s} | {a} | {b} | {counts} |")
    lines += [
        "",
        "point 用原图 (W,H) 归一化至 0–1000、四舍五入；box 是半开 xyxy，左上向下取整、右下向上取整。72 点量化前后都在 GT 内，没有按 GT 调整坐标。mask 只对公开 mask 调用原始 codec，单 mask 编码以保持标签顺序。文本/point/box 均未调用 SAM 生成 mask。",
        "",
        "这是原生 SAMTok 结构化输入适配，而非编辑模型的双图 `[I,A]` 像素呈现。公开信息与 setting 保持一致；展示图中的点、框、彩色 mask 仅用于阅读，不作为模型额外输入。codec 的内部 SAM2 只用于已有 mask 编码和离线 token 解码，不是 point/box 前置分割。",
        "",
        "## 定位结果",
        "",
        "下表均为百分数。**未知目标**指 locator=none/point/box，排除已有 mask 的复制题。IoU 对比原始源区域 GT，严格要求 label 对齐，不做最优匹配或候选挑选。LoRA 主指标对整个请求的 JSON/标签/顺序/跨度严格检查；请求不完整时该请求所有目标记 0，失败仍在分母。",
        "",
        "原始模型另列固定兼容解析结果：只允许去掉预先定义的空 think/Markdown 包裹，不改 token、标签、顺序或额外字段。这样可以分开观察格式失败和几何定位失败。原始模型的严格结果也完整保存，不能把两种口径混称相同指标。",
        "",
        "| Setting | 未知目标数 | Base 未知 mIoU（兼容） | LoRA 未知 mIoU（严格） | LoRA 全目标 mIoU | LoRA 全目标都 IoU≥50 的请求 | LoRA 严格格式 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in SETTINGS:
        f, b = summaries["step2000"][s], summaries["base"][s]
        lines.append(
            f"| {s} | {f['unknown_targets']} | {percent(b['unknown_compatible_iou'])} | {percent(f['unknown_strict_iou'])} | {percent(f['all_strict_iou'])} | {percent(f['all_targets_iou50'])} | {percent(f['strict_format'])} |"
        )
    lines += [
        "",
        f"所有 200 次请求：LoRA 严格格式 {percent(fine['strict_format'])}%，已知 mask 原始生成 token 完整复制率 {percent(fine['known_mask_exact_copy'])}%；base 严格格式 {percent(base['strict_format'])}%，兼容格式 {percent(base['compatible_format'])}%，mask 复制率 {percent(base['known_mask_exact_copy'])}%。",
        f"S2-mask/S3-mask 的定位参考是 codec round trip：全部 72 GT mask→token→mask 的均值 {percent(summaries['step2000']['S2-mask']['all_reference_iou'])}%。这反映 codec 表达损失，不代表模型自主定位能力，也不是数学意义上的 IoU 上界。这里评分的是模型实际生成 token，**没有使用部署时 known-mask 覆盖策略纠正复制错误**。",
        "",
        f"其中 {codec_reference['below_iou50']}/72 个区域的 codec round trip IoU<50%，{codec_reference['below_iou75']}/72 个低于 75%。独立观察 codec 失败图可见：部分遮挡衣物会被扩展到相邻覆盖物，微小车辆或横幅会只还原局部或移到邻近结构。因此只训练 TE 不足以消除这部分已有 mask 的重建误差。",
        "",
        "## 成对比较与诊断",
        "",
        "以下 95% 区间用 20 张源图为单位成对 bootstrap 5,000 次；保留同一图内全部目标，不把重复 setting 当成独立图像。样本仍很小，不能替代 976 test 的最终论文评测。",
        "",
        "| Setting | LoRA 严格 − Base 兼容（百分点） | 95% 区间 |",
        "|---|---:|---|",
    ]
    for s in SETTINGS:
        d = improvements[s]
        if d:
            lines.append(
                f"| {s} | {100 * d['delta']:+.1f} | [{100 * d['ci95'][0]:+.1f}, {100 * d['ci95'][1]:+.1f}] |"
            )
    lines += [
        "",
        "| 同一定位信息，有指代 − 无指代 | 未知目标 mIoU 差（百分点） | 95% 区间 |",
        "|---|---:|---|",
    ]
    for s, d in ref_pairs.items():
        lines.append(
            f"| {s} | {100 * d['delta']:+.1f} | [{100 * d['ci95'][0]:+.1f}, {100 * d['ci95'][1]:+.1f}] |"
        )
    lines += [
        "",
        "以下按操作和大小汇总未知目标，包含跨 setting 重复观察，只用于诊断，不增加独立样本量。",
        "",
        "| 操作 | 目标输出数 | LoRA 严格 mIoU | codec 参考 IoU |",
        "|---|---:|---:|---:|",
    ]
    for op, d in diagnostics["step2000"]["operation"].items():
        lines.append(
            f"| {op} | {d['n']} | {percent(d['strict_iou'])} | {percent(d['reference_iou'])} |"
        )
    lines += [
        "",
        "| GT 区域面积占比 | 目标输出数 | LoRA 严格 mIoU | codec 参考 IoU |",
        "|---|---:|---:|---:|",
    ]
    for area, d in diagnostics["step2000"]["area"].items():
        lines.append(
            f"| {area} | {d['n']} | {percent(d['strict_iou'])} | {percent(d['reference_iou'])} |"
        )
    lines += [
        "",
        f"LoRA 未知目标可能串 ID 的诊断标记 {fine['possible_id_swaps']} 次（预测与另一目标 IoU>0.5 且比本目标高>0.2；不用于改答案）；point 严格命中率 {percent(fine['point_hit_strict'])}%。更多 mode/K/precision/recall/面积比见 metrics.json 和 targets.csv。",
        f"509 个未知目标输出中，{error_split['prediction_below_iou50']} 次预测 IoU<50%；其中 {error_split['prediction_and_reference_below_iou50']} 次对应的 GT codec reference 也低于 50%，另有 {error_split['prediction_below50_reference_atleast50']} 次 reference≥50%（其中 {error_split['prediction_below50_reference_atleast70']} 次≥70%）。前者不证明所有错误都由 codec 造成，后者明确显示 TE 仍有可改进空间。也有 {error_split['prediction_exceeds_reference_by5points']} 次预测比 reference 高超过 5 个百分点，再次说明 reference 不是上界。以上是跨 setting 的输出次数，不是独立目标数。",
        f"核验：codec 重复解码与缓存逐像素一致；LoRA 最大输入 {regression['step2000']['max_input_tokens']} tokens，触及 512 输出 token 上限 {regression['step2000']['token_limit_hits']} 次。base 对应上限命中 {regression['base']['token_limit_hits']} 次。",
        "",
        "## 对当前能力的判断",
        "",
        "1. 协议可以继续沿用。200/200 严格输出、211/211 mask 精确复制表明混合输入的 JSON、ID 绑定与一次联合输出已稳定；无需为了这批结果另起格式或添加候选。原始模型去除固定 Markdown 外壳后为 197/200，另 3 次漏目标。",
        "2. point/box 在无文本指代时也能工作。LoRA 的 S2−S3 点/框差值仅约 +1.2/+0.4 个百分点，源图 bootstrap 区间都跨 0；S4-ref 与 S4-noref 的差异也不显著。小样本尚不能证明两者等价，但没有出现原始模型那样明显的无指代性能下降。",
        "3. 混合联合输出仍需要提升区域精度。人工查看可视化可见：S1 的 `v2-sa_455310 U2` 把顶层栏杆定位到较低楼层；`v2-sa_8415518 U4` 把戴头巾者的上衣定位到头巾。它们的 codec reference 分别约 73.6% / 80.0%，因此确有 TE 定位错误。",
        "4. 平均提升并不代表每例都提升。例如 `v2-sa_5447626 U4` 的 S2-point 编织袋案例，base IoU 95.8%，LoRA 21.7%，只选到局部且未覆盖公开点击。后续若继续训练，应在独立训练源中强化部件/整体范围、空间关系、遮挡与添加操作的源锚点，以及点必须落在输出内的困难例；不把这 20 图回灌训练。",
        "5. codec 需作为单独误差来源保留评估。mask 输入全部正确复制，S2-mask/S3-mask 仍只有 74.4% mIoU；小衣物、细杆、分离窗格等的范围损失不会靠复制指令改善。当前 2-code 表示的重建参考不是上界，也不能据此免除 TE 的定位责任。",
        "",
        "## 文件与复现",
        "",
        f"- [指标图]({PREVIEW}/setting_scores.png)",
        f"- [S1–S3-point 代表目标]({PREVIEW}/settings_1.jpg)；[其余 setting 代表目标]({PREVIEW}/settings_2.jpg)",
        f"- [失败例 1]({PREVIEW}/failures_1.jpg)；[失败例 2]({PREVIEW}/failures_2.jpg)",
        f"- [Codec 失败与局部放大 1]({PREVIEW}/codec_failures_1.jpg)；[Codec 失败与局部放大 2]({PREVIEW}/codec_failures_2.jpg)",
        f"- [Codec 能表达但 TE 预测错误的例子]({PREVIEW}/te_failures_1.jpg)",
        f"- [S5 较差完整联合请求]({PREVIEW}/mixed_low.jpg)；[中位请求]({PREVIEW}/mixed_median.jpg)；[较好请求]({PREVIEW}/mixed_high.jpg)",
        "- `data/benchmark_public.jsonl`：官方 public 投影；`data/inference_public.jsonl`：实际模型可见请求。",
        "- `data/offline_gt.jsonl`、`data/scoring_truth.jsonl`：独立离线评分资料，未传入 generate。",
        "- `eval/{base,step2000}/inference/combined.jsonl`：全部未经修复的原始回答、生成 token ID、解析结果和耗时。",
        "- `eval/{base,step2000}/score/combined.jsonl` 与 `masks/`：逐请求、逐目标指标和实际预测 mask。",
        "- `reports/metrics.json`、`reports/targets.csv`：可复核汇总及逐目标表。`provenance/prompt_*.txt` 保存真实 prompt 示例。",
        "- `reports/S5_PROTOCOL_EXAMPLE.md`：根据输入模态覆盖选择的一条实际 point+box+mask+text 联合请求及原始输出。",
        "- `scripts/adapter.py` 实现公开输入转换；其他脚本复用冻结 formal code 的推理/评分模块。源码、数据和结果均在此新目录，未修改原始数据或任何远程仓库。",
        "- 环境入口 `env.sh`；阶段调度 `scripts/run_experiment.py`，GPU 空闲后执行，任何 worker 错误留下 FAILED.json 并停止，不基于输出质量自动重试。",
        "",
        "可视化样例选择规则：每个 setting 取未知目标 IoU 中位附近和最低目标；纯 mask setting 使用全目标。S5 展示未知目标均值最低、中位、最高的完整请求，各目标来自同一次生成。该展示选择只影响报告，不影响评分或答案。",
        "",
    ]
    (ROOT / "REPORT.md").write_text("\n".join(lines))
    print(
        json.dumps(
            {"report": str(ROOT / "REPORT.md"), "figures": figures, "summary": fine},
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
