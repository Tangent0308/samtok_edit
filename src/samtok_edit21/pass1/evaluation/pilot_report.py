"""Local pilot report and exportable scientific figures from frozen score files."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion

from samtok_edit21.pass1.data.pilot import decode_rle
from samtok_edit21.pass1.evaluation.metrics import describe, paired


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def pct(x):
    return "—" if x is None else f"{100 * x:.2f}%"


def metric(x):
    return "—" if x is None else f"{x:.4f}"


def overlay(ax, image, mask, color, truth=None, crop=None):
    if crop:
        x0, y0, x1, y1 = crop
        image = image[y0:y1, x0:x1]
        mask = mask[y0:y1, x0:x1]
        if truth is not None:
            truth = truth[y0:y1, x0:x1]
    ax.imshow(image)
    rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
    rgba[mask, :3] = color
    rgba[mask, 3] = 0.44
    if truth is not None:
        edge = truth & ~binary_erosion(truth, iterations=2)
        rgba[edge] = [0.1, 1.0, 0.2, 0.95]
    ax.imshow(rgba)
    ax.set_axis_off()


def case_figures(root, base, adapted):
    source = {r["id"]: r for r in rows(root / "data/test.jsonl")}
    bb = {(t["id"], t["label"]): t for r in base for t in r["targets"]}
    aa = {(t["id"], t["label"]): t for r in adapted for t in r["targets"]}
    base_valid = {r["id"]: r["compatible"] for r in base}
    adapter_valid = {r["id"]: r["compatible"] for r in adapted}
    selected = {}
    jobs = []
    colors = {
        "gt": (0.05, 0.85, 0.25),
        "base": (1.0, 0.4, 0.1),
        "adapter": (0.05, 0.65, 1.0),
    }
    for loc in ["none", "point", "box", "mask"]:
        keys = [k for k, t in aa.items() if t["locator"] == loc]
        keys.sort(key=lambda k: (aa[k]["iou"] - bb[k]["iou"], k))
        picked = [keys[-1], keys[len(keys) // 2], keys[0]]
        selected[loc] = [
            {
                "id": k[0],
                "label": k[1],
                "delta_metric": "valid_id_iou",
                "delta": aa[k]["iou"] - bb[k]["iou"],
                "selection": name,
            }
            for k, name in zip(
                picked, ["largest gain", "median delta", "largest regression"]
            )
        ]
        jobs.append(
            (
                loc,
                "Text-only" if loc == "none" else loc.capitalize(),
                picked,
                ["Largest gain", "Median delta", "Largest regression"],
            )
        )
    eligible = [
        r
        for r in adapted
        if len(r["targets"]) == 5
        and {t["locator"] for t in r["targets"]} == {"none", "point", "box", "mask"}
    ]
    if eligible:
        eligible.sort(
            key=lambda r: (
                np.mean([t["iou"] for t in r["targets"] if t["locator"] != "mask"]),
                r["id"],
            )
        )
        mixed = eligible[len(eligible) // 2]
        picked = [(mixed["id"], t["label"]) for t in mixed["targets"]]
        selected["mixed_k5"] = {
            "id": mixed["id"],
            "selection": "median unknown IoU among K5 cases containing all four locator types",
        }
        jobs.append(
            (
                "mixed_k5",
                "Mixed K=5 (median unknown IoU; all four input types)",
                picked,
                [f"Target {i + 1}" for i in range(5)],
            )
        )
        generated = next(
            r
            for r in rows(root / "eval/test_selected/combined.jsonl")
            if r["id"] == mixed["id"]
        )
        (root / "previews/mixed_k5_response.json").write_text(
            json.dumps(
                {
                    "id": mixed["id"],
                    "request": source[mixed["id"]]["request"],
                    "raw": generated["raw"],
                    "generations_per_image": 1,
                },
                indent=2,
            )
            + "\n"
        )
    for filename, heading, picked, kinds in jobs:
        fig, axes = plt.subplots(
            len(picked), 4, figsize=(15, 3.5 * len(picked) + 0.5), layout="constrained"
        )
        for row_index, (key, kind) in enumerate(zip(picked, kinds)):
            row = source[key[0]]
            label = key[1]
            target = next(t for t in row["request"]["targets"] if t["label"] == label)
            loc = target["locator"]
            truth = next(t for t in row["truth"] if t["label"] == label)
            with Image.open(root / "data" / row["image"]) as im:
                image = np.asarray(im.convert("RGB"))
            gt = decode_rle(truth["selection"])
            bm = np.asarray(Image.open(bb[key]["mask_path"])) > 0
            am = np.asarray(Image.open(aa[key]["mask_path"])) > 0
            yy, xx = np.where(gt)
            h, w = gt.shape
            pad = max(24, int(max(xx.max() - xx.min(), yy.max() - yy.min()) * 0.35))
            crop = (
                max(0, int(xx.min()) - pad),
                max(0, int(yy.min()) - pad),
                min(w, int(xx.max()) + pad + 1),
                min(h, int(yy.max()) + pad + 1),
            )
            axes[row_index, 0].imshow(image)
            axes[row_index, 0].set_axis_off()
            if loc == "point":
                x, y = target["region"]["point"]
                axes[row_index, 0].scatter(
                    [x * w / 1000],
                    [y * h / 1000],
                    s=80,
                    c="yellow",
                    edgecolors="black",
                    linewidths=1.5,
                )
            elif loc == "box":
                from matplotlib.patches import Rectangle

                x0, y0, x1, y1 = target["region"]["box"]
                axes[row_index, 0].add_patch(
                    Rectangle(
                        (x0 * w / 1000, y0 * h / 1000),
                        (x1 - x0) * w / 1000,
                        (y1 - y0) * h / 1000,
                        fill=False,
                        edgecolor="yellow",
                        linewidth=2,
                    )
                )
            elif loc == "mask":
                overlay(
                    axes[row_index, 0],
                    image,
                    decode_rle(truth["canonical_dec"]),
                    (0.8, 0.2, 1.0),
                )
            axes[row_index, 0].set_title(
                f"{kind} | {key[0]} / {label}\nK={len(row['truth'])} | {'ref' if target['has_ref'] else 'no-ref'}+{loc} | area {100 * truth['area']:.2f}%",
                fontsize=9,
            )
            for col, mask, c, title in [
                (1, gt, colors["gt"], "Ground truth S"),
                (2, bm, colors["base"], f"Base IoU {bb[key]['iou']:.3f}"),
                (3, am, colors["adapter"], f"LoRA IoU {aa[key]['iou']:.3f}"),
            ]:
                ax = axes[row_index, col]
                overlay(ax, image, mask, c, truth=gt if col > 1 else None)
                valid = (
                    base_valid[key[0]]
                    if col == 2
                    else adapter_valid[key[0]]
                    if col == 3
                    else True
                )
                ax.set_title(
                    title + (" | incomplete protocol" if not valid else ""), fontsize=10
                )
                inset = ax.inset_axes([0.60, 0.015, 0.39, 0.42])
                overlay(inset, image, mask, c, truth=gt if col > 1 else None, crop=crop)
                for spine in inset.spines.values():
                    spine.set_visible(True)
                    spine.set_color("white")
                    spine.set_linewidth(1.5)
            text = "\n".join(textwrap.wrap(target["instruction"], width=60)[:3])
            axes[row_index, 0].text(
                0,
                -0.025,
                text,
                transform=axes[row_index, 0].transAxes,
                fontsize=8,
                va="top",
            )
        caption = (
            "All five targets from the same single model response."
            if filename == "mixed_k5"
            else "Rows chosen by raw valid-ID mask IoU delta, including regressions."
        )
        fig.suptitle(
            f"{heading} | one joint autoregressive response per image\nGreen contour = GT; inset = same target crop. {caption}",
            fontsize=12,
        )
        fig.savefig(root / "previews" / f"cases_{filename}.png", dpi=160)
        plt.close(fig)
    (root / "previews/case_selection.json").write_text(
        json.dumps(selected, indent=2) + "\n"
    )


def charts(root, b, a):
    locs = ["none", "point", "box", "mask"]
    labels = ["Text", "Point", "Box", "Known mask"]
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(10, 4.6), layout="constrained")
    for shift, color, name, s in [
        (-0.24, "#808997", "Base", b),
        (0, "#2588ba", "Pass1 LoRA", a),
    ]:
        vals = [s["by_locator"][k]["compatible_iou"] for k in locs]
        bars = ax.bar(x + shift, vals, width=0.23, label=name, color=color)
        ax.bar_label(bars, fmt="%.3f", fontsize=9, padding=3)
    vals = [a["by_locator"][k]["canonical_iou"] for k in locs]
    bars = ax.bar(
        x + 0.24, vals, width=0.23, label="Canonical token reference", color="#b8cfaa"
    )
    ax.bar_label(bars, fmt="%.3f", fontsize=9, padding=3)
    ax.set_xticks(
        x, [f"{name}\nn={a['by_locator'][key]['n']}" for name, key in zip(labels, locs)]
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean mask IoU (compatibility-normalized protocol)")
    ax.set_title(
        "Held-out test | 128 source images | same prompt / 2 MP / greedy decode"
    )
    ax.legend(loc="upper left", ncol=3, fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    fig.savefig(root / "previews/test_metrics.png", dpi=180)
    plt.close(fig)
    logs = rows(root / "runs/pilot/train.jsonl")
    steps = [r["step"] for r in logs]
    loss = np.array([r["loss"] for r in logs])
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    ax.plot(steps, loss, color="#9dbdce", alpha=0.6, label="Per update")
    ax.plot(
        steps[7:],
        np.convolve(loss, np.ones(8) / 8, mode="valid"),
        color="#187f99",
        label="8-update mean",
    )
    ax.set_xlabel("Optimizer update (64 requests/update)")
    ax.set_ylabel("Grouped assistant CE")
    ax.grid(alpha=0.2)
    ax.legend()
    ax.set_title("Pass1 LoRA pilot: training objective (not a localization metric)")
    fig.savefig(root / "previews/training_loss.png", dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    if (
        not (root / "selection.json").exists()
        or not (root / "eval/COMPLETE.json").exists()
    ):
        raise ValueError("Evaluation is incomplete")
    (root / "previews").mkdir(exist_ok=True)
    base = rows(root / "eval/base_test_scores/combined.jsonl")
    adapted = rows(root / "eval/test_selected_scores/combined.jsonl")
    b = describe(base)
    a = describe(adapted)
    ci = paired(base, adapted)
    selection = read(root / "selection.json")
    manifest = read(root / "data/manifest.json")
    audit = read(root / "runs/pilot/freeze_audit.json")
    env = read(root / "environment.json")
    charts(root, b, a)
    case_figures(root, base, adapted)
    train = rows(root / "runs/pilot/train.jsonl")
    data_audit = read(root / "logs/data_audit.json")
    lines = [
        "# Pass1 混合定位 LoRA 小规模试训结果",
        "",
        f"本地分支：`{env['branch']}`；基于 v4 `{env['base_commit']}`。未提交、未推送。",
        f"实验目录：`{root}`。原始数据/模型只读；新数据、日志、checkpoint、图表均在此目录。",
        "",
        f"按源图隔离：2048 train / 64 dev / 128 test。训练 8192 条请求、15198 个目标、128 次更新；仅训练 174,587,904 个 LM LoRA 参数。开发集选择 step {selection['selected_step']}，随后首次计算测试集指标。",
        "",
        "## 测试结果",
        "",
        "以下定位数值使用 **compatible IoU**：只剥离外层 Markdown fence / 固定 empty-think 前缀，仍严格校验 JSON、ID、顺序、唯一性、字段和 token 范围。不能补目标、改 token、按 GT 重匹配。这样避免把基座的代码块输出习惯当成定位错误。",
        "",
        "| 类型 | 目标数 | Base IoU | LoRA IoU | 配对变化 | 95% CI（按源图 bootstrap） | Canonical token 参考 |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for key, name in [
        ("unknown", "未知目标合计"),
        ("none", "纯文本"),
        ("point", "Point"),
        ("box", "Box"),
        ("mask", "给定 Mask"),
    ]:
        bb = b["unknown"] if key == "unknown" else b["by_locator"][key]
        aa = a["unknown"] if key == "unknown" else a["by_locator"][key]
        p = ci[key]
        lo, hi = p["paired_source_bootstrap_95ci"]
        lines.append(
            f"| {name} | {aa['n']} | {metric(bb['compatible_iou'])} | {metric(aa['compatible_iou'])} | {p['target_delta']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {metric(aa['canonical_iou'])} |"
        )
    lines += [
        "",
        "| 协议/绑定指标 | Base | LoRA |",
        "|---|---:|---:|",
        f"| 严格 JSON/ID/schema 通过率 | {pct(b['strict_rate'])} | {pct(a['strict_rate'])} |",
        f"| 去代码块后的完整协议通过率 | {pct(b['compatible_rate'])} | {pct(a['compatible_rate'])} |",
        f"| 给定 mask 的原始精确复制率 | {pct(b['known']['exact_code'])} | {pct(a['known']['exact_code'])} |",
        f"| 整图所有目标 IoU ≥ .5（compatible） | {pct(b['compatible_all_targets_iou50'])} | {pct(a['compatible_all_targets_iou50'])} |",
        f"| Point 命中率（原始有效 ID span） | {pct(b['point_hit'])} | {pct(a['point_hit'])} |",
        f"| 未知目标 strict end-to-end IoU | {metric(b['unknown']['strict_iou'])} | {metric(a['unknown']['strict_iou'])} |",
        f"| 未知目标有效 ID 解码 IoU（保留部分有效回答） | {metric(b['unknown']['iou'])} | {metric(a['unknown']['iou'])} |",
        "",
        f"只看 K≥2 的混合目标请求，未知目标 compatible IoU：{metric(b['multi_unknown']['compatible_iou'])} → {metric(a['multi_unknown']['compatible_iou'])}（{a['multi_unknown']['n']} 个未知目标）。",
        "",
        "给定 mask 的复制率和 IoU 都来自模型原始回答；没有使用部署 API 的确定性回填来提高这些数字。Canonical token 参考是数据里已有 token 的重建质量，不是数学性能上限。",
        "",
        "![测试定位指标](previews/test_metrics.png)",
        "",
        "## 开发集选择",
        "",
        "| checkpoint | 未知目标 strict IoU | compatible IoU | 严格格式通过率 |",
        "|---|---:|---:|---:|",
    ]
    for s, v in selection["dev_metrics"].items():
        lines.append(
            f"| {s} | {metric(v['strict_iou'])} | {metric(v['compatible_iou'])} | {pct(v['strict_rate'])} |"
        )
    lines += [
        "",
        f"选择规则在训练前固定：{selection['criterion']}。选择时间：{selection['selected_at_utc']}。",
        "训练前生成了基座 dev/test 原始输出，但测试输出的解析统计、解码评分及图像分析均在 checkpoint 选择后执行。没有基于 test 调整 prompt、数据、阈值或挑选 checkpoint。",
        "",
        "## 数据范围与限制",
        "",
        "| split | 独立源图 | 请求 | K=1/2/3/4/5 请求分布 |",
        "|---|---:|---:|---|",
    ]
    for s, v in manifest["summary"].items():
        lines.append(
            f"| {s} | {v['source_groups']} | {v['requests']} | "
            + "/".join(str(v["K"].get(str(k), 0)) for k in range(1, 6))
            + " |"
        )
    lines += [
        "",
        "- 来源为合并数据的 default train：D1/D3/D4/D10–D15；使用 selection S 与已有 canonical token，roundtrip ≥ .5、面积 [.0005,.30]。部分目标组来自同一图像的不同 record，仅作为定位监督，不假设存在联合编辑结果。",
        "- 7 种输入组合均覆盖。实际单目标占训练请求约 46%；四、五目标较少。正式 benchmark 更多四目标和细小部件，不能把此结果视为正式 benchmark 结果。D2 dense pool 尚未纳入本轮。",
        "- 点击在 S 与 canonical 解码区域的交集中选取，且通过 0–1000 坐标回投检查；heldout 采用较深的点。这是一致性较好的 pilot click 分布，并非正式 benchmark 的原始点击分布。",
        "- 同图相同 canonical span 去重，排除了过小/过大和低 roundtrip 的目标；这轮不覆盖全部困难情况。",
        f"- 校验了 {data_audit['verified_image_copies']} 份本地图片的 SHA256 和独立文件 inode；跨 split 图像哈希交集为 0，正式 benchmark source ID/sha256 命中为 0。此检查不声称排除了所有语义近重复图像。",
        "",
        "## 按面积与目标数看未知目标",
        "",
        "| 切片 | 目标数 | Base compatible IoU | LoRA compatible IoU |",
        "|---|---:|---:|---:|",
    ]
    for group in ["by_area_unknown", "by_K_unknown"]:
        for key, v in a[group].items():
            lines.append(
                f"| {group}/{key} | {v['n']} | {metric(b[group][key]['compatible_iou'])} | {metric(v['compatible_iou'])} |"
            )
    lines += [
        "",
        "## 训练与验证",
        "",
        "- r64 / alpha128 / dropout .05；仅 LM q/k/v/o/gate/up/down 共 252 个模块。BF16 基座、FP32 LoRA、SDPA、2MP、assistant-only grouped CE。8 GPU × microbatch 1 × accumulation 8；LR 2e-5，128 updates。",
        f"- 训练耗时（含末尾验证）：{audit['duration_s'] / 60:.1f} 分钟；rank0 峰值 allocated 显存 {max(x['peak_gpu_gb_rank0'] for x in train):.2f} GB。",
        f"- 冻结参数前后 SHA256 一致：`{audit['all_frozen_parameters_unchanged']}`。关闭 LoRA 后 logits 和 TE pre-final-norm 特征均逐元素一致：`{audit['adapter_off_probe']}`。",
        "- 25 条实际 processor 样本覆盖 K=1–5，assistant token 对齐和每目标两个 code 的监督位置通过；26 项相关/文档测试通过（CPU 测试指定 `DIFFSYNTH_ATTENTION_IMPLEMENTATION=torch`）。首次默认 xformers CPU 调度导致两项旧 DiT 测试失败，切换已有 SDPA 开关后通过，未修改 DiT。",
        "- checkpoint 加载先将 LoRA 层建为 FP32，再加载权重，避免途经 BF16 的精度损失。新增回归测试覆盖该问题；真实 504 个 LoRA 张量逐元素重载一致。早期 `dev_step64` 为修复前诊断，已弃用；最终选择使用 `dev_step64_fp32` / `dev_step128_fp32`。",
        "- 点/框/text 输入没有 SAM2 前处理。离线评价才使用冻结 codec 解码；所有方法使用固定 batch=1 的相同 decoder，并缓存同图视觉特征，避免候选数量改变 BF16 batch 数值路径。",
        "- 未运行 DiT 编辑评测；目前验证的是 Pass1 定位、协议和关闭 LoRA 的 TE 一致性。",
        "",
        "![训练曲线](previews/training_loss.png)",
        "",
        "## 可视化",
        "",
        "每种输入按原始有效 ID 的解码 IoU 变化展示收益最大、中位变化、退步最大的目标；选择依据在 `previews/case_selection.json`，不只展示成功例。表格仍使用完整协议 gating 的 compatible IoU。绿色边界是 GT；右下角为相同 GT 区域的局部放大，仅用于阅读。",
        "- [一张图五个混合目标，一次生成](previews/cases_mixed_k5.png)；[该次完整输入和原始回答](previews/mixed_k5_response.json)",
        "- [Point](previews/cases_point.png)",
        "- [Box](previews/cases_box.png)",
        "- [纯文本](previews/cases_none.png)",
        "- [给定 Mask](previews/cases_mask.png)",
        "",
        "## 代码与产物",
        "",
        "- [协议、训练和使用说明](repo/docs/pass1_localization.md)",
        "- [固定试训配置](repo/configs/pass1_pilot.json)",
        "- [数据统计与来源过滤](data/manifest.json)",
        "- [开发集选择记录](selection.json)",
        f"- [选中 LoRA](runs/pilot/step_{selection['selected_step']:04d}/adapter_model.safetensors)",
        "- [测试完整统计与 bootstrap](eval/test_selected_summary.json)",
        "- [基础模型测试统计](eval/base_test_summary.json)",
        "- [冻结参数与 adapter-off 检查](runs/pilot/freeze_audit.json)",
        "- `eval/*/combined.jsonl`：原始输出或逐目标评分；`logs/` 保留全部运行记录。`base_dev_scores` 与 `base_dev_scores_v2` 为早期评分诊断，正式 dev 采用 `base_dev_scores_final`。",
        "",
    ]
    report = "\n".join(lines)
    if (root / "ANALYSIS.md").exists():
        report = report.replace(
            "## 测试结果",
            (root / "ANALYSIS.md").read_text().strip() + "\n\n## 测试结果",
            1,
        )
    (root / "REPORT.md").write_text(report)
    print(
        json.dumps(
            {
                "report": str(root / "REPORT.md"),
                "selected_step": selection["selected_step"],
                "unknown_delta": ci["unknown"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
