"""Summarize frozen dev selection/test results and copy readable previews locally."""

from __future__ import annotations

import collections
import datetime
import json
import os
import shutil
import textwrap
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from samtok_edit21.pass1.data.pilot import decode_rle
from samtok_edit21.pass1.evaluation.metrics import describe, paired
from samtok_edit21.pass1.evaluation.pilot_report import overlay

ROOT = Path(os.environ["PASS1_FORMAL_ROOT"]).expanduser().resolve()
PREVIEW = ROOT / "reports/previews"
DISPLAY = Path(os.environ["PASS1_PREVIEW_ROOT"]).expanduser().resolve()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()]


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def num(v):
    return "—" if v is None else f"{v:.4f}"


def percent(v):
    return "—" if v is None else f"{v * 100:.2f}%"


def cases(base, adapted, predictions):
    sources = {x["id"]: x for x in rows(ROOT / "data/prepared/test.jsonl")}
    original = {(r["id"], t["label"]): t for r in base for t in r["targets"]}
    tuned = {(r["id"], t["label"]): t for r in adapted for t in r["targets"]}
    chosen = {}
    jobs = []
    for loc in ["point", "box", "none", "mask"]:
        keys = sorted(
            [k for k, t in tuned.items() if t["locator"] == loc],
            key=lambda k: (tuned[k]["strict_iou"] - original[k]["strict_iou"], k),
        )
        if not keys:
            continue
        picked = [keys[-1], keys[len(keys) // 2], keys[0]]
        kinds = ["Largest gain", "Median delta", "Largest regression"]
        chosen[loc] = [
            {
                "id": k[0],
                "label": k[1],
                "selection": why,
                "strict_iou_delta": tuned[k]["strict_iou"] - original[k]["strict_iou"],
            }
            for k, why in zip(picked, kinds)
        ]
        jobs.append((loc, loc, picked, kinds))
    mixed = [
        r
        for r in adapted
        if len(r["targets"]) == 5
        and {t["locator"] for t in r["targets"]} == {"none", "point", "box", "mask"}
    ]
    if mixed:
        mixed.sort(
            key=lambda r: (
                np.mean(
                    [t["strict_iou"] for t in r["targets"] if t["locator"] != "mask"]
                ),
                r["id"],
            )
        )
        r = mixed[len(mixed) // 2]
        keys = [(r["id"], t["label"]) for t in r["targets"]]
        chosen["mixed_k5"] = {
            "id": r["id"],
            "selection": "median strict unknown IoU among K5/all-four-modality requests",
        }
        jobs.append(
            (
                "mixed_k5",
                "Mixed K=5, all four input types",
                keys,
                [f"Target {t['label']}" for t in r["targets"]],
            )
        )
        output = next(x for x in rows(predictions) if x["id"] == r["id"])
        write(
            PREVIEW / "mixed_k5_response.json",
            {
                "id": r["id"],
                "request": sources[r["id"]]["request"],
                "raw": output["raw"],
                "generate_calls": 1,
            },
        )
    for name, heading, keys, kinds in jobs:
        fig, axes = plt.subplots(
            len(keys),
            4,
            figsize=(16, 3.7 * len(keys) + 0.5),
            layout="constrained",
            squeeze=False,
        )
        for j, (key, why) in enumerate(zip(keys, kinds)):
            row = sources[key[0]]
            target = next(t for t in row["request"]["targets"] if t["label"] == key[1])
            truth = next(t for t in row["truth"] if t["label"] == key[1])
            gt = decode_rle(truth["selection"])
            with Image.open(row["image"]) as im:
                image = np.asarray(im.convert("RGB"))
            with Image.open(original[key]["mask_path"]) as im:
                bm = np.asarray(im) > 0
            with Image.open(tuned[key]["mask_path"]) as im:
                am = np.asarray(im) > 0
            h, w = gt.shape
            yy, xx = np.where(gt)
            pad = max(24, int(max(np.ptp(xx), np.ptp(yy)) * 0.35))
            crop = (
                max(0, int(xx.min()) - pad),
                max(0, int(yy.min()) - pad),
                min(w, int(xx.max()) + pad + 1),
                min(h, int(yy.max()) + pad + 1),
            )
            ax = axes[j, 0]
            ax.imshow(image)
            ax.set_axis_off()
            loc = target["locator"]
            if loc == "point":
                x, y = target["region"]["point"]
                ax.scatter(
                    [x * w / 1000],
                    [y * h / 1000],
                    s=90,
                    c="yellow",
                    edgecolors="black",
                    linewidths=1.4,
                )
            elif loc == "box":
                x0, y0, x1, y1 = target["region"]["box"]
                ax.add_patch(
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
                overlay(ax, image, decode_rle(truth["canonical_dec"]), (0.8, 0.2, 1.0))
            ax.set_title(
                f"{why} | {key[1]} | {loc}\nK={len(row['truth'])} | area={100 * truth['area']:.3f}%",
                fontsize=10,
            )
            ax.text(
                0,
                -0.04,
                "\n".join(textwrap.wrap(target["instruction"], 55)[:3]),
                transform=ax.transAxes,
                fontsize=8,
                va="top",
            )
            entries = [
                (gt, (0.05, 0.85, 0.25), "Ground truth S"),
                (
                    bm,
                    (1.0, 0.4, 0.1),
                    f"Base strict IoU={original[key]['strict_iou']:.3f}",
                ),
                (
                    am,
                    (0.05, 0.65, 1.0),
                    f"LoRA strict IoU={tuned[key]['strict_iou']:.3f}",
                ),
            ]
            for col, (mask, color, title) in enumerate(entries, 1):
                ax = axes[j, col]
                overlay(ax, image, mask, color, truth=gt if col > 1 else None)
                ax.set_title(title, fontsize=10)
                inset = ax.inset_axes([0.60, 0.015, 0.39, 0.42])
                overlay(
                    inset, image, mask, color, truth=gt if col > 1 else None, crop=crop
                )
                if (
                    col > 1
                    and not (original if col == 2 else tuned)[key]["request_strict"]
                ):
                    ax.text(
                        0.02,
                        0.02,
                        "Protocol incomplete: gated score 0",
                        transform=ax.transAxes,
                        color="white",
                        bbox={"facecolor": "red", "alpha": 0.8},
                        fontsize=8,
                    )
        fig.suptitle(
            heading + " | GT outline: green | consistent GT crop for inspection",
            fontsize=13,
        )
        fig.savefig(PREVIEW / f"cases_{name}.png", dpi=150)
        plt.close(fig)
    write(PREVIEW / "case_selection.json", chosen)


def main():
    PREVIEW.mkdir(parents=True, exist_ok=True)
    DISPLAY.mkdir(parents=True, exist_ok=True)
    selection = read(ROOT / "selection.json")
    data = read(ROOT / "data/prepared/manifest.json")
    material = read(ROOT / "data/materialized/manifest.json")
    artifacts = {
        name: read(ROOT / "eval" / name / "COMPLETE.json")
        for name in ["base_test", "pilot_test", "selected_test"]
    }
    scores = {k: rows(v["scores"]) for k, v in artifacts.items()}
    stats = {k: describe(v) for k, v in scores.items()}
    comparisons = {
        k: paired(scores[k], scores["selected_test"], metric="strict_iou")
        for k in ["base_test", "pilot_test"]
    }
    write(
        ROOT / "reports/test_comparison.json",
        {"metrics": stats, "paired_strict_source_bootstrap": comparisons},
    )
    updates = rows(ROOT / "runs/formal/train.jsonl")
    deduplicated = {r["step"]: r for r in updates}
    updates = [deduplicated[k] for k in sorted(deduplicated)]
    candidates = read(ROOT / "eval/dev_candidates.json")
    audits = [
        read(p) for p in sorted((ROOT / "runs/formal").glob("freeze_audit_*.json"))
    ]
    integration = read(ROOT / "reports/shared_te_audit.json")
    all_frozen = all(
        x["all_frozen_parameters_unchanged"]
        and all(z["equal"] for z in x["adapter_off_probe"].values())
        for x in audits
    )
    fig, ax = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    x = np.array([r["step"] for r in updates])
    y = np.array([r["loss"] for r in updates])
    ax[0].plot(x, y, alpha=0.25, label="update loss")
    if len(y) >= 25:
        ax[0].plot(
            x[24:], np.convolve(y, np.ones(25) / 25, "valid"), label="25-update average"
        )
    ax[0].set(xlabel="Optimizer update", ylabel="Grouped CE")
    ax[0].legend()
    ax[1].plot(x, [r["lr"] for r in updates])
    ax[1].set(xlabel="Optimizer update", ylabel="Learning rate")
    ax[2].plot(
        [r["step"] for r in candidates],
        [r["unknown_strict_iou"] for r in candidates],
        marker="o",
    )
    ax[2].axvline(
        selection["selected_step"], color="red", linestyle=":", label="Selected on dev"
    )
    ax[2].set(xlabel="Optimizer update", ylabel="Dev unknown strict IoU")
    ax[2].legend()
    for a in ax[:2]:
        a.axvline(data["warmup_steps"], color="grey", linestyle="--")
    fig.savefig(PREVIEW / "training_and_dev.png", dpi=160)
    fig.savefig(PREVIEW / "training_and_dev.pdf")
    plt.close(fig)
    cases(
        scores["base_test"],
        scores["selected_test"],
        artifacts["selected_test"]["predictions"],
    )
    a = stats["selected_test"]
    b = stats["base_test"]
    p = stats["pilot_test"]
    lines = [
        "# Pass1 正式定位 LoRA 实验",
        "",
        f"运行目录：`{ROOT}`。本报告在完整 dev 选择和 test 评估完成后生成。",
        f"选中 **step {selection['selected_step']}**；选择指标为 dev 未知区域目标的 strict IoU（{selection['selected_dev_metric']:.4f}），未使用 test 选择模型。",
        "",
        "## 固定测试集结果",
        "",
        "| 指标 | 原始 SAMTok | 之前 pilot step128 | 本次正式 LoRA |",
        "|---|---:|---:|---:|",
    ]
    for label, fn, fmt in [
        ("完整严格协议", lambda z: z["strict_rate"], percent),
        ("未知目标 strict IoU", lambda z: z["unknown"]["strict_iou"], num),
        ("未知目标 compatible IoU", lambda z: z["unknown"]["compatible_iou"], num),
        ("多目标未知区域 strict IoU", lambda z: z["multi_unknown"]["strict_iou"], num),
        ("给定 mask 的 code 精确复制率", lambda z: z["known"]["exact_code"], percent),
        ("整条请求所有目标 IoU≥0.5", lambda z: z["all_targets_iou50"], percent),
    ]:
        lines.append(
            "| " + label + " | " + " | ".join(fmt(fn(z)) for z in [b, p, a]) + " |"
        )
    for loc in ["point", "box", "none", "mask"]:
        lines.append(
            "| "
            + loc
            + " strict IoU | "
            + " | ".join(
                num(z["by_locator"].get(loc, {}).get("strict_iou")) for z in [b, p, a]
            )
            + " |"
        )
    lines += [
        "",
        "strict IoU 将整条请求的协议失败计为 0；不进行 GT 匹配、候选 oracle 选择或生成后修复。一次 generate 联合输出所有目标。",
        "",
        "| 正式 LoRA 相对基线 | 未知目标 strict IoU 增量 | 按源图配对 bootstrap 95% CI |",
        "|---|---:|---:|",
    ]
    for name, comp in comparisons.items():
        v = comp["unknown"]
        lo, hi = v["paired_source_bootstrap_95ci"]
        lines.append(f"| {name} | {v['target_delta']:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    lines += [
        "",
        "## 数据、训练与完整性",
        "",
        f"- 实际训练 {data['training_requests']:,} 条请求、{data['train_sources']:,} 张源图；warmup {data['warmup_steps']} updates + main {data['main_steps']} updates。实际 dev/test 请求数分别为 {data['context']['dev']['requests']}/{data['context']['test']['requests']}。",
        f"- 全量展开接收 {material['accepted_requests']:,}/{material['planned_requests']:,} 条请求；过滤 {material['planned_requests'] - material['accepted_requests']} 条，原因见 data/materialized/manifest.json。反事实有效对数 {material['valid_counterfactual_pairs']:,}。",
        "- 每条训练请求恰好使用一次；反事实伙伴在同一全局 batch；末尾 padding 的损失为 0，按真实样本数归一化。原始数据未修改。",
        "- 8×H100；有效 batch 64；r64/alpha128/dropout .05，252 个 LM 投影模块；冻结 BF16 基座，FP32 LoRA。assistant-only grouped CE，预测 code/copy/format 权重 1/.25/.25。",
        "- 全程 LR 2e-5，3% update 线性 warmup + 连续 cosine；数据阶段切换不重置 LR、优化器或随机状态。100 步完整保存；从全新 LoRA 开始，未继承 pilot/smoke 权重。",
        "- 恢复功能验证通过，FP32 adapter、优化器和各 rank RNG 均保存。额外连续训练对照的严格逐位容差检查未通过：两个新运行在第一次更新已出现 GPU 反向数值差异，尽管前向 loss 和 RNG 状态相同；恢复后 RNG 推进保持一致。不承诺完整训练权重逐位重放；原始诊断保留在 reports/resume_equivalence*.json。",
        f"- 上下文上限 8192，原生 processor 最大 2MP；实际最大上下文 {max(v['max'] for v in data['context'].values())} tokens；rank0 峰值 allocated 显存 {max(r['peak_gpu_gb_rank0'] for r in updates):.2f} GB。",
        f"- 各训练段冻结参数/adapter-off 精确一致性全部通过：{all_frozen}；真实 shared-TE 集成后关闭 LoRA，pre-norm/norm 特征逐元素相同：{all(v['equal'] for v in integration['pass2_features'].values())}。",
        "- point/box/text 直接进入 MLLM；未使用 SAM 把提示转换为 mask。已给 mask 使用已有 codec span；评分阶段才用冻结 codec 解码生成 token。",
        "- dev/test 点只根据 GT S 采样，保留 teacher 未覆盖的点；不能把本次数字直接与此前较容易的 pilot 测试集横向比较，表中 pilot 已在本次固定 test 上重新评估。",
        "- 尚未运行最终论文 benchmark 的 DiT 编辑评测；这里衡量 Pass1 定位、协议遵循与关闭 LoRA 后 TE 一致性。",
        "",
        "## 分组结果",
        "",
        "| 未知目标分组 | N | Base strict IoU | Pilot strict IoU | 正式 strict IoU |",
        "|---|---:|---:|---:|---:|",
    ]
    for group in ["by_K_unknown", "by_operation_unknown", "by_area_unknown"]:
        for key, v in a[group].items():
            lines.append(
                f"| {group}/{key} | {v['n']} | {num(b[group][key]['strict_iou'])} | {num(p[group][key]['strict_iou'])} | {num(v['strict_iou'])} |"
            )
    failures = collections.Counter(
        t["locator"]
        for r in scores["selected_test"]
        for t in r["targets"]
        if t["locator"] != "mask" and t["strict_iou"] < 0.5
    )
    lines += [
        "",
        f"未知目标 strict IoU<0.5 的数量按提示形式统计：`{dict(failures)}`。完整逐目标评分和配对 CI 见 `reports/test_comparison.json`。",
        "",
        "## 可视化与复现",
        "",
        f"面向用户的图和 PDF 已复制到 `{DISPLAY}`。原始图和日志留在本实验目录。",
        "- `reports/previews/training_and_dev.png`：训练曲线及 dev 选择。",
        "- `reports/previews/cases_*.png`：每类按 strict IoU 变化选择最大收益、中位变化、最大退步；K=5 混合例选择未知目标分数中位样本。选例记录与原始回答同步保存。",
        "- `selection.json`：候选 dev 指标与选定检查点。",
        "- `configs/run.json`、`configs/resolved_data.json`、`provenance/`：固定配置、实际数据、源码快照与版本。",
        "- `eval/*/COMPLETE.json`：每次评估的原始输出、评分、统计文件路径。",
        "- `runs/formal/step_*/`：FP32 LoRA、优化器、各 rank RNG 与 COMPLETE 标志。",
        "- `scripts/workflow.py`：durable controller；`logs/` 保存每个子任务命令、退出状态和原始输出。",
        "",
    ]
    (ROOT / "REPORT.md").write_text("\n".join(lines))
    for source in PREVIEW.iterdir():
        if source.is_file():
            shutil.copy2(source, DISPLAY / source.name)
    shutil.copy2(ROOT / "REPORT.md", DISPLAY / "REPORT.md")
    write(
        ROOT / "reports/delivery.json",
        {
            "display_directory": str(DISPLAY),
            "files": [str(p) for p in DISPLAY.iterdir()],
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
    )
    print(
        json.dumps(
            {
                "report": str(ROOT / "REPORT.md"),
                "display_directory": str(DISPLAY),
                "selected_step": selection["selected_step"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
