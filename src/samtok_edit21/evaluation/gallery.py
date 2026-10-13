"""Side-by-side gallery: trained model vs base model on real edit instructions (v3).

    python -m torch.distributed.run --nproc_per_node 8 -m samtok_edit21 ground-gallery generate \\
        --adapter <adapter_dir> --output <compare_dir>
    samtok-edit ground-gallery html --records <compare_dir> --output <report_dir> --name E-A2

``generate`` runs every instruction twice with identical inputs - once with the DiT LoRA and code injector switched
off (base) and once on (trained).  Both see only the source image and the prompt containing the region token.
``html`` writes ``index.html`` (images next to it) and one self-contained file with the images embedded.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from samtok_edit21.preparation.grounding_data import EVAL_PROTOCOL, STAGE2_DATA, heldout, single_token_rows

SECTIONS = {"dev": "benchmark dev：真实编辑指令", "heldout": "预留集：真实编辑指令（带标准答案）",
            "add_box": "add（框 token，未训练，仅作对照）"}
TYPES = {"remove": "删除", "attribute": "属性", "replace": "替换", "action": "动作", "text": "文字", "add": "新增"}
OUTLINE = (255, 0, 255)
EMBED_WIDTH, EMBED_QUALITY = 400, 70


def gallery_jobs(eval_protocol, stage2_data, dev_cases, per_type, add_cases):
    """Benchmark dev cases with and without the phrase, held-out rows per edit type, and a few box-token adds."""
    protocol = Path(eval_protocol)
    compiled = {row["case_id"]: row for row in map(json.loads, open(protocol / "compiled.jsonl"))}
    cases = [json.loads(text) for text in open(protocol / "cases.jsonl")]
    jobs = []
    dev = [c for c in cases if c["split"] == "dev" and compiled[c["case_id"]]["region_kind"] == "mask"][:dev_cases]
    for case in dev:
        entry = compiled[case["case_id"]]
        for variant in ("ref", "noref"):
            template = entry.get("ref_template") if variant == "ref" else entry["noref_template"]
            if template:
                jobs.append({"section": "dev", "case": case["case_id"], "variant": variant,
                             "edit_type": entry["edit_type"], "source": case["source_image"],
                             "mask": case["region"]["mask"], "template": template, "instruction": entry["instruction"]})
    groups = {}
    for _, row, _ in single_token_rows(stage2_data, include_add=True):
        if heldout(row["edit_image"]):
            groups.setdefault((row["edit_type"], row.get("instr_variant")), []).append(row)
    rng = random.Random(7)
    for (edit_type, variant), rows in sorted(groups.items()):
        for row in rng.sample(rows, min(per_type, len(rows))):
            jobs.append({"section": "heldout", "case": Path(row["edit_image"]).stem[:40], "variant": variant,
                         "edit_type": edit_type, "source": row["edit_image"], "target": row["image"],
                         "prompt": row["prompt"]})
    adds = [c for c in cases if c["split"] == "dev" and compiled[c["case_id"]]["region_kind"] != "mask"
            and compiled[c["case_id"]]["edit_type"] == "add"][:add_cases]
    for case in adds:
        entry = compiled[case["case_id"]]
        jobs.append({"section": "add_box", "case": case["case_id"], "variant": "ref", "edit_type": "add",
                     "source": case["source_image"], "mask": case["region"].get("mask"), "box": case["region"].get("box"),
                     "template": entry.get("ref_template") or entry["noref_template"], "instruction": entry["instruction"]})
    return jobs


def outlined(image, region, width=5):
    mask = Image.fromarray((np.asarray(region) > 0).astype(np.uint8) * 255)
    edge = np.asarray(mask.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(width))) > 0
    pixels = np.asarray(image.convert("RGB")).copy()
    pixels[edge] = OUTLINE
    return Image.fromarray(pixels)


def shrink(image, width=560):
    scale = min(1.0, width / image.width)
    if scale == 1:
        return image
    return image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)


def generate(args):
    import torch
    import torch.distributed as dist

    from samtok_edit21.data.protocol import SPAN_RE, relative_box
    from samtok_edit21.evaluation.grounding import generate as run_model
    from samtok_edit21.models.code_injector import trained_weights
    from samtok_edit21.models.codec import load_codec
    from samtok_edit21.models.pipeline import load_pipeline
    from samtok_edit21.training.grounding import load_grounding_adapter

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    device = f"cuda:{local}"
    out = Path(args.output)
    (out / "img").mkdir(parents=True, exist_ok=True)
    pipe = load_pipeline(args.qwen, args.samtok, device=device)
    codec = load_codec(args.samtok, device)
    load_grounding_adapter(pipe, codec, args.adapter)
    pipe.eval()
    jobs = gallery_jobs(args.eval_protocol, args.stage2_data, args.dev_cases, args.per_type, args.add_cases)
    mine = list(enumerate(jobs))[rank::world]
    records, started = [], time.time()
    for done, (index, job) in enumerate(mine):
        try:
            source = Image.open(job["source"]).convert("RGB")
            if "prompt" in job:
                prompt, span = job["prompt"], SPAN_RE.search(job["prompt"]).group(0)
                region = codec.decode_strict(source.convert("RGBA"), span)[0] > 0
            elif job["section"] == "add_box":
                x1, y1, x2, y2 = job["box"]
                box = relative_box((x1, y1, x2, y2), source.width, source.height)
                span = f"<|box_start|>[{box[0]}, {box[1]}, {box[2]}, {box[3]}]<|box_end|>"
                region = np.zeros((source.height, source.width), bool)
                region[y1:y2, x1:x2] = True
                prompt = job["template"].replace("{region}", span)
            else:
                region = np.asarray(Image.open(job["mask"]).convert("L")) > 0
                span = codec.encode(source.convert("RGBA"), [region])[0][0]
                prompt = job["template"].replace("{region}", span)
            record = {**{k: v for k, v in job.items() if k != "box"}, "idx": index, "prompt": prompt, "span": span,
                      "area": float(region.mean())}
            shrink(outlined(source, region)).save(out / "img" / f"{index:04d}_src.jpg", quality=86)
            if job.get("target"):
                with Image.open(job["target"]) as target:
                    shrink(target.convert("RGB")).save(out / "img" / f"{index:04d}_gt.jpg", quality=86)
            before = np.asarray(source, np.float32)
            near = np.asarray(Image.fromarray((region > 0).astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(31))) > 0
            far = ~near
            for model in ("base", "trained"):
                with trained_weights(pipe, model == "trained"):
                    image, _ = run_model(pipe, codec, prompt, source, steps=args.steps)
                shrink(image).save(out / "img" / f"{index:04d}_{model}.jpg", quality=86)
                change = np.abs(np.asarray(image, np.float32) - before).mean(-1)
                record[f"{model}_in"] = float(change[region > 0].mean()) if region.any() else 0.0
                record[f"{model}_out"] = float(change[far].mean()) if far.any() else 0.0
            records.append(record)
        except Exception as error:
            print(f"[rank {rank}] skip {job.get('case')}: {error!r}"[:300], flush=True)
        if rank == 0 and done % 10 == 0:
            print(json.dumps({"done": done, "of": len(mine), "sec": round(time.time() - started)}), flush=True)
    gathered = [None] * world
    dist.all_gather_object(gathered, records)
    if rank == 0:
        merged = sorted((record for part in gathered for record in part), key=lambda record: record["idx"])
        json.dump({"steps": args.steps, "adapter": str(args.adapter), "records": merged},
                  open(out / "records.json", "w"), ensure_ascii=False)
        (out / "GEN_DONE").write_text(str(len(merged)))
        print("wrote", len(merged), "records", flush=True)
    dist.barrier()
    dist.destroy_process_group()


# ----------------------------------------------------------------------------------------------- html

def highlight_examples(data_dir, trained_eval, base_eval, image_dir, counts=(("ho", "noref", 18), ("dev", "noref", 12), ("dev", "ref", 6))):
    """Sample highlight probes present in both evaluations and copy their renders next to the page."""
    from samtok_edit21.evaluation.grounding import metric_key
    from samtok_edit21.evaluation.grounding_report import by_index, is_leaky, load_eval

    items = json.load(open(Path(data_dir) / "eval.json"))
    trained, base = by_index(load_eval(trained_eval)), by_index(load_eval(base_eval))
    rng, examples = random.Random(3), []
    for name, variant, count in counts:
        pool = [i for i, item in enumerate(items) if item["set"] == name and item["variant"] == variant
                and i in trained and i in base and not is_leaky(item)]
        for index in sorted(rng.sample(pool, min(count, len(pool))), key=lambda i: trained[i]["area"]):
            item = items[index]
            for label, root in (("base", base_eval), ("trained", trained_eval)):
                source = Path(root) / "images" / item["set"] / f"{index:05d}.jpg"
                target = Path(image_dir) / f"hl{index:05d}_{label}.jpg"
                if source.exists() and not target.exists():
                    shutil.copy(source, target)
            examples.append({"idx": index, "set": item["set"], "variant": item["variant"],
                             "edit_type": trained[index].get("edit_type", ""), "area": trained[index]["area"],
                             "base_iou": base[index][metric_key(base[index])],
                             "trained_iou": trained[index][metric_key(trained[index])], "phrase": item.get("phrase")})
    return examples


def escape(value):
    return html.escape("" if value is None else str(value))


def mark_tokens(prompt):
    text = escape(prompt)
    for opening, closing in (("&lt;|mt_start|&gt;", "&lt;|mt_end|&gt;"), ("&lt;|box_start|&gt;", "&lt;|box_end|&gt;")):
        text = text.replace(opening, "<span class='tk'>" + opening).replace(closing, closing + "</span>")
    return text


CSS = """:root{--bd:#e2e2e6;--mut:#666}
*{box-sizing:border-box}
body{font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;padding:20px;max-width:1700px;margin:0 auto;color:#1a1a1a;background:#fafafa}
h1{font-size:22px;margin:0 0 6px} h2{font-size:18px;margin:34px 0 8px;border-left:4px solid #c33;padding-left:9px}
.intro{background:#fff;border:1px solid var(--bd);border-radius:8px;padding:14px 16px;margin:12px 0 18px;font-size:13.5px;line-height:1.8}
code{background:#f2f2f5;border-radius:4px;padding:1px 5px;font-size:12.5px;word-break:break-all}
.tk{background:#ffe9b0;border-radius:3px;padding:1px 2px;font-weight:600}
table{border-collapse:collapse;font-size:13px;background:#fff} th,td{border:1px solid var(--bd);padding:5px 10px;text-align:right}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}
.bar{position:sticky;top:0;background:#fafafaee;backdrop-filter:blur(4px);padding:10px 0;border-bottom:1px solid var(--bd);z-index:9;margin-bottom:10px}
.bar button{font:inherit;font-size:12.5px;padding:4px 11px;margin:2px;border:1px solid var(--bd);background:#fff;border-radius:14px;cursor:pointer}
.bar button.on{background:#1a1a1a;color:#fff;border-color:#1a1a1a}
.card{background:#fff;border:1px solid var(--bd);border-radius:8px;padding:12px 14px;margin:12px 0;content-visibility:auto;contain-intrinsic-size:auto 420px}
.hd{display:flex;gap:7px;align-items:center;flex-wrap:wrap;margin-bottom:7px}
.badge{font-size:12px;background:#eef0f4;border-radius:11px;padding:2px 9px}
.b-remove{background:#fde2e2}.b-attribute{background:#e2eefd}.b-replace{background:#e8f6e2}.b-action{background:#f6eee2}.b-text{background:#efe2f6}.b-add{background:#fdf4d0}
.cid{color:var(--mut);font-size:12px;margin-left:auto}
.prompt{margin:6px 0}.lbl{display:inline-block;font-size:11.5px;color:var(--mut);margin-right:7px}
.met{font-size:12.5px;color:#333;margin:7px 0}
.row{display:flex;gap:9px;flex-wrap:wrap}figure{margin:0;flex:1 1 300px;max-width:430px}
figure img{width:100%;min-height:90px;border:1px solid var(--bd);border-radius:5px;display:block;background:#f3f3f5}
figcaption{font-size:12px;color:var(--mut);margin-top:4px;line-height:1.45}.sub{font-size:11px}
.note{font-size:13px;color:#333;margin:4px 0 10px}
@media(max-width:800px){figure{flex:1 1 100%}}"""

SCRIPT = """const secs={"all":"全部","dev":"benchmark dev","heldout":"预留集","add_box":"add（框 token）","highlight":"指认任务"};
const types={"all":"全部类型","remove":"删除","attribute":"属性","replace":"替换","action":"动作","text":"文字","add":"新增","other":"其他"};
let fs="all",ft="all";
function mk(id,obj,cur,set){document.getElementById(id).innerHTML=Object.entries(obj).map(([k,v])=>
 `<button data-k="${k}" class="${k===cur?'on':''}">${v}</button>`).join("");
 document.getElementById(id).querySelectorAll("button").forEach(b=>b.onclick=()=>{set(b.dataset.k);draw();});}
function draw(){mk("fsec",secs,fs,v=>fs=v);mk("ftype",types,ft,v=>ft=v);
 document.querySelectorAll(".card").forEach(c=>{
  const ok=(fs==="all"||c.dataset.sec===fs)&&(ft==="all"||c.dataset.type===ft);
  c.style.display=ok?"":"none";});
 document.querySelectorAll("h2").forEach(h=>{let n=h.nextElementSibling,shown=false;
  while(n&&n.tagName!=="H2"){if(n.classList.contains("card")&&n.style.display!=="none")shown=true;n=n.nextElementSibling;}
  h.style.display=shown?"":"none";
  if(h.nextElementSibling&&h.nextElementSibling.classList.contains("note"))h.nextElementSibling.style.display=shown?"":"none";});}
draw();"""


def build_html(args):
    records_dir, out = Path(args.records), Path(args.output)
    (out / "img").mkdir(parents=True, exist_ok=True)
    payload = json.load(open(records_dir / "records.json"))
    records, name = payload["records"], args.name
    for path in (records_dir / "img").iterdir():
        if not (out / "img" / path.name).exists():
            shutil.copy(path, out / "img" / path.name)
    examples = []
    if args.highlight_data and args.highlight_eval and args.highlight_base:
        examples = highlight_examples(args.highlight_data, args.highlight_eval, args.highlight_base, out / "img")
    embedded = {}

    def embed(file):
        if file not in embedded:
            image = Image.open(out / "img" / file).convert("RGB")
            scale = min(1.0, EMBED_WIDTH / image.width)
            if scale < 1:
                image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, "WEBP", quality=EMBED_QUALITY, method=4)
            embedded[file] = "data:image/webp;base64," + base64.b64encode(buffer.getvalue()).decode()
        return embedded[file]

    def figure(file, caption, inline):
        """One image cell; the browser lazy-loads it natively, with or without JavaScript."""
        if inline:
            return f"<figure><img loading='lazy' decoding='async' src=\"{embed(file)}\" alt=''><figcaption>{caption}</figcaption></figure>"
        return (f"<figure><a href='img/{file}' target='_blank'><img loading='lazy' decoding='async' src=\"img/{file}\" alt=''></a>"
                f"<figcaption>{caption}</figcaption></figure>")

    def card(record, inline):
        index = record["idx"]
        cells = [(f"{index:04d}_src.jpg", "源图 + token 指向的区域<br><span class='sub'>（粉色轮廓仅供查看，未输入模型）</span>")]
        if (out / "img" / f"{index:04d}_gt.jpg").exists():
            cells.append((f"{index:04d}_gt.jpg", "标准答案（数据集目标图）"))
        cells += [(f"{index:04d}_base.jpg", "Base：原版 Qwen-Image-2.1"), (f"{index:04d}_trained.jpg", f"{escape(name)}：训练后")]
        variant = "有物体名" if record["variant"] == "ref" else "无物体名"
        kind = record["edit_type"]
        return (f"<div class='card' data-sec='{record['section']}' data-type='{kind}'>"
                f"<div class='hd'><span class='badge b-{kind}'>{TYPES.get(kind, kind)}</span><span class='badge'>{variant}</span>"
                f"<span class='badge'>区域面积 {record['area'] * 100:.2f}%</span><span class='cid'>{escape(record['case'])}</span></div>"
                f"<div class='prompt'><span class='lbl'>输入给模型的指令</span><code>{mark_tokens(record['prompt'])}</code></div>"
                f"<div class='met'>区域内改动 base {record['base_in']:.0f} → {escape(name)} {record['trained_in']:.0f}　|　"
                f"区域外改动 base {record['base_out']:.1f} → {escape(name)} <b>{record['trained_out']:.1f}</b></div>"
                f"<div class='row'>{''.join(figure(file, caption, inline) for file, caption in cells)}</div></div>")

    def highlight_card(example, inline):
        index = example["idx"]
        cells = "".join(figure(f"hl{index:05d}_{label}.jpg", caption, inline)
                        for label, caption in (("base", "Base：原版"), ("trained", f"{escape(name)}：训练后")))
        variant = "有物体名" if example["variant"] == "ref" else "无物体名"
        reference = f"{escape(example['phrase'])} &lt;M&gt;" if example.get("phrase") else "this region &lt;M&gt;"
        return (f"<div class='card' data-sec='highlight' data-type='{example.get('edit_type') or 'other'}'>"
                f"<div class='hd'><span class='badge'>指认任务</span><span class='badge'>{variant}</span>"
                f"<span class='badge'>区域面积 {example['area'] * 100:.2f}%</span><span class='cid'>{example['set']} #{index}</span></div>"
                f"<div class='prompt'><span class='lbl'>输入给模型的指令</span>"
                f"<code>Apply a {{颜色}} semi-transparent mask to <span class='tk'>{reference}</span> and keep everything else unchanged.</code></div>"
                f"<div class='met'>与 token 所指区域的 IoU：base {example['base_iou']:.2f} → {escape(name)} <b>{example['trained_iou']:.2f}</b></div>"
                f"<div class='row'>{cells}</div></div>")

    sections = {}
    for record in records:
        sections.setdefault(record["section"], []).append(record)
    mean = lambda rows, key: float(np.mean([row[key] for row in rows]))  # noqa: E731
    table = ""
    for section in ("dev", "heldout", "add_box"):
        for kind in sorted({row["edit_type"] for row in sections.get(section, [])}):
            rows = [row for row in sections[section] if row["edit_type"] == kind]
            table += (f"<tr><td>{SECTIONS[section]}</td><td>{TYPES.get(kind, kind)}</td><td>{len(rows)}</td>"
                      f"<td>{mean(rows, 'base_out'):.2f}</td><td><b>{mean(rows, 'trained_out'):.2f}</b></td>"
                      f"<td>{mean(rows, 'base_in'):.0f}</td><td>{mean(rows, 'trained_in'):.0f}</td></tr>")

    def page(inline):
        body = ""
        for section in ("dev", "heldout", "add_box"):
            rows = sections.get(section)
            if rows:
                body += (f"<h2>{SECTIONS[section]}</h2><p class='note'>n={len(rows)}　区域外改动：base {mean(rows, 'base_out'):.2f} → "
                         f"{escape(name)} <b>{mean(rows, 'trained_out'):.2f}</b>　区域内改动：base {mean(rows, 'base_in'):.0f} → "
                         f"{escape(name)} {mean(rows, 'trained_in'):.0f}</p>" + "".join(card(row, inline) for row in rows))
        if examples:
            body += ("<h2>指认任务：模型自己把 token 指的区域涂出来</h2><p class='note'>直接检验模型是否读懂了 token，与编辑质量无关。</p>"
                     + "".join(highlight_card(example, inline) for example in examples))
        note = ("本文件已把全部图片内嵌，可单独下载、离线打开；图片为缩略图（宽 400 px）。原始分辨率版本是同目录的 <code>index.html</code>（需连同 <code>img/</code> 一起拷贝）。"
                if inline else "本页图片在同目录的 <code>img/</code> 下，拷贝时要连目录一起拷；单文件版是 <code>single_file.html</code>。")
        return f"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(name)} 对比：训练后模型 vs 原版</title><style>{CSS}</style></head><body>
<h1>{escape(name)} 对比：训练后模型 vs 原版 Qwen-Image-2.1</h1>
<div class="intro">
<b>推理方式（两个模型完全相同，唯一差别是权重）</b><br>
输入只有两样：<b>源图</b> 和 <b>一句包含 mask token 的指令</b>。mask token 是 SAMTok 把区域编码成的 4 个 token，例如
<code><span class="tk">&lt;|mt_start|&gt;&lt;|mt_0097|&gt;&lt;|mt_0411|&gt;&lt;|mt_end|&gt;</span></code>，直接写在指令里，和普通文字一起过文本编码器进入 DiT。<br>
<b>没有</b>把 token 解码成像素 mask，<b>没有</b>注意力区域偏置，<b>没有</b> latent 融合，<b>没有</b>放大编辑。每张图上的粉色轮廓只是为了让你看出 token 指的是哪里，模型看不到它。<br>
采样：{payload['steps']} 步，CFG 1，seed 0，画布约 1MP。文本编码器两边都是 SAMTok 版。训练后模型 = 原版 DiT + LoRA + 码本向量注入，权重在
<code>{escape(payload['adapter'])}</code>。<br>
<b>指标</b>：区域内/区域外改动 = 输出与源图的平均像素差（0–255）。理想情况是区域内大、区域外小。<br>{note}
</div>
<h2>总览</h2>
<table><tr><th>分组</th><th>编辑类型</th><th>例数</th><th>区域外改动 base</th><th>区域外 {escape(name)}</th><th>区域内 base</th><th>区域内 {escape(name)}</th></tr>{table}</table>
<div class="bar"><b style="font-size:13px">筛选：</b><span id="fsec"></span><br><span id="ftype"></span></div>
{body}
<script>{SCRIPT}</script></body></html>"""

    (out / "index.html").write_text(page(False), encoding="utf-8")
    single = out / "single_file.html"
    single.write_text(page(True), encoding="utf-8")
    print("wrote", out / "index.html")
    print("wrote", single, f"{single.stat().st_size / 1e6:.0f} MB, {len(records) + len(examples)} cards, {len(embedded)} images embedded")


def parser():
    from samtok_edit21.models.pipeline import DEFAULT_QWEN, DEFAULT_SAMTOK

    p = argparse.ArgumentParser(prog="samtok-edit ground-gallery", description=__doc__.split("\n\n")[0])
    commands = p.add_subparsers(dest="action", required=True)
    gen = commands.add_parser("generate", help="Generate base and trained outputs (torchrun)")
    gen.add_argument("--adapter", required=True)
    gen.add_argument("--output", required=True)
    gen.add_argument("--qwen", default=DEFAULT_QWEN)
    gen.add_argument("--samtok", default=DEFAULT_SAMTOK)
    gen.add_argument("--eval-protocol", default=EVAL_PROTOCOL)
    gen.add_argument("--stage2-data", default=STAGE2_DATA)
    gen.add_argument("--dev-cases", type=int, default=999)
    gen.add_argument("--per-type", type=int, default=15, help="Held-out rows per (edit type, variant)")
    gen.add_argument("--add-cases", type=int, default=10)
    gen.add_argument("--steps", type=int, default=40)
    page = commands.add_parser("html", help="Build the pages from a generate output")
    page.add_argument("--records", required=True, help="Directory with records.json and img/")
    page.add_argument("--output", required=True)
    page.add_argument("--name", default="trained")
    page.add_argument("--highlight-data", help="Data directory (eval.json) for the highlight examples")
    page.add_argument("--highlight-eval", help="Evaluation directory of the trained model")
    page.add_argument("--highlight-base", help="Evaluation directory of the base model")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    return generate(args) if args.action == "generate" else build_html(args)


if __name__ == "__main__":
    main()
