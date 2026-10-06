# SAMTokEdit（Qwen-Image-2.1）v2 训练数据盘点

本文记录 v2 训练数据：来源、相对 v1 的转换、最终文件、统计、行格式、校验和复现命令。四个源数据集的筛选、9B noref 转换和 v1 corpus 构建过程不变，详见 v1 存档 [训练数据盘点](archive/v1/04_SAMTokEdit_Qwen21_训练数据盘点.md)。转换代码说明见[代码实现说明第 3 节](01_SAMTokEdit_Qwen21_代码实现说明.md#3-训练数据转换v1--v2)。

## 1. 结论

| 项 | 值 |
|---|---|
| 数据目录 | `/mnt/bn/strategy-mllm-train/intern/users/tanyue/experiments/SAMTokEdit/qwen21_v2/data/train_v2_box_001/` |
| 来源 | v1 `train_full_9b_rules_003`（98,574 个源编辑对，按计划只做格式转换和剔除） |
| 源编辑对 | 保留 97,368；剔除 composite 1,206 |
| Stage 1（`stage1.jsonl`） | 110,079 行：edit_ntp 96,319 + rec_ntp 13,760 |
| Stage 2（`stage2.jsonl`） | 290,006 行：edit 97,368 + edit_umt ref 96,319 + edit_umt noref 96,319 |
| 生成代码 | commit `c411c1c`，`python -m samtok_edit21.preparation.v2_data`，约 3.5 分钟 |
| 校验 | 每条 v1 行与 v1 provenance 的 row hash 一致后才使用；全部输出行通过 v2 `validate_row`；同一命令两次运行输出逐字节一致 |

数据沿用 v1 的已知问题（按计划留到下一阶段处理）：CrispEdit/ScaleEdit 的区域外漂移，以及部分 add 的 mask 与实际新增位置不符（ScaleEdit add 约 64%，见 [v1 分析第 5 节](07_SAMTokEdit_Qwen21_v1分析与v2计划.md#5-数据层面的问题)）。add 框来自这些 mask，所以错位会延续到框。

## 2. 来源与谱系

~~~text
RefEdit / CrispEdit-2M / ScaleEdit / SAMTok Derived Edit
  → 质量筛选、SAM3/grounding 实例 mask（v1：sources.jsonl）
  → Qwen3.5-9B noref 转换 + 规则回退（v1：semantic_runs/qwen21_noref9b_rules_4n_full_003）
  → 发布 codec 编码 mask，生成 NTP/plain/ref/noref 行（v1：train_full_9b_rules_003）
  → v2 转换（本文）：train_v2_box_001
~~~

v1 输入：`/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/qwen21_full4_20260928/data/`

- `train_full_9b_rules_003/`：`input-XX.jsonl`（源记录 + 语义标注）、`encoded/worker-XX/`（已编码行）、`provenance.jsonl`、`metadata_report.json`
- `sources.jsonl`：源图、目标图、实例 RLE、数据集 mask

## 3. 转换规则

| 规则 | 涉及 | 处理 |
|---|---|---|
| 剔除 composite | 1,206 个源（CrispEdit 60、ScaleEdit 1,146）；行数：NTP/ref/noref 各 1,042，plain 1,206 | 整条记录的所有行都不输出 |
| add 改框 | 12,556 个源（RefEdit 2,904、CrispEdit 3,525、ScaleEdit 6,127） | 由生成 v1 span 的同一实例 mask 计算外接框，向外取整到 0–1000；替换 NTP 答案和 ref/noref prompt 中的 span，位置不变；plain 行不变 |
| Derived add → attribute | 6,269 个源（含 27 个仅有 plain 行的源） | 所有行 `edit_type` 改为 attribute；保留宿主实例 ⟨M⟩ 和原 prompt（noref 为 `Add ... in this region ⟨M⟩`，符合 attribute 的 noref 规则） |
| rec_ntp 回放 | 候选 26,569（RefEdit/Derived 单实例的 remove/replace/attribute/action 单元）→ 选 13,760 | 按 `sha256("rec:"+id)` 取前 `round(96,319/7)` 条；框 = 该单元 mask 的外接框；prompt = Qwen3-VL grounding 请求 |
| 其余行 | 323,662 行（含 add 源的 plain 行 12,556 条） | 与 v1 逐字节相同（按 provenance 中 row hash 与 v1 row hash 比对）；改动的行：add 37,668、Derived add 24,995、新增 rec_ntp 13,760 |

所有 add 单元都是单实例（12,556 个单元，每个 1 个框），没有多框 add。

## 4. 文件

| 文件 | 内容 | sha256 |
|---|---|---|
| `stage1.jsonl` | Stage 1：edit_ntp（按源顺序）+ rec_ntp（按选择顺序，在末尾） | `e2312e33…87bf0c` |
| `stage2.jsonl` | Stage 2：edit + edit_umt（按源顺序） | `cb46a3f0…21cd` |
| `provenance.jsonl` | 每个输出行一条：file/line、源 id、数据集、sample_type、edit_type、v1_edit_type、conversion（unchanged/add_box/derived_add_as_attribute/rec_replay）、row hash、v1 row hash | `2fc4dd0f…3fce` |
| `conversion_report.json` | 计数、剔除、转换、add 框统计、回放选择、输入/输出 hash、代码 commit | — |
| `metadata_report.json` | 四机入口检查的摘要（行数与 hash） | — |

图片仍在 v1 的 `assets/` 和 Derived 原路径下（行内为绝对路径），未复制。

## 5. 统计

**Stage 1。**

| 类型 | edit_ntp | rec_ntp |
|---|---:|---:|
| add | 12,556 | — |
| remove | 27,698 | 4,760 |
| replace | 18,198 | 3,232 |
| attribute | 33,217 | 5,768 |
| action | 2,666 | — |
| text | 1,984 | — |
| 合计 | 96,319 | 13,760 |

**Stage 2**（ref 与 noref 行数相同）。

| 类型 | edit | edit_umt ref | edit_umt noref |
|---|---:|---:|---:|
| add | 12,659 | 12,556 | 12,556 |
| remove | 28,059 | 27,698 | 27,698 |
| replace | 18,222 | 18,198 | 18,198 |
| attribute | 33,367 | 33,217 | 33,217 |
| action | 2,995 | 2,666 | 2,666 |
| text | 2,066 | 1,984 | 1,984 |
| 合计 | 97,368 | 96,319 | 96,319 |

**按数据集**（源编辑对 / edit_ntp）：CrispEdit 37,668 / 37,176；Derived 27,957 / 27,923；ScaleEdit 23,939 / 23,428；RefEdit 7,804 / 7,792。rec_ntp：Derived 11,194、RefEdit 2,566。

**add 框**（占图像面积比例的分位数）：

| 数据集 | p10 | p25 | p50 | p75 | p90 |
|---|---:|---:|---:|---:|---:|
| RefEdit | 0.026 | 0.062 | 0.125 | 0.207 | 0.297 |
| CrispEdit | 0.011 | 0.030 | 0.090 | 0.218 | 0.431 |
| ScaleEdit | 0.006 | 0.014 | 0.033 | 0.070 | 0.125 |

mask 占框面积的中位数为 0.51–0.60。全部 add 框的中位宽、高为 214、267（0–1000 单位），评测中 point 输入的默认 add 框用这个尺寸（D5）。

**框与 v1 span 的对应抽查。** 随机 300 个 add 单元：把 v1 span 用 codec 解码，取其外接框，与 v2 框比较 IoU。均值为 RefEdit 0.53、CrispEdit 0.44、ScaleEdit 0.38；IoU 为 0 的只占 0–2%。这与已知的 add 区域 codec 往返损失一致（v1 分析：0.38），说明框与原 span 对应同一实例；框本身直接来自原始 mask，不经过 codec 损失。

## 6. 行格式例子

~~~json
{"sample_type": "edit_ntp", "edit_type": "add", "edit_image": ".../refedit-b150db4078a9da532833f32b/source.img",
 "prompt": "Add a little bluebird perched on the rightmost birdbath",
 "mt_cot": "```json\n[{\"bbox_2d\": [663, 294, 908, 460], \"label\": \"little bluebird perched on the rightmost birdbath\"}]\n```"}
{"sample_type": "edit_umt", "instr_variant": "ref", "edit_type": "add", "edit_image": "...", "image": ".../target.img",
 "prompt": "Add a little bluebird perched on the rightmost birdbath <|box_start|>[663, 294, 908, 460]<|box_end|>"}
{"sample_type": "edit_umt", "instr_variant": "noref", "edit_type": "add", "edit_image": "...", "image": "...",
 "prompt": "Add a little bluebird perched in this region <|box_start|>[663, 294, 908, 460]<|box_end|>."}
{"sample_type": "edit_umt", "instr_variant": "noref", "edit_type": "attribute", "edit_image": "...", "image": "...",
 "prompt": "Change the color of this region <|mt_start|><|mt_0144|><|mt_0279|><|mt_end|> to bright red."}
{"sample_type": "rec_ntp", "edit_type": "remove", "edit_image": ".../refedit-75804bb074668bc00cd1a4e3/source.img",
 "prompt": "Locate the plain stump that is farthest away in this image and output its bbox coordinates in JSON format.",
 "mt_cot": "```json\n[{\"bbox_2d\": [201, 406, 448, 657], \"label\": \"plain stump that is farthest away\"}]\n```"}
~~~

## 7. 训练读取与配比

- Stage 1 只读 `stage1.jsonl`。每个 optimizer update 中 edit_ntp : rec_ntp = 7 : 1（32 卡、accumulation 8 时为 224 : 32），每个 rank 精确满足。正式日程 1,300 update，类型采样 `natural`：全部 110,079 行都被采到，每行约 3.0 次；add/remove/replace/attribute 占 edit_ntp 抽样的 95.2%（add 13.0%、remove 28.8%、replace 18.9%、attribute 34.4%、action 2.7%、text 2.1%）。
- 缓存只读 `stage2.jsonl`；Stage 2 读缓存，ref : noref : plain = 1 : 2 : 1（32 卡、accumulation 4 时为 32 : 64 : 32）。正式日程 1,000 update，类型采样 `main4`：四个主类型在每个池中约占 95%（noref 池：add/remove/replace 各 21.5%、attribute 30.8%，action 2.7%、text 2.0%）；add noref 每行约 1.1 次，remove 0.5 次；共覆盖 126,811 个不同行（44%）。
- 原 v1 类型权重会让 action/text（2,666/1,984 行）占约 24% 的抽样、在 Stage 1 中每行重复 9–12 次，而评测中没有 action、只有 1 个 text case，所以 v2 改为上述方案（2026-10-06 确认）。
- 读取时每行再次经过 `validate_row`；Stage 2 的缓存身份记录 metadata hash 和逐行 hash 摘要。

## 8. smoke 子集

`qwen21_v2/smoke/data_smoke_001/`：每个（数据集, 类型）组按 id hash 取 4 个源的全部行，另加 10 条 rec_ntp；72 个源，Stage 1 80 行（edit_ntp 70，含 add 12）、Stage 2 212 行，覆盖全部 6 种类型、两种区域表示和全部行类型。由 `scripts/diagnostics/make_smoke_data.py` 生成。

## 9. 复现与校验命令

```bash
cd /opt/tiger/tanyue/samtok_edit_qwen-image-2.1-v2
export PYTHONPATH=$PWD/src:$PWD/third_party/diffsynth:$PWD/third_party
PY=/tmp/samtok21-fixes-dUnbt5/venv/bin/python   # 或集群 setup_env.sh 建立的环境
V2=/mnt/bn/strategy-mllm-train/intern/users/tanyue/experiments/SAMTokEdit/qwen21_v2

# v1 → v2 转换（输出目录必须为空；结果与 train_v2_box_001 逐字节一致）
$PY -m samtok_edit21.preparation.v2_data --output $V2/data/<new_dir>
# smoke 子集
$PY scripts/diagnostics/make_smoke_data.py --data $V2/data/train_v2_box_001 --output $V2/smoke/<new_dir> --per-group 4
# 逐行协议校验与图片可读性（较慢：会打开全部图片）
$PY -m samtok_edit21 validate --metadata $V2/data/train_v2_box_001/stage1.jsonl
```
