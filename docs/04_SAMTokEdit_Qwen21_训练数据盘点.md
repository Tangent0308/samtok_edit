# SAMTokEdit（Qwen-Image-2.1）训练数据盘点

- **第 0 节**：v1 源数据经清洗、编辑核验和协议 1.0 适配后得到的新数据，分单目标和多目标两份。这两份数据还没有转成训练行。
- **第 1–9 节**：当前训练在用的 v2 训练数据 `train_v2_box_001`。

第 1–9 节记录 v2 训练数据：来源、相对 v1 的转换、最终文件、统计、行格式、校验和复现命令。四个源数据集的筛选、9B noref 转换和 v1 corpus 构建过程不变，详见 v1 存档 [训练数据盘点](archive/v1/04_SAMTokEdit_Qwen21_训练数据盘点.md)。转换代码说明见[代码实现说明第 3 节](01_SAMTokEdit_Qwen21_代码实现说明.md#3-训练数据转换v1--v2)。

## 0. 协议 1.0 新数据（v1 清洗与适配后，2026-10-15）

清洗与核验的过程见[文档 09](09_SAMTokEdit_Qwen21_训练数据诊断与改进.md)；协议适配和多目标召回的规则见[文档 10](10_SAMTokEdit_Qwen21_训练数据协议适配.md)（第 1–9 节为单目标，第 10 节为多目标）。

### 0.1 总览

| 数据 | 清单 | 数量 | 一行是 |
|---|---|---|---|
| 单目标：一图一处编辑 | `protocol_adapt_qwen38_27b_v1/manifest_protocol.jsonl` | 39,615 对 | 一个编辑单元；其中 1,679 条 add 另带整体宿主视图 |
| 多目标：一图多处编辑 | `protocol_multi_qwen38_27b_v1/manifest_multi.jsonl` | 4,066 对，9,964 个单元 | 一对图及其 2–5 个单元（U1…Uk），一起执行 |
| **合计** | | **43,681 对，49,579 个单元** | 两份数据的图对互不重叠 |

- **根目录**：`/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTokEdit_Cleaned_20261009/`。
  - 两份清单的图片路径都相对根目录（`images/<来源>/<id 末两位>/<id>.source.*`、`.target.*`），图片没有复制；
  - 结果图与源图同尺寸。
- **单元内容**：每个单元都是一条协议 1.0 记录，包括：
  - 语义记录 `instruction_spec`；
  - 由共享协议模块生成的 ref / no-ref 两版指令，与 benchmark 2.4.0 一致；
  - `mask_semantics`；
  - S（`selection`，即输入区域）；
  - `execution_region`、`edited_region`：执行区域和实际改动区域，内部字段，不作模型输入。

  所有 mask 都是源图像素坐标下的 COCO RLE（`rle_size` 为 `[高, 宽]`），附 0–1000 外接框 `box_1000`。
- **代码**：v2 分支。
  - `fff8ca5`：单目标四机全量；
  - `f90d54d`：像素把关与原地修复；
  - `2109371`、`e979335`、`2f68d24`：多目标召回。
  - 除 `fff8ca5` 外，都还没推送。

### 0.2 谱系与各阶段保留量

~~~text
v1 sources.jsonl：98,574 对
  → 像素清洗：对齐、区域外成块改动、贴回检查                                  70,956
  → Qwen3.8-27B 编辑核验                                                      48,437
  → 去掉 action 434、无单一类型 601；text 1,521 按改字 / 删字 / 加字分流         47,402
  → 协议 1.0 适配（四机全量 + 原地修复）：单目标接受 39,615；多目标 5,724；拒收 2,063
  → 多目标召回（本机 8 卡）：接受 4,066 对（9,964 个单元）；其实只有一个目标 253；拒收 1,405
~~~

**按来源**（单位：图对）

| 来源 | v1 | 像素清洗 | 编辑核验 | 协议输入 | 单目标 | 多目标 | 合计 | 合计 / v1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RefEdit | 7,804 | 7,108 | 4,570 | 4,570 | 4,119 | 28 | 4,147 | 53.1% |
| CrispEdit | 37,728 | 22,806 | 12,444 | 12,400 | 8,718 | 2,200 | 10,918 | 28.9% |
| ScaleEdit | 25,085 | 19,368 | 13,519 | 12,528 | 9,683 | 1,778 | 11,461 | 45.7% |
| Derived | 27,957 | 21,674 | 17,904 | 17,904 | 17,095 | 60 | 17,155 | 61.4% |
| **合计** | **98,574** | **70,956** | **48,437** | **47,402** | **39,615** | **4,066** | **43,681** | **44.3%** |

**按原标注类型**（单位：图对；最终的操作类型以结果图为准，见 0.3、0.4）

| 原类型 | v1 | 像素清洗 | 编辑核验 | 单目标 | 多目标 | 合计 / v1 |
|---|---:|---:|---:|---:|---:|---:|
| add | 18,764 | 11,314 | 8,718 | 7,275 | 140 | 7,415（39.5%） |
| remove | 28,124 | 21,558 | 14,823 | 12,199 | 1,916 | 14,115（50.2%） |
| replace | 18,153 | 12,860 | 8,681 | 7,622 | 456 | 8,078（44.5%） |
| attribute | 27,318 | 20,952 | 13,659 | 11,071 | 1,524 | 12,595（46.1%） |
| text | 2,053 | 1,950 | 1,521 | 1,448 | 30 | 1,478（72.0%） |
| action | 2,895 | 1,341 | 434 | 去掉 | — | — |
| 无单一类型 | 1,267 | 981 | 601 | 去掉 | — | — |

### 0.3 单目标数据（`protocol_adapt_qwen38_27b_v1/`）

| 文件 | 内容 | 大小 |
|---|---|---|
| `manifest_protocol.jsonl` | 接受的 39,615 条，训练用 | 355 MB |
| `records.jsonl` | 全部 47,402 行的完整记录（槽位、检查、审核、拒收原因），顺序与输入相同 | 448 MB |
| `multi_object.jsonl` | 5,724 条多目标，已由 0.4 召回 | 7 MB |
| `summary.json`、`config.json`、`README.md` | 统计；模型、提示词、参数、代码；`post_fix` 记录 2026-10-14 的原地修复 | — |
| `runs/protocol_adapt_27b_4n_001/` | 四机运行的日志和逐行结果 | 2.2 GB |
| `_trash_20261014/` | 原地修复前的文件，确认后可删 | 0.8 GB |

**操作 × 来源**

| 来源 | add | remove | replace | attribute | 合计 |
|---|---:|---:|---:|---:|---:|
| RefEdit | 2,099 | 730 | 470 | 820 | 4,119 |
| CrispEdit | 1,237 | 2,876 | 2,138 | 2,467 | 8,718 |
| ScaleEdit | 3,086 | 2,274 | 1,003 | 3,320 | 9,683 |
| Derived | 825 | 6,327 | 3,704 | 6,239 | 17,095 |
| **合计** | **7,247** | **12,207** | **7,315** | **12,846** | **39,615** |

- **text 行的去向**（1,448 条）：
  - 改字 → attribute.text 1,391，其中 7 条实际是改字色或字体；
  - 删字 → remove 12；
  - 加字 → add 24；
  - 图标换成文字 → replace 21。
- **add**：
  - S 的粒度：承载面局部 4,306、整个物体 1,607、部件 1,334；
  - 锚点：直接承载 7,172，参照物（新物体托住 S）75；
  - 4,196 条裁到了新物体所在的局部，1,679 条另带整体宿主视图 `host_view`。
- **attribute**：
  - 属性：color 10,055、text 1,406、material 1,247、pattern 321、finish 238、texture 170、opacity 31、tint 25；
  - 572 条同时改多个属性。
- **remove**：整体 12,084，连同内容物 123。
- **指令长度**：ref 中位数 13 词（p90 17，最长 33）；no-ref 9 词（p90 12，最长 27）。
- **拒收 2,063**：

  | 原因 | 条数 |
  |---|---:|
  | 审核未过 | 1,229 |
  | add 几何检查（承载物没贴着新物体 269、不在下方 145 等） | 434 |
  | 没检测到改动 | 139 |
  | 模型判为不合法编辑 | 91 |
  | SAM3 失败 | 87 |
  | 像素把关（S 落在没变的地方） | 41 |
  | 局部执行区域有误 | 37 |
  | 人工剔除 | 5 |

### 0.4 多目标数据（`protocol_multi_qwen38_27b_v1/`）

| 文件 | 内容 | 大小 |
|---|---|---|
| `manifest_multi.jsonl` | 接受的 4,066 对，训练用 | 86 MB |
| `records.jsonl` | 全部 5,724 对的完整记录，顺序与 `multi_object.jsonl` 相同 | 107 MB |
| `summary.json`、`config.json`、`README.md` | 统计；提示词哈希、参数、代码、人工剔除名单 | — |
| `work/` | 各步骤的逐行结果和日志 | 562 MB |

**单元的操作 × 来源**

| 来源 | 图对 | add | remove | replace | attribute | 单元合计 |
|---|---:|---:|---:|---:|---:|---:|
| RefEdit | 28 | 3 | 5 | 37 | 15 | 60 |
| CrispEdit | 2,200 | 198 | 2,841 | 537 | 2,071 | 5,647 |
| ScaleEdit | 1,778 | 112 | 1,929 | 282 | 1,806 | 4,129 |
| Derived | 60 | 2 | 62 | 29 | 35 | 128 |
| **合计** | **4,066** | **315** | **4,837** | **885** | **3,927** | **9,964** |

- **每对单元数**：2 个 2,820 对，3 个 782 对，4 个 342 对，5 个 122 对。
- **一对之内的操作**：
  - 只有一种操作 3,942 对：全是 remove 1,918、全是 attribute 1,558、全是 replace 336、全是 add 130；
  - 混合 124 对：remove+replace 64、attribute+replace 23、add+remove 15、attribute+remove 10，其余 12。
- **单元**：
  - S 的来源：原标注复用 7,883、从原标注拆出 138、SAM3 重打 1,846、SAM 部件 97；
  - attribute 属性：color 2,980、material 896、finish 131、texture 113、pattern 88、text 53、opacity 10、tint 3；
  - add 的 S 粒度：物体 112、部件 108、承载面局部 95。
- **指令长度**：ref 中位数 11 词（p90 16，最长 30）；no-ref 8 词（p90 11，最长 22）。
- **没进主集**：
  - 其实只有一个目标 253 对，尚未处理；
  - 拒收 1,405 对：

    | 原因 | 对数 |
    |---|---:|
    | 拆不开（超过 5 个单元、物体无法单独指认、姿态变化、几个物体换成一个等） | 982 |
    | 单元的 S 重叠或嵌套 | 207 |
    | 审核未过 | 98 |
    | 单元层面的把关（像素把关、add 几何检查、SAM3 失败等） | 87 |
    | 有改动不属于任何单元 | 18 |
    | 修复后无有效输出 | 10 |
    | 人工剔除 | 3 |

### 0.5 行格式

单目标（一行一个单元，RLE 已省略）：

~~~json
{"source_image": "images/derived/6c/derived-1bc0b1b067646484164a206c.source.png", "target_image": "images/derived/6c/derived-1bc0b1b067646484164a206c.target.png",
 "id": "derived-1bc0b1b067646484164a206c", "dataset": "derived", "type_before": "attribute",
 "instruction_spec": {"version": "1.0", "operation": "attribute", "ref": "the red hoodie in the center", "scope": "hoodie",
                      "attributes": [{"property": "color", "value": "bright blue"}], "placement": {}, "payload": "", "target_part": "", "remove_mode": "whole", "qualifier": ""},
 "instruction_ref": "Change the color of the red hoodie in the center to bright blue.",
 "instruction_noref": "Change the color of the selected hoodie to bright blue.",
 "mask_semantics": {"version": "1.1", "source_role": "target", ...},
 "selection": {"source": "sam3", "rle_size": [1024, 1024], "rle_counts": "...", "box_1000": [275, 216, 614, 520]},
 "host_view": null, "execution_region": {"source": "cleaned_region", ...},
 "edited_region": {"method": "lab-diff-hysteresis-2", "area_frac": 0.067, "stats": {...}, ...},
 "flags": ["fix:part_as_s"], "adapter_version": "protocol-adapt-3", "instruction_protocol_version": "1.0"}
~~~

多目标（一行一对，`units` 中每个单元的字段与单目标相同）：

~~~json
{"source_image": "images/scaleedit/d7/scaleedit-cf891fa1db9cef0653da5ad7.source.jpg", "target_image": "images/scaleedit/d7/scaleedit-cf891fa1db9cef0653da5ad7.target.png",
 "id": "scaleedit-cf891fa1db9cef0653da5ad7", "dataset": "scaleedit", "type_before": "attribute",
 "instruction_before": "Change the color of the white market tent on the left to red and the central white market tent to blue.",
 "units": [
  {"id": "U1", "instruction_ref": "Change the color of the left white market tent canopy to red.",
   "instruction_noref": "Change the color of the selected market tent canopy to red.",
   "instruction_spec": {...}, "mask_semantics": {...}, "selection": {"source": "reused", "box_1000": [52, 512, 328, 590], ...},
   "execution_region": {...}, "edited_region": {"area_frac": 0.017, ...}, "flags": [...]},
  {"id": "U2", "instruction_ref": "Change the color of the central white market tent canopy to blue.",
   "instruction_noref": "Change the color of the selected market tent canopy to blue.", ...}],
 "edited_region": {"area_frac": 0.029, ...}, "flags": ["U2:edited_region_far_changes", ...],
 "adapter_version": "protocol-adapt-4", "multi_version": "protocol-multi-1", "instruction_protocol_version": "1.0"}
~~~

训练时把各单元的句子拼成一个多条指令的请求（benchmark 的写法为 `U1 [...]: ...` 每行一条），每个单元各给自己的区域。

### 0.6 质量

| 数据 | 抽查 | 合格 | 小瑕疵 | 有错 | 按来源加权的估计 |
|---|---|---:|---:|---:|---|
| 单目标 | 分层随机 60 条接受记录 | 50 | 7 | 3 | 合格约 86%、小瑕疵约 9%、有错约 5%（区间约 1%–14%） |
| 多目标 | 全量完成后每个来源 12 对 + 运行中途 21 对，共 69 对 | 57 | 8 | 4 | 合格约 88%、小瑕疵约 7%、有错约 5% |

- **两份数据都通过了完整性检查**：
  - 每个单元的句子都能由共享协议模块重新生成，`mask_semantics` 与契约一致；
  - 所有 mask 能解码、非空，尺寸与原图一致；
  - 引用的图片全部存在（单目标 79,230 张，多目标 8,132 张）。
- **已处理的错误**：
  - 单目标在 2026-10-14 原地修复，剔除 46 条：像素把关拦下 41 条，人工剔除 5 条；另重算了 17 条的改动区域。
  - 多目标抽查看到的 4 条有错中，1 条在修复轮被拒收，3 条人工剔除。
- **仍然存在、目前没有自动手段的错误**：
  - 源数据里有指令没写的附带改动，例如花瓶换色时形状也变了；
  - S 比实际改动大很多。
- **抽查的审阅页**：本机 8790 端口，在 code-server 中打开 `/proxy/8790/…`。
  - `full_run_spotcheck.html`：单目标；
  - `multi_full_spotcheck.html`：多目标完成后；
  - `multi_recall_review.html`：多目标运行中途和试点。

  归档在 `experiments2/SAMTokEdit/qwen21_v2/data/` 的 `protocol_full_run_spotcheck_20261014/` 和 `protocol_multi_pilot_20261015/`。

### 0.7 与 benchmark v2 的差距（转训练行时要注意）

| 维度 | 新数据 | benchmark v2 |
|---|---|---|
| S 面积中位数 | 3.2%（全部 49,579 个单元） | 0.57% |
| S 面积 < 0.5% | 16%（单目标 14%，多目标 22%） | 45% |
| S 面积 ≥ 5% | 38%（单目标 42%，多目标 20%） | 1.4% |
| 一图多处编辑 | 4,066 对，占 9.3%，每对 2–5 处 | 主协议 mixed 为每图 2–5 处 |
| add 所占比例 | 15%（7,562 个单元） | 25%（196 / 785） |
| 区域给法 | 只有 mask 和文字指代 | point、box、mask、无，约 3 : 3 : 3 : 1.5 |

其他偏差：
- Derived 占单目标的 43%；它的换色 87% 是 bright 系高饱和色，光 bright red 就有 3,160 条。
- 没有 action 类。

### 0.8 还没做的

- 转成训练行：区域给法的配比、宿主视图的混入、按面积加权、多单元请求的拼法、固定验证集。
- 多目标召回中判为“其实只有一个目标”的 253 对，可用单目标流程再处理。
- 推送 v2 分支上未推送的 5 个提交。
- 确认后删除 `protocol_adapt_qwen38_27b_v1/_trash_20261014/`。

## 1. 结论

| 项 | 值 |
|---|---|
| 数据目录 | `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v2/data/train_v2_box_001/`（2026-10-06 从 intern 旧根目录逐字节复制，三个 sha256 与 `metadata_report.json` 一致；原件仍在 `/mnt/bn/strategy-mllm-train/intern/users/tanyue/experiments/SAMTokEdit/qwen21_v2/data/train_v2_box_001/`） |
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
V2=/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v2

# v1 → v2 转换（输出目录必须为空；结果与 train_v2_box_001 逐字节一致）
$PY -m samtok_edit21.preparation.v2_data --output $V2/data/<new_dir>
# smoke 子集
$PY scripts/diagnostics/make_smoke_data.py --data $V2/data/train_v2_box_001 --output $V2/smoke/<new_dir> --per-group 4
# 逐行协议校验与图片可读性（较慢：会打开全部图片）
$PY -m samtok_edit21 validate --metadata $V2/data/train_v2_box_001/stage1.jsonl
```
