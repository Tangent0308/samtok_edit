# SAMTokEdit Qwen-Image-2.1 v3：外部编辑数据统一打标管线（CrispEdit、ScaleEdit）

本文是一条新数据管线的完整计划，状态为**设计、待用户确认，代码尚未实现**，截至 2026-10-20。

**目标：**用 CrispEdit、ScaleEdit 上还没处理过的原始编辑对，打标出符合当前标准（指令协议 2.0、mask 语义 2.0、区域外逐像素不变、token 往返合格）的训练数据。数据格式与现有 D3（单目标）、D4（多目标）相同。

**内容：**
- 源数据怎么得到；
- 现有流程每一步做了什么、留下多少；
- 新管线每一步的输入、规则、输出和复用的代码；
- 产量与算力估算；
- 校准、上线和待确认事项。

**背景与依据：**
- 现有数据的整体情况见[文档 01](01_SAMTokEdit_Qwen21_v3数据.md) §2.1–§2.5、§2.12；
- 新数据要满足的要求（C3 计划）见文档 01 §2.11 和[文档 03](03_SAMTokEdit_Qwen21_v3实验.md) §6；
- 逐阶段核查记录在 `V3/analysis/ext_pipeline_20261020/`：`crispedit_flow.md`、`scaleedit_flow.md`、`v1_funnel.py`。

路径记号：`datasets/…` 指 `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/…`，`V3/…` 指 `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v3/…`。

## 0. 一页摘要
- **现有流程：**CrispEdit、ScaleEdit 各下载了 52.9 万、30.0 万个原始对，最后只有 8,124、7,596 条进入训练，端到端 1.5%、2.5%。
- **低产的原因：**
  1. 场景难度筛只留约 9%。这道筛在协议 2.0 的盲指代指令下已不必要。
  2. 27B 每条调用 5–6 次，SAM3 分割两次。
  3. 便宜的像素检查排在最后。
  4. add 的 mask 语义和现在的标准不一致，后面要重做。
- **新管线：**六步，27B 只调用 2 次（规划 + 审核），SAM3 一次，像素预检排在最前面，直接产出协议 2.0 数据。
- **估算：**每 10 万个原始对出 2–3 万条，约 150–250 GPU 小时，四机约 5–8 小时。数字待校准验证。
- **还能用的原始数据：**
  - **本地已下载的数据都被旧流程判过，但约九成只是被场景难度筛丢掉的（§1.4）。**在范围内、质量预筛已通过、从没打过标的，CrispEdit 有 35.4 万对，ScaleEdit 有 17.5 万对（核心五类 12.1 万 + 文字 5.4 万）。第一批可以全部取自本地，不用下载。
  - 上游未下载的：CrispEdit 约 112 万对，ScaleEdit 至少约 88 万对可用候选。

## 1. 源数据：怎么得到

### 1.1 CrispEdit-2M
- **上游：**HF `WeiChow/CrispEdit-2M`，CC BY 4.0（来自 EditMGT 论文 arXiv:2512.11715）。
  - 共 8,971 个 parquet 分片，每片 256 行，约 229 万对，7 类编辑。
  - 下载固定在 revision `764723d87e765bbceeff7e08bb0890f89abecf6b`。
- **一行的内容：**`input_img`（源图，`{bytes, path}`）、`output_img`（编辑后，同格式）、`instruction`（13–294 字符）、`type`（7 类之一）。
  - 最长边中位 1,535 px，约 1.57 MP；
  - 源图和编辑图只有 48% 尺寸完全相同，常差 1–2 px。
- **本地已有：**`datasets/CrispEdit_Labeling/source/CrispEdit-2M/`，2,078 个分片、529,020 行，约 1.1 TB，分 5 批下载（08-13 到 09-23）。
- **各类分片数：**

  | 类型 | 上游分片 | 本地 | 剩余 | 用不用 |
  |---|---:|---:|---:|---|
  | add | 1,213 | 445 | 768 | 用 |
  | color | 1,984 | 445 | 1,539 | 用（改属性） |
  | remove | 1,388 | 445 | 943 | 用 |
  | replace | 1,567 | 445 | 1,122 | 用 |
  | motion change | 128 | 128 | 0 | 不用（动作，不在协议内） |
  | background change | 1,091 | 84 | 1,007 | 不用（全局） |
  | style | 1,600 | 86 | 1,514 | 不用（全局） |

  五类在范围内的剩余约 4,372 个分片、约 112 万行、约 2.4 TB。
- **产量与分片区间强相关：**不同区间用的指令模板不同，旧流程"两道筛都过"的比例差得很远。
  - 例：color 0–549 为 24%，≥ 600 只有 1.1%；replace 150–699 为 30%，0–149 只有 0.4%。
  - 高产区间多是 "X located in the <区域>" 这类模板句。
  - 新管线去掉了场景难度筛（§3.3），区间差异会变小，但仍按区间记录产量。
- **下载脚本：**CrispEdit 打标仓库 `/opt/tiger/tanyue/sam3-crispedit-labeling` 的 `scripts/download_crispedit_more.py --run-dir <新目录> --per-type N`。
  - 它按类型从未下载的分片里均匀挑 N 个，只支持 add、color、remove、replace；
  - 运行目录里已有 `plan.json` 时直接沿用，所以可以指定分片清单。
  - 新管线以只读方式调用它，下载到新数据集目录（§5），不往原有目录里加文件。

### 1.2 ScaleEdit（ScaleEdit-12M 经 filtered-6m 挑选）
- **上游图片：**HF `InternVL-U/ScaleEdit-12M`，CC BY-NC-SA 4.0（非商业、相同方式共享）。
  - 约 1,185 万对，23 类任务，约 6.19 TB，每片约 31 GB。
  - 源图或是内嵌字节，或是网址加 SHA-256（来自网页）。
- **挑选清单：**HF `QingyuShi/scaleedit-filtered-6m` 的 `keep_manifest.parquet`。
  - 6,584,390 行 KEEP，只有清单没有图片；
  - 给出复核过的 `final_task`、`final_instruction`、`instruction_action`、`category_action`、`confidence`。
  - 两边的 revision 记在 `datasets/ScaleEdit_Labeling/runs/download_200k_20260924/download_state.json`。
- **一行的内容（本地 20 列）：**`sample_id`、`edit_task` / `final_task`、`original_instruction` / `final_instruction`、源图和编辑图字节与尺寸等。
  - 中位 2.0 MP；
  - 299,633 行里 253,769 行两图同尺寸，但常有整体缩放：中位 1.075 倍、平移约 42 px。
- **本地已有：**`datasets/ScaleEdit_Labeling/source/ScaleEdit-filtered-source/`，1,155 个分片、299,633 行，约 137 GB，是 filtered-6m 的 4.55%。
  - `part-*`：10 万行，23 类各约 6,152 行，含全局类；
  - `expand-20260924-*`：199,633 行，只含局部类。
- **类型：**
  - 17 类局部编辑可用：物体添加 / 删除 / 替换、改色、改材质、4 种文字编辑（物体表面、建筑表面、GUI、海报）、动作、组合、数量、大小、推理类；
  - 6 类全局编辑不用；
  - 动作、推理类按协议也不用。
- **剩余：**下载器最后一次索引时，140 个未下载的分片里至少还有 882,805 个可用候选。

  | 类型 | 候选 |
  |---|---:|
  | 物体替换 | 347,398 |
  | 物体添加 | 170,284 |
  | 物体表面文字 | 122,387 |
  | 物体删除 | 112,501 |
  | 改色 | 76,120 |
  | 建筑表面文字 | 47,001 |
  | 改材质 | 6,873 |

  推理、海报、GUI 类基本用完。只有网址的源图可以加 `--fetch-source-urls` 去抓。
- **下载脚本：**ScaleEdit 打标仓库的 `/opt/tiger/tanyue/sam3-scaleedit-labeling/scripts/download_scaleedit_filtered.py`。
  - 按清单行在原始分片里用唯一指令匹配，去重，解码检查，按类型配额；
  - 关键参数：`--run-id`、`--run-dir`、`--target-rows`、`--max-download-gb`、`--cache-dir /opt/tiger/tanyue/.cache/scaleedit-expand-<id>`。
  - **它必须整片下载原始分片**（约 31 GB 一片，上次平均每留下 1 行要传约 4 MB）；
  - 去重只看 `--output-dir` 里已有的分片。新管线调用时，要把已有分片的索引一起传给它。

### 1.3 新管线的统一取数
两个来源各写一个读取器，读成同一种行格式：

| 字段 | 内容 |
|---|---|
| `id` | `crispedit-<分片>-<行号>` 或 `scaleedit-<sample_id 的哈希>` |
| `dataset`、`native_type` | 来源与原始类型（CrispEdit 的 `type`；ScaleEdit 的 `final_task`） |
| `instruction` | 原始指令（ScaleEdit 用 `final_instruction`），只作为规划的参考，最终句子由协议 2.0 重新渲染 |
| `source_image`、`target_image` | 解出的图片文件（PNG/JPEG），按 id 分目录 |
| `provenance` | 上游 revision、分片名、行号、下载批次 |

取数时依次排除：
1. 已经处理过的行：CrispEdit 的 529,020 个 (分片, 行号)，ScaleEdit 的 299,633 个 `sample_id`。
2. 不在范围内的类型：全局、风格、背景、视角、动作、推理类。
3. 与 benchmark 2.6.0 的图像隔离：字节哈希，再加感知哈希近重复检索（50k 当时没做）。
4. 来源内部的重复图：多个 id 指向同一张源图时，每张源图每个子集最多保留 3 条。
5. 尺寸：短边 < 512 px 的不要。

### 1.4 本地已下载的数据用到了多少（10-20 核查）
- **核查方式：**逐行对照打标仓库的质量预筛和场景难度筛清单。脚本 `V3/analysis/ext_pipeline_20261020/local_usage.py`，结果 `local_usage.txt`。
  - CrispEdit 的 529,020 行全部做过质量预筛，质量通过的 460,518 行全部做过场景难度筛；
  - ScaleEdit 的 299,633 行同样全部做过两道筛。
- **结论：**本地没有"从没处理过"的行。但两道筛都过、往下打标的只有 CrispEdit 38,971、ScaleEdit 25,664，约九成只是被场景难度筛丢掉的。按新管线的标准（去掉场景难度筛），它们都还能用。
- **质量通过、只被场景难度筛丢掉的行（在范围内）：**

  | 来源 | 类型 | 行数 |
  |---|---|---:|
  | CrispEdit | add | 95,887 |
  | CrispEdit | color（改属性） | 87,716 |
  | CrispEdit | remove | 81,363 |
  | CrispEdit | replace | 88,835 |
  | CrispEdit 小计 | | **353,801** |
  | ScaleEdit | 物体删除 | 22,486 |
  | ScaleEdit | 物体添加 | 22,044 |
  | ScaleEdit | 物体替换 | 25,248 |
  | ScaleEdit | 改色 | 25,288 |
  | ScaleEdit | 改材质 | 26,110 |
  | ScaleEdit 核心五类小计 | | **121,176** |
  | ScaleEdit | 物体表面文字 + 建筑表面文字 | 54,039 |
  | ScaleEdit（可选） | GUI、海报文字 | 43,718 |
  | ScaleEdit（可选） | 组合编辑（多处改动） | 9,880 |

  不在范围内的不计：CrispEdit 的 motion 26,098、background 20,633、style 21,015；ScaleEdit 的动作和推理类。
- **对计划的影响：**
  - 第 1 步取数先用这个本地池，用完再下载。第一批（§6）全部取自本地，存储和下载的风险推后。
  - 这些行已有质量预筛 PASS，规划时仍重新判断。由于质量不合格的约 12% 已经去掉，预计产量比 §5 高：CrispEdit 约 24%，ScaleEdit 约 35%。
  - 只算本地池：CrispEdit 约 8.5 万条，ScaleEdit 约 4.2 万条（核心五类）加约 1.5 万条（文字），合计约 14 万条。这是估计，待校准。
  - 已走完旧流程的 38,971、25,664 行不再重复处理。

## 2. 现有流程：从源数据到 D3/D4

| # | 阶段 | 做什么 | 模型 | CrispEdit | ScaleEdit |
|---|---|---|---|---:|---:|
| 0 | 下载 | §1 | — | 529,020 | 299,633 |
| 1 | 质量预筛 | 看前后图和指令，6 个维度全部 PASS 才留；全局类不调模型直接丢 | Qwen3.8-27B | 460,518 | 265,456 |
| 2 | 场景难度筛 | 只看源图和指令：同类多实例、指令要从中选出一个 | 27B | 38,971 | 25,664 |
| 3 | 观察 + grounding | 列出实际编辑单元，给每个单元框；add 框在目标图 | 27B | 38,456 | 25,553 |
| 4 | SAM3 分割 | 框 + 指代短语；add 标在目标图的新物体上；文字用多边形 | SAM3 | 37,728 | 25,085 |
| 5 | 合并 | 四个子集的 `sources.jsonl` | — | 37,728 | 25,085 |
| 6 | 像素清洗 | 对齐、配准、贴回；区域外成块改动、没改、错位的丢 | CPU | 22,806 | 19,368 |
| 7 | 编辑核验 | 问题清单 + 盲描述两种问法 | 27B | 12,444 | 13,519 |
| 8 | 类型筛 | 去掉 action 和无单一类型；text 按改字、删字、加字分流 | — | 12,400 | 12,528 |
| 9 | 协议 1.0 改写 | 填槽位、确定 S、SAM3 重分割、渲染、两票审核、修一次 | 27B + SAM3 | 单 8,718 / 多 3,351 | 单 9,683 / 多 2,160 |
| 10 | 多目标召回（D4） | 多目标行改写成多单元 case | 27B + SAM3 | 2,200 对 | 1,778 对 |
| 11 | 协议 2.0（D3） | 规则改写：盲指代、site | — | 8,718 | 9,683 |
| 12 | token 往返 ≥ 0.5 | v3 训练可用 | SAMTok | 8,124 | 7,596 |

- **代码位置：**
  - 0–4 在两个打标仓库，二者同一套设计，大部分模块逐字节相同；
  - 6–11 在 v2/v3 仓库的 `scripts/data_cleaning/` 和 `src/samtok_edit21/preparation/`；
  - 12 在 `round2-data`。
- **算力：**
  - 打标约 925 GPU 小时（CrispEdit）和 1,030 GPU 小时（ScaleEdit），其中质量预筛占 70%–80%；
  - 核验约 35 GPU 小时、协议改写约 80 GPU 小时，四个子集合计。
- **问题：**
  1. 27B 调用 5–6 次，SAM3 两次。最贵的质量预筛跑全量，不用 GPU 的像素清洗反而在打标之后。
  2. 场景难度筛只留约 9%（add 只有 3.7%）。它是为"带名指令也必须读 token"设计的，协议 2.0 盲指代下 token 本来就是唯一线索；它还让 CrispEdit 偏向模板句。
  3. add 的 mask 标在新物体上，后面又按锚点重分割。token 往返在最后才算，ScaleEdit 有 22% 在这一步被丢。
  4. 流程分在四处，格式转了四次。四机脚本写死 4 节点和旧路径；CrispEdit 仓库本地有 1 个提交没推，远端还是旧路径。

## 3. 新管线：逐步说明
总体：取数 → 像素预检 → 规划 → 分割 + token → 区域把关 + 贴回 → 渲染、审核、导出。

每一步写自己的分片结果，可断点续跑；被丢掉的行记下原因，进入"拒收清单"。

### 3.1 第 1 步：取数
- **输入：**上游清单与分片（§1）。
- **做什么：**下载新分片 → 解码 → 统一行格式（§1.3）→ 依次排除（§1.3 的 5 条）→ 按类型配额抽取（§6）。
- **输出：**`ingest/<批次>/rows.jsonl` 和图片目录；排除计数写进 `ingest/<批次>/summary.json`。
- **复用：**两个打标仓库的下载脚本（只读调用）；ScaleEdit 的清单匹配逻辑。
- **设备：**CPU、网络。

### 3.2 第 2 步：像素预检（不需要 mask，排在所有 GPU 步骤之前）
- **尺寸：**编辑图缩放到源图尺寸。
- **对齐：**
  - 先在缩小的图上（长边 512，与 `clean_pairs.py` 相同）估计整体错位；
  - 错位明显时用 SIFT + RANSAC 仿射配准（长边 768）；
  - 配准后的覆盖率要 ≥ 98%。
- **改动图：**
  - Lab 色差 ΔE，先做高斯平滑（σ = 1）；
  - 双阈值 6 / 12：低阈值内、且与高阈值区域相连的像素算"改了"；
  - 与现在 `edited_region` 的参数相同（`protocol_adapt.EDITED_PARAMS`）。
- **丢弃规则**（初值，校准时调整）：

  | 规则 | 初值 | 对应旧流程 |
  |---|---|---|
  | 几乎没改 | 改动面积 < 0.05% 或改动强度过弱 | `drop_D` |
  | 改动遍布全图 | 改动面积 > 40% | 全局编辑 |
  | 多处远离的成块改动 | 最大的两块改动相距 > 0.3 × 对角线，且第二块 > 1% | `drop_C` |
  | 配准后仍错位 | 纹理块对齐比例 < 40% | `drop_M` |

- **输出：**对齐后的编辑图、改动图（RLE）、改动外接框、各项指标。
- **复用：**`scripts/data_cleaning/clean_pairs.py` 的 `stats`、`register`、`coherent_outside`、`alignment`；`scripts/data_cleaning/checks.py`；`protocol_adapt.color_difference`。
- **预计：**丢掉 25%–40%。旧流程的像素清洗丢了 CrispEdit 40%、ScaleEdit 23%。
- **设备：**CPU。

### 3.3 第 3 步：规划（Qwen3.8-27B，一次调用）
- **输入视图：**
  - 源图；
  - 对齐后的编辑图；
  - 改动高亮图：源图上用轮廓标出改动区域，并标号；
  - 改动区域的前后特写；
  - 原始指令只作参考。
- **模型要回答（受约束的 JSON）：**
  1. `verdict`：编辑是否成立。问题类别沿用旧核验：`not_done`、`partial`、`wrong_target`、`scope`（改到了别的物体）、`defect`、`global`。
  2. `units`：1–5 个编辑单元。每个单元按协议 2.0 填槽位：`operation`、`ref`（指代 S 的短语）、`scope`、`payload`、`attributes`、`target_part`、`remove_mode`、`placement`、`qualifier`。
  3. 每个单元 S 在源图上的框：
     - add 给锚点，即新物体站、放、挂、贴附的物体或表面；
     - 部件编辑给部件本身（mask 语义 2.0）。
  4. 标签：`comparable_instances`，是否有同类近邻。**只记录，不用来筛**，后面用于构造孪生和近邻反例。
- **合并了旧流程的：**质量预筛的 6 个维度、grounding 的观察与框、编辑核验的问题清单、协议改写的槽位。
- **规则：**
  - 槽位规范化与转成 spec 用 `protocol_adapt.normalize_slots`、`to_spec`；
  - 一堆或一串同类物算一个对象，不同的人、车、建筑等是多个单元（用户 10-12 的规则）；
  - 改字归 `attribute.text`，删字归 remove，加字归 add。
- **服务：**vLLM 0.31，每机一个 DP=8 服务，开前缀缓存。设置与协议改写时相同：`--max-model-len 20000`、`--max-num-seqs 96`、`--limit-mm-per-prompt '{"image": 6}'`。
- **复用：**`protocol_adapt` 的 `Client`、`slot_prompt`、`ground_prompt`、`parse_json`、`normalize_slots`、`to_spec`；`edit_check` 的视图函数与问题清单；打标仓库 `pair_quality` 的维度定义。
- **设备：**GPU，每条约 5 GPU 秒（估算）。

### 3.4 第 4 步：分割 + token
- **SAM3：**`/mnt/bn/strategy-mllm-train/common/models/sam3/sam3.pt`，`confidence_threshold = 0.3`，设置与打标仓库相同：
  - 框小于全图 65% 时，四周留 25% 上下文裁图；
  - 框外扩 2.5%；
  - 三种候选（框提示、文字提示、框加文字），语义优先选择；
  - 框提示要求分数 ≥ 0.65、与框的 IoU ≥ 0.30；文字提示分数 ≥ 0.45。
  - 调用 `crispedit.mask.pipeline.segment_grounded_box`，与 `scripts/data_cleaning/protocol_remask.py` 相同，打标仓库只读导入。
- **add：**先分割锚点，再用 `protocol_adapt.localize_anchor` 裁到落点附近，参数 `ANCHOR_PARAMS`，不再标新物体本身。
- **token：**
  - 用 SAMTok codec（`Qwen3-VL-8B-SAMTok` 目录里的 `mask_tokenizer_256x2.pth`、`sam2.1_hiera_large.pt`）编码 S，再解码；
  - 往返 IoU < 0.5 的不再往下走；
  - 结果写进 token 缓存，格式与 `V3/data/round2_20261016/v1_tokens.jsonl` 相同。
- **复用：**`protocol_remask.py`、`protocol_adapt.sam_prompt`、`localize_anchor`、`samtok_edit21.models.codec`。
- **设备：**GPU，每单元约 1 GPU 秒。

### 3.5 第 5 步：区域把关 + 贴回
- **改动区域：**`edited_region` = 改动图 ∩ 支持区域，算法与参数同 `protocol_adapt.edited_region`。
- **支持区域**（与 50k 相同，替换按 C3 计划放宽）：

  | 操作 | 支持区域 |
  |---|---|
  | attribute | 只取目标 |
  | remove | 目标 + 一圈（约边长的 18%） |
  | replace | 目标 + 外扩约边长的 50% |
  | add | 锚点 + 附着处 |

- **把关：**
  - 协议改写里的几何规则：`edited_region_empty`、`edited_region_misses_s`、add 的 `support_not_touching_item` 和 `support_not_under_item`、`anchor_not_at_item`、`execution_region_covers_host`、`selection_off_edit`。
  - 按 C3 计划加三条：
    - replace 的目标区域改动 ≥ 85%（旧物完全消失）；
    - attribute 的目标区域改动 ≥ 90%；
    - add 的锚点改动 ≤ 60%（附着物不能被改坏）。
- **贴回：**支持区域外换回源图像素，保证区域外逐像素相同。
- **贴回后复查**（`clean_pairs.py` 的 GATE）：

  | 项 | 门限 |
  |---|---|
  | 区域外仍有改动 | ≤ 2% |
  | 区域内改动 | ≥ 10% |
  | 边界被切 | ≤ 0.30 |
  | 接缝 | ≤ 25 |
  | 支持区域占比 | ≤ 60% |

- **多单元：**
  - 每个单元单独算，S 之间不共享、不嵌套；
  - 未解释的改动 ≤ 10%（与 D4 的规则相同）。
- **复用：**`protocol_adapt` 的 `geometry`、`geometry_flags`、`edited_region`、`attach_edited_region`；`clean_pairs.py` 的 `paste`、`ring_metrics`、`stats`。
- **设备：**CPU。

### 3.6 第 6 步：渲染、审核、导出
- **渲染：**用 `instruction_protocol_v2.render` 从 spec 生成带名句和盲指代句；`protocol_adapt.text_flags` 检查措辞。
- **审核：**27B 两票，视图为源图（S 描边）、编辑图、特写。
  - 检查编辑是否完成、关系、目标、轮廓、盲指代句、措辞；
  - 合并了旧编辑核验的盲描述思路：先让模型描述看到的改动，再对照 spec。
  - 不过的带着审核意见重做一次规划或重分割（`protocol_adapt` 的 repair）；仍不过就拒收。
- **导出：**
  - **单目标：**与 D3 的 `manifest_protocol_v2.jsonl` 字段相同，包括 `source_image`、`target_image`、`id`、`dataset`、`instruction_spec`、`instruction_ref`、`instruction_noref`、`selection`（S）、`execution_region`、`edited_region`、`mask_semantics`、`instruction_protocol_version`、`flags`；
  - **多目标：**与 D4 的 `manifest_multi_protocol_v2.jsonl` 相同；
  - 外加 token 缓存，`round2_data` 直接可以构造训练行；
  - 每批写 `README.md`、`summary.json`（各步计数和拒收原因）、抽样审阅图。
- **复用：**`instruction_protocol_v2`、`protocol_adapt.audit_prompt`、`audit_parts`、`build`、`summarize`、`review_sheet`、`protocol_v2_adapt`。
- **设备：**GPU（审核，每条约 3 GPU 秒），CPU（导出）。

## 4. 输出与目录
- **新数据集目录：**`datasets/ExtEdit_Protocol2_<日期>/`，原有的 `CrispEdit_Labeling`、`ScaleEdit_Labeling` 不动。

  | 子目录 | 内容 |
  |---|---|
  | `crispedit/`、`scaleedit/` | 各自的 `ingest/`、`stages/`（每步分片结果）、`final/` |
  | `final/` | `manifest_protocol_v2.jsonl`（单目标）、`manifest_multi_protocol_v2.jsonl`（多目标）、`tokens.jsonl`、`rejected.jsonl`、`summary.json`、`README.md` |

- **代码（v3 仓库，计划）：**
  - `src/samtok_edit21/preparation/ext_pipeline.py`：各步逻辑（尚不存在，待实现）；
  - `scripts/data_pipeline/ext_label.py`（尚不存在，待实现）：命令行，子命令 `ingest`、`precheck`、`plan`、`segment`、`gate`、`audit`、`repair`、`export`、`report`；
  - `src/samtok_edit21/distributed/ext_label.py`（尚不存在，待实现）：四机运行器，写法同现有的 `src/samtok_edit21/distributed/edit_check.py`。
- **环境：**
  - 27B 用 vLLM 0.31；SAM3 用打标仓库的依赖，打标仓库只读导入，设 `PYTHONDONTWRITEBYTECODE=1`；
  - 四机脚本按文档 05 的写法提供可直接粘贴的入口，由用户在 ARNOLD 上提交。

## 5. 产量与算力估算
按旧流程各阶段的通过率相乘，不加场景难度筛：

| 来源 | 预检 | 规划（质量） | 核验 / 审核 | 改写接收 | token | 合计 |
|---|---:|---:|---:|---:|---:|---:|
| CrispEdit | 约 60% | 约 87% | 约 55% | 约 80% | 约 93% | 约 21% |
| ScaleEdit | 约 77% | 约 89% | 约 70% | 约 83% | 约 78% | 约 31% |

- **产量：**每 10 万个范围内的原始对约出 2–3 万条，旧流程约 2,000 条。规划和审核在没经过场景难度筛的行上的通过率可能不同，以校准结果为准。
- **算力：**规划约 5、分割约 1、审核约 3 GPU 秒；预检先丢掉三成，每个原始对约 6–9 GPU 秒，即每 10 万对约 150–250 GPU 小时，四机约 5–8 小时。
- **存储：**
  - CrispEdit 剩余约 2.4 TB；
  - ScaleEdit 每片约 31 GB，按需整片下载、处理完删缓存；
  - 解出的图片约每对 5 MB。

## 6. 第一批的类型配比（建议）
目标是补 C3 的差距（文档 01 §2.11）：替换、大目标改属性、写字。

| 子集 | 来源 | 原始对 | 预计产出 | 说明 |
|---|---|---:|---:|---|
| 替换 | ScaleEdit 物体替换 2.5 万（本地全部）+ CrispEdit replace 3.5 万（本地） | 6 万 | 约 1.6 万 | 优先轮廓变化大的（规划时标出） |
| 改属性 | ScaleEdit 改色 / 改材质 2 万 + CrispEdit color 2 万（本地） | 4 万 | 约 1.1 万 | 预检时按改动面积留一半 ≥ 1% 的大目标 |
| 写字 | ScaleEdit 物体表面文字 + 建筑文字 2 万（本地） | 2 万 | 约 5,500 | 多为改字；要求 OCR 读出与指令一致（§3.6 审核里加一项） |
| 删除、添加 | 两个来源各 1 万（本地） | 4 万 | 约 1.1 万 | 维持比例，防止遗忘 |
| 合计 | | 16 万（全部取自本地，见 §1.4） | 约 4.3 万 | 四机约 1 天，不用下载 |

## 7. 校准与上线
1. **校准**（约 0.5 天，单机）：
   - 每个来源抽 1,000 条已经走完旧流程、各阶段去留已知的行，用新管线重跑；
   - 看两项：旧流程接收的行有多少被新管线接收，旧流程拒收的行有多少被新管线拒收；
   - 抽 100 条新接收的行看图，估计准确率。
   - 放行线：抽查准确率 ≥ 90%，不低于旧流程抽查的约 86%；旧流程接收的行里新管线接收 ≥ 80%。
   - 不达标就改规划提示词、阈值，或恢复旧流程的某一道检查。
2. **冒烟**（单机）：5,000 个新原始对，检查各步计数、格式、token 缓存和 `round2_data` 能否直接读。
3. **正式**（四机）：按 §6 分批，每批写 README 和计数，抽查 100 条后再交给训练。

## 8. 风险与待确认
1. **去掉场景难度筛（产量约 ×10 的主要来源）：**会多出目标唯一的简单编辑。好处是协议 2.0 盲指代下仍需读 token，区域大小也更接近 benchmark；`comparable_instances` 标签保留，可按需提高有近邻的行的比例。
2. **27B 合并成两次调用：**一次调用要同时做判断和填槽位，可能比分开时差。用校准结果判断，必要时把编辑核验单独留一次。
3. **CrispEdit 高产区间是模板句：**指令最终由协议 2.0 重新渲染，原句只作参考，影响有限。
4. **ScaleEdit 许可：**CC BY-NC-SA 4.0，只能非商业使用，训练出的模型和数据的使用范围要确认。
5. **存储与下载：**第一批取自本地，不用下载（§1.4）。之后若要上游的新数据：ScaleEdit 必须整片下载，CrispEdit 剩余部分约 2.4 TB。
6. **两个打标仓库：**只读引用，不修改；CrispEdit 仓库本地未推送的提交与新管线无关。

## 9. 时间表（确认后）
| 日期 | 事项 |
|---|---|
| 10-21 | 写管线代码，单元测试，小样本自测 |
| 10-22 | 校准（§7.1），根据结果调整 |
| 10-23 | 冒烟；给出四机入口脚本 |
| 10-23～10-24 | 四机第一批（§6），用户提交 |
| 10-25 | 抽查、转成训练行，与数据线的新数据（D7–D9）一起进入 C3b |
