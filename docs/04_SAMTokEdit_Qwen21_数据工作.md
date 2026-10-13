# SAMTokEdit（Qwen-Image-2.1）数据工作：训练数据、评测 benchmark、清洗与协议适配

- **更新**：2026-10-13。
- **本文档讲什么**：数据侧的全部现状——数据从哪里来、v2 训练实际用的是什么、数据有什么问题、做了哪些清洗、怎样适配到指令协议 1.0、评测集是什么、哪些目录是中间产物、现在进行到哪一步。方法侧的实验（E-A1、E-A2 等）这里只写它们怎么用数据，实验本身见 [03_SAMTokEdit_Qwen21_v3优化计划与进度.md](03_SAMTokEdit_Qwen21_v3优化计划与进度.md)。
- **当前真正在用的数据看 0.1 节的表。** 中间产物、试验产物和已被取代的产物列在第 7 节，逐条注明了状态；附录 A 是每一条路径的核对记录。
- **核实方式**：文中的路径和数字在 2026-10-13 重新核对过，用的是 `ls`、`wc -l`、`sha256sum`，以及只用标准库的 Python 计数（parquet 行数读文件尾元数据），没有运行仓库代码。只来自文档或笔记、没有重新核对的数字标为「（文档记载，未复核）」。来源之间的不一致列在附录 B。
- **路径约定**：
  - 以 `datasets/`、`experiments/`、`experiments2/` 开头的路径，都相对 `/mnt/bn/strategy-mllm-train/user/tanyue/`。
  - 「v3 仓库」是 `/opt/tiger/tanyue/samtok_edit_qwen-image-2.1-v3`（分支 `qwen-image-2.1-v3`），之后的开发都在这里。「v2 仓库」是 `/opt/tiger/tanyue/samtok_edit_qwen-image-2.1-v2`（分支 `qwen-image-2.1-v2`），数据侧代码在那里开发到提交 `17b2dee`，已经并入 v3 分支，两边的数据侧文件相同。仓库内路径相对仓库根目录。
  - 「docs/04、docs/07、docs/08、docs/09」指 v2 阶段的文档，现在在 v3 仓库的 `docs/archive/v2/` 下。其中 docs/09（训练数据诊断与改进）是逐对审计和清洗的完整记录，本文是它的提要，并补上了之后的进展（协议 1.0 适配）。
  - 「文档 02」「文档 03」指 v3 仓库 `docs/` 下当前的 v2 分析、v3 优化计划与进度。

---

## 0. 一页总览

### 0.1 现在在用的数据

| 用途 | 路径 | 规模 | 状态 |
|---|---|---|---|
| **当前推荐的训练数据底座**（像素清洗 + MLLM 核验 + 去掉 text、action） | `datasets/SAMTokEdit_Cleaned_20261009/edit_check_qwen38_27b_v1/manifest_kept_no_text_action.jsonl` | 46,482 对<br>每对 2 张图，在 `datasets/SAMTokEdit_Cleaned_20261009/images/`<br>269 MB，sha256 `b6de5863…` | 还没转成训练清单<br>协议 1.0 全量改写的输入 |
| 核验后、去 text/action 之前 | `datasets/SAMTokEdit_Cleaned_20261009/edit_check_qwen38_27b_v1/manifest_kept.jsonl` | 48,437 对 | 协议 1.0 改写从这里取回 1,521 条 text |
| **v2 训练清单**（v2 全部模型的训练数据） | `experiments2/SAMTokEdit/qwen21_v2/data/train_v2_box_001/` | `stage1.jsonl` 110,079 行<br>`stage2.jsonl` 290,006 行<br>`provenance.jsonl` 400,085 行<br>来自 97,368 个源编辑对 | 没有清洗过<br>5 个 v2 运行的 `manifest.json` 都指向它<br>目前是 Stage A 的 token 和图片来源 |
| 像素清洗的逐对判定 | `datasets/SAMTokEdit_Cleaned_20261009/decisions.jsonl` | 98,574 行 | Stage A 用来挑「干净真实编辑」：`decision == "keep"`，且来源为 RefEdit 或 Derived |
| GRES/VER 指代数据 | `datasets/SAMTok_Training_Data/mix_gres8k_ver4k_train.parquet` | 12,337 行 | E-A2 和 v3 `grounding_data.py` 的多实例指认来源<br>也是 Derived 的源数据 |
| Stage A 训练和评测数据（E-A1、E-A2 实际用过的） | `experiments2/SAMTokEdit/qwen21_v3/data/stageA_ea1_20261011/`<br>`experiments2/SAMTokEdit/qwen21_v3/data/stageA_ea2_20261012/` | E-A1：`train.jsonl` 42,000 行<br>E-A2：`epoch_rows.jsonl` 每轮 82,000 行，`schedule.json` 164,000 条<br>两者 `eval.json` 各 1,930 项 | 2026-10-13 迁来，迁移记录见 `qwen21_v3/MOVED_20261013.tsv`<br>按 `qwen21_v3/README.md`：E-A2 的数据在用（E-A2b 也用），E-A1 的数据留档 |
| v2 模型评测协议 | `experiments2/SAMTokEdit/qwen21_v2/eval/protocol_v2_001/cases.jsonl`<br>`experiments2/SAMTokEdit/qwen21_v2/eval/protocol_v2_001/compiled.jsonl` | 715 个 case：dev 183、test 532 | v2 的全部评测都用它<br>E-A 的 dev 指认项取自其中 dev 的 mask case |
| 指令协议 1.0 的规范来源 | `datasets/samtok_edit_benchmark_v2/` | 200 张图，785 个编辑目标 | 2026-10-13 升到 release 2.4.0<br>还没有用于模型评测 |
| 协议 1.0 适配的验证样本 | `experiments2/SAMTokEdit/qwen21_v2/data/protocol_v1_pilot_20261012/`<br>`experiments2/SAMTokEdit/qwen21_v2/data/protocol_v1_validation_20261012/` | 试点 40 行<br>r3–r9 共用 4 批样本：230、160、119、206 行 | 只是验证，全量还没跑 |

### 0.2 接下来会用、但还没有产出

1. **协议 1.0 全量改写结果。**
   - 输入 47,402 行，即 46,482 对减去 601 对「无单一类型」，再加上 1,521 条 text。这是按 v2 仓库 `protocol_adapt.load_rows` 的取法算出的。
   - 代码是 v2 仓库提交 `17b2dee`，adapter 版本 `protocol-adapt-2`。
   - 每条接受的记录包含：`instruction_spec`、ref 和 no-ref 两句指令、`selection`（S，即 prompt 里的 ⟨M⟩）、`execution_region`、`edited_region`。多目标行单独记录。
   - 还没开跑，需要用户放行。输出目录也没建；按数据放置约定，应在 `datasets/SAMTokEdit_Cleaned_20261009/` 下新建一个子目录。
2. **训练清单构建器。** 从协议记录到训练清单（v2 的 Stage 1/2 格式，或 v3 需要的格式）的构建器，两个仓库都还没有。v2 的 `src/samtok_edit21/preparation/v2_data.py` 只能转换 v1 语料。
3. **Stage B 数据**：反事实对、原版教师蒸馏目标、干净的 remove 对。这是 E-A2 之后的方法侧计划，还没开始（文档记载，未复核）。
4. **用户另一条线的 SA-1B 新训练数据**：在 `datasets/sa1b_edit_work/`，还在迭代，没有定稿集（1.7 节）。

---

## 1. 数据来源盘点

### 1.1 四个打标子集一览

| 来源 | 最终数据集 | 行数（分片） | 进入 `sources.jsonl` | 像素清洗后 | MLLM 核验后 | 推荐集 46,482 中 |
|---|---|---:|---:|---:|---:|---:|
| RefEdit | `datasets/RefEdit_Labeling/final/RefEdit-mask-prefiltered-qwen38-self-contained/` | 7,804（105） | 7,804 | 7,108 | 4,570 | 4,570 |
| CrispEdit | `datasets/CrispEdit_Labeling/final_dataset_39k/` | 38,971（1,844） | 37,728 | 22,806 | 12,444 | 12,400 |
| ScaleEdit | `datasets/ScaleEdit_Labeling/final/scaleedit_25k/` | 25,664（1,073） | 25,085 | 19,368 | 13,519 | 11,608 |
| SAMTok Derived | `datasets/SAMTok_Derived_Edit_Labeling/combined/manifest.jsonl` | 27,957 | 27,957 | 21,674 | 17,904 | 17,904 |
| **合计** | | 100,396 | **98,574** | **70,956** | **48,437** | **46,482** |

- `sources.jsonl` 是四个最终集合并后的源编辑对，位于 `experiments/SAMTokEdit/qwen21_full4_20260928/data/sources.jsonl`，98,574 行。
- CrispEdit 和 ScaleEdit 进入 `sources.jsonl` 的只有 mask 质检为 OK 的行：
  - CrispEdit：OK 37,728、MASK_REVIEW 910、GROUND_FAIL 333；
  - ScaleEdit：OK 25,085、MASK_REVIEW 472、GROUND_FAIL 107。
- 四个子集的打标代码都在仓库 `sam3-crispedit` 的不同分支。本机的 clone 和 HEAD 如下，与 docs/09 第 7 节引用的提交一致：

  | 子集 | 本机 clone | 分支 | HEAD |
  |---|---|---|---|
  | RefEdit | `/opt/tiger/tanyue/sam3-refedit-labeling` | `refedit-labeling` | 72e0d25 |
  | CrispEdit | `/opt/tiger/tanyue/sam3-crispedit-labeling` | `crispedit-labeling` | 407e913 |
  | ScaleEdit | `/opt/tiger/tanyue/sam3-scaleedit-labeling` | `scaleedit-labeling` | c253733 |
  | Derived | `/opt/tiger/tanyue/sam3-samtok-derived-edit-labeling` | `samtok-derived-edit-labeling` | 2274c50 |

- 每个子集目录下都有 `README.md`，写明目录结构和 10-09 整理前后的路径对照。

### 1.2 RefEdit

| 项 | 内容 |
|---|---|
| 目录 | `datasets/RefEdit_Labeling/`：<br>• `source/RefEdit/`：上游 `bpathir1/RefEdit`，105 个 parquet，18,249 行<br>• `stages/`：质量预筛和两轮 mask<br>• `final/RefEdit-mask-prefiltered-qwen38-self-contained/`：`data/` 下 105 个 parquet、7,804 行，双图内嵌 |
| 任务分布（`run_summary.json`） | color_change 1,663，material_change 799，object_addition 2,909，object_removal 1,163，object_replacement 1,270 |
| 实例与 mask 来源 | 实例 9,836 个<br>mask 来源：pcs 4,488、pvs 2,839、hybrid 460、direct_box 16、negative_space_inverse 1 |
| 打标流程 | Qwen3.8-27B 质量预筛（PASS 7,836）→ Qwen3.5-35B grounding → SAM3 分割（文档记载，未复核） |
| 区域标在哪 | add 的区域是 MLLM 在**目标图**上定位的新物体，再按直接缩放映射回源图坐标；其余类型标在源图上 |
| 指令 | 沿用上游原指令；68% 含明确的空间或序数指代，与评测的指代风格接近（文档记载，未复核） |
| 已知特性 | • add、remove 对齐好：逐对审计中三项全过的比例为 96%、94%<br>• attribute、replace 的目标图在区域外被整图轻微重画，对齐率只有 11–17%，清洗时主要靠贴回保留<br>• replace 的区域是原物体与新物体的并集（3.1 节）<br>• MLLM 核验保留率最低：attribute 37%、replace 51% |

### 1.3 CrispEdit

| 项 | 内容 |
|---|---|
| 目录 | `datasets/CrispEdit_Labeling/`，内部结构与整理前相同：<br>• `source/CrispEdit-2M/`：上游 `WeiChow/CrispEdit-2M`，2,078 个 parquet，529,020 行。按分片名，add/color/remove/replace 各 445 片，motion change 128 片，background change 84 片，style 86 片<br>• `prefilter/quality/`、`prefilter/scene/`<br>• `runs/labeling_4node_crispedit_full_localenv_20260925/`<br>• `iterations/`<br>• `final_dataset_39k/`：`shards/` 下 5 个类别子目录，共 1,844 个 parquet、38,971 行 |
| 最终集按源类型 | add 3,654，color 11,833，motion change 614，remove 13,713，replace 9,157 |
| 打标流程 | Qwen3.8-27B 质量预筛（529,020 → 460,518）→ 只看源图的场景难度筛（→ 38,971）→ 同一模型两遍 grounding → SAM3。中间数字 460,518 是文档记载，未复核；529,020 和 38,971 已核 |
| 区域标在哪 | remove、replace、color、motion 标在源图；add 标在目标图，再缩放回源图 |
| 已知特性 | • 52.4% 的样本源图和目标图尺寸差几个像素，伴随整图重绘（文档记载，未复核）<br>• 主四类按 `far_out ≤ 5%` 算的对齐率是 38–44%；对齐、定位、有改动三项全过的只有 32–38%（docs/09 1.1 节）<br>• 像素清洗保留 58–70%，MLLM 核验再保留 50–63%<br>• 核验丢掉的主要是「改到了指令之外的同类物体」 |

### 1.4 ScaleEdit

| 项 | 内容 |
|---|---|
| 目录 | `datasets/ScaleEdit_Labeling/`：<br>• `source/ScaleEdit-filtered-source/`：1,155 个 parquet，299,633 行；`dataset_manifest.json` 记来源为 `QingyuShi/scaleedit-filtered-6m + InternVL-U/ScaleEdit-12M`<br>• `runs/`<br>• `final/scaleedit_25k/`：`shards/` 下 1,073 个 parquet、25,664 行 |
| 最终集任务（`dataset_manifest.json`） | object_addition 6,077<br>object_removal 5,861<br>object_replacement 2,996<br>color_change 2,560<br>material_change 2,195<br>action_editing 2,327<br>4 种文字编辑合计 2,141<br>compositional 787，size 157，count 54<br>4 种推理类合计 509 |
| 打标流程 | 质量预筛 → 场景难度筛（→ 25,664）→ grounding → SAM3。源图和目标图原样透传，不缩放也不裁剪（文档记载，未复核） |
| 区域标在哪 | add 标在目标图，按直接缩放映射回源图；视角不同时必然错位 |
| 已知特性 | • 同尺寸样本才是真正几何错位的一批：整体缩放约 1.075 倍、平移约 42 px（中位数，文档记载，未复核）<br>• add 只有 10% 可直接用<br>• 清洗主要靠配准后贴回：推荐集中 ScaleEdit 11,608 对里有 8,162 对是配准后贴回（`fix_register`）<br>• 含 text 2,053 对、action 2,407 对、「无单一类型」1,267 对（按 `sources.jsonl` 的类型统计） |

### 1.5 SAMTok Derived

| 项 | 内容 |
|---|---|
| 目录 | `datasets/SAMTok_Derived_Edit_Labeling/`：<br>• `combined/manifest.jsonl`：27,957 行，这是最终数据集<br>• `combined/summary.json`、`by_type/`、`provenance/`、`gallery/`<br>• `runs/four_node/`：两次四机运行<br>• `runs/launchers/`<br>按约定，`combined/` 的路径不能改，因为 v2 清单引用它 |
| 类型 | remove 7,990，add 6,269，replace 5,016，attribute 8,682 |
| 源 | `datasets/SAMTok_Training_Data/mix_gres8k_ver4k_train.parquet`，12,337 行。按 `source_subset`：ver 14,898 行、gres 13,059 行。按（`source_subset`, `parquet_row_index`）去重后共 7,474 张不同的源图。文档另记源数据有 7,671 张图、10,633 个区域（未复核） |
| 源图存储 | 同一张源图存了两份：<br>• `combined/images/sources/remove/`：6,238 个文件<br>• `combined/images/sources/multitype/`：6,936 个文件<br>其中 5,700 个文件名两处都有。抽查 3 对：解码后像素完全相同，文件字节不同（PNG 编码不同） |
| 生成方式 | mask 直接复用 GRES/VER 的实例 mask，不重新分割。<br>• remove：Qwen-Image-2.1 在裁切块上重绘后写回<br>• replace：生成新物体<br>• attribute：只在 mask 内粘贴<br>• add：在宿主上加一个小物件<br>（文档记载，未复核） |
| 质量标签 | `summary.json` 中 `quality_label = model_pass_not_human_verified`。冻结集人工复核 65/96 通过（文档记载，未复核） |
| 已知特性 | • add 是「小贴纸」：6,269 条指令中 5,371 条含 small/tiny，1,925 条含 sticker（已核）<br>• add 的 mask 是宿主实例，协议 1.0 下就是锚点<br>• attribute 有 444 条区域内改动低于 10%（文档记载）<br>• remove 有平涂式填充（文档记载）<br>• v2 转换时 6,269 条 add 全部改归 attribute |

### 1.6 其他相关数据

- **`datasets/SAMTok_Training_Data/`**（SAMTok 的训练数据，与其他用途共用）：
  - `mix_gres8k_ver4k_train.parquet`：12,337 行；
  - `gres8k_train.parquet`：8,337 行；
  - `sam_info.json`：1.32 GB，SAMTok 的 SA-1B 训练图片索引。benchmark v2 用它查图片是否在索引中（6.3 节）。
- **v1 数据中间层**：`experiments/SAMTokEdit/qwen21_full4_20260928/data/`
  - `sources.jsonl`：98,574 行。它是清洗的输入，Stage A 也从这里取 ref 短语。
  - `assets/{crispedit,refedit,scaleedit}/`：每条记录各拷一份源图和目标图。v2 清单引用这些路径，清洗后数据集用硬链接指向它们。
  - `semantic_runs/`：5 个 noref 语义转换运行。生产版是 `qwen21_noref9b_rules_4n_full_003`：接受 97,361 条（LLM 96,593 + 规则 768），失败 1,213 条（文档记载，未复核）。
  - `train_full_9b_rules_003/`：v1 语料，也是 v2 转换的输入。
- **评测的上游原始数据**：
  - `datasets/CompBench/`、`datasets/HumanEdit/`、`datasets/MIRAGE/`：HF 原始下载，benchmark v0 由它们构建；
  - `datasets/PACO_benchmark_candidates/`：benchmark v1 的 goal1k 候选。

### 1.7 `datasets/sa1b_edit_work/`：用户另一条线的 SA-1B 训练数据管线（实验中，不要动）

- **是什么**：用户新的训练数据生成管线的工作目录，基于 SA-1B 图片及其 dense mask。对应仓库是 `sa1b-samtok-edit-data`，本机没有 clone。
- **流程**：
  1. Qwen3.8-27B 按协议 1.0 规划每个 mask 的编辑，指令由同一个渲染模块生成；
  2. Qwen-Image-2.1（vLLM-Omni，40 步，seed 0）生图；
  3. 确定性检查、Qwen3.8 语义审核、助手逐图复核。
- **结构**：25 个子目录，按日期迭代，10-10 至今天。今天还在写入的有：
  - `full30_concise2_20261013/`：30 张源图、1,044 个 mask，07:44 仍在写；
  - `add_anchor_concise_20261013/`；
  - `instruction_concise2_20261013/`。
- **候选池** `disjoint_pool/candidates.jsonl`：1,044 张图、127,123 个 mask。报告写明与 benchmark v2 的 200 张源图 ID 交集为 0、字节哈希交集为 0，**没做近重复检查**。
- **说明文档**：没有顶层 README。只有少数子目录有说明：
  - `natural_edits_v1_20261012/README.md`
  - `protocol1_v5_20261012/质量分析_中文.md`
  - `quality_iteration_v1/*.md`
  - `visual_examples/manual_visual_review.md`
- **现状**：
  - 已知最好的一批是 `quality_iteration_v1/final_release/`：13 张图、44 对通过发布门；
  - `protocol1_v5_20261012` 的结论是「协议一致性已跑通，目标图质量仍不足以无人工筛选地扩量」，其中 add 只达标 2/8。
- 它与本文的 v2 数据线是**两条独立的线**。下文不再展开。

---

## 2. v2 训练实际使用的数据：`train_v2_box_001`

### 2.1 文件

目录：`experiments2/SAMTokEdit/qwen21_v2/data/train_v2_box_001/`

| 文件 | 行数 | sha256（已核） |
|---|---:|---|
| `stage1.jsonl` | 110,079 | `e2312e33…87bf0c` |
| `stage2.jsonl` | 290,006 | `cb46a3f0…21cd` |
| `provenance.jsonl` | 400,085（= stage1 + stage2，每个输出行一条） | `2fc4dd0f…3fce` |
| `conversion_report.json`、`metadata_report.json` | — | — |

- **原件**：10-06 从 `/mnt/bn/strategy-mllm-train/intern/users/tanyue/experiments/SAMTokEdit/qwen21_v2/data/train_v2_box_001/` 逐字节复制而来。原件仍在，三个 sha256 与上表相同（已核）。
- **谁在用**：5 个 v2 运行的 `experiments2/SAMTokEdit/qwen21_v2/runs/*/manifest.json` 都引用 user 路径（已核）：
  - `qwen21_v2_4n_A_s1_b0_001`
  - `qwen21_v2_4n_A2_s2_b0_001`
  - `qwen21_v2_4n_S2_bias_clause_001`
  - `qwen21_v2_4n_S2_embed_001`
  - `qwen21_v2_4n_S2_rope_001`
- **说明文档**：`docs/archive/v2/04_SAMTokEdit_Qwen21_训练数据盘点.md`（v2 阶段的训练数据盘点，已归档）。
- **smoke 子集**：`experiments2/SAMTokEdit/qwen21_v2/smoke/data_smoke_001/`，stage1 80 行、stage2 212 行，覆盖 72 个源。

### 2.2 怎么来的

```text
四个打标子集（mask 质检 OK 的行）
  → sources.jsonl（98,574 个源编辑对）
  → v1 noref 语义转换：Qwen3.5-9B + 规则回退（semantic_runs/qwen21_noref9b_rules_4n_full_003）
  → v1 语料 train_full_9b_rules_003（用发布版 codec 编码 mask）
  → v2 转换：python -m samtok_edit21.preparation.v2_data
     v2 仓库提交 c411c1c；conversion_report.json 记录的 code_commit 是 c411c1cd…
```

v2 转换做的事（`conversion_report.json`、`provenance.jsonl`，已核）：

| 规则 | 涉及 |
|---|---|
| 剔除 composite | 1,206 个源（CrispEdit 60、ScaleEdit 1,146），这些源的所有行都不输出 |
| add 的区域改成框 ⟨B⟩ | 12,556 个源（RefEdit 2,904、CrispEdit 3,525、ScaleEdit 6,127）。框由生成 v1 mask token 的同一实例 mask 计算，向外取整到 0–1000；全部是单实例单框 |
| Derived 的 add 改归 attribute | 6,269 个源；保留宿主 ⟨M⟩ 和原 prompt |
| 新增 rec_ntp 定位回放 | 从 26,569 个候选中按 `sha256("rec:"+id)` 取 13,760 条，只取 Derived 和 RefEdit 的单实例非 add 单元 |
| 其余行 | 与 v1 逐字节相同（`provenance.jsonl` 中 `row_hash == v1_row_hash`） |

### 2.3 行数（全部已核）

**Stage 1**（`stage1.jsonl`）：

| 类型 | edit_ntp | rec_ntp |
|---|---:|---:|
| add | 12,556 | — |
| remove | 27,698 | 4,760 |
| replace | 18,198 | 3,232 |
| attribute | 33,217 | 5,768 |
| action | 2,666 | — |
| text | 1,984 | — |
| 合计 | 96,319 | 13,760 |

预测目标的形式：
- edit_ntp：`mask_2d` 83,763 条，`bbox_2d` 12,556 条（add）；
- rec_ntp：全部是 `bbox_2d`。

**Stage 2**（`stage2.jsonl`；`edit` 是不带区域的原指令行，`edit_umt` 是带区域 token 的行）：

| 类型 | edit | edit_umt ref | edit_umt noref |
|---|---:|---:|---:|
| add | 12,659 | 12,556 | 12,556 |
| remove | 28,059 | 27,698 | 27,698 |
| replace | 18,222 | 18,198 | 18,198 |
| attribute | 33,367 | 33,217 | 33,217 |
| action | 2,995 | 2,666 | 2,666 |
| text | 2,066 | 1,984 | 1,984 |
| 合计 | 97,368 | 96,319 | 96,319 |

**按来源**（`provenance.jsonl`）：

| 来源 | 源编辑对（Stage 2 edit 行） | edit_ntp | edit_umt ref、noref（各） | rec_ntp |
|---|---:|---:|---:|---:|
| CrispEdit | 37,668 | 37,176 | 37,176 | — |
| ScaleEdit | 23,939 | 23,428 | 23,428 | — |
| RefEdit | 7,804 | 7,792 | 7,792 | 2,566 |
| Derived | 27,957 | 27,923 | 27,923 | 11,194 |
| 合计 | 97,368 | 96,319 | 96,319 | 13,760 |

**add 框占图像面积的中位数**：RefEdit 0.125，CrispEdit 0.090，ScaleEdit 0.033。

### 2.4 区域与指令的格式

- **mask token**：`<|mt_start|><|mt_XXXX|><|mt_YYYY|><|mt_end|>`。
  - 每个 mask 是 2 个码本 token。
  - Stage 2 带 mask 的 `edit_umt` 行每种变体 83,763 行，**每行恰好 1 个 span、2 个 token**（已核）。
  - 多实例单元在 v1 时已经取并集，编码成一个 mask（`mask_ids=['union']`，docs/09 7.5 节）。
- **框 token**：`<|box_start|>[x1, y1, x2, y2]<|box_end|>`，坐标是 0–1000 的整数，只用于 add，每种变体 12,556 行。`edit` 行不带任何区域 token。
- **ref 变体**：原指令，把 ⟨M⟩ 或 ⟨B⟩ 插在被指代的短语之后。
  - 例：`Change the leftmost bird's feathers <M> to soft down feathers`
- **noref 变体**：去掉指代短语，改成 `this region ⟨M⟩/⟨B⟩`。100% 的 noref 行含 `this region`（已核）。各类型常见的开头（前 4 个词，已核）：

  | 类型 | 开头 | 行数 |
  |---|---|---:|
  | remove | `Remove the object in`（不区分大小写） | 27,360/27,698（98.8%） |
  | replace | `Replace the object in`（不区分大小写） | 17,283/18,198（95.0%） |
  | attribute（33,217 行） | `Turn this region ⟨M⟩` | 9,316 |
  | | `Change the color of` | 9,257，另有小写开头 771 |
  | | `Change this region ⟨M⟩` | 4,265 |
  | | `Add a small red` / `Add a small round`（来自 Derived 的 add） | 2,476 / 380 |
  | | `Transform this region ⟨M⟩` | 1,262 |
  | | `Change the material of` | 491 |
  | text | `Replace the text in this region ⟨M⟩ with`（区分大小写） | 1,842/1,984（92.8%） |
  | add | 含 `in this region ⟨B⟩` | 12,556/12,556（100%） |

- **noref 改写方式**：Qwen3.5-9B + 规则回退，就是 v1 的语义转换（文档记载）。
- **训练读取配比**（文档记载，未复核）：
  - Stage 1：每个 update 中 edit_ntp : rec_ntp = 7 : 1；
  - Stage 2：ref : noref : plain = 1 : 2 : 1，类型按 `main4` 采样，四个主类型约占 95%。

### 2.5 图片

- Stage 2 的 97,368 个源编辑对，用到 82,585 个不同的源图路径。
  - 69,411 个在 `experiments/SAMTokEdit/qwen21_full4_20260928/data/assets/`，每条记录一份拷贝；
  - 13,174 个在 `datasets/SAMTok_Derived_Edit_Labeling/combined/images/`。
- 同一张图对应多条记录的情况见 3.2 节。

### 2.6 Stage A（E-A1、E-A2、v3）怎么用这份数据

E-A1 和 E-A2 的数据构建脚本现在在：
- `experiments2/SAMTokEdit/qwen21_v3/legacy_code/ea1/ea1_build_data.py`
- `experiments2/SAMTokEdit/qwen21_v3/legacy_code/ea2/ea2_build_data.py`

v3 仓库把同样的做法写成了 `src/samtok_edit21/preparation/grounding_data.py`（命令 `samtok-edit ground-data`，说明见文档 01 的 3.6 节），并核对过它重建出的评测集和 E-A2 训练数据与当时的文件逐字节相同。三者取数方式相同：

- **单元**：从 `stage2.jsonl` 取 `edit_umt` 行，去掉 add，只要**恰好 1 个 mask span、没有框 token** 的行。（图片, span）构成一个单元，ref 和 noref 两行合成同一个单元。
- **干净真实编辑**：在 `datasets/SAMTokEdit_Cleaned_20261009/decisions.jsonl` 中，`decision == "keep"` 且来源为 RefEdit 或 Derived 的对，用 Stage 2 原始图片和原 prompt。注意：
  - 没有用 MLLM 核验结果（`verdicts.jsonl`）；
  - 没有用 `fix_resize`、`fix_register` 的对。
- **`decisions.jsonl` 的 `decision` 取值**（98,574 行，已核）：

  | 取值 | 含义 | 行数 |
  |---|---|---:|
  | `keep` | 原样保留 | 42,154 |
  | `fix_resize` | 贴回 | 5,528 |
  | `fix_register` | 配准后贴回 | 23,274 |
  | `drop_gate` | 贴回后检查不过 | 8,175 |
  | `drop_C` | 区域外有成块改动 | 10,974 |
  | `drop_M` | 错位且配准不可用 | 1,375 |
  | `drop_D` | 区域内几乎没改动 | 7,094 |

  `clean_trial_20261009` 里没有全量判定文件，只有逐对审计 `audit/audit.jsonl` 和 6 批各 640 对的试验判定。
- **多实例指认**（只有 E-A2 和 v3 用）：`mix_gres8k_ver4k_train.parquet`。E-A2 取了 8,997 个 GRES 单元，其中 3,933 个是多实例；每轮 12,000 行。
- **划分**：按 `sha256(源图路径) % 10 == 0` 留出；E-A2 起另加 16×16 均值哈希防泄漏（3.2 节）。

已用过的数据量（各自的 `data_stats.json`，已核）：

| | E-A1（`stageA_ea1_20261011/`） | E-A2（`stageA_ea2_20261012/`） |
|---|---|---|
| Stage 2 单元 | 73,083，其中训练 65,841 | 65,037（按内容哈希排除 529 个与评测图相同的源之后） |
| 可用的干净真实编辑行 | 40,168 | 38,452 |
| 训练行 | 42,000：指认 30,000、程序化编辑 8,000、真实编辑 4,000 | 每轮 82,000，共 2 轮（`schedule.json` 164,000 条）：<br>• 指认 62,000（含 GRES 12,000）<br>• 程序化编辑 12,000<br>• 真实编辑 8,000<br>其中 27,624 行带同图干扰实例 |
| 评测项 `eval.json` | 1,930：留出集 ref 393 / noref 607，dev ref 122 / noref 122，反事实对 686 | 同左；另有 730 项标为 `leaky_ea1` |

E-A2 的 `epoch_rows.jsonl` 在 2026-10-13 迁移时改过一处：GRES 图片路径由绝对路径改成相对路径 `gres_images/xxxxx.jpg`，其余内容不变，仍是 82,000 行。

---

## 3. v2 数据的问题

### 3.1 证据与结论

| 问题 | 证据 | 出处 | 结论 |
|---|---|---|---|
| **源图与目标图不对齐**（主要根因） | • Stage 2 主四类 275,645 行中，CrispEdit 和 ScaleEdit 占 168,454 行（61.1%）；行数已按 provenance 复算<br>• 逐对审计：主四类 48% 可直接用，10% 贴回可修，35% 改动扩散，8% 空操作或过弱<br>• `far_out ≤ 5%` 的对齐率：ScaleEdit add 10%、CrispEdit 主四类 38–44%、RefEdit attribute/replace 11–17% | docs/08 4.1 节；docs/09 第 1 节 | 模型学到整图重绘和重新取景 |
| **ScaleEdit 整体缩放漂移** | • ScaleEdit 同尺寸样本配准：缩放中位 1.075、平移 42 px，配准后区域外改动从 0.354 降到 0.022<br>• 训练对中 `fix_register` 的 ScaleEdit 样本尺度中位 0.929，92% 是缩小<br>• B0 和 E5 不融合时输出漂移率 17–20%，不含区域 token 的纯指令也会漂移 | docs/09 第 2 节；文档 02 §2.2（文档记载，未复核） | 漂移来自训练对本身，是 DiT 的通用退化，不是区域条件引起的 |
| **软补丁** | • 贴回后编辑区的锐度只有周围的 0.62–0.70（CrispEdit、ScaleEdit 的贴回样本），见文档 02 §2.3（文档记载）<br>• 清洗后主四类 66,684 对中贴回样本 26,682 对（40.0%），其中 CrispEdit+ScaleEdit 23,002 对（已核）<br>• 推荐集 46,482 对中贴回 15,921 对（`fix_resize` 2,472 + `fix_register` 13,449，已核） | 文档 02 §2.3 | 清洗把「锐利背景 + 软补丁」变成了显式监督，当前清洗修不掉 |
| **小部件** | • 清洗后 attribute 区域面积中位 6.3%，dev 只有 0.49%（文档 02 §2.4）<br>• 推荐集 46,482 中 attribute 区域中位 6.14%、低于 0.5% 的占 10.3%（已核） | 文档 02 §2.4 | 训练区域普遍比评测的部件大 |
| **add 的区域是新物体本身**（标在目标图） | • 推荐集的 add 8,718 对中：7,825 对只有目标图实例，865 对（Derived）是宿主 mask（已核）<br>• codec 往返 IoU：add 0.38、remove 0.88<br>• add 框内改动比例：CrispEdit 0.42、ScaleEdit 0.08（文档记载）<br>• pass 1 的 add 框 Acc@0.5 只有 0.23（文档记载） | docs/09 7.1 节；docs/08 4.1 节；docs/07 5.2 节 | 新物体在源图里不存在，SAMTok 表示不准，推理时也拿不到。协议 1.0 改为锚点 |
| **区域含目标侧并集**（RefEdit replace） | • 并集比原物体中位大约 16%；一半样本里新物体有超过 20% 落在原物体之外（文档记载）<br>• 推荐集中 RefEdit replace 543 对，532 对有多个实例（已核）<br>• 全部 replace 中有 560 对同时有源图和目标图实例（已核） | docs/09 7.1 节 | 新物体的位置和形状被带进了输入。协议 1.0 要求 S 只取源图实例 |
| **add 任务不对口** | • 评测的 add 是「在同类多实例场景里再加一个」，训练中只占 1–3%（文档记载）<br>• Derived 的 add 是小贴纸：5,371/6,269 含 small/tiny，1,925 含 sticker（已核） | docs/09 4.1 节 | 下游「加出来的物体太小」与之对应 |
| **空操作、平涂 remove、指令与区域不一致** | • Derived attribute 有 444 条改动低于 10%<br>• 平涂 remove：250 个抽样中 17 个区域内灰度标准差 <10<br>• 指令与区域指向不同的例子，如 `refedit:11175`<br>以上均为文档记载 | docs/09 4.2–4.4 节 | MLLM 核验的问题类型里，scope、wrong_target 很多（4.3 节） |
| **text 和 action** | • action：2,895 → 像素清洗后 1,341 → 核验后 434（已核）<br>• text：2,053 → 1,950 → 1,521（已核）<br>• 评测没有 action，只有 1 个 text case；v1 权重下这两类占约 24% 的抽样（文档记载） | docs/04 第 7 节；docs/09 6.15 节 | 10-12 去掉；协议 1.0 适配又把 text 按编辑内容加回（第 5 节） |
| **一个单元标了多个实例**（协议 1.0 要求一单元一实例） | 推荐集 46,482 中 6,477 对有多于 1 个标注实例（已核）。例如：<br>• CrispEdit remove 1,326/4,364<br>• ScaleEdit remove 1,185/3,265<br>• ScaleEdit「无单一类型」391/601 | 本次统计 | 协议适配中一部分会判为多目标而不进主集；一堆同类物品算一个对象 |

### 3.2 同一张图以不同路径多次出现（内容级泄漏）

**现象**

- Stage A（F1/F3/F5、E-A1）的留出集按「源图路径的 sha256 % 10」划分。但同一张图会以多个路径出现：
  - 同一张图对应多条编辑指令，v1 `assets/` 给每条记录各拷一份；
  - Derived 的源图在 `remove/` 和 `multitype/` 各存一份。

**本次核对的数字**

| 项 | 数字 |
|---|---|
| Stage 2 源编辑对 / 不同源图路径 | 97,368 / 82,585 |
| 其中 Derived | 27,957 对 / 13,174 个路径 |
| 82,585 个路径中逐字节相同的文件组 | 3,139 组、6,421 个文件，即多出 3,282 份拷贝 |
| 字节相同组的分布 | 几乎都在同一来源内：CrispEdit 1,644 组、RefEdit 1,361 组、ScaleEdit 123 组；其余 11 组未细查 |
| Derived 两处都存的源图 | 5,700 张。抽查 3 对，像素相同、字节不同，**按文件字节哈希查不出来** |
| 不同图片数（上限） | 按字节去重，82,585 个路径对应 79,303 份不同内容；再把 Derived 两处都存的 5,700 张各算一张，至多约 73,603 张。没有做感知哈希的全量去重，实际可能更少 |

**影响**

- E-A1 用 16×16 均值哈希复查：评测用的 1,160 张图中，363 张以另一个路径出现在训练集里；受影响的有留出集 220 项、343 个反事实对中的 255 对。按干净子集重算后，E-A1 的结论不变（文档 03 的 5.2 节，文档记载，未复核）。
- E-A2 和 v3 `grounding_data.py` 在训练端加了内容防护：16×16 均值哈希与任一评测源图相同的训练源都去掉。E-A2 去掉了 529 个 Stage 2 源和 425 行 GRES，复查后训练与评测重合为 0（已核）。

**建议：按图像内容分组划分**

- 先按图片身份分组，再划分；同一张图的全部记录放进同一侧。
- 图片身份可以用：
  - 解码后像素的哈希，或感知哈希；
  - Derived 可直接用（`source_subset`, `parquet_row_index`）；
  - 三个开源子集也可以用上游 id。
- 文件字节的 sha256 不够，因为 Derived 的重复是重新编码的 PNG。
- 清洗后的数据集和将来的协议 1.0 输出有同样的问题：pair id 唯一，但图片会重复。

---

## 4. 质量清洗

### 4.1 流水线与产物

| 阶段 | 规模 | 做法 | 代码 | 产物 | 性质 |
|---|---|---|---|---|---|
| 1. 逐对审计 | 98,574 | 512 px 下计算 `far_out`、`in_share`、`fill`、`contrast`，分成 A/B/C/D 四类 | `experiments2/SAMTokEdit/qwen21_v2/data/clean_trial_20261009/audit/audit_sources.py` | `clean_trial_20261009/audit/audit.jsonl`（98,574 行）<br>配准试验 `register.jsonl`（750 行） | 诊断 |
| 2. 小批量验证 | 6 批，各 640 对 | 旧规则 `batch_v1`/`batch_v2`/`batch_v3`/`batch_s2`<br>新规则 `batch2_s1`/`batch2_s2` | `clean_trial_20261009/scripts/` | `clean_trial_20261009/results/<batch>/results.jsonl` | 试点 |
| 3. 全量像素清洗 | 98,574 → **70,956** | 4.2 节规则；只用 CPU，19.2 分钟（文档记载） | v2 仓库 `scripts/data_cleaning/clean_pairs.py`、`checks.py`、`verify_cleaned.py`、`reprocess.py`；运行时副本在 `datasets/SAMTokEdit_Cleaned_20261009/scripts/` | `datasets/SAMTokEdit_Cleaned_20261009/`：<br>• `manifest.jsonl` 70,956<br>• `decisions.jsonl` 98,574<br>• `images/` 141,912 个文件<br>• `summary.json`、`verification.json` | 全量 |
| 4. MLLM 核验试点 | 303 对，后扩到 699 对 | 比较 27B 现成打分器、短判定、Flash-Next、27B 问题清单（Q2）和盲描述（Dn2）；人工标注四轮 | 试点脚本在 `experiments2/SAMTokEdit/qwen21_v2/data/mllm_pilot_20261010/` | 同左：<br>• `inputs.jsonl` 303<br>• `q27/labels_final.json`<br>• `flash_next_v2_20261011/round4.json`<br>• `repo_check_20261011/manifest.jsonl` 699 | 试点 |
| 5. 全量 MLLM 核验（方案 2） | 70,956 → **48,437** | Qwen3.8-27B，Q2 和 Dn2 任一判坏即丢；4 机 × 8 卡，约 66 分钟（文档记载） | v2 仓库 `src/samtok_edit21/preparation/edit_check.py`、`src/samtok_edit21/distributed/edit_check.py`、`scripts/edit_check/`；提交 `c1ed983` | `datasets/SAMTokEdit_Cleaned_20261009/edit_check_qwen38_27b_v1/`：<br>• `manifest_kept.jsonl` 48,437<br>• `verdicts.jsonl` 70,956<br>• `summary.json`、`config.json`<br>• `runs/edit_check_27b_4n_002/` | 全量 |
| 6. 去掉 text 和 action | 48,437 → **46,482** | 按 `type` 字段删除，其余行逐字节保留 | v2 仓库 `scripts/data_cleaning/select_types.py`，提交 `727869e` | `edit_check_qwen38_27b_v1/manifest_kept_no_text_action.jsonl`、`.summary.json` | 全量 |

- 四个 manifest 的行与清洗后 `manifest.jsonl` 的对应行逐字节相同，图片路径都相对 `datasets/SAMTokEdit_Cleaned_20261009/`。
- 字段说明见该目录的 `README.md`。所有区域都在源图像素坐标，带 COCO RLE 和 0–1000 的外接框；目标图与源图同尺寸。

### 4.2 像素清洗规则

阈值来自 `summary.json` 的 `rules`（已核）：

| 参数 | 值 |
|---|---|
| `fill_min` | 0.1 |
| `far_out_keep` | 0.02 |
| `blob_max` | 0.01 |
| `align_min` | 0.4 |
| `align_tiles` | 5 |
| `register_gain` | 0.02 |
| `register_cover` | 0.98 |
| `instance_area_min` | 0.9 |

每一对依次判断（docs/09 6.1、6.7.2、6.8.2 节）：

1. **改动过弱就丢**：区域内被明显改动的像素少于 10%，判 `drop_D`。
2. **区域外成块改动就丢**：先做颜色仿射校正，取 |Δ|>30 的像素，用 3 px 开运算去掉细边缘；若离区域 6% 长边以外最大的连通块超过全图 1%，判 `drop_C`。
3. **区域外几乎没改就原样保留**：远区域外的明显改动不超过 2%，判 `keep`。
4. **否则配准后贴回**：SIFT + RANSAC 仿射配准；只有区域外改动下降超过 2 个百分点、且区域被完整覆盖（≥98%）时才采用配准。标在目标图上的实例随同一变换移动；若配准会把这类实例推出画面 10% 以上，就不采用配准。
5. **没有配准的样本做对齐检查**：在 256 px 下取 32×32 的有纹理小块，若位移不超过 1 px 的块少于 40%，判 `drop_M`。
6. **贴回后检查**，任一项不过判 `drop_gate`：
   - 区域外改动 ≤2%；
   - 区域内改动 ≥10%；
   - 边界外一圈被改动的像素 ≤30%；
   - 这一圈的平均差异 ≤25；
   - 贴回区域 ≤ 全图 60%。

后两项阈值是文档记载。

处理方式（`treatment`，已核）：

| 方式 | 对数 | 说明 |
|---|---:|---|
| `link` | 34,733 | 原文件的硬链接 |
| `resize` | 7,421 | 只缩放到源图尺寸 |
| `resize+paste` | 5,528 | 贴回 |
| `register+paste` | 23,274 | 配准后贴回 |

- 更正了坐标的实例：6,307 个，涉及 5,466 对（`verification.json`，已核）。
- `verification.json` 的全量校验：有问题的对数 0；随机 3,000 对重算指标，不一致 0（已核）。
- 处理过的目标图是 PNG，约 102 GB；`images/` 的表观大小约 209 GB（文档记载，未复核）。

### 4.3 MLLM 核验（方案 2）

- **每对的视图**：三张图，原图（红色轮廓标出区域）、编辑后的图、区域放大对照。
- **两项独立核验**：
  - q2：按编辑类型提的问题清单，开思考模式；
  - dn2：先不给指令盲描述改动，再对照指令判定。
- 两项都通过才保留。提示词全文和参数在 `config.json`。

结果（`verdicts.jsonl`，已核）：

| 结论 | 对数 |
|---|---:|
| 保留（两项都通过） | 48,437 |
| 只有 q2 判坏 | 3,393 |
| 只有 dn2 判坏 | 7,633 |
| 两项都判坏 | 11,493 |
| 未核验 | 0 |

判坏时给出的问题类型（已核）：

| 问题 | q2 | dn2 |
|---|---:|---:|
| defect（明显瑕疵） | 4,695 | 6,020 |
| scope（改到了指令之外的物体） | 4,640 | 6,135 |
| partial（只做了一部分） | 2,780 | 2,009 |
| wrong_target（改错对象） | 1,349 | 1,806 |
| not_done（没做） | 1,256 | 2,992 |
| too_weak（改动太弱） | 166 | 160 |
| identity | — | 4 |

人工标注子集上的效果（第四轮标注，84 坏 / 234 好）：抓到 72/84（86%），误判 19/234（8.1%）；粗算核验后坏样本约 5–6%（文档记载，未复核）。

### 4.4 各阶段保留量（全部已核）

| 来源 | 类型 | 源编辑对 | 像素清洗后 | 核验后 | 推荐集 |
|---|---|---:|---:|---:|---:|
| CrispEdit | add | 3,563 | 2,485 | 1,572 | 1,572 |
| | remove | 13,276 | 7,655 | 4,364 | 4,364 |
| | replace | 8,916 | 5,374 | 2,918 | 2,918 |
| | attribute | 11,485 | 7,105 | 3,546 | 3,546 |
| | action | 488 | 187 | 44 | 去掉 |
| ScaleEdit | add | 6,023 | 4,825 | 3,777 | 3,777 |
| | remove | 5,695 | 4,943 | 3,265 | 3,265 |
| | replace | 2,951 | 1,754 | 1,263 | 1,263 |
| | attribute | 4,689 | 3,761 | 2,702 | 2,702 |
| | action | 2,407 | 1,154 | 390 | 去掉 |
| | text | 2,053 | 1,950 | 1,521 | 去掉 |
| | 无单一类型 | 1,267 | 981 | 601 | 601 |
| RefEdit | add | 2,909 | 2,822 | 2,504 | 2,504 |
| | remove | 1,163 | 1,104 | 740 | 740 |
| | replace | 1,270 | 1,075 | 543 | 543 |
| | attribute | 2,462 | 2,107 | 783 | 783 |
| Derived | add | 6,269 | 1,182 | 865 | 865 |
| | remove | 7,990 | 7,856 | 6,454 | 6,454 |
| | replace | 5,016 | 4,657 | 3,957 | 3,957 |
| | attribute | 8,682 | 7,979 | 6,628 | 6,628 |
| **合计** | | **98,574** | **70,956** | **48,437** | **46,482** |

按类型合计：

| 类型 | 推荐集 |
|---|---:|
| add | 8,718 |
| remove | 14,823 |
| replace | 8,681 |
| attribute | 13,659 |
| 无单一类型 | 601 |

推荐集按清洗判定：`keep` 30,561、`fix_resize` 2,472、`fix_register` 13,449。

### 4.5 试点与全量的区分

- **试点和验证**：
  - `clean_trial_20261009`：6 批，各 640 对；
  - `mllm_pilot_20261010`：303 对、699 对，以及 Flash-Next 对比。
- **全量**：
  - 像素清洗：一次，2026-10-09；
  - 方案 2 核验：运行 `edit_check_27b_4n_002`，2026-10-11。之前的 `edit_check_27b_4n_001` 在启动检查时就停了（ARNOLD 镜像缺 DejaVu 字体），没有产生任何结论；
  - 类型筛选：一次，2026-10-12。

---

## 5. 指令协议 1.0 适配

### 5.1 协议 1.0 要求什么

规范正文：
- `datasets/samtok_edit_benchmark_v2/BENCHMARK_V2_ZH.md` 第 5 节（2.4.0 版）；
- `datasets/samtok_edit_benchmark_v2/docs/INSTRUCTION_PROTOCOL_ZH.md`。

代码是无依赖的 `instruction_protocol.py`，vendored 在 v2 和 v3 仓库的 `src/samtok_edit21/preparation/instruction_protocol.py`，sha256 `e93d7d90…`，与 benchmark 2.3.0 时的哈希相同。2.4.0 的变化见 6.3 节。

- **一条语义记录**：字段为 `version`（"1.0"）、`operation`、`ref`、`scope`、`payload`、`target_part`、`remove_mode`、`attributes`、`placement`、`qualifier`。ref 和 no-ref 两句都从它渲染；no-ref 只把 ref 换成 `the selected <scope>`，其余内容相同，用 `validate_pair` 校验。
- **S 与 T**：
  - S 是区域或 ref 选中的源实例，定位信息总是指向 S；
  - T 是实际被编辑的东西：S 本身，或者 S 的一个部件（`target_part`，例如 "the handle of the selected suitcase"）；
  - scope 的粒度必须与区域一致。
- **四类操作**：

  | 操作 | 句式 | 要点 |
  |---|---|---|
  | add | `Add <item> <relation> [<site> of] <S>` | 21 种关系词；`anchor_mode` 默认 support；under/above/below/beside/near 必须是 reference；区域标锚点，不标新物体 |
  | replace | `Replace <T> with <payload>` | 必须有结构或品类变化；只换颜色、材质、图案归 attribute |
  | remove | 六种模式：whole、part（需要 `target_part`）、contents、contents_only、lowerlegs、hem | 删除文字或 logo 归 remove |
  | attribute | 八种属性：color、material、finish、pattern、texture、text、tint、opacity | 同一个 T 上可以有多个属性；tint 和 opacity 只能单独用；修改文字归 `attribute.text` |

  翻转、移动、重新摆姿势不在协议范围内。
- **单元**：一个单元是一个源实例（或其部件）加一种操作；实例或承载关系不清楚时拒收。
- **执行区域**：以下情况必须有显式执行区域，缺失或覆盖整个宿主时拒收（`needs_local_execution`）：
  - 有 `target_part`；
  - remove 的 contents、contents_only、lowerlegs、hem；
  - reference 型 add。
- **2.4.0 新增**（今天）：
  - mask 含义契约 `mask_semantics.version = 1.1`；
  - 写作风格 `benchmark-concise-2-direct-add-1`；
  - add 优先直接承载物，reference 只用于真正紧邻、支撑面明确的情况。

### 5.2 用户为这次适配定的规则

来源：会话笔记 `samtokedit-training-data-state-20261012.md` 与适配目录的 README。这些规则来自笔记，不是从数据推出的。

1. 多目标行不进主集，记录原因，留待做多目标能力时召回。口径：一堆、一串同类物品算一个对象；分开的人、楼、车算多目标。
2. text 按编辑内容加回：
   - 改文字 → `attribute.text`；
   - 删文字 → remove；
   - 加文字 → add。
3. 不扩充数据，只改写已通过清洗的行。这排除了宿主级部件扩充，也排除了从 Derived 的 6,269 条 add 重新筛选。
4. 重打 mask 的情况：
   - add 的 mask 不在锚点上时，重打锚点；
   - remove 或 attribute 的 mask 含目标侧并集时，重打 S。
5. replace 只取原物体。
6. add 锚点（10-13）：
   - 取新物体**直接站、放、挂、附着**的物体、部位或表面，包括地面、水面、天空；
   - 只有承载处被挡住时才用参照物，而且必须紧挨着，不能隔空；
   - S 不能盖住整个大物体，要裁到新物体所在处。

### 5.3 流水线（v2 仓库 `17b2dee` 版本）

入口是 `scripts/data_cleaning/protocol_adapt.py` 的子命令 `pilot / slots / build / audit / report / repair / final`，加上 `scripts/data_cleaning/protocol_remask.py`。每一轮由 `experiments2/SAMTokEdit/qwen21_v2/data/protocol_v1_validation_20261012/scripts/run_round_full.sh` 串起来。

1. **取行**（`load_rows`）：推荐集中有类型的行，加上 `manifest_kept.jsonl` 中的 text 行。text 先按规则草分：attribute 1,479、add 24、remove 18（本次按同一正则复算），最终路由由模型定。
2. **填槽位**（`slots`）：Qwen3.8-27B 看 BEFORE、带编号描边的图、AFTER 和特写，输出受约束的 JSON；不合法时带着问题重问。
3. **定 S**：
   - 原标注已经在 S 上的，逐字节复用；
   - add 的锚点没有对应的源图 mask、S 只在目标图上标过、或点名了部件（`part_as_s`）时，用 SAM3 重打（`protocol_remask.py`，参数与 CrispEdit 打标仓库 407e913 相同）；
   - 17b2dee 起，add 先按新槽位 `s_object` 分割整个物体或表面，再用 `localize_anchor`（`anchor-window-1`）裁到新物体周围，裁前的整块 mask 保存在 `selection.localized.full`。
4. **生成记录**（`build`）：渲染 ref 和 no-ref 两句并校验，做几何把关：
   - 参照型锚点距离超过对角线 1.5% 拒收；
   - 接触类承载超过 3% 拒收；
   - 新物体附近没有 S 拒收；
   - 部件或裁短的执行区域不能覆盖整个宿主；
   - 「连同内容物」要求执行区域包含容器。

   同时计算 `edited_region`（方法 `lab-diff-hysteresis-1`：Lab ΔE、双阈值、按噪声自适应），检不出改动的记录拒收。
5. **独立审核**（`audit`）：27B 投两票，两票都通过才算。检查项：edit、relation、target、outline、noref、wording；add 有专用的 relation、target、outline 规则。
6. **修复**（`repair`）：审核不过的行带着审核意见重填槽位；描边或目标有问题的强制重打 S；几何检查拒收的也一并重问。修复后再审核一次。
7. **把关**（`final`）：只有审核通过、并且带 `edited_region` 的记录保持「接受」。

每条记录的区域字段：

| 字段 | 含义 |
|---|---|
| `selection` | S，即 prompt 里的 ⟨M⟩ |
| `execution_region` | 来自清洗区域、Derived 的 `execution_region`，或重打后的区域 |
| `edited_region` | 像素真正发生变化的区域，RLE 加统计量 |

### 5.4 各轮结果

| 轮次 | 样本 | 代码与改动 | 接受 | 多目标 | 拒收 | 人工看图 |
|---|---|---|---:|---:|---:|---|
| 试点 run1 | 40 行，四个来源 × 四类，加 4 条 text；seed 20261012 | 初版 | 38 | 1 | 1 | 40 条中正确 25、小问题 9、错误 6 |
| 试点 run2 | 同上 | 45ac3d7 | 36 | 4 | 0 | 40 条中正确 35、小问题 4、错误 0、口径待定 1 |
| r3 | 230（试点 40 + 新抽 190） | 加入审核；一堆算一个 | 189 | 41 | 0 | — |
| r4 | 同上 | 修复闭环与把关；部件作为 S | 184 | 41 | 5 | 16 条中正确 12、小问题 2、错误 2（README 记载） |
| r5 | 同上 | 关系、内容物规则 | 186 | 43 | 1 | 20 条中正确 18、小问题 1、错误 1（README 记载） |
| r6 | 160，全新 | 审核改为两票（c03218b） | 130 | 26 | 4 | 25 条中正确 22、小问题 2、错误 1 |
| r6 加 `edited_region` 重建（`final_er2.jsonl`） | 同上 | 818ba2d，只用 CPU | 126 | 26 | 8 | — |
| r7 | 119，全新 | near/beside 几何把关（c03218b） | 98 | 19 | 2 | 20 张图中正确 18、小问题 2、错误 0；另有 7 条只核对文字，全部正确 |
| r7 加 `edited_region` 重建 | 同上 | 818ba2d | 98 | 19 | 2 | — |
| r8 | 206：r6、r7 的全部 add 96 行（含点名的 6 条）+ 新 add 70 行 + 非 add 回归 40 行 | 17b2dee 初版 | 176 | 15 | 15 | 18 条中正确 12、小问题 4、错误 2 |
| r9 | 同 r8 | 17b2dee | 173 | 16 | 17 | 18 条中正确 15、小问题 2、错误 1（两张并排长椅选错一张，crispedit-f049e632…） |

- 判定文件：
  - `protocol_v1_pilot_20261012/review/verdicts_run{1,2}.tsv`
  - `protocol_v1_validation_20261012/review/verdicts_r6_r7.tsv`
  - `protocol_v1_validation_20261012/review/verdicts_r8_r9.tsv`

  以上计数已按文件复算。r4、r5 只有 README 里的结论。
- r9 对比 r6/r7，取两版都接受的 75 条 add：

  | | 旧版 r6/r7 | 新版 r9 |
  |---|---:|---:|
  | 参照型锚点 | 29 | 2 |
  | S 超过新物体面积 4 倍 | 27 | 6 |
  | S 面积中位数（占画面） | 5.9% | 3.5% |

  代价：add 拒收率约 10%（r7 时约 2%）。以上为 README 记载。
- 审阅页在 `protocol_v1_validation_20261012/review_page/`：`r6.html`–`r9.html`、`index.html`。绿色是 S，橙色是执行区域，紫色是实际改动区域；r8、r9 页上另有旧版 S（红）和裁剪前的整块锚点（黄）。

### 5.5 代码与提交

| 提交（v2 仓库） | 时间（UTC） | 内容 | 测试函数数 |
|---|---|---|---:|
| `45ac3d7` | 10-12 13:33 | 适配初版：vendored 协议模块、`protocol_adapt.py`、两个脚本 | 13 |
| `c03218b` | 10-12 22:46 | 审核、修复、把关；`part_as_s`；几何把关 | 19 |
| `818ba2d` | 10-13 04:55 | `edited_region` | 22 |
| `17b2dee` | 10-13 08:47 | add 锚点改为直接承载物并裁到落点；adapter `protocol-adapt-2` | 30（README 记「45 项测试通过」，可能含参数化；未运行） |

涉及的文件：
- `src/samtok_edit21/preparation/protocol_adapt.py`（17b2dee 时 1,201 行）
- `src/samtok_edit21/preparation/instruction_protocol.py`
- `scripts/data_cleaning/protocol_adapt.py`
- `scripts/data_cleaning/protocol_remask.py`
- `tests/test_protocol_adapt.py`

**未提交改动的变化过程**：
- 约 07:40，v2 仓库有 4 个文件未提交：`git diff --stat` 为 4 files changed, 280 insertions(+), 50 deletions(-)；
  - `scripts/data_cleaning/protocol_adapt.py`：31
  - `scripts/data_cleaning/protocol_remask.py`：6
  - `src/samtok_edit21/preparation/protocol_adapt.py`：213
  - `tests/test_protocol_adapt.py`：80
- 内容就是 add 锚点的重新定义：
  - 允许把路面、沙地、水面、天空作为 S；
  - `sam_prompt` 对 add 改用 scope；
  - 新增 `localize_anchor`，以及 `ANCHOR_METHOD` 和 `ANCHOR_PARAMS`；
  - 用 `gap_frac` 做几何把关（`reference_anchor_not_adjacent`、`support_not_touching_item`、`anchor_not_at_item`）；
  - 审核规则按 add 和其他类型分开；
  - `build` 移到并行 worker 里。
- 08:47 这批改动作为 `17b2dee` 提交：同样 4 个文件，365 insertions(+), 59 deletions(-)。08:50 的 `git status` 为空。

**提交与推送状态**（2026-10-13 整理后）：
- 这四个提交都在 v3 分支里，并已随 v3 分支推到 GitHub（`origin/qwen-image-2.1-v3`）。
- 远端的 `qwen-image-2.1-v2` 分支仍停在 `795e818`，没有这四个提交；本地 v2 仓库有。是否也推到 v2 分支由用户决定，不影响使用。
- 之后数据侧的改动请直接在 v3 仓库里做。

### 5.6 现状与待定

- **状态**：全量（47,402 行）没有开跑。08:53 时本机没有在跑的数据任务：27B 的 vLLM 服务已停，只剩保活进程。
- **耗时估计**：4 机数小时，或单机独占 GPU 约 10 小时（会话笔记记载，未复核）。
- **待用户定的事**：
  1. 是否放行全量。
  2. 「无单一类型」的 601 对是否正式丢弃。代码已经不取它们，docs/09 7.6 节第 8 项也建议丢弃，但没有看到用户的正式决定。
  3. 多目标行的召回时机。验证各轮多目标占 7–19%：r3–r7 为 16–19%；r8、r9 以 add 为主，约 7–8%。
  4. 新锚点规则下 add 约 10% 的拒收率是否可以接受。
  5. benchmark 2.4.0 的规范模块已经变了（6.3 节），训练侧 vendored 的副本要不要同步。
  6. 协议记录转成什么训练清单：v2 的 Stage 1/2，还是 v3 的 Stage A/B。这需要和方法侧一起定。
  7. （已处理）`17b2dee` 已并入 v3 分支并推送。

---

## 6. 评测 benchmark

### 6.1 三个版本

| 版本 | 路径 | 规模 | 内容 | 谁在用 |
|---|---|---|---|---|
| v0「SAMTok Fine-Grained Interactive Edit Benchmark」 | `datasets/samtok_edit_benchmark/`（`benchmark.jsonl`） | 656 个 case，655 张源图 | • 来源：CompBench 532、HumanEdit 24、MIRAGE 100<br>• 类型：add 260、remove 268、replace 94、mixed 34<br>• 区域数：单区域 517、双区域 139<br>• 每个区域有 mask、box、point<br>• 构建自 HF 的三个数据集（固定 revision） | v2 评测协议 `protocol_v2_001` 由它生成 |
| v1 | `datasets/samtok_edit_benchmark_v1/`（`benchmark/cases.jsonl`） | 450 | • 类型：add 113、attribute 112、remove 112、replace 113<br>• 区域：466 个，单区域 434、双区域 16<br>• 来源：v0 过滤后 150 条 + goal1k 300 条（PACO-LVIS 153、BURST 58、ADE20K-Part 43、MeViS 26、SA-V 14、MOSEv2 6）<br>• 指令版本 `mask_grounded_single_ops_v4`<br>• 另有 433 条候选在 `candidates/expansion_433_20261008/`，未准入<br>• 没有独立人工审核 | SAMTokEdit v2 评测没有用；实验根目录 `experiments/SAMTokEdit/benchmark_v1/` |
| v2（SA-1B 200） | `datasets/samtok_edit_benchmark_v2/`（`benchmark/cases.jsonl`） | 200 张图，785 个目标 | 6.3 节 | 指令协议 1.0 的规范来源；还没有模型评测 |

### 6.2 `eval/protocol_v2_001`：v2 模型评测实际用的集合

目录：`experiments2/SAMTokEdit/qwen21_v2/eval/protocol_v2_001/`

- **组成**：715 个 case = v0 的单区域 case 517 个 + MIRAGE 的 99 个双区域 case 拆成 198 个单部件 case。
  - 517 = CompBench 492 + HumanEdit 24 + MIRAGE 1；
  - CompBench 的 40 个双区域 case 没有收；
  - `part` 字段：空 517、1 有 99、2 有 99。
- **划分**：按源图固定划分，dev 183（163 张源图）、test 532（452 张源图），两边没有共用的源图（已核）。
- **`cases.jsonl` 的字段**：`eval_index`、`benchmark_id`、`case_id`、`source_dataset`、`source_image`（指向 `datasets/samtok_edit_benchmark/images/source/...`）、`split`、`part`、`benchmark_type`、`instruction`、`region_instruction`、`referent_in_clause`、`region{mask, box, point}`。三种区域 715 个 case 都有。
- **`compiled.jsonl` 的字段**：`case_id`、`instruction`、`method`（llm 714、rule_based 1）、`ref_phrase`、`type_resolution`、`edit_type`、`region_kind`、`noref_template`、`valid_noref`、`invalid_reason`、`ref_template`。
  - 编译用训练数据同款的 9B noref 转换器，add 规则修复后重跑（提交 874c9a1）；
  - 715/715 有效。
- **类型映射**：add→add 263；remove→remove 266；replace→replace 33；replace→attribute 152；replace→text 1。
- **区域种类**：add 用框（263），其余用 mask（452）。

| split | add（box） | remove（mask） | attribute（mask） | replace（mask） | text（mask） | 合计 |
|---|---:|---:|---:|---:|---:|---:|
| dev | 61 | 81 | 32 | 8 | 1 | 183 |
| test | 202 | 185 | 120 | 25 | 0 | 532 |

- **按来源**：dev 中 CompBench 134、HumanEdit 9、MIRAGE 40；test 中 CompBench 358、HumanEdit 15、MIRAGE 159。
- **其他记录**：`cases.summary.json` 记了两个 MIRAGE 指代不一致的 case：`mirage_048#r2`、`mirage_054#r2`。
- **推理 setting**：mask、box、point、text 四种用法和融合开关在推理时决定，见 docs/07 8.7 节（文档记载）。
- **Stage A 的 dev 指认项**：dev 中的 mask case 正好 122 个（= 183 − 61 个 add），E-A 的 dev ref、noref 各 122 项。

### 6.3 benchmark v2 的事实

当前版本 release 2.4.0，`benchmark/release.json` 于 08:31 更新；manifest sha256 `c39a5554…`；状态 `tasks_frozen_pending_independent_review_and_model_evaluation`。

| 项 | 内容 |
|---|---|
| 规模 | 200 张 SA-1B 源图，785 个独立目标；每图目标数：2 个 26 张、3 个 48 张、4 个 41 张、5 个 85 张 |
| 操作 | add 196、attribute 196、remove 197、replace 196；2.4.0 中 add 全是直接锚点（support 196、reference 0） |
| 交互形式 | noref+point 113、ref+box 112、noref+mask 112、ref+none 112、ref+mask 112、noref+box 112、ref+point 112。合计 point 225、box 224、mask 224、无区域 112 |
| 目标面积 | 中位 0.57%；785 个中 535 个小于 1% |
| 审核 | 只有助手逐图查看，独立人工复核 0，模型运行 0 |
| 2.4.0 的变化 | • 原地修正了 134 个单元的指代和直接附着关系，源图、区域、交互形式不变<br>• 2.3.0 的元数据归档在 `archive/release_2.3.0_before_unified_anchor_20261013/`<br>• 新审核包 `evaluation/review_protocol1_release2.4.0/`，新模型输入 `evaluation/inputs_release2.4.0_native/` |
| 规范模块 | 2.4.0 的 `protocol_migration.json` 记 canonical 模块 sha256 为 `ed8c5d7b…`；2.3.0 时是 `e93d7d90…`，即训练仓库 vendored 的那份。**两边现在不一致** |
| 训练目录交集 | `statistics.json` 记 `training_catalog_overlap: 200`、`checkpoint_exposure_audited: false`：200 张图都在 `datasets/SAMTok_Training_Data/sam_info.json`（SAMTok 的 SA-1B 训练索引）里。benchmark 文档自己说明这只是索引成员关系，不能证明某个 checkpoint 训练过，但也不能称为已审计的独立留出集。用户 10-12 说明 benchmark 与编辑训练数据来源不同、不做泄漏检查，这里只转述 benchmark 自己的记录 |
| 仓库 | 文档中写的 `/opt/tiger/samtok_edit_benchmark_v2branch`（BENCHMARK_V2_ZH.md）、`/opt/tiger/tanyue/samtok_edit_benchmark_v2branch`（delivery.json）在本机都不存在。`/opt/tiger/tanyue/samtok_edit_benchmark` 是另一个 clone，分支 `dev`，HEAD 17e806e |

---

## 7. 中间产物清单

状态的含义：
- **在用**：现在的流程或复现要用；
- **仅留档**：证据或记录，不在流程里；
- **可删除候选**：文档或 README 明说已被取代或可忽略。删除需要用户在自己的消息里明确同意，并先证明没有引用。

| 路径 | 由哪一步产生 | 状态 | 理由 |
|---|---|---|---|
| `experiments2/SAMTokEdit/qwen21_v2/data/train_v2_box_001/`（340 MB） | v2 转换（c411c1c） | 在用 | v2 训练数据；Stage A 的单元池 |
| `/mnt/bn/strategy-mllm-train/intern/users/tanyue/experiments/SAMTokEdit/qwen21_v2/data/train_v2_box_001/` | 同上的原件 | 仅留档 | 与 user 副本三个 sha256 相同（已核） |
| `experiments2/SAMTokEdit/qwen21_v2/smoke/data_smoke_001/` | `scripts/diagnostics/make_smoke_data.py` | 仅留档 | smoke 测试用 |
| `experiments2/SAMTokEdit/qwen21_v2/data/clean_trial_20261009/`（45 MB） | 逐对审计、配准试验、6 批小批量验证 | 仅留档 | docs/09 第 1–3 节和 6.1–6.7 节的证据。全量清洗现场重算指标，不读 `audit.jsonl` |
| `experiments2/SAMTokEdit/qwen21_v2/data/reorg_20261009/` | 10-09 的路径整理和删除 | 仅留档 | 移动记录、`deleted_dirs.tsv`、`removed_symlinks.tsv` |
| `experiments2/SAMTokEdit/qwen21_v2/data/mllm_pilot_20261010/`（102 MB） | MLLM 核验试点 | 仅留档 | 人工标注 `q27/labels_final.json`、`flash_next_v2_20261011/round4.json` 可用于以后校准核验 |
| `experiments2/SAMTokEdit/qwen21_v2/data/region_protocol_survey_20261011/`（121 MB） | 区域协议调研（docs/09 第 7 节） | 仅留档 | 方案在冲突处已被协议 1.0 取代。标记清单 `derived/flagged_case_ids.json`、`refedit/replace_flagged.json` 可能仍有用 |
| `experiments2/SAMTokEdit/qwen21_v2/data/protocol_v1_pilot_20261012/`（53 MB） | 协议适配试点 | 仅留档 | run1 已被 run2 取代 |
| `experiments2/SAMTokEdit/qwen21_v2/data/protocol_v1_validation_20261012/`（679 MB，含 r8、r9） | 协议适配验证 r3–r9 | 在用 | 全量前的依据，审阅页也在这里 |
| `experiments2/SAMTokEdit/qwen21_v2/data/protocol_v1_validation_20261012/r9_aborted/`（约 1.1 MB） | 第 9 轮中途停下的半截结果 | **可删除候选** | README 写明「可以忽略」 |
| `experiments2/SAMTokEdit/qwen21_v3/data/stageA_ea2_20261012/` | E-A2 的数据构建 | 在用 | `qwen21_v3/README.md`：E-A2b 也用它 |
| `experiments2/SAMTokEdit/qwen21_v3/data/stageA_ea1_20261011/` | E-A1 的数据构建 | 仅留档 | `qwen21_v3/README.md`：留档；`eval.json` 是 1,930 项评测集的原始定义 |
| `experiments2/SAMTokEdit/qwen21_v3/legacy_code/ea1/`、`legacy_code/ea2/` | E-A 脚本 | 仅留档 | `qwen21_v3/README.md` 写明不要再运行。`ea2_build_data.py` 的常量 `EA1` 仍指向已迁走的 `review_v2_20261011/ea1/data` |
| `experiments2/SAMTokEdit/qwen21_v2/eval/protocol_v2_001/semantic/`、`semantic.log`、`semantic_sources.jsonl` | 评测 prompt 编译 | 在用 | `compiled.jsonl` 的来源 |
| `experiments2/SAMTokEdit/qwen21_v2/eval/protocol_v2_001/compiled_promptv1.jsonl`、`compiled_promptv1.summary.json`、`review_promptv1.md`、`semantic_promptv1/` | add 规则修复（874c9a1）之前的编译 | 仅留档 | 已被 `compiled.jsonl` 取代（docs/07 8.11 节）；体积很小 |
| `experiments2/SAMTokEdit/qwen21_v2/eval/protocol_v2_001/prompt_addfix/`、`prompt_addfix2/`、`prompt_addfix3/` | add 改写规则的三次试编译 | 仅留档 | 同上 |
| `experiments2/SAMTokEdit/qwen21_v2/eval/protocol_v2_001/smoke/` | 评测管线 smoke | 仅留档 | — |
| `datasets/SAMTokEdit_Cleaned_20261009/manifest.jsonl`、`decisions.jsonl`、`images/` | 全量像素清洗 | 在用 | 推荐集的图片都在 `images/`；`decisions.jsonl` 被 Stage A 使用 |
| `datasets/SAMTokEdit_Cleaned_20261009/scripts/`、`logs/run.log`、`logs/verify.log` | 清洗运行时 | 仅留档 | — |
| `datasets/SAMTokEdit_Cleaned_20261009/logs/decisions.before_frame_rule.jsonl`（400 MB，98,574 行） | 加「出画 10%」规则之前的判定 | 仅留档 | 已被 `decisions.jsonl` 取代（docs/09 6.8.2 节：229 对改判为丢弃），但 README 没有说明；要删须用户确认 |
| `datasets/SAMTokEdit_Cleaned_20261009/edit_check_qwen38_27b_v1/runs/edit_check_27b_4n_002/`（117 MB） | 全量核验的运行记录 | 仅留档 | — |
| `datasets/SAMTokEdit_Cleaned_20261009/edit_check_qwen38_27b_v1/runs/edit_check_27b_4n_001/`（192 KB） | 第一次提交，启动时就停了 | **可删除候选** | docs/09 6.14 节：「001 没有产生任何结论，目录里只有日志」。很小，留作失败记录也无妨 |
| `datasets/SAMTok_Derived_Edit_Labeling/combined/images/sources/remove/`<br>`datasets/SAMTok_Derived_Edit_Labeling/combined/images/sources/multitype/` | Derived 的两次四机运行 | 在用 | 5,700 张图存了两份，但 v2 清单两套路径都引用，不能动 |
| `experiments/SAMTokEdit/qwen21_full4_20260928/data/semantic_runs/qwen21_noref9b_rules_4n_full_003/` | v1 noref 转换（生产版） | 在用（上游） | v1 和 v2 语料的指令来源 |
| `experiments/SAMTokEdit/qwen21_full4_20260928/data/semantic_runs/qwen21_noref4n_full_001`、`qwen21_noref4n_full_002`、`qwen21_noref9b_rules_4n_full_001`、`qwen21_noref9b_rules_4n_full_002` | v1 早期 noref 转换 | 仅留档 | v1 文档记载：4B `_001` 竞态失败，`_002` 有示例污染、未用于生产，9B `_001` CUDA 超时；9B `_002` 的状态未查 |
| `datasets/samtok_edit_benchmark_v1/candidates/expansion_433_20261008/` | v1 扩充候选 | 仅留档 | 未正式准入 |
| `datasets/samtok_edit_benchmark_v2/archive/`，含今天新建的 `release_2.3.0_before_unified_anchor_20261013/` | benchmark 历史版本 | 仅留档 | — |
| `datasets/sa1b_edit_work/` 的 25 个子目录 | 用户 SA-1B 新管线的迭代 | 他人在用，不评估 | 没有顶层 README（1.7 节） |
| `/home/tiger/.claude/jobs/1799fb27/tmp/` 下的 `r3`–`r9` 等 | 协议适配会话的本地工作目录 | 会话临时文件 | 08:42 已归档到 `protocol_v1_validation_20261012/` |

---

## 8. 当前进度与待办

### 8.1 已完成

| 日期 | 事项 | 结果 |
|---|---|---|
| 10-06 | v2 训练清单 `train_v2_box_001` | 110,079 + 290,006 行 |
| 10-08 | 98,574 对逐对审计 | 主四类只有 48% 可直接用 |
| 10-09 | 四个子集归到 `datasets/` 下 | 删除约 213 GB 旧流程目录（文档记载） |
| 10-09 | 全量像素清洗 | 70,956 对 |
| 10-11 | 方案 2 全量核验 | 48,437 对 |
| 10-12 | 去掉 text 和 action | 46,482 对 |
| 10-12 至 10-13 | 协议 1.0 适配 | 代码 4 个提交，最新 17b2dee；试点和 r3–r9 验证，最近两轮 r8、r9 各 206 行；`edited_region` 字段 |
| 10-11 至 10-12 | Stage A 的 E-A1、E-A2 | 已用 v2 数据跑完；E-A2 四条放行线通过（文档记载，未复核） |
| 10-13 08:21 | E-A 数据迁到 `qwen21_v3/` | 迁移记录 `qwen21_v3/MOVED_20261013.tsv` |

### 8.2 当前状态（2026-10-13）

- 数据任务：没有在跑的。协议 1.0 的全量改写等用户放行。
- 仓库与文档：2026-10-13 建了 v3 分支，v2 阶段的 docs/01–09 归档到 `docs/archive/v2/`，本文是新的数据文档。E-A1、E-A2 的数据和运行迁到了 `experiments2/SAMTokEdit/qwen21_v3/`（那里的 `README.md` 写明了各项状态）。
- benchmark v2 在 2026-10-13 升到 release 2.4.0（6.3 节）。
- `datasets/sa1b_edit_work/` 是用户另一条线的 SA-1B 数据管线，仍在迭代（1.7 节）。

### 8.3 等用户决定的

1. **协议 1.0 全量**：是否放行（47,402 行）；输出放在哪里，建议在 `datasets/SAMTokEdit_Cleaned_20261009/` 下新建子目录。
2. **601 对「无单一类型」**：是否正式丢弃。
3. **多目标行**：何时召回。
4. **add 约 10% 的拒收率**：是否可以接受。
5. **协议模块同步**：benchmark 2.4.0 的规范模块（`ed8c5d7b…`）是否同步到训练仓库（现在是 `e93d7d90…`）。
6. **训练清单的格式和构建器**：与方法侧一起定（v2 Stage 1/2 还是 v3 Stage A/B）。
7. **数据划分**：统一改为按图像内容分组（3.2 节）。目前只有 Stage A 在训练端做了均值哈希防护。
8. **提交同步**：（已处理）`17b2dee` 已并入 v3 分支并推送。远端 v2 分支是否也更新，由用户决定。
9. **文档更新**：协议 1.0 适配此前只记在实验目录的 README 和提交说明里，本文第 5 节是它第一次进入仓库文档。docs/09 0.6 节第 4 项（Derived 的 add 重新筛选）和 6.15 节（不要 text）与用户 10-12 的决定冲突，需要改写或注明已被取代。

---

## 附录 A：路径核对表

核对时间为 2026-10-13 07:40–08:55 UTC。方式：`ls`、`wc -l`、`sha256sum`、`stat`、标准库 Python（JSON 计数、parquet 读文件尾元数据）。「是」表示核对时存在。

| 路径 | 存在 | 类型 | 行数 / 文件数 / 说明 | 核对方式 |
|---|---|---|---|---|
| `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTokEdit_Cleaned_20261009/edit_check_qwen38_27b_v1/manifest_kept_no_text_action.jsonl` | 是 | 文件 | 46,482 行；sha256 `b6de5863…ca85` | wc、sha256sum、Python 计数 |
| `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTokEdit_Cleaned_20261009/edit_check_qwen38_27b_v1/manifest_kept_no_text_action.summary.json` | 是 | 文件 | 记 46,482 / 1,955 | 读取 |
| `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTokEdit_Cleaned_20261009/edit_check_qwen38_27b_v1/manifest_kept.jsonl` | 是 | 文件 | 48,437 行 | wc、Python |
| `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTokEdit_Cleaned_20261009/edit_check_qwen38_27b_v1/verdicts.jsonl` | 是 | 文件 | 70,956 行 | wc、Python |
| `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTokEdit_Cleaned_20261009/edit_check_qwen38_27b_v1/{README.md,summary.json,config.json}` | 是 | 文件 | — | ls、读取 |
| `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTokEdit_Cleaned_20261009/edit_check_qwen38_27b_v1/runs/edit_check_27b_4n_002/` | 是 | 目录 | 6 项；117 MB；`SUCCESS.json` 记保留 48,437 | ls、du |
| `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTokEdit_Cleaned_20261009/edit_check_qwen38_27b_v1/runs/edit_check_27b_4n_001/` | 是 | 目录 | 3 项；192 KB | ls、du |
| `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTokEdit_Cleaned_20261009/edit_check_qwen38_27b_v1/` | 是 | 目录 | 740 MB | du |
| `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTokEdit_Cleaned_20261009/manifest.jsonl` | 是 | 文件 | 70,956 行 | wc |
| `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTokEdit_Cleaned_20261009/decisions.jsonl` | 是 | 文件 | 98,574 行 | wc、Python |
| `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTokEdit_Cleaned_20261009/images/` | 是 | 目录 | 141,912 个文件 | find \| wc |
| `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTokEdit_Cleaned_20261009/{README.md,summary.json,verification.json}` | 是 | 文件 | — | 读取 |
| `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTokEdit_Cleaned_20261009/scripts/` | 是 | 目录 | 4 项 | ls |
| `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTokEdit_Cleaned_20261009/logs/decisions.before_frame_rule.jsonl` | 是 | 文件 | 98,574 行；400 MB | wc、ls |
| `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v2/data/train_v2_box_001/stage1.jsonl` | 是 | 文件 | 110,079 行；sha256 `e2312e33…` | wc、sha256sum |
| `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v2/data/train_v2_box_001/stage2.jsonl` | 是 | 文件 | 290,006 行；sha256 `cb46a3f0…` | wc、sha256sum |
| `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v2/data/train_v2_box_001/provenance.jsonl` | 是 | 文件 | 400,085 行；sha256 `2fc4dd0f…` | wc、sha256sum |
| `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v2/data/train_v2_box_001/{conversion_report.json,metadata_report.json}` | 是 | 文件 | — | 读取 |
| `/mnt/bn/strategy-mllm-train/intern/users/tanyue/experiments/SAMTokEdit/qwen21_v2/data/train_v2_box_001/` | 是 | 目录 | 3 个 sha256 与上相同 | sha256sum |
| `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v2/smoke/data_smoke_001/` | 是 | 目录 | stage1 80 行、stage2 212 行 | wc |
| `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v2/runs/`（5 个运行） | 是 | 目录 | 每个 `manifest.json` 都引用 user 路径的 `train_v2_box_001` | grep |
| `/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/qwen21_full4_20260928/data/sources.jsonl` | 是 | 文件 | 98,574 行 | wc |
| `/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/qwen21_full4_20260928/data/assets/` | 是 | 目录 | 3 个子目录 | ls |
| `/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/qwen21_full4_20260928/data/train_full_9b_rules_003/` | 是 | 目录 | 17 项 | ls |
| `/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/qwen21_full4_20260928/data/semantic_runs/` | 是 | 目录 | 5 个运行，含 `qwen21_noref9b_rules_4n_full_003` | ls |
| `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Training_Data/mix_gres8k_ver4k_train.parquet` | 是 | 文件 | 12,337 行 | parquet 文件尾 |
| `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Training_Data/gres8k_train.parquet` | 是 | 文件 | 8,337 行 | parquet 文件尾 |
| `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Training_Data/sam_info.json` | 是 | 文件 | 1.32 GB | stat |
| `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/RefEdit_Labeling/final/RefEdit-mask-prefiltered-qwen38-self-contained/data/` | 是 | 目录 | 105 个 parquet，7,804 行 | parquet 文件尾 |
| `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/RefEdit_Labeling/final/RefEdit-mask-prefiltered-qwen38-self-contained/run_summary.json` | 是 | 文件 | 任务分布、mask 来源 | 读取 |
| `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/RefEdit_Labeling/source/RefEdit/data/` | 是 | 目录 | 105 个 parquet，18,249 行 | parquet 文件尾 |
| `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/RefEdit_Labeling/stages/{RefEdit-quality-prefilter-qwen38,RefEdit-mask,RefEdit-mask-prefiltered-qwen38}/` | 是 | 目录 | — | ls |
| `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit_Labeling/final_dataset_39k/shards/` | 是 | 目录 | 5 个子目录，1,844 个 parquet，38,971 行 | parquet 文件尾 |
| `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit_Labeling/final_dataset_39k/dataset_manifest.json` | 是 | 文件 | 源类型和 mask_qc 计数 | 读取 |
| `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit_Labeling/source/CrispEdit-2M/` | 是 | 目录 | 2,081 项，其中 2,078 个 parquet、529,020 行 | ls、parquet 文件尾 |
| `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit_Labeling/{prefilter/quality,prefilter/scene,runs/labeling_4node_crispedit_full_localenv_20260925}/` | 是 | 目录 | — | ls |
| `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/ScaleEdit_Labeling/final/scaleedit_25k/shards/` | 是 | 目录 | 1,073 个 parquet，25,664 行 | parquet 文件尾 |
| `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/ScaleEdit_Labeling/final/scaleedit_25k/dataset_manifest.json` | 是 | 文件 | 任务和 qc 计数 | 读取 |
| `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/ScaleEdit_Labeling/source/ScaleEdit-filtered-source/` | 是 | 目录 | 1,155 个 parquet，299,633 行 | parquet 文件尾 |
| `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/ScaleEdit_Labeling/runs/` | 是 | 目录 | — | ls |
| `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/combined/manifest.jsonl` | 是 | 文件 | 27,957 行 | wc、Python |
| `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/combined/summary.json` | 是 | 文件 | — | 读取 |
| `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/combined/images/sources/remove/` | 是 | 目录 | 6,238 个文件 | ls \| wc |
| `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/combined/images/sources/multitype/` | 是 | 目录 | 6,936 个文件；与 `remove/` 同名 5,700 个 | ls、comm |
| `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/runs/{four_node,launchers}/` | 是 | 目录 | — | ls |
| `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v3/MOVED_20261013.tsv` | 是 | 文件 | 11 条迁移 | 读取 |
| `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v3/README.md` | 是 | 文件 | 09:09 新建；目录说明和各项状态 | 读取 |
| `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v3/data/stageA_ea1_20261011/train.jsonl` | 是 | 文件 | 42,000 行 | wc |
| `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v3/data/stageA_ea1_20261011/{data_stats.json,eval.json,eval_quick.json}` | 是 | 文件 | `eval.json` 1,930 项 | Python |
| `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v3/data/stageA_ea2_20261012/epoch_rows.jsonl` | 是 | 文件 | 82,000 行（08:21 重写过） | wc、Python |
| `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v3/data/stageA_ea2_20261012/{data_stats.json,eval.json,eval_quick_idx.json,schedule.json}` | 是 | 文件 | `eval.json` 1,930 项；schedule 164,000 条 | Python |
| `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v3/data/stageA_ea2_20261012/gres_images/` | 是 | 目录 | 6,962 个文件 | ls \| wc |
| `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v3/legacy_code/ea1/ea1_build_data.py`、`legacy_code/ea2/ea2_build_data.py`、`legacy_code/ea2/README_as_run.md` | 是 | 文件 | — | find |
| `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v3/runs/{ea1_local8_001,ea2_local8_001}/` | 是 | 目录 | — | find |
| `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v2/eval/protocol_v2_001/review_v2_20261011/ea1/`、`ea2/` | **否** | — | 08:21 迁到 `qwen21_v3/` | ls |
| `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v2/eval/protocol_v2_001/review_v2_20261011/SAMTokEdit_v2_复核与下一步方法建议.md` | 是 | 文件 | 内部仍引用 `ea1/…`、`ea2/…` 旧路径 | ls、grep |
| `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v2/eval/protocol_v2_001/cases.jsonl` | 是 | 文件 | 715 行 | wc、Python |
| `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v2/eval/protocol_v2_001/compiled.jsonl` | 是 | 文件 | 715 行 | wc、Python |
| `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v2/eval/protocol_v2_001/{cases.summary.json,compiled.summary.json}` | 是 | 文件 | — | 读取 |
| `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v2/eval/protocol_v2_001/compiled_promptv1.jsonl`、`semantic_sources.jsonl` | 是 | 文件 | 各 715 行 | wc |
| `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v2/eval/protocol_v2_001/{semantic,semantic_promptv1,prompt_addfix,prompt_addfix2,prompt_addfix3,smoke}/` | 是 | 目录 | — | ls |
| `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/samtok_edit_benchmark/benchmark.jsonl` | 是 | 文件 | 656 行 | wc |
| `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/samtok_edit_benchmark/{benchmark_meta.json,validation_report.json}` | 是 | 文件 | — | 读取 |
| `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/samtok_edit_benchmark_v1/benchmark/cases.jsonl` | 是 | 文件 | 450 行 | wc、Python |
| `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/samtok_edit_benchmark_v1/dataset_summary.json` | 是 | 文件 | — | 读取 |
| `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/samtok_edit_benchmark_v1/candidates/expansion_433_20261008/` | 是 | 目录 | 15 项 | ls |
| `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/samtok_edit_benchmark_v2/benchmark/cases.jsonl` | 是 | 文件 | 200 行（08:31 重写为 2.4.0） | wc、Python |
| `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/samtok_edit_benchmark_v2/benchmark/instruction_design.jsonl` | 是 | 文件 | 785 行 | wc |
| `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/samtok_edit_benchmark_v2/benchmark/{release.json,statistics.json,protocol_migration.json,validation.json}` | 是 | 文件 | release 2.4.0 | 读取 |
| `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/samtok_edit_benchmark_v2/BENCHMARK_V2_ZH.md` | 是 | 文件 | 2.4.0 版（08:36） | 读取 |
| `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/samtok_edit_benchmark_v2/README_中文.md` | 是 | 文件 | 仍写 2.3.0 | 读取 |
| `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/samtok_edit_benchmark_v2/docs/{INSTRUCTION_PROTOCOL_ZH.md,INSTRUCTION_ALIGNMENT_20261013_ZH.md}` | 是 | 文件 | — | ls |
| `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/samtok_edit_benchmark_v2/archive/release_2.3.0_before_unified_anchor_20261013/` | 是 | 目录 | 含 2.3.0 的 `protocol_migration.json`（`e93d7d90…`） | ls、grep |
| `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/samtok_edit_benchmark_v2/sa1b_source_selection/training_index_audit.json` | 是 | 文件 | 匹配 2,799/2,800 | 读取 |
| `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/samtok_edit_benchmark_v2/assets/` | 是 | 目录 | 200 项 | ls |
| `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/{CompBench,HumanEdit,MIRAGE,PACO_benchmark_candidates}/` | 是 | 目录 | — | ls |
| `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v2/data/clean_trial_20261009/audit/audit.jsonl` | 是 | 文件 | 98,574 行 | wc |
| `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v2/data/clean_trial_20261009/audit/register.jsonl` | 是 | 文件 | 750 行 | wc |
| `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v2/data/clean_trial_20261009/results/{batch_v1,batch_v2,batch_v3,batch_s2,batch2_s1,batch2_s2}/results.jsonl` | 是 | 文件 | 各 640 行 | wc |
| `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v2/data/reorg_20261009/{deleted_dirs.tsv,removed_symlinks.tsv}` | 是 | 文件 | — | ls |
| `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v2/data/mllm_pilot_20261010/inputs.jsonl` | 是 | 文件 | 303 行 | wc |
| `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v2/data/mllm_pilot_20261010/repo_check_20261011/manifest.jsonl` | 是 | 文件 | 699 行 | wc |
| `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v2/data/mllm_pilot_20261010/q27/labels_final.json`、`flash_next_v2_20261011/round4.json` | 是 | 文件 | — | ls |
| `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v2/data/region_protocol_survey_20261011/{derived/flagged_case_ids.json,derived/group_part_ids.json,refedit/replace_flagged.json}` | 是 | 文件 | — | ls |
| `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v2/data/protocol_v1_pilot_20261012/pilot/rows.jsonl` | 是 | 文件 | 40 行 | wc |
| `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v2/data/protocol_v1_pilot_20261012/run{1,2}/build/records.jsonl` | 是 | 文件 | 各 40 行 | wc、读 summary |
| `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v2/data/protocol_v1_pilot_20261012/review/verdicts_run{1,2}.tsv` | 是 | 文件 | — | Python |
| `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v2/data/protocol_v1_validation_20261012/README.md` | 是 | 文件 | 08:48 更新，含 r8、r9 一节 | 读取 |
| `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v2/data/protocol_v1_validation_20261012/rows/{r3_r4_r5_rows,r6_rows,r7_rows,r8_r9_rows}.jsonl` | 是 | 文件 | 230 / 160 / 119 / 206 行 | wc |
| `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v2/data/protocol_v1_validation_20261012/r{4,5,6,7,8,9}/final.jsonl` | 是 | 文件 | 230 / 230 / 160 / 119 / 206 / 206 行 | wc、读 summary |
| `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v2/data/protocol_v1_validation_20261012/r{6,7}/final_er2.jsonl` | 是 | 文件 | 160 / 119 行 | wc、读 summary |
| `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v2/data/protocol_v1_validation_20261012/r9_aborted/` | 是 | 目录 | 5 个文件 | ls |
| `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v2/data/protocol_v1_validation_20261012/review/verdicts_r6_r7.tsv`、`verdicts_r8_r9.tsv` | 是 | 文件 | — | Python |
| `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v2/data/protocol_v1_validation_20261012/review_page/{index,r6,r7,r8,r9}.html` | 是 | 文件 | — | ls |
| `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v2/data/protocol_v1_validation_20261012/scripts/{run_round_full.sh,server_dp8.sh}` | 是 | 文件 | — | ls |
| `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/sa1b_edit_work/` | 是 | 目录 | 25 个子目录 | ls |
| `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/sa1b_edit_work/disjoint_pool/candidates.jsonl` | 是 | 文件 | 1,044 行 | wc |
| `/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen3.8-27B` | 是 | 目录 | — | ls |
| `/opt/tiger/tanyue/samtok_edit_qwen-image-2.1-v2/docs/{04_SAMTokEdit_Qwen21_训练数据盘点.md,09_SAMTokEdit_Qwen21_训练数据诊断与改进.md}` | 是 | 文件 | 09 共 1,611 行 | wc |
| `/opt/tiger/tanyue/samtok_edit_qwen-image-2.1-v2/docs/assets/v2_data_audit_20261008/` | 是 | 目录 | 25 项 | ls |
| `/opt/tiger/tanyue/samtok_edit_qwen-image-2.1-v2/src/samtok_edit21/preparation/{v2_data,instruction_protocol,protocol_adapt,edit_check,sources,semantic,corpus,fallback,converters}.py` | 是 | 文件 | `instruction_protocol.py` 的 sha256 `e93d7d90…` | ls、sha256sum |
| `/opt/tiger/tanyue/samtok_edit_qwen-image-2.1-v2/src/samtok_edit21/data/{protocol,io,provenance}.py` | 是 | 文件 | — | ls |
| `/opt/tiger/tanyue/samtok_edit_qwen-image-2.1-v2/scripts/data_cleaning/{clean_pairs,checks,verify_cleaned,reprocess,select_types,protocol_adapt,protocol_remask}.py` | 是 | 文件 | — | ls |
| `/opt/tiger/tanyue/samtok_edit_qwen-image-2.1-v2/scripts/edit_check/{run_arnold.sh,setup_env.sh}`、`scripts/annotation/` | 是 | 文件 | — | ls |
| `/opt/tiger/tanyue/samtok_edit_qwen-image-2.1-v2/tests/test_protocol_adapt.py` | 是 | 文件 | 30 个 test 函数（17b2dee） | grep |
| `/opt/tiger/tanyue/samtok_edit_qwen-image-2.1-v3/src/samtok_edit21/preparation/grounding_data.py` | 是 | 文件 | 358 行 | wc |
| `/opt/tiger/tanyue/sam3-{refedit,crispedit,scaleedit,samtok-derived-edit}-labeling` | 是 | 目录 | HEAD 分别为 72e0d25 / 407e913 / c253733 / 2274c50 | git log |
| `/opt/tiger/samtok_edit_benchmark_v2branch`、`/opt/tiger/tanyue/samtok_edit_benchmark_v2branch` | **否** | — | benchmark v2 的文档和 delivery.json 引用了它们 | ls |
| `sa1b-samtok-edit-data` 仓库 | **否**（本机无） | — | — | ls |
| `/home/tiger/.claude/projects/-opt-tiger-tanyue/memory/{samtokedit-training-data-state-20261012,samtok-benchmark-v2-reference,dataset-location-convention}.md` | 是 | 文件 | 第一份 08:53 更新过 | 读取 |

---

## 附录 B：未能核实的事项、来源之间的不一致

### B.1 没有复核的（只来自文档或笔记）

- **上游打标的中间数字**：RefEdit 预筛 PASS 7,836；CrispEdit 预筛 460,518；ScaleEdit 质量预筛的数字。
- **Derived 源数据**有 7,671 张图、10,633 个区域；冻结集人工复核 65/96。
- **清洗目录体积**：`images/` 表观 209 GB、实际新增 102 GB（du 太慢，没有做）。
- **MLLM 核验效果**：在人工标注子集上抓到 86%（72/84）、误判 8.1%（19/234）；全量耗时 66 分钟。
- **方法侧数字**：
  - E-A1 泄漏数字（363/1,160、220 项、255/343 对）；
  - E-A2 放行线结果；
  - docs/07、docs/08 和文档 02 中的模型侧数字：codec IoU 0.38/0.88、Acc@0.5 0.23、漂移率、锐度比、v1 语义转换 97,361/1,213。
- **协议适配**：
  - 全量耗时估计；
  - 17b2dee 的「45 项测试通过」：本文只数了 30 个 test 函数，没有运行测试；
  - r4、r5 的人工抽检结论：只有 README，没找到逐行判定文件。
- **推送状态**：依据 v3 仓库的远程跟踪引用，它的最近一次 fetch 时间不知道。

### B.2 来源之间不一致、或文档已过时（附核实结果）

1. **CrispEdit 上游分片数**。docs/09 8.3 节写「add/color/remove/replace/motion change 各 445 片」。磁盘上 motion change 只有 128 片，另有 background change 84 片、style 86 片；parquet 共 2,078 个。「2,081」是目录项数，其中含 3 个非 parquet 项。
2. **贴回样本的占比**（已改正）。原复核文档写「贴回样本约占清洗后主四类的 35%（23,002/66,684）」。23,002 只是 CrispEdit 和 ScaleEdit 的贴回样本；四个来源合计是 26,682（40.0%）。文档 02 §2.3 已按此改写。
3. **核验结果目录的体积**。docs/09 6.14 节写「共 484 MB」，现在 du 为 740 MB，因为之后加了 269 MB 的 `manifest_kept_no_text_action.jsonl`。
4. **问题类型表少一类**。docs/09 6.14 节的问题类型表没有 dn2 的 `identity`（4 次）。
5. **docs/09 与用户 10-12 的决定冲突**：
   - 6.15 节和 0.1 节说训练数据不再含 text；协议 1.0 适配按用户决定又把 text 加回来了（`load_rows` 也是这样实现的）；
   - 0.6 节第 4 项「Derived 的 add 要重新筛选」与「不扩充数据」冲突；
   - 第 7 节的锚点打标方案已被协议 1.0 适配取代；
   - 协议 1.0 适配此前没有进入仓库文档，只在实验目录的 README 和提交说明里；本文第 5 节补上了。docs/09 已归档，没有再改。
6. **text 的条数**。会话笔记写「re-admitting about 1,480 text rows」，实际 text 有 1,521 条；按规则草分为 attribute 1,479、add 24、remove 18。1,480 大约是其中 attribute 一路的数量。
7. **推送状态的说法已过时**。试点、验证 README 和会话笔记说 45ac3d7、c03218b、818ba2d「只在本地、未推送」。2026-10-13 起这三个提交和 `17b2dee` 都已随 v3 分支推到远端（5.5 节）。
8. **benchmark v2 版本号和规范模块**：
   - `README_中文.md` 和 `delivery.json` 仍写 2.3.0，`release.json` 和 `BENCHMARK_V2_ZH.md` 已是 2.4.0；
   - 2.4.0 的规范模块 sha256 是 `ed8c5d7b…`，训练侧 vendored 的副本和测试钉住的哈希都是 `e93d7d90…`。
9. **找不到的路径**：
   - benchmark v2 文档引用的仓库路径（两个 `samtok_edit_benchmark_v2branch`）本机都不存在；
   - `sa1b-samtok-edit-data` 本机没有；
   - `/opt/tiger/tanyue/samtok_edit_benchmark` 是 dev 分支，不是 v2branch。
10. **v2 仓库 `docs/README.md` 的产物路径过时**。它写实验产物在 `/mnt/bn/strategy-mllm-train/intern/users/tanyue/experiments/SAMTokEdit/qwen21_v2/`，10-06 起实际已改到 `experiments2/`。
11. **E-A 迁移后的旧路径**（已处理）。原复核文档的产物索引用的是 `ea1/…`、`ea2/…` 的旧相对路径；它已拆成文档 02 和文档 03，两份都改用新路径。`legacy_code/ea2/ea2_build_data.py` 里的常量 `EA1` 仍指向旧路径，那是留档的脚本，不再运行。
12. **`audit.jsonl` 是不是清洗的输入**。`clean_trial_20261009/README.md` 说是；docs/09 6.8.2 节说全量脚本改为现场计算，不再读审计文件。前者指的是小批量试验。
13. **类型口径不同**：
    - v2 转换剔除的 composite 是 1,206 个（CrispEdit 60、ScaleEdit 1,146），按 v1 语义类型；
    - 清洗的「无单一类型」是 1,267 个（全在 ScaleEdit），按 `sources.jsonl` 的 `provisional_type`；
    - docs/09 0.3 节自己也注明：主四类 92,359 对与第 1 节的 92,303 对口径不同。
14. **文件在核对过程中变化**：
    - `protocol_v1_validation_20261012/README.md` 从 9,071 字节变成 16,926 字节；
    - `BENCHMARK_V2_ZH.md` 从 50,191 字节变成 34,520 字节；
    - E-A 的 `epoch_rows.jsonl` 被重写，行数不变。

    本文的数字取最后一次读到的版本。
