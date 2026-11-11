# SITEdit 最终方法的数据盘点

更新：2026-11-11。本文盘点**最终交付版本（R2@1000）用到的全部数据**，包括三阶段训练链 S1 → S2（v4）→ R2，以及正式评测数据。方法见 [METHOD.md](METHOD.md)，实验见 [EXPERIMENTS.md](EXPERIMENTS.md)。

**本文的数字来源：**
- 行文件的构成：2026-11-11 直接逐行统计行文件。
- 合板数据、区域池：抄自 v4 数据文档，即 `docs/archive/v4/01_SAMTokEdit_Qwen21_v4数据.md`，在那里有完整字段说明和构建记录。
- 自蒸馏池：直接统计 `audit.jsonl`、`pool_*.jsonl`。

路径记号与 METHOD.md 相同：`NAS/` = `/mnt/bn/strategy-mllm-train/user/tanyue/`，`SE/` = `NAS/experiments2/SAMTokEdit/`，`V3/` = `SE/qwen21_v3/`，`QA/` = `SE/quality_anchor_20261104/`。

---

## 0. 数据流总览

```text
九个来源 D1、D3、D4、D10–D15 ──merge-data（10-29）──▶ 合板 SAMTokEdit_Merged_20261029：410,563 条记录 / 489,416 个单元
SA-1B dense mask ──round2-data pool / pool-ext（10-16、10-30）──▶ 区域池：1,132 + 20,000 张图
                                  │
          ┌───────────────────────┴───────────────────────┐
          ▼                                               ▼
rows_s1.jsonl（192,000 行）── S1                  rows_s2.jsonl（512,000 行）── S2（= v4）
                                                          │ build_rows_v5.py（11-05）：换入高质量合成目标、加 10% 普通编辑
                                                          ▼
                                             rows_v5_s2.jsonl（512,000 行）
                                                          │ build_rows_v6s2.py（11-08）：换掉 NHR-Edit 普通行、单目标中合成:贴回 = 70:30、孪生全部用贴回
                                                          ▼
                                             rows_v6_s2.jsonl（512,000 行）
                                                          │ build_rows_v6s2hqfull.py（11-08）：非孪生的编辑行全部换成合成目标
                                                          ▼
                                             rows_v6_s2hqfull.jsonl（512,000 行）── R2（交付检查点只用前 32,000 行）

原版 Qwen-Image-2.1 整图自蒸馏（v2、v3 两批，各 90,000 条）── 门限 ── 27B 审核 ──▶ v2_final 26,643 对 + v3_final 27,045 对 = 53,688 对
（R2 的合成目标和普通编辑行都来自这里）
```

## 1. 基础数据：合板 `NAS/datasets/SAMTokEdit_Merged_20261029/`

- **规模：**410,563 条记录、489,416 个单元。默认训练视图 `views/default_train.txt`，验证 `views/dev.txt`。
- **格式：**`records.jsonl` 每行一条记录，字段见 v4 数据文档 §2.2。README 在该目录。
- **协议：**全部记录按指令协议 2.0、mask 语义 2.0 记录。每个单元有选区 S、执行区域、实际改动区域、SAMTok token 和带名/盲指代两种句子。
- **构建：**`samtok-edit merge-data`，代码 `src/samtok_edit21/preparation/merge_data.py`。D12–D15 先由 `samtok-edit sa1b-groups` 转换，代码 `src/samtok_edit21/preparation/sa1b_groups.py`。

| 代号 | 内容 | 记录 / 单元 | 质检 | 原始数据 |
|---|---|---:|---|---|
| D1 | SA-1B 50k：原版 Qwen-Image-2.1 裁图生成，再按支持区域羽化贴回 | 50,000 / 50,000 | 确定性检查 + Qwen3.8-27B 审核 | `NAS/datasets/sa1b_samtok_edit_50k_20261014/release/` |
| D3 | v1 清洗后的单目标编辑（RefEdit、CrispEdit、ScaleEdit、SAMTok Derived） | 39,615 / 39,615 | 抽查约 86% 对 | `NAS/datasets/SAMTokEdit_Cleaned_20261009/protocol_adapt_qwen38_27b_v1/protocol_v2_blind/` |
| D4 | v1 清洗后的多目标对 | 4,066 / 9,964 | 抽查约 88% 对 | `NAS/datasets/SAMTokEdit_Cleaned_20261009/protocol_multi_qwen38_27b_v1/protocol_v2_blind/` |
| D10 | extedit 第一批（CrispEdit-2M、ScaleEdit） | 38,346 / 40,791 | 抽 100 条，93 条可用 | `NAS/datasets/ExtEdit_Protocol2_20261021/` |
| D11 | extedit 第二批 | 151,337 / 163,595 | 抽 80 条，79 条可用 | `NAS/datasets/ExtEdit_Protocol2_20261022/` |
| D12 | SA-1B 孪生：同图、同操作的两个目标各自编辑 | 25,022 / 25,022 | 27B 规划 + 证据门 + 两级审核 | `SE/data_pipeline/02_production_runs/sa1b_samtok_group_twins50k_multi100k_20261025/final/twins_balanced.jsonl` |
| D13 | SA-1B 多目标：一张图 2–4 个目标 | 39,814 / 98,066 | 同上 | 同目录 `multi_balanced.jsonl` |
| D14 | SA-1B 附带单目标 | 12,363 / 12,363 | 同上 | 同目录 `singles.jsonl` |
| D15 | SA-1B 单目标 v2.1 | 50,000 / 50,000 | 同上；助手盲评合格 75% | `SE/data_pipeline/02_production_runs/sa1b_samtok_single50k_20261026/final/singles_balanced.jsonl` |

- **许可：**混合许可，整体按最严的条款，只用于非商业研究。SA-1B 为研究用途；CrispEdit-2M 为 CC BY 4.0；ScaleEdit 为 CC BY-NC-SA 4.0（用户 2026-11-07 确认可用）。
- **与评测数据的隔离：**
  - 合板时按 SA-1B 图 id、SHA-1 和 dHash，与 IF-Bench 2.6.0 的 1,084 张源图查重复，重合的记录移出默认视图（D12–D15 共 22 条，D3/D4 共 5 条）。
  - D10、D11 在取数时就排除了 benchmark 近重复图。
  - 开发集的 60 张 SA-1B 图不进训练。
  - 详见 v4 数据文档 §4.4。

## 2. 区域池与 round-2 数据目录

| 目录 | 内容 |
|---|---|
| `V3/data/round2_20261016/` | 开发划分 `dev_split.json`、D2 区域池 `pool/`（1,132 张图、72,857 个区域）、token 缓存（`edit_tokens.jsonl`、`v1_tokens.jsonl`、`d4_tokens.jsonl`）、`support_cache.jsonl`、开发集 `eval_c1.json`（1,320 项）与快速子集 `eval_c2_quick_idx.json` |
| `V3/data/round2_scaleup_20261030/` | S1、S2 的数据目录。`pool/` = 旧 1,132 张 + 新 20,000 张 SA-1B 图（全部 D12 图 12,491 张，加 D15 图 7,509 张），新增可用区域 1,244,227 个（往返 IoU ≥ 0.5）。其余文件硬链接自上一行的目录。含 `rows_s1.jsonl`、`rows_s2.jsonl` |
| `QA/data_round2_v6p/` | R2 的数据目录。`pool/`、`pool_ext/`、`edit_tokens.jsonl`、`d4_tokens.jsonl`、`dev_split.json`、`eval_c1.json`、`eval_c*_quick_idx.json` 都是指向 `round2_scaleup_20261030` 的符号链接。另有各实验的行文件，R2 用 `rows_v6_s2hqfull.jsonl` |

## 3. S1、S2（v4）的训练行

两份都由 `samtok-edit round2-data rows-merged --spec scripts/round2/specs/rows_s{1,2}.json` 生成（10-30，代码 `47c0efd`）。种子 20261030；每次更新 32 行。

### 3.1 `rows_s1.jsonl`（192,000 行 = 6,000 × 32）

| 行种类 | 行数 | 占比 | 内容 |
|---|---:|---:|---|
| `hl`（单 token 盲指代高亮） | 95,040 | 49.5% | 目标 = 源图在区域上叠 50% 纯色 |
| `hl_ref`（单 token 带名高亮） | 20,160 | 10.5% | 同上，句子带类别名 |
| `hlm`（多 token 高亮） | 38,400 | 20.0% | 2–5 个区域，各一种颜色 |
| `edit_merged`（SA-1B token 编辑） | 38,400 | 20.0% | D12 孪生成对、D15 池内单目标、D1，各 12,800 |

- 高亮行合计 80%，其中反事实成对 24,000 对（相邻 11,720、同类实例 7,603、部件/宿主 4,677）。
- 单 token 区域按面积：< 0.1% 14.4%，0.1–0.5% 35.4%，0.5–2% 25.2%，≥ 2% 25.0%。
- 共用到 20,795 张池图。

### 3.2 `rows_s2.jsonl`（512,000 行 = 16,000 × 32）

| 行种类 | 行数 | 占比 |
|---|---:|---:|
| `edit_merged`（token 编辑，全部为贴回或原数据目标） | 460,800 | 90.0% |
| 其中多目标行（带 `order` 字段） | 128,000 | 25.0% |
| 其中孪生行 | 52,414 | 10.2% |
| `hl` / `hl_ref` / `hlm`（高亮回放） | 24,576 / 6,144 / 20,480 | 10.0% |

- **合板行按来源：**D11 145,531、D13 93,126、D15 49,118、D12 42,863、D1 38,554、D3 35,840、D10 35,442、D14 11,862、D4 8,464，共 353,601 条不同记录。
- **按单元操作：**删除 175,558、改属性 159,591、替换 157,413、添加 138,735、改字 14,465。
- S1、S2 都对 D12–D15 按新物体描述封顶（每个描述 50 条），去掉了 8,858 条单目标记录。
- 生成后用 `scripts/round2/check_rows.py` 各抽前 4,000 行核对，问题数 0。

## 4. R2 的训练数据

### 4.1 高质量合成目标：自蒸馏池 `NAS/datasets/SITEdit_SelfDistill_20261104/`

- **生成方式：**用原版 Qwen-Image-2.1（原版 TE，不加 adapter）按记录的带名指令**整图**编辑，40 步，seed 0，CFG 1。之后把选区、执行区域、实际改动三者的并集外扩 2% 对角线作为还原区，高斯羽化，区域外换回源图。
  - 与贴回目标的区别：编辑区内是整图生成的自然结果，光影、纹理与上下文一致；贴回目标是局部生成后贴入，容易平涂、过饱和或有接缝。
- **门限：**在还原前的原始输出上计算（`scripts/r2_data/finalize_v2.py`、`distill_metrics.py`）：
  - 无平移（相位相关）；
  - 区域外改动 ≤ 5%，色调漂移 ≤ 3；
  - 区域内确实改动（添加 ≥ 0.15，其余 ≥ 0.3）；
  - 整体改动 ≤ 40%；
  - 改色类区域内细节比 ≥ 0.7。
- **语义审核：**本地 Qwen3.8-27B 判断五项，全部为真才通过：操作对、目标实例对、要求做到、其他不变、自然（`scripts/r2_data/audit_v2.py`、`audit_server.sh`）。

| 批次 | 选样 | 生成 | 通过门限 | 通过审核 | 目录 |
|---|---|---:|---:|---:|---|
| v2（11-05） | S1 实际用到的单目标记录全部，再按 改色/材质 3、添加 2、替换 1、删除 1 加权从 S2 单目标记录中抽；不含 ScaleEdit（`distill_select_v2.py`） | 90,000（另含试点与扩展 6,050） | 30,412 | **26,643**（87.6%） | `v2/`（生成）、`v2_final/` |
| v3（11-07） | 未生成过的单目标记录，含 ScaleEdit，改色/材质和添加优先（`distill_select_v3.py`） | 90,000 | 30,991 | **27,045** | `v3/`、`v3_final/` |

- **审核通过的按来源分布：**
  - v2：D11 9,309、D15 5,510、D12 4,024、D3 3,045、D1 2,995、D10 877、D14 813、D13 70；
  - v3：D11 16,250、D3 3,889、D10 2,639、D15 2,093、D1 879、D14 653、D12 586、D13 56。
  - D13 的条目是多目标记录中单个目标的单独编辑。
- **每个 `*_final/` 目录的内容：**
  - `audit.jsonl`（审核结果）、`metrics.jsonl`；
  - `pool_token.jsonl`：带 token 的合成行。字段 `id`（合板记录 id）、`target`（合成目标图）、`mask`（实际改动）、`noref`、`twin`、`family`、`op`；
  - `pool_plain.jsonl`：同一批对的无 token 普通编辑。字段 `id`（`sd:` 前缀）、`source`、`target`、`mask`、`prompt`、`category`；
  - `targets/`、`masks/`。
- 由 `distill_pool.py` 建池。生成器是 `scripts/quality/distill_gen.py`；四机生成入口在 `QA/launch/sd_gen_4n.sh`、`sd_gen_v3.sh`，运行目录 `QA/runs4n/sdgen_001`、`sdgen3_001`。

### 4.2 `rows_v5_s2.jsonl`（`QA/data_round2_v5/`，11-05，`scripts/r2_data/build_rows_v5.py`）

- **来源：**以 v4 的 `rows_s2.jsonl` 为底，保持同样的更新切片、顺序和孪生/多目标结构。
- **改动：**
  - 单目标编辑行有对应合成目标的，换成合成目标；平涂目标换成合成池行；
  - 约 10% 的行改成无 token 普通编辑行 `edit_plain`，一半来自 NHR-Edit，一半来自自蒸馏普通池；
  - 高亮回放保留。
- **构成：**`edit_merged` 409,560（其中合成目标 106,067，多目标 128,000，孪生 49,672），`edit_plain` 51,240，`hl` 24,576、`hl_ref` 6,144、`hlm` 20,480。

### 4.3 `rows_v6_s2.jsonl`（`QA/data_round2_v6p/`，11-08，`scripts/r2_data/build_rows_v6s2.py`）

在 `rows_v5_s2.jsonl` 上做四处改动：

1. **换掉 NHR-Edit 普通行。**NHR-Edit 已于 11-07 按用户要求删除。51,240 个普通行中有 25,490 个引用它，全部换成自蒸馏普通池中审核通过的行。
2. **"单目标"编辑行中合成 : 贴回 = 70 : 30。**依据是 mix70 试点：在 100:0 / 70:30 / 50:50 / 0:100 四档中，它的 MIRA Overall 最高、PQ 也最高。
3. **孪生行全部恢复贴回目标。**v5 有 8,712 行孪生带合成目标；S1-A 改动孪生后孪生绑定崩了。
4. **高亮行用原覆盖高亮，不加 style 字段。**留色版本使 PF 降 0.84，IF-Bench S4-noref 降 0.09。

- **构成：**`edit_merged` 409,560（合成目标 251,922，多目标 52,893，孪生 49,672），`edit_plain` 51,240，高亮 51,200。
- **注意：**脚本判断"单目标"的条件是 `kind == edit_merged`、非孪生、没有 `partners`。多目标行也满足这个条件，所以在第 2 步中有约 59% 的多目标行被换成了单目标合成行（128,000 → 52,893）。

### 4.4 `rows_v6_s2hqfull.jsonl`（R2 实际使用，11-08，`scripts/r2_data/build_rows_v6s2hqfull.py`）

- 在 `rows_v6_s2.jsonl` 上，把所有"单目标"编辑行中仍是贴回目标的，全部换成合成池（53,688 对，每对平均约 6.7 次）。
- 与 4.3 的问题相同：剩下的 52,893 行多目标也被当作单目标换掉了。**因此 R2 的训练行里没有多目标编辑行。**

逐行统计（2026-11-11）：

| 范围 | 总行数 | 合成目标 token 编辑 | 贴回目标 token 编辑（= 孪生行） | 多目标行 | 普通编辑（全部自蒸馏） | 高亮 `hl` / `hl_ref` / `hlm` |
|---|---:|---:|---:|---:|---:|---:|
| 全文件 | 512,000 | 359,888（70.3%） | 49,672（9.7%） | 0 | 51,240（10.0%） | 24,576 / 6,144 / 20,480（10.0%） |
| **前 32,000 行（R2@1000 实际训练过的）** | 32,000 | 22,486（70.3%） | 3,117（9.7%） | 0 | 3,187（10.0%） | 1,510 / 395 / 1,305（10.0%） |
| 前 224,000 行（R2 全程 7,000 次） | 224,000 | 157,345 | 21,839 | 0 | 22,416 | 10,708 / 2,692 / 9,000 |

- **token 编辑行按来源**（全文件，按合板 id 前缀）：D11 160,323、D12 72,286、D15 58,658、D3 45,669、D1 40,914、D10 20,508、D14 10,318、D13 884。
- **R2 的多目标能力：**来自初始化的 v4。v4 的 S2 训练过 128,000 行多目标。
- **高亮行：**高亮行的区域项在 R2 中为 0，高亮行只通过整图 FM 和路由项起作用（METHOD.md §3）。

### 4.5 NHR-Edit（已删除，只影响历史行文件）

- **曾用于：**11-04 的 anchor300 探针、四机 B 组、v5 的普通编辑行。
- **删除：**2026-11-07 按用户要求删除。
- **影响：**`rows_v5_s2.jsonl` 中引用它的行已无法训练。R2 的行不引用 NHR-Edit（已逐行核对，51,240 个普通行全部来自自蒸馏池）。

## 5. Pass1 定位 LoRA 的数据

详见 `docs/pass1_formal_data.md`，配置在 `configs/pass1/formal_data.json`、`configs/pass1/artifacts.json`。

- **来源**（只读）：
  - 合板默认训练视图（D1、D3、D4、D10–D15）的原图、公开指令、选区 S 和 canonical token；
  - `round2_scaleup_20261030` 区域池（21,132 张图）。
- **规模：**

  | 划分 | 请求 | 目标 |
  |---|---:|---:|
  | warmup | 31,973 | 54,353 |
  | main | 127,888 | 402,833 |
  | dev | 1,000 | 3,550 |
  | test | 999 | 3,547 |

- **输入模态：**ref/noref × none/point/box/mask 七种组合。目标数 K = 1–5。包含 real / hybrid / dense 三种拼法，以及反事实成对。

## 6. 评测数据

| 评测 | 规模 | 数据位置 | 用途 |
|---|---|---|---|
| **IF-Bench 2.6.0**（自建） | 1,084 张 SA-1B 源图，3,872 个单元（每图 2–5 个），10 种 setting，共 10,840 个请求 | 数据根 `NAS/datasets/samtok_edit_benchmark_v2/`；公共 1MP 运行包 `NAS/.ifbench-tmp/one-mp-full4-20261103-judgeio`（有 `READY.json`）；benchmark 仓库 `Tangent0308/samtok_edit_benchmark` 的 `v2branch` | 主表 |
| IF-Bench 快速子集（开发用） | `QA/ifbench_cases80.txt` 排序后前 40 个 case × S1、S3-mask、S4-noref、S5 = 160 个请求、141 个单元 | 外部基线同口径的数字由正式全量 `REPORT.json` 的 per_case 重算（`QA/scripts/ifb_table.py`） | 选方案、选检查点 |
| **MIRA-Bench v1.1** | 200 张真实图，每图 5 条组合子指令 | `NAS/datasets/MIRA-Bench/benchmark/`（`001.jpg`–`200.jpg`）；运行包 `NAS/.ifbench-tmp/external_bench_20261102/mira_bundle` | 外部画质与执行 |
| MIRA 30 张子集（开发用） | 30 张 | `QA/mira_eval/sub30/SUBSET30.json` | 选方案、选检查点 |
| **CompBench** | 3,481 个请求（2,905 单轮 + 288 组两轮）；客观指标 3,285 个，动作/位置/视角 196 个需 GPT | 共享输入 `NAS/.ifbench-tmp/compbench_full_20261107/bundle/`；运行包 `NAS/.ifbench-tmp/compbench_full5_20261109/package-v1/` | 外部泛化 |
| **MagicBrush** | 535 个会话、1,053 个编辑轮次；无 mask、独立单轮 | `NAS/datasets/MagicBrush`；运行包 `NAS/.ifbench-tmp/magicbrush_4node32_nas_20261106/` | 外部泛化 |
| 开发集探针 | 1,320 项，在 60 张留出的 SA-1B 开发图上：single 600、multi 100、pair 400、edit 100、remove 120 | `V3/data/round2_20261016/eval_c1.json` | token 读取能力 |

- **历史暴露：**IF-Bench 40 例子集和 MIRA 30 张子集在 11-04 到 11-10 之间被反复用于选方案和选检查点。
- **IF-Bench 全量：**合并报告了全部 1,084 张图，没有划分 dev/test。

## 7. 路径总表

| 内容 | 路径 |
|---|---|
| 合板数据 | `NAS/datasets/SAMTokEdit_Merged_20261029/` |
| S1/S2 数据目录与行 | `V3/data/round2_scaleup_20261030/`（`rows_s1.jsonl`、`rows_s2.jsonl` 与各自的 `_stats.json`） |
| 取行配置 | 仓库 `scripts/round2/specs/rows_s1.json`、`rows_s2.json` |
| 自蒸馏池 | `NAS/datasets/SITEdit_SelfDistill_20261104/{v2,v3}`（生成）、`{v2_final,v3_final}`（门限 + 审核后的池） |
| v5 行 | `QA/data_round2_v5/rows_v5_s2.jsonl` |
| v6 / R2 行 | `QA/data_round2_v6p/rows_v6_s2.jsonl`、`rows_v6_s2hqfull.jsonl` |
| R2 数据生成脚本（仓库副本） | `scripts/r2_data/`（NAS 原件在 `QA/scripts/`） |
| Pass1 正式数据 | `SE/pass1_localization_lora_formal_20261030_0939/data/`（运行根目录，含 `ARTIFACTS.json`；说明见 `docs/pass1_formal_data.md`） |
| 评测数据 | 见 §6 |
