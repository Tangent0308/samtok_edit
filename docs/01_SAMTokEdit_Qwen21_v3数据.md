# SAMTokEdit Qwen-Image-2.1 v3：数据

本文是 v3 的数据侧总结，内容包括：
- 统一的指令协议；
- 训练数据的清洗和协议适配；
- 新的数据管线和产出的数据；
- 第二轮训练行的构成；
- 开发评测集和 benchmark 的要点；
- 每份数据的路径。

方法见[文档 02](02_SAMTokEdit_Qwen21_v3方法.md)，实验见[文档 03](03_SAMTokEdit_Qwen21_v3实验.md)。截至 2026-10-24。

路径记号（`scripts/docs/check_paths.py` 会逐一检查）：`datasets/…` 指 `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/…`，`V3/…` 指 v3 实验根 `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v3/…`。

## 0. 总览

| 代号 | 数据 | 规模 | 用在哪里 | 路径 |
|---|---|---|---|---|
| — | v1 原始编辑对（四个打标子集） | 98,574 对 | 清洗的输入；v1/v2 模型 | §2.1 |
| — | 清洗后编辑对 | 46,482 对 | 协议适配的输入 | `datasets/SAMTokEdit_Cleaned_20261009/` |
| D3 | v1 单目标，协议 2.0 | 39,615 条 | C2（`edit_v1`） | `datasets/SAMTokEdit_Cleaned_20261009/protocol_adapt_qwen38_27b_v1/protocol_v2_blind/manifest_protocol_v2.jsonl` |
| D4 | v1 多目标召回，协议 2.0 | 4,066 对 / 9,964 单元 | C2（`edit_multi_v1`） | `datasets/SAMTokEdit_Cleaned_20261009/protocol_multi_qwen38_27b_v1/protocol_v2_blind/manifest_multi_protocol_v2.jsonl` |
| D1 | SA-1B 50k 自蒸馏编辑，协议 2.0 | 50,000 条 / 1,132 张源图 | C1、C1b（编辑 15%）；C2 主体；开发集 | `datasets/sa1b_samtok_edit_50k_20261014/release/annotations/train_protocol_v2.jsonl` |
| D2 | SA-1B dense mask 区域池 | 1,132 张图，训练可用 62,873 个区域 | C1、C1b、C2 的指认；开发集 | `V3/data/round2_20261016/pool/` |
| D5 | 50k 同图拼合多目标 | 训练时生成 | C2（`edit_multi`） | `V3/data/round2_20261016/support_cache.jsonl` |
| D6 | 数据侧构造的孪生与链式多目标 | 计划孪生 1.5 万组 + 多目标 5,000 组 | 尚未交付，未使用 | `experiments2/SAMTokEdit/data_pipeline/02_production_runs/sa1b_samtok_group_train_20k_20261016/README.md` |
| D7 | 写字编辑（数据线新产） | 计划 3,000 条 | C3b（计划，§2.11） | 待交付 |
| D8 | 结构替换（数据线新产） | 计划 4,000 条 | C3b（计划，§2.11） | 待交付 |
| D9 | 整体改属性（数据线新产） | 计划 2,000 条 | C3b（计划，§2.11） | 待交付 |
| D10 | CrispEdit、ScaleEdit 新数据，协议 2.0（extedit 第一批，§2.12） | 38,346 条（单目标 36,430、多目标 1,916）/ 40,791 单元 | C3-D、C3-MD（12,000 行）；合板 | `datasets/ExtEdit_Protocol2_20261021/` |
| D11 | CrispEdit、ScaleEdit 新数据，协议 2.0（extedit 第二批，§2.12） | 151,337 条（单目标 141,626、多目标 9,711）/ 163,595 单元 | 合板 | `datasets/ExtEdit_Protocol2_20261022/` |
| 合板 | D1、D3、D4、D10、D11 合成一份，加文字标记和默认训练视图（§2.13） | 283,364 条 / 303,965 单元；默认视图 258,577 条 / 278,240 单元 | 尚未训练（`edit_merged` 已就绪） | `datasets/SAMTokEdit_Merged_20261024/` |
| — | 第一轮指认数据（E-A1、E-A2） | 42,000 行 / 每轮 82,000 行 | E-A1、E-A2 | `V3/data/stageA_ea1_20261011/`、`V3/data/stageA_ea2_20261012/` |
| — | 第二轮开发评测集 | 1,320 项 | C1、C1b、C2 及基线 | `V3/data/round2_20261016/eval_c1.json` |
| — | benchmark v2，release 2.6.0 | 1,084 张图 / 3,872 单元 | 最终测试（未正式跑）；10-19 拆分 36 例 | `datasets/samtok_edit_benchmark_v2/` |

**D3、D4、D10、D11 按类型的单元数**（10-23 盘点，10-24 加 D11）：

| 类型 | D3 | D4 | D10 | D11 | 合计 |
|---|---:|---:|---:|---:|---:|
| 添加 | 7,247 | 315 | 1,304 | 46,733 | 55,599 |
| 删除 | 12,207 | 4,837 | 3,858 | 42,430 | 63,332 |
| 替换 | 7,315 | 885 | 12,396 | 42,235 | 62,831 |
| 改属性（不含改字） | 11,454 | 3,877 | 10,986 | 32,135 | 58,452 |
| 改字 | 1,392 | 50 | 12,247 | 62 | 13,751 |
| 合计 | 39,615 | 9,964 | 40,791 | 163,595 | 253,965 |

- 改属性以改色为主：D3 改色 9,946、材质 1,109；D10 改色 7,123、材质 3,599。
- 目标 S 面积的中位数：
  - D3 约 3%–5%；
  - D4 约 1%–3%；
  - D10 替换 6.6%、改属性 9.1%、添加 6.6%、删除 3.2%、改字 1.6%，明显大于 D3/D4。
- token 往返 < 0.5（训练时不用）：D3 10.4%，D4 4.8%，D10 为 0（打标时已卡掉）。

**时间线**

| 日期 | 事项 |
|---|---|
| 10-08 | v1 98,574 对逐对审计 |
| 10-09 | 四个子集归位，全量像素清洗 → 70,956 |
| 10-11 | Qwen3.8-27B 编辑核验 → 48,437 |
| 10-12 | 去掉 text/action → 46,482；benchmark 2.3.0 引入协议 1.0 |
| 10-14 | 协议 1.0 单目标适配全量（四机）→ 39,615；50k 生成启动 |
| 10-15 | 多目标召回 4,066 对；50k 发布；协议 2.0 定稿；D3、D1 适配到 2.0；benchmark 2.5.0 |
| 10-16 | 50k 补 `edited_region`；第二轮数据（开发划分、区域池、C1 行、开发集）；benchmark 2.6.0 |
| 10-17 | C1b 行；训练图诊断集 |
| 10-18 | D4 适配到 2.0；v1/D4 token 缓存；C2 行 |
| 10-19 | 从 benchmark 2.6.0 拆出 36 个单指令 case |
| 10-20 | 核查 C2 训练数据（中等尺度不缺；部分目标"做了一半"；写字编辑多是改字）；制定 C3 数据计划（§2.11） |
| 10-20 | 梳理 CrispEdit、ScaleEdit 从源数据到 D3/D4 的完整流程，设计统一打标管线（§2.12） |
| 10-21 | extedit 第一批发布（D10）；10-22 加自然度复查后重新发布 |
| 10-24 | extedit 第二批发布（D11）；合板前统一文字复查；五批合板（§2.13） |

## 1. 统一的指令协议

训练数据和 benchmark 用同一套指令协议。共享模块是 benchmark 仓库的 `samtok_benchmark/v2/instruction_protocol.py`，v3 仓库逐字节复制成 `src/samtok_edit21/preparation/instruction_protocol_v2.py`（sha256 `fe192e8d…`，与 benchmark 2.6.0 一致）。

### 1.1 协议 1.0（benchmark 2.3.0 引入，2.4.0 补充 mask 语义 1.1）
- **一个单元一条语义记录 `instruction_spec`：**字段有 `operation`、`ref`、`scope`、`payload`、`target_part`、`remove_mode`、`attributes`、`placement`、`qualifier`。ref 句和 no-ref 句都由模块从 spec 渲染，不手写。
- **S 和 T：**S 是区域选中的源实例，定位信息总指向 S；T 是实际被编辑的东西，可以是 S 本身，也可以是 S 的部件（`target_part`）。
- **四类操作：**
  - add：`Add <item> <relation> [<site> of] <S>`。区域标锚点（新物体直接站、放、挂、附着的物体或表面），不标新物体本身。
  - replace：必须有结构或品类变化；只换颜色、材质、图案的归 attribute。
  - remove：六种模式 whole、part、contents、contents_only、lowerlegs、hem；删文字或 logo 也归 remove。
  - attribute：color、material、finish、pattern、texture、text、tint、opacity，可同时改多项。
  - 翻转、移动、重新摆姿势不在协议内。
- **单元：**一个源实例（或其部件）+ 一种操作。target_part、contents 类 remove 和参照式 add 必须给出显式执行区域。
- **训练侧补充规则**（用户 10-12/13 决定）：
  - 一堆或一串同类物算一个对象；
  - 改字归 `attribute.text`，删字归 remove，加字归 add；
  - add 的锚点要裁到落点，不能盖住整个大物体。

### 1.2 协议 2.0：no-ref 改为盲指代（用户 10-15 定稿）
- **为什么改：**方法的理想输入是"用户只给区域和编辑要求"，不会说出对象类别。1.0 的 no-ref 写成 `the selected <scope>`（如 `the selected sleeve`），只看文字就能分辨目标，DiT 就不去读 token（v2 同图同句换 token 的跟随率只有 0.06）。
- **三条决定：**
  1. no-ref 一律写成 `the selected region`，不出现类别、部件、宿主、site、方位、序数，也不出现源物体现状的特征。
  2. 被编辑的部分就是定位区域：部件、宿主的描述并入 `the selected region`，源 mask 改为选中被编辑的部分本身（mask 语义 2.0）。
  3. 编辑结果里的类别信息（payload、属性值、结果说明）原样保留，如 `Replace the selected region with a flared bell sleeve.`。
- **mask 语义 2.0 相对 1.0 的变化：**

  | 情形 | 1.0 的 mask | 2.0 的 mask |
  |---|---|---|
  | attribute/replace 带 target_part | 宿主 | 部件本身 |
  | remove part | 宿主 | 被删部件 |
  | remove lowerlegs / hem | 整件衣物 | 被剪掉的那段 |
  | remove contents / contents_only | 容器 | 容器 ∪ 内容物 / 内容物 |
  | add 带 site | 宿主 + 文字 site | site 本身（附着部位） |
  | 整物体、不带 site 的 add、参照式 add | — | 不变 |

- **句式：**
  - `Change the <property> of the selected region to <value>.`
  - `Replace the selected region with <payload>.`
  - `Remove the selected region.`；按 remove 模式在句末加说明，如 `, including its contents.`。
  - `Add <payload> <relation> the selected region.`
- **token 的位置：**训练和本方法推理时，⟨M⟩ 紧跟在 `the selected region` 之后。
- **qualifier 的三种处理：**
  - 用来区分目标的限定并入 `the selected region`；
  - 只描述编辑要求的保留；
  - 提到图中其他物体的保护说明整句去掉。

## 2. 训练数据

### 2.1 v1 原始数据：四个打标子集

| 子集 | 上游 | 最终集 | 行数 | 进入合并源对 |
|---|---|---|---:|---:|
| RefEdit | HF `bpathir1/RefEdit` | `datasets/RefEdit_Labeling/final/RefEdit-mask-prefiltered-qwen38-self-contained/` | 7,804 | 7,804 |
| CrispEdit | HF `WeiChow/CrispEdit-2M` | `datasets/CrispEdit_Labeling/final_dataset_39k/` | 38,971 | 37,728 |
| ScaleEdit | `QingyuShi/scaleedit-filtered-6m` + `InternVL-U/ScaleEdit-12M` | `datasets/ScaleEdit_Labeling/final/scaleedit_25k/` | 25,664 | 25,085 |
| SAMTok Derived | 自建：GRES/VER 实例 mask + Qwen-Image-2.1 生成目标图 | `datasets/SAMTok_Derived_Edit_Labeling/combined/manifest.jsonl` | 27,957 | 27,957 |
| 合计 | | | 100,396 | **98,574** |

- **合并源对：**`experiments/SAMTokEdit/qwen21_full4_20260928/data/sources.jsonl`。CrispEdit、ScaleEdit 只收 mask 质检通过的行。
- **打标流程：**Qwen3.8-27B 质量预筛 → 场景难度筛 → grounding → SAM3 分割。add 在目标图上标新物体，其余类型标在源图上。
- **v1/v2 训练实际用的是未清洗的转换版：**`experiments2/SAMTokEdit/qwen21_v2/data/train_v2_box_001/`（stage1 110,079 行、stage2 290,006 行）。no-ref 当时由 Qwen3.5-9B 改写成 `this region ⟨M⟩`。

### 2.2 v1/v2 数据的问题
来源：2026-10-08 逐对审计（`docs/archive/v2/09_SAMTokEdit_Qwen21_训练数据诊断与改进.md`）。

| 问题 | 证据 |
|---|---|
| 源图与目标图不对齐（主要根因） | 主四类只有 48% 可直接用，10% 贴回可修，35% 改动扩散，8% 空操作或过弱。四个打标流程都没做对齐检查 |
| ScaleEdit 整体缩放漂移 | 同尺寸样本缩放中位 1.075、平移 42 px；v2 模型输出漂移率 17%–20% |
| 贴回造成"软补丁" | 贴回后编辑区锐度只有周围的 0.62–0.70 |
| add 的区域标的是新物体本身 | 8,718 对 add 中 7,825 对只有目标图实例；token 往返 IoU add 0.38，remove 0.88 |
| 任务不对口 | Derived 的 add 多是"小贴纸"（6,269 条中 5,371 条含 small/tiny） |
| 空操作、平涂 remove、指令与区域不一致 | Derived attribute 444 条改动 < 10%；250 个 remove 抽样中 17 个是平涂 |
| 区域偏大 | 清洗后 attribute 区域中位 6.3%，评测 0.49% |
| 同一张图以多个路径出现 | 82,585 个源图路径中 3,139 组字节相同；E-A2 起用 16×16 均值哈希排除评测图 |

### 2.3 清洗流水线（输出 46,482 对）

```
98,574 源对
 → 像素清洗（CPU，10-09）                                   70,956
 → Qwen3.8-27B 编辑核验（四机 32 卡，约 66 分钟，10-11）       48,437
 → 去掉 text、action（10-12）                                 46,482
```

- **像素清洗**，依次判断：
  1. 区域内明显改动 < 10%：丢。
  2. 颜色校正后，离区域较远处有大于全图 1% 的连通改动：丢。
  3. 远区域外的改动 ≤ 2%：保留。
  4. 否则做 SIFT + RANSAC 仿射配准后贴回，配准后的区域外改动必须明显下降。
  5. 不配准的做纹理块对齐检查。
  6. 贴回后再检查：区域外改动、区域内改动、边界环。
- **像素清洗的判定计数：**保留 42,154，贴回 5,528，配准后贴回 23,274；丢弃 27,618。更正坐标的实例 6,307 个。
- **MLLM 核验**（方案 2）：Qwen3.8-27B 看三张图（原图红轮廓、编辑后、区域放大），用两种问法：
  - 按类型的问题清单（思考模式）；
  - 先盲描述改动，再对照指令。

  任一判坏即丢。人工子集上抓到 86% 的坏例，误判 8.1%。
- **按来源（源对 → 像素 → 核验）：**
  - RefEdit 7,804 → 7,108 → 4,570；
  - CrispEdit 37,728 → 22,806 → 12,444；
  - ScaleEdit 25,085 → 19,368 → 13,519；
  - Derived 27,957 → 21,674 → 17,904。
- **路径**（`datasets/SAMTokEdit_Cleaned_20261009/`，README 在 `datasets/SAMTokEdit_Cleaned_20261009/README.md`）：
  - `manifest.jsonl`：像素保留 70,956；
  - `decisions.jsonl`：全部判定；
  - `edit_check_qwen38_27b_v1/manifest_kept_no_text_action.jsonl`：46,482。
- **代码：**v2 分支（数据侧会话在那里开发），即 `scripts/data_cleaning/clean_pairs.py`、`src/samtok_edit21/preparation/edit_check.py`。v3 仓库里有同名副本。
- **已知局限：**像素规则判断不了编辑做得对不对；add 区域只盖住物体一部分时，贴回会切掉其余部分（约占贴回 add 的 5%）。

### 2.4 单目标协议适配（D3：39,615 条）
- **协议 1.0 适配**（10-14，ARNOLD 4×8 卡，约 2.5 小时，代码 v2 分支 fff8ca5 + f90d54d）：
  - **输入：**清洗后的 47,402 行，即 48,437 去掉 action 和"无单一类型"的行，text 行按改字、删字、加字分流。
  - **每行的流程：**
    1. Qwen3.8-27B 看前后图、编号描边图和特写，输出受约束的 JSON，确定 S；
    2. 需要时用 SAM3 重新分割，add 先分割承载物再裁到落点；
    3. 渲染 ref/no-ref 句，做几何把关，算 `edited_region`（Lab ΔE 双阈值 6/12）；
    4. 27B 两票审核，不过的带着意见修一次。
  - **结果：**接受 39,615（修复前 39,661），另有多目标 5,724、拒收 2,063。
    - 接受的按操作：attribute 12,846、remove 12,207、replace 7,315、add 7,247；
    - 按来源：Derived 17,095 占 43%。
  - **抽查：**60 条约 86% 对、9% 小瑕疵、5% 错。S 中位面积约 4%。
  - **路径：**`datasets/SAMTokEdit_Cleaned_20261009/protocol_adapt_qwen38_27b_v1/`。
- **协议 2.0 适配**（10-15，v3 仓库 `scripts/data_cleaning/protocol_v2_adapt.py v1`，规则驱动，不调用 MLLM）：
  - `instruction_noref` 改为盲指代 39,356 条；259 条无法分类的 qualifier 留空，只能作 ref 行。
  - 换 mask：带 site 的 add 裁到附着处 25 条，remove contents 与执行区域取并集 22 条。315 条带 site 的 add 锚点已经贴合，不改。原值存在 `selection_v1`。
  - 结果文件即 D3：`datasets/SAMTokEdit_Cleaned_20261009/protocol_adapt_qwen38_27b_v1/protocol_v2_blind/manifest_protocol_v2.jsonl`。

### 2.5 多目标召回（D4：4,066 对 / 9,964 单元）
- **来源：**单目标适配时判为多目标的 5,724 对，改写成 benchmark 式的多单元 case。每对 2–5 个单元，每个单元有自己的 spec、ref/no-ref、S、执行区域和 `edited_region`。
- **构造：**10-15，本机 8 卡 3 小时，Qwen3.8-27B + SAM3。
- **规则：**
  - 一个单元一个物体；
  - 衣物而非整个人；
  - S 之间不共享、不嵌套；
  - 未解释的改动 ≤ 10%；
  - 整对两票审核。
- **结果：**接受 4,066 对。
  - 2/3/4/5 单元的对数：2,820 / 782 / 342 / 122。
  - 单元操作：remove 4,837、attribute 3,927、replace 885、add 315，以删除和改属性为主。
  - 抽查约 88% 对。
- **协议 2.0 版**（10-18，`scripts/data_cleaning/protocol_v2_adapt.py multi`）：盲指代 9,893 个单元，71 个留空。结果文件即 D4：`datasets/SAMTokEdit_Cleaned_20261009/protocol_multi_qwen38_27b_v1/protocol_v2_blind/manifest_multi_protocol_v2.jsonl`，README 在同目录。
- **合计：**单目标 39,615 + 多目标 4,066 = 43,681 对，占 v1 98,574 的 44.3%。

### 2.6 SA-1B 50k 自蒸馏数据（D1：新管线，50,000 条）
这是用户另一条线的新数据管线：仓库 `sam3-crispedit` 的分支 `sa1b-samtok-edit-data`，本机没有 clone。10-14 启动，10-15 发布，4 机 × 8 卡，墙钟约 20 小时。

1. **规划：**Qwen3.8-Flash-Next-FP8 在 SA-1B dense mask 上逐个 mask 规划一种编辑（协议 1.0），按缺额补齐类型。
2. **生成（教师）：**用 Qwen-Image-2.1 原版（40 步，seed 0）。
   - 围绕支持区域裁图，提示 = ref 指令 + 保持说明 + "目标在裁图中的位置"；
   - 编辑后按支持区域羽化贴回，所以**支持区域外与源图逐像素相同**。
   - 这是"特权自蒸馏"：教师看得到区域和局部放大，学生只看 token 和整图。
   - 支持区域策略：add 取锚点 + 附着处，attribute 只取目标，remove/replace 取目标 + 一圈。
3. **审核：**确定性检查 + Qwen3.8-27B 语义审核（操作对、目标做到、非目标保持、自然）。
4. **规模：**
   - 生成 79,821 条，自动通过率 62.6%（remove 最低，42.1%）；
   - 按四类各 12,500 补齐到 50,000 条；
   - 只覆盖 **1,132 张源图**，每张平均 44 条。
5. **与 benchmark 隔离：**按源图 ID 和字节哈希排除，和 benchmark 1,084 个 case 的交集为 0。没有做近重复检索。没有人工验收。
6. **协议 2.0 适配与补充字段**（10-15/16，原文件不动，另写 `train_protocol_v2.jsonl`）：
   - 盲指代 49,753 条，247 条留空。
   - 带 site 的 add 有 6,754 条，其中 3,606 条把 mask 裁到附着处（新 mask 在 `datasets/sa1b_samtok_edit_50k_20261014/release/masks/region_v2/`）。
   - 补 `edited_region`（实际改动像素，ΔE 双阈值，只取支持区域内），RLE 写在记录里，PNG 在 `datasets/sa1b_samtok_edit_50k_20261014/release/masks/edited/`。168 条为空。
   - **面积中位：add 0.023%、remove 0.12%、replace 0.16%、attribute 0.11%。**区域非常小，这是第二轮"不执行"问题的背景之一（文档 03 §5.4）。
7. **训练目标的完成度（10-20 核查，`V3/analysis/c3_plan_20261020/`）：**C2 用到的 38,225 条里，按 `edited_region` 统计目标区域被改动的比例：
   - replace 有 1,153 条（11.5%）、attribute 有 1,126 条（11.2%）改动不到目标的一半；
   - add 有 861 条（10.2%）的锚点或附着处被改动超过 60%；
   - remove 改动不到一半的只有 1.0%。

   这些行对应 C2 在 benchmark 上"叠加、只改一部分、改坏附着物"的失败。C3 计划不再使用它们（文档 03 §6，待确认）。
8. **路径：**
   - 说明：`datasets/sa1b_samtok_edit_50k_20261014/release/README.md`；
   - 协议 2.0 版即 D1：`datasets/sa1b_samtok_edit_50k_20261014/release/annotations/train_protocol_v2.jsonl`；
   - 原始版：`datasets/sa1b_samtok_edit_50k_20261014/release/annotations/train.jsonl`。

### 2.7 SA-1B dense mask 区域池（D2）与指认数据
- **来源：**50k 同批源图的 SA-1B dense 标注，`experiments2/SAMTokEdit/data_pipeline/02_production_runs/sa1b_samtok_edit_50k_20261014/sources/`（每张图一个 `dense.json`）。
- **区域池**（`V3/data/round2_20261016/pool/`，1,132 个文件，`samtok-edit round2-data` 生成）：
  - 保留面积 0.05%–30% 的 mask；
  - 每个 mask 记录 SAMTok token、token 再解码出的区域（作为监督目标）、往返 IoU、类别词，以及同图关系：同类、宿主/部件、接触、重复标注。
  - 往返 IoU ≥ 0.5 才算可用。训练图 1,072 张，可用区域 62,873 个。
- **面积分桶抽样**（单 token 指认）：

  | 面积 | <0.1% | 0.1–0.25% | 0.25–0.5% | 0.5–1% | 1–2% | 2–5% | ≥5% |
  |---|---|---|---|---|---|---|---|
  | 份额 | 5% | 20% | 20% | 20% | 15% | 12% | 8% |

  < 0.5% 的合计 45%，贴近 benchmark。
- **指认的写法与目标：**
  - 单 token：`Apply a red semi-transparent mask to the selected region ⟨M⟩ and keep everything else unchanged.` 等 4 个模板；带名版用类别词替换 `the selected region`。
  - 多 token：一句话列出 2–5 个区域，各配一种颜色。
  - 颜色 6 种。目标图 = 源图在区域上叠 50% 颜色。

### 2.8 天然孪生与拼合多目标（D5），以及数据侧的 D6
- **天然孪生：**同一源图上，盲指代 no-ref 逐字相同的多条 50k 记录（例如同一张图上的多条 `Remove the selected region.`），只有 token 能区分。
  - 10-17 统计：3,610 组、103,157 对（组内两两组合），93% 是 remove，86% 相距较远。
  - C2 实际使用：组内按框距最近两两配对，6,590 对、13,180 行。
- **拼合多目标（D5）：**同一源图上执行区域不相交的 2–4 条 50k 编辑。50k 的目标在支持区域外与源图逐像素相同，可以直接合成联合目标，不需要新数据。C2 中 11,270 行。
- **D6：**数据侧按需求单（旧文档 06）构造的孪生（训练 1.5 万组 + 评测 200 组）和链式多目标（训练 5,000 组 + 评测 100 组）。正式任务还没交付，C2 没有使用。说明见 `experiments2/SAMTokEdit/data_pipeline/02_production_runs/sa1b_samtok_group_train_20k_20261016/README.md`。

### 2.9 第一轮数据（E-A1、E-A2）
- 单元来自 v2 训练清单 `train_v2_box_001` 的 `edit_umt` 行（去掉 add，恰好一个 mask）。真实编辑取像素清洗判 `keep` 的 RefEdit/Derived 对。按源图路径哈希的 10% 留出。
- **E-A1**（`V3/data/stageA_ea1_20261011/`，20 MB）：
  - 训练 42,000 行 = 指认 30,000（不带物体名 17,029、带名 12,971）+ 程序化局部编辑 8,000 + 真实编辑 4,000；
  - 评测 1,930 项 = 留出 ref 393 / noref 607，dev ref 122 / noref 122，反事实对 686；
  - 快速评测 96 项。
- **E-A2**（`V3/data/stageA_ea2_20261012/`，3.9 GB，含 GRES 图 6,962 张）：
  - 每轮 82,000 行 × 2 轮 = 指认 62,000（含 GRES 多实例 12,000）+ 程序化 12,000 + 真实 8,000；
  - 同图干扰实例 27,624 行；
  - 按均值哈希去掉了 529 个与评测图相同的源；
  - 评测同 E-A1 的 1,930 项，快速评测 160 项。

### 2.10 第二轮训练数据（`V3/data/round2_20261016/`，313 MB）

| 文件 | 内容 |
|---|---|
| `README.md` | 目录说明 |
| `dev_split.json` | 留出的 60 张 SA-1B 源图和 40 条 v1 记录，这些图的编辑和 dense mask 不进任何训练 |
| `pool/` | D2 区域池 |
| `edit_tokens.jsonl` | 50k 每条记录源 mask 的 token、解码区域、往返 IoU（与 D1 同序） |
| `v1_tokens.jsonl`、`d4_tokens.jsonl` | D3、D4 的 token 缓存，另记目标图在区域外改了多少（判断漂移） |
| `support_cache.jsonl` | 50k 每条的支持区域（32 px 网格），用来挑不相交的拼合单元 |
| `rows_c1.jsonl`、`rows_c1b.jsonl`、`rows_c2.jsonl`（各带 `_stats.json`） | 三次训练的训练行 |
| `eval_c1.json`、`eval_c1_quick_idx.json`、`eval_c2_quick_idx.json`、`eval_train_diag.json` | 开发评测集、训练中快速评测子集（各 112 项）、C1 的"训练图 vs 新图"诊断集 |

**token 往返过滤：**往返 IoU < 0.5 的 token 指不到要编辑的区域，用来训练会教模型不看 token，因此不用。

| 数据 | 总数 | 开发集 | 往返 < 0.5 | 可用 |
|---|---:|---:|---:|---:|
| D1 50k 单条 | 50,000 | 2,804（60 张图） | 8,971（add 3,346） | **38,225** |
| D3 v1 单条 | 39,615 | 40 | 4,111 | **35,464** |
| D4 多目标 | 4,066 对 | — | 385 对 | **3,681 对** |

**各次训练的行构成：**每次更新 32 行，行号决定由哪张卡在哪一步处理，见文档 02 §4.12。

| 训练 | 总行数 | 构成 |
|---|---:|---|
| C1（从原版起训，2,000 次更新） | 64,000 | 单 token 指认 25,600、带名 9,600；多 token 指认 19,200；50k 编辑 9,600（盲指代 + token 6,256、物体名 + token 1,944、只有物体名 1,400） |
| C1b（从 C1@1500 起训，800 次更新） | 25,600 | 同 C1 的比例（10,240 / 3,840 / 7,680 / 3,840）；近邻反例（99.9% 的单 token 行有，平均 5.93 个）；2,112 对反事实成对；40% 多 token 行围绕相邻对 |
| C2（从 C1b@800 起训，3,522 次更新） | 112,704 | 见下表 |

C2 的行（"可用数据全部用一次，add 用两次"）：

| 种类 | 行数 | 说明 |
|---|---:|---|
| `edit`（D1） | 46,690 | 38,225 条全用，add ×2；add 16,930 / remove 9,680 / replace 10,066 / attribute 10,014；天然孪生 13,180 行（remove 的 92% 是孪生） |
| `edit_v1`（D3） | 39,768 | 35,464 条全用，add ×2；add 8,608 / remove 11,956 / replace 7,174 / attribute 12,030；9,786 行目标有区域外改动，只算局部损失 |
| `edit_multi_v1`（D4） | 3,681 | 2/3/4/5 单元 = 2,592 / 686 / 302 / 101；1,576 对用局部损失 |
| `edit_multi`（D5） | 11,270 | 2/3/4 单元 = 4,539 / 3,894 / 2,837，单元操作互不相同 |
| 指认 | 11,295 | 单 token 4,955、带名 1,832、多 token 4,508；含 1,018 对反事实成对 |

- **写法：**单条编辑盲指代 + token 60,607 行（70%）、物体名 + token 25,851 行（30%）。没有不带 token 的行，这是用户 10-18 的决定：方法始终带 token。
- **区域大小：**D1 的 token 区域中位 0.16%（73% < 0.5%），D3 中位 4.1%。D1 偏小，但中等尺度并不缺：C2 用到的 50k 编辑里，remove/replace/attribute 有 28%–33% 改动 ≥ 0.5%。这是 10-20 的核查（`V3/analysis/c3_plan_20261020/`）；旧版这里写的"0.2%–5% 缺档"不对。
- **生成命令：**`python -m samtok_edit21 round2-data rows-c2 --out <dir>`，代码 `src/samtok_edit21/preparation/round2_data.py`。

### 2.11 下一步数据计划（C3，10-20 制定，待确认）
- **依据：**C2 在哪里输、为什么输，见文档 03 §5.7；训练怎么用这批数据，见文档 03 §6。
- **数据核查：**脚本与结果在 `V3/analysis/c3_plan_20261020/`。
- **原则：**只补 C2 稳定落后的几类，不伤保持（C2 最大的优势），因此新数据仍按支持区域贴回。

**要补的差距**（benchmark 子集的单元编辑通过率；单元数按全量 benchmark）

| 差距 | C2 | 对手 | 全量单元数 | 数据侧原因 |
|---|---|---|---|---|
| 写字 | 0.42 | FLUX.3 0.74，原版 0.58 | 193（5.0%） | 训练里写字编辑约 1,200 条（v1 约 1,130、50k 77），但 88% 是"把已有的字改成 X"。benchmark 的写字单元 95% 是新加带字物体（63%）或换成带字物体（32%），这两种训练里只有约 150 条 |
| 替换 | 0.74 | 原版 0.84 | 969，其中面积 ≥ 0.25% 的 716 | 50k 替换有 11.5% 改动不到目标的一半；贴回范围只外扩约目标边长的 26%，轮廓变化大的替换进不来 |
| 大目标改属性 | 0.84 | FLUX.3 0.95 | 438（面积 ≥ 1%） | 50k 改属性有 11.2% 改动不到目标的一半，只有 18% 的目标 ≥ 1%（benchmark 44%） |
| 添加时改坏附着物 | 4 个失败 | — | — | 50k 添加有 10.2% 的锚点或附着处被改动超过 60% |

**A. 现有数据清洗（v3 侧，C3a 使用，10-21）**

"改动比例"指训练目标图在目标区域（token 的选区）内改动了多少。

| 来源 | 去掉 | 规则 |
|---|---|---|
| 50k（D1） | replace 1,153 条、attribute 1,126 条 | 改动比例 < 0.5 |
| 50k（D1） | add 861 条 | 锚点或附着处改动比例 > 0.6 |
| v1（D3） | replace 112 条、attribute 365 条 | 改动比例 < 0.5，由记录里的选区和改动区域算出（`v1_coverage.txt`） |
| v1（D3） | add：先目测 50 条再定 | 776 条改动比例 > 0.6。v1 的 add 选区有时就是新物体所在的整个表面，改动比例高不一定是改坏了，所以不直接套用规则 |

- 其余与 C2 相同：孪生、add 加倍、多目标拼合、指认比例。
- 不做放大裁剪：50k 并不缺中等尺度（§2.10）。

**B. 数据线新产（C3b 使用，10-21 起，10-25 交付）**

三个子集共同的要求：

| 项 | 要求 |
|---|---|
| 管线 | 沿用 50k：规划 → 教师生成（原版 Qwen-Image-2.1，40 步）→ 按支持区域贴回 → 审核 |
| 源图 | SA-1B。与 benchmark 隔离：源图 ID、字节哈希，再加近重复检索（50k 当时没做）。不与 50k 的 1,132 张源图重复 |
| 分散 | ≥ 5,000 张不同源图，每张图每个子集 ≤ 3 条（50k 平均每张 44 条） |
| 写法 | 协议 2.0 的 spec，渲染带名和盲指代两种句子；mask 按 mask 语义 2.0：部件目标只圈部件，add 圈附着处 |
| 贴回 | 支持区域外与源图逐像素相同 |
| 记录 | 支持区域、实际改动区域（RLE）、`s_covered`（目标区域被改动的比例）、支持区域外强改动比例 |
| 审核 | 照旧问四项：操作对、目标做到、非目标保持、自然；再加各子集的专项检查，结果写进记录 |
| 交付格式 | 与 50k 的 `train_protocol_v2.jsonl` 字段相同，另加各子集的专项字段 |

三个子集：

| 子集 | 代号 | 通过数量 | 生成量（估） | 内容与配比 | 区域 | 专项检查（全部通过才收） | 专项字段 |
|---|---|---|---|---|---|---|---|
| 写字 | D7 | 3,000 | 约 6,000 | 新加带字的牌子、贴纸、标签、贴花 60%；换成写着某字的牌子 35%；改已有牌子上的字 5%（已有约 1,050 条）。字串 1–3 个词、≤ 20 个字符，大写英文词和数字为主（OPEN、BEACH ACCESS、A-12、28），少量店名路名 | 0.1%–3%；字在 1 MP 画布上清晰可读 | OCR 读出与要求一致（忽略大小写和标点）；没有乱码或多余的字；附着物不变色、不变形 | `text_target`、`ocr_text`、`text_form` |
| 结构替换 | D8 | 4,000 | 约 7,000 | 只做品类或形状变化：T 恤 → 连帽衫、轿车 → 皮卡、招牌 → 灯笼。新物体描述 = 类型 + 1–2 个可核对的属性。配比：人身上的衣帽鞋包 45%、其他物体 35%、招牌 10%、车 10% | 目标 0.25%–5% 为主；贴回范围为目标外扩约边长的 50%（现在约 26%） | `s_covered` ≥ 0.85；看不到旧物的任何部分，新物不是叠在旧物上；新物逐项符合描述；穿戴物合身，尺度合理 | `old_absent`、`payload_checks`、`support_dilation` |
| 整体改属性 | D9 | 2,000 | 约 3,000 | 颜色约 90%，其余色调、材质、图案。多部件物体主体整块改，小部件（logo、拉链、反光条）可保留，这条写进审核问题 | 一半以上目标 ≥ 1%，覆盖到 5%–10% | `s_covered` ≥ 0.9；轮廓、褶皱、logo 不变（结构相似度 + 看图审核）；改动明显（目标内平均色差 ΔE ≥ 15） | `structure_sim`、`mean_delta_e` |

- 合计约 9,000 条，生成约 1.6 万次。50k 生成 8 万次用了约 20 小时，按此估算四机约半天到一天。
- 通过率是估计值，以冒烟批为准再调生成量。

**C. 外部数据**

| 数据集 | 决定 | 原因与条件 |
|---|---|---|
| ScenePair | 不用 | 1,280 对单词裁剪图，是文字编辑的评测集；每对是同图里两个不同的单词，不是整图编辑前后对 |
| Pico-Banana-400K 文字子集 | 合规确认后再处理（由用户确认） | 见下 |
| AnyWord-3M | 本轮不用 | 约 300 万张真实图，带 OCR 文字行标注，可以做"擦掉字作源图、原图作目标"的反向对；数据集许可没写清楚 |

Pico-Banana-400K 文字子集：
- **许可：**数据集 CC BY-NC-ND 4.0（非商业、不得发布修改版），源图来自 Open Images（CC BY 2.0）。
- **数据情况：**文字类约 1–2 万对，没有 mask；Nano-Banana 会把整图重画。
- **处理（确认可用后由 v3 侧做）：**只留局部写字编辑 → 对齐 → 由差异和定位模型 + SAM2 求区域 → 按支持区域贴回 → OCR 一致 → 改写成协议 2.0。
- **预计与限制：**预计留下 2,000–5,000 条，只用于训练，不对外发布。

**D. 进入训练的方式（v3 侧，交付后 1 天）**
- **收货抽查：**每个子集随机看 100 条，统计面积、`s_covered`、OCR 一致率、token 往返 IoU 的分布；不达标的批次退回。
- **编码：**用 256×2 codec 编码 token，往返 IoU < 0.5 的不用；补支持区域缓存。
- **写法比例同 C2：**盲指代 + token 70%，带名 + token 30%。
- **C3b 的行：**共 1,500 次更新 × 32 = 48,000 行。
  - 新数据约 25%，约 12,000 行：D7 按 1.5 倍过采样，D8、D9 各用一遍，再用新数据拼合约 1,500 行多目标；
  - 其余 75% 沿用 C3a 的配比，防止遗忘。

**E. 时间与分工**

| 日期 | 数据线（用户） | v3 侧 |
|---|---|---|
| 10-21 | 改管线：替换的贴回范围、OCR、`s_covered` 阈值、结构检查、近重复检索；每个子集先产 200 条冒烟 | A 的清洗与 C3a 准备；检查冒烟批的格式、token 和样例图 |
| 10-22～10-24 | 正式生产 | C3a 训练与评测；若 Pico-Banana 合规通过，处理其文字子集 |
| 10-25 | 交付 D7、D8、D9 | 转成训练行（D） |
| 10-26 | — | C3b 训练与评测（文档 03 §6） |

- **验收**（C3b 在决策子集上，见文档 03 §6）：写字 ≥ 0.55、替换 ≥ 0.80、大目标改属性 ≥ 0.90，保持不低于 0.87。
- **兜底：**写字补了数据仍明显落后，剩下的原因是文本编码器本身，再考虑双路条件。

### 2.12 CrispEdit、ScaleEdit：现有流程与统一打标管线（10-20 设计；10-22 第一批、10-24 第二批已发布）
> **10-22 更新：**
> - 新管线已实现（labeling 仓库分支 `extedit-protocol2-labeling`）；
> - 第一批 10 万个原始对已跑完，发布为 **D10 = `datasets/ExtEdit_Protocol2_20261021/`**：38,346 条，单目标 36,430、多目标 1,916，共 40,791 个单元（10-22 加自然度复查后）；
> - 按单元：改字 12,247、替换 12,396、其他改属性 10,986、删除 3,858、添加 1,304；
> - 第二批（扩到约 20 万个单元，见文档 07 §12）10-24 发布为 **D11 = `datasets/ExtEdit_Protocol2_20261022/`**：151,337 条，单目标 141,626、多目标 9,711，共 163,595 个单元。按单元：添加 46,733、替换 42,235、删除 42,430、改属性 32,135、改字 62；
> - D10、D11 合计 204,386 个单元：替换 26.7%、添加 23.5%、删除 22.6%、改属性 21.1%、改字 6.0%；
> - 详见文档 07 §10–§11 和该目录的 README。
> - 下面是 10-20 的流程核查与设计。

用户 10-20 要求：
- 把 v1 里来自 CrispEdit、ScaleEdit 的数据，从源数据到现在清洗后的数据，一步一步讲清楚；
- 把这个流程整理、合并成一条新管线，用更多的源数据打标出符合当前标准（协议 2.0）的数据。

**现有流程**（两个打标仓库 `/opt/tiger/tanyue/sam3-crispedit-labeling`、`/opt/tiger/tanyue/sam3-scaleedit-labeling` 同一套设计、代码大部分相同；之后的清洗与改写在 v2/v3 仓库）

| # | 阶段 | 做什么 | 模型 | CrispEdit | ScaleEdit |
|---|---|---|---|---:|---:|
| 0 | 下载 | HF 原始对。CrispEdit-2M（约 229 万对，CC BY 4.0）；ScaleEdit-12M 经 `QingyuShi/scaleedit-filtered-6m` 挑选（CC BY-NC-SA 4.0） | — | 529,020 | 299,633 |
| 1 | 质量预筛 | 看前后图和指令，6 个维度全 PASS 才留；全局类型（背景、风格等）直接丢 | Qwen3.8-27B | 460,518 | 265,456 |
| 2 | 场景难度筛 | 只看源图和指令：同类多实例、指令要从中选出一个才留 | 27B | **38,971** | **25,664** |
| 3 | 观察 + grounding | 列出实际编辑单元，给每个单元框（add 框在目标图） | 27B | 38,456 | 25,553 |
| 4 | SAM3 分割 | 框 + 指代短语分割；add 标在目标图的新物体上；文字用多边形 | SAM3 | 37,728 | 25,085 |
| 5 | 合并 | 进入四个子集的合并源对 `sources.jsonl` | — | 37,728 | 25,085 |
| 6 | 像素清洗 | 对齐、配准、贴回，区域外成块改动和没改的丢（§2.3） | CPU | 22,806 | 19,368 |
| 7 | 编辑核验 | 问题清单 + 盲描述两种问法（§2.3） | 27B | 12,444 | 13,519 |
| 8 | 类型筛 | 去 action 和无单一类型；text 按改字、删字、加字分流 | — | 12,400 | 12,528 |
| 9 | 协议 1.0 改写 | 填槽位、确定 S、SAM3 重分割、渲染句子、两票审核、修一次（§2.4） | 27B + SAM3 | 单目标 8,718，多目标 3,351 | 单目标 9,683，多目标 2,160 |
| 10 | 多目标召回 | 多目标行改写成多单元 case（§2.5，D4） | 27B + SAM3 | 2,200 对 | 1,778 对 |
| 11 | 协议 2.0 | 规则改写：盲指代、site（§2.4） | — | 8,718 | 9,683 |
| 12 | token 往返 ≥ 0.5 | v3 训练可用（§2.10） | SAMTok | **8,124** | **7,596** |

- **端到端：**CrispEdit 约 65 个下载对出 1 条单目标数据（1.5%），ScaleEdit 约 39 个（2.5%）。
- **算力：**
  - 打标约 925 GPU 小时（CrispEdit）和 1,030 GPU 小时（ScaleEdit），其中质量预筛占 70%–80%；
  - 之后的核验约 35 GPU 小时、协议改写约 80 GPU 小时，四个子集合计。
- **还没用的原始数据：**
  - CrispEdit 五类约 112 万对（4,372 个分片，约 2.4 TB）；
  - ScaleEdit 未下载的分片里至少约 88 万对可用候选：替换 34.7 万、添加 17.0 万、物体表面文字 12.2 万、删除 11.3 万、改色 7.6 万、建筑文字 4.7 万。
- **逐阶段细节**（命令、阈值、坏例、能否在新分片上重跑、按分片区间的产量）：`V3/analysis/ext_pipeline_20261020/crispedit_flow.md`、`scaleedit_flow.md`；下游各阶段按来源计数的脚本 `v1_funnel.py`。

**现有流程的问题**
1. **27B 每条要调用 5–6 次，SAM3 分割两次：**预筛、场景、观察、grounding、核验、改写审核。
2. **顺序不经济：**最贵的质量预筛放在最前面跑全量；不用 GPU 的像素清洗放在打标之后，才丢掉 CrispEdit 40%、ScaleEdit 23%。
3. **场景难度筛只留约 9%，而协议 2.0 下已经不需要它：**
   - 这道筛是为了让带名指令必须读 token 才分得清；
   - 盲指代指令里本来就没有物体名，token 是唯一线索；
   - 它还让 CrispEdit 偏向 "X located in the …" 这类模板句。
4. **mask 语义和现在的标准不一致：**add 标的是新物体本身，后面又按锚点重分割一次；token 往返在最后才算，ScaleEdit 有 22% 在这一步被丢掉。
5. **代码和格式分散：**分在四处，格式转了四次；四机脚本写死 4 节点和旧路径；CrispEdit 仓库本地有 1 个提交没推，远端还是旧路径。

**新管线：**完整计划写在[文档 07](07_SAMTokEdit_Qwen21_外部编辑数据打标管线.md)（设计，待确认）。内容包括：源数据怎么得到、新管线六步的输入、规则、输出和复用代码、产量与算力、校准与上线、待确认事项。要点：
- 六步依次是取数、像素预检、规划、分割 + token、区域把关 + 贴回、渲染审核导出；
- 27B 从 5–6 次降到 2 次，SAM3 只跑一次，像素预检排在最前；
- 场景难度筛改为标签；
- 直接产出与 D3、D4 同格式的协议 2.0 数据；
- 估计每 10 万个原始对出 2–3 万条，旧流程约 2,000 条。

### 2.13 合板：五批编辑数据合成一份（10-24，`datasets/SAMTokEdit_Merged_20261024/`）
用户 10-24 的要求：
- 盘点现有的可用数据，合成一批，供后续方法使用；
- 文字类加独立标记；
- 合板前统一做一次文字复查，单机运行，由用户提交。

字段、规则和用法的完整说明在该目录的 README，这里记要点。

**文字复查**
- **说明与入口：**labeling 分支的 `/opt/tiger/tanyue/sam3-extedit-protocol2-labeling/docs/EXTEDIT_TEXT_RECHECK.md`。
- **运行目录：**`experiments2/SAMTokEdit/data_pipeline/02_production_runs/merge_text_recheck_20261024/`，校准在其 `admin/calibration/`。
- **对象：**五批里与文字有关的单元，共 17,305 个：
  - 改字（改属性，属性为 text）13,786 个；
  - 写字候选（添加、替换的新物体描述里有引号或 text、reading 一类的词）3,519 个。
- **两项检查：**
  - 请求检查：只看指令，27B 两票，给出要写的字以及它是否乱码或拼错。ScaleEdit 的源指令常有 "Stadiiom" 这类拼错。
  - 读图复查：同 extedit 的文字复查，BEFORE / AFTER 特写，不告诉模型要写什么，由代码比对。
  - D10、D11 的改字打标时已读图复查过，只做请求检查。
- **运行：**ARNOLD 单机，10-24 10:45 到 11:32 UTC，没有报错。
- **校准（看图）：**
  - 27 条事先判过的单元里，5 条坏的全部拦下；
  - 按结果分组抽 32 条：判乱码的都对；读图不通过的约七到八成确有问题。
- **规则：**
  - 两票里至少一票取出了要写的字，才算写字单元，共 2,181 个；
  - 至少一票判乱码，或读图复查不通过，即不通过；
  - 记录里有一个文字单元不通过，整条移出默认视图。
- **结果：**
  - 改字 13,786 个，通过 13,400 个。D10 通过 99.2%，D3 通过 81%。
  - 写字 2,181 个，通过 1,768 个。D11 通过 82%；D1 只有 53%，原版模型自蒸馏，写字质量差。
  - 794 条记录移出默认视图。

**合板规则**
- **格式：**统一为 D3 / D4 的单元格式，单目标记录也是只有一个单元的 `units`。
  - 每个单元带 SAMTok token（span、解码区域、往返 IoU）和文字标记 `text`：
    - `role` 为 `change`（改字）、`render`（写字）、`remove`（删除文字类对象，按 scope 词表打标）或空；
    - 另有要写的字和复查结论。
  - 每条记录带源图 SHA-1（`group`）、编辑图漂移和局部损失标记、`split`，以及不进默认视图的原因 `exclude`。
- **默认训练视图**（`views/default_train.txt`）去掉的记录：
  - 验证集（C1–C3 的 dev）；
  - token 往返 < 0.5；
  - 做了一半的目标，同 C3 取行；
  - 文字复查不通过；
  - 源图与 benchmark 2.6.0 相同或近重复（dHash ≤ 6 位），或与验证集共用源图；
  - 同一个新物体描述超过 50 次；
  - D1 以外，同一张源图超过 3 条。
- **图片：**全部硬链接到原文件，不占额外空间。

**结果**

| 来源 | 全部记录 | 默认视图记录 | 默认视图单元 | 主要的去掉原因 |
|---|---:|---:|---:|---|
| D1 | 50,000 | 34,602 | 34,602 | 往返低 9,536、做了一半 4,108、验证集 2,804 |
| D3 | 39,615 | 33,658 | 33,658 | 往返低 4,116、同图超限 627、做了一半 529 |
| D4 | 4,066 | 3,578 | 8,690 | 往返低 385、做了一半 101 |
| D10 | 38,346 | 36,915 | 39,334 | 描述超限 712、做了一半 601 |
| D11 | 151,337 | 149,824 | 161,956 | 做了一半 670、描述超限 512、文字 321 |
| 合计 | 283,364 | 258,577 | 278,240 | |

- **默认视图按操作（单元）：**
  - 删除 72,211（26.0%）；
  - 替换 69,775（25.1%）；
  - 改属性 64,837（23.3%）；
  - 添加 58,849（21.2%）；
  - 改字 12,568（4.5%）。
- **多目标与文字：**
  - 多目标 15,082 条、34,745 个单元；
  - 写字 1,687 个单元，删文字类对象 1,845 个。
- **与 benchmark 的近重复：**D3 4 条、D4 1 条，D1 为 0。
- **校验：**
  - id 不重复；
  - 517,860 个图片文件都在，抽 5,000 个都是硬链接；
  - 单元索引、视图与记录一致；
  - 输入文件都早于构建。

**怎么用于训练**
- **训练行类型 `edit_merged`：**单目标、多目标记录都用它，训练时只加载行里用到的记录。
  - D1 记录可带近邻负例和 twins，同 C2。
  - 在 75 条记录上（每个来源 15 条）核对过：它读出的区域、token 和指令，与 C2 原来各类型（`edit`、`edit_v1`、`edit_ext`、多目标）逐像素一致。
- **取行：**`round2-data rows-merged --spec <配置>`。配置写明：
  - 按来源、操作、文字类别的抽取份额；
  - 高亮行和 D1 拼合多目标行的比例。
  - 一份 1,500 次更新的配置生成 48,000 行，约 2 分钟。
- **GPU 训练冒烟（10-25）：**本机 8 卡、3 次更新、96 行（`edit_merged` 84 行，覆盖五个来源、单目标与多目标、D1 的近邻负例和 twins），0 个样本跳过，损失正常。

**代码：**`src/samtok_edit21/preparation/merge_data.py`（`merge-data`，提交 `f7a86b60`），`rows-merged` 与 `edit_merged`（提交 `a542da97`）。

## 3. 评测数据

### 3.1 第二轮开发评测集（`V3/data/round2_20261016/eval_c1.json`，1,320 项）
全部在 60 张留出的开发图上，只给 token。C1、C1b、C2、原版和 E-A2 基线都用这一份。

| 子集 | 项数 | 构造 | 主要指标 |
|---|---:|---|---|
| single | 600 | 按面积分布抽 600 个区域；盲指代 400、带类别名 200 | 指认读回区域与 token 解码区域的 IoU |
| multi | 100 | 2/3/4 个 token 各约 1/3，各一种颜色 | 全部区域 IoU ≥ 0.5 的比例、串色 |
| pair | 400（200 对） | 同类实例 150 对（面积比 ≤ 3）+ 部件对宿主 50 对；同句只换 token | 两次都跟对自己 token 的比例 |
| edit | 100 | 开发图上的 50k 编辑，四类各 25；要求有盲指代、往返 ≥ 0.5、改动区域非空 | 执行率、强度、区域外改动、与参考目标的误差 |
| remove | 120（60 对） | 同类实例对做 `Remove the selected region <M>.` 换 token 探针 | 两次都删对自己区域的比例 |

### 3.2 benchmark v2，release 2.6.0（当前版本）
benchmark 在单独的仓库维护：`Tangent0308/samtok_edit_benchmark`，分支 `v2branch`，2.6.0 对应 commit 98f918d。说明以那里的 README.md、docs/BENCHMARK_OVERVIEW_2_6_0_ZH.md、docs/CONSTRUCTION_2_6_0_ZH.md 为准，下面只摘要点。数据根是 `datasets/samtok_edit_benchmark_v2/`。

> 本机的 `/opt/tiger/tanyue/samtok_edit_benchmark` 是旧的 dev 分支，不是 2.6.0。

- **规模：**1,084 张 SA-1B 源图，每张一个 case；共 3,872 个编辑单元，每图 2–5 个。
  - 2/3/4/5 个单元的 case 数：147 / 255 / 597 / 85。
  - 操作：replace 969、attribute 1,002、remove 921、add 980。add 全部是直接锚点。
- **交互形式：**7 种组合，基本均衡（约 553 个/种）：盲指代 + point/box/mask，物体名 + 无定位/box/mask/point。每图至少两种。
- **区域面积：**中位 0.47%，68% < 1%，最小 0.003%。
- **协议：**指令协议 2.0、mask 语义 2.0，no-ref 自动渲染为 `the selected region`。
- **输入方式：**
  - 源图 + 定位图（只画 point/box/mask）；
  - 源图 + 原生 point/box/mask；
  - 诊断用的灰度标签图。
  - 主评测为 `mixed`，每个单元保留原来的组合。
- **评分：**本地 Qwen3.8-Flash-Next-FP8，每个 case 一次联合判全部单元。
  - 每个单元打 E（编辑）、P（保持）、Q（质量）三项 0–4 分，另有全图的 P、Q；
  - 全部单元和全图都过线才算 JointPass。
- **状态：**构建、资产、几何、协议校验通过；独立人工终审未完成；正式全量模型 judge 结果尚未发布。
- **版本历史：**

  | 版本 | 日期 | 规模 | 主要变化 |
  |---|---|---|---|
  | 2.3.0 | 10-12 | 200 图 / 785 单元 | 引入指令协议 1.0 |
  | 2.4.0 | 10-13 | 同上 | 共享协议模块和写作风格，mask 语义 1.1；add 全部改为直接锚点 |
  | 2.5.0 | 10-15 | 1,084 图 / 3,872 单元 | 新增 884 张图（视觉审核 964 个候选） |
  | 2.6.0 | 10-16 | 1,084 图 / 3,872 单元 | 协议 2.0 盲指代：重渲染全部 no-ref，改写 1,552 条 ref，708 张新 mask，24 个单元改了操作类型；2.5.0 及以前的 no-ref 结果不可直接比较 |

- **更早的 v0 benchmark：**656 例，来自 CompBench/HumanEdit/MIRAGE，路径 `datasets/samtok_edit_benchmark/benchmark.jsonl`。只在 C1b 的纯文字诊断里用过（文档 03 §4.4），评分模型是 Qwen3.8-27B，与 2.6.0 不是同一套评分。

### 3.3 从 benchmark 2.6.0 拆出的单指令 case（10-19）
- **抽样：**固定种子打乱图片，把每张图的多条编辑拆成一条指令一个 case，每张图最多取 3 条。四类各 9 条，共 36 条，来自 13 张图。
- **核对：**benchmark 的 ref/no-ref 文本和 v3 的协议渲染逐字一致（3,872 条全部一致）。
- **case 清单：**`V3/analysis/bench_split_20261019/cases.json`。结果见文档 03 §5.5。

**benchmark 2.6.0 的 101 例子集（10-19，用于快速比对）**
- 从四个官方基线都判过分的 case 里，按每图单元数分层抽样（seed 20261019），去掉上面目测过的 13 张图。共 101 例、360 个单元。
- 区域面积中位 0.43%；带物体名的单元占 57%。官方基线在子集上的分数和全集相差 3 个百分点以内。
- 清单：`V3/analysis/bench_subset_20261019/case_ids.txt`。抽样脚本和统计：`V3/analysis/bench_subset_20261019/select_subset.py`、`V3/analysis/bench_subset_20261019/subset_stats.json`。
- 结果见文档 03 §5.6。

### 3.4 训练数据与评测数据的关系
- **来源：**v1 清洗数据来自 RefEdit/CrispEdit/ScaleEdit/GRES，与 SA-1B 的 benchmark 来源不同，用户决定不做泄漏检查。50k 和区域池与 benchmark 同属 SA-1B，已按源图 ID 和哈希排除，交集为 0。合板时（§2.13）对 D1、D3、D4 补做了近重复检索（dHash ≤ 6 位）：D1 为 0，D3 4 条、D4 1 条，已移出默认视图；D10、D11 打标取数时已查过。
- **分布差距：**
  - 区域大小：D1 中位 0.16%，D3 中位 4.1%，benchmark 中位 0.47%。
  - 多目标：benchmark 每图 2–5 条一起执行，C2 训练中多目标行约 13%（D4 + D5）。
  - 定位方式：benchmark 有 point/box/mask/无定位四种，本方法只覆盖 mask→token。用 point/box 时需要先由 SAM 一类模型得到 mask。

## 4. 路径总表

| 路径 | 内容 | 用在哪里 |
|---|---|---|
| `datasets/SAMTokEdit_Cleaned_20261009/README.md` | 清洗数据总说明 | — |
| `datasets/SAMTokEdit_Cleaned_20261009/edit_check_qwen38_27b_v1/` | MLLM 核验结果，含 46,482 对清单 | 协议适配的输入 |
| `datasets/SAMTokEdit_Cleaned_20261009/protocol_adapt_qwen38_27b_v1/` | 协议 1.0 单目标适配（39,615） | 2.0 适配的输入 |
| `datasets/SAMTokEdit_Cleaned_20261009/protocol_adapt_qwen38_27b_v1/protocol_v2_blind/` | **D3**，含 README、summary | C2；开发集 40 条 |
| `datasets/SAMTokEdit_Cleaned_20261009/protocol_multi_qwen38_27b_v1/` | 协议 1.0 多目标召回 | 2.0 适配的输入 |
| `datasets/SAMTokEdit_Cleaned_20261009/protocol_multi_qwen38_27b_v1/protocol_v2_blind/` | **D4**，含 README、summary | C2 |
| `datasets/ExtEdit_Protocol2_20261021/` | **D10**：CrispEdit、ScaleEdit 新数据按协议 2.0 打标的第一批（38,346 条，含 README、token 缓存） | C3-D、C3-MD 用了其中 12,000 行 |
| `datasets/ExtEdit_Protocol2_20261022/` | **D11**：第二批（151,337 条，含 README、token 缓存） | 合板 |
| `datasets/SAMTokEdit_Merged_20261024/` | **合板**（§2.13）：五批合成一份，含 README、`records.jsonl`、单元索引 `units.parquet`、默认视图、`build.json` | 以后的训练（`edit_merged`） |
| `experiments2/SAMTokEdit/data_pipeline/02_production_runs/merge_text_recheck_20261024/` | 合板前的统一文字复查：清单、记录、校准与判定 | 合板的文字标记 |
| `datasets/sa1b_samtok_edit_50k_20261014/release/` | **D1** 全部：说明、标注、源图 1,132 张、目标图 50,000 张、mask、支持区域、`edited` 和 `region_v2` mask | C1、C1b、C2；开发集 |
| `experiments2/SAMTokEdit/data_pipeline/02_production_runs/sa1b_samtok_edit_50k_20261014/sources/` | SA-1B dense 标注 | D2 区域池 |
| `datasets/SAMTok_Training_Data/mix_gres8k_ver4k_train.parquet` | GRES/VER | Derived 子集来源；E-A2 的 GRES 指认 |
| `experiments2/SAMTokEdit/data_pipeline/02_production_runs/sa1b_samtok_group_train_20k_20261016/README.md` | D6 构造说明（未交付） | 以后 |
| `datasets/samtok_edit_benchmark_v2/` | benchmark 2.6.0 数据根 | 最终测试；拆分 case |
| `datasets/samtok_edit_benchmark/benchmark.jsonl` | 旧 v0 benchmark（656 例） | C1b 纯文字诊断 |
| `experiments2/SAMTokEdit/qwen21_v2/data/train_v2_box_001/` | v2 训练清单（未清洗） | v2 模型；E-A1/E-A2 的单元来源 |
| `V3/data/stageA_ea1_20261011/`、`V3/data/stageA_ea2_20261012/` | 第一轮数据 | E-A1、E-A2 |
| `V3/data/round2_20261016/` | 第二轮全部数据，见 §2.10 | C1、C1b、C2 |
| `V3/analysis/bench_split_20261019/cases.json` | benchmark 拆分 36 例 | 10-19 对比 |

## 5. 勘误：和旧文档不一致的地方

旧文档已移到 `docs/archive/v3_process/`。

1. 旧文档 04（10-13）整体过时：它写协议 1.0 全量还没跑，benchmark 是 200 图版本，也没有 50k、D4、协议 2.0 和第二轮数据。以本文为准。
2. 旧文档 07 §6.1 预计 v1 数据要换 mask 的是 remove contents 123 条、带 site 的 add 363 条。实际只换了 22 条和 25 条，另 315 条 site add 锚点已贴合、不改。
3. 区域池规模：旧文档 08 先后写过约 7.3 万、约 8 万，实际训练可用 62,873。开发图上的 50k 编辑是 2,804 条，不是约 2,600 条。C1 快速评测 112 项，不是 160 项（160 是 E-A2 的）。
4. C1/C1b 的编辑行并不全是"盲指代 + token"：两轮都是 65% / 20% / 15%（盲指代 + token / 物体名 + token / 只有物体名）。C2 去掉了只有物体名的行。
5. "1.6 万张源图"是 50k 管线源池的容量，正式 50k 只覆盖 1,132 张。
6. 天然孪生的两个数字口径不同：3,610 组 / 103,157 对是组内两两组合，6,590 对是 C2 实际使用的最近配对。
7. 协议模块版本：协议 1.0 全量、多目标召回和 50k 生成用的是 benchmark 2.4.0 的模块（`ed8c5d7b…`）；v3 仓库里的 `instruction_protocol.py` 是 2.3.0 版（`e93d7d90…`），不再使用；2.0 适配和第二轮都用 `instruction_protocol_v2.py`（`fe192e8d…`）。
