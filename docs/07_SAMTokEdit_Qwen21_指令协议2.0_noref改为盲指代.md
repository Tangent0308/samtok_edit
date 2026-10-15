# 指令协议 2.0：no-ref 改为盲指代（定稿）

- **日期**：2026-10-15。
- **状态**：定稿。依据你在 2026-10-15 的三条决定（见第 1 节）。
- **读者**：负责改写指令的 agent，分两边：
  - 训练数据：v2 分支的协议适配，以及 SA-1B 数据管线 `sam3-crispedit` 分支 `sa1b-samtok-edit-data`；
  - benchmark：`samtok_edit_benchmark` 分支 v2branch。

  第 2–5 节是两边共用的规则，第 6 节是训练数据的改法，第 7 节是 benchmark 的改法，第 8 节是自检清单。
- **相对于什么改**：指令协议 1.0、mask 语义 1.1、写作风格 `benchmark-concise-2-direct-add-1`。ref 的写法不变；变的是 no-ref 句子，以及源 mask 的选取规则。

---

## 1. 决定

1. **no-ref 全部改成盲指代。** 训练数据和 benchmark 里所有 no-ref 指令都不再提被编辑对象的类别、部件、位置或任何特征，一律写成 `the selected region`。原来带类别名的 no-ref（`the selected sleeve`）**不再保留**。
2. **被编辑的部分就是定位区域。** 编辑的是某个物体的一部分时，"这一部分"的描述也算作指代：它和宿主描述一起并入 `the selected region`，源 mask 改为选中这一部分本身。例如 "the lower legs of the trousers" 整体变成 `the selected region`，mask 选中裤子的下半截。
3. **编辑结果里出现类别信息没有关系。** payload、属性值、结果说明是"要改成什么"，原样保留。例如 `Replace the selected region with a flared bell sleeve.` 里的 "sleeve" 不算泄露。

设定依据：方法的理想输入是"用户只给区域和编辑要求"，不会说出编辑对象的类别名。

---

## 2. 一句话规则

> 把指令拆成"对哪里做"和"做什么"两部分：**"对哪里做"整体换成 `the selected region`，源 mask 选中的就是这个"哪里"；"做什么"原样保留。**

| 属于"对哪里做"，一律并入 `the selected region` | 属于"做什么"，原样保留 |
|---|---|
| 物体类别：sleeve、chair、man | 动作：change、replace、remove、add、tint、make |
| 部件和宿主：the handle of …、the strap from …、the lower legs of … | payload：替换物、新增物 |
| add 的安装部位（site）：the upper back of … | 属性名和新值：the color … to navy blue |
| 位置、方位、序数：left、third from the front、near the door | add 的附着关系：on、attached to、clipped to … |
| 源物体的现状特征：the red coat、the wooden frame | 结果说明：to make knee-length shorts、including its contents |

---

## 3. 源 mask 怎么选（mask 语义 2.0）

**统一规则：源 mask = 实际被编辑的区域 T。** ref 和 no-ref 用同一个 mask；point、box 由这个 mask 生成，构造方式不变：紧致框，点取离边界最远的像素。

| 情形 | 1.0 的 mask | 2.0 的 mask |
|---|---|---|
| 编辑整个物体或部件，S = T | 该物体或部件 | 不变 |
| attribute / replace 带 `target_part`（改宿主的某个部件） | 宿主 | **部件本身** |
| remove `part`（从宿主上删一个部件） | 宿主 | **被删的部件** |
| remove `lowerlegs` / `hem`（剪短衣物） | 整件衣物 | **被剪掉的那一段** |
| remove `contents`（连同内容物一起删） | 容器 | **容器 ∪ 内容物** |
| remove `contents_only`（只删内容物） | 容器 | **内容物** |
| add 不带 site | 直接承载物（锚点） | 不变 |
| add 带 site（例如 "the handle of the suitcase"） | 宿主（锚点）+ 文字 site | **site 本身**，即附着部位，例如提手 |

**add 的说明**：
- add 的 mask 仍然是锚点，即源图里新物体所附着的承载物或部位，**不是新物体的轮廓**。
- 2.0 只是要求锚点精确到附着部位本身，与此前"add 的 S 取最细可命名粒度"的规则一致。
- site 若是一个区段而不是可命名的部件（例如 `the lower portion`、`the front face`），附着部位就是宿主 mask 上新物体实际落下的那一块。有目标图时按 6.2 的方法从目标图求出；没有目标图（benchmark）时人工框定。
- 参照式 add（`under`、`beside`、`near` 等）的 mask 仍是参照物，句式为 `Add <payload> <relation> the selected region.`；benchmark 当前没有这类单元。

**原 mask 不覆盖**：换成部件 mask 的单元，原来的宿主 mask 保留为历史字段（第 6.3、7.2 节）。

---

## 4. 各操作的 no-ref（盲指代）句式

`<R>` 代表 `the selected region`；训练数据里它后面紧跟 mask token：`the selected region ⟨M⟩`。

| 操作 | 句式 |
|---|---|
| attribute：color / material / finish / pattern / texture | `Change the <property> of <R> to <value>.`（texture 写作 `surface texture`） |
| attribute：text | `Change the text on <R> to "<text>".` |
| attribute：tint | `Tint <R> <value>.` |
| attribute：opacity | `Make <R> <value>.` |
| attribute：多个属性 | `Change the <p1> of <R> to <v1> and its <p2> to <v2>.` |
| replace | `Replace <R> with <payload>.` |
| remove：whole、part、contents_only | `Remove <R>.` |
| remove：contents | `Remove <R>, including its contents.` |
| remove：lowerlegs | `Remove <R> to make knee-length shorts.` |
| remove：hem | `Remove <R> to make it knee-length.` |
| add | `Add <payload> <relation> <R>.` |

- **qualifier**：分三种情况处理。
  - 用来区分目标的限定（方位、所属等），并入 `<R>`。
  - 只描述编辑要求、不提图中其他物体的，原样保留，例如 `, preserving the underlying surface`。
  - 提到了图中其他物体的保护说明，例如 `, preserving the plant growing in it and the pots on either side`，这些物体会帮着定位目标，所以整句去掉。保护邻近内容本来就是所有编辑的公共要求，去掉不改变编辑内容。
  - 统计：benchmark 2.5.0 没有 qualifier。50k 数据有 1,406 条，其中 971 条是 `preserving the underlying surface`。
- **payload 的写法**：去掉指回源物体的代词和引用，例如 `a matching one`、`another one like it`。用具体名词写出新物体，这一条与 1.0 相同。

### 4.1 改写示例（1.0 → 2.0）

带部件的 attribute、带部件的 remove、带 site 的 add 三行取自 benchmark 2.5.0 的原句；其余为按协议构造的示例。

| 操作 | ref（不变） | 1.0 no-ref | **2.0 no-ref** | 2.0 mask |
|---|---|---|---|---|
| attribute | `Change the color of the red coat of the woman on the right to navy blue.` | `Change the color of the selected coat to navy blue.` | `Change the color of the selected region to navy blue.` | 外套（不变） |
| attribute（部件） | `Change the color of the wooden frame of the third bar chair from the front to matte black.` | `Change the color of the wooden frame of the selected chair to matte black.` | `Change the color of the selected region to matte black.` | 木框（原为整把椅子） |
| replace | `Replace the outer sleeve of the woman on the left with a flared bell sleeve.` | `Replace the selected sleeve with a flared bell sleeve.` | `Replace the selected region with a flared bell sleeve.` | 衣袖（不变） |
| remove | `Remove the second person on the left.` | `Remove the selected person.` | `Remove the selected region.` | 人（不变） |
| remove（部件） | `Remove the white chest emblem from the foreground drummer's pink T-shirt.` | `Remove the white chest emblem from the selected T-shirt.` | `Remove the selected region.` | 胸前徽标（原为 T 恤） |
| remove（剪短） | `Remove the lower legs of the man's grey trousers to make knee-length shorts.` | `Remove the lower legs of the selected trousers to make knee-length shorts.` | `Remove the selected region to make knee-length shorts.` | 裤子下半截（原为整条裤子） |
| remove（含内容物） | `Remove the basket on the left, including its contents.` | `Remove the selected basket, including its contents.` | `Remove the selected region, including its contents.` | 篮子 ∪ 篮中物（原为篮子） |
| add | `Add a red knit beanie to the man on image-left.` | `Add a red knit beanie to the selected man.` | `Add a red knit beanie to the selected region.` | 人（锚点，不变） |
| add（带 site） | `Add one small round enamel pin on the upper back of the central girl's leather jacket.` | `Add one small round enamel pin on the upper back of the selected jacket.` | `Add one small round enamel pin on the selected region.` | 外套后背上部（原为整件外套） |

### 4.2 中文翻译（benchmark 的 `instruction_noref_zh`）

- "选中的 X" 一律写成 **"选中区域"**。
- 示例：
  - `将选中区域的颜色改为海军蓝。`
  - `将选中区域替换为喇叭袖。`
  - `移除选中区域。`
  - `移除选中区域，使其成为及膝短裤。`
  - `给选中区域加上一枚小圆形珐琅徽章。`

---

## 5. 规范模块（`instruction_protocol.py`）要改的地方

两个仓库共用这个模块，改完后按逐字节一致的方式同步，与 1.0 相同。

1. **`render(spec, has_ref=False)` 改为输出盲指代。**
   - 源指代统一为 `the selected region`。
   - 不再拼接 `target_part`、`site`。
   - remove 各模式按第 4 节的句式输出。
   - `render(spec, has_ref=True)` 不变。
2. **`instruction_spec` 的字段不变。** `ref`、`scope`、`target_part`、`placement.site` 仍然保留，ref 渲染要用它们；只是 no-ref 渲染不再使用。
3. **新增确定性检查**（no-ref 句子必须全部满足）：
   - 指代部分只有 `the selected region`；
   - 不含 `scope`、`target_part`、`site` 里的词，也不含 ref 短语里不属于 payload 和属性值的名词与形容词；
   - 不含方位词和序数词：left、right、top、bottom、upper、lower、front、back、behind、near、next to、middle、center、first、second、third、last 等。
   - 例外：payload、属性值和结果说明里出现这些词是允许的，例如 `to make knee-length shorts`、`a flared bell sleeve`。
4. **版本号**：`instruction_protocol_version` = `2.0`，`mask_semantics.version` = `2.0`，写作风格 = `benchmark-concise-3-blind-1`。2.0 的 no-ref 与 1.0 不兼容，所以升主版本号。
5. **`mask_semantics` 记录**：`source_role` 只取两个值。add 为 `anchor`，其余为 `target`，并且 target 指实际被编辑的区域。

---

## 6. 训练数据怎么改

### 6.1 清洗优化后的 v1 数据（v2 分支的协议适配结果）

- **输入**：`datasets/SAMTokEdit_Cleaned_20261009/protocol_adapt_qwen38_27b_v1/manifest_protocol.jsonl`，39,615 条接受的记录。
- **记录字段**：`instruction_spec`、`instruction_ref`、`instruction_noref`、`selection`（S）、`execution_region`、`edited_region`、`host_view` 等。
- **输出**：在同一目录下新建子目录，例如 `protocol_v2_blind/`，写新的 manifest；原文件不改。

需要做的：

1. **全部 39,615 条 no-ref 重新渲染**：用 2.0 的 `render(spec, False)`，不需要调用 MLLM。
2. **换 mask 的记录**（selection 改为实际被编辑的区域，原 selection 保留为 `selection_v1`）：
   - **remove `contents`（123 条）**：新 selection = 原 selection ∪ `execution_region`（容器加内容物）。
   - **add 带 site（363 条）**：先检查已裁到落点的 selection（`localize_anchor` 的结果）是否已落在 site 上。是就保留，否则按 6.2 第 3 步从目标图求附着部位。
   - **`target_part`**：协议适配已经把 S 换成了部件（`part_as_s`），manifest 里没有 `target_part` 非空的记录，不用处理。
3. **参照式 add（75 条）**：mask 仍是参照物，只重新渲染句子。
4. **`host_view`**（部件级 add 额外保留的"整个宿主 + site"视图）：2.0 里 no-ref 不能写 site，这个视图只能配 ref 句子使用。带 token 的 no-ref 行不用它。
5. **审核**：换了 mask 的记录重做"指代"和"轮廓"两项；其余沿用。
6. **目标图、`edited_region`、`execution_region`**：都不变。

### 6.2 SA-1B 50k 数据（`datasets/sa1b_samtok_edit_50k_20261014/release/`）

- **输入**：`annotations/train.jsonl`，共 50,000 条，四类各 12,500。字段说明见数据管线仓库的 DATASET_50K_ZH.md（本地副本 `datasets/sa1b_samtok_edit_50k_20261014/release/README.md`）。
- **输出**：在 `release/` 下新建 `annotations/train_protocol_v2.jsonl`，新 mask 放在 `masks/region_v2/`。原文件和原 mask 不改：媒体文件是硬链接，原地写会改到共享文件。

需要做的：

1. **全部 50,000 条 no-ref 重新渲染**：用 2.0 的 `render(spec, False)`。
2. **qualifier（1,406 条）**：按第 4 节的三种情况处理。保留的 qualifier 在 spec 里不变；去掉的记 `qualifier_dropped`。
3. **add 带 site（6,754 条）**：mask 现在是整个宿主（例如整栋房子），site 写在句子里（例如 `the flat roof`）。新 mask 取附着部位：
   - **首选，从目标图求**：源图与目标图的差异区域，在宿主 mask 内向外扩一圈，再与宿主 mask 取交集。这就是新物体实际附着的那块宿主表面。扩张半径取新物体短边的约 30%，至少 16 像素。
   - **兜底**：差异区域与宿主几乎不相交（例如新物体完全悬挂在宿主外侧），改用 SAM3 按 site 名称在宿主 mask 内分割。
   - **都失败**：保留原 mask，记 `blind_ineligible_reason = part_mask_missing`。这类记录只能作为 ref、不带 token 的样本。
   - `bbox`、`point` 按新 mask 重算，原值保留为 `bbox_v1`、`point_v1`。
4. **其余 43,246 条**：数据里没有 `target_part` 非空、没有特殊 remove 模式，mask 已经是实际被编辑的区域，只需重新渲染句子。
5. **抽查**：对换了 mask 的 add 按宿主类别分层，人工看 100 条，确认新 mask 落在新物体附着处、不是整个宿主，也不是新物体本身。

### 6.3 训练行怎么用

- **不再有带类别名的 no-ref 行。** 带 token 的行只有两种：
  - `no-ref + token`：`Change the color of the selected region ⟨M⟩ to navy blue.`
  - `ref + token`：ref 短语后接 token，写法与之前相同。
- 另外保留一部分 `ref、不带 token` 的行，保住纯文字指代能力。
- 建议比例：no-ref + token 65%、ref + token 20%、ref 不带 token 15%。
- 孪生组和多目标样本（文档 06）的指令一律用 2.0 的 no-ref。

---

## 7. benchmark 怎么改

以当前 release 2.5.0 为基础（1,084 张图、3,872 个目标），发布一个新 release（例如 2.6.0）。

### 7.1 改指令

- 3,872 个目标的 `instruction_noref` 和 `instruction_noref_zh`，全部按第 4 节重写，直接用 `render(spec, False)` 生成。
- `instruction_ref`、`instruction_ref_zh` 不变。
- 公开输入里的 no-ref 单元（no-ref + point / box / mask，共 1,660 个）使用新句子。

### 7.2 换 mask（299 个目标）

2.5.0 中源 mask 选的是宿主、需要换成实际被编辑部分的目标：

| 类型 | 数量 | 新 mask |
|---|---:|---|
| add 带 site | 210 | 附着部位本身 |
| remove `part` | 56 | 被删的部件 |
| remove `lowerlegs` | 18 | 被剪掉的裤腿下半截 |
| remove `contents` | 11 | 容器 ∪ 内容物 |
| remove `hem` | 3 | 被剪掉的下摆 |
| attribute 带 `target_part` | 1 | 被改的部件 |

- **做法**：先用 SAM3 按部件名（`target_part` 或 `site`）在宿主 mask 内分割，再逐个人工目视确认，保存为新文件，例如 `U3_v2.png`。
- **覆盖范围**：同一目标的所有交互形式都改用新 mask，ref + mask 的单元也一样；`box`、`point` 由新 mask 重新生成。
- **保留旧 mask**：原 mask 和原框、原点保留为 `target.mask_v1`、`box_v1`、`point_v1`，原文件不覆盖。
- **同步其他字段**：`target.geometry`、`statistics.json`、`asset_manifest.jsonl` 跟着更新。
- **分割不出来的目标**：记录在 `known_issues.json` 里，暂时保留旧 mask 并标注，不要静默保留。

### 7.3 其他字段

- `mask_semantics`：按第 5 节第 5 条更新。
- `completion_requirement`、`edit_contract`、`preserve` 写的是评测者看的私有说明，可以继续用完整的 ref 描述，不需要改成盲指代。但其中的范围描述要与新 mask 一致：例如原来写"只删衣袖下半截，mask 是整件衣服"，现在 mask 就是下半截。
- `instruction_length_advice`：按新句子重算。
- benchmark 的 `BENCHMARK_V2_ZH.md` 和协议文档 `datasets/samtok_edit_benchmark_v2/docs/INSTRUCTION_PROTOCOL_ZH.md`（以及仓库里对应的 `docs/` 文件）：第 5 节"Ref / No-ref"一段改为本文第 2–4 节的规则，删掉"No-ref 只把 ref 换成 the selected <scope>"的说法。

### 7.4 评测协议

- 主评测 mixed 不变，只是其中 no-ref 单元的句子换成新写法。
- 2.5.0 及以前的结果用的是旧 no-ref，不能与新 release 的 no-ref 单元直接比较，报告时要注明版本。

---

## 8. 自检清单（改写完成后逐条检查）

1. 每条 no-ref 句子都由 `render(spec, False)` 生成，并通过第 5 节的确定性检查。
2. 没有任何 no-ref 句子含有 `the selected <名词>`，`region` 除外。
3. 每个目标的源 mask 都是实际被编辑的区域（add 是附着部位）。抽查 S ≠ T 的目标：mask 与"要改的那一块"一致，没有残留整个宿主。
4. 换了 mask 的目标：box 和 point 已经重算，旧 mask 已保留。
5. ref 句子与 2.5.0 逐字相同。
6. 中文 no-ref 一律是"选中区域"。
7. 统计：benchmark 换 mask 的 299 个目标都有记录（换成功，或在 `known_issues.json` 里说明原因）；训练数据报告换 mask 成功和失败的条数。
8. 已知的语义问题不由本次改写处理，例如 `Change the color of the selected man to blue`（mask 是整个人，改的其实是衣服）。这类问题按 `known_issues.json` 的流程回到作者记录修正：修正后它属于 S ≠ T 的情形，mask 换成衣服本身。

---

## 附录：给各个 agent 的提示词

下面三段可以直接发给对应的 agent。每段都假设对方能读到本文，路径有两个，任选其一：
- 本机：`/opt/tiger/tanyue/samtok_edit_qwen-image-2.1-v3/docs/07_SAMTokEdit_Qwen21_指令协议2.0_noref改为盲指代.md`
- GitHub：`https://github.com/Tangent0308/samtok_edit/blob/qwen-image-2.1-v3/docs/07_SAMTokEdit_Qwen21_指令协议2.0_noref改为盲指代.md`

### A. 孪生与多目标数据构造（SA-1B 打标管线）

```text
no-ref 指令的标准改了，请在你当前的孪生组与多目标构造工作上做适配，不要推倒重来。
规范在 07_SAMTokEdit_Qwen21_指令协议2.0_noref改为盲指代.md（路径见下），先通读第 1–5 节和第 8 节。
需求文档 06_SAMTokEdit_Qwen21_孪生与多目标数据需求.md 的示例已按新标准更新，其余要求不变。

要点：
1. no-ref 不再写 "the selected <类别>"，一律写 "the selected region"。被编辑对象的类别、部件、安装部位、方位、序数都不能出现在 no-ref 里。payload、属性值、附着关系、结果说明（如 "to make knee-length shorts"）原样保留。
2. 源 mask 必须是实际被编辑的区域。add 仍然是锚点，但要精确到附着部位本身：原来"宿主 mask + site 文字"的写法，mask 要换成 site 那一块；有目标图时，用"源图与目标图的差异区域扩一圈，再与宿主 mask 取交集"求出（07 第 6.2 节第 3 步）。
3. 规范模块 instruction_protocol.py 的 no-ref 渲染按 07 第 5 节改，并和 benchmark 仓库保持逐字节一致；版本号升为 instruction_protocol 2.0、mask_semantics 2.0、写作风格 benchmark-concise-3-blind-1。
4. 孪生组：组内 no-ref 必须逐字相同。按新写法，同一操作、同一 payload 的两个成员自然相同，组级检查"句子相同"保留。暂时保留"同一 scope"的选择条件。
5. 多目标组：每个目标一句 no-ref，各自对应自己的 mask。同一组里出现几句完全相同的句子是允许的。
6. 已经生成的目标图不用重做：生成时用的是 ref 句子。只需重新渲染 no-ref，并按第 2 条补换 mask，原 mask 另存为 *_v1。
7. 规划和审核提示词里关于 no-ref 的说明改成 07 第 2 节的一句话规则；no-ref 的确定性检查按 07 第 5 节第 3 条。

完成后请汇报：改了哪些文件；已有样本中换 mask 成功和失败的条数；每类操作抽 5 条 ref 与新 no-ref 的对照；07 第 8 节自检清单的结果。
```

### B. benchmark（samtok_edit_benchmark，v2branch）

```text
no-ref 指令的标准改了，请在当前 release 2.5.0（1,084 张图、3,872 个目标）的基础上适配，发一个新 release（例如 2.6.0）。
规范在 07_SAMTokEdit_Qwen21_指令协议2.0_noref改为盲指代.md（路径见下），重点看第 1–5 节、第 7 节和第 8 节。

要点：
1. 全部 3,872 个目标的 instruction_noref 和 instruction_noref_zh 按 07 第 4 节重写：英文一律 "the selected region"，中文一律"选中区域"。直接用改过的 render(spec, False) 生成。instruction_ref 和 instruction_ref_zh 不变。
2. 源 mask 必须是实际被编辑的区域。2.5.0 里有 299 个目标的 mask 选的是宿主，需要换：add 带 site 210、remove part 56、remove lowerlegs 18、remove contents 11（新 mask = 容器 ∪ 内容物）、remove hem 3、attribute 带 target_part 1。
   做法：SAM3 按部件名或 site 名在宿主 mask 内分割，再逐个目视确认；site 是区段而非部件时人工框定。新 mask 另存新文件（如 U3_v2.png），所有交互形式都用新 mask，box 和 point 重新生成。原 mask、box、point 保留为 *_v1，不覆盖。做不出来的写进 known_issues.json，不要静默保留旧 mask。
3. 规范模块 instruction_protocol.py 按 07 第 5 节改，与训练侧仓库保持逐字节一致；版本号升为 instruction_protocol 2.0、mask_semantics 2.0、写作风格 benchmark-concise-3-blind-1。
4. mask_semantics、target.geometry、statistics.json、asset_manifest.jsonl、instruction_length_advice 同步更新。completion_requirement、edit_contract、preserve 可以继续用完整描述，但范围要与新 mask 一致。
5. BENCHMARK_V2_ZH.md 和 docs/INSTRUCTION_PROTOCOL_ZH.md 里 "No-ref 只把 ref 换成 the selected <scope>" 的说法改为 07 第 2–4 节的规则。
6. 主评测 mixed 的结构不变，只是 no-ref 单元换成新句子。报告和 release.json 里注明：2.5.0 及以前的 no-ref 结果不能与新版本直接比较。
7. known_issues.json 里已有的语义问题（如 "Change the color of the selected man to blue"）按原流程修正：修正后若改的是衣服，就按第 2 条把 mask 换成衣服。

完成后请汇报：换 mask 成功和失败的数目（按类型）；每类操作抽 5 条 ref 与新 no-ref 的对照；07 第 8 节自检清单的结果；新 release 的 manifest sha256。
```

### C. 已有训练数据的适配（清洗后的 v1 数据和 SA-1B 50k）

```text
no-ref 指令的标准改了，请对两份已有训练数据做适配。只写新文件，不改原文件。
规范在 07_SAMTokEdit_Qwen21_指令协议2.0_noref改为盲指代.md（路径见下），具体步骤在第 6.1 节（v1 数据）和第 6.2 节（50k 数据）。

1. v1 数据：datasets/SAMTokEdit_Cleaned_20261009/protocol_adapt_qwen38_27b_v1/manifest_protocol.jsonl，39,615 条，输出到同目录的 protocol_v2_blind/。
   - 全部重新渲染 no-ref。
   - remove contents 123 条：selection 改为容器 ∪ 内容物。
   - add 带 site 363 条：检查已裁过的 selection 是否落在 site 上，不在就从目标图求附着部位。
   - 参照式 add 75 条：只改句子。
   - host_view 只配 ref 句子使用。
2. 50k 数据：datasets/sa1b_samtok_edit_50k_20261014/release/annotations/train.jsonl，50,000 条，输出 annotations/train_protocol_v2.jsonl 和 masks/region_v2/。
   - 媒体是硬链接，绝不能原地写。
   - 全部重新渲染 no-ref。
   - qualifier 1,406 条按第 4 节的三种情况处理。
   - add 带 site 6,754 条：按第 6.2 节第 3 步从目标图求附着部位 mask，bbox 和 point 重算。
3. 目标图都不改，生成和成图审核结果沿用；换了 mask 的记录重做"指代"和"轮廓"两项审核。

完成后请汇报：两份数据各自的条数；换 mask 成功和失败的条数；qualifier 的处理分布；每类操作抽 5 条对照；人工抽查 100 条换了 mask 的 add 的结论；07 第 8 节自检清单的结果。
```
