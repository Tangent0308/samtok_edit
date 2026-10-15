# 指令协议 1.1 草案：不泄露目标信息的盲指代（blind）

- **日期**：2026-10-15。
- **状态**：草案。第 10 节的 7 个决定需要你确认，确认前不改任何数据和 benchmark。
- **适用**：训练数据（v2 分支的协议适配、SA-1B 数据管线）和 benchmark（`samtok_edit_benchmark` v2branch）。两边共用同一个规范模块，本文只描述规则，不涉及代码。
- **基线**：指令协议 1.0、mask 语义 1.1、写作风格 `benchmark-concise-2-direct-add-1`。benchmark 当前是 release 2.5.0（1,084 张图、3,872 个目标），本文的统计都按它计算。1.0 的 ref 和 no-ref 两种渲染保持不变，本文**新增**第三种渲染 blind。

---

## 1. 为什么要改

### 1.1 现在的 no-ref 仍然泄露目标信息

协议 1.0 的 no-ref 只把 ref 换成 `the selected <scope>`。scope 名词、部件名、安装部位都还留在句子里：

| 操作 | 1.0 no-ref（现状） | 泄露了什么 |
|---|---|---|
| attribute | `Change the color of the selected backpack to yellow.` | 目标是一个背包 |
| replace | `Replace the selected sleeve with a flared bell sleeve.` | 目标是袖子；payload 又重复了一次 "sleeve" |
| remove（part） | `Remove the strap from the selected bag.` | 宿主是包，删的是带子 |
| add | `Add a red tag clipped to the handle of the selected suitcase.` | 宿主是箱子，位置在提手 |
| remove（hem） | `Remove the lower section of the selected skirt to make it knee-length.` | 目标是裙子 |

图里同类物体只有一个时，模型只看 scope 名词就能找到目标，根本不用看定位信息（mask token、point、box）。这带来两个后果：
1. **训练**：模型可以走文字捷径，不学读 token。v2 的教训正是如此：靠文字或外部区域就能定位时，DiT 不读 token。
2. **评测**：no-ref 的分数混合了"读定位"和"认类别"两种能力，无法单独衡量定位输入有没有被用上。

### 1.2 blind 的目标

**指令只说"做什么"，不说"对哪个东西做"。** 目标是什么、在哪里、有什么特征，全部只能从定位输入（mask token / point / box / mask）得到。

---

## 2. 三种渲染的信息量

| 渲染 | 指代部分 | 定位输入 | 用途 |
|---|---|---|---|
| ref（1.0，不变） | 完整指代短语 `the red backpack of the girl on the left` | 可有可无 | 文字指代能力 |
| no-ref（1.0，不变） | `the selected <scope>` | 必须有 | 保留，兼容旧数据和旧结果 |
| **blind（1.1 新增）** | **`the selected region`**，所有操作、所有目标都一样 | **必须有** | 只靠定位输入找目标 |

三种渲染来自同一份 `instruction_spec`。blind 只是换了一个渲染函数，不新增任何语义字段。

---

## 3. blind 的规则

### 3.1 固定的中性指代词

- 所有目标一律写成 `the selected region`，不区分物体、部件、表面。
- 理由：
  - 写 `object` 会泄露"它是整体不是部件"；
  - 写 `part` 会泄露"它是部件"；
  - 一个固定的词最简单，也最容易做确定性检查。
- 训练数据里 mask token 紧跟在这个词后面：`the selected region ⟨M⟩`。benchmark 的定位输入照常单独给出。
- （第 10 节 D1：推荐 `region`；备选 `area`。）

### 3.2 定位区域必须就是被编辑的东西（S = T）

blind 句子里不能出现部件名，所以定位区域必须**直接选中被编辑的那一块**：

| 1.0 的情形 | blind 怎么处理 |
|---|---|
| S = T（mask 就是要改的东西） | 直接渲染 |
| `target_part` 非空（mask 是宿主，改的是其中一个部件） | 定位区域换成部件本身的 mask，`target_part` 置空；没有部件 mask 时，这个单元**不适用 blind** |
| add 带 `site`（例如 "the handle of"） | 定位区域换成那个部位本身，site 置空。协议适配已经把 add 的 S 裁到落点附近，大多数可以直接用 |
| remove 的 `part` 模式 | 同 `target_part`：换成部件 mask，模式改成 `whole` |
| remove 的 `contents` / `contents_only` | 可以保留："contents" 是删除范围，不是类别（见 3.3） |
| remove 的 `lowerlegs` / `hem` | 句子本身就说明了是衣物。推荐：定位区域换成要裁掉的那一段，改写成 `Remove the selected region.`；没有那一段的 mask 时不适用 blind（D5） |

### 3.3 各操作的标准句式

| 操作 | blind 句式 | 示例 |
|---|---|---|
| attribute（color / material / finish / pattern / texture） | `Change the <property> of the selected region to <value>.` | `Change the color of the selected region to bright yellow.` |
| attribute（text） | `Change the text on the selected region to "<value>".` | `Change the text on the selected region to "OPEN".` |
| attribute（tint / opacity） | `Tint the selected region <value>.` / `Make the selected region <value>.` | `Tint the selected region light blue.` |
| 多个属性 | `Change the <p1> of the selected region to <v1> and its <p2> to <v2>.` | `its` 指回 region，不泄露信息，可以保留 |
| replace | `Replace the selected region with <payload>.` | `Replace the selected region with a wooden bar chair.` |
| remove（whole） | `Remove the selected region.` | — |
| remove（contents） | `Remove the selected region, including its contents.` | — |
| remove（contents_only） | `Remove the contents from the selected region.` | — |
| add | `Add <payload> <relation> the selected region.` | `Add a red knit beanie on the selected region.` |

- add 的 relation 只能用 1.0 的直接附着关系：`on`、`in`、`attached to`、`clipped to`、`hanging from`、`resting on`、`mounted on`、`sewn onto`、`tied around` 等。不使用参照式 add（`under`、`beside`、`near` 等）：它的定位区域是参照物，不是新物体的落点，blind 下会含糊。
- `qualifier` 在 blind 中一律去掉。如果去掉之后语义变了，这个单元不适用 blind。

### 3.4 payload 和属性值：允许什么、禁止什么

payload 和属性值描述的是"改成什么"，这是编辑本身，必须保留。但不能借它们透露目标：

| 规则 | 反例 | 合格写法 |
|---|---|---|
| 不能引用目标的身份或现状 | `with a larger version of it`、`to match the other chair`、`another one` | 写出具体的新物体 |
| 不能引用图中其他物体来定位 | `with a lamp like the one on the table` | `with a brass desk lamp` |
| 不能出现方位词、序数词 | `to the color of the leftmost cup` | `to dark green` |
| 可以重复目标类别名，但要打标记（第 10 节 D4） | — | `Replace the selected region with a flared bell sleeve.` 保留原文，记 `payload_mentions_scope`。replace 要求新旧物体属于同一功能层级，新物体的名字常常就是旧物体的类别：benchmark 2.5.0 的 969 个 replace 里有 609 个（63%）如此。硬改写会让指令不自然、甚至改变要求，所以不改写，改为用盲猜检查（5.2）衡量实际泄露，并在报告中单列 |

### 3.5 多目标

- benchmark 的多目标 case 中，每个单元一行，单元编号把句子和它自己的定位输入绑定在一起：`U2 [blind; box]: Remove the selected region.`
- 几个单元可以是一模一样的句子，靠编号和定位输入区分。这是允许的，也正是要测的能力。
- 训练数据的多目标样本中，每个 `the selected region` 后面紧跟它自己的 token：`1. Remove the selected region ⟨M1⟩. 2. Change the color of the selected region ⟨M2⟩ to red.`

---

## 4. 适用性判定

每个单元记录 `blind_eligible`（是否适用 blind）。不适用时，用 `blind_ineligible_reason` 写明原因，可选值如下：

| 原因 | 说明 |
|---|---|
| `part_mask_missing` | `target_part`、remove `part`、add `site` 需要部件 mask，但没有 |
| `garment_section_missing` | `lowerlegs` / `hem` 没有要裁掉那一段的 mask |
| `reference_add` | 参照式 add |
| `qualifier_required` | 去掉 qualifier 后语义改变 |
| `payload_depends_on_target` | payload 违反 3.4 的前三条，且无法改写 |
| `ref_only` | benchmark 的 ref + 无区域单元：没有定位输入，不存在 blind 版本 |

---

## 5. 检查

### 5.1 确定性检查（必须全部通过）

1. 句子里的指代部分只能是 `the selected region`。
2. 句子里不出现以下词：scope 名词及其复数；ref 短语里的名词和形容词；`target_part`；`site`；1.0 里出现过的颜色、材质等源属性描述。payload 和属性值本身除外，按 3.4 处理。
3. 不出现方位词和序数词：left、right、top、bottom、upper、lower、front、back、behind、near、next to、first、second、last、middle 等（另附词表）。
4. 由同一份 spec 渲染：blind 句的操作、payload、属性值、relation 与 ref、no-ref 一致。

### 5.2 盲猜检查（MLLM，记录结果，不作为过滤）

- **做法**：只给 MLLM 看原图和 blind 句子，不给定位输入，让它指出最可能被编辑的东西，再看指对了没有。
- **意义**：衡量剩余的泄露程度。例如图里只有一个东西能戴帽子时，`Add a beanie on the selected region.` 仍然能猜中。
- **记录**：`blind_guessable` 和 MLLM 的选择。
- **benchmark 用法**：分"可猜中 / 不可猜中"两组报告。不可猜中的那组最干净地衡量"是否用了定位输入"。
- （第 10 节 D6：推荐只记录、不过滤；训练数据可以对可猜中的样本降权。）

### 5.3 人工抽查

- 按操作类型分层抽查，每类不少于 30 条。
- 检查两点：句子是否自然；仅凭定位输入时，编辑意图是否仍然完整。

---

## 6. 记录格式的变化

`instruction_spec` 字段**不变**。每个单元新增：

| 字段 | 说明 |
|---|---|
| `instruction_blind` | blind 句子；不适用时为空 |
| `blind_eligible`、`blind_ineligible_reason` | 第 4 节 |
| `blind_selection` | blind 使用的定位区域。S = T 时等于原 `selection`；换成部件 mask 时，给出新 mask 和它的来源（SAM3 重分割、人工标注等） |
| `blind_flags` | 例如 `payload_mentions_scope` |
| `blind_guessable`、`blind_guess` | 第 5.2 节的结果 |
| `instruction_protocol_version` | `1.1`。1.1 是 1.0 的超集：1.0 的 ref 和 no-ref 渲染逐字不变 |

---

## 7. 训练数据怎么更新

1. **每条已接受的记录渲染三种指令**：ref、no-ref、blind（适用时）。已有的 ref 和 no-ref 不需要重跑。
2. **补部件 mask**：对 `target_part` 非空、remove `part`、add 带 site 的记录，用 SAM3 按部件名重新分割出部件 mask，作为 `blind_selection`。协议适配流水线里已有 `part_as_s` 重分割，可以复用。分割失败的记为 `part_mask_missing`。
3. **训练采样比例**（D7，推荐）：blind + token 40%，no-ref + token 25%，ref + token 20%，ref 不带 token 15%。blind 是"必须读 token"的主信号；保留 no-ref 和 ref，避免模型丢掉利用文字的能力。
4. **孪生数据大幅简化**：blind 句子与目标类别无关，孪生组不再要求"同类实例"。同一张图上任意两个可编辑的目标，只要同一句 blind 指令对两者都成立，就能组成一组：
   - remove 几乎总能成立；
   - 改色只要新颜色与两者原色都不同就能成立。

   这会把可用的孪生组扩大很多。同类实例对仍然最难，建议保留一部分（D7 附带）。孪生需求文档（文档 06）要相应补一节，我在你确认后去改。
5. **SA-1B 数据管线**：规划时同时产出 blind 句子并做 5.1 的检查；盲猜检查可以并入现有的 27B 审核。
6. **审核**：blind 不改变目标图，现有的成图审核结果直接沿用。

---

## 8. benchmark 怎么更新

1. **版本**：在当前 2.5.0 之后发一个新 release（例如 2.6.0）。源图、mask、box、point 都不变，只新增每个单元的 blind 字段，以及缺失的部件 mask。
2. **需要补 mask 的单元**（2.5.0 统计，共 3,872 个目标；都另存为新文件，不覆盖原 mask）：
   - 带 `target_part` 的 57 个（remove 子部件 56、attribute 1）：mask 选中的是宿主，需要部件本身的 mask；
   - 带 `site` 的 add 210 个（共 980 个 add）：需要附着部位本身的 mask；
   - remove 的 `lowerlegs` 18 个、`hem` 3 个：需要被裁掉那一段的 mask（D5）；
   - remove 的 `contents` 11 个：不用补，直接渲染。

   标不了的记为 `part_mask_missing` 或 `garment_section_missing`。其余约 3,500 个目标可以直接渲染 blind。
3. **交互形式**：原来的 no-ref + point / box / mask 三种形式（2.5.0 中共 1,660 个目标：point 554、mask 553、box 553），增加对应的 blind + point / box / mask。
4. **主评测怎么用**（D2，推荐）：主评测 mixed 中，原来是 no-ref 的单元**改用 blind**（适用时），不适用的保留 no-ref 并标出。理由是 no-ref 本来的用意就是测"是否用上了定位输入"，blind 才真正做到这一点。2.4.0 的 mixed 作为 `mixed_v24` 保留一个版本，便于和旧结果对照。
5. **报告**：在原有分解（类型、交互形式、目标数）之外，增加两项：
   - blind 对 no-ref 的配对差：同一个单元两种写法的分数差；
   - blind 中可猜中与不可猜中两组的分数。
6. **基线**：原版模型的视觉定位协议（`visual_locator_v2`）同样换成 blind 句子，保证公平。

---

## 9. 兼容与版本

- 1.1 是 1.0 的超集。1.0 的 ref 和 no-ref 逐字不变，旧数据、旧评测结果全部有效。
- 规范模块里新增 blind 渲染、适用性判定和确定性检查。训练仓库和 benchmark 仓库按逐字节一致的方式同步，与 1.0 相同。
- 写作风格版本号建议升为 `benchmark-concise-3-blind-1`。

---

## 10. 需要你确定的

| 编号 | 问题 | 推荐 | 其他选项 |
|---|---|---|---|
| D1 | 中性指代词 | `the selected region` | `the selected area`；按操作分别用 object / region（会泄露粒度，不推荐） |
| D2 | benchmark 主评测里 no-ref 单元怎么办 | 改用 blind，旧版保留为 `mixed_v24` | 主评测不变，blind 只作消融（`all_blind`） |
| D3 | `target_part`、子部件 remove、add site 等 S ≠ T 的单元 | 训练侧用 SAM3 补部件 mask；benchmark 侧人工标注部件 mask。都补不了的不适用 blind | 直接排除，不补 mask（blind 覆盖率会明显下降，尤其是 benchmark 的部件任务） |
| D4 | payload 里重复了目标类别名（如 "a flared bell sleeve"；benchmark 的 replace 中占 63%） | 保留原文，打 `payload_mentions_scope` 标记，用盲猜检查衡量实际泄露，报告时单列 | 尝试改写（会不自然、可能改变要求）；或一律视为不适用 blind（replace 会损失六成以上） |
| D5 | `lowerlegs`、`hem` | 有要裁掉那一段的 mask 就改成 `Remove the selected region.`，否则不适用 blind | 一律不适用 blind |
| D6 | 盲猜检查的结果 | 只记录、分组报告；训练对可猜中的样本降权 | 训练和 benchmark 都过滤掉可猜中的样本 |
| D7 | 训练采样比例 | blind + token 40%、no-ref + token 25%、ref + token 20%、ref 不带 token 15%；孪生组放宽到不同类目标，同类对保留约三成 | 只用 blind + token 和 ref 两种 |
