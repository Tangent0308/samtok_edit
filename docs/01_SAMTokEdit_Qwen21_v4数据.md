# SAMTokEdit Qwen-Image-2.1 v4：数据

本文是 v4 的数据侧文档，盘点合板后的数据：
- 合板数据集的来源、格式、文字复查、默认训练视图和统计；
- 从合板数据生成训练行的方式，包括 v4 新增的 OPSD 行；
- 开发评测集、benchmark 及其子集，训练与评测数据的隔离；
- 路径总表和已知问题。

方法见[文档 02](02_SAMTokEdit_Qwen21_v4方法.md)，实验见[文档 03](03_SAMTokEdit_Qwen21_v4实验.md)。截至 2026-10-26。

- **v4 没有新产数据：**合板数据（10-24 构建）、SA-1B 区域池和 round-2 数据目录都是 v3 时期的产物。v4 的数据代码有三处改动：
  - 取行加了 OPSD 行（提交 `128b1f8a`，§3）；
  - 合板的 `half_done` 不再对改字套用覆盖率规则（`e53c4181`，§2.4）；
  - 重建区域池时读 dense 标注的路径改到现在的位置（`f7db6e4c`，§6）。
- **合板之前的数据史**（v1 清洗、协议适配、第一轮数据、C3 数据计划、外部数据打标管线的设计）见 v3 分支的 `/opt/tiger/tanyue/samtok_edit_qwen-image-2.1-v3/docs/01_SAMTokEdit_Qwen21_v3数据.md` 和 `/opt/tiger/tanyue/samtok_edit_qwen-image-2.1-v3/docs/07_SAMTokEdit_Qwen21_外部编辑数据打标管线.md`，本文只摘要。
- **路径记号**（`scripts/docs/check_paths.py` 逐一检查）：`datasets/…`、`experiments2/…` 指 `/mnt/bn/strategy-mllm-train/user/tanyue/` 下的同名路径；`V3/…` 指 `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v3/…`；`src/…`、`tests/…` 在本仓库（分支 `qwen-image-2.1-v4`）。
- **术语：**一条**记录**是一对源图和目标图；记录里的一个编辑叫一个**单元**，单目标记录只有一个单元。

## 0. 总览

| 数据 | 在 v4 里的作用 | 规模 | 路径 |
|---|---|---|---|
| 合板数据（D1、D3、D4、D10、D11） | 编辑训练行 `edit_merged`、`opsd_merged` 的唯一来源 | 283,364 条 / 303,965 单元；默认训练视图 259,078 条 / 278,748 单元（10-26 重建） | `datasets/SAMTokEdit_Merged_20261024/` |
| D2：SA-1B dense mask 区域池 | 高亮行、D1 的近邻负例、开发集 | 1,132 张图、72,857 个区域；1,072 张训练图上可用 62,873 个 | `V3/data/round2_20261016/pool/` |
| round-2 数据目录 | 取行的 `--out`、训练的 `--data`：开发划分、token 与支持区域缓存、开发集、C1–C3 的行 | 约 325 MB | `V3/data/round2_20261016/` |
| 开发集 | 训练中的快速评测、训练后的开发评测 | 1,320 项，全在 60 张留出图上 | `V3/data/round2_20261016/eval_c1.json` |
| benchmark v2，release 2.6.0 | 最终测试 | 1,084 张图 / 3,872 单元 | `datasets/samtok_edit_benchmark_v2/` |
| 决策子集 | C3 起的方案比较、OPSD 老师测试 | 301 例 / 1,075 单元 | `V3/analysis/c3_eval_20261023/case_ids.txt` |

- **规模：**默认视图 259,078 条记录、278,748 个单元，是全部记录的 91.4%。按单元，外部数据 D10、D11 占 72.4%，D1 占 12.4%，D3 占 12.1%，D4 占 3.1%。
- **操作基本均衡：**删除 25.9%、替换 25.0%、改属性 23.3%、添加 21.1%、改字 4.7%；改字的 93% 来自 D10。
- **多目标：**15,088 条记录、34,758 个单元（占单元的 12.5%），每条 2–5 个单元。
- **token：**每个单元都带 SAMTok token，默认视图里往返 IoU 全部 ≥ 0.5。D10、D11 打标时已卡掉往返低的；D1 有 19.1% 往返低，不进默认视图。
- **区域大小两极：**改动区域的中位数 D1 为 0.13%，其余来源 4%–9%（§2.5）。
- **文字：**改字 13,073、写字 1,688、删文字类对象 1,845 个单元；改字、写字单元在默认视图里全部通过了文字复查。
- **v4 的新用法：**`opsd_merged` 行只用源图、指令和各单元的区域，不读目标图（§3.1）。
- **10-26 重建：**改字不再按覆盖率判"只做了一半"，501 条记录回到默认视图，其余内容与 10-24 的版本逐字节相同（§2.4）。
- **训练：**OPSD 试训 10-26 起在跑（P1 与对照 P0，行文件见 §3.7，设置与结果见文档 03 §8）。

## 1. 指令协议与区域 token

训练数据和 benchmark 用同一套指令协议 2.0（用户 10-15 定稿）。共享模块 `src/samtok_edit21/preparation/instruction_protocol_v2.py` 与 benchmark 2.6.0 的协议模块逐字节相同（sha256 `fe192e8d…`）。

- **单元与 spec：**一个单元是一个源实例（或其部件）加一种操作。语义记录 `instruction_spec` 的槽位是 `operation`、`ref`、`scope`、`payload`、`target_part`、`remove_mode`、`attributes`、`placement`、`qualifier`；两种句子都由 `render` 从 spec 渲染，不手写。
- **四类操作：**
  - 添加：区域圈新物体直接附着的锚点或表面（裁到落点），不圈新物体本身；加字也归添加。
  - 删除：whole、part、contents、contents_only、lowerlegs、hem 六种模式；删字也归删除。
  - 替换：必须有结构或品类变化；只换颜色、材质、图案的归改属性。
  - 改属性：color、material、finish、pattern、texture、text、tint、opacity。改字是 `attribute.text`，合板数据里单列为操作 `text`。
  - 翻转、移动、重摆姿势不在协议内。
- **两种写法**（例子是一条 D11 替换）：

  | 写法 | 字段 | 句子 | 带 token 时 ⟨M⟩ 的位置 |
  |---|---|---|---|
  | 带名（ref） | `instruction_ref` | Replace the cardinal on the left with a blue jay. | 紧跟指代短语：…the cardinal on the left ⟨M⟩ with… |
  | 盲指代（noref） | `instruction_noref` | Replace the selected region with a blue jay. | 紧跟 the selected region |

  - 盲指代句不出现目标的类别、部件、宿主、site、方位、序数和现状特征；编辑结果的信息（payload、属性值）原样保留。
  - token 只绑定被编辑的区域，不绑定 payload。
  - 不能写成盲指代的单元 `instruction_noref` 为空（原因在 `noref_ineligible`），只用带名 + token。
- **token ⟨M⟩：**四个特殊 token，即 `<|mt_start|><|mt_XXXX|><|mt_YYYY|><|mt_end|>`；两级码本各 256 个码，第二级编号加 256。
  - SAMTok codec（`mask_tokenizer_256x2.pth`）把 S 编码成 token；token 再解码出的区域（`dec`）是 token 的实际含义，也是训练的监督目标。
  - 往返 IoU 是 `dec` 与 S 的 IoU。< 0.5 的 token 指不到要编辑的区域，不用于带 token 的训练。
- **三种区域**（mask 语义 2.0）：

  | 区域 | 字段 | 含义 |
  |---|---|---|
  | S | `selection` | 被编辑的源区域，定位信息总指向它。部件编辑只圈部件；删部件圈被删的部分；lowerlegs、hem 圈剪掉的那段；contents 圈容器 ∪ 内容物，contents_only 只圈内容物；添加圈附着处 |
  | 执行区域 | `execution_region` | 允许改动的范围。D1、D10、D11 是贴回用的支持区域：改属性只取目标，删除、替换取目标外扩一圈，添加取锚点 + 附着处；D3、D4 是 v1 清洗时的区域 |
  | 实际改动 | `edited_region` | 源图与目标图的 Lab ΔE 双阈值（6 / 12）连通差分，去掉远处和过小的块；另记 `area_frac`（占全图）和 `stats.s_covered`（S 被改动的比例） |

  - D1、D10、D11 的目标图按支持区域贴回，区域外与源图逐像素相同。
  - D3、D4 的目标图来自其他编辑模型，可能在区域外漂移（§2.2 的 `loss_region`）。
- **多单元指令：**一行总述，每个单元一行编号，各带自己的 token；单元顺序由训练行的 `order` 给出。

  ```text
  Execute all numbered instructions and keep everything else unchanged.
  U1: Replace the selected region <M1> with a blue jay.
  U2: Change the color of the selected region <M2> to light blue.
  ```

## 2. 合板数据

`datasets/SAMTokEdit_Merged_20261024/`：五批协议 2.0 编辑数据合成的一份训练数据（用户 10-24 要求），完整说明在该目录的 README。数据流：

```text
v1 四个打标子集 98,574 对 ──清洗、协议适配（10-09～10-18）──▶ D3 39,615 条、D4 4,066 对
SA-1B dense mask ──数据线 50k 管线（10-14～10-16）──▶ D1 50,000 条；同批 1,132 张图 ──▶ D2 区域池
CrispEdit-2M、ScaleEdit 新原始对 ──extedit 管线（10-21～10-24）──▶ D10 38,346 条、D11 151,337 条
D1、D3、D4、D10、D11 + 统一文字复查 ──merge-data（10-24）──▶ 合板 283,364 条 / 303,965 单元
```

### 2.1 五个来源与区域池

| 代号 | 内容 | 记录 / 单元 | 质检 | 原数据集 |
|---|---|---:|---|---|
| D1 | SA-1B 50k：原版 Qwen-Image-2.1 自蒸馏的编辑 | 50,000 / 50,000 | 确定性检查 + 27B 审核；没有人工验收 | `datasets/sa1b_samtok_edit_50k_20261014/release/` |
| D3 | v1 清洗后的单目标编辑 | 39,615 / 39,615 | 抽查 60 条约 86% 对 | `datasets/SAMTokEdit_Cleaned_20261009/protocol_adapt_qwen38_27b_v1/protocol_v2_blind/` |
| D4 | v1 清洗后的多目标对 | 4,066 / 9,964 | 抽查约 88% 对 | `datasets/SAMTokEdit_Cleaned_20261009/protocol_multi_qwen38_27b_v1/protocol_v2_blind/` |
| D10 | extedit 第一批（CrispEdit、ScaleEdit） | 38,346 / 40,791 | 分层抽 100 条，93 条可用 | `datasets/ExtEdit_Protocol2_20261021/` |
| D11 | extedit 第二批 | 151,337 / 163,595 | 分层抽 80 条，79 条可用 | `datasets/ExtEdit_Protocol2_20261022/` |
| D2 | D1 同批源图的 dense mask 区域池 | 1,132 张图、72,857 个区域 | 往返 ≥ 0.5 才用 | `V3/data/round2_20261016/pool/` |

**D1（SA-1B 50k，用户另一条数据线的新管线，10-14 启动、10-15 发布）**
- **流程：**Qwen3.8-Flash-Next-FP8 在 SA-1B dense mask 上逐个规划一种编辑 → 原版 Qwen-Image-2.1（40 步，seed 0）围绕支持区域裁图生成，提示里带区域位置 → 按支持区域羽化贴回（支持区域外与源图逐像素相同）→ 确定性检查 + Qwen3.8-27B 语义审核（操作对、目标做到、非目标保持、自然）。
- 生成时看得到区域和局部放大，训练时学生只看 token 和整图，相当于"特权自蒸馏"。
- **规模：**生成 79,821 条，自动通过 62.6%（删除最低，42.1%），按四类各 12,500 条补齐；只覆盖 1,132 张源图，平均每张 44 条。
- **协议 2.0 适配（10-15/16）：**盲指代 49,753 条，247 条没有；带 site 的添加有 3,606 条把 S 裁到附着处；补了 `edited_region`，168 条为空。说明见 `datasets/sa1b_samtok_edit_50k_20261014/release/README_protocol_v2.md`。

**D3、D4（v1 清洗与协议适配）**
- **链条：**v1 四个打标子集（RefEdit、CrispEdit、ScaleEdit、SAMTok Derived）98,574 对 → 像素清洗（对齐、配准、贴回）70,956 → Qwen3.8-27B 编辑核验 48,437 → 去掉 action 和无单一类型、文字按改字 / 删字 / 加字分流 47,402 → 协议 1.0 适配（27B + SAM3，两票审核）：单目标 39,615 条即 D3，判为多目标的 5,724 对另行改写、收下 4,066 对即 D4 → 协议 2.0 规则改写（不调用模型）。
- **按上游：**
  - D3：Derived 17,095、ScaleEdit 9,683、CrispEdit 8,718、RefEdit 4,119；
  - D4（对）：CrispEdit 2,200、ScaleEdit 1,778、Derived 60、RefEdit 28。
- **D4：**
  - 规则：一个单元一个物体，S 之间不共享、不嵌套，未解释的改动 ≤ 10%；
  - 2 / 3 / 4 / 5 单元的对 2,820 / 782 / 342 / 122；
  - 不少对的各单元带的是整对的改动区域（1,164 对所有单元的 `edited_region` 完全相同，其中 1,075 对等于整对的改动区域），训练时按单元裁到各自附近（`unit_change`）。
- **漂移：**目标图来自其他编辑模型，只有一部分做过贴回。token 缓存记了区域外的改动 `out_diff`、`out_share8`；任一超阈值（> 1.0 或 > 0.02）的记录标 `loss_region: local`，训练时只在编辑附近算损失。

**D10、D11（extedit 管线，v3 文档 07）**
- **代码：**labeling 仓库 `Tangent0308/sam3-crispedit` 的分支 `extedit-protocol2-labeling`，本地 `/opt/tiger/tanyue/sam3-extedit-protocol2-labeling/`，说明 `/opt/tiger/tanyue/sam3-extedit-protocol2-labeling/docs/EXTEDIT_PROTOCOL2.md`。
- **流程：**取数 → 像素预检（对齐、配准、改动图）→ Qwen3.8-27B 规划（一次调用给出判定、1–5 个单元的槽位和 S 的框）→ SAM3 分割 + SAMTok token（往返 < 0.5 的丢掉）→ 区域把关（含替换、改属性没做完和添加改坏锚点的规则）+ 按支持区域贴回 → 27B 两票审核 → 文字复查（改字）、自然度复查（CrispEdit 添加）→ 导出。
- **取数时排除：**旧流程用过的行、与 benchmark 2.6.0 源图相同或近重复的行、同一源图超过 3 条的行。
- **与旧流程比：**27B 从每条 5–6 次降到 2 次，SAM3 只跑一次，去掉了场景难度筛；收下率第一批 38.3%、第二批 27.7%，旧流程端到端约 1.5%–2.5%。
- **D10（10-21/22）：**
  - 100,166 个原始对 → 38,346 条（单目标 36,430、多目标 1,916）；
  - ScaleEdit 建筑表面文字、物体表面文字两类贡献了几乎全部改字（12,247 个单元）。
- **D11（10-22～10-24）：**
  - 545,713 个原始对 → 151,337 条（单目标 141,626、多目标 9,711）；
  - 用户要求扩到约 20 万个单元，向替换、添加、大目标改属性倾斜；不再加写字，改材质的源数据已用完；
  - 同一新物体描述连同第一批最多 50 次，去掉 6,117 条。
- **共同的质检：**
  - id 唯一，token 缓存与记录一一对应；
  - 协议 2.0，每个单元两票审核，往返 ≥ 0.5，区域外逐像素相同；
  - 自然度复查剔除"改动前"的图有抠图痕迹的 CrispEdit 添加：D10 565 条中 89 条，D11 43,802 条中 7,028 条。

**D2（SA-1B dense mask 区域池，10-16）**
- **来源：**D1 同批 1,132 张源图的 dense 标注，在 `experiments2/SAMTokEdit/data_pipeline/02_production_runs/sa1b_samtok_edit_50k_20261014/sources/`。
- **内容：**每张图一个文件 `pool/<SA-1B 图 id>.json`，保留面积 0.05%–30% 的 mask。每个 mask 记录：
  - SAMTok token、再解码的区域（监督目标）、往返 IoU、类别词；
  - 同图关系：同类、宿主 / 部件、接触、重复标注。
- **规模：**72,857 个区域；往返 ≥ 0.5 且解码非空的 66,247 个；去掉 60 张开发图后，1,072 张训练图上可用 62,873 个。
- **两份：**合板目录的 `pool/` 是这 1,132 个文件的硬链接。代码读的是 `--out` / `--data` 目录里的 `pool/`，即 round-2 目录。

**许可：**混合许可，整体只能按最严的条款用于非商业研究。
- SA-1B 图片（D1、D2）：研究用途；
- CrispEdit-2M：CC BY 4.0；
- ScaleEdit（经 `QingyuShi/scaleedit-filtered-6m` 挑选）：CC BY-NC-SA 4.0；
- D3、D4 的 RefEdit、SAMTok Derived 部分按各自上游。

### 2.2 目录与字段

| 文件 | 内容 |
|---|---|
| `records.jsonl` | 每条记录一行，3.7 GB。每行以 `{"id": "<记录 id>"` 开头，可以只按 id 取行，不必解析全部 |
| `units.parquet` | 每个单元一行的索引，303,965 行、16 MB，供筛选 |
| `views/default_train.txt`、`views/dev.txt` | 默认训练视图的 259,078 个记录 id；验证集的 2,844 个 |
| `stats.json` | 各来源、各操作的计数，文字标记、排除原因、局部损失记录数、硬链接数 |
| `build.json` | 命令、参数、起止时间、代码提交，23 个输入文件的大小与 SHA-1，区域池和 benchmark 源图的计数 |
| `images/D1/source/`、`images/D1/target/<id 前两位>/` | D1 的 1,132 张源图、50,000 张目标图 |
| `images/<来源>/<id 末两位>/<id>.source.<原格式>`、`<id>.target.png` | D3、D4、D10、D11 的源图和目标图 |
| `pool/`、`support_cache.jsonl` | D2 区域池；D1 的粗支持区域（32 px 格子），拼多目标行用 |

- **硬链接：**517,860 个图片文件、1,132 个池文件和 `support_cache.jsonl` 共 518,993 个，都是原文件的硬链接（同一文件系统，不占额外空间）。不要原地改写，否则原数据集里的同一文件也会变；删除本目录不影响原数据集。
- **路径：**记录里的图片路径都相对本目录。

**记录的字段**

| 字段 | 内容 |
|---|---|
| `id` | `<来源>:<原 id>`，如 `D11:crispedit-replace_01354-170`、`D1:0052ac9a85a2a6d097c72152` |
| `family`、`origin`、`source_ref` | 来源简称；上游（sa1b、crispedit、scaleedit、refedit、derived）；原数据集里的 id |
| `group` | 源图的 SHA-1（`sha1:…`），同一张源图的记录相同 |
| `source_image`、`target_image`、`size` | 图片路径；[宽, 高] |
| `units` | 编辑单元列表 |
| `drift` | token 缓存记下的目标图漂移：`out_diff`、`out_share8`、`in_diff`（D1 没有） |
| `loss_region` | `local`：训练时只在编辑附近算损失。默认视图里 9,869 条，全在 D3、D4 |
| `size_ok`、`split`、`exclude`、`default` | 图与 token 尺寸一致；`train` 或 `dev`；不进默认视图的原因；是否在默认视图 |
| 来源特有 | D1：`sa1b`（source_id、sample_id、object_id、bbox、operation）、`provenance`。D3、D4、D10、D11：原清单的 `type_before`、`instruction_before`。D10、D11 另有 `observed`、`treatment`、`provenance`（上游仓库、revision、分片、行号），CrispEdit 添加有 `naturalness_check` |

**单元的字段**

| 字段 | 内容 |
|---|---|
| `uid` | U1、U2… |
| `instruction_spec`、`instruction_ref`、`instruction_noref`、`noref_ineligible` | spec 和两种句子（§1） |
| `op` | add、remove、replace、attribute；改字单列为 text |
| `selection`、`execution_region`、`edited_region` | 三种区域（§1），COCO RLE：`rle_size` 为 [高, 宽]，`rle_counts` 为压缩串，`box_1000` 为 0–1000 归一化的框 |
| `token` | `span`：token 串；`dec`：解码区域（RLE，键为 `size`、`counts`）；`roundtrip`：`dec` 与 S 的 IoU |
| `half_done` | 是否"只做了一半"（§2.4） |
| `text` | 文字标记：`role`（change、render、remove 或空）、`requested`（要写的字）、`verdict`（pass、fail）、`reason`（§2.3） |
| 原样保留 | `mask_semantics`、`flags`、`audit`，以及来源有的 `comparable`、`text_check`、`text_payload`、`protocol_v2_change`、`instruction_protocol_version` |

- **`units.parquet` 的列：**record、uid、family、origin、multi、op、text_role、text_verdict、area（`edited_region.area_frac`）、roundtrip、half_done、noref（有无盲指代句）、loss_local、split、default、exclude（逗号分隔）、group。
- **直接读：**先用 `units.parquet` 选出记录 id，再按行首的 id 取行。训练器的 `Context.merged` 也是这样只解析用到的记录。

  ```python
  import json
  import pandas as pd

  root = "/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTokEdit_Merged_20261024"
  units = pd.read_parquet(f"{root}/units.parquet")
  ids = set(units[units["default"] & (units["family"] == "D11") & (units["op"] == "replace")]["record"])
  records = [json.loads(line) for line in open(f"{root}/records.jsonl") if line[8:line.index('"', 8)] in ids]
  ```

### 2.3 文字复查

- **起因：**用户 10-24 要求文字类单元加独立标记，合板前对所有文字相关的单元统一复查一次（ARNOLD 单机，用户提交）。
- **说明与代码：**`/opt/tiger/tanyue/sam3-extedit-protocol2-labeling/docs/EXTEDIT_TEXT_RECHECK.md`（labeling 分支，代码 `7e472f7`）。
- **运行目录：**`experiments2/SAMTokEdit/data_pipeline/02_production_runs/merge_text_recheck_20261024/`。
  - 清单 `units.jsonl`；
  - 运行写入 `node0/textrecheck/`（`records.jsonl`、`summary.json`）；
  - 校准与判定在 `admin/calibration/`（`verdicts.py` → `verdicts.jsonl`，合板读这份）。
- **运行：**10-24 10:45 提交、11:32 完成，第一次尝试就成功，17,305 个单元都有记录。

**清单**（17,305 个单元）

| 类别 | D1 | D3 | D4 | D10 | D11 | 合计 |
|---|---:|---:|---:|---:|---:|---:|
| 改字（`attribute.text`） | 35 | 1,392 | 50 | 12,247 | 62 | 13,786 |
| 写字候选：添加、替换的新物体描述里有引号或 text、words、reading 一类的词 | 1,196 | 215 | 19 | 171 | 1,918 | 3,519 |

**两项检查**
- **请求检查**（全部单元）：只看指令，27B 两票，取出要写的字，并判断它是否乱码或拼错（ScaleEdit 的源指令常有 "Stadiiom" 这类拼错）。两票都判乱码的 138 个，一票的 79 个。
- **读图复查**（D1、D3、D4 的改字 1,477 个 + 写字单元 2,181 个，共 3,658 个）：
  - 给模型改动前、后的特写，不告诉它要写什么，由代码比对读出的字；
  - 结果：通过 3,027，字对不上 542，字没改 61，旧字残留 25，回答无法解析 3；
  - D10、D11 的改字打标时已读图复查过，这次只做请求检查。

**规则**（`verdicts.py`）
- **写字单元（`render`）：**两票里至少一票取出了要写的字，共 2,181 个；其余 1,338 个候选不算文字类。
- **不通过（`fail`），改字或写字单元满足任一：**
  - 至少一票判乱码；
  - 读图复查不通过：字对不上、字没改、旧字残留、回答无法解析。
  - D10、D11 的改字只看乱码这一条。
- **整条剔除：**一条记录里只要有一个文字单元不通过，整条（含多目标）不进默认视图，共 794 条。
- **删除文字类对象（`remove`）：**删除单元的 scope 含 text、lettering、logo、label、number、name 等词即打标，不复查。删字和删其他物体做法相同，标记只为方便筛选。

**校准**（看图，`experiments2/SAMTokEdit/data_pipeline/02_production_runs/merge_text_recheck_20261024/admin/calibration/README.md`）
- **27 条事先判过的单元：**
  - 5 条坏的全部拦下；
  - 19 条好的里 6 条被误拒：2 条字形本来就有小毛病，4 条漏读字母或读到残影；
  - 3 条边缘的都通过。
- **按结果分组抽 32 条看图：**
  - 判乱码的 16 条都对；
  - 字对不上的 8 条里 6 条该拒、1 条误拒、1 条两可；
  - 判为不需要写字的 4 条、通过的 4 条都对。
- **估计：**读图不通过的单元里约七到八成确有问题。宁可错拒，也不让错字进训练。

**结果**

| 来源 | 改字：通过 / 总数 | 写字：通过 / 总数 | 移出默认视图的记录 |
|---|---:|---:|---:|
| D1 | 26 / 35（74%） | 45 / 85（53%） | 49 |
| D3 | 1,129 / 1,392（81%） | 130 / 152（86%） | 285 |
| D4 | 41 / 50（82%） | 12 / 17（71%） | 12 |
| D10 | 12,143 / 12,247（99.2%） | 137 / 160（86%） | 127 |
| D11 | 61 / 62 | 1,444 / 1,767（82%） | 321 |
| 合计 | 13,400 / 13,786（97.2%） | 1,768 / 2,181（81.1%） | 794 |

- D1 写字只有 53% 通过：原版模型自蒸馏，写字质量差。

### 2.4 默认训练视图

`views/default_train.txt` 是建议的训练范围：满足下表全部条件的记录。

| 条件（`exclude` 里的名字） | 规则 |
|---|---|
| 不在验证集（`split` 为 `train`） | 验证集即 C1–C3 的开发划分：D1 里 60 张留出源图上的全部 2,804 条，D3 的 40 条 |
| `no_token`、`roundtrip_low` | 每个单元都有 token，且往返 IoU ≥ 0.5，与 C1–C3 取行的门槛相同 |
| `size_mismatch` | 图与 token 尺寸一致（实际没有不一致的） |
| `half_done` | 没有"只做了一半"的单元：替换、改属性（不含改字，见下）的改动盖住 S 不到一半（`s_covered` < 0.5）；D1 的添加把锚点改动了 60% 以上。D1 以外的添加不套用锚点规则：v1 的添加选区有时就是整个承载面，D10、D11 打标时已有锚点把关 |
| `text_fail` | 文字复查都通过（§2.3） |
| `benchmark_overlap` | 源图与 benchmark 2.6.0 的 1,084 张源图不同（SHA-1），也不是近重复（dHash 相差 ≤ 6 位）。只查 D1、D3、D4，D10、D11 取数时已查 |
| `dev_overlap` | 源图（SHA-1）不与验证集的记录共用 |
| `payload_cap` | 同一个新物体描述（添加、替换的 payload，小写、去冠词）全体最多用 50 次。多目标记录先计入且不去掉，单目标记录按固定的伪随机顺序计入 |
| `per_image_cap` | D1 以外，同一张源图最多 3 条，按 D10、D11、D3、D4 的顺序保留。D1 的设计就是一张图做很多条，不受此限 |

两个上限只在其余条件都满足的记录里计数。

**不进默认视图的原因与条数**（记录；一条可有多个原因，910 条有两个以上）

| 原因 | D1 | D3 | D4 | D10 | D11 | 合计 |
|---|---:|---:|---:|---:|---:|---:|
| 验证集 | 2,804 | 40 | 0 | 0 | 0 | 2,844 |
| `roundtrip_low` | 9,536 | 4,116 | 385 | 0 | 0 | 14,037 |
| `half_done` | 4,079 | 442 | 97 | 157 | 660 | 5,435 |
| `payload_cap` | 462 | 505 | 0 | 712 | 512 | 2,191 |
| `text_fail` | 49 | 285 | 12 | 127 | 321 | 794 |
| `per_image_cap` | 0 | 627 | 8 | 0 | 10 | 645 |
| `dev_overlap` | 0 | 10 | 0 | 0 | 0 | 10 |
| `benchmark_overlap` | 0 | 4 | 1 | 0 | 0 | 5 |
| 不进默认视图 | 15,381 | 5,918 | 488 | 996 | 1,503 | 24,286 |
| 默认视图 | 34,619 | 33,697 | 3,578 | 37,350 | 149,834 | 259,078 |

- **D1 与 C2、C3 的衔接：**D1 训练记录里往返 ≥ 0.5 的正好是 C2 用的 38,225 条。`half_done` 的 D1 记录 4,079 条，其中 3,116 条在这 38,225 条里（替换 1,153、改属性 1,102、添加 861）。C3 去掉的是 4,108 条，多出的 29 条是改字：C3 对 50k 记录按原始 operation 判断，改字也套用了覆盖率规则。
- **改字曾被误算进 `half_done`，10-26 已修正并重建：**
  - **缺陷：**10-24 构建时，合板按 spec 的 `operation`（attribute）判断，改字也套用"改动盖住 S 不到一半"。改字只改笔画，覆盖率低不说明没做完：D10 抽看 4 条，打标时读回的字有 3 条与要求一致（如 MARKET → STATION）。C3 对 v1 行和 extedit 行都不对改字用这条规则。
  - **受影响：**574 条记录的 `half_done` 只来自改字单元（D10 444、D3 87、D1 29、D11 10、D4 4）。
  - **修正：**`merge_data.half_done` 改为按 `spec_op` 判断，改字不再套用（v4 提交 `e53c4181`）。
  - **重建（用户 10-26 同意）：**用修正后的代码完整重建，与 10-24 的版本逐条对比。
    - 283,364 条记录中 282,790 条逐字节相同；其余 574 条只差改字单元的 `half_done`（578 个单元由 True 变为 False）和记录的 `exclude` 里少了 `half_done`；
    - 501 条进入默认视图（D10 435、D3 39、D1 17、D11 10），没有记录被挤出，两个上限的计数不变；另有 4 条属于验证集，仍不进默认视图；
    - 10-24 的版本在 `datasets/_trash_20261026/`，等用户确认后删除。
- **被排除的记录仍可取用：**它们都在 `records.jsonl` 里，`exclude` 写明原因。例如只因 `roundtrip_low` 被排除的，可以用于不带 token 的训练形式。

### 2.5 统计

以下除特别说明外都是默认视图；由 `units.parquet` 重新算过（10-26），与 `stats.json` 和 README 一致。

**按来源**

| 来源 | 全部记录 | 全部单元 | 其中多目标记录 | 默认视图记录 | 默认视图单元 | 占默认单元 | 默认视图源图 |
|---|---:|---:|---:|---:|---:|---:|---:|
| D1 | 50,000 | 50,000 | 0 | 34,619 | 34,619 | 12.4% | 1,064 |
| D3 | 39,615 | 39,615 | 0 | 33,697 | 33,697 | 12.1% | 28,566 |
| D4 | 4,066 | 9,964 | 4,066 | 3,578 | 8,690 | 3.1% | 3,546 |
| D10 | 38,346 | 40,791 | 1,916 | 37,350 | 39,770 | 14.3% | 37,270 |
| D11 | 151,337 | 163,595 | 9,711 | 149,834 | 161,972 | 58.1% | 149,161 |
| 合计 | 283,364 | 303,965 | 15,693 | 259,078 | 278,748 | 100% | 218,892 |

源图合计按不同的图去重，有 708 张图出现在不止一个来源里。

**按操作（单元）**

| 操作 | D1 | D3 | D4 | D10 | D11 | 合计 | 占比 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 删除 | 9,680 | 11,901 | 4,348 | 3,858 | 42,424 | 72,211 | 25.9% |
| 替换 | 8,904 | 6,451 | 808 | 11,737 | 41,877 | 69,777 | 25.0% |
| 改属性（不含改字） | 8,884 | 10,408 | 3,367 | 10,805 | 31,373 | 64,837 | 23.3% |
| 添加 | 7,130 | 4,115 | 141 | 1,227 | 46,237 | 58,850 | 21.1% |
| 改字 | 21 | 822 | 26 | 12,143 | 61 | 13,073 | 4.7% |

**按上游（单元）**

| 上游 | 删除 | 替换 | 改属性 | 添加 | 改字 | 合计 | 占比 |
|---|---:|---:|---:|---:|---:|---:|---:|
| CrispEdit（D3、D4、D10、D11） | 24,277 | 30,567 | 18,568 | 38,323 | 51 | 111,786 | 40.1% |
| ScaleEdit（D3、D4、D10、D11） | 31,296 | 26,561 | 31,033 | 11,550 | 12,998 | 113,438 | 40.7% |
| SA-1B（D1） | 9,680 | 8,904 | 8,884 | 7,130 | 21 | 34,619 | 12.4% |
| SAMTok Derived（D3、D4） | 6,227 | 3,265 | 5,527 | 400 | 2 | 15,421 | 5.5% |
| RefEdit（D3、D4） | 731 | 480 | 825 | 1,447 | 1 | 3,484 | 1.2% |

**文字标记（单元）**

| 标记 | D1 | D3 | D4 | D10 | D11 | 合计 |
|---|---:|---:|---:|---:|---:|---:|
| `change`：改字 | 21 | 822 | 26 | 12,143 | 61 | 13,073 |
| `render`：写字（添加 1,389、替换 299） | 26 | 86 | 2 | 137 | 1,437 | 1,688 |
| `remove`：删文字类对象 | 184 | 42 | 15 | 140 | 1,464 | 1,845 |
| 无 | 34,388 | 32,747 | 8,647 | 27,350 | 159,010 | 262,142 |

**多目标记录（按单元数）**

| 单元数 | D4 | D10 | D11 | 合计 |
|---|---:|---:|---:|---:|
| 2 | 2,530 | 1,484 | 7,616 | 11,630 |
| 3 | 660 | 297 | 1,475 | 2,432 |
| 4 | 290 | 114 | 524 | 928 |
| 5 | 98 | 0 | 0 | 98 |
| 记录 / 单元 | 3,578 / 8,690 | 1,895 / 4,315 | 9,615 / 21,753 | 15,088 / 34,758 |

**改动区域的大小**（`edited_region.area_frac`，占全图）

| 来源 | 10% 分位 | 25% | 中位 | 75% | 90% | < 1% 的单元 |
|---|---:|---:|---:|---:|---:|---:|
| D1 | 0.015% | 0.042% | 0.13% | 0.56% | 2.4% | 82.1% |
| D3 | 0.48% | 1.7% | 5.0% | 10.8% | 19.3% | 17.1% |
| D4 | 0.70% | 1.7% | 4.2% | 9.9% | 17.6% | 15.1% |
| D10 | 0.95% | 1.6% | 4.8% | 14.6% | 26.8% | 11.4% |
| D11 | 1.3% | 3.4% | 8.7% | 19.0% | 30.7% | 7.2% |
| 全部 | 0.30% | 1.6% | 5.7% | 15.0% | 27.3% | 18.6% |

| 来源 | < 0.1% | 0.1–0.5% | 0.5–1% | 1–5% | 5–10% | 10–30% | ≥ 30% |
|---|---:|---:|---:|---:|---:|---:|---:|
| D1 | 43.9% | 29.7% | 8.6% | 12.4% | 2.7% | 2.1% | 0.7% |
| D3 | 2.1% | 8.2% | 6.8% | 32.6% | 22.8% | 24.2% | 3.3% |
| D4 | 0.4% | 5.8% | 8.9% | 41.0% | 19.4% | 21.9% | 2.7% |
| D10 | 0.1% | 1.8% | 9.5% | 39.4% | 15.2% | 26.5% | 7.5% |
| D11 | 0.2% | 2.8% | 4.2% | 27.2% | 20.1% | 34.8% | 10.7% |
| 全部 | 5.8% | 6.8% | 6.0% | 28.2% | 17.5% | 27.9% | 7.8% |

按操作的中位数：

| 来源 | 删除 | 替换 | 改属性 | 添加 | 改字 |
|---|---:|---:|---:|---:|---:|
| D1 | 0.17% | 0.23% | 0.17% | 0.023% | 0.035% |
| D3 | 4.9% | 7.0% | 5.1% | 4.1% | 1.4% |
| D4 | 3.0% | 6.0% | 6.0% | 2.4% | 1.9% |
| D10 | 5.1% | 11.8% | 9.7% | 6.5% | 1.4% |
| D11 | 5.4% | 12.2% | 8.7% | 9.3% | 2.2% |

- 这是实际改动的面积，不是 S 的面积；替换、添加的改动可以比 S 大或小。benchmark 统计的是 S 的面积（中位 0.47%，68.4% < 1%），两者只能粗比。`units.parquet` 里没有 S 的面积。
- 小区域主要来自 D1：< 0.1% 的单元 94% 来自 D1，< 0.5% 的 73%，< 1% 的 55%。

**其他**
- **token 往返 IoU 的中位数（10% 分位）：**D1 0.77（0.58）、D3 0.92（0.68）、D4 0.89（0.72）、D10 0.88（0.64）、D11 0.90（0.64）。
- **没有盲指代句的单元：**571 个（D3 251、D11 130、D1 113、D4 66、D10 11），只能带名 + token。
- **局部损失：**`loss_region: local` 的记录 9,869 条（D3 8,350、D4 1,519）。
- **每张图的记录数：**D1 中位 28 条、最多 131 条；其余来源受上限约束，每张图最多 3 条。

### 2.6 构建代码与校验

- **代码：**`src/samtok_edit21/preparation/merge_data.py`。命令 `samtok-edit merge-data --out <目录> --workers 48`，`--limit N` 只取每个来源的前 N 条做冒烟构建。
- **构建：**
  - 2026-10-24 12:21–13:24 UTC 在本机运行，v3 分支提交 `f7a86b60`；
  - `build.json` 由提交 `a542da97` 的代码事后补写，`note` 字段有说明：输入文件的大小与 SHA-1 为补写时所算，所有输入的修改时间都早于构建开始；
  - v4 的 `merge_data.py` 与之相比多了补写 `build.json` 的函数和注释改动。构建逻辑只有一处不同：`half_done` 不再对改字使用覆盖率规则（`e53c4181`，§2.4）。
- **10-26 重建：**
  - v4 分支，从提交 `64e6615c` 起跑，09:08–10:16 UTC 在本机运行，`--workers 48`；
  - `build.json` 记的提交是构建结束时的 HEAD `0856f593`，两者之间只改了取行代码，合板逻辑相同；
  - 23 个输入文件的大小与 SHA-1 与 10-24 构建时相同；518,993 个硬链接全部建成。
- **步骤：**
  1. 读五个来源：D1 的 S 和支持区域从 PNG 转成 RLE；token 来自 round-2 目录的 `edit_tokens.jsonl`、`v1_tokens.jsonl`、`d4_tokens.jsonl`，以及 D10、D11 自带的 `tokens.jsonl`、`multi_tokens.jsonl`；文字标记来自 `verdicts.jsonl`。
  2. 源图身份：D1、D3、D4 逐张算 SHA-1 和 dHash；D10、D11 的 SHA-1 取自各生产运行的 `ingest/rows.jsonl`；benchmark 2.6.0 的 1,084 张源图（`datasets/samtok_edit_benchmark_v2/assets/`）同样算一遍。
  3. 逐条记录算排除原因，再算 `dev_overlap`，最后算两个上限。
  4. 图片、区域池、支持区域缓存硬链接进新目录，记录里的路径改为相对路径。
  5. 写 `records.jsonl`、两个视图、`units.parquet`、`stats.json`、`build.json`。
- **构建后的校验（10-24）：**
  - 记录 id 不重复，每行以 id 开头；
  - 引用的 517,860 个图片文件都在，`images/` 下没有多余文件；抽 5,000 个，全部是硬链接；
  - `units.parquet` 的 303,965 行与记录里的单元一一对应；两个视图与 `default`、`split` 一致，两者不相交；
  - 抽 2,000 条记录，每个单元的 S 和 token 解码区域都与图同尺寸；
  - 所有输入文件的修改时间都早于构建开始。
- **训练读法的核对：**`edit_merged` 读出的各区域、token 和指令，与 C2 原来各类型（`edit`、`edit_v1`、`edit_ext`、多目标）的读法逐像素一致：每个来源 15 条，共 75 条。
- **单元测试：**
  - `tests/test_merge_data.py`：排除规则、两个上限、文字标记与 `half_done`、图片命名、`merged_regions` 的区域、取行的匹配；
  - `tests/test_opsd.py`：其中 `test_opsd_rows_fill_whole_step_pairs_on_every_rank` 检查 OPSD 行的整步对排布（§3.5）。

## 3. 训练行：从合板数据到训练

训练器按"行"取数据：一行是一个训练样本的描述，行文件的顺序决定哪张卡在哪一步处理哪一行。合板数据由 `samtok-edit round2-data rows-merged --spec <配置>` 取行（`build_rows_merged`，`src/samtok_edit21/preparation/round2_data.py`）。

### 3.1 行的种类

| 种类 | 来源 | 由配置的哪一项产生 | 训练时用到的数据 |
|---|---|---|---|
| `edit_merged` | 合板默认视图的任一记录，单目标或多目标 | `draws` | 源图、目标图；每个单元的 S、执行区域、实际改动、token（`span`、`dec`）和 spec；记录的 `loss_region`；D1 行另有近邻负例、孪生对方的 S |
| `opsd_merged`（v4 新增） | 同上 | 写了 `"opsd": true` 的 `draws` | 源图；指令；各单元的 S、执行区域、实际改动。不读目标图，没有负例和孪生 |
| `edit_multi` | D1 默认视图里同一张图的 2–4 条编辑拼成一条（v3 称 D5） | `joined_multi` | 各条的源图和目标图（联合目标 = 源图 + 各条的改动）、区域池负例 |
| `hl`、`hl_ref`、`hlm` | D2 区域池，不含 60 张开发图 | `highlights` | 源图、区域的 token 与解码区域、近邻负例；目标 = 源图在区域上叠 50% 的颜色 |
| `edit`、`edit_v1`、`edit_multi_v1`、`edit_ext`、`edit_multi_ext` | 合板前的读法，直接读 D1、D3、D4、D10 的原清单 | 只在 C1–C3 的旧行文件里 | 训练器仍支持；`rows-merged` 不再生成 |

- **`edit_merged` 的区域怎么用**（`merged_regions`，`src/samtok_edit21/training/round2.py`）：
  - 每个单元的损失区域取实际改动，没有时依次取执行区域、S；多目标记录的实际改动裁到该单元附近（外扩全图对角线的 1%）；
  - 写侧路由区域 = S ∪ 实际改动；读侧用 token 的解码区域；
  - `loss_region: local` 的记录只在 S ∪ 执行区域 ∪ 实际改动外扩 2 个 token 的范围内算流匹配损失。
- **`opsd_merged` 的数据需求**（`src/samtok_edit21/training/opsd.py`）：
  - 学生的提示按行的写法渲染；老师的提示把每个单元都写成带名 + token，负分支是同一句去掉 token；
  - 老师可改的范围 = 各单元 S ∪ 执行区域 ∪ 实际改动，在 token 格子上外扩、羽化；范围外老师的估计换成源图；
  - 不读目标图，所以 D3、D4 的目标图漂移、D1 的写字质量不直接影响 OPSD 行；但可改范围里的实际改动仍是由目标图算出来的。
  - 算法见文档 02。

### 3.2 取行配置

配置是一个 JSON：

| 字段 | 含义 | 默认 |
|---|---|---|
| `name` | 输出 `<name>.jsonl`、`<name>_stats.json` | 必填 |
| `seed` | 随机种子 | 20261025 |
| `updates`、`world`、`accumulation` | 总行数 = `updates` × `world` × `accumulation` | `updates` 必填；8、4 |
| `merged` | 合板目录 | `datasets/SAMTokEdit_Merged_20261024/` |
| `draws` | 依次从默认视图抽记录，见下 | 必填 |
| `forms` | 有盲指代句的单元的写法比例 | `{"noref_token": 0.65, "ref_token": 0.35}`，同 C2 |
| `highlights` | `{"single": 比例, "multi": 比例}`：单 token、多 token 高亮行占全部行的比例 | 0 |
| `joined_multi` | D1 拼合多目标行占全部行的比例 | 0 |
| `d1_negatives`、`d1_twins` | D1 的近邻负例、孪生 | 都开 |

**抽取（`draws` 的每一项）**
- **筛选**（`merged_match`）：
  - `family`：一个或几个来源；
  - `op`：add、remove、replace、attribute、text 只取该操作的单目标记录，multi 只取多目标记录，不写则不限；
  - `text_role`：change、render、remove，取至少有一个单元带该标记的记录。
- **数量三选一：**`rows`（行数）、`share`（占全部行的比例）、`rest: true`（剩下的全部，最多一项）。高亮行、拼合多目标行先按比例扣除，各项加起来必须正好填满总行数。
- **不放回：**各项依次从尚未被抽过的记录里随机取；要的比剩下的多时重复抽，`<name>_stats.json` 记下每项的 `available` 和 `repeated`。
- **`"opsd": true`：**这一项的行是 `opsd_merged`。
- **`"forms"`：**这一项自己的写法比例（v4 提交 `0856f593`），不写时用配置的 `forms`。OPSD 抽取项宜多用盲指代；写法名只能是 `noref_token`、`ref_token`，写错会报错。

**例子**（冒烟用，10-26；3 次更新、96 行）：

```json
{"name": "rows_opsd_smoke", "seed": 7, "updates": 3, "world": 8, "accumulation": 4,
 "draws": [{"family": "D11", "op": "replace", "rows": 24, "opsd": true},
           {"family": ["D10", "D11"], "op": "multi", "rows": 8, "opsd": true},
           {"family": "D1", "rows": 8, "opsd": true},
           {"family": "D11", "op": "add", "rows": 16},
           {"rest": true}],
 "highlights": {"single": 0.06, "multi": 0.02}, "joined_multi": 0.04}
```

- **产出：**`opsd_merged` 40 行（D11 31、D1 8、D10 1），`edit_merged` 44 行（D11 26、D1 8、D10 6、D3 4），`hl` 4、`hl_ref` 2、`hlm` 2、`edit_multi` 4 行。
- **可抽的记录**（10-24 版数据）：D11 单目标替换 36,934 条，D10、D11 多目标 11,504 条，D1 34,602 条，D11 单目标添加 43,491 条；没有重复抽。D1 行太少，没有配出孪生。
- **真实规模：**合板 README 里有一份 1,500 次更新的示例配置，生成 48,000 行，约 2 分钟，内存峰值约 8 GB（取行时把默认视图的全部记录读进内存）。

### 3.3 写法、D1 近邻负例与孪生

- **写法：**
  - 有盲指代句的单元按 `forms` 抽：默认盲指代 + token 65%、带名 + token 35%；没有盲指代句的单元（默认视图 571 个）一律带名 + token；
  - 多目标记录每个单元各抽一次写法，单元顺序随机（`order`）；
  - 合板取行没有不带 token 的写法。C1、C1b 的"只有物体名"（`ref_only`）只在旧行文件里。
- **D1 近邻负例：**D1 的单目标 `edit_merged` 行带 `negatives`。
  - 取区域池里 token 解码区域周围的标注区域：框间距 ≤ 对角线的 5%，最近的至多 6 个，加上宿主至多 2 个；
  - 不含被编辑物体本身（IoU ≥ 0.5）和它的部件（≥ 90% 在其内）；
  - 训练时这些区域必须保持不变。
- **D1 孪生：**
  - 在抽到的 D1 单目标行里，同一张图、盲指代句逐字相同的成组，组内按框距最近两两配对（`pair_key`、`pair_kind: twin`、`partners`）；
  - 两行都改成盲指代 + token，只有 token 能区分它们；对方的 S 并入负例；
  - 孪生只在抽到的行之间配，数量取决于抽了多少 D1 行。
- **OPSD 行**没有负例和孪生：老师的可改范围已经管住了保持。

### 3.4 高亮行与 D1 拼合多目标行

- **单 token 高亮（`hl`、`hl_ref`）：**
  - 按面积分桶从区域池抽区域：0.05–0.1% 5%、0.1–0.25% 20%、0.25–0.5% 20%、0.5–1% 20%、1–2% 15%、2–5% 12%、≥ 5% 8%；
  - 其中 30% 成反事实对：同句、同模板、同色，换成相邻的相近大小区域，或部件与宿主（占对的 30%）；
  - 27% 写成带类别词的 `hl_ref`；
  - 例：`Apply a red semi-transparent mask to the selected region ⟨M⟩ and keep everything else unchanged.`（4 个模板）。
- **多 token 高亮（`hlm`）：**2–5 个区域（35% / 30% / 20% / 15%），40% 围绕一对相邻区域构造；每个区域一种颜色（6 色），3 个模板。
- **D1 拼合多目标（`edit_multi`）：**
  - 同一张图上 2–4 条（40% / 35% / 25%）操作互不相同的 D1 默认视图记录，要求都有盲指代句；
  - 执行区域在 32 px 格子上外扩一格后互不相交；
  - D1 的目标在支持区域外与源图逐像素相同，所以联合目标 = 源图 + 各条的改动。

### 3.5 行在卡和步上的位置

- **行号规则：**第 u 次更新、第 s 个累积步、第 r 张卡处理全局第 (u × accumulation + s) × world + r 行；一次更新的批次由行文件决定，与卡数无关。
- **成对的行**（孪生、反事实高亮对）放在同一张卡的第 s、s + 1 步（s 为偶数），保证落在同一次更新里；其余单行两两填空位。
- **OPSD 行（v4 新增）：**
  - 一行 OPSD 的开销约为普通行的 4 倍（1 MP、16 步采样、2 个监督状态、老师引导 2.0 时约 29 秒、41 GB），而每次梯度平均都要等最慢的卡；
  - 所以 OPSD 行按"整步对"放：一个步对在所有卡上各放 2 行（8 卡共 16 行），这些步对均匀分布在整个训练里；
  - 凑不满一个步对的余数并入普通单行；
  - 没有 OPSD 行时，放置方式和随机数的用法与之前的行文件完全相同，旧行文件可以原样复现。
- **冒烟配置按现在的代码生成的结果：**40 行 OPSD 中 32 行占满 2 个步对（第 1 次更新的第 0–1 步、第 2 次更新的第 2–3 步），余下 8 行并入单行。
  - 用这份行在 8 卡上跑了 3 次更新（10-26，冻结老师）：0 个跳过，OPSD 平均损失 0.00508。
  - 这次 3 次更新用了 300 s，与旧排布相同。原因是冒烟规模下余数占 OPSD 行的五分之一，又集中在少数卡上，所以不代表实际训练的耗时。
- **建议：**OPSD 行数取 2 × world 的倍数（8 卡为 16 的倍数），就全部按整步对放；配置里的 `world`、`accumulation` 要与实际运行一致。

### 3.6 怎么用

```bash
# 取行：<目录> 里要有 dev_split.json、pool/、support_cache.jsonl（可以链接到 round-2 目录）
python -m samtok_edit21 round2-data rows-merged --out <目录> --spec <配置>.json
# 训练：--data 用 round-2 目录（edit_tokens.jsonl、pool/、开发集）；--rows 可以是绝对路径
python -m torch.distributed.run --nproc_per_node 8 -m samtok_edit21 round2-train \
    --data <round-2 目录> --rows <目录>/<name>.jsonl --output <run>/train ...
```

- `samtok-edit` 与 `python -m samtok_edit21` 是同一个命令行。
- `<name>_stats.json` 记下行的种类计数、每项抽取的 `available` / `repeated`、各来源的行数、孪生对数、带负例的 D1 行数和写法计数。
- 训练器只加载行里用到的合板记录（孪生的对方也加载）；有 `opsd_merged` 行时会建 OPSD 老师。训练参数见文档 02。

### 3.7 round-2 数据目录与已有的行文件

`V3/data/round2_20261016/`（约 325 MB，`samtok-edit round2-data` 生成）：

| 文件 | 大小 | 内容 |
|---|---:|---|
| `dev_split.json` | 2.5 KB | 留出的 60 张 SA-1B 源图和 40 条 v1 记录，它们的编辑和 dense mask 不进任何训练 |
| `pool/` | 138 MB | D2 区域池，1,132 个文件 |
| `edit_tokens.jsonl` | 44 MB | D1 50,000 条的 token、解码区域、往返 IoU，与 D1 清单同序 |
| `v1_tokens.jsonl`、`d4_tokens.jsonl` | 76 MB、13 MB | D3 39,615 条、D4 4,066 对的 token 缓存，另记目标图在区域外改了多少 |
| `support_cache.jsonl` | 8.1 MB | D1 50,000 条的执行区域（32 px 格子）和框，挑拼合多目标用 |
| `eval_c1.json` 等 | 288 KB | 开发集与快速评测子集（§4.1） |
| `eval_train_diag.json` | 154 KB | C1 的"训练图 vs 新图"诊断集，650 项 |
| `rows_*.jsonl`、`rows_*_stats.json`、`rows_c3*_dropped.json` | 共约 46 MB | C1–C3 的行（下表） |
| `scripts/` | 4 个脚本 | 实际生成 token 缓存和支持区域缓存的批处理脚本 |
| `README.md` | 4.6 KB | 目录说明 |

C1–C3 的行文件（每次更新 32 行 = 8 卡 × 累积 4 步）：

| 行文件 | 训练 | 行数 | 构成（行） |
|---|---|---:|---|
| `rows_c1.jsonl` | C1 | 64,000（2,000 次更新） | 单 token 高亮 25,600、带名 9,600、多 token 19,200；D1 编辑 9,600：盲指代 + token 6,256、带名 + token 1,944、只有物体名 1,400 |
| `rows_c1b.jsonl` | C1b | 25,600（800 次） | 高亮 10,240 / 3,840 / 7,680；D1 编辑 3,840；反事实对 2,112 对（相邻 1,712、部件与宿主 400） |
| `rows_c2.jsonl` | C2 | 112,704（3,522 次） | `edit` 46,690（其中孪生 6,590 对）、`edit_v1` 39,768、`edit_multi` 11,270、`edit_multi_v1` 3,681；高亮 11,295（`hl` 4,955、`hl_ref` 1,832、`hlm` 4,508） |
| `rows_c3m.jsonl` | C3-M | 48,000（1,500 次） | 从 C2 的行里按 C2 的比例抽，先去掉做了一半的目标（D1 4,108 条、D3 442 条）：`edit` 19,885、`edit_v1` 16,937、`edit_multi` 4,800、`edit_multi_v1` 1,568、高亮 4,810 |
| `rows_c3d.jsonl` | C3-D、C3-MD | 48,000（1,500 次） | 同上，但四分之一换成 D10：`edit_ext` 10,500（替换 5,000、改属性 4,000、改字 1,500）、`edit_multi_ext` 1,500；其余 `edit` 14,913、`edit_v1` 12,703、`edit_multi` 3,600、`edit_multi_v1` 1,176、高亮 3,608 |

- C3-D 从 D10 抽的替换要求 `s_covered` ≥ 0.85，改属性要求 `s_covered` ≥ 0.9 且改动 ≥ 1%；可抽的有替换 10,725、改属性 6,609、改字 12,228、多目标 1,916。
- **OPSD 试训的行**（10-26，由 10-26 重建的数据生成；配置、P0 的生成方式和核对见 `V3/analysis/opsd_pilot_20261026/README.md`）：
  - `rows_opsd_pilot_p1.jsonl`：16,000 行，`opsd_merged` 8,000（D11 4,550、D1 1,600、D10 1,289、D4 561），`edit_merged` 6,800，拼合多目标 600，高亮 600；500 个 OPSD 步对在 8 张卡上全部整齐，每次更新各卡的 OPSD 行数相同；OPSD 单元的盲指代占 0.849；
  - `rows_opsd_pilot_p0.jsonl`：P1 的行，OPSD 行改为同一条记录的 `edit_merged`（D1 行补近邻负例），位置不变。

## 4. 评测数据

### 4.1 开发集

`V3/data/round2_20261016/eval_c1.json`，1,320 项，全在 60 张留出的 SA-1B 开发图上，只给 token。C1 以来所有训练和基线的开发评测都用这一份。

| 子集 | 项数 | 构造 | 主要指标 |
|---|---:|---|---|
| single | 600 | 按面积分布抽区域；盲指代 400、带类别名 200 | 高亮读回的区域与 token 解码区域的 IoU |
| multi | 100 | 2 / 3 / 4 个 token 各约三分之一，各一种颜色 | 全部区域 IoU ≥ 0.5 的比例、串色 |
| pair | 400（200 对） | 同类实例 150 对 + 部件与宿主 50 对；同一句只换 token | 两次都跟对自己 token 的比例 |
| edit | 100 | 开发图上的 D1 编辑，四类各 25；要求有盲指代句、往返 ≥ 0.5、改动非空 | 执行率、强度、区域外改动、与参考目标的误差 |
| remove | 120（60 对） | 同类实例对做 `Remove the selected region ⟨M⟩.` 的换 token 探针 | 两次都删对自己区域的比例 |

- **快速评测子集**（训练中用）：
  - `V3/data/round2_20261016/eval_c1_quick_idx.json`：C1、C1b，112 项（single 64、multi 16、pair 32）；
  - `V3/data/round2_20261016/eval_c2_quick_idx.json`：C2 起，112 项（edit 40、remove 20、single 32、multi 8、pair 12）。
- **`eval_train_diag.json`：**650 项（single 400、pair 200、multi 50），C1 时用来比较训练图和新图。

### 4.2 benchmark v2，release 2.6.0

- **数据根：**`datasets/samtok_edit_benchmark_v2/`。
  - 清单 `datasets/samtok_edit_benchmark_v2/benchmark/cases.jsonl`（sha256 `818ce524…`）；
  - 版本与统计 `datasets/samtok_edit_benchmark_v2/benchmark/release.json`、`datasets/samtok_edit_benchmark_v2/benchmark/statistics.json`；
  - 源图与 mask 在 `assets/`。
- **规模：**1,084 张 SA-1B 源图，每张一个 case；3,872 个单元，每图 2–5 个（147 / 255 / 597 / 85 个 case）。
- **操作：**替换 969、改属性 1,002、删除 921、添加 980；添加全部是直接锚点。
- **交互形式：**7 种组合，各约 553 个单元：盲指代 + point / box / mask，带名 + 无定位 / box / mask / point。本方法只覆盖 mask → token。
- **区域：**S 面积中位 0.47%，68.4% < 1%，最小 0.003%。
- **协议：**指令协议 2.0、mask 语义 2.0。benchmark 的两种句子与本仓库协议模块的渲染逐字一致（3,872 个单元全部一致，10-19 核对）。
- **评分：**Qwen3.8-Flash-Next-FP8，每个 case 一次联合判全部单元；每个单元打 E（编辑）、P（保持）、Q（质量），另有整图的 P、Q；全部过线才算 JointPass。
- **状态：**独立人工终审和语义签字还没做（`release.json` 的 `independent_human_verified` 为 false）。
- **代码仓库：**`Tangent0308/samtok_edit_benchmark` 的 `v2branch`。本机 clone `/opt/tiger/tanyue/samtok_edit_benchmark/` 停在 dev 分支，本地的 `origin/v2branch` 也只到 2.3.0，不能当 2.6.0 用；2.6.0 以数据根为准。

### 4.3 benchmark 的子集

| 子集 | 规模 | 怎么抽的 | 用在哪里 | 清单 |
|---|---|---|---|---|
| 拆分的单指令 case（10-19） | 36 例，来自 13 张图，四类各 9 | 固定种子打乱图片，每图最多拆 3 条，每条一个单指令 case | C2 与原版的逐例对比 | `V3/analysis/bench_split_20261019/cases.json` |
| 101 例子集（10-19） | 101 例 / 360 单元 | 四个官方基线都判过分的 915 例里，按每图单元数分层抽（seed 20261019），去掉上面 13 张图 | C2 起的快速比对 | `V3/analysis/bench_subset_20261019/case_ids.txt` |
| 决策子集（10-23） | 301 例 / 1,075 单元 | 101 例，加上同一池里至少含一个 C2 输掉的那几类单元的 200 例（按每图单元数分层，seed 20261023） | C3 评测、A（C3-D + token 引导）、OPSD 老师测试 | `V3/analysis/c3_eval_20261023/case_ids.txt` |

- **101 例子集：**S 面积中位 0.43%，带名的单元占 56.4%（`V3/analysis/bench_subset_20261019/subset_stats.json`）。
- **决策子集**（`V3/analysis/c3_eval_20261023/subset_stats.json`，抽样脚本 `V3/analysis/c3_eval_20261023/select_decision.py`）：
  - "C2 输掉的几类"：面积 ≥ 0.25% 的替换、写字、面积 ≥ 1% 的改属性；
  - 每个 case 2 / 3 / 4 / 5 个单元的 41 / 71 / 165 / 24 例；
  - 单元操作：添加 268、删除 249、替换 278、改属性 280；
  - 切片：写字 68、替换 236、大目标改属性 139 个单元。
- OPSD 老师测试（`V3/analysis/opsd_teacher_20261025/`）用的就是这 301 例。

### 4.4 训练与评测数据的隔离

| 训练数据 | 与 benchmark 2.6.0 | 与开发集 |
|---|---|---|
| D1、D2 | 同属 SA-1B，但不同的图：生成时用冻结的排除清单，发布后按 source ID 和源文件 SHA-256 核对 1,084 张，交集为 0；合板时再查 SHA-1 和 dHash 近重复，也是 0 | 60 张开发图上的 2,804 条 D1 记录 `split` 为 `dev`；取行时区域池去掉这 60 张图 |
| D3、D4 | 上游不同（RefEdit、CrispEdit、ScaleEdit、GRES），原先按用户决定不查；合板时补查，D3 4 条、D4 1 条近重复，已移出默认视图 | 40 条 D3 记录 `split` 为 `dev`；与之共用源图的 10 条 D3 记录标 `dev_overlap` |
| D10、D11 | 取数时排除与 benchmark 源图相同或近重复的行：D10 22 行，D11 四机部分 181 行（本机部分没有计数文件） | 不共用源图（`dev_overlap` 为 0） |

- benchmark 的三个子集都由 benchmark 的 case 组成，上面的检查同样覆盖。
- **分布差距**（仍然存在）：
  - benchmark 每图 2–5 个单元一起执行；默认视图里来自多目标记录的单元只占 12.5%；
  - benchmark 的 S 68.4% < 1%，默认视图的改动区域只有 18.6% < 1%，其中 55% 来自 D1（口径不同，见 §2.5）；
  - benchmark 有 point、box、无定位三种其他输入，本方法只用 mask → token。

## 5. 路径总表

| 路径 | 内容 |
|---|---|
| `datasets/SAMTokEdit_Merged_20261024/` | 合板数据：README、`records.jsonl`、`units.parquet`、`views/`、`stats.json`、`build.json`、`images/`、`pool/`、`support_cache.jsonl` |
| `experiments2/SAMTokEdit/data_pipeline/02_production_runs/merge_text_recheck_20261024/` | 合板前的统一文字复查：清单、运行记录、校准与判定 |
| `datasets/sa1b_samtok_edit_50k_20261014/release/` | D1 原数据：说明、`annotations/train_protocol_v2.jsonl`、图片、mask |
| `datasets/SAMTokEdit_Cleaned_20261009/README.md` | v1 清洗数据总说明 |
| `datasets/SAMTokEdit_Cleaned_20261009/protocol_adapt_qwen38_27b_v1/protocol_v2_blind/` | D3 原数据，含 README |
| `datasets/SAMTokEdit_Cleaned_20261009/protocol_multi_qwen38_27b_v1/protocol_v2_blind/` | D4 原数据，含 README |
| `datasets/ExtEdit_Protocol2_20261021/` | D10 原数据：清单、token 缓存、拒收清单、README |
| `datasets/ExtEdit_Protocol2_20261022/` | D11 原数据 |
| `experiments2/SAMTokEdit/data_pipeline/02_production_runs/extedit_protocol2_100k_local8_20261021/` | D10 的生产运行 |
| `experiments2/SAMTokEdit/data_pipeline/02_production_runs/extedit_protocol2_batch2_local8_20261022/` | D11 的生产运行（本机部分） |
| `experiments2/SAMTokEdit/data_pipeline/02_production_runs/extedit_protocol2_batch2_4n_20261023/` | D11 的生产运行（四机部分） |
| `experiments2/SAMTokEdit/data_pipeline/02_production_runs/sa1b_samtok_edit_50k_20261014/sources/` | SA-1B dense 标注，D2 的来源 |
| `/opt/tiger/tanyue/sam3-extedit-protocol2-labeling/` | extedit 打标与文字复查的代码（分支 `extedit-protocol2-labeling`） |
| `V3/data/round2_20261016/` | round-2 数据目录（§3.7） |
| `V3/data/round2_20261016/eval_c1.json` | 开发集 |
| `datasets/samtok_edit_benchmark_v2/` | benchmark 2.6.0 数据根 |
| `V3/analysis/bench_split_20261019/cases.json` | 36 个拆分的单指令 case |
| `V3/analysis/bench_subset_20261019/case_ids.txt` | 101 例子集 |
| `V3/analysis/c3_eval_20261023/case_ids.txt` | 301 例决策子集 |
| `src/samtok_edit21/preparation/instruction_protocol_v2.py` | 指令协议 2.0 模块 |
| `src/samtok_edit21/preparation/merge_data.py` | 合板构建（`merge-data`） |
| `src/samtok_edit21/preparation/round2_data.py` | 区域池、开发集、C1–C3 与合板的取行（`round2-data`） |
| `src/samtok_edit21/training/round2.py` | `edit_merged` 等行的读法和损失区域 |
| `src/samtok_edit21/training/opsd.py` | `opsd_merged` 行的读法与 OPSD |

## 6. 已知问题与待办

- **合板数据的训练：**OPSD 试训 10-26 起在跑（§3.7、文档 03 §8）。之前只跑过 8 卡 GPU 冒烟：
  - 10-25：从 C2@3522 起 3 次更新、96 行，其中 `edit_merged` 84 行；
  - 10-26：§3.2 的配置，从 C3-D@1500 起，冻结老师 2 次更新后续训到第 3 次，EMA 老师 3 次更新；两次都是 0 个样本跳过；
  - 按整步对排布重新生成的同一配置的行，也在 8 卡上跑了 3 次更新（冻结老师），0 个跳过；
  - 正式训练的取行配比（来源、操作、文字类别、OPSD 行的份额）待定，见文档 03。
- **D1 小目标、单图集中：**
  - 改动区域中位 0.13%；默认视图只有 1,064 张源图，每张中位 28 条、最多 131 条，按图抽样或划分时要注意；
  - 往返低的 19.1% 不进默认视图；
  - 写字只有 53% 通过复查（原版模型自蒸馏），改字 74%。
- **D3、D4 的质量：**
  - 改字复查只通过 81%（D3）、82%（D4）；
  - 目标图来自其他编辑模型，默认视图里 9,869 条有区域外漂移，只能用局部损失；
  - 添加不套用锚点规则（选区有时是整个承载面），没有对应的"改坏附着物"过滤；
  - D4 有 1,164 对的各单元带同一个改动区域（其中 1,075 对就是整对的改动区域），这些单元的改动面积（`units.parquet` 的 area）偏大。训练时按单元裁到各自附近（`unit_change`），不受影响；`half_done` 也不受影响：S 在单元自己的区域里，裁不裁，S 被改动的比例都一样。
- **`half_done` 对改字过严：**已修正，10-26 重建后 501 条改字记录回到默认视图（§2.4）。
- **文字标记的局限：**读图不通过的约两三成是误拒；删除文字类对象只按 scope 词表打标，没有复查。
- **没有人工验收：**D1 只有自动审核；D3、D4、D10、D11 只有抽查。已知的零星问题：D10、D11 的替换偶尔在目标外多改一块，CrispEdit 添加少数仍有轻微抠图痕迹，写字偶有最后一个字母被物体边缘遮住而没被拦下。
- **S 面积没有索引：**`units.parquet` 只有改动面积；要和 benchmark 按 S 面积对齐比较，需要从 `records.jsonl` 算一遍。
- **重建区域池的路径（已修正）：**`round2-data pool` 原来读 `datasets/sa1b_samtok_edit_50k_20261014/sources/`，该目录已不存在。v4 改为读 `experiments2/…/sa1b_samtok_edit_50k_20261014/sources/`（`f7db6e4c`），1,132 张图的 dense 标注都在。
- **在产、未进合板的数据：**
  - v3 计划的数据线新产（D6 孪生与链式多目标，D7 写字、D8 结构替换、D9 整体改属性）都没有进合板；
  - 数据线 10-25 起在四机生产 SA-1B 孪生约 12,500 组和多目标约 10 万个：`experiments2/SAMTokEdit/data_pipeline/02_production_runs/sa1b_samtok_group_twins50k_multi100k_20261025/`，10-26 08:00 进度约 20%，预计还要约 2.5 天；
  - 它的排除名单写的是 benchmark-200 加评测预留 1,200 张，是否覆盖 benchmark 2.6.0 的全部 1,084 张源图还没核对；交付后要先按 SHA-1 和 dHash 查隔离，再决定是否合入。
- **benchmark 本机代码不是 2.6.0：**见 §4.2。
