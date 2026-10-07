# SAMTokEdit（Qwen-Image-2.1）v2 代码实现说明

本文说明分支 `qwen-image-2.1-v2` 的方法实现：数据协议、训练数据转换、两阶段训练、推理和评测分别在哪里实现、做了什么、为什么这样做。v1 的分析和 v2 的决策依据见 [v1 分析与 v2 计划](07_SAMTokEdit_Qwen21_v1分析与v2计划.md)；实验过程与数值见 [实验记录](02_SAMTokEdit_Qwen21_实验记录.md)；训练数据见 [训练数据盘点](04_SAMTokEdit_Qwen21_训练数据盘点.md)；四机启动见 [四机实验运行指南](03_SAMTokEdit_Qwen21_四机实验运行指南.md)。v1 的实现说明已存档在 [archive/v1](archive/v1/01_SAMTokEdit_Qwen21_代码实现说明.md)。

- 本地 checkout：`/opt/tiger/tanyue/samtok_edit_qwen-image-2.1-v2`，分支 `qwen-image-2.1-v2`（远程 `https://github.com/Tangent0308/samtok_edit.git`）。
- 基础：v1 分支 `qwen-image-2.1-dev` 的 commit `a93fe56`；v1 分支本身未改动。
- 文中代码链接指向本分支源码；行号以撰写时的 commit 为准。

## 1. 总览

### 1.1 v2 相对 v1 改了什么

| 部分 | v1 | v2 | 计划编号 |
|---|---|---|---|
| add 的区域表示 | SAMTok mask span ⟨M⟩（codec 往返 IoU 仅 0.38） | Qwen3-VL 框 ⟨B⟩ = `<\|box_start\|>[x1, y1, x2, y2]<\|box_end\|>`，0–1000 相对源图坐标 | 8.2 |
| 其他类型 | ⟨M⟩ | ⟨M⟩（不变） | 8.2 |
| Derived 的 add | add + 宿主实例 mask | 归入 attribute，保留 ⟨M⟩ | D3 |
| composite | 训练 | 本轮剔除 | 8.1 |
| Stage 1 | TE LoRA：NTP×0.05 + 经冻结 DiT 的 FM（含区域加权 C） | 定位 LoRA：纯 NTP（edit_ntp + 1/8 rec_ntp 框 grounding 回放），不加载 DiT/VAE | 方案 A，D6 |
| pass-2 编码 / 缓存 | Stage-1 TE | raw SAMTok TE（推理时关闭 adapter，与 raw 逐位一致） | 方案 A |
| Stage 2 loss | FM + 区域加权 C + attention loss A | 官方 FM；可选结构性绑定 `--binding` | 8.5 |
| 推理 | 无区域保持 | 可选 latent 融合（区域外按当前 σ 注入加噪源图） | 8.6 |
| 评测 | 旧编译器丢属性名词；含 mixed | 与训练同一 noref 转换器；只评原子编辑；MIRAGE 拆分；dev/test 划分 | 8.7，D2/D4/D5/D8/D9 |

所有新增功能关闭时（`--binding none`、不融合、v1 adapter + pass-2 用 Stage-1 TE），v2 代码对 v1 权重的推理输出与 v1 代码**逐位一致**（见[实验记录第 4 节](02_SAMTokEdit_Qwen21_实验记录.md#4-gpu-等价性检查)）。

### 1.2 数据流

~~~text
纯文本：源图 + 指令
  pass 1（TE + 定位 LoRA）→ [{"mask_2d": ⟨M⟩ | "bbox_2d": [..], "label": ...}]
  → 绑定原指令短语，插入 ⟨M⟩/⟨B⟩（ref prompt）
  pass 2（同一 TE，关闭 LoRA = raw SAMTok TE）→ 官方编辑模板编码
  → Qwen-Image-2.1 DiT（+ 可选区域绑定、latent 融合）→ RGBA 编辑图

交互：源图 + 用户区域（mask/框/点）
  → ⟨M⟩（codec 编码，或 SAM2 候选再编码）/ ⟨B⟩（框）插入 noref prompt → 从 pass 2 开始

训练：
  Stage 1：TE LoRA，NTP（edit_ntp: 编辑指令 → 区域 JSON；rec_ntp: grounding 请求 → bbox_2d）
  缓存：raw TE 编码 edit / edit_umt 行 + 源/目标 VAE latent + 区域绑定 payload
  Stage 2：DiT LoRA，官方 FM（+ 可选 --binding）
~~~

Stage 2 不再依赖 Stage 1：条件只用 raw TE，所以 Stage 1 与 Stage 2 可以独立运行。Stage 2 默认在训练时即时计算条件、不建缓存（第 5.1 节），所有消融臂用同一份 metadata；缓存路径仍然保留。

### 1.3 目录与模块

| 路径 | 内容 |
|---|---|
| [data/protocol.py](../src/samtok_edit21/data/protocol.py) | 区域 token（⟨M⟩/⟨B⟩）、定位 JSON、短语绑定、ref/noref 渲染、metadata 行校验 |
| [data/io.py](../src/samtok_edit21/data/io.py) | metadata 读写、两阶段配比与调度 |
| [data/provenance.py](../src/samtok_edit21/data/provenance.py) | 缓存格式 v3、条件身份、pass-2 TE 选择 |
| [preparation/v2_data.py](../src/samtok_edit21/preparation/v2_data.py) | v1 → v2 训练数据转换 |
| [preparation/semantic.py](../src/samtok_edit21/preparation/semantic.py) | 9B noref 转换器（训练数据与评测共用） |
| [models/pipeline.py](../src/samtok_edit21/models/pipeline.py) | 模型装配、pass 1/2、adapter 开关、推理 `edit()` |
| [models/binding.py](../src/samtok_edit21/models/binding.py) | 区域绑定：token 布局、区域图、四种模式、融合区域 |
| [models/codec.py](../src/samtok_edit21/models/codec.py) | 发布的 VQ-SAM2 codec 包装（未改动） |
| [training/objectives.py](../src/samtok_edit21/training/objectives.py) | adapter、缓存条件构造、FM loss |
| [training/engine.py](../src/samtok_edit21/training/engine.py) | DiffSynth runner 适配：Stage 1、缓存、Stage 2 |
| [distributed/training.py](../src/samtok_edit21/distributed/training.py) | ARNOLD 四机编排（按阶段独立运行） |
| [evaluation/](../src/samtok_edit21/evaluation/) | 评测协议 v2：case、编译、推理、judge manifest、汇总 |
| [third_party/diffsynth](../third_party/diffsynth/diffsynth/models/qwen_image_21_dit.py) | vendored DiffSynth 2.1.8：新增绑定钩子与 latent 融合 |
| [scripts/diagnostics/](../scripts/diagnostics/) | 审计、八卡推理 smoke、GPU 等价性检查、smoke 数据 |
| [tests/](../tests/) | CPU 单元测试（45 项） |

v1 的区域监督 C/A 相关模块（`regions/supervision.py`、`regions/build.py`、`training/attention.py`、`training/calibration.py`、`data/preflight.py`）和 18 行 debug harness 已删除；`regions/selection.py`（SAM2 候选）保留供评测使用。

## 2. 数据协议

**需求。** add 放置新物体，SAMTok mask 无法表示"物体应该出现在哪里"；其他类型编辑已有区域，mask 合适。每个类型只能有一种区域表示，训练与推理一致。

**实现。** [`REGION_KIND`](../src/samtok_edit21/data/protocol.py#L71) 是唯一规则：add → `box`，其余原子类型 → `mask`。

- ⟨M⟩：`<|mt_start|><|mt_XXXX|><|mt_YYYY|><|mt_end|>`，code0 ∈ [0,255]，code1 ∈ [256,511]（不变）。
- ⟨B⟩：[`box_of`](../src/samtok_edit21/data/protocol.py#L94) 生成 `<|box_start|>[x1, y1, x2, y2]<|box_end|>`。坐标为 0–1000 相对整数，x1<x2、y1<y2；拒绝前导零、浮点和越界，不做修复。`<|box_start|>`/`<|box_end|>` 是词表已有 token（151648/151649），不新增 token。
- 像素框转相对框用 [`relative_box`](../src/samtok_edit21/data/protocol.py#L143)：向外取整（左上 floor、右下 ceil），保证非空。
- 定位 JSON（[`to_cot`](../src/samtok_edit21/data/protocol.py#L183) / [`parse_cot`](../src/samtok_edit21/data/protocol.py#L202)）：mask 项 `{"mask_2d": "⟨M⟩", "label": ...}`，框项 `{"bbox_2d": [x1, y1, x2, y2], "label": ...}`（Qwen3-VL 原生格式）。解析后两种区域都用 token 字符串表示，渲染统一。
- 一个单元（短语）内不能混用 mask 和框；多实例用多个同类区域直接相连。

**metadata 行类型**（[`validate_row`](../src/samtok_edit21/data/protocol.py#L528)，v2 拒绝 composite）：

| sample_type | 字段 | 用途 | 例子 |
|---|---|---|---|
| `edit_ntp` | edit_image, prompt, mt_cot | Stage 1：编辑指令 → 区域 JSON | `Add a little bluebird perched on the rightmost birdbath` → `[{"bbox_2d": [663, 294, 908, 460], "label": "little bluebird perched on the rightmost birdbath"}]` |
| `rec_ntp` | edit_image, prompt, mt_cot | Stage 1：框 grounding 回放 | `Locate the plain stump that is farthest away in this image and output its bbox coordinates in JSON format.` → `[{"bbox_2d": [201, 406, 448, 657], "label": "plain stump that is farthest away"}]` |
| `edit_umt` ref | edit_image, image, prompt, instr_variant | Stage 2：带位置描述的区域编辑 | `Add a little bluebird perched on the rightmost birdbath <|box_start|>[663, 294, 908, 460]<|box_end|>` |
| `edit_umt` noref | 同上 | Stage 2：只靠区域 token 定位 | `Add a little bluebird perched in this region <|box_start|>[663, 294, 908, 460]<|box_end|>.` |
| `edit` | edit_image, image, prompt | Stage 2：普通编辑 | 原指令 |

- `edit_ntp` 的区域类型必须符合 `REGION_KIND`（add 必须输出 bbox_2d，其他必须输出 mask_2d）。
- `rec_ntp` 的 prompt 必须是 [`REC_TEMPLATE`](../src/samtok_edit21/data/protocol.py#L38) 加 label，答案只有 bbox_2d。它用 Qwen3-VL 原生 grounding 请求，与编辑定位请求 `LOC_REQUEST` 不同，所以"编辑类型 → 输出格式"的规则不会被打乱。
- `edit_umt` 的 [`validate_inline`](../src/samtok_edit21/data/protocol.py#L475)：区域组数、组内类型、空格和 noref 短语（add 必须是 `in this region`）。

## 3. 训练数据转换（v1 → v2）

**需求。** 沿用 v1 数据，只做计划规定的改动：剔除 composite、add 改框、Derived add 改类型、加回放数据，其余行必须与 v1 逐字节一致。

**实现。** [`preparation/v2_data.py`](../src/samtok_edit21/preparation/v2_data.py) 读取 v1 corpus 的构建目录（`input-XX.jsonl` 原始记录 + `encoded/worker-XX` 已编码行），按源索引顺序归并：

1. 每一条 v1 行先与 v1 `provenance.jsonl` 的 row hash 核对，保证读到的就是 v1 训练用的行。
2. composite 记录整条丢弃。
3. add（RefEdit/CrispEdit/ScaleEdit）：[`add_boxes`](../src/samtok_edit21/preparation/v2_data.py#L100) 取生成该 span 的**同一组实例 mask**（`masks_for_record` + codec 的多实例排序 `_ordered_masks`），计算外接框，原位替换 NTP 答案和两条 UMT prompt 中的 span。
4. Derived add：所有行 `edit_type` 改为 attribute，保留 ⟨M⟩。
5. rec_ntp：从 RefEdit/Derived 的单实例非 add 单元（remove/replace/attribute/action）中，按 `sha256("rec:"+id)` 选取 `round(edit_ntp 行数 / 7)` 条，框取该单元 mask 的外接框（[`rec_row`](../src/samtok_edit21/preparation/v2_data.py#L144)）。
6. 输出 `stage1.jsonl`（只有 NTP）、`stage2.jsonl`（edit + edit_umt）、逐行 `provenance.jsonl`（含 v1 row hash 和转换类型）、`conversion_report.json`、`metadata_report.json`。

读 mask 的 I/O 是瓶颈，所以先单遍分类、再用线程池只读需要的 mask；全量转换约 3.5 分钟。数据统计与校验结果见[数据盘点](04_SAMTokEdit_Qwen21_训练数据盘点.md)。

## 4. Stage 1：定位 LoRA（纯 NTP）

**需求。** pass 1 只负责输出准确的区域 JSON；不再通过 FM 改变 pass-2 的编码（v1 中这一项导致普通 prompt 漂移）。

**实现。**

- 只加载 TE（[`SamtokTrainingModule`](../src/samtok_edit21/training/engine.py#L136)，Stage 1 组件仅 `text_encoder`），LoRA 注入所有 LM 层的 q/k/v/o 与 gate/up/down（`TE_TARGETS`），视觉塔冻结。
- [`localization_inputs`](../src/samtok_edit21/models/pipeline.py#L182)：SAMTok chat template，user = 图像 + 文本，assistant 预填空 think。`edit_ntp` 的文本是"指令 + `LOC_REQUEST`"；`rec_ntp`（`request=None`）的文本就是 grounding 请求本身。
- [`ntp_loss`](../src/samtok_edit21/models/pipeline.py#L233)：只监督答案 JSON 加 `<|im_end|>` 的交叉熵，`--ntp-weight` 默认 1.0。
- 调度（[`RATIOS`](../src/samtok_edit21/data/io.py#L20)）：每个 optimizer update 中 edit_ntp : rec_ntp = 7 : 1，每个 rank 都精确满足。类型采样由 `--type-weights` 决定（[`type_probabilities`](../src/samtok_edit21/data/io.py)）：Stage 1 默认 `natural`（按各类型行数，每行被采次数相同；Stage 1 约 3 遍，避免少数类型被反复采样），rec_ntp 始终按自然分布。

## 5. 条件缓存：raw TE + 绑定 payload

**需求。** Stage 2 的条件必须与推理 pass 2 完全相同的编码器（raw TE）；所有绑定臂共用一份缓存，因此每个区域行都要预存绑定信息。

**实现。**

- 缓存阶段加载 TE + VAE + codec，**不加载任何 TE adapter**。[`prepare_fm`](../src/samtok_edit21/training/objectives.py#L233) 对每行编码目标/源图 latent 和 prompt；含区域的行另外生成绑定 payload（[`binding_payload`](../src/samtok_edit21/models/binding.py#L194)）。
  - [`token_layout`](../src/samtok_edit21/models/binding.py#L141)：在官方 PromptEmbedder 实际输入的 token 序列（去掉 system 后）中定位每个区域单元（连续的 ⟨M⟩/⟨B⟩ token）和指令范围（`<|vision_end|>` 之后到序列末尾）。逐单元 decode 回字符串，与 prompt 中的区域组逐字比对，避免绑错 token。
  - [`unit_masks`](../src/samtok_edit21/models/binding.py#L119)：⟨M⟩ 用 codec 在源图上解码（raw logits > 0.5），⟨B⟩ 在源图像素上向外取整栅格化；同一单元取并集。
  - [`coverage_grid`](../src/samtok_edit21/models/binding.py#L73)：双线性抗锯齿缩放到画布、16×16 平均池化（每个 latent token 的覆盖率），再 3×3 最大池化膨胀 1 个 token。目标网格和源网格各存一份（float32）。
  - 区域解码为空的单元标记为 `empty`，训练和推理都不绑定它。
- 缓存格式 v3（[`provenance.py`](../src/samtok_edit21/data/provenance.py)）：identity 记录 `te_adapter: null`、模型文件 hash、codec 与 SAM2 hash、绑定几何版本、metadata hash 和逐行 hash 摘要。[`validate_cache_inputs`](../src/samtok_edit21/data/provenance.py#L164) 要求区域行必须带 payload、普通行不能带，payload 结构由 [`validate_payload`](../src/samtok_edit21/models/binding.py#L214) 检查。
- Stage 2 启动时，各 rank 按 `行号 % world_size` 并行校验全部行（[`verify_cache_shard`](../src/samtok_edit21/data/provenance.py#L201)），因此任意卡数都能使用同一份缓存。
- 存储：1M 像素下每行约 9.6 MB（主要是图像 token 的 TE hidden states，与 v1 相同），全量约 2.8 TB。

### 5.1 不建缓存：Stage 2 即时计算条件（默认）

**需求。** 全量缓存 2.8 TB，user 和 intern 的 NAS 配额都放不下（运行 A 在缓存写到 52% 时因配额失败，见实验记录第 9 节）。而算条件本身很快（集群上约 0.6 s/行/卡），所以 Stage 2 改为每个样本训练前现算条件，不落盘。要求训练结果与"先建缓存再训练"完全相同。

**实现。**

- [`OnlineConditioning`](../src/samtok_edit21/training/engine.py)：在训练进程里另外加载与缓存阶段完全相同的 raw TE + VAE（`load_pipeline(components=("text_encoder", "vae"))`）和 codec，对每个样本调用同一个 `prepare_fm`。保证一致的三点：
  - `prepare_fm` 整个在 `torch.no_grad()` 下运行（缓存阶段的 runner 也是这样调用的）；
  - 结果先拷到 CPU，再用 data loader 的 `send_to_device` 送回 GPU，与缓存 payload"存盘、读回、送 GPU"走同一条路；
  - 整个计算在 `torch.random.fork_rng` 内，不消耗训练的随机数，所以 timestep 和噪声的抽取与读缓存训练逐个相同。
- 它不是 `nn.Module`：冻结的 TE/VAE 不进 DDP、优化器、checkpoint 和 rank 权重 hash，被训练的模块与读缓存时完全相同。
- [`run_train`](../src/samtok_edit21/training/engine.py)：Stage 2 给 `--metadata` 时，读同一个 `stage2.jsonl`，在 rank 0 计算与缓存相同的条件身份（`conditioning_identity`，含模型 hash），用同样的调度。adapter.json 记录的 `conditioning_identity` 与建缓存时逐字相同，推理端的 pass-2 选择不受影响；`run.json` 的 plan 里多记一项 `conditioning: on_the_fly`。
- 每个样本训练时多一次 TE/VAE 前向，每卡显存多约 17 GB（TE 8B，bf16）。

**等价性验证**（实验记录第 9 节）：
- 64 个集群生产缓存 payload（`5443a7b`，1M 像素）和 212 行本地 smoke 缓存：即时计算结果逐张量逐位一致；row hash 和条件身份一致；训练随机数状态不变。
- 读缓存与即时计算各训练 3 个 update（同种子，`none` 与 `bias_clause` 两种绑定）：逐 update 的 metrics、optimizer 日志、各 rank 梯度日志、各 rank 权重 hash、checkpoint 和最终 adapter 逐位一致。

## 6. Stage 2：DiT LoRA + 结构性绑定

**需求。** 去掉 v1 的区域加权 C 和 attention loss A（第 4 节分析：A 退化、监督的不是模型实际使用的通路）；改用训推一致的结构性绑定，按优先级逐一消融。

**loss。** [`flow_loss`](../src/samtok_edit21/training/objectives.py#L261) 与官方 `FlowMatchSFTLoss` 相同：均匀抽 timestep、`x_t = (1−σ)x0 + σε`、目标 `ε − x0`、MSE × 官方时间权重。日志额外记录 `region_row`（该行是否含区域）和 `bound_units`（实际绑定的单元数）。

**绑定模式**（[`BindingConfig`](../src/samtok_edit21/models/binding.py#L57)，训练 `--binding`，参数写入 adapter.json，推理自动沿用）：

| 模式 | 作用 | 参数 |
|---|---|---|
| `none` | B0：不绑定 | — |
| `bias_span` | 目标 query q 对该单元区域 token 的 attention logit 加 `β·log(ε + (1−ε)·m(q))` | β（默认 1.0）、ε（默认 0.05）；ε=0 时下限为 log(1e-6) |
| `bias_clause` | 同上，作用于源图之后的整条指令（含尾部模板 token）；单单元 prompt 时等价于区域化 prompt | 同上；多单元 prompt 报错 |
| `region_embed` | 目标 token q 加 `m(q)·W(h)`，h 为该单元最后一个区域 token 经 `txt_in` 后的特征，W 为秩 64 的低秩映射，上投影零初始化 | rank（默认 64） |
| `region_rope` | 该单元区域 token 的 RoPE (h, w) 索引改为区域质心在目标网格上的居中坐标，帧索引不变 | — |

m(q) 是目标 token q 的区域覆盖率（第 5 节，0–1）。

**DiT 钩子（vendored DiffSynth）。**

- [`QwenImage21DiT.forward(region_binding=...)`](../third_party/diffsynth/diffsynth/models/qwen_image_21_dit.py#L539)：调用 `region_binding.bind(...)` 把 TE 位置映射到联合序列位置（图像 pad 展开为 4 个 latent token），然后 `apply_embedding`、`apply_rope`。
- 无缓存/首步（FlexAttention）：[`score_mod`](../src/samtok_edit21/models/binding.py#L361) 按"key 属于哪个单元 × query 位置"查表加偏置，经各 block 传给 attention processor（[行 247](../third_party/diffsynth/diffsynth/models/qwen_image_21_dit.py#L247)）。
- KV cache 解码步：同样的偏置写成加性 SDPA mask（[`decode_mask`](../src/samtok_edit21/models/binding.py#L378)），一次 forward 只构造一次。
- 只有目标 query 被偏置，region_rope 只改前缀 token 的 RoPE（首步写入 cache），所以首步缓存的前缀 K/V 在后续步骤中仍然有效。
- `region_embed` 对不含区域的行也用零向量调用一次映射，保证 DDP 中每个参数都参与反向（`find_unused_parameters=False`）；新增参数与 LoRA 一起保存在 adapter 中，`load_adapter` 根据 adapter.json 的 `binding` 字段重建模块。
- v1 的 attention probe/LSE 统计接口已从 vendored DiT 中移除；不传 `region_binding` 时计算路径与原实现相同。

**调度。** ref : noref : plain = 1 : 2 : 1，每个 rank 每个 update 精确满足。UMT 池默认 `main4`：其他类型（action、text 等）按自然占比抽取、不被放大，其余份额由 add/remove/replace/attribute 按 v1 的 14:14:14:20 分配；plain 按自然分布（background/global 单类上限 15%）。`v1`（原类型权重）仍可选。

## 7. 推理

**pass 1 / pass 2 的 TE。** [`te_adapters(pipe, enabled)`](../src/samtok_edit21/models/pipeline.py#L160) 临时开关 TE 的全部 PEFT 层，并恢复 requires_grad。`localize` 生成时启用定位 LoRA；[`edit`](../src/samtok_edit21/models/pipeline.py#L359) 的 pass 2 由 `pass2_te` 决定，CLI 根据 Stage 2 adapter 的 conditioning identity 自动选择（[`pass2_text_encoder`](../src/samtok_edit21/data/provenance.py#L93)）：

- v2 adapter（缓存 v3、`te_adapter: null`）→ `raw`：关闭 adapter。关闭后的编码与 raw SAMTok TE 逐位一致（已验证）。
- v1 adapter（缓存 v2）→ `adapter`：必须加载完全相同的 Stage 1 adapter，pass 2 保持启用（复现 v1）。

**推理期绑定（D8）。** [`region_binding_for`](../src/samtok_edit21/models/pipeline.py#L263) 对 pass 2 实际编码的 prompt 重新计算 token 布局和区域图（⟨M⟩ 由 codec 解码，⟨B⟩ 栅格化），计算方式与缓存相同。CLI `--binding` 默认沿用 adapter 的配方，也可在推理时覆盖模式/β/ε（用于 E4：在 B0 上只加推理期偏置）；`region_embed` 只能用于带该模块训练的 adapter。

**latent 融合。** [`blend_outside_region`](../third_party/diffsynth/diffsynth/pipelines/qwen_image_21.py#L138)：每个去噪步之后，`latents = m·latents + (1−m)·((1−σ_{t+1})·x_src + σ_{t+1}·ε)`。x_src 是源图按目标画布精确缩放后的 VAE latent，ε 是初始噪声，最后一步 σ=0（区域外等于源图）。[`blend_inputs`](../src/samtok_edit21/models/pipeline.py#L284) 构造融合区域：源图像素 mask → latent 网格（格内任一像素属于区域即为区域）→ 膨胀 2 个 token → 高斯羽化（σ=1 token）。add 的框先外扩 10%（每边按框宽/高的 10%）。全 1 mask 时融合是 no-op（已验证逐位一致）。

**CLI 例子**（安装后 `samtok-edit`，或 `python -m samtok_edit21`）：

```bash
# 纯文本两遍：pass 1 定位 LoRA，pass 2 raw TE，融合区域取 pass-1 解码区域
python -m samtok_edit21 infer --mode online --image src.png --prompt "Remove the leftmost bird." \
  --te-adapter <run>/stage1/adapter --dit-adapter <run>/stage2/adapter \
  --blend-prompt --output out.png
# 交互 add：框 token 直接写入 prompt，融合区域为外扩 10% 的框
python -m samtok_edit21 infer --mode inline --image src.png \
  --prompt "Add a red ball in this region <|box_start|>[300, 300, 700, 700]<|box_end|>." \
  --dit-adapter <run>/stage2/adapter --blend-box 300 300 700 700 --output out.png
# 在 B0 上只加推理期偏置（E4）
python -m samtok_edit21 infer --mode inline ... --binding bias_clause --binding-beta 2 --binding-eps 0.05
```

## 8. 评测协议 v2（代码）

模块在 [`src/samtok_edit21/evaluation/`](../src/samtok_edit21/evaluation/)，按计划第 8.7 节实现：

| 步骤 | 模块 | 说明 |
|---|---|---|
| case | [cases.py](../src/samtok_edit21/evaluation/cases.py#L63) | 517 个单区域 case + MIRAGE 99 个双区域 case 拆成 198 个原子编辑（[`mirage_atomic`](../src/samtok_edit21/evaluation/cases.py#L35)：带位置指令按 ", and " 拆分，region-only 指令按 "For {region_k}," 拆分，第 k 个子句对应第 k 个区域），共 715 个；按源图 hash 固定划分 30% dev / 70% test（[`split_of`](../src/samtok_edit21/evaluation/cases.py#L23)） |
| 编译（D2） | [compile.py](../src/samtok_edit21/evaluation/compile.py#L41) | `sources` 把 case 写成转换器输入（add/remove 类型钉住，replace 由规则判定）；在 vLLM 0.17.1 环境运行**训练数据用的同一个** `preparation.semantic`（Qwen3.5-9B + 规则回退）；`compile` 生成带 `{region}` 的 noref 模板，失败时退回确定性的 `interactive_prompt`；所有模板用训练 noref 语法校验；`review.md` 每类抽样供人工检查。2026-10-06 起转换器对 add 多一条规则（删除整个放置、保留新物体的姿态/外观，见实验记录 6.3），其他类型的 prompt 不变；训练数据仍是旧 prompt 的结果 |
| 推理 | [run.py](../src/samtok_edit21/evaluation/run.py#L139) | torchrun 每卡一个进程，可断点续跑；设置与区域处理见下表；`--stock` 用 benchmark 仓库的 two-image 定位协议跑 stock |
| judge | [manifest.py](../src/samtok_edit21/evaluation/manifest.py) | 生成原 pair_v2 judge 的 manifest（字段与 input digest 与 judge 自带 prepare 一致）；未改动的 517 个 case 直接复用已有 stock 输出 |
| 汇总 | [score.py](../src/samtok_edit21/evaluation/score.py) | 方法 × setting × 编译类型 × dev/test 的 E/P/Q/strict；`--compare A B` 给出按 case 配对的 bootstrap 区间 |

| setting | mask 类（remove/replace/attribute/...） | box 类（add） |
|---|---|---|
| `mask` | codec(用户 mask)；融合区域 = 用户 mask | 用户 mask 的外接框；融合 = 框外扩 10% |
| `box` | SAM2(框) 最高分候选 → codec；融合 = 用户框 | 用户框；融合 = 框外扩 10% |
| `point` | SAM2(点) 最高分候选 → codec；融合 = 该候选 | 以点为中心、训练 add 框中位尺寸（宽 214、高 267，0–1000）的默认框（D5） |
| `text` | pass 1 → ref prompt；融合 = 解码区域 | pass 1 → ref prompt；融合 = 框外扩 10% |
| `text_plain` | — | 原指令，不加区域（D9 对照） |

每个 setting 可同时跑融合开/关（输出目录 `<setting>` 与 `<setting>+blend`）。交互 setting 默认用 noref 模板；`--prompt-variant ref` 改用 ref 模板（原指令 + 区域 token，输出目录加 `+ref`），用于检查 noref 改写丢失信息的影响（见实验记录 6.1）。画布 = 源图宽高比、面积约 1024²、边长 32 的倍数；40 步、CFG 1、seed 0、KV cache；输出为白底合成的 RGB，缩放回源图尺寸（与 stock 协议相同）。

## 9. 四机编排、审计与诊断脚本

- [`distributed/training.py`](../src/samtok_edit21/distributed/training.py#L217)：`--phases` 取 `stage1,cache,stage2` 的子集，默认 `stage1,stage2`。Stage 2 的输入有三种来源：本次 `cache` 阶段建的缓存；`--cache` 指向的已完成缓存；两者都没有时即时计算（5.1 节，默认）。`--binding*` 传给 Stage 2。每个阶段之后跨节点 barrier；最后由 rank 0 运行审计。非 `--full-training` 时额外跑八卡推理 smoke。
- [`scripts/diagnostics/audit_run.py`](../scripts/diagnostics/audit_run.py)：对已运行的阶段检查 update 数、每个 rank 的精确配比、梯度日志、各 rank 权重 hash 一致、adapter 有限且 LoRA B 已更新、adapter 的绑定配方与参数一致、W&B 完成、缓存行（小缓存全部，全量随机 1000 行）。即时计算的运行没有缓存，改为检查 adapter 记录的条件身份：须是 raw TE 的 v2 协议，且 metadata hash 和分辨率与本次运行一致。
- [`scripts/diagnostics/check_online_conditioning.py`](../scripts/diagnostics/check_online_conditioning.py)：对缓存 payload 逐个用即时计算重算，逐张量比较，并检查 row hash、条件身份和训练随机数状态。
- [`scripts/diagnostics/compare_stage2_runs.py`](../scripts/diagnostics/compare_stage2_runs.py)：比较读缓存与即时计算的两次 Stage 2 训练，要求 metrics、日志、权重 hash、checkpoint、adapter 全部逐位一致。
- [`scripts/diagnostics/smoke_inference8.py`](../scripts/diagnostics/smoke_inference8.py)：八卡分别跑 direct、inline（mask/框 × ref/noref）、online、oracle（mask/框）、推理期绑定覆盖、无 KV cache 循环，含融合。
- [`scripts/diagnostics/check_v2_equivalence.py`](../scripts/diagnostics/check_v2_equivalence.py)：单卡逐位/no-op 检查（第 10 节）。
- [`scripts/diagnostics/make_smoke_data.py`](../scripts/diagnostics/make_smoke_data.py)：每个（数据集, 类型）组取前 N 个源（按 id hash）的全部行，加同比例 rec_ntp。

## 10. 测试与等价性

CPU 单元测试（`pytest tests/`，48 项）覆盖：框格式与外向取整、bbox JSON 往返与类型规则、rec_ntp 校验、v1→v2 转换（合成 v1 corpus：多实例排序、Derived 改类型、回放选择、provenance 漂移检测）、两阶段调度配比、缓存 v3 身份与 payload 校验、真实 tokenizer 下的 token 布局、`score_mod` 与解码 mask 一致、region_embed 零初始化与 DDP 参数参与、region_rope 质心、评测 case 拆分/编译回退/默认框/manifest、Stage 2 输入来源参数（`--cache` 与 `--metadata` 二选一）、即时计算运行的审计条件。

GPU 检查（结果见[实验记录第 4 节](02_SAMTokEdit_Qwen21_实验记录.md#4-gpu-等价性检查)）：

1. pass 2 关闭 adapter 与 raw TE 逐位一致；
2. region_embed 零初始化与无绑定逐位一致；
3. 全 1 融合 mask 与不融合逐位一致；
4. 偏置在 KV cache 与无缓存路径下的差异与无偏置时相同量级；
5. β=0 与无绑定相同（kernel 级差异）；
6. v1 adapter 经 v2 代码（绑定、融合关闭）与 v1 代码输出逐位一致。

## 11. 超参数与默认值

| 项 | Stage 1 | Stage 2 |
|---|---|---|
| 训练对象 | TE LoRA（LM 全部 attention + MLP 线性层） | DiT LoRA（DiffSynth 自动检测的 224 个模块）+ 可选 region_embed |
| rank / alpha / dropout | 64 / 64 / 0.05 | 32 / 32 / 0 |
| LR / 调度 / warmup | 4e-5 / cosine / 4% | 1e-4 / constant / 2.5% |
| weight decay / 梯度裁剪 | 0.05 / 1.0 | 0.01 / 1.0 |
| 每卡 accumulation / 32 卡全局 batch | 8 / 256 | 4 / 128 |
| 配比 | edit_ntp : rec_ntp = 7 : 1 | ref : noref : plain = 1 : 2 : 1 |
| 类型采样（`--type-weights`） | `natural`（四个主类型约 95%，每行约见 3 次） | `main4`（四个主类型约 95%，按 14:14:14:20） |
| optimizer updates | 1,300（约 3 个 epoch） | 1,000（缩减日程 R，选型用） |
| loss | NTP（权重 1.0） | 官方 FM（+ 绑定无额外 loss） |
| max_pixels | 1,048,576 | 同缓存 |

推理：面积约 1024²、40 步、CFG 1、seed 0、KV cache、pass-1 `max_new_tokens=256`、融合膨胀 2 token/羽化 σ=1、add 框外扩 10%。

## 12. 重要路径

| 类别 | 路径 |
|---|---|
| v2 实验根目录 | `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v2/`（2026-10-06 起；代码中为 `data/io.py` 的 `EXPERIMENT_ROOT`） |
| 旧根目录（存档，intern 配额已满） | `/mnt/bn/strategy-mllm-train/intern/users/tanyue/experiments/SAMTokEdit/qwen21_v2/`：本地 smoke 运行和日志（`smoke/runs/`、`smoke/logs/`）只在这里；数据、评测和 smoke 数据已逐字节复制到新根目录 |
| v2 训练数据 | `qwen21_v2/data/train_v2_box_001/` |
| smoke 数据 | `qwen21_v2/smoke/data_smoke_001/`（运行和日志见旧根目录） |
| 正式运行 | `qwen21_v2/runs/<SAMTOK_RUN_ID>/` |
| 评测 | `qwen21_v2/eval/protocol_v2_001/`（cases、转换器输出、编译模板、smoke） |
| v1 数据（转换输入） | `/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/qwen21_full4_20260928/data/train_full_9b_rules_003/` |
| v1 adapter（参照） | Stage 1 `/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/qwen21_full4_20260928/runs/qwen21_full_4n_formal_003/stage1/adapter`；Stage 2 `/mnt/bn/strategy-mllm-train/intern/users/tanyue/experiments/SAMTokEdit/qwen21_full4_20260928/runs/qwen21_full_4n_formal_003_resume_006/stage2/adapter` |
| Qwen-Image-2.1 | `/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen-Image-2.1` |
| SAMTok TE / codec | `/mnt/bn/strategy-mllm-train/user/tanyue/models/SAMTok/Qwen3-VL-8B-SAMTok`（含 `sam2.1_hiera_large.pt`、`mask_tokenizer_256x2.pth`） |
| noref 转换器 / judge 模型 | `.../pretrained_models/Qwen3.5-9B`、`.../pretrained_models/Qwen3.8-27B` |
| benchmark | 数据 `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/samtok_edit_benchmark`；仓库 `/opt/tiger/tanyue/samtok_edit_benchmark`（MIRAGE 原子编辑选择、stock 协议） |
| 已有 stock 输出 | `/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/qwen21_656/inference/qwen21/` |
| judge 代码 | `/mnt/bn/strategy-mllm-train/intern/users/tanyue/experiments/SAMTokEdit/qwen21_stage2_benchmark_noref_aligned_20261004/judge/code` |
| 本地环境（节点 /tmp，非持久） | 训练/推理 `/tmp/samtok21-fixes-dUnbt5/venv`（torch 2.8、transformers 5.12.1、peft 0.20）；转换器 `/tmp/samtok21-layout-annotation-env`（vLLM 0.17.1）；judge `/tmp/samtok21-benchmark-judge-env`（vLLM 0.28）。集群上由 `scripts/training/setup_env.sh`、`scripts/annotation/setup_env.sh` 重建 |
