# SAMTokEdit Qwen-Image-2.1 v4：方法

本文说明 v4 的方法与代码：
- 方法的设计与架构：每个部件是什么、为什么这样做（§0–§4）；
- v4 新增的在线自蒸馏（OPSD）的算法设计（§5）；
- 代码实现：基于哪些开源仓库、改了什么、仓库怎样组织、各部件在哪里实现，以及正确性验证和运行方法（§6）。

数据见[文档 01](01_SAMTokEdit_Qwen21_v4数据.md)，实验设置与结果见[文档 03](03_SAMTokEdit_Qwen21_v4实验.md)。代码路径相对仓库根目录（分支 `qwen-image-2.1-v4`）；`V3/…` 指 `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v3/…`。截至 2026-10-27。

v4 与 v3 的关系：
- **方法：**与 v3 第二轮（C1 → C1b → C2 → C3）相同。代码只保留这条路线，并按模块重新整理。v2 的两阶段方法、v3 第一轮的工具和全部旧文档留在 `qwen-image-2.1-v3` 分支。
- **DiffSynth：**随仓库提供的 DiffSynth 升级到 2.1.12。同一份权重下，训练损失、梯度和推理输出与 v3 逐位相同（§6.6）。
- **新增：**OPSD 训练（§5、§6.5、§6.7）。

## 0. 一句话方法、验收口径与当前最好的配置

**方法。**SAMTok 的 codec 把编辑区域编码成 4 个 token 的 mask 片段 `<|mt_start|><|mt_XXXX|><|mt_YYYY|><|mt_end|>`（下文记作 ⟨M⟩），直接写进编辑指令。例如：

```
Replace the selected region <|mt_start|><|mt_0097|><|mt_0411|><|mt_end|> with a navy polo shirt.
```

- 冻结的 Qwen3-VL-8B-SAMTok 文本编码器（TE）编码整句；码本注入模块（CodeInjector）在 ⟨M⟩ 的位置加上码本向量的投影。
- Qwen-Image-2.1 DiT（原权重 + LoRA）按这个条件编辑源图。
- **推理时不解码像素 mask，不加注意力偏置，不做 latent 融合。**区域只以 token 的形式进入模型。
- mask 只在训练时使用，用途有四：
  - 构造目标图；
  - 损失权重；
  - 注意力监督标签；
  - OPSD 老师的特权输入。

**验收口径（用户 2026-10-18 决定）。**只看带 token 的编辑，必须好于原版。token 来自用户给的区域，或以后由 TE 从文字指令生成。不带 token 时比原版差可以接受，benchmark 的纯文字分数不作验收。

**当前最好的配置**（结果见文档 03）：
- **训练好的权重：**C3-D，`V3/runs/c3d_local8_001/train/adapter_1500`，含 DiT LoRA 和码本注入。
- **推理：**token 引导 2.0（§3.11）；只给文字的单元先由原始 TE 生成 token（§3.12）。
- **结果：**benchmark 2.6.0 的 301 例决策子集上，整图全过（JointPass）为 37.7%（方案 A 那次判分，300 个共同 case；301 例上为 37.9%）。
  - 同一子集在 C3 那次判分里：原版 11.7%，FLUX.3 27.7%，C3-D 35.0%；
  - 老师上限测试那次判分里重判：C3-D 34.2%，C3-D + 引导 37.9%。
- **OPSD 老师（T3）：**同一子集、老师测试那次判分里为 45.8%。老师读 GT mask，这只是上限。
- **OPSD 训练：**代码已实现并验证（§5、§6.7）。试训（10-26 至 10-27）按现在的配方没有带来提升：P1 对 P0 整图全过 −0.7 个点 [−5.3, +4.0]；加引导后 P1@500 为 40.3%，对 C3-D + 引导 +4.0、对 P0 + 引导 +4.3，都不显著，高出的部分来自保持。诊断与修订方向见 §5.11，结果见文档 03 §7 和 `V3/analysis/opsd_pilot_20261026/README.md`。

## 1. 出发点与设计原则

### 1.1 v2 的问题

v2 的 DiT 条件用原始 SAMTok TE，并试过把 codec 解码出的区域作注意力偏置。2026-10-11 复核发现了下表的问题（完整证据见 v3 分支的文档 06）。v3/v4 的每个部件都对应其中一条或几条：

| # | 问题 | 证据（摘要） | 对应的设计 |
|---|---|---|---|
| P1 | **DiT 生成时不读 mask token**（核心） | 同图同句只换 token 的删除 120 对：跟对率只有 0.06–0.20；改用 codec 区域偏置后是 0.75 | 让 token 成为降低损失的唯一途径：路由损失、指认任务、近邻反例 |
| P2 | mask 当输入是捷径 | 训练时带区域偏置的模型更不读 token | mask 只作监督，不作学生输入 |
| P3 | 只用编辑对的 FM 损失学不会 token 的含义 | FM 误差约 97% 在区域外；有物体名时读名词就够 | 盲指代指令、按区域面积归一的损失、逐像素精确的目标 |
| P10 | token 读得出身份，读不出范围 | 读出头约 75% 选对实例，IoU 最多 0.50；码本向量比 DiT 第 12 层更好读（+0.15） | 码本注入 |
| P12 | 原版缺的主要是读 token | 原版加上区域偏置后，attribute 1.00，add 0.82 | 编辑以原版能力为基础，小 LoRA 只补读 token |
| P13 | 训练 TE 会改坏文本条件 | v1：条件余弦相似度 0.786，严格成功率从 0.63 降到 0.27 | 不训练 TE，DiT 条件始终用原始 TE |
| P14 | 架构约束 | 联合序列按块因果，源图 token 看不到 ⟨M⟩ | 读取只可能是 ⟨M⟩→源图、目标→⟨M⟩/源图，路由损失正好监督这两处 |

### 1.2 设计原则
1. **推理形式不变：**区域 → SAMTok token → 写进指令 → DiT。不解码、不偏置、不融合。
2. **mask 只作监督，不作学生的输入。**OPSD 里 mask 是老师的输入，学生仍然只看 token（依据 P1、P2）。
3. **让 token 成为降低损失的唯一途径：**盲指代指令、同类与近邻干扰、按面积归一的损失、精确的目标（依据 P3）。
4. **保住原版能力：**
   - 编辑数据尽量用原版自己生成的结果（SA-1B 50k 自蒸馏，文档 01）；
   - LoRA 小（rank 32）；
   - TE 不训练。
5. **token 只需给出无歧义的身份：**精确边界交给生成器的先验。

## 2. 整体架构

### 2.1 条件路径与联合序列

```
训练与推理共用的条件路径
  指令文字（含 ⟨M⟩）─► Qwen3-VL-8B-SAMTok TE（冻结，取最终 RMSNorm 之前的隐状态）
                        └─► + CodeInjector：⟨M⟩ 的各位置加码本向量的线性投影 ─┐
  源图 ─► VAE ─► 源图 latent ───────────────────────────────────────────────┤
                                                                            ▼
                         Qwen-Image-2.1 DiT（冻结 + LoRA r32）─► 目标 latent ─► VAE ─► 输出
只在训练时用的监督（不进入推理）
  token 的 codec 解码区域 / 实际改动像素 / 邻居与孪生区域
     ├─► 按面积归一的区域 FM 项、邻居 FM 项
     ├─► 注意力路由损失：写（目标 token → ⟨M⟩）、读（⟨M⟩ → 源图 token）
     └─► OPSD 老师的特权输入：允许改动的区域、带名指令（§5）
```

DiT 是单流 MMDiT：32 个 block，宽 4096，32 头，bf16。所有输入排成一条联合序列（`third_party/diffsynth/diffsynth/models/qwen_image_21_dit.py`）：

```
[ 文本前缀 | 源图 latent | 指令文字（⟨M⟩ 是其中 4 个位置） | 目标 latent ]
```

- **网格：**VAE latent 是 64 通道、H/16 × W/16 的网格，每格对应一个 DiT token，即 16×16 像素。TE 里每个源图占位 token 对应 2×2 格，在 DiT 里展开成 4 个 token，填入源图 latent。
- **调制与注意力：**
  - 时间步调制只作用于目标 token，前面的条件部分恒为 t = 0；
  - 注意力按块因果：前缀看不到目标，目标看得到全部前缀；
  - 所以推理时前缀的 K/V 只算一次并缓存（`use_kv_cache=True`），之后每一步只算目标 token。
- **读 token 的途径：**⟨M⟩ 在序列里的位置固定，目标 token 只能通过注意力读它。§3.5 的路由损失监督的正是这条通路。

### 2.2 推理流程
1. **画布：**约 1 MP，保持宽高比，边长取 32 的倍数（`src/samtok_edit21/evaluation/generation.py::canvas`）。
2. **区域 → token：**`src/samtok_edit21/models/codec.py::SamtokCodec` 的 `encode`。用户给的是点或框时，先由 SAM2.1 变成 mask（`src/samtok_edit21/regions/selection.py`）。
3. **写进指令：**按协议 2.0 写入（§3.10）；多个单元写成编号的多行提示（§3.9）。
4. **生成：**`generation.generate` → `pipeline.edit(..., mode="inline", pass2_te="raw")` → DiffSynth 的 `QwenImage21Pipeline.__call__`。默认 CFG 1，或用 token 引导（§3.11）。
5. **导出：**输出合成到白底，缩放回原图尺寸（`generation.export`），与 benchmark 的打分口径一致。

benchmark 适配器 `src/samtok_edit21/evaluation/benchmark_token.py` 在此之上处理 benchmark 的任务格式：
- 每个单元的定位（点、框、mask，或灰度标签图）编码成 token；
- 没有定位的单元先由原始 TE 生成 token（`ground`，§3.12）；
- `token_guidance` 打开 token 引导；
- `teacher` 只用于 OPSD 的老师上限测试。它读 GT，不能用于提交。

### 2.3 训练流程

```
合板数据 / 第二轮数据 ──round2-data──► 训练行文件（jsonl，行序就是训练顺序，见 §3.13）
                                         │
round2-train（torchrun，每卡一个进程）按行处理：
  ├─ 普通行（编辑、指认、多目标……）：prepare_round2 → round2_loss（FM + 区域 + 邻居 + 写/读路由）→ backward
  └─ OPSD 行（opsd_merged）：opsd_backward（学生自己采样 → 老师给目标 → 学生损失）→ backward
每次更新：各卡梯度平均（一次 all-reduce）→ AdamW → 用 EMA 老师时更新老师权重
```

## 3. 部件与损失

本节是 v3 第二轮定型、v4 原样保留的部分。各阶段在哪一步加入见 §7。

### 3.1 底座 DiT 与 LoRA
- **底座：**Qwen-Image-2.1 的 DiT、VAE 和 processor 原样使用，bf16，冻结（`src/samtok_edit21/models/pipeline.py::load_pipeline`）。
- **LoRA：**
  - PEFT LoRA，rank 32，alpha 32，dropout 0；
  - 目标模块是 DiffSynth 自动检测出的 `to_q`、`to_k`、`to_v`、`to_out.0`、`proj`、`out`、`gate_layer`；
  - 共 83,886,080 个参数，可训练参数转成 fp32；
  - 代码：`src/samtok_edit21/models/adapters.py`（`add_adapter`、`load_adapter`、`save_adapter`）。
- **可训练参数合计：**87,044,120 = LoRA + 码本注入 3,158,016 + 写、读两个方向的路由仿射各 12 个。仿射只参与损失，推理不用。
- **为什么：**原版缺的只是读 token 和少量先验（P12），小 LoRA 足够，对原版能力的损伤也小。

### 3.2 文本编码器
- **模型：**Qwen3-VL-8B-SAMTok，冻结。
  - 取最终 RMSNorm 之前的隐状态，用一个临时的 forward pre-hook 读出；
  - 加载时检查隐藏维度 4096、视觉 patch 16，保证和 DiT 接得上；
  - 代码：`SamtokTextEncoder`、`load_pipeline`。
- **分词：**分词器和对话模板换成 SAMTok 的（`build_processor`）。检查 514 个 mask 相关 token（`<|mt_start|>`、512 个码、`<|mt_end|>`）各是词表里不同的单个 token。
- **DiT 条件始终用原始 TE（`pass2_te="raw"`）：**TE 不训练（P13）。码本注入加在 TE 的输出上：

<!-- code: src/samtok_edit21/models/pipeline.py::SamtokTextEncoder.forward -->
```python
def forward(self, **inputs):
    injector = self.code_injector
    if injector is None or not injector.enabled:
        return self.encode(**inputs)[0]
    # The side channel is trained against a frozen backbone: no graph through the text encoder,
    # gradients reach only the injector's projections.
    with torch.no_grad():
        hidden = self.encode(**inputs)[0]
    return hidden + injector(inputs["input_ids"]).to(hidden.dtype)
```
<!-- /code -->

### 3.3 SAMTok codec
- **编码方式：**VQ-SAM2，两级残差量化，每级 256 个码。一个 mask 编成 2 个码：第一级写成 `<|mt_0000|>`–`<|mt_0255|>`，第二级加 256（`src/samtok_edit21/models/codec.py`）。
- **token 的含义以 `decode_strict` 的解码区域为准：**训练目标、路由标签、评测 IoU 都用它。
- **小区域：**解码不准。往返 IoU < 0.5 的样本不进合板数据的默认视图（文档 01）。

### 3.4 码本注入（CodeInjector）
- **注入方式：**
  - ⟨M⟩ 的第一级码位置加 `W0·e0 + b0`，第二级码位置加 `W1·e1 + b1`；
  - `<|mt_end|>` 只在紧跟两级码时加 `W2·(e0 + e1) + b2`，即量化后的 mask 表示；
  - `e0`、`e1` 是 codec 的码本向量，256 维。
- **初始化与参数：**3 个 `Linear(256→4096)`，全零初始化，所以未训练时与原版逐位一致；共 3,158,016 个参数。
- **梯度与校验：**模块加在 TE 输出上，TE 主干不求梯度，梯度只进注入模块。保存时写入码本指纹，指纹不符就拒绝加载。
- **为什么：**码本向量本身比 DiT 中间层更能读出区域（P10）。让 DiT 直接拿到码背后的几何，比硬记 512 个 id 容易。
- **代码：**`src/samtok_edit21/models/code_injector.py`。

<!-- code: src/samtok_edit21/models/code_injector.py::CodeInjector.forward -->
```python
def forward(self, input_ids):
    """``[B, L]`` token ids -> ``[B, L, hidden]`` correction (zero away from mask spans)."""
    index = self.code_index(input_ids)
    length = index.shape[1]
    level0 = (index >= 0) & (index < CODES_PER_LEVEL)
    level1 = index >= CODES_PER_LEVEL
    book = self.codebook

    def rows(position, level):
        low = level * CODES_PER_LEVEL
        return book[position.clamp(low, low + CODES_PER_LEVEL - 1)]

    delta = self.proj[0](rows(index, 0)) * level0[..., None] + self.proj[1](rows(index, 1)) * level1[..., None]
    pad = index.new_full((index.shape[0], 1), -1)
    previous = torch.cat([pad, index], 1)[:, :length]
    before = torch.cat([pad, pad, index], 1)[:, :length]
    closes = ((input_ids.to(index.device) == self.end_token_id)
              & (before >= 0) & (before < CODES_PER_LEVEL) & (previous >= CODES_PER_LEVEL))
    return delta + self.proj[2](rows(before, 0) + rows(previous, 1)) * closes[..., None]
```
<!-- /code -->

### 3.5 注意力路由损失（写、读）
- **监督对象：**block 内真正送进注意力的 query/key 的 pre-softmax 内积。
  - 按 block 的前向重算：norm1 → 时间步调制 → `to_q`/`to_k`（含 LoRA）→ RMS norm → RoPE；
  - 再对头和 ⟨M⟩ 的 4 个 token 取平均；
  - 用 forward pre-hook 读取，不改前向；与非重入的梯度检查点配合（检查点在反传时重跑钩子，所以 `router.disarm()` 放在反传之后）。
- **两个方向：**
  - 写：每个目标 token 对 ⟨M⟩ 的平均 logit；正例是区域覆盖率 > 0.5 的目标 token；
  - 读：⟨M⟩ 对每个源图 token 的 logit；正例在源图 latent 网格上。
  - 一个区域如果没有 token 覆盖率 > 0.5，就取覆盖率最大的 token 作正例，所以小于一个 token 的区域也有正例（`token_labels`）。
- **层与标定：**
  - 监督层是 `(6, 10, 14, 18, 22, 26)`；
  - 每层每个方向一对标定参数 (scale, bias)：`calibrated = scale·(logit − mean)·4 + bias`。
- **损失：**带权二元交叉熵。正例和反例各占总权重的一半，小区域因此不会被淹没；"重反例"的权重乘以 `heavy_weight`（C1b 起为 8）。
  - 写方向的重反例：邻居、干扰、孪生区域，以及同一句里其他 ⟨M⟩ 的区域；
  - 读方向的重反例：其他 ⟨M⟩ 的源图区域，以及源图网格上的邻居区域。
- **系数：**`λ_route = λ_read = 0.05`，不乘时间步权重。
- **代码：**`src/samtok_edit21/training/routing.py`（`route_targets`、`span_attention_logits`、`span_read_logits`、`AttentionRouter`），以及 `src/samtok_edit21/training/round2.py`（`token_labels`、`route_entries`）。
- **为什么：**直接监督"哪些目标 token 该去看 ⟨M⟩"和"⟨M⟩ 该去看源图的哪里"。⟨M⟩ 对干净源图的注意力在所有去噪步都一样，所以读方向的监督很稳定。

### 3.6 指认任务
- **任务：**让模型在 ⟨M⟩ 所指的区域上叠一层 0.5 透明度的颜色，区域外与源图逐像素相同。
  - 输出本身就是区域，损失直接关于 token；
  - 同图有多个实例时，不读 token 就降不下损失。
- **写法：**协议 2.0。no-ref 写成 `the selected region ⟨M⟩`，ref 写成 `the <category> ⟨M⟩`。
- **单 token 与多 token：**多 token 指认一次 2–5 个区域，各用 6 种高区分度颜色中的一种。区域互不重叠，并排除宿主、部件和重复标注。
- **区域池：**SA-1B dense mask，按面积桶抽样；< 0.5% 的区域约占 45%，贴近 benchmark 的区域大小。
- **读回：**把"输出 − 源图"投影到"颜色 − 源图"的方向上得到透明度，再阈值化成 mask。在完美的叠色上，读回 IoU 为 0.99。
- **代码：**`src/samtok_edit21/data/highlight.py`；第二轮的行由 `src/samtok_edit21/preparation/round2_data.py` 生成。
- **参考：**InstructDiffusion 用同样的形式做指代分割。区别是这里的指代对象是 SAMTok token，并用"同句换 token"来验证。

### 3.7 反例、反事实成对、近邻硬反例与孪生
- **近邻硬反例（C1b 起）：**
  - 取框间距不超过图像对角线 5% 的任意类别标注物体，最近的至多 6 个，再加至多 2 个宿主；
  - 读、写两个方向都作为重反例，并用于邻居 FM 项。
  - 起因：C1 单 token 失败中有 35% 是涂到旁边的相似物，而这些近邻几乎没有被用作反例。
- **反事实成对（C1b 起）：**
  - 一对两行同图、同句、同模板、同色，只换 token，各以对方的区域为反例；
  - 两行放在同一张卡、同一次更新的相邻两步（`place_in_slots`）。
- **多 token 相邻组（C1b 起）：**40% 的多 token 指认行围绕一对相邻区域构建。
- **孪生（C2 起）：**
  - 同一张源图上，`instruction_noref` 逐字相同的两条 50k 记录，按框间距最近两两配对，强制用盲指代；
  - 对方的源 mask 作重反例，一对放在同一次更新。
  - 代码：`round2_data.py` 的 `edit_negatives`、`pair_within`。
- **副作用：**50k 的删除行 92% 是孪生，attribute 行 28%（文档 03 的 C2 分析）。

### 3.8 编辑损失

```
loss = w(t)·( FM + λ_region·FM_region + λ_distractor·FM_distractor ) + λ_route·write + λ_read·read
```

- **时间步：**与 DiffSynth 官方的 `FlowMatchSFTLoss` 完全一致。
  - 在 1000 个训练时间步中均匀抽样（Qwen-Image 的 shift 模板）；
  - `x_t = (1−σ)·x0 + σ·ε`，回归目标是 `ε − x0`；
  - `w(t)` 是以 t = 500 为中心的高斯权重。
  - 代码直接调用调度器的 `add_noise`、`training_target`、`training_weight`。
- **FM：**逐 token 平方误差的全图平均。样本带 `loss_mask` 时只在 mask 内平均：目标图在编辑区外也有改动的 v1/D4 行，只在编辑相关像素外扩 2 个 token 的范围内算，不让模型学整图漂移。
- **FM_region：**区域内的平方误差，按覆盖率加权、按面积归一。
  - 编辑行的区域是实际改动的像素（`edited_region`），外扩 1 个 token 作上下文（`--edit-region-dilate 1`）；
  - 多个区域先各自归一再平均（`region_error`）；
  - 多目标行的每个单元用 `unit_change` 切出自己的改动。
- **FM_distractor：**邻居、孪生区域（覆盖率 > 0.25）上的平方误差均值。这些位置的目标就是源图，意思是"别的不许动"。
- **系数：**`λ_region = λ_distractor = 1`，`λ_route = λ_read = 0.05`。
- **代码：**`src/samtok_edit21/training/round2.py`（`prepare_round2`、`round2_loss`）：

<!-- code: src/samtok_edit21/training/round2.py::round2_loss -->
```python
def round2_loss(pipe, sample, router=None, *, lambda_region=1.0, lambda_distractor=1.0, timestep_index=None,
                noise=None, checkpointing=True, heavy_weight=DISTRACTOR_WEIGHT, edit_region_dilate=True):
    """Returns ``(flow, write, read, metrics)``; the caller adds ``lambda_route * write + lambda_read * read``.
    With a router, call ``router.disarm()`` after backward (gradient checkpointing re-runs the hooks)."""
    inputs = sample.inputs
    index = (torch.randint(0, len(pipe.scheduler.timesteps), (1,)) if timestep_index is None
             else torch.tensor([timestep_index]))
    timestep = pipe.scheduler.timesteps[index].to(device=pipe.device, dtype=pipe.torch_dtype)
    clean = inputs["input_latents"]
    noise = torch.randn_like(clean) if noise is None else noise
    noisy = pipe.scheduler.add_noise(clean, noise, timestep)
    target = pipe.scheduler.training_target(clean, noise, timestep)

    def predict():
        return pipe.model_fn(dit=pipe.dit, latents=noisy, timestep=timestep, **inputs, kv_cache=None,
                             use_gradient_checkpointing=checkpointing, **ATTENTION)

    write = read = None
    if router is None or not sample.spans:
        prediction = predict()
    else:
        router.arm_entries(sample.regions[0].numel(), route_entries(sample, heavy_weight))
        with router.recording():
            prediction = predict()
        write, read = router.collect(), router.collect_read()
    if prediction.shape != target.shape:
        raise RuntimeError("FM prediction/target shape mismatch")
    squared_error = (prediction.float() - target.float()).pow(2).mean(1)[0]
    mean = squared_error.mean() if sample.loss_mask is None else \
        (squared_error * sample.loss_mask).sum() / sample.loss_mask.sum().clamp(min=1.0)
    dilate = sample.region_dilate if sample.region_dilate is not None else sample.kind != "edit"
    if not edit_region_dilate and sample.kind in EDIT_KINDS:   # stage C: only the tokens that really change
        dilate = False
    in_region = region_error(squared_error, sample.regions, dilate=dilate)
    other = (sample.distractors > DISTRACTOR_COVERAGE).float()
    on_distractors = (squared_error * other).sum() / other.sum() if float(other.sum()) > 0 else mean.detach() * 0
    weight = pipe.scheduler.training_weight(timestep).to(pipe.device)
    flow = (mean + lambda_region * in_region + lambda_distractor * on_distractors) * weight
    metrics = {"mse": float(mean.detach()), "reg": float(in_region.detach()), "dis": float(on_distractors.detach()),
               "t": float(timestep), "write": float(write.detach()) if write is not None else None,
               "read": float(read.detach()) if read is not None else None}
    return flow, write, read, metrics
```
<!-- /code -->

- **C3-M 的教训：**C3-M 去掉了外扩那一圈（`--edit-region-dilate 0`），不执行率下降，但保持显著变差（P −0.066）。原因是区域项不再加权改动周围没变的部分，改动就溢出到旁边。所以 C3-D 和 v4 保留外扩（文档 03）。

### 3.9 多目标提示与拼合
- **写法：**与 benchmark v2 的单元编号一致。
  - 第一行 `Execute all numbered instructions and keep everything else unchanged.`；
  - 之后每行一条 `U{k}: <句子>`，每行一个 ⟨M⟩，单元顺序随机打乱（`round2_data.py::multi_edit_prompt`）。
- **数据来源：**
  - 50k 同图拼合：每组 2–4 个操作不同、执行区域不相交的单元，联合目标 = 源图 + Σ（各单条目标 − 源图）（`composite_group`）；
  - 合板数据里的真实多目标记录（D4、D10/D11 的多单元记录）。
- **监督：**每个 ⟨M⟩ 的路由以其他单元的区域为重反例；每个单元区域的误差分别归一后再平均。

### 3.10 指令写法（协议 2.0）
- **no-ref（盲指代）：**一律写成 `the selected region ⟨M⟩`。源 mask 是实际被编辑的区域；add 时是附着处。
- **ref（带名）：**⟨M⟩ 接在被指代对象（add 时是锚点）之后。
- **代码：**`src/samtok_edit21/preparation/instruction_protocol_v2.py`（`render(spec, has_ref, selector_token=span)`），与 benchmark 2.6.0 的共享模块逐字节相同。
- **训练行的写法比例：**C2 起，盲指代 + token 占 65%，带名 + token 占 35%；孪生强制用盲指代，实际约为 70/30。

### 3.11 推理时的 token 引导
- **做法：**CFG 的负分支换成去掉全部 ⟨M⟩ 的同一句（`src/samtok_edit21/data/protocol.py::strip_tokens`）：`v = v_neg + w·(v_pos − v_neg)`。只对带 token 的提示生效。
- **意义：**放大"有 token"与"没 token"的差别，也就是"在 token 所指之处做这个编辑"。每步多一次前向。
- **效果（文档 03）：**
  - C2 + 引导 2.0：整图全过 +6.0 个点；
  - C3-D + 引导 2.0：单元 E +0.032（显著），单元 P −0.024（显著），整图全过 +3.0（不显著），是目前最好的系统。
- **OPSD 的老师也用它**，正分支是带名 + token 的指令（§5.3）。

### 3.12 只给文字的单元：第一遍生成 token
- **做法：**benchmark 里没有定位的单元（ref + none），先让原始 SAMTok TE 按 `Can you segment {ref}?` 生成 token（它本来就是 SAMTok 的分割用法），再按 ref + token 读；没有生成有效 token 时保持纯文字（`benchmark_token.py::ground`）。
- **效果：**C2 上只给文字的单元 E 从 0.69 提到 0.86（文档 03）。
- 用户定的后续方向是用 TE LoRA 从纯文字指令生成 token，尚未开始。

### 3.13 其他训练细节
- **画布：**训练最大 1 MP，只缩不放，宽高取 32 的倍数（`src/samtok_edit21/data/io.py::dimensions`，`--max-pixels`）。OPSD 行用推理画布 `canvas`（§5.5）。
- **各卡同步初始化：**模型建好后，把 0 号卡的全部可训练参数广播到所有卡；开始和每次保存时核对各卡参数的校验和（`src/samtok_edit21/training/distributed.py`：`broadcast_parameters`、`rank_divergence`）。
- **梯度平均：**全部可训练参数的梯度拼成一条向量，做一次 all-reduce（`average_gradients`）。
- **批次顺序：**
  - 第 `(u·acc + s)·world + r` 行由第 r 卡在第 u 次更新的第 s 步处理；
  - 所以 8 卡 × 累积 4 和 32 卡 × 累积 1 每次更新用到的样本相同；
  - 行序在建行时由 `place_in_slots` 排好（§5.8）。
- **优化：**
  - AdamW；LoRA 用权重衰减 0.01，码本注入和路由仿射不用；
  - 学习率线性预热后余弦退火到 `lr_min`（`src/samtok_edit21/training/round2_engine.py::learning_rate`）；
  - 第 r 卡的时间步和噪声用种子 `seed + r`。
- **保存与续跑：**
  - `adapter_<n>/`：`adapter.safetensors`、`adapter.json`（完整配方、路由仿射、各卡一致性）、`code_injector.safetensors`、`code_injector.json`（码本指纹）；
  - `resume.pt`：参数、优化器状态、更新次数，有 OPSD 行时还有老师权重；
  - `TRAIN_DONE`：完成标记；
  - `train_metrics.jsonl` 每 10 次更新一行；W&B 记录见 `src/samtok_edit21/training/tracking.py`。

## 4. C3-D：当前最好的训练配方

**权重链：**原版 → C1（2,000 次更新，取 @1500）→ C1b（800）→ C2（3,522，ARNOLD 四机）→ C3-D（1,500，本机 8 卡）。设置和结果见文档 03。

C3-D 的配方取自 `adapter.json` 的 `grounding` 字段：

| 项 | 值 |
|---|---|
| 起点 | `V3/runs/c2_4n_002/train/adapter_3522` |
| 训练行 | `V3/data/round2_20261016/rows_c3d.jsonl`。保持 C2 的类别比例，四分之一换成 D10 的定向子集：替换 5k、整体改属性 4k、写字 1.5k、多单元 1.5k；去掉"做了一半"的目标 |
| 更新次数 × 每次的行数 | 1,500 × 32（8 卡 × 累积 4） |
| 学习率 | 5e-5，预热 100 次，余弦退火到 5e-6 |
| 权重衰减 | 0.01（只对 LoRA） |
| LoRA | rank 32，alpha 32，dropout 0 |
| 损失系数 | `λ_region = 1`，`λ_distractor = 1`，`λ_route = 0.05`，`λ_read = 0.05`，`heavy_weight = 8` |
| 区域外扩 | `edit_region_dilate = 1` |
| 路由层 | 6、10、14、18、22、26 |
| 训练画布 | 最大 1,048,576 像素 |
| 种子 | 2000（第 r 卡用 2000 + r） |
| 时长 | 11.3 h，本机 8 × H100 |

## 5. OPSD：在线自蒸馏（v4 新增）

### 5.1 思路
- **用户 10-25 的设想：**
  - 训练时借用显式 mask 施加硬约束，推理时只走原生的区域 token 通路；
  - 老师和学生共用同一个 DiT；
  - 老师拿到 GT mask，尽量把编辑做对；
  - 学生只看区域 token，在自己采样的轨迹上逼近老师。
- **为什么要在线：**
  - SFT 只在 GT 轨迹上监督，那里的 `x_t` 由 GT 目标图加噪得到；
  - 推理时学生走自己的轨迹，犯的错（不执行、改错实例、溢出到邻居）会把状态带离 GT 轨迹，而 SFT 从不在这些状态上给出纠正；
  - OPSD 在学生自己访问到的状态上，由老师给出纠正方向。
- **针对的失败：**
  - 不执行：C2 开发集上带 token 不执行占 50%，C3-D 48%；
  - 盲指代弱，添加、写字弱。
  - 老师上限测试显示，特权信息恰好补在这几处（§5.3）。
- **数据上的好处（有前提）：**原则上 OPSD 的目标来自模型自己，不需要目标图。但现在的 A 用到了 `edited_region`（源图与目标图的差分）和执行区域，带名指令也来自标注；只有 mask 和指令的数据要另定 A（例如选区外扩），这样的 A 够不够用要另测。

### 5.2 参考的工作与开源实现，以及取舍

| 来源 | 做法要点 | 本实现的取舍 |
|---|---|---|
| OPSD（arXiv 2601.18734，LLM） | 同一模型，老师多看参考解答；学生自己采样，老师在学生轨迹上逐 token 给分布；前向 KL，逐项截断；老师冻结为初始权重；训练短，取最好的检查点 | 老师 = 同一 DiT + 冻结的起点权重 + 特权输入；学生自己采样；短训练，看检查点 |
| D-OPSD（arXiv 2605.05204，流模型） | 学生用推理采样器自己采样，老师在每个状态上前向一次；速度 MSE；EMA（0.9999）或冻结的老师稳定，用正在训练的权重当老师会崩；代价约为 SFT 的 4 倍 FLOPs | 速度误差（在 x0 空间，等价于乘 σ²，§5.4）；默认冻结老师，EMA 可选（`--opsd-teacher-ema`）；不用正在训练的权重当老师 |
| HPSD（arXiv 2608.13205） | 条件-状态失配：老师死守自己的特权内容，纠正方向是拉回它自己的内容 | 特权只给"在哪里、是什么"（区域、名字），不给目标图；列为风险（§5.10） |
| CFG 与在线蒸馏（arXiv 2607.24731，NBA/PDM） | 只匹配合成后的速度时，正负分支的误差会互相抵消；负分支含特权信息时会出问题 | 学生用单次条件预测去匹配老师引导后的速度，相当于把引导内化，推理可以不用 CFG；PDM 没有实现，留作消融 |
| DiffSynth-Studio 的 OPD（上游 `examples/qwen_image_21/model_training/special/opd/train_opd_lora.py`） | 同一 DiT 上，学生（LoRA 开）不求梯度地自己采样，随机取状态，老师（LoRA 关、CFG 4、更密的步）在这个状态上给目标，取 MSE | **骨架照它写：**无梯度采样 → 选状态 → 老师目标 → 学生 MSE。老师换成"冻结的 LoRA + 特权条件"的单次预测（这里不是步数蒸馏）。避开它的两个问题（见下） |
| D-OPSD 仓库的 `flux2-klein-edit-self-distill-gt-ref` | 编辑任务的 OPSD：老师把 GT 目标图作为最后一张参考图 | 我们的特权是 mask 和名字，不是目标图。目标图太强，本身也有瑕疵（方案文档 §3.6） |

**DiffSynth OPD 脚本的两个问题，本实现都避开了：**
- **选状态：**循环里 `idx = torch.randint(...)` 覆盖了按 `min_distill_idx` 不放回抽出的 `idx`，`min_distill_idx` 实际不起作用。本实现的 `pick_states` 在 σ ≥ `min_sigma` 的状态里不放回地抽。
- **反传时的权重：**每个状态先算学生（带梯度）再切到老师（LoRA 系数 0），`forward` 返回时 DiT 停在老师模式，反传在之后才发生。开梯度检查点时，反传会重算前向，用的却是老师的权重，梯度不对。本实现先算老师目标，再算学生的前向并立即反传，反传时 DiT 上就是学生的权重（§5.6）。

调研与方案的完整记录：`V3/analysis/opsd_plan_20261025/README.md`（第 9 节是开源实现调研）。

### 5.3 老师：T3 配置与上限测试的依据

老师是同一个 DiT 和 TE，权重是冻结的起点权重（C3-D），外加三项特权输入。这就是上限测试中的 T3 配置：

| 特权输入 | 做法 | 管什么 |
|---|---|---|
| 允许改动的区域 A | A = 每个单元的 `selection` ∪ `execution_region` ∪ `edited_region`；在 token 网格上外扩 2 格、高斯羽化（σ = 1 格）；A 外老师的干净估计取源图 latent | 保持：在学生溢出、改到邻居、多加东西的状态上，老师给出的纠正是"那里退回原样" |
| 带名指令 | 每个单元都写成带名 + token（`p2.render(spec, True, span)`），学生这一行是盲指代时也一样 | 认对目标 |
| token 引导 | `v_T = v_neg + w·(v_pos − v_neg)`，w = 2；负分支是带名指令去掉 token | 执行：做没做、做没做完 |

**上限测试**（`V3/analysis/opsd_teacher_20261025/`，决策子集 301 例，同一次判分，详见文档 03）：

| 系统 | 单元 E | 单元 P | 整图全过 |
|---|---:|---:|---:|
| T0 学生（C3-D，GT token） | 0.759 | 0.898 | 35.9% |
| T1 = T0 + A 外回填 | 0.734 | 0.927 | 32.2% |
| T2 = T1 + 带名指令 | 0.788 | 0.949 | 42.5% |
| T3 = T2 + token 引导 2.0 | 0.823 | 0.953 | 45.8% |

- **T3 对 T0：**整图全过 +10.0 个点 [+3.3, +16.3]；E、P 都显著提高。
- **三项分别的作用：**
  - 带名指令贡献最大；
  - 引导在回填之下再提高 E，P 不降；
  - 回填单独使用时 P 升、E 降：benchmark 的 S mask 贴得紧，删除和替换会在 mask 外留下残留。
- **对训练的含义：**A 要按数据的实际改动放宽，因此用"选区 ∪ 执行区域 ∪ 改动像素"再外扩，而不是只用选区。

### 5.4 目标与损失

**记号**（与 DiffSynth 的流匹配一致）：
- `x_t = (1−σ)·x0 + σ·ε`，速度 `v = ε − x0`；
- 一个速度预测对应干净估计 `x̂0 = x_t − σ·v`；
- `m ∈ [0,1]` 是 A 在 token 网格上的羽化 mask，`x_src` 是源图在目标网格上的 latent。

**老师的干净估计：**`x0_T = m·(x_t − σ·v_T) + (1 − m)·x_src`，对 `v_T` 停梯度。

**学生在一个状态上的损失：**`L = mean((x0_S − x0_T)²)`，其中 `x0_S = x_t − σ·v_S`。
- **A 内（m = 1）：**`L = σ²·mean((v_S − v_T)²)`，即乘 σ² 的速度 MSE。
- **A 外（m = 0）：**`L = mean((x_t − σ·v_S − x_src)²)`，学生要在一步之内回到源图。
  - 这等价于速度目标 `(x_t − x_src)/σ`，也就是管线里 `blend_outside_region` 的做法，只是把"每步改状态"换成了"训练时给目标"。
  - 但在 x0 空间里，误差就是学生偏离源图的实际幅度，不会因为 `1/σ` 在小 σ 时放大。所以不需要 LLM 版 OPSD 的逐项截断，再加上只监督 σ ≥ `min_sigma` 的状态。
- **一行的损失：**选出的 N 个状态取平均，乘 `λ_opsd`，再除以梯度累积步数，与普通行同一个口径。
- **代码：**`teacher_clean`、`student_loss`（§6.5）。

### 5.5 学生的轨迹与状态采样
- **策略：**学生就是推理时的模型，条件与推理完全相同：源图，加上这一行写法（盲指代或带名）的带 token 指令。
- **画布：**用推理画布 `generation.canvas`（约 1 MP，保持宽高比），而不是训练行的 `dimensions`，这样状态的分布就是推理时的分布。
- **调度：**
  - 复制一份管线的调度器（`deepcopy`，训练用的 1000 步调度不受影响），设为 K 步（默认 16）推理调度；
  - 用同样的 `dynamic_shift_len`（目标网格的 token 数），与 `QwenImage21Pipeline.__call__` 一致。
- **采样：**
  - 不求梯度，DiT 在 eval 模式（与推理一致，没有 LoRA dropout）；
  - 用前缀 KV 缓存；Euler 步就是调度器的 `step`；
  - 默认学生引导为 1（单次条件预测）。`--opsd-student-guidance > 1` 时按 token 引导采样，只改变状态的分布，学生被训练的仍然是条件预测。
- **选状态：**在 σ ≥ `min_sigma`（默认 0.1）的状态里不放回地均匀抽 N 个（默认 2），按时间顺序排列（`pick_states`）。
- **初始噪声：**`torch.randn`，各卡的种子不同。

已验证：同样的初始噪声下，这条轨迹与 benchmark 和开发集用的 `generation.generate` 逐位相同（§6.7）。所以学生被监督的状态，正是它推理时会走到的状态。

### 5.6 老师的权重、交换与梯度隔离
- **老师的权重：**DiT LoRA 和码本注入投影，也就是前向中起作用的全部可训练张量；路由仿射只在损失里用，不需要。
  - 默认冻结为起点权重：广播之后立刻拷一份（`TeacherWeights`）；
  - `--opsd-teacher-ema m`（0 < m < 1）时，每次优化后 `w_T ← m·w_T + (1−m)·w_S`；
  - 老师权重存进 `resume.pt`，续跑时恢复；没有老师权重的旧 `resume.pt` 不能续跑 OPSD（直接报错）。
- **交换：**`TeacherWeights.active()` 把老师权重拷进参数，退出时（包括出错时）拷回学生的值。老师和学生共用一份底座，只多存一份 LoRA 和注入，约 8,700 万参数。
- **顺序：**
  1. 学生采样（学生权重，无梯度）；
  2. 对每个选出的状态：
     - 换上老师权重，无梯度地算 `x0_T`，换回；
     - 学生带梯度前向，立即反传。
- **为什么这样排：**反传时参数里一定是学生的值，非重入梯度检查点的重算也就一定用学生的权重。每个状态的计算图用完就释放，N 个状态的激活不会同时占显存。
- **模式：**采样和老师目标都在 eval 模式（`evaluating`），学生前向在训练模式；一行做完 DiT 回到原来的模式。

### 5.7 一行 OPSD 的完整流程

```
opsd_backward(pipe, ctx, row, teacher, config, scale):
  sample ← prepare_opsd：读记录与源图；推理画布；源图 latent（条件）；A 的羽化 mask 与 x_src（目标网格）；
           学生指令（行的写法）、老师指令（每个单元带名）、老师负分支（老师指令去掉 token）
  scheduler ← deepcopy(pipe.scheduler).set_timesteps(K, dynamic_shift_len = h·w)
  无梯度、eval 模式：
    学生条件 ← encode(学生指令)；states ← rollout(学生, scheduler, noise, 学生引导)
    换上老师权重：老师正、负分支的条件 ← encode(老师指令)、encode(老师负分支)
  对 picked = pick_states(σ ≥ min_sigma, N 个) 中的每个 i：
    无梯度、eval 模式、换上老师权重：v_T ← guided(老师条件, states[i], t_i, w)；x0_T ← teacher_clean(...)
    学生条件 ← encode(学生指令)（带梯度，梯度进码本注入）
    v_S ← predict(学生条件, states[i], t_i, 梯度检查点)
    L_i ← student_loss(states[i], σ_i, v_S, x0_T)；检查有限；(scale·L_i/N).backward()
  返回 {"opsd": 平均损失, "sigmas": [σ_i]}
```

**代价**（1 MP，K = 16，N = 2，老师引导 2，单卡 H100）：
- 每行 29.1 s，峰值显存 41.4 GB；普通编辑行约 7 s。
- 一行约为：16 次无梯度采样前向 + 2 × 2 次老师前向 + 2 次学生前向和反传。

### 5.8 数据与混合
- **建行：**`round2-data rows-merged --spec` 的一个抽取项加上 `"opsd": true`，抽到的记录就成为 `opsd_merged` 行。
  - 单单元记录的行带 `form`；多单元记录的行带 `order` 和 `forms`，与 `edit_merged` 相同；
  - OPSD 行没有近邻反例和孪生，因为老师的 A 已经管住了保持。
  - 详见文档 01。
- **写法：**抽取项可以单独设写法比例（`forms`，提交 `0856f593`）。OPSD 行宜多用盲指代，因为"带名"是老师最起作用的特权。
- **混合：**OPSD 行和普通行在同一个行文件里按比例混合。普通行（GT 流匹配、区域项、路由项）作为锚，比例由抽取项决定。
- **行的排布：**
  - 一个 OPSD 行的代价约为普通行的 4 倍，而每次梯度平均都要等最慢的卡；
  - 所以 `place_in_slots` 让 OPSD 行占满整个"步对"：相邻两步、每张卡各 2 行，均匀分布在整个训练中；
  - 凑不满一个步对的余数（少于 2 × world 行）并入普通行，实际训练里可以忽略；
  - 没有 OPSD 行时，排布和随机数的使用与以前的行文件完全相同。
- **训练器：**`--lambda-opsd` 设权重；`train_metrics.jsonl` 和 W&B 记录 `opsd`（OPSD 行的平均损失）和 `opsd_rows`。

### 5.9 参数与默认值

| 参数 | 默认 | 说明 |
|---|---|---|
| `--lambda-opsd` | 1.0 | OPSD 行损失的权重 |
| `--opsd-steps` | 16 | 学生采样的步数（推理调度） |
| `--opsd-states` | 2 | 每条轨迹监督的状态数 |
| `--opsd-min-sigma` | 0.1 | σ 低于它的状态不监督 |
| `--opsd-guidance` | 2.0 | 老师的 token 引导（1 为关） |
| `--opsd-student-guidance` | 1.0 | 采样策略的 token 引导（1 为条件预测） |
| `--opsd-dilate` | 2 | A 的外扩（token） |
| `--opsd-feather` | 1.0 | A 的高斯羽化（token） |
| `--opsd-teacher-ema` | 0.0 | 0 为冻结老师；0–1 之间为 EMA 的动量 |

- 默认值对应上限测试的 T3（外扩 2、羽化 1、引导 2）和方案文档的建议（K = 12–16，每条轨迹 2–4 个状态）。
- **`--lambda-opsd` 要按混合比例设。**在 C3-D 处实测单行梯度范数（LoRA 与码本注入）的中位数：普通行 0.155（16 行），OPSD 行 0.017（6 行），相差约 9 倍。`λ_opsd = 1` 时，OPSD 在一半一半的混合里只占约十分之一的梯度，所以试训用 8。

### 5.10 风险与待定
- **老师在某些类型上不更强：**这些类型拿不到信号。上限测试里写字、添加提升明显，按类型的细分见文档 03。必要时只对有效的类型抽 OPSD 行。试训证实了这一风险，而且更严重：在训练行上，老师很少比学生好（§5.11）。
- **特权太强或条件-状态失配**（HPSD）：学生学不会，甚至被拉坏。对策是只给温和的特权（区域、名字、引导），用冻结老师，短训练，看检查点。
- **负分支带名字：**老师的负分支是"带名、无 token"的指令，学生看不到这个名字。学生只匹配合成后的速度，不单独匹配各分支。如果不稳定，再试 PDM（正分支和引导方向分开匹配），或去掉老师引导（`--opsd-guidance 1`）。
- **回填的边界：**A 外扩、羽化，要看对比图。
- **代价：**每行约为普通行的 4 倍。试训实测（本机 8 卡、每次更新 32 行、OPSD 行占一半）：每次更新约 71 s，普通训练约 27 s。
- **试训：**10-26 至 10-27，P1（OPSD）与 P0（同一批记录全做普通训练）对照，各 500 次更新，从 C3-D 起。结果和诊断见 §5.11、文档 03 §7。

### 5.11 试训结果与诊断（10-27）
**结果**（文档 03 §7）：
- P1（一半行做 OPSD）对 P0（同一批行全做普通训练），各 500 次更新：决策子集整图全过 −0.7 个点 [−5.3, +4.0]，单元 E、P 都不显著；开发集盲指代 IoU +0.016（显著），不执行率不变。
- 加引导 2.0 后，P1@500 为 40.3%，对 C3-D + 引导 +4.0 [−1.3, +9.3]，不显著；单元 P +0.017 [+0.000, +0.035]，边缘。
- 第二场对 P0 + 引导：+4.3 [−0.3, +9.0]，不显著；单元 E +0.006，单元 P +0.015。引导下多出的部分来自保持，不是执行。

**诊断**（`V3/analysis/opsd_pilot_20261026/diagnosis/`，76 条试训没用过的合板记录）：
1. **§5.4 的损失被 A 外主导。**
   - `mean` 取在整个 latent 上，而 A 占格子的中位比例只有 1.6%（D1）到 31%（添加）。
   - 在起点 C3-D 上，A 内只占损失的 7%–18%、梯度的 7%–26%。A 外每个元素的误差（学生轨迹偏离源图 latent）本身也比 A 内到老师的误差大，所以比单看面积更严重。
   - A 外的目标是 `x_src`，与老师无关，这一项实际上是"照抄源图"的保持损失。P1 学到的主要就是它：A 外误差降到 P0 的约 1/4；A 内在 σ ≥ 0.7 没有变化。
2. **老师的特权在训练行上很少起作用。**
   - 接管测试（学生先走 k 步，再由 T3 走完）：老师从头接手，只把不执行从 8/76 降到 5/76，输出离 GT 目标也不更近。
   - benchmark 上老师高 12.7 个点、盲指代 E 高 0.153。它的优势在小区域、同类多实例的盲指代上；合板记录大多是显著的大目标，学生已经做得到。
3. **执行在高噪声段决定。**
   - 合板记录上，老师在 σ ≥ 0.74 接手还能救回，σ ≤ 0.57 就救不回（只有 3 条可救）；开发集上一半以上的差距在 σ ≥ 0.73 定了，σ 0.59 接手仍能追回约四成（第 4 条）。
   - §5.5 的均匀选状态，约一半落在 σ ≥ 0.74（15 个可选状态里 7 个），但这一段 A 内的梯度占比也只有 14%–26%。
4. **开发集上，老师的优势是真实的（第 1 步，10-27）。**在 100 条小区域盲指代编辑上，不执行：学生 48、学生 + 引导 35、不带名的老师 31、T3 22。超出可部署引导的那部分（−13 个点，离 GT 目标近 4.5 灰度）主要来自带名（−9）。学生先走 8 步（σ 0.73），老师只能追回一半以下。所以 OPSD 的前提在 D1 这类数据上成立，试训的问题在选行和损失（文档 03 §7）。
5. **与外部评审意见的对应：**
   - 面积稀释：证实，而且更严重（第 1 条）；
   - 老师在学生状态上能否纠正：在有差距的记录上能（接管从未变坏），但这样的记录很少；
   - 双重引导、16 步对 40 步：没有单独测。

**修订方向（未实现，待用户定）：**
- **损失分开归一：**`L = Σ_j m_j·e_j / Σ_j m_j + β·Σ_j (1−m_j)·e_j / Σ_j (1−m_j)`，其中 `e_j = (x0_S − x0_T)²_j`，β 约 0.1，两项分开记录。A 外也可以改成不需要老师的保持损失，放到普通行上。
- **选行：**以 D1 为主，侧重删除、添加、改属性（替换上老师没有优势）。其他来源先过一遍学生和老师，只留老师明显更好的行；度量可以是一个高噪声状态上 A 内 x0 的差距，或者短接管。
- **选状态：**侧重 σ ≥ 0.73，其余放在中噪声段。
- **老师保留带名：**超出引导的优势主要来自带名；回填在开发集上作用不显著，但它管住保持，仍保留。
- **对照：**
  - 离线目标：老师目标算在加噪的 GT 状态上；
  - 去掉带名的老师；
  - 用这两组确认起作用的是哪一项特权。

## 6. 代码实现

### 6.1 基于哪些开源代码，改了什么

| 依赖 | 版本 | 放在哪里 | 本项目做了什么 |
|---|---|---|---|
| DiffSynth-Studio | 2.1.12，commit `60b709c4` | `third_party/diffsynth/`（随仓库提供，以独立包 `diffsynth` 安装） | 上游原样，只在 `third_party/diffsynth/diffsynth/pipelines/qwen_image_21.py` 加了两个默认关闭的钩子：`blend_mask`/`blend_latents` 与 `blend_outside_region`（latent 融合，只用于老师上限测试），以及 `PromptEmbedder.process(..., return_token_ids=False)`（返回裁掉系统提示后的 token id，用来定位 ⟨M⟩）。说明见 `third_party/README.md`，版本记录见 `upstream_versions.json` |
| SAMTok | 官方源码 | `third_party/samtok/`（包名 `samtok`） | 源码和码本都没改；只用 `samtok.models`（VQ-SAM2 编解码器），由 `src/samtok_edit21/models/codec.py` 封装 |
| Qwen-Image-2.1 | `Qwen/Qwen-Image-2.1` 的 `b3179ad3` | 权重在 `/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen-Image-2.1` | DiT、VAE、processor 原样使用 |
| Qwen3-VL-8B-SAMTok | `zhouyik/Qwen3-VL-8B-SAMTok` 的 `b78aef11` | 权重在 `/mnt/bn/strategy-mllm-train/user/tanyue/models/SAMTok/Qwen3-VL-8B-SAMTok` | 替换原版 TE；分词器、对话模板换成它的；目录里同时带着 codec 的权重 |
| PEFT、transformers | peft 0.20.0，transformers 5.12.1 | pip 安装 | LoRA 注入；Qwen3-VL 模型 |

- **运行环境：**`scripts/round2/setup_env.sh` 固定为 Python 3.11、torch 2.8.0（cu128）、transformers 5.12.1、accelerate 1.14.0、peft 0.20.0，并检查 FlexAttention 可用。
- **方法代码：**全部在 `src/samtok_edit21/`，是一个独立的包（`samtok-edit21`），通过 DiffSynth 的公开接口使用它：
  - 管线：`QwenImage21Pipeline.from_pretrained`、`__call__`；
  - 单元：`QwenImage21Unit_PromptEmbedder`、`QwenImage21Unit_EditImageEmbedder`；
  - 模型函数 `model_fn_qwen_image_21`；
  - 调度器 `FlowMatchScheduler` 的 `set_timesteps`、`step`、`add_noise`、`training_target`、`training_weight`。
- **注意力路由**要读 DiT 内部的量，用的是 PyTorch 的 forward pre-hook，不改 DiffSynth 源码。

### 6.2 仓库结构

```
samtok_edit_qwen-image-2.1-v4/
├── src/samtok_edit21/            # 本项目的包
│   ├── api.py, __init__.py       # 惰性导出：load_pipeline, edit, generate, load_codec, load_adapter, load_grounding_adapter
│   ├── cli.py, __main__.py       # samtok-edit <命令>
│   ├── models/                   # 模型侧
│   ├── data/                     # 区域 token 协议、指认任务、读写工具
│   ├── preparation/              # 指令协议 2.0、合板、第二轮数据与训练行
│   ├── training/                 # 第二轮目标与训练器、注意力路由、OPSD、多卡工具、W&B
│   ├── evaluation/               # 只给 token 的生成、开发集探针与报告、benchmark 适配器
│   └── regions/                  # SAM2 点/框选区
├── third_party/                  # diffsynth（上游 + 两个钩子）、samtok（官方源码）及说明
├── scripts/round2/               # ARNOLD 入口、环境、运行设置
├── scripts/docs/                 # 文档检查：路径、代码块同步
├── tests/                        # CPU 单元测试（pytest）
├── docs/                         # 数据、方法、实验三份文档
├── pyproject.toml, requirements.txt, constraints-tested.txt, upstream_versions.json
```

| 模块 | 作用 |
|---|---|
| `src/samtok_edit21/models/pipeline.py` | 加载管线并替换 TE（`load_pipeline`、`SamtokTextEncoder`、`build_processor`）；指令编码（`encode_edit`）；源图缩放（`resize_sources`）；推理入口 `edit()`；`ATTENTION`；老师测试用的 `blend_inputs` |
| `src/samtok_edit21/models/codec.py` | SAMTok 编解码器：mask ↔ token（`SamtokCodec`、`load_codec`） |
| `src/samtok_edit21/models/code_injector.py` | 码本注入：模块、挂载、保存与加载、码本指纹 |
| `src/samtok_edit21/models/adapters.py` | DiT LoRA 的注入、保存、加载；`load_grounding_adapter`（LoRA + 码本注入，推理用） |
| `src/samtok_edit21/data/protocol.py` | 区域 token 的文本协议：`span_of`、`spans_in`、`token_layout`、`strip_tokens` |
| `src/samtok_edit21/data/highlight.py` | 指认任务：模板、目标图、读回 |
| `src/samtok_edit21/data/io.py` | `write_json`、`row_hash`、训练画布 `dimensions` |
| `src/samtok_edit21/preparation/instruction_protocol_v2.py` | 协议 2.0 的共享模块（与 benchmark 2.6.0 相同） |
| `src/samtok_edit21/preparation/merge_data.py` | 合板（`samtok-edit merge-data`） |
| `src/samtok_edit21/preparation/round2_data.py` | 开发划分、区域池与 token 缓存、各阶段训练行（C1–C3、合板数据 `build_rows_merged`）、行排布 `place_in_slots`、开发评测集 |
| `src/samtok_edit21/training/round2.py` | 各类行的样本准备（`prepare_round2`）、⟨M⟩ 位置查找、路由标签、`round2_loss` |
| `src/samtok_edit21/training/round2_engine.py` | 训练器（`samtok-edit round2-train`）：多卡、学习率、普通行与 OPSD 行、保存与续跑、快速评测 |
| `src/samtok_edit21/training/opsd.py` | **v4 新增：**OPSD（§5、§6.5） |
| `src/samtok_edit21/training/routing.py` | 注意力路由损失（写、读） |
| `src/samtok_edit21/training/distributed.py` | 参数广播、各卡一致性、梯度平均 |
| `src/samtok_edit21/training/tracking.py` | 多卡下的 W&B 记录 |
| `src/samtok_edit21/evaluation/generation.py` | 只给 token 的生成：`canvas`、`generate`、`export` |
| `src/samtok_edit21/evaluation/round2_eval.py`、`round2_report.py` | 开发集五类探针（`round2-eval`）与配对报告（`round2-report`） |
| `src/samtok_edit21/evaluation/benchmark_token.py` | benchmark v2 适配器（`ground`、`token_guidance`、`teacher`） |
| `src/samtok_edit21/regions/selection.py` | SAM2 的点/框选区 |

### 6.3 推理链路
- **入口：**`load_pipeline` 加载 DiT、VAE 和 SAMTok TE；`load_grounding_adapter(pipe, codec, 权重目录)` 加载 LoRA 和码本注入；然后用 `generate` 或 `edit` 生成。
- **`ATTENTION = {"use_flex_attention": True}`：**DiffSynth 2.1.9 起，Qwen-Image-2.1 默认走分段的 SDPA 注意力；本包的每次模型调用都传这个参数，留在训练权重时用的块因果 FlexAttention 路径上（§6.6）。

<!-- code: src/samtok_edit21/evaluation/generation.py::generate -->
```python
@torch.no_grad()
def generate(pipe, codec, prompt, source_rgb, *, steps=20, seed=0, area=CANVAS_AREA, cfg_scale=1.0, negative_prompt=None,
             blend_region=None, blend_dilate=2, blend_feather=1.0):
    """Token-only inference: inline region tokens, raw text encoder, CFG 1 (or token guidance: ``cfg_scale`` with a
    token-free ``negative_prompt``).  No blending, unless ``blend_region`` (a pixel mask, teacher test only) asks to
    keep the source outside it.  ``codec`` is unused, kept so that every caller keeps its signature."""
    from samtok_edit21.models.pipeline import edit

    height, width = canvas(source_rgb, area)
    extra = {} if negative_prompt is None else {"negative_prompt": negative_prompt}
    image, _ = edit(pipe, prompt, [source_rgb.convert("RGBA")], mode="inline", height=height, width=width,
                    num_inference_steps=steps, cfg_scale=cfg_scale, seed=seed, use_kv_cache=True, pass2_te="raw",
                    blend_region=blend_region, blend_dilate=blend_dilate, blend_feather=blend_feather, **extra)
    return export(image, source_rgb.size), (height, width)
```
<!-- /code -->

`edit()` 检查输入（inline 模式要求指令里有 token、只有一张源图；负提示不能带 token），把 `ATTENTION` 和其余参数交给管线。老师测试时，它还用 `blend_inputs` 准备融合用的 mask 和源图 latent。

### 6.4 训练链路
1. **建行：**`samtok-edit round2-data rows-merged --out <数据目录> --spec <spec.json>`（`build_rows_merged`），写出 `<name>.jsonl` 和 `<name>_stats.json`。行序就是训练顺序（`place_in_slots`）。
2. **训练：**`python -m torch.distributed.run --nproc_per_node 8 -m samtok_edit21 round2-train --data <数据目录> --rows <name>.jsonl ...`（`round2_engine.main`）。
   - **加载：**`load_pipeline`，调度器设为 1000 步训练调度；`--init-adapter` 加载 LoRA 和码本注入，否则新建 LoRA；挂上 `AttentionRouter`。
   - **同步：**广播参数，核对各卡一致；有 OPSD 行时拷出老师权重。
   - **每次更新、每个累积步：**
     - 取第 `(u·acc + s)·world + r` 行；
     - OPSD 行调 `opsd_backward`，其余调 `prepare_round2` → `round2_loss`，再反传 `flow + λ_route·write + λ_read·read`（乘 1/acc）；
     - 出错的行计入 `skipped` 并跳过，一张坏图不会中断长训练。
   - **之后：**`average_gradients` → `optimizer.step()` → 用 EMA 时更新老师 → 记录、保存、快速评测。
3. **评测：**`round2-eval`（开发集）、`round2-report`（配对报告）；benchmark 由 `benchmark_token` 生成，官方判分器打分（文档 03）。

### 6.5 OPSD 的代码

`src/samtok_edit21/training/opsd.py` 的结构：

| 名字 | 作用 |
|---|---|
| `OpsdConfig` | 参数（§5.9），构造时检查取值范围 |
| `TeacherWeights` | 老师权重：`active()` 交换，`update_ema()`，`state()` 存盘 |
| `prompts`、`allowed_region`、`prepare_opsd`、`OpsdSample` | 一行的准备：三条指令、A、推理画布上的源图条件、`x_src` 与羽化 mask |
| `encode`、`predict`、`guided`、`model_timestep` | 条件编码、一次 DiT 预测（同 `model_fn` 与 `ATTENTION`）、CFG 合成（同 DiffSynth 的 `cfg_guided_model_fn`）、时间步格式 |
| `evaluating`、`rollout`、`pick_states` | eval 模式；学生采样（同管线的 Euler 步与 KV 缓存）；选状态 |
| `teacher_clean`、`student_loss` | 老师的干净估计；学生损失 |
| `opsd_backward` | 一行 OPSD（§5.7） |

<!-- code: src/samtok_edit21/training/opsd.py::teacher_clean -->
```python
def teacher_clean(latents, sigma, velocity, mask, source_latents):
    """The teacher's clean estimate: its own inside the allowed region, the source outside (float32)."""
    estimate = latents.float() - sigma * velocity.float()
    return mask * estimate + (1 - mask) * source_latents.float()
```
<!-- /code -->

<!-- code: src/samtok_edit21/training/opsd.py::student_loss -->
```python
def student_loss(latents, sigma, velocity, target):
    """``mean((x0_S - x0_T)^2)`` with ``x0_S = x_t - sigma * v_S``."""
    return (latents.float() - sigma * velocity.float() - target).pow(2).mean()
```
<!-- /code -->

<!-- code: src/samtok_edit21/training/opsd.py::rollout -->
```python
@torch.no_grad()
def rollout(pipe, scheduler, positive, negative, noise, guidance):
    """The student's own trajectory: the latent at the input of every step of ``scheduler`` (Euler steps of the
    pipeline, with the prefix KV cache the pipeline uses at inference)."""
    blocks = len(pipe.dit.transformer_blocks)
    caches = ([{} for _ in range(blocks)], [{} for _ in range(blocks)] if guidance != 1.0 else None)
    latents, states = noise, []
    for timestep in scheduler.timesteps:
        states.append(latents)
        velocity = guided(pipe, positive, negative, latents, model_timestep(pipe, timestep), guidance, caches=caches)
        latents = scheduler.step(velocity, timestep, latents)
    return states
```
<!-- /code -->

<!-- code: src/samtok_edit21/training/opsd.py::TeacherWeights -->
```python
class TeacherWeights:
    """The teacher's copy of the trainable tensors that act in the forward pass (DiT LoRA and code injector).

    ``active()`` puts the copy into the parameters and restores the student's values on exit; ``update_ema`` moves the
    copy towards the student (momentum 0 is never used: a frozen teacher simply never updates)."""

    def __init__(self, parameters, weights=None):
        self.parameters = list(parameters)
        source = self.parameters if weights is None else list(weights)
        if len(source) != len(self.parameters) or any(w.shape != p.shape for w, p in zip(source, self.parameters)):
            raise ValueError("Teacher weights do not match the trainable parameters")
        self.weights = [w.detach().to(device=p.device, dtype=p.dtype).clone() for w, p in zip(source, self.parameters)]

    @contextmanager
    def active(self):
        backup = [p.detach().clone() for p in self.parameters]
        with torch.no_grad():
            for parameter, weight in zip(self.parameters, self.weights):
                parameter.copy_(weight)
        try:
            yield
        finally:
            with torch.no_grad():
                for parameter, value in zip(self.parameters, backup):
                    parameter.copy_(value)

    @torch.no_grad()
    def update_ema(self, momentum):
        if not 0 < momentum < 1:
            raise ValueError("EMA momentum must be in (0, 1)")
        for weight, parameter in zip(self.weights, self.parameters):
            weight.mul_(momentum).add_(parameter.detach(), alpha=1 - momentum)

    def state(self):
        return [w.detach().cpu() for w in self.weights]
```
<!-- /code -->

<!-- code: src/samtok_edit21/training/opsd.py::opsd_backward -->
```python
def opsd_backward(pipe, ctx, row, teacher, config, *, scale, checkpointing=True):
    """One OPSD row: the student's rollout, then for every picked state the teacher target and the student's
    backward of ``scale * loss / states``.  Returns the mean loss and the picked sigmas."""
    sample = prepare_opsd(pipe, ctx, row, config)
    scheduler = deepcopy(pipe.scheduler)            # the trainer's own schedule (1000 training steps) stays untouched
    scheduler.set_timesteps(config.steps, dynamic_shift_len=sample.source_latents.shape[2] * sample.source_latents.shape[3])
    noise = torch.randn(sample.source_latents.shape, device=pipe.device, dtype=pipe.torch_dtype)
    with torch.no_grad(), evaluating(pipe.dit):
        positive = encode(pipe, sample.student_prompt, sample)
        validate_conditioning({**positive, "input_latents": noise})
        negative = None if config.student_guidance == 1.0 else encode(pipe, strip_tokens(sample.student_prompt), sample)
        states = rollout(pipe, scheduler, positive, negative, noise, config.student_guidance)
        with teacher.active():
            teacher_positive = encode(pipe, sample.teacher_prompt, sample)
            teacher_negative = None if config.guidance == 1.0 else encode(pipe, sample.teacher_negative, sample)
    picked = pick_states(scheduler.sigmas, config.states, config.min_sigma)
    losses = []
    for index in picked:
        latents, timestep, sigma = states[index], scheduler.timesteps[index], float(scheduler.sigmas[index])
        with torch.no_grad(), evaluating(pipe.dit), teacher.active():
            velocity = guided(pipe, teacher_positive, teacher_negative, latents, model_timestep(pipe, timestep),
                              config.guidance)
            target = teacher_clean(latents, sigma, velocity, sample.mask, sample.source_latents)
        conditioning = encode(pipe, sample.student_prompt, sample)
        velocity = predict(pipe, conditioning, latents, model_timestep(pipe, timestep), checkpointing=checkpointing)
        loss = student_loss(latents, sigma, velocity, target)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite OPSD loss")
        (scale * loss / len(picked)).backward()
        losses.append(float(loss.detach()))
    return {"opsd": sum(losses) / len(losses), "sigmas": [round(float(scheduler.sigmas[i]), 4) for i in picked]}
```
<!-- /code -->

**训练器里的接入**（`src/samtok_edit21/training/round2_engine.py`）：
- `KIND_INDEX` 加了 `opsd_merged`，`--lambda-opsd` 和 `--opsd-*` 参数构成 `OpsdConfig`；
- 老师的参数是 LoRA 加上 `injector.proj`：广播之后拷出，续跑时从 `resume.pt` 的 `teacher` 恢复；
- 循环里 `opsd_merged` 行调用 `opsd_backward(..., scale=λ_opsd/acc)`，累计 `opsd` 和 `opsd_rows`；
- `optimizer.step()` 之后，EMA 老师调用 `update_ema`；
- 只含普通行的训练，路径和以前完全相同。

**行排布**（`src/samtok_edit21/preparation/round2_data.py`）：

<!-- code: src/samtok_edit21/preparation/round2_data.py::place_in_slots -->
```python
def place_in_slots(units, rng, updates, world, accumulation):
    """Rows in training order: global row (u * accumulation + s) * world + r is processed by rank r at step s of
    update u.  A pair occupies steps s, s + 1 (s even) of one rank, so both members share an update; singles fill
    the remaining slots two by two.  OPSD rows (kind opsd_merged) cost several other rows each, and the ranks wait for
    the slowest one at every gradient average: they fill whole step pairs, two rows on every rank, spread evenly over
    the run; a remainder short of a step pair joins the other singles.  Without OPSD rows the placement (and the use
    of ``rng``) is the one of the earlier rows files."""
    if accumulation % 2:
        raise ValueError("pairs need an even accumulation")
    pairs = [u for u in units if len(u) == 2]
    singles = [u[0] for u in units if len(u) == 1 and u[0].get("kind") != "opsd_merged"]
    heavy = [u[0] for u in units if len(u) == 1 and u[0].get("kind") == "opsd_merged"]
    steps = []                                      # OPSD step pairs: rows 2r, 2r + 1 go to rank r
    if heavy:
        rng.shuffle(heavy)
        full = len(heavy) - len(heavy) % (2 * world)
        steps = [heavy[k:k + 2 * world] for k in range(0, full, 2 * world)]
        singles += heavy[full:]
    rng.shuffle(singles)
    blocks = pairs + [singles[k:k + 2] for k in range(0, len(singles), 2)]
    rng.shuffle(blocks)
    groups = updates * accumulation // 2
    if len(blocks) + len(steps) * world != groups * world or any(len(b) != 2 for b in blocks):
        raise ValueError("units do not fill the slots exactly")
    at = [g for g in range(groups) if (g + 1) * len(steps) // groups > g * len(steps) // groups]
    opsd_at = {g: h for h, g in enumerate(at)}
    rows, k = [None] * (updates * world * accumulation), 0
    for u in range(updates):
        for r in range(world):
            for s in range(0, accumulation, 2):
                g = (u * accumulation + s) // 2
                if g in opsd_at:
                    block = steps[opsd_at[g]][2 * r:2 * r + 2]
                else:
                    block, k = blocks[k], k + 1
                rows[(u * accumulation + s) * world + r], rows[(u * accumulation + s + 1) * world + r] = block
    return rows
```
<!-- /code -->

### 6.6 DiffSynth 升级到 2.1.12 与等价性验证
- **版本：**从 2.1.8（`7686e54d`）升到 2.1.12（`60b709c4`）。
- **本地改动减到两处：**v3 在 DiffSynth 里有四处扩展：采样器、可微 FlexAttention、DiT 的区域绑定挂载点、返回 token id。前三处只为 v2 的方法服务，v4 不再需要。现在只剩管线里的两个钩子（§6.1），都是可选参数，不传时与上游逐位一致。
- **注意力默认值：**上游 2.1.9 把 Qwen-Image-2.1 的注意力默认改成分段 SDPA。v4 统一传 `use_flex_attention=True`（`ATTENTION`），保持训练时的数值路径。
- **验证**（v3 对 v4，升级前后各做一次，临时脚本不入库）：
  - 13 个训练行，覆盖全部 10 类：损失和梯度校验和逐位相同；
  - benchmark 适配器的 9 张输出（C3-D、C3-D + 引导 2.0、老师 T3，各 3 个任务）逐位相同；
  - 5 个开发集探针逐位相同。

### 6.7 OPSD 的正确性验证

在 GPU 上用 C3-D 权重和合板数据的真实记录验证，临时脚本不入库：

| 检查 | 做法 | 结果 |
|---|---|---|
| 恒等 | 老师 = 学生，没有特权：同一指令，引导 1，A 为全图 | 损失 0.0，梯度最大值 0.0 |
| T3 参考值 | C3-D，T3 老师，单例 | 损失 4.77e-4，梯度最大值 7.27e-5 |
| 隔离 | 改动学生权重后重算老师目标；看退出交换后学生的权重 | 目标不变；学生的值恢复 |
| 梯度检查点 | 同样的种子、状态和目标，开、关检查点各算一次（256×256 画布） | 两次损失都是 0.0018055，梯度逐位相同 |
| 在线性 | `opsd_backward` 内的学生轨迹，对比 `generation.generate`（评测用的同一路径），初始噪声相同，16 步 | 单单元和两单元的行、学生引导 1 和 2 四种情况下，每次 DiT 调用的输入、输出和最终 latent 都逐位相同；采样在 eval 模式、无梯度、用 KV 缓存 |
| 模式 | 一行前后的模式 | 学生前向在训练模式且带梯度；采样和老师在 eval 模式；结束后 DiT 回到训练模式 |
| 能降损失 | 固定一行、状态和老师目标，AdamW（lr 1e-4）走 8 步 | 损失 1.30e-3 → 3.86e-4（降到 0.30 倍） |
| 代价 | 1 MP，K = 16，N = 2，引导 2，单卡 H100 | 每行 29.1 s，峰值显存 41.4 GB |
| 8 卡冒烟 | 96 行（40 行 OPSD）。冻结老师先跑 2 次更新，再 `--resume` 到 3；EMA 0.99 跑 3 次 | 都没有跳过的行；OPSD 平均损失为 0.0067（冻结，前 2 次）、0.00181（续跑的第 3 次）、0.00532（EMA，3 次）；写出了 `adapter_2`、`adapter_3`、`resume.pt`、`TRAIN_DONE` |
| 续跑后的老师 | 对比 `resume.pt` 里的老师和起点权重 | 冻结老师与 C3-D 相同：454 个张量的和与平方和一致；EMA 老师已经移动 |
| 行排布 | 用 200 组随机设置对比新旧 `place_in_slots` | 没有 OPSD 行时，行序和随机数状态都与旧版相同；有 OPSD 行时，每个 OPSD 步对在所有卡上整齐，并且均匀分布 |
| 8 卡冒烟（新排布） | 同一配置按整步对排布重新生成行，冻结老师跑 3 次更新 | 没有跳过的行，OPSD 平均损失 0.00508。冒烟规模下余数（8 行）占 OPSD 行的五分之一，耗时不代表实际训练 |
| 单元测试 | `tests/test_opsd.py` | 指令、A、目标与损失、选状态、老师权重、模式、引导、采样、行排布，共 9 项通过；全套 59 项通过 |

### 6.8 怎样运行

**安装**（见 `README.md`）：

```bash
uv venv --python /usr/bin/python3.11 /path/to/samtok21-env
uv pip install --python /path/to/samtok21-env/bin/python -r requirements.txt
```

**推理：**

```python
from PIL import Image
from samtok_edit21 import generate, load_codec, load_grounding_adapter, load_pipeline
from samtok_edit21.data.protocol import strip_tokens

pipe = load_pipeline(device="cuda")
codec = load_codec("/mnt/bn/strategy-mllm-train/user/tanyue/models/SAMTok/Qwen3-VL-8B-SAMTok", "cuda")
load_grounding_adapter(pipe, codec, "<C3-D 权重目录>")
source = Image.open("source.png").convert("RGB")
token = codec.encode(source.convert("RGBA"), [mask])[0][0]          # 布尔 mask -> ⟨M⟩
prompt = f"Replace the selected region {token} with a navy polo shirt."
image, _ = generate(pipe, codec, prompt, source, steps=40, seed=0,
                    cfg_scale=2.0, negative_prompt=strip_tokens(prompt))   # token 引导 2.0
```

**OPSD 训练行：**一个 spec 例子（冒烟用，写法见 `build_rows_merged` 上方的注释和文档 01；试训的配置是 `V3/analysis/opsd_pilot_20261026/spec_p1.json`，其中 OPSD 抽取项另设了 `"forms"`）：

```json
{"name": "rows_opsd_smoke", "seed": 7, "updates": 3, "world": 8, "accumulation": 4,
 "draws": [{"family": "D11", "op": "replace", "rows": 24, "opsd": true},
           {"family": ["D10", "D11"], "op": "multi", "rows": 8, "opsd": true},
           {"family": "D1", "rows": 8, "opsd": true},
           {"family": "D11", "op": "add", "rows": 16},
           {"rest": true}],
 "highlights": {"single": 0.06, "multi": 0.02}, "joined_multi": 0.04}
```

```bash
python -m samtok_edit21 round2-data rows-merged --out <数据目录> --spec spec.json
```

**OPSD 训练**（本机 8 卡，从 C3-D 起；数据目录要有 `round2-data pool` 的产物，以及 `eval_c1.json` 和快速评测项）：

```bash
python -m torch.distributed.run --nproc_per_node 8 -m samtok_edit21 round2-train \
  --data <数据目录> --rows <name>.jsonl --quick-items eval_c2_quick_idx.json --output <run>/train \
  --init-adapter /mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v3/runs/c3d_local8_001/train/adapter_1500 \
  --updates <N> --accumulation 4 --lr 5e-5 --warmup 20 --lr-schedule cosine --lr-min 5e-6 --heavy-weight 8 \
  --lambda-opsd 8 --opsd-steps 16 --opsd-states 2 --opsd-guidance 2.0 --opsd-teacher-ema 0
```

- 其余参数见 `samtok-edit round2-train --help`。
- **四机：**把同样的参数写进 `scripts/round2/bootstrap_arnold.sh` 的 `SAMTOK_TRAIN_ARGS`（运行设置的写法见 `scripts/round2/settings/`）。行文件的 `world` 和 `accumulation` 要与实际运行一致，OPSD 的整步排布才对齐。
- **试训的完整设置：**`V3/runs/opsd_p1_local8_001/scripts/train.sh`（P1）、`V3/runs/opsd_p0_local8_001/scripts/train.sh`（P0）。

### 6.9 测试
- **运行：**`PYTHONPATH=src:third_party/diffsynth:third_party python -m pytest -q tests/`，共 59 项，全部在 CPU 上：
  - `test_opsd.py`：OPSD；
  - `test_round2.py`：第二轮目标、路由与行；
  - `test_grounding.py`：指认、注入、路由、画布；
  - `test_merge_data.py`：合板；
  - `test_stage_c_rows.py`：C3 的行；
  - `test_protocol.py`：协议；
  - `test_benchmark_token.py`：benchmark 适配器；
  - `test_docs.py`：文档代码块与源码一致。
- **文档检查：**
  - `python scripts/docs/sync_code_blocks.py` 从源码重写文档里的代码块，`--check` 只检查（测试会跑）；
  - `python scripts/docs/check_paths.py` 检查文档里的路径是否存在（需要共享盘）。

## 7. 方法演进

| 阶段 | 相对上一阶段改了什么 | 为什么 |
|---|---|---|
| E-A1 | 原版 + 新 LoRA；单 token 指认 + 程序化局部编辑 + 少量真实编辑；区域 FM 项 | 验证"指认任务能不能让 DiT 读 token"（P1、P3） |
| E-A2 | 加码本注入、写方向路由、邻居 FM 项 | E-A1 只到部分可用（同句换 token 跟随 0.53）；需要更直接的监督（P10） |
| C1 | 协议 2.0 写法；读方向路由；多 token 指认；SA-1B dense mask 区域池；从原版重训 | benchmark 区域小、一图多目标、四种定位方式 |
| C1b | 近邻硬反例（×8）；30% 反事实成对；余弦学习率 | C1 失败中 35% 是近邻 |
| C2 | 编辑为主，用全部可用编辑数据；区域项改为实际改动像素；局部损失；多目标编辑；天然孪生 | 编辑能力至少不降，尽量用全数据 |
| C3-D | 四分之一换成 D10 定向数据（替换、整体属性、写字、多单元）；去掉做了一半的目标 | C2 输给原版和 FLUX.3 的地方 |
| 推理：token 引导、第一遍 token | CFG 负分支为去掉 token 的同一句；只给文字的单元先由原始 TE 生成 token | 不执行；只给文字的单元弱 |
| **v4：OPSD** | 学生自己采样，冻结老师 + 特权条件（A、带名、引导）在学生的状态上给目标 | 老师上限比学生高 10 个点；SFT 不纠正学生自己的错误状态 |

## 8. 参考工作

| 工作 | 是什么 | 借鉴了什么 / 区别 |
|---|---|---|
| SAMTok（arXiv 2601.16093） | 把任意 mask 编成 2 个词，用于多模态大模型的理解和分割 | 区域表示和 codec 的来源；它不涉及生成侧怎样使用这些词 |
| InstructDiffusion（arXiv 2309.03895） | 把指代分割写成"给物体叠半透明色块" | 指认任务形式的先例；它用语言指代，目的是分割 |
| Emu Edit（arXiv 2311.10089） | 多任务联合训练 | 感知任务和编辑混训的证据 |
| TokenCompose（CVPR 2024） | 用分割 mask 监督名词 token 的交叉注意力 | 注意力路由监督的先例；本方法针对离散的区域 token |
| RePlan（arXiv 2512.16864） | VLM 输出区域框，免训练地注入 MMDiT 注意力 | 区域作为输入；v2 的注意力偏置与它重叠，v3 起不再这样做 |
| OPSD（arXiv 2601.18734） | LLM 的在线自蒸馏：同一模型，老师多看参考解答 | §5 的出发点 |
| D-OPSD（arXiv 2605.05204） | 流模型的在线自蒸馏，速度 MSE、EMA/冻结老师 | 损失形式、老师权重 |
| HPSD（arXiv 2608.13205） | 视频模型自蒸馏，条件-状态失配 | 风险与特权强度的取舍 |
| NBA/PDM（arXiv 2607.24731） | CFG 与在线蒸馏 | 引导内化的风险；PDM 作为备选 |
| DiffSynth-Studio OPD | Qwen-Image-2.1 的在线步数蒸馏脚本 | OPSD 的实现骨架 |

## 9. 已知局限与下一步

| 局限 | 现状（文档 03） | 下一步（未执行，待用户确认） |
|---|---|---|
| 不执行 | C3-D 开发集带 token 不执行 48%；引导能补一部分；OPSD 试训没有降低（P1 46%，P0 46%） | 按老师与学生的差距选 OPSD 行、改损失后再试（§5.11） |
| 盲指代、添加、写字弱 | 老师的提升集中在这几类；试训里 P1 的盲指代 E 没有提升 | 老师的优势在合板训练行上很少出现，要选小区域、同类多实例的行（§5.11） |
| 大面积属性修改 | 0.90 的目标没有一个部署系统达到（C3-MD 0.885、P1@500 + 引导 0.885 最接近） | 待定 |
| 换 TE 的代价 | 原版只换 TE 时，替换和写字明显变差 | 老师用原版 TE 作为 OPSD 的后续选项（方案文档 §3.5） |
| 判分噪声 | 跨场逐例的整图全过一致率 78%，总数稳定 | 比较放在同一次判分里做 |
| 第一遍 token | 原始 TE 生成，已是 benchmark 默认 | TE LoRA 从纯文字生成 token，是用户定的后续方向 |
