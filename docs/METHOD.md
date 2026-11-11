# SITEdit（SAMTokEdit）最终方法：设计、代码实现与权重

更新：2026-11-11。本文描述**当前最终交付版本**（论文里的 SITEdit，代号 R2@1000）。数据见 [DATA.md](DATA.md)，从 v4 开始的全部实验见 [EXPERIMENTS.md](EXPERIMENTS.md)。

**路径记号**
- `NAS/…` = `/mnt/bn/strategy-mllm-train/user/tanyue/…`
- `SE/…` = `NAS/experiments2/SAMTokEdit/…`
- `V3/…` = `SE/qwen21_v3/…`（v4 及以前的实验根，S1、S2 在这里）
- `QA/…` = `SE/quality_anchor_20261104/…`（11-04 之后的画质实验根，R2 在这里）
- 仓库内路径（`src/…`、`scripts/…`）相对本仓库分支 `qwen-image-2.1-r2-delivery` 的根目录。

---

## 0. 一页概要

- **任务：**交互式、细粒度、多目标的图像编辑。用户在一张图上给出 1–5 个编辑单元，每个单元一条指令，加上可选的区域提示（无、点、框或 mask）。要求所有单元一起完成，其余内容保持不变。
- **核心思想：**把每个编辑区域编码成 SAMTok 的离散区域 token（Space Token，4 个特殊 token），直接写进该单元的子指令。Qwen-Image-2.1 的 DiT 只凭这些 token 定位和编辑。推理时不用像素 mask，不加注意力偏置，也不做 latent 融合或贴回。mask 只在训练时作为监督出现。
- **两轮推理：**
  - **Pass1（定位）：**Qwen3-VL-8B-SAMTok 加一个定位 LoRA，一次自回归调用，为全部单元输出区域 token。输入可以是文字、点、框，或已给的 mask。
  - **Pass2（编辑）：**Qwen-Image-2.1 DiT 加本方法的 DiT LoRA 和码本注入器（CodeInjector），文本编码器换成原版 SAMTok TE。一次联合生成完成全部单元，并用 token 引导（γ=2）。
- **训练：**DiT 部分训练三个阶段，Pass1 单独训练。

  | 阶段 | 初始化 | 训练行 | 更新 × 行/次 | 学习率 | 关键损失差异 | 权重 |
  |---|---|---|---|---|---|---|
  | S1：学读 token | 原版（新 LoRA） | `rows_s1.jsonl`：80% 高亮读回 + 20% SA-1B token 编辑 | 6,000 × 32 | 1e-4，余弦到 1e-5，预热 50 | 高亮行与编辑行都加区域项 | `V3/runs/s1_4n_001/train/adapter_6000` |
  | S2：学编辑（即 v4） | S1@6000 | `rows_s2.jsonl`：90% token 编辑（多来源、多目标 25%、孪生）+ 10% 高亮回放 | 16,000 × 32 | 5e-5，余弦到 5e-6，预热 100 | 同 S1 | `V3/runs/s2_4n_001/train/adapter_16000` |
  | **R2：补画质（最终）** | S2@16000 | `rows_v6_s2hqfull.jsonl`：80% token 编辑（非孪生全部用高质量合成目标）+ 10% 无 token 普通编辑 + 10% 高亮 | **1,000 × 32 = 32,000 行**（日程按 7,000 次设计，取第 1,000 次的检查点） | 8e-6，余弦到 8e-7，预热 100（第 1,000 次时约 7.70e-6） | **高亮行的区域项为 0** | **`QA/runs_v5/v6_full/r2/train/adapter_1000`** |
  | Pass1 定位 LoRA | Qwen3-VL-8B-SAMTok（冻结） | Pass1 正式数据（warmup 31,973 + main 127,888 个请求） | 见 §4.4 | 峰值 2e-5 | 只训 LM LoRA，NTP 交叉熵 | `SE/pass1_localization_lora_formal_20261030_0939/runs/formal/step_2000` |

- **最终结果（详见 EXPERIMENTS.md §9–§10）：**
  - **IF-Bench 全量**（1,084 张图 × 10 种 setting，canonical Judge）：十种平均 UnitPass 80.3%、JointPass 49.9%。v4 为 81.1% / 50.3%，两者配对差在判分噪声内。最强基线 Qwen-Image-2.1 为 64.0% / 29.4%。
  - **MIRA-Bench 全量**（200 张图，GPT-5.4）：PF 7.88、Cons 7.83，均高于原版（7.44 / 7.76）。同时段重判的 PQ 为 R2 7.19、原版 7.21，差 −0.02。v4 的 PQ 比原版低约 1.0。
  - **MagicBrush、CompBench：**MagicBrush 五项指标都显著好于 v4。CompBench 的 LC-T 略好；LC-I、PSNR、SSIM 普遍略低于 v4，属于小幅差距。

---

## 1. 输入、输出与指令协议

- **区域 token（SAMTok，mask 语义 2.0）：**`<|mt_start|><|mt_XXXX|><|mt_YYYY|><|mt_end|>`。
  - 两级残差量化码本，各 256 个码，第二级编号加 256。
  - 编解码器是 `NAS/models/SAMTok/Qwen3-VL-8B-SAMTok` 中的 `mask_tokenizer_256x2.pth`，代码在 `src/samtok_edit21/models/codec.py`。
  - token 解码出的区域（`dec`）就是 token 的实际含义，也是训练时的监督区域。
- **指令协议 2.0**（`src/samtok_edit21/preparation/instruction_protocol_v2.py`）：
  - **带名（ref）：**token 紧跟指代短语。例：`Replace the cardinal on the left <M> with a blue jay.`
  - **盲指代（noref）：**token 紧跟 `the selected region`。例：`Replace the selected region <M> with a blue jay.`
- **多单元提示**（训练和推理一致，`round2_data.multi_edit_prompt`）：

  ```text
  Execute all numbered instructions and keep everything else unchanged.
  U1: {子指令 1，含区域 token}
  U2: {子指令 2，含区域 token}
  ```

- **无 token 普通编辑（`edit_plain`，R2 新增）：**同一种编辑，指令里不带 token。用来锚定画质，见 §3.2。

## 2. 模型部件

| 部件 | 内容 | 是否训练 | 代码 |
|---|---|---|---|
| 编辑器主干 | Qwen-Image-2.1 的 DiT 与 VAE，`NAS/models/pretrained_models/Qwen-Image-2.1` | 冻结，只加 LoRA | `src/samtok_edit21/models/pipeline.py`（DiffSynth 2.1.12，vendored 在 `third_party/diffsynth`） |
| 文本编码器 | 原版 SAMTok TE：Qwen3-VL-8B-SAMTok 替换原 TE，能读区域 token | 冻结 | `models/pipeline.py`（`load_pipeline`、`encode_edit`） |
| DiT LoRA | rank 32，alpha 32，dropout 0；作用于 `to_q / to_k / to_v / to_out.0 / proj / out / gate_layer` | 训练 | `src/samtok_edit21/models/adapters.py`（`add_adapter`、`load_grounding_adapter`） |
| 码本注入器 CodeInjector | 把区域 token 的两级码本向量（code_dim 256，2 级 × 256 码）经线性层投到 4096 维，加到 TE 隐状态里对应 token 的位置。码本 sha256 `00df2a27…bbf9e67` | 训练 | `src/samtok_edit21/models/code_injector.py` |
| 注意力路由 | 训练时的辅助损失，在 DiT 第 6/10/14/18/22/26 层监督"写"（图像 query 对 token key 的注意力落在区域里）和"读"（token query 读区域）。每层一个可学的仿射标定；推理时不起作用 | 训练（仿射参数） | `src/samtok_edit21/training/routing.py` |
| Pass1 定位 LoRA | Qwen3-VL-8B-SAMTok 的语言模型 LoRA：rank 64，alpha 128，dropout 0.05，fp32 LoRA，bf16 底座 | 训练（独立） | `src/samtok_edit21/pass1/`（`grounder.py`、`model.py`、`protocol.py`、`training/formal.py`） |

- DiT LoRA 与 CodeInjector 一起保存在 adapter 目录：`adapter.json`、`adapter.safetensors`、`code_injector.json`、`code_injector.safetensors`。
- `adapter.json` 的 `grounding` 字段记下完整训练配方、路由仿射参数（`route_affine`、`read_affine`）和多卡一致性检查值。

## 3. 训练目标

损失定义在 `src/samtok_edit21/training/round2.py` 的 `round2_loss`；训练器是 `src/samtok_edit21/training/round2_engine.py`。

总损失为

```
L = FM_whole + λ_region · FM_region + λ_distractor · FM_distractor + λ_route · L_write + λ_read · L_read
```

- **`FM_whole`：**整图的 flow-matching 损失。输入是源图 latent 加带 token 的指令，目标是目标图 latent。
- **`FM_region`：**只在该行的损失区域内算的 FM 误差。
  - 编辑行的损失区域：实际改动区域外扩 1 个 latent token（`--edit-region-dilate 1`）；
  - 高亮行的损失区域：被高亮的区域。
- **`FM_distractor`：**近邻硬反例区域（同图相邻、同类实例、孪生的对方等）上的误差，要求这些区域保持不变，权重 `--heavy-weight 8`。
- **`L_write`、`L_read`：**注意力路由的两个方向，λ 都是 0.05。

### 3.1 各阶段的损失超参

| 超参 | S1 | S2（v4） | R2 |
|---|---|---|---|
| `--lambda-region`（编辑行） | 1 | 1 | 1 |
| `--lambda-region-hl`（高亮行；不写时等于 `--lambda-region`） | 不写（= 1） | 不写（= 1） | **0** |
| `--lambda-distractor` | 1 | 1 | 1 |
| `--lambda-route` / `--lambda-read` | 0.05 / 0.05 | 0.05 / 0.05 | 0.05 / 0.05 |
| `--heavy-weight` | 8 | 8 | 8 |
| `--edit-region-dilate` | 1 | 1 | 1 |
| 高亮渲染 `--highlight-style` | overlay（50% 纯色叠加） | overlay | overlay |
| 优化器 | AdamW，LoRA 的 weight decay 0.01，注入器与仿射为 0 | 同左 | 同左 |
| 分辨率上限 | 约 1 MP | 同左 | 同左 |

### 3.2 R2 为什么这样改（证据见 EXPERIMENTS.md §4–§8）

1. **高亮行上的区域项是画质下降的主因。**同样的高亮行，在路由项之外再加区域项和干扰物项，MIRA PQ 降 1.23 [−1.87, −0.65]。只加路由项不伤画质，也不伤读 token。所以 R2 把高亮行的区域项设为 0，编辑行保留区域项。区域项是模型学会"在区域内编辑"的必要条件。
2. **贴回（paste-back）目标压低画质上限。**只用贴回目标训练，画质上限约 PQ 6.5；只用整图生成后还原区域外的高质量合成目标，约 7.1–7.3。所以 R2 中非孪生单目标编辑行全部改用高质量合成目标，只有孪生行（9.7%）保留贴回目标，以保住孪生绑定。
3. **从零重训出不来 bench 能力。**从零按新配方训练，IF-Bench 快速子集的 UnitPass 停在 0.72–0.77，与训练量无关。从 v4 出发低学习率续训，可以同时保住 v4 的 bench 能力并补回画质。
4. **续训量要小。**3.2 万行时两个指标同时最好，再训下去 bench 下降、画质也缓慢变差，属于过训。学习率在 8e-6 / 2e-5 / 5e-5 三档中选 8e-6，因为它最能守住 JointPass。

## 4. 训练配方（逐项可复现）

### 4.1 S1（`V3/runs/s1_4n_001/`，2026-10-30）

- **代码：**v4 分支提交 `47c0efd16fe4bf458d98db41425b159bec444960`。
- **启动设置：**`scripts/round2/settings/s1_4n.sh`，加入口 `scripts/round2/bootstrap_arnold.sh`。NAS 上的完整拼接版是 `V3/launch/s1_4n_full.sh`。
- **硬件：**ARNOLD 4 机 × 8 卡，accumulation 1，每次更新 32 行。
- **参数：**
  ```
  --rows rows_s1.jsonl --quick-items eval_c2_quick_idx.json --updates 6000 --accumulation 1 --warmup 50
  --lr 1e-4 --lr-schedule cosine --lr-min 1e-5 --heavy-weight 8 --checkpoints 1000,...,6000
  ```
  其余参数为默认：λ_region 1、λ_distractor 1、λ_route 0.05、λ_read 0.05、LoRA rank 32。
- **数据目录：**`V3/data/round2_scaleup_20261030`。
- **耗时：**24,814 秒，跳过 0 行。按开发集选第 6,000 次。

### 4.2 S2 = v4（`V3/runs/s2_4n_001/`，2026-10-31）

- **代码：**同 S1（`47c0efd`）。
- **启动设置：**`scripts/round2/settings/s2_4n.sh`，NAS 完整版 `V3/launch/s2_4n_full.sh`。
- **参数：**
  ```
  --rows rows_s2.jsonl --init-adapter V3/runs/s1_4n_001/train/adapter_6000 --updates 16000 --accumulation 1
  --warmup 100 --lr 5e-5 --lr-schedule cosine --lr-min 5e-6 --heavy-weight 8 --checkpoints 2000,...,16000
  ```
- **耗时：**69,193 秒。
- **选点：**按开发集选第 16,000 次。决策子集上对 C3-D + 引导的 JointPass +6.6% [+1.0%, +12.3%]。

### 4.3 R2（最终，`QA/runs_v5/v6_full/r2/`，2026-11-10）

- **代码：**快照 `22650e4b30830f95a11a3af8f4558326d98edb2f`，即 quality-anchor 线的 `3de0b33`。
  - 快照 bundle：`QA/code/quality-anchor-snapshot-22650e4.bundle`，分支名 `quality-anchor`。
  - 训练相关代码已整理进本分支，见 §6.2。
- **启动脚本：**`scripts/round2/settings/r2_4n.sh`，与 NAS 上实际运行的 `QA/launch/v6_r2.sh` 逐字相同。
- **参数**（从 `adapter_1000/adapter.json` 读出，与脚本一致）：
  ```
  --data QA/data_round2_v6p --rows rows_v6_s2hqfull.jsonl --quick-items eval_c2_quick_idx.json --accumulation 1
  --init-adapter V3/runs/s2_4n_001/train/adapter_16000
  --updates 7000 --warmup 100 --lr 8e-6 --lr-schedule cosine --lr-min 8e-7 --heavy-weight 8
  --checkpoints 1000,2000,3000,5000,7000 --quick-eval 99999
  --lambda-region 1 --lambda-region-hl 0 --lambda-distractor 1 --lambda-route 0.05 --lambda-read 0.05
  ```
  world 32，seed 2000，route_layers [6, 10, 14, 18, 22, 26]，highlight_style overlay，max_pixels 1,048,576。
- **交付检查点 `adapter_1000`：**
  - 是 7,000 次余弦日程中第 1,000 次的检查点，不是独立的 1,000 次训练，此时学习率约 7.70e-6；
  - 只用到行文件的前 32,000 行。
  - 要逐步复现它，必须保留 `--updates 7000`，同时加 `--stop-after 1000`（代码已支持）。只把 `--updates` 改成 1000 会改变学习率日程。
- **全程：**7,000 次更新，26,828 秒，跳过 0 行。检查点 1000/2000/3000/5000/7000 都保留在 `train/` 下。

### 4.4 Pass1 定位 LoRA（`SE/pass1_localization_lora_formal_20261030_0939/`）

- **训练记录：**`configs/pass1/formal_training_record.json`。
  - 新 LM LoRA 加在冻结的 SAMTok 底座上。8 卡，每卡 microbatch 1，梯度累积 8，有效 batch 64。
  - 峰值学习率 2e-5，AdamW，weight decay 0.01，梯度裁剪 1.0。3% 线性预热，再连续余弦降到峰值的 10%。
  - 数据课程：先过一遍 warmup，再过一遍 main。
- **选点：**按开发集 `unknown.strict_iou` 选 step 2000。
- **代码与数据：**代码在 `src/samtok_edit21/pass1/`；数据见 `docs/pass1_formal_data.md`；Pass1 的方法、实验与复现见 `docs/pass1_*.md`。

## 5. 推理（所有正式评测共用）

实现：`src/samtok_edit21/evaluation/ifbench.py`（IF-Bench 的公开请求）、`src/samtok_edit21/evaluation/generation.py`（`generate`）。配置：`configs/ifbench/main_row.json`。

| 项 | 设置 |
|---|---|
| Pass1 | `Qwen3-VL-8B-SAMTok` + LoRA step 2000；`do_sample=False`，`max_new_tokens=512`，`pass1_max_pixels=2097152`；一次调用定位全部单元。给定 mask 的单元直接用 codec 编码，不调用 Pass1 |
| Pass2 | Qwen-Image-2.1 DiT/VAE + **R2@1000 的 DiT LoRA（强度 1.0）与 CodeInjector** + 原版 SAMTok TE |
| 步数、种子 | 40 步，seed 42，每个请求只采样一次 |
| token 引导 | γ = 2.0。负分支是同一句提示去掉全部区域 token；每步算两次 DiT，共 80 次 forward。不带 token 的路径用 CFG 1 |
| 画布 | 约 1 MP，保持源图宽高比。IF-Bench 用 benchmark 的公共灰边画布，生成后回源 |
| 其他 | KV cache 开启；无 latent 融合、无贴回、无注意力偏置 |

正式评测中 IF-Bench、MIRA、CompBench、MagicBrush 都用这一套配置，只有 adapter 换成 R2@1000。

## 6. 代码实现

### 6.1 目录与模块

```
src/samtok_edit21/
  models/        pipeline.py（Qwen-Image-2.1 + SAMTok TE）、codec.py（SAMTok 编解码）、code_injector.py、adapters.py（LoRA）
  data/          protocol.py（区域 token 协议）、highlight.py（高亮/留色/虚化目标与提示）、io.py
  preparation/   instruction_protocol_v2.py、merge_data.py（合板数据）、sa1b_groups.py（D12–D15 转换）、round2_data.py（区域池、开发集、取行）
  training/      round2.py（行的读法与损失）、round2_engine.py（训练器）、routing.py、distributed.py、tracking.py
  evaluation/    generation.py、ifbench.py（IF-Bench 两轮推理）、ifbench_report.py（配对统计与合并）、round2_eval.py / round2_report.py（开发集探针）
  regions/       edited.py（实际改动区域）
  pass1/         Pass1 定位：protocol/model/grounder，data/、training/、evaluation/
scripts/
  round2/        ARNOLD 入口 bootstrap_arnold.sh / run_arnold.sh / setup_env.sh；settings/{s1,s2,r2}_4n.sh；specs/rows_s{1,2}.json；check_rows.py
  r2_data/       R2 训练数据的生成脚本（自蒸馏选样、门限、审核、建池、取行），见 DATA.md §4
  quality/       distill_gen.py（自蒸馏整图生成）
  ifbench/       IF-Bench 四机入口与 settings/r2_full_4n.sh、judge.py、补判与合并脚本、paper_tables.py
  pass1/         Pass1 数据、正式训练与 dev20 评测脚本
configs/         ifbench/main_row.json（推理配置，dit_adapter = R2@1000）、pass1/*
third_party/     diffsynth（vendored 2.1.12 + 两处 pipeline hook）、samtok
```

命令行入口有两个：`samtok-edit`（等价于 `python -m samtok_edit21`），子命令为 `round2-data / merge-data / sa1b-groups / round2-train / round2-eval / round2-report`；`samtok-pass1`（Pass1）。

### 6.2 本分支与 v4、R2 训练快照的关系

- **基点：**`qwen-image-2.1-v4` 的 `aa5a414`（IF-Bench 全量所用的 v4 代码）。
- **quality-anchor 线的提交（R2 训练所用），已 cherry-pick：**
  - `77a9c5b`：无 token 普通编辑行 `edit_plain`、自蒸馏目标覆盖 `target_override`、LoRA 强度探针；
  - `1093b60`：自蒸馏生成器 `scripts/quality/distill_gen.py`；
  - `f96c4d5`：着色高亮（tint）、与风格无关的探针读回、`--stop-after`；
  - `9b8c530`：高亮行单独的区域项权重 `--lambda-region-hl`；
  - `3de0b33`：留色与虚化两种读 token 任务。
- **未合入：**`22a5253`（LND，低噪声段向原版 DiT 蒸馏，实验无效）。它只在 `--lnd-sigma > 0` 时生效，R2 用的是默认 0，所以去掉它不影响 R2 的训练路径。唯一可见的差别是 `train_metrics.jsonl` 不再记录 `lnd_used`、`lnd_eligible` 两个字段。
- **合并 `qwen-image-2.1-v5`：**IF-Bench 补判与合并报告（`ifbench_report.py`、`rejudge_baselines.py`、`paper_tables.py`），以及 R2 的 IF-Bench 全量入口。
- **本次清理删除的内容**（只在本分支删除，其他分支不动）：
  - OPSD（`training/opsd.py`、训练器里的 OPSD 参数与分支、取行时的 OPSD 排布，`tests/test_opsd.py`）；
  - 旧 benchmark 2.x 适配器 `evaluation/benchmark_token.py` 与它唯一依赖的 `regions/selection.py`；
  - C2 启动设置；
  - 错误的 `r2_ifbench_4n.sh`（它跑的是开发探针，不是 IF-Bench）；
  - 数字有误的 `docs/04、05`。
  - 去掉 OPSD 不改变没有 OPSD 行时的取行结果：被删的分支只处理 OPSD 行，也不消耗随机数。
- **仍保留的旧行种类：**训练器仍能读取 `edit`、`edit_v1`、`edit_multi`、`edit_ext` 等 C1–C3 时代的行种类，这些读法和 S1/S2 共用同一套损失代码。最终方法只用 `hl`、`hl_ref`、`hlm`、`edit_merged`、`edit_plain`。

### 6.3 验证（2026-11-11，本机 CPU）

- 全部模块可导入；`samtok-edit round2-train --help` 正常，包含 `--lambda-region-hl` 和 `--stop-after`。
- `pytest tests`：119 项通过（含文档代码块同步检查 `tests/test_docs.py`），9 项跳过（需要 GPU 或 Pass1 环境）。2 项失败都是因为本机环境没装 `wandb`（`test_round2_tracker_*`），与代码改动无关。
- 没有做 GPU 重训。R2 的权重就是 `22650e4` 训练出的原文件，SHA256 见 §7。

## 7. 权重与模型路径

| 用途 | 路径 | 主文件 SHA256 |
|---|---|---|
| **最终 DiT LoRA（R2@1000）** | `QA/runs_v5/v6_full/r2/train/adapter_1000/adapter.safetensors` | `c0d15a0c81ca63713c72d8ab0ee23e7f475c4ebe86642601d74b546227e734b8` |
| **最终 CodeInjector（R2@1000）** | `QA/runs_v5/v6_full/r2/train/adapter_1000/code_injector.safetensors` | `205f2c28184e4f2d01018d6f0c9e2252c5c546b697688a29dae5c1af00296fb9` |
| R2 的初始化 = v4（S2@16000） | `V3/runs/s2_4n_001/train/adapter_16000/adapter.safetensors` | `da5069b8590f2977e4f9ac1b6486b3e5b146cfbad099e610c57c3cf0263772dc` |
| S1@6000 | `V3/runs/s1_4n_001/train/adapter_6000/adapter.safetensors` | `0fab862e6992d46bff742ae06e06b0a56d377b783789de7cb999df2d09ea9362` |
| **Pass1 定位 LoRA** | `SE/pass1_localization_lora_formal_20261030_0939/runs/formal/step_2000/adapter_model.safetensors` | `c857149a46cd9c14ab47cb1eede5d8ac2cc03d701a96217c1a4fd77565d75a5e` |
| Qwen-Image-2.1 | `NAS/models/pretrained_models/Qwen-Image-2.1` | — |
| SAMTok（TE、codec、Pass1 底座） | `NAS/models/SAMTok/Qwen3-VL-8B-SAMTok` | — |
| IF-Bench Judge | `NAS/models/pretrained_models/Qwen3.8-Flash-Next-FP8`（revision `236dfdf2…`） | — |

R2 的其他检查点 `adapter_{2000,3000,5000,7000}` 和 `resume.pt` 在同一目录，只用于实验分析，不是交付权重。

## 8. 运行

### 8.1 环境

- **编辑与训练：**
  ```bash
  uv venv --python /usr/bin/python3.11 ENV
  uv pip install --python ENV/bin/python -r requirements.txt
  ```
  四机入口里的 `scripts/round2/setup_env.sh` 自动完成这一步。从源码直接运行时，把 `PYTHONPATH` 设为 `src:third_party/diffsynth:third_party`。
- **Pass1：**用单独的环境（`transformers==5.19.0`，见 `configs/pass1/requirements-tested.txt`），不要与 `constraints-tested.txt` 混用。
- **本机现成环境：**
  - `/tmp/ifbench/external_bench_20261102/env_v4`（编辑）；
  - `/tmp/ifbench/external_bench_20261102/env_pass1`（Pass1）。
  - 机器回收后，需从 NAS 上的环境包还原，见 `QA/scripts/restore_runtimes.sh`。

### 8.2 单图编辑（Python）

```python
from PIL import Image
from samtok_edit21 import generate, load_codec, load_grounding_adapter, load_pipeline

pipe = load_pipeline(device="cuda")
codec = load_codec("/mnt/bn/strategy-mllm-train/user/tanyue/models/SAMTok/Qwen3-VL-8B-SAMTok", "cuda")
load_grounding_adapter(pipe, codec, "<QA>/runs_v5/v6_full/r2/train/adapter_1000")   # DiT LoRA + CodeInjector
source = Image.open("source.png").convert("RGB")
token = codec.encode(source.convert("RGBA"), [mask])[0][0]                            # 布尔 mask -> 区域 token
prompt = f"Replace the selected region {token} with a navy polo shirt."
image, _ = generate(pipe, codec, prompt, source, steps=40, seed=42,
                    cfg_scale=2.0, negative_prompt="Replace the selected region with a navy polo shirt.")
```

### 8.3 训练（四机 ARNOLD）

- **S1、S2：**把 `scripts/round2/settings/{s1,s2}_4n.sh` 与 `scripts/round2/bootstrap_arnold.sh` 拼接后作为启动命令。
- **R2：**`bash scripts/round2/settings/r2_4n.sh`。脚本从 NAS bundle 克隆快照 `22650e4`，每次提交换一个新的 `V5_RUN_ID`；中断后从 `resume.pt` 续训。

### 8.4 IF-Bench 全量评测（四机）

```bash
bash scripts/ifbench/settings/r2_full_4n.sh
```

- 只跑我方方法，七个基线已完成。
- 这个脚本在 11-10/11 以 attempt `r2_run002`、代码 `8beaffb` 跑完，结果在 `SE/ifbench_r2_full_20261110/`（有 `SUCCESS.json`）。
- 重新提交时换一个新的 `SAMTOK_ATTEMPT`。

## 9. 局限与注意事项

1. **R2 的训练行里没有多目标编辑行。**两个取行脚本（`scripts/r2_data/build_rows_v6s2.py`、`build_rows_v6s2hqfull.py`）用"非孪生且无 partners"来判断"单目标"，多目标行也满足这个条件，于是全部被换成了单目标合成行。v4 的 S2 有 128,000 行多目标；v6 S2 基础行剩 52,893 行；R2 为 0（详见 DATA.md §4.4）。R2 的多目标能力继承自 v4。IF-Bench 全量上 R2 与 v4 的 JointPass 差 −0.4 个百分点 [−1.9, +1.1]，没有可见的退化。但这与取行脚本 docstring 写的意图不同，写论文时应按实际情况描述。
2. **IF-Bench 上只有 S3-point 显著低于 v4：**JointPass −3.5 [−6.4, −0.6]，UnitPass −2.5 [−3.9, −1.2]。其他 9 种 setting 都在噪声内。
3. **MIRA 的 GPT-5.4 判分对同一张图波动大：**30 张逐字节相同的 R2 输出，两次判分 PQ 均差 +0.39，逐图平均绝对差 0.76。所以与原版比较必须同时段重判；最终的 PQ 7.19 对 7.21 就是同时段重判的结果。
4. **开发期间用过 MIRA 30 张子集和 IF-Bench 40 例子集选方案、选检查点，**属于在这两个 benchmark 上的历史暴露，论文需要披露。
5. **CompBench 上 R2 与 v4 互有高低：**LC-T 略高；LC-I、PSNR、SSIM 普遍略低，局部编辑 PSNR −0.86 dB。没有做配对区间。
