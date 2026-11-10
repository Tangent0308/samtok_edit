# SITEdit v5：方法

本文记录 2026-11-01 至 11-10 敲定的最终训练方法。配套实验见 [文档 05](05_SITEdit_v5实验.md)。

**路径记号**
- `V3/…` 指 `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v3/…`；
- `runs_v5/…` 指 `V3/runs_v5/…`；
- 仓库内路径（`src/…`、`scripts/…`）相对本仓库（分支 `qwen-image-2.1-v5`）根目录。

---

## 0. 一分钟版

**方法名称：SITEdit（Space-in-Token Editing）。**

把源区域选择编码成 Space Token（4 个离散 RQ 码），直接拼入文字指令，和子指令绑定成统一条件，送进 DiT 编辑器。推理时不解码 token、不加注意力偏置、不融合 latent；Space Token 只在训练时对应 ground-truth mask。

**交付检查点：**`runs_v5/v6_full/r2/train/adapter_1000`

**训练：三阶段。**

| 阶段 | 目的 | 步数 | 行数 | 初始化 | lr |
|---|---|---|---|---|---|
| S1 Read | 让模型理解 token 含义 | 6,000 | 192,000（约 10% read-token + 90% edit） | 原版 LoRA 随机 | 1e-4 余弦 → 1e-5 |
| S2 Edit | 强化多目标编辑能力 | 16,000 | 512,000（同 rows_s2.jsonl） | S1@6000 | 5e-5 余弦 → 5e-6 |
| R2 Refine | 恢复画质、维持执行力 | **1,000**（交付） | 64,000（rows_v6_s2hqfull.jsonl 前 1,000 步 × 32） | S2@16000 | **8e-6 余弦 → 8e-7** |

---

## 1. 模型结构

与 v4 完全相同；本分支代码快照为 `22650e4b30830f95a11a3af8f4558326d98edb2f`（源自 `3de0b33`）。

- **LoRA**：r32、α32，作用于 DiT 的 `to_q/to_k/to_v/to_out.0/proj/out/gate_layer`；AdamW，weight decay 0.01，约 1 MP。
- **码本注入（CodeInjector）**：把 Space Token 的 RQ 嵌入加到文字 token 的 embedding 上，权重独立。
- **Space Token Guidance（推理）**：γ = 2，用 v⁺（带 token）− v⁻（不带 token）放大空间条件。LoRA 强度：1.0（v4 正式 IF-Bench 跑的是 0.75）。

---

## 2. R2 训练数据（`rows_v6_s2hqfull.jsonl`）

共 512,000 行，构成：

| 行类型 | 比例 | 目标图来源 | 说明 |
|---|---|---|---|
| 读 token 行（overlay highlight） | 10% | 源图本身 | 让模型保持 token 定位能力 |
| token 编辑行（非孪生） | 80%（其中 ~9.7% 孪生） | **高合成目标（composite）** | 自蒸馏合成对，53,688 对，来自 v2+v3 审核通过池 |
| 孪生行（paste-back） | 9.7% | paste-back | 确保保持能力的下界 |
| 纯文字编辑行（无 token） | 10% | 合成目标 | |

**关键决定：非孪生单目标编辑全用合成目标，消除 paste-back 造成的质量天花板。**
- paste-back 比例从 31%（S2 阶段）降到 9.7%（仅孪生行保留），MIRA PQ +0.91。
- 合成目标来自同一个 DiT 编辑器对审核通过样本的前向推理结果，质量接近真实编辑输出。

---

## 3. R2 损失设置

与 S2 相同，区别仅在于读 token 行不加区域项高亮损失：

| 损失项 | 编辑行 λ | 读 token 行 λ |
|---|---|---|
| FM（flow matching） | 1 | 1 |
| 区域项（region） | 1 | 1 |
| 区域高亮项（region_hl） | 0 | **0**（读 token 行本身不改图） |
| distractor | 1 | 1 |
| 路由（route） | 0.05 | 0.05 |

---

## 4. R2 超参敲定过程（见文档 05 §2–§4）

- **步数**：@1000 是 UnitPass 和 JointPass 同时最优的峰值。@2000 起单调下降（0.835 → 0.807 → 0.780 @ 3000/5000）。
- **学习率**：在 {4e-6, 8e-6, 1.6e-5} 三点扫，8e-6 是唯一保住 JointPass 的点（0.519，其他两个 ≈ 0.481）。
- **排除的配方**（全部实测，均不如交付方案）：
  - 读 token 行加 color-leave 任务
  - 读 token 行比例 20%（vs 10%）
  - 读 token 行加区域项
  - 孪生比例 20%（vs 9.7%）
  - 合成孪生目标（vs paste-back 孪生）
  - paste-back 比例 31%
  - 训练中途切换课程
  - 纹理重建辅助阶段（R1t）
  - 强度路由（E1）
  - 仅增加训练规模（无合成目标）
  - 推理 LoRA 强度降低（0.75 → 0.5 等）

---

## 5. 推理配方

```
采样步数：28
Space Token Guidance：γ = 2（带 token 和不带 token 各推一遍）
LoRA 强度：1.0
CodeInjector：常开
```

---

## 6. 检查点路径

| 用途 | 路径 |
|---|---|
| **交付** | `V3/runs_v5/v6_full/r2/train/adapter_1000` |
| R2 初始化（v4 Edit 终点） | `V3/runs/s2_4n_001/train/adapter_16000` |
| S1 终点 | `V3/runs/s1_4n_001/train/adapter_6000` |
| 四机启动脚本 | `scripts/round2/settings/r2_4n.sh` |
| 训练行文件 | `V3/data_round2_v6p/rows_v6_s2hqfull.jsonl` |
