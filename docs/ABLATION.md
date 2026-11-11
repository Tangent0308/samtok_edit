# 消融实验（2026-11-12 启动）

本文件记录论文消融实验的设计、代码、数据、运行方式与结果位置。所有数字以 `ABLATION_ROOT/REPORT/ABLATION.json` 为准。

- `ABLATION_ROOT` = `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/ablation_20261112`
- 代码：分支 `qwen-image-2.1-r2-ablation`（基于交付分支 `qwen-image-2.1-r2-delivery`），目录 `scripts/ablation/`
- 唯一定义来源：`scripts/ablation/plan.py`（所有臂、系统、路径、训练参数）

## 1. 用户决定（11-12）

- 只评测 IF-Bench（加开发集探针），不评 MIRA。
- 不做"无精修目标"对照（先不做），不做"文字坐标"对照（方法以 SAMTok token 协议为前提）。
- 数据规模只做数据量：25% / 50% 记录，不做训练量曲线。
- IF-Bench 子集取 200 张图。
- 其余按建议：缩减协议的组件消融、最终模型的推理诊断、所有系统一次判分。

## 2. 实验列表

| 块 | 系统 | 改动 | 评测 setting | 参照 |
|---|---|---|---|---|
| 组件（缩减协议） | `abl_ref` | 完整配方 | S1、S2-mask、S3-mask、S4-noref | — |
| | `abl_no_identification` | 目标识别行全部换成编辑行（只用编辑对训练） | 同上 | `abl_ref` |
| | `abl_no_attention` | `--lambda-route 0 --lambda-read 0`（前向不挂路由器） | 同上 | `abl_ref` |
| | `abl_no_twins` | 孪生编辑对两成员都换成普通单目标编辑行 | 同上 | `abl_ref` |
| | `abl_no_region` | `--lambda-region 0 --lambda-region-hl 0` | 同上 | `abl_ref` |
| | `abl_no_injector` | `--no-code-injector`（token 只作为文本编码器词表进入 DiT） | 同上 | `abl_ref` |
| | `abl_no_multi` | Edit 阶段多目标编辑行换成单目标编辑行（Read 阶段本来没有） | 同上 | `abl_ref` |
| | `abl_no_neighbor` | `--lambda-distractor 0` | 同上 | `abl_ref` |
| 最终模型诊断 | `final` | R2@1000，正式推理配置 | S1、S2-mask、S3-mask、S4-noref | — |
| | `final_notoken` | 推理时提示中不写 Space Token（也就没有 token 引导） | S1、S2-mask | `final` |
| | `final_g1` | γ = 1（关闭 token 引导） | 四个 setting | `final` |
| 数据规模 | `s25` | 25% 记录，4,000 次更新 | 十个 setting | `s100` |
| | `s50` | 50% 记录，8,000 次更新 | 十个 setting | `s100` |
| | `s100` | 正式 S2@16000（100% 记录，16,000 次更新） | 十个 setting | — |

- `final_notoken` 只评 S1、S2-mask：盲指代 setting 去掉 token 后指令中没有目标。S1 去掉 token 等于不用第一遍定位。
- 开发集探针（`round2-eval`，1,320 项，20 步，判分无关）评：原版、`final`、8 个组件臂、`s25`、`s50`、`s100`。其中 paired following 即"同一张图换成另一个实例的 token，输出是否跟着走"。

## 3. 协议

**组件消融（缩减协议）：**
- 从原版 DiT 起训，Read 250 次 + Edit 750 次更新，每次 32 行（四机 32 卡 × 累积 1；单机 8 卡 × 累积 4 时每次更新的行完全相同）。
- 行取正式 `rows_s1.jsonl` 前 8,000 行、`rows_s2.jsonl` 前 24,000 行，即正式两阶段的前段；`abl_ref` 就是缩短的正式配方。
- 学习率：Read 1e-4（预热 25，余弦到 1e-5），Edit 5e-5（预热 50，余弦到 5e-6）；Edit 从同一臂的 Read@250 热启动。
- 损失（S1/S2 的默认）：区域项 1（含目标识别行）、近邻保持 1、写/读路由 0.05、硬反例 8、编辑区域外扩一个 token、overlay 高亮。
- 每 250 次存检查点，中断后从检查点续训。

**数据规模：**
- 从 S1@6000 起训，只对编辑记录抽子集；S2 的取行配置（`scripts/round2/specs/rows_s2.json`，同一随机种子、同一组份额）在 4,000 / 8,000 次更新上重新取行，即与 S2 相同的 epoch 数。
- 子集单位是 `group`（同一源图的记录共享），`sha256("ablation-scale-20261112:" + group)` 决定是否入选，25% ⊂ 50%（嵌套），孪生和多目标记录不会被拆开。
- 目标识别回放来自 SA-1B 区域池，不属于 385K 记录，三点相同。Read 阶段（S1）用过全量数据中约 3.8 万条编辑行，三点共用。
- 学习率同 S2：5e-5，预热 100，余弦到 5e-6。100% 点直接用正式 S2@16000。

**评测：**
- IF-Bench 200 张图（`ABLATION_ROOT/cases200.txt`，`select_cases.py`）：排除 80 个开发 case（`quality_anchor_20261104/ifbench_cases80.txt`），按单元数分层（2/3/4/5 个单元：28/47/110/15 张，共 712 个单元），按固定哈希顺序抽取，在任何消融输出之前冻结。
- Pass1 复用 R2 全量评测（`ifbench_r2_full_20261110/pass1`）的记录：Pass1 与编辑器无关，所有系统用同一份 token。
- 生成与正式评测相同：40 步、种子 42、γ = 2（`final_g1` 除外），每个请求一次联合生成。本机核对：消融流程生成的 `final` 与正式 R2 全量评测输出逐字节相同（`v2-sa_10030491` 的 S1、S2-mask）。
- 判分：`scripts/ablation/judge_many.py`，每台机器一个 Judge 引擎判所有系统（同一份 bundle 配置、同一个图像分配）；`judge.py` 的判分与报告代码不变。
- 统计：每个 setting 的 UnitPass、JointPass 用完整分母（生成失败、判分缺失都算不通过，缺失单独列出）；均值是系统所含 setting 的算术平均；区间为按源图配对的 bootstrap 95% 区间（10,000 次，种子 20261027，所有系统共用一组抽样）。

## 4. 数据（`ABLATION_ROOT/data`）

- 链接正式 S1/S2 数据目录 `qwen21_v3/data/round2_scaleup_20261030` 的全部内容，并新增下列行文件（`build_data.py`）。
- 组件行都在对应前段上**原位替换**：每次更新的其余行不变，成对的行仍在同一次更新内；替换行取自同一文件未用到的后段。

| 文件 | 行数 | 替换 | 说明 |
|---|---:|---:|---|
| `rows_abl_noid_s1.jsonl` | 8,000 | 6,360 | 目标识别行（hl、hl_ref、hlm、高亮对）→ 非孪生编辑行 |
| `rows_abl_noid_s2.jsonl` | 24,000 | 2,387 | 同上 |
| `rows_abl_notwin_s1.jsonl` | 8,000 | 726 | 孪生编辑对成员 → 普通单目标编辑行 |
| `rows_abl_notwin_s2.jsonl` | 24,000 | 2,412 | 同上 |
| `rows_abl_nomulti_s2.jsonl` | 24,000 | 5,987 | 多目标编辑行 → 普通单目标编辑行 |
| `rows_abl_s25.jsonl` | 128,000 | — | 25% 记录（统计见 `scale_s25_subset.json`、`rows_abl_s25_stats.json`） |
| `rows_abl_s50.jsonl` | 256,000 | — | 50% 记录（统计见 `scale_s50_subset.json`、`rows_abl_s50_stats.json`） |

- 每个文件的种类计数在同名 `_stats.json`；前段（未改动）的计数在 `prefix_read_stats.json`、`prefix_edit_stats.json`。
- 不经模型的行检查（`scripts/round2/check_rows.py`，各取前 800 行）：五个组件行文件问题数均为 0（`ABLATION_ROOT/data_checks/`）。

## 5. 代码改动

- `training/round2_engine.py`：
  - `--no-code-injector`：不挂 CodeInjector、不保存 `code_injector.safetensors`；从带 CodeInjector 的权重热启动时报错。推理端 `load_grounding_adapter` 本来就按目录里有无 CodeInjector 加载。
  - 写、读路由权重都为 0 时，前向不挂路由器（`round2_loss(router=None)`），不再计算注意力监督项；写路由权重为 0 时不再把写项加进损失。两者都为正时行为不变。
- `evaluation/ifbench.py`：方法配置可选 `"drop_tokens": true`，提示只用公开指令（不写 token，因此也没有 token 引导）。
- `scripts/ablation/`：`plan.py`（定义）、`build_data.py`、`select_cases.py`、`judge_many.py`、`report.py`、`run_4n.sh`（四机入口）、`rehearse.sh`（本机预演）。
- 测试：`tests/test_ablation_plan.py`。

## 6. 运行

- **四机入口：**`ABLATION_ROOT/launch/run_4n.sh`，整段原样粘贴成 ARNOLD 四个 worker（4 × 8 卡）的启动命令；代码来自同目录 `code/` 下的 git bundle，按提交号检出。
- **执行顺序：**组件 8 臂训练 → 探针（原版、最终模型、8 臂）→ IF-Bench 生成（最终模型三个、8 臂）→ 规模 2 臂训练 → 规模三点的探针与生成 → 所有系统一次判分 → `REPORT/ABLATION.md`。
- **预计时长**（按 S1/S2/R2 实测速度）：组件训练约 11.5 h，规模训练约 14.5 h，探针约 1.5 h，生成约 4.5 h，判分约 3.5 h，共约 36 h。
- **中断后：**把 `ABL_RUN_ID` 换成新值重新提交。已完成的步骤按结果文件跳过（`TRAIN_DONE`、`EVAL_DONE`、`state/*.done`），训练从最近检查点续训，生成从未完成的请求继续（上一次提交遗留的认领会记录到 `RELEASES.jsonl` 后释放）。
- **输出：**
  - 训练 `train/<臂>/{read,edit}/`、`scale/<臂>/train/`
  - 探针 `probe/<系统>/eval.json`
  - IF-Bench `ifbench/<系统>/`（`REPORT.json`）
  - 汇总 `REPORT/ABLATION.{json,md}`
  - 日志 `submissions/<ABL_RUN_ID>/`

## 7. 验证记录

- 单元测试：119 通过、9 跳过，2 个失败均为本机环境缺 `wandb`（与改动前相同）；新增 `tests/test_ablation_plan.py` 3 项通过。
- 训练冒烟（本机 2 卡）：无 CodeInjector 臂 Read → Edit、无注意力监督臂 Read，各 2 次更新，0 个样本跳过；无 CodeInjector 的权重目录中没有 CodeInjector 文件。
- 生成冒烟（本机 1 卡）：`final_notoken` 提示中无 token、cfg 1.0；`final` 与正式 R2 输出逐字节相同。
