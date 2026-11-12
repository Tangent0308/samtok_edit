# 消融实验（2026-11-12 启动）

本文件记录论文消融实验的设计、代码、数据、运行方式与结果位置。所有数字以 `ABLATION_ROOT/v2/REPORT/ABLATION.json` 为准。

- `ABLATION_ROOT` = `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/ablation_20261112`
- **现行版本是 v2**，输出在 `ABLATION_ROOT/v2/`。v1（运行 `abl_001`，Read 只有 250 次更新）已停止，见 §8。数据 `ABLATION_ROOT/data/`、评测图 `ABLATION_ROOT/cases200.txt`、代码快照 `ABLATION_ROOT/code/` 两版共用。
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
| | `abl_no_identification` | Space Token：目标识别行全部换成编辑行（只用编辑对训练） | 同上 | `abl_ref` |
| | `abl_no_aux_losses` | Loss：区域均衡、近邻保持、写/读注意力监督全关（`--lambda-region 0 --lambda-region-hl 0 --lambda-distractor 0 --lambda-route 0 --lambda-read 0`），只剩整图 flow matching | 同上 | `abl_ref` |
| | `abl_no_pipeline` | 数据管线：孪生编辑对与多目标记录换成普通单目标编辑行 | 同上 | `abl_ref` |
| | `abl_no_injector` | Space Token：`--no-code-injector`（token 只作为文本编码器词表进入 DiT） | 同上 | `abl_ref` |
| 最终模型诊断 | `final` | R2@1000，正式推理配置 | S1、S2-mask、S3-mask、S4-noref | — |
| | `final_notoken` | 推理时提示中不写 Space Token（也就没有 token 引导） | S1、S2-mask | `final` |
| | `final_g1` | γ = 1（关闭 token 引导） | 四个 setting | `final` |
| 数据规模 | `s25` | 25% 记录，4,000 次更新 | 十个 setting | `s100` |
| | `s50` | 50% 记录，8,000 次更新 | 十个 setting | `s100` |
| | `s100` | 正式 S2@16000（100% 记录，16,000 次更新） | 十个 setting | — |

- 组件臂按目的合并（用户 11-12）：原来的 8 臂里，无注意力监督、无区域项、无近邻项合并为 `abl_no_aux_losses`；无孪生、无多目标合并为 `abl_no_pipeline`。合并后的一行看不出组内哪一部分起作用，需要时可在补充材料里再拆。
- `final_notoken` 只评 S1、S2-mask：盲指代 setting 去掉 token 后指令中没有目标。S1 去掉 token 等于不用第一遍定位。
- 开发集探针（`round2-eval`，1,320 项，20 步，判分无关）评：原版、`final`、8 个组件臂、`s25`、`s50`、`s100`。其中 paired following 即"同一张图换成另一个实例的 token，输出是否跟着走"。

## 3. 协议

**组件消融（缩减协议 v2）：**
- 从原版 DiT 起训，每次 32 行（四机 32 卡 × 累积 1；单机 8 卡 × 累积 4 时每次更新的行完全相同）。
- **Read = 正式 S1 的前 1,000 次更新：**同样的行（`rows_s1.jsonl` 前 32,000 行），同样的学习率日程（1e-4，预热 50，按 6,000 次余弦到 1e-5），在第 1,000 次停下（`--updates 6000 --stop-after 1000`）。参照臂的 Read 应复现正式 S1@1000（快速探针单 token 盲指代 0.310），可用来检验当前代码与正式训练一致。
- **Edit：**`rows_s2.jsonl` 前 24,000 行，750 次更新，5e-5（预热 50，余弦到 5e-6），从同一臂的 Read@1000 热启动。
- `abl_ref` 就是缩短的正式配方。
- v1 的 Read 只有 250 次（`rows_s1` 前 8,000 行，自己的 250 次余弦），参照臂学到的读 token 太少，见 §8。
- 损失（S1/S2 的默认）：区域项 1（含目标识别行）、近邻保持 1、写/读路由 0.05、硬反例 8、编辑区域外扩一个 token、overlay 高亮。
- 每 250 次存检查点（Read 存 250/500/750/1000），中断后从检查点续训。

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
| `rows_abl_noid_s1_r1000.jsonl`（v2） | 192,000（训练只用前 32,000） | 25,570 | 目标识别行（hl、hl_ref、hlm、高亮对）→ 非孪生编辑行；`rows_s1` 后段只有约 18,350 行，循环补足，7,218 行出现两次；孪生行 2,764 与参照相同 |
| `rows_abl_notwin_s1_r1000.jsonl`（v2） | 192,000（训练只用前 32,000） | 2,764 | 孪生编辑对成员 → 普通单目标编辑行 |
| `rows_abl_noid_s1.jsonl`（v1） | 8,000 | 6,360 | v1 的 Read 文件，v2 不用 |
| `rows_abl_noid_s2.jsonl` | 24,000 | 2,387 | 目标识别行 → 非孪生编辑行（两版共用） |
| `rows_abl_notwin_s1.jsonl`（v1） | 8,000 | 726 | v1 的 Read 文件，v2 不用 |
| `rows_abl_notwin_s2.jsonl` | 24,000 | 2,412 | 同上 |
| `rows_abl_nomulti_s2.jsonl` | 24,000 | 5,987 | 多目标编辑行 → 普通单目标编辑行（合并前的臂，现在不用） |
| `rows_abl_nopipe_s2.jsonl` | 24,000 | 8,399 | 孪生（2,412）与多目标（5,987）编辑行 → 普通单目标编辑行（`abl_no_pipeline` 的 Edit；其 Read 用 `rows_abl_notwin_s1_r1000`，`rows_s1` 没有多目标行） |
| `rows_abl_s25.jsonl` | 128,000 | — | 25% 记录：可用默认记录 95,211 / 384,533，行中不同记录 87,632（S2 为 353,601），孪生 6,410 对，多目标 32,000 行 |
| `rows_abl_s50.jsonl` | 256,000 | — | 50% 记录：可用默认记录 192,516，行中不同记录 177,238，孪生 13,129 对，多目标 64,000 行 |

- 规模两份行文件与 S2 结构一致：编辑 90%、目标识别 10%；每个抽取的可用量都约为 S2 的 25% / 50%，重复倍数与 S2 相同（例如多目标 32,000 行取自 13,505 条，约 2.37 倍；S2 为 128,000 / 53,802，约 2.38 倍）。各来源的入选比例在 `scale_s{25,50}_subset.json`（25%：0.23–0.26；50%：0.50–0.52）。

- v2 的两个 Read 文件在训练用的 32,000 行之后接上 `rows_s1` 未改动的其余行：训练器检查行数要覆盖整个 6,000 次日程，这些行不会被训练。
- 每个文件的种类计数在同名 `_stats.json`；未改动前段的计数在 `prefix_read_1000_stats.json`（v2）、`prefix_read_stats.json`（v1）、`prefix_edit_stats.json`。
- 不经模型的行检查（`scripts/round2/check_rows.py`，各取前 800 行）：五个组件行文件问题数均为 0（`ABLATION_ROOT/data_checks/`）。

## 5. 代码改动

- `training/round2_engine.py`：
  - `--no-code-injector`：不挂 CodeInjector、不保存 `code_injector.safetensors`；从带 CodeInjector 的权重热启动时报错。推理端 `load_grounding_adapter` 本来就按目录里有无 CodeInjector 加载。
  - 写、读路由权重都为 0 时，前向不挂路由器（`round2_loss(router=None)`），不再计算注意力监督项；写路由权重为 0 时不再把写项加进损失。两者都为正时行为不变。
- `evaluation/ifbench.py`：方法配置可选 `"drop_tokens": true`，提示只用公开指令（不写 token，因此也没有 token 引导）。
- `scripts/ablation/`：`plan.py`（定义）、`build_data.py`、`select_cases.py`、`judge_many.py`、`report.py`、`run_4n.sh`（四机入口）、`rehearse.sh`（本机预演）。
- 测试：`tests/test_ablation_plan.py`。

## 6. 运行

- **四机入口（v2，两个四机作业并行，用户 11-12 要求）：**
  - 组件块：`ABLATION_ROOT/launch/run_4n_v2_component.sh`（默认运行编号 `abl_v2c_001`）：5 个组件臂的训练、探针、生成、判分，约 14.5 h。
  - 规模块：`ABLATION_ROOT/launch/run_4n_v2_scale.sh`（`abl_v2s_001`）：原版与最终模型探针、最终模型 3 个诊断的生成、规模 2 臂训练、3 个规模点的探针与生成、判分，约 20 h。
  - 每个作业的四个 worker 都 `bash` 自己的文件；进度 `bash ABLATION_ROOT/launch/status_v2.sh`。
  - 分块由 `ABL_BLOCK`（`component` / `scale` / `all`）决定，定义在 `plan.lists()`。每块只判自己的系统，块内比较不跨判分会话；两块都完成后 `REPORT/ABLATION.md` 包含全部三块。
  - 代码来自 `ABLATION_ROOT/code/` 下的单提交快照 bundle，按提交号检出（本机仓库是浅克隆，带历史的 bundle 无法 clone）。v2 快照与对应分支提交的代码树逐字节相同。
  - v1 的入口 `launch/run_4n.sh`（快照 `b264fca` = 分支 `5782a36`）保留不动。
- **执行顺序**（`ABL_BLOCK=all` 时一个作业跑全部，顺序如下；分块时每块跳过不属于自己的步骤）：组件 8 臂训练 → 探针（原版、最终模型、8 臂）→ IF-Bench 生成（最终模型三个、8 臂）→ 规模 2 臂训练 → 规模三点的探针与生成 → 判分 → `REPORT/ABLATION.md`。
- **预计时长**（按 v1 实测 4.24 s 一次更新）：组件块约 14.5 h（5 臂，训练 11.6 h，每臂约 2.3 h）；规模块约 20 h（训练 14.6 h）。
- **中断后：**把 `ABL_RUN_ID` 换成新值重新提交。已完成的步骤按结果文件跳过（`TRAIN_DONE`、`EVAL_DONE`、`state/*.done`），训练从最近检查点续训，生成从未完成的请求继续（上一次提交遗留的认领会记录到 `RELEASES.jsonl` 后释放）。
- **输出**（v2 都在 `ABLATION_ROOT/v2/` 下）：
  - 训练 `train/<臂>/{read,edit}/`、`scale/<臂>/train/`
  - 探针 `probe/<系统>/eval.json`
  - IF-Bench `ifbench/<系统>/`（`REPORT.json`）
  - 汇总 `REPORT/ABLATION.{json,md}`
  - 日志 `submissions/<ABL_RUN_ID>/`

## 7. 验证记录

- 单元测试：119 通过、9 跳过，2 个失败均为本机环境缺 `wandb`（与改动前相同）；新增 `tests/test_ablation_plan.py` 3 项通过。
- 训练冒烟（本机 2 卡）：无 CodeInjector 臂 Read → Edit、无注意力监督臂 Read，各 2 次更新，0 个样本跳过；无 CodeInjector 的权重目录中没有 CodeInjector 文件。
- 生成冒烟（本机 1 卡）：`final_notoken` 提示中无 token、cfg 1.0；`final` 与正式 R2 输出逐字节相同。
- 汇总脚本：用已知通过率的人工判分结果核对，点估计、配对差、区间和缺失计数都正确。
- **完整预演**（`scripts/ablation/rehearse.sh`，本机 4 卡，代码同四机：快照 bundle `b264fca`）：
  - 每个训练阶段 2 次更新、2 张评测图、探针每类 1 项、Judge 只列出不加载引擎；
  - 11-11 18:20–20:20（UTC）从检出到出表全部跑通，`SUCCESS.json` 写出；
  - 14 个系统的生成、13 个探针、14 个 REPORT.json 与汇总表都齐全；训练器读到的行与设计一致（例如无目标识别臂的 Read 阶段 64/64 行是编辑行，无注意力监督臂写/读项为 0）。

## 8. 四机运行记录

**v1：`abl_001`（11-11 19:05 UTC 启动，用户 11-12 决定停止）**
- 四机汇合成功（world 32、累积 1），前 10 次更新 0 个跳过，行构成与 `rows_s1` 一致；编译后 4.24 s 一次更新。
- 参照臂（Read 250 + Edit 750）训完后的快速探针（112 项）：

  | 模型 | 单 token 盲指代 IoU | 带名 IoU | 多 token 区域 IoU | 区域外改动 |
  |---|---|---|---|---|
  | 原版 | 0.067 | 0.264 | 0.037 | 6.57 |
  | v1 参照臂 Read@250 | 0.122 | 0.308 | 0.021 | 2.25 |
  | v1 参照臂 Read@250 + Edit@750 | 0.155 | 0.387 | 0.028 | 2.47 |
  | 正式 S1@1000（同样 3.2 万行） | 0.310 | 0.431 | 0.194 | 1.87 |
  | 正式 S2@16000（v4） | 0.467 | 0.597 | 0.310 | 1.93 |

- 结论：Read 太短（目标识别行约 8,800 行），读 token 只学到正式 S1@1000 的一半，多 token 几乎没学到；读 token 相关的组件臂可能和参照拉不开差距。用户选择把 Read 延长为正式 S1 的前 1,000 次后重启（v2）。
- v1 的输出（`ABLATION_ROOT/train/ref/`、`train/no_identification/read` 的部分结果、`submissions/abl_001/`）保留，不用于论文。

**v2：**11-12 下午由用户分两个四机作业提交（`abl_v2c_001`、`abl_v2s_001`），快照 `81de09f`（= `0ed8852`）。启动后在此补记。
