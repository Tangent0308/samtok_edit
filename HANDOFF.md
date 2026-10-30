# SAMTokEdit 交接说明

写于 2026-10-28，原开发机回收之前。读者是没有上下文的新会话：先读完本文，再按 §2.2 的指引读文档。本文只给方向和入口，细节以文档和 NAS 上的 README 为准，两者不一致时以盘上的结果文件为准。

**路径记号**
- `NAS/…` 指 `/mnt/bn/strategy-mllm-train/user/tanyue/…`（共享盘，机器回收后仍在）；
- `V3/…` 指 `NAS/experiments2/SAMTokEdit/qwen21_v3/…`（第二轮全部实验产物，文档里也用这个记号）；
- `分支:路径` 指 GitHub 仓库 `Tangent0308/samtok_edit` 某个分支里的文件，可用 `git show origin/<分支>:<路径>` 读；
- 仓库内路径（`docs/…`、`src/…`）都相对本仓库（分支 `qwen-image-2.1-v4`）根目录。

## 0. 一分钟版

- **任务：**SAMTokEdit，在 Qwen-Image-2.1 编辑管线上做区域编辑。SAMTok 把编辑区域编码成 4 个 token 的 mask token，直接写进指令；DiT（原权重 + LoRA + 码本注入）只凭 token 定位和编辑。推理时不解码 mask、不加注意力偏置、不融合 latent；mask 只在训练时用。详见 `docs/02_SAMTokEdit_Qwen21_v4方法.md` §0–§4。
- **验收口径（用户 10-18 定）：**只看带 token 的编辑，必须好于原版。token 来自用户给的区域，或由 TE 从文字生成。不带 token 时比原版差可以接受。
- **时间线：**目标 CVPR 2027。注册 11-10、投稿 11-16、补充材料 11-23（AoE）。之前的计划是 11-05 前后冻结方法。
- **当前最好的结果：**
  - 训练权重：C3-D，`V3/runs/c3d_local8_001/train/adapter_1500`。
  - 默认系统：C3-D + 第一步定位 + token 引导 2.0。在 benchmark 2.6.0 的 301 例决策子集上，官方判分的整图全过约 36%–38%；同一子集上原版 11.7%，FLUX.3 klein 27.7%。
  - 分数最高的学生是 OPSD 试训的 P1@500 + 引导（两次判分分别为 40.3%、38.9%）。但它对 P0 + 引导只高 +4.3 个点 [−0.3, +9.0]，不显著，所以没有换默认系统。
  - 第二次 OPSD 试训的判分里同场重判，C3-D + 引导为 37.2%，高于那次的所有学生系统。
- **最近的工作：OPSD（在线自蒸馏），两次试训后停掉（10-29）。**同一个 DiT 当老师，老师看 GT mask、物体名，并加引导；学生只看 token，在自己的轨迹上向老师靠拢。
  - 第一次（10-26/27）没有提升。原因是损失被区域外的"照抄源图"项主导，而且试训行上老师很少比学生好。
  - 第二次（10-28/29）按诊断修了选行、损失和选状态，事先写定的两条判据都没过：开发集不执行 P1′ 45 对 P0′ 47。
  - 根子在极小区域：剩下的不执行集中在改动 < 0.1% 的区域，模型在那里读 token 的 IoU 只有 0.1–0.2。老师的优势来自物体名和回填，学生推不出来。
  - 全部写在 `docs/03_SAMTokEdit_Qwen21_v4实验.md` §7。
- **当前：**两阶段重训（文档 03 §11）。10-30 扩池完成（新增可用区域 124 万个），`rows_s1.jsonl`、`rows_s2.jsonl` 已生成并核对（问题数 0）；代码修复、两阶段冒烟、推理冒烟都已做完。下一步：用户把 `V3/launch/s1_4n_full.sh` 全文作为四机启动命令开 S1（`s1_4n_001`，约 8–10 小时），训完按开发集挑点后开 S2（`scripts/round2/settings/s2_4n.sh`）。合板数据 D12–D15 10-29 并入（§4.1b）。

## 1. 协作约定（用户的规则，必须遵守）

1. **语言：**给用户的所有最终输出都用中文，包括回复、进度、确认请求和仓库文档，英文术语尽量少。commit message 用英文。
2. **提交：**
   - 格式 `<type>: <subject>`（MaiXiaochai CommitMessageGuidelines），type 取 feat、fix、docs、refactor、test、chore 等；
   - **绝不**加 Claude 署名、`Co-Authored-By` 或 "Generated with Claude Code"。这条覆盖工具默认给的署名提示；
   - 推送前检查 `git log <远程>..HEAD --format=%B | grep -ciE "claude|co-authored|anthropic"`，结果必须是 0。
3. **密钥：**
   - W&B key 用 ARNOLD 注入的 `WANDB_API_KEY`，否则读 `NAS/experiments2/SAMTokEdit/.secrets/wandb.env`；
   - 绝不打印、绝不提交，仓库是公开的；
   - 也不要把其他进程的命令行整段打出来（可能带凭据）。
4. **可以改的仓库：**
   - `qwen-image-2.1-v4`（当前开发）、`qwen-image-2.1-v3`、labeling 仓库的 `extedit-protocol2-labeling` 分支；
   - 其他仓库、数据集、别的会话的临时目录都只读。
5. **删除：**
   - 删除既有数据必须等用户在自己的消息里明确同意；
   - 没有同意时，移到 `_trash_<日期>/` 并写 MANIFEST，等确认；
   - 自己产生的冗余产物可以直接删。
6. **放置：**
   - 正式数据放 `NAS/datasets/<名字>/`，一个数据一个目录，带 README；
   - 实验与数据管线的产物放 `NAS/experiments2/SAMTokEdit/` 下的对应目录（第二轮实验在 `qwen21_v3/`，数据管线在 `data_pipeline/`），带 README；
   - 文档里要写清路径。
7. **运行仓库代码：**
   - 用 `PYTHONDONTWRITEBYTECODE=1 python -B`，不要用 `-I`（会在仓库里留下 pyc）；
   - 临时脚本、核对脚本放自己的临时目录，不入库；仓库的单元测试在 `tests/`。
8. **先方案后执行：**训练、全量运行、试训都要用户同意；只问方案时只给方案。
   - **单机入口：**用户 10-28 要求每个实验写好单机入口，结果输出到 NAS 对应的目录，由用户另开机器启动来加速。新机器只跑实验，本机可以在会话里调整。
   - 模板是 `V3/analysis/opsd_pilot2_20261028/` 的 `entry.sh`（新机器克隆代码、建环境）、`worker.sh`（按 NAS 上的原子认领和心跳领任务）和 `tasks.sh`（任务表），见该目录 README 的"多机运行"一节。
   - 给用户一条启动命令，并说明开几台机器有用。
9. **长任务：**
   - 盯住进度，出错及时修复；
   - 预计完成时间用北京时间报；
   - 不要动别人的 GPU 进程。原机器上用户常驻一个 `run.py --size 8000 --gpus 8 …`，它会拖慢判分，要暂停由用户自己决定。
10. **交付形式：**这位用户那里 Artifact 页面打不开，结果给本地或 NAS 路径。
11. **飞书进度文档：**先在本地起草，用户确认后再推。链接和读写方法见记忆 `lark-progress-doc.md`（§2.4）。
12. **文档规范：**
    - 三份文档分工：01 数据、02 方法、03 实验。
    - 每个实验按"目的、设置、评测、结果、分析与结论、图、路径"写，结论附配对 95% 区间。
    - 文档里的代码块由 `scripts/docs/sync_code_blocks.py` 从源码同步；路径用 `scripts/docs/check_paths.py` 检查（`tests/test_docs.py` 会查代码块）。
    - 图放 `docs/assets/v4/`，图注写原图路径。
    - 作图用 `V3/tools/fonts/` 的 Noto Sans CJK。

## 2. 资源地图

### 2.1 代码（GitHub）

| 仓库与分支 | 用途 | 状态（10-28） |
|---|---|---|
| `Tangent0308/samtok_edit` 的 `qwen-image-2.1-v4` | **当前开发**：第二轮方法 + OPSD 训练器；DiffSynth 2.1.12 放在 `third_party/diffsynth`（加了两个可选钩子） | 已推送，含本文 |
| 同仓库 `qwen-image-2.1-v3` | 第二轮的原始代码和 v3 文档（文档 03 §2–§6 的实验都用它跑）。该分支 `docs/` 下：04 代码说明，05 四机（ARNOLD）运行指南，07 外部数据打标管线 | 已推送，不再开发 |
| 同仓库 `qwen-image-2.1-v2` | v2 方法；数据清洗与协议适配（`protocol_adapt`） | 10-28 推送了积压的 6 个提交（`4bbac7f..5c06d7a`） |
| 同仓库 `qwen-image-2.1-dev`、`dev`、`main` 等 | 更早的版本 | 只读 |
| `Tangent0308/sam3-crispedit` 的 `extedit-protocol2-labeling` | 外部编辑数据（CrispEdit、ScaleEdit）协议 2.0 打标与文字复查管线 | 已推送 |
| 同仓库 `crispedit-labeling`、`refedit-labeling`、`scaleedit-labeling`、`samtok-derived-edit-labeling` | 早期各数据源的打标 | 与远程一致，只读 |
| `Tangent0308/samtok_edit_benchmark` 的 `v2branch` | 用户的 benchmark 与统一指令协议；判分用的是提交 `bf2283d` | 只读 |

### 2.2 文档（按需读）

| 文档 | 内容 | 什么时候读 |
|---|---|---|
| `docs/03_SAMTokEdit_Qwen21_v4实验.md` §0 | 当前结论，所有训练与评测一览 | **最先读** |
| 同上 §7 | OPSD 试训、四项诊断、第 1 步（开发集老师差距）、第二场（P0 + 引导） | 继续 OPSD 之前必读 |
| 同上 §9、§10 | 结论汇总与下一步；产物路径总表 | 找结论、找产物 |
| `docs/02_SAMTokEdit_Qwen21_v4方法.md` §0–§4 | 方法、部件与损失、C3-D 配方 | 动方法之前 |
| 同上 §5、§6 | OPSD 设计（§5.11 是试训诊断与修订方向）；代码实现与验证；怎样运行（§6.8） | 改 OPSD 代码之前 |
| `docs/01_SAMTokEdit_Qwen21_v4数据.md` | 合板训练数据（来源、格式、默认视图、统计）、训练行构建（含 OPSD 行）、评测数据；§6 是已知问题与待办 | 取行、动数据之前 |
| `V3/README.md` | 实验树索引 | 找某次运行的产物 |
| `V3/analysis/opsd_pilot_20261026/README.md` | 试训的全过程：设计、判定口径、结果、诊断、下一步、运行方法 | 继续 OPSD 之前 |
| `NAS/datasets/SAMTokEdit_Merged_20261029/README.md` | 合板数据说明（10-29，含 D12–D15） | 取行 |
| `NAS/experiments2/SAMTokEdit/data_pipeline/README.md` | 数据管线产物的目录规范 | 动数据管线 |
| `qwen-image-2.1-v3:docs/05_SAMTokEdit_Qwen21_四机实验运行指南.md` | 在 ARNOLD 上提交四机训练 | 需要多机时 |

### 2.3 NAS

| 路径 | 内容 |
|---|---|
| `NAS/datasets/SAMTokEdit_Merged_20261029/` | 合板训练数据（10-29）：九个来源 D1、D3、D4、D10、D11、D12–D15；默认视图 384,533 条记录、462,273 个单元。代码的 `MERGED_ROOT` 指向它。图片是指向各来源的硬链接，不要原地改 |
| `NAS/experiments2/SAMTokEdit/cleanup_20261029/` | 10-29 经用户同意的清理（约 830 GB）：文件清单、日志、脚本；合板 10-26 版、CrispEdit-2M 原始分片、未完成的 2 万孪生/多目标生产、判分裁图、续跑保存点、没用到的中间权重 |
| `NAS/datasets/samtok_edit_benchmark_v2/` | benchmark v2 2.6.0：`benchmark/cases.jsonl`，1,084 例 |
| `NAS/datasets/ExtEdit_Protocol2_20261021/`、`NAS/datasets/ExtEdit_Protocol2_20261022/` | D10、D11：外部编辑数据，协议 2.0 |
| `NAS/datasets/SAMTokEdit_Cleaned_20261009/` | 早期清洗与协议 1.0 适配的产物 |
| `V3/` | 第二轮全部实验：数据与训练行、权重、开发集评测、决策子集评测、分析；见 `V3/README.md` |
| `NAS/experiments2/SAMTokEdit/data_pipeline/` | 数据管线的运行与验证（SA-1B 50k 在 `02_production_runs/sa1b_samtok_edit_50k_20261014/`） |
| `NAS/experiments2/SAMTokEdit/benchmark_v2_2.6.0_baselines/`、`NAS/experiments2/SAMTokEdit/benchmark_v2_1084_baselines/` | 官方基线（原版、FLUX.3 等）的输入、输出和判分 |
| `NAS/experiments2/SAMTokEdit/handoff_20261028/` | 本次回收前的备份（§7） |
| `NAS/models/pretrained_models/Qwen-Image-2.1`、`NAS/models/SAMTok/Qwen3-VL-8B-SAMTok` | 底座与 SAMTok（`src/samtok_edit21/models/pipeline.py` 的默认路径） |
| `NAS/models/pretrained_models/Qwen3.8-Flash-Next-FP8` | benchmark 官方判分模型 |

### 2.4 Claude 的记忆

- 原机器上 Claude Code 的项目记忆是一组 md 文件，`MEMORY.md` 是索引。里面记着每个阶段用户的决定和原话（数据协议规则、验收口径、W&B 与飞书的用法等），比文档细。
- 备份在 `NAS/experiments2/SAMTokEdit/handoff_20261028/claude_memory/`。
- 恢复方法：复制到新机器的项目记忆目录 `~/.claude/projects/<工作目录路径把 / 换成 - >/memory/`。原工作目录 `/opt/tiger/tanyue` 对应 `-opt-tiger-tanyue`。
- 记忆反映的是写入时的情况；与文档或盘上的结果冲突时，以后者为准。
- 会话记录（jsonl）备份在 `NAS/experiments2/SAMTokEdit/handoff_20261028/claude_project_state/`，一般不用读。

## 3. 进展（截至 10-29）

文档 03 §0.1 有完整的当前结论，这里只列脉络：

| 阶段 | 日期 | 结论 | 文档 03 |
|---|---|---|---|
| 第一轮 E-A1、E-A2 | 10-11 至 10-13 | 只用编辑对训练，DiT 不会读 token；加指认任务和路由监督后学会 | §2 |
| C1、C1b | 10-16 至 10-17 | 在 SA-1B dense mask 上学定位；瓶颈在 DiT 区分近邻的精度，不在数据量 | §3 |
| C2（四机） | 10-18 至 10-20 | 以带 token 的编辑为主。benchmark 101 例整图全过 34.7%（原版 12.9%）；输在只给文字的单元、替换、写字、大目标改属性 | §4 |
| C3 | 10-23 至 10-25 | 改数据（C3-D）有效且不伤保持：决策子集 35.0%，对 C2 +5.0，显著。改区域项（C3-M）伤保持。推理时加 token 引导，几次判分里再高 1.7–3.7 个点（不显著），以少量保持换执行 | §5 |
| 数据线 | 10-20 至 10-26 | 外部数据 D10、D11 打标；五个来源合板，10-26 修正规则后重建 | 文档 01 |
| OPSD 老师上限 | 10-25 至 10-26 | 老师 T3（回填 + 带名 + 引导）比学生高 10 个点，保持也更好；v4 分支实现了 OPSD 训练器 | §6；文档 02 §5、§6 |
| OPSD 试训与诊断 | 10-26 至 10-27 | 见下 | §7 |
| OPSD 第二次试训 | 10-28 至 10-29 | 修了选行、损失和选状态，机制、部署两条判据都没过，停掉 OPSD | §7.2 |

**OPSD 试训的要点**（P1 = 一半行做 OPSD，P0 = 同一批行全做普通训练；都从 C3-D@1500 起训 500 次）
- **没有提升：**
  - P1 对 P0：整图全过 −0.7 [−5.3, +4.0]；
  - P1 + 引导对 P0 + 引导：+4.3 [−0.3, +9.0]。高出的部分来自保持（单元 P +0.015），不是执行（单元 E +0.006）。
- **原因一：损失被区域外主导。**
  - 损失在整个 latent 上取平均，区域内只占损失的 7%–18%、梯度的 7%–26%；
  - 区域外的目标就是源图 latent，所以 P1 主要学会了"照抄源图"，这一点不需要老师。
- **原因二：训练行上老师很少更好。**
  - 试训用的合板行大多是显著的大目标，学生已经做得到：76 条留出记录里，老师从头接手只救回 3 条；
  - 而且结果在高噪声段就定了。
- **第 1 步（开发集 100 条 SA-1B 编辑，小区域盲指代）：**
  - 不执行：学生 48，学生 + 引导 35，不带名的老师 31，T3 22；
  - 超出引导的部分主要来自带名（−9 个点）；
  - 学生先走 8 步（σ 0.73），老师只能追回一半以下。
  - 结论：OPSD 的前提在 D1 这类数据上成立，问题在选行和损失。
- **默认系统不换：**按试训前写定的口径，只有 P1 + 引导明显好过 P0 + 引导才换。

## 4. 下一步（新的训练和全量运行都要用户先同意）

### 4.1 OPSD 第二次试训（10-28 至 10-29，已完成：停掉 OPSD）
- **做了什么：**
  - 按诊断改了三处：选行（OPSD 行全取 D1 小区域的删除、添加、改属性）；损失（A 内外分开归一，`--opsd-loss split`）；选状态（75% 取自 σ ≥ 0.73）。`λ_opsd` 重新标定为 2。
  - P1′ 对 P0′，从 C3-D@1500 起各 500 次更新。
  - 两台机器跑，共 13.2 小时：本机训练 P1′ 并判分，另一台训练 P0′、做生成和开发集评测。
- **结果**（`V3/analysis/opsd_pilot2_20261028/criteria.txt`、`report.txt`）：
  - 机制没过：开发集 100 条小区域编辑的不执行，P1′ 45、P0′ 47、C3-D 48，P1′ − P0′ = −2 [−7, +3]；
  - 部署没过：决策子集 P1′ + 引导对 P0′ + 引导 +2.7 [−2.7, +7.6]，保持不变；
  - 同场重判的 C3-D + 引导 37.2%，高于两者。
- **诊断：**
  - 区域内的蒸馏误差在训练中几乎不降，学生没有靠近老师。
  - 不执行集中在改动 < 0.1% 的极小区域：49 条里 C3-D 35 条不执行，老师 T3 17 条。模型在那里读 token 的 IoU 只有 0.1–0.2。
  - 老师的优势来自物体名和回填，学生从 token 推不出来。
- **结论：**按事先写定的判定停掉 OPSD。两次试训的权重都不换上去，代码里的 OPSD 选项保留、默认关闭。
- **详情：**`V3/analysis/opsd_pilot2_20261028/README.md`（含结果和多机运行）；文档 03 §7.2；文档 02 §5.11。

### 4.1b 新数据（10-29 已并入合板）
- **已完成：**数据线的孪生 12,511 组、多目标 39,814 张图（100,117 个目标）、附带单目标 12,363 条，10-29 并入合板，成为 D12、D13、D14。
  - 默认视图新增 76,366 条记录、134,436 个单元；旧来源逐字节不变。
  - 用户的决定：添加的选区在落点还原不了时改用宿主物体；尽可能用上全部生产数据；附带单目标也用。
  - 详见文档 01 §2.7；转换的运行目录 `NAS/experiments2/SAMTokEdit/data_pipeline/02_production_runs/merge_sa1b_groups_20261029/`；盘点 `V3/analysis/new_data_inventory_20261029/README.md`。
- **单目标 v2.1（5 万条）：**10-29 完成，当天并入为 D15（默认视图 49,089 条）。
- **以下是 10-28 时的计划：**10 万条多目标、2.5 万条孪生在打标；另计划 10 万条单目标。
- 用户 10-28 问过"是否用 scale 后的数据从头重训第一阶段、再调第二阶段"。我的建议（记忆 `samtokedit-data-scaling-question-20261028.md`）：
  - 不串行：先在 C3-D 上修好第二阶段（即 §4.1）；
  - 新数据先按标志分层抽检（孪生要查两个实例能否被 token 区分、改的是不是对的那个），再并进合板数据；
  - 做 C4：从 C3-D（或更好的第一阶段）续训，以多目标为主、加旧数据回放，对照用同样步数不加新数据，同一次判分比较；
  - 从头重训只在冻结方法后做一次，和论文消融一起。
- 依据：数据量不是瓶颈（默认视图 259,078 条，C2 一次只用了 112,704 行）；起作用的是对准失败类型的数据（C3-D 换四分之一的行就 +5 个点）。
- 10-29 补充：OPSD 已停，孪生、多目标数据直接做普通训练。剩下的不执行集中在改动 < 0.1% 的极小区域（文档 03 §7.2），所以：
  - 抽检时按改动面积分层，统计极小区域的占比；
  - C4 适当提高极小区域、同类多实例行的比重，开发集按面积分档报告（文档 03 §9 第 3 条）。

### 4.2 定型与论文（无论 OPSD 结果如何都要做）
- **选定最终系统：**默认是 C3-D + 第一步定位 + 引导 2.0。两次 OPSD 试训的模型都没有显著好过它。
- **全量评测：**benchmark 1,084 例（约 4 h 生成，另加判分），与官方基线同场判分。3 个点以下的差别只有全量才测得出。
- **论文消融：**edit-only、去掉路由、去掉注入（文档 03 §9 第 5 条）。
- **还没解决的问题**（文档 03 §9）：大目标改属性没到 0.90；盲指代改到旁边的同类物体；写字拼错；替换做成叠加。

### 4.3 其他（优先级较低）
- **仓库修复（10-30 已修）：**`scripts/round2/cuda_readiness.py` 原来按 v3 已删的模块名起探测，新机器建环境必然失败；v4 `e6d3887b` 改为以脚本自身起探测（文档 03 §9 第 7 条）。
- **保持损失：**区域项只算改动处，同时给外圈单独加保持项（文档 03 §9 第 3 条）。
- **双路条件：**原版 TE 管文字语义，SAMTok TE 只给 token。这会改变用户 10-18 定的"第二遍只用 SAMTok TE"，需要用户拍板。
- **TE LoRA：**从纯文字指令生成第一遍 token，是用户定的后续方向。
- **数据线待办：**见文档 01 §6。

## 5. 在新机器上恢复

1. **代码：**按 `README.md` 的 Install 一节 clone `qwen-image-2.1-v4` 并建环境（Python 3.11，`uv pip install -r requirements.txt`；在 H100 / CUDA 12.8 上验证过的版本见 `constraints-tested.txt`）。不装包时，在仓库根目录设 `PYTHONPATH=src:third_party/diffsynth:third_party`。
2. **自检：**

   ```bash
   PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:third_party/diffsynth:third_party python -B -m pytest -q -p no:cacheprovider tests
   PYTHONDONTWRITEBYTECODE=1 python -B scripts/docs/check_paths.py
   ```

   第一条跑全套 63 项测试，只用 CPU；第二条检查文档路径，需要挂 NAS。
3. **判分环境与 benchmark 代码：**按 `NAS/experiments2/SAMTokEdit/handoff_20261028/envs/README.md` 重建，里面有 vLLM 0.31.0+cu129 的安装包。benchmark 代码用 `v2branch` 的 `bf2283d`。
4. **Claude 的记忆：**见 §2.4。
5. **旧脚本里写死的本机路径要换掉。**试训、诊断、判分脚本里有：

   | 原路径 | 换成 |
   |---|---|
   | `/opt/tiger/tanyue/samtok_edit_qwen-image-2.1-v4` | 新的 v4 clone |
   | `/tmp/samtok21-fixes-dUnbt5/venv` | 新的主环境 |
   | `/tmp/claude_judge_env129` | 新的判分环境 |
   | `/tmp/claude_bench/samtok_edit_benchmark_v2branch` | 新的 benchmark clone（`bf2283d`） |
   | `/tmp/claude_v4/diag` | `V3/analysis/opsd_pilot_20261026/diagnosis` |

6. **开跑之前：**用 `nvidia-smi` 看清 GPU 上有没有别人的进程。

## 6. 经验与坑

**评测**
- **判分噪声大：**
  - 同一批输出隔开两次判分，逐例的整图全过只有 78% 一致；同一份 P1 + 引导两场差 1.4 个点。
  - 系统比较一律放在同一次判分里。
  - 300 例上整图全过的配对区间约 ±5 个点，3 个点以下的差别要用全量。
- **判定口径在跑之前写进 README，**不事后挑指标。
- **开发集与 benchmark 量的不是一回事：**开发集的"不执行"（改动像素内的变化减区域外变化 < 5）和 benchmark 的 E、P 不同，两边都要看。

**方法（OPSD）**
- **特权信息要学生推得出来：**先按面积等维度看老师的优势落在哪里，再看学生在那里能不能从自己的输入读出对应的信息。第二次试训里，老师的优势在极小区域、来自物体名，学生在那里读不准 token，蒸馏就传不过去。
- **损失要分开归一：**在整个 latent 上取平均，小区域的信号会被区域外淹没；区域内外要分开归一、分开记录。
- **蒸馏前先测老师：**先确认老师在训练行上真的更好（接管测试很便宜），并找出结果在哪个噪声段决定。

**工程**
- **多机运行**（第二次试训）：
  - 按任务认领（NAS 上 mkdir 原子认领 + 心跳，超时接管）比固定串联灵活；
  - 最长的串行步骤（判分，每个系统约 40–60 分钟）要最先排，来一个判一个；
  - 配对比较的两组输出尽量放在同一套环境里做；
  - 监视要能发现远端机器死掉：看各机器存活文件的新旧、未完成任务的认领者是否还在，不能只看全局有没有新文件。
- **新机器的入口要从头测到尾，包括 CUDA 检查：**v4 的 `setup_env.sh` 最后一步调用了已删掉的模块，两台新机器都在这一步失败（§4.3）。
- **被停掉的训练要带 `--resume` 重开：**训练只走到 W&B 初始化就被停时，W&B 记录已经建了（id 由输出路径决定），不带 `--resume` 会被拒绝；没有 `resume.pt` 时 `--resume` 仍从头训练。
- **无人值守的串联脚本：**
  - `set -u` 下 `shift` 之后再用 `$1`，脚本会静默退出；
  - 等待脚本要同时盯状态文件、驱动进程的 PID，以及"多久没有新文件"；
  - 正式启动前，先在临时沙盒里把重活换成桩，跑通整条串联。
- **`pgrep -f` 的模式**不要能匹配到别的等待脚本自己的命令行，用 `[r]un_s2.sh` 这种写法。
- **原机器的 `find` 是 bfs，**不支持 `-newermt '-45 minutes'`，要用 `-mmin`。
- **vLLM 判分有逐例缓存，**可以杀掉进程组暂停，再从缓存续跑。
- **按块提交：**`git apply --unidiff-zero` 会把新增行放错位置，要用带上下文的 diff 并检查暂存区。
- **测试容差：**float32 下批量计算与逐行计算的线性层有 1e-6 量级的差，相关测试要固定种子、放宽容差。
- **ARNOLD 镜像缺 `libxcb.so.1`，**`import cv2` 会失败（已改用 SciPy 的 EDT）。
- **第一轮的 LoRA 初始化各卡不一致，**`V3/legacy_code/` 只作留档，不要再运行。
- **DiffSynth 2.1.9 起** Qwen-Image-2.1 默认改用分段 SDPA；v4 统一传 `use_flex_attention=True`，保持训练时的数值路径。

**数据**
- **复查要按标志分层抽，**不能只随机抽：罕见失败（0.06%）在 1,200 行的随机样本里看不到。
- **全量之前，**先在接近生产分布的样本上演练。

**安全**
- **备份或打包之前先扫凭据。**W&B key 是 40 位十六进制，和 git 提交哈希长得一样，要用"是不是已知提交"来区分。

## 7. 回收前的备份与 git 状态（10-28）

`NAS/experiments2/SAMTokEdit/handoff_20261028/`：

| 内容 | 说明 |
|---|---|
| `HANDOFF.md` | 本文的副本（以仓库里的为准） |
| `local_workspace_backup/` | 原工作区 `/opt/tiger/tanyue/` 下用户自己的笔记、压缩包、`local_archive/`、`samtok_stage2_case_display_20261004/`；dev clone 未提交的文档 06 与 `assets/`（附 diff）；DiffSynth clone 里的解读笔记。清单 `MANIFEST.tsv` |
| `tmp_scratch_20261028.tar`、`tmp_scratch_20261028.list.txt` | `/tmp` 下各临时目录的小文件（脚本、日志、小结果，3,389 个，2.0 GB）；不含模型副本、环境、缓存和凭据 |
| `tmp_samtok21_fixes_20260926.tar` | 9-26 的旧验证（129 个小文件） |
| `redacted_job_scripts/` | ARNOLD 任务配置和入口脚本（19 个），凭据已换成 `<REDACTED>` |
| `envs/` | 各环境的依赖清单、判分环境的安装日志、vLLM 安装包、重建方法 `README.md` |
| `claude_memory/`、`claude_project_state/` | Claude 的记忆、会话记录 |

- **也搬进了实验树：**
  - v4 开发时的核对（等价、OPSD 正确性、合板重建）→ `V3/analysis/v4_code_checks_20261026/`；
  - OPSD 诊断 → `V3/analysis/opsd_pilot_20261026/diagnosis/`。
- **没有备份：**
  - `/tmp` 下的模型副本（原件在 `NAS/models/`）；
  - 虚拟环境（按 `envs/README.md` 重建）和各种缓存；
  - 两份 `wandb.env`（key 已在 `.secrets/`）；
  - 一个 `.pem` 文件。
- **git 状态：**
  - `qwen-image-2.1-v4`：已推送，含本文；
  - `qwen-image-2.1-v2`：推送了积压的 6 个提交；
  - v3、labeling、benchmark 各分支：与远程一致；
  - 两处未提交的改动只备份到了 NAS，没有提交：
    - `qwen-image-2.1-dev` 的本地 clone 有 10-05 的改动（`docs/README.md`、`docs/06_…`、`assets/`）；
    - 工作区根目录的 git 仓库（code.byted.org 上的 `tanyue`）`README.md` 有改动。
