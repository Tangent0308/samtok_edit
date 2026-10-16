# SAMTokEdit Qwen-Image-2.1 文档（v3 分支）

当前文档共八份，按用途分工，互相不重复：

| 文档 | 回答的问题 |
|---|---|
| [01 代码实现说明](01_SAMTokEdit_Qwen21_代码实现说明.md) | 当前方法是怎样在官方 DiffSynth 和 SAMTok 代码之上实现的；每一块在哪个文件、关键代码是什么；命令、格式、默认超参数、测试 |
| [02 v2 分析](02_SAMTokEdit_Qwen21_v2分析.md) | v2 为什么不如原版；v2 的 DiT 到底有没有用 mask token。每个结论对应的证据、图和产物路径 |
| [03 v3 优化计划与进度](03_SAMTokEdit_Qwen21_v3优化计划与进度.md) | 相关工作与创新点；v3 的优化方案和实验计划；每个实验做了什么、结果如何；发现的问题与修正；下一步 |
| [04 数据工作](04_SAMTokEdit_Qwen21_数据工作.md) | 训练数据和评测 benchmark 的盘点；v2 数据的问题；清洗和协议适配的方案与进度；哪些路径是当前真正在用的，哪些是中间产物 |
| [05 四机实验运行指南](05_SAMTokEdit_Qwen21_四机实验运行指南.md) | 在 ARNOLD 上怎样启动一次 v3 实验（以 E-A2b 为样例）；怎样改参数；产物、进度查看和失败处理 |
| [06 孪生与多目标数据需求](06_SAMTokEdit_Qwen21_孪生与多目标数据需求.md) | 给数据侧的需求：反事实孪生组（同一张图、同一句 no-ref 指令、两个同类实例各自一张目标图），以及多目标联合编辑样本（同一张图 2–5 个目标，链式生成）。怎样选目标、写指令、生成、审核，交付哪些字段、多大规模 |
| [07 指令协议 2.0：no-ref 改为盲指代](07_SAMTokEdit_Qwen21_指令协议2.0_noref改为盲指代.md) | 训练数据和 benchmark 共用的指令改写规则（定稿）：no-ref 一律写成 "the selected region"，源 mask 是实际被编辑的区域；各操作句式与改写示例、规范模块要改什么、两边各自的改法和自检清单 |
| [08 第二轮实验计划](08_SAMTokEdit_Qwen21_第二轮实验计划.md) | 换成协议 2.0 的新数据、从原版重新训练：每个实验用什么数据、什么任务、训练哪些部件、从什么权重起训、方法设计、规模、耗时和放行线 |

维护约定：
- 代码有改动时同步改文档 01。它的代码块由 `python scripts/docs/sync_code_blocks.py` 从源码生成，`pytest tests/test_docs.py` 检查两者是否一致。
- 新的实验结果记在文档 03 第 5 节，并更新它第 0 节的状态表；数据侧的进展记在文档 04。
- 文档里的图放在 `assets/` 下对应的子目录（`v2_review/`、`v3_progress/` 等）。

路径：
- 代码：`/opt/tiger/tanyue/samtok_edit_qwen-image-2.1-v3`，分支 `qwen-image-2.1-v3`（从 `qwen-image-2.1-v2` 分出，v2 分支保持不变）。
- v3 实验产物：`/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v3/`。
- v2 实验产物：`/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v2/`。

存档：
- [`archive/v2/`](archive/v2/)：v2 阶段的九份文档，原样保留（内部链接已改成能从新位置打开）。
  - 01 代码实现说明（v2）、02 实验记录、03 四机实验运行指南（v2 的 Stage 1 / Stage 2 / MLLM 编辑核验入口，仍然有效）、04 训练数据盘点；
  - 05 独立案例研究、06 add 失败原因分析、07 v1 分析与 v2 计划、08 v2 阶段分析、09 训练数据诊断与改进（逐对审计的完整记录，文档 04 是它的提要）。
- [`archive/v1/`](archive/v1/) 和 [`archive/`](archive/) 下的其余文件：v1 及更早的记录。
