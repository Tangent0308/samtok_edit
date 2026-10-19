# SAMTokEdit Qwen-Image-2.1 文档（v3 分支）

先读前三份：数据、方法、实验。

| 文档 | 回答的问题 |
|---|---|
| [01 数据](01_SAMTokEdit_Qwen21_v3数据.md) | 统一的指令协议（1.0 → 2.0 盲指代）；v1 数据的问题、清洗和协议适配；新的数据管线（SA-1B 50k 自蒸馏、多目标召回、dense mask 区域池）和产出的数据量；第二轮训练行构成；开发评测集与 benchmark 2.6.0 要点；全部数据路径 |
| [02 方法](02_SAMTokEdit_Qwen21_v3方法.md) | v2 的问题；v3 每个部件是什么、为什么加、哪一步加入、代码在哪；各阶段的方法演进；参考的工作；已知局限 |
| [03 实验](03_SAMTokEdit_Qwen21_v3实验.md) | E-A1、E-A2、C1、C1b、C2 的数据、设置、权重和结果；各项评测的设置与指标；诊断和分析（注意力探针、纯文字判分、不执行诊断、原版对比、benchmark 拆分 case）；结论与下一步；整理后的产物目录 |
| [04 代码实现说明](04_SAMTokEdit_Qwen21_代码实现说明.md) | 方法在官方 DiffSynth 和 SAMTok 代码之上怎样实现：每一块在哪个文件、关键代码（自动同步）、命令、默认超参数、测试 |
| [05 四机实验运行指南](05_SAMTokEdit_Qwen21_四机实验运行指南.md) | 在 ARNOLD 上怎样启动实验（第 7 节是当前使用的第二轮入口，含可直接粘贴的完整脚本）；改设置、看进度、失败处理 |
| [06 v2 分析](06_SAMTokEdit_Qwen21_v2分析.md) | v2 为什么不如原版、v2 的 DiT 有没有用 mask token；证据、图和产物路径（v3 设计的出发点） |

维护约定：
- 代码有改动时同步改文档 04。它的代码块和文档 05 的完整脚本由 `python scripts/docs/sync_code_blocks.py` 从源码生成，`pytest tests/test_docs.py` 检查两者是否一致。
- 新的实验结果记在文档 03（相应章节和第 6 节结论），数据侧的变化记在文档 01。
- 文档引用的路径用 `python scripts/docs/check_paths.py` 检查（需要挂着共享盘）。
- 文档里的图放在 `assets/` 下对应的子目录：`v3_progress/` 是第一轮，`v3_round2/` 是第二轮，`v2_review/` 是 v2 分析。

路径：
- 代码：`/opt/tiger/tanyue/samtok_edit_qwen-image-2.1-v3`，分支 `qwen-image-2.1-v3`（从 `qwen-image-2.1-v2` 分出，v2 分支保持不变）。
- v3 实验产物：`/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v3/`，目录说明见其中的 `README.md`。
- v2 实验产物：`/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v2/`。
- benchmark：仓库 `Tangent0308/samtok_edit_benchmark` 的 `v2branch`，数据根 `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/samtok_edit_benchmark_v2`。

存档：
- [`archive/v3_process/`](archive/v3_process/)：v3 进行中写的计划、需求和进度（原 03、04、06、07、08 号文档），内容已整理进 01–03。
- [`archive/v2/`](archive/v2/)：v2 阶段的九份文档。
- [`archive/v1/`](archive/v1/) 和 [`archive/`](archive/) 下的其余文件：v1 及更早的记录。
