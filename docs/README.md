# SAMTokEdit Qwen-Image-2.1 文档（v4 分支）

| 文档 | 内容 |
|---|---|
| [01 数据](01_SAMTokEdit_Qwen21_v4数据.md) | 合板后的训练数据盘点（来源、格式、文字复查、默认视图、统计），训练行的构建（含 OPSD 行），评测数据 |
| [02 方法](02_SAMTokEdit_Qwen21_v4方法.md) | 方法的设计与架构、各部件与损失、C3-D 配方、OPSD 算法设计；代码实现：基于 DiffSynth 和 SAMTok 的组织方式、模块、链路、正确性验证、运行方法 |
| [03 实验](03_SAMTokEdit_Qwen21_v4实验.md) | 训练与评测实验的设置、结果、分析与结论，关键的可视化，产物路径 |

- 图片在 `docs/assets/v4/`，图注里给出原图路径。
- 文档里的代码块由 `scripts/docs/sync_code_blocks.py` 从源码同步，测试会检查是否过时；文档里的路径用 `scripts/docs/check_paths.py` 检查。
- v3 及更早的文档（v2 分析、四机运行指南、外部编辑数据打标管线等）留在 `qwen-image-2.1-v3` 分支。
