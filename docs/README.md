# SAMTokEdit Qwen-Image-2.1 文档（v4 分支）

## Pass1：混合目标定位（10-31 由 `pass1` 分支合入）

Pass1 的数字是定位指标，不等于 DiT 编辑成绩。

| Pass1 文档 | 内容 |
|---|---|
| [方法与协议](pass1_localization.md) | 两轮架构图、字段含义、一次联合输出、NTP 损失 |
| [代码实现](pass1_implementation.md) | 规范目录、关键路径/源码块、冻结和恢复 |
| [正式数据](pass1_formal_data.md) | 来源、实际物化数量、筛选/隔离与覆盖缺口 |
| [实验结果](pass1_experiments.md) | pilot、正式 test、benchmark 十种 setting；训练曲线和前后对比图 |
| [问题分析](pass1_analysis.md) | 失败案例、根因证据、局限及尚未实施的改进 |
| [运行与复现](pass1_reproduction.md) | 安装、CLI/API、训练恢复、评测与产物路径 |

## v4 编辑方法

| 文档 | 内容 |
|---|---|
| [01 数据](01_SAMTokEdit_Qwen21_v4数据.md) | 合板后的训练数据盘点（来源、格式、文字复查、默认视图、统计），训练行的构建（含 OPSD 行），评测数据 |
| [02 方法](02_SAMTokEdit_Qwen21_v4方法.md) | 方法的设计与架构、各部件与损失、C3-D 配方、OPSD 算法设计与试训诊断；代码实现：基于 DiffSynth 和 SAMTok 的组织方式、模块、链路、正确性验证、运行方法 |
| [03 实验](03_SAMTokEdit_Qwen21_v4实验.md) | 训练与评测实验的设置、结果、分析与结论，关键的可视化，产物路径 |

- v4 图片在 `docs/assets/v4/`；Pass1 图片和汇总指标在 `docs/assets/pass1/`，manifest 记录原图路径与 SHA256。
- 文档里的代码块由 `scripts/docs/sync_code_blocks.py` 从源码同步，测试会检查是否过时；文档里的路径用 `scripts/docs/check_paths.py` 检查。
- v3 及更早的文档（v2 分析、四机运行指南、外部编辑数据打标管线等）留在 `qwen-image-2.1-v3` 分支。
