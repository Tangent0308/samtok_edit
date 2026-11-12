# 文档（分支 `qwen-image-2.1-r2-delivery`，最终交付版本）

| 文档 | 内容 |
|---|---|
| [METHOD.md](METHOD.md) | 最终方法 SITEdit（R2@1000）：设计、部件、损失、三阶段配方、推理、代码实现、权重路径与 SHA256、运行命令、局限 |
| [EXPERIMENTS.md](EXPERIMENTS.md) | 从 v4 两阶段重训（10-30）到 R2 定稿与外部 benchmark 全量（11-11）的全部训练与评测实验，每项都附结果文件路径；末尾列出已更正的错误 |
| [DATA.md](DATA.md) | 最终方法用到的数据：合板数据 D1–D15、区域池、S1/S2/R2 的训练行（逐行统计）、自蒸馏高质量目标池、Pass1 数据、评测数据 |

**Pass1（定位）**是最终方法的一部分，它的文档单独保留：

| Pass1 文档 | 内容 |
|---|---|
| [方法与协议](pass1_localization.md) | 两轮架构、字段含义、一次联合输出、NTP 损失 |
| [代码实现](pass1_implementation.md) | 目录、关键路径与源码块、冻结与恢复 |
| [正式数据](pass1_formal_data.md) | 来源、实际物化数量、筛选、隔离与覆盖缺口 |
| [实验结果](pass1_experiments.md) | pilot、正式 test、benchmark 十种 setting |
| [问题分析](pass1_analysis.md) | 失败案例、根因、局限 |
| [运行与复现](pass1_reproduction.md) | 安装、CLI/API、训练恢复、评测与产物路径 |

**历史文档**放在 `archive/v4/`，只作追溯，不代表当前方法：
- v4 的数据、方法、实验三份文档，覆盖 E-A、C1–C3、OPSD、S1/S2、IF-Bench 全量的过程记录；
- v4 的交接说明 `HANDOFF_v4.md`。

其他约定：
- 图片在 `assets/`。
- 文档里的代码块由 `scripts/docs/sync_code_blocks.py` 从源码同步（只检查 `docs/*.md`）。
- CVPR 投稿的整体交接文档在 NAS：`/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/CVPR2027_HANDOFF.md`（11-12 起只在 NAS 上更新）。
