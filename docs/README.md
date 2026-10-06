# SAMTokEdit Qwen-Image-2.1 文档

当前主要文档：

1. [代码实现说明](01_SAMTokEdit_Qwen21_代码实现说明.md)：方法、官方起点、项目改动、代码索引和关键代码。
2. [实验记录](02_SAMTokEdit_Qwen21_实验记录.md)：debug、noref 转换、四机排错和全量准备结果。
3. [四机实验运行指南](03_SAMTokEdit_Qwen21_四机实验运行指南.md)：ARNOLD/W&B 环境、远程 clone、完整入口命令和产物验收。
4. [训练数据盘点](04_SAMTokEdit_Qwen21_训练数据盘点.md)：四个源数据集、过滤、字段映射、最终 metadata 和统计。

5. [独立案例研究](05_SAMTokEdit_Qwen21_独立案例研究.md)：逐例视觉观察和方法相对 baseline 的表现。
6. [add 失败原因分析](06_SAMTokEdit_Qwen21_add失败原因分析.md)：add 评测、数据与 loss 诊断、attention 证据和改进建议。
7. [v1 分析与 v2 计划](07_SAMTokEdit_Qwen21_v1分析与v2计划.md)：v1 的逐 case 复核、where 信号链路、TE/DiT 探针、反事实与 knockout、逐级消融、训练数据对齐审计；第 8 节为 v2 实施计划与进度。

历史原始记录保存在 [`archive/`](archive/)；其中的命令和路径用于追溯，不作为当前入口。

当前开发 checkout：`/opt/tiger/tanyue/samtok_edit_qwen-image-2.1-v2`，分支 `qwen-image-2.1-v2`（基于 `qwen-image-2.1-dev` 的 a93fe56）。`qwen-image-2.1-dev` 保持不变。实现说明链接指向当前 src/third_party；实验历史保留旧运行证据。
