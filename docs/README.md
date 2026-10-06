# SAMTokEdit Qwen-Image-2.1 文档（v2 分支）

v2 主要文档：

1. [代码实现说明](01_SAMTokEdit_Qwen21_代码实现说明.md)：v2 方法与代码（区域协议、数据转换、Stage 1 定位、raw-TE 缓存、Stage 2 结构性绑定、推理融合、评测协议、测试与路径）。
2. [实验记录](02_SAMTokEdit_Qwen21_实验记录.md)：v2 的数据转换、单元测试、GPU 等价性检查、八卡 smoke、评测管线 smoke、问题与处理。
3. [四机实验运行指南](03_SAMTokEdit_Qwen21_四机实验运行指南.md)：ARNOLD 完整入口、运行 A（Stage 1 + 缓存 + B0）、Stage 2 消融臂、四机 smoke、产物与失败处理。
4. [训练数据盘点](04_SAMTokEdit_Qwen21_训练数据盘点.md)：v2 训练数据 `train_v2_box_001` 的来源、转换规则、统计、行格式与复现命令。

v1 的分析与 v2 计划：

5. [独立案例研究](05_SAMTokEdit_Qwen21_独立案例研究.md)：v1 逐例视觉观察和方法相对 baseline 的表现。
6. [add 失败原因分析](06_SAMTokEdit_Qwen21_add失败原因分析.md)：v1 add 评测、数据与 loss 诊断、attention 证据和改进建议。
7. [v1 分析与 v2 计划](07_SAMTokEdit_Qwen21_v1分析与v2计划.md)：v1 的逐 case 复核、探针、反事实与 knockout、逐级消融、数据审计；第 8 节为 v2 实施计划、决策与进度。

存档：v1 版本的 01–04 在 [`archive/v1/`](archive/v1/)（代码链接指向 v1 commit a93fe56）；更早的原始记录在 [`archive/`](archive/)。

- 代码：`/opt/tiger/tanyue/samtok_edit_qwen-image-2.1-v2`，分支 `qwen-image-2.1-v2`（基于 `qwen-image-2.1-dev` 的 a93fe56；`qwen-image-2.1-dev` 保持不变）。
- 实验产物：`/mnt/bn/strategy-mllm-train/intern/users/tanyue/experiments/SAMTokEdit/qwen21_v2/`（`data/`、`smoke/`、`runs/`、`eval/`）。
