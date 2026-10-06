# SAMTokEdit（Qwen-Image-2.1）v2 实验记录

本文按时间记录 v2 分支的实验：做了什么、命令、结果、产物路径、遇到的问题和处理。方法与代码见[代码实现说明](01_SAMTokEdit_Qwen21_代码实现说明.md)，计划与进度见 [v1 分析与 v2 计划](07_SAMTokEdit_Qwen21_v1分析与v2计划.md)第 8 节。v1 的全部实验记录已存档在 [archive/v1](archive/v1/02_SAMTokEdit_Qwen21_实验记录.md)。

2026-10-06 起，v2 产物写在 `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v2/`（下文记为 `$V2`）。此前的产物在 intern 旧根目录 `/mnt/bn/strategy-mllm-train/intern/users/tanyue/experiments/SAMTokEdit/qwen21_v2/`（下文记为 `$V2_OLD`）：数据、评测和 smoke 数据已逐字节复制到 `$V2`，本地 smoke 运行和日志只在 `$V2_OLD`（第 8 节）。

## 0. 状态（2026-10-06）

| 里程碑 | 状态 |
|---|---|
| M0：代码、测试、数据转换、评测编译器 | 完成，本地八卡 smoke 全部通过；`review.md` 待人工抽检 |
| M1：E1 Stage 1 + pass-1 评测 | 运行 A 首次提交（2026-10-06）因 intern 配额已满在入口处失败（第 7 节）；入口已修复，实验根目录已移到 user 目录（第 8 节），待重新提交 |
| M2–M4 | 未开始 |

## 1. 环境

| 项 | 值 |
|---|---|
| 机器 | 本地 1 节点 × 8 × 80 GB GPU |
| 代码 | `/opt/tiger/tanyue/samtok_edit_qwen-image-2.1-v2`，分支 `qwen-image-2.1-v2` |
| 训练/推理环境 | `/tmp/samtok21-fixes-dUnbt5/venv`：Python 3.11、torch 2.8.0、transformers 5.12.1、accelerate 1.14.0、peft 0.20.0、byted-wandb 0.13.98（与集群锁定版本相同） |
| noref 转换器环境 | `/tmp/samtok21-layout-annotation-env`：vLLM 0.17.1、torch 2.10（生产标注锁定版本） |
| judge 环境 | `/tmp/samtok21-benchmark-judge-env`：vLLM 0.28 |

本地包未以 editable 方式安装，运行时设置 `PYTHONPATH=$REPO/src:$REPO/third_party/diffsynth:$REPO/third_party`（集群上由 `pip install -e .` 提供 `samtok` 包）。

注意：实验期间 8 张 GPU 上一直有一个不属于本实验的常驻进程（`/mnt/bn/strategy-mllm-train/user/tanyue/run.py --size 8000 --gpus 8`，每卡 1.6 GB、约 41% 利用率）。它不影响正确性，但下文的耗时都是在共享 GPU 上测得的，集群独占时应更快。

## 2. 训练数据转换（v1 → v2）

```bash
cd /opt/tiger/tanyue/samtok_edit_qwen-image-2.1-v2
PYTHONPATH=src:third_party/diffsynth:third_party /tmp/samtok21-fixes-dUnbt5/venv/bin/python \
  -m samtok_edit21.preparation.v2_data --output $V2/data/train_v2_box_001
```

- 结果：Stage 1 110,079 行（edit_ntp 96,319 + rec_ntp 13,760），Stage 2 290,006 行；剔除 composite 1,206 个源；add 改框 12,556 个源；Derived add 改为 attribute 6,269 个源。耗时 3.5 分钟（commit `c411c1c`）。
- 一致性：每条 v1 行与 v1 provenance 的 row hash 核对通过；同一命令先写入 `/tmp/v2conv_full`，再写入正式目录，两次的 `stage1.jsonl`/`stage2.jsonl` 逐字节一致。按 provenance 比对，323,662 行与 v1 完全相同；改动的行为 add 37,668、Derived add 24,995、新增 rec_ntp 13,760。
- 框与 v1 span 的对应抽查（随机 300 个 add 单元，codec 解码 v1 span 后取外接框）：与 v2 框的 IoU 均值 RefEdit 0.53、CrispEdit 0.44、ScaleEdit 0.38，IoU=0 的比例 0–2%。与 v1 分析中 add 区域的 codec 往返损失（0.38）一致，说明框和原 span 对应同一实例。
- 首次试转换（前 2,000 个源）耗时 100 s，几乎全在共享盘读 mask PNG；改为先分类、只读需要的 mask 并用线程池后降到 8.5 s，输出不变。

统计与格式见[训练数据盘点](04_SAMTokEdit_Qwen21_训练数据盘点.md)。

## 3. 单元测试

```bash
PYTHONPATH=src:third_party/diffsynth:third_party /tmp/samtok21-fixes-dUnbt5/venv/bin/python -m pytest -q tests/
```

45 项全部通过（约 8 s，CPU）。覆盖内容见[代码实现说明第 10 节](01_SAMTokEdit_Qwen21_代码实现说明.md#10-测试与等价性)。token 布局测试使用真实的 Qwen-Image-2.1 processor 和 SAMTok tokenizer，并通过官方 PromptEmbedder 得到实际输入 id（TE 用桩替代）。

## 4. GPU 等价性检查

单卡（GPU 0），512² 画布、4 步；pass-1 adapter 用 v1 的 Stage 1 adapter（权重非零）。

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:third_party/diffsynth:third_party /tmp/samtok21-fixes-dUnbt5/venv/bin/python \
  scripts/diagnostics/check_v2_equivalence.py \
  --te-adapter /mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/qwen21_full4_20260928/runs/qwen21_full_4n_formal_003/stage1/adapter \
  --image <benchmark 源图> --output <report.json>
```

| 检查 | 结果 |
|---|---|
| pass 2 关闭 adapter vs raw TE | 逐位一致；开启 adapter 时特征最大差 600（确实生效） |
| region_embed 零初始化 vs 无绑定 | 输出图逐位一致 |
| 全 1 融合 mask vs 不融合 | 逐位一致 |
| bias_span：KV cache vs 无缓存 | 像素平均绝对差 0.126（无偏置时同一比较为 0.128）；偏置本身使输出变化 0.96 |
| bias_clause：KV cache vs 无缓存 | 0.140；偏置本身使输出变化 2.40 |
| β=0 vs 无绑定 | 0.147（kernel 级差异，与上面同量级） |
| region_rope 生效 | 输出变化 1.47 |

**v1/v2 一致性。** 同一组 v1 adapter（Stage 1 + Stage 2 `resume_006`）、同一 inline prompt、512²、6 步，分别用 v1 checkout（`/opt/tiger/tanyue/samtok_edit_qwen-image-2.1-dev`）和 v2 代码（`pass2_te=adapter`，绑定、融合关闭）推理：两张 PNG 逐字节相同（sha256 `8e7c9863…09bd`）。说明 vendored DiT 的改动（移除 attention probe、加入绑定钩子）不改变原计算。

## 5. 八卡 smoke

数据：`$V2/smoke/data_smoke_001`（72 个源，Stage 1 80 行、Stage 2 212 行，覆盖全部类型和两种区域表示）。运行器为正式四机入口使用的同一编排（`distributed.training --local`），W&B offline。

### 5.1 256² 全流程（B0）

```bash
cd /opt/tiger/tanyue/samtok_edit_qwen-image-2.1-v2
export PYTHONPATH=$PWD/src:$PWD/third_party/diffsynth:$PWD/third_party SAMTOK_LOCAL_PORT=<空闲端口>
/tmp/samtok21-fixes-dUnbt5/venv/bin/python -m samtok_edit21.distributed.training --local \
  --run-root $V2_OLD/smoke/runs/smoke_002_b0 --data $V2/smoke/data_smoke_001 --wandb-mode offline \
  --phases stage1,cache,stage2 --binding none --stage1-steps 2 --stage2-steps 3 \
  --stage1-save-steps 8 --stage2-save-steps 4 --max-pixels 65536 --timeout 7200
```

结果：`SUCCESS.json`，`audit.json` 通过。

| 阶段 | 墙钟 | 检查要点 |
|---|---|---|
| collectives | 27 s | 8 rank all-reduce/broadcast |
| Stage 1 | 107 s | 每个 update：edit_ntp 56、rec_ntp 8（7:1，每 rank 7:1）；NTP loss 0.79 → 0.85（2 update，含 bbox 目标） |
| 缓存 | 91 s | 212 行，格式 v3，`te_adapter: null`；140 个区域行全部带 payload（mask 单元 4 token，框单元约 21 token），普通行无 payload；无空区域 |
| Stage 2 | 101 s | 每个 update：ref 8、noref 16、plain 8；`region_row` 0.75；loss_fm 0.16/0.126/0.128 |
| 审计 | 69 s | update 数、配比、梯度、权重一致、adapter 有限、缓存 212 行全部校验 |
| 推理 smoke | 79 s | 8 种推理路径全部输出有限、非恒定的 RGBA 图 |

推理 smoke 的 8 个用例：direct；inline ref mask；inline noref 框 + 融合；online 两遍 + 融合；oracle mask + 融合；oracle 框；inline noref mask + 推理期绑定覆盖（B0 上加 bias_clause，绑定 1 个单元）+ 融合；inline ref 框 + 无 KV cache。online 用例中，只训练 2 步的 Stage 1 对 add 指令仍输出 mask_2d；这是正式 E1 要解决的格式问题，不是代码问题。

### 5.2 256² 四个绑定臂（复用 5.1 的缓存与 Stage 1 adapter）

`--phases stage2 --cache $V2_OLD/smoke/runs/smoke_002_b0/cache --stage1-adapter $V2_OLD/smoke/runs/smoke_002_b0/stage1/adapter --binding <arm> --binding-beta 2 --binding-eps 0.05 --stage2-steps 3`

| 臂 | 审计 | 训练时 bound_units | adapter | 推理 smoke 绑定单元数（8 例） |
|---|---|---|---|---|
| bias_span | 通过 | 0.75（全部区域行） | LoRA | 0,1,1,1,1,1,0,1（第 7 例覆盖为 none） |
| bias_clause | 通过 | 0.75 | LoRA | 同上 |
| region_embed | 通过 | 0.75 | LoRA + `region_embed.down/up` | 0,1,1,1,1,1,1,1（region_embed 不能在推理时关闭） |
| region_rope | 通过 | 0.75 | LoRA | 0,1,1,1,1,1,0,1 |

第 1 例为 direct（无区域），所以绑定单元数为 0。

### 5.3 全分辨率（1M 像素）计时与显存

同一 smoke 数据，`--max-pixels 1048576`：B0 全流程（Stage 1 3 update、缓存 212 行、Stage 2 3 update），再用该缓存跑 bias_clause、region_embed、region_rope 各 3 update。四个运行全部 `SUCCESS.json`、审计通过。

| 项 | 稳态耗时 | 峰值显存 |
|---|---|---|
| Stage 1 NTP | 1.1–1.2 s/microstep（约 10 s/update） | 20.5 GiB/卡 |
| 缓存 | 约 1.7 s/行/卡（含 codec 解码） | — |
| Stage 2 B0 | 6.7–7.3 s/microstep（约 28 s/update） | 20.6 GiB/卡 |
| Stage 2 bias_clause | 7.7–8.1 s/microstep（约 +10–15%） | 20.6 GiB/卡 |
| Stage 2 region_embed / region_rope | 6–7 s/microstep | 20.6 GiB/卡 |

FlexAttention 的 score_mod 在长序列（约 8k token）、梯度检查点下正常编译，无重编译告警。每个缓存行约 9.6 MB，全量 290,006 行约 2.8 TB。

产物：`$V2_OLD/smoke/runs/smoke_002_{b0,bias_span,bias_clause,region_embed,region_rope}`、`$V2_OLD/smoke/runs/smoke_003_fullres_{b0,bias_clause,region_embed,region_rope}`；日志 `$V2_OLD/smoke/logs/`。

### 5.4 问题与处理

| 问题 | 原因 | 处理 |
|---|---|---|
| `smoke_001_b0` 在 collectives 阶段失败：`EADDRINUSE` 端口 29541 | 本机该端口已被占用 | 用 `SAMTOK_LOCAL_PORT` 指定空闲端口；新 run ID `smoke_002_b0`（编排器拒绝复用 run 目录） |
| 本地启动报 `No module named samtok_edit21` / codec 找不到 `samtok` | 本地环境未 editable 安装 | 本地设置 `PYTHONPATH`；集群 `setup_env.sh` 新增 codec 导入检查（缓存阶段首次在集群使用 codec） |
| 评测转换器 vLLM 报 `No such file or directory: 'ninja'` | 需把环境 `bin/` 放进 PATH（标注入口原本如此） | 运行时 `PATH=/tmp/samtok21-layout-annotation-env/bin:...` |
| 首次评测 manifest 在 `--require-complete` 下报缺失 | 非 add case 本来就不跑 `text_plain`（D9） | manifest 增加 `--compiled`，只对 add 期望 `text_plain` |
| 一次本地提交意外带入了已暂存的删除文件 | 提交顺序 | 推送前重做本地提交，使每个提交只含对应改动 |

## 6. 评测管线 smoke

评测目录：`$V2/eval/protocol_v2_001/`。

### 6.1 case 与编译

```bash
REPO=/opt/tiger/tanyue/samtok_edit_qwen-image-2.1-v2; EVAL=$V2/eval/protocol_v2_001
PY=/tmp/samtok21-fixes-dUnbt5/venv/bin/python
PYTHONPATH=$REPO/src $PY -m samtok_edit21.evaluation.cases --output $EVAL/cases.jsonl
PYTHONPATH=$REPO/src $PY -m samtok_edit21.evaluation.compile sources --cases $EVAL/cases.jsonl --output $EVAL/semantic_sources.jsonl
CUDA_VISIBLE_DEVICES=0 VLLM_WORKER_MULTIPROC_METHOD=spawn \
PATH=/tmp/samtok21-layout-annotation-env/bin:/tmp/samtok21-qwen35-compare-env/bin:$PATH PYTHONPATH=$REPO/src \
  /tmp/samtok21-layout-annotation-env/bin/python -m samtok_edit21.preparation.semantic \
  --sources $EVAL/semantic_sources.jsonl --output $EVAL/semantic --batch-size 64
PYTHONPATH=$REPO/src $PY -m samtok_edit21.evaluation.compile compile --cases $EVAL/cases.jsonl \
  --annotations $EVAL/semantic --output $EVAL/compiled.jsonl
```

- case：715 个（CompBench add 233 / remove 237 / replace 22，HumanEdit 24，MIRAGE 单区域 1 + 拆分 198）。dev 183 个（163 张源图）、test 532 个（452 张源图）。MIRAGE 拆分中有 2 个子句与选择文件中的 refer_object 措辞不同（`mirage_048#r2`、`mirage_054#r2`），按子句顺序对应区域，仅作诊断标记。
- 转换器：709/715 接受（模型 707、规则 2），6 个失败退回确定性模板；耗时 38 s（含模型加载 27 s）。
- 编译类型：add 263、remove 266、replace 33、attribute 152、text 1。MIRAGE 的 "replace" 原子编辑大多是颜色/材质修改，按训练规则编为 attribute；全部 715 个模板通过训练 noref 语法校验。
- 6 个模板回退都是较长的 CompBench add 指令（如 `add a fish on the right of the second fish ...`），模板保留了位置描述并追加 `in this region {region}`。
- 转换器对部分 add 指令会删去外观/姿态描述（如 `... with similar color but opposite direction` → `add a goldfish in this region`）；训练 noref 数据由同一转换器生成，分布一致，但需在人工抽检中留意。
- 人工抽检表：`$EVAL/review.md`（每类最多 50 条，含 noref 与 ref 两种模板）。

**初步人工抽检（2026-10-06，作者本人）。**

| 类型 | 结论 |
|---|---|
| remove（266） | 全部为 `remove the object in this region {region}`，正确 |
| replace（33）、text（1） | 逐条检查，旧对象替换为 `the object in this region`，新内容保留，正确 |
| attribute（152） | 抽查 40 条，属性名词保留（`Change the color/material/texture of this region {region} to X`），修复了 v1 评测编译器丢失属性名词的问题 |
| add（263） | 随机 30 条：15 条正确（只替换放置描述，保留新物体外观/姿态）；6 条（20%）把新物体的姿态/外观一起删了（如 `with its back to us and head facing left`、`with its back facing`、`that is the same as other planes`）；4–6 条保留了部分放置描述并追加 `in this region` |

add 的问题来自 9B 转换器对放置短语之后的从句的处理：CompBench 的 add 指令常把新物体的姿态写在放置短语之后（`add a yellow cow on the right side of the brown cow with its back to us ...`），转换器把放置短语连同其后内容一起删掉。训练数据的指令多把外观写在放置短语之前：随机抽查训练集 30 条 add 的原指令与 noref 行，没有一条丢失新物体自身的外观/姿态（6 条保留了部分放置描述）。所以这是评测指令风格带来的问题，交互 setting 下模型收到的要求会少于 judge 看到的原指令，在这一点上评测 noref prompt 与训练分布并不一致。为此编译结果同时提供 `ref_template`（原指令 + 区域 token，同样是训练格式，709/715 个 case 可用，其余 6 个是模板回退的 case），推理可用 `--prompt-variant ref`（输出目录加 `+ref` 后缀）。建议交互 setting 下的 add 两种都评：noref 为计划默认，ref 作对照；若两者差异明显，以 ref 结果作为 add 的主要参考。

### 6.3 noref 转换 prompt 的 add 规则优化

6.1 中的 add 问题是 9B 转换器没有遵守已有要求（保留新物体外观/姿态，只删放置描述）：姿态写在放置短语之后时，会连同放置一起删掉。优化过程（每次都用 Qwen3.5-9B、vLLM 0.17.1、temperature 0，同时跑 715 个评测 case 和训练集随机 300 条 RefEdit/CrispEdit/ScaleEdit add 指令；产物 `$EVAL/prompt_addfix{,2,3}/`）：

| 尝试 | 改动 | 评测 add（抽查的 30 条） | 训练风格 add（299 条，与 v1 结果比较） | 非 add |
|---|---|---|---|---|
| 1 | 在共享 prompt 中加 add 规则，换 add 示例 | 7 条修复，但 5 条把放置短语移到 `in this region` 之后 | 135 条改变，较多变差（放置残留、丢 "leaping off the cliff" 等） | 19 条改变，其中 1 条丢属性名词（"Change the fur color of…" → "Change this region to blue"） |
| 2 | 共享 prompt 不变；只在 add 专用部分加规则和两个示例（删除整个放置、保留尾部姿态） | 8 条改变全部为修复；放置残留消失 | 34 条内容改变，约一半变差（"perched/hanging/nestled/parked" 随放置一起被删） | 不变 |
| 3（采用） | 在 2 的规则中加一句：放置前的姿态词保留（"perched on the log" → "perched in this region"） | 8 条改变全部为修复（如 `Add a yellow cow in this region ⟨B⟩ with its back to us and head facing left.`）；仍漏 2 条歧义从句 | 263 条不变；36 条改变中约 24 条更好（删净放置、保留 perched/sitting/standing/walking 等）、约 9 条中性、3 条变差（1 条放置残留、丢 "nestled"、丢 "medium size"） | 内容不变 |

采用尝试 3（commit `874c9a1`）。用它重跑评测编译：715/715 由转换器接受（模型 714、规则 1），不再有模板回退，全部有 ref 模板；相对旧 prompt，263 个 add 中 67 个改写改变，非 add 不变。当前 `$EVAL/compiled.jsonl`、`review.md`、`semantic/` 为新结果；旧结果保留为 `compiled_promptv1.jsonl`、`review_promptv1.md`、`semantic_promptv1/`。

训练数据按决定暂不改动，仍是旧 prompt 的转换结果。如果下一轮数据改进用新 prompt 重跑，约 12% 的 add noref 行会改变，按上面的抽查以改进为主。由于评测与训练现在用的转换 prompt 只在 add 部分不同，评测 noref 的 add 改写比训练行略"干净"（放置删得更彻底、姿态保留更完整），两者仍是同一种格式。

### 6.2 推理、stock、judge 与汇总

用全分辨率 smoke 的 B0 adapter（只训练 3 步，结果数值无意义，只检查流程），6 个 dev case（CompBench add 2、remove 2，MIRAGE attribute 1、add 1）× 全部 setting × 融合开/关，共 51 张图（8 卡，约 25 s/张）：

```bash
PYTHONPATH=$REPO/src:$REPO/third_party/diffsynth:$REPO/third_party \
$PY -m torch.distributed.run --standalone --nproc_per_node 8 -m samtok_edit21.evaluation.run \
  --cases $EVAL/smoke/cases.jsonl --compiled $EVAL/compiled.jsonl --output $EVAL/smoke/smoke_b0 \
  --te-adapter <run>/stage1/adapter --dit-adapter <run>/stage2/adapter --split all --blend both
```

- 各 setting 的区域 token 与计划一致：add 在 mask/box/point 下都是框（point 用默认框）；remove/attribute 为 codec span（box/point 先经 SAM2）；text 走 pass 1；`text_plain` 只对 add。融合面积合理（例如 add 框外扩后约 0.5，remove 小目标约 0.04）。
- 已知局限：MIRAGE 部件编辑在 point setting 下，SAM2 最高分候选可能是整个人而非裤子（候选面积 0.10）。计划中可选的"部件类另报最小候选"尚未实现。
- stock（MIRAGE 原子 case，`--stock`）：用 benchmark 仓库的 `render_annotation` 和 `two_image_locator_prompt` 生成单区域的两图协议输入，例如 `Edit Image 1. For the region marked by the red mask in Image 2, change the texture of the pants to striped. ...`。
- stock 复现性：用 v2 的 stock 路径重跑已有输出 `qwen21_656` 的 case 0000（mask、text 两个 setting），prompt 与输入完全相同，但输出不是逐位相同：像素平均绝对差 0.42（mask）、0.27（text），44–46% 像素完全相同。差异量级与 kernel 级差异相当（第 4 节 KV cache 对比为 0.13），推测来自运行环境不同。复用的 517 个 stock 输出与新跑的 198 个 MIRAGE stock 输出可以比较；如需完全同一环境，可用 `--stock` 重跑全部 715 个 case。
- judge manifest：75 行（smoke_b0 51 + stock 24，其中 16 行链接 `qwen21_656` 已有输出），无缺失。用未改动的 pair_v2 judge（Qwen3.8-27B，4 卡）评 30 行子集（mask、mask+blend、text），30/30 成功；`evaluation.score` 生成分组表和配对 bootstrap。

```bash
PYTHONPATH=$REPO/src:$REPO/third_party/diffsynth:$REPO/third_party $PY -m samtok_edit21.evaluation.manifest \
  --cases $EVAL/smoke/cases.jsonl --compiled $EVAL/compiled.jsonl --run smoke_b0=$EVAL/smoke/smoke_b0 \
  --stock $EVAL/smoke/stock --split all --output $EVAL/smoke/judge_manifest.jsonl --require-complete
J=/mnt/bn/strategy-mllm-train/intern/users/tanyue/experiments/SAMTokEdit/qwen21_stage2_benchmark_noref_aligned_20261004/judge/code
cd $J && PATH=/tmp/samtok21-benchmark-judge-env/bin:$PATH PYTHONPATH=$J /tmp/samtok21-benchmark-judge-env/bin/python \
  -m evaluation.metrics.launch --run-dir $EVAL/smoke/judge --manifest <manifest> --devices 0,1,2,3 \
  --variants pair_v2 --split all --max-tokens 4096 --max-pixels 1048576 --batch-size 8
PYTHONPATH=$REPO/src $PY -m samtok_edit21.evaluation.score --records $EVAL/smoke/judge/all/records \
  --compiled $EVAL/compiled.jsonl --output $EVAL/smoke/report.json --compare stock smoke_b0
```

## 7. 四机运行 A 首次提交（2026-10-06）

入口为[四机指南第 2 节](03_SAMTokEdit_Qwen21_四机实验运行指南.md#2-运行-astage-1e1-缓存e2-stage-2-b0-seed-1e3)，run ID `qwen21_v2_4n_A_s1_b0_001`，代码 `5443a7b`。

- **现象**：Merlin 作业 `65f2a3f151b31f87`（trial `303524216`）13:49 提交，13:54 以 exit code 1 失败。ARNOLD 日志页的最后几行是常驻脚本 `run0926.sh` 的输出，看不到入口报错；共享盘上没有 `runs/qwen21_v2_4n_A_s1_b0_001/`。
- **原因**：intern 目录的 NAS 配额已满。入口在 `mkdir -p "$RUN/bootstrap"` 处收到 `Disk quota exceeded`，以 exit code 1 退出。这一步在节点日志创建之前，所以共享盘上没有任何记录。ARNOLD 用 `bash /tmp/full_script_bash_file.sh` 执行用户脚本，stdout 和 stderr 分开记录（`/opt/tiger/rh2/rh2/init/bootstrap/user_script/runner.py`），而 mkdir 的报错只写到了 stderr。
- **证据**：
  - 作业实际执行的脚本（用 `mlx job get` 取回）与指南相同，只在开头多了一行 `bash /mnt/bn/strategy-mllm-train/user/tanyue/run0926.sh`。`mlx job log` 连不上（websocket bad handshake），未能取回 stderr。
  - 14:20 在开发机上测试：`qwen21_v2/runs/` 和 `intern/users/tanyue/` 下新建文件都报 `Disk quota exceeded`，`user/tanyue/` 下可以写。`df` 显示整卷只用了 6%，看不出这个配额。
  - 退出码对照（本机 bash 5.2）：`mkdir` 失败为 1；入口里其他的显式检查都是 2，只有 `${VAR:?}` 和未定义变量也会得到 1，而这几个变量与 v1 调通的作业相同。
- **为什么不是我们自己占满的**：我们在 intern 下共约 40 GB（`qwen21_v2` 19.7 GB、`qwen21_full4_20260928` 3.4 GB、两个 benchmark 输出目录约 16 GB）。v1 缓存（2.8 TB）11:57 已删除，12:09 仍能正常写入。所以这个配额按比 `intern/users/tanyue` 更大的范围计算，满了之后需要管理员清理或扩容，我们这边删东西解决不了。
- **修复**（commit `9530dbb`，只改入口脚本和指南，训练代码不变，`SAMTOK_EDIT_COMMIT` 仍为 `5443a7b`）：
  - 入口第一行起把 stderr 并入 stdout，所有报错都会出现在 ARNOLD 日志页上。
  - 创建 run 目录前先在 `$SAMTOK_EXPERIMENT/runs/` 下写一个探针文件；写不进去就打印原因、以退出码 3 退出，不留下任何文件。
  - 本地验证（伪造四机 ARNOLD 变量，临时 run ID）：对当前的 intern 根目录，退出码 3，stdout 中有 `Disk quota exceeded` 和说明，stderr 为空，没有新建任何文件；对可写的临时目录并故意让 clone 失败，正常认领节点，写出 `failure.json`（阶段 checkout，退出码 128）。
- **后续**：实验根目录改到 user 目录（第 8 节）。这次什么都没有写，新根目录下也没有这个 run，可以沿用同一个 run ID 重新提交。

## 8. user 目录清理与实验根目录迁移（2026-10-06）

### 8.1 实验根目录

intern 配额满了以后，按你的决定把实验根目录改为 `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v2/`（新建 `experiments2/`，项目文件夹 `SAMTokEdit/`，其下仍按系列分 `qwen21_v2/`）。commit `60f875c`。

- **复制到新根目录**（intern 只需可读，原件保留）：`data/train_v2_box_001/`（0.36 GB，5 个文件）、`eval/protocol_v2_001/`（0.07 GB，241 个文件）、`smoke/data_smoke_001/`（3 个文件）。两边逐文件 sha256 一致；`stage1.jsonl`、`stage2.jsonl`、`provenance.jsonl` 的 sha256 与 `metadata_report.json` 一致。数据报告里只有 hash、没有绝对路径，训练入口按 hash 校验，所以复制件可以直接使用。smoke 数据的 `metadata_report.json` 里 `smoke_of` 仍指向 intern 原件，只是来源记录，不参与校验。
- **留在旧根目录**：本地 smoke 运行和日志（`$V2_OLD/smoke/runs/`、`$V2_OLD/smoke/logs/`，约 19 GB），作为第 5 节的证据，不再写入。
- **代码与文档**：`data/io.py` 的 `EXPERIMENT_ROOT`（只作参照，训练入口不读它）、四机指南、代码实现说明、数据盘点和计划中的路径已改为新根目录。训练代码不变，`SAMTOK_EDIT_COMMIT` 仍为 `5443a7b`。
- 新根目录已确认可写；能否放下约 2.8 TB 的缓存取决于 user 目录配额，开发机上看不到（四机指南第 1 节）。

### 8.2 清理 `user/tanyue/experiments/SAMTokEdit/`

经你确认，删除了旧管线（Qwen-Image-Edit-2511 + Qwen2.5-VL-7B-SAMTok，8–9 月）的数据和 checkpoint、各 smoke 运行和 v1 Stage 1 的中间 checkpoint，共 239.1 GB；目录总量从约 423 GB 降到 184 GB。删除记录（逐项路径、大小、理由、核对结果）：`/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/DELETED_2026-10-06.json`。

| 删除项 | 大小 |
|---|---:|
| `crispedit_refined/stage1_full/data/`、`crispedit_refined/stage2_full/data/`（旧管线物化的 CrispEdit 图片与 metadata） | 206.8 GB |
| 旧管线 checkpoint：`crispedit_refined/{stage1_full/train_8gpu_1ep,stage2_full/stage2_dit_lora}`、`crispedit_refined_4node/*-run2/stage*_lora`、`stage2_full_edit_mt/stage2_dit_lora`、`stage1_20k_mt` 中的 `*.safetensors`（30 个） | 16.1 GB |
| v1 Stage 1 中间 checkpoint `qwen21_full_4n_formal_003/stage1/step-*.safetensors`（13 个；最终 `adapter/` 保留） | 9.1 GB |
| smoke 运行：`crispedit_refined/stage{1,2}_8gpu_smoke`、`stage1_single_gpu_smoke`、`stage1_8gpu_smoke`、`stage2_8gpu_smoke`、`qwen_image_2_1_dev_smoke` | 6.6 GB |
| 旧数据：`validation_edit_mt_64/data`、`crispedit_refined_4node/*/data` | 0.5 GB |
| 写入中断留下的 5 个 `.tmp`（v1 region 缓存 3 个、9B 转换进度 2 个） | 36 KB |

- **删除前的检查**：v2 训练数据只引用 `qwen21_full4_20260928/data/assets/` 和 `datasets/SAMTok_Derived_Edit_Labeling`；v1 数据清单只引用 `assets/`；v2 评测只引用 benchmark 数据集和 `qwen21_656`；v2 代码、benchmark 清单和 benchmark 仓库都不引用删除项。抽查 757 个 v2 图片路径都是普通文件、不在删除范围内，768 个 asset 分片目录里没有软链接。3 个 region `.tmp` 对应的行在 manifest 中指向的 `coverage/*.pt` 都存在。
- **删除后的核对**：计划中的路径全部不存在；v1 Stage 1 adapter（698,425,744 字节）、3 × 256 个 asset 分片、v1 数据文件、benchmark 与评测输出、旧运行的日志和 loss 曲线、`.secrets` 都在；抽查 931 个 v2 训练图片全部可读。
- **保留**：`qwen21_full4_20260928/data/`（v2 训练图片和 v1 数据）、v1 Stage 1 adapter 与日志、benchmark 相关输出、旧模型的评测输出与可解释性分析、旧训练的 reports/logs/loss 曲线。归档文档（`docs/archive/`）里对 `qwen_image_2_1_dev_smoke` 的 2 处引用现已失效。
- 32 路并行，用时 45 秒。

## 9. 下一步

1. 人工复核 `$EVAL/review.md`（已用 6.3 的新 prompt 重新生成；确认 add 的交互 setting 是否同时评 ref 变体）。
2. 用新根目录重新提交四机运行 A（E1 + E2 + E3 seed 1），随后 B0 seed 2（[四机指南](03_SAMTokEdit_Qwen21_四机实验运行指南.md)）。
3. E1 完成后：pass-1 评测（非 add 的 mask IoU 不低于 v1：remove 0.72、replace 0.66；add 的 bbox 格式率 ≥ 95%）。
4. E3 完成后：dev 评测（B0 × 2 seed、融合开/关）、E4 推理期偏置扫描，然后 E5–E7。
