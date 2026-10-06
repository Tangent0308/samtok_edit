# SAMTokEdit（Qwen-Image-2.1）v2 实验记录

本文按时间记录 v2 分支的实验：做了什么、命令、结果、产物路径、遇到的问题和处理。方法与代码见[代码实现说明](01_SAMTokEdit_Qwen21_代码实现说明.md)，计划与进度见 [v1 分析与 v2 计划](07_SAMTokEdit_Qwen21_v1分析与v2计划.md)第 8 节。v1 的全部实验记录已存档在 [archive/v1](archive/v1/02_SAMTokEdit_Qwen21_实验记录.md)。

所有 v2 产物都在 `/mnt/bn/strategy-mllm-train/intern/users/tanyue/experiments/SAMTokEdit/qwen21_v2/`（下文记为 `$V2`）。

## 0. 状态（2026-10-06）

| 里程碑 | 状态 |
|---|---|
| M0：代码、测试、数据转换、评测编译器 | 完成，本地八卡 smoke 全部通过；`review.md` 待人工抽检 |
| M1：E1 Stage 1 + pass-1 评测 | 待四机运行 A（[四机指南第 2 节](03_SAMTokEdit_Qwen21_四机实验运行指南.md#2-运行-astage-1e1-缓存e2-stage-2-b0-seed-1e3)） |
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
  --run-root $V2/smoke/runs/smoke_002_b0 --data $V2/smoke/data_smoke_001 --wandb-mode offline \
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

`--phases stage2 --cache $V2/smoke/runs/smoke_002_b0/cache --stage1-adapter $V2/smoke/runs/smoke_002_b0/stage1/adapter --binding <arm> --binding-beta 2 --binding-eps 0.05 --stage2-steps 3`

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

产物：`$V2/smoke/runs/smoke_002_{b0,bias_span,bias_clause,region_embed,region_rope}`、`$V2/smoke/runs/smoke_003_fullres_{b0,bias_clause,region_embed,region_rope}`；日志 `$V2/smoke/logs/`。

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

add 的问题来自 9B 转换器对放置短语后的从句的处理。训练 noref 数据由同一转换器生成，所以评测与训练分布一致（D2 的初衷）；但交互 setting 下模型收到的要求可能少于 judge 看到的原指令。为此编译结果同时提供 `ref_template`（原指令 + 区域 token，同样是训练格式，709/715 个 case 可用，其余 6 个是模板回退的 case），推理可用 `--prompt-variant ref`（输出目录加 `+ref` 后缀）。建议交互 setting 下的 add 两种都评：noref 为计划默认，ref 作对照。训练数据中 add noref 行的同类信息丢失，记为下一阶段数据改进项。

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

## 7. 下一步

1. 人工复核 `$EVAL/review.md`（初查结果见 6.1；确认 add 的交互 setting 是否同时评 ref 变体）。
2. 四机运行 A（E1 + E2 + E3 seed 1），随后 B0 seed 2（[四机指南](03_SAMTokEdit_Qwen21_四机实验运行指南.md)）。
3. E1 完成后：pass-1 评测（非 add 的 mask IoU 不低于 v1：remove 0.72、replace 0.66；add 的 bbox 格式率 ≥ 95%）。
4. E3 完成后：dev 评测（B0 × 2 seed、融合开/关）、E4 推理期偏置扫描，然后 E5–E7。
