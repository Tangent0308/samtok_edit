# SAMTokEdit Qwen-Image-2.1 实验记录

> 下方原有第 1–6 节为历史记录。2026-09-26 的独立审计发现，历史 cache identity/非零梯度检查的充分性曾被高估；其旧环境路径也已失效。当前实现、修复归属和新验收以第 7 节及实现文档为准，历史内容保留供追溯。

## 实验环境

- GPU：8 × H100 80 GB
- Python 环境：`/opt/tiger/tanyue/samtok_edit_qwen_image_2_1/.venv`
- PyTorch：2.8.0+cu128；Transformers 5.12.1；Accelerate 1.14.0；PEFT 0.20.0
- Qwen3-VL：`/mnt/bn/strategy-mllm-train/user/tanyue/models/SAMTok/Qwen3-VL-8B-SAMTok`
- Qwen-Image-2.1：`/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen-Image-2.1`
- 调试源数据：`/mnt/bn/strategy-mllm-train/user/tanyue/datasets/RefEdit-mask-prefiltered-qwen38-self-contained`
- 实验根目录：`/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/qwen_image_2_1_dev_smoke`

源 parquet 只读。构造脚本物化 8 个有效样本，生成 Stage 1 8 行（3 NTP、2 UMT-ref、2 UMT-noref、1 plain）和 Stage 2 8 行（2 UMT-ref、4 UMT-noref、2 plain）。

## 1. 数据构造与协议校验

运行：

```bash
PYTHONPATH=.:DiffSynth-Studio python tests/eight_gpu_smoke/prepare_refedit.py \
  --source /mnt/bn/strategy-mllm-train/user/tanyue/datasets/RefEdit-mask-prefiltered-qwen38-self-contained \
  --output /mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/qwen_image_2_1_dev_smoke/refedit_data \
  --samtok /mnt/bn/strategy-mllm-train/user/tanyue/models/SAMTok/Qwen3-VL-8B-SAMTok \
  --unique-rows 8
PYTHONPATH=.:DiffSynth-Studio python -m samtok_edit21.cli validate \
  --metadata /mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/qwen_image_2_1_dev_smoke/refedit_data/stage1.jsonl \
  --base-path /mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/qwen_image_2_1_dev_smoke/refedit_data
```

结果：`{"rows": 8, "decoded_images": 6, "passed": true}`。`refedit_data/build_report.json`、`provenance.json` 和物化图片记录了来源、mask checksum 信息和 smoke-only 标注。

SAMTok codec 启动时打印标准 SAM2 backbone 的 mask encoder missing keys，随后从 `mask_tokenizer_256x2.pth` 严格加载并输出 `Loaded checkpoint successfully`；8 个 mask 均成功编码。这是发布 checkpoint 的两段式加载行为，后续正式数据转换仍需保留启动日志审计。

## 2. 分布式设备与采样探针

Stage 1：

```bash
accelerate launch --main_process_port 45680 --num_processes 8 \
  tests/eight_gpu_smoke/probe_distributed.py \
  --metadata .../refedit_data/stage1.jsonl --stage stage1 --accumulation 8 --steps 1
```

8 个 rank 分别报告 `cuda:0` 至 `cuda:7`，每 rank 8 个局部样本，类型计数为 NTP=3、UMT=4（其中 ref/noref 各 2）、plain=1；全局报告为 NTP=24、UMT-ref=16、UMT-noref=16、plain=8。

Stage 2 使用 port 45681、`--stage stage2 --accumulation 4`。8 个 rank 均得到 4 个样本；全局报告为 UMT-ref=8、UMT-noref=16、plain=8。说明 schedule 在多卡切片后仍保持目标比例。

## 3. 发现的 bug 与修复

### 3.1 可选 XTuner 导入阻塞 codec

最初导入 `samtok.models` 会同时加载 XTuner 感知模型，和当前 Transformers 版本冲突，导致 mask codec 尚未初始化就失败。修复：`samtok/models/__init__.py` 只直接加载 VQ-SAM2，其他类通过 `__getattr__` 懒加载；SAM2/losses 内部改为相对导入。`mmengine` 加入运行环境依赖。

### 3.2 设备错误：8 个进程全部占用 GPU 0

第一次真实 8 卡训练在模型加载后观察到 GPU0 约 74 GB、其余 GPU 约 2 GB，原因是 `device_map={"":"cuda"}` 将字符串 `cuda` 解析为 device 0。该运行在 forward 前中断，没有产生训练 checkpoint。修复：`train.py:run_train` 和 `run_cache` 先构造 `Accelerator`，再把 `args.device` 设置为 `str(accelerator.device)`，每个进程显式加载到自己的 `cuda:<local_rank>`。之后设备探针确认映射正确。

### 3.3 端口占用

端口 29500 和 29601 已有外部监听，导致 launch 在初始化阶段报 `EADDRINUSE`；改用 45680 以上空闲端口。端口失败没有修改代码，也没有产生模型结果。

## 4. 真实 8 卡训练记录

以下命令用于最终 smoke；输出目录应使用新的空目录，避免和历史失败尝试混合：

```bash
accelerate launch --main_process_port 45682 --num_processes 8 --mixed_precision bf16 \
  -m samtok_edit21.train train --stage stage1 \
  --metadata .../refedit_data/stage1.jsonl --base-path .../refedit_data \
  --output .../stage1 --max-pixels 262144 --steps 1 --accumulation 8 \
  --save-steps 64 --num-workers 0 --seed 20260925
```

验收项：8 卡显存近似均衡；`schedule.json` 的全局计数正确；loss 有限；梯度审计中 LoRA 梯度非零且冻结参数梯度为 0；`adapter/adapter.safetensors` 和 `adapter.json` 存在且可重新加载。

Stage 2 cache：

```bash
accelerate launch --main_process_port 45683 --num_processes 8 --mixed_precision bf16 \
  -m samtok_edit21.train cache --metadata .../refedit_data/stage2.jsonl \
  --base-path .../refedit_data --te-adapter .../stage1/adapter \
  --output .../cache --max-pixels 262144 --num-workers 0
```

验收项：8 个 rank 的 cache shard、sidecar 和 `manifest.json` 齐全；manifest row hash、SHA256、TE adapter identity 和 conditioning shape 均通过 `verify_cache`。

Stage 2 training：

```bash
accelerate launch --main_process_port 45684 --num_processes 8 --mixed_precision bf16 \
  -m samtok_edit21.train train --stage stage2 --cache .../cache \
  --output .../stage2 --steps 1 --accumulation 4 --save-steps 32 --num-workers 0
```

验收项与 Stage 1 相同，另检查只加载 DiT、cache 条件不再重复运行 TE/VAE。

本次成功产物：

- Stage 1：`.../stage1_run3/`，`loss.csv` 8 条有限 loss，梯度审计报告 504 个可训练梯度张量、252 个非零张量、冻结梯度 0；adapter 约 698 MB。
- Cache：`.../cache_run1/`，`manifest.json` 含 8 行，`0/` 到 `7/` 每个 rank 各有一个 `.pth` 和 sidecar；`verify_cache` 返回 `True`。
- Stage 2：`.../stage2_run2/`，4 条有限 loss，梯度审计报告 464 个可训练梯度张量、232 个非零张量、冻结梯度 0；`adapter.json` 已包含 cache 的 `conditioning_identity`。

在发现 runner sampler 属性名问题后，最终验收使用了新产物：`stage1_final/`、`cache_final/`、`stage2_final/`。最终 Stage 1/Stage 2 schedule report 分别为全局 3:2:2:1 和 1:2:1，且 runner 已实际读取 `schedule_sampler`；最终 Stage 2 loss 为 0.2754、0.0129、0.0904、0.0940，均有限。

Stage 2 第一次重跑使用端口 45688 时遇到 `EADDRINUSE`，换用 45689 后成功。第一次 Stage 2 adapter 没有 conditioning identity，已在 `train.py:run_train` 修复并以 `stage2_run2` 重跑。

## 5. 推理 smoke

使用训练得到的 Stage 1 adapter 运行 `localize`，检查输出 JSON 能被 `parse_generated_cot` 接受；再使用 Stage 1/Stage 2 adapter 运行 `infer`，检查 PNG 可读、尺寸为目标尺寸且无 NaN。推理产物放在 `.../inference/`，不写入源数据目录。

实际运行：

```bash
python -m samtok_edit21.cli localize --image .../refedit_00_source.png \
  --prompt 'Change the selected object in the image.' \
  --te-adapter .../stage1_final/adapter --output .../inference/localize.json \
  --device cuda:0 --height 256 --width 256 --max-new-tokens 64
```

输出包含一个严格 JSON mask item 和可用的 inline conditioning prompt。使用 `stage2_final/adapter` 运行 inline inference（2 steps、256×256）成功，输出 `.../inference/final.png`，PIL 校验为 `RGBA (256, 256)`。第一次 inference 暴露了 CLI 对嵌套 `te_adapter_identity` 的解析错误；`cli.py` 现已同时支持嵌套 identity 和旧式路径字段。

## 6. 结果结论与限制

数据协议、8 卡 device 绑定、采样比例、Stage 1 NTP/FM 梯度链路、Stage 2 cache 完整性和推理入口都纳入 smoke 验收。smoke 数据只有 8 行，不能评价收敛、lambda 最优值或最终编辑质量；正式训练前仍需使用过滤和人工审核后的完整数据，并在新数据上重新运行协议和 cache 审计。

## 后续实验

后续每次运行追加日期、git commit、命令、输出路径、指标、异常和修复，不覆盖本页历史记录。

## 7. 2026-09-26：独立审计、上游归属确认与修复验收

### 7.1 版本、范围与证据位置

起点：分支 `qwen-image-2.1-dev`，HEAD `1859744d618611f8e203edbb7b0d7be42500a0b7`。本节实现改动尚未 git commit，属于该 HEAD 上的工作区修复；没有提交、覆盖源数据或改写历史实验产物。用户原有 README 的规划链接和未跟踪的区域约束规划文档保留；README 仅额外修正环境/localize 示例。

三批证据均在临时目录，**系统清理 /tmp 后可能消失**；本文保留必要的结果和复现命令，不将临时产物当长期模型发布：

- 修改前独立审计：`/tmp/samtok21-audit-tKUbmC/`，REPORT.md、RUNS.md、audit_gpu/extra/codec_audit/final_checks.json、单/双卡完整日志。
- 上游归属与方案：`/tmp/samtok21-upstream-xq4pvu/`，REPAIR_PLAN.md、官方固定版本文件、GitHub commit JSON、attention 数值对照。
- 本轮实现与复验：`/tmp/samtok21-fixes-dUnbt5/`，test_regressions.py、attention_regression.py、attention_geometry.py、runner_rng_probe.py、integration_checks.py、final_artifact_checks.py，以及下述日志/产物。

本轮独立执行了完整发布模型的小批量单卡/双卡训练与推理；控制流程反例、LR/RNG 探针和 attention 语义对照另用明确标注的小模型。未重跑本页历史 8 卡训练，没有做长训练、收敛分析或正式质量基准。

实际可用环境：Python 3.11.2、torch 2.8.0+cu128、torchvision 0.23.0、transformers 5.12.1、accelerate 1.14.0、peft 0.20.0，H100 80GB，nvidia-smi 报驱动 535.161.08。旧文档 venv 路径已不存在；系统 Python 的 torch 2.14/cu130、transformers 4.48 不能直接承担本项目。本轮先恢复隔离审计环境，随后在 `/tmp/samtok21-fixes-dUnbt5/venv` **从零按修订 requirements 安装**，用该环境通过单测、Stage 2、推理和 codec。安装日志 `install.log` 显示解析 92 packages；完整版本约束已纳入 repo 的 constraints-tested.txt。

### 7.2 修改前真正验证过的正确部分

修改前默认 recipe 可以运行，但不能因此认定没有 bug。审计证据：

| 检查 | 修改前实测 |
|---|---|
| 原有单测 | 23 passed |
| 完整 Stage 1 | 单卡 1 update、8 microsteps；双卡 2 updates、每 rank 16 microsteps |
| 完整 Stage 2 | 单/双卡各 2 updates、每 rank 8 microsteps |
| default adapters | Stage 1：174,587,904 trainable params、504 tensors；Stage 2：85,995,520 params、464 tensors；fp32、有限，LoRA B 均有非零更新 |
| NTP 独立梯度 | raw CE 1.6321014，TE LoRA grad norm 1.48124；冻结梯度 0 |
| FM 独立梯度 | loss 0.2755177，TE LoRA grad norm 0.151525；冻结梯度 0 |
| norm / NTP 对照 | 手动 norm 与 post-norm 输出差 0；HF 原生 labels CE 1.6322559，约 1.5e-4 bf16 数值差；hook 数量回到 0 |
| 官方 FM 对照 | 同 RNG 的自定义/官方 loss 都为 0.1936758161，差 0 |
| 条件/cache | 单卡缓存与重编码、双卡 uneven cache 对齐后 embeddings/source/target 最大差均 0 |
| 模板与几何 | system prefix 实测 14 tokens；256² source 为 16×16 latent，64 pads×4；双图 320×224/224×320 为 14×20/20×14，140 pads×4=560 |
| codec | 单/批编码解码一致，RefEdit:0 codes [17,322]，IoU=0.96685217；SAM2 box 最佳 IoU=0.9785588 |
| 在线推理 | 新 adapter reload 后生成合法 JSON，并输出 2-step 256² RGBA PNG |

旧 bf16 KV/uncached velocity max diff=0.03125、mean=0.0050964，不宣称 bitwise 相同或完整多步质量等价。codec 的 raw logits > 0.5 来自发布实现；改为 >0 在单样本改变 1349 pixels、IoU=0.96409，没有证据应在迁移中擅自改变阈值。

### 7.3 问题归属、规划与实际处理

上游通过 GitHub API 与 ls-remote 核实：旧固定版本 `d2d684ad1f912949eae08453b9411ae40c5ec0ab`；当次核查官方 main 为 `7686e54d41d25c0e8ed5f1318acc23b6bb832654`，2026-09-21，PR #1697。后者父提交就是前者，且只改 DiT。官方 runner 在两版本中相同。

| ID | 复现的问题 / 归属 | 本轮处理及兼容性 |
|---|---|---|
| F1 | 项目 warm-start 实际 rank=2/dropout=.15，却保存默认 rank=32/dropout=0，reload size mismatch | 实际 PEFT 配置导出、显式参数冲突检查、key/shape/finite/recipe 指纹；双阶段非默认 warm-start/reload 复验 |
| F2 | 项目 verifier 只查 checksum/部分 shape，base-A shard 配 base-B manifest 仍通过；非法 sample_type 也通过 | cache-v2 完整内容身份、真实 row index、payload/sidecar/manifest 一致性、唯一完整行覆盖、协议/逐图几何检查；训练/推理前核对实际基座和 adapter |
| F4 | 项目两个 train/cache CLI 各走一套实现，v1 名同格式不同，旧 dict identity 被当路径而 TypeError | CLI 统一委托 train.py；删除旧循环；明确 v1 只读审计，缺完整来源需重建，不伪造迁移证明 |
| F6 | 旧官方 fallback 将 source block 当纯 causal，且绕过 checkpoint | 同步最新官方 #1697 原文件，不替换本地 runner 扩展；state_dict schema 不变 |
| F5 | 项目 converter 去冠词使 the cat/a cat 变同名 cat，完美预测 GT 仍无法回填 | 保留唯一精确 phrase，显式多实例分组；NTP-only 也执行 GT→绑定→ref 的 round-trip |
| F3 | 项目 online/oracle 始终 ref，没实现 proposal 默认 noref | 共享语义绑定器，默认请求 noref，审核 units/保守语法子集，actual variant 与 fallback reason；strict 模式拒绝 unsupported |
| A1 | 官方仍用默认 ConstantLR 的 1/3 初始 LR；项目没有选定明确 recipe | 项目显式 constant=1；可选 warmup/cosine 按实际 optimizer update 推进并记 LR；不改其他官方调用的默认行为 |
| A2 | 项目 seed 只用于 schedule | 同 seed 初始化、DDP 后 seed+rank 全 RNG；固定设备数重复运行验证 |
| A3 | 项目审计允许缺失/全零梯度，旧 accumulation 可掩盖断链 | 每次 backward 的梯度 hook 单独审计；全零/无梯度/非有限/冻结有梯度报错 |
| A4 | 项目 masked inline/direct 可多图，与训练协议冲突 | 公共 edit 与 CLI 单图约束；direct/stock 明确拒绝 mask；plain 多图保留 |
| A5/A6 | 项目环境路径失效、缺直接依赖、示例漏 --prompt | constraints-tested + 补依赖，干净 venv 验证，修正文档与 README |
| A7 | 项目没有 proposal 的 benchmark 白底/恢复尺寸 wrapper | 显式 --benchmark-output，保留 raw RGBA、输出 RGB 参考尺寸，多图需显式参考索引 |

F1/F2/F3/F4/F5 和大部分 A 项是项目实现/契约问题，不是升级官方库就会消失的问题。F6 是旧上游缺陷，最新官方已经修复；A1 是上游仍存在的默认行为，不是 PyTorch scheduler 违反定义。没有加入新的空间 loss、attention bias 或其他规划中的科研扩展。

### 7.4 本轮修复后验收

| 验收 | 实际结果 / 产物 |
|---|---|
| 基础单测 + 临时回归 | **61 passed**，`final_acceptance_pytest.log`；包含原 23 项（1 个旧 cache 测试夹具升级为 v2）和 38 个新增反例/契约用例 |
| 干净依赖环境 | 从 requirements 安装完成；在新 venv 重跑测试与真实模型任务通过 |
| 单卡 Stage 1 | `stage1/`，rank=2/dropout=.15，8 microsteps、1 update，第一步实际 LR=4e-5 |
| 双卡 Stage 1 warm-start | `stage1_warm_ddp/`，不重传 rank/dropout，自动继承 2/.15；每 rank 8 microsteps；新梯度 hook 每次有非零观测；reload localize 成功 |
| Stage 1 正式绑定契约的小样本 | `reviewed_stage1/`：真实 RefEdit:0 source/target/codes，converter 生成 4 种 row，ref/noref 真正不同；8 microsteps、1 update 通过，保存 recipe 指纹 |
| 单/双卡 cache-v2 | `cache_single/`、`cache_ddp/`，同 3 行 FM metadata；双卡 2+1 不整除分片，完整 row index 0/1/2；按行对齐 tensors 逐元素相同 |
| 单/双卡 Stage 2 | `stage2/`、`stage2_ddp/`，各 2 updates、每 rank 8 microsteps，rank=2/dropout=.15，cosine/warmup=1；有限 loss、464 梯度张量、冻结梯度 0 |
| 双卡 Stage 2 warm-start | `stage2_warm_ddp/`，继承非默认配置且核对相同 conditioning identity，1 update、每 rank 4 microsteps，通过 |
| adapter 完整性 | 七套产物（含最终追加的 stage2_final）均 rank=2/dropout=.15、fp32/有限；Stage 1 有 504 tensors/252 非零 LoRA B；Stage 2 有 464/232；`final_artifact_checks.json` |
| 重编码一致性 | 新缓存与实际 adapter 在线重新编码，embeddings/source/target 最大差 0；`integration_checks.json` |
| no-Flex 完整模型 | 强制 FALLBACK，完整 DiT FM loss=0.0525325127，464 gradient tensors，norm=0.0208680491；checkpoint backward 通过 |
| 严格 oracle noref | no-Flex 完整模型 2-step 推理：`Make this region <M> blue.`，actual=noref、无 fallback；`oracle_fallback.png` |
| online + benchmark | `online.raw.png` 为 256×256 RGBA；`online.png` 为源图原尺寸 1024×1024 RGB；白底全透明像素测试也通过 |
| online 语义边界 | 模型生成 label=`leftmost bird blue.`，所以 requested=noref、actual=ref，并记录保守语法不支持原因；**不计为 noref 成功** |
| interactive + codec | `interactive.png`，真实 mask 经 codec 得 [17,322]，生成 inline prompt 并完成 2-step 256² RGBA 推理 |
| RNG / LR 分布式探针 | `rng_ddp_b/c` 两次双卡小模型运行，初始化、每 rank RNG draws、最终参数逐项一致；rank0/1 随机流不同、更新后参数同步；单/双卡实际 LR 都为 [0.005,0.01,0.01,0.005] |

中间最早 `stage1/`、`stage1_warm_ddp/` 产生于追加 recipe fingerprint 校验之前，rank/dropout 已正确但没有该字段；后续 `reviewed_stage1/` 和 Stage 2 产物已含 fingerprint。这些均为修复过程中的 smoke 产物，不作为正式训练起点发布。

官方 patch 的额外数值验证（fp32、2-layer 小 DiT）：

| 项目 | 最大绝对差 |
|---|---:|
| 旧 fallback vs Flex，无 padding / 有 padding | 0.0242956 / 0.0391016 |
| 新 fallback vs Flex，无 padding / 有 padding | 4.768e-7 / 2.384e-7 |
| 新 Flex vs 旧 Flex | 0 |
| 新 fallback cached vs uncached | ≤2.384e-7 |
| fallback checkpoint 开/关 input/parameter gradients | 0 |
| 双图非方形 fallback vs Flex | 2.980e-7 |
| 双图非方形 cached vs uncached | 2.384e-7 |
| prefix 不看变化的 target / 首 text 不看未来 text / target 忽略 padding | 均为 0 |

另验证同 source block 的第一个 token 会受到最后一个 token 变化影响（max delta=0.00666262），排除了旧纯三角可见性。新本地 DiT 与官方下载文件 SHA256 同为 `54639031f91e74d3f418e35572d2d98c6dad9ab8ce2ca54ffdf5146473744e0a`。证据：latest_attention_check.json、attention_geometry.json。

### 7.5 实际命令与重新运行方式

以下均在 repo 根目录执行。重跑时另选空输出目录；不要覆盖本节产物。

```bash
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$PWD:DiffSynth-Studio
export CHECK=/tmp/samtok21-fixes-dUnbt5
export PY=$CHECK/venv/bin/python
export DATA=/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/qwen_image_2_1_dev_smoke/refedit_data
export TORCHINDUCTOR_CACHE_DIR=$CHECK/inductor

$PY -m pytest -p no:cacheprovider -q tests $CHECK/test_regressions.py

CUDA_VISIBLE_DEVICES=0 $PY -m samtok_edit21.cli train --stage stage1 \
  --metadata $DATA/stage1.jsonl --base-path $DATA --output $CHECK/stage1 \
  --max-pixels 65536 --steps 1 --rank 2 --dropout 0.15 --seed 926

CUDA_VISIBLE_DEVICES=0,1 $PY -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m samtok_edit21.train train --stage stage1 --metadata $DATA/stage1.jsonl \
  --base-path $DATA --output $CHECK/stage1_warm_ddp --max-pixels 65536 \
  --steps 1 --init-adapter $CHECK/stage1/adapter --seed 926

CUDA_VISIBLE_DEVICES=2,3 $PY -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m samtok_edit21.cli cache --metadata /tmp/samtok21-audit-tKUbmC/cache_three.jsonl \
  --base-path $DATA --te-adapter $CHECK/stage1/adapter \
  --output $CHECK/cache_ddp --max-pixels 65536

CUDA_VISIBLE_DEVICES=5 $PY -m samtok_edit21.train cache \
  --metadata /tmp/samtok21-audit-tKUbmC/cache_three.jsonl --base-path $DATA \
  --te-adapter $CHECK/stage1/adapter --output $CHECK/cache_single --max-pixels 65536

CUDA_VISIBLE_DEVICES=4 $PY -m samtok_edit21.cli train --stage stage2 \
  --cache $CHECK/cache_ddp --output $CHECK/stage2 --steps 2 \
  --rank 2 --dropout 0.15 --lr-schedule cosine --warmup-steps 1 --seed 926

CUDA_VISIBLE_DEVICES=2,3 $PY -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m samtok_edit21.train train --stage stage2 --cache $CHECK/cache_ddp \
  --output $CHECK/stage2_ddp --steps 2 --rank 2 --dropout 0.15 \
  --lr-schedule cosine --warmup-steps 1 --seed 926

CUDA_VISIBLE_DEVICES=5,6 $PY -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m samtok_edit21.train train --stage stage2 --cache $CHECK/cache_ddp \
  --init-adapter $CHECK/stage2/adapter --output $CHECK/stage2_warm_ddp --steps 1 --seed 926

$PY $CHECK/prepare_reviewed.py
$PY -m samtok_edit21.cli validate --metadata $CHECK/reviewed_stage1.jsonl \
  --base-path $DATA --check-bindings
CUDA_VISIBLE_DEVICES=3 $PY -m samtok_edit21.train train --stage stage1 \
  --metadata $CHECK/reviewed_stage1.jsonl --base-path $DATA --output $CHECK/reviewed_stage1 \
  --max-pixels 65536 --steps 1 --rank 2 --dropout 0.15 --seed 926

CUDA_VISIBLE_DEVICES=0 $PY -m samtok_edit21.cli infer \
  --image $DATA/images/refedit_00_source.png --prompt 'Make the leftmost bird blue.' \
  --te-adapter $CHECK/stage1/adapter --dit-adapter $CHECK/stage2/adapter \
  --height 256 --width 256 --steps 2 --max-new-tokens 128 \
  --benchmark-output --output $CHECK/online.png

CUDA_VISIBLE_DEVICES=1 $PY -m samtok_edit21.cli localize \
  --image $DATA/images/refedit_00_source.png --prompt 'Make the leftmost bird blue.' \
  --te-adapter $CHECK/stage1_warm_ddp/adapter --height 256 --width 256 \
  --max-new-tokens 128 --output $CHECK/localize.json

CUDA_VISIBLE_DEVICES=2 $PY -m samtok_edit21.cli infer --mode interactive \
  --image $DATA/images/refedit_00_source.png --mask /tmp/samtok21-audit-tKUbmC/codec_gt.png \
  --prompt 'Make this region blue.' --te-adapter $CHECK/stage1/adapter \
  --dit-adapter $CHECK/stage2/adapter --height 256 --width 256 --steps 2 \
  --output $CHECK/interactive.png

CUDA_VISIBLE_DEVICES=4 $PY $CHECK/integration_checks.py
CUDA_VISIBLE_DEVICES=6 $PY $CHECK/attention_regression.py
CUDA_VISIBLE_DEVICES=7 $PY $CHECK/attention_geometry.py
CUDA_VISIBLE_DEVICES=6,7 $PY -m torch.distributed.run --standalone --nproc_per_node=2 \
  $CHECK/runner_rng_probe.py $CHECK/rng_ddp_b
# 同命令用新目录 rng_ddp_c 再跑一次；单卡 --nproc_per_node=1 输出 rng_single_gpu。
$PY $CHECK/final_artifact_checks.py
```

若临时脚本已清理，repo 自带 tests 仍可运行；上述新增脚本没有加入 repo，以遵守调试文件放临时目录的要求。正式长跑应先使用审核数据、生成全新 v2 cache，而不是复用本节 smoke 产物。

### 7.6 本轮失败尝试与未覆盖项

- 升级 cache-v2 后，初次 pytest 为 22 pass / 1 fail：旧测试用没有 identity 的 v1 shard 期待通过。该期待正是需要修复的旧行为，故修改测试夹具为真实 v2 envelope；不是放宽验证来迎合旧测试。
- 临时 RNG 探针最初把 Parameter 直接挂在根 module；官方 DiffusionTrainingModule.to 只移动子模块，导致双卡 CPU/GPU mismatch。改为探针显式调用 nn.Module.to 后重跑；项目实际 pipe 是子模块，不受此夹具问题影响。失败日志 rng_ddp_a.log 保留。
- 早期完成任务时出现未显式 destroy_process_group 的 NCCL warning；已在项目 run_train/run_cache 结束调用 accelerator.end_training，后续 Stage 2 warm-start 验证正常。
- 旧 8 行 smoke 的 NTP 第 3 行（0-based row=2）label 为 `white candle in the middle candlestick`，不出现在原指令中；只读扫描报告 0 matches。未改源数据。另建真实 RefEdit:0 的审核样本，noref 为 `Change this region <M> to soft down feathers`，通过 round-trip 和真实训练。
- 新 cache-v2 是有意的兼容性收紧：缺完整历史身份的 v1 cache/Stage 2 adapter 不再静默接受。已有 adapter rank 错误不自动猜测修复；已提供按原配置查证、另存新目录的操作原则。
- noref 已实现受限、可审核的正确路径，不宣称任意自然语言/任意 composite 已能自动解析。若要普遍自动化，需决定 pass-1 结构化协议升级或单独语义解析方案；这属于方法设计变更。
- 有限 loss、非零梯度、PNG 可读不是质量证据。没有检验正式全量数据、完整 40-step 评测、长时间稳定性或收敛，更不能证明 mask token 已形成精确区域控制。

### 7.7 最终复查补充

最终源码复查又收紧了两处实现，并对本轮新增实现做了扩展性修正：

- cache identity 最初包含整份 row_hashes 列表；重复写入每 shard 会产生 O(N²) 元数据。这是本轮实现过程发现的问题，已改为常量大小的 `rows_sha256`；manifest 保留有序 rows，逐行 hash/index 与该摘要联合验证。早期 v2 smoke 格式仍可只读验证，新写入只用紧凑摘要。
- source 几何检查从“所有 image-pad 总数”进一步升级为“每个连续 image-pad block 与对应 source latent 网格一一匹配”，并校验 source 数量，拒绝总数相同但分配到各图错误的情况。
- 自动 noref 不再把带 and/then/while/also 的复合句当单个 replace；remove 不再任意删除整个 from ... 尾部，以免吞掉 carefully 等 how 信息。复杂 where 必须在审核的 ref_phrase 中明确标注。
- compact identity/语义保护两项新增回归先使测试达到 58；最后增加 3 项 latent/embedding 浮点 dtype 拒绝测试，最终 **61 passed**（`final_acceptance_pytest.log`）。

用最后版本重新生成 `cache_final/`：来源为 `reviewed_stage1/adapter` + `reviewed_stage2.jsonl`，双卡 3 行 uneven cache，compact identity 验证通过。再从该缓存完成 `stage2_final/` 单卡 1 update（4 microsteps），rank=2/dropout=.15；这是完整审核绑定数据→最终 cache 契约→DiT 训练的追加验收，不是覆写旧产物。

对应追加命令：

```bash
CUDA_VISIBLE_DEVICES=0,1 $PY -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m samtok_edit21.train cache --metadata $CHECK/reviewed_stage2.jsonl --base-path $DATA \
  --te-adapter $CHECK/reviewed_stage1/adapter --output $CHECK/cache_final --max-pixels 65536
CUDA_VISIBLE_DEVICES=4 $PY -m samtok_edit21.train train --stage stage2 \
  --cache $CHECK/cache_final --output $CHECK/stage2_final --steps 1 --rank 2 --dropout 0.15 --seed 926
```

最终 strict oracle CLI 验收也完成：加载 `reviewed_stage1/adapter` 与 `stage2_final/adapter`，自动核对 compact conditioning identity，读取审核 units，2-step 输出 `oracle_final.png`（RGBA 256×256）。报告 requested=noref、actual=noref、fallback_reason=null，实际指令为 `Change this region <M> to soft down feathers`。命令：

```bash
CUDA_VISIBLE_DEVICES=4 $PY -m samtok_edit21.cli infer --mode oracle \
  --image $DATA/images/refedit_00_source.png \
  --prompt "Change the leftmost bird's feathers to soft down feathers" \
  --cot-file $CHECK/reviewed_cot.txt --units-file $CHECK/reviewed_units.json --strict-noref \
  --te-adapter $CHECK/reviewed_stage1/adapter --dit-adapter $CHECK/stage2_final/adapter \
  --height 256 --width 256 --steps 2 --output $CHECK/oracle_final.png
```

真实 TE/VAE 双图非方形路径在最终逐图校验下再次通过：320×224 / 224×320，source latents 为 [1,64,14,20] / [1,64,20,14]，140 pads，总计 560 latent tokens。证据 `multi_geometry_final.json`。最终产物汇总脚本再次校验七套 adapter、三份缓存及所有推理 PNG，输出 `final_artifact_checks.json` / `final_artifact_checks_v2.log`；源码 `git diff --check` 无错误。

文档交付前还对 README、实现文档、实验记录中的 32 条具体 CLI 命令做了 argparse 解析检查，全部通过（略过仅用于说明入口等价性的 `...` 伪命令）。证据 `check_doc_cli.py / check_doc_cli.log`。最终训练、缓存、推理、产物汇总进程均已确认 exit code=0；代码未 git commit。
