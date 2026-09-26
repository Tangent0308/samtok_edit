# SAMTokEdit：Qwen-Image-2.1 实现与使用

本文描述当前代码，不把历史“接口跑通”当作实现无误的证明。2026-09-26 完成独立检查并按 F4→F1→F2→F6/复现性→F5/F3 的顺序实施修复；实际测试、失败尝试和数值结果见 [实验记录](SAMTokEdit_Qwen21_实验记录.md)。以 research proposal 最后的 **Qwen-Image-2.1 两次前向方案**为准，不沿用早期 2511 assistant-CoT 条件化版本。

## 1. 官方代码与项目扩展的边界

| 部分 | 来源 / 项目改造 |
|---|---|
| DiT、VAE、flow scheduler、基础 pipeline | 仓库内 DiffSynth，当前固定到官方 `7686e54d41d25c0e8ed5f1318acc23b6bb832654`（2.1.8） |
| DiT attention | 完整同步官方 [#1697 提交](https://github.com/modelscope/DiffSynth-Studio/commit/7686e54d41d25c0e8ed5f1318acc23b6bb832654)，本项目没有另写 attention 数学实现 |
| 训练/cache 生命周期 | 官方 `diffsynth/diffusion/runner.py`；保留项目的精确 sampler、梯度审计/裁剪、data_process 默认值修复；新增可选 scheduler factory 与 rank seed |
| SAMTok TE 适配 | `samtok_edit21/model.py`：原生 Qwen3-VL 包装、processor/tokenizer、两种模板、pre/post norm 分流、在线两步推理 |
| 两阶段训练入口 | `samtok_edit21/train.py`：模型 forward、采样、实际配置、分布式身份校验；两个 CLI 入口均委托这里 |
| 共享训练工具 | `training.py`：add/load/save LoRA、几何检查、prepare_fm、flow_loss；旧的独立训练/cache 循环已移除 |
| 数据与绑定 | `protocol.py / prepare.py / data.py`：严格 v2 行协议、标签 round-trip、ref/noref 改写、采样 |
| 产物来源校验 | 新 `provenance.py`：内容指纹、cache-v2 验证、历史 identity 解析与只读审计 |
| mask codec | `codec.py` + `samtok/` 的发布 VQ-SAM2；没有在 DiT 内解码空间 mask |

上一个固定版本是 `d2d684ad1f912949eae08453b9411ae40c5ec0ab`；与新版本之间官方只修改了 DiT 文件。本地旧 DiT 与旧官方文件字节相同，所以旧 fallback 的纯 causal 图像前缀问题来自上游，**最新官方已修复**。不能用上游 runner 覆盖本地 runner，否则会丢失项目扩展。版本及扩展记录在 `upstream_versions.json`。

## 2. 方法如何落到代码

1. Pass 1：无 system 的 SAMTok 原生 chat template，源图 + 编辑指令 + `Please identify and segment the region to be edited in this image.`，输出严格 fenced JSON list，每项仅有 `mask_2d/label`。
2. Pass 2：使用官方 2.1 的 `Comprehend and analyze the provided prompt.` system 与 image template，将 mask span 内联到 user 指令；JSON 本身不送给 DiT。
3. FM 读取末层 RMSNorm **之前**的 4096 维特征；NTP 读取 RMSNorm **之后**的特征再经过冻结 lm_head。scoped pre-hook 在前向结束后移除。
4. NTP 从 `prefix-1` 位置预测第一个 assistant token，监督包含末尾 im_end，前缀不参与 CE。FM 使用官方 timestep/noise-clean target/fp32 MSE/scheduler weighting，不通过离散定位采样反传。
5. target/source VAE latent 为 `[1,64,H/16,W/16]`；一个 TE image-pad placeholder 展开 4 个 DiT latent tokens。两侧必须使用同一套 resize 结果，不能独立缩图。

SAMTok span：`<|mt_start|><|mt_0000|><|mt_0256|><|mt_end|>`，两层有序码本分别取 [0,255]、[256,511]，共 4 个原子 tokens；额外特殊 tokens 总数 514。

## 3. 数据协议与标签契约

| sample_type | 必需含义 | 禁止项 |
|---|---|---|
| `edit_ntp` | source、原始 prompt、canonical mt_cot | target image、inline mask、instr_variant |
| `edit_umt` | source/target、含 mask 的 prompt、ref/noref instr_variant | mt_cot |
| `edit` | source/target、普通 prompt | mask、mt_cot、instr_variant |

mask 行只能有一个 source；普通 edit 支持多图。截断 span、错码本、控制字符、歧义/重叠引用均拒绝，不默认取第一处匹配。

`convert_record` 接收人工审核的 `units=[{ref_phrase, edit_type, mask_codes, anchor_phrase?}]`。现在保留精确引用，不无条件去掉 the/a/an：`the cat` 与 `a cat` 不能都变成 `cat`。多实例必须用显式 `one of ...` label 语义分组。转换总会执行：

`GT JSON → parse → grouped_units → ref render`

核对 unit 数量、代码顺序和引用位置；NTP-only 也必须通过。不同 atomic unit 不可因为 label 相同而静默合并。无法唯一绑定的数据应修订为新数据集或拒绝，不覆盖源数据。

```bash
python -m samtok_edit21.cli convert --input /path/reviewed_records.jsonl --output /path/new_rows.jsonl
python -m samtok_edit21.cli validate --metadata /path/new_rows.jsonl --base-path /path/data --check-bindings
```

`--check-bindings` 只读扫描全部 NTP 并列出失败行。默认 validate 保留历史接口协议检查，因此历史 smoke 行可能通过基础检查但不通过绑定检查；正式数据必须增加此项。既有 `tests/eight_gpu_smoke/prepare_refedit.py` 仍是 smoke-only：其 ref/noref 不是可靠的正式 noref 对照，不能据此声称方法有效。

## 4. 训练、保存与可复现性

| 配置 | Stage 1 | Stage 2 |
|---|---|---|
| 可训练部分 | Qwen3 language_model attention/MLP LoRA | DiT 全部 nn.Linear LoRA |
| 默认 rank / alpha / dropout | 64 / 64 / 0.05 | 32 / 32 / 0 |
| 默认 LR / weight decay | 4e-5 / 0.05 | 1e-4 / 0.01 |
| 默认 accumulation | 8 | 4 |
| 每步采样比例 | NTP:ref:noref:plain = 3:2:2:1 | ref:noref:plain = 1:2:1 |
| loss | NTP × 0.05；FM × 1.0，各样本独立分支 | FM × 1.0 |

LoRA 参数与优化器更新为 fp32，基座冻结。Stage 1 的 VAE 在 no_grad；FM 条件保留 TE 梯度，冻结 DiT 仍允许反传到 TE。Stage 2 只加载 DiT；TE/VAE 不重复运行。cache 任务始终 eval，包括显式传入 `--stage stage1` 的情况。

`--init-adapter` 仅 warm-start 权重，**不是完整 resume**：optimizer、scheduler、采样进度重新开始。未显式传 rank/dropout 时继承 adapter；显式冲突在训练前报错。保存从实际 PEFT 导出 rank/alpha/dropout/target_modules，写 schema_version=2、recipe_sha256、base_identity，并验证 tensor key/shape/有限值。Stage 2 另存 conditioning_identity；不再依赖可能过期的 CLI 默认值。已有 rank 写错的 adapter 不自动修复：rank 可从 A/B shape 查证，dropout 不能从 tensor 推断，必须结合原配置另存修复目录。

两个命令等价：
```bash
python -m samtok_edit21.train train ...
python -m samtok_edit21.cli train ...
```
cache 也相同；`--save-every` 是 `--save-steps` 的别名。checkpoint 间隔沿用官方 **microstep** 计数；`--steps` 是 **optimizer update** 数，两者不能混用。

### LR 与随机数

项目默认显式选择 constant=1，即第一步就是传入 LR。官方未扩展调用仍保留其原始 ConstantLR 默认行为（factor=1/3），不改变其他项目 recipe。

支持 `--lr-schedule constant|cosine --warmup-steps N`。调度器只在实际、未跳过的 optimizer update 后推进一次，不随 world size 额外推进。warmup 的第 k 次更新使用 LR × k/N（k 从 1 起）；cosine 在 warmup 后下降。记录实际用于更新的 LR 于 `optimizer_steps.jsonl`，其中 loss 字段明确是最后一个 microstep，而非全局平均。

adapter 初始化前所有 rank 使用共同 seed；DDP prepare 后使用 seed+rank 控制 Python/NumPy/torch/CUDA 随机流；schedule 用独立共享 seed。固定设备数、版本和 seed 的短运行可复验，不保证跨硬件 bitwise deterministic。

每次 backward 的参数 hook 单独观察当前分支梯度，避免旧 accumulation 梯度掩盖断链；审计拒绝当前分支全零/无梯度、非有限或冻结参数有梯度。**不要求每个张量每步非零**，初始 LoRA A 零梯度正常。裁剪只发生在同步更新前。

## 5. 训练与推理超参数：当前项目和官方实现对照

本节只记录代码中实际生效的默认值、官方公开配置和适用范围。项目当前基座是 Qwen-Image-2.1 + `zhouyik/Qwen3-VL-8B-SAMTok`；表中“当前项目”指**未通过 CLI 覆盖、且未用 `--init-adapter` 继承已有 adapter 配方**时的默认值。下文训练 `--steps` 指 optimizer update，推理 `--steps` 指去噪步数，二者不相同；实验记录中的 rank 2、256²、2-step 推理是缩减的 smoke 配置，不是本节默认值。

### 5.1 DiffSynth Qwen-Image-2.1 图像编辑训练与项目 Stage 2

对照基准为仓库固定的 DiffSynth `7686e54d`：[官方 2.1 LoRA 脚本](https://github.com/modelscope/DiffSynth-Studio/blob/7686e54d41d25c0e8ed5f1318acc23b6bb832654/examples/qwen_image_21/model_training/lora/Qwen-Image-2.1.sh)、[训练入口](DiffSynth-Studio/examples/qwen_image_21/model_training/train.py)、[公共参数](DiffSynth-Studio/diffsynth/diffusion/parsers.py)、[LoRA 注入](DiffSynth-Studio/diffsynth/diffusion/training_module.py)、[FM loss](DiffSynth-Studio/diffsynth/diffusion/loss.py)。脚本前半段实际执行的是文生图示例（`dataset_repeat=50`）；其 `# Edit` 下的图像编辑命令**全部被注释**。下表的“官方编辑示例”只抄录该注释命令及其未覆盖的公共默认值，不代表官方另行发布了 SAMTok/TE 两阶段编辑配方。

| 参数 | DiffSynth 2.1 官方 `# Edit` 注释示例 / 公共默认 | 当前项目 Stage 2 |
|---|---|---|
| 训练对象 | Qwen-Image-2.1 DiT LoRA；`lora_base_model=dit` | 同一基座的 DiT LoRA；TE/VAE 冻结，使用 Stage 1 adapter 产生的条件 cache |
| 数据入口 | `data_file_keys=image,edit_image`，`extra_inputs=edit_image`；示例数据为 Qwen-Image-Edit-2511 | `edit_umt:ref`、`edit_umt:noref`、`edit` 的 cache-v2；不读取 NTP 行 |
| LoRA target / rank / alpha / dropout | `lora_target_modules=""` → 自动检测重复 `ModuleList` 内、`min(in_features,out_features)≥512` 的 Linear，**不等于全部 Linear**；32 / 32 / 0（alpha=rank 和 dropout=0 来自注入实现/PEFT 默认） | DiT 全部 `nn.Linear`；32 / 32 / 0 |
| 优化器 / LR / weight decay | AdamW / `1e-4` / `0.01`（weight decay 为公共默认） | AdamW / `1e-4` / `0.01`；AdamW 默认 `betas=(0.9,0.999)`、`eps=1e-8` |
| 单卡 microbatch / 累积 | 1 / 1（示例未传累积参数，公共默认 1）；全局 batch = GPU 数 | 1 / 4；全局 batch = `4 × GPU 数`，8 卡时为 32 |
| 训练长度 / 数据重复 | `num_epochs=5`、`dataset_repeat=100` | `num_epochs=1` 遍历生成的 schedule；`--steps` 未给时按各采样池容量计算 update 数，池耗尽可循环抽取；无固定 repeat=100 |
| 图像尺寸 | 动态尺寸，`max_pixels=1048576`，宽高按 32 对齐 | `max_pixels=1048576`，宽高按 32 对齐；Stage 2 实际采用 cache manifest 的尺寸上限 |
| 精度 | pipeline BF16；官方 LoRA 注入将可训练参数转换到 pipeline dtype（BF16） | 冻结基座 BF16；LoRA 可训练参数与优化器状态 FP32 |
| 样本混合 / loss | 无 SAMTok mask 或 NTP 分支；正条件 FM，训练 `cfg_scale=1` | `ref:noref:plain=1:2:1`；FM × 1.0 |
| FM 时间步 / 目标 | 1000 个训练时间步均匀抽样；`noise−clean` target，FP32 MSE 乘 scheduler weight | 相同的 1000 步抽样、target、MSE 和 scheduler weight |
| LR 调度 / warmup | runner 未扩展调用使用 PyTorch `ConstantLR` 默认配置，初始 factor=1/3，`total_iters=5`；示例未指定 warmup | 默认 `constant`、warmup 0，从首个 optimizer update 使用设定 LR；可选 `cosine` 与 `--warmup-steps` |
| 梯度检查点 / 裁剪 | 启用 DiT gradient checkpointing；公共 runner 不传 `max_grad_norm`，默认不裁剪 | 启用 DiT gradient checkpointing；同步 optimizer update 前裁剪 norm=1.0 |
| seed / 保存 | 训练 seed 未在官方命令中指定；`save_steps=None` 时按 epoch 保存 | seed=`20260920`；`--save-steps=100` 按 microstep 计数；最终另存带配方和身份信息的 adapter |

项目 Stage 1 的冻结 DiT FM 路径也使用同一 DiffSynth 2.1 scheduler/loss 定义，但其 NTP/FM 联合目标和 TE LoRA 并非上述官方编辑示例的一部分。当前项目参数解析、分支 loss、采样与 scheduler 分别见 [train.py](samtok_edit21/train.py)、[training.py](samtok_edit21/training.py)、[data.py](samtok_edit21/data.py)。

### 5.2 SAMTok Qwen3-VL 训练与项目 Stage 1

对照资料分三层，不能合并成一个“官方 8B LoRA recipe”：① [SAMTok 论文附录 B](https://arxiv.org/html/2601.16093)给出跨 Qwen-VL 模型的 VLM SFT 设置；② 官方代码中的 [Xtuner 4B 配置](samtok/configs/qwen3vl_4b_mt256x2.py) 和 [MS-Swift 4B 脚本](samtok/swift/sft_qwen3vl_4b.sh) 是两个具体的 **4B** LoRA 示例；③ [Qwen3-VL-8B-SAMTok 发布页](https://huggingface.co/zhouyik/Qwen3-VL-8B-SAMTok)提供权重和推理示例，未列出该 8B checkpoint 的完整训练超参数。论文中 tokenizer 训练的 LR `4e-5`、global batch 1024 是训练独立 **mask tokenizer**，不是 Qwen3-VL SFT 的参数。

| 参数 | SAMTok 论文 VLM SFT | SAMTok Xtuner 4B 示例 | SAMTok MS-Swift 4B 示例 | 当前项目 Stage 1（8B） |
|---|---|---|---|---|
| 训练对象 / 损失 | 冻结视觉编码器，微调投影层和 LLM；NTP SFT | Qwen3-VL-4B，视觉编码器冻结；LLM LoRA，NTP | Qwen3-VL-4B，冻结 ViT/aligner；`all-linear` LoRA，另保存 `embed_tokens/lm_head`；NTP | 冻结视觉编码器、投影层、embedding、lm_head 和基座；LM attention `q/k/v/o_proj` + MLP `gate/up/down_proj` LoRA；NTP 与经过冻结 DiT 回传至 TE 的 FM |
| LoRA rank / alpha / dropout | 论文未列 | 128 / 256 / 0.05 | 64 / 128 / 脚本未显式传 dropout | 64 / 64 / 0.05 |
| 优化器 / LR / weight decay | AdamW / `2e-5` / 未列 | AdamW / `2e-5` / `0.05`；`betas=(0.9,0.999)` | LR `2e-5`；其余未在脚本中显式传入 | AdamW / `4e-5` / `0.05`；`betas=(0.9,0.999)`、`eps=1e-8` |
| batch / 累积 | global batch 256 | per-device 4 / 累积 1；global batch 取决于实际 GPU 数 | 8 GPU × per-device 4 × 累积 2 = global batch 64 | per-device microbatch 1 / 累积 8；global batch = `8 × GPU 数`，8 卡时为 64 |
| 长度 / 图像输入 | 论文未列 VLM SFT epoch、最大文本长度 | 1 epoch；`model_max_length=8192` | 1 epoch；`max_length=8192`，`IMAGE_MAX_TOKEN_NUM=2048` | `--steps` 未给时由 schedule 推导，runner 遍历 1 次 schedule；无 CLI 文本长度上限；训练图像 `max_pixels=1048576` |
| 精度 | 论文未列 | BF16 AMP | `torch_dtype=bfloat16` | 冻结基座 BF16；LoRA/优化器状态 FP32 |
| 调度 / warmup | cosine；warmup 比例未列 | 前 5% warmup，随后 cosine | `warmup_ratio=0.05`；脚本未显式列调度器类型 | 默认 constant、warmup 0；可选 cosine 和指定 warmup update 数 |
| gradient checkpoint / clip | 论文未列 | 配置中 gradient clip norm=1 | 启用 gradient checkpoint；脚本未显式列 clip | LM/DiT gradient checkpoint；同步 update 前 clip norm=1.0 |
| seed / data workers / 保存 | 论文未列 | seed 未列 / 4 / 每 1000 step | seed 未列 / 4 / 每 1000 step | `20260920` / 0 / 每 100 microstep；最终另存 adapter |
| mask 及任务比例 | NTP 的 mask-token 生成/理解；无 FM | 示例为 mask generation 数据 | 示例为 `mask_generation_gres` | 预训练 8B SAMTok mask token 保持不变；`NTP:ref:noref:plain=3:2:2:1`；NTP × 0.05、FM × 1.0 |

### 5.3 推理：DiffSynth 编辑、SAMTok 定位与当前两次前向

DiffSynth 对照 [官方 Qwen-Image-2.1 pipeline](https://github.com/modelscope/DiffSynth-Studio/blob/7686e54d41d25c0e8ed5f1318acc23b6bb832654/diffsynth/pipelines/qwen_image_21.py) 及 [官方推理示例](DiffSynth-Studio/examples/qwen_image_21/model_inference/Qwen-Image-2.1.py)；SAMTok 对照 [8B 发布页 Quickstart](https://huggingface.co/zhouyik/Qwen3-VL-8B-SAMTok) 与本地 [Qwen3-VL demo](samtok/demo/qwen3vl_samtok_infer.py)。DiffSynth 原生推理没有 SAMTok 定位步；SAMTok 原生推理没有图像扩散编辑步。

| 参数 | DiffSynth 2.1 原生编辑 | SAMTok Qwen3-VL-8B 原生定位 / mask 解码 | 当前项目默认 online 推理 |
|---|---|---|---|
| 输入 / 前向 | prompt + `edit_image`，一次扩散编辑 | 单次 Qwen3-VL image+question chat；需要可视化时再用 VQ-SAM2 解码 mask | Pass 1 原生 SAMTok chat 定位；Pass 2 将 mask tokens 内联指令后交给 Qwen 2.1 编辑 |
| VLM 生成 | 不适用 | demo/8B Quickstart：`max_new_tokens=512`、`do_sample=False`、`top_p=1.0` | 定位 `max_new_tokens=256`、`do_sample=False`、`use_cache=True`、默认 `candidates=1`；遇到 im_end 特殊 token 停止 |
| 采样温度 | 不适用 | 贪心生成，无生效温度 | 默认贪心，无生效温度；仅 `localize --candidates >1` 采样时传 `temperature=0.8` |
| mask codec | 不适用 | 两级、每级 256 code；`DirectResize(1024)`；解码后的 raw mask logits `>0.5` 二值化 | 同一 256×2 codebook、`DirectResize(1024)` 和 raw logits `>0.5`；定位阶段通常只传 mask token，不执行空间 mask 解码 |
| 输出大小 / 去噪步 | pipeline 默认 1024×1024 / 40 | 无扩散去噪步；mask 可插值回原图大小 | 默认 1024×1024 / 40；`--height/--width/--steps` 可覆盖 |
| CFG / negative prompt | `cfg_scale=1.0` / 单空格 `" "` | 不适用 | `--cfg=1.0`；未覆盖 pipeline 的单空格 negative prompt |
| seed / 随机数设备 | pipeline `seed=None`、`rand_device="cpu"`；官方编辑示例显式 `seed=1` | demo 未显式设 seed | `--seed=0`；沿用 pipeline 的 CPU 随机数设备 |
| KV cache / VAE tiling | KV cache 开；VAE tiling 关，若开启则 tile 256、stride 192 | VLM `generate` 常规 cache；无扩散 VAE tiling | KV cache 开（可用 `--no-kv-cache` 关闭）；VAE tiling 沿用关闭 |
| 模式 / 输出 | 原生 `pipe(...)` 生成 RGBA 图 | 文本及可选解码 mask | `--mode=online`、`--variant=noref`，失败时可按协议回退 ref/plain；RGBA PNG；`--benchmark-output` 另作白底/原尺寸 RGB 后处理 |

## 6. cache-v2 与迁移

`samtok21-cache-v2` 是唯一新写入/训练格式：

- identity 包含 Qwen DiT/VAE/scheduler、SAMTok TE 权重及 tokenizer 配置、官方 processor 的逐文件 SHA256；未使用的官方 TE、SAM2/codec 不在 diffusion conditioning 指纹中。
- 包含 TE adapter 权重和配置 hash、preprocessing 协议标识、max_pixels、metadata 文件 hash、有序 row hashes 的紧凑摘要 rows_sha256。模型目录路径不是身份，同内容搬目录可接受，同路径换内容会失败。
- 大权重哈希由主 rank 计算并广播；它是真实完整文件 hash，会增加启动 I/O，不依赖 mtime 或不可信的旧 hash 缓存。
- dataset 把真实 row_index 随输出传递；payload 与 sidecar 同时记录 identity/row_hash/index，manifest 按真实行号排序。不得根据 shard 枚举猜测行号。完整行列表只在 manifest 中出现，shard identity 的大小不随数据集行数增长；兼容读取本轮早期使用完整 row_hashes 的 v2 smoke 产物。
- 发布 manifest 前、Stage 2 前校验协议、NTP 排除、checksum、来源、唯一且完整的行号、路径不越界、TE 维度/mask、source/target latent 网格。官方全有效 text mask 可以是 None。
- Stage 2 加载模型前核对实际模型内容；推理加载 DiT adapter 前核对 TE/processor/基座/adapter 内容。cache 的 max_pixels 随 Stage 2 配方保存，推理可选不同输出尺寸，不强制同分辨率。

训练/cache 输出必须是新目录（允许已存在的空目录）；中途失败不会发布成功 manifest，重试用新目录。源数据始终只读。

旧两种 cache-v1 布局只能通过 `provenance.audit_legacy_cache` 只读审计。旧 `te_adapter_identity={...}`、`te_adapter={...}` 可规范化；path-only 明确报告缺少历史 hash，而不是把 dict 当路径或用当前文件伪造历史证明。由于旧格式没有完整模型/配置身份，本次不自动迁移为可信 v2：**重建缓存**。旧 Stage 2 adapter 缺乏 v2 provenance 时推理明确拒绝；需基于可核实历史记录作独立迁移评审，不能仅补当前 hash。单纯同步 attention patch 不改变权重 schema/TE 条件，不要求因该项本身重训。

## 7. 环境与训练命令

在 repo 根目录，使用 Python 3.11；`constraints-tested.txt` 锁定本次实际验证的 Linux x86_64/H100/CUDA 12.8 环境（torch 2.8.0、torchvision 0.23.0、transformers 5.12.1、accelerate 1.14.0、peft 0.20.0）。不是所有平台通用锁。依赖已补 ijson、pyarrow、Hydra、OpenCV、SciPy；无需依赖 user-site。

```bash
python3.11 -m venv /path/to/samtok21-venv
source /path/to/samtok21-venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$PWD:DiffSynth-Studio
export PYTHONDONTWRITEBYTECODE=1
```

旧文档中的 `/opt/tiger/tanyue/samtok_edit_qwen_image_2_1/.venv` 在 2026-09-26 检查时不存在，不要默认 source 它。模型路径可用 `--qwen/--samtok` 覆盖；默认路径见 model.py。

以下 `/path/...` 均需替换为实际路径；单卡用普通 python，多卡建议 torchrun 显式指定数量，不依赖外部 accelerate 配置：

```bash
torchrun --standalone --nproc_per_node=2 -m samtok_edit21.train train --stage stage1 \
  --metadata /path/stage1.jsonl --base-path /path/data \
  --output /path/new-stage1 --steps 1000 --accumulation 8 --seed 20260926

torchrun --standalone --nproc_per_node=2 -m samtok_edit21.train cache \
  --metadata /path/stage2.jsonl --base-path /path/data \
  --te-adapter /path/new-stage1/adapter --output /path/new-cache

torchrun --standalone --nproc_per_node=2 -m samtok_edit21.train train --stage stage2 \
  --cache /path/new-cache --output /path/new-stage2 --steps 1000 --accumulation 4 --seed 20260926
```

多卡 Stage 1 accumulation 必须为 8 的倍数，Stage 2 为 4 的倍数，保证每 rank 的比例一致。两阶段最终可消费产物均为 `adapter/{adapter.json,adapter.safetensors}`；官方 step checkpoint 只含训练权重，不是独立完整 resume/adapter 包。

## 8. 推理、noref 边界与评测

```bash
python -m samtok_edit21.cli localize --image /path/source.png \
  --prompt 'Make the leftmost bird blue.' \
  --te-adapter /path/new-stage1/adapter --output /path/localize.json

python -m samtok_edit21.cli infer --image /path/source.png \
  --prompt 'Make the leftmost bird blue.' --variant noref \
  --te-adapter /path/new-stage1/adapter --dit-adapter /path/new-stage2/adapter \
  --height 1024 --width 1024 --output /path/result.png
```

online/oracle 默认 requested_variant=noref；两者使用同一 `condition_localization`。报告 `requested_variant/actual_variant/fallback_reason`：

- 已审核 `--units-file` 提供每个定位分组的 `ref_phrase/edit_type/anchor_phrase?`，顺序与分组一致，ref_phrase 必须精确对应绑定后的 label；mask codes 来自定位结果而非这个文件。
- 没有审核信息时，只对少量明确的英文全句语法推断 remove/replace/text、颜色属性、简单动作和无空间 anchor 的 add；global 使用 this image。不是通用语义解析器。
- add 必须保留新增物体；复杂 add 需要审核的末尾 anchor。text 必须保留目标文字。composite 需要逐 unit 的审核语义。审核 ref_phrase 应覆盖完整 where；不要把 what/how 包入待删除的引用范围。未标注的其他补语保留，不再任意删除整段 from ...；未审核的 and/then/while 等复合语句会拒绝自动推断。
- noref 改写不可靠时回退 ref；定位解析/绑定失败时 online 回退 plain。严格实验加 `--strict-noref`，无法正确生成 noref 就报错，不能把 fallback 混进 noref 得分；该标志只用于 online/oracle。
- localize 输出是候选报告列表；oracle 的 `--cot-file` 需要其中的 raw canonical mask JSON 内容（或另行提供的 canonical JSON），不是整份候选报告列表。

审核 units 示例：
```json
[{"ref_phrase":"a red ball next to the chair","edit_type":"add","anchor_phrase":"next to the chair"}]
```

`direct/stock` 定义为无 mask 普通编辑；`inline` 必须已有合法 mask span；`interactive` 使用一张源图和所选 mask，按选区顺序调用 codec 后插入指代短语。所有 masked 模式统一要求单 source，CLI 在加载模型前拒绝 masked 多图；普通 direct/stock 仍可多图。

原生输出保持 RGBA PNG。评测时显式传 `--benchmark-output`：保留 `result.raw.png`，白底 alpha composite 后转 RGB，并 resize 至参考源图原始尺寸，JSON 记录 raw/final 尺寸。多图必须指定 `--reference-image-index`（0-based），不能猜参考图；所有基线必须使用相同后处理。

## 9. 科研边界与后续更新规范

两个 code 能被 codec 解码，不意味着 DiT 已有硬性空间约束或背景保护；本项目未加入 attention supervision、regional FM、inference attention bias。现有规划见 [mask 区域约束实现规划](SAMTokEdit_Qwen21_mask区域约束实现规划.md)，不混入本轮修 bug。

noref 不保证所有空间信息只来自 mask；what/how 与图像仍可能泄露目标。正式评测应分开统计格式成功率、绑定成功率、noref 覆盖率、定位 IoU、区域内编辑与区域外保真，并做正确/交换/随机/无 code 对照。当前 JSON 不携带 atomic edit_type，因此任意自然语言自动 noref 仍需独立决定协议升级或语义解析步骤，本轮采用“可审核输入 + 保守子集 + 显式回退/strict”的有界实现。

codec 保持发布实现的 raw logits > 0.5 阈值；这不等于 sigmoid > 0.5，也不是 2.1 迁移 bug。以后改变阈值需作为独立实验记录。

### 2026-09-26 变更登记

F1/F2/F4：统一入口、真实 adapter 配置、cache-v2 内容身份、旧产物显式拒绝/审计。
F6：同步官方 #1697。A1/A2/A3：显式 LR、全 RNG seed、当前 backward 梯度审计。
F5/F3：唯一标签 round-trip、可审核 noref。A4/A7：单图绑定、独立 benchmark 后处理。
A5/A6：实测依赖约束、可执行环境/命令说明及 localize prompt。

后续任何代码/数据协议更新必须同步维护本文当前行为，在实验记录追加日期、代码版本、来源归属、兼容性影响、命令、产物、实测结果与未覆盖项。影响 TE 条件或 resize 的改动必须升级 preprocessing 标识并重建缓存；不要覆盖历史失败记录，也不要把计划中的功能写成已验收。
