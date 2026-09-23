# SAMTokEdit Qwen-Image-2.1：基于官方 DiffSynth 接口的适配实现

## 1. 实现目标

本分支从 `samtok_edit/main` 创建，分支名为 `qwen-image-2.1-official-api`。实现目标是保留研究方案中的 Localize-then-Edit 方法，同时把训练的基础设施尽量放回 DiffSynth 官方接口：

```text
Stage 1：Qwen3-VL-8B-SAMTok TE LoRA
         NTP localization CE + Qwen-Image-2.1 FM loss

Stage 2a：官方 data-process runner
          冻结 Stage 1 TE adapter，缓存 TE 条件和 VAE latent

Stage 2b：官方 train runner
          只训练 Qwen-Image-2.1 DiT LoRA
```

官方 runner 负责 DDP、optimizer、gradient accumulation、日志和 checkpoint。SAMTok 专用的 TE、数据协议、损失组合、比例 schedule 和 cache 审计由 `samtok_edit21/official_api.py` 提供。

官方 runner 的优化器、`ConstantLR` 学习率调度器、累积和 checkpoint 路径均保留；本项目只通过显式参数覆盖学习率、权重衰减和梯度裁剪，不另造一套训练循环。

## 2. 上游代码

`DiffSynth-Studio/` 从 main 中的旧版本更新到官方 DiffSynth 2.1.8，固定 revision 为 `d2d684ad`。Qwen-Image-2.1 和 Qwen3-VL-8B-SAMTok 的本地 revision 记录在 [upstream_versions.json](upstream_versions.json)。

本适配没有修改 Qwen-Image-2.1 的 DiT、VAE 或 attention 结构。官方 Qwen-Image-2.1 pipeline、scheduler、VAE、PromptEmbedder 和 `model_fn_qwen_image_21` 直接复用。

## 3. 对官方 runner 的扩展

### 3.1 确定性 sampler

官方 [launch_training_task](DiffSynth-Studio/diffsynth/diffusion/runner.py) 默认创建 `DataLoader(shuffle=True)`。研究方法要求每个 optimizer window 保持固定类型比例，因此 runner 增加了一个兼容扩展：

```python
sampler = getattr(dataset, "official_sampler", None)
```

普通数据集仍走官方 `shuffle=True`。如果数据集提供 `official_sampler`，runner 使用该 sampler，其他 DDP、optimizer 和 logger 逻辑保持官方实现。

`ScheduledMetadata` 和 `ScheduledCache` 使用这一入口。schedule 由原有 `make_schedule()` 生成，按 rank 交错组织，使多卡上每张卡得到相同类型比例。

Stage 1 的每卡 accumulation 必须是 8 的倍数；Stage 2 必须是 4 的倍数：

```text
Stage 1：edit_ntp : edit_umt(ref) : edit_umt(noref) : edit
        = 3 : 2 : 2 : 1

Stage 2：edit_umt(ref) : edit_umt(noref) : edit
        = 1 : 2 : 1
```

例如 2 卡时：

```text
Stage 1 accumulation=8：每张卡 3/2/2/1，共 8 个 micro-step
Stage 2 accumulation=4：每张卡 1/2/1，共 4 个 micro-step
```

如果多卡使用了无法整除类型 block 的 accumulation，官方适配入口会直接报错，避免误把全局比例当成每卡比例。

### 3.2 梯度审计和梯度裁剪

官方 runner 增加了两个可选扩展：

- `after_backward_audit()`：在 optimizer step 前检查可训练梯度数量、梯度范数、冻结参数梯度和有限值；
- `max_grad_norm`：保留研究方案默认的 `1.0` 梯度裁剪，参数为空时保持官方行为。

这些扩展只在模型提供对应属性时生效，官方其他模型的训练行为不变。

## 4. SAMTok 模型接入

`samtok_edit21/model.py` 中的 `load_pipeline()` 做以下工作：

1. 通过官方 `QwenImage21Pipeline.from_pretrained()` 加载 2.1 DiT/VAE；
2. 通过 `Qwen3VLForConditionalGeneration.from_pretrained()` 加载完整的 Qwen3-VL-8B-SAMTok；
3. 将 Qwen3-SAMTok tokenizer 和 chat template 接入 Qwen-Image-2.1 processor；
4. 检查 hidden size、patch size、image token id 和 514 个 mask token 是否匹配；
5. 用 `SamtokTextEncoder` 区分两个特征出口：
   - FM 读取最终 RMSNorm 之前的 4096 维特征；
   - NTP 读取最终 RMSNorm 之后的 hidden，再经过冻结 lm_head。

编辑条件使用官方 `QwenImage21Unit_PromptEmbedder`。图像 resize 和 source VAE latent 使用官方 `QwenImage21Unit_EditImageEmbedder` 的逻辑。TE 的 RGB 白底合成和 VAE 的 RGBA 输入保持官方 2.1 语义。

## 5. Stage 1 实现

`OfficialSamtokTrainingModule` 在官方 `DiffusionTrainingModule` 上实现 `forward()`：

### NTP 分支

`edit_ntp` 样本只读取 source image 和 `mt_cot`。模型输入是：

```text
原图 + 原始编辑指令 + localization request
```

监督目标是 canonical mask JSON 加 `<|im_end|>`。代码只对 assistant 预测区间计算 CE，并使用：

```text
loss = 0.05 * loss_ntp
```

### FM 分支

`edit_umt` 和 `edit` 样本使用官方 Qwen-Image-2.1 flow matching：

```text
timestep：1000 个训练 timestep 中均匀采样
noisy：scheduler.add_noise(clean, noise, timestep)
target：scheduler.training_target(clean, noise, timestep)
prediction：官方 Qwen-Image-2.1 model_fn
loss：fp32 MSE × scheduler.training_weight(timestep)
```

FM 的梯度通过冻结的 DiT 反传到 TE LoRA，因此 Stage 1 不能把 TE FM forward 放在 `no_grad` 中。

最终 loss 为：

```text
L_stage1 = 0.05 * L_NTP + 1.0 * L_FM
```

Stage 1 默认参数：

```text
TE LoRA rank / alpha：64 / 64
dropout：0.05
learning rate：4e-5
weight decay：0.05
gradient clipping：1.0
base model：bf16
LoRA 参数和 optimizer state：fp32
```

## 6. Stage 2a：官方缓存接口

命令入口：

```bash
PYTHONPATH=.:DiffSynth-Studio python -m samtok_edit21.official_api cache \
  --metadata "$E/data/stage2.jsonl" \
  --base-path "$E/data" \
  --te-adapter "$E/stage1/step-000100" \
  --output "$E/official_cache" \
  --max-pixels 1048576
```

该命令调用官方 `launch_data_process_task`。SAMTok 专用 model forward 生成并保存：

```text
input_latents
edit_latents
prompt_embeds
prompt_embeds_mask
edit_image_pad_mask
```

Stage 2a 不加载 DiT，也不计算 loss。Stage 1 TE LoRA 以冻结 adapter 挂载，保证 cache 与推理时的 TE 计算一致。

官方 runner 生成的 rank 子目录会额外生成 sidecar 和 `manifest.json`。sidecar 保存 row hash、文件 SHA256、模型路径、TE adapter 和分辨率预算。Stage 2 启动前会重新检查所有 hash、tensor 有限值和几何恒等式：

```text
sum(edit_image_pad_mask) * 4
    == sum(source_latent_height * source_latent_width)
```

## 7. Stage 2b：官方训练接口

命令入口：

```bash
PYTHONPATH=.:DiffSynth-Studio python -m samtok_edit21.official_api train \
  --stage stage2 \
  --cache "$E/official_cache" \
  --output "$E/stage2_official" \
  --accumulation 4 \
  --rank 32 \
  --lr 1e-4 \
  --weight-decay 0.01
```

该命令调用官方 `launch_training_task`。训练进程只加载 Qwen-Image-2.1 DiT，cache 中的条件特征和 latent 直接输入官方 FM loss。TE 和 VAE 不加载到 Stage 2 训练进程中。

Stage 2 默认参数：

```text
DiT LoRA rank / alpha：32 / 32
dropout：0
learning rate：1e-4
weight decay：0.01
gradient clipping：1.0
```

DiT LoRA target 使用 2.1 官方空 target 语义，即所有 `nn.Linear`。2511 双流模块名称不再使用。

## 8. 初始化和 checkpoint

Stage 1 从完整的 Qwen3-VL-8B-SAMTok checkpoint 开始，在 language model 的 q/k/v/o 和 gate/up/down projection 上注入新 LoRA。Stage 2 从 Qwen-Image-2.1 预训练 DiT 开始，注入新的 DiT LoRA。

官方 `ModelLogger` 继续保存 `step-*.safetensors`。训练结束后适配层还会导出为 `output/adapter/{adapter.json,adapter.safetensors}`，供 Stage 1 cache 和 `--init-adapter` 直接使用；这样不会把官方 step checkpoint 格式误当成 adapter 目录。

`--init-adapter` 只加载 LoRA 权重，optimizer、scheduler 和数据进度重新开始。Stage 2 checkpoint 的 manifest 固定记录它依赖的 Stage 1 TE adapter SHA256，推理加载时检查这两个身份是否匹配。

## 9. 测试边界

当前实现包含三层检查：

1. JSONL 协议、mask code 范围、ref/noref 字段和图像路径检查；
2. cache row hash、文件 checksum、tensor finite 和 TE/VAE geometry 检查；
3. 官方 sampler、optimizer、checkpoint 和单步梯度审计检查。

本分支的 CPU 测试覆盖协议、缓存损坏、每卡比例、官方 sampler 接入、官方 runner checkpoint 和官方 cache geometry。完整测试结果见实验记录。
