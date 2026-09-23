# SAMTokEdit Qwen-Image-2.1 官方接口适配实验记录

## 1. 实验分支和输入

```text
仓库：/opt/tiger/tanyue/samtok_edit_qwen_image_2_1_official
分支：qwen-image-2.1-official-api
起点：samtok_edit/main
DiffSynth：d2d684ad / 2.1.8
```

模型使用用户提供的本地路径：

```text
/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen-Image-2.1
/mnt/bn/strategy-mllm-train/user/tanyue/models/SAMTok/Qwen3-VL-8B-SAMTok
```

源数据路径保持只读。本次 smoke metadata 位于仓库内 `smoke_data/`，图像引用用户提供的已有调试数据目录，没有复制或修改源图像。

## 2. 协议和 schedule 检查

使用已有 CrispEdit 调试图像构造了：

```text
Stage 1：8 行
  edit_ntp       3 行
  edit_umt/ref   2 行
  edit_umt/noref 2 行
  edit           1 行

Stage 2：4 行
  edit_umt/ref   1 行
  edit_umt/noref 2 行
  edit           1 行
```

协议检查：

```text
rows=8
decoded_images=9
passed=true
```

在模拟 2 卡、Stage 1 `accumulation=8` 时，每张卡得到：

```text
edit_ntp=3
edit_umt/ref=2
edit_umt/noref=2
edit=1
```

在模拟 2 卡、Stage 2 `accumulation=4` 时，每张卡得到：

```text
edit_umt/ref=1
edit_umt/noref=2
edit=1
```

如果多卡的 accumulation 不能整除对应类型 block，官方适配入口会拒绝启动。这避免了不同 rank 消费不同类型比例的问题。

## 3. 自动化测试

运行命令：

```bash
source /opt/tiger/tanyue/samtok_edit_qwen_image_2_1/.venv/bin/activate
python -m pytest -q
```

结果：

```text
23 passed
```

测试覆盖：

- canonical mask code 和 JSONL 协议；
- cache row hash、sidecar 和 checksum；
- source latent 与 image-pad geometry；
- 多卡 schedule 的每卡类型比例；
- 官方 runner 的自定义 sampler 接入；
- 官方 optimizer/accumulation/checkpoint 路径；
- 官方 cache manifest 的 geometry 和损坏检测。

其中官方 runner 的 CPU dummy smoke 生成了：

```text
loss.csv
step-4.safetensors
```

这验证了修改后的 `launch_training_task` 可以继续完成 optimizer、gradient accumulation、logger 和 checkpoint。

## 4. 真实模型链路状态

本环境当前没有可用 NVIDIA driver：

```text
nvidia-smi: failed to communicate with the NVIDIA driver
torch.cuda.is_available(): False
```

因此本次会话完成了真实 metadata、processor 配置和训练代码的静态检查、CPU 协议/cache/runner smoke，但没有在真实 Qwen3-SAMTok + Qwen-Image-2.1 权重上执行 GPU forward、反向传播和显存 smoke。之前的 qwen-image-2.1 custom runner 已对相同模型权重完成过 GPU loss、梯度、DDP 和 cache geometry 审计；本分支新增的官方 runner 层已由上述 CPU runner 和 schedule 测试验证。恢复 GPU 环境后，建议按下面顺序运行完整 smoke：

```bash
PYTHONPATH=.:DiffSynth-Studio python -m samtok_edit21.cli validate \
  --metadata smoke_data/stage1.jsonl \
  --base-path /mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/crispedit_refined/stage1_8gpu_smoke/data/crispedit_samtok

accelerate launch --num_processes 1 \
  -m samtok_edit21.official_api train \
  --stage stage1 \
  --metadata smoke_data/stage1.jsonl \
  --base-path /mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/crispedit_refined/stage1_8gpu_smoke/data/crispedit_samtok \
  --output /mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/qwen_image_2_1_official_api/stage1 \
  --steps 1 --accumulation 8

accelerate launch --num_processes 1 \
  -m samtok_edit21.official_api cache \
  --metadata smoke_data/stage2.jsonl \
  --base-path /mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/crispedit_refined/stage1_8gpu_smoke/data/crispedit_samtok \
  --te-adapter /mnt/.../stage1/adapter \
  --output /mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/qwen_image_2_1_official_api/cache

accelerate launch --num_processes 1 \
  -m samtok_edit21.official_api train \
  --stage stage2 \
  --cache /mnt/.../cache \
  --output /mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/qwen_image_2_1_official_api/stage2 \
  --steps 1 --accumulation 4
```

正式 GPU smoke 需要额外检查：

1. Stage 1 NTP 和 FM 的 LoRA 梯度均非零；
2. 冻结的 vision、lm_head、DiT、VAE 无梯度；
3. Stage 1 checkpoint 可以生成官方 cache；
4. Stage 2 cache manifest 与 TE adapter SHA256 一致；
5. Stage 2 DiT LoRA 梯度非零，TE/VAE 不加载到训练进程；
6. 2 卡时每卡的 schedule 报告分别为 3:2:2:1 和 1:2:1；
7. 两卡参数同步且没有 NaN/Inf/OOM。

## 5. 结果解释

当前测试证明的是官方接口扩展的控制流、数据协议、cache 格式、采样 schedule 和 CPU optimizer/checkpoint 路径。它没有替代真实 GPU 模型 smoke，也没有用这 8 行调试数据判断方法效果。恢复 GPU 后，官方 runner 和旧 custom runner 应使用同一固定 seed、noise、timestep、adapter 和 cache 做一条样本级对照，以确认条件 tensor、FM loss、梯度范数和单步 LoRA 更新。
