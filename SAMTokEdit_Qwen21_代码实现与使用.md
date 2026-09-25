# SAMTokEdit：Qwen-Image-2.1 实现与使用

本文记录本仓库从 `main` 中的 DiffSynth 与 SAMTok 快照开始，到 Qwen-Image-2.1 两阶段训练、缓存和推理的完整实现。代码基于仓库内固定的 DiffSynth 版本开发；上游训练器仍负责分布式、优化器、梯度累积、裁剪、日志和 checkpoint，SAMTokEdit 只扩展模型输入、数据协议、损失组合与缓存校验。

## 1. 代码来源与模型组合

- `DiffSynth-Studio/`：Qwen-Image-2.1 的 DiT、VAE、scheduler、pipeline 和训练 runner。
- `samtok/`：SAMTok 的 VQ-SAM2 mask tokenizer 以及 Qwen3-VL 处理代码。`samtok/models/__init__.py` 对 XTuner 相关模块采用懒加载，避免 codec 在推理和数据转换时导入可选训练栈。
- `samtok_edit21/model.py:95-136`：用 DiffSynth `QwenImage21Pipeline` 加载 DiT/VAE，用 `Qwen3-VL-8B-SAMTok` 加载 4096 维文本编码器，并检查 hidden size 4096、视觉 patch size 16。
- `samtok_edit21/codec.py`：调用 Qwen3-VL-8B-SAMTok 发布的 `sam2.1_hiera_large.pt` 与 `mask_tokenizer_256x2.pth`，把二值 mask 编码为两个 256 码本 token。加载标准 SAM2 backbone 时会报告 mask encoder 缺失键；mask tokenizer 权重随后补齐这些层，最终使用 `strict=True` 加载 tokenizer 状态并完成编码。

Qwen-Image-2.1 的 TE 与 DiT 都是 2.1 接口：TE 输出最后维度 4096，视觉图像经过 Qwen3-VL processor 的 patch/grid 处理；DiT 侧 VAE latent 为 `[B,64,H/16,W/16]`，由同一组 `height/width` 决定。`samtok_edit21/training.py:116-149` 的 `validate_conditioning` 检查 TE visual grid 与 VAE grid 的关系，禁止两侧独立 resize。

## 2. 数据协议

`protocol.py` 定义协议版本 2：

- `edit`：普通图像编辑，只有 `edit_image/image/prompt/sample_type/edit_type`。
- `edit_ntp`：Stage 1 的定位 NTP，使用 `edit_image/prompt/mt_cot`。`mt_cot` 是严格 JSON fenced list，每个元素只有 `mask_2d` 和 `label`。
- `edit_umt`：带 mask 的统一编辑；`instr_variant` 为 `ref` 或 `noref`，mask span 内联到 prompt。
- mask span 必须是 `<|mt_start|><|mt_0000|><|mt_0256|><|mt_end|>` 形式，第一层码在 `[0,255]`，第二层码在 `[256,511]`。
- `parse_cot`、`spans_in`、`render_units` 拒绝截断 span、非法码、控制字符、重叠或歧义引用，不自动修复训练标签。

源数据路径只读。`tests/eight_gpu_smoke/prepare_refedit.py` 从 `RefEdit-mask-prefiltered-qwen38-self-contained` 读取嵌入式 `source_img/target_img/mask_png`，在实验目录中物化 PNG 和 JSONL；生产数据转换应在人工审核的 phrase rewrite 后使用 `render_units`，smoke builder 中的 `"in this region"` 仅用于调通接口。

## 3. LoRA 与两阶段训练

`samtok_edit21/training.py:29-51` 实现 `add_adapter`：

- Stage 1：Qwen3-VL language model 的 attention 和 MLP 投影层 LoRA，默认 `r=64, alpha=64, dropout=0.05`。
- Stage 2：Qwen-Image-2.1 DiT 中每个 `nn.Linear` LoRA，默认 `r=32, alpha=32, dropout=0`。2.1 没有 2511 的 `add_q_proj/txt_mlp/txt_mod` 分支，因此不复用旧目标名。
- LoRA 参数转成 fp32 保存和更新，基础权重保持冻结。`load_adapter` 严格检查 adapter schema；`save_adapter` 用 safetensors 并检查有限值。

Stage 1 的 `SamtokTrainingModule.forward`（`train.py:134-207`）按样本类型分流：

1. `edit_ntp`：Qwen3-VL chat template + image processor，assistant 区间交叉熵由 `model.py:193-208` 计算，乘 `--ntp-weight`，默认 0.05。
2. 其他编辑样本：VAE 目标/源图 latent 在 `no_grad` 中计算，TE 条件保持梯度，调用 Qwen-Image-2.1 scheduler 与 `model_fn` 的 Flow Matching SFT loss，乘 `--fm-weight`，默认 1.0。

因此默认 FM:NTP 的数值权重为 20:1。真实训练中应结合每种 loss 的原始尺度和梯度 norm 调整；smoke test 只验证有限 loss 与非零 LoRA 梯度，不代表最终权重最优。

Stage 2 只加载 DiT 和 cache 条件，继续使用官方 scheduler 的 flow matching loss。默认 `lr=1e-4, weight_decay=0.01, accumulation=4`；Stage 1 默认 `lr=4e-5, weight_decay=0.05, accumulation=8`。`--init-adapter` 只 warm-start LoRA 权重，optimizer、scheduler 和数据进度重新开始。

## 4. 精确采样与分布式实现

`data.py:make_schedule` 先按 `sample_type`/`instr_variant` 建池，再按固定 seed 产生全局 schedule：

- Stage 1 每个 optimizer step 的全局 batch 为 `world_size * accumulation`，比例为 NTP 3、UMT-ref 2、UMT-noref 2、plain edit 1。
- Stage 2 比例为 UMT-ref 1、UMT-noref 2、plain edit 1。
- `train.py:ScheduledMetadata` 和 `ScheduledCache` 暴露 `schedule_sampler` 顺序 sampler；扩展后的 DiffSynth runner 再按 rank 切片，因此 8 卡、Stage 1 accumulation 8 时每张卡都得到 3/2/2/1，Stage 2 accumulation 4 时每张卡都得到 1/2/1。runner 同时兼容旧项目中的临时 sampler 属性名。
- `train.py:after_backward_audit` 检查可训练参数梯度有限、非零，且冻结参数没有梯度；`runner.py` 在同步梯度时才执行裁剪，避免 accumulation 中间步改变梯度。
- 每个 rank 的 device 在加载模型前由 `Accelerator.device` 显式绑定为 `cuda:<local_rank>`。不能把 Transformers 的 `device_map={"":"cuda"}` 留给多卡任务，否则所有进程会落到 GPU 0。

## 5. 官方 runner 扩展与 cache

`DiffSynth-Studio/diffsynth/diffusion/runner.py` 保留上游 `launch_training_task`/`launch_data_process_task` 的生命周期，并支持 dataset 暴露 `schedule_sampler`、可选 `load_from_cache`、可选 `after_backward_audit`。`train.py` 仅实现模型 forward 和数据集：

```bash
source /opt/tiger/tanyue/samtok_edit_qwen_image_2_1/.venv/bin/activate
export PYTHONPATH=$PWD:DiffSynth-Studio
```

Stage 1：

```bash
accelerate launch --num_processes 8 --mixed_precision bf16 \
  -m samtok_edit21.train train --stage stage1 \
  --metadata /path/stage1.jsonl --base-path /path/data \
  --output /path/stage1 --steps 1000 --accumulation 8
```

Stage 1 adapter 输出为 `/path/stage1/adapter/{adapter.json,adapter.safetensors}`。

Stage 2 cache：

```bash
accelerate launch --num_processes 8 --mixed_precision bf16 \
  -m samtok_edit21.train cache \
  --metadata /path/stage2.jsonl --base-path /path/data \
  --te-adapter /path/stage1/adapter --output /path/cache
```

cache 每个 rank 产生 `.pth` 和 sidecar；主进程写 `manifest.json`，记录 metadata hash、模型路径、TE adapter hash、shape 配置和每行 hash。`verify_cache` 在 Stage 2 训练前检查所有 shard、sidecar、SHA256、协议字段和 conditioning shape。

Stage 2：

```bash
accelerate launch --num_processes 8 --mixed_precision bf16 \
  -m samtok_edit21.train train --stage stage2 \
  --cache /path/cache --output /path/stage2 --steps 1000 --accumulation 4
```

## 6. 推理流程

`cli.py` 提供三步：

1. `validate`：检查 JSONL、图片、mask span 和协议字段。
2. `localize`：加载 Qwen3-VL adapter，对单张源图生成严格 `mt_cot`，可用 codec 解码 mask。
3. `infer`：加载 TE adapter 和 DiT adapter，在线生成或接受已定位的 inline prompt，调用 Qwen-Image-2.1 pipeline 输出 PNG。

示例：

```bash
python -m samtok_edit21.cli validate --metadata /path/stage1.jsonl --base-path /path/data
python -m samtok_edit21.cli localize --image /path/source.png \
  --te-adapter /path/stage1/adapter --output /path/localize.json
python -m samtok_edit21.cli infer --image /path/source.png \
  --prompt '...' --te-adapter /path/stage1/adapter \
  --dit-adapter /path/stage2/adapter --output /path/result.png
```

## 7. Debug smoke 流程

测试脚本全部位于 `tests/eight_gpu_smoke/`，产物位于：

`/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/qwen_image_2_1_dev_smoke/`

```bash
python tests/eight_gpu_smoke/prepare_refedit.py \
  --source /mnt/bn/strategy-mllm-train/user/tanyue/datasets/RefEdit-mask-prefiltered-qwen38-self-contained \
  --output /mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/qwen_image_2_1_dev_smoke/refedit_data \
  --samtok /mnt/bn/strategy-mllm-train/user/tanyue/models/SAMTok/Qwen3-VL-8B-SAMTok \
  --unique-rows 8
```

先运行 `probe_distributed.py` 检查 rank/device 和采样比例，再运行 Stage 1、cache、Stage 2 和单卡推理。完整命令、日志、失败尝试与修复见 [`SAMTokEdit_Qwen21_实验记录.md`](SAMTokEdit_Qwen21_实验记录.md)。

## 8. 后续修改记录

后续任何代码或数据协议更新都应在本节追加日期、文件、原因、兼容性影响和新的运行命令；不要覆盖历史实验记录。
