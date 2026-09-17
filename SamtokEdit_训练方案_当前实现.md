# SamtokEdit 训练方案（当前实现）

> SAMTok mask-token CoT 引导的 Qwen-Image-Edit-2511 细粒度编辑实现说明。
>
> 本文是当前仓库的自包含实现规范，完整描述方法目标、四类数据 schema、数据构建、
> 两阶段训练、推理评测、实现路径和运行约束；代码片段和命令均以当前实现为准，
> 不依赖仓库外的设计说明文档。

当前代码实现包含 Stage 1/Stage 2 训练入口、canonical CoT 数据管线、SAMTok codec 构建器、
Qwen-Image-Edit-2511 / SAMTok gres-ft 模型适配和 Stage 1/Stage 2 评测入口。实验运行过程与结果单独记录在
`SamtokEdit_实验记录.md`。

---

## 方法概述

**问题**：编辑模型容易出现指代性定位失败，例如多实例选错对象、局部编辑引发全局漂移，
导致细粒度编辑能力不足。

**方法**：使用 Qwen-Image-Edit-2511 的 Qwen2.5-VL text encoder，并替换为带有 SAMTok
词表和 mask-token 生成能力的 `QwenImageSamtokTextEncoder`。给定源图和编辑指令，
SAMTok 将一个二值 mask 编码成两个离散 token；text encoder 在 assistant 段自回归生成
mask-token CoT，并对“编辑模板 + CoT”整条序列做一次 forward，取最后一层 hidden
作为 `prompt_emb` 注入 Qwen-Image DiT。

**当前固定设定**：

1. 基座是 `Qwen-Image-Edit-2511`，不是 2509；训练和推理均显式启用 `zero_cond_t`。
2. text encoder 使用 `/mnt/bn/strategy-mllm-train/user/tanyue/models/SAMTok/Qwen2.5-VL-7B-SAMTok-gres-ft`。
3. DiT/VAE 使用 `/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen-Image-Edit-2511`。
4. tokenizer 和 processor 使用 `prepare_samtok_te_dir.py` 生成的合并目录，当前验证目录为
   `/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/artifacts/merged_samtok_te`。
5. processor 类为 `Qwen2VLProcessor`；合并词表大小为 `152179`，新增 SAMTok token
   范围为 `<|mt_start|>`、`<|mt_0000|>`…`<|mt_0511|>`、`<|mt_end|>`。
6. 当前 GRES 生成模板全部为英文；CrispEdit 原始 instruction 会被保留，构建器支持
   `--ascii_only` 对 prompt 和非空 CoT label 做 ASCII 约束。
7. 采样分布和 Qwen-Image-Edit-2511 的 flow-matching 设定不改；Stage 1 当前默认权重
   `ntp_loss_weight=0.05`、`fm_loss_weight=1.0`，均可由 launcher 环境变量覆盖。

训练分为两阶段：

- **Stage 1**：text encoder LLM 层 LoRA，在线 forward，同时计算 NTP 和 FM；CoT 从 metadata
  离线写入，训练时不在线生成。
- **Stage 2**：先融合 Stage 1 text-encoder LoRA 并离线缓存 prompt embedding 和 latent，
  再只训练 DiT LoRA，使用纯 FM loss。

---

## 运行环境与复现

当前验证环境是 Debian 12、Python 3.11.2、PyTorch 2.8.0+cu128、CUDA runtime
12.8、Transformers 5.12.1、Accelerate 1.14.0 和 8 张 H100 80GB；容器镜像为
`aliyun-va-hub.byted.org/reckon/data.reckon.mlx.image_4551_sg:ada11081bb25b40d8e1899588b696f24`。
镜像与宿主 NVIDIA driver 不属于 Python 项目依赖，需要由运行平台预先提供；
PyTorch wheel 按 `cu128` 安装。

仓库根目录的 `pyproject.toml` 精确固定当前直接训练、数据构建和评测依赖；
`DiffSynth-Studio` 保持普通代码目录，由 `setup_env.sh` 以 editable 本地包安装，
并在安装后校验 `diffsynth.__file__` 确实指向当前仓库，避免命中用户环境中遗留的
DiffSynth 2.0.18 editable 记录。默认构建命令是：

```bash
cd /opt/tiger/tanyue/samtok_edit
bash setup_env.sh
source .venv/bin/activate
```

脚本默认使用 Python 3.11、uv 0.11.32（仅在未安装 uv 时用该版本引导）、
ByteDance 内部 PyPI 和 `cu128` PyTorch backend；可通过 `bash setup_env.sh --help`
查看可覆盖参数。安装后执行依赖一致性、关键版本、CUDA、本地 DiffSynth 来源和
仓库单元测试校验。

如果默认内部 PyPI 暂未同步固定的 uv 版本，bootstrap 会只针对同一个精确 uv 版本回退到
公共 PyPI；项目其余依赖仍使用 `SAMTOK_EDIT_INDEX`。回退源可由
`SAMTOK_EDIT_UV_BOOTSTRAP_INDEX` 覆盖。因此在具备对应网络访问的干净 Python 3.11 机器上，
仍只需运行一次 `bash setup_env.sh`。

按要求不保留 lockfile：`setup_env.sh` 使用 `uv pip install`，不使用会生成
`uv.lock` 的 `uv sync`，`.gitignore` 也忽略根目录 `uv.lock`。直接依赖版本是精确的；
但无 lock 时传递依赖不能保证 bit-for-bit 不变，这是不使用 lockfile 的必然权衡。

---

## 统一序列与对齐规则

### 1.1 统一序列 S

编辑路径的模板段由 `DiffSynth-Studio/diffsynth/pipelines/qwen_image_samtok.py`
中的 `build_edit_model_inputs` 唯一构造：

```text
<|im_start|>system
Describe the key features of the input image (color, shape, size, texture, objects, background),
then explain how the user's text instruction should alter or modify the image. Generate a new image
that meets the user's requirements while maintaining consistency with the original input where appropriate.
<|im_end|>
<|im_start|>user
Picture 1: <|vision_start|><|image_pad|><|vision_end|>{edit instruction}<|im_end|>
<|im_start|>assistant
```

其中图片前缀由 `IMAGE_PROMPT_TEMPLATE = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"`
生成，条件图先按面积 `384 * 384` 等比 resize，并交给同一个 `Qwen2VLProcessor`。

令 processor 输出的模板段 token 为 `T`，长度为 `L_T`；CoT 段为单独 tokenizer 得到的
`C`，其内容为：

```text
mt_cot + "<|im_end|>"
```

当 `mt_cot` 不为 `None` 时，实际输入为：

```text
input_ids = cat(T, C)
attention_mask = cat(mask(T), ones_like(C))
```

`prompt_emb` 使用 text encoder 最后一层 hidden，并在 `EDIT_DROP_IDX = 64` 之后截取；
NTP 使用同一张 hidden 的位移切片：

```text
hidden[:, L_T - 1 : L_T - 1 + L_C]  -> lm_head -> labels = C
```

因此 NTP label 覆盖整个 canonical CoT 以及末尾的 `<|im_end|>`，prompt embedding 和
NTP 监督共享同一个 text-encoder forward。

纯文本兜底路径仍保留 Qwen-Image 原始 `encode_prompt`，使用其独立模板和 `drop_idx=34`；
本文讨论的编辑训练/推理路径使用 `EDIT_DROP_IDX=64`。

⟨M⟩ 的形式为：

```text
<|mt_start|><|mt_XXXX|><|mt_YYYY|><|mt_end|>
```

其中第一个 code 在 `[0, 255]`，第二个 code 已加 `256` 偏移，位于 `[256, 511]`。

### 1.2 对齐四律（训推一致性的全部来源）

**R1｜模板段和 CoT 段分段 tokenize，再在 ids 层拼接。** 模板段由 processor 处理，
包括视觉 token、`pixel_values` 和 `image_grid_thw`；CoT 段用 tokenizer 的
`add_special_tokens=False` 单独处理。禁止把 `prompt + mt_cot` 拼成字符串后二次整体 tokenize，
以免 BPE 跨段合并和 NTP 边界漂移。

**R2｜pass-1 和 pass-2 共用 `build_edit_model_inputs`。** 两者使用相同的 system prompt、
`Picture {i}:` 前缀、条件图 resize 和 processor。pass-1 的生成输入就是 pass-2 的模板段。

**R3｜CoT 只有一个 canonical 序列化入口。**
`DiffSynth-Studio/diffsynth/core/data/samtok_dataset.py` 中的 `to_cot` 用于 metadata
写入和生成结果重新序列化；推理 raw text 必须先经过
`parse_and_canonicalize_mt_cot`，pass-2 不直接消费 raw text。

**R4｜NTP 位移一位。** 模板末尾位置 `L_T-1` 的 hidden 预测 CoT 的第一个 token；label
是完整 `cot_ids`，包含 `<|im_end|>`。`QwenImageUnit_SamtokPromptEmbedder` 只对
CoT 切片计算 `lm_head`，不创建完整序列的 vocabulary logits。

### 1.3 推理：两轮 pass（全部在 pipeline 内部完成）

调用入口为 `scripts/inference/infer_samtok_edit.py` 中的 `run_edit`：

```text
用户 prompt + edit_image=[PIL.Image]
        │
QwenImageUnit_EditImageEmbedder 等原生 unit
        │  edit_image_auto_resize=True
        ▼
QwenImageUnit_SamtokEmbedder（pass-1）
        │ mt_cot 显式提供：canonicalize 后直接使用
        │ mt_cot=None 且 enable_samtok_cot=True：greedy generate，eos=<|im_end|>
        │ parse_and_canonicalize_mt_cot：strict → item → span → no target
        ▼
QwenImageUnit_SamtokPromptEmbedder（pass-2）
        │ 模板 T + 分段 tokenize 的 CoT C
        │ 一次 text_encoder.encode，输出 prompt_emb 和可选 NTP hidden
        ▼
Qwen-Image DiT denoise（zero_cond_t=True）→ VAE decode
```

`QwenImageSamtokPipeline.__call__` 支持：

- `mt_cot`：显式 GT CoT，主要用于 GT-CoT ablation；
- `enable_samtok_cot`：关闭后不在线生成，行为退化到无 CoT 条件路径；
- `samtok_max_new_tokens`：在线 pass-1 的生成上限，默认 `128`；
- `last_mt_cot`、`last_pass1_raw`、`last_parse_layer`：供日志、可视化和评测读取。

负分支不会拼接 CoT；训练时 `cfg_scale=1`，因此只使用正分支。

### 1.4 训练：两阶段

```text
Stage 1（TE LoRA，task=sft）
  metadata: edit_mt : edit_ntp : edit : edit_umt = 4 : 2 : 1 : 1
  edit_ntp: edit_image + prompt + 非空 CoT，input_image=None，只有 NTP
  edit:     source edit_image + target image，无 CoT，只有 FM
  edit_umt: source edit_image + target image，prompt 内嵌一个 mask span，无 CoT，只有 FM
  edit_mt:  source edit_image + target image + 非空 CoT，单次 text encoder forward，NTP + FM
  可训：text encoder 的 model.language_model.layers.* LoRA A/B，fp32
  冻结：vision tower、embed/lm_head 原权重、DiT、VAE
  runner：顺序 DataLoader + DDP-aware schedule，避免 shuffle 打散子步类型

Stage 2a（sft:data_process）
  加载 SAMTok TE + VAE，融合 Stage 1 TE LoRA
  按 stage2.jsonl 运行所有 pipeline units
  缓存 prompt_emb、prompt_emb_mask、input_latents、edit_latents 等到 .pth
  need_ntp=False，不缓存 NTP hidden/label
  8 卡按 metadata index 精确分片；每个 .pth 旁写同名 .json provenance/shape/dtype sidecar

Stage 2b（sft:train）
  只加载 Qwen-Image-Edit-2511 DiT
  metadata_path=None，UnifiedDataset 读取 Stage 2a .pth cache
  pipe.units=[]，FlowMatchSFTLoss 直接消费缓存
  只训练 DiT LoRA，纯 FM loss
  smoke debug runner 逐步审计 8 卡取样、FM tensor、梯度、参数更新和卡间一致性
```

Stage 1 的 LoRA 是 text encoder 的 28 层 LLM 部分，必须保留梯度到 `prompt_emb`；Stage 2
把 text encoder 计算移到离线阶段，因此可以只让 DiT 参与在线训练。

---

## 数据格式

### 2.1 metadata 规范

当前数据构建脚本输出 DiffSynth JSONL。路径相对于 `dataset_base_path` 时由数据算子解析；
GRES builder 默认写绝对 `edit_image` 路径，CrispEdit builder 默认写 output root 下的相对路径。

`edit_mt`（带 target image 和 canonical CoT）：

```json
{
  "image": "images/add_00000/000007_target.jpg",
  "edit_image": "images/add_00000/000007_source.jpg",
  "prompt": "Add a green bottle near the cupcakes",
  "sample_type": "edit_mt",
  "mt_cot": "```json\n[{\"mask_2d\": \"<|mt_start|><|mt_0037|><|mt_0368|><|mt_end|>\", \"label\": \"green bottle near the cupcakes\"}]\n```",
  "provenance": {
    "source_parquet": "add_00000.parquet",
    "row_idx": 7,
    "edit_type": "add",
    "qc_flag": "OK"
  }
}
```

`edit_ntp` 没有 `image`，只需要源图、编辑 prompt 和 CoT：

```json
{
  "edit_image": "/mnt/bn/strategy-mllm-train/intern/common_datasets/Sa2VA-Training/osprey-724k/xxx.jpg",
  "prompt": "change the left cat to blue",
  "mt_cot": "```json\n[{\"mask_2d\": \"<|mt_start|><|mt_0001|><|mt_0257|><|mt_end|>\", \"label\": \"left cat\"}]\n```",
  "sample_type": "edit_ntp"
}
```

`edit` 没有 `mt_cot`，保存原始编辑三元组：

```json
{
  "image": "images/add_00000/000008_target.jpg",
  "edit_image": "images/add_00000/000008_source.jpg",
  "prompt": "Add a hat to the man",
  "sample_type": "edit"
}
```

`edit_umt` 同样没有 `mt_cot`，但把原 prompt 中一个可明确定位的指代短语替换为
source mask 的四个原子 token：

```json
{
  "image": "images/remove_00000/000008_target.jpg",
  "edit_image": "images/remove_00000/000008_source.jpg",
  "prompt": "Remove <|mt_start|><|mt_0037|><|mt_0368|><|mt_end|>.",
  "sample_type": "edit_umt",
  "provenance": {
    "source_parquet": "remove_00000.parquet",
    "row_idx": 8,
    "original_prompt": "Remove the red cup.",
    "umt_replaced_text": "the red cup",
    "umt_rewrite_method": "remove_object"
  }
}
```

约束：

- `edit_mt` 和 `edit_ntp` 的 `mt_cot` 必须是 `to_cot` 产生的**非空** canonical strict 形式；
- refined 训练数据彻底禁止空 mask、global/noop 和 `to_cot([])`；style/background 也必须有
  有效 mask span，否则构建阶段丢弃；
- `edit` 和 `edit_umt` 必须完全省略 `mt_cot`；`edit_umt.prompt` 必须恰好包含一个合法
  `<|mt_start|> code0 code1 <|mt_end|>` span；
- `edit_ntp` 省略 `image`，训练入口会令 `input_image=None`，从而不计算 FM；
- `edit_image` 在训练算子中无论输入是字符串还是路径列表，最终都会成为非空 PIL image list；
- 真实 mask 的 SAMTok 编码在源图上进行，避免 pass-1 只看源图而训练 CoT 却来自 target 图。

训练数据由 `compose_training_metadata.py` 生成 Stage 1/Stage 2 JSONL；输出目录不在代码中
硬编码，由 launcher 的 `DATASET_BASE`、`STAGE1_METADATA` 和 `OUTPUT_PATH` 参数指定。

### 2.2 四类样本

**edit_mt / edit_umt** 来自新的带 mask CrispEdit 数据：

1. 从原始 `CrispEdit-2M` parquet 读取 input/source、output/target、instruction、type；
2. 从 `CrispEdit-2M-mask` 按同名 parquet 和 `row_idx` 对齐 mask；
3. 仅保留 `filter_decision == "keep"` 且 `mask_png` 解码非空、`mask_sum > 0` 的行；
4. 对 source/target 图片进行落盘；
5. 非空 mask 由 `SamtokCodec.encode_single_batch` 编码，写入一条 mask span；
6. `edit_mt` 将 span 放入非空 canonical CoT；
7. 对能无歧义定位指代短语的行，以确定性语法模板生成 `edit_umt`，只替换该短语，原始
   instruction、被替换文本和规则名写入 provenance；不可靠的 UMT 改写直接丢弃；
8. 同时生成共享同一 source/target 图片的 `edit_mt.jsonl` 与 `edit_umt.jsonl`。

当前构建器按每个 mask parquet row 处理一张 mask；它不是原方案中尚未实现的“多表达式、多 mask
分组编排器”。后续如要支持多 mask，需要在 `build_edit_mt_metadata.py` 中扩展输入 schema，
并继续通过 `to_cot` 写 canonical 结果。

**edit_ntp** 来自 GRES/SAMTok 发布数据：

- 输入默认是 `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Training_Data/mask_generation_gres209k.json`；
- 图片根目录默认是 `/mnt/bn/strategy-mllm-train/intern/common_datasets/Sa2VA-Training/osprey-724k`；
- 从 release conversation 的 CoT 或 segmentation question 中提取 expression；
- 对已存在的 mask 复用原 CoT 重新 canonicalize，并将 prompt 改写为英文编辑模板；
- 先过滤空 CoT、非法 span 和非 ASCII 文本，再从完整合格池进行固定种子抽样；
- 不再合成 global/noop 行，也不存在 `global_ratio` 或 `GLOBAL_TEMPLATES`。

**edit** 从 `CrispEdit-2M-fact-prefilter/manifest` 读取 filter-only manifest，并与原始
`CrispEdit-2M` 的同名 parquet/`row_idx` 严格 join；只使用 fact prefilter 的 keep 行，保留
原始 source、target 和 instruction，不带 CoT，作为通用编辑/FM 保持项。

三个 CrispEdit 分支 `edit_mt/edit/edit_umt` 允许在原始 source identity 上重合；正式训练量下
不再把它们互斥或最小化重合作为约束。来源重合是明确允许的采样策略，不是数据泄漏。

### 2.3 CoT canonical 格式

当前唯一序列化函数是
`DiffSynth-Studio/diffsynth/core/data/samtok_dataset.py:to_cot`。

训练 metadata 只允许非空 item：

```json
{"mask_2d": "<|mt_start|><|mt_0001|><|mt_0257|><|mt_end|>", "label": "left cat"}
```

约束：

- `mask_2d` key 在前，`label` key 在后；
- item 用 `",\n"` 连接；
- `sanitize_label` 替换引号、反斜杠和反引号，压缩空白，限制 80 字符，空值回退 `target`；
- `make_labels(expr, 1)` 产生原 expression，多个 mask 时产生统一的 `one of the {expr}`；
- `to_cot` 拒绝 codebook 范围外 span；
- codebook 1 为 `[0,255]`，codebook 2 写作 `[256,511]`；
- `samtok_codec.py` 的 malformed-span fixer 只用于 decode/可视化，绝不进入条件路径。

### 2.4 pass-1 输出解析（分层恢复）

`parse_and_canonicalize_mt_cot` 位于 `samtok_dataset.py`，当前顺序为：

1. 字面空表 `[]`；
2. strict JSON list；
3. item 片段恢复；
4. 合法 span 拾取；
5. 仅在没有任何合法 span 时识别 `No target.`；
6. 其他情况返回 `None`，pipeline 对显式非法 CoT 抛出 `ValueError`。

恢复过程只丢弃信息：过滤非法 code、去重、保序、清洗 label；不会猜缺失 code，也不会注入
词表外 `<|mt_9999|>`。解析层记录到 `pipe.last_parse_layer`，供推理和评测统计。
这里的 `empty/no target` 解析只为兼容历史 checkpoint/评测输出；refined builder、composer、
validator 和训练 Dataset 都会拒绝空 CoT，它不会进入新的 Stage 1/Stage 2 训练数据。

---

## 代码组织

```text
/opt/tiger/tanyue/samtok_edit/
├── pyproject.toml                           # Python 3.11 直接依赖的精确版本
├── setup_env.sh                            # uv 环境构建、本地 DiffSynth 和验收
├── DiffSynth-Studio/                         # 普通目录，官方 main@fed7b18f 的 vendored tree
│   └── diffsynth/
│       ├── models/qwen_image_text_encoder_samtok.py
│       ├── utils/state_dict_converters/qwen_image_text_encoder_samtok.py
│       ├── configs/model_configs.py
│       ├── configs/vram_management_module_maps.py
│       ├── core/data/samtok_dataset.py
│       ├── pipelines/qwen_image_samtok.py
│       └── diffusion/loss.py
├── scripts/
│   ├── data/
│   │   ├── prepare_samtok_te_dir.py
│   │   ├── samtok_codec.py
│   │   ├── build_edit_ntp_metadata.py
│   │   ├── build_edit_mt_metadata.py
│   │   ├── build_edit_metadata.py
│   │   ├── build_stage1_refined_full.sh
│   │   ├── build_stage2_refined_full.sh
│   │   ├── audit_refined_metadata.py
│   │   ├── audit_stage1_schedule.py
│   │   ├── sanitize_stage2_validation_content.py
│   │   ├── compose_training_metadata.py
│   │   ├── validate_training_metadata.py
│   │   └── validate_metadata_disjointness.py
│   ├── train/
│   │   ├── train_samtok_edit.py
│   │   ├── stage1_te_lora.sh
│   │   ├── stage2_data_process.sh
│   │   ├── stage2_dit_lora.sh
│   │   ├── audit_stage1_training_log.py
│   │   ├── audit_stage2_training_log.py
│   │   ├── audit_stage2_cache.py
│   │   └── run_stage2_8gpu_pipeline.sh
│   ├── inference/infer_samtok_edit.py
│   ├── inference/validate.py
│   ├── eval/run_eval.py                   # S1–S8 统一评测入口
│   ├── eval/run_stage1_eval_8gpu.sh
│   ├── eval/run_stage2_eval_8gpu.sh
│   ├── eval/run_stage2_eval.py
│   ├── eval/run_scaleedit_refined_eval_8gpu.sh
│   ├── eval/run_stage2_four_node_checkpoint_eval_8gpu.sh
│   ├── eval/analyze_stage1_eval.py
│   ├── eval/analyze_stage1_cot_masks.py
│   ├── eval/analyze_eight_setting_eval.py
│   └── eval/make_stage1_category_comparisons.py
├── tests/test_samtok_edit.py
└── SamtokEdit_训练方案_当前实现.md
```

所有自定义脚本会把 `DiffSynth-Studio` 放入 `sys.path`；shell launcher 额外设置
`PYTHONPATH`。单卡时 launcher 直接运行 `python`，多卡时才调用
`accelerate launch --multi_gpu`。

---

## 逐文件实现

### 4.1 TE wrapper：`DiffSynth-Studio/diffsynth/models/qwen_image_text_encoder_samtok.py`

`QwenImageSamtokTextEncoder` 继承 `Qwen2_5_VLForConditionalGeneration`，保留原生 HF
`forward`/`generate`，避免破坏 KV cache 和视觉输入逻辑。当前实现：

- `_base_config(vocab_size)` 内置 Qwen-Image 7B text/vision architecture config；
- 默认 `vocab_size=152179`，但仍允许通过 `extra_kwargs` 传入；
- `generation_config` 使用 greedy、`use_cache=True`、`eos_token_id=[<|im_end|>]`；
- `encode(...)` 调用 `self.model(..., output_hidden_states=True, use_cache=False)`，返回 hidden states；
- `ntp_logits(hidden_slice)` 只对短 CoT slice 调 `lm_head`；
- 训练时 gradient checkpointing 由 trainer 开启，并显式调用 `text_encoder.train()`。

这比原方案中的“覆盖 forward”更保守：forward 仍是 Transformers 原生实现，SAMTok 专用行为
通过 `encode`、`ntp_logits` 和词表配置实现。

### 4.2 State dict converter

文件：
`DiffSynth-Studio/diffsynth/utils/state_dict_converters/qwen_image_text_encoder_samtok.py`。

`QwenImageSamtokTextEncoderStateDictConverter` 只依赖 key iteration 和下标读取，因此兼容
普通 safetensors mapping 与 DiffSynth `DiskMap`：

- `visual.*` → `model.visual.*`；
- `model.language_model.*`/`model.visual.*` 原样保留；
- 其他 `model.*` 旧布局映射到 `model.language_model.*`；
- 缺失 `lm_head.weight` 时从 `model.language_model.embed_tokens.weight` 补齐。

### 4.3 模型注册（hash 注册制）

`DiffSynth-Studio/diffsynth/configs/model_configs.py` 的 `qwen_image_series` 已注册：

```python
{
    "model_hash": "7792f327a564edcc922f747808b18fb6",
    "model_name": "qwen_image_text_encoder",
    "model_class": "diffsynth.models.qwen_image_text_encoder_samtok.QwenImageSamtokTextEncoder",
    "state_dict_converter": "diffsynth.utils.state_dict_converters.qwen_image_text_encoder_samtok.QwenImageSamtokTextEncoderStateDictConverter",
    "extra_kwargs": {"vocab_size": 152179},
}
```

`vram_management_module_maps.py` 复用官方 `QwenImageTextEncoder` 的 module map 和
version checker，避免新增一套 VRAM wrapper。

hash 是由 `prepare_samtok_te_dir.py` 对 `model*.safetensors` 的 key/shape 计算，当前合并目录
manifest 记录：

```text
processor_class: Qwen2VLProcessor
tokenizer_length: 152179
model_vocab_size: 152179
te_model_hash: 7792f327a564edcc922f747808b18fb6
```

### 4.4 数据模块：`DiffSynth-Studio/diffsynth/core/data/samtok_dataset.py`

一个文件包含两部分：canonical CoT 工具和 Stage 1 调度 Dataset。

#### 4.4.1 上半：canonical 文本工具（R3）

导出常量/函数：

```text
MT_START, MT_END, MT_FMT
CODEBOOK_SIZE=256, CODEBOOK_DEPTH=2
span_of, valid_span_codes, is_valid_span
sanitize_label, make_labels, to_cot
parse_and_canonicalize_mt_cot
```

`to_cot` 是唯一 serializer；`parse_and_canonicalize_mt_cot` 是唯一 pass-1 parser。实现
包含 strict/item/span/empty 分层、去重、合法 code 检查和 canonical round-trip 校验。

#### 4.4.2 下半：精确比例 Dataset

`SamtokEditingDataset` 继承 `UnifiedDataset`。

- `type_ratio` 默认 `edit_mt:4,edit_ntp:2,edit:1,edit_umt:1`；
- `metadata_path` 非空且 ratio 非 `none` 时建立 Stage 1 schedule；
- `metadata_path=None` 时走父类 cache 行为；ratio=`none` 时保留 metadata 原顺序，供 Stage 2a
  精确分片，但仍执行全部 schema/CoT 校验；
- 构造期检查 `edit_mt/edit_ntp` 的 CoT 必须为非空 canonical `strict`，`edit/edit_umt` 不得
  含 CoT，且 UMT prompt 恰有一个合法 span；
- 每个 optimizer step 的 A 个 micro-step 类型由配比块重复并随机排列；
- 每个 micro-step 连续放置 P 个相同类型样本，保证 DDP rank 同型；
- 要求 `gradient_accumulation_steps % ratio_block_size == 0`，不要求卡数整除；
- 每类样本 pool 独立 shuffle，取尽后循环重洗；
- DataLoader 必须 `shuffle=False`，否则破坏 schedule；
- `__len__` 返回 schedule 长度，避免 repeat 被父类再次乘一次。
- `__getitem__` 额外写入仅在运行时使用的 `_samtok_schedule_position` 和
  `_samtok_source_row_id`，用于 debug 模式核对 Accelerate 的 rank 分片，不改变 metadata 文件。

当 `P=1,A=8` 时，schedule contract 要求每个累积窗口包含 4 条 `edit_mt`、2 条 `edit_ntp`、
1 条 `edit`、1 条 `edit_umt`；当 `P=8` 时每个同型 micro-step 连续占据 8 个位置。

### 4.5 Pipeline：`DiffSynth-Studio/diffsynth/pipelines/qwen_image_samtok.py`

该文件继承官方 `qwen_image.py` 的 pipeline/unit，并保留原生 ShapeChecker、NoiseInitializer、
EditImageEmbedder、Inpaint、EntityControl、BlockwiseControlNet 等分支。

#### 4.5.1 共享模板构造

`build_edit_model_inputs(pipe, prompt, edit_image, condition_image_area=384*384)` 是 pass-1/pass-2
唯一入口。它要求非空 `list[PIL.Image]`，按宽高比计算 `/32` 对齐尺寸，构造 `Picture i:` 前缀，
然后调用 `pipe.processor(..., padding=True, return_tensors="pt").to(pipe.device)`。

#### 4.5.2 `QwenImageUnit_SamtokEmbedder`（pass-1）

处理优先级：

1. 显式 `mt_cot`（包括推理脚本传入的 GT CoT）先 canonicalize；
2. `mt_cot=None`、`samtok_online_cot=True` 且有条件图时，调用 text encoder `generate`；
3. 生成文本交给 `parse_and_canonicalize_mt_cot`；
4. 正分支写回 canonical `mt_cot`，负分支强制写 `None`；
5. 记录 `last_mt_cot`、`last_pass1_raw`、`last_parse_layer`。

训练入口显式传 `samtok_online_cot=False`，所以训练不在线生成；推理入口默认开启。

#### 4.5.3 `QwenImageUnit_SamtokPromptEmbedder`（pass-2 + NTP hidden）

这是当前实现中的实际类名，继承官方 `QwenImageUnit_PromptEmbedder`，并通过
`PipelineUnit.__init__` 注册正分支 `prompt/mt_cot` 和 NTP 输出。

- `edit_image` 为 list 时走 `encode_prompt_edit_multi`；
- `mt_cot` 不为空时，独立 tokenizer 后与模板 ids 拼接；
- `pipe.text_encoder.encode` 得到 final-normalized hidden；
- `extract_masked_hidden` 后去掉 `EDIT_DROP_IDX=64`；
- `samtok_need_ntp=True` 时输出 `samtok_cot_hidden` 和 `samtok_cot_labels`；
- `samtok_need_ntp=False` 时不产生 NTP hidden，Stage 2 cache 不会存大词表监督张量；
- prompt embedding pad/stack 逻辑保持官方 Qwen-Image 结构。

NTP 切片由 `shifted_cot_supervision(hidden, cot_ids, template_length)` 统一生成。该函数检查
batch/sequence shape 和边界，严格取 `hidden[:, L_T-1:L_T-1+L_C]`；运行时还会确认
hidden 数量与 label 数量相同、最后一个 label 为 `<|im_end|>`，并记录完整边界信息。

#### 4.5.4 `QwenImageSamtokPipeline.__call__` 与 `from_pretrained`

`from_pretrained` 仍通过 DiffSynth ModelPool 按 hash 取得：

```text
qwen_image_text_encoder
qwen_image_dit
qwen_image_vae
```

随后从 `tokenizer_config` 载入 `Qwen2Tokenizer`，从 `processor_config` 载入
`Qwen2VLProcessor`，检查 `<|mt_start|>` 存在，并记录 `<|im_end|>` id。

`__call__` 在进入父类 denoise 前设置内部 flags，退出时清理；推理侧调用方式为：

```python
pipe(
    prompt,
    edit_image=[image],
    edit_image_auto_resize=True,
    zero_cond_t=True,
    mt_cot=None,
    enable_samtok_cot=True,
)
```

当前 `DiffSynth-Studio` 已同步官方 `Qwen-Image` KV-cache 修复
`db5b335a`，`QwenImageTransformerBlock.forward` 正式接收并向 attention 转发 `kv_cache`；
仓库不再保留旧的本地临时补丁。

#### 4.5.5 units 顺序

当前顺序为：

```text
ShapeChecker
NoiseInitializer
InputImageEmbedder
Inpaint
EditImageEmbedder
LayerInputImageEmbedder
ContextImageEmbedder
SamtokEmbedder
SamtokPromptEmbedder
SamtokEntityControl
BlockwiseControlNet
```

`SamtokEmbedder` 在 `EditImageEmbedder` 之后，确保 pass-1 和 pass-2 使用相同的 auto-resize
条件图。

### 4.6 Loss：`DiffSynth-Studio/diffsynth/diffusion/loss.py`

新增 `SamtokNTPLoss` 和 `SamtokEditingLoss`。

```text
edit_ntp -> ntp_weight * SamtokNTPLoss
edit     -> fm_weight  * FlowMatchSFTLoss
edit_umt -> fm_weight  * FlowMatchSFTLoss
edit_mt  -> ntp_weight * SamtokNTPLoss + fm_weight * FlowMatchSFTLoss
```

`SamtokNTPLoss` 只对 `samtok_cot_hidden` 过 `pipe.text_encoder.ntp_logits`，然后对 labels
做 cross entropy。`FlowMatchSFTLoss` 沿用上游 flow-matching 计算，使用同一份
`input_latents` 加噪并调用 `pipe.model_fn`；DiT 冻结但梯度仍经 `prompt_emb` 回传 text
encoder LoRA。当前实现额外在 fp32 计算 MSE，并记录 timestep、training weight、latent/noise/
prediction/target 的 shape 与 dtype，供 Stage 2 smoke 强校验。

`pipe.last_loss_log` 保存未加权的 `loss_ntp`/`loss_fm` 分量，`last_loss_debug` 记录
loss dtype 和实际权重；Stage 1 logger 将分量、total loss 和标量 debug metric 写入
CSV/W&B/SwanLab（SAMTok 训练默认启用 W&B；其他 logger 按参数启用）。训练循环同时校验
`loss_total = ntp_weight * loss_ntp + fm_weight * loss_fm`。

### 4.7 训练入口：`scripts/train/train_samtok_edit.py`

这是 Stage 1、Stage 2a、Stage 2b 的统一入口，使用上游 parser/runner，并新增：

- `SamtokEditingDataset` 和 `edit_image` 类型路由；
- `sample_type` → 输入图、是否需要 NTP、loss 的分派；
- `QwenImageSamtokTrainingModule` 的分片路径兼容层；
- text encoder LoRA 的 `lora_dropout=0.05` 和 fp32 cast；
- TE gradient checkpointing 后显式 `train()`；
- `sft:data_process`、`sft`、`sft:train` 三种 task；
- 顺序 Stage 1 runner：`shuffle=False`、同步步裁剪、按 optimizer step 手动 scheduler；
- Stage 2a runner：验证 8 卡 metadata index/类型分片、cache tuple 必需字段和 tensor
  finiteness，并为每份 cache 写 provenance sidecar；
- Stage 2b debug runner：保持官方 `shuffle=True + AdamW + ConstantLR` 训练路径，同时按
  cache sidecar 审计每个 DDP step；非 debug 模式仍使用 DiffSynth 官方训练 runner；
- 训练 seed 使用 `device_specific=True`：metadata schedule 在所有 rank 相同，而 timestep/noise
  RNG 按 rank 分开；
- `--debug_train_metrics` 和 `--debug_log_steps`。

新版 DiffSynth 的 `parse_model_configs` 会把每个 `model_paths` 项直接用于 quantization map
查找。分片路径是 `list[str]`，不可 hash。当前 trainer 只在遇到嵌套 list 时做路径 key
序列化和 `ModelConfig(path=list)` 构造；普通字符串路径仍调用官方 parser，`quant_options`
也会继续透传。

Stage 1 debug 模式的审计内容：

- 可训练 tensor/参数数量和 dtype；
- 只有 `pipe.text_encoder.*.lora_A/B` 可训练；
- 每个 micro-step gather 8 个 rank 的 sample type、schedule position、source row 和 loss；
- 卡间类型同型、schedule 连续分片和各 rank loss finite 的强制校验；
- NTP/FM 分派、加权 loss 恒等式、NTP shift 边界和 `<|im_end|>` label 校验；
- UMT prompt span 在 processor 模板中必须完整保留为四个原子 token，并且 UMT 只走 FM；
- FM 的 timestep、training weight、fp32 MSE 和 latent/noise/target shape 会进入 debug 记录；
- `input_latents` 只来自 metadata `image` 目标图，`edit_latents` 来自 `edit_image` 条件图；
- CoT hidden/label、prompt embedding、input/edit latent 的 shape 和 dtype；
- 每个 rank 的梯度有限性、非零梯度张量、冻结梯度张量；
- accumulation slot、同步步、optimizer step、learning rate、clip return norm；
- 首个 LoRA B 张量的实际 update L2 norm，以及同步步后所有 rank 的参数一致性；
- 每个累积窗口第一个 micro-step 的梯度未混入旧梯度，可作为 NTP/FM 加权尺度的调参观测。

`scripts/train/audit_stage1_training_log.py` 会解析全部 `[SamtokDebug]` JSON，严格检查四类比例、
loss 路由与恒等式、GT source/target、CoT/UMT token 位置、bf16/fp32、梯度累积、冻结参数梯度、
每次 optimizer update 和 DDP 参数同步，并记录 checkpoint 大小/SHA256 与分类型 loss 统计。

Stage 2 debug 模式的审计内容：

- 只有 `pipe.dit.*.lora_A/B` 可训练，text encoder/VAE 可训练参数为 0；
- DiT LoRA tensor 为 bf16，并统计 12 组官方 target module family 的命中数量；
- 每个 epoch 的物理 cache 在 `dataset_repeat` 后恰好按配置次数消费；新 Stage 2 的类型比例为
  `edit_mt:edit:edit_umt=2:1:1`；
- 每步 gather 8 卡的 sample type、metadata index、FM loss、timestep、梯度范数、probe
  update 和参数范数；
- FM loss 必须为 finite fp32，`input_latents/noise_pred/training_target` shape 必须一致；
- 每卡可训练梯度必须 finite 且非零，冻结参数不得出现梯度；
- optimizer step 后 LoRA probe update 必须非零，8 卡参数范数必须保持一致；
- 官方全层 target list 会覆盖最后一个 block 的 text-only 输出分支；由于最终 DiT 只返回
  image stream，这些末层参数可能无梯度，必须结合 `find_unused_parameters=True` 和最终
  checkpoint 零 tensor 名单解释，不能误判为整个 LoRA 未更新；
- runtime audit 明确记录 optimizer、betas、weight decay、scheduler、有效 batch、bf16、
  gradient checkpointing、`zero_cond_t`、`find_unused_parameters` 和是否梯度裁剪。

### 4.8 训练 shell 与超参

三个 launcher 的默认模型路径均已写成用户当前实际路径，也允许环境变量覆盖：

```text
QWEN_2511=/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen-Image-Edit-2511
SAMTOK_TE=/mnt/bn/strategy-mllm-train/user/tanyue/models/SAMTok/Qwen2.5-VL-7B-SAMTok-gres-ft
```

#### `stage1_te_lora.sh` —— TE LoRA，NTP+FM 双损失

文件：`scripts/train/stage1_te_lora.sh`。

Stage 1 和 Stage 2b 标准训练默认启用 WandB，并且在模型加载前强制检查以下环境变量：

```bash
export WANDB_API_KEY=<your-api-key>
export WANDB_ENTITY=<your-user-or-team>
export WANDB_PROJECT=<your-project>
```

`WANDB_API_KEY` 只从环境读取，不会写入 `training_args.json`、CSV 或普通训练日志；
`WANDB_ENTITY` 会传给 `wandb.init(entity=...)`。可选的 `WANDB_RUN_NAME` 会作为 run name。
缺少任一必需变量时 launcher 在加载模型前以退出码 2 终止。WandB 本地文件写入
`$OUTPUT_PATH/wandb_log/`，曲线同时由 WandB SDK 同步到指定 project/entity。

只有明确的离线调试才设置：

```bash
ENABLE_WANDB_LOG=0 bash scripts/train/stage1_te_lora.sh
```

Python 直启时对应使用 `--disable_wandb_log`；默认仍为开启。不要把 API key 写入 shell
脚本、仓库文件或命令行参数。

默认关键参数：

```text
sample_type_ratio=edit_mt:4,edit_ntp:2,edit:1,edit_umt:1
lora_base_model=text_encoder
lora_rank=64
lora_dropout=0.05
learning_rate=4e-5
weight_decay=0.05
warmup_ratio=0.05
max_grad_norm=1.0
gradient_accumulation_steps=8
ntp_loss_weight=0.05
fm_loss_weight=1.0
zero_cond_t=True
find_unused_parameters=False
```

训练入口调用示例（路径由环境变量指定）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NUM_PROCESSES=8 MAIN_PROCESS_PORT=50673 DATASET_WORKERS=8 \
DATASET_BASE=/path/to/dataset_base \
STAGE1_METADATA=/path/to/dataset_base/stage1.jsonl \
MERGED_TE_DIR=/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/artifacts/merged_samtok_te \
OUTPUT_PATH=/path/to/experiment/stage1_te_lora \
MAX_PIXELS=1048576 GRADIENT_ACCUMULATION_STEPS=8 NUM_EPOCHS=1 \
NTP_LOSS_WEIGHT=0.05 FM_LOSS_WEIGHT=1.0 \
WANDB_API_KEY=<your-api-key> WANDB_ENTITY=<your-user-or-team> WANDB_PROJECT=<your-project> \
bash scripts/train/stage1_te_lora.sh
```

多卡时将 `NUM_PROCESSES` 设为大于 1，launcher 才会调用 Accelerate；
`MAIN_PROCESS_PORT` 可避免同机任务的 rendezvous 端口冲突。`gradient_accumulation_steps` 需要是 8 的
倍数，以保持每个窗口的 4:2:1:1 类型结构。NTP 和 FM 都经过全部 TE LoRA，因此
`find_unused_parameters` 默认关闭；仅在修改可训练图后确有 unused parameter 时设
`FIND_UNUSED_PARAMETERS=1`。其他可覆盖项包括 `SAMPLE_TYPE_RATIO`、`LEARNING_RATE`、
`WEIGHT_DECAY`、`WARMUP_RATIO`、`MAX_GRAD_NORM`、`LORA_RANK`、`LORA_DROPOUT`、
`NTP_LOSS_WEIGHT`、`FM_LOSS_WEIGHT` 和 `SEED`。

#### `stage2_data_process.sh` —— 融合 TE LoRA，缓存 prompt/latent

运行前必须设置：

```bash
export TE_LORA_PATH=/path/to/stage1_te_lora.safetensors
```

该脚本加载 SAMTok TE 和 Qwen VAE，processor/tokenizer 指向合并目录，
`--preset_lora_path "$TE_LORA_PATH" --preset_lora_model text_encoder`，任务为
`sft:data_process`；该缓存步骤不初始化训练 logger，shell 会显式传入
`--disable_wandb_log`。数据 `stage2.jsonl` 由 `compose_training_metadata.py` 生成，包含
`edit_mt + edit + edit_umt`，精确比例为 2:1:1。

当前 Stage 2a runner 要求 metadata 行数能被 world size 整除，逐批检查 rank 收到的
metadata index 是否连续覆盖且类型总数不变。每个 `<rank>/<local_id>.pth` 旁写
`<rank>/<local_id>.json`，保存 metadata index、sample type、prompt、源数据 provenance、
图片尺寸、实际融合的 TE LoRA 路径，以及 cache tensor 的 key/shape/dtype/finiteness。
可使用 `scripts/train/audit_stage2_cache.py` 再次加载全部 `.pth` 并生成结构化验收报告。
正式审计默认使用 32 个 process，每个 sidecar/cache pair 在同一 task 中验证；`.pth`
只从存储读取一次，同一份 bytes 同时用于单文件 SHA256、`torch.load`、tensor 结构、
bf16 与 finiteness 检查。各文件 hash 再按相对路径有序合成 manifest SHA256，因此不需要
第二次读取全部 cache。日志定期 flush `processed/percent/rate/read_gib/elapsed/eta/errors`，
完整报告以 atomic replace 写入 `stage2_cache_audit.json`。

默认输出为 `$REPO_ROOT/models/stage2_cache`；实际运行时建议将 `OUTPUT_PATH` 指向
`/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/<run>/stage2_cache`。

#### `stage2_dit_lora.sh` —— 只训 DiT LoRA

该脚本只加载 Qwen-Image-Edit-2511 transformer shards，读取 Stage 2a `.pth` cache：

```text
task=sft:train
sample_type_ratio=none
lora_base_model=dit
lora_rank=32
learning_rate=1e-4
num_epochs=1
dataset_repeat=2
gradient_accumulation_steps=1
weight_decay=0.01
zero_cond_t=True
use_gradient_checkpointing=True
find_unused_parameters=True
```

目标模块为官方 2511 edit LoRA 配方：

```text
to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,
to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1
```

Stage 2b 也默认启用 W&B，并使用与 Stage 1 相同的账户环境变量门禁；CSV 始终同步写入
`$OUTPUT_PATH/loss.csv`。设置 `DEBUG_TRAIN_METRICS=1` 后启用上述 Stage 2 强审计 runner。
32 行 8 卡 smoke 在 `dataset_repeat=2,num_epochs=5` 时，每卡每 epoch 8 个 micro-step，
全局有效 batch 为 8，总计 40 个 optimizer step。
`audit_stage2_training_log.py` 对 debug 日志、CSV 和最终 safetensors 做离线一致性验收：检查
三类数据的逐 rank/逐 epoch 消费、FM target/预测 shape 与 dtype、fp32 MSE、有限梯度、
LoRA probe 每步更新和八卡同步、DiT-only trainable graph、学习率、checkpoint schema 及 W&B
正常结束。它还显式识别 Qwen flow-matching 在 `timestep=1000` 时 training weight 为零的
合法端点样本；这类单 rank loss 为零不会被误报成数值故障。
该 `num_epochs=5` 仅用于 smoke；当前正式 Stage 2b 默认为 `num_epochs=1`，实际 optimizer
step 数由新 2:1:1 metadata 行数、`dataset_repeat`、world size 与 gradient accumulation 共同决定。

`scripts/train/run_stage2_8gpu_pipeline.sh` 用于正式运行的可复现串行编排：先执行
Stage 2a，再由 `audit_stage2_cache.py` 全量反序列化 cache，检查数量、来源、必需 tensor、
bf16 精度与 finiteness，只有审计通过才启动 Stage 2b。编排脚本要求显式传入数据、
Stage 1 TE LoRA、合并 TE 目录和权限为 `600` 的 W&B env 文件；可选 SHA256 门禁在加载
模型前拒绝错误 metadata 或 TE checkpoint。各阶段使用独立日志，编排层只记录阶段状态；
脚本本身不包含 W&B API key。`START_PHASE=cache|audit|train` 支持在已验收的阶段边界恢复；
`audit` 和 `train` 恢复都要求已有 cache，进入训练前还会强制检查结构化审计报告
`passed=true`。审计并行度可由 `CACHE_AUDIT_WORKERS`、`CACHE_AUDIT_TORCH_THREADS`、
`CACHE_AUDIT_CHUNKSIZE` 和 `CACHE_AUDIT_LOG_EVERY` 覆盖。缓存审计默认用 32 个 process，
每个 process 固定 1 个 Torch CPU thread；每个 `.pth` 只读一次，在同一 worker 内完成原始文件
SHA256、反序列化、结构、dtype 和 finiteness 检查，并按固定相对路径顺序合成 manifest SHA256。
审计日志按默认每 500 条实时 flush `processed/percent/rate/read_gib/elapsed/eta/errors`。
分布式 rendezvous 默认端口使用系统临时端口范围之外的 `20051/20052`；预检会分别尝试绑定
IPv4 与 IPv6 wildcard，避免 IPv6 出站连接占用端口但只检查 IPv4 loopback 时产生的漏检。
正式 cache 加载时，当前 DiffSynth `UnifiedDataset` 使用迭代式 `os.scandir` 发现 `.pth`，
避免旧 `os.listdir + os.path.isdir` 对每个 `.pth`/sidecar 发起额外远端 stat；global rank 0
每跨过约 25,000 条向训练日志 flush 一条 `cache_discovery found=...` 进度。

#### 超参数和资源

Stage 1 的 TE LoRA 使用 fp32 参数、bf16 基座/激活、AdamW `(0.9,0.999)`、weight decay
`0.05`、dropout `0.05`、5% warmup 后 cosine 到 0、同步步梯度裁剪 `1.0`。Stage 2 沿用
官方 DiT LoRA：bf16 rank-32 LoRA、AdamW `(0.9,0.999)`、weight decay `0.01`、学习率
`1e-4`，以及 PyTorch `ConstantLR` 默认的 `factor=1/3,total_iters=5`；官方 runner 不做
额外梯度裁剪。需要注意，官方 runner 会把 scheduler 传给 `accelerator.prepare`；当
`split_batches=False` 时，`AcceleratedScheduler.step()` 每个 optimizer step 内部推进
`num_processes` 次。因而 8 卡实测只在第一个 global optimizer step 使用 `lr/3`，第二步
起即回到 `1e-4`，并不是连续 5 个 global step 的低因子。NTP cross entropy 和 FM MSE
都在 fp32 计算；Accelerate 本身不再额外开 autocast，pipeline 显式以 bf16 运行基座和
Stage 2 LoRA。生产训练可用多卡 DDP，单卡 launcher 适合调试。

### 4.9 数据构建脚本

输入数据：

```text
/mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M
/mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-fact-prefilter/manifest
/mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-mask
/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Training_Data/mask_generation_gres209k.json
/mnt/bn/strategy-mllm-train/intern/common_datasets/Sa2VA-Training/osprey-724k
```

#### 4.9.1 `prepare_samtok_te_dir.py` —— 合并 tokenizer/processor 和生成 manifest

```bash
python scripts/data/prepare_samtok_te_dir.py \
  --samtok_dir /mnt/bn/strategy-mllm-train/user/tanyue/models/SAMTok/Qwen2.5-VL-7B-SAMTok-gres-ft \
  --qwen_2511_dir /mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen-Image-Edit-2511 \
  --output_dir /mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/artifacts/merged_samtok_te
```

脚本从 SAMTok 复制 tokenizer/config，从 Qwen-Image-Edit-2511 复制
`preprocessor_config.json`，用 `Qwen2VLProcessor.from_pretrained` 校验四个边界 token，
校验 tokenizer length 和 config vocab size，并写 `samtok_edit_manifest.json`。

#### 4.9.2 `samtok_codec.py` —— VQ-SAM2 封装

当前实现依赖已安装的 `samtok.models`，而不是把模型源码复制进仓库；默认 codec checkpoint
来自 `Qwen2.5-VL-7B-SAMTok-gres-ft` 目录下的：

```text
sam2.1_hiera_large.pt
mask_tokenizer_256x2.pth
```

`SamtokCodec`：

- 强制 released VQ-SAM2 在 fp32 工作，避免 SAM prompt encoder 的 dtype mismatch；
- 检查 mask 非空、尺寸与源图一致；
- 通过 bounding box 按 left-to-right/top-to-bottom 排序；
- `encode` 支持一图多 mask，`encode_single_batch` 支持多图单 mask；
- `decode` 仅用于可视化/评测，malformed fixer 不进入条件路径。

#### 4.9.3 `build_edit_ntp_metadata.py` —— GRES → edit_ntp

默认命令：

```bash
python scripts/data/build_edit_ntp_metadata.py \
  --input_json /mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Training_Data/mask_generation_gres209k.json \
  --image_root /mnt/bn/strategy-mllm-train/intern/common_datasets/Sa2VA-Training/osprey-724k \
  --output_jsonl /mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/data/crispedit_samtok/edit_ntp_gres.jsonl \
  --sample_rows 21158 --ascii_only --seed 0 --check_images
```

主循环读取 conversation，恢复 release CoT，提取 expression，随机选择英文
`EDIT_VERB_TEMPLATES`；空/非法 CoT 在抽样前丢弃，不创建任何 global/noop 样本。默认路径写
绝对图片路径；传 `--relative_image_paths` 可保持相对 GRES path。

#### 4.9.4 CrispEdit 三类 metadata

`build_edit_mt_metadata.py` 负责带 mask 的 CrispEdit parquet，并同时产出：

```text
edit_mt.jsonl  # image + edit_image + prompt + mt_cot
edit_umt.jsonl # image + edit_image + prompt-with-mask-span，不含 mt_cot
```

构建器调用示例：

```bash
python scripts/data/build_edit_mt_metadata.py \
  --crispedit_dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M \
  --mask_dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-mask \
  --output_root /path/to/experiment/data/crispedit_samtok \
  --sample_rows 20000 --seed 0 --ascii_only \
  --device cuda --dtype float32 --codec_batch_size 32
```

`--all_eligible` 用于全量正式构建：它会先对全部 paired parquet 执行与全局抽样相同的
keep、instruction join、phrases、canonical type、ASCII 和 source exclusion 预检，再把全部
合格行交给 codec。`--ascii_only` 或 `--exclude_metadata_jsonl` 必须与 `--sample_rows` 或
`--all_eligible` 一起使用，避免参数在旧的 prefix 模式中被静默忽略。全量 parquet 内可能
连续出现超过 64 个非空 mask；当前 released SAM2 在 H100 上使用 batch 64 可能触发 SDPA
kernel configuration error，因此正式全量构建使用 `--codec_batch_size 32`。

`build_edit_metadata.py` 用于不需要 mask 的纯 edit 数据。它把
`CrispEdit-2M-fact-prefilter/manifest` 与原始图片 parquet 做严格长度、row_idx、instruction
对齐，只保留 manifest keep 行；并支持与 mask builder 一致的
全局 `--sample_rows/--seed/--ascii_only`、hard `--exclude_metadata_jsonl`、多 worker 原子
shard、`--resume` 和 `--combine_only`。每行写入原始 parquet/row/type provenance。
`--image_subdir` 可把图片写入 output root 下的独立相对目录，默认为 `images`；正式 Stage 2
使用 `images_edit`，从而可以只读复用已经通过 codec 构建的 mask 图片池，且不会向该池写入
纯 edit 文件。
`--deprioritize_metadata_jsonl` 用于数据池不足以完全互斥时最小化来源重合：先使用所有不在
deprioritized metadata 中的合格行，仅从重合池随机补足必要差额。refined 正式方案允许三类
CrispEdit 数据重合，因此不传 exclusion/deprioritization 参数。两个脚本都将图像 bytes 原子落盘为
`images/<shard>/...`，避免只生成 JSONL 而缺图。

放大构建时不要使用按文件名/行号截断的 `--max_rows`：文件名按 edit type 排序，会造成类型
偏置。当前 `build_edit_mt_metadata.py` 支持 `--sample_rows N --seed S`，先在全部 keep rows
中建立候选集再全局随机抽样；`--ascii_only` 会排除 prompt 或非空 CoT label 含非 ASCII
字符的行。全量构建使用 `--all_eligible`，而不是省略 sampling 参数后进入 legacy prefix
模式。GRES builder 同样支持固定种子的 `--sample_rows` 和 `--ascii_only`。

构建训练外验证集时，可重复传入 `--exclude_metadata_jsonl`。构建器优先读取
`provenance.source_parquet + row_idx`；纯 `edit` 行没有 provenance 时，从
`images/<parquet-stem>/<row>_source.*` 恢复 source identity。原始 parquet 名中的空格与图片
目录的下划线会先 canonicalize，再在全局随机抽样前排除。启用 `--sample_rows` 时，codec
阶段只读取实际命中的 parquet，不再为小样本扫描全部含图 raw shard。

大数据构建还支持多 GPU parquet 分区：随机抽样时每个进程使用相同的全局抽样集合；
`--all_eligible` 时每个 worker 只扫描自己的确定性 parquet 分区，避免 N 卡重复扫描全量。
传
`--num_workers 8 --worker_index 0..7 --skip_combine --resume` 后只写互不重叠的原子 shard；
所有进程完成后用 `--combine_only` 检查全部预期 shard 并合并，不再次加载 codec。
多进程 codec 构建应同时设置
`OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8`；否则每个
PyTorch/SAM2 进程可能各自创建约 250 个 host thread，造成严重 CPU oversubscription 和
吞吐下降。该限制只影响 host 并行度，不改变 codec 数值。

旧数据版本为验证集做内容隔离时使用过以下工具；它不用于 refined 三个训练分支之间的互斥。
source identity 互斥不足以排除 CrispEdit 内部“不同 parquet row 复用同一原图”的情况。
`sanitize_stage2_validation_content.py` 在 Stage 2 source pools compose 前执行内容级净化：先对
验证集 source/target 建立 SHA256 集合，只对训练池中 size-compatible 的唯一路径做 hash，
删除任何 source/target 内容命中的行；随后保留全部安全 `edit_mt`，并按
`ceil(edit_mt/2)` 选择纯 `edit`。纯 edit 始终先使用不在安全 `edit_mt` identity 集中的行，
再以固定 seed 从重合池补足，因此训练分支 source overlap 达到理论最小；脚本原子输出两个
`*_train.jsonl` 和包含排除/重合统计的 JSON 报告。

`compose_training_metadata.py` 将四类输入校验、抽样、打乱并生成 Stage 1/Stage 2 JSONL：

```bash
python scripts/data/compose_training_metadata.py \
  --edit_mt_jsonl .../edit_mt.jsonl \
  --edit_ntp_jsonl .../edit_ntp_gres.jsonl \
  --edit_jsonl .../edit.jsonl \
  --edit_umt_jsonl .../edit_umt.jsonl \
  --stage1_output .../stage1.jsonl \
  --stage2_output .../stage2.jsonl \
  --max_edit_mt 16 --max_edit_ntp 8 --max_edit 4 --max_edit_umt 4 --seed 0
```

它只负责 metadata 级别的比例和随机化；Stage 1 运行时的精确 4:2:1:1 由
`SamtokEditingDataset` schedule 再次保证。

`--stage1_output` 和 `--stage2_output` 现在可独立选择；仅构建 Stage 2 时不需要
`--edit_ntp_jsonl`。Stage 2 metadata 包含 `edit_mt + edit + edit_umt`，比例 2:1:1，并
打乱。针对多卡 data-process，可传 `--stage2_num_shards P`；构建器会先将每种类型
均分到 P 个 shard，再按 position-major 顺序写文件，使 Accelerate 的
`rows[rank::P]` 分片在每个 rank 都保持相同类型比例。各类型数量必须可被 P
整除；若全量 source pool 因奇数或过滤后只能接近 2:1，可额外传
`--pad_stage2_to_shards`。构建器保留全部输入行，求满足最终
`edit_mt=2*edit=2*edit_umt` 且三类均可被
P 整除的最小不小于输入的计数，再按固定 seed 复制缺少的行；复制行写入
`schedule_padding` 及原因/ordinal，避免 DDP sampler 隐式 padding。报告会同时给出未
padding 的 source counts、最终 counts 和 padding counts。Stage 1 的最大池总数不是 4 的
倍数时可用 `--pad_stage1_to_ratio` 做最小显式 padding；再传 `--stage1_num_processes 8`
会把比例 block 对齐到 8 卡 optimizer step，避免 Dataset 在 epoch 尾部隐式回卷。例如全部
42,313 个合格 `edit_mt` 会保留，四类 source pool 为 42,313:21,158:10,579:10,579，最终
显式 padding 23/10/5/5 行得到 42,336:21,168:10,584:10,584。Stage 2 的 8 卡 smoke
可单独构建 16:8:8 的 32 行：

```bash
python scripts/data/compose_training_metadata.py \
  --edit_mt_jsonl .../edit_mt.jsonl \
  --edit_jsonl .../edit.jsonl \
  --edit_umt_jsonl .../edit_umt.jsonl \
  --stage2_output .../stage2.jsonl \
  --max_edit_mt 16 --max_edit 8 --max_edit_umt 8 --stage2_num_shards 8 --seed 8
```

#### 4.9.5 `validate_training_metadata.py` —— 构建产物验收

验收器不改变数据，只检查 JSONL schema、sample type、canonical CoT、文本字符集、图片路径
和随机图片解码；支持 `--expected_counts`、`--check_paths`、`--decode_image_sample`、
`--report_json`，路径检查和图片解码使用有界线程池。

`audit_refined_metadata.py` 进一步逐条回查原始 CrispEdit、mask、fact manifest 和 GRES：
校验 prompt/CoT/UMT rewrite provenance、允许且统计三类 CrispEdit identity 重合、检查显式
schedule padding、可对全部落盘图片做原始 bytes 等值比较，并可随机重新运行 SAMTok codec
确认 mask token span 没有改变。`--stage stage2 --world_size 8` 还会严格检查 2:1:1 比例、
Stage 2 padding 原因，以及 `rows[rank::8]` 的逐卡同比例；不含 GRES 的 Stage 2 数据不会加载
GRES 大 JSON。源 parquet/图片核验默认按 8 个 shard 并发执行，并定期把
完成 shard 数写入日志；GRES 图片存在性检查由 builder 使用 32 个 I/O 线程并行完成，绝对
路径构造不触发重复远端 `resolve`。`build_stage1_refined_full.sh` 把 8 GPU codec 构建、纯 edit
的 8-worker 原子分片、NTP 构建、compose、通用验证和上述强审计串成单一失败即停的入口。
`audit_stage1_schedule.py` 还会在
不加载图片/模型的情况下验证每个 8 卡 accumulation window 的 32:16:8:8 消费、rank 同型、
所有 metadata 行恰好使用一次且不存在隐式 pool recycling。

refined Stage 1 全量入口：

```bash
cd /opt/tiger/tanyue/samtok_edit
RUN_ROOT=/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/crispedit_refined/stage1_full \
  bash scripts/data/build_stage1_refined_full.sh
```

默认 8 个 GPU worker、codec batch 32，并将每个 worker 的 OMP/MKL/OpenBLAS/NumExpr 线程限制为
8，避免 96 核机器被数百线程/进程过度订阅。脚本支持对 atomic shard 直接 `--resume`。

refined Stage 2 全量入口：

```bash
cd /opt/tiger/tanyue/samtok_edit
RUN_ROOT=/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/crispedit_refined/stage2_full \
  bash scripts/data/build_stage2_refined_full.sh
```

该入口保留全部 42,313 条合格 `edit_mt`，固定种子抽取 21,157 条纯 `edit` 和 21,157 条
`edit_umt`，再为 2:1:1 与八卡 strided shard 整除做最小显式 padding，最终为
42,320:21,160:21,160，共 84,640 行。完整 mask metadata 和图片复用已经由真实 builder/codec
生成的 refined Stage 1 数据池，入口先做固定 SHA256 门禁；mask 图片目录只读链接到该池，
纯 edit 图片独立写入 `images_edit/`。最终仍会对 Stage 2 实际选中的全部 CrispEdit identity
重新执行 source prompt/type/mask/manifest、落盘图片 bytes 等值检查，并抽样重跑 128 条 codec，
因此复用不跳过内容验收。该脚本只构建和审计数据，不启动 cache 或训练。

它适用于 smoke、正式训练数据以及后续重新构建的 metadata；实验命令和报告位置记录在
`SamtokEdit_实验记录.md`。

`validate_metadata_disjointness.py` 专门检查训练/验证 split：按 source identity 分别比较
`edit_mt` 与纯 `edit`，比较 source/target 相对引用，并对验证集全部图片和训练集所有
size-compatible 图片计算 SHA256。报告同时检查验证集内部 identity/内容重复；任一交集非零
都会写出失败报告并返回非零退出码。

### 4.10 推理脚本：`scripts/inference/infer_samtok_edit.py`

默认路径已经指向 2511 和 gres-ft。典型命令：

```bash
python scripts/inference/infer_samtok_edit.py \
  --prompt "change the left cat to blue" \
  --image_path /path/to/source.png \
  --save_path /mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/inference/out.png \
  --merged_te_dir /mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/artifacts/merged_samtok_te \
  --te_lora /path/to/stage1_te_lora.safetensors \
  --dit_lora /path/to/stage2_dit_lora.safetensors \
  --num_inference_steps 40 --cfg_scale 4.0
```

脚本自动加载 DiT shards、SAMTok TE shards 和 VAE，默认 `edit_image=[image]`、
`edit_image_auto_resize=True`、`zero_cond_t=True`、在线 CoT 开启。stdout 会打印
`mt_cot`、解析层和 raw pass-1 文本；若同时提供 `--sam2_ckpt` 与
`--mask_tokenizer_ckpt`，还会写 `_pass1_mask` 可视化。

`scripts/inference/validate.py` 在同一 pipeline 上读取 JSONL，支持 `--use_gt_cot` 和
`--disable_cot` 两个消融开关，并写 `results.json`。

### 4.11 统一评测入口：`scripts/eval/run_eval.py`

该脚本现在是 S1–S8 的统一评测入口；本节先列出已完成的 Stage 1 五组图像编辑对照。
S1–S5 共用 stock Qwen-Image-Edit-2511 DiT/VAE，不加载 DiT LoRA：

1. `s1_qwen2511_stock`：2511 原始 TE、processor 和官方 DiffSynth
   `QwenImagePipeline`，直接编辑；
2. `s2_samtok_initial_direct`：gres-ft 初始 SAMTok TE（不加载 Stage 1
   LoRA，等价于 LoRA 初始状态），直接编辑，不生成 CoT；
3. `s3_stage1_te_direct`：gres-ft + Stage 1 TE LoRA，直接编辑，不生成
   CoT；
4. `s4_stage1_te_online_cot`：同一 Stage 1 TE 先 greedy 自回归生成 CoT，
   canonical parser 处理后再用 template + CoT 编码出图；
5. `s5_stage1_te_gt_cot`：不做自回归生成，直接把验证行的 `mt_cot`
   追加到 template 后编码出图。

第 2/3 组均显式传 `enable_samtok_cot=False, mt_cot=None`，因此只做一次 TE
forward，不是 two-pass。第 4 组是完整方法推理，第 5 组是 GT-CoT oracle。
五组均使用同一 `seed = base_seed + metadata_index`、bf16、40 steps、CFG 4.0、
`edit_image=[source]`、`edit_image_auto_resize=True` 和 `zero_cond_t=True`；高宽从 source
取得，pipeline 按 16 的倍数向上对齐。第 4 组 greedy 生成的默认上限为
`samtok_max_new_tokens=128`，可由命令行显式调整并写入 run config。

脚本默认路径已绑定当前 64 条验证集、2511、gres-ft、merged processor 和
Stage 1 `step-5000.safetensors`。先做不加载模型的完整预检：

```bash
cd /opt/tiger/tanyue/samtok_edit
python scripts/eval/run_eval.py --dry_run
```

真正出图时直接运行：

```bash
python scripts/eval/run_eval.py
```

可用 `--settings 1 3 4`、`--start_index`、`--max_samples` 选子集；中断后使用完全
相同参数加 `--resume`。非空输出目录默认拒绝覆盖，resume 时也会校验
`run_config.json` 完全一致。输出按 setting 分目录，每张 PNG 都有独立 JSON
sidecar，记录 seed、prompt、GT/实际使用 CoT、raw pass-1、parser layer、耗时和
provenance；同时生成每组 `results.jsonl`、总 `report.json` 以及
source/target/S1–S5 对照 panel。最终 panel 写入 `panels_with_instruction/`：顶部包含
metadata index、edit type 和完整英文 instruction，七列明确标为 Source、Target、
S1 Stock 2511、S2 Initial direct、S3 Stage-1 direct、S4 Online CoT 和 S5 GT CoT；
`overview_representative_7types.jpg` 额外拼接每种 edit type 的首个代表样本。

8 卡完整评测由 `scripts/eval/run_stage1_eval_8gpu.sh` 统一调度。controller
严格按 setting 1→2→3→4→5 串行启动五次独立 `torchrun`；每次只加载
当前 setting 的模型，8 个 rank 用 `selected_rows[rank::8]` 分片，因此 64 条数据
每卡精确处理 8 条。rank 0 在该 setting 的 64 个 PNG/JSON sidecar 全部完整后
写 `results.jsonl` 和 setting report，controller 才进入下一组。五组完成后，
`--finalize_only` 验收全部 320 个结果、校验 metadata hash/world size，最后生成
总 report 和 panel。该调度器可通过 `RESUME=1` 使用逐样本 sidecar 续跑。

完成后可在不加载模型的情况下重新验收并生成 panel：

```bash
python scripts/eval/run_eval.py \
  --settings 1 2 3 4 5 \
  --output_dir /mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/stage1_evaluation/five_settings \
  --finalize_only
```

`scripts/eval/analyze_stage1_eval.py` 对已有 sidecar/PNG 做离线审计，不加载任何模型：

```bash
python scripts/eval/analyze_stage1_eval.py
```

审计内容包括五组配置逐样本一致性、全部 PNG 解码与唯一 hash、online/GT CoT 的
canonical、空/非空、对象数量、label 和 mask 精确一致性，以及 setting 间逐字节相同
输出和归一化 RGB MAE。RGB MAE 只用于确认输出是否随条件变化，不作为语义编辑质量
指标。默认报告写入 `five_settings/analysis/quantitative_audit.json`。

`scripts/eval/analyze_stage1_cot_masks.py` 进一步使用 released VQ-SAM2 codec，把
online CoT 和 GT CoT 的非空 mask span 都在同一张 source image 上 decode 成二值 mask：

```bash
CUDA_VISIBLE_DEVICES=7 python scripts/eval/analyze_stage1_cot_masks.py --device cuda:0
```

脚本统计 pixel IoU、Dice、bbox IoU、归一化质心距离和预测/GT 面积比，并按 edit type
汇总；同时从原始 CrispEdit mask parquet 恢复 raster annotation，分别检查 online decoded
mask 和 GT decoded mask 相对原始标注的重合度，以区分 Online CoT 误差与 codec 本身的
有损重建误差。默认输出到 `five_settings/analysis/decoded_mask_overlap/`，其中
`report.json` 保存逐样本/汇总指标，`panels/` 保存 Source、Online decoded、GT decoded、
raw annotation 和 Online-vs-GT overlap 可视化，颜色分别为红、绿、蓝。

`scripts/eval/make_stage1_category_comparisons.py` 在上述完整出图和 mask decode 审计之后，
按 `provenance.edit_type` 生成分类汇总图：

```bash
python scripts/eval/make_stage1_category_comparisons.py
```

该步骤不加载任何模型。每个类别生成两张大图：`<type>_final_results.jpg` 逐行展示该类
全部样本，七列固定为 Original、GT edited image、S1–S5；`<type>_mask_comparison.jpg`
逐行展示 raw GT raster mask、GT token decode 和 online token decode。三种 mask 都在各自
独立的 source-image overlay 面板中展示，不使用把三种 mask 混到同一张图上的 overlap
表示；颜色分别为蓝、绿、红。空 CoT 使用原图加显式 `EMPTY ... TOKEN MASK` 标记。
每行顶部保留 metadata index 和完整 instruction。脚本同时写 `manifest.json`，记录每类
样本 index、非空 CoT/raw mask 数量、图片尺寸、绝对路径和 SHA256；默认输出目录是
`five_settings/analysis/category_comparisons/`。

同一脚本也负责 S1–S8 的分类汇总，不另设 Stage 2 可视化实现。传入
`--stage2_root <S6-S8结果目录>` 后，最终结果大图扩展为 Original、GT edited image、
S1–S8 共十列；mask 大图扩展为四个彼此独立的 source-image overlay：raw GT raster、
GT token decode、Stage 1 online token decode、Stage 2 online token decode。S7 的在线
CoT 仍由与 S4 相同的 Stage 1 TE 产生；脚本会先对 64 条 S4/S7 的 metadata、输入、GT、
conditioned CoT、pass-1 原文和 parser 层逐字段硬校验。全部相等时，两列各自展示同一组
真实 codec decode 的独立副本，并在 manifest 记录复用依据；任一字段不同则拒绝生成，
避免把未经 decode 的 Stage 2 token 错配进图中。默认输出到同 step 的
`eight_settings_comparison/analysis/category_comparisons/`。

若需要跨 Stage 2 checkpoint 比较，可在主 `--stage2_root` 之外重复传入
`--additional_stage2_root`。脚本从每个 root 的 `preflight.json` 读取并校验 checkpoint
step，按传入顺序为每个 checkpoint 追加 direct、online CoT、GT CoT 三列，并为每个
checkpoint 的 online mask-token decode 追加一个独立 overlay 列。例如 step-4,000 与
step-8,000 同时比较时，最终结果图为 Original、GT、S1–S11 共 13 列，mask 图为 raw GT、
GT token、Stage 1 online、step-4,000 online、step-8,000 online 共 5 列；布局、单元尺寸、
instruction 和空 mask 标记与单 checkpoint 模式保持一致。

### 4.12 Stage 2 三 setting 评测与八 setting 汇总

`scripts/eval/run_eval.py` 是 S1–S8 共用的统一入口。它在与 Stage 1 完全相同的 64 条验证集
和生成参数上评测显式传入的 Stage 2
DiT LoRA checkpoint。三个新 setting 都加载 gres-ft、Stage 1 正式
`step-5000` TE LoRA、Qwen-Image-Edit-2511 DiT/VAE 和同一个 Stage 2 DiT LoRA：

6. `s6_stage2_direct`：关闭在线 CoT，显式传 `mt_cot=None`，只做原生 direct edit；
7. `s7_stage2_online_cot`：先由 Stage 1 TE 在线 greedy 生成并 canonicalize CoT，再出图；
8. `s8_stage2_gt_cot`：关闭在线生成，直接使用验证 metadata 的 canonical GT CoT。

这三组继续使用 `seed=base_seed+metadata_index`、bf16、40 steps、CFG 4.0、
`edit_image_auto_resize=True` 和 `zero_cond_t=True`。预检除 Stage 1 的数据/模型检查外，还强制
检查 Stage 2 checkpoint 为 1,440 个 BF16 DiT LoRA tensor、720 对 A/B、235,929,600 个参数，
并精确覆盖官方 12 类 target module；checkpoint 文件名中的 optimizer step 与训练 world size
还会用于记录截至该 checkpoint 的含重复样本消费量。

统一入口根据 setting spec 自动决定 stock/SAMTok text encoder、Stage 1 TE LoRA 和 Stage 2
DiT LoRA 的加载边界；数据预检、分片、逐样本 sidecar、resume、finalize 和 panel 代码只有一份。
`scripts/eval/run_stage2_eval_8gpu.sh` 只是面向 S6–S8 的调度 wrapper：它在启动瞬间从正式训练输出中按 step 选择最新完整
checkpoint，并把路径固定给整个 run；随后以三个独立 8-rank `torchrun` 严格串行执行 S6→S7→S8，
避免训练继续写出更新 checkpoint 时混用权重。每个 setting 内仍按
`selected_rows[rank::8]` 分片，每卡 8 条，支持逐 sidecar `RESUME=1` 续跑。

全部完成后，`scripts/eval/analyze_eight_setting_eval.py` 只读取既有 S1–S5 与新 S6–S8 的
sidecar/PNG，不加载模型。它强制核对 8 组的 metadata、seed、steps、CFG、prompt、source、
target、GT CoT、输出尺寸和 world size；检查 direct/online/GT 路由，比较 S4/S7 的同一 TE
pass-1 输出，并对 512 张输出做解码、唯一 hash、28 个 setting pair 的 byte identity 和归一化
RGB MAE 审计。最后生成 64 张包含 Source、Target、S1–S8 和完整 instruction 的十列 panel，
以及 7 类代表样本总览。RGB MAE 只用于输出敏感性/完整性检查，不作为语义编辑质量指标。

### 4.13 Refined Stage 2 ScaleEdit 评测

`scripts/data/build_scaleedit_stage2_validation.py` 从
`/mnt/bn/strategy-mllm-train/user/tanyue/datasets/ScaleEdit-200-samples` 的 200 条已审计
source/edited/mask 数据中构建固定 32 条验证集。样本不是随机前缀：先逐张查看 source、GT edited
和 raw mask overlay，再固定 `sample_id`，四个主类别各 8 条：`small_object`、
`fine_grained`、`multi_instance`、`precise_edit`。主类别用于平衡和分类出图；一条样本还可带
多个 `selection_tags`。

构建器执行以下门禁：

- source/target 直接使用 parquet 内原始 bytes 落盘，raw mask 保持原 PNG；
- prompt 必须为非空 ASCII English，mask 必须是 `qc_flag=OK` 的非空局部 mask；
- 使用 released VQ-SAM2 codec 在 source 上真实编码 mask，并通过唯一 serializer `to_cot`
  生成 non-empty canonical strict CoT；
- `edit_umt` 使用人工复核过且在 instruction 中只出现一次的精确指代短语，将其替换为一个
  `<|mt_start|> code0 code1 <|mt_end|>` span，不修改 instruction 的其他内容；
- codec 编码和 GT token decode 均按小 batch 执行，`SamtokCodec.decode_single_batch` 支持
  不同 source image 的单 span 批量反解；每条 decode 必须尺寸正确且非空；
- 扫描 refined Stage 1/Stage 2 正式 metadata 的全部去重图片引用，先并行检查 byte size，
  对 size-compatible 文件计算 SHA256；任何 source/target exact-content 重合都会使构建失败。

输出包含一个评测主视图和两个标准数据视图：

```text
data/scaleedit_samtok/validation.jsonl          # 评测字段、GT CoT、UMT prompt、mask 路径
data/scaleedit_samtok/validation_edit_mt.jsonl  # 标准 edit_mt schema
data/scaleedit_samtok/validation_edit_umt.jsonl # 标准 edit_umt schema
data/scaleedit_samtok/images/                   # 32 source + 32 GT edited
data/scaleedit_samtok/masks/                    # raw GT + GT token decode
reports/data_build_report.json
```

`scripts/eval/run_stage2_eval.py` 实现三组严格对照：

1. `s1_qwen2511_stock`：只加载原始 Qwen-Image-Edit-2511 TE/DiT/VAE 和官方 processor，
   使用 `QwenImagePipeline` 直接编辑；
2. `s2_stage2_online_cot`：加载 gres-ft TE、refined Stage 1
   `step-10584.safetensors` TE LoRA 和 refined Stage 2 `step-21160.safetensors` DiT LoRA，
   用原 instruction greedy 自回归生成 mask-token CoT，canonicalize 后再出图；
3. `s3_stage2_edit_umt`：加载与第 2 组完全相同的 TE/DiT 权重，关闭在线 CoT，把
   `edit_umt_prompt` 中的 GT mask span 作为 user instruction 的一部分直接出图。

三组均固定 `seed=base_seed+eval_index`、bf16、40 inference steps、CFG 4.0、
`edit_image=[source]`、`edit_image_auto_resize=True`、`zero_cond_t=True`。输出保持 source
宽高比、目标面积为官方约 `1024*1024`，宽高各自按 32 对齐；例如 2250×1500 输入统一生成
1248×832，避免把原始大图尺寸直接用于 diffusion latent。输入条件图仍由官方
`edit_image_auto_resize=True` 独立缩放到约 1024²。
第 2 组 `do_sample=False`、默认最多生成 128 token；第 3 组运行时额外检查 mask span 被 tokenizer
处理为四个原子 token，并实际出现在 2511 user template 内。由于 CFG 会先后调用同一 prompt
embedder 编码 positive/negative prompt，pipeline 每次调用都会重置审计状态，并保留正向 user
prompt 的非空 mask-span 审计，防止空 negative prompt 覆盖结果。sidecar 保存原 prompt、实际
conditioning prompt、raw pass-1、canonical CoT、parser layer、seed、耗时和模型 provenance。

`scripts/eval/run_scaleedit_refined_eval_8gpu.sh` 默认依次运行 1--3，每次只启动一个 8-rank torchrun，所有卡
按 `rows[rank::8]` 处理不同样本，完成后再聚合。`SETTING_SEQUENCE` 可指定需要执行的 setting，
配合 `RESUME=1` 定点补跑失败组而不重新生成已完成图片，例如
`SETTING_SEQUENCE=3 RESUME=1`。评测前可运行不加载模型的完整门禁：

```bash
python scripts/eval/run_stage2_eval.py --dry_run
```

正式 8 卡入口为：

```bash
bash scripts/eval/run_scaleedit_refined_eval_8gpu.sh
```

最终出图不计算图像质量指标。`run_stage2_eval.py --finalize_only` 为每条样本生成 Source、GT、
Stock、Online CoT、edit_umt 五列对比，并按四个主类别生成 overview；
`scripts/eval/visualize_stage2_eval_masks.py` 使用 released codec 反解 online CoT，分别以三个
独立 source overlay 展示 raw GT mask、GT mask-token decode 和 online mask-token decode，
不会把三种 mask 混合在同一个 overlay 中。完整 instruction 会写在每一行图的标题中；报告只
记录 parser/decode 是否有效和文件位置，不对编辑图计算自动指标。

### 4.14 Arnold 四机 32 卡入口

四机入口是独立新增实现，不改变 4.8 节的单机 8 卡脚本。目标拓扑固定为 4 台机器、
每机 8 卡、全局 32 个 DDP process，直接读取 Arnold 注入的
`ARNOLD_WORKER_HOSTS`、`ARNOLD_WORKER_NUM`、`ARNOLD_WORKER_GPU` 和 `ARNOLD_ID`。
`scripts/train/arnold_4node_env.sh` 同时支持 `host:port` 和 `[IPv6]:port` 两种
`ARNOLD_WORKER_HOSTS` 格式，并转换为 Accelerate 所需参数：

```text
--num_processes 32
--num_machines 4
--machine_rank $ARNOLD_ID
--main_process_ip $ARNOLD_WORKER_0_HOST
--main_process_port <Arnold allocated port>
--rdzv_backend static
--same_network
```

各节点只可见本机 `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`；Accelerate 的
`num_processes` 表示全局进程数，而不是每机进程数。

新增文件及职责如下：

- `scripts/train/arnold_4node_env.sh`：解析/校验 Arnold 拓扑并封装多机 Accelerate；
- `scripts/train/launch_4node.sh`：提供 `stage1`、`stage2_cache`、`stage2_train`
  三个独立的 32-rank launch phase，模型与训练参数和单机入口一致；
- `scripts/data/prepare_4node_metadata.sh`：复用已验收 refined component JSONL 和落盘
  图片，只生成并验证新的 ws32 metadata，不重新运行 codec；
- `scripts/train/run_arnold_4node_pipeline.sh`：按 Stage 1 training → Stage 2 cache →
  rank-0 cache audit → Stage 2 training 串行编排，使用共享 marker 做跨节点阶段同步；
- `scripts/train/bootstrap_arnold_4node.sh`：可作为 Arnold 从裸 worker 开始的完整入口。每个
  worker 安装 `ffmpeg/libsm6/libxext6/tmux/htop`；rank 0 清除 proxy 后使用已在 Arnold worker
  实测成功的 GitHub 直连，在共享 `/mnt/bn` 路径 clone `dev`，随后执行
  `python3.11 -m pip install --user uv==0.11.32`、`cd` 进入仓库并运行
  `setup_env.sh`。其他 worker 通过共享 marker 等待，环境就绪后四节点共同进入
  pipeline。

ws32 metadata 使用独立实验目录，不覆盖单机 metadata。固定统计为：

```text
Stage 1: edit_mt=42368, edit_ntp=21184, edit=10592, edit_umt=10592
         total=84736
Stage 2: edit_mt=42368, edit=21184, edit_umt=21184
         total=84736
```

Stage 1 的 32-rank schedule 每个 optimizer step 仍由 8 个 homogeneous micro-step
组成，全局类型计数为 `128:64:32:32`，331 optimizer steps 内每条 metadata 恰好使用
一次。Stage 2 metadata 的 32 个 strided shards 每份均为
`edit_mt=1324,edit=662,edit_umt=662`。Stage 2a 必须重新生成 32 个 rank 目录下的
cache，不能复用 sidecar 中记录 `world_size=8` 的 cache；全量 cache audit 只在 node 0
执行，其余节点等待 audit pass marker 后才进入 Stage 2b。

数学训练 setting 保持原实现：Stage 1 为 1 epoch、gradient accumulation 8、
`lr=4e-5`、`weight_decay=0.05`、`lambda_ntp=0.05`；Stage 2 为 1 epoch、
dataset repeat 2、gradient accumulation 1、`lr=1e-4`、`weight_decay=0.01`。
由于 world size 从 8 增至 32，有效 global batch 分别从 64/8 变为 256/32；四机入口
不自动缩放 learning rate。每 rank dataloader worker 默认从 8 降为 2，使全局 worker
总数仍为 64，这只改变 I/O 并发度，不改变优化语义。

完整 Arnold 入口在脚本顶部保留用户填写区：必填 `WANDB_API_KEY` 和每次唯一的
`SAMTOK_RUN_ID`；`WANDB_ENTITY` 默认为 `2200012743-peking-university`，`WANDB_PROJECT`
默认为 `samtok-edit`。入口不调用 `wandb login`，
API key 不会写进 metadata、`training_args.json` 或控制 marker。该入口
必须在 clone 前已经可被 Arnold 读取（推荐把完整内容直接粘贴到 Arnold entry）；不能在
尚未 clone 时通过目标仓库路径调用它。完整可复制版见
`SamtokEdit_四机训练运行指南.md`：

```bash
export WANDB_API_KEY=""   # 必填
export WANDB_ENTITY=2200012743-peking-university
export WANDB_PROJECT=samtok-edit
export SAMTOK_RUN_ID=""   # 必填，每次实验使用新名称

# 实际任务中由四个 worker 执行已粘贴到 Arnold entry 的完整
# scripts/train/bootstrap_arnold_4node.sh 内容。
```

同一入口脚本必须由 Arnold 在四台 worker 上执行。输出默认位于
`/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/`
`crispedit_refined_4node/$SAMTOK_RUN_ID`；每个分布式阶段按 node 分日志，最终
`reports/run_manifest.json` 记录两阶段 metadata/checkpoint SHA256 和 cache audit 结果。实验
名称统一使用 `crispedit-refined-4node-<唯一后缀>`，不使用 `v1`；W&B 两个 run 分别追加
`-stage1` 和 `-stage2`。

bootstrap 从确定实验目录开始，将四个 worker 的完整 stdout/stderr 分别 tee 到
`$RUN_ROOT/logs/bootstrap.node0.log`--`bootstrap.node3.log`，覆盖 apt、Git clone、uv 安装、
环境校验、metadata、训练和最终验收。bootstrap marker 位于 `$RUN_ROOT/bootstrap_control/`；
pipeline 只允许该次 bootstrap 预先创建的 `logs/bootstrap.node*.log`、`environment.ok` 和
`git_commit.txt`，发现其他旧产物仍会拒绝复用目录。phase 自身的细分日志继续写入
`$RUN_ROOT/logs/<stage>.node<N>.log`。

### 4.15 DiT mask-token 反事实与注意力分析

解释性入口由以下文件组成：

- `scripts/eval/prepare_mask_token_interventions.py`：从统一 fine-grained benchmark 中选取
  12 个类别互不重复的同类多实例 case；`--selection_set all` 表示完整 12-case 协议，
  `core/additional` 仅用于复现两次已完成 GPU job 的调度分片，不表示不同实验版本。脚本保留 benchmark 原 mask A，
  并调用 gres-ft codec 内已加载的 SAM2.1 Hiera-L，以另一个同类实例上的正点和
  原目标点上的负点得到 mask B；A/B 均经过真实 SAMTok codec encode/decode；
- `scripts/eval/run_mask_token_interpretability.py`：使用 refined 四机训练结束的 Stage 1 TE
  LoRA 和 Stage 2 DiT LoRA 执行成对 `edit_umt` 推理，并以只读 forward pre-hook 捕获 DiT
  joint attention；
- `scripts/eval/summarize_mask_token_interpretability.py`：验证输出与注意力归一化，计算定位/迁移
  指标并生成逐 case 与总览图；
- `scripts/eval/visualize_mask_token_attention_clear.py`：在不改变原 attention 数值的前提下生成
  标注完整 A/B conditioning prompt 的高对比度九列总览；
- `scripts/eval/consolidate_mask_token_interpretability.py`：严格校验两个已完成 source job，
  合并为唯一的 12-case manifest、metrics、report 和总览；逐 case 九列图使用 symlink，
  不复制原始推理图或 attention NPZ；
- `scripts/eval/run_mask_token_interpretability_3gpu.sh`：三卡完整编排，写入 prepare、inference、
  summarize、clear visualization 日志和 `controller.status`；
- `scripts/eval/run_mask_token_interpretability_additional_8gpu.sh`：第二调度分片的完整编排，
  默认用 8 卡按 case 并行，卡数由 `CUDA_VISIBLE_DEVICES` 动态推导；该命名仅为已完成作业的
  兼容入口，最终结果不按分片区分；
- `tests/test_mask_token_interpretability.py`：覆盖固定选择、top-area mask、重合指标和双向注意力
  抽取。

实验使用 benchmark 的 `instruction.region_only` 模板。可读模板只含
`Remove {region_1}.` 或 `Replace {region_1} with <object/background>.`；实际 prompt 将
`{region_1}` 替换为 canonical 四原子 token
span，绝不加入 left/right/序数等位置指代。A/B 共享 source、location-free 文本、seed、checkpoint
与全部 diffusion 参数，唯一变化是 mask span。这里关闭 pass-1（`enable_samtok_cot=False`），
因为待测因素是用户直接提供的 mask token；仍然走本方法的 Stage 1 TE + Stage 2 DiT
`edit_umt` inference，而不是 stock pipeline。

alternate mask 的构造不是手工涂抹：脚本复用 released gres-ft codec 中的 SAM2.1 Hiera-L，
正点落在另一个同类实例、负点落在 benchmark 原目标点。构建门禁要求 case 标记为
`same_class_multi_instance`，alternate mask 面积在合理范围，raw A/B IoU 小于 0.2，四 token
必须变化，location-free 模板不能泄漏位置词，raw mask 与各自 token decode 的
IoU 不得低于 0.5，decoded A/B IoU 也必须小于 0.2。manifest 同时保存 raw mask 和
token decode；
分析时以 decode mask 为主，因为它才是四个离散 token 实际表达的区域，raw mask 指标仅作为
补充。

注意力 probe 不修改 `DiffSynth-Studio` 的 forward，也不替换 flash-attention 输出。它在指定
`QwenDoubleStreamAttention` 层的输入处，使用该层真实的 Q/K projection、RMSNorm、Qwen RoPE
和 `1/sqrt(head_dim)` scale 重算所需概率，并只记录 CFG positive branch。输出/noisy latent token
位于 image sequence 前部，source/edit-image latent token 紧随其后；probe 按实际 latent shape
明确切出 source 范围，避免把生成 latent 当成输入图。mask token 位置由实际 Qwen2VL processor
输入定位，并按 `EDIT_DROP_IDX` 转换为进入 DiT 的 text sequence 坐标。

每个 condition 在 zero-based DiT layer `5,15,30,45,59` 与 denoising step
`0,10,20,30,39` 的笛卡尔积上记录两种方向：

1. `mask_query_to_source`：四个 mask-token query 对 source-image key 的 attention；
2. `source_query_to_mask`：每个 source-image query 对四个 mask-token key 的 attention 总和。

对每个被探测层，先用该层真实 `to_q/to_k/add_q_proj/add_k_proj`、Q/K RMSNorm 和
Qwen RoPE 重算 `Q` 与 `K`，再计算
`P = softmax(Q K^T / sqrt(head_dim) + attention_mask)`。softmax 的 key 轴是完整的
`[text keys, output-image keys, source-image keys]`，不是只在 source 区域内先做 softmax。
`mask_query_to_source` 从 `P` 中取四个 mask-token query 和 source key 子矩阵，跨
batch、24 heads 和 4 tokens 平均；`source_query_to_mask` 则取 source query 对四个
mask-token key 的概率之和，再跨 batch 和 head 平均。两者最后都只在 64×64 source
grid 上归一化为和 1；每个 condition 有 25 张/方向，共 50 张图。

指标默认使用 codec-decoded mask，因为这才是四个离散 token 实际表示的区域；raw
mask 只作辅助审计。对每张已归一化 heatmap `H`：

- `target_mass = sum(H[target])`，`other_mass = sum(H[other])`；25 张分别计算后取平均；
- `routing_margin = mean(target_mass) - mean(other_mass)`，正值表示当前 condition 更偏向
  它应该表示的实例，但会受 A/B 面积差异影响；
- `attention_density = mean(target_mass) / target_area_fraction`，`density_margin` 是 target
  与 other 的单位面积 density 之差；
- `top-area IoU`：以 target 在 64×64 grid 上的格子数 `k` 取 heatmap 最高的 `k`
  个格子，与 target 求 IoU，再对 25 张图取平均；它避免了人工设置 heatmap 阈值；
- 如 target 面积比例为 `p`，同面积独立随机选择的 plug-in chance IoU 为
  `p / (2 - p)`，`IoU lift = measured IoU / chance IoU`；
- `peak-inside rate`：25 张 heatmap 的 argmax 落在 target 内的比例；
- 显示/shift 使用 25 张图等权平均后再归一化的 aggregate heatmap。`shift cosine`
  是 attention 质心从 A 到 B 的位移向量与 decoded-mask 质心位移向量的余弦；
- `switch score = [H_B(B)-H_A(B)] + [H_A(A)-H_B(A)]`，即换成 B token 后 B 区域
  获得的 attention 加上 A 区域失去的 attention。

注意力只提供描述性证据，不能单独证明因果。更强的行为证据来自只替换 mask
token 后编辑落点是否随之改变。另外，DiT 接收的是 TE contextualized embeddings，
mask 信息可能已扩散到其他 text token，仅探测四个字面 token 位置可能低估模型的使用。

清晰版可视化不改变 heatmap 数值，只改变显示方式。每个 case 的主图严格固定为九列：原图、
raw mask A overlay、mask-token A decode overlay、SAM2 修改后的 raw mask B overlay、
mask-token B decode overlay、mask-token A query→source attention、mask-token B query→source
attention、A 编辑结果、B 编辑结果。两个 attention panel 不放任何
mask fill、mask boundary 或 top-k 轮廓，只把原图压暗并叠加高对比度强度色；显示值为
source-space attention probability 除以均匀注意力 `1/(H*W)` 后的 enrichment，因此均匀水平是
`1×`。展示的是 `mask_query_to_source` 在 25 个 layer-step 上的等权平均，不是
`source_query_to_mask`；后者只进入数值报告。A/B 共享一个上限，上限是两组原始
25 张 enrichment 的 99.5 percentile 与 `2×` 中的较大值；色标下限固定为 `0.5×`，
显示强度再做 `((E-0.5)/(high-0.5))^0.52` 的 gamma 增强。这只影响显示，不影响指标。
黄色/白色表示高关注。图头逐字写出送入 TE 并编码为 DiT conditioning 的完整 A/B user
prompt，包括 `<|mt_start|>`、两个离散 code token 和 `<|mt_end|>`，因此可以直接核对反事实
条件具体替换了哪些 mask token。

已完成的 source job 可分别用下列命令复现；二者只负责不同 case 的 GPU 调度，实验协议与
最终分析完全相同：

```bash
cd /opt/tiger/tanyue/samtok_edit
source .venv/bin/activate
bash scripts/eval/run_mask_token_interpretability_3gpu.sh
bash scripts/eval/run_mask_token_interpretability_additional_8gpu.sh
```

GPU 数量只用于 case 级并行，不执行 DDP collective。source job 分别保留
`data/interventions.jsonl`、`runs/*/{original,alternate}_{output,attention,record}`、
`analysis/{metrics.jsonl,report.json}` 和 `visualizations_clear/`，确保原始产物可追溯。
最终结果不再按 source job 分开展示，而由以下命令建立唯一入口：

```bash
cd /opt/tiger/tanyue/samtok_edit
source .venv/bin/activate
.venv/bin/python scripts/eval/consolidate_mask_token_interpretability.py
```

统一输出目录为：

```text
/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/crispedit_refined/interpretability/dit_mask_token_counterfactual_12case
```

其中 `manifest.jsonl`、`metrics.jsonl`、`report.json` 和
`visualizations/overview_12case.jpg` 是文档和人工检查使用的 canonical 入口。总览每行顶部
固定增加 `UNIFIED CASE 00--11`、类别和 benchmark ID 横幅，使两个 source job 内部均从 0
开始的 provenance 编号不会被误解为统一编号；
`report.json` 同时记录底层 source experiment root 与校验和。旧版临时 panel 和
layer-step grid 不保留或引用。

---

## 代码回归入口

```bash
cd /opt/tiger/tanyue/samtok_edit
python -m unittest tests/test_samtok_edit.py tests/test_mask_token_interpretability.py
```

测试覆盖 canonical CoT、分层 parser、DDP schedule、非 canonical 拒绝、codec 空 mask 拒绝、
英文模板、全局随机抽样与 worker 分区、state-dict converter、KV-cache 转发和新版 DiffSynth
分片路径兼容，以及分类 mask 大图从审计 panel 中提取 GT/online 独立面板的列顺序、
S1–S8 分类汇总对 S4/S7 在线 CoT 完全一致性的强制校验、ScaleEdit 32 条选择平衡、
Stage 2 三 setting 的 CoT/UMT 调用契约、CFG negative
prompt 不覆盖正向 UMT span 审计和官方约 1MP 输出尺寸换算。
解释性测试另行覆盖反事实 case 固定选择、双向 attention map 的 source-range 抽取与归一化、
top-area IoU 和 routing metric。
测试运行结果和训练/数据实验结果统一记录在 `SamtokEdit_实验记录.md`。
