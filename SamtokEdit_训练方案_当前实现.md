# SamtokEdit 训练方案（当前实现）

> SAMTok mask-token CoT 引导的 Qwen-Image-Edit-2511 细粒度编辑实现说明。
>
> 本文基于 `/opt/tiger/tanyue/SamtokEdit 训练方案.md` 更新，描述当前仓库
> `/opt/tiger/tanyue/samtok_edit` 的实际代码、数据规范、训练入口和实现约束。
> 原始方案中的目标与章节组织保留；代码片段和命令以当前实现为准。

当前代码实现包含 Stage 1/Stage 2 训练入口、canonical CoT 数据管线、SAMTok codec 构建器、
Qwen-Image-Edit-2511 / SAMTok gres-ft 模型适配和 Stage 1 评测入口。实验运行过程与结果单独记录在
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
  metadata: edit_mt : edit_ntp : edit = 2 : 1 : 1
  edit_ntp: edit_image + prompt + CoT，input_image=None，只有 NTP
  edit:     source edit_image + target image，无 CoT，只有 FM
  edit_mt:  source edit_image + target image + CoT，单次 text encoder forward，NTP + FM
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

约束：

- `edit_mt` 和 `edit_ntp` 的 `mt_cot` 必须已经是 `to_cot` 产生的 canonical 形式；
- 全局编辑或空 mask 使用 `to_cot([])`，即 ````json\n[]\n````，不是缺失值；
- `edit_ntp` 省略 `image`，训练入口会令 `input_image=None`，从而不计算 FM；
- `edit_image` 在训练算子中无论输入是字符串还是路径列表，最终都会成为非空 PIL image list；
- 真实 mask 的 SAMTok 编码在源图上进行，避免 pass-1 只看源图而训练 CoT 却来自 target 图。

训练数据由 `compose_training_metadata.py` 生成 Stage 1/Stage 2 JSONL；输出目录不在代码中
硬编码，由 launcher 的 `DATASET_BASE`、`STAGE1_METADATA` 和 `OUTPUT_PATH` 参数指定。

### 2.2 三类样本

**edit_mt** 来自带 mask 的 CrispEdit 数据：

1. 从原始 `CrispEdit-2M` parquet 读取 input/source、output/target、instruction、type；
2. 从 `CrispEdit-2M-mask-parquet-101697` 按同名 parquet 和 `row_idx` 对齐 mask；
3. 仅保留 `filter_decision == "keep"`；
4. 对 source/target 图片进行落盘；
5. 非空 mask 由 `SamtokCodec.encode_single_batch` 编码，写入一条 mask span；
6. global/noop 或空 mask 写成 `to_cot([])`；
7. 同时生成 `edit_mt.jsonl` 和无 CoT 的 `edit.jsonl`。

当前构建器按每个 mask parquet row 处理一张 mask；它不是原方案中尚未实现的“多表达式、多 mask
分组编排器”。后续如要支持多 mask，需要在 `build_edit_mt_metadata.py` 中扩展输入 schema，
并继续通过 `to_cot` 写 canonical 结果。

**edit_ntp** 来自 GRES/SAMTok 发布数据：

- 输入默认是 `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Training_Data/mask_generation_gres209k.json`；
- 图片根目录默认是 `/mnt/bn/strategy-mllm-train/intern/common_datasets/Sa2VA-Training/osprey-724k`；
- 从 release conversation 的 CoT 或 segmentation question 中提取 expression；
- 对已存在的 mask 复用原 CoT 重新 canonicalize，并将 prompt 改写为英文编辑模板；
- 默认按 `global_ratio=0.10` 增加全局英文编辑行，CoT 为 `to_cot([])`；
- 当前 `EDIT_VERB_TEMPLATES` 和 `GLOBAL_TEMPLATES` 全部 ASCII 英文，不使用中文模板。

**edit** 使用 CrispEdit 原始 instruction，不带 CoT，作为通用编辑/FM 保持项。当前脚本支持
按 mask parquet 的 `filter_decision=keep` 过滤，也支持 `--max_files`、`--max_rows` 做构建烟测。

### 2.3 CoT canonical 格式

当前唯一序列化函数是
`DiffSynth-Studio/diffsynth/core/data/samtok_dataset.py:to_cot`。

骨架：

````text
```json
[]
```
````

非空 item：

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
│   │   ├── sanitize_stage2_validation_content.py
│   │   ├── compose_training_metadata.py
│   │   ├── validate_training_metadata.py
│   │   └── validate_metadata_disjointness.py
│   ├── train/
│   │   ├── train_samtok_edit.py
│   │   ├── stage1_te_lora.sh
│   │   ├── stage2_data_process.sh
│   │   ├── stage2_dit_lora.sh
│   │   ├── audit_stage2_cache.py
│   │   └── run_stage2_8gpu_pipeline.sh
│   ├── inference/infer_samtok_edit.py
│   ├── inference/validate.py
│   ├── eval/run_stage1_eval.py
│   └── eval/run_stage1_eval_8gpu.sh
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

- `type_ratio` 默认 `edit_mt:2,edit_ntp:1,edit:1`；
- `metadata_path` 非空且 ratio 非 `none` 时建立 Stage 1 schedule；
- `metadata_path=None` 或 ratio=`none` 时不建立 schedule，走父类 cache 行为；
- 构造期检查 `edit_mt/edit_ntp` 的 CoT 必须已 canonicalize，且只能是 `strict` 或 `empty`；
- 每个 optimizer step 的 A 个 micro-step 类型由配比块重复并随机排列；
- 每个 micro-step 连续放置 P 个相同类型样本，保证 DDP rank 同型；
- 要求 `gradient_accumulation_steps % ratio_block_size == 0`，不要求卡数整除；
- 每类样本 pool 独立 shuffle，取尽后循环重洗；
- DataLoader 必须 `shuffle=False`，否则破坏 schedule；
- `__len__` 返回 schedule 长度，避免 repeat 被父类再次乘一次。
- `__getitem__` 额外写入仅在运行时使用的 `_samtok_schedule_position` 和
  `_samtok_source_row_id`，用于 debug 模式核对 Accelerate 的 rank 分片，不改变 metadata 文件。

当 `P=1,A=4` 时，schedule contract 要求每个累积窗口包含 2 条 `edit_mt`、1 条 `edit_ntp`、
1 条 `edit`；当 `P=8` 时每个同型 micro-step 连续占据 8 个位置。

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
- `input_latents` 只来自 metadata `image` 目标图，`edit_latents` 来自 `edit_image` 条件图；
- CoT hidden/label、prompt embedding、input/edit latent 的 shape 和 dtype；
- 每个 rank 的梯度有限性、非零梯度张量、冻结梯度张量；
- accumulation slot、同步步、optimizer step、learning rate、clip return norm；
- 首个 LoRA B 张量的实际 update L2 norm，以及同步步后所有 rank 的参数一致性；
- 每个累积窗口第一个 micro-step 的梯度未混入旧梯度，可作为 NTP/FM 加权尺度的调参观测。

Stage 2 debug 模式的审计内容：

- 只有 `pipe.dit.*.lora_A/B` 可训练，text encoder/VAE 可训练参数为 0；
- DiT LoRA tensor 为 bf16，并统计 12 组官方 target module family 的命中数量；
- 每个 epoch 的 24 个物理 cache 在 `dataset_repeat=2` 后恰好各消费两次，总类型为
  `edit_mt=32, edit=16`；
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
sample_type_ratio=edit_mt:2,edit_ntp:1,edit:1
lora_base_model=text_encoder
lora_rank=64
lora_dropout=0.05
learning_rate=4e-5
weight_decay=0.05
warmup_ratio=0.05
max_grad_norm=1.0
gradient_accumulation_steps=4
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
MAX_PIXELS=1048576 GRADIENT_ACCUMULATION_STEPS=4 NUM_EPOCHS=1 \
NTP_LOSS_WEIGHT=0.05 FM_LOSS_WEIGHT=1.0 \
WANDB_API_KEY=<your-api-key> WANDB_ENTITY=<your-user-or-team> WANDB_PROJECT=<your-project> \
bash scripts/train/stage1_te_lora.sh
```

多卡时将 `NUM_PROCESSES` 设为大于 1，launcher 才会调用 Accelerate；
`MAIN_PROCESS_PORT` 可避免同机任务的 rendezvous 端口冲突。`gradient_accumulation_steps` 需要是 4 的
倍数，以保持每个窗口的 2:1:1 类型结构。NTP 和 FM 都经过全部 TE LoRA，因此
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
`--disable_wandb_log`。数据 `stage2.jsonl` 由 `compose_training_metadata.py` 生成，默认由
`edit_mt + edit` 构成。

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
24 行 8 卡 smoke 在 `dataset_repeat=2,num_epochs=5` 时，每卡每 epoch 6 个 micro-step，
全局有效 batch 为 8，总计 30 个 optimizer step。
该 `num_epochs=5` 仅是已完成 smoke 的历史验证配置；当前正式 Stage 2b 默认为
`num_epochs=1`。对 165,960 个物理 cache、`dataset_repeat=2`、8 卡、每卡 batch 1 和
gradient accumulation 1，正式运行为 41,490 个 global optimizer step。

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
/mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-mask-parquet-101697
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
  --global_ratio 0.10 --seed 0 --check_images
```

主循环读取 conversation，恢复 release CoT，提取 expression，随机选择英文
`EDIT_VERB_TEMPLATES`；global 行使用英文 `GLOBAL_TEMPLATES` 和 `to_cot([])`。默认路径写
绝对图片路径；传 `--relative_image_paths` 可保持相对 GRES path。

#### 4.9.4 CrispEdit 三类 metadata

`build_edit_mt_metadata.py` 负责带 mask 的 CrispEdit parquet，并同时产出：

```text
edit_mt.jsonl  # image + edit_image + prompt + mt_cot
edit.jsonl     # image + edit_image + prompt
```

构建器调用示例：

```bash
python scripts/data/build_edit_mt_metadata.py \
  --crispedit_dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M \
  --mask_dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-mask-parquet-101697 \
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

`build_edit_metadata.py` 用于不需要 mask 的纯 edit 数据，并已支持与 mask builder 一致的
全局 `--sample_rows/--seed/--ascii_only`、hard `--exclude_metadata_jsonl`、多 worker 原子
shard、`--resume` 和 `--combine_only`。每行写入原始 parquet/row/type provenance。
`--deprioritize_metadata_jsonl` 用于数据池不足以完全互斥时最小化来源重合：先使用所有不在
deprioritized metadata 中的合格行，仅从重合池随机补足必要差额。`--filter_with_mask_parquets`
仍可限制到 mask parquet 的 keep rows。两个脚本都将图像 bytes 原子落盘为
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

大数据构建还支持多 GPU parquet 分区：每个进程使用相同的全局抽样集合，传
`--num_workers 8 --worker_index 0..7 --skip_combine --resume` 后只写互不重叠的原子 shard；
所有进程完成后用 `--combine_only` 检查全部预期 shard 并合并，不再次加载 codec。
多进程 codec 构建应同时设置
`OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8`；否则每个
PyTorch/SAM2 进程可能各自创建约 250 个 host thread，造成严重 CPU oversubscription 和
吞吐下降。该限制只影响 host 并行度，不改变 codec 数值。

source identity 互斥不足以排除 CrispEdit 内部“不同 parquet row 复用同一原图”的情况。
`sanitize_stage2_validation_content.py` 在 Stage 2 source pools compose 前执行内容级净化：先对
验证集 source/target 建立 SHA256 集合，只对训练池中 size-compatible 的唯一路径做 hash，
删除任何 source/target 内容命中的行；随后保留全部安全 `edit_mt`，并按
`ceil(edit_mt/2)` 选择纯 `edit`。纯 edit 始终先使用不在安全 `edit_mt` identity 集中的行，
再以固定 seed 从重合池补足，因此训练分支 source overlap 达到理论最小；脚本原子输出两个
`*_train.jsonl` 和包含排除/重合统计的 JSON 报告。

`compose_training_metadata.py` 将三类输入校验、抽样、打乱并生成 Stage 1/Stage 2 JSONL：

```bash
python scripts/data/compose_training_metadata.py \
  --edit_mt_jsonl .../edit_mt.jsonl \
  --edit_ntp_jsonl .../edit_ntp_gres.jsonl \
  --edit_jsonl .../edit.jsonl \
  --stage1_output .../stage1.jsonl \
  --stage2_output .../stage2.jsonl \
  --max_edit_mt 8 --max_edit_ntp 4 --max_edit 4 --seed 0
```

它只负责 metadata 级别的比例和随机化；Stage 1 运行时的精确 2:1:1 由
`SamtokEditingDataset` schedule 再次保证。

`--stage1_output` 和 `--stage2_output` 现在可独立选择；仅构建 Stage 2 时不需要
`--edit_ntp_jsonl`。Stage 2 metadata 只包含 `edit_mt + edit`，默认依次随机抽样并
打乱。针对多卡 data-process，可传 `--stage2_num_shards P`；构建器会先将每种类型
均分到 P 个 shard，再按 position-major 顺序写文件，使 Accelerate 的
`rows[rank::P]` 分片在每个 rank 都保持相同类型比例。各类型数量必须可被 P
整除；若全量 source pool 因奇数或过滤后只能接近 2:1，可额外传
`--pad_stage2_to_shards`。构建器保留全部输入行，求满足最终 `edit_mt=2*edit` 且两类均可被
P 整除的最小不小于输入的计数，再按固定 seed 复制缺少的行；复制行写入
`schedule_padding` 及原因/ordinal，避免 DDP sampler 隐式 padding。报告会同时给出未
padding 的 source counts、最终 counts 和 padding counts。例如 Stage 2 的 8 卡 smoke
可单独构建 16:8 的 24 行：

```bash
python scripts/data/compose_training_metadata.py \
  --edit_mt_jsonl .../edit_mt.jsonl \
  --edit_jsonl .../edit.jsonl \
  --stage2_output .../stage2.jsonl \
  --max_edit_mt 16 --max_edit 8 --stage2_num_shards 8 --seed 8
```

#### 4.9.5 `validate_training_metadata.py` —— 构建产物验收

验收器不改变数据，只检查 JSONL schema、sample type、canonical CoT、文本字符集、图片路径
和随机图片解码；支持 `--expected_counts`、`--check_paths`、`--decode_image_sample`、
`--report_json`，路径检查和图片解码使用有界线程池。

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

### 4.11 Stage 1 评测：`scripts/eval/run_stage1_eval.py`

当前脚本专用于 Stage 1 的五组图像编辑对照，所有 setting 共用 stock
Qwen-Image-Edit-2511 DiT/VAE，不加载 DiT LoRA：

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
python scripts/eval/run_stage1_eval.py --dry_run
```

真正出图时直接运行：

```bash
python scripts/eval/run_stage1_eval.py
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
python scripts/eval/run_stage1_eval.py \
  --settings all \
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

---

## 代码回归入口

```bash
cd /opt/tiger/tanyue/samtok_edit
python -m unittest tests/test_samtok_edit.py
```

测试覆盖 canonical CoT、分层 parser、DDP schedule、非 canonical 拒绝、codec 空 mask 拒绝、
英文模板、全局随机抽样与 worker 分区、state-dict converter、KV-cache 转发和新版 DiffSynth
分片路径兼容。测试运行结果和训练/数据实验结果统一记录在 `SamtokEdit_实验记录.md`。
