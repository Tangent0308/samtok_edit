# SAMTokEdit（Qwen-Image-2.1）代码实现说明（v3）

- **分支**：`qwen-image-2.1-v3`，本地 `/opt/tiger/tanyue/samtok_edit_qwen-image-2.1-v3`。
- **本文档讲什么**：当前方法是怎样在官方 DiffSynth 和官方 SAMTok 代码之上实现的，每一块在哪个文件、关键代码是什么。方法为什么这样设计见[文档 02](02_SAMTokEdit_Qwen21_v3方法.md)，实验和结果见[文档 03](03_SAMTokEdit_Qwen21_v3实验.md)，数据见[文档 01](01_SAMTokEdit_Qwen21_v3数据.md)。第 1–3.8 节写的是第一轮（`ground-*`，E-A2）的实现，第二轮（`round2-*`，C1/C1b/C2）见 3.9 节。
- **维护约定**：代码有改动时同步改本文档。文中的代码块不是手抄的，而是由 `scripts/docs/sync_code_blocks.py` 从源码生成；改完代码后运行一次该脚本即可更新，测试 `tests/test_docs.py` 会检查两者是否一致。文末有更新记录。
- **v2 的实现说明**：v2 独有的部分（Stage 1 定位、结构性绑定、缓存、v2 训练引擎）仍在代码里，本文第 7 节只列出它们的位置，细节见 [archive/v2/01_SAMTokEdit_Qwen21_代码实现说明.md](archive/v2/01_SAMTokEdit_Qwen21_代码实现说明.md)。

---

## 1. 总览

### 1.1 方法与数据流

编辑区域由 SAMTok 编码成一个 **mask token 片段**（4 个 token），直接写在指令里：

```
Remove the second person on the left <|mt_start|><|mt_0097|><|mt_0411|><|mt_end|>
```

推理时模型的输入只有源图和这句指令：

```
源图 ──┐
       ├─► 文本编码器（Qwen3-VL-8B-SAMTok，冻结）──► 逐 token 特征 ──(+ 码本向量注入)──┐
指令 ──┘                                                                              ├─► DiT（原版权重 + LoRA）──► 编辑结果
源图 ──► VAE ──► 源图 latent ─────────────────────────────────────────────────────────┘
```

没有把 token 解码成像素 mask，没有注意力区域偏置，没有 latent 融合。mask 只在**训练时**出现，用来构造目标图和损失权重。

v3 相对原版模型新增的可训练部分只有两样：
1. **DiT 上的 LoRA**（rank 32，83.9M 参数）。
2. **码本向量注入**（3 个线性层，3.16M 参数）：把 SAMTok 码本里这两个码对应的向量，经零初始化投影后加到文本编码器在这 4 个 token 位置的输出上。

训练时另有 12 个只用于损失标定的参数（注意力路由损失的逐层仿射），推理不用。

### 1.2 基于哪些官方代码，各改了什么

| 依赖 | 版本 | 放在哪里 | 本项目做了什么 |
|---|---|---|---|
| DiffSynth-Studio | 2.1.8，commit `7686e54d` | `third_party/diffsynth/`（整库收进仓库，以独立包 `diffsynth` 安装） | 四处**默认关闭**的扩展，不传相应参数时与上游逐位一致：`diffusion/runner.py`（调度采样、梯度审计）、`core/attention/`（可微的 FlexAttention）、`models/qwen_image_21_dit.py`（`region_binding` 挂载点）、`pipelines/qwen_image_21.py`（返回裁剪后的 token id）。清单见 `third_party/README.md` 和 `upstream_versions.json` |
| SAMTok | 官方源码 | `third_party/samtok/`（包名 `samtok`） | 源码和码本都没有改。只用它的 `samtok.models`（VQ-SAM2 编解码器），由 `src/samtok_edit21/models/codec.py` 包一层 |
| Qwen-Image-2.1 | `b3179ad3` | 权重在 `/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen-Image-2.1` | DiT、VAE、processor 原样使用 |
| Qwen3-VL-8B-SAMTok | `b78aef11` | 权重在 `/mnt/bn/strategy-mllm-train/user/tanyue/models/SAMTok/Qwen3-VL-8B-SAMTok` | 替换原版文本编码器；目录里同时带着 codec 的两个权重文件 |

运行环境由 `scripts/training/setup_env.sh` 固定：Python 3.11、torch 2.8.0（cu128）、transformers 5.12.1、accelerate 1.14.0、peft 0.20.0。

**v3 的方法代码全部在 `src/samtok_edit21/` 里，没有再改 `third_party/` 下的任何文件。** 注意力路由损失需要读 DiT 内部的量，用的是 PyTorch 的 forward pre-hook，不改 DiffSynth 源码。

### 1.3 目录与模块

| 路径 | 作用 | 来源 |
|---|---|---|
| `src/samtok_edit21/models/pipeline.py` | 加载模型、替换文本编码器、指令编码、推理入口 `edit()` | v2，v3 加了注入的接入点 |
| `src/samtok_edit21/models/codec.py` | SAMTok 编解码器的封装：mask ↔ token | v2，v3 加了 `load_codec` |
| `src/samtok_edit21/models/code_injector.py` | **码本向量注入**，以及 v3 权重的开关 | **v3 新增** |
| `src/samtok_edit21/data/highlight.py` | **指认任务**：指令模板、目标图、从输出读回区域 | **v3 新增** |
| `src/samtok_edit21/training/routing.py` | **注意力路由损失** | **v3 新增** |
| `src/samtok_edit21/training/grounding.py` | **样本准备与损失**，v3 权重的加载 | **v3 新增** |
| `src/samtok_edit21/training/grounding_engine.py` | **训练循环**（多机多卡） | **v3 新增** |
| `src/samtok_edit21/preparation/grounding_data.py` | **训练数据与评测集的构建**，含内容级泄漏防护 | **v3 新增** |
| `src/samtok_edit21/evaluation/grounding.py` | **只给 token 的评测**（三种探针） | **v3 新增** |
| `src/samtok_edit21/evaluation/grounding_report.py` | **配对分析报告** | **v3 新增** |
| `src/samtok_edit21/evaluation/gallery.py` | **训练后模型与原版的逐例对比页** | **v3 新增** |
| `src/samtok_edit21/cli.py` | 命令行入口，v3 的命令以 `ground-` 开头 | v2，v3 加了分发 |
| `scripts/grounding/` | v3 的四机入口脚本 | **v3 新增** |
| `scripts/docs/sync_code_blocks.py` | 文档代码块与源码同步 | **v3 新增** |
| `tests/test_grounding.py`、`tests/test_docs.py` | v3 的单元测试 | **v3 新增** |
| `src/samtok_edit21/training/objectives.py` | LoRA 的注入、保存、加载；v2 的 FM 损失 | v2 |
| `src/samtok_edit21/data/protocol.py`、`data/io.py` | 区域 token 的文本协议；读写与图像尺寸 | v2 |
| 其余（`models/binding.py`、`training/engine.py`、`distributed/`、`evaluation/` 的其他文件、`preparation/` 的其他文件、`regions/`） | v2 的两阶段方法、评测协议 v2、数据准备与清洗 | v2，见第 7 节 |

---

## 2. 基础层：模型加载、token 协议与官方 pipeline

这一层 v2 就有，v3 原样沿用。读懂它才能看懂第 3 节在哪里动了手。

### 2.1 模型加载与文本编码器替换

文件：`src/samtok_edit21/models/pipeline.py`，函数 `load_pipeline`。

- DiT 和 VAE 用官方的 `QwenImage21Pipeline.from_pretrained(...)` 从 Qwen-Image-2.1 目录加载，bf16。
- 文本编码器换成 `Qwen3VLForConditionalGeneration.from_pretrained(<SAMTok 目录>)`，外面包一层 `SamtokTextEncoder`；加载时检查隐藏维度是 4096、视觉 patch 是 16，保证和 Qwen-Image-2.1 的 DiT 接得上。
- processor 用 Qwen-Image-2.1 自带的，但分词器和对话模板换成 SAMTok 的（`build_processor`），并检查 `<|mt_start|>`、512 个 `<|mt_XXXX|>`、`<|mt_end|>` 这 514 个 token 都是词表里互不相同的单个 token。
- 加载完整个 pipeline 冻结（`requires_grad_(False)`、`eval()`）。

DiT 吃的是文本编码器**最后一层归一化之前**的特征。`SamtokTextEncoder.encode` 用一个临时的 forward pre-hook 在最终 norm 的入口取这份特征，取完就摘掉钩子：

<!-- code: src/samtok_edit21/models/pipeline.py::SamtokTextEncoder.encode -->
```python
def encode(self, **inputs):
    captured = []
    norm = self.model.model.language_model.norm
    handle = norm.register_forward_pre_hook(
        lambda module, args: captured.append(args[0])
    )
    try:
        self.model.model.rope_deltas = None
        output = self.model.model(**inputs, use_cache=False, return_dict=True)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(
            f"Expected one final norm invocation, got {len(captured)}"
        )
    return captured[0], output.last_hidden_state
```
<!-- /code -->

### 2.2 区域 token 的写法

文件：`src/samtok_edit21/data/protocol.py`。

- **mask token 片段**：`<|mt_start|><|mt_XXXX|><|mt_YYYY|><|mt_end|>`。SAMTok 的码本是两级残差量化，每级 256 个码；第一级的码写成 `0000`–`0255`，第二级的码加 256 后写成 `0256`–`0511`（`span_of`、`valid_span_codes`）。
- **框 token**：`<|box_start|>[x1, y1, x2, y2]<|box_end|>`，坐标是 0–1000 的相对值。v2 的 add 用它；v3 的指认训练不用框（见 3.6）。
- `spans_in(text)`、`boxes_in(text)`、`regions_in(text)` 从一段文字里取出这些片段；`SPAN_RE` 是对应的正则。

### 2.3 SAMTok 编解码器

文件：`src/samtok_edit21/models/codec.py`，类 `SamtokCodec`，辅助函数 `load_codec(samtok_dir, device)`。

- `codec.encode(image, [mask, ...])` → 每个 mask 一个 token 片段。交互式场景下用户给的 mask 就是这样变成 token 的。
- `codec.decode_strict(image, text)` → 按顺序把文字里的每个片段解码回像素 mask。**"token 指的区域"在整个项目里的定义就是它**：训练目标、损失权重、评测打分都以这个解码结果为准，而不是原始标注，所以监督和 token 天然一致。
- codec 始终是 fp32。它的码本（`codec.vq.quantizer.codebooks[0/1].weight`）就是 3.2 节注入用的向量来源。

### 2.4 指令编码与 DiT 的联合序列

文件：`src/samtok_edit21/models/pipeline.py`。

- `resize_sources(pipe, images, height, width)`：用官方 `QwenImage21Unit_EditImageEmbedder` 把源图缩放到与目标画布匹配的尺寸。
- `encode_edit(pipe, prompt, images, return_ids=False)`：先检查每个 mask 片段确实被切成 4 个 token，再调用官方 `QwenImage21Unit_PromptEmbedder().process(...)`。返回 `prompt_embeds`（逐 token 特征）、`edit_image_pad_mask`（哪些位置是源图占位 token），`return_ids=True` 时还返回 `prompt_input_ids`（裁掉系统提示后的 token id，本项目加的可选输出，3.4 节用它定位片段）。

DiT 是单流结构（32 个 block、宽度 4096、32 个头），把所有东西排成一条**联合序列**（`third_party/diffsynth/diffsynth/models/qwen_image_21_dit.py` 的 `QwenImage21DiT.forward`）：

```
[ 文本前缀 | 源图 latent | 指令文字（其中 4 个位置是 mask 片段） | 目标图 latent ]
```

- 文本编码器输出里的每个源图占位 token 展开成 4 个 DiT token，这些位置填入源图 latent；目标图的 token 接在最后。
- 时间步调制只作用在目标 token 上，前面的条件部分恒为 t=0；注意力是按块因果的：前缀看不到目标，目标能看到全部前缀。所以推理时前缀只算一次、缓存 K/V（`use_kv_cache=True`），之后每一步只算目标 token。
- **含义**：mask 片段的 4 个 token 在序列里的位置是固定的，目标 token 通过注意力读它。3.3 节的路由损失监督的就是这条"目标 token → 片段"的注意力。

### 2.5 推理入口 `edit()`

`edit(pipe, instruction, images, mode=..., binding=..., blend_region=..., **kwargs)` 是所有推理的统一入口。v3 的"只给 token"推理就是其中的 `mode="inline"`（指令里已经带着 token），不传绑定、不传融合区域：

<!-- code: src/samtok_edit21/evaluation/grounding.py::generate -->
```python
@torch.no_grad()
def generate(pipe, codec, prompt, source_rgb, *, steps=20, seed=0, area=CANVAS_AREA, cfg_scale=1.0, negative_prompt=None,
             blend_region=None, blend_dilate=2, blend_feather=1.0):
    """Token-only inference: inline region tokens, raw text encoder, no binding, CFG 1 (or token guidance: ``cfg_scale``
    with a token-free ``negative_prompt``).  No blending, unless ``blend_region`` (a pixel mask, teacher test only)
    asks to keep the source outside it."""
    from samtok_edit21.models.binding import BindingConfig
    from samtok_edit21.models.pipeline import edit

    height, width = canvas(source_rgb, area)
    extra = {} if negative_prompt is None else {"negative_prompt": negative_prompt}
    image, _ = edit(pipe, prompt, [source_rgb.convert("RGBA")], mode="inline", height=height, width=width,
                    num_inference_steps=steps, cfg_scale=cfg_scale, seed=seed, use_kv_cache=True, pass2_te="raw",
                    binding=BindingConfig("none"), codec=codec, blend_region=blend_region, blend_dilate=blend_dilate,
                    blend_feather=blend_feather, **extra)
    return export(image, source_rgb.size), (height, width)
```
<!-- /code -->

`mode="online"` / `"oracle"` 是 v2 的两遍推理（先让文本编码器生成区域，再编辑），`binding` 和 `blend_region` 是 v2 的结构性绑定与 latent 融合，v3 都不用。

### 2.6 LoRA 的注入、保存与加载

文件：`src/samtok_edit21/training/objectives.py`。

- `add_adapter(model, "stage2", rank, dropout)`：用 PEFT 给 DiT 注入 LoRA，目标模块用 DiffSynth 的自动检测结果（`to_q`、`to_k`、`to_v`、`to_out.0`、`proj`、`out`、`gate_layer`），alpha 等于 rank；可训练参数转成 fp32。
- `save_adapter(model, directory, config)` 写 `adapter.safetensors` 和 `adapter.json`（含 rank、目标模块和配方指纹）；`load_adapter(model, directory, trainable=False)` 按 `adapter.json` 重新注入并核对每个张量的名字和形状。

---

## 3. v3 方法的实现

### 3.1 指认任务：让输出本身就是区域

文件：`src/samtok_edit21/data/highlight.py`。

训练行有三种，都用同一种方式指代区域——带物体名时写 `<短语> <M>`，不带时写 `this region <M>`（`region_reference`）：

| 类型（`kind`） | 指令 | 目标图 |
|---|---|---|
| `hl` 指认 | "Apply a red semi-transparent mask to X and keep everything else unchanged." 等 4 个模板 × 8 种颜色 | 源图在 token 所指区域上叠 50% 透明度的颜色 |
| `proc` 程序化局部编辑 | "Convert X to grayscale …" 等 5 种（灰度、提亮、压暗、模糊、反色） | 只在区域内做这个精确变换 |
| `real` 真实编辑 | 数据集原本的指令 | 数据集的目标图 |

指认和程序化编辑的目标图在训练时现场生成，画在**模型实际看到的那张缩放后的源图**上，所以区域外与源图逐像素相同：

<!-- code: src/samtok_edit21/data/highlight.py::render_target -->
```python
def render_target(prepared_rgb, mask, row):
    """Target image of a highlight or procedural row, built on the resized source the model is conditioned on."""
    rgb = np.asarray(prepared_rgb.convert("RGB"), np.float32)
    weight = resize_mask(mask, prepared_rgb.size)
    if row["kind"] == "hl":
        out = overlay(rgb, weight, PALETTE[row["color"]])
    elif row["kind"] == "proc":
        out = procedural_edit(rgb, weight, row["op"])
    else:
        raise ValueError(f"Row kind {row['kind']!r} has a real target image")
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))
```
<!-- /code -->

同一个定义反过来就是评测用的**读回**：让模型生成一张指认图，再从输出里把涂色区域恢复出来，和 token 的区域算 IoU。对精确叠色，`输出 − 源图` 在 `颜色 − 源图` 方向上的投影就是透明度：

<!-- code: src/samtok_edit21/data/highlight.py::extract_mask -->
```python
def extract_mask(source_rgb, output_rgb, color, threshold=0.5):
    """Recover the highlighted region from a generated image.

    For an exact overlay ``out = src + alpha (color - src)``; projecting ``out - src`` on ``color - src`` gives alpha,
    scaled here so a 50% overlay reads 1.  Pixels already close to the colour carry no signal and read 0.
    """
    source = np.asarray(source_rgb, np.float32) / 255.0
    output = np.asarray(output_rgb.resize(source_rgb.size, Image.BILINEAR), np.float32) / 255.0
    direction = np.asarray(color, np.float32)[None, None] / 255.0 - source
    energy = (direction * direction).sum(-1)
    alpha = ((output - source) * direction).sum(-1) / np.maximum(energy, 1e-3) / OVERLAY_ALPHA
    alpha[energy < 0.02] = 0.0
    smooth = Image.fromarray((alpha.clip(0, 1) * 255).astype(np.uint8)).filter(ImageFilter.BoxBlur(2))
    return np.asarray(smooth, np.float32) / 255.0 > threshold
```
<!-- /code -->

评测时颜色取离图像平均色最远的一种（`pick_color`），避免源图本身就是那个颜色。

### 3.2 码本向量注入

文件：`src/samtok_edit21/models/code_injector.py`。

**动机。** 文本编码器只把 `<|mt_0097|>` 当作词表里的一个 id；而这个码在 SAMTok 码本里对应一个 256 维向量，两级向量之和就是量化后的 mask 表示。与其让 DiT 去硬记 512 个 id 的含义，不如把这些向量直接送到它面前。

**做法。** 对一个片段的 4 个位置：

| 位置 | 加到文本编码器输出上的量 |
|---|---|
| `<|mt_start|>` | 无 |
| 第一级码 | `W0 · e0 + b0` |
| 第二级码 | `W1 · e1 + b1` |
| `<|mt_end|>` | `W2 · (e0 + e1) + b2`，即量化后的 mask 表示 |

三个线性层（256 → 4096）**全部零初始化**，所以没训练时模型与原版完全相同。

<!-- code: src/samtok_edit21/models/code_injector.py::CodeInjector.forward -->
```python
def forward(self, input_ids):
    """``[B, L]`` token ids -> ``[B, L, hidden]`` correction (zero away from mask spans)."""
    index = self.code_index(input_ids)
    length = index.shape[1]
    level0 = (index >= 0) & (index < CODES_PER_LEVEL)
    level1 = index >= CODES_PER_LEVEL
    book = self.codebook

    def rows(position, level):
        low = level * CODES_PER_LEVEL
        return book[position.clamp(low, low + CODES_PER_LEVEL - 1)]

    delta = self.proj[0](rows(index, 0)) * level0[..., None] + self.proj[1](rows(index, 1)) * level1[..., None]
    pad = index.new_full((index.shape[0], 1), -1)
    previous = torch.cat([pad, index], 1)[:, :length]
    before = torch.cat([pad, pad, index], 1)[:, :length]
    closes = ((input_ids.to(index.device) == self.end_token_id)
              & (before >= 0) & (before < CODES_PER_LEVEL) & (previous >= CODES_PER_LEVEL))
    return delta + self.proj[2](rows(before, 0) + rows(previous, 1)) * closes[..., None]
```
<!-- /code -->

- `CodeInjector.from_codec(codec, tokenizer)` 从 codec 里取出两级码本拼成 `[512, 256]`，并建立 token id → 码本行号的查找表。
- `<|mt_end|>` 只有紧跟在"第一级码 + 第二级码"之后才加量，孤立的结束符不处理。

**接入点。** `attach_code_injector(pipe, injector)` 把模块挂到 `pipe.text_encoder.code_injector`，`SamtokTextEncoder.forward` 在输出上加它。文本编码器主干放在 `no_grad` 里，梯度只流进注入模块：

<!-- code: src/samtok_edit21/models/pipeline.py::SamtokTextEncoder.forward -->
```python
def forward(self, **inputs):
    injector = self.code_injector
    if injector is None or not injector.enabled:
        return self.encode(**inputs)[0]
    # The side channel is trained against a frozen backbone: no graph through the text encoder,
    # gradients reach only the injector's projections.
    with torch.no_grad():
        hidden = self.encode(**inputs)[0]
    return hidden + injector(inputs["input_ids"]).to(hidden.dtype)
```
<!-- /code -->

因为注入加在文本编码器的输出上，官方的指令编码流程（裁剪系统提示、拼接、缓存前缀）完全不用改，训练和推理走同一段代码。

**保存与加载。** `save_code_injector` 写 `code_injector.safetensors` 和 `code_injector.json`；后者记录码本和 token 映射的指纹（`CodeInjector.fingerprint`），加载时指纹不符就拒绝，防止把权重用到另一个码本上。`load_code_injector` 也能读 E-A2 当时的保存格式 `ea2_extra.pt`。

**开关。** `trained_weights(pipe, enabled)` 是一个上下文管理器，同时开关 DiT 的 LoRA 和注入模块。`enabled=False` 时就是"同一个文本编码器下的原版模型"，所有配对比较的基线都这样得到，不需要加载两份模型。

### 3.3 注意力路由损失

文件：`src/samtok_edit21/training/routing.py`。

**要监督什么。** FM 损失只奖励"图对不对"，从不告诉模型某个片段对应哪里。路由损失直接监督这条联系：在选定的几层上，**区域内的目标 token 对片段的注意力，应当高于区域外的目标 token**；落在同图其他实例上的目标 token 是最重要的反例。

**读的是模型自己的量。** 被监督的是 block 内部真正送进注意力的 query 和 key 的内积（softmax 之前的 logit）。`span_attention_logits` 按 block 的 `forward` 原样重算到 query/key 为止——norm1、时间步调制、带 LoRA 的 `to_q` / `to_k`、RMS norm、RoPE——然后对头和片段的 4 个 token 取平均：

<!-- code: src/samtok_edit21/training/routing.py::span_attention_logits -->
```python
def span_attention_logits(block, hidden_states, modulation, target_token_mask, rotary_emb, n_target, span_positions):
    """Mean attention logit from each target token to the span tokens, as this block computes it.

    Mirrors ``QwenImage21TransformerBlock.forward`` up to the query/key that enter attention (norm1, timestep
    modulation, to_q/to_k with their LoRA, RMS norm, RoPE).  The target tokens are the last ``n_target`` positions of
    the joint sequence.  Returns ``[n_target]`` for batch element 0: the scaled dot product averaged over heads and
    over the span's tokens.
    """
    attention = block.attn
    scale_shift = modulation.chunk(2, dim=-1)[0]
    states, _ = block._modulate(block.img_norm1(hidden_states), scale_shift, target_token_mask)
    start = states.shape[1] - n_target
    query = attention.norm_q(attention.to_q(states[:, start:]).unflatten(-1, (attention.heads, -1))).to(states.dtype)
    query = apply_rotary_emb_qwen(query, rotary_emb[start:], use_real=False)
    key = attention.norm_k(attention.to_k(states[:, span_positions]).unflatten(-1, (attention.heads, -1))).to(states.dtype)
    key = apply_rotary_emb_qwen(key, rotary_emb[span_positions], use_real=False)
    logits = torch.einsum("bnhd,bshd->bn", query.float(), key.float())[0]
    return logits / (math.sqrt(query.shape[-1]) * query.shape[2] * key.shape[1])
```
<!-- /code -->

测试 `test_routing_logit_is_the_blocks_own_attention_logit` 截获 block 内部真实的 query/key 做了核对，两者一致。

**标签和权重**（`route_targets`）。正例是区域覆盖率超过 0.5 的目标 token，其余是反例；正、反两类各占一半总权重，这样小区域不会被淹没；落在其他实例上的反例权重乘 3：

<!-- code: src/samtok_edit21/training/routing.py::route_targets -->
```python
def route_targets(region, distractors, distractor_weight=DISTRACTOR_WEIGHT):
    """Labels and balanced weights on the target latent grid.

    ``region`` / ``distractors`` are coverage maps in [0, 1].  Positives (inside the region) and negatives (outside)
    each carry half of the total weight, so a small region is not drowned out; negatives lying on other instances of
    the image count ``distractor_weight`` times.
    """
    labels = (region > 0.5).float().flatten()
    negative = 1 - labels
    weight_neg = negative + (distractor_weight - 1) * (distractors > 0.5).float().flatten() * negative
    weights = torch.zeros_like(labels)
    if labels.sum() > 0:
        weights = weights + labels * (0.5 / labels.sum())
    if weight_neg.sum() > 0:
        weights = weights + weight_neg * (0.5 / weight_neg.sum())
    return labels, weights
```
<!-- /code -->

**损失。** 每层的 logit 先减去均值、乘 4，再过一个可学习的逐层仿射（初值 scale=1、bias=0），最后做带权的二元交叉熵。仿射的作用是让损失去适应每层 logit 的量级，而不是逼注意力本身取某个绝对值。各层的损失取平均。

**怎样挂到 DiT 上。** `AttentionRouter` 给选定的 block 注册 forward pre-hook，不改 DiffSynth 源码，也不改变前向结果（钩子只读不写）：

<!-- code: src/samtok_edit21/training/routing.py::AttentionRouter._hook -->
```python
def _hook(self, index):
    def before_block(block, args, kwargs):
        context = self._context
        if context is None:
            return
        for entry in context["entries"]:
            if entry.get("labels") is not None:
                logits = span_attention_logits(
                    block, kwargs["hidden_states"], kwargs["modulation"], kwargs["target_token_mask"],
                    kwargs["rotary_emb"], context["n_target"], entry["span"])
                scale, bias = self.affine[index]
                calibrated = scale * (logits - logits.mean()) * self.logit_scale + bias
                term = F.binary_cross_entropy_with_logits(
                    calibrated, entry["labels"], weight=entry["weights"], reduction="sum")
                if self._recording:
                    self._terms.append(term)
            if entry.get("source_labels") is not None and self.read_affine is not None:
                logits = span_read_logits(
                    block, kwargs["hidden_states"], kwargs["modulation"], kwargs["target_token_mask"],
                    kwargs["rotary_emb"], entry["span"], entry["source_positions"])
                scale, bias = self.read_affine[index]
                calibrated = scale * (logits - logits.mean()) * self.logit_scale + bias
                term = F.binary_cross_entropy_with_logits(
                    calibrated, entry["source_labels"], weight=entry["source_weights"], reduction="sum")
                if self._recording:
                    self._read_terms.append(term)

    return before_block
```
<!-- /code -->

**与梯度检查点配合的注意事项。** DiT 训练时每个 block 都走非重入的梯度检查点（`torch.utils.checkpoint`，`use_reentrant=False`），`backward` 时会把 block 再执行一遍。钩子在第二遍里必须做完全相同的运算，否则 autograd 会发现保存的张量数量对不上而报错。所以：
- `arm(...)` 之后钩子一直生效，直到 `backward` 完成后调用 `disarm()`；
- 只有在 `with router.recording():` 里（也就是第一遍前向）才把损失项记下来，第二遍只重算不记录。

用法固定为：`arm` → `with recording(): 前向` → `collect()` 取损失 → `backward` → `disarm()`。测试 `test_router_gradients_survive_checkpointing_and_disarm_is_inert` 验证了开、关检查点时梯度一致，以及钩子不改变前向结果。

默认层是第 6、10、14、18、22、26 层（`DEFAULT_LAYERS`）。

### 3.4 样本准备与损失

文件：`src/samtok_edit21/training/grounding.py`。

`prepare_sample(pipe, codec, row)` 把一行数据变成一次前向需要的全部东西：

1. 用 codec 把这一行的 token 解码成区域；把 `others`（同图其他 token，最多 3 个）也解码，并集去掉本区域后就是**干扰实例区域**。
2. 确定画布（约 1MP，边长取 32 的倍数），用官方流程缩放源图；指认和程序化编辑现场生成目标图，真实编辑读数据集的目标图。
3. VAE 编码源图和目标图；`encode_edit` 编码指令（这一步带梯度，但只有注入模块可训练）。
4. 把区域换算到目标 token 网格上：每个目标 token 对应 16×16 像素，取覆盖率（`grid_coverage`）。
5. 算出片段的 4 个 token 在联合序列里的位置（`span_token_positions`）。指令里的位置来自 `prompt_input_ids`；每个源图占位 token 在 DiT 里展开成 4 个，所以要按展开后的长度累加偏移。

<!-- code: src/samtok_edit21/training/grounding.py::span_token_positions -->
```python
def span_token_positions(tokenizer, prompt, conditioning):
    """Positions of the single mask span in the DiT's joint sequence.

    ``prompt_input_ids`` index the trimmed text-encoder sequence; every image placeholder token there expands to four
    DiT tokens (``QwenImage21DiT.forward``), all of which precede the instruction.
    """
    ids = conditioning["prompt_input_ids"][0]
    layout = token_layout(tokenizer, ids[ids >= 0], prompt)
    if len(layout["units"]) != 1 or layout["units"][0][1] - layout["units"][0][0] != 4:
        raise ValueError("A grounding row carries exactly one mask-token span")
    start, end = layout["units"][0]
    repeats = torch.where(conditioning["edit_image_pad_mask"][0], 4, 1)
    offsets = (repeats.cumsum(0) - repeats).tolist()
    return [offsets[index] for index in range(start, end)]
```
<!-- /code -->

`grounding_loss(pipe, sample, router)` 的损失是

```
loss = w(t) · ( FM + λ_region · FM_region + λ_distractor · FM_distractor ) + λ_route · routing
```

- 时间步采样、加噪、目标和权重 `w(t)` 与官方 `FlowMatchSFTLoss` 完全相同；
- `FM` 是逐 token 平方误差的全图平均；
- `FM_region` 是同一份误差只在区域内（向外扩一个 token）取平均，这样小区域和大区域一样重要；
- `FM_distractor` 是在干扰实例上取平均，这些位置的目标就是源图本身，等于要求"别的实例不许动"；
- `routing` 来自 3.3 节。

<!-- code: src/samtok_edit21/training/grounding.py::region_terms -->
```python
def region_terms(squared_error, region, distractors):
    """Region-normalised and distractor-normalised means of a per-token squared error map."""
    mean = squared_error.mean()
    weight = F.max_pool2d(region[None, None].float(), 3, 1, 1)[0, 0]   # one token of context around the region
    in_region = (squared_error * weight).sum() / weight.sum() if float(weight.sum()) > 0 else mean.detach() * 0
    other = (distractors > DISTRACTOR_COVERAGE).float()
    on_distractors = (squared_error * other).sum() / other.sum() if float(other.sum()) > 0 else mean.detach() * 0
    return mean, in_region, on_distractors
```
<!-- /code -->

`timestep_index` 和 `noise` 两个可选参数用来固定随机性，供测试使用。

### 3.5 训练循环

文件：`src/samtok_edit21/training/grounding_engine.py`，命令 `samtok-edit ground-train`，在 `torchrun` 下运行。

**并行方式。** 每张卡是一个进程，各自处理自己的样本；每次更新时把各卡的梯度取平均，每张卡执行同一步优化。有效 batch = 卡数 × `--accumulation`。

- 数据顺序由数据目录里的 `schedule.json` 决定，第 r 张卡取 `order[r::world]`。所以**总卡数 × accumulation 相同，每次更新用到的样本就完全相同**：8 卡 × 4 和 32 卡 × 1 是同一批样本。
- 梯度平均（`average_gradients`）：把全部可训练参数的梯度拼成一条向量做一次 all-reduce。

**各卡权重必须一致。** LoRA 的 A 矩阵是随机初始化的。模型建好后，把 0 号卡的全部可训练参数广播给所有卡；开始时和每次保存权重时核对各卡参数的校验和，不一致就报错停下：

<!-- code: src/samtok_edit21/training/grounding_engine.py::broadcast_parameters -->
```python
def broadcast_parameters(parameters, dist):
    """Give every rank rank 0's values (LoRA A matrices are random and would otherwise differ per rank)."""
    flat = torch.cat([p.detach().flatten().float() for p in parameters])
    dist.broadcast(flat, src=0)
    offset = 0
    with torch.no_grad():
        for p in parameters:
            p.copy_(flat[offset:offset + p.numel()].view_as(p).to(p.dtype))
            offset += p.numel()
```
<!-- /code -->

（E-A1、E-A2 当时的脚本没有这一步，各卡的初始化不一致，见文档 03 §2.3。`--unsynced-init` 能重现当时的行为，只用于回归核对。）

**优化设置。** AdamW，学习率线性预热后保持不变；LoRA 用权重衰减，注入模块和路由仿射不用。每 10 次更新往 `train_metrics.jsonl` 写一行各项损失的平均值；在 `--quick-eval` 指定的更新次数上，用数据目录里的 160 条快速评测项跑一次指认评测，写进 `quick_summary.jsonl`。

**每个保存点写什么**（`<output>/adapter_<更新次数>/`）：

| 文件 | 内容 |
|---|---|
| `adapter.safetensors`、`adapter.json` | DiT 的 LoRA；`adapter.json` 的 `grounding` 字段记录完整训练配方、路由仿射的值、各卡一致性校验结果 |
| `code_injector.safetensors`、`code_injector.json` | 码本向量注入的三个投影，以及码本指纹 |

另外 `<output>/resume.pt` 保存优化器状态用于 `--resume` 续跑，`<output>/TRAIN_DONE` 标记完成并记下最终权重目录名。

**主要参数**：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--data` / `--output` | 必填 | 数据目录 / 训练输出目录 |
| `--updates`、`--accumulation` | 1300、4 | 优化步数；每卡每步的样本数 |
| `--lr`、`--warmup`、`--weight-decay` | 1e-4、33、0.01 | |
| `--rank` | 32 | 新建 LoRA 的 rank |
| `--init-adapter` | 无 | 从一个已有的 v3 权重目录热启动（阶段 B 用）：加载 LoRA、码本向量注入，路由层相同时还恢复路由仿射 |
| `--lambda-route`、`--lambda-region`、`--lambda-distractor` | 0.05、1、1 | 三项损失的系数；`--lambda-route 0` 时不挂钩子 |
| `--route-layers` | 6,10,14,18,22,26 | 路由损失加在哪些 block |
| `--no-code-injector` | 关 | 消融：不用码本向量注入 |
| `--checkpoints`、`--quick-eval` | 325,650,975,1300；0,325,650,975,1300 | 保存点；快速评测点 |
| `--seed` | 2000 | 第 r 张卡的时间步和噪声用 `seed + r` |
| `--resume` | 关 | 从 `resume.pt` 续跑 |

### 3.6 训练数据与评测集的构建

文件：`src/samtok_edit21/preparation/grounding_data.py`，命令 `samtok-edit ground-data train` 和 `samtok-edit ground-data eval-set`。数据来源和各文件的清单见 [04_SAMTokEdit_Qwen21_数据工作.md](archive/v3_process/04_SAMTokEdit_Qwen21_数据工作.md)。

**基本单位是"一张图上的一个 token"（unit）。** 来源有两处：
- v2 的 stage-2 清单里只带一个 mask 片段、不带框的行（`single_token_rows`）。add 行选的是"新物体放哪里"，不是一个已有的实例，所以不用。
- GRES 指代数据（parquet 里的 `answer` 和内嵌图片）。它的图通常有多个带名字的实例，是"同图多实例"的主要来源。

每个 unit 记下同一张图上的其他 token（`others`，最多 3 个），训练时它们就是干扰实例。然后每个 unit 生成指认行、程序化编辑行；清洗后判定保留的 RefEdit / Derived 真实编辑对生成真实编辑行。

**泄漏防护。** 预留集按源图**路径**的哈希划分（`heldout`），但同一张图会以不同路径出现（同图多条指令、同一批图存了两份），只按路径划分会泄漏。所以构建时还按**图像内容**排除：训练源图的 16×16 均值哈希只要与任何一张评测图相同就丢掉，排除后再核对一次，仍有重合就报错。

<!-- code: src/samtok_edit21/preparation/grounding_data.py::average_hash -->
```python
def average_hash(image):
    gray = np.asarray(image.convert("L").resize((16, 16), Image.BILINEAR), np.float32)
    return np.packbits((gray > gray.mean()).flatten()).tobytes().hex()
```
<!-- /code -->

**确定性。** 行的顺序、颜色、模板、操作都由带种子的随机数决定，随机数的调用顺序是数据集定义的一部分。同样的输入和种子会产出逐字节相同的文件（E-A2 的数据就是这样复现核对的）。

**输出目录的内容**：

| 文件 | 内容 |
|---|---|
| `epoch_rows.jsonl` | 一轮的全部训练行 |
| `schedule.json` | 各轮打乱后的行号序列，训练按它取样 |
| `eval.json` | 评测项；可带 `leaky_<名字>` 标记，表示该图在某个更早的训练集里出现过 |
| `eval_quick_idx.json` | 训练中快速评测用的评测项下标 |
| `gres_images/` | 从 GRES parquet 里取出的图片；行里用相对路径引用，所以数据目录可以整体搬动 |
| `data_stats.json` | 统计和输入路径 |

**训练行的格式**（一行一个 JSON）：

```json
{"kind": "hl", "edit_image": "<源图路径>", "span": "<|mt_start|>…<|mt_end|>", "others": ["…"], "color": "red", "variant": "noref", "prompt": "Apply a red semi-transparent mask to this region <|mt_start|>…<|mt_end|> and keep everything else unchanged."}
{"kind": "proc", "edit_image": "…", "span": "…", "others": [], "op": "blur", "variant": "ref", "prompt": "Blur the left cup <|mt_start|>…<|mt_end|> and keep everything else unchanged."}
{"kind": "real", "edit_image": "…", "image": "<目标图路径>", "span": "…", "prompt": "<数据集指令>", "variant": "ref", "edit_type": "remove", "others": []}
```

**评测项的格式**：预留集项带 `span`；benchmark 项带 `gt_mask`（用户 mask，评测时现场编码成 token）；反事实对的项带 `span` 和 `other`（同图的另一个 token）。

### 3.7 评测、报告与对比页

**评测**（`src/samtok_edit21/evaluation/grounding.py`，命令 `samtok-edit ground-eval`，在 `torchrun` 下运行）。三种探针喂给模型的都只有源图和带 token 的指令，mask 只在事后打分时用：

| 探针 | 做法 | 指标 |
|---|---|---|
| 指认 | 生成指认图，读回区域 | 与 token 区域的 IoU（benchmark 项对用户 mask 算）；反事实对上看读回的区域更像自己的实例还是另一个 |
| remove 探针 | 同一张图、同一句 "Remove the object in this region ⟨M⟩."，只换 token | 改动落在自己实例上是否多于另一个实例；两个 token 都对才算这一对"跟随" |
| 编辑的区域内外改动 | benchmark 的真实指令 | 区域内、远离区域处的平均像素改动 |

给了 `--adapter` 时，remove 探针和编辑探针会在"训练后"和"原版"（用 `trained_weights` 关掉 LoRA 和注入）下各跑一次；不给 `--adapter` 就是评测原版模型，得到所有配对比较的基线。

**报告**（`grounding_report.py`，命令 `samtok-edit ground-report`）。所有比较都是配对的：只用本次运行和所有参考运行都有的评测项，差值给出 bootstrap 95% 区间；预留集另报"无泄漏子集"；最后逐条给出放行线是否通过。用 `--reference 名字=路径` 指定参考评测，第一个是原版。

**对比页**（`gallery.py`，命令 `samtok-edit ground-gallery generate` 和 `html`）。对 benchmark 的真实指令和各编辑类型的预留样本，各用原版和训练后模型生成一次，生成一个可筛选的页面和一个图片内嵌的单文件。

### 3.8 用 v3 权重做推理

Python：

```python
from PIL import Image
from samtok_edit21 import edit, load_codec, load_grounding_adapter, load_pipeline
from samtok_edit21.models.binding import BindingConfig

pipe = load_pipeline(device="cuda")                      # Qwen-Image-2.1 DiT/VAE + SAMTok 文本编码器
codec = load_codec("/mnt/bn/strategy-mllm-train/user/tanyue/models/SAMTok/Qwen3-VL-8B-SAMTok", "cuda")
load_grounding_adapter(pipe, codec, "<权重目录>")         # DiT LoRA + 码本向量注入
pipe.eval()

source = Image.open("source.png").convert("RGBA")
span = codec.encode(source, [mask])[0][0]                # 用户给的布尔 mask -> token 片段
image, _ = edit(pipe, f"Remove the object in this region {span}.", [source], mode="inline",
                height=1024, width=1024, num_inference_steps=40, cfg_scale=1.0, seed=0,
                pass2_te="raw", binding=BindingConfig("none"), codec=codec, blend_region=None)
```

命令行（`--mask` 给一张 mask 图，程序把它编码成 token 写进指令）：

```bash
samtok-edit infer --mode interactive --image source.png --mask mask.png \
  --prompt "Remove the object in this region." --dit-adapter <权重目录> --output out.png
```

`samtok-edit infer` 看到权重目录里有码本向量注入的权重时，会自动走 `load_grounding_adapter`。

---

### 3.9 第二轮代码（`round2-*`，C1、C1b、C2）

第二轮在第一轮的基础层、码本注入和路由代码上，新增了一套数据、训练、评测和报告模块。方法和设计理由见[文档 02](02_SAMTokEdit_Qwen21_v3方法.md) §4，设置和结果见[文档 03](03_SAMTokEdit_Qwen21_v3实验.md) §3、§5。

| 文件 | 内容 |
|---|---|
| `src/samtok_edit21/preparation/round2_data.py` | 开发划分、SA-1B dense mask 区域池、token 缓存；C1/C1b/C2 训练行（指认、近邻反例、反事实成对、天然孪生、50k 拼合多目标、v1/D4 编辑行与漂移判断）；C3 训练行（`build_rows_c3`：按 C2 比例抽行、去掉做了一半的样本、D10 定向子集）；合板数据的取行（`build_rows_merged`：按 JSON 配置从默认视图抽行，见文档 01 §2.13）；开发评测集 `eval_c1.json` |
| `src/samtok_edit21/preparation/merge_data.py` | 合板（文档 01 §2.13）：五批数据统一为 D3/D4 单元格式，token 内嵌，文字标记，源图 SHA-1 分组，默认视图的排除规则（`hard_exclusions`、`apply_caps`），图片硬链接，`units.parquet`、`stats.json`、`build.json` |
| `src/samtok_edit21/training/round2.py` | 每类训练行的样本准备（`_prepare_*`；D3/D4 格式的单条与多目标记录共用 `_prepare_record`、`_prepare_multi_record`，C3 起也读 D10 的 `edit_ext`、`edit_multi_ext`；合板数据的单条与多目标记录都用 `edit_merged`：`merged_regions`、`_prepare_merged`，只加载行里用到的记录）、多 span 的位置查找、路由标签（`token_labels`，无 token 覆盖率 > 0.5 时取最大者）、`round2_loss`（FM + 区域 + 邻居 + 写/读路由；`edit_region_dilate=False` 时编辑行的区域项不外扩）、`unit_change`、局部损失 `loss_mask` |
| `src/samtok_edit21/training/round2_engine.py` | 训练循环：`--init-adapter` 热启动、余弦学习率、`--heavy-weight`、`--edit-region-dilate`（C3-M）、按行号的确定性分片、快速评测、W&B |
| `src/samtok_edit21/training/routing.py` | 第一轮的写方向路由，加上读方向（⟨M⟩→源图）和读方向仿射 `read_affine`；两个方向都是带权 BCE |
| `src/samtok_edit21/training/tracking.py` | 多卡下的 W&B 记录（`DistributedTracker`） |
| `src/samtok_edit21/evaluation/round2_eval.py` | 开发集五类探针：单 token、多 token、换 token 成对、编辑、remove 探针 |
| `src/samtok_edit21/evaluation/round2_report.py` | 配对报告：bootstrap 区间、扣漂移的不执行率（`NOOP_MARGIN = 5`）、按面积分桶；`--wandb` 把开发集结果写进训练运行的 summary |
| `src/samtok_edit21/preparation/instruction_protocol_v2.py` | 协议 2.0 共享模块（与 benchmark 2.6.0 逐字节相同）；`render(spec, has_ref, selector_token=span)` 把 ⟨M⟩ 写进句子 |
| `scripts/round2/` | 四机入口 `bootstrap_arnold.sh`、`run_arnold.sh`、运行设置 `settings/`（见文档 05 第 7 节） |

命令：

| 命令 | 作用 | 是否多卡 |
|---|---|---|
| `round2-data <step>` | 建第二轮数据：`pool`（开发划分、区域池、token 缓存，多卡）、`rows`（C1）、`rows-c1b`、`rows-c2`、`rows-c3m`、`rows-c3d`、`rows-merged --spec <配置>`（合板数据）、`quick-c2`、`eval`（开发评测集） | `pool` 多卡 |
| `merge-data --out <目录>` | 合板：写出 `datasets/SAMTokEdit_Merged_20261024/` 这样的数据目录（`--limit N` 每个来源只取前 N 条，试跑用） | 否（多进程） |
| `round2-train` | 训练 | 是 |
| `round2-eval` | 在开发集上评测一个权重，或不给权重时评测原版 | 是 |
| `round2-report` | 配对报告 | 否 |

与第一轮默认值的差别：
- 可训练参数多读方向仿射 12 个，共 87,044,120；
- 学习率可选余弦（`--lr-schedule cosine --lr-min 1e-5`）；
- 重反例权重用 `--heavy-weight` 设置（C1 为 3，C1b/C2 为 8）；
- 默认 2,000 次更新、预热 50 次。

C2 实际用的完整参数见 `scripts/round2/settings/c2_4n.sh`。

---

## 4. 命令一览

所有命令既可以写成 `samtok-edit <命令>`，也可以写成 `python -m samtok_edit21 <命令>`；需要多卡的命令放在 `python -m torch.distributed.run --nproc_per_node 8 -m samtok_edit21 <命令>` 里。

| 命令 | 作用 | 是否多卡 |
|---|---|---|
| `ground-data eval-set` | 建评测集 | 否 |
| `ground-data train` | 建训练数据目录 | 否（多进程算图像哈希） |
| `ground-train` | 训练 | 是 |
| `ground-eval` | 评测一个权重目录，或不给权重时评测原版 | 是 |
| `ground-report` | 出配对分析报告 | 否 |
| `ground-gallery generate` / `html` | 生成对比图 / 建对比页 | 是 / 否 |
| `infer` | 单张推理（v2 就有，已支持 v3 权重） | 否 |

单机 8 卡从数据到报告的完整例子：

```bash
V3=/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v3
RUN=$V3/runs/<新的 run id>
python -m torch.distributed.run --nproc_per_node 8 -m samtok_edit21 ground-train \
  --data $V3/data/stageA_ea2_20261012 --output $RUN/train
python -m torch.distributed.run --nproc_per_node 8 -m samtok_edit21 ground-eval \
  --data $V3/data/stageA_ea2_20261012 --adapter $RUN/train/adapter_1300 --output $RUN/eval
samtok-edit ground-report --eval $RUN/eval --data $V3/data/stageA_ea2_20261012 --train $RUN/train \
  --name <名字> --reference stock=$V3/baselines/stock_highlight_eval --output $RUN/analysis.txt
```

四机入口是 `scripts/grounding/bootstrap_arnold.sh`（粘贴到 ARNOLD 的启动命令）和它调用的 `scripts/grounding/run_arnold.sh`，用法见 [05_SAMTokEdit_Qwen21_四机实验运行指南.md](05_SAMTokEdit_Qwen21_四机实验运行指南.md)。

---

## 5. 默认超参数（第一轮，E-A2 的配方；第二轮见 3.9 节和文档 03）

| 项 | 值 |
|---|---|
| 基座 | 原版 Qwen-Image-2.1 DiT，SAMTok 文本编码器，不加绑定、不融合 |
| 可训练参数 | 87.04M：LoRA 83.89M（rank 32，alpha 32）+ 注入 3.16M + 路由仿射 12 |
| 优化 | AdamW，lr 1e-4，预热 33 步后不变；LoRA 权重衰减 0.01，其余 0 |
| 规模 | 1300 次更新，有效 batch 32 |
| 损失系数 | λ_region 1，λ_distractor 1，λ_route 0.05 |
| 路由 | 第 6/10/14/18/22/26 层；logit 乘 4；其他实例上的反例权重 3 |
| 分辨率 | 画布约 1MP；一个目标 token 对应 16×16 像素 |
| 训练时间步 | 官方训练调度的 1000 个时间步，均匀采样 |
| 评测生成 | 20 步，CFG 1，seed 0；对比页 40 步 |
| 实测开销 | 约 6 秒/样本/卡；8 卡 batch 32 约 25 秒/更新；训练显存约 40–50 GB/卡 |

---

## 6. 测试与正确性验证

- `tests/test_grounding.py`（17 项，全部在 CPU 上跑）：指认任务的读回精度、注入模块各位置的行为与保存加载、路由 logit 与 block 真实注意力 logit 的一致性、检查点下的梯度、热启动时路由标定的恢复、损失各项的合成、片段位置（用真实分词器）、数据构建的泄漏防护与确定性、报告的配对与放行逻辑。路由相关的测试用的是真实 DiT 类的一个小号实例，走的是真实的 block 代码。
- `tests/test_docs.py`：文档代码块与源码一致。
- 运行：`pytest tests`（当前共 113 项，含数据侧的测试）。
- `python scripts/docs/check_paths.py`：检查 `docs/` 下各文档引用的路径是否都存在（需要挂着共享盘）。
- `tests/test_round2.py`：第二轮的读方向路由、多 span、训练行构建、损失和读回。
- 与 E-A2 当时脚本的等价性核对（逐位一致的数值与梯度、逐字节相同的数据文件、逐数字相同的报告）和 GPU 端到端回归（训练记录逐位复现、重跑评测差值为 0、四机入口单机跑通）的结果，见文档 03 §2.3 和 `V3/legacy_code/`。

---

## 7. v2 保留的部分

下面这些 v3 没有改，也不在 v3 的训练和推理路径上（除了标注"v3 复用"的），留着是为了复现 v2 的结果和做基线。实现细节见 [archive/v2/01_SAMTokEdit_Qwen21_代码实现说明.md](archive/v2/01_SAMTokEdit_Qwen21_代码实现说明.md)。

| 路径 | 内容 |
|---|---|
| `models/binding.py` | v2 的结构性区域绑定（注意力偏置、region-RoPE、区域嵌入）和 latent 融合用的区域。其中 `token_layout`（在编码后的 token 序列里定位区域片段）**v3 复用** |
| `models/pipeline.py` 的 `localize`、`ntp_loss`、`te_adapters` | v2 的两遍推理：文本编码器带定位 LoRA 先生成区域 |
| `training/engine.py`、`training/objectives.py` 的 `prepare_fm` / `flow_loss` | v2 的两阶段训练引擎（Stage 1 NTP、条件缓存、Stage 2 FM），建在 DiffSynth 的 runner 和 accelerate 上。`objectives.py` 里的 LoRA 注入、保存、加载和 `validate_conditioning` **v3 复用** |
| `training/tracking.py`、`training/gradient_audit.py` | W&B 记录、梯度审计（v2 引擎用） |
| `distributed/training.py` 与 `scripts/training/` | v2 的四机编排。其中 `scripts/training/setup_env.sh`（装环境、检查 CUDA）**v3 复用** |
| `evaluation/cases.py`、`compile.py`、`localize.py`、`manifest.py`、`run.py`、`score.py` | 评测协议 v2（benchmark 的 case 编译、生成、judge 清单与打分） |
| `preparation/`（除 `grounding_data.py`）、`scripts/data_cleaning/`、`scripts/edit_check/`、`scripts/annotation/` | 数据侧：来源转换、v1→v2 转换、清洗、MLLM 编辑核验、指令协议 1.0/2.0 适配。见文档 01 |
| `regions/selection.py` | 用 SAM2 从点或框得到候选 mask |

---

## 8. 重要路径

| 内容 | 路径 |
|---|---|
| 代码 | `/opt/tiger/tanyue/samtok_edit_qwen-image-2.1-v3`（分支 `qwen-image-2.1-v3`） |
| Qwen-Image-2.1 权重 | `/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen-Image-2.1` |
| SAMTok 文本编码器与 codec 权重 | `/mnt/bn/strategy-mllm-train/user/tanyue/models/SAMTok/Qwen3-VL-8B-SAMTok` |
| v3 实验产物 | `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v3/`（索引见文档 03 第 7 节） |
| 当前最好的 v3 权重 | C2：`/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v3/runs/c2_4n_002/train/adapter_3522`（第一轮最好的是 E-A2 `…/runs/ea2_local8_001/train/adapter_1300`） |
| v2 实验产物 | `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/qwen21_v2/` |

---

## 9. 更新记录

| 日期 | 内容 |
|---|---|
| 2026-10-24 | 3.9 节补合板代码：`merge-data`、`round2-data rows-merged`、训练行类型 `edit_merged` |
| 2026-10-19 | 文档重排：本文由 01 改为 04；补 3.9 节第二轮代码；路径与引用改到新的文档 01–03 |
| 2026-10-13 | v3 首版。把 E-A2 的方法（码本向量注入、注意力路由损失、干扰实例项、指认任务、数据构建与泄漏防护、三种评测探针、报告、对比页）写进仓库；训练循环加入各卡权重同步；新增四机入口和文档代码块同步工具 |
