# SAMTokEdit：Qwen-Image-2.1 实现与使用

> **v1 存档。** 本文描述 v1（分支 `qwen-image-2.1-dev`，commit `a93fe56`）的实现与记录，代码链接指向该 commit。当前 v2 文档见 [docs/README.md](../../README.md)。

本文是当前仓库的实现说明。2026-10-01 在独立克隆中整理为可安装的 src 包，保留此前实现的训练注意力监督 A 与区域加权 FM C。目标是说明三件事：SAMTok 区域表示如何接入 Qwen-Image-2.1 编辑；哪些官方组件被直接复用、哪些文件由本项目实现或扩展；数据准备、两阶段训练与推理如何运行。实验过程、失败尝试和数值验收另外记录在 [实验记录](02_SAMTokEdit_Qwen21_实验记录.md)，阅读本文不需要先了解历史修复编号。

当前方法使用 **Qwen3-VL-8B-SAMTok 作为同一个可定位、可编码编辑条件的 TE**，使用 **Qwen-Image-2.1 的 DiT 和 VAE**。纯文本推理先生成区域 tokens，再把它们插入 user 编辑指令，由同一 TE 编码后交给 DiT；交互式输入则直接把选区编码为 tokens，跳过定位。训练分成 TE LoRA 的 NTP/FM 联合适配、冻结 TE 后的 DiT LoRA FM 适配。这里的“两次前向”描述推理的数据流，不是每条训练样本都先在线生成 mask 再反向传播。

阅读顺序：第 1 节看总体方法与官方边界；第 2 节逐模块核对需求、官方起点、项目改动与关键代码；第 3 节准备数据；第 4–5 节核对训练机制和参数；第 6 节理解产物关联；第 7–8 节运行基础训练与推理；第 9 节说明实现边界；第 10 节集中给出 A/C 的监督定义、公式、参数与启用流程。

## 1. 实现总览

### 1.1 方法和两阶段训练

本项目把 Qwen3-VL-8B-SAMTok 同时用作区域定位器和 Qwen-Image-2.1 的编辑条件 TE。它在纯文本推理时先预测带 label 的 SAMTok mask codes，再把 codes 绑定到原指令中的对象短语；第二遍将完整编辑指令交给 Qwen-Image-2.1 DiT。交互选区则由冻结的 VQ-SAM2 codec 将用户提供的 mask 编成相同的 codes，直接进入第二遍。定位 JSON 只用于绑定，不作为 DiT 的文本条件；DiT 读取的是官方编辑模板下的连续 TE hidden states 和源图 VAE latents。

~~~text
纯文本：源图 + 指令
  → SAMTok TE 生成 [{mask_2d, label}, ...]
  → label 绑定原指令，插入 SAMTok codes
  → DiffSynth PromptEmbedder + 同一 SAMTok TE
  → Qwen-Image-2.1 DiT + 源图 VAE latents → RGBA 编辑图

交互选区：源图 + 用户 mask
  → VQ-SAM2 codec 编码 codes → 插入指令
  → 从 PromptEmbedder 开始，跳过定位
~~~

训练不在每条样本上运行上述离散生成链。Stage 1 将已标注的 NTP 行和 FM 行混合：NTP 学习从指令生成区域 JSON，FM 用已编码的 mask prompt 学习编辑条件，并通过冻结 DiT 把梯度传回 **TE LoRA**。Stage 2 先冻结 Stage 1 TE 构建条件缓存，再只训练 **DiT LoRA** 的 FM。可选的区域加权 C 作用于两阶段的合格局部 FM 行；可选的注意力监督 A 仅作用于 Stage 2。分支入口见 [SamtokTrainingModule.forward](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/training/engine.py#L224)，A/C 的计算式和开关见第 10 节。

| 部分 | 官方现有能力 | 本项目新增或修改 |
|---|---|---|
| SAMTok | Qwen3-VL [定位数据模板](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/third_party/samtok/datasets/qwen3vl_dataset.py#L53)、[VQ-SAM2 mask codec](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/third_party/samtok/models/sam2.py#L4055) | [模型接入](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/models/pipeline.py#L71)、[协议与短语绑定](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/data/protocol.py#L62)、[codec 包装](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/models/codec.py#L112)；不修改发布的码本 |
| Qwen-Image-2.1 | [DiffSynth pipeline](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/third_party/diffsynth/diffsynth/pipelines/qwen_image_21.py#L17) 的编辑模板、源图缩放、VAE/DiT、scheduler、去噪主循环 | [TE 条件接入](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/models/pipeline.py#L97)、[两阶段训练目标](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/training/objectives.py#L227)；默认编辑路径继续调用官方 pipeline |
| DiffSynth 内部扩展 | PromptEmbedder、block-causal attention、训练 runner | 在 vendored 文件中新增可选 [实际 token IDs](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/third_party/diffsynth/diffsynth/pipelines/qwen_image_21.py#L224)、[Q/K/LSE 统计](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/third_party/diffsynth/diffsynth/models/qwen_image_21_dit.py#L218)和 [optimizer-step 回调](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/third_party/diffsynth/diffsynth/diffusion/runner.py#L164)；未启用 A 时保持原返回形式 |
| 工程层 | 官方示例提供单一图像编辑训练流程 | 项目增加 [数据转换](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/preparation/converters.py#L161)、[任务混采](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/data/io.py#L101)、[cache 身份校验](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/data/provenance.py#L99)、[CLI 模式](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/cli.py#L11) |

版本基准在 [upstream_versions.json](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/upstream_versions.json)：本文的“DiffSynth 官方”指固定的 2.1.8 / commit 7686e54，不指随时间变化的主分支。权重分别来自 Qwen-Image-2.1 与 Qwen3-VL-8B-SAMTok；项目只在选用 stock 基线时加载 Qwen-Image-2.1 原始 TE。完整数据字段见第 3 节，训练参数和命令见第 4–8 节。

### 1.2 读代码的顺序

先看 [pipeline.py](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/models/pipeline.py#L71) 如何装配官方 pipeline 与 SAMTok TE，再看 [protocol.py](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/data/protocol.py#L62) 的 mask span/指令绑定；训练从 [engine.py](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/training/engine.py#L119) 进入 [objectives.py](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/training/objectives.py#L227)。需要追 A/C 时，先读 [supervision.py](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/regions/supervision.py#L32) 和 [attention.py](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/training/attention.py#L21)，最后查看 vendored DiffSynth 的可选统计接口。下文按这条运行路径说明每个模块的需求、官方起点和实际改动；代码块只摘关键行，链接指向完整实现。

### 1.3 当前目录组织与库扩展边界（2026-10-01）

当前开发目录为 `/opt/tiger/tanyue/samtok_edit_qwen-image-2.1-dev`，分支 `qwen-image-2.1-dev`。当前目录是唯一保留的本地 checkout；它替换了原先同名目录中的旧内容。正在运行的旧实验使用节点 `/tmp` 中的独立 checkout。此次只调整模块与入口组织，不改数据协议、mask、模型计算、loss、梯度更新、scheduler 或训练配方。

```text
pyproject.toml                     # 安装包、可选依赖、samtok-edit CLI
src/samtok_edit21/
  __init__.py, api.py               # 延迟加载的公共 API
  __main__.py, cli.py               # 统一 python -m / 命令行入口
  data/                            # 协议、metadata I/O、provenance、预检
  models/                          # pipeline.py 接入、codec.py 包装
  training/                        # engine.py runner 适配、objectives.py、A、审计与指标
  regions/                         # 冻结 coverage、SAMTok 编码、用户选区
  preparation/                     # 来源筛选、语义转换、规则回退、最终 corpus
  distributed/                     # 四机训练/标注编排、CUDA/NCCL 探针
third_party/
  diffsynth/                       # 固定 2.1.8 的项目扩展版；单独安装
  samtok/                          # 保留官方命名空间和源码；codec 模块随包安装
scripts/
  training/                        # ARNOLD bootstrap / run / 环境
  annotation/                      # 独立 vLLM 环境与 ARNOLD 入口，无 W&B
  diagnostics/                     # 实验结果审计与八卡推理
examples/metadata/                 # 历史最小 metadata 示例
tests/                            # 安装后执行的现有单元测试
docs/                              # 四份主文档 + 历史 archive
```

项目逻辑只位于 `src/samtok_edit21`；顶层不再留 `train.py` 等兼容转发文件。训练/推理从已安装包导入，不再由 engine 临时修改 `sys.path`。DiffSynth 保留独立库身份；本项目注册自己的训练 module、conditioning 与监督接口，调用现有 runner 和 pipeline。现有 DiffSynth 可选扩展保留在 `third_party/diffsynth`，源码内容与整理前一致；不是任意未修改的 PyPI DiffSynth 都能替代它。SAMTok 发布码本和源码也未改动。

### 1.4 可复用 API：方法、官方起点与新增入口

**方法需要**：在其他 Python 项目中加载此方法、加载两阶段 adapter 并编辑，不依赖从仓库根目录运行。

**官方起点**：DiffSynth 的 `QwenImage21Pipeline` 是实际去噪 pipeline，SAMTok 提供 TE/codec；项目的原始 `load_pipeline/edit/localize` 完成组合，参数与返回值维持原样。

**新增内容**：[pyproject.toml](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/pyproject.toml#L1) 声明 `src` 包、命令行入口与可选依赖；[api.py](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/api.py#L10) 延迟导出原始函数对象，未再包装前向；[paths.py](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/paths.py#L5) 仅为 checkout 内的四机脚本解析 repo 根目录。一般模型 API 和训练 engine 不需要 checkout 脚本。公共 API 示例：

```python
from PIL import Image
from samtok_edit21 import load_pipeline, load_adapter, edit

pipe = load_pipeline(device="cuda")
load_adapter(pipe.text_encoder, "/path/stage1/adapter")
load_adapter(pipe.dit, "/path/stage2/adapter")
pipe.eval()
image, report = edit(
    pipe, "Make the leftmost bird blue.",
    [Image.open("/path/source.png").convert("RGBA")],
    mode="online", height=1024, width=1024, seed=0,
)
image.save("/path/result.png")
```

对整理前后进行了源码和实际运行对照：176 个方法定义结构一致，另外 3 个只调整 checkout 路径或身份记录；1,719 个 vendored 文件字节一致。完整基座八卡两阶段 LoRA 权重 hash，以及三种推理的输出像素均与整理前一致。验证命令与边界见[实验记录第 19 节](02_SAMTokEdit_Qwen21_实验记录.md#19-2026-10-01独立克隆中的安装包整理与行为一致性验证)。

包安装后，`samtok-edit infer ...` 与 `python -m samtok_edit21 infer ...` 路由至相同 CLI。详细方法实现仍在下文按“需求 → 官方能力 → 本项目改动 → 代码索引/关键代码块”逐项说明。


## 2. 各模块的具体实现

### 2.1 装配 SAMTok TE 与官方 Qwen-Image-2.1 编辑管线

**方法与需求。** 同一 Qwen3-VL-8B-SAMTok TE 要能生成区域 codes，也要给 DiT 提供 4096 维编辑条件；mask tokens 必须是 tokenizer 中 514 个互不相同的原子词项。FM 需要末层 RMSNorm 前的特征，NTP 需要归一化后经过冻结 lm_head 的 logits。

**官方起点与项目改动。** [DiffSynth QwenImage21Pipeline](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/third_party/diffsynth/diffsynth/pipelines/qwen_image_21.py#L17) 已有 DiT、VAE、scheduler、编辑模板与图像处理单元。项目在 [load_pipeline](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/models/pipeline.py#L97) 中加载官方 DiT/VAE，另用 Transformers 加载 SAMTok TE；[build_processor](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/models/pipeline.py#L71) 保留 Qwen-Image-2.1 图像 processor，换入 SAMTok tokenizer/chat template，并校验 514 个 added tokens、image-pad ID、hidden size 和视觉 patch size。基座全部冻结，LoRA 范围由后续阶段决定。

~~~python
# 摘自 src/samtok_edit21/models/pipeline.py：build_processor
processor = AutoProcessor.from_pretrained(
    str(Path(qwen_dir) / "processor"), local_files_only=True
)
if samtok_dir:
    processor.tokenizer = AutoTokenizer.from_pretrained(
        samtok_dir, local_files_only=True
    )
    processor.chat_template = Path(samtok_dir, "chat_template.jinja").read_text()
~~~

装配入口将加载的 HF 权重放入项目包装器，并把替换后的 processor 交给原 pipeline；对应代码在 [load_pipeline](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/models/pipeline.py#L132)：

~~~python
pipe.text_encoder = SamtokTextEncoder(hf)
pipe.processor = build_processor(qwen_dir, samtok_dir)
pipe.requires_grad_(False)
~~~

DiffSynth [原有 TE 包装](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/third_party/diffsynth/diffsynth/models/qwen_image_21_text_encoder.py#L85) 也读取最终 norm 前特征，但会运行完整 lm_head；项目的 [SamtokTextEncoder.encode](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/models/pipeline.py#L46) 直接调用 HF backbone，以 scoped pre-hook 取 FM 特征，同时返回归一化后的 NTP 特征，并在每次调用后移除 hook、清理 rope_deltas。两种训练目标因此共享权重而不共享错误的输出层。

~~~python
# 摘自 src/samtok_edit21/models/pipeline.py：SamtokTextEncoder.encode
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
return captured[0], output.last_hidden_state
~~~

### 2.2 SAMTok span、训练行与数据转换

**方法与需求。** 每个 mask 用两个有序码本 code 表示为四个原子 tokens；NTP 行存原指令和 canonical JSON，UMT 行存 source/target 与内联 codes，普通 edit 行没有 mask。数据集原始类型映射是转换层的职责，训练层只消费统一协议。字段和四行转换例子见第 3 节。

**官方起点与项目改动。** SAMTok 发布 codec 负责 mask↔codes，DiffSynth 原本没有这套区域 token 协议。项目在 [protocol.py](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/data/protocol.py#L62) 定义码范围、span 提取、JSON 和短语绑定；[prepare.convert_record](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/preparation/converters.py#L161) 与 [native_edit_type](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/preparation/converters.py#L59) 将不同数据源转换为统一行；[validate_row](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/data/protocol.py#L389) 在读入时执行字段和引用校验。mask 编解码调用冻结的 [SamtokCodec](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/models/codec.py#L112)，不重新训练或改写发布 codebook。

~~~python
# 摘自 src/samtok_edit21/data/protocol.py
def valid_span_codes(c0, c1):
    return 0 <= c0 < 256 and 256 <= c1 < 512

def span_of(codes):
    if len(codes) != 2 or not valid_span_codes(*codes):
        raise ValueError("SAMTok needs code0 in [0,255], offset code1 in [256,511]")
    return f"<|mt_start|><|mt_{codes[0]:04d}|><|mt_{codes[1]:04d}|><|mt_end|>"
~~~

[codec.encode](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/models/codec.py#L112) 接受源图坐标的二值 mask，生成两个 code 并对第二级加 256 偏移；[decode_strict](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/models/codec.py#L180) 给训练区域预处理提供严格解码，[decode](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/models/codec.py#L191) 给可视化提供容错解码。源图内部经过 DirectResize(1024)，解码 logits 再插值回原始源图尺寸；此 1024 不决定扩散输出画布。

### 2.3 Pass 1：区域定位和 Stage 1 NTP

**方法与需求。** 对源图及原始编辑指令，自回归生成 mask_2d/label JSON。训练时只监督 JSON 与终止 token，固定空思考块属于已给定前缀；推理时用相同前缀调用 generate。这里训练的是 TE LoRA，且没有目标图或扩散 FM。

**官方起点与项目改动。** SAMTok 的 [Qwen3-VL 数据模板](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/third_party/samtok/datasets/qwen3vl_dataset.py#L53) 使用原生无 system chat；DiffSynth 的 [编辑模板](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/third_party/diffsynth/diffsynth/pipelines/qwen_image_21.py#L133) 另带 system 和 image1，不能拿来做定位。项目新增 [localization_inputs](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/models/pipeline.py#L149)：图后直接连接编辑指令与固定定位请求，assistant 前缀后填空思考块；[ntp_loss](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/models/pipeline.py#L198) 用 prefix−1 的 causal shift，只对 mt_cot + im_end 计算 CE。推理的 [localize](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/models/pipeline.py#L235) 复用同一个输入构造器。

~~~python
# 摘自 src/samtok_edit21/models/pipeline.py：localization_inputs
content = [
    {"type": "image"},
    {"type": "text", "text": instruction.strip() + "\n" + LOC_REQUEST},
]
text = pipe.processor.apply_chat_template(
    [{"role": "user", "content": content}],
    tokenize=False,
    add_generation_prompt=True,
)
text += EMPTY_THINK
~~~

~~~python
# 摘自 src/samtok_edit21/models/pipeline.py：ntp_loss
inputs, prefix, labels = localization_inputs(pipe, instruction, images, cot=cot)
_, normalized = pipe.text_encoder.encode(**inputs)
supervised = normalized[:, prefix - 1 : prefix - 1 + labels.shape[1]]
logits = pipe.text_encoder.model.lm_head(supervised)
~~~

监督 label 的 tokenization 和 CE 见 [localization_inputs](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/models/pipeline.py#L178)、[ntp_loss](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/models/pipeline.py#L202)；冻结 lm_head 仍允许梯度回到 TE LoRA。定位结果的 label 通过 [grouped_units](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/data/protocol.py#L147) 与 [condition_localization](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/data/protocol.py#L202) 绑定到原指令，不把 JSON 字符串直接交给 DiT。

### 2.4 Pass 2：官方编辑条件与 Stage 1 FM

**方法与需求。** 对已有 GT mask codes 的 FM 行，把四-token span 插在对象短语后，编码完整编辑指令和源图，令 Qwen-Image-2.1 DiT 预测目标图的 flow target。Stage 1 冻结 DiT/VAE，只通过 DiT 的输入梯度更新 TE LoRA；不从 argmax 或采样出的离散 codes 反传。

**官方起点与项目改动。** [DiffSynth PromptEmbedder](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/third_party/diffsynth/diffsynth/pipelines/qwen_image_21.py#L133) 保留原 system/image1 模板，[EditImageEmbedder](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/third_party/diffsynth/diffsynth/pipelines/qwen_image_21.py#L262) 保留源图缩放及 VAE 条件。本项目 [encode_edit](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/models/pipeline.py#L215) 调用官方 PromptEmbedder，并校验每个 SAMTok span 在实际 tokenizer 下确为四个原子 token；[prepare_fm](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/training/objectives.py#L227) 复用同一缩放后 source 给 TE/VAE，目标和源图 VAE 编码均不建图，TE 前向按 te_grad 决定是否保留计算图。

~~~python
# 摘自 src/samtok_edit21/training/objectives.py：prepare_fm
images, target, height, width = load_images(row, base_path, max_pixels)
images = resize_sources(pipe, images, height, width)
with torch.no_grad():
    target_latent = pipe.vae.encode(pipe.preprocess_image(target))
    source_latents = [pipe.vae.encode(pipe.preprocess_image(im)) for im in images]
with torch.enable_grad() if te_grad else torch.no_grad():
    cond = encode_edit(pipe, row["prompt"], images, return_positions=True) if supervision is not None and supervision["eligible"] else encode_edit(pipe, row["prompt"], images)
~~~

[flow_loss](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/training/objectives.py#L250) 复现官方 [FlowMatchSFTLoss](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/third_party/diffsynth/diffsynth/diffusion/loss.py#L5) 的 timestep/noise/noisy latent/training target/scheduler weight；实际调用 [pipeline model_fn](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/third_party/diffsynth/diffsynth/pipelines/qwen_image_21.py#L321)。基础 FM 是目标 latent 上的 FP32 MSE × scheduler weight。可选 A/C 只替换或补充后面的监督项，见 2.7–2.8 和第 10 节。

### 2.5 两阶段 LoRA、任务分支与离线缓存

**方法与需求。** Stage 1 混合 edit_ntp、局部 UMT 与普通 edit；只训练 TE attention/MLP LoRA。Stage 2 冻结 TE，从同一 Stage 1 adapter 生成缓存，只训练 DiT LoRA。Stage 1 的 NTP/ref/noref/plain 采样份额为 3/2/2/1，Stage 2 的 ref/noref/plain 为 1/2/1；这是不同样本行的混采，不是每行同时计算 NTP+FM。

**官方起点与项目改动。** 官方 DiffSynth 提供 runner、自动 DiT target 检测和训练模块接口；SAMTok 的原生 SFT 不包含扩散 FM。项目 [add_adapter](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/training/objectives.py#L57) 只放开 LoRA 参数并转 FP32，Stage 2 的 [target 检测](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/training/objectives.py#L49) 调用 DiffSynth 的自动检测；[SamtokTrainingModule](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/training/engine.py#L119) 根据阶段只加载所需组件。[make_schedule](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/data/io.py#L101) 实现上述配比，[forward](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/training/engine.py#L224) 按行选择目标。

~~~python
# 摘自 src/samtok_edit21/training/engine.py：SamtokTrainingModule.forward
if self.stage == "stage2":
    if inputs is None:
        raise ValueError("Stage 2 training expects cached inputs")
    validate_conditioning(inputs)
    loss, metrics = self._flow(inputs)
elif data["sample_type"] == "edit_ntp":
    images, _, height, width = load_images(
        data, self.args.base_path, self.args.max_pixels
    )
    images = resize_sources(self.pipe, images, height, width)
    loss, metrics = ntp_loss(self.pipe, data["prompt"], images, data["mt_cot"])
    loss = loss * self.args.ntp_weight
~~~

同一 [forward 分支](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/training/engine.py#L252) 中，其余 Stage 1 FM 行保留 TE 的梯度连接：

~~~python
else:
    prepared = prepare_fm(
        self.pipe, data, self.args.base_path, self.args.max_pixels, te_grad=True,
        supervision=self._supervision(data)
    )
    if not prepared["prompt_embeds"].requires_grad:
        raise RuntimeError("FM lost its gradient connection to TE")
    loss, metrics = self._flow(prepared)
    loss = loss * self.args.fm_weight
~~~

缓存由 [run_cache](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/training/engine.py#L704) 用冻结 TE+adapter 和 VAE 调用 prepare_fm(te_grad=False)，保存 prompt_embeds、source/target latents 与可选 region_supervision；噪声和 timestep 不保存，Stage 2 的 [flow_loss](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/training/objectives.py#L261) 每次重新采样。缓存发布前，[分片汇总](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/training/engine.py#L596) 由每个 DDP rank 并行核对自己的 payload、sidecar checksum 和 row identity，主 rank 合并紧凑索引并原子发布 `manifest.json`；[verify_cache_shard](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/data/provenance.py#L185) 在 Stage 2 启动时并行完成同样的 payload/模型/预处理检查。最终 adapter 的 recipe 与 conditioning identity 由 [save_adapter](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/training/objectives.py#L102) 和 [run_train](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/training/engine.py#L443) 保存；产物字段见第 6 节。

### 2.6 推理入口、两次调用与分辨率

**方法与需求。** online 模式先定位再编辑；oracle 接给定 JSON，inline 接已有 tokens，interactive 先将用户 mask 编码，direct/stock 做无 mask 对照。所有图像生成最终进入同一 QwenImage21Pipeline。online 的两遍应按同一个输出画布面积处理源图，避免定位 TE 和编辑 TE 看到不同的缩放结果。

**官方起点与项目改动。** DiffSynth [pipeline](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/third_party/diffsynth/diffsynth/pipelines/qwen_image_21.py#L61) 只负责单次图像编辑，默认 height=width=1024；[ShapeChecker](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/third_party/diffsynth/diffsynth/pipelines/qwen_image_21.py#L121) 先把非 32 倍数的输出宽高向上取整。本项目 [edit](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/models/pipeline.py#L283) 新增 online/oracle/inline/direct 调度；[CLI inference](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/cli.py#L212) 加上 interactive、stock 与保存报告。online 的 localize 先调用同一个 shape checker，再按该画布面积缩放源图；Pass 2 的官方 pipeline 也如此处理。因此默认 1024² 和自定义非 32 倍数尺寸都能保持两遍的源图尺度一致。

~~~python
# 摘自 src/samtok_edit21/models/pipeline.py：localize
height, width = pipe.check_resize_height_width(height, width)
prepared = resize_sources(pipe, images, height, width)
inputs, prefix, _ = localization_inputs(pipe, instruction, prepared)
~~~

交互选区在 [CLI interactive 分支](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/cli.py#L311) 中逐个编码用户 mask、组成内联 prompt，然后以 inline 模式调用同一个 edit；因此不会运行 localize：

~~~python
masks = [np.asarray(Image.open(p).convert("L")) > 0 for p in args.mask]
groups = [codec.encode(images[0], [m])[0] for m in masks]
prompt = interactive_prompt(
    prompt, groups, whole_image=len(masks) == 1 and bool(masks[0].all())
)
mode = "inline"
~~~

点/框输入先由 [regions.segment](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/regions/selection.py#L13) 调用原始 SAM2.1 生成候选，再用 [save_candidates](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/regions/selection.py#L74) 交给用户选一个 mask；候选不会直接绕过用户选择进入编辑。

训练时 [load_images](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/data/io.py#L86) 依据目标图（NTP 用源图）确定约 1M 像素、接近 32 倍数的画布；官方 [resize_edit_image](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/third_party/diffsynth/diffsynth/pipelines/qwen_image_21.py#L287) 依据目标画布面积和每张源图自身宽高比缩放源图。推理 [CLI 默认参数](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/cli.py#L79) 不自动采用源图宽高；例如请求宽×高 1000×750，实际生成画布为 1024×768。[benchmark-output](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/cli.py#L348) 只是出图后的白底合成与参考源尺寸 resize。TE 图像输入在 [PromptEmbedder](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/third_party/diffsynth/diffsynth/pipelines/qwen_image_21.py#L165) 中白底合成 RGB，VAE 使用同一缩放后图像的 RGBA；[validate_conditioning](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/training/objectives.py#L165) 检查视觉 token 与 latent 网格的 1:4 对应。

### 2.7 区域监督预处理与加权 FM（C）

**方法与需求。** C 希望把 FM 误差的部分权重移到局部编辑区域及其补区域，同时按实际权重总和归一，维持基础 loss 的量级。A/C 使用的区域先冻结为 source/target 两份 latent 网格覆盖率；输入来自 metadata 中已经编码的 SAMTok codes，转换到网格是训练目标所需的坐标映射，不重新标注数据集。

**官方起点与项目改动。** DiffSynth 的 [FlowMatchSFTLoss](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/third_party/diffsynth/diffsynth/diffusion/loss.py) 是全画布均匀 MSE，没有区域标签。项目新增 [prepare-regions](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/regions/supervision.py#L171)，在原始 source 上 [decode_strict](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/models/codec.py#L180)，再用 [coverage_grid](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/regions/supervision.py#L32) 分别生成 source 和 target 覆盖率；[RegionStore](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/regions/supervision.py#L131) 固定这些监督数据。[region_fm_loss](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/regions/supervision.py#L53) 用各组 target coverage 的逐格最大值作区域联合权重，仅在合格局部 UMT 且 region_weight>0 时由 [flow_loss](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/training/objectives.py#L302) 调用。

~~~python
# 摘自 src/samtok_edit21/regions/supervision.py：coverage_grid
pixels = torch.as_tensor(mask, dtype=torch.float32)[None, None]
pixels = F.interpolate(pixels, (height, width), mode="bilinear",
                       align_corners=False, antialias=True)
coverage = F.max_pool2d(F.avg_pool2d(pixels, 16, 16), 3, 1, 1)[0, 0]
return coverage.clamp(0, 1).contiguous()
~~~

~~~python
# 摘自 src/samtok_edit21/regions/supervision.py：region_fm_loss
si, so = m.sum(dims), (1 - m).sum(dims)
di, do = si.clamp_min(n_min), so.clamp_min(n_min)
inside, outside = (m * e).sum(dims) / di, ((1 - m) * e).sum(dims) / do
z = 1 + weight * (si / di + so / do)
loss = ((e.mean(dims) + weight * (inside + outside)) / z).mean()
~~~

具体的适用性、alignment、空区域处理与完整公式在第 10.2、10.4 节。不开 C 时 [flow_loss](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/training/objectives.py#L297) 保留官方全画布 FM。

### 2.8 训练注意力监督（A）及 DiffSynth 的可选接口

**方法与需求。** A 约束三枚已可见的 mask tokens 与目标/源图区域之间的注意力比例：target queries→mask keys 为主项，mask queries→source keys 为辅助项。分母必须是同次前向中 query 对全部可见 keys 的注意力，而不能在选取的 keys 上另做 softmax。A 只在 Stage 2 局部 UMT 上计算，不改变推理 attention logits。

**官方起点与项目改动。** DiffSynth 原 attention API 默认只返回 output；项目在 vendored [attention.py](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/third_party/diffsynth/diffsynth/core/attention/attention.py#L221) 增加可选的可微 natural-log LSE，在 [PromptEmbedder](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/third_party/diffsynth/diffsynth/pipelines/qwen_image_21.py#L224) 可选返回与 hidden states 同步裁剪的实际 token IDs，在 [DiT attention processor](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/third_party/diffsynth/diffsynth/models/qwen_image_21_dit.py#L218) 增加仅选定训练层触发的 attention_probe。无 probe 时保持原 Tensor 返回、原 block-causal mask 和推理 KV-cache 路径。

~~~python
# 摘自 third_party/diffsynth/diffsynth/pipelines/qwen_image_21.py
if return_token_ids:
    ids = [sample_ids[sample_mask.bool()][self._drop_idx:]
           for sample_ids, sample_mask in zip(model_inputs.input_ids, model_inputs.attention_mask)]
    result["prompt_input_ids"] = torch.stack([
        torch.cat([sample, sample.new_full((max_seq_len - len(sample),), -1)]) for sample in ids
    ])
~~~

项目 [span_positions](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/training/attention.py#L21) 用上述实际 IDs 定位每组四-token span；[bind_layout](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/training/attention.py#L45) 把文本位置映射到 DiT joint 序列，排除尚未看见 codes 的 mt_start。选定层用同次 post-norm/post-RoPE Q/K 和 [FlexAttention LSE](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/third_party/diffsynth/diffsynth/models/qwen_image_21_dit.py#L270) 计算小型可微统计；[BoundRegionProbe](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/training/attention.py#L73) 在 log space 求主辅分子/分母，[attention_loss](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/training/attention.py#L102) 跨层、头、位置先加总，再取比例和平方。

~~~python
# 摘自 third_party/diffsynth/diffsynth/models/qwen_image_21_dit.py：QwenImage21AttnProcessor
result = _attention(query, key, value, attn_mask=attention_mask, use_flex=True,
                    return_lse=attention_probe is not None)
if attention_probe is not None:
    hidden_states, lse = result
    statistics = attention_probe(query[:, :seq_len_q], key[:, :seq_len_kv], lse[:, :, :seq_len_q])
else:
    hidden_states = result
~~~

~~~python
# 摘自 src/samtok_edit21/training/attention.py：BoundRegionProbe.__call__
nt = torch.logsumexp((log_t + self.target_regions.log()[:, None, :, None]).flatten(1), 1)
dt = torch.logsumexp(log_t.flatten(1), 1)
ns = torch.logsumexp((log_s + self.source_regions.log()[:, None, None, :]).flatten(1), 1)
ds = torch.logsumexp(log_s.flatten(1), 1)
return torch.stack((nt, dt, ns, ds), -1)
~~~

[flow_loss](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/training/objectives.py#L314) 将 A 作为不乘 timestep weight 的辅助项加到 FM；[SamtokTrainingModule._flow](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/training/engine.py#L183) 根据成功 optimizer updates 做独立 warmup。A 需要 PyTorch ≥2.8、可微 FlexAttention LSE、无 KV cache 和 non-reentrant checkpoint；不满足时显式报错。计算方向、数学式、默认层和校准见第 10.3–10.5 节。

### 2.9 官方 runner 的最小扩展与产物链

**方法与需求。** 精确混采、梯度累积与 A warmup 必须依照真实 optimizer update，而不是 microstep 计数。训练输出要能证明 Stage 2 cache 来自哪份 Stage 1 adapter，推理应拒绝错配的 TE/DiT 条件。

**官方起点与项目改动。** 项目继续使用 DiffSynth [runner](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/third_party/diffsynth/diffsynth/diffusion/runner.py#L53) 和 [ModelLogger](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/third_party/diffsynth/diffsynth/diffusion/logger.py#L71) 的 optimizer/DDP/checkpoint 生命周期；没有另起一套训练循环。vendored runner 增加按 [schedule_sampler](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/third_party/diffsynth/diffsynth/diffusion/runner.py#L96) 取样、同步 update 时裁剪梯度及推进项目 LR scheduler、backward 审计，并在累积窗口结束调用可选回调。项目 [on_optimizer_step](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/training/engine.py#L192) 用完成的 update 数驱动 A warmup 和监督指标汇总。

~~~python
# 摘自 third_party/diffsynth/diffsynth/diffusion/runner.py
if accelerator.sync_gradients:
    update_hook = getattr(accelerator.unwrap_model(model), "on_optimizer_step", None)
    if update_hook is not None:
        update_hook(optimizer_step, accelerator, skipped=accelerator.optimizer_step_was_skipped)
~~~

[conditioning_identity](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/data/provenance.py#L39) 关联基座、SAMTok TE、processor、Stage 1 adapter、max_pixels 与 metadata；[verify_cache](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/data/provenance.py#L99) 核对分片及行覆盖；[assert_inference_identity](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/data/provenance.py#L80) 在推理前核对 Stage 2 adapter 的条件身份。带 A/C 的缓存另有监督身份，详见第 6 节和第 10.2 节。工程依赖见 [requirements.txt](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/requirements.txt) 与 [constraints-tested.txt](../constraints-tested.txt)；行为测试入口见 [tests](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/tests)。

## 3. 数据协议与标签契约

| sample_type | 必需含义 | 禁止项 |
|---|---|---|
| `edit_ntp` | source、原始 prompt、canonical mt_cot | target image、inline mask、instr_variant |
| `edit_umt` | source/target、含 mask 的 prompt、ref/noref instr_variant | mt_cot |
| `edit` | source/target、普通 prompt | mask、mt_cot、instr_variant |

每行只允许表中字段加 `sample_type/edit_type/edit_image/prompt`：NTP 另有 `mt_cot`，FM 另有 `image`，UMT 再有 `instr_variant`。来源、id、units、质量信息只存 manifest。mask 行只能有一个 source；普通 edit 支持多图。截断 span、错码本、控制 token、歧义/重叠引用均拒绝，不默认取第一处匹配。基础 [`validate_row`](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/data/protocol.py#L389) 已执行标签唯一绑定，不必依靠额外开关才能发现此类问题；四-token 码范围与抽取入口分别见 [`valid_span_codes`](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/data/protocol.py#L62) 和 [`spans_in`](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/data/protocol.py#L77)。

### 3.1 定位与编辑序列

定位前缀（图像占位由 processor 展开）：

```text
<|im_start|>user
<|vision_start|><|image_pad|>…<|vision_end|>{instruction.strip()}
Please identify and segment the region to be edited in this image.<|im_end|>
<|im_start|>assistant
<think>

</think>

```

后接 `mt_cot + <|im_end|>`；`mt_cot` 本身不含思考块。canonical JSON 为 `` ```json\n[{"mask_2d": "…", "label": "…"},\n…]\n``` ``，键序固定，键值之间冒号加空格，每项一个 mask。NTP 不接受空列表或 `No target.`。生成解析只允许移除空思考块，不接受非空推理文字。

编辑序列保持 DiffSynth 官方模板：system=`Comprehend and analyze the provided prompt.`；user 图像以 `<image1>` 开头；序列止于 `<|im_start|>assistant\n`，没有思考块/JSON/assistant 内容。两条序列使用同一图像 processor、32 对齐 resize 和透明区域白底合成。每组 mask 跟在完整短语后一个空格，同组多个 mask 直接相连，后接单词有一个空格、后接标点无空格；不能直接接在冠词或介词后。

### 3.2 类型、引用和 mask 的合同

| edit_type | mask 的标注语义（全部在源图坐标） | label / ref 挂靠 | noref |
|---|---|---|---|
| add | 目标新增内容分割 ∩ diff，再映射源图 | 新增内容短语，包含放置锚点 | 只把 anchor 换成 `in this region`；无 anchor 则补在内容短语后 |
| remove | 源对象/部件 | 被删对象，含仅起定位作用的 `from …` | `the object in this region` |
| replace | 源对象 ∪ 目标新对象 | 源对象 | `the object in this region`，保留 with/to 后的新对象 |
| attribute | 源对象/部件，不取目标侧 | 对象/部件 | `this region`，保留 color/material/texture 等属性名词 |
| action | 源对象 ∪ 目标同一对象；移动含起止位置 | 对象 | `the object in this region`，保留动作 |
| text | 源文字 ∪ 目标文字 | 原文字连同引号；否则文字载体 | `the text in this region`，保留新文字 |
| background | 膨胀后的稳定前景之补集 | background/scene/backdrop 或新底短语 | `this region`；抠图换底需上游审核改写 |
| global | 全图 | 固定 label=`this image`；ref 挂靠实际整图短语 | `this image`；无整图短语补 `to this image`；独立 `Colorize` 等补宾语 |

单 unit 用原子类型；≥2 units 用 `composite`，各 unit 类型仍为原子类型。global/background 仅一个 mask。label 去掉句首 the/a/an，必须是原文唯一连续片段；去冠词后出现歧义（如同句 the cat/a cat）不产生 NTP/ref，不保留冠词来绕过协议。同短语多实例写 `one of the {ref_phrase}`；unit 按指令顺序，同组 mask 按外接框中心先 x 后 y 排序。新构建数据按空间排序；codec.encode、批量 corpus 编码以及携带 `mask_paths` 的转换都会执行排序。现有 code-only 输入不从 token 文本猜空间位置，保留已存的组内顺序并保持 span/label/coverage 对齐；历史全量数据的 75 组非规范排序例外见第 12 节和数据盘点。

局部六类原始 mask 面积限定 0.05%–60%，background 限定 20%–97% 且等于膨胀稳定前景的补集，global 必须全图。`validate_mask_geometry` 检查这些像素条件；`convert_record` 的 unit 可带与 `mask_codes` 一一配对的 `mask_paths`，background 再带 `stable_foreground_path`，路径使用绝对路径，转换会检查原图尺寸、面积和空间顺序。纯 token 行不含几何信息，不能仅靠 `validate` 证明面积、目标语义、SAM3/diff 的标注正确性；这些必须由上游打标 QC 保证并写入 manifest。`build-debug` 没有稳定前景证据时会跳过 background，而非默认通过。SAM3 打标/跨图映射、语义复核不由训练 dataloader 重新执行。

### 3.3 转换、改写与来源

[`convert_record`](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/preparation/converters.py#L161) 接收 `instruction/edit_image/image?/units`，unit 字段为 `ref_phrase/edit_type/mask_codes/anchor_phrase?`，输出 NTP/ref/noref/plain（无 target 时仅 NTP）。检查 `GT JSON → parse → grouped_units → ref render` 的 unit 数、code 顺序及绑定位置。绑定失败只去掉 NTP/ref；noref 失败只去掉 noref；合法 plain 保留。ref/noref 文本相同只保留 noref。

规则无法覆盖的句式由上游 Qwen3-VL-8B 改写并审核，可在打标 record 中传 `noref_instruction`。其中用 `{mask_0}`、`{mask_1}` 指向原始 units 存储顺序，每个占位必须出现一次、紧跟该 unit 对应的 noref 短语；转换器使用真实 codes 替换，不允许改写器生成 codes。例如背景抠图记录可给 `Turn this region {mask_0} into a plain white background, product photography style`。该字段仅留在 provenance，不写训练行。未提供审核改写且规则失败时记录 `rewrite_errors`，不自动运行未审核的语言模型改写。

同一入口也接受存量训练行：`edit_mt` 非空 JSON 拆 NTP/ref/noref，空 JSON 只转 plain、不伪造全图 code；直接用 mask 替换名词的 `edit_umt` 补类型对应的 noref 短语（去冠词，add 去介词）；`edit` 补类型，已有合法行严格复核。CrispEdit 路径 color/motion/style/add/remove/replace/background 前缀可补类型；无法确定类型或 composite 缺审核 atomic units 时报告错误。GRES 的空思考块剥除后套编辑动词模板，原 codes/labels 保留，`No target.` 跳过。

`type/raw_type/final_task` 中已知 CrispEdit/ScaleEdit 原生类别由 [`native_edit_type`](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/preparation/converters.py#L59) 映射（含编号前缀、四类 `*_text_editing`、part_extraction、tone_adjustment 等），不能被 unit 的人工类型静默覆盖；冲突报错。RefEdit modify、reasoning/compositional 等需要已审核 units，不根据编辑文本猜类型。text unit 的 ref_phrase 若包含一个带引号原文，归一为该原文连同引号。

```bash
python -m samtok_edit21.cli convert --input /path/reviewed_records.jsonl --output /path/rows.jsonl \
  --mask-tokenizer-sha256 ENCODER_CHECKSUM_FROM_SOURCE_BUILD_REPORT
python -m samtok_edit21.cli validate --metadata /path/rows.jsonl --base-path /path/data --check-bindings
```

必须提供源编码报告中的 `--mask-tokenizer-sha256`，与 `--samtok` 下实际 `mask_tokenizer_256x2.pth` 完整 SHA256 一致才转换；不能把当前权重 hash 冒充未知 codes 的历史来源。来源不明/不一致的 codes 先从原始 mask 重编码。输出必须使用不存在的路径，产物为 metadata、`.provenance.json`、`.report.json`；manifest 存 source_dataset、完整原始 record、source_index、derived_row_hashes、rewrite_errors 及输入自带的质量字段，报告存输入文件和 tokenizer 校验和。以派生行 hash 关联，不用图片路径作为记录 id。语义/格式失败按源记录报告，不影响其他源记录。既有 `tests/eight_gpu_smoke/prepare_refedit.py` 是历史 smoke-only，不可作为正式打标器；不合规范的输出会被严格校验拒绝。

`build-debug` 不再隐式选择 standalone GRES 文件；需要显式给 `--gres`（Qwen3-VL-SAMTok 的 GRES/GRefCOCO conversations JSON）及来自其编码记录的 `--gres-mask-tokenizer-sha256`。不做 GRES replay 时传 `--gres-count 0`。未提供可核实的编码身份时，在加载 codec、写输出前拒绝启动。通用 convert manifest 的 `geometry_qc=checked` 表示实际检查了输入 raw mask；`upstream_required` 表示 token-only 导入，仍需上游 QC 证据，不等于面积检查通过。

### 3.4 从标注 record 到训练行：一个完整例子

转换器输入是一行一个 JSON record。以下 codes 只用于展示格式，真实数据必须使用该 source 上的 mask 编码值，不能直接复制示例数字：

```json
{
  "source_dataset": "reviewed_edits",
  "edit_image": "images/source.png",
  "image": "images/target.png",
  "instruction": "Change the color of the left cat to blue.",
  "units": [{
    "edit_type": "attribute",
    "ref_phrase": "left cat",
    "mask_codes": ["<|mt_start|><|mt_0001|><|mt_0257|><|mt_end|>"],
    "anchor_phrase": null
  }]
}
```

生成下面四行；source_dataset 和 units 只保留在 provenance，不进入训练 metadata：

```jsonl
{"sample_type":"edit_ntp","edit_type":"attribute","edit_image":"images/source.png","prompt":"Change the color of the left cat to blue.","mt_cot":"```json\n[{\"mask_2d\": \"<|mt_start|><|mt_0001|><|mt_0257|><|mt_end|>\", \"label\": \"left cat\"}]\n```"}
{"sample_type":"edit","edit_type":"attribute","edit_image":"images/source.png","image":"images/target.png","prompt":"Change the color of the left cat to blue."}
{"sample_type":"edit_umt","edit_type":"attribute","edit_image":"images/source.png","image":"images/target.png","prompt":"Change the color of the left cat <|mt_start|><|mt_0001|><|mt_0257|><|mt_end|> to blue.","instr_variant":"ref"}
{"sample_type":"edit_umt","edit_type":"attribute","edit_image":"images/source.png","image":"images/target.png","prompt":"Change the color of this region <|mt_start|><|mt_0001|><|mt_0257|><|mt_end|> to blue.","instr_variant":"noref"}
```

Stage 1 metadata 可以使用完整转换结果；Stage 2 metadata 必须过滤掉 edit_ntp，再用 Stage 1 adapter 构建 cache。Stage 2 不接受 NTP，也不会自动忽略它。生成样本的数量比例不需要直接等于训练比例，实际每 update 的组成由 sampler 决定。

### 3.5 采样如何对应数据协议

实现位置是 `data.RATIOS`、`protocol.TYPE_WEIGHTS`、`data.make_schedule`，不是 DiffSynth 的 dataset_repeat：

| 池 / 层级 | Stage 1 | Stage 2 |
|---|---|---|
| 每个完整 global update 的样本配比 | NTP:ref:noref:plain = 3:2:2:1 | ref:noref:plain = 1:2:1 |
| NTP/ref/noref 池内 edit_type 权重 | add/remove/replace/attribute/action/text/background/global/composite = 14/14/14/20/10/10/6/6/6 | 同左，但没有 NTP 池 |
| plain 池 | 按已有类型行数的自然比例，background、global 各封顶 15% | 同左 |
| dataset 来源 | 不单独设置来源比例；GRES replay 只进 NTP | 不使用 GRES NTP |

任务池比例每 update 精确；edit_type 是加权随机抽样，有限 batch 内只近似。缺少某个 edit_type 时在现有子类型间归一化，并写入 `absent_edit_types`；缺整个必要任务池则报错，不自动修改任务比例。每个子类型池先 shuffle，耗尽后重新 shuffle 循环，因此小池可能多次曝光。plain 池只有 background/global、无法满足各 15% 上限时会拒绝采样。`schedule.json` 记录实际抽样量、覆盖率和重复次数，不能只看 metadata 行数推测训练曝光。

## 4. 训练、保存与可复现性

| 配置 | Stage 1 | Stage 2 |
|---|---|---|
| 可训练部分 | Qwen3 language_model attention/MLP LoRA | DiT 官方空 target 配置的自动检测结果：32 个 block × 7 个 Linear = 224 个模块 |
| 默认 rank / alpha / dropout | 64 / 64 / 0.05 | 32 / 32 / 0 |
| 默认 LR / weight decay | 4e-5 / 0.05 | 1e-4 / 0.01 |
| 默认 LR 调度 / warmup | cosine / 总 optimizer updates 的 4% | constant / 总 optimizer updates 的 2.5%；可切 cosine 作同 cache 对照 |
| 默认 accumulation | 8 | 4 |
| 每步采样比例 | NTP:ref:noref:plain = 3:2:2:1 | ref:noref:plain = 1:2:1 |
| loss | NTP × 0.05；FM × 1.0，各样本独立分支 | FM × 1.0 |

LoRA 参数与优化器更新为 fp32，基座冻结。Stage 1 的 VAE 在 no_grad；FM 条件保留 TE 梯度，冻结 DiT 仍允许反传到 TE。Stage 2 只加载 DiT；TE/VAE 不重复运行。cache 任务始终 eval，包括显式传入 `--stage stage1` 的情况。

LoRA 的实际挂载位置如下，不能将“训练 TE/DiT”理解为该模型所有参数都参与更新：

| 阶段 | 匹配路径（i 为层号） | 当前基座模块数 | 明确不挂载的部分 |
|---|---|---|---|
| Stage 1 | `model.model.language_model.layers.i.self_attn.{q_proj,k_proj,v_proj,o_proj}`；同层 `mlp.{gate_proj,up_proj,down_proj}` | 36 层 × 7 = 252 个 Linear，对应 504 个 LoRA A/B 参数张量 | visual encoder/merger、embed_tokens、lm_head、norm；DiT、VAE、codec 全部不更新 |
| Stage 2 | `transformer_blocks.i.attn.{to_q,to_k,to_v,to_out.0}`；同 block `img_mlp.{proj,out,gate_layer}` | 32 blocks × 7 = 224 个 Linear，对应 448 个 LoRA A/B 参数张量 | block 外输入/输出、time/text embedding 和全局 modulation；TE、VAE 不加载到 Stage 2 训练模型 |

Stage 2 的具体列表不是硬编码在项目中，而是运行时调用官方自动检测；表中是当前 Qwen-Image-2.1 结构的结果。LoRA tensor 数不等于可训练标量参数量，后者随 rank 和各 Linear 维度变化。

`--init-adapter` 仅 warm-start 权重，**不是完整 resume**：optimizer、scheduler、采样进度重新开始。未显式传 rank/dropout 时继承 adapter；显式冲突在训练前报错。保存从实际 PEFT 导出 rank/alpha/dropout/target_modules，写 schema_version=2、recipe_sha256、base_identity，并验证 tensor key/shape/有限值。Stage 2 另存 conditioning_identity；不再依赖可能过期的 CLI 默认值。已有 rank 写错的 adapter 不自动修复：rank 可从 A/B shape 查证，dropout 不能从 tensor 推断，必须结合原配置另存修复目录。

Stage 2 **新建** adapter 的 target 直接调用 DiffSynth `DiffusionTrainingModule.auto_detect_lora_target_modules`，与官方 `--lora_target_modules ""` 同一规则。Qwen-Image-2.1 当前结构下是 224 个 block 内 Linear；旧代码枚举全部 232 个 Linear，多出的 8 个为 `img_in`、`modulation.1`、`norm_out.linear`、`proj_out`、两个 `time_text_embed.timestep_embedder.linear_*` 和两个 `txt_in.*_layer`。旧 232-target adapter 的 `target_modules` 已写入 `adapter.json`，`load_adapter` / `--init-adapter` 仍按保存的配方加载，不会静默转成 224-target；需要官方范围时使用新建 adapter。PEFT 保存时可把 224 个完整名称压缩为 7 个匹配后缀，实际注入模块数应以 adapter tensor key 核验。此变更只影响新建 Stage 2 LoRA，不改变 Stage 1、cache 内容或基座权重。

两个命令等价：

```bash
python -m samtok_edit21.training.engine train ...
python -m samtok_edit21.cli train ...
```
cache 也相同；`--save-every` 是 `--save-steps` 的别名。正式训练必须显式提供 `--steps`（**optimizer update** 数），不再按最大归一化采样池自动推导训练长度；cache 命令不需要该参数。checkpoint 间隔沿用 DiffSynth **每 rank microstep** 计数，但本项目默认改为 `--save-steps=2000`，并要求正数且可被 `--accumulation` 整除，保证定期保存落在完整 update 后。Stage 1 默认累积 8，对应每 250 updates 保存；Stage 2 默认累积 4，对应每 500 updates 保存。训练结束仍另存最终 `adapter/`；step 权重和最终 adapter 均不含 optimizer、scheduler、采样进度，不能精确 resume。旧默认 100 microsteps 的 Stage 1 checkpoint 可能位于累积中途，不应把历史 `step-100` 解释为 100 次参数更新。

训练前可在相同数据和参数下加 `--plan-only`：完成 metadata/cache 来源校验并构造真实 schedule，输出 `training_plan` JSON，**不加载训练模型，也不创建/修改 `--output` 目录**。正常训练会在加载模型前打印同一计划，并把详细统计写入 `schedule.json`；`run.json` 另记每 rank microstep 数和预计 step 权重文件数。`pool_exposure` 对每类及其 `edit_type` 给出源行数、实际抽取数、平均抽取次数、已见/未见行数、单行最小/最大抽取次数。平均抽取次数只是池级暴露指标，不等于每条样本都均匀遍历；抽取子类型使用既定权重与循环队列，正式训练应结合覆盖情况审核。

### LR 与随机数

项目通过 DiffSynth runner 的 `scheduler_factory` 显式选择调度器；不修改官方未扩展调用的 `ConstantLR` 默认行为（初始 factor=1/3）。Stage 1 默认 `--lr-schedule cosine --warmup-ratio 0.04`；Stage 2 默认 `--lr-schedule constant --warmup-ratio 0.025`。两阶段均可选 `constant|cosine`，因此 Stage 2 两组可共享同一份 cache、warmup、初始化 seed 和其他训练参数，仅改变 warmup 后的 LR 曲线。历史实验使用旧默认 constant/0 warmup，不能与新默认混称。

`--warmup-ratio R` 按实际总 optimizer update 数计算 `ceil(R × updates)`；`--warmup-steps N` 是互斥的显式覆盖，允许 `N=0` 关闭 warmup。比例需在 `[0,1]` 且有限，步数不得超过总 update 数。解析后的 `effective_warmup_steps` 和 `optimizer_updates` 写入 `run.json`。调度器只在实际、未跳过的 optimizer update 后推进一次，不随 world size 或梯度累积的 microstep 额外推进。warmup 第 k 次更新使用 LR × k/N（k 从 1 起），之后 constant 保持 LR，cosine 从峰值逐步下降；记录实际用于更新的 LR 于 `optimizer_steps.jsonl`，其中 loss 字段是最后一个 microstep 而非全局平均。极短 smoke 若 `ceil(R × updates)=updates`，没有实际衰减区间，不能用来比较两种曲线。

adapter 初始化前所有 rank 使用共同 seed；DDP prepare 后使用 seed+rank 控制 Python/NumPy/torch/CUDA 随机流；schedule 用独立共享 seed。固定设备数、版本和 seed 的短运行可复验，不保证跨硬件 bitwise deterministic。

每次 backward 的参数 hook 单独观察当前分支梯度，避免旧 accumulation 梯度掩盖断链；审计接受已连接且有限的零梯度，拒绝当前 backward 未触达可训练参数、非有限梯度或冻结参数有梯度。初始 LoRA A 零梯度、官方 FM timestep 权重为零均正常；判定与日志详见第 14 节。裁剪只发生在同步更新前。

## 5. 训练与推理超参数：当前项目和官方实现对照

本节只记录代码中实际生效的默认值、官方公开配置和适用范围。项目当前基座是 Qwen-Image-2.1 + `zhouyik/Qwen3-VL-8B-SAMTok`；表中“当前项目”指**未通过 CLI 覆盖、且未用 `--init-adapter` 继承已有 adapter 配方**时的默认值。下文训练 `--steps` 指 optimizer update，推理 `--steps` 指去噪步数，二者不相同；实验记录中的 rank 2、256²、2-step 推理是缩减的 smoke 配置，不是本节默认值。

### 5.1 DiffSynth Qwen-Image-2.1 图像编辑训练与项目 Stage 2

对照基准为仓库固定的 DiffSynth `7686e54d`：[官方 2.1 LoRA 脚本](https://github.com/modelscope/third_party/diffsynth/blob/7686e54d41d25c0e8ed5f1318acc23b6bb832654/examples/qwen_image_21/model_training/lora/Qwen-Image-2.1.sh)、[训练入口](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/third_party/diffsynth/examples/qwen_image_21/model_training/train.py)、[公共参数](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/third_party/diffsynth/diffsynth/diffusion/parsers.py)、[LoRA 注入](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/third_party/diffsynth/diffsynth/diffusion/training_module.py)、[FM loss](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/third_party/diffsynth/diffsynth/diffusion/loss.py)。脚本前半段实际执行的是文生图示例（`dataset_repeat=50`）；其 `# Edit` 下的图像编辑命令**全部被注释**。下表的“官方编辑示例”只抄录该注释命令及其未覆盖的公共默认值，不代表官方另行发布了 SAMTok/TE 两阶段编辑配方。

| 参数 | DiffSynth 2.1 官方 `# Edit` 注释示例 / 公共默认 | 当前项目 Stage 2 |
|---|---|---|
| 训练对象 | Qwen-Image-2.1 DiT LoRA；`lora_base_model=dit` | 同一基座的 DiT LoRA；TE/VAE 冻结，使用 Stage 1 adapter 产生的条件 cache |
| 数据入口 | `data_file_keys=image,edit_image`，`extra_inputs=edit_image`；示例数据为 Qwen-Image-Edit-2511 | `edit_umt:ref`、`edit_umt:noref`、`edit` 的 cache-v2；不读取 NTP 行 |
| LoRA target / rank / alpha / dropout | `lora_target_modules=""` → 自动检测重复 `ModuleList` 内、`min(in_features,out_features)≥512` 的 Linear；当前 DiT 为 224 个；32 / 32 / 0（alpha=rank 和 dropout=0 来自注入实现/PEFT 默认） | 直接调用同一自动检测方法，默认 224 个；32 / 32 / 0。旧 232-target adapter 按已保存配方加载 |
| 优化器 / LR / weight decay | AdamW / `1e-4` / `0.01`（weight decay 为公共默认） | AdamW / `1e-4` / `0.01`；AdamW 默认 `betas=(0.9,0.999)`、`eps=1e-8` |
| 单卡 microbatch / 累积 | 1 / 1（示例未传累积参数，公共默认 1）；全局 batch = GPU 数 | 1 / 4；全局 batch = `4 × GPU 数`，8 卡时为 32 |
| 训练长度 / 数据重复 | `num_epochs=5`、`dataset_repeat=100` | 必须显式传 `--steps`（optimizer updates）；`num_epochs=1` 遍历固定长度 schedule；池耗尽可循环抽取，无固定 repeat=100 |
| 图像尺寸 | 动态尺寸，`max_pixels=1048576`，宽高按 32 对齐 | `max_pixels=1048576`，宽高按 32 对齐；Stage 2 实际采用 cache manifest 的尺寸上限 |
| 精度 | pipeline BF16；官方 LoRA 注入将可训练参数转换到 pipeline dtype（BF16） | 冻结基座 BF16；LoRA 可训练参数与优化器状态 FP32 |
| 样本混合 / loss | 无 SAMTok mask 或 NTP 分支；正条件 FM，训练 `cfg_scale=1` | `ref:noref:plain=1:2:1`；FM × 1.0 |
| FM 时间步 / 目标 | 1000 个训练时间步均匀抽样；`noise−clean` target，FP32 MSE 乘 scheduler weight | 相同的 1000 步抽样、target、MSE 和 scheduler weight |
| LR 调度 / warmup | runner 未扩展调用使用 PyTorch `ConstantLR` 默认配置，初始 factor=1/3，`total_iters=5`；示例未指定 warmup | 默认 `constant`，warmup 为总 update 数的 2.5%（向上取整）；可选 `cosine`，同 warmup 对照；可用 `--warmup-steps` 覆盖 |
| 梯度检查点 / 裁剪 | 启用 DiT gradient checkpointing；公共 runner 不传 `max_grad_norm`，默认不裁剪 | 启用 DiT gradient checkpointing；同步 optimizer update 前裁剪 norm=1.0 |
| seed / 保存 | 训练 seed 未在官方命令中指定；`save_steps=None` 时按 epoch 保存 | seed=`20260920`；`--save-steps=2000` 按每 rank microstep 计数且必须对齐 accumulation；最终另存带配方和身份信息的 adapter |

项目 Stage 1 的冻结 DiT FM 路径也使用同一 DiffSynth 2.1 scheduler/loss 定义，但其 NTP/FM 联合目标和 TE LoRA 并非上述官方编辑示例的一部分。当前项目参数解析、分支 loss、采样与 scheduler 分别见 [engine.py](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/training/engine.py)、[objectives.py](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/training/objectives.py)、[data.py](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/data/io.py)。

### 5.2 SAMTok Qwen3-VL 训练与项目 Stage 1

对照资料分三层，不能合并成一个“官方 8B LoRA recipe”：① [SAMTok 论文附录 B](https://arxiv.org/html/2601.16093)给出跨 Qwen-VL 模型的 VLM SFT 设置；② 官方代码中的 [Xtuner 4B 配置](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/third_party/samtok/configs/qwen3vl_4b_mt256x2.py) 和 [MS-Swift 4B 脚本](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/third_party/samtok/swift/sft_qwen3vl_4b.sh) 是两个具体的 **4B** LoRA 示例；③ [Qwen3-VL-8B-SAMTok 发布页](https://huggingface.co/zhouyik/Qwen3-VL-8B-SAMTok)提供权重和推理示例，未列出该 8B checkpoint 的完整训练超参数。论文中 tokenizer 训练的 LR `4e-5`、global batch 1024 是训练独立 **mask tokenizer**，不是 Qwen3-VL SFT 的参数。

| 参数 | SAMTok 论文 VLM SFT | SAMTok Xtuner 4B 示例 | SAMTok MS-Swift 4B 示例 | 当前项目 Stage 1（8B） |
|---|---|---|---|---|
| 训练对象 / 损失 | 冻结视觉编码器，微调投影层和 LLM；NTP SFT | Qwen3-VL-4B，视觉编码器冻结；LLM LoRA，NTP | Qwen3-VL-4B，冻结 ViT/aligner；`all-linear` LoRA，另保存 `embed_tokens/lm_head`；NTP | 冻结视觉编码器、投影层、embedding、lm_head 和基座；LM attention `q/k/v/o_proj` + MLP `gate/up/down_proj` LoRA；NTP 与经过冻结 DiT 回传至 TE 的 FM |
| LoRA rank / alpha / dropout | 论文未列 | 128 / 256 / 0.05 | 64 / 128 / 脚本未显式传 dropout | 64 / 64 / 0.05 |
| 优化器 / LR / weight decay | AdamW / `2e-5` / 未列 | AdamW / `2e-5` / `0.05`；`betas=(0.9,0.999)` | LR `2e-5`；其余未在脚本中显式传入 | AdamW / `4e-5` / `0.05`；`betas=(0.9,0.999)`、`eps=1e-8` |
| batch / 累积 | global batch 256 | per-device 4 / 累积 1；global batch 取决于实际 GPU 数 | 8 GPU × per-device 4 × 累积 2 = global batch 64 | per-device microbatch 1 / 累积 8；global batch = `8 × GPU 数`，8 卡时为 64 |
| 长度 / 图像输入 | 论文未列 VLM SFT epoch、最大文本长度 | 1 epoch；`model_max_length=8192`；代码写 `max_pixels=2048×28×28`、`min_pixels=4×28×28`（这里记录配置原值，不把 28 当作本项目 patch size） | 1 epoch；`max_length=8192`，`IMAGE_MAX_TOKEN_NUM=2048` | 必须显式给 `--steps`，runner 遍历 1 次固定长度 schedule；无 CLI 文本长度上限；训练图像 `max_pixels=1048576`、32 对齐 |
| 精度 | 论文未列 | BF16 AMP | `torch_dtype=bfloat16` | 冻结基座 BF16；LoRA/优化器状态 FP32 |
| TE attention backend | 论文未列 | `attn_implementation=flash_attention_2` | `attn_impl=flash_attn` | 原生 HF Qwen3-VL 使用 `attn_implementation=sdpa`；不是复用 Xtuner/Swift attention 配置 |
| 调度 / warmup | cosine；warmup 比例未列 | 前 5% warmup，随后 cosine | `warmup_ratio=0.05`；脚本未显式列调度器类型 | 默认 cosine，warmup 为总 update 数的 4%（向上取整）；可用 `--warmup-steps` 覆盖 |
| gradient checkpoint / clip | 论文未列 | 配置中 gradient clip norm=1 | 启用 gradient checkpoint；脚本未显式列 clip | LM/DiT gradient checkpoint；同步 update 前 clip norm=1.0 |
| seed / data workers / 保存 | 论文未列 | seed=None / 4 / 每 1000 step，最多保留 2 份 | seed 未列 / 4 / 每 1000 step，最多保留 2 份 | `20260920` / 0 / 每 2000 microstep；不自动删除快照，最终另存 adapter |
| mask 及任务比例 | NTP 的 mask-token 生成/理解；无 FM | 示例为 mask generation 数据 | 示例为 `mask_generation_gres` | 预训练 8B SAMTok mask token 保持不变；`NTP:ref:noref:plain=3:2:2:1`；NTP × 0.05、FM × 1.0 |

### 5.3 推理：DiffSynth 编辑、SAMTok 定位与当前两次前向

DiffSynth 对照 [官方 Qwen-Image-2.1 pipeline](https://github.com/modelscope/third_party/diffsynth/blob/7686e54d41d25c0e8ed5f1318acc23b6bb832654/diffsynth/pipelines/qwen_image_21.py) 及 [官方推理示例](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/third_party/diffsynth/examples/qwen_image_21/model_inference/Qwen-Image-2.1.py)；SAMTok 对照 [8B 发布页 Quickstart](https://huggingface.co/zhouyik/Qwen3-VL-8B-SAMTok) 与本地 [Qwen3-VL demo](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/third_party/samtok/demo/qwen3vl_samtok_infer.py)。DiffSynth 原生推理没有 SAMTok 定位步；SAMTok 原生推理没有图像扩散编辑步。

| 参数 | DiffSynth 2.1 原生编辑 | SAMTok Qwen3-VL-8B 原生定位 / mask 解码 | 当前项目默认 online 推理 |
|---|---|---|---|
| 输入 / 前向 | prompt + `edit_image`，一次扩散编辑 | 单次 Qwen3-VL image+question chat；需要可视化时再用 VQ-SAM2 解码 mask | Pass 1 无 system、prefill 空思考块的 SAMTok chat；Pass 2 将 mask tokens 内联指令后交给 Qwen 2.1 编辑 |
| VLM 生成 | 不适用 | demo/8B Quickstart：`max_new_tokens=512`、`do_sample=False`、`top_p=1.0` | 定位 `max_new_tokens=256`、`do_sample=False`、`use_cache=True`、默认 `candidates=1`；遇到 im_end 特殊 token 停止 |
| 采样温度 | 不适用 | 贪心生成，无生效温度 | 默认贪心，无生效温度；仅 `localize --candidates >1` 采样时传 `temperature=0.8` |
| mask codec | 不适用 | 两级、每级 256 code；`DirectResize(1024)`；解码后的 raw mask logits `>0.5` 二值化 | 同一 256×2 codebook、`DirectResize(1024)` 和 raw logits `>0.5`；定位阶段通常只传 mask token，不执行空间 mask 解码 |
| 输出大小 / 去噪步 | pipeline 默认 1024×1024 / 40 | 无扩散去噪步；mask 可插值回原图大小 | 默认 1024×1024 / 40；`--height/--width/--steps` 可覆盖 |
| CFG / negative prompt | `cfg_scale=1.0` / 单空格 `" "` | 不适用 | `--cfg=1.0`；未覆盖 pipeline 的单空格 negative prompt |
| seed / 随机数设备 | pipeline `seed=None`、`rand_device="cpu"`；官方编辑示例显式 `seed=1` | demo 未显式设 seed | `--seed=0`；沿用 pipeline 的 CPU 随机数设备 |
| KV cache / VAE tiling | KV cache 开；VAE tiling 关，若开启则 tile 256、stride 192 | VLM `generate` 常规 cache；无扩散 VAE tiling | KV cache 开（可用 `--no-kv-cache` 关闭）；VAE tiling 沿用关闭 |
| 模式 / 输出 | 原生 `pipe(...)` 生成 RGBA 图 | 文本及可选解码 mask | `--mode=online`、`--variant=ref`，定位失败回退 plain；noref 仅用于已知单元类型的消融；RGBA PNG；`--benchmark-output` 另作白底/原尺寸 RGB 后处理 |

### 5.4 参数取值在源码中的位置与归属

| 参数 / 行为 | 生效位置 | 与官方的对齐或区别 |
|---|---|---|
| TE LoRA rank/alpha/dropout=64/64/0.05 | `train.normalize_args`、`training.add_adapter` | 项目配置；rank 与 Swift 示例相同、alpha 不同；dropout 与 Xtuner 示例相同；不是已公开的 8B checkpoint 完整配方 |
| Stage 1 LR=4e-5、WD=0.05 | `train.normalize_args` | WD 与 Xtuner 示例相同；LR 不同于官方 VLM SFT 的 2e-5；不等同于 SAMTok mask tokenizer 的 4e-5 训练任务 |
| Stage 1 cosine/4%，Stage 2 constant/2.5% | `train.scheduler_factory/resolve_warmup_steps` | 本项目训练设置；SAMTok Xtuner 是 cosine/5%；DiffSynth 默认 ConstantLR 不等于项目 warmup 后的 constant |
| DiT 224 targets、rank/alpha=32/32 | `training.stage2_target_modules/add_adapter` | target 检测、rank 与固定 DiffSynth 编辑示例一致；本项目 trainable 参数转 fp32 |
| Stage 2 LR=1e-4、WD=0.01 | `train.normalize_args` → 官方 runner AdamW | 与 DiffSynth 编辑示例/公共默认一致 |
| batch=1、accumulation=8/4、精确混采 | 官方 DataLoader 默认 batch=1；`train._accelerator`；`data.make_schedule` | 本项目两阶段配置；不能仅因 8 卡 global batch=64 就称为相同 SAMTok recipe |
| NTP=0.05、FM=1.0 | `SamtokTrainingModule.forward` | 项目联合目标；SAMTok 通用 SFT 没有 FM，DiffSynth 编辑没有 NTP；Stage 2 直接用 FM，不应用 Stage 1 的系数开关 |
| training timesteps=1000、FM target/weight | 训练类初始化、`training.flow_loss` | 复用官方 scheduler 定义；不是把推理去噪步数设成 1000 |
| max_pixels=1048576、resize=32 对齐 | `train._parser`、`data.dimensions`、官方 EditImageEmbedder | 面积默认与 DiffSynth 示例一致；SAMTok 官方示例的数据像素预算另有设置；当前不是固定 1024² 训练 |
| save_steps=2000、steps 必填 | `train._parser/validate_training_length_and_saves` | microstep 计数沿用 DiffSynth；间隔和必须显式指定 update 数是项目约束，非官方默认 |
| inference 1024²/40 steps/CFG 1.0 | `cli.main` → `model.edit` → 官方 pipeline | 这三项与 DiffSynth pipeline 默认一致；项目 seed=0、max_new_tokens=256 是 CLI 设置 |
| greedy / sampled localization | `model.localize`、`cli.inference` | greedy 与 SAMTok demo 对齐；项目预填空思考块并限制生成长度；多候选 temperature=0.8 只在采样时生效 |
| codec FP32、resize 1024、raw logits >0.5 | `codec.SamtokCodec` | 使用发布 codec 的编码/解码约定；点框 proposal 的原始 SAM2 阈值则为 raw logits >0，在 `regions.segment` 中，不应混为一个阈值 |

未在官方公开配置中明确列出的值，在表中标为未列，不把框架可能随版本变化的隐含默认补写成 SAMTok 8B 实测参数。当前项目的真实运行配置以 `run.json`、`adapter.json` 和 `optimizer_steps.jsonl` 为准，文档默认值不覆盖 CLI 指定值或 warm-start 保存配方。

## 6. 缓存、adapter 与产物关联

### 6.1 训练/cache 目录里保存了什么

| 产物 | 写入者 | 用途 |
|---|---|---|
| `run.json` | `train.run_train` | 生效参数、模型内容身份、总 updates、warmup、保存预算、seed 规则 |
| `schedule.json` | `data.make_schedule` → `train.run_train` | 任务池/子类型实际抽样与覆盖统计 |
| `optimizer_steps.jsonl` | 扩展后的官方 runner | 本次 update 的实际 LR 和最后 microstep loss |
| `supervision_metrics.jsonl` | 项目 `on_optimizer_step` 回调 | 启用 A/C 后的跨 rank 逐样本均值、各指标计数、跳过原因与成功 update 编号 |
| `step-*.safetensors` | 官方 ModelLogger | 按 microstep 保存的权重快照；最后不足一个间隔也会补存；没有 optimizer/scheduler 状态 |
| `adapter/adapter.safetensors`、`adapter/adapter.json` | `training.save_adapter` | 最终 LoRA 权重、真实 rank/alpha/dropout/targets、recipe hash、base identity；Stage 2 另带 conditioning identity；项目加载入口使用这个目录 |
| cache 的 `<rank>/<index>.pth` 与 `.json` sidecar | 官方 data-process runner + `train._cache_manifest` | 条件/latents、源 row_index/hash、identity、shard checksum |
| cache 的 `manifest.json` | `train._cache_manifest` | 全部分片验证通过后原子发布；按源行号关联 metadata 与分片 |

cache payload 的 `inputs` 包括 `input_latents`（target）、`edit_latents`（sources）、`prompt_embeds`、`prompt_embeds_mask`、`edit_image_pad_mask`；构建时传 `--region-cache` 还会保存 `region_supervision`，合格样本含 FP32 S/T 覆盖率与 Long span 位置。Stage 2 的随机噪声和 timestep 不在 cache 中，每次训练重新生成。Stage 1 final adapter → cache identity → Stage 2 adapter conditioning_identity → 推理身份校验构成连续的来源链。

### 6.2 内容身份与兼容边界

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

### 7.1 环境与模型目录

在 repo 根目录，使用 Python 3.11；`constraints-tested.txt` 锁定本次实际验证的 Linux x86_64/H100/CUDA 12.8 环境（torch 2.8.0、torchvision 0.23.0、transformers 5.12.1、accelerate 1.14.0、peft 0.20.0）。不是所有平台通用锁。依赖已补 ijson、pyarrow、Hydra、OpenCV、SciPy；无需依赖 user-site。

```bash
python3.11 -m venv /path/to/samtok21-venv
source /path/to/samtok21-venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$PWD/src:third_party/diffsynth
export PYTHONDONTWRITEBYTECODE=1
```

模型路径可用 `--qwen/--samtok` 覆盖；当前默认值在 `model.py`：

| 参数 | 默认目录 | 必需内容 |
|---|---|---|
| `--qwen` | `/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen-Image-2.1` | transformer/、vae/、processor/、scheduler/；stock 模式另外使用 text_encoder/ |
| `--samtok` | `/mnt/bn/strategy-mllm-train/user/tanyue/models/SAMTok/Qwen3-VL-8B-SAMTok` | HF 模型权重、config/tokenizer/chat_template；codec/交互另需 sam2.1_hiera_large.pt 和 mask_tokenizer_256x2.pth |

模型按本地文件加载，不会在缺权重时静默改用在线模型。训练设备由 Accelerator/local rank 决定，实际覆盖 `--device`；多卡通过 torchrun/accelerate 分配，单卡可用 CUDA_VISIBLE_DEVICES 选择可见卡。推理与 codec 命令的 `--device` 直接控制设备。

### 7.2 准备并检查 Stage 1 / Stage 2 metadata

先按第 3 节得到审核后的 JSONL。下面的 checksum 必须来自输入 codes 的真实编码记录，不是为了通过校验随便填当前文件 hash：

```bash
python -m samtok_edit21.cli convert \
  --input /path/reviewed_records.jsonl --output /path/stage1.jsonl \
  --mask-tokenizer-sha256 ENCODER_CHECKSUM_FROM_SOURCE_BUILD_REPORT
python -m samtok_edit21.cli validate \
  --metadata /path/stage1.jsonl --base-path /path/data --check-bindings
```

Stage 2 只保留其中 FM 行，可用仓库的数据读写函数过滤（输出路径不要指向源数据）：

```python
from samtok_edit21.data.io import read_rows, write_rows

rows = read_rows("/path/stage1.jsonl")
write_rows("/path/stage2.jsonl", [r for r in rows if r["sample_type"] != "edit_ntp"])
```

```bash
python -m samtok_edit21.cli validate \
  --metadata /path/stage2.jsonl --base-path /path/data
```

相对的 image/edit_image 路径由训练和 validate 的 `--base-path` 解析；通用 converter 不读取普通 source/target 像素，除非提供 raw mask 做几何 QC，此时按第 3 节使用绝对路径。转换后检查 `.report.json` 的失败/跳过记录和 `.provenance.json`，不能只看命令退出成功。确认 Stage 1 有 NTP/ref/noref/plain 四个池，Stage 2 有 ref/noref/plain 三个池，再规划训练。

### 7.3 Stage 1 → cache → Stage 2

以下 `/path/...` 均需替换为实际路径；单卡用普通 python，多卡建议 torchrun 显式指定数量，不依赖外部 accelerate 配置：

```bash
# 先只读预览 Stage 1；--output 不会被创建，可与正式训练复用。
torchrun --standalone --nproc_per_node=2 -m samtok_edit21.training.engine train --stage stage1 \
  --metadata /path/stage1.jsonl --base-path /path/data \
  --output /path/new-stage1 --steps 1000 --accumulation 8 --plan-only

torchrun --standalone --nproc_per_node=2 -m samtok_edit21.training.engine train --stage stage1 \
  --metadata /path/stage1.jsonl --base-path /path/data \
  --output /path/new-stage1 --steps 1000 --accumulation 8 --seed 20260926 \
  --lr-schedule cosine --warmup-ratio 0.04

torchrun --standalone --nproc_per_node=2 -m samtok_edit21.training.engine cache \
  --metadata /path/stage2.jsonl --base-path /path/data \
  --te-adapter /path/new-stage1/adapter --output /path/new-cache

# Stage 2 两组共用同一份 cache；先预览任意一组的采样计划。
torchrun --standalone --nproc_per_node=2 -m samtok_edit21.training.engine train --stage stage2 \
  --cache /path/new-cache --output /path/new-stage2-constant --steps 1000 \
  --accumulation 4 --seed 20260926 --plan-only

torchrun --standalone --nproc_per_node=2 -m samtok_edit21.training.engine train --stage stage2 \
  --cache /path/new-cache --output /path/new-stage2-constant --steps 1000 \
  --accumulation 4 --seed 20260926 --lr-schedule constant --warmup-ratio 0.025

torchrun --standalone --nproc_per_node=2 -m samtok_edit21.training.engine train --stage stage2 \
  --cache /path/new-cache --output /path/new-stage2-cosine --steps 1000 \
  --accumulation 4 --seed 20260926 --lr-schedule cosine --warmup-ratio 0.025
```

多卡 Stage 1 accumulation 必须为 8 的倍数，Stage 2 为 4 的倍数，保证每 rank 的比例一致。两阶段最终可消费产物均为 `adapter/{adapter.json,adapter.safetensors}`；官方 step checkpoint 只含训练权重，不是独立完整 resume/adapter 包。

给定池大小 `N_k`、global batch `world_size × accumulation` 和该池比例份额 `r_k/Σr`，池级平均抽取次数为 `steps × global_batch × r_k/(Σr × N_k)`。例如 8 卡、4000 updates 时，Stage 1 的 global batch=64，NTP/ref/noref/plain 分别抽取 96k/64k/64k/32k 次；Stage 2 的 global batch=32，ref/noref/plain 为 32k/64k/32k 次。使用默认 `--save-steps=2000`，Stage 1 共 32k microsteps → 16 个 step 权重文件，Stage 2 共 16k → 8 个；另有最终 adapter。当前不自动删除旧 step 文件，需在开跑前预算磁盘。是否训练 4000 updates 应由实际池覆盖、验证表现和计算预算决定，不是默认长度。

后续 Stage 2 ablation 提醒：以上两个命令必须使用**同一份已验收 cache**和相同的 base/Stage 1 条件、schedule seed、训练 update 数、rank/dropout、LR、batch/accumulation 等设置，仅改变 `--lr-schedule`；输出目录必须不同。保存两组 `run.json`、`schedule.json`、`optimizer_steps.jsonl` 和 adapter，先核对有效 warmup/update 数与 LR 轨迹，再在同一验证集、相同推理参数和随机种子下比较。短程 smoke 只验证软件路径，不代替正式质量/收敛结论。

### 7.4 训练后如何连接到推理

推理必须同时指定用于该 cache 的 `--te-adapter /path/new-stage1/adapter` 和由该 cache 训练的 `--dit-adapter /path/new-stage2-constant/adapter`。不能只换 TE adapter 而沿用另一套 TE 条件下训练的 DiT adapter；程序会在加载前检查内容身份。`step-*.safetensors` 不能直接当作 `--te-adapter/--dit-adapter` 目录，正式推理使用带 adapter.json 的最终目录。

只加载 TE adapter 做 `localize` 可以单独检查定位；只加载 TE adapter、不加载 DiT adapter 的 infer 是使用冻结基座 DiT 的中间检查，不等同于完成两阶段后的最终模型。`--init-adapter` 继续适配已有权重也不等同于恢复中断的训练状态。

## 8. 推理、noref 边界与评测

### 8.1 模式对应关系

| 命令 / mode | 需要的输入 | 是否运行 pass-1 | 主要用途 |
|---|---|---|---|
| localize | 单 source + 原始 prompt，可选 TE adapter | 是；只输出定位报告，不运行 DiT | 检查 JSON、引用绑定、候选 mask |
| infer / online（默认） | 单 source + 原始 prompt | 是，默认 ref | 完整纯文本两次前向编辑 |
| infer / oracle | 单 source + 原始 prompt + `--cot-file` | 否，使用给定 JSON | 将定位误差与编辑误差分开检查；noref 消融可加审核 units |
| infer / inline | 单 source + 已含 mask spans 的 prompt | 否 | 已经完成区域绑定的编辑 |
| infer / interactive | 单 source + `--mask` PNG 列表 + prompt | 否；现场执行 codec.encode | 用户直接指定区域；多个选区需对应多个指代词 |
| infer / direct | source（可多图）+ 无 mask prompt | 否 | 同 SAMTok TE 条件下去掉区域的对照；可加载项目 adapter |
| infer / stock | source（可多图）+ 无 mask prompt | 否；使用 Qwen-Image-2.1 原始 TE | 官方权重基线，禁止加载项目 TE/DiT adapter |

所有图像生成模式最终调用官方 `QwenImage21Pipeline`。online 的 JSON 解析/绑定失败回退原始 plain prompt；oracle 是显式提供条件的检查入口，JSON/引用错误直接报错，不把错误 GT 自动当成成功 oracle。

### 8.2 纯文本定位和完整编辑

```bash
python -m samtok_edit21.cli localize --image /path/source.png \
  --prompt 'Make the leftmost bird blue.' \
  --te-adapter /path/new-stage1/adapter --output /path/localize.json

python -m samtok_edit21.cli infer --image /path/source.png \
  --prompt 'Make the leftmost bird blue.' --variant ref \
  --te-adapter /path/new-stage1/adapter --dit-adapter /path/new-stage2-constant/adapter \
  --height 1024 --width 1024 --output /path/result.png
```

online/oracle 默认 requested_variant=ref；两者使用同一 `condition_localization`。报告 `requested_variant/actual_variant/fallback_reason`：

- 已审核 `--units-file` 提供每个定位分组的 `ref_phrase/edit_type/anchor_phrase?`，顺序与分组一致，ref_phrase 必须精确对应绑定后的 label；mask codes 来自定位结果而非这个文件。
- 纯文本 pass-1 JSON 没有 edit_type，默认只按 label 绑定原指令，不推断类型。`--variant noref` 必须提供已审核 `--units-file` 才能实际走 noref；缺失时返回 ref 并记录原因，`--strict-noref` 则报错。
- add 必须保留新增物体；复杂 add 需要审核的末尾 anchor。text 必须保留目标文字。composite 需要逐 unit 的审核语义。审核 ref_phrase 应覆盖完整 where；不要把 what/how 包入待删除的引用范围。未标注的其他补语保留，不任意删除整段 from ...。
- noref 改写不可靠时回退 ref；定位解析/绑定失败时 online 回退 plain。严格实验加 `--strict-noref`，无法正确生成 noref 就报错，不能把 fallback 混进 noref 得分；该标志只用于 online/oracle。
- localize 输出是候选报告列表；oracle 的 `--cot-file` 需要其中的 raw canonical mask JSON 内容（或另行提供的 canonical JSON），不是整份候选报告列表。

与上面 bird 指令匹配的审核 units 文件内容示例：

```json
[{"ref_phrase":"leftmost bird","edit_type":"attribute"}]
```

该文件必须匹配实际定位/GT JSON 的 label，不能用固定例子匹配所有生成结果。add 的 unit 另可指定 `anchor_phrase`，例如 `ref_phrase="red ball next to the chair"`、`anchor_phrase="next to the chair"`。

### 8.3 Oracle、noref 消融与用户选区

`/path/gt_cot.txt` 保存第 3 节格式的 fenced JSON list，mask codes/label 必须对应本次 source 和 prompt；`/path/reviewed_units.json` 保存上面的 units 列表。

```bash
# 用真实 GT 区域隔离定位误差：ref。
python -m samtok_edit21.cli infer --mode oracle --variant ref \
  --image /path/source.png --prompt 'Make the leftmost bird blue.' \
  --cot-file /path/gt_cot.txt --te-adapter /path/new-stage1/adapter \
  --dit-adapter /path/new-stage2-constant/adapter --output /path/oracle-ref.png

# 同一 GT codes 的 noref 对照，严格禁止回退混入结果。
python -m samtok_edit21.cli infer --mode oracle --variant noref --strict-noref \
  --image /path/source.png --prompt 'Make the leftmost bird blue.' \
  --cot-file /path/gt_cot.txt --units-file /path/reviewed_units.json \
  --te-adapter /path/new-stage1/adapter \
  --dit-adapter /path/new-stage2-constant/adapter --output /path/oracle-noref.png

# 用户已提供源图尺寸的二值 mask。
python -m samtok_edit21.cli infer --mode interactive \
  --image /path/source.png --mask /path/selected-mask.png --prompt 'turn into gold' \
  --te-adapter /path/new-stage1/adapter \
  --dit-adapter /path/new-stage2-constant/adapter --output /path/interactive.png

# 点/框先得到候选，再由用户挑选一个候选 PNG 传给上面的 --mask。
python -m samtok_edit21.cli regions --image /path/source.png \
  --point 100 120 1 --output /path/region-candidates

# 采样两份完整定位假设，并解码每份假设中的各个目标。
python -m samtok_edit21.cli localize --image /path/source.png \
  --prompt 'Make the leftmost bird blue.' --candidates 2 --decode-masks \
  --te-adapter /path/new-stage1/adapter --output /path/candidates.json
```

点坐标示例必须换成实际图内位置，label=1/0 分别为正/负点；框参数为 `--box X0 Y0 X1 Y1`，使用源图像素坐标。`regions` 返回 `candidates.json` 和 PNG，selected_index 是按 SAM2 score 得到的默认候选索引，并不会替用户自动启动编辑；传多个 `--mask` 表示同时选择多个区域，不是让程序从中挑一个。单次 localize JSON 的多项也表示同一假设中的多个目标。

`direct/stock` 定义为无 mask 普通编辑；`inline` 必须已有合法 mask span；`interactive` 使用一张源图和所选 mask，按选区顺序调用 codec 后插入指代短语。所有 masked 模式统一要求单 source，CLI 在加载模型前拒绝 masked 多图；普通 direct/stock 仍可多图。

交互绑定识别 this region/area/object/image、the selected region/area/object、here/this/it，按文本顺序与选区一一对应，数量不匹配报错。没有指代词时只允许单选区：add/insert/place/put/draw 句尾补 `in this region`；remove/delete/erase/get rid of/replace/swap 动词后补 `the object in this region`；文字编辑补 `the text in this region`；apply 补 `to this region`；make/turn/change/paint/color 等动词后补 `this region`；非动词开头在句首补。全图选区使用 `this image`。多候选定位每次都 prefill 相同空思考块；一次 JSON 的多个项目默认同时编辑，并非候选替代项。

原生输出保持 RGBA PNG。评测时显式传 `--benchmark-output`：保留 `result.raw.png`，白底 alpha composite 后转 RGB，并 resize 至参考源图原始尺寸，JSON 记录 raw/final 尺寸。多图必须指定 `--reference-image-index`（0-based），不能猜参考图；所有基线必须使用相同后处理。

### 8.4 输出检查

`localize --output` 写候选报告列表；`infer --output result.png` 还会写 `result.json`，其中记录实际 conditioning_prompt、raw 定位输出、items、requested_variant/actual_variant、fallback_reason，以及输出模式/尺寸/后处理。判断运行走了 ref、noref 还是 plain，以 actual_variant 为准。`--height/--width` 控制画布，`--steps` 控制去噪步数，`--max-new-tokens` 控制 pass-1 长度，不要把这三个概念混用。

## 9. 当前实现边界与验证入口

两个 code 能被 codec 解码，不意味着 DiT 已有硬性空间约束或背景保护。训练时 A（attention supervision）和 C（regional FM）已作为显式可选功能实现，见第 10 节；推理时 B（attention logits 软 mask 偏置）尚未实现。A/C 是训练目标，不保证区域外像素严格不变。

noref 不保证所有空间信息只来自 mask；what/how 与图像仍可能泄露目标。正式评测应分开统计格式成功率、绑定成功率、noref 覆盖率、定位 IoU、区域内编辑与区域外保真，并做正确/交换/随机/无 code 对照。当前 JSON 不携带 atomic edit_type，因此默认纯文本走 ref，noref 消融使用已审核单元类型，不做在线语法猜测。

codec 保持发布实现的 raw logits > 0.5 阈值；这不等于 sigmoid > 0.5，也不是 2.1 迁移 bug。以后改变阈值需作为独立实验记录。

当前验证覆盖数据协议、真实 tokenizer/processor、LoRA 范围、NTP 监督位置、TE/DiT 梯度、cache 身份与张量一致性、两阶段小批量训练，以及 online/oracle/interactive 出图。验收结果与命令见实验记录：第 8 节为 DiT 224-target，第 9 节为 LR scheduler，第 10 节为训练长度/保存/曝光，第 11 节为数据合同与端到端检查。基础测试可在 repo 根目录运行：

```bash
python -m pytest -p no:cacheprovider -q tests
```

小批量训练与少步出图只证明执行路径，不证明定位语义、编辑质量或收敛；SAMTok 生成一个可绑定 label 也不等于它找到了正确对象。全量标注、独立质量评测、复杂指令的上游审核改写、精确训练状态 resume 不包含在当前已完成的功能中。没有 raw mask 的 token-only metadata 无法由本地字段检查推断几何 QC 是否通过。

后续维护时，改条件模板查 model.py 与官方 PromptEmbedder；改字段/绑定查 protocol.py 和 prepare.py；改采样查 data.py；改参数/阶段逻辑查 train.py；改 LoRA/FM 查 training.py；改来源校验查 provenance.py。代码变动须同步更新本文的当前行为，实验结果另记实验记录。影响实际 TE 条件、adapter 内容或 resize 的变更必须重新判断 cache 身份和是否重建，不能靠改路径沿用不匹配产物。定位前缀只作用于 NTP/pass-1，单独改变它不等于 pass-2 模板变化；但重新训练的 TE adapter 内容变化后，必须重建其 Stage 2 cache。

## 10. A/C 监督定义、公式与启用流程

### 10.1 作用范围与官方接口边界

本节实现 `/opt/tiger/tanyue/SAMTok_mask_attention_constraints.md` 的训练部分；不实现推理软 mask B。CLI 默认 `--region-weight 0 --attention-weight 0`，保留基础实验行为；启用命令见 10.6。区域监督不修改 metadata 的字段、定位 JSON、编辑 prompt、LoRA target 名称或推理 API。

| 分支 | C | A | 参数更新 |
|---|---|---|---|
| Stage 1 / edit_ntp | 不计算 | 不计算 | 原 NTP → TE LoRA |
| Stage 1 / 局部 edit_umt（ref/noref） | 显式启用且区域合格时计算 | 禁止启用 | C 经冻结 DiT 反传至 TE LoRA |
| Stage 2 / 局部 edit_umt（ref/noref） | 可单独启用 | 可单独启用或与 C 同用 | DiT LoRA，仍为官方自动检测的 224 个 Linear |
| edit、global、明确不对齐或显式跳过空区域 | 保持基础 FM | 不计算 | 对应阶段的原训练分支 |

项目复用官方 attention 计算并读取统计，不替换 processor 的可见性规则、不注入 logits bias，也不保存完整 $S\times S$ 注意力矩阵。

A/C 的逐文件代码改动和关键摘录已集中在第 2.7–2.9 节；这里继续说明监督数据、位置映射、精确公式和启用条件。

A 要求 PyTorch ≥ 2.8、支持可微 LSE 的 FlexAttention、block-causal、无 KV cache、PyTorch non-reentrant checkpoint。版本/后端不满足时直接报错，不 detach LSE、不偷偷切成近似。配置了 DeepSpeed activation checkpointing 时也明确拒绝此路径。当前验证环境为 PyTorch 2.8.0+cu128；其他后端未作通过承诺。

### 10.2 冻结区域数据：先于 Stage 1，独立于 TE adapter

[`prepare-regions` 主循环](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/regions/supervision.py#L171)逐行读取合法 metadata，按 prompt 中出现顺序解析全部四-token spans（不按码去重），调用发布 codec 在**原始 source** 上解码：[`DirectResize(1024)`](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/models/codec.py#L76) → [raw logits 插值回原图 → `>0.5`](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/models/codec.py#L204)。不使用原始标注 mask 替代 token 解码结果，也不先把 source 缩成训练尺寸再解码。

对每组原图二值 mask，[`coverage_grid`](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/regions/supervision.py#L32) 分别按实际 source canvas 与 target canvas 执行：FP32 bilinear resize（`align_corners=False, antialias=True`）→ `avg_pool2d(kernel=16,stride=16)` → `max_pool2d(kernel=3,stride=1,padding=1)`。这与当前 TE/VAE 几何一致，输出 `[K,H/16,W/16]` 的覆盖率 `coverage_source/coverage_target`，不再次二值化。source 和 target 大小不同则分别计算，不能复用错误尺寸的网格；[构建这两份网格的代码](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/regions/supervision.py#L224)紧挨着保存逻辑。

区域缓存保存外扩后的原始覆盖率 $\tilde m$；C 使用 target 各组的逐格最大值并集。A 在读取时分别计算 source、target 各组的 $\hat m=\tilde m/\max_i\tilde m_i$。最大值归一化使 A 的最优比例可达 1，但**并不消除其对最大覆盖率位置的偏好，也不要求注意力铺满区域**。

浮点边界处理：antialias resize 对全 1 mask 可能产生约 `1+2.4e-7` 的舍入越界；最终 coverage 投影回 `[0,1]`，不重新二值化，也不改已在合法范围内的覆盖率值。

源目标对齐必须由调用者明确认证：`--assume-aligned` 表示该批所有局部 UMT 全图坐标对齐；或提供 `--alignment-manifest`，JSON 为 `{完整row_hash: true/false}`，必须覆盖每个局部 UMT。false 的样本仍训练基础 FM，不参与 A/C。相同宽高不能证明对齐；crop、视角变换等数据不能仅靠 resize 获得正确目标区域。本实现不猜测或自动估计变换。

空解码/下采样后空区域默认报错；`--skip-empty` 显式允许跳过**整条样本**的 A/C，并计入 `empty_region`，不是丢弃某个 mask 后静默减少 K。格式错误、码范围错误、缺 alignment 记录等仍报错。

区域 `manifest.json` 记录 codec/SAM2 权重 SHA256、原 metadata SHA256、几何配置、source 最小像素数、alignment 认证和 skip 策略。逐样本记录源/目标图 SHA256、完整行 hash、有序 spans、eligibility/reason、coverage 和 shard checksum。`RegionStore.load` 核验 manifest/payload 的身份与适用性，并核验图像内容（同一进程内成功核验的文件避免重复读取）。不得训练途中改写输入文件。

`--mask-tokenizer-sha256` 必须填**数据编码阶段记录的 checksum**；本地解码权重须与其相等。现场计算当前文件 hash 只能确认当前文件身份，不能独立证明历史数据由该权重编码。

Stage 2 构建 conditioning cache 时，把区域 supervision 与其 `supervision_identity` 并入缓存；这是独立于 diffusion `conditioning_identity` 的训练标签身份。改变 lambda 不需要重算 TE；改变 codec、mask 几何、metadata/图像等需要重建区域及带监督的 conditioning cache。推理不加载区域缓存或 codec 来施加约束，仍只核验原模型/TE adapter 身份。

### 10.3 从实际 TE token IDs 到 DiT joint 位置

[DiffSynth `PromptEmbedder`](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/third_party/diffsynth/diffsynth/pipelines/qwen_image_21.py#L224) 在同一次前向中，按与 hidden states 相同的 attention-mask 去 padding、system 前缀裁剪和 batch padding 规则返回 IDs；补齐位置用 -1。[`span_positions`](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/training/attention.py#L21) 对照 tokenizer 得到的每组四个原子 ID，逐组检查顺序/数量/连续性，不在原始 prompt 中按字符数猜 token offset。

缓存的 `span_positions` 为 `[K,4]` 的 Long Tensor，坐标在 `prompt_embeds` 中。进入 DiT 后，[`AttentionSupervision.bind_layout`](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/training/attention.py#L45) 利用运行时 `repeats=where(img_mask,4,1)`，以 `cumsum(repeats)-repeats` 将位置映射到 joint 序列。源图索引来自 `image_ids==0`，目标图索引来自 `target_token_mask`；核验单 source、网格 token 数和 mask 位置均为有效文本、不是 padding/image token。

A 只取每组的 code1、code2、mt_end，排除因果可见性上还看不到该组码的 mt_start。监督方向为 target queries → 三个 mask keys，以及三个 mask queries → source keys；不是 source → mask，也不是 mask → target。

### 10.4 C 与 A 的精确计算

令 $e_i$ 为 64 个 latent channel 上的 FP32 平方误差均值，$m_i=\max_k\tilde m_i^{(k)}$。记

$$
s_{in}=\sum_i m_i,\quad s_{out}=\sum_i(1-m_i),\quad d_{in}=\max(s_{in},n_{min}),\quad d_{out}=\max(s_{out},n_{min}).
$$

$$
L_C=w(t)\frac{\operatorname{mean}_i(e_i)+\lambda_C\left(\frac{\sum_i m_ie_i}{d_{in}}+\frac{\sum_i(1-m_i)e_i}{d_{out}}\right)}{1+\lambda_C(s_{in}/d_{in}+s_{out}/d_{out})}.
$$

分母是实际位置权重之和，触发 `n_min` 截断时也不能固定成 $1+2\lambda_C$。FP32 计算覆盖率求和、误差和归一化。恒定误差严格保持基础 FM 尺度；$\lambda_C=0$ 恢复原 loss。对于 soft coverage，二值 mask 的“区域内 token 权重份额”简式不能直接当成参数梯度份额。

上述 C 的逐项计算在 [`region_fm_loss`](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/regions/supervision.py#L53)；[`flow_loss`](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/training/objectives.py#L303) 对多组 target coverage 取逐格最大值，仅在区域合格且 `region_weight > 0` 时替换基础 FM 的位置聚合，最后仍乘 scheduler 的 timestep weight。

A 使用与前向一致的投影、q/k norm、RoPE 后 Q/K：

$$
\log A_{ij}=q_i^\top k_j/\sqrt d-\operatorname{LSE}_i.
$$

LSE 来自同一次 FlexAttention，分母包含 query **全部可见 keys**，保留梯度；辅助项也不是 source-only softmax。每层逐组返回 `log(N_T), log(D_T), log(N_S), log(D_S)` 四个 FP32 数；log-space `logsumexp` 在头、位置、三个 mask tokens 及所选层上求和，防止极小 attention mass 下溢，与先求和 N/D 的公式等价。

$$
r^T_k=\frac{\sum_{l,h,i\in T,j\in\mathcal M_k}\hat m_i^{(k)}A^{l,h}_{ij}}{\sum_{l,h,i\in T,j\in\mathcal M_k}A^{l,h}_{ij}},\qquad
r^S_k=\frac{\sum_{l,h,j\in\mathcal M_k,s\in S}\hat m_s^{(k)}A^{l,h}_{js}}{\sum_{l,h,j\in\mathcal M_k,s\in S}A^{l,h}_{js}}.
$$

$$
L_A=\frac1K\sum_k(1-r^T_k)^2+\mu\frac1K\sum_k(1-r^S_k)^2,\qquad
L=L_C+\lambda_A(u)L_A.
$$

未启用 C 时上式 $L_C$ 替换为基础 FM。A 不乘 $w(t)$；全部采样到的 t 都计算。聚合是“跨头、跨层先加总，再取比例/平方，最后平均 K”，不是逐层 loss 平均。

调用链为 `processor → block → non-reentrant checkpoint → DiT → model_fn → flow_loss`，每一步显式返回 Tensor/tuple；不使用 attention hook、全局列表或 forward side effect 收集统计。checkpoint 重算不会重复累计监督值。未启用 probe 时所有原输出形式不变。

A 的分子/分母统计见 [`BoundRegionProbe.__call__`](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/training/attention.py#L73)，跨层汇总及两项平方损失见 [`attention_loss`](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/training/attention.py#L102)；[DiT attention processor](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/third_party/diffsynth/diffsynth/models/qwen_image_21_dit.py#L219) 在选定层取同次前向的 Q/K/LSE，[`flow_loss`](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/training/objectives.py#L314) 接收返回值并与 FM 相加。

### 10.5 超参数、梯度校准与日志

| 参数 | CLI 默认 | 启用方案的设置 / 语义 | 来源 |
|---|---|---|---|
| `--region-weight` | 0（关闭 C） | 0.5；Stage 1/2 分别显式启用 | 本项目方案，非两个官方训练默认 |
| `--region-n-min` | 16 | 内外有效面积分母下限，单位 latent token | 本项目方案 |
| `--attention-weight` | 0（关闭 A） | 使用校准 JSON 的 `attention_weight`，不预设万能常数 | 本项目方案；仅 Stage 2 |
| `--attention-read-weight` | 0.5 | $\mu$，三个 mask queries → source 辅助项系数 | 本项目方案 |
| `--attention-layers` | `7 11 15 19 23` | 0-based；全部 32 个头；校验不重复且在 [0,31]，按序存储 | 本项目方案 |
| `--attention-warmup-steps` | 500 | 成功 optimizer updates，不是 microsteps；0 表示立即满权重 | 本项目方案；独立于 LR warmup |
| `calibrate-attention --target-ratio` | 0.2 | A 梯度 / 实际 FM 分支梯度的目标比例 | 方案 0.1–0.3 区间内的校准默认 |
| 校准 samples / timestep indices | 8 / `100 500 900` | manifest 顺序的前至多 8 个合格缓存行，每行三个 scheduler 索引 | 小样本测量入口，不是正式训练采样分布 |

已有学习率、LR scheduler、采样配比、梯度累积、224-module DiT LoRA、保存间隔和训练长度规则保持第 4–5 节的定义。SAMTok Qwen3-VL 官方任务与 DiffSynth 固定版编辑训练都没有本项目的这组 A/C 目标；这里不是复制它们的 loss 参数。

`calibrate-attention` 在同一组可训练 DiT LoRA 参数上，分别测基础 FM、实际 C（含 scheduler weight）和未乘 lambda/warmup 的 A 的裁剪前梯度范数。不执行 optimizer update。每组使用同一噪声、t 和 RNG 状态；为兼容 compiled FlexAttention 的 donated backward buffers，三个目标分别重建前向并反传一次，不使用 `retain_graph=True` 或改全局 compiler 配置。每个测量建议值为 `target_ratio * norm(C) / norm(A)`，最终取中位数；JSON 保存逐测量 norm、建议值及最终系数对应的实测 ratios，而不是宣称每条数据都恰为 0.2。

校准须匹配正式 Stage 2 的 rank、初始化 adapter、C/read/layers 设置。`--init-adapter` 可校准 warm-start，核验其 Stage 2 conditioning identity。零/非有限梯度直接报错，不生成看似有效的系数。正式训练应使用覆盖面积、编辑类型与 t 的代表性 cache；单个 RefEdit 图的 smoke 系数只能验证执行路径。

令 u 为本窗口开始前已成功完成的 optimizer updates：

$$
\lambda_A(u)=\lambda_A^*\min(u/500,1).
$$

因此默认首个累积窗口 A 系数为 0，同一窗口各 microsteps 相同；跳过 optimizer step 不增加 u。启用 A 时即便 warmup 系数暂为 0 仍计算并记录 A。`--init-adapter` 是权重 warm-start 而非完整 resume，u 从 0 开始。

启用任一项后每个窗口写 `supervision_metrics.jsonl`：`optimizer_step/skipped`、跨 rank 的逐样本均值、每个指标的实际样本数 `counts`、`skip_reasons`。基础 FM/C、区域内外 MSE、覆盖面积、截断比例、mask 最大值、均匀注意力基线、A 主辅项、聚合与逐层 r、逐层 mask attention mass 都被记录。不同指标可能仅覆盖合格子集，要结合 counts 解读，不把 NTP 或 plain 缺失的 A 指标补零。`target_mass` 是 $D_T/(heads\cdot|T|)$；`source_mass` 是 $D_S/(heads\cdot3)$；另存 log target mass 便于观察很小数值。原 `optimizer_steps.jsonl` 仍只是 LR/最后 microstep loss。

### 10.6 完整启用顺序与命令

以下是正式流程模板，先把路径和编码身份替换成自己的已审核数据；输出目录必须全新。单卡示例可改为原有 torchrun/accelerate 启动方式。Stage 1 metadata 需要包含随后 Stage 2 使用的相同行（行字段改变后 hash 也改变）。

```bash
export RUN=/path/to/new_run
export DATA=/path/to/data_root
export S1=/path/to/stage1.jsonl
export S2=/path/to/stage2.jsonl
export ENCODER_SHA=checksum_recorded_by_the_data_encoder

# 仅在确认 source/target 全图对齐时使用此认证；否则传 alignment manifest。
python -m samtok_edit21.cli prepare-regions \
  --metadata "$S1" --base-path "$DATA" --output "$RUN/regions" \
  --max-pixels 1048576 --mask-tokenizer-sha256 "$ENCODER_SHA" --assume-aligned

python -m samtok_edit21.training.engine train --stage stage1 \
  --metadata "$S1" --base-path "$DATA" --output "$RUN/stage1" \
  --steps 4000 --save-steps 2000 --max-pixels 1048576 \
  --region-cache "$RUN/regions" --region-weight 0.5

python -m samtok_edit21.training.engine cache \
  --metadata "$S2" --base-path "$DATA" --output "$RUN/cache" \
  --max-pixels 1048576 --te-adapter "$RUN/stage1/adapter" \
  --region-cache "$RUN/regions"

python -m samtok_edit21.cli calibrate-attention \
  --cache "$RUN/cache" --output "$RUN/calibration.json" \
  --rank 32 --samples 8 --timesteps 100 500 900 \
  --region-weight 0.5 --target-ratio 0.2

# 填写 calibration.json 中的 attention_weight；它依赖本次数据/初始化。
export ATTENTION_WEIGHT=value_from_calibration_json
python -m samtok_edit21.training.engine train --stage stage2 \
  --cache "$RUN/cache" --output "$RUN/stage2_ac" \
  --steps 4000 --save-steps 2000 --rank 32 \
  --region-weight 0.5 --attention-weight "$ATTENTION_WEIGHT" \
  --attention-layers 7 11 15 19 23 --attention-read-weight 0.5 \
  --attention-warmup-steps 500
```

Stage 2 训练不传 `--region-cache`，区域已内嵌 conditioning cache。消融在**同一份**带监督 cache、相同 seed/初始化/长度上分别用：基础 FM（两个 weight 都为 0）、C（0.5/0）、A（0/校准系数）、A+C（0.5/校准系数）。A-only 应按实际基础 FM 分支另校准；不能把 C 校准比值当成其实际比值。原 constant/cosine LR 对照仍可继续，除 `--lr-schedule` 外保持条件一致。两阶段 C 的效果若也要隔离，需分别训练 Stage 1 并各自重建 conditioning cache，不能混用 TE adapter。

推理继续用第 8 节命令加载对应 Stage 1/2 adapter，不传 A/C 参数。权重中学到的区域行为与未来 B 的推理约束是两回事。小批量数值验收、实际训练、开销和限制见实验记录第 12 节。


## 11. 四机训练与 W&B

四机实现的总体流程、逐模块修改、关键代码块与源码行号、ARNOLD/W&B 配置、完整调试入口和日志约定统一见 [四机训练运行指南](03_SAMTokEdit_Qwen21_四机实验运行指南.md)。现有两阶段更新范围与数据计算保持不变，新增节点编排、全局 optimizer-update 指标汇总、逐 rank 梯度与训练后参数一致性检查。普通 train CLI 默认不开启 W&B，四机入口明确使用 online 模式。

## 12. 全量数据与 noref 转换实现

**方法要求。** 使用四个数据集的最终通过样本和现成 mask；训练之前物化 NTP/plain/ref/noref 行。转换失败的 1,213 条仅保留 plain。官方 SAMTok/DiffSynth 不提供这四个数据集的适配器，本节代码属于项目新增的数据工程。

**具体实现。** [full_data.accepted](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/preparation/sources.py#L33) 按各源最终字段过滤；`image_asset` 保留图像原始字节、校验可解码性，不重新计算 mask。[annotate_full.model_input](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/preparation/semantic.py#L76) 只把原指令和源类型发送给文本模型；`task_prompt` 添加当前类型的一个例子；模型只生成两字段 JSON。

```python
value = {'instruction': source['instruction'],
         'edit_type': source.get('provisional_type') or source.get('native_type')}
# 模型输出：{"ref_phrase": ["..."], "noref_instruction": "..."}
```

[annotate_full.py](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/preparation/semantic.py#L84) 的后处理绑定已有 mask IDs，规范化 region 指代，检查原文匹配、内容保留和协议。原子类型沿用源标签，仅对未映射的粗类别做细化。生成失败后，[rule_fallback.py](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/preparation/fallback.py#L1) 尝试从原指令截取完整 referent，并保留新内容；输出仍经过同一验证器。自动 accepted 不是人工语义金标。

[corpus.encode_chunk](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/preparation/corpus.py#L154) 的 split → encode-worker → merge 将源/语义结果按 ID 对齐，使用冻结 SAMTok codec 编码，验证每行后写出两阶段 metadata。[regions/build.py](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/regions/build.py#L1) 另生成 coverage，ref/noref 共享同一内容缓存。两者与 Stage 1 adapter 生成的 conditioning cache 是不同产物。

```text
sources.jsonl + semantic_runs/..._003/annotations.jsonl
  → split（成功四行，失败 plain-only）
  → encode-worker（现成 mask → SAMTok code）
  → merge（stage1.jsonl / stage2.jsonl / provenance.jsonl）
  → full_regions（冻结 coverage）
  → Stage 1 → conditioning cache → Stage 2
```

**相同 ref/noref 指令的去重。** `convert_record` 在两种 UMT 指令完全相同时只保留 noref 行，常见于 global 编辑。例如 `Apply a watercolor style to this image.` 的 ref/noref 都是 `Apply a watercolor style to this image <mask>.`，因此产出 NTP、plain、noref 三行。原批量编码器固定要求四行，会误拒绝这种合法结果。2026-10-01 的复核修复了这一边界；[具体校验](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/preparation/corpus.py#L196) 从 NTP 重新构造 ref，确认确实与 noref 一致才接受三行，真正缺失的 ref 和转换错误仍拒绝：

```python
expected = [('edit_ntp', None), ('edit', None),
            ('edit_umt', 'ref'), ('edit_umt', 'noref')]
if not errors and kinds == [expected[0], expected[1], expected[3]]:
    reference = render_units(source['instruction'], grouped_units(
        source['instruction'], parse_cot(rows[0]['mt_cot'])))
    if reference == rows[-1]['prompt']:
        expected.pop(2)
if errors or kinds != expected:
    raise ValueError(...)
```

**批量编码的组内空间排序。** 冻结 codec 的单图 `encode` 会排序，但原 `encode_chunk` 直接展平 source 的 mask_ids 顺序，漏掉了这一步。[批量编码循环](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/preparation/corpus.py#L166) 现在对多 mask 组复用同一 `_ordered_masks`，仅调整已有 mask 的顺序，随后仍使用批量编码：

```python
for group in masks:
    if len(group) > 1:
        _, order = codec._ordered_masks(group)
        group = [group[index] for index in order]
    for mask in group:
        pairs.append((image, mask))
```

这不修改 annotation 的 mask_ids，不改变 mask 内容，也不改变 unit 按指令短语顺序的规则。现有全量文件中的 254 个多 mask 组有 75 组采用历史来源顺序；作为已编码输入，仍按同组连续 span 的已有顺序读取，label、span positions 与 coverage 一致绑定。新准备的数据采用 x 中心优先、y 中心次之的规范顺序。此轮不覆写已启动正式训练所用的历史数据；当前数据的排序现状已明确记录到数据盘点。

当前四源全量数据没有 global 样本，文件内容、行数与现有训练配比不变。全部复核过程与范围见[实验记录第 20 节](02_SAMTokEdit_Qwen21_实验记录.md#20-2026-10-01全量资产复核与新-adapter-完整链路验证)。

**共享文件系统 cache 的续写。** DiffSynth 的 `launch_data_process_task` 仍负责模型前向和 cache 数据格式；[项目适配层](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/training/engine.py#L711) 增加 `--resume-cache`，并把写盘参数传给官方 runner。[写盘实现](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/third_party/diffsynth/diffsynth/diffusion/runner.py#L175) 在重启时按 `_row_index` 读取并复用已经完成的 payload，损坏或身份不符的文件重新前向；新 payload 先写进 rank/进程唯一的临时文件，`fsync` 后用 `os.replace` 发布，`torch.save`/rename 的共享盘异常按指数退避和 rank jitter 重试。cache 汇总阶段由 [_cache_manifest_shard](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/training/engine.py#L596) 让每个 rank 并行校验和生成 sidecar，再由 [_merge_cache_manifest](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/training/engine.py#L649) 合并；Stage 2 入口由 [_distributed_cache_validation](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/training/engine.py#L688) 并行验证各 rank 的 payload。已有且内容匹配的 sidecar 会复用，未通过就不会发布 `manifest.json` 或进入 Stage 2。

四机编排器的 `--stage1-adapter` 会跳过 Stage 1 DDP，直接校验并复用已完成的 adapter；`--cache-output` 可以指向失败作业的 partial cache，`--resume-cache` 继续写入同一目录。新 run 根下只建立旧 Stage 1/cache 的引用别名，使正式全量审计仍能检查原 Stage 1 记录和续写后的 cache。入口和故障实例见[四机指南第 2.1 节](03_SAMTokEdit_Qwen21_四机实验运行指南.md#21-stage-1-已完成后的四机续训入口)。

[annotation_cluster.py](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/distributed/annotation.py#L1) 在四机上启动 32 个独立 TP=1 vLLM worker，按 `inputs[rank::32]` 分片，按源 ID 合并。训练的 32-rank DDP 与此独立副本模式不同。准确目录、统计和真实行例子见[训练数据盘点](04_SAMTokEdit_Qwen21_训练数据盘点.md)，完整可直接提交的 ARNOLD 命令见[四机指南](03_SAMTokEdit_Qwen21_四机实验运行指南.md)。


## 13. 正式训练复用已准备数据的验收报告

**方法与需求。** 全量数据已在训练之前完成图片物化、逐行协议验证、mask code 编码和区域 coverage 构建。正式启动应读取这些产物并开始训练，避免 32 个 rank 各自提前遍历整套图片/coverage。两阶段的更新参数、任务配比、loss 和优化器不变。

**官方起点与项目改动。** DiffSynth runner 不负责本项目的 region cache 验收。项目原 [Stage 1 入口](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/training/engine.py#L417) 在加载模型前对全部非 NTP 行调用 `RegionStore.load`；全量 390,657 行在每个 rank 重复执行，其中 194,618 行会读取 coverage 和图像内容。现由 [cluster.run](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/distributed/training.py#L226) 在 `--full-training` 时自动传入 `--prepared-data-report <data>/metadata_report.json`，由 [verify_prepared_region_report](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/data/preflight.py#L11) 验证离线报告与当前 metadata/manifest 的身份一致。

```python
# src/samtok_edit21/distributed/training.py：仅正式 Stage 1 自动启用
prepared = (["--prepared-data-report", str(data / "metadata_report.json")]
            if a.full_training else [])
```

[预检分支](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/training/engine.py#L395) 仅让全局 rank 0 核对报告，使用已有 `_main_rank_result` 把结果或错误广播给全部 rank：

```python
if args.prepared_data_report:
    result = _main_rank_result(accelerator, lambda: verify_prepared_region_report(
        args.prepared_data_report, args.metadata, args.region_cache,
        args.max_pixels, len(rows)))
else:
    # 未提供离线报告的普通/debug 入口保留逐行预检。
    store = RegionStore(args.region_cache, args.max_pixels)
    for row in rows:
        if row["sample_type"] != "edit_ntp":
            store.load(row, args.base_path)
```

报告须属于同一版本目录，且 `training_ready`、`region_cache_ready` 均为 true；重算 Stage 1 JSONL 和 region manifest 的 SHA256，匹配报告；检查 schema、geometry、max_pixels、metadata hash、identity hash、总行数和三类区域计数。检查读取 JSONL/manifest，不逐条打开图片或 `.pt`。正式 report 的实测验证耗时约 2.40 秒；本次没有新四机训练耗时结论。

训练正常读取 metadata 并构造真实 schedule；[RegionStore.load](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/regions/supervision.py#L146) 在首次消费相应文件时仍核对 coverage 和图像 hash、张量形状与协议。报告复用的是准备阶段的验证结果，不意味着启动时重新检查了每一个资产文件。Stage 1 后新生成的 conditioning cache 仍走既有 merge/Stage 2 完整性验证，因为它不属于此前离线准备的产物。

[启动日志](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/training/engine.py#L385) 在各 rank 输出 metadata_load、region_preflight、model_identity、wandb_init、model_load、training 的开始/完成；Stage 2 另记录 conditioning_cache_validation。rank 0 将对应记录写入阶段目录的 `startup.jsonl`，`run.json.data_preflight` 保存报告核对结果。`--plan-only` 只打印，不写输出目录。[节点编排](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/distributed/training.py#L142) 每 60 秒输出子进程日志路径、大小、距最后修改秒数，并写 `nodes/<node>/<phase>.progress.json`；这个心跳证明编排器仍在等待，是否完成 optimizer update 要看 `training_metrics.jsonl`。


## 14. loss、梯度更新与 scheduler：零权重修复

### 14.1 总体说明与官方边界

方法仍是 Stage 1 更新 TE language LoRA，Stage 2 冻结缓存条件、更新 DiT LoRA。NTP/FM、区域加权 C、注意力监督 A 的公式和正式系数不变；梯度累积、裁剪、AdamW 和 LR scheduler 的更新时钟不变。此次改动修复项目额外添加的梯度审计：**有限、保持计算图连接的零 loss/零梯度合法**。审计用于调试和故障诊断，不是 DiffSynth 或 SAMTok 官方要求。

DiffSynth 的 [官方 set_training_weight](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/third_party/diffsynth/diffsynth/diffusion/flow_match.py#L365) 在 1000 个训练 timestep 下把曲线减去最小值再归一，因此最高噪声端点 `t=1000` 的 `training_weight=0`。[官方 FlowMatchSFTLoss](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/third_party/diffsynth/diffsynth/diffusion/loss.py#L5) 均匀抽 timestep，并先转为 pipeline dtype 再计算权重；项目沿用这一行为。BF16 下抽到索引 0–4，原始 timestep 约为 1000、999.558、999.116、998.674、998.231，都会舍入为 1000，因此这五种抽样产生零 FM 权重。不会排除这些 timestep，不给权重加 epsilon，也不 detach 零 loss。

### 14.2 各 loss 实际进入 backward 的方式

[SamtokTrainingModule.forward](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/training/engine.py#L224) 负责 Stage 1 分支系数；[flow_loss](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/training/objectives.py#L250) 负责 FM/C/A。`w(t)` 是官方训练权重，`L_C` 是第 10.4 节按实际总权重归一后的区域 FM；不满足 C 条件时用全画布 MSE。正式配置为：

| 阶段/行类型 | 单条样本用于 backward 的 loss |
|---|---|
| Stage 1 NTP | `0.05 × L_NTP` |
| Stage 1 UMT ref/noref | `1 × w(t) × L_C`，C=0.5；区域不合格时退到 MSE |
| Stage 1 plain edit | `1 × w(t) × MSE` |
| Stage 2 UMT ref/noref | `w(t) × L_C + λ_A(u) × L_A`；区域不合格时退到基础 FM |
| Stage 2 plain edit | `w(t) × MSE` |

`λ_A(u)=0.1×min(u/500,1)`，`u` 是本窗口开始前已成功完成的 optimizer updates。[A warmup](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/training/engine.py#L183) 在同一累积窗口使用同一个 u；第一窗口 A 系数为 0，第二窗口为 0.0002。A 不乘 w(t)，所以 Stage 2 在 w=0 且 A 已启用时仍可有梯度。C 在 FM 内部，因此随 w(t) 一起为零。Stage 1 的 `ntp_weight/fm_weight` 属于混合任务系数，Stage 2 当前配方直接使用 FM+A。

```python
# 摘自 src/samtok_edit21/training/objectives.py：不改官方 timestep 的实际计算值
original_t = pipe.scheduler.timesteps[i]
t = original_t.to(device=pipe.device, dtype=pipe.torch_dtype)
weight = pipe.scheduler.training_weight(t).to(pipe.device)
basic_fm = mse * weight
# 合格区域且启用 C 时：fm = regional * weight
loss = fm + attention_weight * aux
```

例如 Stage 1 同一全局窗口的三类 NTP/ref/noref/plain 数为 96/64/64/32。累计目标是所有 256 条加权样本的均值，即 `3/8×0.05×mean(NTP)+5/8×mean(FM)`。日志中的 `loss_ntp`、`loss_fm` 是各自出现样本上的分支均值，`weighted_total` 是实际 backward loss 的全样本均值，不能直接把两个分支均值相加当总 loss。

### 14.3 修复梯度审计与事后验收

旧项目检查 `grad_norm>0 && current_backward_grad_peak>0`，会把官方零权重样本误判为断链。新模块 [audit_backward](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/training/gradient_audit.py#L30) 将合法性与非零性分开：至少有可训练参数的 `.grad`、当前 backward 的参数 hook 确实被执行、累计 norm 与当前 hook peak 有限、冻结参数没有 `.grad`。[当前 backward 的参数 hook](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/training/engine.py#L175) 观察当前 backward，避免前面 microstep 的累积梯度掩盖本步未触达参数。LoRA A 初始梯度为零仍合法。

```python
# 摘自 src/samtok_edit21/training/gradient_audit.py：零值不是错误条件
if not norms:
    errors.append("missing_trainable_gradients")
if not peaks:
    errors.append("missing_current_backward_hooks")
if not math.isfinite(total) or not math.isfinite(peak):
    errors.append("nonfinite_gradient")
if frozen:
    errors.append("frozen_parameters_with_gradients")
```

每 rank 的 `gradients-rank<RANK>.jsonl` 在报错前也写完整记录：microstep、已完成更新数、分支、row_sha256、实际 timestep、抽样索引、转 BF16 前 timestep、training_weight、loss、当前 hook peak 与累积 norm。`current_backward_zero=true` 不代表整个累积窗口没有梯度；`accumulated_grad_zero` 独立记录此状态。原因字段区分 `fm_scheduler_weight_zero` 与 `finite_zero_backward`，后者也允许正权重下恰好为零的合法导数。

[validate_gradient_record](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/training/gradient_audit.py#L13) 和 [audit_gradient_logs](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/training/gradient_audit.py#L83) 是训练及事后验收共享的判据，[正式实验事后验收](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/scripts/diagnostics/audit_full_training.py#L60) 与 [debug 事后验收](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/scripts/diagnostics/audit_debug_run.py#L28) 均调用，防止运行成功后又被旧的“非零”规则拒绝。历史缺少新增 hook 计数字段的日志可只读复核；新运行记录实际 hook 数。[on_optimizer_step](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/training/engine.py#L192) 聚合每个窗口的零梯度原因；[W&B 日志](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/training/tracking.py#L97) 记录 `train/zero_backward`、`train/zero_weight_fm` 和 `gradient_zero/<reason>`。这两个 train 指标分别是全样本零 backward 比率和全样本零 FM 权重比率，不是累计 loss 权重。

### 14.4 累积、AdamW 与 LR 的更新时钟

[runner 更新段](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/third_party/diffsynth/diffsynth/diffusion/runner.py#L137) 沿用 Accelerate 的累积语义：`accelerator.backward(loss)` 自动除以 accumulation；非同步 microstep 的包装 optimizer 不执行真正 step/zero_grad，保留累积梯度；同步边界进行 DDP 梯度平均、max_grad_norm=1 裁剪、AdamW 更新，然后清梯度。Stage 1 为 8 microsteps/update，Stage 2 为 4；32 ranks 对应全局 batch 256/128。

项目 [scheduler_factory](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/training/engine.py#L358) 使用自定义 LambdaLR。[自定义 scheduler 与 Accelerate 的准备路径](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/third_party/diffsynth/diffsynth/diffusion/runner.py#L111) 不把这个自定义 scheduler 交给 AcceleratedScheduler，从而不会按 world_size 重复推进；它只在同步且 optimizer 未被 scaler 跳过时推进一次。零 loss/零梯度不是 scaler 跳步：即便整个窗口恰好为零，AdamW 的历史动量和 weight decay 仍可能更新参数，LR、成功 update 数及 A warmup 正常推进。只有实际 optimizer 跳步时，这三个时钟都不推进。

```python
# 摘自 DiffSynth runner：项目传入 scheduler_factory 的路径
effective_lr = optimizer.param_groups[0]["lr"]  # 本次真正使用的 LR
optimizer.step()
if accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped:
    scheduler.step()
    optimizer_step += 1
# 同步窗口结束后回调，再 zero_grad；Accelerate 保留非同步步的累积梯度
```

正式 warmup 经 [resolve_warmup_steps](https://github.com/Tangent0308/samtok_edit/blob/a93fe568bcbaa8e767778d1d594d6e7c0010b28a/src/samtok_edit21/training/engine.py#L345) 向上取整：Stage 1 `ceil(3081×0.04)=124`，Stage 2 `ceil(3081×0.025)=78`。第 u 次更新（从 1 开始）在 warmup 内使用 `base_lr×u/warmup_steps`；Stage 1 此后 cosine，Stage 2 此后 constant。`optimizer_steps.jsonl.lr` 和 W&B `train/lr` 记录本次应用的 LR，不是 step 后下一次 LR。checkpoint 保存间隔仍按 microsteps 计数，adapter 快照不包含 optimizer/scheduler resume 状态。

本地验证结果和完整复现指令见[实验记录第 18 节](02_SAMTokEdit_Qwen21_实验记录.md#18-2026-09-30正式-_002-零梯度误报与更新链路验证)。四机使用新的 run ID 和新分支代码，入口见[四机指南第 2 节](03_SAMTokEdit_Qwen21_四机实验运行指南.md#2-正式全量训练入口)。
