> 历史归档：保留当时的实验与命令；当前启动入口及数据状态以 [四份主文档](../README.md) 为准。旧 run ID 不应直接重用。

# noref prompt 优化与 Qwen3 / Qwen3.5 对照实验

实验日期：2026-09-29。本页是上一轮 prompt 和模型选择实验；**当前精简 prompt 与 thinking 开关实测**见[三例与 thinking 对照](SAMTokEdit_Qwen21_noref三例Prompt与Thinking对照.md)，完整当前规则见[两字段转换说明第 3 节](SAMTokEdit_Qwen21_noref两字段转换与模型对比.md#3-当前完整-prompt)。9B 的上一轮拒绝与人工式文本审阅漏检实例见[9B 未通过样本审计](SAMTokEdit_Qwen21_9B未通过样本审计.md)。本记录接续[失败分析](SAMTokEdit_Qwen21_noref失败分析与修复.md)，所有模型仅输出 `ref_phrase` 与 `noref_instruction`；不生成类型、mask ID 或自评。继续使用数据集已有 mask，不判断几何准确性。

## 1. 方法、比较口径与样本

本轮先比较精简 prompt 和小幅定向修改，再用选定 prompt 比较 Qwen3-4B-Instruct-2507、Qwen3-8B 和 Qwen3.5-9B。三者不是同一个发行版仅改变参数量，因此结果代表具体模型的任务表现。

- 对照输入 312 条：246 条新分层抽样 + 66 条历史问题回归。前者按四个数据集与原生类型分组，每组最多 8 条；后者刻意包含困难案例。因此整体程序通过率不能外推到全量数据。
- 对照集数据集数量：CrispEdit 61、Derived 35、RefEdit 45、ScaleEdit 171。
- 在生成前固定其中 62 条供逐条文本审阅，每个数据集×原生类型组 2 条。
- 另取 60 条留出样本，与前 312 条无 ID 重叠，未用于本轮 prompt 选择；CrispEdit 10、Derived 8、RefEdit 10、ScaleEdit 32。
- 严格文本审阅要求：完整原文 reference、正确对象/部件、保留 NEW 内容与额外约束、消除 OLD 定位、不把比较对象当作额外编辑单元、region 一一对应。审阅由助手逐条阅读完成，**不是独立人工金标，也不是统计上已验证的全量准确率**。程序拒绝但语义合理的同义改写，单独计入“文本审阅通过”；真正可进入候选的是“程序与审阅同时通过”。

本机已有用户 GPU 常驻负载（`run.py --size 8000 --gpus 8 --interval 0.0005`），本轮保持原状。三个模型依次测试、互不同时加载，但不是无其他负载的隔离性能基准；吞吐只能作为当前机器条件下的比较。

三个模型都关闭 thinking，H100 80GB 同一张 GPU 依次运行，BF16 / TP=1 / batch=64 / temperature=0 / seed=20260928 / max_model_len=8192 / max_tokens=512 / 最多三次尝试。重试包含上一条输出与错误信息。9B 仅启用语言模型，不加载视觉模块，也不开 MTP。

## 2. Prompt：保留主要约束，只改容易误解的边界

### 2.1 总体说明

试验过较短共享 prompt、补回属性约束的精简 prompt、按类型拆分的短 prompt；程序通过率与逐条审阅没有显示出稳定收益。最终保留原先有效约束，只增加一句 add 需要保留衣着、外观和姿态，并替换两个例子。

当前共享 prompt 276 个英文词（旧版 274），每条仅附本类一个例子。按 4B tokenizer 和本次类型分布计算，规则加例子平均约 434 tokens，旧版约 424；没有堆叠更多全类型 few-shot。完整 prompt 见[两字段转换说明第 3 节](SAMTokEdit_Qwen21_noref两字段转换与模型对比.md#3-当前完整-prompt)。

### 2.2 详细实现

方法要求 add 的 NEW 内容仍出现在 noref，而 OLD 位置由 region 代替；replace 要分清描述旧物体的 `with` 与引出替换物的 `with`。[annotate_full.py:19](../../samtok_edit21/annotation/annotate_full.py#L19) 的共享规则新增：

```text
For addition keep ALL new content, including clothing/appearance/pose;
replace only placement with "in this region".
```

[annotate_full.py:48](../../samtok_edit21/annotation/annotate_full.py#L48) 中仅改 add / replace 示例：

```text
Add a woman in a green coat and white boots beside the bus.
ref_phrase = ["woman in a green coat and white boots beside the bus"]
noref_instruction = "Add a woman in a green coat and white boots in this region."

Replace the house with a tiled roof on the left with a glass tower.
ref_phrase = ["house with a tiled roof on the left"]
noref_instruction = "Replace this region with a glass tower."
```

输入仍只需 instruction 与数据集已有 edit_type；输出 schema 不增加字段。程序的类型映射、原 mask 绑定、后处理与语义检查在 prompt 对照期间保持不变，没有通过放宽校验提高通过率。

### 2.3 旧环境下 4B / 8B 的 prompt 对照

以下使用同一个 vLLM 0.10.2 环境，模型初始化不计入转换耗时。

| 模型 / prompt | 程序通过 / 312 | 固定 62 条文本审阅通过 | 转换时间 | 输入条/秒 |
|---|---:|---:|---:|---:|
| 4B 原 prompt | 239 | 43 | 12.51 秒 | 24.93 |
| 4B 当前小改 | 236 | 48 | 12.68 秒 | 24.61 |
| 8B 原 prompt | 265 | 47 | 15.21 秒 | 20.52 |
| 8B 当前小改 | 259 | 45 | 15.43 秒 | 20.23 |

当前小改在额外 60 条留出集上：4B 文本审阅通过 47 条、程序与审阅同时通过 45 条；8B 两者均为 46 条。合计 122 条文本审阅，4B 为 95，8B 为 91。8B 没有显示稳定质量优势。这里的模型选择结论仅针对这批样本，不把程序 accepted 直接称为准确。

精简方案的程序通过数 / 312：最初精简 4B=226 / 8B=236；补约束精简 228 / 239；按类型拆分短 prompt 226 / 223。因此没有采用这些版本。

## 3. Qwen3.5-9B 文件与运行适配

### 3.1 模型文件

用户指定目录：

```text
/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen3.5-9B/
```

官方仓库 `Qwen/Qwen3.5-9B`，固定 revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`。共 16 个上游文件，19,329,393,661 bytes，另存 `download_manifest.json`。共享目录已有与该 revision 一致的文件，因此本次实际采用**校验官方摘要后复制缓存**，没有重复从网络下载 19GB 权重。四个权重分片与 tokenizer 的 LFS 文件逐个校验 SHA256，其他文件校验 Git blob SHA1；目标文件再次校验后才原子改名为正式文件。

官方参考：[模型卡](https://huggingface.co/Qwen/Qwen3.5-9B)、[vLLM Qwen3.5-9B recipe](https://recipes.vllm.ai/Qwen/Qwen3.5-9B)、[纯文本模式说明](https://docs.vllm.ai/en/v0.17.0/models/supported_models/)。9B 是带视觉组件的模型；此次任务只使用文本。

### 3.2 代码适配

新建隔离环境 `/tmp/samtok21-qwen35-compare-env`，vLLM 0.17.1 / torch 2.10.0+cu128 / transformers 4.57.6。训练与原四机标注环境不在此处升级。环境锁定清单保存在实验目录 `requirements.lock.txt`。项目另需 pyarrow；启动必须将该环境的 bin 加入 PATH，让 FlashInfer 能找到 ninja。首次测试分别暴露缺 pyarrow 与 PATH 未包含 ninja，补齐后重新完整运行；失败启动不计入正式吞吐。

[annotate_full.py:409](../../samtok_edit21/annotation/annotate_full.py#L409) 兼容新版 JSON 约束 API，同时保留原 vLLM 0.10.2 路径：

```python
if 'structured_outputs' in SamplingParams.__struct_fields__:
    json_decoding = {'structured_outputs': StructuredOutputsParams(json=OUTPUT_SCHEMA)}
else:
    json_decoding = {'guided_decoding': GuidedDecodingParams(json=OUTPUT_SCHEMA)}
# 同一个 OUTPUT_SCHEMA，仍只有 ref_phrase 与 noref_instruction。
```

[模型识别](../../samtok_edit21/annotation/annotate_full.py#L437)根据 config 的 `model_type` 识别 Qwen3.5，传入 `language_model_only=True`；若环境不支持，提前报清晰错误。模型初始化参数保持一致。[运行身份](../../samtok_edit21/annotation/annotate_full.py#L454)增加 vLLM / torch / transformers 版本与纯文本模式，避免跨环境结果被无提示续写到同一目录。

```python
if model_config.get('model_type') in {'qwen3_5', 'qwen3_5_moe'}:
    engine_options['language_model_only'] = True
# apply_chat_template(..., enable_thinking=False) 继续适用于三个模型。
```

## 4. 同一新环境下三个模型的实测

### 4.1 结果与计时口径

| 指标 | Qwen3-4B-Instruct-2507 | Qwen3-8B | Qwen3.5-9B |
|---|---:|---:|---:|
| 程序通过 / 312 对照输入 | 234/312 | 258/312 | 270/312 |
| 程序通过 / 60 留出输入 | 49/60 | 54/60 | 56/60 |
| 严格文本审阅通过 / 固定 62 条 | 48/62 | 46/62 | 51/62 |
| 严格文本审阅通过 / 留出 60 条 | 48/60 | 46/60 | 53/60 |
| 合计严格文本审阅通过 | 96/122（78.7%） | 92/122（75.4%） | 104/122（85.2%） |
| 程序与文本审阅同时通过 | 94/122（77.0%） | 92/122（75.4%） | 102/122（83.6%） |
| 122 条中被程序接受但审阅失败 | 8 | 17 | 9 |
| 312 条实际转换耗时，含重试 | 12.98 秒 | 14.91 秒 | 26.33 秒 |
| 输入吞吐 / 秒 | 24.04 条 | 20.92 条 | 11.85 条 |
| 生成请求数 / 输出 tokens | 519 / 17039 | 456 / 15513 | 423 / 15220 |
| LLM 初始化耗时（独立列出） | 51.62 秒 | 37.29 秒 | 29.91 秒 |
| 完整子进程耗时，含 import / 模型哈希 / 初始化 | 79.37 秒 | 73.95 秒 | 80.72 秒 |

**本批结果支持优先考虑 9B 获取更好的转换质量；4B 保留速度优势，8B 没有显示稳定收益。** 9B 文本审阅比 4B 多通过 8 条（分母 122），转换耗时约为 4B 的 2.03 倍、吞吐低约 50.7%。这些是分层小样本结果，没有把它外推为全量准确率。

9B 第一次成功跑 312 条时，生成阶段含 FlashInfer GDN 首次 JIT，总计 183.72 秒。表内采用内核缓存完成后的独立重跑：新建 LLM、新建输出目录、相同 312 条与生成设置；不是利用旧 accepted 跳过数据。冷/热两轮逐 ID 对照：312 条 status 全部一致；5 条记录的原始答案/重试文本有变化，分别为 1 条 JSON 空白变化、3 条冠词变化、1 条 catcher 样本去掉旧位置的语义改善。固定 62 条审阅中只有箭头样本的冠词变化，仍因 add 只引用载体而失败，审阅计数不变。固定 seed / temperature=0 不意味着跨进程输出逐字节一致。4B / 8B 的 torch.compile 发生在 LLM 初始化中，已从实际转换耗时排除。表中初始化数字会受编译缓存影响，不将它用于模型稳态速度排名。

相比旧 vLLM，4B / 8B 的部分句子有小变化：在相同固定 122 条上，与已逐条审阅的旧输出比较，仅有 2 / 4 条两字段内容不同；这些变更已重新逐条阅读，其余沿用完全相同文本的审阅结论。没有把新环境的程序计数与旧环境的语义计数混用。

所有六组输出均验证行数、唯一 ID、源 ID 集合与 complete 文件 SHA256；9B 热重跑另做同样检查。新旧环境各 30 项协议回归检查通过；旧环境额外实际运行两条输入验证兼容路径。模型 index 的 775 个张量条目均指向已校验的四个分片。运行结束时出现 xgrammar/nanobind 的退出期警告，正式进程退出码均为 0，结果完整；本轮没有做长时间常驻服务内存测试。


### 4.2 具体改进与尚未解决的例子

本轮直接比较同一输入下的实际两字段结果；下面的 9B 输出均来自保存的日志。

**正确定位部件**：原文 `Change the reading material of the person on the left to a magazine`，9B 提取 `reading material of the person on the left`，生成 `Change this region to a magazine.`，没有把整个 person 当作编辑对象，也没有在 region 后重复 reading material。

**正确去除 OLD 文字和载体**：原文 `Change the text on the white sign held by the woman in the center from 'STOP RACISM' to 'STOP EQUALITY'.`，9B 只引用 `'STOP RACISM'`，生成 `Replace this region with 'STOP EQUALITY'.`。旧文字与其位置不再残留。

**正确区分两个 with**：`Replace the large building with a green roof on the left with a modern glass skyscraper.`，三个模型在当前 prompt 下均能生成 `Replace this region with a modern glass skyscraper.`，旧建筑的 green roof 不再误当作新增要求。

**add 内容修复仍有边界残留**：此前 `add the right model in the denim jacket over a light blue shirt and bright orange pants` 的衣着被删掉；当前三个模型均保留了完整衣着，但生成文本仍保留 `right model`。这里只能确认“丢衣着”得到改善，不能将整条样本声称为完美 noref。

**仍被程序接受的旧位置残留**：9B 对 `Replace the text 'DEC 10 MON 8/7c' with 'NOV15-TUE9/8P' under the ARROW title.` 生成 `Replace this region with 'NOV15-TUE9/8P' under the ARROW title.`。NEW 正确，但旧载体定位仍在；程序接受，严格文本审阅不通过。

**仍有复杂 add 与比较对象错误**：给两片叶子分别加 smudge / spot，9B 的 ref 仍可能只选原有叶子而不含新增内容；缩小左包以匹配右包，仍可能把不编辑的右包也写进 ref 列表。后者 region 数不匹配会被程序拒绝，前者存在漏检。

**单字母画符号仍有类型边界问题**：`Draw an 'O' in the bottom-right cell of the tic-tac-toe board.`，三个模型都只引用格子，保留 `Draw an 'O' in this region`。语义接近文字插入，但当前程序从原始粗类型细化为 add，按 add 协议 ref 应包括新增内容；单字符 O 未被已有词项检查可靠捕获。不能把 accepted 当作该类型映射已经正确的证明。本轮保留该问题，没有为了模型比较临时更改映射或 mask。

此外，部分拒绝是现有检查过严：9B 将 `A character lowers their pointing arm.` 转为 ref=`pointing arm`、noref=`A character lowers this region.`，文本审阅认为操作与目标部件合理，但词项检查报缺少 `their`。因此报告区分纯文本通过与真正被程序接受的可用候选。

## 5. 结果追溯与复现

Prompt 对照目录 `/tmp/samtok21-noref-prompt-ab-20260929/` 保存所有 prompt 代码快照、输入、原始输出、计时、审阅记录。9B 与同环境复测目录 `/tmp/samtok21-qwen35-compare/` 保存模型下载摘要、环境锁、代码快照和每个模型的完整输出；调试脚本与实验数据均不放入 repo。

```bash
cd /opt/tiger/tanyue/samtok_edit_qwen-image-2.1-dev
PATH="/tmp/samtok21-qwen35-compare-env/bin:$PATH" \
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
  /tmp/samtok21-qwen35-compare-env/bin/python -m samtok_edit21.annotate_full \
  --sources /tmp/samtok21-noref-prompt-ab-20260929/benchmark.jsonl \
  --output /tmp/samtok21-qwen35-compare/manual-new-run \
  --model /mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen3.5-9B \
  --batch-size 64
```

这是本地比较命令，无 W&B。当前[四机入口](SAMTokEdit_Qwen21_四机训练运行指南.md#73-完整-arnold-入口)已选 9B 与 vLLM 0.17.1；四节点全量仍需 ARNOLD 作业验证。全量训练数据目录没有被本轮小样本结果覆盖。
