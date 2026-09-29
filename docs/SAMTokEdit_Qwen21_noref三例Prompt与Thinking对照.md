# noref 精简 prompt 与 Qwen3.5-9B thinking 对照

实验日期：2026-09-29。结论：保留**关闭 thinking**作为四机默认；采用较短的共享规则，每条只附其类型对应的一个示例。向每条输入统一附三个示例的候选已实际测试，但在本批样本上比按类型附一个示例差，因此没有进入默认配置。模型始终只输出 `ref_phrase` 与 `noref_instruction`；编辑类型、mask ID 与已有 mask 均由原有程序处理，没有重算或核对 mask。

## Prompt 与三个直观例子

[共享规则](../samtok_edit21/annotate_full.py#L19)保留原版有效的边界：reference 必须是原文完整对象或部件（add 为完整新对象加摆放位置）；noref 消去旧定位，但保留新内容、数量、约束和未编辑的比较对象。[task_prompt](../samtok_edit21/annotate_full.py#L51)只附当前类型的一个示例，避免把其他类型的模式带入答案。**共享规则全文**和全部类型示例见[两字段转换说明第 3 节](SAMTokEdit_Qwen21_noref两字段转换与模型对比.md#3-当前完整-prompt)。典型三例：

| 输入 | `ref_phrase` | `noref_instruction` |
|---|---|---|
| `Add a woman in a green coat and white boots beside the bus.` | `["woman in a green coat and white boots beside the bus"]` | `Add a woman in a green coat and white boots in this region.` |
| `Replace the house with a tiled roof on the left with a glass tower.` | `["house with a tiled roof on the left"]` | `Replace this region with a glass tower.` |
| `Replace the text 'Exit' with 'Open' on the sign.` | `["'Exit'"]` | `Replace this region with 'Open'.` |

与上一版相同的七个类型示例仍存放于[示例映射](../samtok_edit21/annotate_full.py#L40)中，每次请求仅选一个；粗类别没有可靠映射时不强加示例。使用 Qwen3.5-9B tokenizer，含单个示例的规则由 451–475 tokens 缩至 373–397 tokens，**各类型恰少 78 tokens**（约 16%–17%）。这是 prompt token 数，不包括输入 instruction、chat 模板和重试反馈。

## 数据、判断口径与 prompt 选择

同一份 312 条分层/回归样本用于选择，固定其中 62 条逐条文本审阅；另有 **ID 不重叠**的 60 条留出样本。四个数据集均在其中，历史抽样与旧版 9B 的逐条记录见[上一轮实验](SAMTokEdit_Qwen21_noref提示词优化与模型复测.md)。以下程序通过只表示通过现有确定性检查，不代表语义或 mask 几何正确。严格文本审阅检查完整原文指代、NEW 内容和约束、OLD 位置消除、region 与编辑目标的一一对应；这是助手逐条文本检查，**没有独立人工金标**。

| 配置 | 312 条程序通过 | 60 条留出程序通过 | 固定 122 条文本审阅通过 | 固定 122 条程序和审阅均通过 | 312 条生成/检查耗时 |
|---|---:|---:|---:|---:|---:|
| 原版规则 + 按类型一个示例，thinking 关 | 270 | 56 | 104 | 102 | 26.33 秒 |
| 精简规则 + 按类型一个示例，thinking 关（采用） | **273** | **58** | **108** | **107** | 26.10 秒 |

精简版固定 62 条中有 52 条审阅通过、51 条同时通过；留出 60 条分别为 56 与 56。旧版相应为 51/50、53/52。312 条含重试请求从旧版 423 次变为 403 次；虽然 prompt 缩短，输出 tokens 从 15,220 变为 16,920，因此这批稳态总耗时几乎没有变化。首次 CUDA/JIT 与模型加载未计入生成耗时；两次用同型号 H100 和同一 vLLM 0.17.1 环境，非隔离性能基准。

真实例子说明改善与边界：`The person lifts the kitten closer to their face.` 旧版把未编辑的 kitten 也列成 region，精简版只引用 `person`，动作与 kitten 仍留在 noref。`Remove the middle lunch box from the stack...` 精简版把 `from the stack` 纳入旧 reference 并从 noref 移除。`Draw a frowning face ... in the bottom right square` 精简版保留新增脸的细节并引用新增内容加位置。反例也存在：`Add more liquid to the left test tube ... with the right test tube` 在 312 条中把不变的右试管额外列成编辑目标；`The person in the center lowers the rifle...` 仍留下 `in the center`。程序会接受这种文本问题，不能把 273/312 当成真实准确率。

尝试的三个**所有输入统一附三个示例**版本，在同一 312 条上分别通过 266、267、249 条；另一版以更简短措辞附 add/action/text 三例，通过 261 条。它们都未被采用。各候选的代码快照、输出和对照脚本保存在 `/tmp/samtok21-threecase-thinking-20260929/`；旧版输出在 `/tmp/samtok21-qwen35-compare/`。最终精简版原始结果为 `benchmark_off_v5/`、`holdout_off_v5/`，逐条审阅为 `reviewed_v5.jsonl`。

## thinking 开关：同 prompt、同 16 条四数据集样本

[生成路径](../samtok_edit21/annotate_full.py#L501)默认 `enable_thinking=False`、`max_tokens=512`。调试时可加 `--thinking --thinking-max-tokens 3072`，只支持已测试的 Qwen3.5-9B / vLLM 0.17.1；启用 `qwen3` reasoning parser。Qwen3.5 的 `<think>` 开始标记由 chat 模板放入输入，因此原始生成文本可能直接从推理正文开始，[parse_output](../samtok_edit21/annotate_full.py#L257)只取 `</think>` 后的 JSON。没有结束标记或 JSON 时保留失败记录，不把推理文本当作标注。[输出解析回归检查](../tests/test_annotation_output.py#L1)覆盖这个边界。

从留出集中预先固定 16 条、每个数据集 4 条，两个开关都只尝试一次，temperature=0、BF16、TP=1、batch=16、同一 JSON schema、同一个精简 prompt。关闭 thinking 用生产的 512-token 输出上限；打开 thinking 给 3072 tokens，以容纳推理和答案。模型初始化不计时。

```bash
cd /opt/tiger/tanyue/samtok_edit_qwen-image-2.1-dev
export PATH="/tmp/samtok21-qwen35-compare-env/bin:$PATH"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
MODEL=/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen3.5-9B
SOURCE=/tmp/samtok21-threecase-thinking-20260929/balanced16.jsonl
CUDA_VISIBLE_DEVICES=5 python -m samtok_edit21.annotate_full \
  --sources "$SOURCE" --output /tmp/samtok21-threecase-thinking-20260929/reproduce_off \
  --model "$MODEL" --batch-size 16 --attempts 1
CUDA_VISIBLE_DEVICES=4 python -m samtok_edit21.annotate_full \
  --sources "$SOURCE" --output /tmp/samtok21-threecase-thinking-20260929/reproduce_on \
  --model "$MODEL" --batch-size 16 --attempts 1 --thinking --thinking-max-tokens 3072
```

| 指标 | thinking 关 | thinking 开 |
|---|---:|---:|
| 通过确定性检查 / 16 | **16** | **5** |
| 未产生完整 JSON / 16 | 0 | **11** |
| 生成 tokens | 686 | 46,968 |
| 生成/检查时间 | 2.68 秒 | 53.67 秒 |

打开后 11 条输出恰好达到 3072-token 上限，全部停留在推理正文，没有 `</think>` 或 JSON；剩余 5 条的两字段结果与关闭时语义相同，没有观察到新纠正的案例。较低的 1536-token 预算也曾在另一三例候选上试跑四数据集 32 条，三次重试后仅 2 条通过，耗时 162.98 秒。**这只是本模型、本 prompt 与 vLLM 结构化输出配置的结果**；不能推断所有 thinking 模式都无益。以现有配置开展 98,574 条全量转换时不应打开 thinking。四机入口没有传 `--thinking`，仍用默认关闭。

本次没有改变训练数据协议、原 mask、类型映射、语义校验或四机分片逻辑。生产新 prompt 的身份哈希与旧 prompt 不同；[四机指南](SAMTokEdit_Qwen21_四机训练运行指南.md#73-完整-arnold-入口)要求使用新的 run ID，不可把旧 prompt 的 accepted 记录混入同一续跑。实际输出仍落在 `$SAMTOK_DATA_EXPERIMENT/data/semantic_runs/$SAMTOK_ANNOTATION_RUN_ID/`。
