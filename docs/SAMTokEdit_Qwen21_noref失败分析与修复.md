# noref 转换失败分析与修复（2026-09-29）

最新 prompt 优化及 4B / 8B / Qwen3.5-9B 复测见[模型复测记录](SAMTokEdit_Qwen21_noref提示词优化与模型复测.md)。本页历史实验数字保留原口径。

## 1. 结论与范围

**已证实旧 prompt 的 color 示例会污染输出；同时存在 reference 校验误判和语义校验漏检。** 本次修改了 prompt、纠错反馈和确定性检查，使用本地 Qwen3-4B-Instruct-2507 + vLLM 0.10.2 完成实际生成、八卡分片与严格汇总验证。模型输出仍只有 ref_phrase 和 noref_instruction；不输出编辑类型、anchor、mask ID。

旧四机 `qwen21_noref4n_full_002` 已正常处理完 98,574 条，其中 96,270 accepted、2,304 failed。这里的 failed 是逐条转换未通过检查，**不是 ARNOLD/vLLM 作业崩溃**。更早 `001` 的 progress 文件瞬时消失属于共享盘读取竞争，已另行修复，与 prompt 无关。

本次完整重试原先 2,304 条失败记录：**1,422 accepted、882 failed**。旧 979 条异常 color 中，842 条恢复为 accepted。八卡完成覆盖、归属、身份与 hash 校验，写出 SUCCESS；没有 CANDIDATES_COMPLETE。accepted 仍只是候选，不能直接宣称语义全部正确。

**尚未完成全量语义验收，也未把本次输出与旧 96,270 条自动合并或用于训练。** 35 条分类型 accepted 抽查发现 3 条需继续复核，详情见第 5 节。旧 accepted 的保守复查也筛出 12,362 条疑点，其中包含校验误报，不能称为 12,362 条已确认错误。

## 2. 为什么会失败

### 2.1 示例污染有实际生成与消融证据

旧 prompt 给所有编辑类别同时展示 color/grapes/OLD/NEW 等示例。4B 会套用示例结构，甚至把例子的内容带到不相关输入中：

| 实际输入意图 | 旧输出问题 | 正确方向 |
|---|---|---|
| rightmost vase：glass → metallic | 生成 `Change the color of this region to metallic`，材质变成颜色 | `Change this region from glass to metallic` |
| 给左侧人物增加 orange and black AED | 生成示例中的 basket/grapes | 保留 AED 及其颜色；只消除放置位置 |
| 增加 picnic basket | 错加 grapes | 只保留 picnic basket |

旧 2,304 条的末次错误中：异常新增 color 979 条、唯一原文指代匹配失败 550 条、reference/region 数量不一致 193 条、复合操作需要列表 146 条；其余包括新增 material、OLD/NEW 混淆和已有 mask 绑定歧义。因此不能简单把 color 加入无条件白名单。

消融固定 378 条开发失败样本 + 124 条历史 accepted 对照，使用同一个 4B、BF16、temperature=0、batch=64、最多三次尝试：

| 版本 | 开发失败样本恢复 | 历史 accepted 对照通过 | 含义 |
|---|---:|---:|---|
| 原 prompt 重放 | 3/378 | 124/124 | 大部分问题能稳定重现 |
| 只删除全部示例，保留旧检查 | 209/378 | 110/124 | 证明示例影响很大；删光示例也有退化 |
| 最终实现，含更严格检查 | 208/378 | 99/124 | 检查口径已改变，不能解释为准确率下降或提升 |

最终版的开发样本结果来自八卡完整重试；对照 124 条单独生成。旧消融与最终检查不是同一判定口径，不比较其通过率作为语义准确率。最终失败集另外 1,926 条未用于早期 prompt 调试，其中 1,214 条通过；全量重试后的抽查发现了第 5 节列出的局限，所以这也不是独立人工金标准评估。

### 2.2 程序问题

- 完整指代可合法以介词结尾，例如 `crate the young boy is standing on`。旧规则只看 mask 前最后一个词，误拒绝 `on/with/behind`。
- `Fill in this region` 的 `in` 是句子介词，不应因为后缀长匹配被误认为只能属于 add。
- region 规范化曾丢失非 add 的 `in`，也可能产生 `the the text`。
- text 的 ref 如果缺少原文外层引号，会影响带撇号文字与 OLD 校验。
- 重试只发送错误原因，没有发送上一次输出；temperature=0 下常重复同一答案。
- 原检查只防“新增内容”，对遗漏约束、部件重复、旧文字保留、add 指向已有承载物等覆盖不足。

## 3. 实现：方法要求、代码位置与改动

### 3.1 两字段生成与按类型示例

方法要求模型提取完整原文指代并消除旧定位，保持操作和新内容。类型来自数据集；模型不重新分类。

[annotate_full.py:19](../samtok_edit21/annotate_full.py#L19)、[task_prompt](../samtok_edit21/annotate_full.py#L59)：共享规则 + 仅本类一个格式示例。属性示例改为木碗变光滑，不再给每条请求套 `color of`。reasoning/count 粗类别不默认展示 composite 示例。ref_phrase 的生成 schema 统一为列表，历史字符串读取仍兼容。

```json
{"ref_phrase": ["middle vase"], "noref_instruction": "Change this region from glass to ceramic"}
```

完整实际 prompt 和示例字典见[两字段实现第 3 节](SAMTokEdit_Qwen21_noref两字段转换与模型对比.md#3-当前完整-prompt)。正常请求仍只生成一次，通过检查即结束；失败才带 `correction.error` 和 `correction.previous_output` 重试。没有额外模型自审调用。

```python
model_input(source, {
    "error": previous_attempt["error"],
    "previous_output": previous_attempt["output"],
})
```

### 3.2 程序规范化与原 mask 沿用

[normalize_annotation](../samtok_edit21/annotate_full.py#L179)：原文唯一连续片段匹配、顺序与 region 数量检查仍保留。text 外层引号只有在原文确实存在精确匹配的一对时才补回；不猜测文字。去掉 region 前重复冠词，非 add 的 `in this region` 保留语法介词。

[bind_masks](../samtok_edit21/annotate_full.py#L146)：单操作仍用原 `union`；复合操作仍只绑定已有 instance IDs，歧义保持失败。本轮不生成 mask，不检查原 mask 几何准确性，也不放宽复合实例绑定。

例如实际 material 样本转换为：

```json
{
  "units": [{"edit_type": "attribute", "ref_phrase": "middle vase",
             "type_resolution": "dataset_mapping", "mask_ids": ["union"]}],
  "noref_instruction": "Change this region {mask_0} from glass to ceramic"
}
```

占位符仍要在后续真实 SAMTok codec 物化时替换成 mask tokens。此时未生成最终训练 metadata。

### 3.3 协议误判修复

[protocol.py:364](../samtok_edit21/protocol.py#L364)：ref 允许完整原文指代以介词结尾，noref 仍不允许 mask 直接接在孤立介词后。类型短语检查只给 attribute/background 的 `in this region` 语法前缀留出空间；`the object in this region` 仍不能冒充 attribute 的规范短语。

```python
forbidden = (r"the|a|an" if variant == "ref" else
             r"the|a|an|to|of|on|in|at|with|from|near|under|over|behind|beside")
grammatical_in = matched == "in this region" and "this region" in choices
if matched not in choices and not grammatical_in:
    raise ValueError("Noref mask must follow the type-specific region phrase")
```

### 3.4 校验加强及边界

[verify_semantic_review](../samtok_edit21/annotate_full.py#L322) 新增：编辑动作吞入 ref、原词遗漏、text 附加约束丢失、重复 part-of、OLD text 残留及 NEW 被填入 from-OLD 子句、add 缺少新内容等检查。发生问题反馈给模型；三次仍未通过则显式保存失败。

这些是保守的词项/结构检查，**不是语义等价证明**。例如原文 `wear blue clothing` 改为 `blue clothing` 可能被词项守恒拒绝；另一方面，错误地把旧属性搬成新属性，词都还在，也可能逃过检查。代码与报告继续写 `semantic_quality_verified=false`。

## 4. 实际验证与产物

本地环境：8×H100 80GB；Python 3.11；torch 2.8；vLLM 0.10.2；4B 每 GPU 一个 TP=1 副本，BF16，batch=64，max_tokens=512，max_model_len=8192，temperature=0，最多 3 次。无 W&B。

实际使用 `annotation_cluster --local` 完成单机八卡运行，不只是调用单条生成函数。8 个分片各 288 条，合计 2,304 条；最终源 ID 覆盖唯一且完整、各 complete SHA256 与结果一致，聚合器再次执行检查。最慢 worker 处理阶段 15.33 秒，八卡生成耗时总和 107.26 GPU·秒；这些值不包含模型复制、加载、编译、控制器轮询，不能当作完整作业墙钟耗时。

| 数据集 | 本次恢复 accepted |
|---|---:|
| RefEdit | 392 |
| CrispEdit | 363 |
| ScaleEdit | 562 |
| Derived | 105 |
| 合计 | 1,422 |

882 条余留失败的主因：原文唯一匹配 473、已有 mask 绑定歧义/无明确匹配 232、reference/region 数量 77、词项遗漏 58、粗类型无法细化 17、add reference 不含新内容 14，其余 11。不伪造 mask，不将失败记录静默改成 plain-only。

既有协议测试 + 临时回归测试 **30 passed**；覆盖完整介词结尾指代、noref 类型仍严格、材质不得变颜色、引号/撇号、OLD/NEW、保留例外条件、部件重复、add 保留新对象、规范化不丢介词。调试代码、日志和结果均在：

```text
/tmp/samtok21-noref-prompt-debug-20260929/
  all_failed.jsonl                  # 原失败 2304 条，来源原 run failed.jsonl
  failure_dev.jsonl                 # 378 条开发分层抽样
  failure_holdout.jsonl             # 其余 1926 条
  control_sources.jsonl            # 四数据集31个原生类型，共124条旧accepted对照
  eval_*/                          # 消融原始输出、失败、耗时与日志
  validated_8gpu/                  # 最终代码的完整八卡结果
    annotations.jsonl             # 1422 个候选
    failed.jsonl                  # 882 个失败，含输入和每次错误
    conversion_report.json
    SUCCESS.json
    shards/ logs/ nodes/
  final_controls/                 # 最终代码对照124条：99 accepted，25 failed
  final_review_sample.jsonl       # 分类型35条人工阅读式检查（assistant，非独立人工金标）
  old_accepted_recheck_*.json*      # 对旧96270条的保守检查
  test_regressions.py
```

最终候选 annotations SHA256：`366fac39a6c8d0f206494b9c0fb5659490af2f044f5211df19ffbd605025b960`。源码身份及协议依赖 SHA256 记录于 conversion_report，不用旧 Git HEAD 代替未提交测试时的实际源码 hash。

## 5. 还没有解决的语义问题

按 7 个输出类型各抽 5 条 accepted，固定随机种子 1729，逐条阅读输入与输出，发现以下 3 条需继续复核：

| ID | 观察 |
|---|---|
| crispedit-d69d7c33752ffde62db53dd0 | add 保留了人物，但 noref 丢失 denim jacket / light blue shirt / bright orange pants；这些词留在 ref，词项守恒未拦住 |
| scaleedit-663cab876aa709fb9c27ac74 | Replace 指令把旧建筑的 green roof / on the left 错写进新替换内容；词项仍在但角色错了 |
| scaleedit-5e5499b4d63cea7e6c96f83f | Draw O 的 noref 意图保留，但 add reference 仅指向已有格子；单字符 O 不在原 `_words` 检查范围 |

其余 32 条阅读未见明显编辑意图错误，但不是独立人工准确率；NTP 指代完整性、旧定位是否完全消去仍需更系统验收。本次不据此把 1,422 条全部标成语义已核验，更不将其自动并入训练。

对旧 accepted 的复查只调用新增保守检查，筛出 12,362 条，其中 12,202 条是词项遗漏。既有漏掉方位限定的 reference，也有合理同义表达被拒绝。**该数是需复核数，不是错误率。** 原始 `_002` 的所有数据保持原样。

## 6. 四机入口与训练数据目录

[完整四机入口](SAMTokEdit_Qwen21_四机训练运行指南.md#73-完整-arnold-入口)与[bootstrap](../scripts/train/bootstrap_arnold_annotation_4node.sh#L5)已同步：新 run 默认 `qwen21_noref4n_full_003`，resume 默认为空，远端分支 clone 到每个节点 /tmp，继续无 W&B。若此前环境变量仍指向旧 run/commit/resume，启动前必须改为新版值。

prompt、schema、实现与协议 hash 都已改变，**不能从旧 001/002 直接 resume**；那会正确触发身份不一致。当前没有代用户启动新的四机全量作业。新版四机默认输出仍在原全量数据根目录：

```text
/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/qwen21_full4_20260928/data/
  semantic_sources.jsonl                 # 98574条文本输入，不含图片副本
  sources.jsonl                          # 原图/mask路径与描述，保持不变
  assets/                                # 已准备的原资源
  semantic_runs/qwen21_noref4n_full_003/  # 新四机run启动后才产生
```

本次调试未修改正式数据清单、图片或 mask，未运行训练、未改变训练损失或超参数。新版全量生成还需要检查新 accepted/failed 和语义样本，训练就绪标记仍为 false。
