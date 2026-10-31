# Pass1 实现与代码导航

当前方法与输入协议见 [方法设计](pass1_localization.md)，实验数字见 [实验结果](pass1_experiments.md)，可执行步骤见 [复现说明](pass1_reproduction.md)。本分支基于 v4 `6aa4e10`；GPU 实验使用的原始代码快照仍保留在实验目录，当前整理不重新训练或改写历史结果。

## 1. 模块边界

```text
src/samtok_edit21/pass1/
  protocol.py              # 公开请求、坐标、输出解析、known-mask 最终绑定
  model.py                 # processor、assistant token 对齐、LM LoRA、分组 NTP
  grounder.py              # 一次 generate；独立加载或借用 Pass2 的 TE
  io.py                    # 无模型依赖的文件哈希
  cli.py / __main__.py     # samtok-pass1 / python -m samtok_edit21.pass1
  data/
    pilot.py               # 历史 8k 试训数据
    selection.py           # 扩池选数、源图划分、模式配额、反事实配对
    materialization.py     # S/canonical 校验、point/box、冲突隔离
    curriculum.py          # context 校验、精简训练行、固定 batch 序列
  training/
    checkpoint.py          # FP32 adapter 保存、共享 TE 的 adapter-off probe
    pilot.py               # 历史 128-update 试训
    formal.py              # DDP、完整状态恢复、末尾 batch、冻结审计
  evaluation/
    inference.py           # 图像 + 公开请求 → 原始回答；每请求一次生成
    scoring.py             # 离线 codec 解码、IoU/复制/格式/点击诊断
    metrics.py             # 分组指标、按源图 bootstrap
    shards.py              # 独立 GPU worker、退出状态、完整性合并
    ifbench.py             # IF-Bench 公开输入适配；仅 mask 可读 token cache
    audit.py               # checkpoint 重载和共享 TE 一致性
    pilot_report.py        # 历史试训报告
```

`configs/pass1/` 保存配置与产物索引；`scripts/pass1/data/` 保存数据盘点/源图审计/最终隔离工具；`scripts/pass1/formal/` 保存正式实验的报告和恢复核验工具；`scripts/pass1/ifbench_dev20/` 保存固定 dev20 实验的准备、调度、分析和作图步骤。实验特定脚本没有混入通用模型 API。

原来的 `pass1.formal_train`、`pass1.infer` 等模块仅保留显式 re-export 与 CLI 兼容入口；内部实现和新文档使用上面的规范路径。已有实验命令仍可解析，不需要更改原始日志或快照。

## 2. 公开输入与一次生成

`protocol.validate_request` 拒绝私有或未知字段；`from_native_job` 只读取已公开的 locator。`evaluation.ifbench.adapt` 首先调用 benchmark 官方 `verify`，保持 ID、指令、has_ref 和模态不变，仅转换坐标。没有将隐藏 GT 的类别、scope、mask 路径传入模型。

实际联合生成的关键代码如下。`encode` 经 `prompt()` 先验证公开协议；整个 targets 数组进入同一个上下文。

<!-- code: src/samtok_edit21/pass1/grounder.py::Pass1Grounder.ground -->
```python
@torch.inference_mode()
def ground(self, image, request, *, max_new_tokens=512):
    inputs = {
        k: v.to(self.device)
        for k, v in encode(self.processor, image, request).items()
    }
    with adapter_enabled(self.model, True):
        self.model.model.rope_deltas = None
        ids = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True
        )
    generated = ids[0, inputs["input_ids"].shape[1] :]
    raw = self.processor.tokenizer.decode(generated, skip_special_tokens=False)
    parsed = parse_answer(raw, request)
    return {
        "raw": raw,
        "parsed": parsed,
        "request": request,
        "spans": finalize_spans(parsed, request) if parsed["complete"] else None,
    }
```
<!-- /code -->

这里的 `spans` 是部署结果：`finalize_spans` 对用户已给的 mask 使用输入 token 做确定性绑定。**评估使用原始 parsed 回答，在这一绑定之前计算 mask-copy/IoU**，不会通过该步骤提高复制成绩。

## 3. NTP 监督、LoRA 和 Pass2 隔离

`model.encode` 使用原生 Qwen3-VL processor/chat template 展开图像 token，再定位完整 assistant suffix。预测位置为标签位置的前一位；只对 assistant 输出及 EOS 监督。每个目标仅两个 code token，开始/结束标记归格式组。

<!-- code: src/samtok_edit21/pass1/model.py::grouped_loss -->
```python
def grouped_loss(logits, labels, groups, mode="grouped"):
    ce = F.cross_entropy(logits.float(), labels, reduction="none")
    if mode == "ce":
        return ce.mean(), {"ce": float(ce.detach().mean())}
    weight = (1.0, 0.25, 0.25)
    pieces, stats, total = [], {}, 0.0
    for g, w in enumerate(weight):
        active = groups == g
        if active.any():
            v = ce[active].mean()
            pieces.append(w * v)
            stats[("prediction", "copy", "format")[g]] = float(v.detach())
            total += w
    return sum(pieces) / total, stats
```
<!-- /code -->

这是三组平均交叉熵的加权平均：未知 code=1、mask-copy code=0.25、格式=0.25；缺少某组时只用实际存在的权重归一化。没有 IoU/Dice loss，也没有反事实配对排序项。IoU 用于标签质量筛选、离线评价和 checkpoint 选择。

`model.TARGET_RE` 只匹配语言模型 36 层的 q/k/v/o 与 gate/up/down，共 252 个投影。r64、alpha128、dropout 0.05；基座 BF16、adapter FP32。视觉编码器、merger、embedding、lm_head、codec 和 DiT 均不更新。

`attach_lora` 先创建 FP32 LoRA 层再装入保存的张量，避免加载时途经 BF16 造成精度损失。`adapter_enabled` 在正常返回和异常路径都恢复 adapter 状态及 requires_grad；`Pass1Grounder.from_pipeline` 借用共享 TE，Pass1 之外保持新 LoRA 关闭。该 LoRA 不 merge 到基座。

## 4. 正式数据和训练

| 关键函数 | 负责的约束 |
|---|---|
| `data.selection.eligible_modes` | dense 类别不冒充 ref；host/site 与父子粒度的保守模态限制 |
| `data.selection.canonicalize_teachers` | 同图同 S 的不同缓存 token 选择较高重建质量者，保留原指令与来源 |
| `data.selection.add_counterfactual_pairs` | 同图同类目标间只切换一个 no-ref point/box |
| `data.materialization.sample_point` | 坐标量化后仍在 S；训练时还须在教师解码内；dev/test 不按教师筛点 |
| `data.materialization.conflicting_requests` | 隔离相同公开输入对应不同答案的请求，同时移除配对伙伴 |
| `data.curriculum.context_length` | 解析图像 token 展开长度，并抽样与真实 processor 对照 |
| `data.curriculum.pack_batches` | 所有真实行恰好一次；配对同 update；末尾用 −1 padding |
| `training.formal.rank_samples` | DDP 下零损失 padding，按真实请求数归一化梯度 |
| `training.formal.main` | 全局 LR、不重置的 curriculum、恢复 optimizer/RNG、原子 checkpoint |

配对的实际 batch 打包逻辑：

<!-- code: src/samtok_edit21/pass1/data/curriculum.py::pack_batches -->
```python
def pack_batches(rows, indices, capacity, seed, stage):
    """Keep both counterfactual members within a global optimizer update."""
    rng = random.Random(seed)
    pairs = collections.defaultdict(list)
    blocks = []
    for index in indices:
        row = rows[index]
        if row.get("pair_id"):
            pairs[row["pair_id"]].append(index)
        else:
            blocks.append([index])
    if any(len(v) != 2 for v in pairs.values()):
        raise ValueError("Incomplete counterfactual pair")
    blocks += list(pairs.values())
    rng.shuffle(blocks)
    batches = []
    current = []
    for block in blocks:
        if len(current) + len(block) > capacity:
            batches.append(
                {"stage": stage, "indices": current + [-1] * (capacity - len(current))}
            )
            current = []
        current.extend(block)
    if current:
        batches.append(
            {"stage": stage, "indices": current + [-1] * (capacity - len(current))}
        )
    flat = [i for b in batches for i in b["indices"] if i >= 0]
    if sorted(flat) != sorted(indices):
        raise AssertionError("Batch packing dropped or duplicated requests")
    return batches
```
<!-- /code -->

`training.formal` 的前向调用 `grouped_loss(out.logits[0], labels, groups)`，然后乘以该真实样本的梯度归一化系数、除以 accumulation 反传。checkpoint 含 adapter、optimizer、每 rank RNG、global step、batch-plan SHA256 和 `COMPLETE.json`；恢复时检查数据与计划身份。保存完成之前使用 `.partial` 目录。

两步恢复、FP32 加载和关闭 LoRA 的 TE 一致性通过。GPU 反向数值存在非确定性，连续/恢复运行的严格逐位权重重放没有通过；文档不承诺 bitwise training replay。

## 5. 评价边界与复核

`evaluation.scoring.decode_stable` 使用每个 span 独立的 batch=1 原生解码，同图只复用冻结 image features，避免候选数改变 BF16 batch 路径。strict 模式对整条请求做 schema gating；compatible 只允许移除固定 empty-think 前缀或外层 Markdown fence，不修 token、补 ID 或用 GT 重新匹配。

IF-Bench 推理只接收 `inference_public.jsonl`，评分读取单独的 `scoring_truth.jsonl`。`canonical_cached_equal` 是逐像素比较，包含空 mask 情况；`canonical_iou` 是原 GT→编码→解码的参考，不能当作严格性能上限。

CPU 检查位于 `tests/test_pass1*.py`，覆盖私有字段隔离、坐标边界、顺序/重复 ID、分组损失梯度、异常恢复 adapter 状态、FP32 保存加载、反事实与末尾 batch/恢复。文档关键代码块由 `scripts/docs/sync_code_blocks.py` 同步，并由测试检查漂移。

本次代码整理验收（2026-10-31）：完整 CPU suite 105 项通过；15 个命令 help、15 段文档代码同步和387条路径检查通过。11个关键函数与原实验快照AST一致；新目录中重新执行根因统计，JSON与原结果完全一致。没有重跑GPU训练。测试环境、检查范围和复核记录见 [validation.json](assets/pass1/validation.json)。
