# SAMTokEdit Qwen-Image-2.1 训练数据盘点

本文盘点当前正式训练使用的四个源数据集、过滤规则、路径、字段映射、noref 转换、最终文件结构和训练读取方式。源数据 mask 视为数据集提供的准确标注；项目只读取、物化和编码，不重新计算或核对 mask 几何。

## 1. 四个源数据集

| 数据集 | 大致内容 | 源路径 | 发布行 | 最终保留 |
|---|---|---|---:|---:|
| RefEdit | source/target 图像、编辑指令和预筛选 mask 的对象/属性编辑 | `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/RefEdit-mask-prefiltered-qwen38-self-contained` | 7,804 | 7,804 |
| CrispEdit | add/remove/replace/color/motion 编辑，含最终 mask QC | `/mnt/bn/strategy-mllm-train/user/tanyue/CrispEdit-labeling/final_dataset_39k` | 38,971 | 37,728 |
| ScaleEdit | action、text、material 和 reasoning 来源编辑任务 | `/mnt/bn/strategy-mllm-train/user/tanyue/scaleedit_25k` | 25,664 | 25,085 |
| SAMTok Derived | combined 最终通过编辑对，含 task_type 和 COCO RLE | `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/combined` | 27,957 | 27,957 |
| 合计 | 四个来源的最终质量通过编辑对 |  | 100,396 | **98,574** |

最终过滤字段：

```python
RefEdit: prefilter_verdict == PASS and grounding_status == OK and qc_flag == OK
CrispEdit: quality__prefilter_verdict == PASS and scene__scene_pass is True and mask__qc_flag == OK
ScaleEdit: quality__verdict == PASS and quality__keep is True and
           scene__verdict == PASS and scene__keep is True and mask__qc_flag == OK
Derived: planning_status == accepted and audit_status == pass
```

CrispEdit 排除 1,243 条，ScaleEdit 排除 579 条；不能只看 quality/scene 而漏掉最终 mask QC。

## 2. 本地准备目录

正式数据根目录：

```text
/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/qwen21_full4_20260928/data
```

组织方式：

```text
sources.jsonl                 # 98,574 source rows、图片路径、尺寸、mask 来源、source_id
assets/refedit/               # RefEdit 原始图像字节 source.img/target.img
assets/crispedit/             # CrispEdit 原始图像字节 source.img/target.img
assets/scaleedit/             # ScaleEdit 原始图像字节 source.img/target.img
semantic_sources.jsonl        # noref 纯文本转换输入
semantic_runs/..._003/        # 9B + rule fallback 的逐条结果
train_full_9b_rules_003/      # 正式 metadata、编码、区域 cache
```

训练使用 metadata + 图片路径/缓存；不是把所有图片嵌入 JSONL。前三个数据集图片原始字节被物化到 `assets/<dataset>/<ID 后两位>/<source ID>/source.img|target.img`（由 Pillow 检测实际编码），Derived 沿用 combined 原始图片路径和 RLE 引用。RLE/PNG 只在编码阶段读取，SAMTok code 和 coverage 进入训练 metadata/cache。

## 3. noref 结果和失败策略

生产 run：`data/semantic_runs/qwen21_noref9b_rules_4n_full_003/`。

```text
input 98,574
accepted 97,361 = LLM 96,593 + rule fallback 768
failed 1,213
```

模型只输出 `ref_phrase`、`noref_instruction`；`edit_type` 来自原数据，mask IDs 来自 source manifest。1,213 条失败样本只进入 plain `edit` 行，不生成伪造的 ref/noref/NTP 行。

## 4. 最终训练文件和统计

版本目录：

```text
/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/qwen21_full4_20260928/data/train_full_9b_rules_003/
```

核心文件：`inputs.json`、`stage1.jsonl`、`stage2.jsonl`、`provenance.jsonl`、`encoded/worker-00..07/`、`regions/manifest.json`、`regions/coverage/*.pt`、`metadata_report.json`。

| 文件/类型 | 行数 |
|---|---:|
| source rows | 98,574 |
| Stage 1 `edit_ntp` | 97,361 |
| Stage 1 `edit_umt/ref` | 97,361 |
| Stage 1 `edit_umt/noref` | 97,361 |
| Stage 1 plain `edit` | 98,574 |
| **Stage 1 合计** | **390,657** |
| Stage 2 `edit_umt/ref` | 97,361 |
| Stage 2 `edit_umt/noref` | 97,361 |
| Stage 2 plain `edit` | 98,574 |
| **Stage 2 合计** | **293,296** |

关键 hash：

```text
stage1_sha256 = 70f00fd267ec26d7e26437207aec4d832c90c2cae892f03a2e6e569a0ddadfd5
stage2_sha256 = b26fef1e76fa0c1360248b4a09f486feb430bdb4052b1a9b4e9249d99ce6e24a
provenance_sha256 = 9417f2a7f660251f613b260e78939d980015ad5ba61ee5ff35c3ae7801767997
region_manifest_sha256 = 2a67273b6e2fe06822a4a39bf9ef4cfcf64cf12c07bc34d568330369e3f43278
```

## 5. 最终 row 字段和例子

正式 JSONL 只包含训练协议字段；来源审计字段在 `provenance.jsonl`：

以下四行来自同一实际 source，保留完整路径：

```json
{"edit_image": "/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/qwen21_full4_20260928/data/assets/refedit/46/refedit-c05d9726f751cd4e25b5fe46/source.img", "edit_type": "replace", "sample_type": "edit_ntp", "prompt": "Change the leftmost bird's feathers to soft down feathers", "mt_cot": "```json\n[{\"mask_2d\": \"<|mt_start|><|mt_0017|><|mt_0322|><|mt_end|>\", \"label\": \"leftmost bird's feathers\"}]\n```"}
{"edit_image": "/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/qwen21_full4_20260928/data/assets/refedit/46/refedit-c05d9726f751cd4e25b5fe46/source.img", "edit_type": "replace", "image": "/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/qwen21_full4_20260928/data/assets/refedit/46/refedit-c05d9726f751cd4e25b5fe46/target.img", "sample_type": "edit", "prompt": "Change the leftmost bird's feathers to soft down feathers"}
{"edit_image": "/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/qwen21_full4_20260928/data/assets/refedit/46/refedit-c05d9726f751cd4e25b5fe46/source.img", "edit_type": "replace", "image": "/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/qwen21_full4_20260928/data/assets/refedit/46/refedit-c05d9726f751cd4e25b5fe46/target.img", "sample_type": "edit_umt", "prompt": "Change the leftmost bird's feathers <|mt_start|><|mt_0017|><|mt_0322|><|mt_end|> to soft down feathers", "instr_variant": "ref"}
{"edit_image": "/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/qwen21_full4_20260928/data/assets/refedit/46/refedit-c05d9726f751cd4e25b5fe46/source.img", "edit_type": "replace", "image": "/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/qwen21_full4_20260928/data/assets/refedit/46/refedit-c05d9726f751cd4e25b5fe46/target.img", "sample_type": "edit_umt", "prompt": "Change the object in this region <|mt_start|><|mt_0017|><|mt_0322|><|mt_end|> to soft down feathers", "instr_variant": "noref"}
```

NTP 行使用原始编辑指令并增加 `mt_cot`；plain 行没有 `instr_variant` 或 mask span；ref/noref 行使用同一 source mask code，只有文本指代不同。`anchor`、`mask_ids` 是准备阶段/来源 sidecar 字段：anchor 绑定 add 的位置语义，mask_ids 将 unit 绑定到已有实例/aggregate mask；它们不是当前模型 JSONL 的必需字段。

## 6. 类型映射

协议支持 `add/remove/replace/attribute/action/text/background/global/composite` 九类；本次最终数据只有前六类和 composite，background/global 为 0。CrispEdit `add/remove/replace/color/motion change` 映射为 add/remove/replace/attribute/action；ScaleEdit action/text 标签按最终指令映射；Derived 使用显式 `task_type` adapter；RefEdit 的 material/color/object addition/removal 映射为 attribute/attribute/add/remove。

当前实现保留已映射的原子源标签（例如 RefEdit object_replacement→replace），不让 LLM 重判；reasoning/count 等粗类别由 `resolve_type` 结合指令/子句细化，复合操作记录 composite。独立操作需能唯一绑定已有 mask，绑定失败不生成合格的 UMT/NTP。本表是代码真实落地结果，不把历史语义审计建议误写成另一套已实现分类。完整原生类别和真实指令例子见[来源审计历史](archive/SAMTokEdit_Qwen21_全量数据盘点与转换审计.md#3-类型盘点暂定映射与必须复核的例外)。


### 最终来源与类型细分

| 来源 | plain | NTP/ref/noref 各自 | plain-only | Stage 1 总行 | Stage 2 总行 |
|---|---:|---:|---:|---:|---:|
| refedit | 7,804 | 7,792 | 12 | 31,180 | 23,388 |
| crispedit | 37,728 | 37,236 | 492 | 149,436 | 112,200 |
| scaleedit | 25,085 | 24,410 | 675 | 98,315 | 73,905 |
| derived | 27,957 | 27,923 | 34 | 111,726 | 83,803 |

| edit_type | plain | NTP/ref/noref 各自 |
|---|---:|---:|
| add | 18,928 | 18,798 |
| remove | 28,059 | 27,698 |
| replace | 18,222 | 18,198 |
| attribute | 27,098 | 26,975 |
| action | 2,995 | 2,666 |
| text | 2,066 | 1,984 |
| composite | 1,206 | 1,042 |

## 7. 区域监督缓存

`regions/manifest.json` 覆盖 Stage 1 全部 390,657 行：

```text
task          195,935
eligible      194,618
empty_region      104
```

coverage 文件按 source/target/span 内容去重，ref/noref 行共享文件。`RegionStore` 首次读取时校验 coverage 和 source/target 图像 SHA256；`coverage_grid` 对齐到 latent grid，C 进行区域/背景 FP32 加权，A 使用 mask span positions。

## 8. 训练读取和配比

实际实现位于 [`samtok_edit21/schema/data.py`](../samtok_edit21/schema/data.py) 的 `make_schedule`，根入口 `samtok_edit21.data` 仅为兼容 alias：

```text
Stage 1 global batch 256: edit_ntp=96, ref=64, noref=64, plain=32
Stage 2 global batch 128: ref=32, noref=64, plain=32
```

各分支内按 edit_type 的权重池采样：add/remove/replace 各 14，attribute 20，action/text 各 10，background/global/composite 各 6；不存在的类型不参与并重新归一。plain 对 background/global 各有 15% 概率上限处理（本次均无对应行），见 `capped_plain_weights`。没有额外按源数据集均衡采样，来源实际曝光由各类型池中的样本构成决定。类型行已在训练前生成，DataLoader 只验证、采样和加载，不在线运行 noref 模型。

3081 optimizer updates 是近似一轮长度，严格保证每个 update 的分支比例，但加权随机采样不承诺每条 row 在单轮内出现。每阶段 `schedule.json` 记录 `source_rows`、`draws`、`unique_rows`、`unseen_rows`；训练前可用 `--plan-only` 检查暴露率。NTP 只在 Stage 1，A 只在 Stage 2，plain/NTP 的区域监督返回 `reason=task`。

## 9. 验证和重建命令

```bash
set -Eeuo pipefail
export DATA_ROOT=/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/qwen21_full4_20260928/data
export TRAIN_DATA=$DATA_ROOT/train_full_rebuild_NEW_VERSION
[[ ! -e "$TRAIN_DATA" ]] || { echo "Use a fresh output directory" >&2; exit 2; }
export CODEC=/mnt/bn/strategy-mllm-train/user/tanyue/models/SAMTok/Qwen3-VL-8B-SAMTok
PYTHONPATH=.:DiffSynth-Studio python -m samtok_edit21.full_training_data split \
  --source-root "$DATA_ROOT" \
  --semantic-run "$DATA_ROOT/semantic_runs/qwen21_noref9b_rules_4n_full_003" \
  --output "$TRAIN_DATA" --workers 8
pids=()
for rank in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$rank PYTHONPATH=.:DiffSynth-Studio python -m samtok_edit21.full_training_data encode-worker \
    --output "$TRAIN_DATA" --rank $rank --codec-root "$CODEC" --device cuda:0 --batch-size 16 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid" || exit 1; done
PYTHONPATH=.:DiffSynth-Studio python -m samtok_edit21.full_training_data merge --output "$TRAIN_DATA"
```

当前 `metadata_report.json` 为 `training_ready=true`、`region_cache_ready=true`。重建应使用新版本目录，不覆盖正式数据。
