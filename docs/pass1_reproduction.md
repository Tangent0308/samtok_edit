# Pass1 运行、复现与产物

方法、实现和结果分别见 [方法](pass1_localization.md)、[代码](pass1_implementation.md)、[实验](pass1_experiments.md)。这里的命令会创建**新的实验**，不是要求覆盖已经完成的结果。

## 1. 环境与入口

Pass1 的代码在 `qwen-image-2.1-v4` 分支（2026-10-31 由 `pass1` 分支合入）。模型路径由 `SAMTOK_PASS1_MODEL` 设置，所有模型读取均使用本地权重；不隐式下载。正式实验使用的关键版本记录在 [requirements-tested.txt](../configs/pass1/requirements-tested.txt)。v4 根目录的 `requirements.txt` / `constraints-tested.txt` 锁定 `transformers==5.12.1`，Pass1 用 `5.19.0`，所以 Pass1 要单独建环境，不能把两套版本约束同时用于同一个环境。

```bash
git clone --branch qwen-image-2.1-v4 --single-branch \
  https://github.com/Tangent0308/samtok_edit.git samtok_edit_qwen-image-2.1-v4
cd samtok_edit_qwen-image-2.1-v4
# 在已具备 CUDA/PyTorch 的独立 Python 3.11 环境内：
python -m pip install -c configs/pass1/requirements-tested.txt \
  -e ./third_party/diffsynth -e '.[training,data,pass1,test]'
export SAMTOK_PASS1_MODEL=/path/to/Qwen3-VL-8B-SAMTok
export PASS1_ROOT=/path/to/new_pass1_experiment
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$PASS1_ROOT/tmp" "$PASS1_ROOT/cache"
export TMPDIR="$PASS1_ROOT/tmp" XDG_CACHE_HOME="$PASS1_ROOT/cache"
python -m samtok_edit21.pass1 --help
```

安装后的 `samtok-pass1` 等价于 `python -m samtok_edit21.pass1`。未安装包时，可将 `src:third_party:third_party/diffsynth` 加到 `PYTHONPATH`。路径示例需要替换为本机已有数据和新输出目录；模型/原始数据只读。

## 2. 数据流程

已有正式实验的冻结 selection 和源图审计可以直接用于复现物化。若重新选数，顺序为 inventory → selection → source audit → finalize → materialization → curriculum，不能跳过源图隔离。

```bash
# 两次 inventory 使用相同的四个只读来源参数。
python scripts/pass1/data/inventory.py pool --output "$PASS1_ROOT/inventory" \
  --pool-root "$POOL" --merged-source "$MERGED" --pilot-root "$PILOT" \
  --benchmark-manifest "$BENCH_MANIFEST"
python scripts/pass1/data/inventory.py merged --output "$PASS1_ROOT/inventory" \
  --pool-root "$POOL" --merged-source "$MERGED" --pilot-root "$PILOT" \
  --benchmark-manifest "$BENCH_MANIFEST"
python -m samtok_edit21.pass1 select-data \
  --inventory "$PASS1_ROOT/inventory" --config configs/pass1/formal_data.json \
  --output "$PASS1_ROOT/selection_initial"
python scripts/pass1/data/audit_sources.py \
  --selection "$PASS1_ROOT/selection_initial" --output "$PASS1_ROOT/source_audit" \
  --pilot-splits "$PILOT/data/split_sources.json" \
  --benchmark-manifest "$BENCH_MANIFEST" --benchmark-dataset "$BENCH_DATASET"
python scripts/pass1/data/finalize_selection.py \
  --selection "$PASS1_ROOT/selection_initial" --audit "$PASS1_ROOT/source_audit" \
  --inventory "$PASS1_ROOT/inventory" --config configs/pass1/formal_data.json \
  --output "$PASS1_ROOT/selection"
python -m samtok_edit21.pass1 materialize-data \
  --selection "$PASS1_ROOT/selection" --merged-source "$MERGED" \
  --image-hashes "$PASS1_ROOT/source_audit/image_hashes.parquet" \
  --output "$PASS1_ROOT/materialized" --workers 12 --quarantine-conflicts
python -m samtok_edit21.pass1 prepare-curriculum \
  --materialized "$PASS1_ROOT/materialized" --output "$PASS1_ROOT/prepared" \
  --world-size 8 --accum 8
```

`POOL` 是含 pool/pool_ext 的扩池根目录，`MERGED` 是含 records.jsonl/units.parquet 的合板，`PILOT` 指已冻结的试训根目录（用于排除其留出图），`BENCH_MANIFEST/BENCH_DATASET` 用于 benchmark 源图隔离。来源绝对路径记录在 [产物索引](../configs/pass1/artifacts.json) 和 [正式数据说明](pass1_formal_data.md)。

物化时超过 1% 请求被拒绝会在 curriculum 准备阶段停止。根据实际接收数据生成 batch plan，不能把预选数量写死为已训练数量。原选数所用输入快照和软件版本必须同时保存，否则重新盘点未必产生逐字相同的清单。

## 3. 正式训练与恢复

```bash
torchrun --standalone --nproc_per_node=8 -m samtok_edit21.pass1.training.formal \
  --data "$PASS1_ROOT/prepared/train.jsonl" \
  --batch-plan "$PASS1_ROOT/prepared/batch_plan.json" \
  --output "$PASS1_ROOT/train" --stop-after-step 500 \
  --rank 64 --accum 8 --lr 2e-5 --max-pixels 2097152 --workers 0 --save-every 100
# 评估 dev 后，按同一计划继续至下一个节点：
torchrun --standalone --nproc_per_node=8 -m samtok_edit21.pass1.training.formal \
  --data "$PASS1_ROOT/prepared/train.jsonl" \
  --batch-plan "$PASS1_ROOT/prepared/batch_plan.json" \
  --output "$PASS1_ROOT/train" --resume "$PASS1_ROOT/train/step_0500" \
  --stop-after-step 1000 --workers 0
```

原始实验依次在 500/1000/1500/2000/2500/2502 评估 dev，按 unknown strict IoU 选 2000；相同分数取较早 checkpoint，选择后再评估 test。新数据的节点和总步数以 prepared manifest 为准。完整 run 参数的历史记录是 [formal_training_record.json](../configs/pass1/formal_training_record.json)，它是记录，不是会自动被 Trainer 消费的配置文件。

训练中只跑自己的进程；恢复必须使用带 COMPLETE 标志的 checkpoint、相同 batch plan、相同 world size。`workers=0` 避免原节点 NAS 临时路径过长触发 DataLoader 的 Unix socket 问题。

## 4. 联合推理、评分与 Python API

```bash
python -m samtok_edit21.pass1 shards infer \
  --data "$PUBLIC_REQUESTS" --adapter "$ADAPTER" \
  --output-dir "$PASS1_ROOT/eval/inference" --gpus 0,1,2,3
python -m samtok_edit21.pass1 shards score \
  --data "$SCORING_TRUTH" --predictions "$PASS1_ROOT/eval/inference/combined.jsonl" \
  --output-dir "$PASS1_ROOT/eval/score" --gpus 0,1,2,3
python -m samtok_edit21.pass1 summarize \
  --scores "$PASS1_ROOT/eval/score/combined.jsonl" --output "$PASS1_ROOT/eval/metrics.json"
```

省略 adapter 就是基座。每条请求一次 greedy generate，默认输出上限 512；推理只读取 image 和 request，benchmark 的 GT 使用单独评分文件。

```python
from PIL import Image
from samtok_edit21.pass1.grounder import Pass1Grounder

grounder = Pass1Grounder(adapter="/path/step_2000", base_model="/path/Qwen3-VL-8B-SAMTok")
result = grounder.ground(Image.open("/path/source.jpg"), public_pass1_request)
if not result["parsed"]["complete"]:
    raise ValueError(result["parsed"]["errors"])
tokens_by_id = result["spans"]
```

已有 Pass2 pipeline 可用 `Pass1Grounder.from_pipeline(pipe, adapter, base_model=...)` 借用 TE。Pass1 调用结束后关闭新 LoRA，再用既有指令/token renderer 交给 Pass2。此分支没有自动替换原 benchmark 编辑入口，也没有用当前 LoRA 重新评测 DiT 编辑质量。

## 5. 固定 IF-Bench dev20 实验

脚本位于 `scripts/pass1/ifbench_dev20/`；它们重现固定 release、20 张图、72 个目标和 step2000 对照，不能直接称作任意 benchmark 的通用 runner。需要安装/加入 `PYTHONPATH` 的 benchmark checkout 提交为 `b6ea7e4e8705927ecb7df987c0ebb94d57b57e17`。

```bash
export PASS1_FORMAL_ROOT=/path/to/completed_formal_experiment
export PASS1_BENCH_ROOT=/path/to/new_dev20_experiment
export PASS1_BENCH_REPO=/path/to/samtok_edit_benchmark
export PASS1_BENCH_DATASET=/path/to/samtok_edit_benchmark_v2
export PASS1_DEV20_BUNDLE=/path/to/frozen_dev20_selection_bundle
export PASS1_PREVIEW_ROOT=/opt/tiger/tanyue/visualization_tmp/pass1_dev20_reproduction
export PASS1_ADAPTER="$PASS1_FORMAL_ROOT/runs/formal/step_2000"
export PYTHONPATH="$PASS1_BENCH_REPO${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$PASS1_BENCH_ROOT/logs" "$PASS1_BENCH_ROOT/data" "$PASS1_BENCH_ROOT/provenance"
python scripts/pass1/ifbench_dev20/prepare_cpu.py
python scripts/pass1/ifbench_dev20/run_experiment.py
python scripts/pass1/ifbench_dev20/make_before_after_gallery.py
python scripts/pass1/ifbench_dev20/analyze_root_causes.py
```

`run_experiment.py` 沿用原实验的 H100 空闲显存门槛并最多取四张卡；它不会终止其他任务。先冻结公开输入/隔离检查，再编码明确提供的 mask；生成完毕才解码评分。`report.py` 和 gallery 使用固定的展示规则；分析脚本的 step 2000 曝光数断言对应本次正式训练，不适用于其他 checkpoint。

原始运行的 controller/monitor、日志、全部数据与权重保留在 NAS；它们是历史运行产物，不作为包导入或提交到仓库。该分支提供数据/训练/推理/评分入口和报告程序，原始 controller 的命令记录用于核对当时的执行顺序。

## 6. 检查与产物位置

完整 CPU suite 包含旧 v4 的 W&B 离线记录测试，需要可用的 W&B SDK。本次在独立测试依赖目录使用官方 `wandb==0.19.11`，并显式启用其 service；本机原有 byted SDK 与 protobuf 的导入冲突没有通过修改训练代码绕过。Pass1 正式训练本身不依赖 W&B。

```bash
# 在独立测试环境安装；不要覆盖集群中其他任务的 byted-wandb。
python -m pip install wandb==0.19.11
WANDB_DISABLE_SERVICE=false DIFFSYNTH_ATTENTION_IMPLEMENTATION=torch python -m pytest -q
python scripts/docs/sync_code_blocks.py --check
python scripts/docs/check_paths.py
```

所有当前实验路径、模型位置、checkpoint SHA256、benchmark manifest 和预检记录见 [artifacts.json](../configs/pass1/artifacts.json)。仓库只保存实现、配置、汇总指标和精选可视化；原始样本、日志、检查点留在原实验目录。

面向用户的展示副本位于 `/opt/tiger/tanyue/visualization_tmp/pass1_branch_cleanup_20261031/figures/`（在 pass1 的开发机上，v4 当前开发机本机没有）。`docs/assets/pass1/manifest.json` 记录每张仓库图片的原始文件、SHA256 和字节数；复制不改变图片内容。
