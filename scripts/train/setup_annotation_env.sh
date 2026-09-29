#!/usr/bin/env bash
# Dedicated vLLM environment; never install this lock over the training env.
set -Eeuo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${SAMTOK_ENV:?Set a node-local annotation environment}"
PYTHON_BIN="${SAMTOK_PYTHON:-/usr/bin/python3.11}"
INDEX="${SAMTOK_INDEX:-https://bytedpypi.byted.org/simple/}"
UV="${UV_BIN:-$(command -v uv || true)}"
if [[ -z "$UV" ]]; then
  "$PYTHON_BIN" -m pip install --index-url "$INDEX" --target "${SAMTOK_ENV}-uv" uv==0.12.15
  UV="${SAMTOK_ENV}-uv/bin/uv"
fi
if [[ ! -f "$SAMTOK_ENV/pyvenv.cfg" ]]; then
  "$UV" venv --python "$PYTHON_BIN" --no-python-downloads "$SAMTOK_ENV"
fi
MODEL="${SAMTOK_ANNOTATION_MODEL:-/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen3.5-9B}"
MODEL_TYPE="$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["model_type"])' "$MODEL/config.json")"
[[ "$MODEL_TYPE" == qwen3_5 ]] || { echo 'Use the selected Qwen3.5-9B annotation model' >&2; exit 2; }
LOCK="$REPO/requirements-annotation-qwen35-lock.txt"
EXPECT_TORCH=2.10.0 EXPECT_TRANSFORMERS=4.57.6 EXPECT_VLLM=0.17.1
"$UV" pip sync --python "$SAMTOK_ENV/bin/python" --index-url "$INDEX" "$LOCK"
"$UV" pip check --python "$SAMTOK_ENV/bin/python"
EXPECT_TORCH="$EXPECT_TORCH" EXPECT_TRANSFORMERS="$EXPECT_TRANSFORMERS" EXPECT_VLLM="$EXPECT_VLLM" "$SAMTOK_ENV/bin/python" - <<'PY'
import os
import sys, torch, transformers, vllm
assert sys.version_info[:2] == (3, 11)
assert torch.__version__.split('+')[0] == os.environ['EXPECT_TORCH']
assert transformers.__version__ == os.environ['EXPECT_TRANSFORMERS']
assert vllm.__version__ == os.environ['EXPECT_VLLM']
print(f'Annotation environment validated: Python 3.11 / torch {torch.__version__} / vLLM {vllm.__version__}')
PY
export PATH="$SAMTOK_ENV/bin:$PATH"
PYTHONPATH="$REPO" "$SAMTOK_ENV/bin/python" -m samtok_edit21.cuda_readiness \
  --output "${SAMTOK_CUDA_DIAGNOSTICS:-${SAMTOK_ENV}-cuda}" --expected 8 \
  --timeout "${SAMTOK_CUDA_READY_TIMEOUT:-600}" --interval 15
