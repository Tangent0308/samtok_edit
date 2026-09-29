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
"$UV" pip sync --python "$SAMTOK_ENV/bin/python" --index-url "$INDEX" "$REPO/requirements-annotation-lock.txt"
"$UV" pip check --python "$SAMTOK_ENV/bin/python"
"$SAMTOK_ENV/bin/python" - <<'PY'
import sys, torch, transformers, vllm
assert sys.version_info[:2] == (3, 11)
assert torch.__version__.split('+')[0] == '2.8.0'
assert transformers.__version__ == '4.55.2'
assert vllm.__version__ == '0.10.2'
print('Annotation environment validated: Python 3.11 / torch 2.8 / vLLM 0.10.2')
PY
PYTHONPATH="$REPO" "$SAMTOK_ENV/bin/python" -m samtok_edit21.cuda_readiness \
  --output "${SAMTOK_CUDA_DIAGNOSTICS:-${SAMTOK_ENV}-cuda}" --expected 8 \
  --timeout "${SAMTOK_CUDA_READY_TIMEOUT:-600}" --interval 15
