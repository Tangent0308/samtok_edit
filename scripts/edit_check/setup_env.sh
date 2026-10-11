#!/usr/bin/env bash
# Node-local vLLM environment of the MLLM edit check; never install this lock over another env.
set -Eeuo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${SAMTOK_ENV:?Set a node-local edit-check environment}"
PYTHON_BIN="${SAMTOK_PYTHON:-/usr/bin/python3.11}"
INDEX="${SAMTOK_INDEX:-https://bytedpypi.byted.org/simple/}"
TORCH_INDEX="${SAMTOK_TORCH_INDEX:-https://download.pytorch.org/whl/cu129}"
export UV_LINK_MODE=copy
UV="${UV_BIN:-$(command -v uv || true)}"
if [[ -z "$UV" ]]; then
  "$PYTHON_BIN" -m pip install --index-url "$INDEX" --target "${SAMTOK_ENV}-uv" uv==0.12.15
  UV="${SAMTOK_ENV}-uv/bin/uv"
fi
if [[ ! -f "$SAMTOK_ENV/pyvenv.cfg" ]]; then
  "$UV" venv --python "$PYTHON_BIN" --no-python-downloads "$SAMTOK_ENV"
fi
# torch/torchvision/torchaudio/torchcodec are CUDA 12.9 builds; vLLM is the v0.28.0 cu129 release wheel.
"$UV" pip sync --python "$SAMTOK_ENV/bin/python" --index-url "$INDEX" --extra-index-url "$TORCH_INDEX" \
  --index-strategy unsafe-best-match "$REPO/requirements-mllm-check-lock.txt"
"$UV" pip install --python "$SAMTOK_ENV/bin/python" --index-url "$INDEX" --no-deps -e "$REPO"
"$UV" pip check --python "$SAMTOK_ENV/bin/python"
"$SAMTOK_ENV/bin/python" - <<'PY'
import sys, torch, transformers, vllm, aiohttp, pycocotools
assert sys.version_info[:2] == (3, 11)
assert torch.__version__ == '2.13.0+cu129', torch.__version__
assert transformers.__version__ == '5.15.1', transformers.__version__
assert vllm.__version__ == '0.28.0', vllm.__version__
print(f'Edit-check environment validated: Python 3.11 / torch {torch.__version__} / vLLM {vllm.__version__}')
PY
export PATH="$SAMTOK_ENV/bin:$PATH"
PYTHONPATH="$REPO/src" "$SAMTOK_ENV/bin/python" -m samtok_edit21.distributed.cuda_readiness \
  --output "${SAMTOK_CUDA_DIAGNOSTICS:-${SAMTOK_ENV}-cuda}" --expected 8 \
  --timeout "${SAMTOK_CUDA_READY_TIMEOUT:-600}" --interval 15
