#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${SAMTOK_ENV:?Set SAMTOK_ENV to a node-local virtualenv path}"
PYTHON_BIN="${SAMTOK_PYTHON:-/usr/bin/python3.11}"
INDEX="${SAMTOK_INDEX:-https://bytedpypi.byted.org/simple/}"
UV="${UV_BIN:-$(command -v uv || true)}"
if [[ -z "$UV" ]]; then
  BOOTSTRAP="${SAMTOK_ENV}-uv"
  "$PYTHON_BIN" -m pip install --index-url "$INDEX" --target "$BOOTSTRAP" uv==0.12.15
  UV="$BOOTSTRAP/bin/uv"
fi
cd "$REPO"
if [[ ! -f "$SAMTOK_ENV/pyvenv.cfg" ]]; then
  "$UV" venv --python "$PYTHON_BIN" --no-python-downloads "$SAMTOK_ENV"
fi
"$UV" pip install --python "$SAMTOK_ENV/bin/python" --index-url "$INDEX" \
  --torch-backend cu128 -r requirements.txt -r requirements-cluster.txt
"$UV" pip check --python "$SAMTOK_ENV/bin/python"
PYTHONPATH="$REPO/src:$REPO/third_party/diffsynth" "$SAMTOK_ENV/bin/python" - <<'PY'
import sys, torch, wandb, transformers, accelerate, peft
assert sys.version_info[:2] == (3, 11)
assert torch.__version__.split('+')[0] == '2.8.0'
assert transformers.__version__ == '5.12.1'
assert accelerate.__version__ == '1.14.0'
assert peft.__version__ == '0.20.0'
assert wandb.__version__ == '0.13.98'
from samtok_edit21.models.binding import require_flex_attention
require_flex_attention()
# The v2 cache decodes region maps with the released SAMTok codec.
import samtok.models  # noqa: F401
from samtok_edit21.models.codec import SamtokCodec  # noqa: F401
print('Cluster packages validated: Python 3.11 / torch 2.8 / W&B / FlexAttention / SAMTok codec')
PY
PYTHONPATH="$REPO/src:$REPO/third_party/diffsynth" "$SAMTOK_ENV/bin/python" -m samtok_edit21.distributed.cuda_readiness \
  --output "${SAMTOK_CUDA_DIAGNOSTICS:-${SAMTOK_ENV}-cuda-diagnostics}" \
  --timeout "${SAMTOK_CUDA_READY_TIMEOUT:-600}" --interval "${SAMTOK_CUDA_READY_INTERVAL:-15}"
