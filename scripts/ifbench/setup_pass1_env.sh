#!/usr/bin/env bash
# The Pass1 environment (transformers 5.19), separate from the v4 environment of Pass2 (scripts/round2/setup_env.sh, which
# runs first and provides uv when the machine has none).  Versions: configs/ifbench/requirements-pass1-lock.txt.
#   SAMTOK_ENV=<v4 env> bash scripts/ifbench/setup_pass1_env.sh <node-local env dir>
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_DIR="${1:?Pass1 environment directory}"
PYTHON_BIN="${SAMTOK_PYTHON:-/usr/bin/python3.11}"
INDEX="${SAMTOK_INDEX:-https://bytedpypi.byted.org/simple/}"
UV="${UV_BIN:-$(command -v uv || true)}"
[[ -n "$UV" ]] || UV="${SAMTOK_ENV:?Set SAMTOK_ENV or UV_BIN}-uv/bin/uv"
cd "$REPO"
[[ -f "$ENV_DIR/pyvenv.cfg" ]] || "$UV" venv --python "$PYTHON_BIN" --no-python-downloads "$ENV_DIR"
"$UV" pip install --python "$ENV_DIR/bin/python" --index-url "$INDEX" --torch-backend cu128 \
  -c configs/ifbench/requirements-pass1-lock.txt -e ./third_party/diffsynth -e '.[training,data,pass1,test]'
"$UV" pip check --python "$ENV_DIR/bin/python"
PYTHONPATH="$REPO/src:$REPO/third_party/diffsynth:$REPO/third_party" "$ENV_DIR/bin/python" -B - <<'PY'
import sys, torch, transformers, peft
assert sys.version_info[:2] == (3, 11)
assert torch.__version__.split('+')[0] == '2.8.0'
assert transformers.__version__ == '5.19.0'
assert peft.__version__ == '0.20.0'
import samtok.models  # noqa: F401
from samtok_edit21.models.codec import SamtokCodec  # noqa: F401
from samtok_edit21.pass1.grounder import Pass1Grounder  # noqa: F401
print('Pass1 environment validated: Python 3.11 / torch 2.8 / transformers 5.19 / SAMTok codec')
PY
