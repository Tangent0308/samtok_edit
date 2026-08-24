#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
DIFFSYNTH_DIR="$REPO_ROOT/DiffSynth-Studio"

show_help() {
  cat <<'EOF'
Create the SAMTokEdit Python 3.11 training environment with uv.

Usage:
  bash setup_env.sh

Environment overrides:
  SAMTOK_EDIT_VENV           virtualenv path (default: <repo>/.venv)
  SAMTOK_EDIT_PYTHON         Python 3.11 executable (default: python3.11)
  SAMTOK_EDIT_INDEX          package index (default: ByteDance PyPI)
  SAMTOK_EDIT_UV_VERSION     uv version used only when bootstrapping uv
  SAMTOK_EDIT_TORCH_BACKEND  uv PyTorch backend (default: cu128)
  SAMTOK_EDIT_REQUIRE_CUDA   require a usable CUDA device: 1 or 0 (default: 1)
  SAMTOK_EDIT_RUN_TESTS      run the repository unit tests: 1 or 0 (default: 1)

This script deliberately uses the uv pip interface, not uv sync, so it never
creates uv.lock. The vendored DiffSynth-Studio tree is installed editable.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  show_help
  exit 0
fi
if (( $# != 0 )); then
  echo "Unexpected arguments: $*" >&2
  show_help >&2
  exit 2
fi

VENV_PATH="${SAMTOK_EDIT_VENV:-$REPO_ROOT/.venv}"
PACKAGE_INDEX="${SAMTOK_EDIT_INDEX:-https://bytedpypi.byted.org/simple/}"
UV_VERSION="${SAMTOK_EDIT_UV_VERSION:-0.11.32}"
TORCH_BACKEND="${SAMTOK_EDIT_TORCH_BACKEND:-cu128}"
REQUIRE_CUDA="${SAMTOK_EDIT_REQUIRE_CUDA:-1}"
RUN_TESTS="${SAMTOK_EDIT_RUN_TESTS:-1}"

case "$REQUIRE_CUDA" in
  0|1) ;;
  *) echo "SAMTOK_EDIT_REQUIRE_CUDA must be 0 or 1, got: $REQUIRE_CUDA" >&2; exit 2 ;;
esac
case "$RUN_TESTS" in
  0|1) ;;
  *) echo "SAMTOK_EDIT_RUN_TESTS must be 0 or 1, got: $RUN_TESTS" >&2; exit 2 ;;
esac

if [[ ! -f "$REPO_ROOT/pyproject.toml" ]]; then
  echo "Missing $REPO_ROOT/pyproject.toml" >&2
  exit 1
fi
if [[ ! -f "$DIFFSYNTH_DIR/pyproject.toml" ]]; then
  echo "Missing vendored DiffSynth project: $DIFFSYNTH_DIR/pyproject.toml" >&2
  exit 1
fi

if [[ -n "${SAMTOK_EDIT_PYTHON:-}" ]]; then
  PYTHON_BIN="$SAMTOK_EDIT_PYTHON"
else
  PYTHON_BIN="$(command -v python3.11 || true)"
fi
if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "Python 3.11 was not found. Set SAMTOK_EDIT_PYTHON to its executable." >&2
  exit 1
fi

PYTHON_VERSION="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$PYTHON_VERSION" != "3.11" ]]; then
  echo "SAMTokEdit requires Python 3.11, got $PYTHON_VERSION from $PYTHON_BIN" >&2
  exit 1
fi

if [[ -n "${UV_BIN:-}" ]]; then
  UV_EXECUTABLE="$UV_BIN"
else
  UV_EXECUTABLE="$(command -v uv || true)"
fi
if [[ -z "$UV_EXECUTABLE" && -x "${HOME}/.local/bin/uv" ]]; then
  UV_EXECUTABLE="${HOME}/.local/bin/uv"
fi
if [[ -z "$UV_EXECUTABLE" ]]; then
  echo "[setup] uv not found; installing uv==$UV_VERSION into the user site"
  if ! "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
    echo "Cannot bootstrap uv because $PYTHON_BIN has no pip." >&2
    echo "Install uv==$UV_VERSION, then rerun with UV_BIN=/path/to/uv." >&2
    exit 1
  fi
  "$PYTHON_BIN" -m pip install --user --index-url "$PACKAGE_INDEX" "uv==$UV_VERSION"
  UV_EXECUTABLE="${HOME}/.local/bin/uv"
fi
if [[ ! -x "$UV_EXECUTABLE" ]]; then
  echo "uv is not executable: $UV_EXECUTABLE" >&2
  exit 1
fi

echo "[setup] repo=$REPO_ROOT"
echo "[setup] python=$PYTHON_BIN ($PYTHON_VERSION)"
echo "[setup] uv=$UV_EXECUTABLE ($("$UV_EXECUTABLE" --version))"
echo "[setup] venv=$VENV_PATH"
echo "[setup] index=$PACKAGE_INDEX torch_backend=$TORCH_BACKEND"

if [[ -e "$VENV_PATH" && ! -f "$VENV_PATH/pyvenv.cfg" ]]; then
  echo "Refusing to overwrite a non-virtualenv path: $VENV_PATH" >&2
  exit 1
fi
if [[ ! -f "$VENV_PATH/pyvenv.cfg" ]]; then
  mkdir -p "$(dirname "$VENV_PATH")"
  "$UV_EXECUTABLE" venv "$VENV_PATH" \
    --python "$PYTHON_BIN" \
    --no-python-downloads \
    --default-index "$PACKAGE_INDEX" \
    --seed
else
  EXISTING_VERSION="$("$VENV_PATH/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  if [[ "$EXISTING_VERSION" != "3.11" ]]; then
    echo "Existing environment uses Python $EXISTING_VERSION, expected 3.11: $VENV_PATH" >&2
    exit 1
  fi
  echo "[setup] reusing existing Python 3.11 environment"
fi

VENV_PYTHON="$VENV_PATH/bin/python"
"$UV_EXECUTABLE" pip install \
  --python "$VENV_PYTHON" \
  --default-index "$PACKAGE_INDEX" \
  --torch-backend "$TORCH_BACKEND" \
  --compile-bytecode \
  --strict \
  --editable "$REPO_ROOT" \
  --editable "$DIFFSYNTH_DIR"

"$UV_EXECUTABLE" pip check --python "$VENV_PYTHON"

SAMTOK_EDIT_REPO_ROOT="$REPO_ROOT" \
SAMTOK_EDIT_REQUIRE_CUDA="$REQUIRE_CUDA" \
PYTHONPATH="$DIFFSYNTH_DIR:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
"$VENV_PYTHON" - <<'PY'
from importlib.metadata import version
import os
from pathlib import Path

import torch
import wandb
import diffsynth

expected = {
    "accelerate": "1.14.0",
    "byted-wandb": "0.13.98",
    "datasets": "5.0.1",
    "diffsynth": "2.1.2",
    "peft": "0.20.0",
    "torch": "2.8.0",
    "torchvision": "0.23.0",
    "transformers": "5.12.1",
}
for distribution, wanted in expected.items():
    actual = version(distribution)
    # CUDA wheels append a local version such as +cu128 to the pinned public
    # torch/torchvision version.
    if actual.split("+", 1)[0] != wanted:
        raise RuntimeError(f"{distribution}: expected {wanted}, got {actual}")

if torch.version.cuda != "12.8":
    raise RuntimeError(f"Expected a CUDA 12.8 PyTorch build, got torch.version.cuda={torch.version.cuda}")
if os.environ["SAMTOK_EDIT_REQUIRE_CUDA"] == "1" and not torch.cuda.is_available():
    raise RuntimeError("CUDA is required but torch.cuda.is_available() is false")

repo_root = Path(os.environ["SAMTOK_EDIT_REPO_ROOT"]).resolve()
diffsynth_file = Path(diffsynth.__file__).resolve()
if not diffsynth_file.is_relative_to(repo_root / "DiffSynth-Studio"):
    raise RuntimeError(f"diffsynth was not imported from the vendored tree: {diffsynth_file}")

print(
    "[verify] "
    f"torch={torch.__version__} cuda_runtime={torch.version.cuda} "
    f"cuda_available={torch.cuda.is_available()} wandb={wandb.__version__}"
)
if torch.cuda.is_available():
    print(f"[verify] gpu_count={torch.cuda.device_count()} gpu0={torch.cuda.get_device_name(0)}")
print(f"[verify] diffsynth={version('diffsynth')} source={diffsynth_file}")
PY

if [[ "$RUN_TESTS" == "1" ]]; then
  echo "[setup] running repository unit tests"
  (
    cd "$REPO_ROOT"
    PYTHONPATH="$DIFFSYNTH_DIR:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
      "$VENV_PYTHON" -m unittest -v tests/test_samtok_edit.py
  )
fi

echo
echo "Environment ready. Activate it with:"
printf '  source %q\n' "$VENV_PATH/bin/activate"
echo "Training launchers already prepend the vendored DiffSynth-Studio source tree."
echo "Set WANDB_API_KEY, WANDB_ENTITY, and WANDB_PROJECT before Stage 1/Stage 2 training."
