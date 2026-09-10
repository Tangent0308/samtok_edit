#!/usr/bin/env bash
set -euo pipefail

# This file is a self-contained Arnold job entry. It is intentionally able to
# run before the SAMTokEdit repository exists at SAMTOK_EDIT_REPO_DIR.

show_help() {
  cat <<'EOF'
Clone SAMTokEdit, create its uv environment, and run Stage 1 then Stage 2 on
four Arnold workers with eight GPUs each.

The same script must be used as the entry command on every Arnold worker.
Arnold supplies:
  ARNOLD_WORKER_HOSTS ARNOLD_WORKER_NUM ARNOLD_WORKER_GPU ARNOLD_ID

Required user values (export them in the Arnold entry before this script):
  WANDB_API_KEY WANDB_ENTITY WANDB_PROJECT

Useful overrides:
  SAMTOK_RUN_ID             unique shared run name; defaults to crispedit-refined-4node-<job-id>
  SAMTOK_EDIT_REPO_URL      default: GitHub SAMTokEdit repository
  SAMTOK_EDIT_BRANCH        default: dev_crispedit_refined
  SAMTOK_EDIT_REPO_DIR      shared clone destination
  SAMTOK_EDIT_VENV          shared uv virtualenv destination
  RUN_ROOT                  shared experiment output directory
  SAMTOK_SKIP_APT=1         skip per-node system package installation
  SAMTOK_GIT_HTTP_PROXY     optional proxy used only for git clone; default: direct
  SAMTOK_EDIT_INDEX         uv/Python package index
  SAMTOK_EDIT_UV_VERSION    uv version installed before setup_env.sh
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  show_help
  exit 0
fi
if (( $# != 0 )); then
  show_help >&2
  exit 2
fi

# Defaults used by the previous SAMTokEdit experiments. The API key remains a
# required user-supplied value and is never stored in the repository.
WANDB_ENTITY="${WANDB_ENTITY:-2200012743-peking-university}"
WANDB_PROJECT="${WANDB_PROJECT:-samtok-edit}"

: "${ARNOLD_WORKER_HOSTS:?Arnold must provide ARNOLD_WORKER_HOSTS}"
: "${ARNOLD_WORKER_NUM:?Arnold must provide ARNOLD_WORKER_NUM}"
: "${ARNOLD_WORKER_GPU:?Arnold must provide ARNOLD_WORKER_GPU}"
: "${ARNOLD_ID:?Arnold must provide ARNOLD_ID}"
: "${WANDB_API_KEY:?Fill WANDB_API_KEY in the Arnold entry before running}"

if [[ ! "$ARNOLD_WORKER_NUM" =~ ^[0-9]+$ || "$ARNOLD_WORKER_NUM" != "4" ]]; then
  echo "This entry requires ARNOLD_WORKER_NUM=4, got $ARNOLD_WORKER_NUM" >&2
  exit 2
fi
if [[ ! "$ARNOLD_WORKER_GPU" =~ ^[0-9]+$ || "$ARNOLD_WORKER_GPU" != "8" ]]; then
  echo "This entry requires ARNOLD_WORKER_GPU=8, got $ARNOLD_WORKER_GPU" >&2
  exit 2
fi
if [[ ! "$ARNOLD_ID" =~ ^[0-9]+$ ]] || (( ARNOLD_ID < 0 || ARNOLD_ID >= ARNOLD_WORKER_NUM )); then
  echo "ARNOLD_ID must be in [0, 3], got $ARNOLD_ID" >&2
  exit 2
fi

FIRST_WORKER="${ARNOLD_WORKER_HOSTS%%,*}"
if [[ "$FIRST_WORKER" =~ ^\[([^]]+)\]:([0-9]+)$ ]]; then
  INFERRED_MASTER_ADDR="${BASH_REMATCH[1]}"
  INFERRED_MASTER_PORT="${BASH_REMATCH[2]}"
elif [[ "$FIRST_WORKER" =~ ^([^:]+):([0-9]+)$ ]]; then
  INFERRED_MASTER_ADDR="${BASH_REMATCH[1]}"
  INFERRED_MASTER_PORT="${BASH_REMATCH[2]}"
else
  echo "ARNOLD_WORKER_HOSTS must contain host:port or [IPv6]:port entries, got: $ARNOLD_WORKER_HOSTS" >&2
  exit 2
fi
MASTER_ADDR="${MASTER_ADDR:-${ARNOLD_WORKER_0_HOST:-$INFERRED_MASTER_ADDR}}"
if [[ "$MASTER_ADDR" =~ ^\[([^]]+)\]$ ]]; then
  MASTER_ADDR="${BASH_REMATCH[1]}"
fi
MASTER_PORT="${MASTER_PORT:-${PORT:-$INFERRED_MASTER_PORT}}"
NNODES="$ARNOLD_WORKER_NUM"
NODE_RANK="$ARNOLD_ID"
GPUS_PER_NODE="$ARNOLD_WORKER_GPU"

RUN_SUFFIX="${ARNOLD_JOB_ID:-${ARNOLD_TASK_ID:-$MASTER_PORT}}"
SAMTOK_RUN_ID="${SAMTOK_RUN_ID:-crispedit-refined-4node-${RUN_SUFFIX}}"
if [[ ! "$SAMTOK_RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "SAMTOK_RUN_ID may contain only letters, numbers, dot, underscore, and dash: $SAMTOK_RUN_ID" >&2
  exit 2
fi

SAMTOK_EDIT_REPO_URL="${SAMTOK_EDIT_REPO_URL:-https://github.com/Tangent0308/samtok_edit.git}"
SAMTOK_EDIT_BRANCH="${SAMTOK_EDIT_BRANCH:-dev_crispedit_refined}"
SAMTOK_EDIT_REPO_DIR="${SAMTOK_EDIT_REPO_DIR:-/mnt/bn/strategy-mllm-train/user/tanyue/workspaces/samtok_edit_${SAMTOK_RUN_ID}}"
SAMTOK_EDIT_VENV="${SAMTOK_EDIT_VENV:-$SAMTOK_EDIT_REPO_DIR/.venv}"
RUN_ROOT="${RUN_ROOT:-/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTokEdit/crispedit_refined_4node/$SAMTOK_RUN_ID}"
BOOTSTRAP_CONTROL="${BOOTSTRAP_CONTROL:-$RUN_ROOT/bootstrap_control}"
BOOTSTRAP_TIMEOUT_SECONDS="${BOOTSTRAP_TIMEOUT_SECONDS:-7200}"
SAMTOK_GIT_HTTP_PROXY="${SAMTOK_GIT_HTTP_PROXY:-}"
SAMTOK_EDIT_INDEX="${SAMTOK_EDIT_INDEX:-https://bytedpypi.byted.org/simple/}"
SAMTOK_EDIT_UV_VERSION="${SAMTOK_EDIT_UV_VERSION:-0.11.32}"

export WANDB_API_KEY WANDB_ENTITY WANDB_PROJECT
export MASTER_ADDR MASTER_PORT NNODES NODE_RANK GPUS_PER_NODE
export SAMTOK_RUN_ID SAMTOK_EDIT_VENV RUN_ROOT BOOTSTRAP_CONTROL
# The pipeline may initialize a RUN_ROOT that contains only bootstrap logs and
# bootstrap coordination files created by this entry script.
export SAMTOK_ALLOW_BOOTSTRAP_RUN_ROOT=1

timestamp() {
  date -u +'%Y-%m-%dT%H:%M:%SZ'
}

log() {
  printf '[%s][bootstrap-node=%s] %s\n' "$(timestamp)" "$NODE_RANK" "$*"
}

# Capture apt, clone, uv setup, environment verification, orchestration, and
# training output in the shared experiment result directory. Each worker owns
# one file, so concurrent writes never target the same log.
mkdir -p "$RUN_ROOT/logs" "$BOOTSTRAP_CONTROL"
BOOTSTRAP_LOG="$RUN_ROOT/logs/bootstrap.node${NODE_RANK}.log"
exec > >(tee -a "$BOOTSTRAP_LOG") 2>&1
log "capturing complete worker output in $BOOTSTRAP_LOG"

write_marker() {
  local path="$1"
  local value="${2:-ok}"
  local temporary="${path}.tmp.${NODE_RANK}.$$"
  printf '%s\n' "$value" >"$temporary"
  mv "$temporary" "$path"
}

wait_for_bootstrap() {
  local started=$SECONDS
  while true; do
    if [[ -f "$BOOTSTRAP_CONTROL/environment.failed" ]]; then
      echo "Controller environment setup failed: $(<"$BOOTSTRAP_CONTROL/environment.failed")" >&2
      return 1
    fi
    if [[ -f "$BOOTSTRAP_CONTROL/environment.ok" ]]; then
      return 0
    fi
    if (( SECONDS - started >= BOOTSTRAP_TIMEOUT_SECONDS )); then
      echo "Timed out waiting for shared clone/environment after ${BOOTSTRAP_TIMEOUT_SECONDS}s" >&2
      return 1
    fi
    sleep 2
  done
}

if [[ "${SAMTOK_SKIP_APT:-0}" != "1" ]]; then
  log "installing per-node system libraries"
  sudo apt-get install ffmpeg libsm6 libxext6 tmux htop -y
fi

# Direct GitHub access is the verified default on the Arnold workers. A proxy
# is used only when the caller explicitly supplies SAMTOK_GIT_HTTP_PROXY.
if [[ -n "$SAMTOK_GIT_HTTP_PROXY" ]]; then
  export http_proxy="$SAMTOK_GIT_HTTP_PROXY"
  export https_proxy="$SAMTOK_GIT_HTTP_PROXY"
  export HTTP_PROXY="$SAMTOK_GIT_HTTP_PROXY"
  export HTTPS_PROXY="$SAMTOK_GIT_HTTP_PROXY"
  export no_proxy=".byted.org"
  export NO_PROXY=".byted.org"
  log "using caller-supplied proxy for git clone"
else
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
  log "using verified direct GitHub access for git clone"
fi

if (( NODE_RANK == 0 )); then
  mkdir -p "$(dirname "$SAMTOK_EDIT_REPO_DIR")"
  if [[ -e "$SAMTOK_EDIT_REPO_DIR" ]]; then
    write_marker "$BOOTSTRAP_CONTROL/environment.failed" "repository destination already exists: $SAMTOK_EDIT_REPO_DIR"
    echo "Refusing to overwrite existing repository destination: $SAMTOK_EDIT_REPO_DIR" >&2
    exit 1
  fi

  log "cloning $SAMTOK_EDIT_REPO_URL branch $SAMTOK_EDIT_BRANCH"
  set +e
  env -u WANDB_API_KEY \
    git clone --branch "$SAMTOK_EDIT_BRANCH" --single-branch \
      "$SAMTOK_EDIT_REPO_URL" "$SAMTOK_EDIT_REPO_DIR"
  CLONE_STATUS=$?
  set -e
  if (( CLONE_STATUS != 0 )); then
    write_marker "$BOOTSTRAP_CONTROL/environment.failed" "git clone exit code $CLONE_STATUS"
    exit "$CLONE_STATUS"
  fi

  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
  log "installing uv==$SAMTOK_EDIT_UV_VERSION after clearing clone proxies"
  BOOTSTRAP_PYTHON="${SAMTOK_EDIT_PYTHON:-$(command -v python3.11 || true)}"
  if [[ -z "$BOOTSTRAP_PYTHON" || ! -x "$BOOTSTRAP_PYTHON" ]]; then
    write_marker "$BOOTSTRAP_CONTROL/environment.failed" "Python 3.11 is unavailable"
    echo "Python 3.11 was not found; set SAMTOK_EDIT_PYTHON." >&2
    exit 1
  fi
  set +e
  env \
    -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u WANDB_API_KEY \
    "$BOOTSTRAP_PYTHON" -m pip install --user \
      --index-url "$SAMTOK_EDIT_INDEX" \
      "uv==$SAMTOK_EDIT_UV_VERSION"
  UV_STATUS=$?
  set -e
  if (( UV_STATUS != 0 )); then
    write_marker "$BOOTSTRAP_CONTROL/environment.failed" "uv installation exit code $UV_STATUS"
    exit "$UV_STATUS"
  fi
  UV_USER_BASE="$("$BOOTSTRAP_PYTHON" -c 'import site; print(site.getuserbase())')"
  UV_EXECUTABLE="$UV_USER_BASE/bin/uv"
  if [[ ! -x "$UV_EXECUTABLE" ]]; then
    write_marker "$BOOTSTRAP_CONTROL/environment.failed" "uv executable is missing: $UV_EXECUTABLE"
    echo "uv installation completed but executable is missing: $UV_EXECUTABLE" >&2
    exit 1
  fi

  cd "$SAMTOK_EDIT_REPO_DIR"
  log "creating the pinned uv environment with setup_env.sh"
  set +e
  env \
    -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u WANDB_API_KEY \
    UV_BIN="$UV_EXECUTABLE" \
    SAMTOK_EDIT_VENV="$SAMTOK_EDIT_VENV" \
    SAMTOK_EDIT_INDEX="$SAMTOK_EDIT_INDEX" \
    SAMTOK_EDIT_UV_VERSION="$SAMTOK_EDIT_UV_VERSION" \
    SAMTOK_EDIT_REQUIRE_CUDA=1 \
    SAMTOK_EDIT_RUN_TESTS=1 \
    bash setup_env.sh
  SETUP_STATUS=$?
  set -e
  if (( SETUP_STATUS != 0 )); then
    write_marker "$BOOTSTRAP_CONTROL/environment.failed" "setup_env.sh exit code $SETUP_STATUS"
    exit "$SETUP_STATUS"
  fi

  git -C "$SAMTOK_EDIT_REPO_DIR" rev-parse HEAD >"$BOOTSTRAP_CONTROL/git_commit.txt"
  write_marker "$BOOTSTRAP_CONTROL/environment.ok"
  log "shared clone and environment are ready"
else
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
  while [[ ! -d "$BOOTSTRAP_CONTROL" ]]; do sleep 2; done
  log "waiting for node 0 to clone and install the environment"
  wait_for_bootstrap
fi

wait_for_bootstrap
[[ -x "$SAMTOK_EDIT_VENV/bin/python" ]] || { echo "Shared uv environment is not executable: $SAMTOK_EDIT_VENV" >&2; exit 1; }
[[ -x "$SAMTOK_EDIT_REPO_DIR/scripts/train/run_arnold_4node_pipeline.sh" ]] || {
  echo "The cloned branch does not contain the four-node pipeline script." >&2
  echo "Commit and push the new scripts before submitting this Arnold entry." >&2
  exit 1
}

cd "$SAMTOK_EDIT_REPO_DIR"
# The pipeline also resolves binaries from SAMTOK_EDIT_VENV explicitly, but
# activation keeps the entry behavior identical to a documented manual run.
# shellcheck disable=SC1091
source "$SAMTOK_EDIT_VENV/bin/activate"
log "starting the four-node SAMTokEdit pipeline; run_root=$RUN_ROOT"
set +e
bash scripts/train/run_arnold_4node_pipeline.sh
PIPELINE_STATUS=$?
set -e
if (( PIPELINE_STATUS != 0 )); then
  write_marker "$BOOTSTRAP_CONTROL/pipeline.node${NODE_RANK}.failed" "$PIPELINE_STATUS"
  exit "$PIPELINE_STATUS"
fi
write_marker "$BOOTSTRAP_CONTROL/pipeline.node${NODE_RANK}.done"
log "pipeline completed successfully"
