#!/usr/bin/env bash
# IF-Bench full run in one ARNOLD job: SAMTokEdit first, then the benchmark's seven frozen models.  Paste
# scripts/ifbench/settings/full_4n.sh followed by this whole file as the start command of every worker (4 x 8 GPUs).
# Each worker clones the repository at the pinned commit into node-local /tmp and builds its environments there; inputs,
# weights and results are on the shared disk, so nothing depends on a development machine.  Order: the bundle's
# no-inference preflight, our method (skipped once the run has its SUCCESS.json), then the bundle's own start.sh.
set -Eeuo pipefail
exec 2>&1  # ARNOLD's log page shows stdout; route every shell error there as well.

# ===== run settings =====
# A fresh ID per submission, identical on all workers; the benchmark's attempts are named after it.
export SAMTOK_ATTEMPT="${SAMTOK_ATTEMPT:?Set one fresh common SAMTOK_ATTEMPT on all workers}"
export SAMTOK_EDIT_COMMIT="${SAMTOK_EDIT_COMMIT:?Set the full commit SHA of the code}"
export IFBENCH_BUNDLE="${IFBENCH_BUNDLE:?Set the published benchmark bundle}"
# Our run directory on the shared disk; it stays the same across submissions.
export SAMTOK_IFBENCH_RUN="${SAMTOK_IFBENCH_RUN:-/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/ifbench_one_mp_full4}"
export SAMTOK_NNODES="${SAMTOK_NNODES:-4}"
# 1: preflight and the seven models around our method; 0: our method only.
export SAMTOK_IFBENCH_CHAIN="${SAMTOK_IFBENCH_CHAIN:-1}"
export SAMTOK_EDIT_REPO_URL="${SAMTOK_EDIT_REPO_URL:-https://github.com/Tangent0308/samtok_edit.git}"
export SAMTOK_EDIT_BRANCH="${SAMTOK_EDIT_BRANCH:-qwen-image-2.1-v4}"

# ===== ARNOLD and input checks (before anything is cloned or installed) =====
: "${ARNOLD_WORKER_NUM:?ARNOLD must inject ARNOLD_WORKER_NUM}"
: "${ARNOLD_WORKER_GPU:?ARNOLD must inject ARNOLD_WORKER_GPU}"
: "${ARNOLD_ID:?ARNOLD must inject ARNOLD_ID for this worker}"
[[ "$ARNOLD_WORKER_NUM" == "$SAMTOK_NNODES" ]] || { echo "Expected $SAMTOK_NNODES workers, ARNOLD reports $ARNOLD_WORKER_NUM" >&2; exit 2; }
[[ "$ARNOLD_ID" =~ ^[0-9]+$ && "$ARNOLD_ID" -lt "$SAMTOK_NNODES" ]] || { echo "ARNOLD_ID must be 0..$((SAMTOK_NNODES - 1))" >&2; exit 2; }
[[ "$SAMTOK_ATTEMPT" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,50}$ ]] || { echo 'Invalid SAMTOK_ATTEMPT' >&2; exit 2; }
[[ "$SAMTOK_EDIT_COMMIT" =~ ^[a-fA-F0-9]{40}$ ]] || { echo 'SAMTOK_EDIT_COMMIT must be a full commit SHA' >&2; exit 2; }
if [[ "$SAMTOK_IFBENCH_CHAIN" == 1 ]]; then
  [[ "$ARNOLD_WORKER_NUM" == 4 && "$ARNOLD_WORKER_GPU" == 8 ]] || { echo 'The benchmark run needs 4 workers x 8 GPUs' >&2; exit 2; }
  [[ -f "$IFBENCH_BUNDLE/READY.json" ]] || { echo "Bundle is not published: $IFBENCH_BUNDLE" >&2; exit 2; }
fi
for needed in configuration.json bootstrap.json repo/src assets/shared_inputs/requests; do
  [[ -e "$IFBENCH_BUNDLE/$needed" ]] || { echo "Missing $needed in $IFBENCH_BUNDLE" >&2; exit 2; }
done

NODE="$ARNOLD_ID"
export SAMTOK_NODE="$NODE"
ATTEMPT="$SAMTOK_IFBENCH_RUN/attempts/$SAMTOK_ATTEMPT"
REPO="/tmp/samtok-edit-ifbench-$SAMTOK_ATTEMPT-node$NODE"
mkdir -p "$ATTEMPT/failures"
# Atomic per-node claim: reject scheduler retries BEFORE cloning/installing or appending logs.
if ! mkdir "$ATTEMPT/node$NODE.claimed"; then
  echo "Worker $NODE already started attempt $SAMTOK_ATTEMPT. Use a NEW common SAMTOK_ATTEMPT." >&2
  exit 2
fi
exec > >(tee -a "$ATTEMPT/node$NODE.log") 2>&1
bootstrap_failed() {
  local result="${1:-$?}"
  trap - ERR TERM INT
  printf '{"error":"failed during %s; see node%s.log","exit_code":%d}\n' \
    "$(cat "$ATTEMPT/node$NODE.phase" 2>/dev/null || echo unknown)" "$NODE" "$result" > "$ATTEMPT/failures/node$NODE.json"
  exit "$result"
}
trap bootstrap_failed ERR
trap 'bootstrap_failed 143' TERM
trap 'bootstrap_failed 130' INT
echo "[$(date -u +%FT%TZ)] node $NODE of $SAMTOK_NNODES: start, attempt $SAMTOK_ATTEMPT"

if [[ "$SAMTOK_IFBENCH_CHAIN" == 1 ]]; then
  echo benchmark-preflight > "$ATTEMPT/node$NODE.phase"
  bash "$IFBENCH_BUNDLE/start.sh" --attempt "$SAMTOK_ATTEMPT-validate" --validate-only
fi
if [[ -f "$SAMTOK_IFBENCH_RUN/SUCCESS.json" ]]; then
  echo "Our method already finished: $SAMTOK_IFBENCH_RUN/SUCCESS.json"
else
  echo checkout > "$ATTEMPT/node$NODE.phase"
  export GIT_TERMINAL_PROMPT=0
  git clone --branch "$SAMTOK_EDIT_BRANCH" --single-branch "$SAMTOK_EDIT_REPO_URL" "$REPO"
  git -C "$REPO" checkout --detach "$SAMTOK_EDIT_COMMIT"
  git -C "$REPO" rev-parse HEAD > "$ATTEMPT/node$NODE.commit.txt"
  bash "$REPO/scripts/ifbench/run_arnold.sh"
  rm -rf "$REPO"
fi
if [[ "$SAMTOK_IFBENCH_CHAIN" == 1 ]]; then
  echo benchmark-seven-models > "$ATTEMPT/node$NODE.phase"
  trap - ERR TERM INT
  exec bash "$IFBENCH_BUNDLE/start.sh" --attempt "$SAMTOK_ATTEMPT"
fi
echo "[$(date -u +%FT%TZ)] node $NODE: done"
