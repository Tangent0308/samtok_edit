#!/usr/bin/env bash
# Start the local audit VLM (Qwen3.8-27B, vLLM OpenAI server) on the given GPUs; prints the server PID once healthy.
# usage: audit_server.sh <gpus e.g. 0,1> <port 39xxx> <log file>      stop: kill <pid>
set -euo pipefail
GPUS=$1; PORT=$2; LOG=$3
(( PORT < 29000 || PORT > 29999 )) || { echo 'ports 29xxx do not bind on this machine' >&2; exit 2; }
JPY=/tmp/ifbench/fairness-dev20-runtime-20261030/judge/python
MODEL=/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen3.8-27B
IFS=, read -r -a CARDS <<< "$GPUS"
cd /tmp
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPUS" nohup "$JPY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name Qwen3.8-27B --tensor-parallel-size "${#CARDS[@]}" \
  --host 127.0.0.1 --port "$PORT" --max-model-len 16384 --gpu-memory-utilization 0.85 \
  --limit-mm-per-prompt '{"image": 2}' --seed 0 > "$LOG" 2>&1 &
PID=$!
for _ in $(seq 1 180); do
  curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { echo "$PID"; exit 0; }
  kill -0 "$PID" 2>/dev/null || { echo "server exited during start-up; see $LOG" >&2; exit 1; }
  sleep 10
done
echo "server not healthy after 30 min; see $LOG" >&2; exit 1
