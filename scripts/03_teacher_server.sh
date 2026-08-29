#!/usr/bin/env bash
set -euo pipefail

# Optional standalone teacher server for throughput/token-logprob debugging.
# Native GRPO+OPD training in scripts/11_run_grpo_opd.sh launches its own
# dedicated veRL TeacherServerActor and does not require this process.

TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen2.5-Coder-32B-Instruct}"
TEACHER_GPUS="${TEACHER_GPUS:-0,1,2,3}"
TEACHER_TP="${TEACHER_TP:-4}"
PORT="${PORT:-18001}"

export CUDA_VISIBLE_DEVICES="$TEACHER_GPUS"

python3 -m vllm.entrypoints.openai.api_server \
  --model "$TEACHER_MODEL" \
  --served-model-name reasonsql-teacher \
  --tensor-parallel-size "$TEACHER_TP" \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 \
  --max-num-batched-tokens 16384 \
  --dtype bfloat16 \
  --port "$PORT"

