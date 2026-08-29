#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

[[ -f "$ROOT/data/processed/bird_sft.jsonl" ]] || {
  echo "Run scripts/01_prepare_data.py first." >&2
  exit 2
}
command -v llamafactory-cli >/dev/null || {
  echo "llamafactory-cli is not installed." >&2
  exit 2
}

export CUDA_VISIBLE_DEVICES="${TEACHER_GPUS:-0,1,2,3}"
export FORCE_TORCHRUN=1
export TOKENIZERS_PARALLELISM=false

llamafactory-cli train configs/teacher_lora_sft.yaml "$@"

