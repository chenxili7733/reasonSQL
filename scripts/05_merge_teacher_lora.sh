#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

[[ -d "$ROOT/checkpoints/teacher_lora_qwen2.5_coder_32b" ]] || {
  echo "Teacher LoRA checkpoint is missing." >&2
  exit 2
}

llamafactory-cli export configs/teacher_lora_merge.yaml "$@"

