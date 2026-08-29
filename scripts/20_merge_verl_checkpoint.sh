#!/usr/bin/env bash
set -euo pipefail

# Convert a veRL FSDP actor checkpoint into a Hugging Face model directory.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_DIR="${1:?usage: $0 <global_step_N/actor> [target_dir]}"
TARGET_DIR="${2:-${LOCAL_DIR%/actor}/actor_hf}"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

python3 -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "$LOCAL_DIR" \
  --target_dir "$TARGET_DIR"

echo "Merged checkpoint: $TARGET_DIR"

