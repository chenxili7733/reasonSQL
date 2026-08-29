#!/usr/bin/env bash
set -euo pipefail

# Evaluate multiple HuggingFace checkpoints with identical settings.
#
# Usage:
#   bash eval/eval_series.sh \
#     checkpoints/student_sft_qwen2.5_coder_7b \
#     checkpoints/grpo/global_step_100/actor_hf \
#     checkpoints/grpo_opd/global_step_100/actor_hf

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

[[ "$#" -ge 1 ]] || {
  echo "usage: $0 <model_dir> [model_dir...]" >&2
  exit 2
}

BACKEND="${BACKEND:-vllm}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
BATCH_SIZE="${BATCH_SIZE:-8}"
BIRD_DIR="${BIRD_DIR:-data/bird}"
OUT_DIR="${OUT_DIR:-data/eval/series}"

mkdir -p "$OUT_DIR"

for model_dir in "$@"; do
  model_name="$(basename "$(cd "$model_dir" && pwd -P)")"
  safe_name="$(printf '%s' "$model_name" | tr '/ ' '__')"
  output="$OUT_DIR/${safe_name}.json"
  echo "==================================================================="
  echo "Evaluating $model_dir -> $output"
  python3 eval/eval_bird.py \
    --model "$model_dir" \
    --backend "$BACKEND" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --batch-size "$BATCH_SIZE" \
    --bird-dir "$BIRD_DIR" \
    --output "$output" \
    --raw-predictions "$OUT_DIR/${safe_name}_predictions.jsonl"
done

echo "==================================================================="
echo "Series results saved under $OUT_DIR"
python3 - "$OUT_DIR" <<'PY'
import json
import pathlib
import sys

out_dir = pathlib.Path(sys.argv[1])
rows = []
for path in sorted(out_dir.glob("*.json")):
    if path.name.endswith("_predictions.jsonl"):
        continue
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = payload.get("summary", {})
    except Exception as exc:
        rows.append((path.stem, None, str(exc)))
        continue
    rows.append(
        (
            path.stem,
            summary.get("execution_accuracy"),
            f"{summary.get('correct', 0)}/{summary.get('total', 0)}",
        )
    )
print(f"\n{'checkpoint':45s} {'EX':>8s}  correct/total")
for name, accuracy, counts in sorted(rows, key=lambda item: (item[1] is None, -(item[1] or 0.0))):
    if accuracy is None:
        print(f"{name:45s} {'ERR':>8s}  {counts}")
    else:
        print(f"{name:45s} {accuracy:8.2%}  {counts}")
PY

