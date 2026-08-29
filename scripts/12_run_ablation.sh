#!/usr/bin/env bash
set -euo pipefail

# Run the two principal ablations sequentially. The OPD run expects eight
# visible GPUs; the GRPO run uses four and leaves the remaining GPUs idle.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SEED="${SEED:-42}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-reasonsql-seed-$SEED}"

OUTPUT_DIR="$ROOT/checkpoints/grpo_seed_${SEED}" \
  bash scripts/10_run_grpo.sh \
  "data.seed=$SEED" \
  "trainer.experiment_name=grpo-seed-$SEED"

OUTPUT_DIR="$ROOT/checkpoints/grpo_opd_seed_${SEED}" \
  bash scripts/11_run_grpo_opd.sh \
  "data.seed=$SEED" \
  "trainer.experiment_name=grpo-opd-seed-$SEED"

