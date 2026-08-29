#!/usr/bin/env bash
set -euo pipefail

# Install the exact veRL release targeted by ReasonSQL's native OPD launcher.
# Run this inside a CUDA-enabled Python environment.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERL_TAG="${VERL_TAG:-v0.8.0}"
VERL_SRC="${VERL_SRC:-$ROOT/third_party/verl}"

mkdir -p "$(dirname "$VERL_SRC")"
if [[ ! -d "$VERL_SRC/.git" ]]; then
  git clone https://github.com/verl-project/verl.git "$VERL_SRC"
fi

git -C "$VERL_SRC" fetch --tags
git -C "$VERL_SRC" checkout "$VERL_TAG"
python3 -m pip install -e "$VERL_SRC"

echo "Installed veRL $VERL_TAG from $VERL_SRC"
echo "For coefficient annealing, run:"
echo "  python3 $ROOT/patches/apply_patch.py --verl-src $VERL_SRC"

