#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Arctic's pinned pyserini version was validated on Python 3.9.  Override this
# when the compatible interpreter has another name, e.g. PYTHON=python3.9.
PYTHON="${PYTHON:-python}"
exec "$PYTHON" scripts/prepare_arctic_bird.py "$@"
