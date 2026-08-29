#!/usr/bin/env bash
set -euo pipefail

# Download BIRD train plus a configurable dev archive mirror.
# Override URLs/repositories when the benchmark publishes a newer snapshot.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
BIRD_DIR="${BIRD_DIR:-$ROOT/data/bird}"
TRAIN_URL="${BIRD_TRAIN_URL:-https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip}"
DEV_REPO="${BIRD_DEV_REPO:-thanhdath/finer-sql-training-bundle}"
DEV_FILE="${BIRD_DEV_FILE:-bird_dev.tar.gz}"

command -v unzip >/dev/null || { echo "unzip is required" >&2; exit 2; }
mkdir -p "$BIRD_DIR"

download() {
  local url="$1"
  local output="$2"
  if command -v curl >/dev/null; then
    curl -fL --retry 5 --retry-delay 5 -C - -o "$output" "$url"
  elif command -v wget >/dev/null; then
    wget -c -O "$output" "$url"
  else
    echo "curl or wget is required" >&2
    exit 2
  fi
}

if [[ ! -f "$BIRD_DIR/train/train.json" || ! -d "$BIRD_DIR/train/train_databases" ]]; then
  archive="$BIRD_DIR/train.zip"
  echo "[1/2] Downloading BIRD train..."
  download "$TRAIN_URL" "$archive"
  unzip -q "$archive" -d "$BIRD_DIR"
  rm -f "$archive"
else
  echo "[1/2] BIRD train already exists."
fi

if [[ ! -f "$BIRD_DIR/dev/dev.json" || ! -d "$BIRD_DIR/dev/dev_databases" ]]; then
  command -v huggingface-cli >/dev/null || {
    echo "huggingface-cli is required for the default dev mirror." >&2
    echo "Install huggingface_hub[cli], or manually place dev.json and dev_databases/ under $BIRD_DIR/dev." >&2
    exit 2
  }
  temp_dir="$(mktemp -d)"
  trap 'rm -rf "$temp_dir"' EXIT
  echo "[2/2] Downloading BIRD dev mirror $DEV_REPO/$DEV_FILE..."
  huggingface-cli download "$DEV_REPO" "$DEV_FILE" --repo-type dataset --local-dir "$temp_dir"
  tar -xzf "$temp_dir/$DEV_FILE" -C "$temp_dir"
  mkdir -p "$BIRD_DIR/dev"
  dev_json="$(find "$temp_dir" -type f -name dev.json -print -quit)"
  dev_db="$(find "$temp_dir" -type d -name dev_databases -print -quit)"
  [[ -n "$dev_json" && -n "$dev_db" ]] || {
    echo "Downloaded archive does not contain dev.json and dev_databases/." >&2
    exit 2
  }
  cp "$dev_json" "$BIRD_DIR/dev/dev.json"
  cp -R "$dev_db" "$BIRD_DIR/dev/dev_databases"
else
  echo "[2/2] BIRD dev already exists."
fi

python3 - <<PY
from reasonsql.data import discover_split
for split in ("train", "dev"):
    found = discover_split("$BIRD_DIR", split)
    print(f"{split}: {found.json_path} | {found.database_dir}")
PY
