#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$ROOT/log/checkpoint_detection.tar}"
DATA_DIR="${DATA_DIR:-$ROOT/example_data}"

python "$ROOT/demo.py" \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --data-dir "$DATA_DIR" \
  --image-prefix test_ \
  --top_down_grasp \
  --debug \
  "$@"
