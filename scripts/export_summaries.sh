#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"

exec "$PYTHON_BIN" "$SCRIPT_DIR/../src/cli/main.py" build-summaries \
  --raw-root "$SCRIPT_DIR/../data/raw" \
  --parsed-root "$SCRIPT_DIR/../data/parsed" \
  --graph-root "$SCRIPT_DIR/../data/graph" \
  --summary-root "$SCRIPT_DIR/../data/summaries" \
  "$@"
