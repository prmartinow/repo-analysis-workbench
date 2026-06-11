#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"

exec "$PYTHON_BIN" "$SCRIPT_DIR/../src/cli/main.py" build-search \
  --workspace-root "$ROOT_DIR" \
  --raw-root "$SCRIPT_DIR/../data/raw" \
  --parsed-root "$SCRIPT_DIR/../data/parsed" \
  --search-root "$SCRIPT_DIR/../data/search" \
  "$@"
