#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"

exec "$PYTHON_BIN" "$SCRIPT_DIR/../src/cli/main.py" export-benchmark-prompts \
  --search-root "$SCRIPT_DIR/../data/search" \
  --graph-root "$SCRIPT_DIR/../data/graph" \
  --parsed-root "$SCRIPT_DIR/../data/parsed" \
  --eval-root "$SCRIPT_DIR/../data/eval" \
  "$@"