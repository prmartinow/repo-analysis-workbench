#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"

exec "$PYTHON_BIN" "$SCRIPT_DIR/../src/cli/main.py" parse-repos \
  --workspace-root "$ROOT_DIR" \
  --output-root "$SCRIPT_DIR/../data/raw" \
  "$@"
