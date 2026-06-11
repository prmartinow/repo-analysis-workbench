#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"

printf '[build-embeddings.sh] starting search_root=%s args=%s\n' \
  "$SCRIPT_DIR/../data/search" \
  "$*" >&2

exec env PYTHONUNBUFFERED=1 "$PYTHON_BIN" "$SCRIPT_DIR/../src/cli/main.py" build-embeddings \
  --search-root "$SCRIPT_DIR/../data/search" \
  "$@"