#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "Bootstrapping repo-analysis workspace..."
mkdir -p data/{raw,parsed,graph,search,summaries,eval}
mkdir -p data/eval/prompt_exports
for d in raw parsed graph search summaries eval; do
  touch "data/$d/.gitkeep"
done
touch "data/eval/prompt_exports/.gitkeep"

if command -v python3 >/dev/null 2>&1; then
  echo "python3: $(python3 --version 2>&1)"
  if python3 - <<'PY' >/dev/null 2>&1
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("pyarrow") else 1)
PY
  then
    echo "pyarrow: available"
  else
    echo "pyarrow: not installed (parquet export will emit status only)"
  fi
else
  echo "WARNING: python3 not found; inventory tooling will not run." >&2
fi

if command -v cargo >/dev/null 2>&1; then
  echo "cargo: $(cargo --version 2>&1)"
else
  echo "WARNING: cargo not found; upstream Rust builds cannot be verified locally." >&2
fi

if command -v rustc >/dev/null 2>&1; then
  echo "rustc: $(rustc --version 2>&1)"
else
  echo "WARNING: rustc not found; compiler-backed AST probing is unavailable locally." >&2
fi

if command -v node >/dev/null 2>&1; then
  echo "node: $(node --version 2>&1)"
else
  echo "WARNING: node not found; package workspace tooling is unavailable locally." >&2
fi

echo "Done."
