#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ALL_ARGS=("$@")
REPO_ARGS=()
INDEX_ARGS=()

for arg in "${ALL_ARGS[@]}"; do
  if [[ "$arg" == "--help" || "$arg" == "-h" ]]; then
    PYTHON_BIN="${PYTHON_BIN:-python3}"
    exec "$PYTHON_BIN" "$SCRIPT_DIR/../src/cli/main.py" run-benchmarks --help
  fi
done

for ((i = 0; i < ${#ALL_ARGS[@]}; i++)); do
  case "${ALL_ARGS[$i]}" in
    --repo)
      REPO_ARGS+=("${ALL_ARGS[$i]}" "${ALL_ARGS[$((i + 1))]}")
      INDEX_ARGS+=("${ALL_ARGS[$i]}" "${ALL_ARGS[$((i + 1))]}")
      ((i += 1))
      ;;
    --path-prefix)
      INDEX_ARGS+=("${ALL_ARGS[$i]}" "${ALL_ARGS[$((i + 1))]}")
      ((i += 1))
      ;;
  esac
done

"$SCRIPT_DIR/parse_repos.sh" "${REPO_ARGS[@]}"
"$SCRIPT_DIR/build_index.sh" "${INDEX_ARGS[@]}"
"$SCRIPT_DIR/build_search.sh" "${REPO_ARGS[@]}"
"$SCRIPT_DIR/build_embeddings.sh" "${REPO_ARGS[@]}"
"$SCRIPT_DIR/export_summaries.sh" "${REPO_ARGS[@]}"
"$SCRIPT_DIR/precompute_eval_cache.sh" "${REPO_ARGS[@]}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

exec "$PYTHON_BIN" "$SCRIPT_DIR/../src/cli/main.py" run-benchmarks \
  --search-root "$SCRIPT_DIR/../data/search" \
  --graph-root "$SCRIPT_DIR/../data/graph" \
  --parsed-root "$SCRIPT_DIR/../data/parsed" \
  --eval-root "$SCRIPT_DIR/../data/eval" \
  "${ALL_ARGS[@]}"