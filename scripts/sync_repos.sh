#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

cat <<EOF
repo-analysis-workbench does not manage target repositories as submodules.

Place target repositories next to this workbench or pass --workspace-root to
the build scripts. Current workspace:

  $ROOT_DIR

Detected sibling repositories:
EOF

find "$ROOT_DIR" -mindepth 1 -maxdepth 1 -type d \
  ! -name repo-analysis-workbench \
  ! -name '.*' \
  -printf '  %f\n' | sort
