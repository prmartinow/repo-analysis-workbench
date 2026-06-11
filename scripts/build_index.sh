#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
STATE_FILE_DEFAULT="$SCRIPT_DIR/../data/parsed/.build_index.resume.sqlite3"

ALL_ARGS=("$@")
PASSTHROUGH_ARGS=()
HAS_REPO=0
RESUME_ENABLED=1
STATE_FILE="$STATE_FILE_DEFAULT"

for ((i = 0; i < ${#ALL_ARGS[@]}; i++)); do
  arg="${ALL_ARGS[$i]}"
  case "$arg" in
    --help|-h)
      exec "$PYTHON_BIN" "$SCRIPT_DIR/../src/cli/main.py" build-index --help
      ;;
    --repo)
      HAS_REPO=1
      PASSTHROUGH_ARGS+=("$arg" "${ALL_ARGS[$((i + 1))]}")
      ((i += 1))
      ;;
    --resume-state-file)
      STATE_FILE="${ALL_ARGS[$((i + 1))]}"
      ((i += 1))
      ;;
    --no-resume)
      RESUME_ENABLED=0
      ;;
    *)
      PASSTHROUGH_ARGS+=("$arg")
      ;;
  esac
done

run_single_repo() {
  local repo_name="$1"
  echo "[build-index.sh] running repo=${repo_name}"
  "$PYTHON_BIN" "$SCRIPT_DIR/../src/cli/main.py" build-index \
    --workspace-root "$ROOT_DIR" \
    --raw-root "$SCRIPT_DIR/../data/raw" \
    --parsed-root "$SCRIPT_DIR/../data/parsed" \
    --graph-root "$SCRIPT_DIR/../data/graph" \
    --repo "$repo_name" \
    "${PASSTHROUGH_ARGS[@]}"
}

if [[ "$HAS_REPO" -eq 1 || "$RESUME_ENABLED" -eq 0 ]]; then
  exec "$PYTHON_BIN" "$SCRIPT_DIR/../src/cli/main.py" build-index \
    --workspace-root "$ROOT_DIR" \
    --raw-root "$SCRIPT_DIR/../data/raw" \
    --parsed-root "$SCRIPT_DIR/../data/parsed" \
    --graph-root "$SCRIPT_DIR/../data/graph" \
    "${PASSTHROUGH_ARGS[@]}"
fi

mkdir -p "$(dirname "$STATE_FILE")"

init_state_db() {
  "$PYTHON_BIN" - <<PY
import sqlite3
db_path = r"""$STATE_FILE"""
with sqlite3.connect(db_path) as connection:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS build_index_checkpoints (
            repo TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL DEFAULT (datetime('now')),
            last_pid INTEGER,
            last_exit_code INTEGER
        )
        """
    )
PY
}

is_done() {
  local repo_name="$1"
  "$PYTHON_BIN" - <<PY
import sqlite3
db_path = r"""$STATE_FILE"""
repo = r"""$repo_name"""
with sqlite3.connect(db_path) as connection:
    row = connection.execute(
        "SELECT status FROM build_index_checkpoints WHERE repo = ?",
        [repo],
    ).fetchone()
raise SystemExit(0 if row and row[0] == "completed" else 1)
PY
}

mark_running() {
  local repo_name="$1"
  local pid="$2"
  "$PYTHON_BIN" - <<PY
import sqlite3
db_path = r"""$STATE_FILE"""
repo = r"""$repo_name"""
pid = int("$pid")
with sqlite3.connect(db_path) as connection:
    connection.execute(
        """
        INSERT INTO build_index_checkpoints(repo, status, attempts, updated_at, last_pid, last_exit_code)
        VALUES(?, 'running', 1, datetime('now'), ?, NULL)
        ON CONFLICT(repo) DO UPDATE SET
            status='running',
            attempts=build_index_checkpoints.attempts + 1,
            updated_at=datetime('now'),
            last_pid=excluded.last_pid,
            last_exit_code=NULL
        """,
        [repo, pid],
    )
PY
}

mark_done() {
  local repo_name="$1"
  "$PYTHON_BIN" - <<PY
import sqlite3
db_path = r"""$STATE_FILE"""
repo = r"""$repo_name"""
with sqlite3.connect(db_path) as connection:
    connection.execute(
        """
        INSERT INTO build_index_checkpoints(repo, status, attempts, updated_at, last_pid, last_exit_code)
        VALUES(?, 'completed', 1, datetime('now'), NULL, 0)
        ON CONFLICT(repo) DO UPDATE SET
            status='completed',
            updated_at=datetime('now'),
            last_exit_code=0
        """,
        [repo],
    )
PY
}

mark_failed() {
  local repo_name="$1"
  local exit_code="$2"
  "$PYTHON_BIN" - <<PY
import sqlite3
db_path = r"""$STATE_FILE"""
repo = r"""$repo_name"""
exit_code = int("$exit_code")
with sqlite3.connect(db_path) as connection:
    connection.execute(
        """
        INSERT INTO build_index_checkpoints(repo, status, attempts, updated_at, last_pid, last_exit_code)
        VALUES(?, 'failed', 1, datetime('now'), NULL, ?)
        ON CONFLICT(repo) DO UPDATE SET
            status='failed',
            updated_at=datetime('now'),
            last_exit_code=excluded.last_exit_code
        """,
        [repo, exit_code],
    )
PY
}

CURRENT_REPO=""
RUN_IN_PROGRESS=0

on_interrupt() {
  local exit_code=$?
  if [[ "$RUN_IN_PROGRESS" -eq 1 && -n "$CURRENT_REPO" ]]; then
    echo "[build-index.sh] interrupted repo=${CURRENT_REPO} exit_code=${exit_code}; checkpointing failure"
    mark_failed "$CURRENT_REPO" "$exit_code"
  fi
  exit "$exit_code"
}
trap on_interrupt INT TERM

init_state_db

mapfile -t DEFAULT_REPOS < <(find "$SCRIPT_DIR/../data/raw" -mindepth 1 -maxdepth 1 -type d ! -name '.*' -printf '%f\n' | sort)
if [[ "${#DEFAULT_REPOS[@]}" -eq 0 ]]; then
  echo "[build-index.sh] no raw inventories found; run parse_repos.sh first or pass --repo" >&2
  exit 1
fi

for repo in "${DEFAULT_REPOS[@]}"; do
  if is_done "$repo"; then
    echo "[build-index.sh] skipping repo=${repo} (already completed in $STATE_FILE)"
    continue
  fi
  CURRENT_REPO="$repo"
  RUN_IN_PROGRESS=1
  mark_running "$repo" "$$"
  if run_single_repo "$repo"; then
    :
  else
    exit_code=$?
    mark_failed "$repo" "$exit_code"
    exit "$exit_code"
  fi
  RUN_IN_PROGRESS=0
  mark_done "$repo"
done

echo "[build-index.sh] all repos complete. resume state: $STATE_FILE"
