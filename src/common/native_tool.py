from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional, Sequence


REPO_ANALYSIS_ROOT = Path(__file__).resolve().parents[2]
NATIVE_CRATE_DIR = REPO_ANALYSIS_ROOT / "native"
NATIVE_MANIFEST_PATH = NATIVE_CRATE_DIR / "Cargo.toml"
NATIVE_FAILURE_MARKER = NATIVE_CRATE_DIR / ".build_failed.json"
_NATIVE_DISABLED_REASON: str | None = None


def native_binary_path() -> Path:
    binary_name = "repo-analysis-native.exe" if os.name == "nt" else "repo-analysis-native"
    return NATIVE_CRATE_DIR / "target" / "debug" / binary_name


def native_worker_available() -> bool:
    return (
        _NATIVE_DISABLED_REASON is None
        and persisted_failure_reason() is None
        and NATIVE_MANIFEST_PATH.exists()
        and shutil.which("cargo") is not None
    )


def ensure_native_binary() -> Optional[Path]:
    global _NATIVE_DISABLED_REASON
    if not native_worker_available():
        return None

    binary_path = native_binary_path()
    manifest_mtime = latest_native_source_mtime()
    if binary_path.exists():
        try:
            if binary_path.stat().st_mtime >= manifest_mtime:
                clear_failure_marker()
                return binary_path
        except OSError:
            pass

    try:
        subprocess.run(
            ["cargo", "build", "--manifest-path", str(NATIVE_MANIFEST_PATH), "--quiet"],
            cwd=NATIVE_CRATE_DIR,
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        _NATIVE_DISABLED_REASON = stderr.splitlines()[-1] if stderr else str(exc)
        write_failure_marker(_NATIVE_DISABLED_REASON)
        return None
    clear_failure_marker()
    return binary_path if binary_path.exists() else None


def latest_native_source_mtime() -> float:
    latest = 0.0
    if NATIVE_MANIFEST_PATH.exists():
        latest = max(latest, NATIVE_MANIFEST_PATH.stat().st_mtime)
    for path in NATIVE_CRATE_DIR.rglob("*.rs"):
        latest = max(latest, path.stat().st_mtime)
    return latest


def run_native_json(
    args: Sequence[str],
    *,
    input_payload: object | None = None,
    timeout: int = 300,
) -> Dict[str, object]:
    binary_path = ensure_native_binary()
    if binary_path is None:
        raise RuntimeError("native worker is unavailable")

    stdin = None
    if input_payload is not None:
        stdin = json.dumps(input_payload)

    result = subprocess.run(
        [str(binary_path), *args],
        cwd=REPO_ANALYSIS_ROOT,
        input=stdin,
        capture_output=True,
        text=True,
        check=True,
        timeout=timeout,
    )
    return json.loads(result.stdout or "{}")


def probe_native_worker() -> Dict[str, object]:
    if not native_worker_available():
        return {
            "available": False,
            "invoked": False,
            "reason": _NATIVE_DISABLED_REASON or persisted_failure_reason() or "cargo or native manifest missing",
        }
    try:
        payload = run_native_json(["worker-info"], timeout=120)
    except Exception as exc:  # pragma: no cover - defensive fallback
        return {
            "available": False,
            "invoked": False,
            "reason": str(exc),
        }
    payload["invoked"] = True
    return payload


def probe_rust_file_native(path: Path) -> Optional[Dict[str, object]]:
    try:
        return run_native_json(["inspect-rust", "--path", str(path)], timeout=120)
    except Exception:
        return None


def build_bm25_index(documents_path: Path, output_dir: Path) -> Dict[str, object]:
    return run_native_json(
        [
            "build-bm25",
            "--documents",
            str(documents_path),
            "--output-dir",
            str(output_dir),
        ],
        timeout=300,
    )


def query_bm25_index(
    index_dir: Path,
    query: str,
    *,
    limit: int = 10,
    kinds: Sequence[str] = (),
    path_prefix: str | None = None,
    symbol_id: str | None = None,
) -> list[Dict[str, object]]:
    args = [
        "query-bm25",
        "--index-dir",
        str(index_dir),
        "--limit",
        str(limit),
    ]
    if query:
        args.extend(["--query", query])
    for kind in kinds:
        args.extend(["--kind", kind])
    if path_prefix:
        args.extend(["--path-prefix", path_prefix])
    if symbol_id:
        args.extend(["--symbol-id", symbol_id])
    payload = run_native_json(args, timeout=120)
    return list(payload.get("results", []))


def list_bm25_docs(
    index_dir: Path,
    *,
    offset: int = 0,
    limit: int = 10_000,
    timeout: int = 300,
) -> Dict[str, object]:
    payload = run_native_json(
        [
            "list-bm25-docs",
            "--index-dir",
            str(index_dir),
            "--offset",
            str(offset),
            "--limit",
            str(limit),
        ],
        timeout=timeout,
    )
    return {
        "results": list(payload.get("results", [])),
        "total_docs": int(payload.get("total_docs") or 0),
        "next_offset": payload.get("next_offset"),
    }


def persisted_failure_reason() -> str | None:
    if not NATIVE_FAILURE_MARKER.exists():
        return None
    try:
        payload = json.loads(NATIVE_FAILURE_MARKER.read_text(encoding="utf-8"))
    except Exception:
        return None
    marker_mtime = float(payload.get("source_mtime") or 0.0)
    if marker_mtime < latest_native_source_mtime():
        return None
    return str(payload.get("reason") or "native build previously failed")


def write_failure_marker(reason: str) -> None:
    payload = {
        "reason": reason,
        "source_mtime": latest_native_source_mtime(),
    }
    NATIVE_FAILURE_MARKER.write_text(json.dumps(payload), encoding="utf-8")


def clear_failure_marker() -> None:
    try:
        NATIVE_FAILURE_MARKER.unlink()
    except FileNotFoundError:
        pass