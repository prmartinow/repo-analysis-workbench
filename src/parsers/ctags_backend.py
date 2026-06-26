from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable, Dict, List, Sequence

from common.inventory import is_generated_path
from symbols.schema import normalize_symbol_record


CTAGS_CODE_EXTENSIONS = {
    ".bash",
    ".c",
    ".cc",
    ".cpp",
    ".go",
    ".h",
    ".hpp",
    ".java",
    ".js",
    ".jsx",
    ".mjs",
    ".py",
    ".sh",
    ".ts",
    ".tsx",
    ".zsh",
}
CTAGS_EXCLUDED_PATH_PARTS = {
    ".next",
    "coverage",
    "monaco-editor",
    "public",
    "static",
    "vendor",
}
CTAGS_LARGE_BUNDLE_EXTENSIONS = {".js", ".mjs"}
CTAGS_LARGE_BUNDLE_BYTES = 1_000_000


def probe_universal_ctags(
    repo_name: str,
    repo_root: Path,
    repo_map: Dict[str, object],
    *,
    path_prefixes: Sequence[str] = (),
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> Dict[str, object]:
    started = time.perf_counter()
    binary = shutil.which("ctags")
    if binary is None:
        return {
            "backend": "universal_ctags",
            "available": False,
            "used": False,
            "parsed": False,
            "files": 0,
            "symbols": 0,
            "symbol_records": [],
            "diagnostics": ["ctags executable was not found"],
            "latency_ms": elapsed_ms(started),
        }

    files = discover_ctags_files(repo_map, path_prefixes=path_prefixes)
    if not files:
        return {
            "backend": "universal_ctags",
            "available": True,
            "used": False,
            "parsed": True,
            "files": 0,
            "symbols": 0,
            "symbol_records": [],
            "diagnostics": ["no ctags-supported files matched the inventory"],
            "latency_ms": elapsed_ms(started),
        }

    run = runner or subprocess.run
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=True) as list_file:
        for relative_path in files:
            list_file.write(f"{relative_path}\n")
        list_file.flush()
        args = [
            binary,
            "--output-format=json",
            "--fields=+nelK",
            "--extras=+q",
            "--sort=no",
            "--tag-relative=always",
            "-f",
            "-",
            "-L",
            list_file.name,
        ]
        result = run(
            args,
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )

    diagnostics: List[str] = []
    if result.stderr:
        diagnostics.extend(line for line in result.stderr.splitlines() if line.strip())
    if result.returncode != 0:
        return {
            "backend": "universal_ctags",
            "available": True,
            "used": True,
            "parsed": False,
            "files": len(files),
            "symbols": 0,
            "symbol_records": [],
            "diagnostics": diagnostics or [f"ctags exited with code {result.returncode}"],
            "latency_ms": elapsed_ms(started),
        }

    symbol_records, parse_errors = parse_ctags_json_lines(repo_name, result.stdout)
    diagnostics.extend(parse_errors)
    return {
        "backend": "universal_ctags",
        "available": True,
        "used": True,
        "parsed": not parse_errors,
        "files": len(files),
        "symbols": len(symbol_records),
        "symbol_records": symbol_records,
        "diagnostics": diagnostics,
        "latency_ms": elapsed_ms(started),
        "samples": [
            {
                "path": symbol["path"],
                "name": symbol["name"],
                "kind": symbol["kind"],
                "language": symbol["language"],
            }
            for symbol in symbol_records[:10]
        ],
    }


def discover_ctags_files(repo_map: Dict[str, object], *, path_prefixes: Sequence[str] = ()) -> List[str]:
    files = []
    for item in repo_map.get("files", []):
        path = str(item.get("path") or "")
        if not path:
            continue
        if should_skip_ctags_file(path, item):
            continue
        suffix = Path(path).suffix.lower()
        if suffix not in CTAGS_CODE_EXTENSIONS:
            continue
        if path.endswith(".rs"):
            continue
        if path_prefixes and not matches_path_prefix(path, path_prefixes):
            continue
        files.append(path)
    return sorted(dict.fromkeys(files))


def should_skip_ctags_file(path: str, item: Dict[str, object]) -> bool:
    if bool(item.get("generated")) or is_generated_path(path):
        return True
    parts = set(Path(path).parts)
    if parts & CTAGS_EXCLUDED_PATH_PARTS:
        return True
    suffix = Path(path).suffix.lower()
    if suffix in CTAGS_LARGE_BUNDLE_EXTENSIONS:
        try:
            size = int(item.get("size") or 0)
        except (TypeError, ValueError):
            size = 0
        if size >= CTAGS_LARGE_BUNDLE_BYTES:
            return True
    return False


def parse_ctags_json_lines(repo_name: str, output: str) -> tuple[List[Dict[str, object]], List[str]]:
    symbols = []
    errors = []
    seen_ids = set()
    for line_number, line in enumerate(output.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_number}: invalid ctags JSON: {exc}")
            continue
        if row.get("_type") != "tag":
            continue
        symbol = ctags_row_to_symbol(repo_name, row)
        if symbol["symbol_id"] in seen_ids:
            continue
        seen_ids.add(symbol["symbol_id"])
        symbols.append(symbol)
    return symbols, errors


def ctags_row_to_symbol(repo_name: str, row: Dict[str, object]) -> Dict[str, object]:
    path = str(row.get("path") or row.get("input") or "")
    name = str(row.get("name") or "")
    kind = str(row.get("kind") or row.get("kindName") or "symbol")
    language = str(row.get("language") or "Unknown")
    line = positive_int(row.get("line"), 1)
    end_line = positive_int(row.get("end"), line)
    scope_name = str(row.get("scope") or "")
    qualified_name = f"{scope_name}::{name}" if scope_name else name
    symbol_id = stable_id("sym", repo_name, "ctags", path, kind, qualified_name, str(line), str(end_line))
    record = {
        "symbol_id": symbol_id,
        "repo": repo_name,
        "path": path,
        "language": language,
        "kind": kind,
        "name": name,
        "qualified_name": qualified_name,
        "range": {
            "start_line": line,
            "start_column": 1,
            "end_line": end_line,
            "end_column": 1,
        },
        "signature": clean_ctags_pattern(str(row.get("pattern") or "")),
        "scope": {
            "symbol_id": None,
            "qualified_name": scope_name or None,
            "path": path,
        },
        "ctags": {
            key: row.get(key)
            for key in ("scope", "scopeKind", "kind", "language", "pattern")
            if row.get(key) is not None
        },
    }
    return normalize_symbol_record(record, provider="ctags", confidence=0.65)


def clean_ctags_pattern(value: str) -> str:
    text = value.strip()
    if text.startswith("/^") and text.endswith("$/;\""):
        text = text[2:-4]
    elif text.startswith("/^") and text.endswith("$/"):
        text = text[2:-2]
    return text.replace("\\/", "/").strip()


def stable_id(prefix: str, *parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


def matches_path_prefix(relative_path: str, path_prefixes: Sequence[str]) -> bool:
    return any(relative_path == prefix or relative_path.startswith(f"{prefix.rstrip('/')}/") for prefix in path_prefixes)


def positive_int(value: object, fallback: int) -> int:
    try:
        return max(int(value if value is not None else fallback), 1)
    except (TypeError, ValueError):
        return max(int(fallback), 1)


def elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)
