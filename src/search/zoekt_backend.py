from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List


def build_zoekt_index(
    repo_name: str,
    repo_root: Path,
    zoekt_root: Path,
    *,
    zoekt_index_bin: str | None = None,
    runner=None,
) -> Dict[str, object]:
    started = time.perf_counter()
    binary = resolve_binary("zoekt-index", override=zoekt_index_bin, env_name="REPO_ANALYSIS_ZOEKT_INDEX_BIN")
    index_dir = zoekt_root / repo_name
    if binary is None:
        return {
            "repo": repo_name,
            "backend": "zoekt",
            "available": False,
            "built": False,
            "index_dir": index_dir.as_posix(),
            "diagnostics": ["zoekt-index executable was not found"],
            "elapsed_ms": elapsed_ms(started),
        }

    index_dir.mkdir(parents=True, exist_ok=True)
    run = runner or subprocess.run
    args = [binary, "-index", str(index_dir), str(repo_root)]
    result = run(args, check=False, capture_output=True, text=True, timeout=600)
    diagnostics = stderr_lines(result)
    if result.returncode != 0:
        return {
            "repo": repo_name,
            "backend": "zoekt",
            "available": True,
            "built": False,
            "index_dir": index_dir.as_posix(),
            "command": args,
            "diagnostics": diagnostics or [f"zoekt-index exited with code {result.returncode}"],
            "elapsed_ms": elapsed_ms(started),
        }

    metadata = {
        "repo": repo_name,
        "backend": "zoekt",
        "index_dir": index_dir.as_posix(),
        "repo_root": repo_root.as_posix(),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (index_dir / "repo-analysis-zoekt.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "repo": repo_name,
        "backend": "zoekt",
        "available": True,
        "built": True,
        "index_dir": index_dir.as_posix(),
        "command": args,
        "diagnostics": diagnostics,
        "elapsed_ms": elapsed_ms(started),
    }


def search_zoekt_index(
    repo_name: str,
    zoekt_root: Path,
    query: str,
    *,
    limit: int = 10,
    symbol: bool = False,
    zoekt_bin: str | None = None,
    runner=None,
) -> Dict[str, object]:
    started = time.perf_counter()
    binary = resolve_binary("zoekt", override=zoekt_bin, env_name="REPO_ANALYSIS_ZOEKT_BIN")
    index_dir = zoekt_root / repo_name
    if binary is None:
        return {
            "repo": repo_name,
            "query": query,
            "backend": "zoekt",
            "available": False,
            "results": [],
            "diagnostics": ["zoekt executable was not found"],
            "elapsed_ms": elapsed_ms(started),
        }
    if not index_dir.exists():
        return {
            "repo": repo_name,
            "query": query,
            "backend": "zoekt",
            "available": True,
            "results": [],
            "diagnostics": [f"Zoekt index directory does not exist: {index_dir}"],
            "elapsed_ms": elapsed_ms(started),
        }

    args = [binary, "-index_dir", str(index_dir), "-jsonl"]
    if symbol:
        args.append("-sym")
    args.append(query)
    run = runner or subprocess.run
    result = run(args, check=False, capture_output=True, text=True, timeout=120)
    diagnostics = stderr_lines(result)
    if result.returncode != 0:
        return {
            "repo": repo_name,
            "query": query,
            "backend": "zoekt",
            "available": True,
            "results": [],
            "command": args,
            "diagnostics": diagnostics or [f"zoekt exited with code {result.returncode}"],
            "elapsed_ms": elapsed_ms(started),
        }

    results, parse_errors = parse_zoekt_jsonl(result.stdout, limit=limit)
    diagnostics.extend(parse_errors)
    return {
        "repo": repo_name,
        "query": query,
        "backend": "zoekt",
        "available": True,
        "results": results,
        "command": args,
        "diagnostics": diagnostics,
        "elapsed_ms": elapsed_ms(started),
    }


def parse_zoekt_jsonl(output: str, *, limit: int = 10) -> tuple[List[Dict[str, object]], List[str]]:
    results = []
    diagnostics = []
    for line_number, line in enumerate(output.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            diagnostics.append(f"line {line_number}: invalid Zoekt JSONL: {exc}")
            continue
        if not isinstance(row, dict):
            continue
        results.append(normalize_zoekt_file_match(row))
        if len(results) >= limit:
            break
    return results, diagnostics


def normalize_zoekt_file_match(row: Dict[str, object]) -> Dict[str, object]:
    line_matches = []
    for match in row.get("LineMatches") or []:
        if not isinstance(match, dict):
            continue
        line_matches.append(
            {
                "line_number": int(match.get("LineNumber") or 0),
                "preview": decode_zoekt_bytes(match.get("Line")),
                "score": float(match.get("Score") or 0.0),
                "file_name": bool(match.get("FileName")),
            }
        )
    return {
        "path": str(row.get("FileName") or ""),
        "repository": str(row.get("Repository") or ""),
        "language": str(row.get("Language") or ""),
        "score": float(row.get("Score") or 0.0),
        "line_matches": line_matches,
    }


def decode_zoekt_bytes(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        try:
            return bytes(int(item) for item in value).decode("utf-8", errors="replace")
        except (TypeError, ValueError):
            return ""
    text = str(value)
    try:
        return base64.b64decode(text).decode("utf-8", errors="replace")
    except Exception:
        return text


def resolve_binary(name: str, *, override: str | None, env_name: str) -> str | None:
    if override:
        return override
    env_value = os.environ.get(env_name)
    if env_value:
        return env_value
    return shutil.which(name)


def stderr_lines(result: subprocess.CompletedProcess[str]) -> List[str]:
    return [line for line in str(result.stderr or "").splitlines() if line.strip()]


def elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)
