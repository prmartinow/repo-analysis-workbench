from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from functools import lru_cache
from pathlib import Path
from queue import Queue
from typing import Dict, Iterable, List, Optional, Sequence


SYMBOL_KIND_MAP = {
    2: "module",
    5: "class",
    6: "method",
    10: "enum",
    11: "interface",
    12: "function",
    13: "variable",
    14: "constant",
    19: "constant",
    22: "struct",
    23: "event",
    24: "operator",
    25: "type_parameter",
    26: "trait",
}


@lru_cache(maxsize=1)
def rust_analyzer_available() -> bool:
    command = rust_analyzer_command()
    if not command:
        return False
    try:
        result = subprocess.run(
            [command, "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False
    return bool(result.stdout.strip())


def probe_rust_analyzer(path: Path, source: str, workspace_root: Path) -> Dict[str, object]:
    started = time.perf_counter()
    command = rust_analyzer_command()
    if not command:
        return unavailable_payload(path, started, ["rust-analyzer not configured"])
    if not rust_analyzer_available():
        return unavailable_payload(path, started, ["rust-analyzer is not available in the active toolchain"])

    try:
        result = document_symbol_probe(command, path, source, workspace_root)
    except Exception as exc:  # pragma: no cover - optional backend
        return {
            "backend": "rust-analyzer-lsp",
            "available": True,
            "used": True,
            "parsed": False,
            "path": path.as_posix(),
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "item_counts": [],
            "statement_counts": [],
            "control_counts": [],
            "diagnostics": [f"rust-analyzer probe failed: {exc}"],
        }

    result["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
    return result


def aggregate_rust_analyzer_probes(file_probes: Sequence[Dict[str, object]]) -> Dict[str, object]:
    parsed_files = sum(1 for probe in file_probes if probe.get("parsed"))
    available = any(probe.get("available") for probe in file_probes) if file_probes else rust_analyzer_available()
    return {
        "backend": "rust-analyzer-lsp",
        "available": available,
        "used": bool(file_probes),
        "files": len(file_probes),
        "parsed_files": parsed_files,
        "item_counts": aggregate_counts(file_probes, "item_counts"),
        "statement_counts": [],
        "control_counts": [],
        "document_symbols": sum(len(probe.get("document_symbols", [])) for probe in file_probes),
        "samples": [
            {
                "path": probe["path"],
                "parsed": probe["parsed"],
                "latency_ms": probe["latency_ms"],
            }
            for probe in file_probes[:10]
        ],
    }


def rust_analyzer_command() -> str | None:
    command = os.environ.get("RUST_ANALYZER_BIN", "rust-analyzer").strip()
    return command or None


def unavailable_payload(path: Path, started: float, diagnostics: List[str]) -> Dict[str, object]:
    return {
        "backend": "rust-analyzer-lsp",
        "available": False,
        "used": False,
        "parsed": False,
        "path": path.as_posix(),
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "item_counts": [],
        "statement_counts": [],
        "control_counts": [],
        "document_symbols": [],
        "diagnostics": diagnostics,
    }


def document_symbol_probe(command: str, path: Path, source: str, workspace_root: Path) -> Dict[str, object]:
    with subprocess.Popen(
        [command],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
    ) as process:
        assert process.stdin is not None
        assert process.stdout is not None
        queue: Queue[Dict[str, object]] = Queue()
        stderr_queue: Queue[str] = Queue()
        reader = threading.Thread(target=read_lsp_messages, args=(process.stdout, queue), daemon=True)
        stderr_reader = threading.Thread(target=read_stderr, args=(process.stderr, stderr_queue), daemon=True)
        reader.start()
        stderr_reader.start()

        request_id = 1
        send_lsp_message(
            process.stdin,
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "initialize",
                "params": {
                    "processId": None,
                    "rootUri": workspace_root.as_uri(),
                    "capabilities": {},
                    "workspaceFolders": [{"uri": workspace_root.as_uri(), "name": workspace_root.name}],
                },
            },
        )
        wait_for_response(queue, request_id, timeout=10.0)
        send_lsp_message(process.stdin, {"jsonrpc": "2.0", "method": "initialized", "params": {}})

        send_lsp_message(
            process.stdin,
            {
                "jsonrpc": "2.0",
                "method": "textDocument/didOpen",
                "params": {
                    "textDocument": {
                        "uri": path.as_uri(),
                        "languageId": "rust",
                        "version": 1,
                        "text": source,
                    }
                },
            },
        )

        request_id += 1
        send_lsp_message(
            process.stdin,
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "textDocument/documentSymbol",
                "params": {"textDocument": {"uri": path.as_uri()}},
            },
        )
        response = wait_for_response(queue, request_id, timeout=10.0)

        request_id += 1
        send_lsp_message(process.stdin, {"jsonrpc": "2.0", "id": request_id, "method": "shutdown", "params": None})
        try:
            wait_for_response(queue, request_id, timeout=5.0)
        finally:
            send_lsp_message(process.stdin, {"jsonrpc": "2.0", "method": "exit", "params": None})
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:  # pragma: no cover - defensive cleanup
                process.kill()

    diagnostics = drain_queue(stderr_queue)
    symbols = normalize_document_symbols(response.get("result") or [])
    return {
        "backend": "rust-analyzer-lsp",
        "available": True,
        "used": True,
        "parsed": True,
        "path": path.as_posix(),
        "latency_ms": 0.0,
        "item_counts": summarize_symbol_counts(symbols),
        "statement_counts": [],
        "control_counts": [],
        "document_symbols": symbols,
        "diagnostics": diagnostics[:20],
    }


def send_lsp_message(handle: object, payload: Dict[str, object]) -> None:
    body = json.dumps(payload).encode("utf-8")
    handle.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii"))
    handle.write(body)
    handle.flush()


def read_lsp_messages(handle: object, queue: Queue[Dict[str, object]]) -> None:
    while True:
        headers: Dict[str, str] = {}
        while True:
            line = handle.readline()
            if not line:
                return
            if line == b"\r\n":
                break
            key, value = line.decode("utf-8").split(":", 1)
            headers[key.strip().lower()] = value.strip()
        length = int(headers.get("content-length", "0"))
        if length <= 0:
            continue
        body = handle.read(length)
        if not body:
            return
        queue.put(json.loads(body.decode("utf-8")))


def read_stderr(handle: object, queue: Queue[str]) -> None:
    if handle is None:
        return
    for raw_line in iter(handle.readline, b""):
        line = raw_line.decode("utf-8", errors="replace").strip()
        if line:
            queue.put(line)


def wait_for_response(queue: Queue[Dict[str, object]], request_id: int, timeout: float) -> Dict[str, object]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        remaining = max(deadline - time.monotonic(), 0.1)
        payload = queue.get(timeout=remaining)
        if payload.get("id") == request_id:
            return payload
    raise TimeoutError(f"Timed out waiting for rust-analyzer response {request_id}")


def drain_queue(queue: Queue[str]) -> List[str]:
    values: List[str] = []
    while not queue.empty():
        values.append(queue.get_nowait())
    return values


def normalize_document_symbols(
    items: Iterable[Dict[str, object]],
    *,
    parent_qualified_name: Optional[str] = None,
) -> List[Dict[str, object]]:
    normalized: List[Dict[str, object]] = []
    for item in items:
        name = str(item.get("name") or "")
        kind = normalize_symbol_kind(item.get("kind"))
        if not name or not kind:
            normalized.extend(
                normalize_document_symbols(
                    item.get("children", []),
                    parent_qualified_name=parent_qualified_name,
                )
            )
            continue

        qualified_name = f"{parent_qualified_name}::{name}" if parent_qualified_name else name
        selection_range = item.get("selectionRange") or item.get("range") or {}
        full_range = item.get("range") or selection_range

        normalized.append(
            {
                "name": name,
                "kind": kind,
                "qualified_name": qualified_name,
                "range": normalize_range(full_range),
                "selection_range": normalize_range(selection_range),
                "container_qualified_name": parent_qualified_name,
            }
        )
        normalized.extend(
            normalize_document_symbols(
                item.get("children", []),
                parent_qualified_name=qualified_name,
            )
        )
    return normalized


def normalize_symbol_kind(kind_value: object) -> Optional[str]:
    try:
        kind = int(kind_value or 0)
    except (TypeError, ValueError):
        return None

    mapped = SYMBOL_KIND_MAP.get(kind)
    if mapped == "class":
        return "struct"
    if mapped == "interface":
        return "trait"
    if mapped == "event":
        return "struct"
    if mapped == "constant":
        return "const"
    if mapped == "variable":
        return "local"
    return mapped


def normalize_range(value: object) -> Dict[str, int]:
    if not isinstance(value, dict):
        return {
            "start_line": 1,
            "start_column": 1,
            "end_line": 1,
            "end_column": 1,
        }

    start = value.get("start", {})
    end = value.get("end", {})
    return {
        "start_line": int(start.get("line", 0)) + 1,
        "start_column": int(start.get("character", 0)) + 1,
        "end_line": int(end.get("line", 0)) + 1,
        "end_column": int(end.get("character", 0)) + 1,
    }


def summarize_symbol_counts(items: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    counts: Dict[str, int] = {}
    for item in items:
        kind = str(item.get("kind") or "unknown")
        counts[kind] = counts.get(kind, 0) + 1
    return [{"kind": kind, "count": count} for kind, count in sorted(counts.items())]


def aggregate_counts(file_probes: Sequence[Dict[str, object]], field: str) -> List[Dict[str, object]]:
    counts: Dict[str, int] = {}
    for probe in file_probes:
        for item in probe.get(field, []):
            kind = str(item["kind"])
            counts[kind] = counts.get(kind, 0) + int(item["count"])
    return [{"kind": kind, "count": count} for kind, count in sorted(counts.items())]
