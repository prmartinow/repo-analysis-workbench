from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from symbols.schema import normalize_symbol_record


SCIP_DEFINITION_ROLE = 0x1
SCIP_IMPORT_ROLE = 0x2
SCIP_WRITE_ACCESS_ROLE = 0x4
SCIP_READ_ACCESS_ROLE = 0x8
SCIP_FORWARD_DEFINITION_ROLE = 0x40
SCIP_KIND_BY_VALUE = {
    7: "class",
    8: "constant",
    9: "constructor",
    11: "enum",
    12: "enum_member",
    15: "field",
    16: "file",
    17: "function",
    21: "interface",
    25: "macro",
    26: "method",
    28: "message",
    29: "module",
    30: "namespace",
    35: "package",
    37: "parameter",
    41: "property",
    49: "struct",
    53: "trait",
    54: "type",
    55: "type_alias",
    59: "union",
    61: "variable",
    66: "abstract_method",
    67: "method",
    68: "method",
    69: "method",
    70: "method",
    71: "method",
    80: "method",
    81: "property",
    82: "variable",
}
LANGUAGE_NAMES = {
    "bash": "Shell",
    "go": "Go",
    "java": "Java",
    "javascript": "JavaScript",
    "python": "Python",
    "rust": "Rust",
    "shell": "Shell",
    "typescript": "TypeScript",
    "tsx": "TSX",
    "yaml": "YAML",
}


def probe_scip_indexes(
    repo_name: str,
    repo_root: Path,
    repo_map: Dict[str, object],
    *,
    scip_indexes: Sequence[Path] = (),
    path_prefixes: Sequence[str] = (),
    runner=None,
) -> Dict[str, object]:
    started = time.perf_counter()
    explicit = bool(scip_indexes)
    index_paths = resolve_scip_index_paths(repo_root, scip_indexes)
    if not index_paths:
        return {
            "backend": "scip",
            "available": False,
            "used": False,
            "parsed": not explicit,
            "index_files": [],
            "files": 0,
            "symbols": 0,
            "references": 0,
            "relationships": 0,
            "file_records": [],
            "symbol_records": [],
            "reference_records": [],
            "diagnostics": ["no SCIP index files discovered" if not explicit else "explicit SCIP index files were not found"],
            "latency_ms": elapsed_ms(started),
        }

    diagnostics: List[str] = []
    file_records: List[Dict[str, object]] = []
    symbol_records: List[Dict[str, object]] = []
    reference_records: List[Dict[str, object]] = []
    relationship_count = 0
    parsed_indexes = 0
    repo_files_by_path = {str(item.get("path") or ""): item for item in repo_map.get("files", [])}

    for index_path in index_paths:
        payload, load_diagnostics = load_scip_index_payload(index_path, runner=runner)
        diagnostics.extend(load_diagnostics)
        if payload is None:
            continue
        parsed_indexes += 1
        parsed = parse_scip_payload(repo_name, payload, repo_files_by_path, path_prefixes=path_prefixes)
        file_records.extend(parsed["file_records"])
        symbol_records.extend(parsed["symbol_records"])
        reference_records.extend(parsed["reference_records"])
        relationship_count += int(parsed["relationships"])

    symbol_records = unique_records(symbol_records, "symbol_id")
    reference_records = unique_records(reference_records, "reference_id")
    file_records = unique_records(file_records, "path")
    return {
        "backend": "scip",
        "available": parsed_indexes > 0,
        "used": bool(index_paths),
        "parsed": parsed_indexes == len(index_paths),
        "index_files": [path.as_posix() for path in index_paths],
        "files": len(file_records),
        "symbols": len(symbol_records),
        "references": len(reference_records),
        "relationships": relationship_count,
        "file_records": file_records,
        "symbol_records": symbol_records,
        "reference_records": reference_records,
        "diagnostics": sorted(dict.fromkeys(diagnostics)),
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


def resolve_scip_index_paths(repo_root: Path, scip_indexes: Sequence[Path]) -> List[Path]:
    if scip_indexes:
        paths = []
        for path in scip_indexes:
            candidate = path if path.is_absolute() else repo_root / path
            if candidate.exists():
                paths.append(candidate.resolve())
        return sorted(dict.fromkeys(paths))

    candidates = [
        repo_root / "index.scip",
        repo_root / "index.scip.json",
        repo_root / ".scip" / "index.scip",
        repo_root / ".scip" / "index.scip.json",
    ]
    candidates.extend(sorted(repo_root.glob("*.scip")))
    candidates.extend(sorted(repo_root.glob("*.scip.json")))
    candidates.extend(sorted((repo_root / ".scip").glob("*.scip")) if (repo_root / ".scip").exists() else [])
    return sorted(dict.fromkeys(path.resolve() for path in candidates if path.exists()))


def load_scip_index_payload(path: Path, *, runner=None) -> Tuple[Dict[str, object] | None, List[str]]:
    if path.suffix == ".json" or path.name.endswith(".scip.json"):
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            return None, [f"{path}: failed to load SCIP JSON: {exc}"]
        return payload if isinstance(payload, dict) else None, []

    binary = shutil.which("scip")
    if binary is None:
        return None, [f"{path}: scip executable was not found; install scip or pass a .scip.json file"]
    run = runner or subprocess.run
    result = run(
        [binary, "print", "--json", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        return None, [f"{path}: scip print exited with code {result.returncode}: {stderr}"]
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return None, [f"{path}: scip print produced invalid JSON: {exc}"]
    return payload if isinstance(payload, dict) else None, []


def parse_scip_payload(
    repo_name: str,
    payload: Dict[str, object],
    repo_files_by_path: Dict[str, Dict[str, object]],
    *,
    path_prefixes: Sequence[str] = (),
) -> Dict[str, object]:
    documents = [doc for doc in get_list(payload, "documents") if isinstance(doc, dict)]
    symbol_infos: Dict[str, Dict[str, object]] = {}
    symbol_context: Dict[str, Dict[str, object]] = {}
    definitions: Dict[str, Dict[str, object]] = {}
    file_records: Dict[str, Dict[str, object]] = {}

    for doc in documents:
        path = get_text(doc, "relative_path", "relativePath")
        if not path or (path_prefixes and not matches_path_prefix(path, path_prefixes)):
            continue
        language = normalize_language(get_text(doc, "language") or repo_files_by_path.get(path, {}).get("language"))
        occurrences = [occ for occ in get_list(doc, "occurrences") if isinstance(occ, dict)]
        symbol_rows = [info for info in get_list(doc, "symbols") if isinstance(info, dict)]
        file_records[path] = {
            "path": path,
            "crate": None,
            "package_name": None,
            "module_path": None,
            "language": language,
            "symbols": len(symbol_rows),
            "imports": 0,
            "primary_parser_backend": "scip",
            "content_hash": repo_files_by_path.get(path, {}).get("content_hash"),
        }

        for info in symbol_rows:
            symbol = get_text(info, "symbol")
            if not symbol:
                continue
            symbol_infos.setdefault(symbol, info)
            symbol_context.setdefault(symbol, {"path": path, "language": language})

        for occurrence in occurrences:
            symbol = get_text(occurrence, "symbol")
            if not symbol:
                continue
            if is_definition_occurrence(occurrence):
                definitions.setdefault(symbol, {"path": path, "language": language, "occurrence": occurrence})
                symbol_infos.setdefault(symbol, {"symbol": symbol})
                symbol_context.setdefault(symbol, {"path": path, "language": language})

    symbol_records = []
    symbol_id_by_scip: Dict[str, str] = {}
    for symbol, info in sorted(symbol_infos.items()):
        context = definitions.get(symbol) or symbol_context.get(symbol)
        if not context:
            continue
        record = scip_symbol_to_record(repo_name, symbol, info, context)
        symbol_records.append(record)
        symbol_id_by_scip[symbol] = record["symbol_id"]

    definition_ranges_by_path = build_definition_ranges(definitions, symbol_id_by_scip)
    reference_records = []
    for doc in documents:
        path = get_text(doc, "relative_path", "relativePath")
        if not path or (path_prefixes and not matches_path_prefix(path, path_prefixes)):
            continue
        language = normalize_language(get_text(doc, "language") or repo_files_by_path.get(path, {}).get("language"))
        for occurrence in get_list(doc, "occurrences"):
            if not isinstance(occurrence, dict):
                continue
            symbol = get_text(occurrence, "symbol")
            if not symbol or is_definition_occurrence(occurrence):
                continue
            reference_records.append(
                scip_occurrence_to_reference(
                    repo_name,
                    path,
                    language,
                    occurrence,
                    symbol,
                    symbol_infos.get(symbol, {"symbol": symbol}),
                    symbol_id_by_scip,
                    definition_ranges_by_path.get(path, []),
                )
            )

    return {
        "file_records": list(file_records.values()),
        "symbol_records": symbol_records,
        "reference_records": reference_records,
        "relationships": sum(len(get_list(symbol.get("scip", {}), "relationships")) for symbol in symbol_records),
    }


def scip_symbol_to_record(
    repo_name: str,
    scip_symbol: str,
    info: Dict[str, object],
    context: Dict[str, object],
) -> Dict[str, object]:
    occurrence = context.get("occurrence") if isinstance(context.get("occurrence"), dict) else {}
    occurrence_range = scip_occurrence_range(occurrence)
    enclosing_range = scip_occurrence_range(occurrence, enclosing=True) if occurrence else None
    symbol_range = enclosing_range or occurrence_range or default_range()
    selection_range = occurrence_range or symbol_range
    display_name = get_text(info, "display_name", "displayName") or infer_scip_display_name(scip_symbol)
    signature = scip_signature(info) or display_name
    raw_relationships = [item for item in get_list(info, "relationships") if isinstance(item, dict)]
    record = {
        "symbol_id": stable_id("sym", repo_name, "scip", scip_symbol),
        "repo": repo_name,
        "path": str(context.get("path") or ""),
        "language": str(context.get("language") or "Unknown"),
        "kind": normalize_scip_kind(info.get("kind")),
        "name": display_name,
        "qualified_name": scip_symbol,
        "range": symbol_range,
        "selection_range": selection_range,
        "signature": signature,
        "scope": {
            "symbol_id": None,
            "qualified_name": get_text(info, "enclosing_symbol", "enclosingSymbol") or None,
            "path": str(context.get("path") or ""),
        },
        "scip_symbol": scip_symbol,
        "scip": {
            "symbol": scip_symbol,
            "display_name": display_name,
            "documentation": get_list(info, "documentation"),
            "relationships": normalize_scip_relationships(raw_relationships),
        },
    }
    return normalize_symbol_record(record, provider="scip", confidence=0.95)


def scip_occurrence_to_reference(
    repo_name: str,
    path: str,
    language: str,
    occurrence: Dict[str, object],
    scip_symbol: str,
    info: Dict[str, object],
    symbol_id_by_scip: Dict[str, str],
    definition_ranges: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    occurrence_range = scip_occurrence_range(occurrence) or default_range()
    container_symbol_id = find_enclosing_definition_symbol_id(definition_ranges, occurrence_range)
    reference_kind = scip_reference_kind(occurrence)
    target_symbol_id = symbol_id_by_scip.get(scip_symbol)
    target_name = get_text(info, "display_name", "displayName") or infer_scip_display_name(scip_symbol)
    return {
        "reference_id": stable_id(
            "ref",
            repo_name,
            "scip",
            path,
            scip_symbol,
            reference_kind,
            occurrence_range["start_line"],
            occurrence_range["start_column"],
        ),
        "repo": repo_name,
        "path": path,
        "crate": None,
        "module_path": None,
        "language": language,
        "kind": reference_kind,
        "name": target_name,
        "qualified_name_hint": scip_symbol,
        "span": occurrence_range,
        "container_symbol_id": container_symbol_id,
        "container_qualified_name": None,
        "scope_symbol_id": container_symbol_id,
        "target_symbol_id": target_symbol_id,
        "target_qualified_name": scip_symbol,
        "target_kind": normalize_scip_kind(info.get("kind")),
        "provider": "scip",
        "scip": {
            "symbol": scip_symbol,
            "symbol_roles": int(get_value(occurrence, "symbol_roles", "symbolRoles") or 0),
        },
    }


def build_definition_ranges(
    definitions: Dict[str, Dict[str, object]],
    symbol_id_by_scip: Dict[str, str],
) -> Dict[str, List[Dict[str, object]]]:
    by_path: Dict[str, List[Dict[str, object]]] = {}
    for scip_symbol, context in definitions.items():
        occurrence = context.get("occurrence")
        if not isinstance(occurrence, dict):
            continue
        symbol_id = symbol_id_by_scip.get(scip_symbol)
        if not symbol_id:
            continue
        path = str(context.get("path") or "")
        occurrence_range = scip_occurrence_range(occurrence) or default_range()
        enclosing_range = scip_occurrence_range(occurrence, enclosing=True) or occurrence_range
        by_path.setdefault(path, []).append(
            {
                "symbol_id": symbol_id,
                "range": enclosing_range,
                "selection_range": occurrence_range,
            }
        )
    for ranges in by_path.values():
        ranges.sort(key=lambda item: range_size(item["range"]))
    return by_path


def scip_occurrence_range(occurrence: Dict[str, object], *, enclosing: bool = False) -> Dict[str, int] | None:
    if not occurrence:
        return None
    if enclosing:
        single = get_value(occurrence, "single_line_enclosing_range", "singleLineEnclosingRange")
        multi = get_value(occurrence, "multi_line_enclosing_range", "multiLineEnclosingRange")
        legacy = get_value(occurrence, "enclosing_range", "enclosingRange")
    else:
        single = get_value(occurrence, "single_line_range", "singleLineRange")
        multi = get_value(occurrence, "multi_line_range", "multiLineRange")
        legacy = get_value(occurrence, "range")

    if isinstance(single, dict):
        line = positive_int(single.get("line"), 0)
        start = positive_int(get_value(single, "start_character", "startCharacter"), 0)
        end = positive_int(get_value(single, "end_character", "endCharacter"), start)
        return to_one_based_range(line, start, line, end)
    if isinstance(multi, dict):
        start_line = positive_int(get_value(multi, "start_line", "startLine"), 0)
        start = positive_int(get_value(multi, "start_character", "startCharacter"), 0)
        end_line = positive_int(get_value(multi, "end_line", "endLine"), start_line)
        end = positive_int(get_value(multi, "end_character", "endCharacter"), start)
        return to_one_based_range(start_line, start, end_line, end)
    if isinstance(legacy, list):
        values = [positive_int(value, 0) for value in legacy]
        if len(values) == 3:
            return to_one_based_range(values[0], values[1], values[0], values[2])
        if len(values) >= 4:
            return to_one_based_range(values[0], values[1], values[2], values[3])
    return None


def to_one_based_range(start_line: int, start_column: int, end_line: int, end_column: int) -> Dict[str, int]:
    return {
        "start_line": start_line + 1,
        "start_column": start_column + 1,
        "end_line": end_line + 1,
        "end_column": end_column + 1,
    }


def find_enclosing_definition_symbol_id(
    definition_ranges: Sequence[Dict[str, object]],
    occurrence_range: Dict[str, int],
) -> str | None:
    for item in definition_ranges:
        symbol_range = item.get("range")
        if isinstance(symbol_range, dict) and range_contains(symbol_range, occurrence_range):
            return str(item["symbol_id"])
    return None


def range_contains(outer: Dict[str, int], inner: Dict[str, int]) -> bool:
    outer_start = (int(outer["start_line"]), int(outer["start_column"]))
    outer_end = (int(outer["end_line"]), int(outer["end_column"]))
    inner_start = (int(inner["start_line"]), int(inner["start_column"]))
    inner_end = (int(inner["end_line"]), int(inner["end_column"]))
    return outer_start <= inner_start and inner_end <= outer_end


def range_size(value: Dict[str, int]) -> Tuple[int, int]:
    return (
        max(int(value["end_line"]) - int(value["start_line"]), 0),
        max(int(value["end_column"]) - int(value["start_column"]), 0),
    )


def normalize_scip_relationships(rows: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    relationships = []
    for row in rows:
        symbol = get_text(row, "symbol")
        if not symbol:
            continue
        relationships.append(
            {
                "symbol": symbol,
                "is_reference": bool(get_value(row, "is_reference", "isReference")),
                "is_implementation": bool(get_value(row, "is_implementation", "isImplementation")),
                "is_type_definition": bool(get_value(row, "is_type_definition", "isTypeDefinition")),
                "is_definition": bool(get_value(row, "is_definition", "isDefinition")),
            }
        )
    return relationships


def scip_signature(info: Dict[str, object]) -> str:
    signature = get_value(info, "signature_documentation", "signatureDocumentation")
    if isinstance(signature, dict):
        text = get_text(signature, "text")
        if text:
            return text
    docs = get_list(info, "documentation")
    return str(docs[0]) if docs else ""


def normalize_scip_kind(value: object) -> str:
    if isinstance(value, int):
        return SCIP_KIND_BY_VALUE.get(value, "symbol")
    text = str(value or "").strip()
    if not text:
        return "symbol"
    return text.replace("Kind", "").replace("_", " ").replace("-", " ").strip().lower().replace(" ", "_")


def scip_reference_kind(occurrence: Dict[str, object]) -> str:
    roles = int(get_value(occurrence, "symbol_roles", "symbolRoles") or 0)
    if roles & SCIP_IMPORT_ROLE:
        return "import"
    if roles & SCIP_WRITE_ACCESS_ROLE:
        return "write"
    if roles & SCIP_READ_ACCESS_ROLE:
        return "read"
    return "use"


def is_definition_occurrence(occurrence: Dict[str, object]) -> bool:
    roles = int(get_value(occurrence, "symbol_roles", "symbolRoles") or 0)
    return bool(roles & (SCIP_DEFINITION_ROLE | SCIP_FORWARD_DEFINITION_ROLE))


def infer_scip_display_name(scip_symbol: str) -> str:
    token = scip_symbol.rsplit(" ", 1)[-1]
    matches = [match.group(0) for match in re.finditer(r"[A-Za-z_$][A-Za-z0-9_$]*", token)]
    return matches[-1] if matches else scip_symbol


def normalize_language(value: object) -> str:
    text = str(value or "Unknown")
    return LANGUAGE_NAMES.get(text.lower(), text)


def get_value(payload: Dict[str, object], *keys: str) -> object:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def get_text(payload: Dict[str, object], *keys: str) -> str:
    value = get_value(payload, *keys)
    return str(value or "")


def get_list(payload: Dict[str, object], *keys: str) -> List[object]:
    value = get_value(payload, *keys)
    return value if isinstance(value, list) else []


def default_range() -> Dict[str, int]:
    return {"start_line": 1, "start_column": 1, "end_line": 1, "end_column": 1}


def positive_int(value: object, fallback: int) -> int:
    try:
        return max(int(value if value is not None else fallback), 0)
    except (TypeError, ValueError):
        return max(int(fallback), 0)


def matches_path_prefix(relative_path: str, path_prefixes: Sequence[str]) -> bool:
    return any(relative_path == prefix or relative_path.startswith(f"{prefix.rstrip('/')}/") for prefix in path_prefixes)


def unique_records(records: Iterable[Dict[str, object]], key: str) -> List[Dict[str, object]]:
    seen = set()
    unique = []
    for record in records:
        value = str(record.get(key) or "")
        if not value or value in seen:
            continue
        seen.add(value)
        unique.append(record)
    return unique


def stable_id(prefix: str, *parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


def elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)
