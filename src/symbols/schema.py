from __future__ import annotations

from typing import Dict, Optional


DEFAULT_RANGE = {
    "start_line": 1,
    "start_column": 1,
    "end_line": 1,
    "end_column": 1,
}


def normalize_symbol_record(
    record: Dict[str, object],
    *,
    provider: str,
    confidence: float,
) -> Dict[str, object]:
    normalized = dict(record)
    normalized["provider"] = str(normalized.get("provider") or provider)
    normalized["language"] = str(normalized.get("language") or "Unknown")
    normalized["kind"] = str(normalized.get("kind") or "symbol")
    normalized["path"] = str(normalized.get("path") or "")
    normalized["name"] = str(normalized.get("name") or normalized.get("qualified_name") or "")
    normalized["qualified_name"] = str(normalized.get("qualified_name") or normalized.get("name") or "")
    normalized["range"] = normalize_range(normalized.get("range") or normalized.get("span"))
    normalized.setdefault("span", dict(normalized["range"]))
    normalized["scope"] = normalize_scope(normalized)
    normalized["confidence"] = normalize_confidence(normalized.get("confidence"), fallback=confidence)
    return normalized


def normalize_range(value: object) -> Dict[str, int]:
    if not isinstance(value, dict):
        return dict(DEFAULT_RANGE)
    return {
        "start_line": normalize_positive_int(value.get("start_line"), DEFAULT_RANGE["start_line"]),
        "start_column": normalize_positive_int(value.get("start_column"), DEFAULT_RANGE["start_column"]),
        "end_line": normalize_positive_int(value.get("end_line"), value.get("start_line") or DEFAULT_RANGE["end_line"]),
        "end_column": normalize_positive_int(
            value.get("end_column"),
            value.get("start_column") or DEFAULT_RANGE["end_column"],
        ),
    }


def normalize_scope(record: Dict[str, object]) -> Dict[str, Optional[str]]:
    existing = record.get("scope")
    if isinstance(existing, dict):
        return {
            "symbol_id": optional_text(existing.get("symbol_id")),
            "qualified_name": optional_text(existing.get("qualified_name")),
            "path": optional_text(existing.get("path") or record.get("path")),
        }
    return {
        "symbol_id": optional_text(record.get("container_symbol_id") or record.get("scope_symbol_id")),
        "qualified_name": optional_text(record.get("container_qualified_name")),
        "path": optional_text(record.get("path")),
    }


def normalize_confidence(value: object, *, fallback: float) -> float:
    try:
        parsed = float(value if value is not None else fallback)
    except (TypeError, ValueError):
        parsed = float(fallback)
    return round(max(0.0, min(parsed, 1.0)), 3)


def normalize_positive_int(value: object, fallback: object) -> int:
    try:
        parsed = int(value if value is not None else fallback)
    except (TypeError, ValueError):
        parsed = int(DEFAULT_RANGE["start_line"])
    return max(parsed, 1)


def optional_text(value: object) -> Optional[str]:
    text = str(value or "")
    return text or None
