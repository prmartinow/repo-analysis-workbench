from __future__ import annotations

import importlib
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from common.native_tool import probe_rust_file_native

ITEM_NODE_TYPES = {
    "const": {"const_item"},
    "enum": {"enum_item"},
    "function": {"function_item"},
    "impl": {"impl_item"},
    "module": {"mod_item"},
    "static": {"static_item"},
    "struct": {"struct_item"},
    "trait": {"trait_item"},
    "type": {"type_item", "type_alias"},
    "union": {"union_item"},
    "use": {"use_declaration"},
}

STATEMENT_NODE_TYPES = {
    "expr": {"expression_statement"},
    "let": {"let_declaration"},
    "return": {"return_expression"},
}

CONTROL_NODE_TYPES = {
    "for": {"for_expression"},
    "if": {"if_expression"},
    "loop": {"loop_expression"},
    "match": {"match_expression"},
    "while": {"while_expression"},
}

SYMBOL_NODE_TYPES = {
    "const_item": "const",
    "enum_item": "enum",
    "function_item": "function",
    "mod_item": "module",
    "static_item": "static",
    "struct_item": "struct",
    "trait_item": "trait",
    "type_item": "type",
    "type_alias": "type",
}
NAME_NODE_TYPES = {"identifier", "field_identifier", "type_identifier"}
GENERIC_RE = re.compile(r"<[^<>]*>")
IMPL_HEAD_RE = re.compile(
    r"^\s*impl(?:\s*<[^>]+>\s*)?(?:(?P<trait>[A-Za-z_][A-Za-z0-9_:<>]*)\s+for\s+)?(?P<target>[A-Za-z_][A-Za-z0-9_:<>]*)"
)


def probe_tree_sitter(path: Path, source: str) -> Dict[str, object]:
    started = time.perf_counter()
    native_probe = probe_rust_file_native(path)
    if native_probe:
        native_probe["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
        return native_probe

    parser, diagnostics = load_rust_parser()
    if parser is None:
        return unavailable_payload(path, diagnostics, started)

    try:
        tree = parser.parse(source.encode("utf-8"))
    except Exception as exc:  # pragma: no cover - depends on optional backend
        return {
            "backend": "tree-sitter-rust",
            "available": True,
            "used": True,
            "parsed": False,
            "path": path.as_posix(),
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "item_counts": [],
            "statement_counts": [],
            "control_counts": [],
            "error_nodes": 0,
            "diagnostics": [f"tree-sitter parse failed: {exc}"],
        }

    root = tree.root_node
    node_types = list(iter_node_types(root))
    source_bytes = source.encode("utf-8")
    symbols = extract_symbols(root, source_bytes)
    return {
        "backend": "tree-sitter-rust",
        "available": True,
        "used": True,
        "parsed": not root.has_error,
        "path": path.as_posix(),
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "item_counts": summarize_counts(node_types, ITEM_NODE_TYPES),
        "statement_counts": summarize_counts(node_types, STATEMENT_NODE_TYPES),
        "control_counts": summarize_counts(node_types, CONTROL_NODE_TYPES),
        "symbols": symbols,
        "error_nodes": sum(1 for node_type in node_types if node_type == "ERROR"),
        "diagnostics": diagnostics,
    }


def aggregate_tree_sitter_probes(file_probes: Iterable[Dict[str, object]]) -> Dict[str, object]:
    probes = list(file_probes)
    available = any(probe.get("available") for probe in probes) or bool(load_rust_parser()[0])
    parsed_files = sum(1 for probe in probes if probe.get("parsed"))
    return {
        "backend": "tree-sitter-rust",
        "available": available,
        "used": bool(probes),
        "files": len(probes),
        "parsed_files": parsed_files,
        "item_counts": aggregate_counts(probes, "item_counts"),
        "statement_counts": aggregate_counts(probes, "statement_counts"),
        "control_counts": aggregate_counts(probes, "control_counts"),
        "symbols": sum(len(probe.get("symbols", [])) for probe in probes),
        "error_nodes": sum(int(probe.get("error_nodes") or 0) for probe in probes),
        "samples": [
            {
                "path": probe["path"],
                "parsed": probe["parsed"],
                "latency_ms": probe["latency_ms"],
            }
            for probe in probes[:10]
        ],
    }


@lru_cache(maxsize=1)
def load_rust_parser() -> Tuple[object | None, Tuple[str, ...]]:
    diagnostics: List[str] = []

    try:
        module = importlib.import_module("tree_sitter_languages")
        parser = module.get_parser("rust")
        return parser, ("loaded parser from tree_sitter_languages",)
    except Exception as exc:  # pragma: no cover - optional import path
        diagnostics.append(f"tree_sitter_languages unavailable: {exc}")

    try:
        tree_sitter = importlib.import_module("tree_sitter")
        parser = tree_sitter.Parser()
    except Exception as exc:  # pragma: no cover - optional import path
        diagnostics.append(f"tree_sitter unavailable: {exc}")
        return None, tuple(diagnostics)

    language = load_tree_sitter_rust_language(diagnostics)
    if language is None:
        return None, diagnostics

    try:
        if hasattr(parser, "set_language"):
            parser.set_language(language)
        else:  # tree-sitter >= 0.22
            parser.language = language
    except Exception as exc:  # pragma: no cover - optional backend
        diagnostics.append(f"failed to configure tree-sitter parser: {exc}")
        return None, tuple(diagnostics)
    return parser, tuple(diagnostics)


def load_tree_sitter_rust_language(diagnostics: List[str]) -> object | None:
    for module_name in ("tree_sitter_rust", "tree_sitter_languages"):
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - optional import path
            diagnostics.append(f"{module_name} unavailable: {exc}")
            continue

        try:
            if module_name == "tree_sitter_rust" and hasattr(module, "language"):
                tree_sitter = importlib.import_module("tree_sitter")
                if hasattr(tree_sitter, "Language"):
                    return tree_sitter.Language(module.language())
            if module_name == "tree_sitter_languages" and hasattr(module, "get_language"):
                return module.get_language("rust")
        except Exception as exc:  # pragma: no cover - optional backend
            diagnostics.append(f"failed loading rust language from {module_name}: {exc}")
    return None


def unavailable_payload(path: Path, diagnostics: List[str], started: float) -> Dict[str, object]:
    return {
        "backend": "tree-sitter-rust",
        "available": False,
        "used": False,
        "parsed": False,
        "path": path.as_posix(),
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "item_counts": [],
        "statement_counts": [],
        "control_counts": [],
        "symbols": [],
        "error_nodes": 0,
        "diagnostics": diagnostics or ["tree-sitter rust grammar is not available"],
    }


def iter_node_types(node: object) -> Iterable[str]:
    stack = [node]
    while stack:
        current = stack.pop()
        yield str(getattr(current, "type"))
        children = list(getattr(current, "children", []))
        stack.extend(reversed(children))


def summarize_counts(node_types: Iterable[str], mapping: Dict[str, set[str]]) -> List[Dict[str, object]]:
    counts: List[Dict[str, object]] = []
    values = list(node_types)
    for kind, node_type_names in sorted(mapping.items()):
        count = sum(1 for node_type in values if node_type in node_type_names)
        if count:
            counts.append({"kind": kind, "count": count})
    return counts


def aggregate_counts(file_probes: Iterable[Dict[str, object]], field: str) -> List[Dict[str, object]]:
    counts: Dict[str, int] = {}
    for probe in file_probes:
        for item in probe.get(field, []):
            kind = str(item["kind"])
            counts[kind] = counts.get(kind, 0) + int(item["count"])
    return [{"kind": kind, "count": count} for kind, count in sorted(counts.items())]


def extract_symbols(root: object, source_bytes: bytes) -> List[Dict[str, object]]:
    symbols: List[Dict[str, object]] = []
    visit_symbols(root, source_bytes, (), None, symbols)
    return sorted(
        symbols,
        key=lambda item: (
            item["qualified_name"].count("::"),
            item["selection_range"]["start_line"],
            item["selection_range"]["start_column"],
            item["qualified_name"],
        ),
    )


def visit_symbols(
    node: object,
    source_bytes: bytes,
    module_segments: Tuple[str, ...],
    container_qualified_name: Optional[str],
    symbols: List[Dict[str, object]],
) -> None:
    node_type = str(getattr(node, "type", ""))

    if node_type == "mod_item":
        name = symbol_name(node, source_bytes)
        if name:
            qualified_name = "::".join(module_segments + (name,))
            symbols.append(
                make_symbol_payload(
                    node,
                    source_bytes,
                    "module",
                    name,
                    qualified_name,
                    container_qualified_name,
                )
            )
            child_module_segments = module_segments + (name,)
        else:
            child_module_segments = module_segments
        for child in getattr(node, "children", []):
            visit_symbols(child, source_bytes, child_module_segments, None, symbols)
        return

    if node_type == "trait_item":
        name = symbol_name(node, source_bytes)
        trait_qname = "::".join(module_segments + (name,)) if name else None
        if name and trait_qname:
            symbols.append(
                make_symbol_payload(
                    node,
                    source_bytes,
                    "trait",
                    name,
                    trait_qname,
                    container_qualified_name,
                )
            )
        for child in getattr(node, "children", []):
            visit_symbols(child, source_bytes, module_segments, trait_qname, symbols)
        return

    if node_type == "impl_item":
        impl_owner = infer_impl_owner(node, source_bytes)
        for child in getattr(node, "children", []):
            visit_symbols(child, source_bytes, module_segments, impl_owner, symbols)
        return

    symbol_kind = SYMBOL_NODE_TYPES.get(node_type)
    if symbol_kind is not None:
        name = symbol_name(node, source_bytes)
        if name:
            if node_type == "function_item" and container_qualified_name:
                symbol_kind = "method"
                qualified_name = f"{container_qualified_name}::{name}"
            else:
                qualified_name = "::".join(module_segments + (name,))
            symbols.append(
                make_symbol_payload(
                    node,
                    source_bytes,
                    symbol_kind,
                    name,
                    qualified_name,
                    container_qualified_name if symbol_kind == "method" else None,
                )
            )

    for child in getattr(node, "children", []):
        visit_symbols(child, source_bytes, module_segments, None, symbols)


def make_symbol_payload(
    node: object,
    source_bytes: bytes,
    kind: str,
    name: str,
    qualified_name: str,
    container_qualified_name: Optional[str],
) -> Dict[str, object]:
    name_node = find_name_node(node) or node
    return {
        "name": name,
        "kind": kind,
        "qualified_name": qualified_name,
        "container_qualified_name": container_qualified_name,
        "selection_range": node_range(name_node),
        "range": node_range(node),
        "signature": signature_for_node(node, source_bytes),
    }


def symbol_name(node: object, source_bytes: bytes) -> Optional[str]:
    name_node = find_name_node(node)
    if name_node is None:
        return None
    return node_text(name_node, source_bytes).strip() or None


def find_name_node(node: object) -> object | None:
    if hasattr(node, "child_by_field_name"):
        try:
            name_node = node.child_by_field_name("name")
        except Exception:  # pragma: no cover - depends on optional backend API shape
            name_node = None
        if name_node is not None:
            return name_node
    for child in getattr(node, "children", []):
        if str(getattr(child, "type", "")) in NAME_NODE_TYPES:
            return child
    return None


def infer_impl_owner(node: object, source_bytes: bytes) -> Optional[str]:
    text = collapse_whitespace(node_text(node, source_bytes))
    match = IMPL_HEAD_RE.match(text)
    if not match:
        return None
    target = normalize_type_expr(match.group("target"))
    trait = normalize_type_expr(match.group("trait"))
    return target or trait


def normalize_type_expr(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    normalized = GENERIC_RE.sub("", value)
    normalized = normalized.replace("&", " ").replace("*", " ")
    normalized = re.sub(r"\b(?:dyn|impl|mut|ref)\b", " ", normalized)
    normalized = re.sub(r"\s+", "", normalized).strip(":")
    return normalized or None


def signature_for_node(node: object, source_bytes: bytes) -> str:
    text = node_text(node, source_bytes)
    first_line = text.splitlines()[0] if text else ""
    return first_line.strip()


def node_range(node: object) -> Dict[str, int]:
    start_row, start_column = getattr(node, "start_point", (0, 0))
    end_row, end_column = getattr(node, "end_point", (0, 0))
    return {
        "start_line": int(start_row) + 1,
        "start_column": int(start_column) + 1,
        "end_line": int(end_row) + 1,
        "end_column": int(end_column) + 1,
    }


def node_text(node: object, source_bytes: bytes) -> str:
    start_byte = int(getattr(node, "start_byte", 0))
    end_byte = int(getattr(node, "end_byte", 0))
    return source_bytes[start_byte:end_byte].decode("utf-8", errors="replace")


def collapse_whitespace(value: str) -> str:
    return " ".join(value.split())
