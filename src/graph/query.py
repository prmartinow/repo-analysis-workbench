from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Sequence

from backends.graph_backend import get_graph_backend
from common.telemetry import increment_counter, trace_operation


EDGE_DEFAULTS = {
    "who_imports": ("IMPORTS",),
    "callers_of": ("CALLS",),
    "callees_of": ("CALLS",),
    "implements_of": ("IMPLEMENTS",),
    "inherits_of": ("INHERITS",),
    "reads_of": ("READS",),
    "writes_of": ("WRITES",),
    "refs_of": ("REFS", "REFERENCES"),
}
EDGE_DIRECTIONS = {
    "who_imports": "incoming",
    "callers_of": "incoming",
    "callees_of": "outgoing",
    "implements_of": "incoming",
    "inherits_of": "outgoing",
    "reads_of": "outgoing",
    "writes_of": "outgoing",
    "refs_of": "outgoing",
}


def graph_root_from_parsed(parsed_root: Path) -> Path:
    return parsed_root.parent / "graph"


def execute_graph_query(
    search_root: Path,
    parsed_root: Path,
    graph_root: Path,
    repo_name: str,
    request: Dict[str, object],
) -> Dict[str, object]:
    graph_backend = get_graph_backend(str(graph_root.resolve()), repo_name)
    backend_response = graph_backend.execute(request)
    if backend_response is None:
        raise FileNotFoundError(f"Missing graph backend artifact for repo '{repo_name}' under {graph_root / repo_name}")
    return backend_response


def where_defined(
    search_root: Path,
    parsed_root: Path,
    repo_name: str,
    symbol_query: str,
    *,
    limit: int = 10,
) -> Dict[str, object]:
    with trace_operation("where_defined"):
        response = execute_graph_query(
            search_root,
            parsed_root,
            graph_root=graph_root_from_parsed(parsed_root),
            repo_name=repo_name,
            request={"operation": "where_defined", "seed": symbol_query, "limit": limit},
        )
        return {
            "repo": repo_name,
            "query": symbol_query,
            "matches": response.get("results", []),
        }


def who_imports(
    search_root: Path,
    parsed_root: Path,
    graph_root: Path,
    repo_name: str,
    symbol_query: str,
    *,
    limit: int = 20,
) -> Dict[str, object]:
    return _neighbors_wrapper(search_root, parsed_root, graph_root, repo_name, symbol_query, operation="who_imports", limit=limit)


def adjacent_symbols(
    search_root: Path,
    parsed_root: Path,
    graph_root: Path,
    repo_name: str,
    symbol_query: str,
    *,
    edge_types: Sequence[str] = (),
    direction: str = "both",
    limit: int = 20,
) -> Dict[str, object]:
    response = execute_graph_query(
        search_root,
        parsed_root,
        graph_root,
        repo_name,
        {
            "operation": "neighbors",
            "seed": symbol_query,
            "edge_types": list(edge_types),
            "direction": direction,
            "depth": 1,
            "limit": limit,
        },
    )
    return {
        "repo": repo_name,
        "query": symbol_query,
        "matches": response.get("seeds", []),
        "neighbors": response.get("results", []),
    }


def callers_of(
    search_root: Path,
    parsed_root: Path,
    graph_root: Path,
    repo_name: str,
    symbol_query: str,
    *,
    limit: int = 20,
) -> Dict[str, object]:
    with trace_operation("callers_of"):
        return _neighbors_wrapper(search_root, parsed_root, graph_root, repo_name, symbol_query, operation="callers_of", limit=limit)


def callees_of(
    search_root: Path,
    parsed_root: Path,
    graph_root: Path,
    repo_name: str,
    symbol_query: str,
    *,
    limit: int = 20,
) -> Dict[str, object]:
    with trace_operation("callees_of"):
        return _neighbors_wrapper(search_root, parsed_root, graph_root, repo_name, symbol_query, operation="callees_of", limit=limit)


def reads_of(
    search_root: Path,
    parsed_root: Path,
    graph_root: Path,
    repo_name: str,
    symbol_query: str,
    *,
    limit: int = 20,
) -> Dict[str, object]:
    return _neighbors_wrapper(search_root, parsed_root, graph_root, repo_name, symbol_query, operation="reads_of", limit=limit)


def writes_of(
    search_root: Path,
    parsed_root: Path,
    graph_root: Path,
    repo_name: str,
    symbol_query: str,
    *,
    limit: int = 20,
) -> Dict[str, object]:
    return _neighbors_wrapper(search_root, parsed_root, graph_root, repo_name, symbol_query, operation="writes_of", limit=limit)


def refs_of(
    search_root: Path,
    parsed_root: Path,
    graph_root: Path,
    repo_name: str,
    symbol_query: str,
    *,
    limit: int = 20,
) -> Dict[str, object]:
    return _neighbors_wrapper(search_root, parsed_root, graph_root, repo_name, symbol_query, operation="refs_of", limit=limit)


def implements_of(
    search_root: Path,
    parsed_root: Path,
    graph_root: Path,
    repo_name: str,
    symbol_query: str,
    *,
    limit: int = 20,
) -> Dict[str, object]:
    return _neighbors_wrapper(search_root, parsed_root, graph_root, repo_name, symbol_query, operation="implements_of", limit=limit)


def inherits_of(
    search_root: Path,
    parsed_root: Path,
    graph_root: Path,
    repo_name: str,
    symbol_query: str,
    *,
    limit: int = 20,
) -> Dict[str, object]:
    return _neighbors_wrapper(search_root, parsed_root, graph_root, repo_name, symbol_query, operation="inherits_of", limit=limit)


def statement_slice(
    search_root: Path,
    parsed_root: Path,
    graph_root: Path,
    repo_name: str,
    symbol_query: str,
    *,
    limit: int = 20,
    window: int = 8,
) -> Dict[str, object]:
    with trace_operation("statement_slice"):
        response = execute_graph_query(
            search_root,
            parsed_root,
            graph_root,
            repo_name,
            {
                "operation": "statement_slice",
                "seed": symbol_query,
                "limit": limit,
                "window": window,
            },
        )
        return {
            "repo": repo_name,
            "query": symbol_query,
            "matches": response.get("seeds", []),
            "statements": response.get("results", []),
        }


def path_between(
    search_root: Path,
    parsed_root: Path,
    graph_root: Path,
    repo_name: str,
    source_query: str,
    target_query: str,
    *,
    edge_types: Sequence[str] = (),
    direction: str = "both",
    limit: int = 5,
) -> Dict[str, object]:
    with trace_operation("path_between"):
        response = execute_graph_query(
            search_root,
            parsed_root,
            graph_root,
            repo_name,
            {
                "operation": "path_between",
                "seed": source_query,
                "target": target_query,
                "edge_types": list(edge_types),
                "direction": direction,
                "limit": limit,
            },
        )
        return {
            "repo": repo_name,
            "source_query": source_query,
            "target_query": target_query,
            "matches": response.get("seeds", []),
            "targets": response.get("targets", []),
            "paths": response.get("results", []),
        }


def symbol_summary(
    search_root: Path,
    parsed_root: Path,
    graph_root: Path,
    repo_name: str,
    symbol_query: str,
    *,
    limit: int = 10,
) -> Dict[str, object]:
    with trace_operation("symbol_summary"):
        response = execute_graph_query(
            search_root,
            parsed_root,
            graph_root,
            repo_name,
            {
                "operation": "symbol_summary",
                "seed": symbol_query,
                "limit": limit,
            },
        )
        return {
            "repo": repo_name,
            "query": symbol_query,
            "matches": response.get("seeds", []),
            "summaries": response.get("results", []),
        }


def inspect_graph_backend_payload(graph_root: Path, repo_name: str) -> Dict[str, object]:
    return _load_graph_backend_payload_cached(str(graph_root.resolve()), repo_name)


def reset_graph_view_cache() -> None:
    _load_graph_backend_payload_cached.cache_clear()


@lru_cache(maxsize=8)
def _load_graph_backend_payload_cached(graph_root: str, repo_name: str) -> Dict[str, object]:
    root = Path(graph_root)
    graph_backend = get_graph_backend(str(root.resolve()), repo_name)
    if hasattr(graph_backend, "load_payload"):
        return graph_backend.load_payload()
    raise FileNotFoundError(f"Missing graph artifact for repo '{repo_name}' under {root / repo_name}")


def inspect_graph_backend_payload_uncached(graph_root: Path, repo_name: str) -> Dict[str, object]:
    increment_counter("full_graph_payload_loads")
    with trace_operation("inspect_graph_backend_payload_uncached"):
        graph_backend = get_graph_backend(str(graph_root.resolve()), repo_name)
        if hasattr(graph_backend, "load_payload"):
            return graph_backend.load_payload()
        raise FileNotFoundError(f"Missing graph artifact for repo '{repo_name}' under {graph_root / repo_name}")


def _neighbors_wrapper(
    search_root: Path,
    parsed_root: Path,
    graph_root: Path,
    repo_name: str,
    symbol_query: str,
    *,
    operation: str,
    limit: int = 20,
) -> Dict[str, object]:
    response = execute_graph_query(
        search_root,
        parsed_root,
        graph_root,
        repo_name,
        {
            "operation": operation,
            "seed": symbol_query,
            "edge_types": list(EDGE_DEFAULTS.get(operation, ())),
            "direction": EDGE_DIRECTIONS.get(operation, "both"),
            "depth": 1,
            "limit": limit,
        },
    )
    return {
        "repo": repo_name,
        "query": symbol_query,
        "matches": response.get("seeds", []),
        "neighbors": response.get("results", []),
    }
