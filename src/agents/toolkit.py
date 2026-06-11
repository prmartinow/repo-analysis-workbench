from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from backends.graph_backend import get_graph_backend
from backends.metadata_store import get_metadata_store
from backends.search_backend import get_search_backend
from common.retrieval import semantic_activity_score
from common.telemetry import snapshot_telemetry, trace_operation
from retrieval.engine import retrieve_context
from retrieval.planner import (
    plan_query as build_query_plan,
    prepare_answer_bundle as build_answer_bundle,
    retrieve_iterative as build_iterative_bundle,
)
from symbols.indexer import stable_id


DEFAULT_REPOS: tuple[str, ...] = ()
CALLABLE_SYMBOL_KINDS = ("function", "method", "associated_function")
LOCAL_SYMBOL_KINDS = {"local"}
GRAPH_NOISE_KINDS = {"statement", "symbol_ref"}
NOMINAL_SYMBOL_KINDS = {"trait", "struct", "enum", "type"}


def execute_graph_backend_request(
    graph_root: Path,
    repo_name: str,
    request: Dict[str, object],
) -> Dict[str, object]:
    backend = get_graph_backend(str(graph_root.resolve()), repo_name)
    response = backend.execute(request)
    if response is None:
        raise FileNotFoundError(f"Missing graph backend artifact for repo '{repo_name}' under {graph_root / repo_name}")
    return response


def graph_root_from_parsed(parsed_root: Path) -> Path:
    return parsed_root.parent / "graph"


def graph_neighbors_response(
    search_root: Path,
    parsed_root: Path,
    graph_root: Path,
    repo_name: str,
    symbol_query: str,
    *,
    operation: Optional[str] = None,
    edge_types: Sequence[str] = (),
    direction: str = "both",
    depth: int = 1,
    node_kinds: Sequence[str] = (),
    limit: int = 20,
) -> Dict[str, object]:
    seed, resolved_symbol, candidates, resolution = resolve_graph_seed(
        search_root,
        parsed_root,
        repo_name,
        symbol_query,
        limit=max(limit, 10),
    )
    if resolution == "ambiguous":
        return {
            "repo": repo_name,
            "query": symbol_query,
            "resolved_symbol": None,
            "candidates": candidates,
            "matches": [],
            "neighbors": [],
            "error": f"ambiguous symbol query: {symbol_query}",
        }

    effective_node_kinds = tuple(node_kinds)
    if not effective_node_kinds and operation in {"callers_of", "callees_of"}:
        effective_node_kinds = CALLABLE_SYMBOL_KINDS

    request: Dict[str, object] = {
        "operation": operation or "neighbors",
        "seed": seed,
        "limit": limit,
    }
    if operation is None or edge_types:
        request["edge_types"] = list(edge_types)
    if operation is None or direction != "both":
        request["direction"] = direction
    if operation is None or depth != 1:
        request["depth"] = depth
    if effective_node_kinds:
        request["node_kinds"] = list(effective_node_kinds)

    response = execute_graph_backend_request(graph_root, repo_name, request)
    neighbors = clean_graph_neighbors(
        operation,
        resolved_symbol,
        response.get("results", []),
        allowed_kinds=effective_node_kinds,
    )
    return {
        "repo": repo_name,
        "query": symbol_query,
        "resolved_symbol": resolved_symbol,
        "candidates": candidates,
        "matches": response.get("seeds", []),
        "neighbors": neighbors,
    }


def repo_overview(parsed_root: Path, repo_name: str) -> Dict[str, object]:
    metadata_store = get_metadata_store(str(parsed_root.resolve()), repo_name)
    project_summary = metadata_store.get_summary_by_id(stable_id("sum", repo_name, "project")) or {}
    return {
        "repo": repo_name,
        "project": project_summary,
        "summary_counts": {},
    }


def find_symbol(search_root: Path, repo_name: str, query: str, *, limit: int = 10) -> Dict[str, object]:
    search_backend = get_search_backend(str(search_root.resolve()), repo_name)
    return {
        "repo": repo_name,
        "query": query,
        "results": search_backend.search(query, limit=limit, kinds=("symbol",)),
    }


def find_file(search_root: Path, repo_name: str, path_pattern: str, *, limit: int = 20) -> Dict[str, object]:
    search_backend = get_search_backend(str(search_root.resolve()), repo_name)
    return {
        "repo": repo_name,
        "path_pattern": path_pattern,
        "results": search_backend.find_file(path_pattern, limit=limit),
    }


def search_lexical(
    search_root: Path,
    repo_name: str,
    query: str,
    *,
    limit: int = 10,
    kinds: Sequence[str] = (),
    path_prefix: Optional[str] = None,
) -> Dict[str, object]:
    search_backend = get_search_backend(str(search_root.resolve()), repo_name)
    return {
        "repo": repo_name,
        "query": query,
        "scope": {
            "kinds": list(kinds),
            "path_prefix": path_prefix,
        },
        "results": search_backend.search(query, limit=limit, kinds=kinds, path_prefix=path_prefix),
    }


def trace_calls(
    search_root: Path,
    graph_root: Path,
    parsed_root: Path,
    repo_name: str,
    symbol_query: str,
    *,
    limit: int = 10,
) -> Dict[str, object]:
    resolved = where_defined(search_root, parsed_root, repo_name, symbol_query, limit=1)
    if not resolved["matches"]:
        return {
            "repo": repo_name,
            "query": symbol_query,
            "error": "symbol not found",
        }
    callers = callers_of(search_root, parsed_root, graph_root, repo_name, symbol_query, limit=limit)
    callees = callees_of(search_root, parsed_root, graph_root, repo_name, symbol_query, limit=limit)
    return {
        "repo": repo_name,
        "query": symbol_query,
        "resolved_symbol": resolved["matches"][0],
        "callers": callers["neighbors"],
        "callees": callees["neighbors"],
    }


def compare_repos(
    search_root: Path,
    graph_root: Path,
    parsed_root: Path,
    query: str,
    *,
    repos: Sequence[str] = DEFAULT_REPOS,
    limit: int = 5,
) -> Dict[str, object]:
    comparisons = []
    for repo_name in repos:
        metadata_store = get_metadata_store(str(parsed_root.resolve()), repo_name)
        project_summary = metadata_store.get_summary_by_id(stable_id("sum", repo_name, "project")) or {}
        cached = compare_repo_from_agent_cache(search_root, repo_name, query, project_summary, limit=limit)
        if cached is not None:
            comparisons.append(cached)
            continue

        bundle = prepare_answer_bundle(
            search_root,
            graph_root,
            parsed_root,
            query,
            repo_name=repo_name,
            limit=limit,
        )
        repo_bundle = bundle["bundles"][0]
        comparisons.append(
            {
                "repo": repo_name,
                "focus": repo_bundle["focus"],
                "top_context": repo_bundle["selected_context"],
                "bundle_summary": {
                    **repo_bundle["bundle_summary"],
                    "source": "answer_bundle_fallback",
                },
            }
        )
    return {"query": query, "comparisons": comparisons}
def summarize_path(parsed_root: Path, repo_name: str, path: str) -> Dict[str, object]:
    metadata_store = get_metadata_store(str(parsed_root.resolve()), repo_name)
    path_summaries = metadata_store.get_summary_by_path(path)
    if path_summaries:
        exact_kind = "file" if PurePosixPath(path).suffix else "directory"
        return {"repo": repo_name, "path": path, "kind": exact_kind, "summary": path_summaries[0]}

    matching_prefix = None
    prefixes = parent_paths(path)
    for prefix in prefixes:
        prefix_summaries = metadata_store.get_summary_by_path(prefix)
        for summary in prefix_summaries:
            if str(summary.get("scope") or "") != "directory":
                continue
            if matching_prefix is None or len(str(summary.get("path") or "")) > len(str(matching_prefix.get("path") or "")):
                matching_prefix = summary
    return {
        "repo": repo_name,
        "path": path,
        "kind": "directory" if matching_prefix else "unknown",
        "summary": matching_prefix,
    }


def get_summary(
    search_root: Path,
    graph_root: Path,
    parsed_root: Path,
    repo_name: str,
    node_id: str,
) -> Dict[str, object]:
    metadata_store = get_metadata_store(str(parsed_root.resolve()), repo_name)
    summary_by_id = metadata_store.get_summary_by_id(node_id)
    if summary_by_id is not None:
        return {"repo": repo_name, "node_id": node_id, "summary": summary_by_id}
    if node_id == stable_repo_id(repo_name):
        project_summary = metadata_store.get_summary_by_id(stable_id("sum", repo_name, "project"))
        return {"repo": repo_name, "node_id": node_id, "summary": project_summary}

    graph_response = execute_graph_backend_request(
        graph_root,
        repo_name,
        {"operation": "symbol_summary", "seed": {"node_id": node_id}, "limit": 1},
    )
    if graph_response.get("results"):
        return {"repo": repo_name, "node_id": node_id, "summary": graph_response["results"][0]}
    return {"repo": repo_name, "node_id": node_id, "summary": None}


def get_symbol_signature(
    search_root: Path,
    parsed_root: Path,
    repo_name: str,
    symbol_query: str,
) -> Dict[str, object]:
    with trace_operation("get_symbol_signature"):
        symbol = resolve_symbol_query(search_root, parsed_root, repo_name, symbol_query)
        return {
            "repo": repo_name,
            "query": symbol_query,
            "symbol": symbol,
            "signature": symbol.get("signature") if symbol else None,
        }


def get_symbol_body(
    search_root: Path,
    parsed_root: Path,
    repo_name: str,
    symbol_query: str,
) -> Dict[str, object]:
    with trace_operation("get_symbol_body"):
        symbol = resolve_symbol_query(search_root, parsed_root, repo_name, symbol_query)
        if not symbol:
            return {"repo": repo_name, "query": symbol_query, "symbol": None, "body": None}

        search_backend = get_search_backend(str(search_root.resolve()), repo_name)
        documents = search_backend.lookup_symbol_docs(
            symbol["symbol_id"],
            kinds=("function_body", "type_body"),
            limit=4,
        )
        return {
            "repo": repo_name,
            "query": symbol_query,
            "symbol": describe_symbol_row(symbol),
            "body": documents[0] if documents else None,
        }


def telemetry_snapshot() -> Dict[str, object]:
    return snapshot_telemetry()


def get_enclosing_context(
    search_root: Path,
    graph_root: Path,
    parsed_root: Path,
    repo_name: str,
    symbol_query: str,
) -> Dict[str, object]:
    symbol = resolve_symbol_query(search_root, parsed_root, repo_name, symbol_query)
    if not symbol:
        return {"repo": repo_name, "query": symbol_query, "context": None}

    metadata_store = get_metadata_store(str(parsed_root.resolve()), repo_name)
    container = metadata_store.get_symbol(str(symbol.get("container_symbol_id") or ""))
    path_summary = metadata_store.get_summary_by_path(symbol["path"])
    return {
        "repo": repo_name,
        "query": symbol_query,
        "context": {
            "symbol": describe_symbol_row(symbol),
            "container": describe_symbol_row(container) if container else None,
            "path_summary": path_summary[0] if path_summary else None,
            "statement_slice": statement_slice(
                search_root,
                parsed_root,
                graph_root,
                repo_name,
                symbol_query,
                limit=8,
                window=8,
            ),
        },
    }


def prepare_context(
    search_root: Path,
    graph_root: Path,
    parsed_root: Path,
    task: str,
    *,
    repo_name: Optional[str] = None,
    limit: int = 8,
) -> Dict[str, object]:
    bundle = prepare_answer_bundle(
        search_root,
        graph_root,
        parsed_root,
        task,
        repo_name=repo_name,
        limit=limit,
    )
    contexts = []
    for repo_bundle in bundle["bundles"]:
        contexts.append(
            {
                "repo": repo_bundle["repo"],
                "focus": repo_bundle["focus"],
                "project_summary": repo_bundle["project_summary"],
                "selected_context": repo_bundle["selected_context"],
            }
        )
    return {
        "task": task,
        "contexts": contexts,
    }


def where_defined(
    search_root: Path,
    parsed_root: Path,
    repo_name: str,
    symbol_query: str,
    *,
    limit: int = 10,
) -> Dict[str, object]:
    metadata_store = get_metadata_store(str(parsed_root.resolve()), repo_name)
    matches: List[Dict[str, object]] = []
    for symbol_id in metadata_store.resolve_qname(symbol_query):
        symbol = metadata_store.get_symbol(symbol_id)
        if symbol:
            matches.append(describe_symbol_row(symbol) or {})
    if not matches:
        for symbol_id in metadata_store.resolve_name(symbol_query, repo=repo_name):
            symbol = metadata_store.get_symbol(symbol_id)
            if symbol:
                matches.append(describe_symbol_row(symbol) or {})
                if len(matches) >= limit:
                    break
    if not matches:
        response = execute_graph_backend_request(
            graph_root_from_parsed(parsed_root),
            repo_name,
            {"operation": "where_defined", "seed": symbol_query, "limit": limit},
        )
        return {
            "repo": repo_name,
            "query": symbol_query,
            "matches": response.get("results", []),
        }
    return {
        "repo": repo_name,
        "query": symbol_query,
        "matches": matches[:limit],
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
    return graph_neighbors_response(search_root, parsed_root, graph_root, repo_name, symbol_query, operation="who_imports", limit=limit)


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
    return graph_neighbors_response(
        search_root,
        parsed_root,
        graph_root,
        repo_name,
        symbol_query,
        edge_types=edge_types,
        direction=direction,
        depth=1,
        limit=limit,
    )


def callers_of(
    search_root: Path,
    parsed_root: Path,
    graph_root: Path,
    repo_name: str,
    symbol_query: str,
    *,
    limit: int = 20,
) -> Dict[str, object]:
    return graph_neighbors_response(search_root, parsed_root, graph_root, repo_name, symbol_query, operation="callers_of", limit=limit)


def callees_of(
    search_root: Path,
    parsed_root: Path,
    graph_root: Path,
    repo_name: str,
    symbol_query: str,
    *,
    limit: int = 20,
) -> Dict[str, object]:
    return graph_neighbors_response(search_root, parsed_root, graph_root, repo_name, symbol_query, operation="callees_of", limit=limit)


def reads_of(
    search_root: Path,
    parsed_root: Path,
    graph_root: Path,
    repo_name: str,
    symbol_query: str,
    *,
    limit: int = 20,
) -> Dict[str, object]:
    return graph_neighbors_response(search_root, parsed_root, graph_root, repo_name, symbol_query, operation="reads_of", limit=limit)


def writes_of(
    search_root: Path,
    parsed_root: Path,
    graph_root: Path,
    repo_name: str,
    symbol_query: str,
    *,
    limit: int = 20,
) -> Dict[str, object]:
    return graph_neighbors_response(search_root, parsed_root, graph_root, repo_name, symbol_query, operation="writes_of", limit=limit)


def refs_of(
    search_root: Path,
    parsed_root: Path,
    graph_root: Path,
    repo_name: str,
    symbol_query: str,
    *,
    limit: int = 20,
) -> Dict[str, object]:
    return graph_neighbors_response(search_root, parsed_root, graph_root, repo_name, symbol_query, operation="refs_of", limit=limit)


def implements_of(
    search_root: Path,
    parsed_root: Path,
    graph_root: Path,
    repo_name: str,
    symbol_query: str,
    *,
    limit: int = 20,
) -> Dict[str, object]:
    return graph_neighbors_response(search_root, parsed_root, graph_root, repo_name, symbol_query, operation="implements_of", limit=limit)


def inherits_of(
    search_root: Path,
    parsed_root: Path,
    graph_root: Path,
    repo_name: str,
    symbol_query: str,
    *,
    limit: int = 20,
) -> Dict[str, object]:
    return graph_neighbors_response(search_root, parsed_root, graph_root, repo_name, symbol_query, operation="inherits_of", limit=limit)


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
        seed, resolved_symbol, candidates, resolution = resolve_graph_seed(
            search_root,
            parsed_root,
            repo_name,
            symbol_query,
            limit=max(limit, 10),
        )
        if resolution == "ambiguous":
            return {
                "repo": repo_name,
                "query": symbol_query,
                "resolved_symbol": None,
                "candidates": candidates,
                "matches": [],
                "statements": [],
                "error": f"ambiguous symbol query: {symbol_query}",
            }

        response = execute_graph_backend_request(
            graph_root,
            repo_name,
            {
                "operation": "statement_slice",
                "seed": seed,
                "limit": limit,
                "window": window,
            },
        )
        return {
            "repo": repo_name,
            "query": symbol_query,
            "resolved_symbol": resolved_symbol,
            "candidates": candidates,
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
    limit: int = 5,
    edge_types: Sequence[str] = (),
    direction: str = "both",
) -> Dict[str, object]:
    with trace_operation("path_between"):
        source_seed, resolved_source_symbol, source_candidates, source_resolution = resolve_graph_seed(
            search_root,
            parsed_root,
            repo_name,
            source_query,
            limit=max(limit, 10),
        )
        if source_resolution == "ambiguous":
            return {
                "repo": repo_name,
                "source_query": source_query,
                "target_query": target_query,
                "resolved_source_symbol": None,
                "resolved_target_symbol": None,
                "source_candidates": source_candidates,
                "target_candidates": [],
                "matches": [],
                "targets": [],
                "paths": [],
                "error": f"ambiguous source symbol query: {source_query}",
            }

        target_seed, resolved_target_symbol, target_candidates, target_resolution = resolve_graph_seed(
            search_root,
            parsed_root,
            repo_name,
            target_query,
            limit=max(limit, 10),
        )
        if target_resolution == "ambiguous":
            return {
                "repo": repo_name,
                "source_query": source_query,
                "target_query": target_query,
                "resolved_source_symbol": resolved_source_symbol,
                "resolved_target_symbol": None,
                "source_candidates": source_candidates,
                "target_candidates": target_candidates,
                "matches": [],
                "targets": [],
                "paths": [],
                "error": f"ambiguous target symbol query: {target_query}",
            }

        response = execute_graph_backend_request(
            graph_root,
            repo_name,
            {
                "operation": "path_between",
                "seed": source_seed,
                "target": target_seed,
                "limit": limit,
                "edge_types": list(edge_types),
                "direction": direction,
            },
        )
        return {
            "repo": repo_name,
            "source_query": source_query,
            "target_query": target_query,
            "resolved_source_symbol": resolved_source_symbol,
            "resolved_target_symbol": resolved_target_symbol,
            "source_candidates": source_candidates,
            "target_candidates": target_candidates,
            "matches": response.get("seeds", []),
            "targets": response.get("targets", []),
            "paths": response.get("results", []),
        }


def execute_graph_query(
    search_root: Path,
    parsed_root: Path,
    graph_root: Path,
    repo_name: str,
    request: Dict[str, object],
) -> Dict[str, object]:
    return execute_graph_backend_request(graph_root, repo_name, request)


def expand_subgraph(
    search_root: Path,
    parsed_root: Path,
    graph_root: Path,
    repo_name: str,
    seed: object,
    *,
    edge_types: Sequence[str] = (),
    direction: str = "both",
    depth: int = 1,
    node_kinds: Sequence[str] = (),
    budget: int = 20,
) -> Dict[str, object]:
    return execute_graph_backend_request(
        graph_root,
        repo_name,
        {
            "operation": "neighbors",
            "seed": seed,
            "edge_types": list(edge_types),
            "direction": direction,
            "depth": depth,
            "node_kinds": list(node_kinds),
            "limit": budget,
        },
    )


def symbol_summary(
    search_root: Path,
    parsed_root: Path,
    graph_root: Path,
    repo_name: str,
    symbol_query: str,
    *,
    limit: int = 5,
) -> Dict[str, object]:
    with trace_operation("symbol_summary"):
        seed, resolved_symbol, candidates, resolution = resolve_graph_seed(
            search_root,
            parsed_root,
            repo_name,
            symbol_query,
            limit=max(limit, 10),
        )
        if resolution == "ambiguous":
            return {
                "repo": repo_name,
                "query": symbol_query,
                "resolved_symbol": None,
                "candidates": candidates,
                "matches": [],
                "summaries": [],
                "error": f"ambiguous symbol query: {symbol_query}",
            }

        response = execute_graph_backend_request(
            graph_root,
            repo_name,
            {
                "operation": "symbol_summary",
                "seed": seed,
                "limit": limit,
            },
        )
        return {
            "repo": repo_name,
            "query": symbol_query,
            "resolved_symbol": resolved_symbol,
            "candidates": candidates,
            "matches": response.get("seeds", []),
            "summaries": response.get("results", []),
        }


def plan_query(
    search_root: Path,
    graph_root: Path,
    parsed_root: Path,
    task: str,
    *,
    repo_name: Optional[str] = None,
    limit: int = 8,
) -> Dict[str, object]:
    return build_query_plan(
        search_root,
        graph_root,
        parsed_root,
        task,
        repo_name=repo_name,
        limit=limit,
    )


def prepare_answer_bundle(
    search_root: Path,
    graph_root: Path,
    parsed_root: Path,
    task: str,
    *,
    repo_name: Optional[str] = None,
    limit: int = 8,
    refinement_hints: Sequence[str] = (),
) -> Dict[str, object]:
    return build_answer_bundle(
        search_root,
        graph_root,
        parsed_root,
        task,
        repo_name=repo_name,
        limit=limit,
        refinement_hints=refinement_hints,
    )


def retrieve_iterative(
    search_root: Path,
    graph_root: Path,
    parsed_root: Path,
    task: str,
    *,
    repo_name: Optional[str] = None,
    limit: int = 8,
    prior_bundle: Optional[Dict[str, object]] = None,
    refinement_hints: Sequence[str] = (),
) -> Dict[str, object]:
    return build_iterative_bundle(
        search_root,
        graph_root,
        parsed_root,
        task,
        repo_name=repo_name,
        limit=limit,
        prior_bundle=prior_bundle,
        refinement_hints=refinement_hints,
    )


def score_external_answers(
    eval_root: Path,
    answers_path: Path,
) -> Dict[str, object]:
    from evaluation.harness import score_external_answers as score_external_answers_payload

    return score_external_answers_payload(eval_root, answers_path)


def compare_repo_from_agent_cache(
    search_root: Path,
    repo_name: str,
    query: str,
    project_summary: Dict[str, object],
    *,
    limit: int,
) -> Optional[Dict[str, object]]:
    search_backend = get_search_backend(str(search_root.resolve()), repo_name)
    top_context = search_backend.compare_repo_candidates(query, limit=limit)
    if not top_context:
        return None
    return {
        "repo": repo_name,
        "focus": str(project_summary.get("focus") or ""),
        "top_context": top_context,
        "bundle_summary": {
            "selected_context": len(top_context),
            "graph_neighborhoods": 0,
            "statement_slices": 0,
            "evidence_items": len(top_context),
            "source": "search_backend",
            "cache_entries": len(top_context),
        },
    }


def query_agent_cache(entries: Sequence[Dict[str, object]], query: str, *, limit: int) -> List[Dict[str, object]]:
    query_tokens = normalize_query_tokens(query)
    scored = []
    for entry in entries:
        score = score_agent_cache_entry(entry, query, query_tokens)
        if score <= 0:
            continue
        scored.append((score, entry))

    scored.sort(
        key=lambda item: (
            -item[0],
            kind_compare_rank(str(item[1].get("kind") or "")),
            str(item[1].get("path") or ""),
            str(item[1].get("qualified_name") or item[1].get("name") or ""),
        )
    )

    results = []
    seen = set()
    for score, entry in scored:
        dedupe_key = (
            str(entry.get("symbol_id") or ""),
            str(entry.get("path") or ""),
            str(entry.get("qualified_name") or entry.get("name") or ""),
            str(entry.get("kind") or ""),
        )
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        result = dict(entry)
        result["score"] = round(float(score), 3)
        result.pop("search_text", None)
        results.append(result)
        if len(results) >= limit:
            break
    return results


def score_agent_cache_entry(entry: Dict[str, object], raw_query: str, query_tokens: Sequence[str]) -> float:
    search_text = str(entry.get("search_text") or "")
    name = str(entry.get("name") or "")
    qualified_name = str(entry.get("qualified_name") or "")
    path = str(entry.get("path") or "")
    tags = [str(item) for item in entry.get("metadata", {}).get("tags", []) or ()]

    score = 0.0
    if raw_query == qualified_name:
        score += 120.0
    if raw_query == name:
        score += 100.0
    lowered_query = raw_query.lower()
    if lowered_query == qualified_name.lower():
        score += 90.0
    if lowered_query == name.lower():
        score += 75.0
    if lowered_query and lowered_query in path.lower():
        score += 30.0

    token_hits = 0
    for token in query_tokens:
        if token in search_text:
            score += 8.0
            token_hits += 1
        if token in (name.lower(), qualified_name.lower()):
            score += 12.0
        if token in path.lower():
            score += 6.0
        if token in tags:
            score += 6.0

    if token_hits == len(query_tokens) and query_tokens:
        score += 20.0

    visibility = str(entry.get("metadata", {}).get("visibility") or "")
    if visibility.startswith("pub") or visibility == "public":
        score += 5.0
    return score


def normalize_query_tokens(query: str) -> List[str]:
    tokens = []
    for raw_token in query.replace("::", " ").replace("/", " ").replace("-", " ").replace(".", " ").split():
        normalized = "".join(char for char in raw_token.lower() if char.isalnum() or char == "_")
        if normalized:
            tokens.append(normalized)
    return tokens


def kind_compare_rank(kind: str) -> int:
    ranking = {
        "repo": 0,
        "package": 1,
        "directory": 2,
        "file": 3,
        "symbol": 4,
        "type_body": 5,
        "function_body": 6,
        "doc": 7,
    }
    return ranking.get(kind, 99)


def clean_graph_neighbors(
    operation: Optional[str],
    resolved_symbol: Optional[Dict[str, object]],
    neighbors: Sequence[Dict[str, object]],
    *,
    allowed_kinds: Sequence[str] = (),
) -> List[Dict[str, object]]:
    allowed_kind_set = {str(kind).lower() for kind in allowed_kinds if str(kind).strip()}
    resolved_symbol_id = str((resolved_symbol or {}).get("symbol_id") or "")

    cleaned: List[Dict[str, object]] = []
    seen = set()
    for neighbor in neighbors:
        if not isinstance(neighbor, dict):
            continue
        if is_graph_noise_neighbor(neighbor):
            continue

        neighbor_kind = str(neighbor.get("kind") or "").lower()
        if allowed_kind_set and neighbor_kind not in allowed_kind_set:
            continue

        neighbor_symbol_id = str(neighbor.get("symbol_id") or "")
        neighbor_node_id = str(neighbor.get("node_id") or "")
        if resolved_symbol_id and (neighbor_symbol_id == resolved_symbol_id or neighbor_node_id == resolved_symbol_id):
            continue

        if operation in {"callers_of", "callees_of"} and is_transitive_call_neighbor(neighbor):
            continue

        dedupe_key = (
            neighbor_symbol_id,
            neighbor_node_id,
            str(neighbor.get("qualified_name") or ""),
            str(neighbor.get("path") or ""),
            neighbor_kind,
            str((neighbor.get("edge_metadata") or {}).get("line") or ((neighbor.get("edge") or {}).get("metadata") or {}).get("line") or ""),
        )
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        cleaned.append(neighbor)

    return cleaned


def is_graph_noise_neighbor(neighbor: Dict[str, object]) -> bool:
    neighbor_kind = str(neighbor.get("kind") or "").lower()
    node_id = str(neighbor.get("node_id") or "")
    if neighbor_kind in GRAPH_NOISE_KINDS:
        return True
    if node_id.startswith("stmt:") or node_id.startswith("ref:"):
        return True
    return False


def is_transitive_call_neighbor(neighbor: Dict[str, object]) -> bool:
    edge_type = str(neighbor.get("edge_type") or ((neighbor.get("edge") or {}).get("type") or "")).upper()
    if edge_type != "CALLS":
        return False
    edge_metadata = neighbor.get("edge_metadata") or ((neighbor.get("edge") or {}).get("metadata") or {})
    semantic_level = str(edge_metadata.get("semantic_level") or "").lower()
    return semantic_level == "transitive"


def resolve_graph_seed(
    search_root: Path,
    parsed_root: Path,
    repo_name: str,
    symbol_query: str,
    *,
    limit: int = 10,
) -> Tuple[object, Optional[Dict[str, object]], List[Dict[str, object]], Optional[str]]:
    query = str(symbol_query or "").strip()
    if not query:
        return symbol_query, None, [], None

    candidates = resolve_symbol_candidates(search_root, parsed_root, repo_name, query, limit=max(limit, 10))
    if not candidates:
        return query, None, [], None

    lowered_query = query.lower()
    query_segments = split_symbol_query_segments(query)

    exact_qname = [candidate for candidate in candidates if str(candidate.get("qualified_name") or "").lower() == lowered_query]
    exact_name = [candidate for candidate in candidates if str(candidate.get("name") or "").lower() == lowered_query]
    suffix_matches = [candidate for candidate in candidates if qualified_name_endswith_query(str(candidate.get("qualified_name") or ""), lowered_query)]
    segment_matches = [
        candidate
        for candidate in candidates
        if query_segments
        and candidate_matches_query_segments(
            query_segments,
            split_symbol_query_segments(str(candidate.get("qualified_name") or "")),
        )
    ]

    selected: Optional[Dict[str, object]] = None
    resolution: Optional[str] = None

    if len(exact_qname) == 1:
        selected = exact_qname[0]
        resolution = "exact_qualified_name"
    elif len(exact_name) == 1:
        selected = exact_name[0]
        resolution = "exact_name"
    elif len(suffix_matches) == 1:
        selected = suffix_matches[0]
        resolution = "qualified_name_suffix"
    elif len(segment_matches) == 1:
        selected = segment_matches[0]
        resolution = "qualified_name_segments"
    elif len(candidates) == 1:
        selected = candidates[0]
        resolution = "single_candidate"
    else:
        top_score = float(candidates[0].get("_candidate_score") or 0.0)
        second_score = float(candidates[1].get("_candidate_score") or 0.0)
        if top_score >= 120.0 and (top_score - second_score) >= 25.0:
            selected = candidates[0]
            resolution = "strong_top_candidate"
        else:
            resolution = "ambiguous"

    public_candidates = select_public_symbol_candidates(
        query,
        candidates,
        selected,
        resolution,
        limit=limit,
        exact_qname=exact_qname,
        exact_name=exact_name,
        suffix_matches=suffix_matches,
        segment_matches=segment_matches,
    )

    if selected is None:
        return query, None, public_candidates, resolution

    symbol_id = str(selected.get("symbol_id") or "")
    if not symbol_id:
        return query, None, public_candidates, resolution
    return {"node_id": symbol_id}, strip_candidate_score(selected), public_candidates, resolution


def resolve_symbol_candidates(
    search_root: Path,
    parsed_root: Path,
    repo_name: str,
    symbol_query: str,
    *,
    limit: int = 10,
) -> List[Dict[str, object]]:
    metadata_store = get_metadata_store(str(parsed_root.resolve()), repo_name)
    candidate_rows: List[Dict[str, object]] = []
    allow_locals = query_explicitly_targets_local(symbol_query)

    def add_symbol(symbol_id: str, *, search_score: float = 0.0) -> None:
        symbol = metadata_store.get_symbol(symbol_id)
        if symbol is None:
            return
        described = describe_symbol_row(symbol)
        if described is None:
            return
        kind = str(described.get("kind") or "").lower()
        if kind in LOCAL_SYMBOL_KINDS and not allow_locals:
            return
        row = dict(described)
        row["_search_score"] = float(search_score)
        row["visibility"] = symbol.get("visibility")
        row["module_path"] = symbol.get("module_path")
        row["crate"] = symbol.get("crate")
        row["package_name"] = symbol.get("package_name")
        row["is_test"] = bool(symbol.get("is_test"))
        row["semantic_summary"] = symbol.get("semantic_summary", {})
        container_symbol_id = str(symbol.get("container_symbol_id") or "")
        if container_symbol_id:
            container_symbol = metadata_store.get_symbol(container_symbol_id)
            if container_symbol is not None:
                row["_container_kind"] = container_symbol.get("kind")
                row["_container_name"] = container_symbol.get("name")
        candidate_rows.append(row)

    if symbol_query.startswith("sym:"):
        add_symbol(symbol_query, search_score=500.0)

    for symbol_id in metadata_store.resolve_qname(symbol_query):
        add_symbol(symbol_id, search_score=400.0)

    for symbol_id in metadata_store.resolve_name(symbol_query, repo=repo_name):
        add_symbol(symbol_id, search_score=300.0)

    search_backend = get_search_backend(str(search_root.resolve()), repo_name)
    lexical_results = search_backend.search(
        symbol_query,
        limit=max(limit * 4, 20),
        kinds=("symbol",),
    )
    for result in lexical_results:
        symbol_id = str(result.get("symbol_id") or "")
        if not symbol_id:
            continue
        add_symbol(symbol_id, search_score=float(result.get("score") or 0.0))

    return rank_symbol_candidates(symbol_query, candidate_rows, limit=limit)


def rank_symbol_candidates(
    symbol_query: str,
    candidates: Sequence[Dict[str, object]],
    *,
    limit: int,
) -> List[Dict[str, object]]:
    scored: List[Tuple[float, Dict[str, object]]] = []
    seen = set()

    for candidate in candidates:
        symbol_id = str(candidate.get("symbol_id") or "")
        if not symbol_id or symbol_id in seen:
            continue
        seen.add(symbol_id)

        scored_candidate = dict(candidate)
        scored_candidate["_candidate_score"] = score_symbol_candidate(symbol_query, scored_candidate)
        scored.append((float(scored_candidate["_candidate_score"]), scored_candidate))

    scored.sort(
        key=lambda item: (
            -item[0],
            str(item[1].get("path") or ""),
            str(item[1].get("qualified_name") or item[1].get("name") or ""),
        )
    )
    return [candidate for _score, candidate in scored[:limit]]


def score_symbol_candidate(symbol_query: str, candidate: Dict[str, object]) -> float:
    raw_query = str(symbol_query or "").strip()
    lowered_query = str(symbol_query or "").strip().lower()
    query_tokens = normalize_query_tokens(symbol_query)
    query_segments = split_symbol_query_segments(symbol_query)
    single_token_query = len(query_tokens) == 1 and len(query_segments) <= 1

    candidate_name = str(candidate.get("name") or "")
    name = candidate_name.lower()
    qualified_name = str(candidate.get("qualified_name") or "").lower()
    container_qualified_name = str(candidate.get("container_qualified_name") or "").lower()
    path = str(candidate.get("path") or "").lower()
    kind = str(candidate.get("kind") or "").lower()
    visibility = str(candidate.get("visibility") or "").lower()
    container_kind = str(candidate.get("_container_kind") or "").lower()
    semantic_summary = candidate.get("semantic_summary", {}) or {}

    qname_segments = split_symbol_query_segments(qualified_name)
    container_segments = split_symbol_query_segments(container_qualified_name)

    haystack = " ".join(part for part in (name, qualified_name, container_qualified_name, path, kind) if part)
    score = float(candidate.get("_search_score") or 0.0)

    if symbol_query.startswith("sym:") and str(candidate.get("symbol_id") or "") == symbol_query:
        score += 1000.0
    if lowered_query == qualified_name:
        score += 400.0
    if lowered_query == name:
        score += 250.0
    if qualified_name_endswith_query(qualified_name, lowered_query):
        score += 180.0
    if lowered_query and lowered_query in qualified_name:
        score += 80.0
    if lowered_query and lowered_query in container_qualified_name:
        score += 24.0
    if lowered_query and lowered_query in path:
        score += 30.0

    if len(query_segments) >= 2:
        tail = query_segments[-1]
        prefix_segments = query_segments[:-1]

        if tail and name == tail:
            score += 35.0

        if candidate_matches_query_segments(query_segments, qname_segments):
            score += 180.0
        elif prefix_segments and candidate_matches_query_segments(prefix_segments, qname_segments) and name == tail:
            score += 120.0
        elif prefix_segments and candidate_matches_query_segments(prefix_segments, container_segments) and name == tail:
            score += 100.0
        elif name == tail:
            score -= 20.0

    token_hits = 0
    for token in query_tokens:
        token_hit = False
        if token and token in qualified_name:
            score += 18.0
            token_hit = True
        elif token and token in name:
            score += 15.0
            token_hit = True
        elif token and token in container_qualified_name:
            score += 10.0
            token_hit = True
        elif token and token in path:
            score += 8.0
            token_hit = True
        elif token and token in haystack:
            score += 6.0
            token_hit = True
        if token_hit:
            token_hits += 1

    if query_tokens and token_hits == len(query_tokens):
        score += 40.0
    elif query_tokens:
        score -= (len(query_tokens) - token_hits) * 8.0

    if single_token_query and looks_like_nominal_symbol_query(raw_query):
        if candidate_name == raw_query and kind in NOMINAL_SYMBOL_KINDS:
            score += 180.0
        elif name == lowered_query and kind in NOMINAL_SYMBOL_KINDS:
            score += 120.0
        elif kind in CALLABLE_SYMBOL_KINDS:
            score -= 110.0
        if candidate_name and candidate_name[:1].islower():
            score -= 50.0

    if single_token_query:
        activity_score = semantic_activity_score(semantic_summary)
        score += activity_score
        if visibility == "pub":
            score += 18.0
        if container_kind == "trait":
            score -= 60.0
        elif container_kind == "impl":
            score += 14.0
        if path.startswith("examples/") or path.startswith("tests/"):
            score -= 80.0
        elif path.startswith("metrics/"):
            score -= 45.0
        if kind in CALLABLE_SYMBOL_KINDS:
            score += 10.0

    if kind in LOCAL_SYMBOL_KINDS and not query_explicitly_targets_local(symbol_query):
        score -= 150.0

    return score
def looks_like_nominal_symbol_query(symbol_query: str) -> bool:
    raw = str(symbol_query or "").strip()
    if not raw or "::" in raw or "(" in raw or raw.startswith("sym:"):
        return False
    return raw[:1].isupper() and any(char.islower() for char in raw)


def query_explicitly_targets_local(symbol_query: str) -> bool:
    lowered = str(symbol_query or "").strip().lower()
    if not lowered:
        return False
    if lowered.startswith("sym:"):
        return True
    if "@l" in lowered:
        return True
    return "::" in lowered and "@" in lowered


def qualified_name_endswith_query(qualified_name: str, lowered_query: str) -> bool:
    if not qualified_name or not lowered_query:
        return False
    qualified_name = qualified_name.lower()
    return qualified_name == lowered_query or qualified_name.endswith(f"::{lowered_query}")


def split_symbol_query_segments(value: str) -> List[str]:
    raw = str(value or "").strip().lower()
    if not raw:
        return []
    parts = []
    for segment in raw.split("::"):
        normalized = "".join(char for char in segment if char.isalnum() or char == "_")
        if normalized:
            parts.append(normalized)
    return parts


def candidate_matches_query_segments(query_segments: Sequence[str], candidate_segments: Sequence[str]) -> bool:
    if not query_segments or not candidate_segments:
        return False
    if len(candidate_segments) < len(query_segments):
        return False

    window = len(query_segments)
    for start in range(0, len(candidate_segments) - window + 1):
        if list(candidate_segments[start : start + window]) == list(query_segments):
            return True
    return False


def select_public_symbol_candidates(
    symbol_query: str,
    candidates: Sequence[Dict[str, object]],
    selected: Optional[Dict[str, object]],
    resolution: Optional[str],
    *,
    limit: int,
    exact_qname: Sequence[Dict[str, object]] = (),
    exact_name: Sequence[Dict[str, object]] = (),
    suffix_matches: Sequence[Dict[str, object]] = (),
    segment_matches: Sequence[Dict[str, object]] = (),
) -> List[Dict[str, object]]:
    cap = max(1, min(limit, 8))
    selected_id = str((selected or {}).get("symbol_id") or "")
    selected_name = str((selected or {}).get("name") or "").lower()
    top_score = float((selected or {}).get("_candidate_score") or 0.0)

    if resolution == "exact_qualified_name":
        pool = list(exact_qname)
    elif resolution == "exact_name":
        pool = list(exact_name)
    elif resolution in {"qualified_name_suffix", "qualified_name_segments"}:
        pool = list(suffix_matches or segment_matches)
    elif resolution == "single_candidate":
        pool = [selected] if selected else []
    elif resolution == "strong_top_candidate":
        pool = []
        for candidate in candidates:
            candidate_score = float(candidate.get("_candidate_score") or 0.0)
            candidate_name = str(candidate.get("name") or "").lower()
            candidate_id = str(candidate.get("symbol_id") or "")
            if selected_name and candidate_id != selected_id and candidate_name != selected_name:
                continue
            if top_score and candidate_score < (top_score - 30.0):
                continue
            pool.append(candidate)
        if not pool and selected is not None:
            pool = [selected]
    else:
        pool = list(candidates[:cap])

    if selected is not None and selected_id and not any(str(candidate.get("symbol_id") or "") == selected_id for candidate in pool):
        pool.insert(0, selected)

    focused: List[Dict[str, object]] = []
    seen = set()
    for candidate in pool:
        candidate_id = str(candidate.get("symbol_id") or "")
        if not candidate_id or candidate_id in seen:
            continue
        seen.add(candidate_id)

        candidate_name = str(candidate.get("name") or "").lower()
        if resolution in {"exact_qualified_name", "qualified_name_suffix", "qualified_name_segments", "single_candidate"}:
            if selected_id and candidate_id != selected_id:
                continue
        elif resolution in {"exact_name", "strong_top_candidate"}:
            if selected_name and candidate_id != selected_id and candidate_name != selected_name:
                continue

        focused.append(strip_candidate_score(candidate))
        if len(focused) >= cap:
            break

    if focused:
        return focused
    if selected is not None:
        return [strip_candidate_score(selected)]
    return [strip_candidate_score(candidate) for candidate in candidates[:cap]]


def strip_candidate_score(candidate: Dict[str, object]) -> Dict[str, object]:
    stripped = dict(candidate)
    stripped.pop("_candidate_score", None)
    stripped.pop("_search_score", None)
    return stripped


def resolve_symbol_query(search_root: Path, parsed_root: Path, repo_name: str, symbol_query: str) -> Optional[Dict[str, object]]:
    metadata_store = get_metadata_store(str(parsed_root.resolve()), repo_name)
    if symbol_query.startswith("sym:"):
        return metadata_store.get_symbol(symbol_query)

    qname_matches = metadata_store.resolve_qname(symbol_query)
    if qname_matches:
        return metadata_store.get_symbol(qname_matches[0])

    name_matches = metadata_store.resolve_name(symbol_query, repo=repo_name)
    if name_matches:
        return metadata_store.get_symbol(name_matches[0])

    ranked_candidates = resolve_symbol_candidates(search_root, parsed_root, repo_name, symbol_query, limit=5)
    if ranked_candidates:
        symbol_id = str(ranked_candidates[0].get("symbol_id") or "")
        if symbol_id:
            return metadata_store.get_symbol(symbol_id)

    matches = where_defined(search_root, parsed_root, repo_name, symbol_query, limit=1)["matches"]
    if not matches:
        return None
    symbol_id = matches[0]["symbol_id"]
    if not symbol_id:
        return None
    return metadata_store.get_symbol(symbol_id)


def describe_symbol_row(symbol: Optional[Dict[str, object]]) -> Optional[Dict[str, object]]:
    if symbol is None:
        return None
    return {
        "symbol_id": symbol["symbol_id"],
        "name": symbol["name"],
        "qualified_name": symbol["qualified_name"],
        "kind": symbol["kind"],
        "path": symbol["path"],
        "signature": symbol.get("signature"),
        "summary_id": symbol.get("summary_id"),
        "normalized_body_hash": symbol.get("normalized_body_hash"),
        "container_symbol_id": symbol.get("container_symbol_id"),
        "container_qualified_name": symbol.get("container_qualified_name"),
    }


def stable_repo_id(repo_name: str) -> str:
    from symbols.indexer import stable_id

    return stable_id("repo", repo_name)


def stable_file_id(repo_name: str, path: str) -> str:
    from symbols.indexer import stable_id

    return stable_id("file", repo_name, path)


def stable_directory_id(repo_name: str, path: str) -> str:
    from symbols.indexer import stable_id

    return stable_id("dir", repo_name, path)


def stable_package_id(repo_name: str, package_name: str) -> str:
    from symbols.indexer import stable_id

    return stable_id("pkg", repo_name, package_name)


def parent_paths(path: str) -> List[str]:
    raw = str(path or "").strip("/")
    if not raw:
        return ["."]
    parts = raw.split("/")
    return ["/".join(parts[:index]) for index in range(len(parts), 0, -1)]
