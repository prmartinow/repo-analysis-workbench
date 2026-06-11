from __future__ import annotations

from pathlib import PurePosixPath, Path
from typing import Dict, List, Optional, Sequence, Tuple

from backends.graph_backend import get_graph_backend
from backends.metadata_store import get_metadata_store
from backends.search_backend import get_search_backend
from common.retrieval import (
    BODY_STAGE_KINDS,
    EMBEDDING_MIN_TOKENS,
    EXACT_SYMBOL_MAX_TOKENS,
    GRAPH_EDGE_BONUS,
    GRAPH_STAGE_EDGE_TYPES,
    MAX_BODY_SEEDS,
    MAX_SUMMARY_SCOPES,
    MAX_SYMBOL_SEEDS,
    SUMMARY_FANOUT_MULTIPLIER,
    SUMMARY_STAGE_KINDS,
    SYMBOL_FANOUT_MULTIPLIER,
    SYMBOL_STAGE_KINDS,
)
from common.telemetry import trace_operation
from common.text import tokenize
from embeddings.indexer import query_embedding_index
from rerank.fusion import rerank_candidates
from symbols.indexer import stable_id


DOC_HINTS = {"doc", "docs", "documentation", "guide", "guides", "overview", "readme", "summary"}
GRAPH_HINTS = {
    "callers",
    "caller",
    "callees",
    "callee",
    "references",
    "reference",
    "refs",
    "implements",
    "implementations",
    "imports",
    "neighbors",
    "dependency",
    "dependencies",
    "path",
}
BODY_HINTS = {"body", "implementation", "code", "logic", "source"}
STATEMENT_HINTS = {"statement", "statements", "line", "lines", "expr", "let", "local", "locals"}
PATH_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".go",
    ".java",
    ".js",
    ".json",
    ".md",
    ".py",
    ".rb",
    ".rs",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}
GRAPH_NOISE_KINDS = {
    "statement",
    "symbol_ref",
    "module_ref",
    "type_ref",
    "trait_ref",
}


def retrieve_context(
    search_root: Path,
    graph_root: Path,
    parsed_root: Path,
    repo_name: str,
    query: str,
    *,
    limit: int = 8,
    depth: int = 1,
    kinds: Sequence[str] = (),
    use_graph: bool = True,
    use_embeddings: bool = True,
    use_rerank: bool = True,
    use_summaries: bool = False,
    selective_retrieval: bool = True,
    max_graph_fanout: int = 32,
) -> Dict[str, object]:
    with trace_operation("retrieve_context"):
        search_backend = get_search_backend(str(search_root.resolve()), repo_name)
        graph_backend = get_graph_backend(str(graph_root.resolve()), repo_name)
        metadata_store = get_metadata_store(str(parsed_root.resolve()), repo_name)

        tokens = tokenize(query)
        query_profile = classify_query(tokens, query)
        requested_kinds = {str(kind) for kind in kinds if str(kind)}

        summary_kinds = stage_kinds(SUMMARY_STAGE_KINDS, requested_kinds)
        symbol_kinds = stage_kinds(SYMBOL_STAGE_KINDS, requested_kinds)
        body_kinds = stage_kinds(BODY_STAGE_KINDS, requested_kinds)

        with trace_operation("retrieve_context.summary_search"):
            summary_results = (
                search_backend.search(
                    query,
                    limit=max(limit * SUMMARY_FANOUT_MULTIPLIER, 12),
                    kinds=summary_kinds,
                )
                if summary_kinds
                else []
            )

        summary_scopes = derive_summary_scopes(
            query,
            summary_results,
            max_scopes=max(MAX_SUMMARY_SCOPES, 1),
        )
        exact_symbol_results = resolve_exact_symbol_results(repo_name, metadata_store, query)
        gate = retrieval_gate(
            query,
            tokens,
            query_profile,
            summary_results,
            exact_symbol_results,
            selective_retrieval=selective_retrieval,
        )

        symbol_results: List[Dict[str, object]] = []
        with trace_operation("retrieve_context.symbol_search"):
            for result in exact_symbol_results:
                annotate_scope(result, summary_scopes)
                symbol_results.append(result)

            if symbol_kinds:
                if query_profile["intent"] == "exploration" and summary_scopes:
                    scope_symbol_results = enumerate_scope_symbols(
                        search_backend,
                        summary_scopes,
                        limit=max(limit * 3, 12),
                    )
                    for result in scope_symbol_results:
                        annotate_scope(result, summary_scopes)
                        symbol_results.append(result)

                scoped_results = search_scoped_symbol_results(
                    search_backend,
                    query,
                    symbol_kinds=symbol_kinds,
                    summary_scopes=summary_scopes,
                    limit=max(limit * SYMBOL_FANOUT_MULTIPLIER, 16),
                )
                for result in scoped_results:
                    annotate_scope(result, summary_scopes)
                    symbol_results.append(result)

                if gate["use_global_symbol_search"]:
                    global_results = search_backend.search(
                        query,
                        limit=max(limit * SYMBOL_FANOUT_MULTIPLIER, 16),
                        kinds=symbol_kinds,
                    )
                    for result in global_results:
                        annotate_scope(result, summary_scopes)
                        symbol_results.append(result)

        localized_symbol_results = (
            rerank_candidates(symbol_results, tokens, query_profile=query_profile)
            if use_rerank
            else sort_candidates(symbol_results)
        )

        candidates: Dict[Tuple[str, str], Dict[str, object]] = {}
        add_stage_results(candidates, summary_results, stage_reason="summary")
        add_stage_results(candidates, localized_symbol_results, stage_reason="symbol")

        graph_results: List[Dict[str, object]] = []
        body_results: List[Dict[str, object]] = []
        embedding_results: List[Dict[str, object]] = []

        seed_nodes = build_seed_nodes(localized_symbol_results, limit=max(MAX_SYMBOL_SEEDS, 1))

        with trace_operation("retrieve_context.graph_expansion"):
            if use_graph and gate["use_graph"] and seed_nodes:
                graph_results = expand_graph_candidates(
                    graph_backend,
                    repo_name,
                    seed_nodes,
                    depth=max(depth, 1),
                    max_fanout=max_graph_fanout,
                )
                if not query_profile["explicit_statement"]:
                    graph_results = [item for item in graph_results if str(item.get("kind") or "") not in GRAPH_NOISE_KINDS]
                for result in graph_results:
                    annotate_scope(result, summary_scopes)

        with trace_operation("retrieve_context.metadata_hydration"):
            for candidate in graph_results:
                hydrate_symbol_candidate(candidate, metadata_store)
            add_stage_results(candidates, graph_results, stage_reason="graph")

            if body_kinds and gate["use_body"]:
                body_results = hydrate_body_results(
                    search_backend,
                    localized_symbol_results,
                    limit=max(MAX_BODY_SEEDS, 1),
                    body_kinds=body_kinds,
                    summary_scopes=summary_scopes,
                )
                add_stage_results(candidates, body_results, stage_reason="body")

            if use_embeddings and gate["use_embeddings"]:
                embedding_results = query_embedding_index(search_root, repo_name, query, limit=max(limit, 5))
                for result in embedding_results:
                    annotate_scope(result, summary_scopes)
                add_stage_results(candidates, embedding_results, stage_reason="embedding")

        if use_summaries:
            with trace_operation("retrieve_context.summary_enrichment"):
                annotate_summary_relevance(candidates, metadata_store, tokens)

        ranked_candidates = list(candidates.values())
        ranked = (
            rerank_candidates(ranked_candidates, tokens, query_profile=query_profile)[:limit]
            if use_rerank
            else sort_candidates(ranked_candidates)[:limit]
        )

        return {
            "repo": repo_name,
            "query": query,
            "summary_results": summary_results[:limit],
            "symbol_results": localized_symbol_results[:limit],
            "graph_results": sort_candidates(graph_results)[:limit],
            "body_results": sort_candidates(body_results)[:limit],
            "embedding_results": sort_candidates(embedding_results)[:limit],
            "lexical_results": localized_symbol_results[:limit],
            "selected_context": ranked,
            "summary": {
                "summary_results": len(summary_results),
                "symbol_results": len(symbol_results),
                "graph_results": len(graph_results),
                "body_results": len(body_results),
                "embedding_results": len(embedding_results),
                "selected": len(ranked),
                "graph_enabled": bool(use_graph and gate["use_graph"]),
                "embeddings_enabled": bool(use_embeddings and gate["use_embeddings"]),
                "body_enabled": bool(gate["use_body"]),
                "rerank_enabled": use_rerank,
                "summaries_enabled": bool(use_summaries),
                "retrieval_gate": gate,
                "query_profile": query_profile,
                "summary_scopes": [item["path_prefix"] for item in summary_scopes],
            },
        }


def stage_kinds(default_kinds: Sequence[object], requested_kinds: set[str]) -> Tuple[str, ...]:
    normalized = tuple(str(kind) for kind in default_kinds if str(kind))
    if not requested_kinds:
        return normalized
    return tuple(kind for kind in normalized if kind in requested_kinds)


def derive_summary_scopes(
    query: str,
    summary_results: Sequence[Dict[str, object]],
    *,
    max_scopes: int,
) -> List[Dict[str, object]]:
    scopes: List[Dict[str, object]] = []
    seen = set()
    normalized_query = str(query or "").strip().lower()
    for result in summary_results:
        path = str(result.get("path") or "").strip()
        if not path:
            continue
        prefix = path
        if result.get("kind") in {"file", "doc"} and normalized_query != path.lower():
            prefix = str(PurePosixPath(path).parent)
            if prefix == ".":
                prefix = path
        prefix = prefix.strip()
        if not prefix or prefix in seen:
            continue
        seen.add(prefix)
        scopes.append(
            {
                "path_prefix": prefix,
                "kind": str(result.get("kind") or ""),
                "path": path,
                "score": float(result.get("score") or 0.0),
            }
        )
        if len(scopes) >= max_scopes:
            break
    return scopes


def resolve_exact_symbol_results(repo_name: str, metadata_store: object, query: str) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    seen = set()

    symbol_ids: List[str] = []
    if query.startswith("sym:"):
        symbol_ids.append(query)
    symbol_ids.extend(metadata_store.resolve_qname(query))
    symbol_ids.extend(metadata_store.resolve_name(query, repo=repo_name))

    for index, symbol_id in enumerate(symbol_ids):
        normalized_id = str(symbol_id or "")
        if not normalized_id or normalized_id in seen:
            continue
        seen.add(normalized_id)
        symbol = metadata_store.get_symbol(normalized_id)
        if symbol is None:
            continue
        results.append(symbol_record_to_result(repo_name, symbol, score=420.0 - (index * 20.0), reason="symbol-localization"))
    return results


def symbol_record_to_result(repo_name: str, symbol: Dict[str, object], *, score: float, reason: str) -> Dict[str, object]:
    symbol_id = str(symbol.get("symbol_id") or "")
    return {
        "doc_id": f"symbol:{symbol_id}",
        "kind": "symbol",
        "repo": repo_name,
        "path": symbol.get("path"),
        "name": symbol.get("name"),
        "qualified_name": symbol.get("qualified_name"),
        "symbol_id": symbol_id,
        "title": symbol.get("qualified_name") or symbol.get("name"),
        "preview": symbol.get("signature") or symbol.get("qualified_name") or symbol.get("name"),
        "score": float(score),
        "metadata": {
            "kind": symbol.get("kind"),
            "visibility": symbol.get("visibility"),
            "container_qualified_name": symbol.get("container_qualified_name"),
            "semantic_summary": symbol.get("semantic_summary", {}),
        },
        "reasons": [reason],
    }


def search_scoped_symbol_results(
    search_backend: object,
    query: str,
    *,
    symbol_kinds: Sequence[str],
    summary_scopes: Sequence[Dict[str, object]],
    limit: int,
) -> List[Dict[str, object]]:
    merged: Dict[str, Dict[str, object]] = {}
    for scope in summary_scopes:
        scope_prefix = str(scope.get("path_prefix") or "").strip()
        if not scope_prefix:
            continue
        for result in search_backend.search(
            query,
            limit=limit,
            kinds=symbol_kinds,
            path_prefix=scope_prefix,
        ):
            doc_id = candidate_identity(result)
            existing = merged.get(doc_id)
            if existing is None or float(result.get("score") or 0.0) > float(existing.get("score") or 0.0):
                merged[doc_id] = dict(result)
    return list(merged.values())


def enumerate_scope_symbols(
    search_backend: object,
    summary_scopes: Sequence[Dict[str, object]],
    *,
    limit: int,
) -> List[Dict[str, object]]:
    merged: Dict[str, Dict[str, object]] = {}
    remaining = max(limit, 1)
    for scope in summary_scopes:
        if remaining <= 0:
            break
        scope_prefix = str(scope.get("path_prefix") or "").strip()
        if not scope_prefix:
            continue

        scoped_docs = search_backend.list_documents(
            limit=min(remaining, max(limit, 12)),
            kinds=("symbol",),
            path_prefix=scope_prefix,
        )
        scope_score = float(scope.get("score") or 0.0)
        for index, result in enumerate(scoped_docs):
            doc_id = candidate_identity(result)
            candidate = dict(result)
            candidate["score"] = max(float(candidate.get("score") or 0.0), scope_score + 12.0 - (index * 0.35))
            existing = merged.get(doc_id)
            if existing is None or float(candidate.get("score") or 0.0) > float(existing.get("score") or 0.0):
                merged[doc_id] = candidate
        remaining -= len(scoped_docs)
    return list(merged.values())


def build_seed_nodes(symbol_results: Sequence[Dict[str, object]], *, limit: int) -> List[Tuple[str, float]]:
    seeds: List[Tuple[str, float]] = []
    seen = set()
    for result in sort_candidates(symbol_results):
        node_id = result_to_node_id(str(result.get("repo") or ""), result)
        if not node_id or node_id in seen:
            continue
        seen.add(node_id)
        seeds.append((node_id, float(result.get("score") or 0.0)))
        if len(seeds) >= limit:
            break
    return seeds


def hydrate_body_results(
    search_backend: object,
    symbol_results: Sequence[Dict[str, object]],
    *,
    limit: int,
    body_kinds: Sequence[str],
    summary_scopes: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    hydrated: List[Dict[str, object]] = []
    seen = set()
    for symbol_result in sort_candidates(symbol_results):
        symbol_id = str(symbol_result.get("symbol_id") or "")
        if not symbol_id or symbol_id in seen:
            continue
        seen.add(symbol_id)
        docs = search_backend.lookup_symbol_docs(symbol_id, kinds=body_kinds, limit=1)
        for item in docs:
            hydrated_item = dict(item)
            annotate_scope(hydrated_item, summary_scopes)
            metadata = dict(hydrated_item.get("metadata", {}) or {})
            metadata.setdefault("parent_symbol_id", symbol_id)
            hydrated_item["metadata"] = metadata
            hydrated_item["reasons"] = list(dict.fromkeys([*symbol_result.get("reasons", []), "body"]))
            hydrated.append(hydrated_item)
            break
        if len(hydrated) >= limit:
            break
    return hydrated


def result_to_node_id(repo_name: str, result: Dict[str, object]) -> Optional[str]:
    if result.get("symbol_id"):
        return str(result["symbol_id"])
    if result.get("kind") == "file" and result.get("path"):
        return stable_id("file", repo_name, result["path"])
    return None


def expand_graph_candidates(
    graph_backend: object,
    repo_name: str,
    seed_nodes: Sequence[Tuple[str, float]],
    depth: int,
    *,
    max_fanout: int,
) -> List[Dict[str, object]]:
    expanded = []
    for node_id, base_score in seed_nodes:
        response = graph_backend.execute(
            {
                "operation": "neighbors",
                "seed": {"node_id": node_id},
                "edge_types": GRAPH_STAGE_EDGE_TYPES,
                "direction": "both",
                "depth": max(depth, 1),
                "limit": max(max_fanout, 1),
            }
        )
        for item in (response or {}).get("results", []):
            edge = dict(item.get("edge") or {})
            edge_type = str(edge.get("type") or item.get("edge_type") or "NEIGHBOR")
            direction = str(item.get("direction") or edge.get("metadata", {}).get("direction") or "both")
            graph_depth = int(item.get("depth") or 1)
            score = round(base_score + GRAPH_EDGE_BONUS.get(edge_type, 0.5) - (graph_depth - 1) * 0.35, 6)
            expanded.append(node_to_candidate(repo_name, item, score, edge, direction, graph_depth))
    return expanded


def node_to_candidate(
    repo_name: str,
    node: Dict[str, object],
    score: float,
    edge: Dict[str, object],
    direction: str,
    graph_depth: int,
) -> Dict[str, object]:
    edge_id = str(edge.get("edge_id") or stable_id("edge", repo_name, node.get("node_id") or "unknown"))
    edge_type = str(edge.get("type") or node.get("edge_type") or "NEIGHBOR")
    kind = str(node.get("kind") or "unknown")
    if "path" in node and kind not in {
        "statement",
        "file",
        "directory",
        "repository",
        "package",
        "dependency",
        "test",
        "project_summary",
        "package_summary",
        "directory_summary",
        "file_summary",
        "symbol_summary",
        "module_ref",
        "symbol_ref",
        "type_ref",
        "trait_ref",
    }:
        kind = "symbol"

    return {
        "doc_id": stable_id("cand", repo_name, node["node_id"], edge_id, direction),
        "kind": kind,
        "repo": repo_name,
        "path": node.get("path"),
        "name": node.get("name"),
        "qualified_name": node.get("qualified_name"),
        "symbol_id": node["node_id"] if kind == "symbol" and str(node["node_id"]).startswith("sym:") else None,
        "title": node.get("qualified_name") or node.get("path") or node.get("name"),
        "preview": f"{edge_type} via {direction}",
        "score": score,
        "metadata": {
            "node_kind": node.get("kind"),
            "edge_type": edge_type,
            "direction": direction,
            "graph_distance": graph_depth,
        },
        "reasons": [f"graph:{edge_type}:{direction}"],
    }


def add_stage_results(
    candidates: Dict[Tuple[str, str], Dict[str, object]],
    results: Sequence[Dict[str, object]],
    *,
    stage_reason: str,
) -> None:
    for result in results:
        candidate = dict(result)
        reasons = list(candidate.get("reasons", []))
        if stage_reason not in reasons:
            reasons.append(stage_reason)
        candidate["reasons"] = reasons
        add_candidate(candidates, candidate)


def add_candidate(candidates: Dict[Tuple[str, str], Dict[str, object]], candidate: Dict[str, object]) -> None:
    key = (str(candidate.get("kind") or "unknown"), candidate_identity(candidate))
    existing = candidates.get(key)
    if not existing:
        candidates[key] = candidate
        return

    existing["score"] = max(float(existing.get("score") or 0.0), float(candidate.get("score") or 0.0))
    existing_reasons = list(existing.get("reasons", []))
    for reason in candidate.get("reasons", []):
        if reason not in existing_reasons:
            existing_reasons.append(reason)
    existing["reasons"] = existing_reasons
    existing_metadata = dict(existing.get("metadata", {}) or {})
    existing_metadata.update(candidate.get("metadata", {}) or {})
    existing["metadata"] = existing_metadata
    for field in ("preview", "name", "qualified_name", "path", "symbol_id", "title"):
        if not existing.get(field) and candidate.get(field):
            existing[field] = candidate[field]


def candidate_identity(candidate: Dict[str, object]) -> str:
    return str(candidate.get("symbol_id") or candidate.get("path") or candidate.get("doc_id") or "")


def retrieval_gate(
    query: str,
    tokens: Sequence[str],
    query_profile: Dict[str, object],
    summary_results: Sequence[Dict[str, object]],
    exact_symbol_results: Sequence[Dict[str, object]],
    *,
    selective_retrieval: bool,
) -> Dict[str, object]:
    if not selective_retrieval:
        return {
            "enabled": False,
            "reason": "disabled",
            "use_global_symbol_search": True,
            "use_graph": True,
            "use_body": True,
            "use_embeddings": True,
        }

    exact_symbol_hit = has_exact_symbol_hit(query, exact_symbol_results)
    exact_path_hit = has_exact_path_hit(query, summary_results)
    token_count = len(tokens)

    if query_profile["explicit_docs"] or exact_path_hit:
        return {
            "enabled": True,
            "reason": "path-or-docs",
            "use_global_symbol_search": False,
            "use_graph": False,
            "use_body": False,
            "use_embeddings": False,
        }

    if exact_symbol_hit and token_count <= EXACT_SYMBOL_MAX_TOKENS:
        return {
            "enabled": True,
            "reason": "exact-symbol",
            "use_global_symbol_search": False,
            "use_graph": bool(query_profile["explicit_graph"]),
            "use_body": bool(query_profile["explicit_body"]),
            "use_embeddings": False,
        }

    if query_profile["explicit_statement"]:
        return {
            "enabled": True,
            "reason": "statement-query",
            "use_global_symbol_search": True,
            "use_graph": True,
            "use_body": True,
            "use_embeddings": False,
        }

    exploratory = query_profile["intent"] == "exploration"
    return {
        "enabled": True,
        "reason": "exploration" if exploratory else "generic",
        "use_global_symbol_search": not bool(summary_results) or query_profile["explicit_symbol"],
        "use_graph": exploratory or query_profile["explicit_graph"] or bool(exact_symbol_results),
        "use_body": exploratory or query_profile["explicit_body"] or bool(exact_symbol_results),
        "use_embeddings": exploratory and token_count >= EMBEDDING_MIN_TOKENS and not summary_results[:1],
    }


def classify_query(tokens: Sequence[str], query: str) -> Dict[str, object]:
    token_set = {token.lower() for token in tokens}
    stripped = str(query or "").strip()
    lowered_query = stripped.lower()
    explicit_docs = bool(token_set.intersection(DOC_HINTS))
    explicit_graph = bool(token_set.intersection(GRAPH_HINTS))
    explicit_body = bool(token_set.intersection(BODY_HINTS))
    explicit_statement = bool(token_set.intersection(STATEMENT_HINTS)) or "@l" in lowered_query
    explicit_path = looks_like_path_query(stripped)
    explicit_symbol = stripped.startswith("sym:") or "::" in stripped or (
        len(token_set) <= 2 and not explicit_docs and not explicit_path and not explicit_statement and not explicit_graph
    )

    if explicit_docs or explicit_path:
        intent = "docs"
    elif explicit_symbol:
        intent = "symbol"
    else:
        intent = "exploration"

    return {
        "intent": intent,
        "explicit_docs": explicit_docs,
        "explicit_graph": explicit_graph,
        "explicit_body": explicit_body,
        "explicit_statement": explicit_statement,
        "explicit_path": explicit_path,
        "explicit_symbol": explicit_symbol,
    }


def looks_like_path_query(query: str) -> bool:
    if not query:
        return False
    if "/" in query:
        return True
    suffix = PurePosixPath(query).suffix.lower()
    return suffix in PATH_EXTENSIONS


def has_exact_symbol_hit(query: str, results: Sequence[Dict[str, object]]) -> bool:
    lowered_query = str(query or "").strip().lower()
    if not lowered_query:
        return False
    for result in results:
        name = str(result.get("name") or "").lower()
        qualified_name = str(result.get("qualified_name") or "").lower()
        if lowered_query == name or lowered_query == qualified_name or qualified_name.endswith(f"::{lowered_query}"):
            return True
    return False


def has_exact_path_hit(query: str, results: Sequence[Dict[str, object]]) -> bool:
    normalized_query = str(query or "").strip().lower()
    if not normalized_query:
        return False
    for result in results[:3]:
        path = str(result.get("path") or "").strip().lower()
        if path and path == normalized_query:
            return True
    return False


def annotate_scope(candidate: Dict[str, object], summary_scopes: Sequence[Dict[str, object]]) -> None:
    path = str(candidate.get("path") or "").strip()
    if not path or not summary_scopes:
        return

    best_depth: Optional[int] = None
    best_prefix: Optional[str] = None
    for scope in summary_scopes:
        prefix = str(scope.get("path_prefix") or "").strip()
        if not prefix:
            continue
        if path == prefix:
            depth = 0
        elif path.startswith(prefix.rstrip("/") + "/"):
            depth = max(path.count("/") - prefix.count("/"), 1)
        else:
            continue
        if best_depth is None or depth < best_depth:
            best_depth = depth
            best_prefix = prefix

    if best_depth is None:
        return
    metadata = dict(candidate.get("metadata", {}) or {})
    metadata["containment_depth"] = best_depth
    metadata["scope_prefix"] = best_prefix
    candidate["metadata"] = metadata


def hydrate_symbol_candidate(candidate: Dict[str, object], metadata_store: object) -> None:
    symbol_id = str(candidate.get("symbol_id") or "")
    if not symbol_id:
        return
    symbol = metadata_store.get_symbol(symbol_id)
    if symbol is None:
        return
    candidate.setdefault("preview", symbol.get("signature") or symbol.get("qualified_name"))
    candidate["name"] = symbol.get("name")
    candidate["qualified_name"] = symbol.get("qualified_name")
    candidate["path"] = symbol.get("path")
    metadata = dict(candidate.get("metadata", {}) or {})
    metadata["kind"] = symbol.get("kind")
    metadata["visibility"] = symbol.get("visibility")
    metadata["container_qualified_name"] = symbol.get("container_qualified_name")
    metadata["semantic_summary"] = symbol.get("semantic_summary", {})
    candidate["metadata"] = metadata


def annotate_summary_relevance(
    candidates: Dict[Tuple[str, str], Dict[str, object]],
    metadata_store: object,
    tokens: Sequence[str],
) -> None:
    normalized_tokens = tuple(dict.fromkeys(token.lower() for token in tokens if token))
    if not normalized_tokens:
        return

    for candidate in candidates.values():
        summary_text = summary_text_for_candidate(candidate, metadata_store)
        if not summary_text:
            continue
        haystack = summary_text.lower()
        overlap = sum(1 for token in normalized_tokens if token in haystack)
        if overlap <= 0:
            continue
        metadata = dict(candidate.get("metadata", {}) or {})
        metadata["summary_relevance"] = overlap / max(len(normalized_tokens), 1)
        candidate["metadata"] = metadata
        reasons = list(candidate.get("reasons", []))
        if "summary" not in reasons:
            reasons.append("summary")
        candidate["reasons"] = reasons


def summary_text_for_candidate(candidate: Dict[str, object], metadata_store: object) -> str:
    pieces: List[str] = []
    if candidate.get("kind") in {"repo", "package", "directory", "file", "doc"}:
        pieces.extend(
            str(candidate.get(field) or "")
            for field in ("title", "preview", "searchable")
            if candidate.get(field)
        )
    path = str(candidate.get("path") or "")
    if path:
        for summary in metadata_store.get_summary_by_path(path):
            pieces.append(str(summary.get("summary") or ""))
            pieces.append(str(summary.get("focus") or ""))
    symbol_id = str(candidate.get("symbol_id") or "")
    if symbol_id:
        for summary in metadata_store.get_summary_by_symbol(symbol_id):
            pieces.append(str(summary.get("summary") or ""))
            pieces.append(str(summary.get("focus") or ""))
    return " ".join(piece for piece in pieces if piece).strip()


def sort_candidates(candidates: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    return sorted(
        candidates,
        key=lambda item: (
            -float(item.get("score") or 0.0),
            str(item.get("path") or ""),
            str(item.get("qualified_name") or item.get("title") or ""),
        ),
    )
