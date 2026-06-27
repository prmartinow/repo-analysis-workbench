from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from backends.search_backend import get_search_backend
from common.text import tokenize
from embeddings.indexer import query_embedding_index
from embeddings.units import aggregation_key_for, build_retrieval_units
from rerank.fusion import (
    HEURISTIC_RERANK_PROVIDERS,
    QWEN_RERANK_MAX_CANDIDATES,
    QWEN_RERANK_PROVIDERS,
    allow_heuristic_rerank_fallback,
    candidate_to_rerank_document,
    qwen_rerank,
    rerank_candidates,
)


DEFAULT_LEXICAL_POOL = 1000
DEFAULT_PRE_RANK_POOL = 100
DEFAULT_RERANK_POOL = 50
MAX_UNITS_PER_SOURCE = 4


def retrieve_paper_pipeline(
    search_root: Path,
    repo_name: str,
    query: str,
    *,
    limit: int = 10,
    lexical_pool: int | None = None,
    pre_rank_pool: int | None = None,
    rerank_pool: int | None = None,
) -> Dict[str, object]:
    lexical_pool_size = lexical_pool or int(os.environ.get("REPO_ANALYSIS_PAPER_LEXICAL_POOL", DEFAULT_LEXICAL_POOL))
    pre_rank_pool_size = pre_rank_pool or int(os.environ.get("REPO_ANALYSIS_PAPER_PRE_RANK_POOL", DEFAULT_PRE_RANK_POOL))
    rerank_pool_size = rerank_pool or int(os.environ.get("REPO_ANALYSIS_PAPER_RERANK_POOL", DEFAULT_RERANK_POOL))

    search_backend = get_search_backend(str(search_root.resolve()), repo_name)
    lexical_results = search_backend.search(query, limit=max(lexical_pool_size, limit))
    embedding_results = query_embedding_index(search_root, repo_name, query, limit=max(pre_rank_pool_size, limit))
    embedding_by_key = build_embedding_score_map(embedding_results)
    embedding_unit_ids = build_embedding_unit_id_set(embedding_results)

    unit_candidates = build_unit_candidates(repo_name, query, lexical_results, embedding_by_key, embedding_unit_ids)
    pre_ranked_units = select_diverse_units(sort_units(unit_candidates), limit=max(pre_rank_pool_size, rerank_pool_size, limit))
    rerank_input = pre_ranked_units[: max(rerank_pool_size, limit)]
    reranked_units = rerank_unit_candidates(rerank_input, query)
    selected = aggregate_units_maxp(reranked_units, repo_name, limit=limit)

    return {
        "repo": repo_name,
        "query": query,
        "selected_context": selected,
        "lexical_results": lexical_results[:limit],
        "embedding_results": embedding_results[:limit],
        "unit_results": reranked_units[:limit],
        "summary": {
            "mode": "paper_pipeline",
            "lexical_pool": lexical_pool_size,
            "lexical_candidates": len(lexical_results),
            "unit_candidates": len(unit_candidates),
            "pre_rank_pool": pre_rank_pool_size,
            "pre_ranked_units": len(pre_ranked_units),
            "rerank_pool": rerank_pool_size,
            "reranked_units": len(reranked_units),
            "selected": len(selected),
            "embedding_aggregation": "maxp",
            "rerank_provider": normalized_rerank_provider(),
        },
    }


def build_unit_candidates(
    repo_name: str,
    query: str,
    lexical_results: Sequence[Dict[str, object]],
    embedding_by_key: Dict[str, Dict[str, object]],
    embedding_unit_ids: set[str],
) -> List[Dict[str, object]]:
    query_tokens = tokenize(query)
    candidates: List[Dict[str, object]] = []
    per_source_counts: Dict[str, int] = {}
    for rank, document in enumerate(lexical_results, start=1):
        normalized_document = lexical_result_to_document(document)
        aggregation_key = aggregation_key_for(normalized_document)
        if per_source_counts.get(aggregation_key, 0) >= MAX_UNITS_PER_SOURCE:
            continue
        embedding_match = embedding_by_key.get(aggregation_key, {})
        for unit in build_retrieval_units(normalized_document):
            if per_source_counts.get(aggregation_key, 0) >= MAX_UNITS_PER_SOURCE:
                break
            candidate = unit_to_candidate(
                repo_name,
                unit,
                source_document=document,
                lexical_rank=rank,
                query_tokens=query_tokens,
                embedding_match=embedding_match,
                embedding_unit_ids=embedding_unit_ids,
            )
            candidates.append(candidate)
            per_source_counts[aggregation_key] = per_source_counts.get(aggregation_key, 0) + 1
    return candidates


def lexical_result_to_document(document: Dict[str, object]) -> Dict[str, object]:
    content = str(document.get("searchable") or document.get("content") or "").strip()
    if not content:
        content = " ".join(
            str(value or "")
            for value in (
                document.get("qualified_name"),
                document.get("name"),
                document.get("path"),
                document.get("title"),
                document.get("preview"),
            )
            if value
        )
    return {
        "doc_id": document.get("doc_id") or f"{document.get('kind')}:{document.get('path')}:{document.get('qualified_name') or document.get('name') or ''}",
        "kind": document.get("kind") or "unknown",
        "path": document.get("path"),
        "name": document.get("name"),
        "qualified_name": document.get("qualified_name"),
        "symbol_id": document.get("symbol_id"),
        "title": document.get("title"),
        "preview": document.get("preview"),
        "content": content,
    }


def unit_to_candidate(
    repo_name: str,
    unit: Dict[str, object],
    *,
    source_document: Dict[str, object],
    lexical_rank: int,
    query_tokens: Sequence[str],
    embedding_match: Dict[str, object],
    embedding_unit_ids: set[str],
) -> Dict[str, object]:
    lexical_score = float(source_document.get("score") or 0.0)
    embedding_score = float(embedding_match.get("score") or 0.0)
    token_overlap = count_token_overlap(query_tokens, unit)
    exact_boost = exact_query_boost(query_tokens, unit)
    path_penalty = generated_or_test_penalty(str(unit.get("path") or ""), query_tokens)
    unit_id = str(unit.get("unit_id") or unit.get("doc_id") or "")
    unit_hit_boost = 2.0 if unit_id in embedding_unit_ids else 0.0
    pre_rank_score = (
        lexical_score
        + embedding_score * 40.0
        + token_overlap * 3.0
        + exact_boost
        + unit_hit_boost
        + path_penalty
        - lexical_rank * 0.002
    )
    metadata = {
        "mode": "paper_pipeline",
        "stage": "unit_pre_rank",
        "lexical_rank": lexical_rank,
        "lexical_score": lexical_score,
        "embedding_score": embedding_score,
        "embedding_unit_matched": unit_id in embedding_unit_ids,
        "pre_rank_score": round(pre_rank_score, 6),
        "embedding_aggregation_key": unit.get("aggregation_key"),
        "embedding_aggregation_kind": unit.get("aggregation_kind"),
        "embedding_unit_id": unit_id,
        "embedding_unit_kind": unit.get("unit_kind"),
        "embedding_unit_text": unit.get("content"),
        "embedding_unit_preview": unit.get("preview"),
        "embedding_unit_start_line": unit.get("start_line"),
        "embedding_unit_end_line": unit.get("end_line"),
        "embedding_unit_token_estimate": unit.get("token_estimate"),
    }
    return {
        "doc_id": unit_id,
        "kind": unit.get("source_kind") or unit.get("kind"),
        "repo": repo_name,
        "path": unit.get("path"),
        "name": unit.get("name"),
        "qualified_name": unit.get("qualified_name"),
        "symbol_id": unit.get("symbol_id"),
        "title": unit.get("title"),
        "preview": unit.get("preview"),
        "score": round(pre_rank_score, 6),
        "metadata": metadata,
        "reasons": ["bm25-large-pool", "retrieval-unit", "embedding-fusion"],
    }


def rerank_unit_candidates(candidates: Sequence[Dict[str, object]], query: str) -> List[Dict[str, object]]:
    provider = normalized_rerank_provider()
    query_tokens = tokenize(query)
    if provider in HEURISTIC_RERANK_PROVIDERS:
        return rerank_candidates(candidates, query_tokens)
    if provider not in QWEN_RERANK_PROVIDERS:
        raise ValueError(f"Unsupported rerank provider: {provider}")
    if not candidates or not query.strip():
        return list(candidates)

    try:
        return qwen_rerank_in_batches(query, candidates)
    except RuntimeError:
        if allow_heuristic_rerank_fallback():
            return rerank_candidates(candidates, query_tokens)
        raise


def qwen_rerank_in_batches(query: str, candidates: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    reranked: List[Dict[str, object]] = []
    for batch_start in range(0, len(candidates), QWEN_RERANK_MAX_CANDIDATES):
        batch = list(candidates[batch_start : batch_start + QWEN_RERANK_MAX_CANDIDATES])
        documents = [candidate_to_rerank_document(candidate) for candidate in batch]
        results = qwen_rerank(query, documents)
        seen = set()
        for result in results:
            index = int(result.get("index", -1))
            if index < 0 or index >= len(batch) or index in seen:
                continue
            seen.add(index)
            qwen_score = float(result.get("score") or 0.0)
            candidate = dict(batch[index])
            metadata = dict(candidate.get("metadata", {}) or {})
            metadata["qwen_rerank_score"] = qwen_score
            metadata["stage"] = "unit_neural_rerank"
            candidate["metadata"] = metadata
            candidate["score"] = round(qwen_score + float(metadata.get("pre_rank_score") or 0.0) * 0.01, 6)
            reranked.append(candidate)
        for index, candidate in enumerate(batch):
            if index in seen:
                continue
            reranked.append(candidate)
    return sort_units(reranked)


def aggregate_units_maxp(candidates: Sequence[Dict[str, object]], repo_name: str, *, limit: int) -> List[Dict[str, object]]:
    grouped: Dict[str, Dict[str, object]] = {}
    grouped_hits: Dict[str, List[Dict[str, object]]] = {}
    for candidate in candidates:
        metadata = dict(candidate.get("metadata", {}) or {})
        aggregation_key = str(metadata.get("embedding_aggregation_key") or candidate.get("path") or candidate.get("doc_id"))
        hit = {
            "unit_id": metadata.get("embedding_unit_id") or candidate.get("doc_id"),
            "unit_kind": metadata.get("embedding_unit_kind"),
            "score": candidate.get("score"),
            "preview": metadata.get("embedding_unit_preview") or candidate.get("preview"),
            "start_line": metadata.get("embedding_unit_start_line"),
            "end_line": metadata.get("embedding_unit_end_line"),
            "qwen_rerank_score": metadata.get("qwen_rerank_score"),
        }
        hits = [*grouped_hits.get(aggregation_key, []), hit]
        grouped_hits[aggregation_key] = sorted(hits, key=lambda item: -float(item.get("score") or 0.0))[:3]

        existing = grouped.get(aggregation_key)
        if existing is not None and float(existing.get("score") or 0.0) >= float(candidate.get("score") or 0.0):
            existing_metadata = dict(existing.get("metadata", {}) or {})
            existing_metadata["paper_unit_hits"] = grouped_hits[aggregation_key]
            existing["metadata"] = existing_metadata
            continue

        source_metadata = dict(metadata)
        source_metadata["stage"] = "source_maxp"
        source_metadata["paper_unit_hits"] = grouped_hits[aggregation_key]
        grouped[aggregation_key] = {
            "doc_id": source_doc_id(candidate),
            "kind": candidate.get("kind"),
            "repo": repo_name,
            "path": candidate.get("path"),
            "name": candidate.get("name"),
            "qualified_name": candidate.get("qualified_name"),
            "symbol_id": candidate.get("symbol_id"),
            "title": candidate.get("title"),
            "preview": candidate.get("preview"),
            "score": candidate.get("score"),
            "metadata": source_metadata,
            "reasons": ["paper-pipeline", "maxp"],
        }

    return sort_units(grouped.values())[:limit]


def source_doc_id(candidate: Dict[str, object]) -> str:
    metadata = dict(candidate.get("metadata", {}) or {})
    aggregation_key = str(metadata.get("embedding_aggregation_key") or "")
    if aggregation_key.startswith("symbol:"):
        return aggregation_key
    if aggregation_key.startswith("path:"):
        return aggregation_key
    return str(candidate.get("doc_id") or aggregation_key)


def build_embedding_score_map(results: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    score_map: Dict[str, Dict[str, object]] = {}
    for result in results:
        metadata = dict(result.get("metadata", {}) or {})
        keys = {
            str(metadata.get("embedding_aggregation_key") or ""),
            f"symbol:{result.get('symbol_id')}" if result.get("symbol_id") else "",
            f"path:{result.get('path')}" if result.get("path") else "",
            str(result.get("doc_id") or ""),
        }
        for key in keys:
            if not key:
                continue
            existing = score_map.get(key)
            if existing is None or float(result.get("score") or 0.0) > float(existing.get("score") or 0.0):
                score_map[key] = result
    return score_map


def build_embedding_unit_id_set(results: Sequence[Dict[str, object]]) -> set[str]:
    unit_ids: set[str] = set()
    for result in results:
        metadata = dict(result.get("metadata", {}) or {})
        if metadata.get("embedding_unit_id"):
            unit_ids.add(str(metadata["embedding_unit_id"]))
        for hit in metadata.get("embedding_unit_hits", []) or []:
            if hit.get("unit_id"):
                unit_ids.add(str(hit["unit_id"]))
    return unit_ids


def select_diverse_units(candidates: Sequence[Dict[str, object]], *, limit: int) -> List[Dict[str, object]]:
    selected: List[Dict[str, object]] = []
    counts: Dict[str, int] = {}
    for candidate in candidates:
        key = str(candidate.get("path") or candidate.get("symbol_id") or candidate.get("doc_id") or "")
        if counts.get(key, 0) >= MAX_UNITS_PER_SOURCE:
            continue
        selected.append(candidate)
        counts[key] = counts.get(key, 0) + 1
        if len(selected) >= limit:
            break
    return selected


def sort_units(candidates: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    return sorted(
        candidates,
        key=lambda item: (
            -float(item.get("score") or 0.0),
            str(item.get("path") or ""),
            str(item.get("qualified_name") or item.get("title") or ""),
            str(item.get("doc_id") or ""),
        ),
    )


def count_token_overlap(query_tokens: Sequence[str], unit: Dict[str, object]) -> int:
    haystack = " ".join(
        str(value or "").lower()
        for value in (
            unit.get("name"),
            unit.get("qualified_name"),
            unit.get("path"),
            unit.get("title"),
            unit.get("preview"),
            unit.get("content"),
        )
    )
    return sum(1 for token in query_tokens if token and token.lower() in haystack)


def exact_query_boost(query_tokens: Sequence[str], unit: Dict[str, object]) -> float:
    query = " ".join(query_tokens).lower()
    if not query:
        return 0.0
    boost = 0.0
    for value in (unit.get("qualified_name"), unit.get("name"), unit.get("path"), unit.get("title")):
        lowered = str(value or "").lower()
        if lowered == query:
            boost += 80.0
        elif query in lowered:
            boost += 12.0
    return boost


def generated_or_test_penalty(path: str, query_tokens: Sequence[str]) -> float:
    lowered_path = path.lower()
    token_set = {token.lower() for token in query_tokens}
    penalty = 0.0
    if any(marker in lowered_path for marker in ("/dist/", "/vendor/", "/node_modules/", ".min.")):
        penalty -= 8.0
    if any(marker in lowered_path for marker in ("/tests/", "/test/", ".snap")) and not token_set.intersection({"test", "tests", "snap", "schema"}):
        penalty -= 4.0
    return penalty


def normalized_rerank_provider() -> str:
    return os.environ.get("REPO_ANALYSIS_RERANK_PROVIDER", "qwen").strip().lower()
