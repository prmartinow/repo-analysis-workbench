from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import PurePosixPath
from typing import Dict, Iterable, List

from common.retrieval import RANK_KIND_WEIGHTS, RANK_SYMBOL_KIND_WEIGHTS, semantic_activity_score


KIND_PRIORITY = {
    "symbol": 6,
    "type_body": 5,
    "function_body": 4,
    "file": 3,
    "doc": 3,
    "directory": 2,
    "package": 1,
    "repo": 0,
    "statement": -2,
    "module_ref": -3,
    "symbol_ref": -3,
    "type_ref": -3,
}
CALLABLE_SYMBOL_KINDS = {"function", "method", "associated_function"}
NOMINAL_SYMBOL_KINDS = {"trait", "struct", "enum", "type"}
LOCAL_LIKE_KINDS = {"field", "local", "parameter", "variable"}
QWEN_RERANK_URL = "http://127.0.0.1:18200/rerank"
QWEN_RERANK_MAX_CANDIDATES = 5
QWEN_RERANK_PROVIDERS = {"qwen", "qwen-local", "local-qwen", "auto", ""}
HEURISTIC_RERANK_PROVIDERS = {"heuristic", "builtin", "local", "none", "off", "false"}
TRUTHY_VALUES = {"1", "true", "yes", "on"}


def identifier_terms(value: str) -> List[str]:
    if not value:
        return []
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
    tokens = re.split(r"[^A-Za-z0-9]+", spaced.lower())
    return [token for token in tokens if token]


def rerank_candidates(
    candidates: Iterable[Dict[str, object]],
    query_tokens: List[str],
    *,
    query_profile: Dict[str, object] | None = None,
) -> List[Dict[str, object]]:
    query_profile = query_profile or {}
    explicit_statement = bool(query_profile.get("explicit_statement"))
    exploratory = query_profile.get("intent") == "exploration"
    query_text = " ".join(query_tokens).strip().lower()

    ranked = []
    for candidate in candidates:
        metadata = dict(candidate.get("metadata", {}) or {})
        kind = str(candidate.get("kind") or "")
        symbol_kind = str(
            metadata.get("kind")
            or metadata.get("symbol_kind")
            or metadata.get("node_kind")
            or ""
        ).lower()
        name = str(candidate.get("name") or "")
        qualified_name = str(candidate.get("qualified_name") or "")
        path = str(candidate.get("path") or "")
        title = str(candidate.get("title") or "")
        preview = str(candidate.get("preview") or "")
        visibility = str(metadata.get("visibility") or "").lower()
        graph_distance = metadata.get("graph_distance")
        containment_depth = metadata.get("containment_depth")
        summary_relevance = float(metadata.get("summary_relevance") or 0.0)
        semantic_summary = metadata.get("semantic_summary", {}) or {}

        lowered_name = name.lower()
        lowered_qname = qualified_name.lower()
        searchable = " ".join(
            value.lower()
            for value in (name, qualified_name, path, title, preview)
            if value
        )

        score = float(candidate.get("score") or 0.0)

        if query_text:
            if query_text == lowered_qname:
                score += 180.0
            if query_text == lowered_name:
                score += 150.0
            if lowered_qname.endswith(f"::{query_text}"):
                score += 110.0

        identifier_overlap_terms = set(identifier_terms(name))
        if qualified_name:
            identifier_overlap_terms.update(identifier_terms(qualified_name.split("::")[-1]))

        token_hits = 0
        for token in query_tokens:
            if token and token in lowered_qname:
                score += 16.0
                token_hits += 1
            elif token and token in lowered_name:
                score += 14.0
                token_hits += 1
            elif token and token in title.lower():
                score += 9.0
                token_hits += 1
            elif token and token in preview.lower():
                score += 7.0
                token_hits += 1
            elif token and token in path.lower():
                score += 6.0
                token_hits += 1
            elif token and token in searchable:
                score += 4.0
                token_hits += 1

            if token and token in identifier_overlap_terms:
                score += 5.0

        if query_tokens and token_hits == len(query_tokens):
            score += 26.0
        elif query_tokens and token_hits == 0:
            score -= 18.0

        score += RANK_KIND_WEIGHTS.get(kind, 0.0)
        score += RANK_SYMBOL_KIND_WEIGHTS.get(symbol_kind, 0.0)
        score += KIND_PRIORITY.get(kind, 0) * 0.5

        if graph_distance is not None:
            score += max(20.0 - (float(graph_distance) - 1.0) * 6.0, 2.0)
        if containment_depth is not None:
            score += max(18.0 - float(containment_depth) * 4.0, 4.0)

        if visibility == "pub":
            score += 8.0
        elif visibility == "private" and kind == "symbol":
            score -= 2.0

        if kind == "symbol" and symbol_kind in NOMINAL_SYMBOL_KINDS:
            score += semantic_activity_score(semantic_summary) * (0.12 if exploratory else 0.08)
        elif kind == "symbol" and symbol_kind == "impl":
            score += semantic_activity_score(semantic_summary) * (0.16 if exploratory else 0.1)
        elif kind == "symbol" and symbol_kind in CALLABLE_SYMBOL_KINDS:
            score += semantic_activity_score(semantic_summary) * (0.34 if exploratory else 0.14)

        score += summary_relevance * 28.0

        if kind in {"module_ref", "symbol_ref", "type_ref"}:
            score -= 30.0
        if kind == "statement" and not explicit_statement:
            score -= 60.0
        if symbol_kind in LOCAL_LIKE_KINDS and not explicit_statement:
            score -= 28.0
        if kind in {"function_body", "type_body"} and symbol_kind not in {"", "trait", "struct", "enum", "type", "impl"}:
            score -= 6.0

        suffix = PurePosixPath(path).suffix.lower() if path else ""
        if suffix == ".md" and kind not in {"file", "doc"} and not query_profile.get("explicit_docs"):
            score -= 4.0

        updated = dict(candidate)
        updated["score"] = round(score, 6)
        ranked.append(updated)

    heuristic_ranked = sorted(
        ranked,
        key=lambda item: (
            -float(item["score"]),
            -KIND_PRIORITY.get(str(item.get("kind") or ""), 0),
            str(item.get("path") or ""),
            str(item.get("qualified_name") or item.get("title") or ""),
        ),
    )
    return maybe_qwen_rerank(heuristic_ranked, query_text)


def maybe_qwen_rerank(ranked: List[Dict[str, object]], query_text: str) -> List[Dict[str, object]]:
    provider = os.environ.get("REPO_ANALYSIS_RERANK_PROVIDER", "qwen").strip().lower()
    if provider in HEURISTIC_RERANK_PROVIDERS:
        return ranked
    if provider not in QWEN_RERANK_PROVIDERS:
        raise ValueError(f"Unsupported rerank provider: {provider}")
    if not ranked or not query_text:
        return ranked

    limit = min(
        int(os.environ.get("REPO_ANALYSIS_QWEN_RERANK_LIMIT", QWEN_RERANK_MAX_CANDIDATES)),
        QWEN_RERANK_MAX_CANDIDATES,
        len(ranked),
    )
    top = ranked[:limit]
    documents = [candidate_to_rerank_document(candidate) for candidate in top]
    try:
        results = qwen_rerank(query_text, documents)
    except RuntimeError:
        if allow_heuristic_rerank_fallback():
            return ranked
        raise
    if not results:
        if allow_heuristic_rerank_fallback():
            return ranked
        raise RuntimeError("Qwen rerank returned no results")

    reranked: List[Dict[str, object]] = []
    seen_indexes = set()
    for result in results:
        index = int(result.get("index", -1))
        if index < 0 or index >= len(top) or index in seen_indexes:
            continue
        seen_indexes.add(index)
        updated = dict(top[index])
        metadata = dict(updated.get("metadata", {}) or {})
        metadata["qwen_rerank_score"] = float(result.get("score") or 0.0)
        updated["metadata"] = metadata
        updated["score"] = round(float(updated.get("score") or 0.0) + float(result.get("score") or 0.0), 6)
        reranked.append(updated)

    for index, candidate in enumerate(top):
        if index not in seen_indexes:
            reranked.append(candidate)
    return reranked + ranked[limit:]


def allow_heuristic_rerank_fallback() -> bool:
    return os.environ.get("REPO_ANALYSIS_ALLOW_HEURISTIC_RERANK_FALLBACK", "").strip().lower() in TRUTHY_VALUES


def candidate_to_rerank_document(candidate: Dict[str, object]) -> str:
    parts = [
        str(candidate.get("qualified_name") or candidate.get("name") or ""),
        str(candidate.get("path") or ""),
        str(candidate.get("title") or ""),
        str(candidate.get("preview") or ""),
    ]
    return "\n".join(part for part in parts if part)[:2500]


def qwen_rerank(query: str, documents: List[str]) -> List[Dict[str, object]]:
    request = urllib.request.Request(
        os.environ.get("REPO_ANALYSIS_QWEN_RERANK_URL", QWEN_RERANK_URL),
        data=json.dumps({"query": query, "documents": documents}).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "X-Caller": "repo-analysis",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=None) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Qwen rerank request failed: {exc.code} {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Qwen rerank request failed: {exc}") from exc
    return list(payload.get("results", []))
