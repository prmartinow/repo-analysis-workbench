from __future__ import annotations

import re
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

    return sorted(
        ranked,
        key=lambda item: (
            -float(item["score"]),
            -KIND_PRIORITY.get(str(item.get("kind") or ""), 0),
            str(item.get("path") or ""),
            str(item.get("qualified_name") or item.get("title") or ""),
        ),
    )
