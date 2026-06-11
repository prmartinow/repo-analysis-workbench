# repo-analysis/src/backends/tantivy/search.py
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from backends.metadata_store import get_metadata_store
from common.native_tool import query_bm25_index
from common.retrieval import RANK_KIND_WEIGHTS, RANK_SYMBOL_KIND_WEIGHTS, semantic_activity_score
from common.text import tokenize
from search.indexer import list_documents

MIN_SEARCH_FANOUT = 120
DEFAULT_SEARCH_KINDS = ("repo", "package", "directory", "file", "symbol", "function_body", "type_body", "doc")
CALLABLE_SYMBOL_KINDS = {"function", "method", "associated_function"}
NOMINAL_SYMBOL_KINDS = {"struct", "trait", "enum", "type"}
LOCAL_LIKE_KINDS = {"field", "local", "parameter", "variable"}


class TantivySearchBackend:
    """Tantivy-native search adapter for the interactive hot path."""

    def __init__(self, search_root: Path, repo_name: str) -> None:
        self.search_root = search_root
        self.repo_name = repo_name

    def _metadata_store(self):
        parsed_root = self.search_root.parent / "parsed"
        return get_metadata_store(str(parsed_root.resolve()), self.repo_name)

    def search(
        self,
        query: str,
        *,
        limit: int,
        kinds: Sequence[str] = (),
        path_prefix: Optional[str] = None,
    ) -> List[Dict[str, object]]:
        normalized_query = " ".join(tokenize(query))
        if not normalized_query:
            return []

        tantivy_dir = self.search_root / self.repo_name / "tantivy"
        if not tantivy_dir.exists():
            return []

        fanout = max(limit * 12, MIN_SEARCH_FANOUT)
        merged: Dict[str, Dict[str, object]] = {}
        effective_kinds = tuple(kinds) if kinds else DEFAULT_SEARCH_KINDS

        for query_variant in build_query_variants(query):
            variant = " ".join(tokenize(query_variant))
            if not variant:
                continue

            results = query_bm25_index(
                tantivy_dir,
                variant,
                limit=fanout,
                kinds=effective_kinds,
                path_prefix=path_prefix,
            )
            for item in results:
                doc_id = str(item.get("doc_id") or "")
                if not doc_id:
                    doc_id = f"{item.get('kind')}::{item.get('path')}::{item.get('qualified_name') or item.get('name') or ''}"

                native_score = float(item.get("score") or 0.0)
                existing = merged.get(doc_id)
                if existing is None:
                    candidate = dict(item)
                    candidate["_best_native_score"] = native_score
                    candidate["_variant_hits"] = 1
                    merged[doc_id] = candidate
                    continue

                existing["_variant_hits"] = int(existing.get("_variant_hits") or 1) + 1
                existing["_best_native_score"] = max(float(existing.get("_best_native_score") or 0.0), native_score)
                if native_score > float(existing.get("score") or 0.0):
                    for key, value in item.items():
                        existing[key] = value

        return rerank_search_results(query, list(merged.values()), limit=limit)

    def find_file(self, path_pattern: str, *, limit: int) -> List[Dict[str, object]]:
        normalized_pattern = path_pattern.strip()
        if not normalized_pattern:
            return self.list_documents(limit=limit, kinds=("directory", "file"))
        metadata_store = self._metadata_store()
        exact = metadata_store.get_file(normalized_pattern)
        if exact is not None:
            return [file_record_to_result(self.repo_name, exact)]
        prefix = normalized_pattern.rstrip("*").rstrip("/")
        if prefix:
            prefixed = metadata_store.find_files_by_prefix(prefix, limit=limit)
            if prefixed:
                return [file_record_to_result(self.repo_name, item) for item in prefixed[:limit]]
        docs = self.search(
            path_pattern,
            limit=max(limit * 8, 40),
            kinds=("directory", "file"),
        )
        normalized_pattern = normalized_pattern.lower().replace("*", "")
        if not normalized_pattern:
            return docs[:limit]
        ranked = []
        for item in docs:
            path = str(item.get("path") or "")
            if not path:
                continue
            haystack = path.lower()
            score = float(item.get("score") or 0.0)
            if normalized_pattern in haystack:
                score += 2.0
            ranked.append((score, item))
        ranked.sort(key=lambda pair: (-pair[0], str(pair[1].get("path") or "")))
        return [item for _, item in ranked[:limit]]

    def list_documents(
        self,
        *,
        limit: int,
        kinds: Sequence[str] = (),
        path_prefix: Optional[str] = None,
    ) -> List[Dict[str, object]]:
        return list_documents(
            self.search_root,
            self.repo_name,
            limit=limit,
            kinds=kinds,
            path_prefix=path_prefix,
        )

    def lookup_symbol_docs(self, symbol_id: str, *, kinds: Sequence[str] = (), limit: int = 20) -> List[Dict[str, object]]:
        metadata_store = self._metadata_store()
        symbol = metadata_store.get_symbol(symbol_id)
        body = metadata_store.get_symbol_body(symbol_id)
        if symbol is not None and body is not None:
            result = body_payload_to_result(symbol, body)
            if not kinds or str(result.get("kind") or "") in kinds:
                return [result]
        tantivy_dir = self.search_root / self.repo_name / "tantivy"
        if tantivy_dir.exists():
            docs = query_bm25_index(
                tantivy_dir,
                "",
                limit=max(limit * 4, 20),
                kinds=kinds,
                symbol_id=symbol_id,
            )
            exact = [item for item in docs if str(item.get("symbol_id") or "") == symbol_id]
            if exact:
                return exact[:limit]
        return []

    def compare_repo_candidates(self, query: str, *, limit: int) -> List[Dict[str, object]]:
        return self.search(
            query,
            limit=max(limit * 4, 20),
            kinds=("symbol", "function_body", "type_body", "doc", "file", "directory", "package"),
        )[:limit]

    def artifact_fingerprint(self) -> str:
        tracked_paths = [
            self.search_root / self.repo_name / "tantivy",
        ]
        snapshot = []
        for path in tracked_paths:
            if path.exists():
                snapshot.extend(snapshot_artifact(path))
        return hashlib.sha1(json.dumps(snapshot, sort_keys=True).encode("utf-8")).hexdigest()


def build_query_variants(query: str) -> List[str]:
    base_tokens = [token for token in tokenize(query) if token]
    if not base_tokens:
        return []

    variants: List[str] = [" ".join(base_tokens)]
    if len(base_tokens) >= 3:
        for window_size in (2, 3):
            if len(base_tokens) < window_size:
                continue
            for start in range(0, len(base_tokens) - window_size + 1):
                variants.append(" ".join(base_tokens[start : start + window_size]))

    deduped: List[str] = []
    seen = set()
    for variant in variants:
        normalized = " ".join(tokenize(variant))
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def rerank_search_results(
    query: str,
    results: Sequence[Dict[str, object]],
    *,
    limit: int,
) -> List[Dict[str, object]]:
    query_tokens = normalize_query_tokens(query)
    explicit_statement = query_explicitly_targets_statement(query_tokens, query)

    scored: List[Tuple[float, Dict[str, object]]] = []
    for item in results:
        reranked = dict(item)
        score = score_search_result(
            query,
            query_tokens,
            reranked,
            explicit_statement=explicit_statement,
        )
        reranked["score"] = round(float(score), 6)
        reranked.pop("_best_native_score", None)
        reranked.pop("_variant_hits", None)
        scored.append((float(reranked["score"]), reranked))

    scored.sort(
        key=lambda pair: (
            -pair[0],
            kind_rank(str(pair[1].get("kind") or "")),
            str(pair[1].get("path") or ""),
            str(pair[1].get("qualified_name") or pair[1].get("name") or ""),
        )
    )
    return [item for _score, item in scored[:limit]]


def score_search_result(
    query: str,
    query_tokens: Sequence[str],
    item: Dict[str, object],
    *,
    explicit_statement: bool,
) -> float:
    lowered_query = str(query or "").strip().lower()
    name = str(item.get("name") or "")
    qualified_name = str(item.get("qualified_name") or "")
    title = str(item.get("title") or "")
    preview = str(item.get("preview") or "")
    searchable = str(item.get("searchable") or "")
    path = str(item.get("path") or "")
    metadata = dict(item.get("metadata", {}) or {})
    kind = str(item.get("kind") or "")
    symbol_kind = str(metadata.get("kind") or "").lower()
    semantic_summary = metadata.get("semantic_summary", {}) or {}
    visibility = str(metadata.get("visibility") or "").lower()

    lowered_name = name.lower()
    lowered_qname = qualified_name.lower()
    haystack = " ".join(
        value.lower()
        for value in (name, qualified_name, title, preview, searchable, path)
        if value
    )
    native_score = float(item.get("_best_native_score") or item.get("score") or 0.0)
    score = native_score

    if lowered_query:
        if lowered_query == lowered_qname:
            score += 180.0
        if lowered_query == lowered_name:
            score += 150.0
        if lowered_qname.endswith(f"::{lowered_query}"):
            score += 110.0
        if lowered_query in title.lower():
            score += 18.0
        if lowered_query in preview.lower():
            score += 12.0
        if lowered_query in path.lower():
            score += 16.0

    exact_token_hits = 0
    for token in query_tokens:
        if token and token in lowered_qname:
            score += 16.0
            exact_token_hits += 1
        elif token and token in lowered_name:
            score += 14.0
            exact_token_hits += 1
        elif token and token in title.lower():
            score += 10.0
            exact_token_hits += 1
        elif token and token in preview.lower():
            score += 8.0
            exact_token_hits += 1
        elif token and token in path.lower():
            score += 7.0
            exact_token_hits += 1
        elif token and token in haystack:
            score += 4.0
            exact_token_hits += 1

    if query_tokens and exact_token_hits == len(query_tokens):
        score += 24.0
    elif query_tokens and exact_token_hits == 0:
        score -= 18.0

    score += RANK_KIND_WEIGHTS.get(kind, 0.0)
    score += RANK_SYMBOL_KIND_WEIGHTS.get(symbol_kind, 0.0)

    if visibility == "pub":
        score += 8.0

    if kind == "symbol" and symbol_kind in NOMINAL_SYMBOL_KINDS:
        score += semantic_activity_score(semantic_summary) * 0.08
    elif kind == "symbol" and symbol_kind == "impl":
        score += semantic_activity_score(semantic_summary) * 0.1
    elif kind == "symbol" and symbol_kind in CALLABLE_SYMBOL_KINDS:
        score += semantic_activity_score(semantic_summary) * 0.14

    variant_hits = int(item.get("_variant_hits") or 1)
    if variant_hits > 1:
        score += min(variant_hits - 1, 3) * 4.0

    if kind == "statement" and not explicit_statement:
        score -= 70.0
    if kind in {"module_ref", "symbol_ref", "type_ref"}:
        score -= 30.0
    if symbol_kind in LOCAL_LIKE_KINDS and not explicit_statement:
        score -= 30.0

    return score


def normalize_query_tokens(query: str) -> List[str]:
    return [token.lower() for token in tokenize(query) if token]


def query_explicitly_targets_statement(query_tokens: Sequence[str], query: str) -> bool:
    lowered_query = str(query or "").strip().lower()
    if "@l" in lowered_query:
        return True
    token_set = {str(token).strip().lower() for token in query_tokens}
    return bool(token_set.intersection({"statement", "statements", "line", "lines", "expr", "let", "local", "locals"}))


def kind_rank(kind: str) -> int:
    ranking = {
        "symbol": 0,
        "type_body": 1,
        "function_body": 2,
        "file": 3,
        "directory": 4,
        "package": 5,
        "doc": 6,
    }
    return ranking.get(kind, 99)


def snapshot_artifact(path: Path) -> List[Dict[str, object]]:
    if path.is_dir():
        rows: List[Dict[str, object]] = []
        for child in sorted(path.rglob("*")):
            if not child.is_file():
                continue
            stat = child.stat()
            rows.append(
                {
                    "path": str(child),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
        return rows
    stat = path.stat()
    return [
        {
            "path": str(path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    ]


def body_payload_to_result(symbol: Dict[str, object], body: Dict[str, object]) -> Dict[str, object]:
    symbol_kind = str(symbol.get("kind") or "")
    body_kind = "type_body" if symbol_kind in {"trait", "struct", "enum", "type", "impl"} else "function_body"
    statements = body.get("statements") or []
    preview = " ".join(str(item.get("text") or "") for item in statements[:4]).strip()
    if not preview:
        preview = str(symbol.get("signature") or symbol.get("qualified_name") or "")
    return {
        "doc_id": f"body:{symbol.get('symbol_id')}",
        "kind": body_kind,
        "repo": symbol.get("repo"),
        "path": symbol.get("path"),
        "name": symbol.get("name"),
        "qualified_name": symbol.get("qualified_name"),
        "symbol_id": symbol.get("symbol_id"),
        "title": f"{symbol.get('qualified_name') or symbol.get('name')} body",
        "preview": preview,
        "score": 1.0,
        "metadata": {
            "kind": symbol_kind,
            "body_kind": body_kind,
            "visibility": symbol.get("visibility"),
            "container_qualified_name": symbol.get("container_qualified_name"),
            "semantic_summary": symbol.get("semantic_summary", {}),
        },
    }


def file_record_to_result(repo_name: str, file_record: Dict[str, object]) -> Dict[str, object]:
    path = str(file_record.get("path") or "")
    return {
        "doc_id": f"file:{path}",
        "kind": "file",
        "repo": repo_name,
        "path": path,
        "name": Path(path).name if path else None,
        "qualified_name": None,
        "symbol_id": None,
        "title": path,
        "preview": f"{file_record.get('language') or 'file'} in {file_record.get('crate') or file_record.get('package_name') or 'repo'}",
        "score": 1.0,
        "metadata": {
            "language": file_record.get("language"),
            "crate": file_record.get("crate"),
            "package_name": file_record.get("package_name"),
            "module_path": file_record.get("module_path"),
        },
    }
