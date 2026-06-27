from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Sequence

from common.native_tool import list_bm25_docs
from common.text import tokenize
from embeddings.units import build_retrieval_units
from embeddings.providers import (
    DEFAULT_HASHING_MODEL,
    embed_with_openai,
    embed_with_qwen,
    openai_embeddings_available,
    qwen_embeddings_available,
    resolve_embedding_provider,
)
from symbols.indexer import timestamp_now


SCHEMA_VERSION = "0.3.0"
DIMENSIONS = 256
BATCH_SIZE = 32
LIST_DOCS_BATCH_SIZE = 10_000
KIND_PRIORITY = {
    "symbol": 0.2,
    "statement": 0.15,
    "file": 0.1,
    "repo": 0.05,
    "directory": 0.0,
}


ProgressCallback = Callable[[Dict[str, object]], None]


def build_embedding_index(
    search_root: Path,
    repo_name: str,
    *,
    provider: str | None = None,
    model: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> Dict[str, object]:
    started_at = timestamp_now()

    def emit(event: str, **extra: object) -> None:
        if progress_callback is None:
            return
        progress_callback(
            {
                "event": event,
                "repo": repo_name,
                **extra,
            }
        )

    provider_config = resolve_embedding_provider(provider, model)
    provider_name = str(provider_config["provider"])
    model_name = str(provider_config["model"])

    emit(
        "build_started",
        provider=provider_name,
        model=model_name,
        started_at=started_at,
    )

    repo_search_root = search_root / repo_name
    tantivy_dir = repo_search_root / "tantivy"
    if not tantivy_dir.exists():
        raise FileNotFoundError(f"Missing Tantivy search documents for {repo_name}: {tantivy_dir}")

    if provider_name == "openai":
        payload = build_openai_embedding_payload(
            search_root,
            repo_name,
            model_name,
            progress_callback=progress_callback,
        )
    elif provider_name == "qwen":
        payload = build_qwen_embedding_payload(
            search_root,
            repo_name,
            model_name,
            progress_callback=progress_callback,
        )
    else:
        payload = build_hashing_embedding_payload(
            search_root,
            repo_name,
            model_name,
            progress_callback=progress_callback,
        )

    if not payload["documents"]:
        raise FileNotFoundError(f"Missing Tantivy search documents for {repo_name}: {tantivy_dir}")

    emit(
        "writing_outputs",
        provider=payload["provider"],
        model=payload["model"],
        documents=payload["summary"]["documents"],
    )

    repo_root = search_root / repo_name
    repo_root.mkdir(parents=True, exist_ok=True)
    with (repo_root / "embedding_index.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")
    with (repo_root / "embedding_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "schema_version": SCHEMA_VERSION,
                "repo": repo_name,
                "generated_at": payload["generated_at"],
                "provider": payload["provider"],
                "model": payload["model"],
                "model_backed": payload["model_backed"],
                "dimensions": payload["dimensions"],
                "vector_format": payload["vector_format"],
                "summary": payload["summary"],
            },
            handle,
            indent=2,
            sort_keys=False,
        )
        handle.write("\n")

    if payload["provider"] == "qwen":
        clear_qwen_embedding_checkpoint(search_root, repo_name, model_name)

    emit(
        "build_completed",
        provider=payload["provider"],
        model=payload["model"],
        documents=payload["summary"]["documents"],
        nonzero_dimensions=payload["summary"]["nonzero_dimensions"],
        vector_format=payload["vector_format"],
    )
    return payload


def build_hashing_embedding_payload(
    search_root: Path,
    repo_name: str,
    model_name: str,
    *,
    progress_callback: ProgressCallback | None = None,
) -> Dict[str, object]:
    def emit(event: str, **extra: object) -> None:
        if progress_callback is None:
            return
        progress_callback(
            {
                "event": event,
                "repo": repo_name,
                **extra,
            }
        )

    emit("hashing_scan_started", provider="hashing", model=model_name or DEFAULT_HASHING_MODEL)

    document_frequency: Counter[str] = Counter()
    unit_count = 0
    scan_batches = 0
    total_docs_hint: int | None = None

    for batch in iter_search_documents(search_root, repo_name, batch_size=LIST_DOCS_BATCH_SIZE):
        scan_batches += 1
        if batch:
            first_item_total = batch[0].get("_total_docs")
            if first_item_total is not None:
                total_docs_hint = int(first_item_total)

        for document in iter_retrieval_units(batch):
            tokens = tokenize(str(document["content"]))
            unit_count += 1
            for token in set(tokens):
                document_frequency[token] += 1

        emit(
            "hashing_scan_progress",
            provider="hashing",
            model=model_name or DEFAULT_HASHING_MODEL,
            batch_index=scan_batches,
            batch_docs=len(batch),
            processed_docs=unit_count,
            total_docs=None,
            source_docs_total=total_docs_hint,
        )

    emit(
        "hashing_scan_completed",
        provider="hashing",
        model=model_name or DEFAULT_HASHING_MODEL,
        documents=unit_count,
        batches=scan_batches,
    )

    embedded_documents = []
    nonzero_dimensions = 0
    processed_docs = 0
    embed_batches = 0

    emit(
        "hashing_embed_started",
        provider="hashing",
        model=model_name or DEFAULT_HASHING_MODEL,
        total_docs=unit_count,
    )

    for batch in iter_search_documents(search_root, repo_name, batch_size=LIST_DOCS_BATCH_SIZE):
        embed_batches += 1
        retrieval_units = list(iter_retrieval_units(batch))
        for document in retrieval_units:
            tokens = tokenize(str(document["content"]))
            raw_vector = embed_tokens(tokens, document_frequency, unit_count)
            norm = vector_norm(raw_vector)
            vector = normalize_sparse_vector(raw_vector, norm)
            nonzero_dimensions += len(vector)
            record = build_embedded_unit_record(document, 1.0 if vector else 0.0)
            record["vector"] = {str(index): round(value, 8) for index, value in sorted(vector.items()) if value}
            embedded_documents.append(record)
            processed_docs += 1
        emit(
            "hashing_embed_progress",
            provider="hashing",
            model=model_name or DEFAULT_HASHING_MODEL,
            batch_index=embed_batches,
            batch_docs=len(retrieval_units),
            processed_docs=processed_docs,
            total_docs=unit_count,
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "repo": repo_name,
        "generated_at": timestamp_now(),
        "provider": "hashing",
        "model": model_name or DEFAULT_HASHING_MODEL,
        "model_backed": False,
        "dimensions": DIMENSIONS,
        "vector_format": "sparse",
        "documents": embedded_documents,
        "summary": {
            "documents": len(embedded_documents),
            "nonzero_dimensions": nonzero_dimensions,
        },
    }


def build_openai_embedding_payload(
    search_root: Path,
    repo_name: str,
    model_name: str,
    *,
    progress_callback: ProgressCallback | None = None,
) -> Dict[str, object]:
    if not openai_embeddings_available():
        raise RuntimeError("OpenAI embedding provider requested but OPENAI_API_KEY is not set")

    def emit(event: str, **extra: object) -> None:
        if progress_callback is None:
            return
        progress_callback(
            {
                "event": event,
                "repo": repo_name,
                **extra,
            }
        )

    emit("openai_embed_started", provider="openai", model=model_name)

    embedded_documents = []
    processed_docs = 0
    batch_index = 0
    total_docs_hint = count_retrieval_units(search_root, repo_name)

    for search_batch in iter_search_documents(search_root, repo_name, batch_size=LIST_DOCS_BATCH_SIZE):
        retrieval_units = list(iter_retrieval_units(search_batch))
        for batch in batched(retrieval_units, BATCH_SIZE):
            batch_index += 1
            vectors = embed_with_openai([str(document["content"] or "") for document in batch], model_name)
            for document, vector in zip(batch, vectors):
                record = build_embedded_unit_record(document, 1.0 if vector else 0.0)
                record["vector"] = [round(float(value), 8) for value in vector]
                embedded_documents.append(record)
            processed_docs += len(batch)
            emit(
                "openai_embed_progress",
                provider="openai",
                model=model_name,
                batch_index=batch_index,
                batch_docs=len(batch),
                processed_docs=processed_docs,
                total_docs=total_docs_hint,
            )

    dimensions = len(embedded_documents[0]["vector"]) if embedded_documents else 0
    return {
        "schema_version": SCHEMA_VERSION,
        "repo": repo_name,
        "generated_at": timestamp_now(),
        "provider": "openai",
        "model": model_name,
        "model_backed": True,
        "dimensions": dimensions,
        "vector_format": "dense",
        "documents": embedded_documents,
        "summary": {
            "documents": len(embedded_documents),
            "nonzero_dimensions": dimensions * len(embedded_documents),
        },
    }


def build_qwen_embedding_payload(
    search_root: Path,
    repo_name: str,
    model_name: str,
    *,
    progress_callback: ProgressCallback | None = None,
) -> Dict[str, object]:
    if not qwen_embeddings_available():
        raise RuntimeError("Qwen embedding provider requested but no endpoint URL is configured")

    def emit(event: str, **extra: object) -> None:
        if progress_callback is None:
            return
        progress_callback(
            {
                "event": event,
                "repo": repo_name,
                **extra,
            }
        )

    emit("qwen_embed_started", provider="qwen", model=model_name)

    checkpoint_path, checkpoint_meta_path = qwen_embedding_checkpoint_paths(search_root, repo_name, model_name)
    checkpoint_records = load_qwen_embedding_checkpoint(
        checkpoint_path,
        checkpoint_meta_path,
        repo_name=repo_name,
        model_name=model_name,
    )
    if checkpoint_records:
        emit(
            "qwen_embed_checkpoint_loaded",
            provider="qwen",
            model=model_name,
            documents=len(checkpoint_records),
            checkpoint=str(checkpoint_path),
        )
    initialize_qwen_embedding_checkpoint(
        checkpoint_path,
        checkpoint_meta_path,
        repo_name=repo_name,
        model_name=model_name,
    )

    embedded_documents = list(checkpoint_records)
    seen_doc_ids = {str(document.get("doc_id") or "") for document in embedded_documents}
    processed_docs = len(embedded_documents)
    batch_index = math.ceil(processed_docs / BATCH_SIZE)
    total_docs_hint = count_retrieval_units(search_root, repo_name)

    for search_batch in iter_search_documents(search_root, repo_name, batch_size=LIST_DOCS_BATCH_SIZE):
        retrieval_units = [
            unit
            for unit in iter_retrieval_units(search_batch)
            if str(unit.get("doc_id") or "") not in seen_doc_ids
        ]
        for batch in batched(retrieval_units, BATCH_SIZE):
            batch_index += 1
            emit(
                "qwen_embed_batch_started",
                provider="qwen",
                model=model_name,
                batch_index=batch_index,
                batch_docs=len(batch),
                processed_docs=processed_docs,
                total_docs=total_docs_hint,
            )
            try:
                vectors = embed_with_qwen([str(document["content"] or "") for document in batch], model_name)
            except Exception as exc:
                emit(
                    "qwen_embed_batch_failed",
                    provider="qwen",
                    model=model_name,
                    batch_index=batch_index,
                    batch_docs=len(batch),
                    processed_docs=processed_docs,
                    total_docs=total_docs_hint,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                raise
            if len(vectors) != len(batch):
                raise RuntimeError(f"Qwen returned {len(vectors)} vectors for {len(batch)} embedding inputs")
            emit(
                "qwen_embed_batch_completed",
                provider="qwen",
                model=model_name,
                batch_index=batch_index,
                batch_docs=len(batch),
                processed_docs=processed_docs + len(batch),
                total_docs=total_docs_hint,
                returned_vectors=len(vectors),
            )
            new_records = []
            for document, vector in zip(batch, vectors):
                record = build_embedded_unit_record(document, 1.0 if vector else 0.0)
                record["vector"] = [round(float(value), 8) for value in vector]
                new_records.append(record)
                seen_doc_ids.add(str(record["doc_id"]))
            append_qwen_embedding_checkpoint(checkpoint_path, new_records)
            embedded_documents.extend(new_records)
            processed_docs += len(batch)
            emit(
                "qwen_embed_progress",
                provider="qwen",
                model=model_name,
                batch_index=batch_index,
                batch_docs=len(batch),
                processed_docs=processed_docs,
                total_docs=total_docs_hint,
            )

    dimensions = len(embedded_documents[0]["vector"]) if embedded_documents else 0
    return {
        "schema_version": SCHEMA_VERSION,
        "repo": repo_name,
        "generated_at": timestamp_now(),
        "provider": "qwen",
        "model": model_name,
        "model_backed": True,
        "dimensions": dimensions,
        "vector_format": "dense",
        "documents": embedded_documents,
        "summary": {
            "documents": len(embedded_documents),
            "nonzero_dimensions": dimensions * len(embedded_documents),
        },
    }


def qwen_embedding_checkpoint_paths(search_root: Path, repo_name: str, model_name: str) -> tuple[Path, Path]:
    repo_root = search_root / repo_name
    safe_model_name = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in model_name)
    base = repo_root / f"embedding_index.qwen.{safe_model_name}.checkpoint"
    return base.with_suffix(".jsonl"), base.with_suffix(".meta.json")


def initialize_qwen_embedding_checkpoint(
    checkpoint_path: Path,
    checkpoint_meta_path: Path,
    *,
    repo_name: str,
    model_name: str,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "repo": repo_name,
        "provider": "qwen",
        "model": model_name,
        "vector_format": "dense",
        "unit_schema": "retrieval_units",
    }
    with checkpoint_meta_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=False)
        handle.write("\n")
    checkpoint_path.touch(exist_ok=True)


def load_qwen_embedding_checkpoint(
    checkpoint_path: Path,
    checkpoint_meta_path: Path,
    *,
    repo_name: str,
    model_name: str,
) -> List[Dict[str, object]]:
    if not checkpoint_path.exists() or not checkpoint_meta_path.exists():
        return []
    try:
        metadata = json.loads(checkpoint_meta_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    expected = {
        "schema_version": SCHEMA_VERSION,
        "repo": repo_name,
        "provider": "qwen",
        "model": model_name,
        "vector_format": "dense",
        "unit_schema": "retrieval_units",
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        return []

    records: List[Dict[str, object]] = []
    seen_doc_ids = set()
    with checkpoint_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            doc_id = str(record.get("doc_id") or "")
            if not doc_id or doc_id in seen_doc_ids:
                continue
            if not record.get("unit_id") or not record.get("source_doc_id"):
                continue
            if not isinstance(record.get("vector"), list):
                continue
            records.append(record)
            seen_doc_ids.add(doc_id)
    return records


def append_qwen_embedding_checkpoint(checkpoint_path: Path, records: Sequence[Dict[str, object]]) -> None:
    if not records:
        return
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("a", encoding="utf-8") as handle:
        for record in records:
            json.dump(record, handle, sort_keys=False)
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def clear_qwen_embedding_checkpoint(search_root: Path, repo_name: str, model_name: str) -> None:
    for path in qwen_embedding_checkpoint_paths(search_root, repo_name, model_name):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def query_embedding_index(
    search_root: Path,
    repo_name: str,
    query: str,
    *,
    limit: int = 10,
) -> List[Dict[str, object]]:
    index_path = search_root / repo_name / "embedding_index.json"
    if not index_path.exists():
        raise FileNotFoundError(
            f"Missing embedding index for {repo_name}: {index_path}. "
            "Run build-embeddings before semantic retrieval."
        )

    with index_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    documents = payload.get("documents", [])
    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    provider = str(payload.get("provider") or "hashing")
    vector_format = str(payload.get("vector_format") or "sparse")
    if provider == "openai":
        if not openai_embeddings_available():
            raise RuntimeError("Embedding index uses OpenAI, but OPENAI_API_KEY is not set")
        query_vector = embed_with_openai([query], str(payload["model"]))[0]
        query_norm = 1.0 if query_vector else 0.0
    elif provider == "qwen":
        if not qwen_embeddings_available():
            raise RuntimeError("Embedding index uses Qwen, but no endpoint URL is configured")
        query_vector = embed_with_qwen([query], str(payload["model"]))[0]
        query_norm = 1.0 if query_vector else 0.0
    else:
        raw_query_vector = embed_tokens(query_tokens, None, max(len(documents), 1))
        query_norm = vector_norm(raw_query_vector)
        query_vector = normalize_sparse_vector(raw_query_vector, query_norm)

    if query_norm == 0:
        return []

    grouped_results: Dict[str, Dict[str, object]] = {}
    grouped_unit_hits: Dict[str, List[Dict[str, object]]] = {}
    for document in documents:
        if vector_format == "dense":
            similarity = dense_dot_product(query_vector, [float(value) for value in document.get("vector", [])])
        else:
            doc_vector = {int(index): float(value) for index, value in document["vector"].items()}
            similarity = dot_product(query_vector, doc_vector)
        if similarity <= 0:
            continue
        searchable = " ".join(
            str(item or "").lower()
            for item in (
                document.get("name"),
                document.get("qualified_name"),
                document.get("path"),
                document.get("title"),
                document.get("preview"),
            )
        )
        overlap_bonus = 0.03 * sum(1 for token in query_tokens if token in searchable)
        kind_bonus = KIND_PRIORITY.get(str(document.get("kind") or ""), 0.0)
        path_value = str(document.get("path") or "").lower()
        path_penalty = 0.0
        if any(marker in path_value for marker in ("/tests/", "/test/", ".snap")) and not any(
            token in {"test", "tests", "snap", "schema"} for token in query_tokens
        ):
            path_penalty -= 0.12
        score = similarity + overlap_bonus + kind_bonus + path_penalty
        if score <= 0:
            continue
        result = {
            "doc_id": document.get("source_doc_id") or document["doc_id"],
            "kind": document.get("source_kind") or document["kind"],
            "repo": repo_name,
            "path": document.get("path"),
            "name": document.get("name"),
            "qualified_name": document.get("qualified_name"),
            "symbol_id": document.get("symbol_id"),
            "title": document.get("title"),
            "preview": document.get("preview"),
            "score": round(score, 6),
            "metadata": {
                "provider": payload["provider"],
                "model": payload["model"],
                "dimensions": payload["dimensions"],
                "model_backed": bool(payload.get("model_backed")),
                "embedding_aggregation": "maxp",
                "embedding_aggregation_key": document.get("aggregation_key"),
                "embedding_aggregation_kind": document.get("aggregation_kind"),
                "embedding_unit_id": document.get("unit_id") or document.get("doc_id"),
                "embedding_unit_kind": document.get("unit_kind"),
                "embedding_unit_score": round(score, 6),
                "embedding_unit_preview": document.get("preview"),
                "embedding_unit_text": document.get("content"),
                "embedding_unit_start_line": document.get("start_line"),
                "embedding_unit_end_line": document.get("end_line"),
                "embedding_unit_token_estimate": document.get("token_estimate"),
            },
        }
        aggregation_key = str(document.get("aggregation_key") or result["doc_id"])
        current = grouped_results.get(aggregation_key)
        if current is None or float(result["score"]) > float(current["score"]):
            grouped_results[aggregation_key] = result
            current = result
        unit_hits = list(grouped_unit_hits.get(aggregation_key, []))
        unit_hits.append(
            {
                "unit_id": document.get("unit_id") or document.get("doc_id"),
                "unit_kind": document.get("unit_kind"),
                "score": round(score, 6),
                "preview": document.get("preview"),
                "start_line": document.get("start_line"),
                "end_line": document.get("end_line"),
            }
        )
        grouped_unit_hits[aggregation_key] = sorted(unit_hits, key=lambda item: -float(item.get("score") or 0.0))[:3]
        metadata = dict(current.get("metadata", {}) or {})
        metadata["embedding_unit_hits"] = grouped_unit_hits[aggregation_key]
        current["metadata"] = metadata

    return sorted(
        grouped_results.values(),
        key=lambda item: (
            -item["score"],
            str(item.get("path") or ""),
            str(item.get("qualified_name") or item.get("title") or ""),
        ),
    )[:limit]


def iter_retrieval_units(documents: Iterable[Dict[str, object]]) -> Iterator[Dict[str, object]]:
    for document in documents:
        yield from build_retrieval_units(document)


def count_retrieval_units(search_root: Path, repo_name: str) -> int:
    total_units = 0
    for batch in iter_search_documents(search_root, repo_name, batch_size=LIST_DOCS_BATCH_SIZE):
        for document in batch:
            total_units += len(build_retrieval_units(document))
    return total_units


def build_embedded_unit_record(document: Dict[str, object], norm: float) -> Dict[str, object]:
    return {
        "doc_id": document["doc_id"],
        "source_doc_id": document.get("source_doc_id"),
        "source_kind": document.get("source_kind"),
        "unit_id": document.get("unit_id"),
        "unit_kind": document.get("unit_kind"),
        "aggregation_key": document.get("aggregation_key"),
        "aggregation_kind": document.get("aggregation_kind"),
        "kind": document["kind"],
        "path": document["path"],
        "name": document["name"],
        "qualified_name": document["qualified_name"],
        "symbol_id": document["symbol_id"],
        "title": document["title"],
        "preview": document["preview"],
        "content": document["content"],
        "start_line": document.get("start_line"),
        "end_line": document.get("end_line"),
        "token_estimate": document.get("token_estimate"),
        "char_count": document.get("char_count"),
        "norm": norm,
    }


def iter_search_documents(
    search_root: Path,
    repo_name: str,
    *,
    batch_size: int = LIST_DOCS_BATCH_SIZE,
) -> Iterator[List[Dict[str, object]]]:
    tantivy_dir = search_root / repo_name / "tantivy"
    if not tantivy_dir.exists():
        return

    offset = 0
    while True:
        payload = list_bm25_docs(
            tantivy_dir,
            offset=offset,
            limit=batch_size,
        )
        batch = payload.get("results", [])
        if not batch:
            return

        total_docs = payload.get("total_docs")
        normalized_batch = [normalize_search_document(item, total_docs=total_docs) for item in batch]
        yield normalized_batch

        next_offset = payload.get("next_offset")
        if next_offset is None:
            return
        offset = int(next_offset)

def load_search_documents(search_root: Path, repo_name: str) -> List[Dict[str, object]]:
    documents: List[Dict[str, object]] = []
    for batch in iter_search_documents(search_root, repo_name):
        documents.extend(batch)
    return documents


def normalize_search_document(item: Dict[str, object], *, total_docs: int | None = None) -> Dict[str, object]:
    return {
        "doc_id": item["doc_id"],
        "kind": item["kind"],
        "path": item.get("path"),
        "name": item.get("name"),
        "qualified_name": item.get("qualified_name"),
        "symbol_id": item.get("symbol_id"),
        "title": item.get("title"),
        "preview": item.get("preview"),
        "content": item.get("searchable") or "",
        "_total_docs": total_docs,
    }


def compute_document_frequency(document_tokens: Sequence[Sequence[str]]) -> Counter[str]:
    frequency: Counter[str] = Counter()
    for tokens in document_tokens:
        for token in set(tokens):
            frequency[token] += 1
    return frequency


def embed_tokens(tokens: Sequence[str], document_frequency: Counter[str] | None, document_count: int) -> Dict[int, float]:
    term_frequency = Counter(tokens)
    vector: Dict[int, float] = {}
    for token, tf in term_frequency.items():
        index, sign = hashed_dimension(token)
        if document_frequency is None:
            idf = 1.0
        else:
            idf = math.log((document_count + 1) / (document_frequency[token] + 1)) + 1.0
        vector[index] = vector.get(index, 0.0) + sign * tf * idf
    return vector


def hashed_dimension(token: str) -> tuple[int, float]:
    digest = hashlib.sha1(token.encode("utf-8")).digest()
    index = int.from_bytes(digest[:4], "big") % DIMENSIONS
    return index, 1.0


def normalize_sparse_vector(vector: Dict[int, float], norm: float | None = None) -> Dict[int, float]:
    actual_norm = vector_norm(vector) if norm is None else norm
    if actual_norm == 0:
        return {}
    return {index: value / actual_norm for index, value in vector.items()}


def vector_norm(vector: Dict[int, float]) -> float:
    return math.sqrt(sum(value * value for value in vector.values()))


def dot_product(left: Dict[int, float], right: Dict[int, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(index, 0.0) for index, value in left.items())


def dense_dot_product(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(left, right))


def batched(values: Sequence[Dict[str, object]], size: int) -> Iterable[Sequence[Dict[str, object]]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]
