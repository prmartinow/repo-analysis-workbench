from __future__ import annotations

import hashlib
import re
from typing import Dict, Iterable, List


DEFAULT_EMBED_TOKEN_LIMIT = 900
CODE_EMBED_TOKEN_LIMIT = 350
MIN_CODE_WINDOW_TOKENS = 80
EMBED_CHAR_LIMIT = 10_000
TOKEN_RE = re.compile(r"\w+|[^\s\w]", re.UNICODE)
CODE_LIKE_KINDS = {"file", "function_body", "type_body", "statement"}
DOC_LIKE_KINDS = {"doc", "repo", "directory", "package"}


def build_retrieval_units(document: Dict[str, object]) -> List[Dict[str, object]]:
    """Split a search document into model-sized retrieval units with source provenance."""
    content = str(document.get("content") or "")
    if not content.strip():
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
    if not content.strip():
        return []

    kind = str(document.get("kind") or "")
    if kind in CODE_LIKE_KINDS:
        chunks = chunk_code_like_text(content)
        default_unit_kind = "line_window" if len(chunks) > 1 or kind == "file" else "symbol_body"
    elif kind in DOC_LIKE_KINDS:
        chunks = chunk_document_text(content)
        default_unit_kind = "doc_section" if len(chunks) > 1 else f"{kind}_summary"
    else:
        chunks = chunk_text(content, DEFAULT_EMBED_TOKEN_LIMIT)
        default_unit_kind = "semantic_text"

    units = []
    for ordinal, chunk in enumerate(chunks):
        unit_text = chunk["text"].strip()
        if not unit_text:
            continue
        unit_kind = default_unit_kind
        if kind in {"function_body", "type_body"}:
            unit_kind = kind
        elif kind == "statement":
            unit_kind = "statement"
        unit_id = stable_unit_id(str(document["doc_id"]), ordinal, unit_text)
        aggregation_key = aggregation_key_for(document)
        units.append(
            {
                "unit_id": unit_id,
                "doc_id": unit_id,
                "source_doc_id": document["doc_id"],
                "source_kind": kind,
                "unit_kind": unit_kind,
                "aggregation_key": aggregation_key,
                "aggregation_kind": aggregation_kind_for(document),
                "kind": kind,
                "path": document.get("path"),
                "name": document.get("name"),
                "qualified_name": document.get("qualified_name"),
                "symbol_id": document.get("symbol_id"),
                "title": document.get("title"),
                "preview": preview_text(unit_text, fallback=str(document.get("preview") or "")),
                "content": unit_text,
                "start_line": chunk.get("start_line"),
                "end_line": chunk.get("end_line"),
                "token_estimate": estimate_tokens(unit_text),
                "char_count": len(unit_text),
            }
        )
    return units


def chunk_code_like_text(text: str) -> List[Dict[str, object]]:
    if estimate_tokens(text) <= CODE_EMBED_TOKEN_LIMIT and len(text) <= EMBED_CHAR_LIMIT:
        return [{"text": text, "start_line": 1, "end_line": line_count(text)}]

    chunks: List[Dict[str, object]] = []
    current_lines: List[str] = []
    current_tokens = 0
    start_line = 1
    line_no = 0
    for line_no, line in enumerate(text.splitlines(), start=1):
        line_tokens = max(estimate_tokens(line), 1)
        would_exceed = current_lines and (
            current_tokens + line_tokens > CODE_EMBED_TOKEN_LIMIT
            or sum(len(value) + 1 for value in current_lines) + len(line) > EMBED_CHAR_LIMIT
        )
        if would_exceed:
            chunks.append(
                {
                    "text": "\n".join(current_lines),
                    "start_line": start_line,
                    "end_line": line_no - 1,
                }
            )
            overlap = trailing_overlap(current_lines)
            current_lines = overlap
            current_tokens = estimate_tokens("\n".join(current_lines))
            start_line = max(1, line_no - len(current_lines))
        current_lines.append(line)
        current_tokens += line_tokens

    if current_lines:
        chunks.append(
            {
                "text": "\n".join(current_lines),
                "start_line": start_line,
                "end_line": line_no,
            }
        )
    return split_oversize_chunks(chunks, CODE_EMBED_TOKEN_LIMIT)


def chunk_document_text(text: str) -> List[Dict[str, object]]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if not paragraphs:
        return chunk_text(text, DEFAULT_EMBED_TOKEN_LIMIT)

    chunks: List[Dict[str, object]] = []
    current: List[str] = []
    current_tokens = 0
    for paragraph in paragraphs:
        paragraph_tokens = estimate_tokens(paragraph)
        if current and (
            current_tokens + paragraph_tokens > DEFAULT_EMBED_TOKEN_LIMIT
            or sum(len(value) + 2 for value in current) + len(paragraph) > EMBED_CHAR_LIMIT
        ):
            chunks.append({"text": "\n\n".join(current), "start_line": None, "end_line": None})
            current = []
            current_tokens = 0
        if paragraph_tokens > DEFAULT_EMBED_TOKEN_LIMIT or len(paragraph) > EMBED_CHAR_LIMIT:
            chunks.extend(chunk_text(paragraph, DEFAULT_EMBED_TOKEN_LIMIT))
            continue
        current.append(paragraph)
        current_tokens += paragraph_tokens
    if current:
        chunks.append({"text": "\n\n".join(current), "start_line": None, "end_line": None})
    return split_oversize_chunks(chunks, DEFAULT_EMBED_TOKEN_LIMIT)


def chunk_text(text: str, token_limit: int) -> List[Dict[str, object]]:
    tokens = TOKEN_RE.findall(text)
    if len(tokens) <= token_limit and len(text) <= EMBED_CHAR_LIMIT:
        return [{"text": text, "start_line": None, "end_line": None}]
    if len(tokens) <= token_limit:
        return char_chunks(text)

    chunks: List[Dict[str, object]] = []
    for start in range(0, len(tokens), max(token_limit // 2, 1)):
        part = " ".join(tokens[start : start + token_limit]).strip()
        if part:
            if len(part) > EMBED_CHAR_LIMIT:
                chunks.extend(char_chunks(part))
            else:
                chunks.append({"text": part, "start_line": None, "end_line": None})
        if start + token_limit >= len(tokens):
            break
    return chunks


def char_chunks(text: str) -> List[Dict[str, object]]:
    chunks: List[Dict[str, object]] = []
    stride = max(EMBED_CHAR_LIMIT - 1000, 1)
    for start in range(0, len(text), stride):
        part = text[start : start + EMBED_CHAR_LIMIT].strip()
        if part:
            chunks.append({"text": part, "start_line": None, "end_line": None})
        if start + EMBED_CHAR_LIMIT >= len(text):
            break
    return chunks


def split_oversize_chunks(chunks: Iterable[Dict[str, object]], token_limit: int) -> List[Dict[str, object]]:
    safe_chunks: List[Dict[str, object]] = []
    for chunk in chunks:
        text = str(chunk.get("text") or "")
        if estimate_tokens(text) <= token_limit and len(text) <= EMBED_CHAR_LIMIT:
            safe_chunks.append(chunk)
            continue
        for part in chunk_text(text, token_limit):
            updated = dict(part)
            updated["start_line"] = chunk.get("start_line")
            updated["end_line"] = chunk.get("end_line")
            safe_chunks.append(updated)
    return safe_chunks


def trailing_overlap(lines: List[str]) -> List[str]:
    overlap: List[str] = []
    tokens = 0
    target = max(MIN_CODE_WINDOW_TOKENS, CODE_EMBED_TOKEN_LIMIT // 3)
    for line in reversed(lines):
        line_tokens = max(estimate_tokens(line), 1)
        if overlap and tokens + line_tokens > target:
            break
        overlap.append(line)
        tokens += line_tokens
    return list(reversed(overlap))


def estimate_tokens(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def line_count(text: str) -> int:
    return max(len(text.splitlines()), 1)


def aggregation_key_for(document: Dict[str, object]) -> str:
    symbol_id = document.get("symbol_id")
    if symbol_id:
        return f"symbol:{symbol_id}"
    path = document.get("path")
    if path:
        return f"path:{path}"
    return f"doc:{document['doc_id']}"


def aggregation_kind_for(document: Dict[str, object]) -> str:
    if document.get("symbol_id"):
        return "symbol"
    if document.get("path"):
        return "path"
    return "document"


def stable_unit_id(source_doc_id: str, ordinal: int, text: str) -> str:
    digest = hashlib.sha1(f"{source_doc_id}\0{ordinal}\0{text}".encode("utf-8")).hexdigest()[:16]
    return f"{source_doc_id}:unit:{ordinal}:{digest}"


def preview_text(text: str, *, fallback: str = "") -> str:
    collapsed = " ".join(text.split())
    if not collapsed:
        collapsed = fallback
    return collapsed[:240]
