from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from typing import Iterable, List, Sequence


DEFAULT_HASHING_MODEL = "hashing-tfidf-v2"
DEFAULT_OPENAI_MODEL = "text-embedding-3-small"
OPENAI_EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"


def resolve_embedding_provider(provider: str | None = None, model: str | None = None) -> dict[str, object]:
    normalized = (provider or os.environ.get("REPO_ANALYSIS_EMBEDDING_PROVIDER") or "auto").strip().lower()
    model_name = (model or os.environ.get("REPO_ANALYSIS_EMBEDDING_MODEL") or "").strip() or None

    if normalized in {"auto", ""}:
        if os.environ.get("OPENAI_API_KEY"):
            return {
                "provider": "openai",
                "model": model_name or DEFAULT_OPENAI_MODEL,
                "model_backed": True,
            }
        return {
            "provider": "hashing",
            "model": model_name or DEFAULT_HASHING_MODEL,
            "model_backed": False,
        }

    if normalized == "openai":
        return {
            "provider": "openai",
            "model": model_name or DEFAULT_OPENAI_MODEL,
            "model_backed": True,
        }

    return {
        "provider": "hashing",
        "model": model_name or DEFAULT_HASHING_MODEL,
        "model_backed": False,
    }


def openai_embeddings_available() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def embed_with_openai(texts: Sequence[str], model: str) -> List[List[float]]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    request = urllib.request.Request(
        OPENAI_EMBEDDINGS_URL,
        data=json.dumps({"input": list(texts), "model": model}).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI embeddings request failed: {exc.code} {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenAI embeddings request failed: {exc}") from exc

    data = payload.get("data", [])
    vectors = [item.get("embedding", []) for item in sorted(data, key=lambda item: int(item.get("index", 0)))]
    if len(vectors) != len(texts):
        raise RuntimeError("OpenAI embeddings response size did not match request size")
    return [normalize_dense_vector([float(value) for value in vector]) for vector in vectors]


def normalize_dense_vector(vector: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return [0.0 for _ in vector]
    return [float(value) / norm for value in vector]
