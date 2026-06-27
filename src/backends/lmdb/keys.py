from __future__ import annotations

import hashlib


LMDB_SAFE_KEY_BYTES = 480
HASHED_KEY_PREFIX = "__sha256__:"


def encode_key(value: str) -> bytes:
    raw = value.encode("utf-8")
    if len(raw) <= LMDB_SAFE_KEY_BYTES:
        return raw
    digest = hashlib.sha256(raw).hexdigest()
    return f"{HASHED_KEY_PREFIX}{digest}".encode("ascii")
