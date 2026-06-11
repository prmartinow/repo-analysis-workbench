from __future__ import annotations

import json
from typing import Dict


def encode_payload(payload: Dict[str, object]) -> bytes:
    return json.dumps(payload, separators=(",", ":"), sort_keys=False).encode("utf-8")


def decode_payload(blob: bytes | str | None) -> Dict[str, object]:
    if blob is None:
        return {}
    if isinstance(blob, bytes):
        blob = blob.decode("utf-8")
    if not blob:
        return {}
    return json.loads(blob)

