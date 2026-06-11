from __future__ import annotations

from pathlib import PurePosixPath
from typing import List


def tokenize(query: str) -> List[str]:
    tokens = []
    for raw_token in query.replace("::", " ").replace("/", " ").replace("-", " ").replace(".", " ").split():
        normalized = "".join(char for char in raw_token.lower() if char.isalnum() or char == "_")
        if normalized:
            tokens.append(normalized)
    return tokens


def path_terms(path: str, *, limit: int = 8) -> List[str]:
    parts = PurePosixPath(path).parts
    if not parts:
        return []

    terms: List[str] = []
    seen = set()
    last_index = len(parts) - 1
    for index, part in enumerate(parts):
        if not part or part == ".":
            continue
        segment = PurePosixPath(part).stem if index == last_index else part
        for token in tokenize(segment):
            if token in seen:
                continue
            seen.add(token)
            terms.append(token)
            if len(terms) >= limit:
                return terms
    return terms
