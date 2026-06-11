from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence

from backends.tantivy.search import TantivySearchBackend


class SearchBackend(Protocol):
    def search(
        self,
        query: str,
        *,
        limit: int,
        kinds: Sequence[str] = (),
        path_prefix: Optional[str] = None,
    ) -> List[Dict[str, object]]:
        ...

    def find_file(self, path_pattern: str, *, limit: int) -> List[Dict[str, object]]:
        ...

    def list_documents(
        self,
        *,
        limit: int,
        kinds: Sequence[str] = (),
        path_prefix: Optional[str] = None,
    ) -> List[Dict[str, object]]:
        ...

    def lookup_symbol_docs(self, symbol_id: str, *, kinds: Sequence[str] = (), limit: int = 20) -> List[Dict[str, object]]:
        ...

    def compare_repo_candidates(self, query: str, *, limit: int) -> List[Dict[str, object]]:
        ...

    def artifact_fingerprint(self) -> str:
        ...


@lru_cache(maxsize=16)
def get_search_backend(search_root: str, repo_name: str) -> SearchBackend:
    return TantivySearchBackend(Path(search_root), repo_name)
