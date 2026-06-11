from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Protocol

from backends.ryugraph.queries import RyuGraphBackend


class GraphBackend(Protocol):
    def execute(self, request: Dict[str, object]) -> Optional[Dict[str, object]]:
        ...

    def artifact_fingerprint(self) -> str:
        ...


@lru_cache(maxsize=16)
def get_graph_backend(graph_root: str, repo_name: str) -> GraphBackend:
    return RyuGraphBackend(Path(graph_root), repo_name)
