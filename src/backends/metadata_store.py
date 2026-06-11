from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence

from backends.lmdb.store import LmdbMetadataStore


class MetadataStore(Protocol):
    def get_file(self, path: str) -> Optional[Dict[str, object]]:
        ...

    def find_files_by_prefix(self, path_prefix: str, *, limit: int) -> List[Dict[str, object]]:
        ...

    def get_symbol(self, symbol_id: str) -> Optional[Dict[str, object]]:
        ...

    def get_symbols(self, symbol_ids: Sequence[str]) -> Dict[str, Dict[str, object]]:
        ...

    def resolve_qname(self, qname: str) -> List[str]:
        ...

    def resolve_name(self, name: str, *, repo: str | None = None) -> List[str]:
        ...

    def get_symbol_body(self, symbol_id: str) -> Optional[Dict[str, object]]:
        ...

    def get_summary_by_id(self, summary_id: str) -> Optional[Dict[str, object]]:
        ...

    def get_summary_by_path(self, path: str) -> List[Dict[str, object]]:
        ...

    def get_summary_by_symbol(self, symbol_id: str) -> List[Dict[str, object]]:
        ...

    def get_artifact_metadata(self, key: str) -> Optional[Dict[str, object]]:
        ...

    def get_eval_case(self, case_name: str, fingerprint: str) -> Optional[Dict[str, object]]:
        ...

    def put_eval_case(
        self,
        case_name: str,
        *,
        repo: str,
        task_type: str,
        query: str,
        limit_value: int,
        artifact_fingerprint: str,
        cache_fingerprint: str,
        bundle: Dict[str, object],
        prompt_payload: Dict[str, object],
        bundle_score: Dict[str, object],
    ) -> None:
        ...

    def artifact_fingerprint(self) -> str:
        ...


@lru_cache(maxsize=16)
def get_metadata_store(parsed_root: str, repo_name: str, eval_root: str | None = None) -> MetadataStore:
    return LmdbMetadataStore(
        parsed_root=Path(parsed_root),
        repo_name=repo_name,
        eval_root=Path(eval_root) if eval_root else None,
    )
