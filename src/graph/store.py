from __future__ import annotations

from pathlib import Path
from typing import Dict

from backends.ryugraph.loader import load_ryugraph_database
from graph.query import reset_graph_view_cache


def write_graph_database(
    output_root: Path,
    repo_name: str,
    payload: Dict[str, object],
) -> Path:
    target = load_ryugraph_database(output_root, repo_name, payload)
    reset_graph_view_cache()
    return target
