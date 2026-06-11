from __future__ import annotations

from typing import Dict, Tuple


SUMMARY_STAGE_KINDS: Tuple[str, ...] = ("repo", "package", "directory", "file", "doc")
SYMBOL_STAGE_KINDS: Tuple[str, ...] = ("symbol",)
BODY_STAGE_KINDS: Tuple[str, ...] = ("function_body", "type_body")
GRAPH_STAGE_EDGE_TYPES: Tuple[str, ...] = (
    "DECLARES",
    "CONTAINS",
    "CALLS",
    "READS",
    "WRITES",
    "REFERENCES",
    "IMPLEMENTS",
    "INHERITS",
    "IMPORTS",
    "SUMMARIZED_BY",
)

GRAPH_EDGE_BONUS: Dict[str, float] = {
    "SUMMARIZED_BY": 2.1,
    "DECLARES": 1.9,
    "CONTAINS": 1.6,
    "IMPLEMENTS": 1.5,
    "INHERITS": 1.4,
    "CALLS": 1.2,
    "READS": 0.9,
    "WRITES": 0.9,
    "REFERENCES": 0.8,
    "IMPORTS": 0.7,
}

SUMMARY_FANOUT_MULTIPLIER = 3
SYMBOL_FANOUT_MULTIPLIER = 4
MAX_SUMMARY_SCOPES = 3
MAX_SYMBOL_SEEDS = 4
MAX_BODY_SEEDS = 3
EXACT_SYMBOL_MAX_TOKENS = 3
EMBEDDING_MIN_TOKENS = 5

RANK_KIND_WEIGHTS: Dict[str, float] = {
    "symbol": 30.0,
    "type_body": 16.0,
    "function_body": 12.0,
    "file": 8.0,
    "doc": 8.0,
    "directory": 2.0,
    "package": 1.0,
    "repo": 0.0,
    "statement": -35.0,
    "module_ref": -25.0,
    "symbol_ref": -25.0,
    "type_ref": -25.0,
}

RANK_SYMBOL_KIND_WEIGHTS: Dict[str, float] = {
    "trait": 24.0,
    "struct": 22.0,
    "enum": 20.0,
    "type": 18.0,
    "impl": 14.0,
    "module": 9.0,
    "associated_function": 7.0,
    "method": 6.0,
    "function": 5.0,
    "const": -8.0,
    "static": -8.0,
    "field": -18.0,
    "local": -24.0,
    "parameter": -20.0,
    "variable": -20.0,
}


def semantic_activity_score(semantic_summary: Dict[str, object]) -> float:
    direct_calls = len(semantic_summary.get("direct_calls", []) or [])
    reads = len(semantic_summary.get("reads", []) or [])
    writes = len(semantic_summary.get("writes", []) or [])
    interprocedural_reads = len(semantic_summary.get("interprocedural_reads", []) or [])
    interprocedural_writes = len(semantic_summary.get("interprocedural_writes", []) or [])
    interprocedural_references = len(semantic_summary.get("interprocedural_references", []) or [])
    transitive_calls = len(semantic_summary.get("transitive_calls", []) or [])

    weighted = (
        direct_calls * 4.0
        + reads * 1.25
        + writes * 1.25
        + interprocedural_reads * 1.0
        + interprocedural_writes * 1.0
        + interprocedural_references * 0.75
        + transitive_calls * 0.35
    )
    return min(weighted, 240.0) * 0.7
