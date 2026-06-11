from __future__ import annotations

from pathlib import Path
import unittest


SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
RUNTIME_FILES = (
    SRC_ROOT / "retrieval" / "engine.py",
    SRC_ROOT / "retrieval" / "planner.py",
    SRC_ROOT / "agents" / "toolkit.py",
    SRC_ROOT / "evaluation" / "harness.py",
)
FORBIDDEN_SNIPPETS = (
    "import sqlite3",
    "from search.indexer import",
    "load_symbol_index(",
    "load_symbol_by_id(",
    "inspect_graph_backend_payload_uncached(",
    "documents.jsonl",
    "search_manifest.json",
    "graph_manifest.json",
    "query_manifest.json",
    "symbols.sqlite3",
    "summary.sqlite3",
)


class RuntimeBoundaryTests(unittest.TestCase):
    def test_runtime_modules_do_not_bypass_backend_boundaries(self) -> None:
        violations = []
        for path in RUNTIME_FILES:
            contents = path.read_text(encoding="utf-8")
            for snippet in FORBIDDEN_SNIPPETS:
                if snippet in contents:
                    violations.append(f"{path.name}: {snippet}")
        self.assertEqual([], violations, f"Runtime boundary violations detected: {violations}")
