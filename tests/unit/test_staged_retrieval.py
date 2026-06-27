import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evaluation.harness import SUPPORTED_BENCHMARK_MODES, run_case  # noqa: E402
from retrieval.staged import retrieve_paper_pipeline  # noqa: E402


class FakeSearchBackend:
    def __init__(self, results: list[dict[str, object]]) -> None:
        self.results = results
        self.calls: list[dict[str, object]] = []

    def search(self, query: str, *, limit: int, kinds=(), path_prefix=None) -> list[dict[str, object]]:
        self.calls.append(
            {
                "query": query,
                "limit": limit,
                "kinds": kinds,
                "path_prefix": path_prefix,
            }
        )
        return self.results[:limit]


class StagedRetrievalTest(unittest.TestCase):
    def test_paper_pipeline_is_supported_benchmark_mode(self) -> None:
        self.assertIn("paper_pipeline", SUPPORTED_BENCHMARK_MODES)

    def test_paper_pipeline_uses_large_bm25_pool_batches_rerank_and_aggregates_maxp(self) -> None:
        lexical_results = [
            {
                "doc_id": f"doc-{index}",
                "kind": "file",
                "path": "src/winner.py" if index < 2 else f"src/other_{index}.py",
                "name": "winner.py" if index < 2 else f"other_{index}.py",
                "qualified_name": None,
                "symbol_id": None,
                "title": "src/winner.py" if index < 2 else f"src/other_{index}.py",
                "preview": "candidate",
                "searchable": f"session state sync target passage {index}",
                "score": 100.0 - index,
            }
            for index in range(8)
        ]
        fake_search = FakeSearchBackend(lexical_results)
        qwen_calls: list[list[str]] = []

        def fake_qwen_rerank(_query: str, documents: list[str]) -> list[dict[str, object]]:
            qwen_calls.append(documents)
            return [
                {
                    "index": index,
                    "score": 10.0 - index,
                }
                for index, _document in enumerate(documents)
            ]

        with (
            mock.patch.dict(os.environ, {"REPO_ANALYSIS_RERANK_PROVIDER": "qwen"}),
            mock.patch("retrieval.staged.get_search_backend", return_value=fake_search),
            mock.patch(
                "retrieval.staged.query_embedding_index",
                return_value=[
                    {
                        "doc_id": "path:src/winner.py",
                        "kind": "file",
                        "path": "src/winner.py",
                        "score": 0.9,
                        "metadata": {
                            "embedding_aggregation_key": "path:src/winner.py",
                            "embedding_unit_hits": [],
                        },
                    }
                ],
            ),
            mock.patch("retrieval.staged.qwen_rerank", side_effect=fake_qwen_rerank),
        ):
            payload = retrieve_paper_pipeline(
                Path("/tmp/search"),
                "demo",
                "session state sync",
                limit=3,
                lexical_pool=8,
                pre_rank_pool=8,
                rerank_pool=7,
            )

        self.assertEqual(fake_search.calls[0]["limit"], 8)
        self.assertEqual([len(call) for call in qwen_calls], [5, 2])
        self.assertEqual(payload["summary"]["mode"], "paper_pipeline")
        self.assertEqual(payload["summary"]["embedding_aggregation"], "maxp")
        self.assertLessEqual(len(payload["selected_context"]), 3)
        winner = payload["selected_context"][0]
        self.assertEqual(winner["path"], "src/winner.py")
        self.assertEqual(winner["metadata"]["stage"], "source_maxp")
        self.assertIn("paper_unit_hits", winner["metadata"])

    def test_run_case_can_execute_paper_pipeline_mode(self) -> None:
        case = {
            "name": "demo-case",
            "repo": "demo",
            "task_type": "retrieval",
            "query": "session state sync",
            "expected_path": "src/winner.py",
            "expected_name": "",
            "expected_paths": ["src/winner.py"],
            "expected_symbols": [],
            "expected_terms": [],
        }

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch(
                "evaluation.harness.retrieve_paper_pipeline",
                return_value={
                    "selected_context": [
                        {
                            "doc_id": "path:src/winner.py",
                            "kind": "file",
                            "repo": "demo",
                            "path": "src/winner.py",
                            "name": "winner.py",
                            "qualified_name": None,
                            "symbol_id": None,
                            "title": "src/winner.py",
                            "preview": "session state sync",
                            "score": 10.0,
                            "metadata": {"mode": "paper_pipeline"},
                        }
                    ],
                    "summary": {"mode": "paper_pipeline"},
                },
            ),
            mock.patch(
                "evaluation.harness.prepare_answer_bundle",
                return_value={
                    "bundles": [
                        {
                            "selected_context": [
                                {
                                    "path": "src/winner.py",
                                    "name": "winner.py",
                                    "title": "src/winner.py",
                                    "preview": "session state sync",
                                }
                            ],
                            "evidence": [],
                            "graph_neighborhoods": [],
                            "statement_slices": [],
                        }
                    ]
                },
            ),
        ):
            root = Path(tmpdir)
            run = run_case(case, "paper_pipeline", root / "search", root / "graph", root / "parsed", limit=5)

        self.assertTrue(run["path_hit"])
        self.assertEqual(run["context_summary"]["mode"], "paper_pipeline")


if __name__ == "__main__":
    unittest.main()
