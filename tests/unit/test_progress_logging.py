import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock


SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cli.main import enrich_progress_estimate
from embeddings.indexer import build_qwen_embedding_payload
from embeddings.providers import embed_with_qwen
from rerank.fusion import qwen_rerank


class ProgressLoggingTest(unittest.TestCase):
    def test_enrich_progress_estimate_adds_rate_percent_and_eta(self) -> None:
        payload = enrich_progress_estimate(
            {"processed_docs": 25, "total_docs": 100},
            elapsed_ms=5_000,
        )

        self.assertEqual(payload["progress_unit"], "docs")
        self.assertEqual(payload["percent_complete"], 25.0)
        self.assertEqual(payload["rate_per_sec"], 5.0)
        self.assertEqual(payload["eta_seconds"], 15.0)

    def test_qwen_embedding_requests_do_not_set_client_timeout(self) -> None:
        response = mock.Mock()
        response.read.return_value = json.dumps({"data": [{"index": 0, "embedding": [1.0, 0.0]}]}).encode("utf-8")
        response.__enter__ = mock.Mock(return_value=response)
        response.__exit__ = mock.Mock(return_value=None)

        with (
            mock.patch("embeddings.providers.qwen_embeddings_url", return_value="http://127.0.0.1:18200/v1/embeddings"),
            mock.patch("embeddings.providers.urllib.request.urlopen", return_value=response) as urlopen,
        ):
            embed_with_qwen(["hello"], "text")

        self.assertIsNone(urlopen.call_args.kwargs["timeout"])
        request = urlopen.call_args.args[0]
        self.assertEqual(request.get_header("X-workload"), "batch")

    def test_qwen_embedding_requests_include_safe_metadata_headers(self) -> None:
        response = mock.Mock()
        response.read.return_value = json.dumps({"data": [{"index": 0, "embedding": [1.0, 0.0]}]}).encode("utf-8")
        response.__enter__ = mock.Mock(return_value=response)
        response.__exit__ = mock.Mock(return_value=None)

        with (
            mock.patch("embeddings.providers.qwen_embeddings_url", return_value="http://127.0.0.1:18200/v1/embeddings"),
            mock.patch("embeddings.providers.urllib.request.urlopen", return_value=response) as urlopen,
        ):
            embed_with_qwen(["hello"], "text", headers={"X-Repo-Analysis-Repo": "demo", "X-Ignored": None})

        request = urlopen.call_args.args[0]
        self.assertEqual(request.get_header("X-repo-analysis-repo"), "demo")
        self.assertIsNone(request.get_header("X-ignored"))

    def test_qwen_rerank_requests_do_not_set_client_timeout(self) -> None:
        response = mock.Mock()
        response.read.return_value = json.dumps({"results": [{"index": 0, "score": 1.0}]}).encode("utf-8")
        response.__enter__ = mock.Mock(return_value=response)
        response.__exit__ = mock.Mock(return_value=None)

        with (
            mock.patch.dict("os.environ", {"REPO_ANALYSIS_QWEN_RERANK_URL": "http://127.0.0.1:18200/rerank"}),
            mock.patch("rerank.fusion.urllib.request.urlopen", return_value=response) as urlopen,
        ):
            qwen_rerank("helper", ["demo::helper"])

        self.assertIsNone(urlopen.call_args.kwargs["timeout"])

    def test_qwen_embedding_build_marks_batch_in_flight_before_request(self) -> None:
        events = []
        documents = [
            {
                "doc_id": "doc-demo",
                "kind": "file",
                "path": "README.md",
                "name": "README.md",
                "qualified_name": "README.md",
                "symbol_id": None,
                "title": "README",
                "preview": "hello",
                "content": "hello",
                "_total_docs": 1,
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                mock.patch("embeddings.indexer.qwen_embeddings_available", return_value=True),
                mock.patch("embeddings.indexer.iter_search_documents", return_value=[documents]),
                mock.patch("embeddings.indexer.embed_with_qwen", return_value=[[1.0, 0.0]]) as embed,
            ):
                build_qwen_embedding_payload(Path(tmpdir), "demo", "text", progress_callback=events.append)

        self.assertEqual(events[0]["event"], "qwen_embed_started")
        self.assertEqual(events[1]["event"], "qwen_embed_batch_started")
        self.assertEqual(events[1]["batch_index"], 1)
        self.assertEqual(events[1]["processed_docs"], 0)
        self.assertEqual(events[1]["total_docs"], 1)
        self.assertEqual(events[2]["event"], "qwen_embed_batch_completed")
        self.assertEqual(events[2]["processed_docs"], 1)
        self.assertEqual(events[2]["returned_vectors"], 1)
        self.assertEqual(events[3]["event"], "qwen_embed_progress")
        embed.assert_called_once()
        self.assertEqual(embed.call_args.args, (["hello"], "text"))
        self.assertEqual(embed.call_args.kwargs["headers"]["X-Repo-Analysis-Repo"], "demo")
        self.assertEqual(embed.call_args.kwargs["headers"]["X-Repo-Analysis-Batch-Docs"], 1)

    def test_qwen_embedding_build_emits_heartbeat_while_batch_waits(self) -> None:
        events = []
        documents = [
            {
                "doc_id": "doc-demo",
                "kind": "file",
                "path": "README.md",
                "name": "README.md",
                "qualified_name": "README.md",
                "symbol_id": None,
                "title": "README",
                "preview": "hello",
                "content": "hello",
                "_total_docs": 1,
            }
        ]

        def slow_embed(_texts, _model, **_kwargs):
            time.sleep(0.05)
            return [[1.0, 0.0]]

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                mock.patch.dict("os.environ", {"REPO_ANALYSIS_QWEN_HEARTBEAT_SECONDS": "0.01"}),
                mock.patch("embeddings.indexer.qwen_embeddings_available", return_value=True),
                mock.patch("embeddings.indexer.iter_search_documents", return_value=[documents]),
                mock.patch("embeddings.indexer.embed_with_qwen", side_effect=slow_embed),
            ):
                build_qwen_embedding_payload(Path(tmpdir), "demo", "text", progress_callback=events.append)

        heartbeat_events = [event for event in events if event["event"] == "qwen_embed_batch_waiting"]
        self.assertGreaterEqual(len(heartbeat_events), 1)
        self.assertEqual(heartbeat_events[0]["processed_docs"], 0)
        self.assertEqual(heartbeat_events[0]["total_docs"], 1)
        self.assertEqual(heartbeat_events[0]["batch_docs"], 1)
        self.assertEqual(heartbeat_events[0]["batch_input_chars"], 5)
        self.assertGreater(float(heartbeat_events[0]["batch_elapsed_ms"]), 0.0)

    def test_qwen_embedding_build_marks_batch_failure_without_retrying(self) -> None:
        events = []
        documents = [
            {
                "doc_id": "doc-demo",
                "kind": "file",
                "path": "README.md",
                "name": "README.md",
                "qualified_name": "README.md",
                "symbol_id": None,
                "title": "README",
                "preview": "hello",
                "content": "hello",
                "_total_docs": 1,
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                mock.patch("embeddings.indexer.qwen_embeddings_available", return_value=True),
                mock.patch("embeddings.indexer.iter_search_documents", return_value=[documents]),
                mock.patch("embeddings.indexer.embed_with_qwen", side_effect=RuntimeError("queue wait timed out")) as embed,
            ):
                with self.assertRaises(RuntimeError):
                    build_qwen_embedding_payload(Path(tmpdir), "demo", "text", progress_callback=events.append)

        self.assertEqual([event["event"] for event in events], [
            "qwen_embed_started",
            "qwen_embed_batch_started",
            "qwen_embed_batch_failed",
        ])
        self.assertEqual(events[2]["error_type"], "RuntimeError")
        embed.assert_called_once()
        self.assertEqual(embed.call_args.args, (["hello"], "text"))


if __name__ == "__main__":
    unittest.main()
