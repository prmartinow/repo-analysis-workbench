import json
import sys
import unittest
from pathlib import Path
from unittest import mock


SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cli.main import enrich_progress_estimate
from embeddings.indexer import build_qwen_embedding_payload
from embeddings.providers import embed_with_qwen


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

        with (
            mock.patch("embeddings.indexer.qwen_embeddings_available", return_value=True),
            mock.patch("embeddings.indexer.iter_search_documents", return_value=[documents]),
            mock.patch("embeddings.indexer.embed_with_qwen", return_value=[[1.0, 0.0]]) as embed,
        ):
            build_qwen_embedding_payload(Path("/tmp/search"), "demo", "text", progress_callback=events.append)

        self.assertEqual(events[0]["event"], "qwen_embed_started")
        self.assertEqual(events[1]["event"], "qwen_embed_batch_started")
        self.assertEqual(events[1]["batch_index"], 1)
        self.assertEqual(events[1]["processed_docs"], 0)
        self.assertEqual(events[1]["total_docs"], 1)
        self.assertEqual(events[2]["event"], "qwen_embed_progress")
        embed.assert_called_once_with(["hello"], "text")


if __name__ == "__main__":
    unittest.main()
