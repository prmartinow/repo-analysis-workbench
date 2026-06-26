import json
import sys
import unittest
from pathlib import Path
from unittest import mock


SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cli.main import enrich_progress_estimate
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


if __name__ == "__main__":
    unittest.main()
