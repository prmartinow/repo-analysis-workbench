import base64
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from search.zoekt_backend import build_zoekt_index, search_zoekt_index


class ZoektBackendTest(unittest.TestCase):
    def test_build_reports_missing_zoekt_index_binary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("search.zoekt_backend.shutil.which", return_value=None):
                payload = build_zoekt_index("demo", Path(tmpdir), Path(tmpdir) / "zoekt")

        self.assertFalse(payload["available"])
        self.assertFalse(payload["built"])
        self.assertIn("not found", payload["diagnostics"][0])

    def test_build_runs_zoekt_index_with_expected_index_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir) / "repo"
            repo_root.mkdir()
            zoekt_root = Path(tmpdir) / "zoekt"

            def fake_runner(args, **kwargs):
                self.assertEqual(args[0], "/bin/zoekt-index")
                self.assertEqual(args[1], "-index")
                self.assertEqual(args[2], str(zoekt_root / "demo"))
                self.assertEqual(args[3], str(repo_root))
                return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

            payload = build_zoekt_index(
                "demo",
                repo_root,
                zoekt_root,
                zoekt_index_bin="/bin/zoekt-index",
                runner=fake_runner,
            )

        self.assertTrue(payload["available"])
        self.assertTrue(payload["built"])

    def test_search_parses_zoekt_jsonl_line_matches(self) -> None:
        line = base64.b64encode(b"hello world").decode("ascii")
        stdout = json.dumps(
            {
                "FileName": "README.md",
                "Repository": "demo",
                "Language": "Markdown",
                "Score": 3.5,
                "LineMatches": [
                    {
                        "Line": line,
                        "LineNumber": 7,
                        "Score": 1.25,
                    }
                ],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            zoekt_root = Path(tmpdir)
            (zoekt_root / "demo").mkdir()

            def fake_runner(args, **kwargs):
                self.assertEqual(args[:3], ["/bin/zoekt", "-index_dir", str(zoekt_root / "demo")])
                self.assertIn("-jsonl", args)
                return subprocess.CompletedProcess(args, 0, stdout=stdout + "\n", stderr="")

            payload = search_zoekt_index(
                "demo",
                zoekt_root,
                "hello",
                zoekt_bin="/bin/zoekt",
                runner=fake_runner,
            )

        result = payload["results"][0]
        self.assertTrue(payload["available"])
        self.assertEqual(result["path"], "README.md")
        self.assertEqual(result["score"], 3.5)
        self.assertEqual(result["line_matches"][0]["preview"], "hello world")
        self.assertEqual(result["line_matches"][0]["line_number"], 7)


if __name__ == "__main__":
    unittest.main()
