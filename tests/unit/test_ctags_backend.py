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

from parsers.ctags_backend import discover_ctags_files, probe_universal_ctags


class CtagsBackendTest(unittest.TestCase):
    def test_probe_reports_unavailable_when_ctags_is_missing(self) -> None:
        with mock.patch("parsers.ctags_backend.shutil.which", return_value=None):
            payload = probe_universal_ctags("demo", Path("/tmp/demo"), {"files": []})

        self.assertFalse(payload["available"])
        self.assertFalse(payload["used"])
        self.assertIn("not found", payload["diagnostics"][0])

    def test_probe_normalizes_json_tags(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            (repo_root / "main.py").write_text("def run():\n    return 1\n", encoding="utf-8")
            repo_map = {
                "files": [
                    {
                        "path": "main.py",
                        "extension": ".py",
                        "language": "Python",
                        "generated": False,
                    }
                ]
            }

            def fake_runner(args, **kwargs):
                self.assertIn("--output-format=json", args)
                self.assertEqual(kwargs["cwd"], repo_root)
                self.assertIsNone(kwargs.get("timeout"))
                stdout = json.dumps(
                    {
                        "_type": "tag",
                        "name": "run",
                        "path": "main.py",
                        "language": "Python",
                        "kind": "function",
                        "line": 1,
                        "end": 2,
                        "pattern": "/^def run():$/;\"",
                    }
                )
                return subprocess.CompletedProcess(args, 0, stdout=stdout + "\n", stderr="")

            with mock.patch("parsers.ctags_backend.shutil.which", return_value="/usr/bin/ctags"):
                payload = probe_universal_ctags("demo", repo_root, repo_map, runner=fake_runner)

            symbol = payload["symbol_records"][0]
            self.assertTrue(payload["available"])
            self.assertTrue(payload["used"])
            self.assertEqual(symbol["provider"], "ctags")
            self.assertEqual(symbol["language"], "Python")
            self.assertEqual(symbol["range"]["end_line"], 2)
            self.assertEqual(symbol["confidence"], 0.65)

    def test_discover_ctags_files_skips_public_bundles(self) -> None:
        repo_map = {
            "files": [
                {
                    "path": "src/main.ts",
                    "size": 120,
                    "generated": False,
                },
                {
                    "path": "web/public/monaco-editor/vs/editor/editor.main.js",
                    "size": 3_766_654,
                    "generated": False,
                },
                {
                    "path": "dist/bundle.js",
                    "size": 200,
                    "generated": False,
                },
            ]
        }

        self.assertEqual(discover_ctags_files(repo_map), ["src/main.ts"])


if __name__ == "__main__":
    unittest.main()
