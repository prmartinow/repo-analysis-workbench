import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock


SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common.native_tool import build_bm25_index, run_native_json


class NativeToolTimeoutTest(unittest.TestCase):
    def test_native_json_calls_have_no_subprocess_timeout_by_default(self) -> None:
        result = subprocess.CompletedProcess(["native"], 0, stdout=json.dumps({"ok": True}), stderr="")

        with (
            mock.patch("common.native_tool.ensure_native_binary", return_value=Path("/bin/repo-analysis-native")),
            mock.patch("common.native_tool.subprocess.run", return_value=result) as run,
        ):
            payload = run_native_json(["worker-info"])

        self.assertEqual(payload, {"ok": True})
        self.assertNotIn("timeout", run.call_args.kwargs)

    def test_bm25_build_uses_no_subprocess_timeout(self) -> None:
        result = subprocess.CompletedProcess(["native"], 0, stdout=json.dumps({"built": True}), stderr="")

        with (
            mock.patch("common.native_tool.ensure_native_binary", return_value=Path("/bin/repo-analysis-native")),
            mock.patch("common.native_tool.subprocess.run", return_value=result) as run,
        ):
            payload = build_bm25_index(Path("/tmp/docs.jsonl"), Path("/tmp/index"))

        self.assertEqual(payload, {"built": True})
        self.assertNotIn("timeout", run.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
