import sys
import tempfile
import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from parsers.tree_sitter_backend import discover_tree_sitter_files, probe_tree_sitter_tags


class FakeNode:
    def __init__(
        self,
        node_type,
        start_byte,
        end_byte,
        start_point,
        end_point,
        *,
        children=(),
        fields=None,
        has_error=False,
    ):
        self.type = node_type
        self.start_byte = start_byte
        self.end_byte = end_byte
        self.start_point = start_point
        self.end_point = end_point
        self.children = list(children)
        self.fields = fields or {}
        self.has_error = has_error

    def child_by_field_name(self, name):
        return self.fields.get(name)


class FakeTree:
    def __init__(self, root_node):
        self.root_node = root_node


class FakeParser:
    def __init__(self, root_node):
        self.root_node = root_node

    def parse(self, _source_bytes):
        return FakeTree(self.root_node)


class TreeSitterBackendTest(unittest.TestCase):
    def test_probe_normalizes_nested_python_symbols(self) -> None:
        source = "class Worker:\n    def run(self):\n        return 1\n"
        worker_start = source.index("Worker")
        run_start = source.index("run")
        worker_name = FakeNode("identifier", worker_start, worker_start + len("Worker"), (0, 6), (0, 12))
        run_name = FakeNode("identifier", run_start, run_start + len("run"), (1, 8), (1, 11))
        run_node = FakeNode(
            "function_definition",
            source.index("    def"),
            len(source),
            (1, 4),
            (2, 16),
            children=[run_name],
            fields={"name": run_name},
        )
        class_node = FakeNode(
            "class_definition",
            0,
            len(source),
            (0, 0),
            (2, 16),
            children=[worker_name, run_node],
            fields={"name": worker_name},
        )
        root = FakeNode("module", 0, len(source), (0, 0), (2, 16), children=[class_node])

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            (repo_root / "src").mkdir()
            (repo_root / "src" / "app.py").write_text(source, encoding="utf-8")
            repo_map = {
                "files": [
                    {
                        "path": "src/app.py",
                        "extension": ".py",
                        "language": "Python",
                        "generated": False,
                        "size": len(source),
                    }
                ]
            }

            payload = probe_tree_sitter_tags(
                "demo",
                repo_root,
                repo_map,
                parser_loader=lambda language: (FakeParser(root), (f"loaded {language}",)),
            )

        symbols = payload["symbol_records"]
        self.assertTrue(payload["available"])
        self.assertEqual(payload["parsed_files"], 1)
        self.assertEqual(payload["languages"], ["python"])
        self.assertEqual([symbol["qualified_name"] for symbol in symbols], ["Worker", "Worker::run"])
        self.assertEqual(symbols[0]["provider"], "tree_sitter_tags")
        self.assertEqual(symbols[0]["confidence"], 0.78)
        self.assertEqual(symbols[1]["scope"]["qualified_name"], "Worker")
        self.assertEqual(symbols[1]["range"]["start_line"], 2)

    def test_probe_reports_missing_language_parser(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            (repo_root / "main.go").write_text("package main\nfunc main() {}\n", encoding="utf-8")
            repo_map = {
                "files": [
                    {
                        "path": "main.go",
                        "extension": ".go",
                        "language": "Go",
                        "generated": False,
                        "size": 28,
                    }
                ]
            }

            payload = probe_tree_sitter_tags(
                "demo",
                repo_root,
                repo_map,
                parser_loader=lambda _language: (None, ("not installed",)),
            )

        self.assertFalse(payload["available"])
        self.assertTrue(payload["used"])
        self.assertEqual(payload["missing_languages"], ["go"])
        self.assertEqual(payload["symbols"], 0)
        self.assertIn("go: not installed", payload["diagnostics"])

    def test_discover_tree_sitter_files_filters_generated_and_large_files(self) -> None:
        repo_map = {
            "files": [
                {"path": "src/app.ts", "size": 20, "generated": False},
                {"path": "dist/bundle.js", "size": 20, "generated": False},
                {"path": "src/generated/client.py", "size": 20, "generated": False},
                {"path": "web/public/app.js", "size": 20, "generated": False},
                {"path": "src/huge.py", "size": 1_000_000, "generated": False},
                {"path": "README.md", "size": 20, "generated": False},
            ]
        }

        self.assertEqual(discover_tree_sitter_files(repo_map), ["src/app.ts"])


if __name__ == "__main__":
    unittest.main()
