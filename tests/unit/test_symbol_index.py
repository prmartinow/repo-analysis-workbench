import json
import sys
import tempfile
import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from graph.builder import build_graph_artifact
from search.indexer import build_documents
from symbols.indexer import build_symbol_index
from symbols.schema import normalize_symbol_record


class SymbolIndexTest(unittest.TestCase):
    def test_build_symbol_index_uses_raw_roots_and_resolves_containers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            repo_root = workspace / "demo"
            raw_root = workspace / "raw"

            (repo_root / "src").mkdir(parents=True)
            (repo_root / "Cargo.toml").write_text(
                '[package]\nname = "demo-crate"\nversion = "0.1.0"\n',
                encoding="utf-8",
            )
            (repo_root / "src" / "lib.rs").write_text(
                "\n".join(
                    [
                        "use crate::support::Helper;",
                        "",
                        "pub fn helper() -> u64 {",
                        "    7",
                        "}",
                        "",
                        "pub struct Demo;",
                        "",
                        "impl Demo {",
                        "    pub fn answer(&self) -> u64 {",
                        "        helper()",
                        "    }",
                        "}",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            (raw_root / "demo").mkdir(parents=True)
            (raw_root / "demo" / "manifest.json").write_text(
                json.dumps({"parser_relevant_source_roots": ["src"]}),
                encoding="utf-8",
            )

            artifact = build_symbol_index(
                "demo",
                repo_root,
                raw_root,
                path_prefixes=("src/lib.rs",),
            )

            self.assertEqual(artifact["summary"]["rust_files"], 1)
            self.assertEqual(artifact["files"][0]["crate"], "demo-crate")
            self.assertEqual(artifact["imports"][0]["target"], "crate::support::Helper")
            self.assertGreater(artifact["summary"]["references"], 0)

            impl_symbol = next(item for item in artifact["symbols"] if item["kind"] == "impl")
            method_symbol = next(item for item in artifact["symbols"] if item["name"] == "answer")
            call_reference = next(item for item in artifact["references"] if item["kind"] == "call")

            self.assertEqual(method_symbol["container_symbol_id"], impl_symbol["symbol_id"])
            self.assertEqual(method_symbol["module_path"], "demo_crate")
            self.assertEqual(method_symbol["provider"], "rust_static")
            self.assertEqual(method_symbol["confidence"], 1.0)
            self.assertEqual(method_symbol["range"], method_symbol["span"])
            self.assertEqual(method_symbol["scope"]["symbol_id"], impl_symbol["symbol_id"])
            self.assertEqual(call_reference["container_symbol_id"], method_symbol["symbol_id"])

    def test_provider_neutral_shallow_symbol_feeds_graph_and_search_docs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir) / "demo"
            repo_root.mkdir()
            (repo_root / "main.py").write_text("def run():\n    return 1\n", encoding="utf-8")
            symbol = normalize_symbol_record(
                {
                    "symbol_id": "sym-python-run",
                    "repo": "demo",
                    "path": "main.py",
                    "language": "Python",
                    "kind": "function",
                    "name": "run",
                    "qualified_name": "run",
                    "range": {"start_line": 1, "start_column": 1, "end_line": 2, "end_column": 13},
                },
                provider="ctags",
                confidence=0.65,
            )
            symbol_index = {
                "repo": "demo",
                "files": [{"path": "main.py", "language": "Python", "symbols": 1}],
                "symbols": [symbol],
                "imports": [],
                "references": [],
                "statements": [],
                "summary": {"rust_files": 0, "symbols": 1, "imports": 0, "statements": 0},
            }
            repo_map = {
                "files": [{"path": "main.py", "language": "Python", "generated": False}],
                "directories": [{"path": ".", "depth": 0}],
            }

            graph = build_graph_artifact(symbol_index)
            symbol_node = next(item for item in graph["nodes"] if item["node_id"] == "sym-python-run")
            documents = list(build_documents("demo", repo_root, {"language_mix": []}, repo_map, symbol_index))
            symbol_doc = next(item for item in documents if item.get("symbol_id") == "sym-python-run")

            self.assertEqual(symbol_node["provider"], "ctags")
            self.assertEqual(symbol_node["range"]["start_line"], 1)
            self.assertEqual(symbol_doc["metadata"]["provider"], "ctags")
            self.assertEqual(symbol_doc["metadata"]["confidence"], 0.65)

    def test_build_symbol_index_reuses_cached_file_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            repo_root = workspace / "demo"
            raw_root = workspace / "raw"
            cache_root = workspace / "parsed" / "demo"

            (repo_root / "src").mkdir(parents=True)
            (repo_root / "Cargo.toml").write_text(
                '[package]\nname = "demo-crate"\nversion = "0.1.0"\n',
                encoding="utf-8",
            )
            (repo_root / "src" / "lib.rs").write_text(
                "pub fn answer() -> u64 {\n    42\n}\n",
                encoding="utf-8",
            )

            (raw_root / "demo").mkdir(parents=True)
            (raw_root / "demo" / "manifest.json").write_text(
                json.dumps({"parser_relevant_source_roots": ["src"]}),
                encoding="utf-8",
            )

            first = build_symbol_index(
                "demo",
                repo_root,
                raw_root,
                path_prefixes=("src/lib.rs",),
                cache_root=cache_root,
            )
            second = build_symbol_index(
                "demo",
                repo_root,
                raw_root,
                path_prefixes=("src/lib.rs",),
                cache_root=cache_root,
            )

            self.assertEqual(first["summary"], second["summary"])
            self.assertEqual(first["symbols"], second["symbols"])
            self.assertEqual(second["build_metrics"]["cache_hits"], 1)
            self.assertEqual(second["build_metrics"]["cache_misses"], 0)
            cache_file = cache_root / "file-cache" / "src" / "lib.rs.json"
            self.assertTrue(cache_file.exists(), cache_file)


if __name__ == "__main__":
    unittest.main()
