import json
import sys
import tempfile
import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from graph.builder import build_graph_artifact
from symbols.indexer import build_symbol_index
from symbols.persistence import load_symbol_index, write_metadata_bundle


class SymbolSnapshotTest(unittest.TestCase):
    def test_semantic_fixture_matches_golden_snapshot(self) -> None:
        tests_root = Path(__file__).resolve().parents[1]
        fixture_path = tests_root / "fixtures" / "rust" / "semantic_sample.rs"
        golden_path = tests_root / "golden" / "rust" / "semantic_sample.snapshot.json"

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
                fixture_path.read_text(encoding="utf-8"),
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
            graph = build_graph_artifact(artifact)

            snapshot = {
                "summary": artifact["summary"],
                "parser_backends": {
                    "rustc_ast_probe": {
                        "available": artifact["parser_backends"]["rustc_ast_probe"]["available"],
                        "used": artifact["parser_backends"]["rustc_ast_probe"]["used"],
                        "files": artifact["parser_backends"]["rustc_ast_probe"]["files"],
                        "parsed_files": artifact["parser_backends"]["rustc_ast_probe"]["parsed_files"],
                        "item_counts": artifact["parser_backends"]["rustc_ast_probe"]["item_counts"],
                        "statement_counts": artifact["parser_backends"]["rustc_ast_probe"]["statement_counts"],
                        "control_counts": artifact["parser_backends"]["rustc_ast_probe"]["control_counts"],
                    }
                },
                "symbols": [
                    {
                        "kind": symbol["kind"],
                        "qualified_name": symbol["qualified_name"],
                        "statement_id": bool(symbol["statement_id"]),
                    }
                    for symbol in artifact["symbols"]
                ],
                "imports": [
                    {
                        "target": entry["target"],
                        "target_qualified_name": entry["target_qualified_name"],
                    }
                    for entry in artifact["imports"]
                ],
                "references": [
                    {
                        "kind": entry["kind"],
                        "container": entry["container_qualified_name"],
                        "target": entry["target_qualified_name"],
                    }
                    for entry in artifact["references"]
                ],
                "statements": [
                    {
                        "kind": entry["kind"],
                        "container": entry["container_qualified_name"],
                        "text": entry["text"],
                        "defines": [item["target_qualified_name"] for item in entry["defines"]],
                        "reads": [item["target_qualified_name"] for item in entry["reads"]],
                        "writes": [item["target_qualified_name"] for item in entry["writes"]],
                        "calls": [item["target_qualified_name"] for item in entry["calls"]],
                    }
                    for entry in artifact["statements"]
                ],
                "graph_edge_counts": graph["summary"]["edge_counts"],
            }

            expected = json.loads(golden_path.read_text(encoding="utf-8"))
            self.assertEqual(snapshot, expected)

    def test_persistence_writes_metadata_lmdb(self) -> None:
        tests_root = Path(__file__).resolve().parents[1]
        fixture_path = tests_root / "fixtures" / "rust" / "semantic_sample.rs"

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            repo_root = workspace / "demo"
            raw_root = workspace / "raw"
            output_root = workspace / "parsed"

            (repo_root / "src").mkdir(parents=True)
            (repo_root / "Cargo.toml").write_text(
                '[package]\nname = "demo-crate"\nversion = "0.1.0"\n',
                encoding="utf-8",
            )
            (repo_root / "src" / "lib.rs").write_text(
                fixture_path.read_text(encoding="utf-8"),
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

            write_metadata_bundle(output_root, "demo", artifact)

            metadata_path = output_root / "demo" / "metadata.lmdb"
            self.assertTrue(metadata_path.exists(), metadata_path)

            restored = load_symbol_index(output_root, "demo")
            self.assertEqual(len(restored["symbols"]), artifact["summary"]["symbols"])
            self.assertEqual(len(restored["imports"]), artifact["summary"]["imports"])
            self.assertEqual(len(restored["references"]), artifact["summary"]["references"])
            self.assertEqual(len(restored["statements"]), artifact["summary"]["statements"])
