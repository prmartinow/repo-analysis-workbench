import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from symbols.indexer import build_symbol_index


def write_raw_manifest(raw_root: Path, parser_roots: list[str]) -> None:
    (raw_root / "demo").mkdir(parents=True)
    (raw_root / "demo" / "manifest.json").write_text(
        json.dumps({"parser_relevant_source_roots": parser_roots}),
        encoding="utf-8",
    )


class ParserFusionTest(unittest.TestCase):
    def test_tree_sitter_can_become_primary_backend_when_rust_analyzer_is_unavailable(self) -> None:
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
                        "pub struct Demo;",
                        "",
                        "pub fn helper() -> Demo {",
                        "    Demo",
                        "}",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            write_raw_manifest(raw_root, ["src"])

            with patch(
                "symbols.indexer.probe_rust_ast",
                return_value={
                    "backend": "rustc-ast-tree",
                    "available": True,
                    "used": True,
                    "parsed": True,
                    "path": "src/lib.rs",
                    "latency_ms": 0.1,
                    "item_counts": [{"kind": "function", "count": 1}],
                    "statement_counts": [],
                    "control_counts": [],
                    "diagnostics": [],
                },
            ), patch(
                "symbols.indexer.probe_rust_analyzer",
                return_value={
                    "backend": "rust-analyzer-lsp",
                    "available": False,
                    "used": False,
                    "parsed": False,
                    "path": "src/lib.rs",
                    "latency_ms": 0.1,
                    "item_counts": [],
                    "statement_counts": [],
                    "control_counts": [],
                    "document_symbols": [],
                    "diagnostics": ["unavailable"],
                },
            ), patch(
                "symbols.indexer.probe_tree_sitter",
                return_value={
                    "backend": "tree-sitter-rust",
                    "available": True,
                    "used": True,
                    "parsed": True,
                    "path": "src/lib.rs",
                    "latency_ms": 0.1,
                    "item_counts": [{"kind": "function", "count": 1}, {"kind": "struct", "count": 1}],
                    "statement_counts": [],
                    "control_counts": [],
                    "symbols": [
                        {
                            "name": "Demo",
                            "kind": "struct",
                            "qualified_name": "Demo",
                            "container_qualified_name": None,
                            "selection_range": {"start_line": 1, "start_column": 12},
                            "range": {"start_line": 1, "start_column": 1, "end_line": 1, "end_column": 17},
                            "signature": "pub struct Demo;",
                        },
                        {
                            "name": "helper",
                            "kind": "function",
                            "qualified_name": "helper",
                            "container_qualified_name": None,
                            "selection_range": {"start_line": 3, "start_column": 8},
                            "range": {"start_line": 3, "start_column": 1, "end_line": 5, "end_column": 2},
                            "signature": "pub fn helper() -> Demo {",
                        },
                    ],
                    "error_nodes": 0,
                    "diagnostics": [],
                },
            ):
                artifact = build_symbol_index("demo", repo_root, raw_root, path_prefixes=("src/lib.rs",))

            self.assertEqual(artifact["files"][0]["primary_parser_backend"], "tree_sitter_rust")
            self.assertEqual(artifact["primary_parser_backends"], [{"kind": "tree_sitter_rust", "count": 1}])
            self.assertIn("demo_crate::helper", {item["qualified_name"] for item in artifact["symbols"]})

    def test_workspace_dependency_aliases_resolve_cross_crate_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            repo_root = workspace / "demo"
            raw_root = workspace / "raw"

            (repo_root / "crate-a" / "src").mkdir(parents=True)
            (repo_root / "crate-b" / "src").mkdir(parents=True)

            (repo_root / "Cargo.toml").write_text(
                "\n".join(
                    [
                        "[workspace]",
                        'members = ["crate-a", "crate-b"]',
                        'resolver = "2"',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (repo_root / "crate-a" / "Cargo.toml").write_text(
                "\n".join(
                    [
                        "[package]",
                        'name = "crate-a"',
                        'version = "0.1.0"',
                        "",
                        "[dependencies]",
                        'helper_dep = { path = "../crate-b", package = "crate-b" }',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (repo_root / "crate-b" / "Cargo.toml").write_text(
                '[package]\nname = "crate-b"\nversion = "0.1.0"\n',
                encoding="utf-8",
            )
            (repo_root / "crate-b" / "src" / "lib.rs").write_text(
                "\n".join(
                    [
                        "pub struct Helper;",
                        "",
                        "pub fn external() -> Helper {",
                        "    Helper",
                        "}",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (repo_root / "crate-a" / "src" / "lib.rs").write_text(
                "\n".join(
                    [
                        "use helper_dep::{external, Helper};",
                        "",
                        "pub fn run() -> Helper {",
                        "    external()",
                        "}",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            write_raw_manifest(raw_root, ["crate-a/src", "crate-b/src"])

            artifact = build_symbol_index("demo", repo_root, raw_root)

            imports = {
                (item["path"], item["target"], item["target_qualified_name"])
                for item in artifact["imports"]
                if item["path"] == "crate-a/src/lib.rs"
            }
            self.assertIn(
                ("crate-a/src/lib.rs", "helper_dep::external", "crate_b::external"),
                imports,
            )

            cross_crate_calls = [
                item
                for item in artifact["references"]
                if item["path"] == "crate-a/src/lib.rs"
                and item["kind"] == "call"
                and item["target_qualified_name"] == "crate_b::external"
            ]
            self.assertEqual(len(cross_crate_calls), 1)


if __name__ == "__main__":
    unittest.main()
