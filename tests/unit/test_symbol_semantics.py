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


class SymbolSemanticsTest(unittest.TestCase):
    def test_build_symbol_index_tracks_backends_trait_inheritance_and_self_member_resolution(self) -> None:
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
                        "pub trait Base {",
                        "    fn reset(&mut self);",
                        "}",
                        "",
                        "pub trait Derived: Base {",
                        "    fn increment(&mut self) -> u64;",
                        "}",
                        "",
                        "pub struct Service {",
                        "    count: u64,",
                        "}",
                        "",
                        "pub fn build_service() -> Service {",
                        "    Service { count: 0 }",
                        "}",
                        "",
                        "impl Base for Service {",
                        "    fn reset(&mut self) {",
                        "        self.count = 0;",
                        "    }",
                        "}",
                        "",
                        "impl Derived for Service {",
                        "    fn increment(&mut self) -> u64 {",
                        "        self.reset();",
                        "        self.count = self.count + 1;",
                        "        self.count",
                        "    }",
                        "}",
                        "",
                        "#[cfg(test)]",
                        "mod tests {",
                        "    use super::*;",
                        "",
                        "    #[test]",
                        "    fn smoke() {",
                        "        let service = build_service();",
                        "        let _ = service;",
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

            artifact = build_symbol_index("demo", repo_root, raw_root, path_prefixes=("src/lib.rs",))
            graph = build_graph_artifact(artifact)

            self.assertIn("tree_sitter_rust", artifact["parser_backends"])
            self.assertIn("rust_analyzer_lsp", artifact["parser_backends"])

            derived_trait = next(
                symbol for symbol in artifact["symbols"] if symbol["kind"] == "trait" and symbol["name"] == "Derived"
            )
            self.assertEqual(derived_trait["super_traits"], ["Base"])
            self.assertEqual(derived_trait["resolved_super_traits"][0]["target_qualified_name"], "demo_crate::Base")

            references = {(item["kind"], item["target_qualified_name"]) for item in artifact["references"]}
            self.assertIn(("call", "demo_crate::Service::reset"), references)
            self.assertIn(("use", "demo_crate::Service::count"), references)

            statement_targets = {
                target["target_qualified_name"]
                for statement in artifact["statements"]
                for target in list(statement["reads"]) + list(statement["writes"])
            }
            self.assertIn("demo_crate::Service::count", statement_targets)

            edge_types = {item["type"] for item in graph["summary"]["edge_counts"]}
            self.assertIn("INHERITS", edge_types)
            self.assertIn("TESTS", edge_types)


if __name__ == "__main__":
    unittest.main()
