import json
import sys
import tempfile
import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from graph.builder import build_graph_artifact
from parsers.scip_backend import probe_scip_indexes


WORKER_SYMBOL = "scip-python python demo 1.0 src/app.py/Worker#"
RUN_SYMBOL = "scip-python python demo 1.0 src/app.py/Worker#run()."
BASE_SYMBOL = "scip-python python demo 1.0 src/base.py/Base#"


class ScipBackendTest(unittest.TestCase):
    def test_probe_scip_json_normalizes_symbols_references_and_relationships(self) -> None:
        payload = {
            "metadata": {"project_root": "file:///demo"},
            "documents": [
                {
                    "relative_path": "src/app.py",
                    "language": "python",
                    "symbols": [
                        {
                            "symbol": WORKER_SYMBOL,
                            "display_name": "Worker",
                            "kind": 7,
                            "relationships": [
                                {
                                    "symbol": BASE_SYMBOL,
                                    "is_implementation": True,
                                }
                            ],
                        },
                        {
                            "symbol": RUN_SYMBOL,
                            "display_name": "run",
                            "kind": 17,
                            "signature_documentation": {"language": "python", "text": "def run(self): int"},
                        },
                    ],
                    "occurrences": [
                        {
                            "symbol": WORKER_SYMBOL,
                            "range": [0, 6, 12],
                            "enclosing_range": [0, 0, 2, 23],
                            "symbol_roles": 1,
                        },
                        {
                            "symbol": RUN_SYMBOL,
                            "range": [1, 8, 11],
                            "enclosing_range": [1, 4, 2, 23],
                            "symbol_roles": 1,
                        },
                        {
                            "symbol": WORKER_SYMBOL,
                            "range": [2, 15, 21],
                            "symbol_roles": 8,
                        },
                    ],
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            (repo_root / "src").mkdir()
            (repo_root / "src" / "app.py").write_text(
                "class Worker:\n    def run(self):\n        return Worker()\n",
                encoding="utf-8",
            )
            scip_json = repo_root / "index.scip.json"
            scip_json.write_text(json.dumps(payload), encoding="utf-8")

            probe = probe_scip_indexes(
                "demo",
                repo_root,
                {
                    "files": [
                        {
                            "path": "src/app.py",
                            "language": "Python",
                            "content_hash": "abc",
                        }
                    ]
                },
                scip_indexes=[Path("index.scip.json")],
            )

        symbols = {symbol["scip_symbol"]: symbol for symbol in probe["symbol_records"]}
        run_symbol = symbols[RUN_SYMBOL]
        reference = probe["reference_records"][0]
        self.assertTrue(probe["available"])
        self.assertEqual(probe["symbols"], 2)
        self.assertEqual(probe["references"], 1)
        self.assertEqual(probe["relationships"], 1)
        self.assertEqual(probe["file_records"][0]["primary_parser_backend"], "scip")
        self.assertEqual(run_symbol["provider"], "scip")
        self.assertEqual(run_symbol["signature"], "def run(self): int")
        self.assertEqual(run_symbol["range"]["start_line"], 2)
        self.assertEqual(reference["container_symbol_id"], run_symbol["symbol_id"])
        self.assertEqual(reference["target_symbol_id"], symbols[WORKER_SYMBOL]["symbol_id"])
        self.assertEqual(reference["kind"], "read")

        symbol_index = {
            "repo": "demo",
            "files": probe["file_records"],
            "symbols": probe["symbol_records"],
            "imports": [],
            "references": probe["reference_records"],
            "statements": [],
        }
        graph = build_graph_artifact(symbol_index)
        edge_types = {edge["type"] for edge in graph["edges"]}
        self.assertIn("IMPLEMENTS", edge_types)
        self.assertIn("USES", edge_types)


if __name__ == "__main__":
    unittest.main()
