import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from embeddings.indexer import (  # noqa: E402
    append_qwen_embedding_checkpoint,
    build_embedded_unit_record,
    build_qwen_embedding_payload,
    embed_tokens,
    initialize_qwen_embedding_checkpoint,
    normalize_sparse_vector,
    qwen_embedding_checkpoint_paths,
    query_embedding_index,
)
from embeddings.units import (  # noqa: E402
    CODE_EMBED_TOKEN_LIMIT,
    EMBED_CHAR_LIMIT,
    build_retrieval_units,
)
from rerank.fusion import candidate_to_rerank_document  # noqa: E402


class EmbeddingUnitsTest(unittest.TestCase):
    def test_code_documents_split_into_bounded_line_windows(self) -> None:
        document = {
            "doc_id": "doc-large",
            "kind": "file",
            "path": "src/large.py",
            "name": "large.py",
            "qualified_name": None,
            "symbol_id": None,
            "title": "src/large.py",
            "preview": "large file",
            "content": "\n".join(f"def helper_{index}(): return value_{index}" for index in range(800)),
        }

        units = build_retrieval_units(document)

        self.assertGreater(len(units), 1)
        self.assertTrue(all(unit["unit_kind"] == "line_window" for unit in units))
        self.assertTrue(all(int(unit["token_estimate"]) <= CODE_EMBED_TOKEN_LIMIT for unit in units))
        self.assertTrue(all(int(unit["char_count"]) <= EMBED_CHAR_LIMIT for unit in units))
        self.assertTrue(all(unit["aggregation_key"] == "path:src/large.py" for unit in units))
        self.assertEqual(units[0]["source_doc_id"], "doc-large")
        self.assertEqual(units[0]["source_kind"], "file")
        self.assertIsNotNone(units[0]["start_line"])
        self.assertIsNotNone(units[0]["end_line"])

    def test_long_unbroken_text_is_split_by_char_limit(self) -> None:
        document = {
            "doc_id": "doc-minified",
            "kind": "file",
            "path": "dist/bundle.js",
            "name": "bundle.js",
            "qualified_name": None,
            "symbol_id": None,
            "title": "dist/bundle.js",
            "preview": "minified bundle",
            "content": "x" * (EMBED_CHAR_LIMIT * 2 + 100),
        }

        units = build_retrieval_units(document)

        self.assertGreater(len(units), 1)
        self.assertTrue(all(int(unit["char_count"]) <= EMBED_CHAR_LIMIT for unit in units))

    def test_qwen_embedding_payload_embeds_retrieval_units_not_whole_document(self) -> None:
        documents = [
            {
                "doc_id": "doc-demo",
                "kind": "file",
                "path": "src/large.py",
                "name": "large.py",
                "qualified_name": None,
                "symbol_id": None,
                "title": "src/large.py",
                "preview": "large file",
                "content": "\n".join(f"def helper_{index}(): return value_{index}" for index in range(800)),
                "_total_docs": 1,
            }
        ]
        embedded_inputs = []

        def fake_embed(inputs: list[str], _model_name: str, **_kwargs) -> list[list[float]]:
            embedded_inputs.extend(inputs)
            return [[1.0, 0.0] for _value in inputs]

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                mock.patch("embeddings.indexer.qwen_embeddings_available", return_value=True),
                mock.patch("embeddings.indexer.iter_search_documents", return_value=[documents]),
                mock.patch("embeddings.indexer.embed_with_qwen", side_effect=fake_embed),
            ):
                payload = build_qwen_embedding_payload(Path(tmpdir), "demo", "text")

        self.assertGreater(len(embedded_inputs), 1)
        self.assertTrue(all(len(value) <= EMBED_CHAR_LIMIT for value in embedded_inputs))
        self.assertEqual(payload["summary"]["documents"], len(embedded_inputs))
        self.assertEqual(payload["documents"][0]["source_doc_id"], "doc-demo")
        self.assertEqual(payload["documents"][0]["source_kind"], "file")
        self.assertIn("unit_id", payload["documents"][0])

    def test_qwen_embedding_payload_resumes_checkpointed_units(self) -> None:
        documents = [
            {
                "doc_id": "doc-a",
                "kind": "file",
                "path": "src/a.py",
                "name": "a.py",
                "qualified_name": None,
                "symbol_id": None,
                "title": "src/a.py",
                "preview": "alpha",
                "content": "alpha",
                "_total_docs": 2,
            },
            {
                "doc_id": "doc-b",
                "kind": "file",
                "path": "src/b.py",
                "name": "b.py",
                "qualified_name": None,
                "symbol_id": None,
                "title": "src/b.py",
                "preview": "beta",
                "content": "beta",
                "_total_docs": 2,
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            search_root = Path(tmpdir)
            checkpoint_path, checkpoint_meta_path = qwen_embedding_checkpoint_paths(search_root, "demo", "text")
            initialize_qwen_embedding_checkpoint(
                checkpoint_path,
                checkpoint_meta_path,
                repo_name="demo",
                model_name="text",
            )
            checkpointed_unit = build_retrieval_units(documents[0])[0]
            checkpointed_record = build_embedded_unit_record(checkpointed_unit, 1.0)
            checkpointed_record["vector"] = [0.25, 0.75]
            append_qwen_embedding_checkpoint(checkpoint_path, [checkpointed_record])

            events = []
            embedded_inputs = []

            def fake_embed(inputs: list[str], _model_name: str, **_kwargs) -> list[list[float]]:
                embedded_inputs.extend(inputs)
                return [[1.0, 0.0] for _value in inputs]

            with (
                mock.patch("embeddings.indexer.qwen_embeddings_available", return_value=True),
                mock.patch("embeddings.indexer.iter_search_documents", return_value=[documents]),
                mock.patch("embeddings.indexer.embed_with_qwen", side_effect=fake_embed),
            ):
                payload = build_qwen_embedding_payload(
                    search_root,
                    "demo",
                    "text",
                    progress_callback=events.append,
                )

        self.assertEqual(embedded_inputs, ["beta"])
        self.assertEqual(payload["summary"]["documents"], 2)
        self.assertTrue(any(event["event"] == "qwen_embed_checkpoint_loaded" for event in events))
        self.assertEqual(
            {document["source_doc_id"] for document in payload["documents"]},
            {"doc-a", "doc-b"},
        )

    def test_query_embedding_index_aggregates_unit_hits_by_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            search_root = Path(tmpdir)
            repo_root = search_root / "demo"
            repo_root.mkdir()
            vector = normalize_sparse_vector(embed_tokens(["needle"], None, 2))
            payload = {
                "provider": "hashing",
                "model": "hashing",
                "model_backed": False,
                "dimensions": 256,
                "vector_format": "sparse",
                "documents": [
                    {
                        "doc_id": "doc-a:unit:0",
                        "source_doc_id": "doc-a",
                        "source_kind": "file",
                        "unit_id": "doc-a:unit:0",
                        "unit_kind": "line_window",
                        "aggregation_key": "path:src/a.py",
                        "aggregation_kind": "path",
                        "kind": "file",
                        "path": "src/a.py",
                        "name": "a.py",
                        "qualified_name": None,
                        "symbol_id": None,
                        "title": "src/a.py",
                        "preview": "needle first",
                        "content": "needle first",
                        "start_line": 1,
                        "end_line": 10,
                        "token_estimate": 2,
                        "vector": {str(index): value for index, value in vector.items()},
                    },
                    {
                        "doc_id": "doc-a:unit:1",
                        "source_doc_id": "doc-a",
                        "source_kind": "file",
                        "unit_id": "doc-a:unit:1",
                        "unit_kind": "line_window",
                        "aggregation_key": "path:src/a.py",
                        "aggregation_kind": "path",
                        "kind": "file",
                        "path": "src/a.py",
                        "name": "a.py",
                        "qualified_name": None,
                        "symbol_id": None,
                        "title": "src/a.py",
                        "preview": "needle second",
                        "content": "needle second",
                        "start_line": 11,
                        "end_line": 20,
                        "token_estimate": 2,
                        "vector": {str(index): value for index, value in vector.items()},
                    },
                ],
            }
            (repo_root / "embedding_index.json").write_text(json.dumps(payload), encoding="utf-8")

            results = query_embedding_index(search_root, "demo", "needle", limit=10)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["doc_id"], "doc-a")
        self.assertEqual(results[0]["metadata"]["embedding_aggregation"], "maxp")
        self.assertEqual(len(results[0]["metadata"]["embedding_unit_hits"]), 2)

    def test_rerank_document_prefers_embedding_unit_text(self) -> None:
        document = candidate_to_rerank_document(
            {
                "qualified_name": "demo.helper",
                "path": "src/demo.py",
                "title": "demo helper",
                "preview": "outer preview",
                "metadata": {"embedding_unit_text": "inner semantic passage"},
            }
        )

        self.assertIn("inner semantic passage", document)
        self.assertLessEqual(len(document), 2500)


if __name__ == "__main__":
    unittest.main()
