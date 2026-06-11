import json
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agents.toolkit import (
    compare_repos,
    execute_graph_query,
    expand_subgraph,
    find_file,
    find_symbol,
    get_enclosing_context,
    get_summary,
    get_symbol_body,
    get_symbol_signature,
    implements_of,
    path_between,
    plan_query,
    prepare_answer_bundle,
    prepare_context,
    refs_of,
    repo_overview,
    retrieve_iterative,
    search_lexical,
    statement_slice,
    summarize_path,
    trace_calls,
    where_defined,
    rank_symbol_candidates,
)
from backends.metadata_store import get_metadata_store
from backends.tantivy.search import TantivySearchBackend, build_query_variants, rerank_search_results
from common.telemetry import reset_telemetry, snapshot_telemetry
from embeddings.indexer import build_embedding_index, query_embedding_index
from evaluation.harness import export_benchmark_prompts, run_benchmarks, score_answer_bundles, score_external_answers
from evaluation.harness import benchmark_interactive_commands
from graph.builder import build_graph_artifact
from graph.query import inspect_graph_backend_payload_uncached
from graph.store import write_graph_database
from retrieval.engine import retrieve_context
from rerank.fusion import rerank_candidates
from search.indexer import build_search_index, search_documents
from summaries.builder import build_summary_artifacts, sync_summary_state
from symbols.indexer import build_symbol_index
from symbols.persistence import load_summary_bundle_from_metadata, load_symbol_index, write_metadata_bundle


def seed_demo_workspace(root: Path) -> dict[str, Path]:
    repo_root = root / "demo"
    raw_root = root / "raw"
    parsed_root = root / "parsed"
    graph_root = root / "graph"
    search_root = root / "search"
    summary_root = root / "summaries"

    (repo_root / "src").mkdir(parents=True)
    (repo_root / "Cargo.toml").write_text(
        '[package]\nname = "demo-crate"\nversion = "0.1.0"\n',
        encoding="utf-8",
    )
    (repo_root / "src" / "lib.rs").write_text(
        "\n".join(
            [
                "pub trait ProvidesAnswer {",
                "    fn answer(&self) -> u64;",
                "}",
                "",
                "/// Return the canonical demo answer.",
                "pub fn helper() -> u64 {",
                "    7",
                "}",
                "",
                "pub struct Demo;",
                "",
                "impl ProvidesAnswer for Demo {",
                "    fn answer(&self) -> u64 {",
                "        helper()",
                "    }",
                "}",
                "",
                "#[cfg(test)]",
                "mod tests {",
                "    use super::*;",
                "",
                "    #[test]",
                "    fn demo_answer() {",
                "        let demo = Demo;",
                "        assert_eq!(ProvidesAnswer::answer(&demo), 7);",
                "    }",
                "}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (raw_root / "demo").mkdir(parents=True)
    (raw_root / "demo" / "manifest.json").write_text(
        json.dumps(
            {
                "repo": "demo",
                "notes": ["demo repo"],
                "language_mix": [{"language": "Rust", "files": 1, "bytes": 64}],
                "build_commands": ["cargo build"],
                "test_commands": ["cargo test"],
                "module_graph_seeds": {
                    "analysis_surfaces": ["src"],
                },
                "parser_relevant_source_roots": ["src"],
            }
        ),
        encoding="utf-8",
    )
    (raw_root / "demo" / "repo_map.json").write_text(
        json.dumps(
            {
                "repo": "demo",
                "directories": [
                    {"path": ".", "depth": 0},
                    {"path": "src", "depth": 1},
                ],
                "files": [
                    {
                        "path": "src/lib.rs",
                        "size": 120,
                        "extension": ".rs",
                        "language": "Rust",
                        "generated": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    symbol_index = build_symbol_index("demo", repo_root, raw_root, path_prefixes=("src/lib.rs",))
    write_metadata_bundle(
        parsed_root,
        "demo",
        symbol_index,
        artifact_metadata={
            "parsed_build": {
                "schema_version": symbol_index.get("schema_version"),
                "repo": "demo",
                "generated_at": symbol_index.get("generated_at"),
                "parser": symbol_index.get("parser"),
                "summary": symbol_index.get("summary", {}),
            }
        },
    )
    graph = build_graph_artifact(symbol_index)
    write_graph_database(graph_root, "demo", graph)
    build_search_index("demo", repo_root, raw_root, parsed_root, search_root)
    build_embedding_index(search_root, "demo")
    summary_artifacts = build_summary_artifacts("demo", raw_root, parsed_root, graph_root)
    sync_summary_state(
        parsed_root,
        graph_root,
        "demo",
        summary_artifacts,
        artifact_metadata={
            "summary_build": {
                "schema_version": summary_artifacts.get("schema_version"),
                "repo": "demo",
                "generated_at": summary_artifacts.get("manifest", {}).get("generated_at"),
                "summary": summary_artifacts.get("summary", {}),
                "summary_graph_nodes": True,
            }
        },
    )

    return {
        "repo_root": repo_root,
        "raw_root": raw_root,
        "parsed_root": parsed_root,
        "graph_root": graph_root,
        "search_root": search_root,
        "summary_root": summary_root,
    }


class SearchAndSummaryTest(unittest.TestCase):
    def test_reranker_prefers_trait_symbol_over_field_for_trait_queries(self) -> None:
        candidates = [
            {
                "kind": "symbol",
                "name": "provider",
                "qualified_name": "acme_system::Coordinator::provider",
                "path": "crates/system/src/coordinator.rs",
                "score": 4.2,
                "metadata": {"kind": "field"},
                "reasons": ["lexical"],
            },
            {
                "kind": "symbol",
                "name": "ProviderTrait",
                "qualified_name": "acme_system::provider::ProviderTrait",
                "path": "crates/system/src/providers.rs",
                "score": 3.9,
                "metadata": {"kind": "trait"},
                "reasons": ["lexical", "symbol-localization"],
            },
        ]

        ranked = rerank_candidates(
            candidates,
            ["system", "provider", "trait"],
            query_profile={
                "intent": "architecture",
                "prefer_tags": ["system", "provider", "trait"],
                "prefer_symbol_kinds": ["provider", "trait"],
                "requested_symbol_kinds": ["trait"],
                "prefer_symbols": False,
                "prefer_docs": False,
                "type_intent": True,
                "member_intent": False,
            },
        )

        self.assertEqual(ranked[0]["name"], "ProviderTrait")

    def test_search_index_returns_symbol_hits(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = seed_demo_workspace(Path(tmpdir))
            results = search_documents(paths["search_root"], "demo", "helper", limit=5)
            symbol_hits = [item for item in results if item["kind"] == "symbol"]

            self.assertGreater(len(results), 0)
            self.assertTrue(any(item["name"] == "helper" for item in symbol_hits))
            self.assertTrue((paths["search_root"] / "demo" / "tantivy").exists())

    def test_tantivy_hot_path_works_without_search_sqlite(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = seed_demo_workspace(Path(tmpdir))
            tantivy_dir = paths["search_root"] / "demo" / "tantivy"
            if not tantivy_dir.exists():
                self.skipTest("native Tantivy index unavailable in this environment")

            symbol_lookup = find_symbol(paths["search_root"], "demo", "helper", limit=5)
            self.assertTrue(any(item["name"] == "helper" for item in symbol_lookup["results"]))

            file_lookup = find_file(paths["search_root"], "demo", "src/lib.rs", limit=5)
            self.assertTrue(any(item["path"] == "src/lib.rs" for item in file_lookup["results"]))

            lexical = search_lexical(paths["search_root"], "demo", "answer helper", limit=5, kinds=("symbol",))
            self.assertGreater(len(lexical["results"]), 0)

            body = get_symbol_body(paths["search_root"], paths["parsed_root"], "demo", "answer")
            self.assertEqual(body["body"]["kind"], "function_body")

    def test_default_tantivy_search_excludes_statement_docs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            search_root = Path(tmpdir) / "search"
            tantivy_dir = search_root / "demo" / "tantivy"
            tantivy_dir.mkdir(parents=True)

            backend = TantivySearchBackend(search_root, "demo")
            with mock.patch("backends.tantivy.search.query_bm25_index", return_value=[] ) as query_mock:
                backend.search("uniqueness guard", limit=5)

            self.assertTrue(query_mock.called)
            kinds = query_mock.call_args.kwargs["kinds"]
            self.assertNotIn("statement", kinds)
            self.assertIn("symbol", kinds)
            self.assertIn("type_body", kinds)

    def test_tantivy_search_uses_broad_fanout_for_small_limits(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            search_root = Path(tmpdir) / "search"
            tantivy_dir = search_root / "demo" / "tantivy"
            tantivy_dir.mkdir(parents=True)

            backend = TantivySearchBackend(search_root, "demo")
            with mock.patch("backends.tantivy.search.query_bm25_index", return_value=[] ) as query_mock:
                backend.search("slot replay capabilities", limit=5)

            self.assertTrue(query_mock.called)
            self.assertGreaterEqual(query_mock.call_args.kwargs["limit"], 100)

    def test_build_query_variants_adds_adjacent_phrase_windows(self) -> None:
        variants = build_query_variants("slot replay capabilities")
        self.assertIn("slot replay", variants)
        self.assertIn("replay capabilities", variants)
        self.assertIn("slot replay capabilities", variants)

    def test_search_index_indexes_markdown_file_contents(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = seed_demo_workspace(Path(tmpdir))
            readme_path = paths["repo_root"] / "README.md"
            readme_path.write_text(
                "The demo service provides never miss data guarantees via slot replay capabilities.\n",
                encoding="utf-8",
            )

            repo_map_path = paths["raw_root"] / "demo" / "repo_map.json"
            repo_map = json.loads(repo_map_path.read_text(encoding="utf-8"))
            repo_map["files"].append(
                {
                    "path": "README.md",
                    "size": readme_path.stat().st_size,
                    "extension": ".md",
                    "language": "Markdown",
                    "generated": False,
                }
            )
            repo_map_path.write_text(json.dumps(repo_map), encoding="utf-8")

            build_search_index(
                "demo",
                paths["repo_root"],
                paths["raw_root"],
                paths["parsed_root"],
                paths["search_root"],
            )

            lexical = search_lexical(
                paths["search_root"],
                "demo",
                "never miss data guarantees",
                limit=5,
            )
            top_paths = [item["path"] for item in lexical["results"]]
            self.assertIn("README.md", top_paths)

    def test_rerank_search_results_prefers_canonical_symbols_over_noise(self) -> None:
        results = [
            {
                "doc_id": "guard-struct",
                "kind": "symbol",
                "name": "UniquenessGuard",
                "qualified_name": "acme_system::filter::UniquenessGuard",
                "title": "acme_system::filter::UniquenessGuard",
                "preview": "pub struct UniquenessGuard {",
                "path": "crates/system/src/filter.rs",
                "searchable": "UniquenessGuard uniqueness guard filter",
                "score": 50.0,
                "metadata": {"kind": "struct", "tags": []},
            },
            {
                "doc_id": "guard-impl",
                "kind": "symbol",
                "name": "impl Filter for UniquenessGuard",
                "qualified_name": "acme_system::filter::impl Filter for UniquenessGuard",
                "title": "acme_system::filter::impl Filter for UniquenessGuard",
                "preview": "impl Filter for UniquenessGuard {",
                "path": "crates/system/src/filter.rs",
                "searchable": "impl Filter for UniquenessGuard uniqueness guard",
                "score": 52.0,
                "metadata": {"kind": "impl", "tags": []},
            },
            {
                "doc_id": "guard-method",
                "kind": "symbol",
                "name": "allow_event",
                "qualified_name": "acme_system::filter::UniquenessGuard::allow_event",
                "title": "acme_system::filter::UniquenessGuard::allow_event",
                "preview": "fn allow_event(&self, ...) -> FilterResult {",
                "path": "crates/system/src/filter.rs",
                "searchable": "UniquenessGuard allow event uniqueness",
                "score": 49.0,
                "metadata": {"kind": "method", "tags": []},
            },
            {
                "doc_id": "guard-body",
                "kind": "type_body",
                "name": "impl Filter for UniquenessGuard",
                "qualified_name": "acme_system::filter::impl Filter for UniquenessGuard",
                "title": "acme_system::filter::impl Filter for UniquenessGuard body",
                "preview": "impl Filter for UniquenessGuard { fn allow_event(...) }",
                "path": "crates/system/src/filter.rs",
                "searchable": "impl Filter for UniquenessGuard body uniqueness guard",
                "score": 53.0,
                "metadata": {"kind": "impl", "tags": []},
            },
            {
                "doc_id": "guard-field",
                "kind": "symbol",
                "name": "seen_events",
                "qualified_name": "acme_system::filter::UniquenessGuard::seen_events",
                "title": "acme_system::filter::UniquenessGuard::seen_events",
                "preview": "seen_events: Arc<RwLock<...>>",
                "path": "crates/system/src/filter.rs",
                "searchable": "seen_events uniqueness guard events",
                "score": 54.0,
                "metadata": {"kind": "field", "tags": []},
            },
            {
                "doc_id": "guard-statement",
                "kind": "statement",
                "name": "let@L126",
                "qualified_name": "acme_system::filter::UniquenessGuard::allow_event",
                "title": "crates/system/src/filter.rs:126",
                "preview": "let key = (sig, path)",
                "path": "crates/system/src/filter.rs",
                "searchable": "statement let key uniqueness guard",
                "score": 60.0,
                "metadata": {"kind": "let", "tags": []},
            },
        ]

        ranked = rerank_search_results("uniqueness guard", results, limit=6)
        ranked_ids = [item["doc_id"] for item in ranked]

        self.assertEqual(ranked_ids[0], "guard-struct")
        self.assertIn("guard-impl", ranked_ids[:3])
        self.assertIn("guard-method", ranked_ids[:5])
        self.assertGreater(ranked_ids.index("guard-field"), ranked_ids.index("guard-method"))
        self.assertEqual(ranked_ids[-1], "guard-statement")

    def test_rank_symbol_candidates_prefers_central_concrete_method_for_short_query(self) -> None:
        candidates = [
            {
                "symbol_id": "trait-run",
                "name": "run",
                "qualified_name": "acme_system::traits::Runnable::run",
                "kind": "method",
                "path": "crates/system/src/traits.rs",
                "visibility": "private",
                "_container_kind": "trait",
                "semantic_summary": {},
                "_search_score": 280.0,
            },
            {
                "symbol_id": "pipe-run",
                "name": "run",
                "qualified_name": "acme_system::workers::TaskWorker::run",
                "kind": "method",
                "path": "crates/system/src/worker.rs",
                "visibility": "private",
                "_container_kind": "impl",
                "semantic_summary": {
                    "direct_calls": [{}] * 6,
                    "reads": [{}] * 16,
                    "writes": [],
                    "interprocedural_reads": [{}] * 7,
                    "interprocedural_writes": [],
                    "interprocedural_references": [{}] * 7,
                    "transitive_calls": [],
                },
                "_search_score": 290.0,
            },
            {
                "symbol_id": "engine-run",
                "name": "run",
                "qualified_name": "acme_system::engine::ExecutionEngine::run",
                "kind": "method",
                "path": "crates/system/src/engine.rs",
                "visibility": "pub",
                "_container_kind": "impl",
                "semantic_summary": {
                    "direct_calls": [{}] * 38,
                    "reads": [{}] * 59,
                    "writes": [{}] * 3,
                    "interprocedural_reads": [{}] * 249,
                    "interprocedural_writes": [{}] * 4,
                    "interprocedural_references": [{}] * 396,
                    "transitive_calls": [{}] * 124,
                },
                "_search_score": 260.0,
            },
        ]

        ranked = rank_symbol_candidates("run", candidates, limit=3)

        self.assertEqual(ranked[0]["qualified_name"], "acme_system::engine::ExecutionEngine::run")

    def test_rank_symbol_candidates_prefers_nominal_symbol_for_pascal_case_query(self) -> None:
        candidates = [
            {
                "symbol_id": "provider-method",
                "name": "provider",
                "qualified_name": "acme_system::builder::ExecutionBuilder::provider",
                "kind": "method",
                "path": "crates/system/src/builder.rs",
                "visibility": "pub",
                "_container_kind": "impl",
                "semantic_summary": {
                    "direct_calls": [{}] * 12,
                    "reads": [{}] * 8,
                    "writes": [{}] * 2,
                    "interprocedural_reads": [{}] * 12,
                    "interprocedural_writes": [],
                    "interprocedural_references": [{}] * 16,
                    "transitive_calls": [{}] * 8,
                },
                "_search_score": 320.0,
            },
            {
                "symbol_id": "provider-trait",
                "name": "Provider",
                "qualified_name": "acme_system::provider::Provider",
                "kind": "trait",
                "path": "crates/system/src/provider.rs",
                "visibility": "pub",
                "_container_kind": "",
                "semantic_summary": {},
                "_search_score": 290.0,
            },
        ]

        ranked = rank_symbol_candidates("Provider", candidates, limit=2)

        self.assertEqual(ranked[0]["qualified_name"], "acme_system::provider::Provider")

    def test_parser_probe_and_embedding_sidecar_are_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = seed_demo_workspace(Path(tmpdir))
            symbols = load_symbol_index(paths["parsed_root"], "demo")
            metadata_store = get_metadata_store(str(paths["parsed_root"].resolve()), "demo")
            search_build = metadata_store.get_artifact_metadata("search_build") or {}

            self.assertIn("parser_backends", symbols)
            self.assertTrue(symbols["parser_backends"]["rustc_ast_probe"]["available"])
            self.assertGreater(symbols["summary"]["statements"], 0)
            self.assertEqual(search_build.get("search_backend"), "tantivy")
            self.assertIsNotNone(search_build.get("search_root"))
            self.assertTrue(all(item.get("summary_id") for item in symbols["symbols"]))
            self.assertTrue(all(item.get("normalized_body_hash") for item in symbols["symbols"]))
            self.assertTrue((paths["parsed_root"] / "demo" / "metadata.lmdb").exists())

    def test_lmdb_metadata_store_serves_symbol_body_and_summary_without_sqlite(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = seed_demo_workspace(Path(tmpdir))
            metadata_store = get_metadata_store(
                str(paths["parsed_root"].resolve()),
                "demo",
                eval_root=str((Path(tmpdir) / "eval").resolve()),
            )

            symbol_ids = metadata_store.resolve_qname("demo_crate::helper")
            self.assertEqual(len(symbol_ids), 1)
            helper = metadata_store.get_symbol(symbol_ids[0])
            self.assertEqual(helper["qualified_name"], "demo_crate::helper")

            body = metadata_store.get_symbol_body(symbol_ids[0])
            self.assertEqual(body["qualified_name"], "demo_crate::helper")
            self.assertTrue(any("7" in statement["text"] for statement in body["statements"]))

            summary_id = str(helper["summary_id"])
            symbols_sqlite = paths["parsed_root"] / "demo" / "symbols.sqlite3"
            if symbols_sqlite.exists():
                symbols_sqlite.unlink()
            summary_sqlite = paths["summary_root"] / "demo" / "summary.sqlite3"
            if summary_sqlite.exists():
                summary_sqlite.unlink()

            summary = metadata_store.get_summary_by_id(summary_id)
            self.assertIsNotNone(summary)
            self.assertEqual(summary["qualified_name"], "demo_crate::helper")
            summaries_by_symbol = metadata_store.get_summary_by_symbol(symbol_ids[0])
            self.assertEqual(len(summaries_by_symbol), 1)

    def test_exact_lookup_commands_use_lmdb_without_sqlite_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = seed_demo_workspace(Path(tmpdir))
            symbols_sqlite = paths["parsed_root"] / "demo" / "symbols.sqlite3"
            if symbols_sqlite.exists():
                symbols_sqlite.unlink()
            summary_sqlite = paths["summary_root"] / "demo" / "summary.sqlite3"
            if summary_sqlite.exists():
                summary_sqlite.unlink()

            where = where_defined(paths["search_root"], paths["parsed_root"], "demo", "demo_crate::helper", limit=5)
            self.assertEqual(where["matches"][0]["qualified_name"], "demo_crate::helper")

            signature = get_symbol_signature(paths["search_root"], paths["parsed_root"], "demo", "demo_crate::helper")
            self.assertEqual(signature["signature"], "pub fn helper() -> u64 {")

            context = get_enclosing_context(
                paths["search_root"],
                paths["graph_root"],
                paths["parsed_root"],
                "demo",
                "demo_crate::helper",
            )
            self.assertIsNotNone(context["context"]["path_summary"])
            self.assertEqual(context["context"]["path_summary"]["path"], "src/lib.rs")

    def test_telemetry_tracks_full_payload_hydration_and_hot_path_timing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = seed_demo_workspace(Path(tmpdir))
            reset_telemetry()

            retrieve_context(
                paths["search_root"],
                paths["graph_root"],
                paths["parsed_root"],
                "demo",
                "helper callers",
                limit=4,
            )

            telemetry = snapshot_telemetry()
            counters = telemetry.get("counters", {})
            timings = telemetry.get("timings", {})

            self.assertEqual(int(counters.get("full_symbol_payload_loads", 0)), 0)
            self.assertEqual(int(counters.get("full_graph_payload_loads", 0)), 0)
            self.assertIn("retrieve_context", timings)
            self.assertIn("retrieve_context.summary_search", timings)
            self.assertIn("retrieve_context.symbol_search", timings)

    def test_lexical_toolkit_commands_do_not_trigger_full_payload_hydration(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = seed_demo_workspace(Path(tmpdir))
            reset_telemetry()

            find_symbol(paths["search_root"], "demo", "helper", limit=5)
            find_file(paths["search_root"], "demo", "src/lib.rs", limit=5)
            search_lexical(paths["search_root"], "demo", "helper", limit=5, kinds=("symbol",))

            telemetry = snapshot_telemetry()
            counters = telemetry.get("counters", {})

            self.assertEqual(int(counters.get("full_symbol_payload_loads", 0)), 0)
            self.assertEqual(int(counters.get("full_graph_payload_loads", 0)), 0)

            metadata_store = get_metadata_store(str(paths["parsed_root"].resolve()), "demo")
            parsed_build = metadata_store.get_artifact_metadata("parsed_build") or {}
            summary_build = metadata_store.get_artifact_metadata("summary_build") or {}
            self.assertEqual(parsed_build.get("repo"), "demo")
            self.assertEqual(summary_build.get("repo"), "demo")
            self.assertTrue(summary_build.get("summary_graph_nodes"))

            embedding_results = query_embedding_index(paths["search_root"], "demo", "helper answer", limit=5)
            self.assertGreater(len(embedding_results), 0)
            self.assertTrue(any(item.get("name") in {"helper", "answer"} for item in embedding_results))

    def test_retrieval_and_toolkit_use_search_and_graph_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = seed_demo_workspace(Path(tmpdir))

            context = retrieve_context(
                paths["search_root"],
                paths["graph_root"],
                paths["parsed_root"],
                "demo",
                "answer helper",
                limit=5,
            )
            self.assertGreater(len(context["selected_context"]), 0)
            self.assertTrue(any(item.get("name") == "answer" for item in context["selected_context"]))

            symbol_lookup = find_symbol(paths["search_root"], "demo", "answer", limit=5)
            self.assertTrue(any(item["name"] == "answer" for item in symbol_lookup["results"]))

            call_trace = trace_calls(
                paths["search_root"],
                paths["graph_root"],
                paths["parsed_root"],
                "demo",
                "helper",
            )
            self.assertEqual(call_trace["resolved_symbol"]["name"], "helper")
            self.assertTrue(any(item["name"] == "answer" for item in call_trace["callers"]))

            prepared = prepare_context(
                paths["search_root"],
                paths["graph_root"],
                paths["parsed_root"],
                "find the helper call path",
                repo_name="demo",
                limit=5,
            )
            self.assertEqual(prepared["contexts"][0]["repo"], "demo")
            self.assertGreater(len(prepared["contexts"][0]["selected_context"]), 0)

    def test_graph_heavy_interactive_commands_do_not_trigger_full_payload_hydration(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = seed_demo_workspace(Path(tmpdir))
            reset_telemetry()

            statement_slice(
                paths["search_root"],
                paths["parsed_root"],
                paths["graph_root"],
                "demo",
                "answer",
                limit=5,
            )
            path_between(
                paths["search_root"],
                paths["parsed_root"],
                paths["graph_root"],
                "demo",
                "answer",
                "helper",
                limit=3,
            )
            prepare_answer_bundle(
                paths["search_root"],
                paths["graph_root"],
                paths["parsed_root"],
                "find the helper call path",
                repo_name="demo",
                limit=5,
            )
            retrieve_iterative(
                paths["search_root"],
                paths["graph_root"],
                paths["parsed_root"],
                "find the helper call path",
                repo_name="demo",
                limit=5,
                refinement_hints=("answer method",),
            )
            compare_repos(
                paths["search_root"],
                paths["graph_root"],
                paths["parsed_root"],
                "find the helper implementation",
                repos=("demo",),
                limit=5,
            )

            telemetry = snapshot_telemetry()
            counters = telemetry.get("counters", {})
            timings = telemetry.get("timings", {})

            self.assertEqual(int(counters.get("full_symbol_payload_loads", 0)), 0)
            self.assertEqual(int(counters.get("full_graph_payload_loads", 0)), 0)
            self.assertIn("statement_slice", timings)
            self.assertIn("path_between", timings)
            self.assertIn("prepare_answer_bundle", timings)
            self.assertIn("retrieve_context.graph_expansion", timings)
            self.assertIn("retrieve_context.metadata_hydration", timings)

    def test_summary_outputs_cover_repo_and_path_rollups(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = seed_demo_workspace(Path(tmpdir))

            overview = repo_overview(paths["parsed_root"], "demo")
            self.assertEqual(overview["repo"], "demo")
            self.assertIn("Rust files", overview["project"]["summary"])

            file_summary = summarize_path(paths["parsed_root"], "demo", "src/lib.rs")
            self.assertEqual(file_summary["kind"], "file")
            self.assertEqual(file_summary["summary"]["path"], "src/lib.rs")

            directory_summary = summarize_path(paths["parsed_root"], "demo", "src")
            self.assertEqual(directory_summary["kind"], "directory")
            self.assertEqual(directory_summary["summary"]["path"], "src")
            packages = load_summary_bundle_from_metadata(paths["parsed_root"], "demo")["packages"]
            self.assertTrue(any(item["package_name"] == "demo-crate" for item in packages))

    def test_benchmark_harness_reports_answer_quality(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = seed_demo_workspace(root)
            eval_root = root / "eval"

            payload = run_benchmarks(
                paths["search_root"],
                paths["graph_root"],
                paths["parsed_root"],
                eval_root,
                repos=("demo",),
                modes=("lexical_graph_rerank_summaries",),
                benchmarks=[
                    {
                        "name": "demo_answer",
                        "repo": "demo",
                        "task_type": "symbol_lookup",
                        "query": "answer helper",
                        "expected_path": "src/lib.rs",
                        "expected_name": "answer",
                        "expected_terms": ["answer", "helper"],
                    }
                ],
            )

            run = payload["runs"][0]
            self.assertIn("answer_quality", run)
            self.assertGreater(run["answer_quality"]["score"], 0.5)
            self.assertIn("avg_answer_score", payload["summary"]["modes"][0])
            self.assertTrue((eval_root / "benchmarks.json").exists())

    def test_interactive_benchmark_report_captures_stage1_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = seed_demo_workspace(root)
            eval_root = root / "eval"

            report = benchmark_interactive_commands(
                paths["search_root"],
                paths["graph_root"],
                paths["parsed_root"],
                eval_root,
                scenarios=[
                    {
                        "repo": "demo",
                        "symbol_query": "answer",
                        "file_query": "src/lib.rs",
                        "call_symbol": "answer",
                        "path_source": "answer",
                        "path_target": "helper",
                        "statement_symbol": "answer",
                        "bundle_query": "find the helper call path",
                    }
                ],
                limit=5,
            )

            self.assertEqual(report["summary"]["runs"], 10)
            commands = {item["command"] for item in report["summary"]["commands"]}
            self.assertIn("prepare-answer-bundle", commands)
            self.assertIn("statement-slice", commands)
            self.assertTrue((eval_root / "interactive_benchmark_report.json").exists())

    def test_graph_query_and_bundle_planner_return_deterministic_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = seed_demo_workspace(Path(tmpdir))

            neighbors = execute_graph_query(
                paths["search_root"],
                paths["parsed_root"],
                paths["graph_root"],
                "demo",
                {
                    "operation": "callers_of",
                    "seed": "helper",
                    "limit": 5,
                },
            )
            self.assertEqual(neighbors["operation"], "callers_of")
            self.assertTrue(any(item["name"] == "answer" for item in neighbors["results"]))

            package_search = search_lexical(
                paths["search_root"],
                "demo",
                "demo-crate",
                limit=5,
                kinds=("package",),
            )
            self.assertTrue(any(item["kind"] == "package" for item in package_search["results"]))

            file_lookup = find_file(paths["search_root"], "demo", "src/lib.rs", limit=5)
            self.assertTrue(any(item["path"] == "src/lib.rs" for item in file_lookup["results"]))

            signature = get_symbol_signature(paths["search_root"], paths["parsed_root"], "demo", "helper")
            self.assertIn("helper()", signature["signature"])

            body = get_symbol_body(paths["search_root"], paths["parsed_root"], "demo", "answer")
            self.assertEqual(body["body"]["kind"], "function_body")

            context = get_enclosing_context(
                paths["search_root"],
                paths["graph_root"],
                paths["parsed_root"],
                "demo",
                "answer",
            )
            self.assertEqual(context["context"]["symbol"]["name"], "answer")

            refs = refs_of(
                paths["search_root"],
                paths["parsed_root"],
                paths["graph_root"],
                "demo",
                "answer",
                limit=10,
            )
            self.assertGreaterEqual(len(refs["neighbors"]), 1)

            impls = implements_of(
                paths["search_root"],
                paths["parsed_root"],
                paths["graph_root"],
                "demo",
                "ProvidesAnswer",
                limit=10,
            )
            self.assertTrue(any(item["name"] == "Demo" or item["kind"] == "impl" for item in impls["neighbors"]))

            subgraph = expand_subgraph(
                paths["search_root"],
                paths["parsed_root"],
                paths["graph_root"],
                "demo",
                "helper",
                edge_types=("CALLS", "USES_TYPE", "NEIGHBOR"),
                depth=2,
                budget=10,
            )
            self.assertEqual(subgraph["operation"], "neighbors")

            graph_payload = inspect_graph_backend_payload_uncached(paths["graph_root"], "demo")["payload"]
            node_kinds = {item["kind"] for item in graph_payload["nodes"]}
            self.assertIn("directory", node_kinds)
            self.assertIn("package", node_kinds)
            self.assertIn("test", node_kinds)
            self.assertIn("symbol_summary", node_kinds)
            edge_types = {item["type"] for item in graph_payload["edges"]}
            self.assertIn("SUMMARIZED_BY", edge_types)
            self.assertIn("OVERRIDES", edge_types)
            self.assertIn("NEIGHBOR", edge_types)

            summary_payload = get_summary(
                paths["search_root"],
                paths["graph_root"],
                paths["parsed_root"],
                "demo",
                graph_payload["nodes"][0]["node_id"],
            )
            self.assertEqual(summary_payload["repo"], "demo")

            slice_payload = statement_slice(
                paths["search_root"],
                paths["parsed_root"],
                paths["graph_root"],
                "demo",
                "answer",
                limit=5,
            )
            self.assertGreater(len(slice_payload["statements"]), 0)
            self.assertTrue(any(item["calls"] for item in slice_payload["statements"]))

            path_payload = path_between(
                paths["search_root"],
                paths["parsed_root"],
                paths["graph_root"],
                "demo",
                "answer",
                "helper",
                limit=3,
            )
            self.assertGreater(len(path_payload["paths"]), 0)
            self.assertEqual(path_payload["paths"][0]["target"]["name"], "helper")

            plan_payload = plan_query(
                paths["search_root"],
                paths["graph_root"],
                paths["parsed_root"],
                "find the helper implementation",
                repo_name="demo",
                limit=5,
            )
            self.assertEqual(plan_payload["plans"][0]["repo"], "demo")

            bundle = prepare_answer_bundle(
                paths["search_root"],
                paths["graph_root"],
                paths["parsed_root"],
                "find the helper call path",
                repo_name="demo",
                limit=5,
            )
            repo_bundle = bundle["bundles"][0]
            self.assertGreater(len(repo_bundle["evidence"]), 0)
            self.assertTrue(all(item["provenance"]["path"] for item in repo_bundle["evidence"]))

            refined = retrieve_iterative(
                paths["search_root"],
                paths["graph_root"],
                paths["parsed_root"],
                "find the helper call path",
                repo_name="demo",
                limit=5,
                prior_bundle=bundle,
                refinement_hints=("answer method",),
            )
            self.assertEqual(refined["iteration"]["iteration_count"], 1)
            self.assertGreater(len(refined["bundles"][0]["selected_context"]), 0)

    def test_prompt_export_and_bundle_scoring_are_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = seed_demo_workspace(root)
            eval_root = root / "eval"
            benchmark = [
                {
                    "name": "demo_answer_bundle",
                    "repo": "demo",
                    "task_type": "symbol_lookup",
                    "query": "answer helper",
                    "expected_path": "src/lib.rs",
                    "expected_name": "answer",
                    "expected_terms": ["answer", "helper"],
                }
            ]

            prompts = export_benchmark_prompts(
                paths["search_root"],
                paths["graph_root"],
                paths["parsed_root"],
                eval_root,
                repos=("demo",),
                limit=5,
                benchmarks=benchmark,
            )
            self.assertEqual(prompts["summary"]["exports"], 1)
            self.assertTrue((eval_root / "prompt_exports" / "demo_answer_bundle.json").exists())
            self.assertTrue((eval_root / "eval.lmdb").exists())

            metadata_store = get_metadata_store(
                str(paths["parsed_root"].resolve()),
                "demo",
                eval_root=str(eval_root.resolve()),
            )
            cached_case = metadata_store.get_eval_case("demo_answer_bundle", "")
            self.assertIsNotNone(cached_case)

            bundle_scores = score_answer_bundles(
                paths["search_root"],
                paths["graph_root"],
                paths["parsed_root"],
                eval_root,
                repos=("demo",),
                limit=5,
                benchmarks=benchmark,
            )
            self.assertGreater(bundle_scores["scores"][0]["score"], 0.5)

            with mock.patch(
                "evaluation.harness.prepare_answer_bundle",
                side_effect=AssertionError("cache should satisfy prompt and score requests"),
            ):
                cached_prompts = export_benchmark_prompts(
                    paths["search_root"],
                    paths["graph_root"],
                    paths["parsed_root"],
                    eval_root,
                    repos=("demo",),
                    limit=5,
                    benchmarks=benchmark,
                )
                self.assertEqual(cached_prompts["summary"]["exports"], 1)

                cached_scores = score_answer_bundles(
                    paths["search_root"],
                    paths["graph_root"],
                    paths["parsed_root"],
                    eval_root,
                    repos=("demo",),
                    limit=5,
                    benchmarks=benchmark,
                )
                self.assertGreater(cached_scores["scores"][0]["score"], 0.5)

            answers_path = eval_root / "answers.json"
            answers_path.write_text(
                json.dumps(
                    {
                        "answers": [
                            {
                                "name": "demo_answer_bundle",
                                "answer": "The answer method in src/lib.rs calls helper.",
                                "cited_paths": ["src/lib.rs"],
                                "cited_symbols": ["answer", "helper"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            external_scores = score_external_answers(eval_root, answers_path, benchmarks=benchmark)
            self.assertGreater(external_scores["scores"][0]["score"], 0.7)


if __name__ == "__main__":
    unittest.main()
