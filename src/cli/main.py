from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional


SRC_ROOT = Path(__file__).resolve().parents[1]
REPO_ANALYSIS_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = REPO_ANALYSIS_ROOT / "data"

DEFAULT_RAW_ROOT = DATA_ROOT / "raw"
DEFAULT_PARSED_ROOT = DATA_ROOT / "parsed"
DEFAULT_GRAPH_ROOT = DATA_ROOT / "graph"
DEFAULT_SEARCH_ROOT = DATA_ROOT / "search"
DEFAULT_SUMMARY_ROOT = DATA_ROOT / "summaries"
DEFAULT_EVAL_ROOT = DATA_ROOT / "eval"
DEFAULT_ZOEKT_ROOT = DATA_ROOT / "zoekt"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agents.toolkit import (
    adjacent_symbols,
    callees_of,
    callers_of,
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
    inherits_of,
    path_between,
    plan_query,
    prepare_context,
    prepare_answer_bundle,
    reads_of,
    refs_of,
    repo_overview,
    retrieve_iterative,
    search_lexical,
    score_external_answers,
    summarize_path,
    statement_slice,
    trace_calls,
    where_defined,
    writes_of,
    who_imports,
)
from common.inventory import InventoryProfile, build_inventory, write_inventory
from common.native_tool import probe_native_worker
from embeddings.indexer import build_embedding_index, query_embedding_index
from evaluation.harness import (
    benchmark_interactive_commands,
    export_benchmark_prompts,
    load_benchmark_cases,
    run_benchmarks,
    score_answer_bundles,
)
from graph.builder import build_graph_artifact
from graph.store import write_graph_database
from retrieval.staged import retrieve_paper_pipeline
from search.indexer import build_search_index
from search.zoekt_backend import build_zoekt_index, search_zoekt_index
from summaries.builder import build_summary_artifacts, sync_summary_state
from symbols.indexer import build_symbol_index, current_rss_mb
from symbols.persistence import update_lmdb_artifact_metadata, write_metadata_bundle


DEFAULT_ANALYSIS_SURFACES = ("src", "lib", "crates", "packages", "apps", "tests", "docs")
DEFAULT_PROGRESS_PAIRS = (
    ("index", "total", "items"),
    ("processed_docs", "total_docs", "docs"),
    ("completed_runs", "total_runs", "runs"),
    ("files_processed", "total_files", "files"),
    ("directories_processed", "total_directories", "directories"),
    ("packages_processed", "total_packages", "packages"),
    ("symbols_processed", "total_symbols", "symbols"),
    ("statements_processed", "total_statements", "statements"),
)


def generic_inventory(repo_root: Path, repo_name: str) -> dict:
    profile = InventoryProfile(
        repo=repo_name,
        expected_ref=None,
        analysis_surfaces=tuple(path for path in DEFAULT_ANALYSIS_SURFACES if (repo_root / path).exists()),
        build_commands=(),
        test_commands=(),
        notes=("Generic repository profile generated from the local workspace path.",),
    )
    return build_inventory(repo_root, profile)


def discover_workspace_repos(workspace_root: Path) -> List[str]:
    ignored = {".git", "repo-analysis-workbench", "repo-analysis", "data", "node_modules", ".venv"}
    repos = []
    for path in sorted(workspace_root.iterdir()):
        if not path.is_dir() or path.name in ignored or path.name.startswith("."):
            continue
        if (path / ".git").exists() or any((path / surface).exists() for surface in DEFAULT_ANALYSIS_SURFACES):
            repos.append(path.name)
    return repos


def discover_artifact_repos(root: Path) -> List[str]:
    if not root.exists():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir() and not path.name.startswith("."))


def add_workspace_root_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--workspace-root",
        default=str(WORKSPACE_ROOT),
        help=f"Workspace root. Default: {WORKSPACE_ROOT}",
    )


def add_raw_root_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--raw-root",
        default=str(DEFAULT_RAW_ROOT),
        help=f"Raw inventory root. Default: {DEFAULT_RAW_ROOT}",
    )


def add_parsed_root_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--parsed-root",
        default=str(DEFAULT_PARSED_ROOT),
        help=f"Parsed metadata root. Default: {DEFAULT_PARSED_ROOT}",
    )


def add_graph_root_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--graph-root",
        default=str(DEFAULT_GRAPH_ROOT),
        help=f"Graph root. Default: {DEFAULT_GRAPH_ROOT}",
    )


def add_search_root_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--search-root",
        default=str(DEFAULT_SEARCH_ROOT),
        help=f"Search root. Default: {DEFAULT_SEARCH_ROOT}",
    )


def add_summary_root_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--summary-root",
        default=str(DEFAULT_SUMMARY_ROOT),
        help=f"Summary root. Default: {DEFAULT_SUMMARY_ROOT}",
    )


def add_zoekt_root_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--zoekt-root",
        default=str(DEFAULT_ZOEKT_ROOT),
        help=f"Zoekt sidecar index root. Default: {DEFAULT_ZOEKT_ROOT}",
    )


def add_eval_root_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--eval-root",
        default=str(DEFAULT_EVAL_ROOT),
        help=f"Eval root. Default: {DEFAULT_EVAL_ROOT}",
    )


def add_benchmark_case_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cases-root",
        default=str(DEFAULT_EVAL_ROOT / "cases"),
        help=f"Benchmark case directory with JSON/JSONL files. Default: {DEFAULT_EVAL_ROOT / 'cases'}",
    )
    parser.add_argument(
        "--case",
        dest="case_paths",
        action="append",
        help="Benchmark case JSON/JSONL file or directory. May be passed multiple times.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="repo-analysis operator CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_repos = subparsers.add_parser(
        "parse-repos",
        help="Collect normalized raw repo inventory for repositories in a workspace.",
    )
    add_workspace_root_arg(parse_repos)
    parse_repos.add_argument(
        "--output-root",
        default=str(DEFAULT_RAW_ROOT),
        help=f"Directory where raw inventory JSON files should be written. Default: {DEFAULT_RAW_ROOT}",
    )
    parse_repos.add_argument(
        "--repo",
        action="append",
        help="Restrict inventory generation to one or more named repos.",
    )

    build_index = subparsers.add_parser(
        "build-index",
        help="Parse Rust source roots into symbol and graph artifacts.",
    )
    add_workspace_root_arg(build_index)
    add_raw_root_arg(build_index)
    add_parsed_root_arg(build_index)
    add_graph_root_arg(build_index)
    build_index.add_argument(
        "--repo",
        action="append",
        help="Restrict indexing to one or more named repos.",
    )
    build_index.add_argument(
        "--path-prefix",
        action="append",
        help="Optional repo-relative file or directory prefix to narrow indexing scope.",
    )
    build_index.add_argument(
        "--scip-index",
        action="append",
        help="Optional repo-relative or absolute .scip/.scip.json index to import during build-index.",
    )
    build_index.add_argument(
        "--progress-interval",
        type=int,
        default=100,
        help="Emit progress logs every N parsed files.",
    )

    build_search = subparsers.add_parser(
        "build-search",
        help="Build lexical search artifacts over raw and parsed outputs.",
    )
    add_workspace_root_arg(build_search)
    add_raw_root_arg(build_search)
    add_parsed_root_arg(build_search)
    add_search_root_arg(build_search)
    build_search.add_argument("--repo", action="append")

    build_zoekt = subparsers.add_parser("build-zoekt", help="Build a Zoekt sidecar index for local repos.")
    add_workspace_root_arg(build_zoekt)
    add_zoekt_root_arg(build_zoekt)
    build_zoekt.add_argument("--repo", action="append", help="Restrict Zoekt indexing to one or more repos.")
    build_zoekt.add_argument("--zoekt-index-bin", help="Path to zoekt-index. Defaults to PATH or REPO_ANALYSIS_ZOEKT_INDEX_BIN.")

    build_summaries = subparsers.add_parser(
        "build-summaries",
        help="Build deterministic project/directory/file/symbol summaries.",
    )
    add_raw_root_arg(build_summaries)
    add_parsed_root_arg(build_summaries)
    add_graph_root_arg(build_summaries)
    add_summary_root_arg(build_summaries)
    build_summaries.add_argument("--repo", action="append")

    build_embeddings = subparsers.add_parser(
        "build-embeddings",
        help="Build the mandatory semantic embedding sidecar over search documents.",
    )
    add_search_root_arg(build_embeddings)
    build_embeddings.add_argument("--repo", action="append")
    build_embeddings.add_argument("--provider", choices=("auto", "hashing", "openai", "qwen"), default="qwen")
    build_embeddings.add_argument("--model")

    run_eval = subparsers.add_parser(
        "run-benchmarks",
        help="Run the lightweight retrieval benchmark slice.",
    )
    add_search_root_arg(run_eval)
    add_graph_root_arg(run_eval)
    add_parsed_root_arg(run_eval)
    add_eval_root_arg(run_eval)
    add_benchmark_case_args(run_eval)
    run_eval.add_argument("--repo", action="append")
    run_eval.add_argument("--mode", action="append")
    run_eval.add_argument("--limit", type=int, default=5)

    repo_overview_cmd = subparsers.add_parser("repo-overview", help="Show the repo-level summary.")
    add_parsed_root_arg(repo_overview_cmd)
    repo_overview_cmd.add_argument("--repo", required=True)

    find_symbol_cmd = subparsers.add_parser("find-symbol", help="Search indexed symbols.")
    add_search_root_arg(find_symbol_cmd)
    find_symbol_cmd.add_argument("--repo", required=True)
    find_symbol_cmd.add_argument("query")
    find_symbol_cmd.add_argument("--limit", type=int, default=10)

    find_file_cmd = subparsers.add_parser("find-file", help="Find indexed files and directories by path pattern.")
    add_search_root_arg(find_file_cmd)
    find_file_cmd.add_argument("--repo", required=True)
    find_file_cmd.add_argument("path_pattern")
    find_file_cmd.add_argument("--limit", type=int, default=20)

    search_lexical_cmd = subparsers.add_parser("search-lexical", help="Run scoped lexical search over indexed documents.")
    add_search_root_arg(search_lexical_cmd)
    search_lexical_cmd.add_argument("--repo", required=True)
    search_lexical_cmd.add_argument("query")
    search_lexical_cmd.add_argument("--kind", action="append")
    search_lexical_cmd.add_argument("--path-prefix")
    search_lexical_cmd.add_argument("--limit", type=int, default=10)

    search_zoekt_cmd = subparsers.add_parser("search-zoekt", help="Run a query against a Zoekt sidecar index.")
    add_zoekt_root_arg(search_zoekt_cmd)
    search_zoekt_cmd.add_argument("--repo", required=True)
    search_zoekt_cmd.add_argument("query")
    search_zoekt_cmd.add_argument("--limit", type=int, default=10)
    search_zoekt_cmd.add_argument("--symbol", action="store_true", help="Use Zoekt symbol search mode.")
    search_zoekt_cmd.add_argument("--zoekt-bin", help="Path to zoekt. Defaults to PATH or REPO_ANALYSIS_ZOEKT_BIN.")

    compare_search_cmd = subparsers.add_parser("compare-search-backends", help="Compare Tantivy lexical and Zoekt sidecar results.")
    add_search_root_arg(compare_search_cmd)
    add_zoekt_root_arg(compare_search_cmd)
    compare_search_cmd.add_argument("--repo", required=True)
    compare_search_cmd.add_argument("query")
    compare_search_cmd.add_argument("--limit", type=int, default=10)
    compare_search_cmd.add_argument("--zoekt-bin", help="Path to zoekt. Defaults to PATH or REPO_ANALYSIS_ZOEKT_BIN.")

    embedding_search_cmd = subparsers.add_parser("embedding-search", help="Run semantic search over embedding vectors.")
    add_search_root_arg(embedding_search_cmd)
    embedding_search_cmd.add_argument("--repo", required=True)
    embedding_search_cmd.add_argument("query")
    embedding_search_cmd.add_argument("--limit", type=int, default=10)

    paper_pipeline_cmd = subparsers.add_parser(
        "search-paper-pipeline",
        help="Run BM25 fanout, retrieval-unit rerank, and maxp source aggregation.",
    )
    add_search_root_arg(paper_pipeline_cmd)
    paper_pipeline_cmd.add_argument("--repo", required=True)
    paper_pipeline_cmd.add_argument("query")
    paper_pipeline_cmd.add_argument("--limit", type=int, default=10)
    paper_pipeline_cmd.add_argument("--lexical-pool", type=int)
    paper_pipeline_cmd.add_argument("--pre-rank-pool", type=int)
    paper_pipeline_cmd.add_argument("--rerank-pool", type=int)

    trace_calls_cmd = subparsers.add_parser("trace-calls", help="Trace callers and callees for a symbol.")
    add_search_root_arg(trace_calls_cmd)
    add_graph_root_arg(trace_calls_cmd)
    add_parsed_root_arg(trace_calls_cmd)
    trace_calls_cmd.add_argument("--repo", required=True)
    trace_calls_cmd.add_argument("symbol")
    trace_calls_cmd.add_argument("--limit", type=int, default=10)

    where_defined_cmd = subparsers.add_parser("where-defined", help="Resolve where a symbol is defined.")
    add_search_root_arg(where_defined_cmd)
    add_parsed_root_arg(where_defined_cmd)
    where_defined_cmd.add_argument("--repo", required=True)
    where_defined_cmd.add_argument("symbol")
    where_defined_cmd.add_argument("--limit", type=int, default=10)

    who_imports_cmd = subparsers.add_parser("who-imports", help="Find importers of a symbol or module.")
    add_search_root_arg(who_imports_cmd)
    add_parsed_root_arg(who_imports_cmd)
    add_graph_root_arg(who_imports_cmd)
    who_imports_cmd.add_argument("--repo", required=True)
    who_imports_cmd.add_argument("symbol")
    who_imports_cmd.add_argument("--limit", type=int, default=20)

    adjacent_symbols_cmd = subparsers.add_parser("adjacent-symbols", help="List graph-adjacent symbols for a symbol query.")
    add_search_root_arg(adjacent_symbols_cmd)
    add_parsed_root_arg(adjacent_symbols_cmd)
    add_graph_root_arg(adjacent_symbols_cmd)
    adjacent_symbols_cmd.add_argument("--repo", required=True)
    adjacent_symbols_cmd.add_argument("symbol")
    adjacent_symbols_cmd.add_argument("--edge-type", action="append")
    adjacent_symbols_cmd.add_argument("--direction", choices=("incoming", "outgoing", "both"), default="both")
    adjacent_symbols_cmd.add_argument("--limit", type=int, default=20)

    compare_repos_cmd = subparsers.add_parser("compare-repos", help="Compare retrieval context across repos.")
    add_search_root_arg(compare_repos_cmd)
    add_graph_root_arg(compare_repos_cmd)
    add_parsed_root_arg(compare_repos_cmd)
    compare_repos_cmd.add_argument("--repo", action="append")
    compare_repos_cmd.add_argument("query")
    compare_repos_cmd.add_argument("--limit", type=int, default=5)

    summarize_path_cmd = subparsers.add_parser("summarize-path", help="Summarize a file or directory path.")
    add_parsed_root_arg(summarize_path_cmd)
    summarize_path_cmd.add_argument("--repo", required=True)
    summarize_path_cmd.add_argument("path")

    prepare_context_cmd = subparsers.add_parser("prepare-context", help="Prepare a compact retrieval context for a task.")
    add_search_root_arg(prepare_context_cmd)
    add_graph_root_arg(prepare_context_cmd)
    add_parsed_root_arg(prepare_context_cmd)
    prepare_context_cmd.add_argument("--repo")
    prepare_context_cmd.add_argument("task")
    prepare_context_cmd.add_argument("--limit", type=int, default=8)

    graph_query_cmd = subparsers.add_parser("graph-query", help="Execute a graph query request.")
    add_search_root_arg(graph_query_cmd)
    add_parsed_root_arg(graph_query_cmd)
    add_graph_root_arg(graph_query_cmd)
    graph_query_cmd.add_argument("--repo", required=True)
    graph_query_cmd.add_argument("--request-file")
    graph_query_cmd.add_argument("--request-json")
    graph_query_cmd.add_argument("--operation")
    graph_query_cmd.add_argument("--seed")
    graph_query_cmd.add_argument("--target")
    graph_query_cmd.add_argument("--edge-type", action="append")
    graph_query_cmd.add_argument("--direction", choices=("incoming", "outgoing", "both"))
    graph_query_cmd.add_argument("--depth", type=int, default=1)
    graph_query_cmd.add_argument("--node-kind", action="append")
    graph_query_cmd.add_argument("--limit", type=int, default=20)
    graph_query_cmd.add_argument("--window", type=int, default=8)

    path_between_cmd = subparsers.add_parser("path-between", help="Find graph paths between two symbols or files.")
    add_search_root_arg(path_between_cmd)
    add_parsed_root_arg(path_between_cmd)
    add_graph_root_arg(path_between_cmd)
    path_between_cmd.add_argument("--repo", required=True)
    path_between_cmd.add_argument("source")
    path_between_cmd.add_argument("target")
    path_between_cmd.add_argument("--edge-type", action="append")
    path_between_cmd.add_argument("--direction", choices=("incoming", "outgoing", "both"), default="both")
    path_between_cmd.add_argument("--limit", type=int, default=5)

    statement_slice_cmd = subparsers.add_parser("statement-slice", help="Show a statement-level slice for a symbol.")
    add_search_root_arg(statement_slice_cmd)
    add_parsed_root_arg(statement_slice_cmd)
    add_graph_root_arg(statement_slice_cmd)
    statement_slice_cmd.add_argument("--repo", required=True)
    statement_slice_cmd.add_argument("symbol")
    statement_slice_cmd.add_argument("--limit", type=int, default=20)
    statement_slice_cmd.add_argument("--window", type=int, default=8)

    callers_of_cmd = subparsers.add_parser("callers-of", help="List callers of a symbol.")
    add_search_root_arg(callers_of_cmd)
    add_parsed_root_arg(callers_of_cmd)
    add_graph_root_arg(callers_of_cmd)
    callers_of_cmd.add_argument("--repo", required=True)
    callers_of_cmd.add_argument("symbol")
    callers_of_cmd.add_argument("--limit", type=int, default=20)

    callees_of_cmd = subparsers.add_parser("callees-of", help="List callees of a symbol.")
    add_search_root_arg(callees_of_cmd)
    add_parsed_root_arg(callees_of_cmd)
    add_graph_root_arg(callees_of_cmd)
    callees_of_cmd.add_argument("--repo", required=True)
    callees_of_cmd.add_argument("symbol")
    callees_of_cmd.add_argument("--limit", type=int, default=20)

    reads_of_cmd = subparsers.add_parser("reads-of", help="List read relationships for a symbol.")
    add_search_root_arg(reads_of_cmd)
    add_parsed_root_arg(reads_of_cmd)
    add_graph_root_arg(reads_of_cmd)
    reads_of_cmd.add_argument("--repo", required=True)
    reads_of_cmd.add_argument("symbol")
    reads_of_cmd.add_argument("--limit", type=int, default=20)

    writes_of_cmd = subparsers.add_parser("writes-of", help="List write relationships for a symbol.")
    add_search_root_arg(writes_of_cmd)
    add_parsed_root_arg(writes_of_cmd)
    add_graph_root_arg(writes_of_cmd)
    writes_of_cmd.add_argument("--repo", required=True)
    writes_of_cmd.add_argument("symbol")
    writes_of_cmd.add_argument("--limit", type=int, default=20)

    refs_of_cmd = subparsers.add_parser("refs-of", help="List reference relationships for a symbol.")
    add_search_root_arg(refs_of_cmd)
    add_parsed_root_arg(refs_of_cmd)
    add_graph_root_arg(refs_of_cmd)
    refs_of_cmd.add_argument("--repo", required=True)
    refs_of_cmd.add_argument("symbol")
    refs_of_cmd.add_argument("--limit", type=int, default=20)

    implements_of_cmd = subparsers.add_parser("implements-of", help="List implementations of a trait or interface symbol.")
    add_search_root_arg(implements_of_cmd)
    add_parsed_root_arg(implements_of_cmd)
    add_graph_root_arg(implements_of_cmd)
    implements_of_cmd.add_argument("--repo", required=True)
    implements_of_cmd.add_argument("symbol")
    implements_of_cmd.add_argument("--limit", type=int, default=20)

    inherits_of_cmd = subparsers.add_parser("inherits-of", help="List inherited parents for a trait or type symbol.")
    add_search_root_arg(inherits_of_cmd)
    add_parsed_root_arg(inherits_of_cmd)
    add_graph_root_arg(inherits_of_cmd)
    inherits_of_cmd.add_argument("--repo", required=True)
    inherits_of_cmd.add_argument("symbol")
    inherits_of_cmd.add_argument("--limit", type=int, default=20)

    get_summary_cmd = subparsers.add_parser("get-summary", help="Resolve a summary payload for a graph or summary node.")
    add_search_root_arg(get_summary_cmd)
    add_graph_root_arg(get_summary_cmd)
    add_parsed_root_arg(get_summary_cmd)
    get_summary_cmd.add_argument("--repo", required=True)
    get_summary_cmd.add_argument("node_id")

    get_symbol_signature_cmd = subparsers.add_parser("get-symbol-signature", help="Return a resolved symbol signature.")
    add_search_root_arg(get_symbol_signature_cmd)
    add_parsed_root_arg(get_symbol_signature_cmd)
    get_symbol_signature_cmd.add_argument("--repo", required=True)
    get_symbol_signature_cmd.add_argument("symbol")

    get_symbol_body_cmd = subparsers.add_parser("get-symbol-body", help="Return the indexed body chunk for a symbol.")
    add_search_root_arg(get_symbol_body_cmd)
    add_parsed_root_arg(get_symbol_body_cmd)
    get_symbol_body_cmd.add_argument("--repo", required=True)
    get_symbol_body_cmd.add_argument("symbol")

    get_enclosing_context_cmd = subparsers.add_parser("get-enclosing-context", help="Return enclosing summary and statement context for a symbol.")
    add_search_root_arg(get_enclosing_context_cmd)
    add_graph_root_arg(get_enclosing_context_cmd)
    add_parsed_root_arg(get_enclosing_context_cmd)
    get_enclosing_context_cmd.add_argument("--repo", required=True)
    get_enclosing_context_cmd.add_argument("symbol")

    expand_subgraph_cmd = subparsers.add_parser("expand-subgraph", help="Expand a bounded graph neighborhood from a seed.")
    add_search_root_arg(expand_subgraph_cmd)
    add_parsed_root_arg(expand_subgraph_cmd)
    add_graph_root_arg(expand_subgraph_cmd)
    expand_subgraph_cmd.add_argument("--repo", required=True)
    expand_subgraph_cmd.add_argument("seed")
    expand_subgraph_cmd.add_argument("--edge-type", action="append")
    expand_subgraph_cmd.add_argument("--direction", choices=("incoming", "outgoing", "both"), default="both")
    expand_subgraph_cmd.add_argument("--depth", type=int, default=1)
    expand_subgraph_cmd.add_argument("--node-kind", action="append")
    expand_subgraph_cmd.add_argument("--budget", type=int, default=20)

    plan_query_cmd = subparsers.add_parser("plan-query", help="Plan the retrieval recipe for a task.")
    add_search_root_arg(plan_query_cmd)
    add_graph_root_arg(plan_query_cmd)
    add_parsed_root_arg(plan_query_cmd)
    plan_query_cmd.add_argument("--repo")
    plan_query_cmd.add_argument("task")
    plan_query_cmd.add_argument("--limit", type=int, default=8)

    prepare_bundle_cmd = subparsers.add_parser("prepare-answer-bundle", help="Prepare an answer bundle for an external LLM consumer.")
    add_search_root_arg(prepare_bundle_cmd)
    add_graph_root_arg(prepare_bundle_cmd)
    add_parsed_root_arg(prepare_bundle_cmd)
    prepare_bundle_cmd.add_argument("--repo")
    prepare_bundle_cmd.add_argument("task")
    prepare_bundle_cmd.add_argument("--hint", action="append")
    prepare_bundle_cmd.add_argument("--limit", type=int, default=8)

    retrieve_iterative_cmd = subparsers.add_parser("retrieve-iterative", help="Refine retrieval using a prior answer bundle and new hints.")
    add_search_root_arg(retrieve_iterative_cmd)
    add_graph_root_arg(retrieve_iterative_cmd)
    add_parsed_root_arg(retrieve_iterative_cmd)
    retrieve_iterative_cmd.add_argument("--repo")
    retrieve_iterative_cmd.add_argument("task")
    retrieve_iterative_cmd.add_argument("--prior-bundle")
    retrieve_iterative_cmd.add_argument("--hint", action="append")
    retrieve_iterative_cmd.add_argument("--limit", type=int, default=8)

    export_prompts_cmd = subparsers.add_parser("export-benchmark-prompts", help="Export deterministic benchmark prompt packages.")
    add_search_root_arg(export_prompts_cmd)
    add_graph_root_arg(export_prompts_cmd)
    add_parsed_root_arg(export_prompts_cmd)
    add_eval_root_arg(export_prompts_cmd)
    add_benchmark_case_args(export_prompts_cmd)
    export_prompts_cmd.add_argument("--repo", action="append")
    export_prompts_cmd.add_argument("--limit", type=int, default=8)

    score_bundles_cmd = subparsers.add_parser("score-answer-bundles", help="Score whether prepared answer bundles are sufficient.")
    add_search_root_arg(score_bundles_cmd)
    add_graph_root_arg(score_bundles_cmd)
    add_parsed_root_arg(score_bundles_cmd)
    add_eval_root_arg(score_bundles_cmd)
    add_benchmark_case_args(score_bundles_cmd)
    score_bundles_cmd.add_argument("--repo", action="append")
    score_bundles_cmd.add_argument("--limit", type=int, default=8)

    score_external_cmd = subparsers.add_parser("score-external-answers", help="Score externally produced answers against benchmark expectations.")
    add_eval_root_arg(score_external_cmd)
    add_benchmark_case_args(score_external_cmd)
    score_external_cmd.add_argument("--answers-path", required=True)

    benchmark_interactive_cmd = subparsers.add_parser("benchmark-interactive", help="Emit a baseline latency and telemetry report for interactive commands.")
    add_search_root_arg(benchmark_interactive_cmd)
    add_graph_root_arg(benchmark_interactive_cmd)
    add_parsed_root_arg(benchmark_interactive_cmd)
    add_eval_root_arg(benchmark_interactive_cmd)
    benchmark_interactive_cmd.add_argument("--repo", action="append")
    benchmark_interactive_cmd.add_argument("--limit", type=int, default=5)

    return parser


def handle_parse_repos(args: argparse.Namespace) -> int:
    workspace_root = Path(args.workspace_root).resolve()
    output_root = Path(args.output_root).resolve()
    repo_names = args.repo or discover_workspace_repos(workspace_root)
    if not repo_names:
        raise RuntimeError("No repositories found. Pass --repo or set --workspace-root to a directory containing repos.")

    output_root.mkdir(parents=True, exist_ok=True)

    for repo_name in repo_names:
        repo_root = workspace_root / repo_name
        if not repo_root.exists():
            raise FileNotFoundError(f"Missing repo root: {repo_root}")
        inventory = generic_inventory(repo_root, repo_name)
        write_inventory(output_root, repo_name, inventory)
        print(f"Wrote raw inventory for {repo_name} to {output_root / repo_name}")

    return 0


def handle_build_index(args: argparse.Namespace) -> int:
    workspace_root = Path(args.workspace_root).resolve()
    raw_root = Path(args.raw_root).resolve()
    parsed_root = Path(args.parsed_root).resolve()
    graph_root = Path(args.graph_root).resolve()
    repo_names = args.repo or discover_artifact_repos(raw_root)
    if not repo_names:
        raise RuntimeError("No raw inventories found. Run parse-repos first or pass --repo.")
    path_prefixes = tuple(args.path_prefix or [])
    scip_indexes = tuple(Path(value) for value in (args.scip_index or []))

    parsed_root.mkdir(parents=True, exist_ok=True)
    graph_root.mkdir(parents=True, exist_ok=True)

    for repo_name in repo_names:
        repo_root = workspace_root / repo_name
        if not repo_root.exists():
            raise FileNotFoundError(f"Missing repo root: {repo_root}")

        repo_progress_path = parsed_root / repo_name / "build_progress.json"
        progress_started = time.perf_counter()
        emit_build_progress(
            repo_progress_path,
            {
                "event": "repo_started",
                "repo": repo_name,
                "path_prefixes": list(path_prefixes),
                "started_at": timestamp_now(),
                "elapsed_ms": 0.0,
            },
            log_message=(
                f"[build-index] repo={repo_name} status=started "
                f"path_prefixes={list(path_prefixes) or ['<all>']}"
            ),
        )

        def progress_callback(event: Dict[str, object]) -> None:
            should_log = False
            log_message = None
            event = enrich_progress_estimate(
                event,
                elapsed_ms=float(event.get("elapsed_ms") or 0.0),
            )
            event_name = str(event.get("event") or "")
            if event_name == "repo_scan_started":
                log_message = (
                    f"[build-index] repo={repo_name} status=scanning "
                    f"rust_files={event.get('rust_files_total')} parser_roots={event.get('parser_roots')} "
                    f"rss_mb={event.get('rss_mb')}"
                )
                should_log = True
            elif event_name == "file_parsed":
                index = int(event.get("index") or 0)
                total = int(event.get("total") or 0)
                file_elapsed_ms = float(event.get("file_elapsed_ms") or 0.0)
                if index == 1 or index == total or (args.progress_interval > 0 and index % args.progress_interval == 0):
                    should_log = True
                if file_elapsed_ms >= 10000:
                    should_log = True
                if should_log:
                    log_message = (
                        f"[build-index] repo={repo_name} progress={index}/{total} "
                        f"file={event.get('path')} file_ms={file_elapsed_ms:.1f} "
                        f"elapsed_ms={float(event.get('elapsed_ms') or 0.0):.1f} "
                        f"backend_failures={json.dumps(event.get('backend_failures', {}), sort_keys=True)}"
                    )
                    log_message = append_progress_estimate(log_message, event)
            elif event_name == "stage_progress":
                should_log = True
                log_message = (
                    f"[build-index] repo={repo_name} stage={event.get('stage')} "
                    f"elapsed_ms={float(event.get('elapsed_ms') or 0.0):.1f} "
                    f"rss_mb={float(event.get('rss_mb') or 0.0):.1f} "
                    f"contexts={int(event.get('contexts') or 0)}"
                )
                log_message = append_progress_estimate(log_message, event)
            emit_build_progress(repo_progress_path, event, log_message=log_message if should_log else None)

        native_worker = probe_native_worker()
        try:
            symbol_index = build_symbol_index(
                repo_name,
                repo_root,
                raw_root,
                path_prefixes=path_prefixes,
                progress_callback=progress_callback,
                cache_root=parsed_root / repo_name,
                scip_indexes=scip_indexes,
            )
            emit_build_progress(
                repo_progress_path,
                {
                    "event": "stage_progress",
                    "repo": repo_name,
                    "stage": "building_graph_artifact",
                    "elapsed_ms": round((time.perf_counter() - progress_started) * 1000, 3),
                    "rss_mb": current_rss_mb(),
                    "symbols": symbol_index.get("summary", {}).get("symbols", 0),
                    "references": symbol_index.get("summary", {}).get("references", 0),
                    "statements": symbol_index.get("summary", {}).get("statements", 0),
                },
                log_message=(
                    f"[build-index] repo={repo_name} stage=building_graph_artifact "
                    f"elapsed_ms={round((time.perf_counter() - progress_started) * 1000, 3):.1f} "
                    f"rss_mb={current_rss_mb():.1f}"
                ),
            )
            graph_artifact = build_graph_artifact(symbol_index)
        except Exception as exc:
            emit_build_progress(
                repo_progress_path,
                {
                    "event": "repo_failed",
                    "repo": repo_name,
                    "elapsed_ms": round((time.perf_counter() - progress_started) * 1000, 3),
                    "failed_at": timestamp_now(),
                    "error": {
                        "type": exc.__class__.__name__,
                        "message": str(exc),
                    },
                },
                log_message=f"[build-index] repo={repo_name} status=failed error={exc.__class__.__name__}: {exc}",
            )
            raise

        emit_build_progress(
            repo_progress_path,
            {
                "event": "stage_progress",
                "repo": repo_name,
                "stage": "writing_artifacts",
                "elapsed_ms": round((time.perf_counter() - progress_started) * 1000, 3),
                "rss_mb": current_rss_mb(),
                "graph_nodes": graph_artifact.get("summary", {}).get("nodes", 0),
                "graph_edges": graph_artifact.get("summary", {}).get("edges", 0),
            },
            log_message=(
                f"[build-index] repo={repo_name} stage=writing_artifacts "
                f"elapsed_ms={round((time.perf_counter() - progress_started) * 1000, 3):.1f} "
                f"rss_mb={current_rss_mb():.1f} "
                f"graph_edges={graph_artifact.get('summary', {}).get('edges', 0)}"
            ),
        )
        write_metadata_bundle(
            parsed_root,
            repo_name,
            symbol_index,
            artifact_metadata={
                "parsed_build": {
                    "schema_version": symbol_index.get("schema_version"),
                    "repo": repo_name,
                    "generated_at": symbol_index.get("generated_at"),
                    "parser": symbol_index.get("parser"),
                    "primary_parser_backends": symbol_index.get("primary_parser_backends", []),
                    "parser_backends": symbol_index.get("parser_backends", {}),
                    "source_roots": symbol_index.get("source_roots", []),
                    "path_prefixes": list(path_prefixes),
                    "summary": symbol_index.get("summary", {}),
                }
            },
        )
        remove_file_if_exists(parsed_root / repo_name / "symbols.json")
        remove_file_if_exists(parsed_root / repo_name / "symbols.sqlite3")
        remove_file_if_exists(parsed_root / repo_name / "parquet_status.json")
        remove_file_if_exists(parsed_root / repo_name / "query_manifest.json")
        remove_file_if_exists(graph_root / repo_name / "graph.json")
        graph_db_path = write_graph_database(graph_root, repo_name, graph_artifact)
        update_lmdb_artifact_metadata(
            parsed_root,
            repo_name,
            {
                "graph_build": {
                    "schema_version": graph_artifact.get("schema_version"),
                    "repo": repo_name,
                    "generated_at": graph_artifact.get("generated_at"),
                    "graph_backend": "ryugraph",
                    "graph_database": graph_db_path.name,
                    "summary": graph_artifact.get("summary", {}),
                    "native_worker": native_worker,
                }
            },
        )

        completed_event = {
            "event": "repo_completed",
            "repo": repo_name,
            "elapsed_ms": round((time.perf_counter() - progress_started) * 1000, 3),
            "summary": symbol_index.get("summary", {}),
            "build_metrics": symbol_index.get("build_metrics", {}),
            "graph_summary": graph_artifact.get("summary", {}),
            "completed_at": timestamp_now(),
        }
        emit_build_progress(
            repo_progress_path,
            completed_event,
            log_message=(
                f"[build-index] repo={repo_name} status=completed "
                f"symbols={symbol_index.get('summary', {}).get('symbols')} "
                f"statements={symbol_index.get('summary', {}).get('statements')} "
                f"graph_edges={graph_artifact.get('summary', {}).get('edges')} "
                f"elapsed_ms={completed_event['elapsed_ms']}"
            ),
        )

        print(f"Wrote parsed symbols for {repo_name} to {parsed_root / repo_name}", flush=True)
        print(f"Wrote graph artifact for {repo_name} to {graph_root / repo_name}", flush=True)

    return 0


def enrich_progress_estimate(
    payload: Dict[str, object],
    *,
    elapsed_ms: float,
    progress_pairs: tuple[tuple[str, str, str], ...] = DEFAULT_PROGRESS_PAIRS,
) -> Dict[str, object]:
    enriched = dict(payload)
    elapsed_seconds = max(elapsed_ms / 1000.0, 0.0)
    if elapsed_seconds <= 0:
        return enriched

    for processed_key, total_key, unit in progress_pairs:
        processed = enriched.get(processed_key)
        total = enriched.get(total_key)
        if processed is None or total is None:
            continue
        try:
            processed_count = int(processed)
            total_count = int(total)
        except (TypeError, ValueError):
            continue
        if processed_count <= 0 or total_count <= 0:
            continue

        rate_per_sec = processed_count / elapsed_seconds
        remaining = max(total_count - processed_count, 0)
        enriched.setdefault("progress_unit", unit)
        enriched.setdefault("percent_complete", round(min(processed_count / total_count, 1.0) * 100, 2))
        enriched.setdefault("rate_per_sec", round(rate_per_sec, 3))
        enriched.setdefault("eta_seconds", round(remaining / rate_per_sec, 1) if rate_per_sec > 0 else None)
        break
    return enriched


def append_progress_estimate(message: str, payload: Dict[str, object]) -> str:
    if payload.get("percent_complete") is not None:
        message += f" pct={float(payload.get('percent_complete') or 0.0):.2f}"
    if payload.get("rate_per_sec") is not None:
        unit = str(payload.get("progress_unit") or "items")
        message += f" rate_per_sec={float(payload.get('rate_per_sec') or 0.0):.3f}/{unit}"
    if payload.get("eta_seconds") is not None:
        message += f" eta_seconds={float(payload.get('eta_seconds') or 0.0):.1f}"
    if payload.get("rss_mb") is not None:
        message += f" rss_mb={float(payload.get('rss_mb') or 0.0):.1f}"
    return message


def emit_build_progress(progress_path: Path, payload: Dict[str, object], *, log_message: str | None = None) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"recorded_at": timestamp_now(), **payload}
    progress_path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    with progress_path.with_suffix(".jsonl").open("a", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=False)
        handle.write("\n")
    if log_message:
        print(log_message, flush=True, file=sys.stderr)


def timestamp_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def remove_file_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def handle_build_search(args: argparse.Namespace) -> int:
    workspace_root = Path(args.workspace_root).resolve()
    raw_root = Path(args.raw_root).resolve()
    parsed_root = Path(args.parsed_root).resolve()
    search_root = Path(args.search_root).resolve()
    repo_names = args.repo or discover_artifact_repos(raw_root)

    search_root.mkdir(parents=True, exist_ok=True)
    for repo_name in repo_names:
        repo_root = workspace_root / repo_name
        progress_path = search_root / repo_name / "build_progress.json"
        started = time.perf_counter()
        emit_build_progress(
            progress_path,
            {
                "event": "build_started",
                "repo": repo_name,
                "elapsed_ms": 0.0,
            },
            log_message=f"[build-search] repo={repo_name} status=started",
        )

        def progress_callback(event: Dict[str, object]) -> None:
            stage = str(event.get("event") or "")
            event = enrich_progress_estimate(
                event,
                elapsed_ms=float(event.get("elapsed_ms") or 0.0),
            )
            message = (
                f"[build-search] repo={repo_name} stage={stage} "
                f"elapsed_ms={float(event.get('elapsed_ms') or 0.0):.1f}"
            )
            if event.get("documents") is not None:
                message += f" documents={int(event.get('documents') or 0)}"
            if event.get("files") is not None:
                message += f" files={int(event.get('files') or 0)}"
            if event.get("symbols") is not None:
                message += f" symbols={int(event.get('symbols') or 0)}"
            if event.get("statements") is not None:
                message += f" statements={int(event.get('statements') or 0)}"
            for processed_key, total_key, label in (
                ("files_processed", "total_files", "files"),
                ("directories_processed", "total_directories", "directories"),
                ("packages_processed", "total_packages", "packages"),
                ("symbols_processed", "total_symbols", "symbols"),
                ("statements_processed", "total_statements", "statements"),
            ):
                if event.get(processed_key) is not None:
                    if event.get(total_key) is not None:
                        message += f" {label}={int(event.get(processed_key) or 0)}/{int(event.get(total_key) or 0)}"
                    else:
                        message += f" {label}={int(event.get(processed_key) or 0)}"
            for total_key in ("total_files", "total_directories", "total_packages", "total_symbols", "total_statements"):
                if event.get(total_key) is not None and not any(total_key == pair[1] and event.get(pair[0]) is not None for pair in DEFAULT_PROGRESS_PAIRS):
                    message += f" {total_key}={int(event.get(total_key) or 0)}"
            if event.get("current_path"):
                message += f" current_path={event.get('current_path')}"
            if event.get("built") is not None:
                message += f" built={str(bool(event.get('built'))).lower()}"
            if event.get("reason"):
                message += f" reason={event.get('reason')}"
            message = append_progress_estimate(message, event)
            emit_build_progress(progress_path, event, log_message=message)

        payload = build_search_index(
            repo_name,
            repo_root,
            raw_root,
            parsed_root,
            search_root,
            progress_callback=progress_callback,
        )
        emit_build_progress(
            progress_path,
            {
                "event": "repo_completed",
                "repo": repo_name,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                "documents": payload["summary"]["documents"],
            },
            log_message=(
                f"[build-search] repo={repo_name} status=completed "
                f"elapsed_ms={round((time.perf_counter() - started) * 1000, 3):.1f} "
                f"documents={payload['summary']['documents']}"
            ),
        )
    return 0


def handle_build_zoekt(args: argparse.Namespace) -> int:
    workspace_root = Path(args.workspace_root).resolve()
    zoekt_root = Path(args.zoekt_root).resolve()
    repo_names = args.repo or discover_workspace_repos(workspace_root)
    if not repo_names:
        raise RuntimeError("No repositories found. Pass --repo or set --workspace-root to a directory containing repos.")

    zoekt_root.mkdir(parents=True, exist_ok=True)
    ok = True
    for repo_name in repo_names:
        repo_root = workspace_root / repo_name
        if not repo_root.exists():
            raise FileNotFoundError(f"Missing repo root: {repo_root}")
        payload = build_zoekt_index(
            repo_name,
            repo_root,
            zoekt_root,
            zoekt_index_bin=args.zoekt_index_bin,
        )
        print_json(payload)
        ok = ok and bool(payload.get("built"))
    return 0 if ok else 1


def compare_search_backends(search_root: Path, zoekt_root: Path, repo_name: str, query: str, *, limit: int, zoekt_bin: str | None) -> Dict[str, object]:
    lexical = search_lexical(search_root, repo_name, query, limit=limit)
    zoekt = search_zoekt_index(repo_name, zoekt_root, query, limit=limit, zoekt_bin=zoekt_bin)
    lexical_paths = [str(result.get("path") or "") for result in lexical.get("results", [])]
    zoekt_paths = [str(result.get("path") or "") for result in zoekt.get("results", [])]
    lexical_set = {path for path in lexical_paths if path}
    zoekt_set = {path for path in zoekt_paths if path}
    overlap = sorted(lexical_set & zoekt_set)
    return {
        "repo": repo_name,
        "query": query,
        "limit": limit,
        "lexical": lexical,
        "zoekt": zoekt,
        "comparison": {
            "lexical_paths": lexical_paths,
            "zoekt_paths": zoekt_paths,
            "overlap_paths": overlap,
            "lexical_only_paths": sorted(lexical_set - zoekt_set),
            "zoekt_only_paths": sorted(zoekt_set - lexical_set),
            "overlap_at_k": len(overlap),
        },
    }


def handle_build_summaries(args: argparse.Namespace) -> int:
    raw_root = Path(args.raw_root).resolve()
    parsed_root = Path(args.parsed_root).resolve()
    graph_root = Path(args.graph_root).resolve()
    summary_root = Path(args.summary_root).resolve()
    repo_names = args.repo or discover_artifact_repos(parsed_root)

    summary_root.mkdir(parents=True, exist_ok=True)
    for repo_name in repo_names:
        progress_path = summary_root / repo_name / "build_progress.json"
        started = time.perf_counter()
        emit_build_progress(
            progress_path,
            {
                "event": "build_started",
                "repo": repo_name,
                "elapsed_ms": 0.0,
            },
            log_message=f"[build-summaries] repo={repo_name} status=started",
        )

        def progress_callback(event: Dict[str, object]) -> None:
            stage = str(event.get("event") or "")
            message = (
                f"[build-summaries] repo={repo_name} stage={stage} "
                f"elapsed_ms={float(event.get('elapsed_ms') or 0.0):.1f}"
            )
            for key in ("files", "directories", "nodes", "edges", "symbols", "statements", "packages"):
                if event.get(key) is not None:
                    message += f" {key}={int(event.get(key) or 0)}"
            emit_build_progress(progress_path, event, log_message=message)

        payload = build_summary_artifacts(
            repo_name,
            raw_root,
            parsed_root,
            graph_root,
            progress_callback=progress_callback,
        )
        emit_build_progress(
            progress_path,
            {
                "event": "syncing_summary_state",
                "repo": repo_name,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            },
            log_message=(
                f"[build-summaries] repo={repo_name} stage=syncing_summary_state "
                f"elapsed_ms={round((time.perf_counter() - started) * 1000, 3):.1f}"
            ),
        )
        emit_build_progress(
            progress_path,
            {
                "event": "persisting_summary_metadata",
                "repo": repo_name,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            },
            log_message=(
                f"[build-summaries] repo={repo_name} stage=persisting_summary_metadata "
                f"elapsed_ms={round((time.perf_counter() - started) * 1000, 3):.1f}"
            ),
        )
        sync_summary_state(
            parsed_root,
            graph_root,
            repo_name,
            payload,
            artifact_metadata={
                "summary_build": {
                    "schema_version": payload.get("schema_version"),
                    "repo": repo_name,
                    "generated_at": payload.get("manifest", {}).get("generated_at"),
                    "summary": payload.get("summary", {}),
                    "summary_graph_nodes": True,
                    "summary_json_exports": False,
                }
            },
        )
        remove_file_if_exists(parsed_root / repo_name / "query_manifest.json")
        emit_build_progress(
            progress_path,
            {
                "event": "repo_completed",
                "repo": repo_name,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                **payload["summary"],
            },
            log_message=(
                f"[build-summaries] repo={repo_name} status=completed "
                f"elapsed_ms={round((time.perf_counter() - started) * 1000, 3):.1f} "
                f"packages={payload['summary']['packages']} "
                f"directories={payload['summary']['directories']} "
                f"files={payload['summary']['files']} "
                f"symbols={payload['summary']['symbols']}"
            ),
        )
    return 0


def handle_build_embeddings(args: argparse.Namespace) -> int:
    search_root = Path(args.search_root).resolve()
    repo_names = args.repo or discover_artifact_repos(search_root)

    for repo_name in repo_names:
        progress_path = search_root / repo_name / "embedding_build_progress.json"
        started = time.perf_counter()

        emit_build_progress(
            progress_path,
            {
                "event": "build_started",
                "repo": repo_name,
                "provider": args.provider,
                "model": args.model,
                "elapsed_ms": 0.0,
            },
            log_message=(
                f"[build-embeddings] repo={repo_name} status=started "
                f"provider={args.provider} model={args.model or '<default>'}"
            ),
        )

        def progress_callback(event: Dict[str, object]) -> None:
            event_name = str(event.get("event") or "")
            elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
            payload = {
                **event,
                "elapsed_ms": elapsed_ms,
            }
            payload = enrich_progress_estimate(payload, elapsed_ms=elapsed_ms)

            message = (
                f"[build-embeddings] repo={repo_name} stage={event_name} "
                f"elapsed_ms={elapsed_ms:.1f}"
            )

            if payload.get("provider") is not None:
                message += f" provider={payload.get('provider')}"
            if payload.get("model") is not None:
                message += f" model={payload.get('model')}"

            processed_docs = payload.get("processed_docs")
            total_docs = payload.get("total_docs")
            if processed_docs is not None:
                if total_docs is not None:
                    message += f" processed_docs={processed_docs}/{total_docs}"
                else:
                    message += f" processed_docs={processed_docs}"

            for key in (
                "documents",
                "batch_docs",
                "batch_index",
                "batches",
                "batch_elapsed_ms",
                "batch_input_chars",
                "nonzero_dimensions",
                "vector_format",
            ):
                if payload.get(key) is not None:
                    message += f" {key}={payload.get(key)}"

            message = append_progress_estimate(message, payload)
            emit_build_progress(progress_path, payload, log_message=message)

        payload = build_embedding_index(
            search_root,
            repo_name,
            provider=args.provider,
            model=args.model,
            progress_callback=progress_callback,
        )

        completed_elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        emit_build_progress(
            progress_path,
            {
                "event": "repo_completed",
                "repo": repo_name,
                "provider": payload["provider"],
                "model": payload["model"],
                "documents": payload["summary"]["documents"],
                "nonzero_dimensions": payload["summary"]["nonzero_dimensions"],
                "vector_format": payload["vector_format"],
                "elapsed_ms": completed_elapsed_ms,
            },
            log_message=(
                f"[build-embeddings] repo={repo_name} status=completed "
                f"elapsed_ms={completed_elapsed_ms:.1f} "
                f"documents={payload['summary']['documents']} "
                f"vector_format={payload['vector_format']}"
            ),
        )

        print(
            f"Wrote embeddings for {repo_name} to {search_root / repo_name} "
            f"({payload['summary']['documents']} documents)",
            flush=True,
        )

    return 0


def handle_run_benchmarks(args: argparse.Namespace) -> int:
    eval_root = Path(args.eval_root).resolve()
    progress_path = eval_root / "benchmarks_progress.json"
    started = time.perf_counter()

    emit_build_progress(
        progress_path,
        {
            "event": "run_started",
            "elapsed_ms": 0.0,
        },
        log_message="[run-benchmarks] status=started",
    )

    def progress_callback(event: Dict[str, object]) -> None:
        event_name = str(event.get("event") or "")
        event = enrich_progress_estimate(
            event,
            elapsed_ms=float(event.get("elapsed_ms") or 0.0),
        )
        message = (
            f"[run-benchmarks] event={event_name} "
            f"elapsed_ms={float(event.get('elapsed_ms') or 0.0):.1f}"
        )
        if event.get("case_name"):
            message += f" case={event.get('case_name')}"
        if event.get("repo"):
            message += f" repo={event.get('repo')}"
        if event.get("mode"):
            message += f" mode={event.get('mode')}"
        if event.get("completed_runs") is not None and event.get("total_runs") is not None:
            message += f" progress={int(event.get('completed_runs') or 0)}/{int(event.get('total_runs') or 0)}"
        if event.get("latency_ms") is not None:
            message += f" case_ms={float(event.get('latency_ms') or 0.0):.1f}"
        if event.get("exact_hit") is not None:
            message += f" exact_hit={str(bool(event.get('exact_hit'))).lower()}"
        if event.get("path_hit") is not None:
            message += f" path_hit={str(bool(event.get('path_hit'))).lower()}"
        message = append_progress_estimate(message, event)
        emit_build_progress(progress_path, event, log_message=message)

    payload = run_benchmarks(
        Path(args.search_root).resolve(),
        Path(args.graph_root).resolve(),
        Path(args.parsed_root).resolve(),
        eval_root,
        repos=tuple(args.repo or ()),
        limit=args.limit,
        modes=tuple(args.mode or ()),
        benchmarks=load_benchmark_cases_from_args(args),
        progress_callback=progress_callback,
    )
    emit_build_progress(
        progress_path,
        {
            "event": "run_completed",
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            "runs": payload["summary"]["runs"],
        },
        log_message=(
            f"[run-benchmarks] status=completed "
            f"elapsed_ms={round((time.perf_counter() - started) * 1000, 3):.1f} "
            f"runs={payload['summary']['runs']}"
        ),
    )
    print_json(payload)
    return 0


def load_benchmark_cases_from_args(args: argparse.Namespace) -> List[Dict[str, object]]:
    return load_benchmark_cases(
        cases_root=Path(args.cases_root).resolve() if getattr(args, "cases_root", None) else None,
        case_paths=tuple(Path(path).resolve() for path in (getattr(args, "case_paths", None) or ())),
    )


def handle_graph_query(args: argparse.Namespace) -> int:
    request: Dict[str, object]
    if args.request_file:
        request = json.loads(Path(args.request_file).read_text(encoding="utf-8"))
    elif args.request_json:
        request = json.loads(args.request_json)
    else:
        request = {
            "operation": args.operation or "neighbors",
            "seed": args.seed,
            "target": args.target,
            "edge_types": list(args.edge_type or ()),
            "direction": args.direction or "both",
            "depth": args.depth,
            "node_kinds": list(args.node_kind or ()),
            "limit": args.limit,
            "window": args.window,
        }
    return print_json(
        execute_graph_query(
            Path(args.search_root).resolve(),
            Path(args.parsed_root).resolve(),
            Path(args.graph_root).resolve(),
            args.repo,
            request,
        )
    )


def print_json(payload: object) -> int:
    print(json.dumps(payload, indent=2, sort_keys=False))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "parse-repos":
        return handle_parse_repos(args)
    if args.command == "build-index":
        return handle_build_index(args)
    if args.command == "build-search":
        return handle_build_search(args)
    if args.command == "build-zoekt":
        return handle_build_zoekt(args)
    if args.command == "build-summaries":
        return handle_build_summaries(args)
    if args.command == "build-embeddings":
        return handle_build_embeddings(args)
    if args.command == "run-benchmarks":
        return handle_run_benchmarks(args)
    if args.command == "repo-overview":
        return print_json(repo_overview(Path(args.parsed_root).resolve(), args.repo))
    if args.command == "find-symbol":
        return print_json(find_symbol(Path(args.search_root).resolve(), args.repo, args.query, limit=args.limit))
    if args.command == "find-file":
        return print_json(find_file(Path(args.search_root).resolve(), args.repo, args.path_pattern, limit=args.limit))
    if args.command == "search-lexical":
        return print_json(
            search_lexical(
                Path(args.search_root).resolve(),
                args.repo,
                args.query,
                limit=args.limit,
                kinds=tuple(args.kind or ()),
                path_prefix=args.path_prefix,
            )
        )
    if args.command == "search-zoekt":
        payload = search_zoekt_index(
            args.repo,
            Path(args.zoekt_root).resolve(),
            args.query,
            limit=args.limit,
            symbol=args.symbol,
            zoekt_bin=args.zoekt_bin,
        )
        print_json(payload)
        return 0 if payload.get("available") else 1
    if args.command == "compare-search-backends":
        payload = compare_search_backends(
            Path(args.search_root).resolve(),
            Path(args.zoekt_root).resolve(),
            args.repo,
            args.query,
            limit=args.limit,
            zoekt_bin=args.zoekt_bin,
        )
        print_json(payload)
        return 0 if payload.get("zoekt", {}).get("available") else 1
    if args.command == "embedding-search":
        return print_json(
            {
                "repo": args.repo,
                "query": args.query,
                "results": query_embedding_index(Path(args.search_root).resolve(), args.repo, args.query, limit=args.limit),
            }
        )
    if args.command == "search-paper-pipeline":
        return print_json(
            retrieve_paper_pipeline(
                Path(args.search_root).resolve(),
                args.repo,
                args.query,
                limit=args.limit,
                lexical_pool=args.lexical_pool,
                pre_rank_pool=args.pre_rank_pool,
                rerank_pool=args.rerank_pool,
            )
        )
    if args.command == "trace-calls":
        return print_json(
            trace_calls(
                Path(args.search_root).resolve(),
                Path(args.graph_root).resolve(),
                Path(args.parsed_root).resolve(),
                args.repo,
                args.symbol,
                limit=args.limit,
            )
        )
    if args.command == "where-defined":
        return print_json(
            where_defined(
                Path(args.search_root).resolve(),
                Path(args.parsed_root).resolve(),
                args.repo,
                args.symbol,
                limit=args.limit,
            )
        )
    if args.command == "who-imports":
        return print_json(
            who_imports(
                Path(args.search_root).resolve(),
                Path(args.parsed_root).resolve(),
                Path(args.graph_root).resolve(),
                args.repo,
                args.symbol,
                limit=args.limit,
            )
        )
    if args.command == "adjacent-symbols":
        return print_json(
            adjacent_symbols(
                Path(args.search_root).resolve(),
                Path(args.parsed_root).resolve(),
                Path(args.graph_root).resolve(),
                args.repo,
                args.symbol,
                edge_types=tuple(args.edge_type or ()),
                direction=args.direction,
                limit=args.limit,
            )
        )
    if args.command == "compare-repos":
        return print_json(
            compare_repos(
                Path(args.search_root).resolve(),
                Path(args.graph_root).resolve(),
                Path(args.parsed_root).resolve(),
                args.query,
                repos=tuple(args.repo or ()),
                limit=args.limit,
            )
        )
    if args.command == "summarize-path":
        return print_json(summarize_path(Path(args.parsed_root).resolve(), args.repo, args.path))
    if args.command == "prepare-context":
        return print_json(
            prepare_context(
                Path(args.search_root).resolve(),
                Path(args.graph_root).resolve(),
                Path(args.parsed_root).resolve(),
                args.task,
                repo_name=args.repo,
                limit=args.limit,
            )
        )
    if args.command == "graph-query":
        return handle_graph_query(args)
    if args.command == "path-between":
        return print_json(
            path_between(
                Path(args.search_root).resolve(),
                Path(args.parsed_root).resolve(),
                Path(args.graph_root).resolve(),
                args.repo,
                args.source,
                args.target,
                limit=args.limit,
                edge_types=tuple(args.edge_type or ()),
                direction=args.direction,
            )
        )
    if args.command == "statement-slice":
        return print_json(
            statement_slice(
                Path(args.search_root).resolve(),
                Path(args.parsed_root).resolve(),
                Path(args.graph_root).resolve(),
                args.repo,
                args.symbol,
                limit=args.limit,
                window=args.window,
            )
        )
    if args.command == "callers-of":
        return print_json(
            callers_of(
                Path(args.search_root).resolve(),
                Path(args.parsed_root).resolve(),
                Path(args.graph_root).resolve(),
                args.repo,
                args.symbol,
                limit=args.limit,
            )
        )
    if args.command == "callees-of":
        return print_json(
            callees_of(
                Path(args.search_root).resolve(),
                Path(args.parsed_root).resolve(),
                Path(args.graph_root).resolve(),
                args.repo,
                args.symbol,
                limit=args.limit,
            )
        )
    if args.command == "reads-of":
        return print_json(
            reads_of(
                Path(args.search_root).resolve(),
                Path(args.parsed_root).resolve(),
                Path(args.graph_root).resolve(),
                args.repo,
                args.symbol,
                limit=args.limit,
            )
        )
    if args.command == "writes-of":
        return print_json(
            writes_of(
                Path(args.search_root).resolve(),
                Path(args.parsed_root).resolve(),
                Path(args.graph_root).resolve(),
                args.repo,
                args.symbol,
                limit=args.limit,
            )
        )
    if args.command == "refs-of":
        return print_json(
            refs_of(
                Path(args.search_root).resolve(),
                Path(args.parsed_root).resolve(),
                Path(args.graph_root).resolve(),
                args.repo,
                args.symbol,
                limit=args.limit,
            )
        )
    if args.command == "implements-of":
        return print_json(
            implements_of(
                Path(args.search_root).resolve(),
                Path(args.parsed_root).resolve(),
                Path(args.graph_root).resolve(),
                args.repo,
                args.symbol,
                limit=args.limit,
            )
        )
    if args.command == "inherits-of":
        return print_json(
            inherits_of(
                Path(args.search_root).resolve(),
                Path(args.parsed_root).resolve(),
                Path(args.graph_root).resolve(),
                args.repo,
                args.symbol,
                limit=args.limit,
            )
        )
    if args.command == "get-summary":
        return print_json(
            get_summary(
                Path(args.search_root).resolve(),
                Path(args.graph_root).resolve(),
                Path(args.parsed_root).resolve(),
                args.repo,
                args.node_id,
            )
        )
    if args.command == "get-symbol-signature":
        return print_json(
            get_symbol_signature(
                Path(args.search_root).resolve(),
                Path(args.parsed_root).resolve(),
                args.repo,
                args.symbol,
            )
        )
    if args.command == "get-symbol-body":
        return print_json(
            get_symbol_body(
                Path(args.search_root).resolve(),
                Path(args.parsed_root).resolve(),
                args.repo,
                args.symbol,
            )
        )
    if args.command == "get-enclosing-context":
        return print_json(
            get_enclosing_context(
                Path(args.search_root).resolve(),
                Path(args.graph_root).resolve(),
                Path(args.parsed_root).resolve(),
                args.repo,
                args.symbol,
            )
        )
    if args.command == "expand-subgraph":
        return print_json(
            expand_subgraph(
                Path(args.search_root).resolve(),
                Path(args.parsed_root).resolve(),
                Path(args.graph_root).resolve(),
                args.repo,
                args.seed,
                edge_types=tuple(args.edge_type or ()),
                direction=args.direction,
                depth=args.depth,
                node_kinds=tuple(args.node_kind or ()),
                budget=args.budget,
            )
        )
    if args.command == "plan-query":
        return print_json(
            plan_query(
                Path(args.search_root).resolve(),
                Path(args.graph_root).resolve(),
                Path(args.parsed_root).resolve(),
                args.task,
                repo_name=args.repo,
                limit=args.limit,
            )
        )
    if args.command == "prepare-answer-bundle":
        return print_json(
            prepare_answer_bundle(
                Path(args.search_root).resolve(),
                Path(args.graph_root).resolve(),
                Path(args.parsed_root).resolve(),
                args.task,
                repo_name=args.repo,
                limit=args.limit,
                refinement_hints=tuple(args.hint or ()),
            )
        )
    if args.command == "retrieve-iterative":
        prior_bundle = None
        if args.prior_bundle:
            prior_bundle = json.loads(Path(args.prior_bundle).read_text(encoding="utf-8"))
        return print_json(
            retrieve_iterative(
                Path(args.search_root).resolve(),
                Path(args.graph_root).resolve(),
                Path(args.parsed_root).resolve(),
                args.task,
                repo_name=args.repo,
                limit=args.limit,
                prior_bundle=prior_bundle,
                refinement_hints=tuple(args.hint or ()),
            )
        )
    if args.command == "export-benchmark-prompts":
        return print_json(
            export_benchmark_prompts(
                Path(args.search_root).resolve(),
                Path(args.graph_root).resolve(),
                Path(args.parsed_root).resolve(),
                Path(args.eval_root).resolve(),
                repos=tuple(args.repo or ()),
                limit=args.limit,
                benchmarks=load_benchmark_cases_from_args(args),
            )
        )
    if args.command == "score-answer-bundles":
        return print_json(
            score_answer_bundles(
                Path(args.search_root).resolve(),
                Path(args.graph_root).resolve(),
                Path(args.parsed_root).resolve(),
                Path(args.eval_root).resolve(),
                repos=tuple(args.repo or ()),
                limit=args.limit,
                benchmarks=load_benchmark_cases_from_args(args),
            )
        )
    if args.command == "score-external-answers":
        return print_json(
            score_external_answers(
                Path(args.eval_root).resolve(),
                Path(args.answers_path).resolve(),
                benchmarks=load_benchmark_cases_from_args(args),
            )
        )
    if args.command == "benchmark-interactive":
        return print_json(
            benchmark_interactive_commands(
                Path(args.search_root).resolve(),
                Path(args.graph_root).resolve(),
                Path(args.parsed_root).resolve(),
                Path(args.eval_root).resolve(),
                repos=tuple(args.repo or ()),
                limit=args.limit,
            )
        )

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
