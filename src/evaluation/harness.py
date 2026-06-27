from __future__ import annotations

import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from agents.toolkit import (
    callers_of,
    callees_of,
    find_file,
    find_symbol,
    get_symbol_body,
    get_symbol_signature,
    path_between,
    prepare_answer_bundle as prepare_answer_bundle_tool,
    statement_slice,
    where_defined,
)
from backends.graph_backend import get_graph_backend
from backends.metadata_store import get_metadata_store
from backends.search_backend import get_search_backend
from common.telemetry import reset_telemetry, snapshot_telemetry
from embeddings.indexer import query_embedding_index
from retrieval.engine import retrieve_context
from retrieval.planner import prepare_answer_bundle
from retrieval.staged import retrieve_paper_pipeline
from symbols.indexer import timestamp_now


SCHEMA_VERSION = "0.4.0"
EVAL_CACHE_SCHEMA_VERSION = "0.1.0"
DEFAULT_CASES_ROOT = Path("data/eval/cases")
BENCHMARK_CASE_SUFFIXES = {".json", ".jsonl"}
METRIC_K_VALUES = (1, 3, 5, 10)
DEFAULT_MODES = (
    "semantic_graph_rerank",
    "semantic_graph_rerank_summaries",
    "selective_on",
    "selective_off",
)
LEXICAL_BENCHMARK_MODES = {"lexical_only"}
SEMANTIC_BENCHMARK_MODES = set(DEFAULT_MODES) | {"embedding_only", "paper_pipeline"}
SUPPORTED_BENCHMARK_MODES = SEMANTIC_BENCHMARK_MODES | LEXICAL_BENCHMARK_MODES
SUMMARY_MODES = {
    "semantic_graph_rerank_summaries",
    "lexical_graph_vector_rerank_summaries",
    "selective_on",
    "selective_off",
}
DEFAULT_BENCHMARKS: List[Dict[str, object]] = []

DEFAULT_INTERACTIVE_SCENARIOS: List[Dict[str, object]] = []


def load_benchmark_cases(
    *,
    cases_root: Optional[Path] = None,
    case_paths: Sequence[Path] = (),
) -> List[Dict[str, object]]:
    source_paths: List[Path] = []
    if cases_root is not None and cases_root.exists():
        source_paths.extend(
            path
            for path in sorted(cases_root.rglob("*"))
            if path.is_file() and path.suffix.lower() in BENCHMARK_CASE_SUFFIXES
        )
    for path in case_paths:
        if not path.exists():
            raise FileNotFoundError(f"Benchmark case file does not exist: {path}")
        if path.is_dir():
            source_paths.extend(
                child
                for child in sorted(path.rglob("*"))
                if child.is_file() and child.suffix.lower() in BENCHMARK_CASE_SUFFIXES
            )
        else:
            source_paths.append(path)

    cases: List[Dict[str, object]] = []
    for path in source_paths:
        cases.extend(load_benchmark_cases_file(path))
    return cases


def load_benchmark_cases_file(path: Path) -> List[Dict[str, object]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        cases = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                payload = json.loads(stripped)
                if not isinstance(payload, dict):
                    raise ValueError(f"Benchmark case at {path}:{line_number} must be a JSON object")
                cases.append(normalize_benchmark_case(payload, source=path, ordinal=line_number))
        return cases
    if suffix == ".json":
        payload = load_json(path)
        raw_cases = payload.get("benchmarks", payload.get("cases", payload)) if isinstance(payload, dict) else payload
        if isinstance(raw_cases, dict):
            raw_cases = [raw_cases]
        if not isinstance(raw_cases, list):
            raise ValueError(f"Benchmark JSON file must contain a case object or list: {path}")
        return [
            normalize_benchmark_case(raw_case, source=path, ordinal=index)
            for index, raw_case in enumerate(raw_cases, start=1)
        ]
    raise ValueError(f"Unsupported benchmark case suffix for {path}; expected .json or .jsonl")


def normalize_benchmark_case(raw_case: Dict[str, object], *, source: Path, ordinal: int) -> Dict[str, object]:
    case = dict(raw_case)
    for key in ("name", "repo", "query"):
        if not str(case.get(key) or ""):
            raise ValueError(f"Benchmark case {source}:{ordinal} is missing required field: {key}")
    case.setdefault("task_type", "retrieval")

    expected_paths = normalize_string_list(case.get("expected_paths"))
    if not expected_paths and case.get("expected_path"):
        expected_paths = [str(case["expected_path"])]
    expected_symbols = normalize_string_list(case.get("expected_symbols"))
    if not expected_symbols and case.get("expected_name"):
        expected_symbols = [str(case["expected_name"])]
    if not expected_symbols and case.get("expected_symbol"):
        expected_symbols = [str(case["expected_symbol"])]

    if not expected_paths and not expected_symbols:
        raise ValueError(
            f"Benchmark case {source}:{ordinal} must define expected_path(s) or expected_symbol(s)"
        )

    case["name"] = str(case["name"])
    case["repo"] = str(case["repo"])
    case["query"] = str(case["query"])
    case["task_type"] = str(case["task_type"])
    case["expected_paths"] = expected_paths
    case["expected_symbols"] = expected_symbols
    case["expected_path"] = str(case.get("expected_path") or (expected_paths[0] if expected_paths else ""))
    case["expected_name"] = str(case.get("expected_name") or (expected_symbols[0] if expected_symbols else ""))
    case["expected_terms"] = normalize_string_list(case.get("expected_terms"))
    case["source"] = str(source)
    return case


def normalize_benchmark_cases(raw_cases: Sequence[Dict[str, object]], *, source: Path = Path("<memory>")) -> List[Dict[str, object]]:
    return [
        normalize_benchmark_case(raw_case, source=source, ordinal=index)
        for index, raw_case in enumerate(raw_cases, start=1)
    ]


def normalize_string_list(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def run_benchmarks(
    search_root: Path,
    graph_root: Path,
    parsed_root: Path,
    eval_root: Path,
    *,
    repos: Sequence[str] = (),
    limit: int = 5,
    modes: Sequence[str] = DEFAULT_MODES,
    benchmarks: Optional[Sequence[Dict[str, object]]] = None,
    progress_callback=None,
) -> Dict[str, object]:
    started = time.perf_counter()
    benchmark_cases = normalize_benchmark_cases(DEFAULT_BENCHMARKS if benchmarks is None else benchmarks)
    if not benchmark_cases:
        raise ValueError(
            "No benchmark cases configured; add JSON/JSONL cases under data/eval/cases or pass benchmarks explicitly"
        )
    selected_repos = set(repos or [item["repo"] for item in benchmark_cases])
    cases = [item for item in benchmark_cases if item["repo"] in selected_repos]
    if not cases:
        raise ValueError(f"No benchmark cases matched repos: {', '.join(sorted(selected_repos))}")
    selected_modes = tuple(modes or DEFAULT_MODES)
    unsupported_modes = sorted(set(selected_modes) - SUPPORTED_BENCHMARK_MODES)
    if unsupported_modes:
        raise ValueError(f"Unsupported benchmark modes: {', '.join(unsupported_modes)}")
    total_runs = len(cases) * len(selected_modes)

    def emit(event: str, **extra: object) -> None:
        if progress_callback is None:
            return
        progress_callback(
            {
                "event": event,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                "cases": len(cases),
                "modes": len(selected_modes),
                "total_runs": total_runs,
                **extra,
            }
        )

    emit("run_started", repos=sorted(selected_repos))

    runs = []
    completed_runs = 0
    for case in cases:
        for mode in selected_modes:
            emit("case_started", repo=case["repo"], case_name=case["name"], mode=mode, completed_runs=completed_runs)
            runs.append(run_case(case, mode, search_root, graph_root, parsed_root, limit))
            completed_runs += 1
            latest = runs[-1]
            emit(
                "case_completed",
                repo=case["repo"],
                case_name=case["name"],
                mode=mode,
                completed_runs=completed_runs,
                exact_hit=latest["exact_hit"],
                path_hit=latest["path_hit"],
                symbol_hit=latest["symbol_hit"],
                recall_at_k=latest["retrieval_metrics"]["recall_at_k"],
                mrr=latest["retrieval_metrics"]["mrr"],
                ndcg=latest["retrieval_metrics"]["ndcg"],
                latency_ms=latest["latency_ms"],
            )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": timestamp_now(),
        "summary": summarize_runs(runs),
        "runs": runs,
    }

    eval_root.mkdir(parents=True, exist_ok=True)
    write_json(eval_root / "benchmarks.json", payload)
    emit("run_completed", completed_runs=completed_runs)
    return payload


def benchmark_interactive_commands(
    search_root: Path,
    graph_root: Path,
    parsed_root: Path,
    eval_root: Path,
    *,
    repos: Sequence[str] = (),
    scenarios: Optional[Sequence[Dict[str, object]]] = None,
    limit: int = 5,
) -> Dict[str, object]:
    selected_repos = set(repos or [item["repo"] for item in (scenarios or DEFAULT_INTERACTIVE_SCENARIOS)])
    selected_scenarios = [item for item in (scenarios or DEFAULT_INTERACTIVE_SCENARIOS) if item["repo"] in selected_repos]
    runs = []
    for scenario in selected_scenarios:
        repo_name = str(scenario["repo"])
        command_specs = [
            (
                "find-symbol",
                lambda: find_symbol(search_root, repo_name, str(scenario["symbol_query"]), limit=limit),
            ),
            (
                "find-file",
                lambda: find_file(search_root, repo_name, str(scenario["file_query"]), limit=limit),
            ),
            (
                "where-defined",
                lambda: where_defined(search_root, parsed_root, repo_name, str(scenario["symbol_query"]), limit=limit),
            ),
            (
                "get-symbol-signature",
                lambda: get_symbol_signature(search_root, parsed_root, repo_name, str(scenario["symbol_query"])),
            ),
            (
                "get-symbol-body",
                lambda: get_symbol_body(search_root, parsed_root, repo_name, str(scenario["symbol_query"])),
            ),
            (
                "callers-of",
                lambda: callers_of(search_root, parsed_root, graph_root, repo_name, str(scenario["call_symbol"]), limit=limit),
            ),
            (
                "callees-of",
                lambda: callees_of(search_root, parsed_root, graph_root, repo_name, str(scenario["call_symbol"]), limit=limit),
            ),
            (
                "path-between",
                lambda: path_between(
                    search_root,
                    parsed_root,
                    graph_root,
                    repo_name,
                    str(scenario["path_source"]),
                    str(scenario["path_target"]),
                    limit=limit,
                ),
            ),
            (
                "statement-slice",
                lambda: statement_slice(search_root, parsed_root, graph_root, repo_name, str(scenario["statement_symbol"]), limit=limit),
            ),
            (
                "prepare-answer-bundle",
                lambda: prepare_answer_bundle_tool(
                    search_root,
                    graph_root,
                    parsed_root,
                    str(scenario["bundle_query"]),
                    repo_name=repo_name,
                    limit=limit,
                ),
            ),
        ]
        for command_name, runner in command_specs:
            reset_telemetry()
            started = time.perf_counter()
            payload = runner()
            elapsed_ms = (time.perf_counter() - started) * 1000
            runs.append(
                {
                    "repo": repo_name,
                    "command": command_name,
                    "elapsed_ms": round(elapsed_ms, 3),
                    "telemetry": snapshot_telemetry(),
                    "result_summary": summarize_benchmark_result(payload),
                }
            )

    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": timestamp_now(),
        "summary": summarize_interactive_benchmark_runs(runs),
        "runs": runs,
    }
    eval_root.mkdir(parents=True, exist_ok=True)
    write_json(eval_root / "interactive_benchmark_report.json", report)
    return report


def export_benchmark_prompts(
    search_root: Path,
    graph_root: Path,
    parsed_root: Path,
    eval_root: Path,
    *,
    repos: Sequence[str] = (),
    limit: int = 8,
    benchmarks: Optional[Sequence[Dict[str, object]]] = None,
) -> Dict[str, object]:
    benchmark_cases = normalize_benchmark_cases(DEFAULT_BENCHMARKS if benchmarks is None else benchmarks)
    selected_repos = set(repos or [item["repo"] for item in benchmark_cases])
    cases = [item for item in benchmark_cases if item["repo"] in selected_repos]
    export_root = eval_root / "prompt_exports"
    export_root.mkdir(parents=True, exist_ok=True)
    cache_path = ensure_eval_cache_database(eval_root)

    prompt_exports = []
    for case in cases:
        cached_case = load_or_build_cached_case(
            cache_path,
            search_root,
            graph_root,
            parsed_root,
            case,
            limit=limit,
        )
        prompt_payload = cached_case["prompt_payload"]
        write_json(export_root / f"{case['name']}.json", prompt_payload)
        prompt_exports.append(
            {
                "name": case["name"],
                "repo": case["repo"],
                "path": f"data/eval/prompt_exports/{case['name']}.json",
            }
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": timestamp_now(),
        "summary": {
            "exports": len(prompt_exports),
            "repos": sorted(selected_repos),
        },
        "exports": prompt_exports,
    }
    write_json(export_root / "manifest.json", manifest)
    return manifest


def score_answer_bundles(
    search_root: Path,
    graph_root: Path,
    parsed_root: Path,
    eval_root: Path,
    *,
    repos: Sequence[str] = (),
    limit: int = 8,
    benchmarks: Optional[Sequence[Dict[str, object]]] = None,
) -> Dict[str, object]:
    benchmark_cases = normalize_benchmark_cases(DEFAULT_BENCHMARKS if benchmarks is None else benchmarks)
    selected_repos = set(repos or [item["repo"] for item in benchmark_cases])
    cases = [item for item in benchmark_cases if item["repo"] in selected_repos]
    cache_path = ensure_eval_cache_database(eval_root)
    scores = []
    for case in cases:
        cached_case = load_or_build_cached_case(
            cache_path,
            search_root,
            graph_root,
            parsed_root,
            case,
            limit=limit,
        )
        scores.append(cached_case["bundle_score"])
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": timestamp_now(),
        "summary": summarize_bundle_scores(scores),
        "scores": scores,
    }
    eval_root.mkdir(parents=True, exist_ok=True)
    write_json(eval_root / "bundle_scores.json", payload)
    return payload


def summarize_benchmark_result(payload: Dict[str, object]) -> Dict[str, object]:
    if "results" in payload and isinstance(payload["results"], list):
        return {"results": len(payload["results"])}
    if "matches" in payload and isinstance(payload["matches"], list):
        return {"matches": len(payload["matches"])}
    if "neighbors" in payload and isinstance(payload["neighbors"], list):
        return {"neighbors": len(payload["neighbors"])}
    if "paths" in payload and isinstance(payload["paths"], list):
        return {"paths": len(payload["paths"])}
    if "statements" in payload and isinstance(payload["statements"], list):
        return {"statements": len(payload["statements"])}
    if "bundles" in payload and isinstance(payload["bundles"], list):
        return {"bundles": len(payload["bundles"])}
    return {}


def summarize_interactive_benchmark_runs(runs: Sequence[Dict[str, object]]) -> Dict[str, object]:
    by_command: Dict[str, List[float]] = {}
    for run in runs:
        by_command.setdefault(str(run["command"]), []).append(float(run["elapsed_ms"]))
    commands = []
    for command_name, latencies in sorted(by_command.items()):
        ordered = sorted(latencies)
        median = ordered[len(ordered) // 2] if ordered else 0.0
        commands.append(
            {
                "command": command_name,
                "runs": len(ordered),
                "median_ms": round(median, 3),
                "max_ms": round(max(ordered), 3) if ordered else 0.0,
            }
        )
    return {
        "runs": len(runs),
        "commands": commands,
    }


def score_external_answers(
    eval_root: Path,
    answers_path: Path,
    *,
    benchmarks: Optional[Sequence[Dict[str, object]]] = None,
) -> Dict[str, object]:
    benchmark_cases = {
        item["name"]: item
        for item in normalize_benchmark_cases(DEFAULT_BENCHMARKS if benchmarks is None else benchmarks)
    }
    answers_payload = load_json(answers_path)
    raw_answers = answers_payload.get("answers", answers_payload)
    if isinstance(raw_answers, list):
        answers_by_name = {item["name"]: item for item in raw_answers}
    else:
        answers_by_name = raw_answers

    results = []
    for case_name, case in sorted(benchmark_cases.items()):
        answer = answers_by_name.get(case_name, {})
        results.append(score_external_answer(case, answer))

    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": timestamp_now(),
        "summary": summarize_external_answer_scores(results),
        "scores": results,
    }
    eval_root.mkdir(parents=True, exist_ok=True)
    write_json(eval_root / "external_answer_scores.json", payload)
    return payload


def run_case(
    case: Dict[str, object],
    mode: str,
    search_root: Path,
    graph_root: Path,
    parsed_root: Path,
    limit: int,
) -> Dict[str, object]:
    started = time.perf_counter()
    if mode == "embedding_only":
        selected = query_embedding_index(search_root, case["repo"], case["query"], limit=limit)
        context_summary = {"mode": "embedding_only"}
    elif mode == "paper_pipeline":
        context = retrieve_paper_pipeline(search_root, case["repo"], case["query"], limit=limit)
        selected = context["selected_context"]
        context_summary = context["summary"]
    elif mode == "lexical_only":
        search_backend = get_search_backend(str(search_root.resolve()), str(case["repo"]))
        selected = search_backend.search(str(case["query"]), limit=limit)
        context_summary = {
            "mode": "lexical_only",
            "search_backend": "tantivy",
            "selected": len(selected),
            "embeddings_enabled": False,
            "graph_enabled": False,
            "rerank_enabled": False,
        }
    else:
        context = retrieve_context(
            search_root,
            graph_root,
            parsed_root,
            case["repo"],
            case["query"],
            limit=limit,
            use_graph=True,
            use_rerank=True,
            use_summaries=mode in SUMMARY_MODES,
            selective_retrieval=mode == "selective_on",
        )
        selected = context["selected_context"]
        context_summary = context["summary"]
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)

    retrieval_metrics = score_retrieval(case, selected, limit)
    exact_hit = bool(retrieval_metrics["exact_hit"])
    path_hit = bool(retrieval_metrics["expected_path_hit"])
    symbol_hit = bool(retrieval_metrics["expected_symbol_hit"])
    answer_quality = grade_answer_quality(case, selected)
    bundle_quality = None
    if mode not in LEXICAL_BENCHMARK_MODES:
        bundle = prepare_answer_bundle(
            search_root,
            graph_root,
            parsed_root,
            case["query"],
            repo_name=case["repo"],
            limit=limit,
        )
        bundle_quality = score_bundle(case, bundle)

    return {
        "name": case["name"],
        "repo": case["repo"],
        "task_type": case["task_type"],
        "query": case["query"],
        "mode": mode,
        "expected_path": case["expected_path"],
        "expected_name": case["expected_name"],
        "expected_paths": case.get("expected_paths", []),
        "expected_symbols": case.get("expected_symbols", []),
        "latency_ms": elapsed_ms,
        "exact_hit": exact_hit,
        "path_hit": path_hit,
        "symbol_hit": symbol_hit,
        "retrieval_metrics": retrieval_metrics,
        "files_opened": count_unique_paths(selected),
        "prepared_tokens": estimate_prepared_tokens(selected),
        "selected_count": len(selected),
        "answer_quality": answer_quality,
        "bundle_quality": bundle_quality,
        "context_summary": context_summary,
        "selected": selected,
    }


def summarize_runs(runs: Sequence[Dict[str, object]]) -> Dict[str, object]:
    by_mode: Dict[str, List[Dict[str, object]]] = {}
    for run in runs:
        by_mode.setdefault(run["mode"], []).append(run)

    mode_summaries = []
    for mode, mode_runs in sorted(by_mode.items()):
        mode_summaries.append(
            {
                "mode": mode,
                "runs": len(mode_runs),
                "exact_hits": sum(1 for run in mode_runs if run["exact_hit"]),
                "path_hits": sum(1 for run in mode_runs if run["path_hit"]),
                "symbol_hits": sum(1 for run in mode_runs if run["symbol_hit"]),
                "avg_recall_at_k": average(run["retrieval_metrics"]["recall_at_k"] for run in mode_runs),
                "avg_mrr": average(run["retrieval_metrics"]["mrr"] for run in mode_runs),
                "avg_ndcg": average(run["retrieval_metrics"]["ndcg"] for run in mode_runs),
                "avg_latency_ms": average(run["latency_ms"] for run in mode_runs),
                "avg_files_opened": average(run["files_opened"] for run in mode_runs),
                "avg_prepared_tokens": average(run["prepared_tokens"] for run in mode_runs),
                "avg_answer_score": average(run["answer_quality"]["score"] for run in mode_runs),
                "avg_bundle_score": average(
                    run["bundle_quality"]["score"] for run in mode_runs if run.get("bundle_quality") is not None
                ),
            }
        )

    retrieval_metrics = {
        "runs": len(runs),
        "exact_hits": sum(1 for run in runs if run["exact_hit"]),
        "path_hits": sum(1 for run in runs if run["path_hit"]),
        "symbol_hits": sum(1 for run in runs if run["symbol_hit"]),
        "avg_recall_at_k": average(run["retrieval_metrics"]["recall_at_k"] for run in runs),
        "avg_mrr": average(run["retrieval_metrics"]["mrr"] for run in runs),
        "avg_ndcg": average(run["retrieval_metrics"]["ndcg"] for run in runs),
        "avg_latency_ms": average(run["latency_ms"] for run in runs),
    }
    consumer_metrics = {
        "avg_answer_score": average(run["answer_quality"]["score"] for run in runs),
        "avg_bundle_score": average(
            run["bundle_quality"]["score"] for run in runs if run.get("bundle_quality") is not None
        ),
        "runs_with_bundle_scores": sum(1 for run in runs if run.get("bundle_quality") is not None),
    }

    return {
        "runs": len(runs),
        "modes": mode_summaries,
        "task_types": summarize_task_types(runs),
        "retrieval_metrics": retrieval_metrics,
        "consumer_readiness": consumer_metrics,
    }


def score_retrieval(case: Dict[str, object], selected: Sequence[Dict[str, object]], limit: int) -> Dict[str, object]:
    expected_paths = [normalize_path_match(path) for path in case.get("expected_paths", []) if str(path)]
    expected_symbols = [normalize_symbol_match(symbol) for symbol in case.get("expected_symbols", []) if str(symbol)]
    expected_units = [f"path:{path}" for path in expected_paths] + [f"symbol:{symbol}" for symbol in expected_symbols]
    top_selected = list(selected[:limit])

    path_ranks = ranks_for_expected_paths(top_selected, expected_paths)
    symbol_ranks = ranks_for_expected_symbols(top_selected, expected_symbols)
    matched_units_at_k = set()
    exact_hit = False
    relevance_by_rank = []
    for item in top_selected:
        item_path = normalize_path_match(item.get("path"))
        item_symbols = item_symbol_matches(item)
        item_units = set()
        if item_path in expected_paths:
            item_units.add(f"path:{item_path}")
        for symbol in expected_symbols:
            if symbol in item_symbols:
                item_units.add(f"symbol:{symbol}")
        if item_path in expected_paths and item_symbols.intersection(expected_symbols):
            exact_hit = True
        matched_units_at_k.update(item_units)
        relevance_by_rank.append(1.0 if item_units else 0.0)

    expected_unit_count = len(expected_units)
    first_relevant_rank = min([*path_ranks.values(), *symbol_ranks.values()], default=None)
    recall_by_k = {
        f"recall_at_{k}": recall_at_k(top_selected, case, k)
        for k in METRIC_K_VALUES
        if k <= limit
    }
    recall_by_k["recall_at_k"] = round(len(matched_units_at_k) / expected_unit_count, 3) if expected_unit_count else 0.0

    return {
        "k": limit,
        **recall_by_k,
        "mrr": round(1.0 / first_relevant_rank, 3) if first_relevant_rank else 0.0,
        "ndcg": ndcg(relevance_by_rank, expected_unit_count),
        "expected_path_hit": bool(path_ranks),
        "expected_symbol_hit": bool(symbol_ranks),
        "exact_hit": exact_hit,
        "path_recall_at_k": round(len(path_ranks) / len(expected_paths), 3) if expected_paths else 0.0,
        "symbol_recall_at_k": round(len(symbol_ranks) / len(expected_symbols), 3) if expected_symbols else 0.0,
        "first_relevant_rank": first_relevant_rank,
        "path_ranks": path_ranks,
        "symbol_ranks": symbol_ranks,
        "expected_units": expected_unit_count,
        "matched_units": len(matched_units_at_k),
    }


def recall_at_k(selected: Sequence[Dict[str, object]], case: Dict[str, object], k: int) -> float:
    expected_paths = [normalize_path_match(path) for path in case.get("expected_paths", []) if str(path)]
    expected_symbols = [normalize_symbol_match(symbol) for symbol in case.get("expected_symbols", []) if str(symbol)]
    expected_units = {f"path:{path}" for path in expected_paths} | {f"symbol:{symbol}" for symbol in expected_symbols}
    if not expected_units:
        return 0.0
    matched_units = set()
    for item in selected[:k]:
        item_path = normalize_path_match(item.get("path"))
        if item_path in expected_paths:
            matched_units.add(f"path:{item_path}")
        item_symbols = item_symbol_matches(item)
        for symbol in expected_symbols:
            if symbol in item_symbols:
                matched_units.add(f"symbol:{symbol}")
    return round(len(matched_units) / len(expected_units), 3)


def ranks_for_expected_paths(selected: Sequence[Dict[str, object]], expected_paths: Sequence[str]) -> Dict[str, int]:
    ranks: Dict[str, int] = {}
    for rank, item in enumerate(selected, start=1):
        item_path = normalize_path_match(item.get("path"))
        if item_path in expected_paths and item_path not in ranks:
            ranks[item_path] = rank
    return ranks


def ranks_for_expected_symbols(selected: Sequence[Dict[str, object]], expected_symbols: Sequence[str]) -> Dict[str, int]:
    ranks: Dict[str, int] = {}
    for rank, item in enumerate(selected, start=1):
        item_symbols = item_symbol_matches(item)
        for expected_symbol in expected_symbols:
            if expected_symbol in item_symbols and expected_symbol not in ranks:
                ranks[expected_symbol] = rank
    return ranks


def item_symbol_matches(item: Dict[str, object]) -> set[str]:
    values = {
        item.get("name"),
        item.get("qualified_name"),
        item.get("symbol_id"),
        item.get("title"),
        item.get("doc_id"),
    }
    return {normalize_symbol_match(value) for value in values if str(value or "")}


def normalize_path_match(value: object) -> str:
    text = str(value or "").strip()
    while text.startswith("./"):
        text = text[2:]
    return text


def normalize_symbol_match(value: object) -> str:
    return str(value or "").strip().lower()


def ndcg(relevance_by_rank: Sequence[float], expected_unit_count: int) -> float:
    if expected_unit_count <= 0:
        return 0.0
    dcg = 0.0
    for index, relevance in enumerate(relevance_by_rank, start=1):
        if relevance:
            dcg += float(relevance) / log2(index + 1)
    ideal_count = min(expected_unit_count, len(relevance_by_rank))
    idcg = sum(1.0 / log2(index + 1) for index in range(1, ideal_count + 1))
    return round(dcg / idcg, 3) if idcg else 0.0


def log2(value: int) -> float:
    return math.log2(value)


def score_bundle(case: Dict[str, object], bundle: Dict[str, object]) -> Dict[str, object]:
    repo_bundle = bundle["bundles"][0]
    selected = repo_bundle["selected_context"]
    evidence = repo_bundle["evidence"]
    graph_neighborhoods = repo_bundle["graph_neighborhoods"]
    statement_slices = repo_bundle["statement_slices"]
    all_text = normalize_answer_text(
        " ".join(
            str(part or "")
            for item in [*selected, *evidence]
            for part in (
                item.get("path"),
                item.get("name"),
                item.get("qualified_name"),
                item.get("title"),
                item.get("preview"),
                item.get("why_included"),
            )
        )
    )

    expected_terms = [str(term) for term in case.get("expected_terms", [])]
    expected_name = str(case.get("expected_name") or "")
    expected_path = str(case.get("expected_path") or "")
    path_credit = 1.0 if any(str(item.get("path") or "") == expected_path for item in selected) else 0.0
    name_credit = 1.0 if expected_name and expected_name.lower() in all_text else 0.0
    term_hits = sum(1 for term in expected_terms if normalize_answer_text(term) in all_text)
    term_coverage = round(term_hits / len(expected_terms), 3) if expected_terms else 0.0
    provenance_credit = 1.0 if all(item.get("provenance", {}).get("path") for item in evidence[:3]) else 0.0
    graph_credit = 1.0 if graph_neighborhoods else 0.0
    statement_credit = 1.0 if statement_slices else 0.0
    score = round(
        path_credit * 0.3
        + name_credit * 0.2
        + term_coverage * 0.2
        + provenance_credit * 0.1
        + graph_credit * 0.1
        + statement_credit * 0.1,
        3,
    )
    return {
        "score": score,
        "path_credit": path_credit,
        "name_credit": name_credit,
        "term_coverage": term_coverage,
        "provenance_credit": provenance_credit,
        "graph_credit": graph_credit,
        "statement_credit": statement_credit,
        "selected_context": len(selected),
        "evidence_items": len(evidence),
    }


def score_external_answer(case: Dict[str, object], answer: Dict[str, object]) -> Dict[str, object]:
    answer_text = normalize_answer_text(str(answer.get("answer") or ""))
    cited_paths = {str(item) for item in answer.get("cited_paths", [])}
    cited_symbols = {str(item) for item in answer.get("cited_symbols", [])}
    expected_terms = [str(term) for term in case.get("expected_terms", [])]
    expected_name = str(case.get("expected_name") or "")
    expected_path = str(case.get("expected_path") or "")
    name_credit = 1.0 if expected_name.lower() in answer_text else 0.0
    path_credit = 1.0 if expected_path in cited_paths or expected_path in answer_text else 0.0
    symbol_credit = 1.0 if expected_name in cited_symbols else 0.0
    term_hits = sum(1 for term in expected_terms if normalize_answer_text(term) in answer_text)
    term_coverage = round(term_hits / len(expected_terms), 3) if expected_terms else 0.0
    score = round(path_credit * 0.35 + name_credit * 0.25 + symbol_credit * 0.2 + term_coverage * 0.2, 3)
    return {
        "name": case["name"],
        "repo": case["repo"],
        "score": score,
        "path_credit": path_credit,
        "name_credit": name_credit,
        "symbol_credit": symbol_credit,
        "term_coverage": term_coverage,
    }


def summarize_bundle_scores(scores: Sequence[Dict[str, object]]) -> Dict[str, object]:
    return {
        "cases": len(scores),
        "avg_bundle_score": average(item["score"] for item in scores),
        "full_path_hits": sum(1 for item in scores if item["path_credit"] >= 1.0),
    }


def summarize_external_answer_scores(scores: Sequence[Dict[str, object]]) -> Dict[str, object]:
    return {
        "cases": len(scores),
        "avg_score": average(item["score"] for item in scores),
        "path_hits": sum(1 for item in scores if item["path_credit"] >= 1.0),
        "name_hits": sum(1 for item in scores if item["name_credit"] >= 1.0),
    }


def ensure_eval_cache_database(eval_root: Path) -> Path:
    eval_root.mkdir(parents=True, exist_ok=True)
    path = eval_root / "eval.lmdb"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_or_build_cached_case(
    cache_path: Path,
    search_root: Path,
    graph_root: Path,
    parsed_root: Path,
    case: Dict[str, object],
    *,
    limit: int,
) -> Dict[str, object]:
    repo_name = str(case["repo"])
    artifact_fingerprint = compute_repo_artifact_fingerprint(
        repo_name,
        search_root=search_root,
        graph_root=graph_root,
        parsed_root=parsed_root,
    )
    cache_fingerprint = compute_case_cache_fingerprint(case, artifact_fingerprint, limit=limit)

    metadata_store = get_metadata_store(
        str(parsed_root.resolve()),
        repo_name,
        eval_root=str(cache_path.parent.resolve()),
    )
    cached = metadata_store.get_eval_case(str(case["name"]), cache_fingerprint)
    if cached is not None:
        return {
            "bundle": cached["bundle"],
            "prompt_payload": cached["prompt_payload"],
            "bundle_score": cached["bundle_score"],
            "cache": "hit",
        }

    bundle = prepare_answer_bundle(
        search_root,
        graph_root,
        parsed_root,
        str(case["query"]),
        repo_name=repo_name,
        limit=limit,
    )
    prompt_payload = build_prompt_payload(case, bundle)
    bundle_score = score_bundle(case, bundle)
    metadata_store.put_eval_case(
        str(case["name"]),
        repo=repo_name,
        task_type=str(case["task_type"]),
        query=str(case["query"]),
        limit_value=int(limit),
        artifact_fingerprint=artifact_fingerprint,
        cache_fingerprint=cache_fingerprint,
        bundle=bundle,
        prompt_payload=prompt_payload,
        bundle_score=bundle_score,
    )
    return {
        "bundle": bundle,
        "prompt_payload": prompt_payload,
        "bundle_score": bundle_score,
        "cache": "miss",
    }


def compute_case_cache_fingerprint(case: Dict[str, object], artifact_fingerprint: str, *, limit: int) -> str:
    payload = {
        "schema_version": EVAL_CACHE_SCHEMA_VERSION,
        "artifact_fingerprint": artifact_fingerprint,
        "limit": int(limit),
        "case": {
            "name": str(case["name"]),
            "repo": str(case["repo"]),
            "task_type": str(case["task_type"]),
            "query": str(case["query"]),
            "expected_path": str(case["expected_path"]),
            "expected_name": str(case["expected_name"]),
            "expected_terms": [str(term) for term in case.get("expected_terms", [])],
        },
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def compute_repo_artifact_fingerprint(
    repo_name: str,
    *,
    search_root: Path,
    graph_root: Path,
    parsed_root: Path,
) -> str:
    search_backend = get_search_backend(str(search_root.resolve()), repo_name)
    graph_backend = get_graph_backend(str(graph_root.resolve()), repo_name)
    metadata_store = get_metadata_store(
        str(parsed_root.resolve()),
        repo_name,
    )
    snapshot = {
        "search": search_backend.artifact_fingerprint(),
        "graph": graph_backend.artifact_fingerprint(),
        "metadata": metadata_store.artifact_fingerprint(),
    }
    return hashlib.sha1(json.dumps(snapshot, sort_keys=True).encode("utf-8")).hexdigest()


def build_prompt_payload(case: Dict[str, object], bundle: Dict[str, object]) -> Dict[str, object]:
    return {
        "name": case["name"],
        "repo": case["repo"],
        "task_type": case["task_type"],
        "query": case["query"],
        "expected_path": case["expected_path"],
        "expected_name": case["expected_name"],
        "expected_terms": case.get("expected_terms", []),
        "prompt": build_prompt_text(case, bundle),
        "answer_bundle": bundle,
        "provenance_requirements": {
            "must_cite_path": case["expected_path"],
            "should_cite_symbol": case["expected_name"],
        },
    }


def build_prompt_text(case: Dict[str, object], bundle: Dict[str, object]) -> str:
    repo_bundle = bundle["bundles"][0]
    return (
        "Answer the repository question using only the provided bundle.\n"
        f"Question: {case['query']}\n"
        f"Expected task type: {case['task_type']}\n"
        "Cite file paths and symbol names used as evidence.\n"
        f"Project summary: {repo_bundle['project_summary']}\n"
        f"Evidence items: {json.dumps(repo_bundle['evidence'][:5], sort_keys=False)}\n"
    )


def count_unique_paths(selected: Sequence[Dict[str, object]]) -> int:
    return len({str(item.get("path")) for item in selected if item.get("path")})


def estimate_prepared_tokens(selected: Sequence[Dict[str, object]]) -> int:
    total = 0
    for item in selected:
        text = " ".join(
            str(part or "")
            for part in (item.get("title"), item.get("preview"), item.get("qualified_name"), item.get("path"))
        )
        total += len(text.split())
    return total


def average(values: Sequence[float] | Sequence[int]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return round(sum(float(value) for value in values) / len(values), 3)


def summarize_task_types(runs: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for run in runs:
        grouped.setdefault(str(run["task_type"]), []).append(run)
    return [
        {
            "task_type": task_type,
            "runs": len(task_runs),
            "exact_hits": sum(1 for run in task_runs if run["exact_hit"]),
            "path_hits": sum(1 for run in task_runs if run["path_hit"]),
            "symbol_hits": sum(1 for run in task_runs if run["symbol_hit"]),
            "avg_recall_at_k": average(run["retrieval_metrics"]["recall_at_k"] for run in task_runs),
            "avg_mrr": average(run["retrieval_metrics"]["mrr"] for run in task_runs),
            "avg_ndcg": average(run["retrieval_metrics"]["ndcg"] for run in task_runs),
            "avg_answer_score": average(run["answer_quality"]["score"] for run in task_runs),
            "avg_bundle_score": average(
                run["bundle_quality"]["score"] for run in task_runs if run.get("bundle_quality") is not None
            ),
        }
        for task_type, task_runs in sorted(grouped.items())
    ]


def grade_answer_quality(case: Dict[str, object], selected: Sequence[Dict[str, object]]) -> Dict[str, object]:
    synthesized_answer = synthesize_answer(selected)
    haystack = normalize_answer_text(synthesized_answer)
    expected_terms = [str(term) for term in case.get("expected_terms", [])]
    expected_name = str(case.get("expected_name") or "")
    expected_path = str(case.get("expected_path") or "")

    path_credit = 1.0 if any(str(item.get("path") or "") == expected_path for item in selected) else 0.0
    name_credit = 1.0 if expected_name and expected_name.lower() in haystack else 0.0
    term_hits = sum(1 for term in expected_terms if normalize_answer_text(term) in haystack)
    term_coverage = round(term_hits / len(expected_terms), 3) if expected_terms else 0.0
    top_hit = bool(
        selected
        and str(selected[0].get("path") or "") == expected_path
        and str(selected[0].get("name") or "") == expected_name
    )
    score = round(path_credit * 0.35 + name_credit * 0.35 + term_coverage * 0.2 + (0.1 if top_hit else 0.0), 3)

    return {
        "score": score,
        "path_credit": path_credit,
        "name_credit": name_credit,
        "term_coverage": term_coverage,
        "top_hit": top_hit,
        "expected_terms": expected_terms,
        "synthesized_answer": synthesized_answer,
    }


def synthesize_answer(selected: Sequence[Dict[str, object]]) -> str:
    parts: List[str] = []
    for item in selected[:3]:
        part_bits = []
        if item.get("qualified_name"):
            part_bits.append(str(item["qualified_name"]))
        elif item.get("name"):
            part_bits.append(str(item["name"]))
        elif item.get("title"):
            part_bits.append(str(item["title"]))
        if item.get("path"):
            part_bits.append(f"in {item['path']}")
        if item.get("preview"):
            part_bits.append(str(item["preview"]))
        if part_bits:
            parts.append(" ".join(part_bits))
    return " | ".join(parts)


def normalize_answer_text(value: str) -> str:
    return re.sub(r"[^a-z0-9_:/.-]+", " ", value.lower()).strip()


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")
