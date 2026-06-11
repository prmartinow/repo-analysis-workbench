from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from backends.graph_backend import get_graph_backend
from backends.metadata_store import get_metadata_store
from common.text import tokenize
from common.telemetry import trace_operation
from retrieval.engine import classify_query, retrieve_context
from symbols.indexer import stable_id


DEFAULT_REPOS: tuple[str, ...] = ()


def plan_query(
    search_root: Path,
    graph_root: Path,
    parsed_root: Path,
    task: str,
    *,
    repo_name: Optional[str] = None,
    limit: int = 8,
) -> Dict[str, object]:
    repos = (repo_name,) if repo_name else DEFAULT_REPOS
    tokens = tokenize(task)
    query_profile = classify_query(tokens, task)
    plans = []
    for current_repo in repos:
        preview_context = retrieve_context(
            search_root,
            graph_root,
            parsed_root,
            current_repo,
            task,
            limit=min(max(limit, 3), 6),
            use_graph=False,
            use_embeddings=False,
            use_rerank=False,
            use_summaries=True,
            selective_retrieval=False,
        )
        retrieval_recipe = choose_recipe(query_profile)
        plans.append(
            {
                "repo": current_repo,
                "intent": query_profile["intent"],
                "candidate_seed_types": candidate_seed_types(query_profile),
                "retrieval_recipe": retrieval_recipe,
                "stopping_criteria": stopping_criteria(query_profile, limit),
                "summary_preview": preview_context["summary_results"],
                "symbol_preview": preview_context["symbol_results"],
                "lexical_preview": preview_context["selected_context"],
            }
        )

    return {
        "task": task,
        "query_profile": query_profile,
        "plans": plans,
    }


def prepare_answer_bundle(
    search_root: Path,
    graph_root: Path,
    parsed_root: Path,
    task: str,
    *,
    repo_name: Optional[str] = None,
    limit: int = 8,
    refinement_hints: Sequence[str] = (),
) -> Dict[str, object]:
    with trace_operation("prepare_answer_bundle"):
        normalized_task = normalize_task(task, refinement_hints)
        plan = plan_query(
            search_root,
            graph_root,
            parsed_root,
            normalized_task,
            repo_name=repo_name,
            limit=limit,
        )
        bundles = []
        for repo_plan in plan["plans"]:
            current_repo = repo_plan["repo"]
            graph_backend = get_graph_backend(str(graph_root.resolve()), current_repo)
            metadata_store = get_metadata_store(
                str(parsed_root.resolve()),
                current_repo,
            )
            context = retrieve_context(
                search_root,
                graph_root,
                parsed_root,
                current_repo,
                normalized_task,
                limit=limit,
                use_summaries=True,
                selective_retrieval=True,
            )
            project_summary = metadata_store.get_summary_by_id(stable_id("sum", current_repo, "project")) or {}
            selected_context = context["selected_context"]
            symbol_candidates = context["symbol_results"] or [item for item in selected_context if item.get("symbol_id")]
            file_candidates = [item for item in selected_context if item.get("path")]

            graph_neighborhoods = []
            statement_slices = []
            for candidate in symbol_candidates[: min(3, limit)]:
                query_text = str(candidate.get("qualified_name") or candidate.get("name") or "")
                if not query_text:
                    continue
                graph_response = graph_backend.execute(
                    {
                        "operation": "neighbors",
                        "seed": {"symbol_id": candidate["symbol_id"]},
                        "edge_types": recommended_edge_types(repo_plan["intent"]),
                        "direction": "both",
                        "depth": 1,
                        "limit": 8,
                    },
                )
                graph_neighborhoods.append(
                    {
                        "seed": query_text,
                        "results": (graph_response or {}).get("results", []),
                    }
                )
                statement_response = graph_backend.execute(
                    {
                        "operation": "statement_slice",
                        "seed": {"symbol_id": candidate["symbol_id"]},
                        "limit": 8,
                        "window": 8,
                    },
                )
                if statement_response and statement_response.get("results"):
                    statement_slices.append(
                        {
                            "seed": query_text,
                            "results": statement_response.get("results", []),
                        }
                    )

            evidence = []
            seen_evidence = set()
            for item in selected_context:
                key = (item.get("symbol_id"), item.get("path"), item.get("kind"))
                if key in seen_evidence:
                    continue
                seen_evidence.add(key)
                evidence.append(
                    {
                        "kind": item.get("kind"),
                        "path": item.get("path"),
                        "qualified_name": item.get("qualified_name"),
                        "symbol_id": item.get("symbol_id"),
                        "title": item.get("title"),
                        "preview": item.get("preview"),
                        "reasons": list(item.get("reasons", [])),
                        "provenance": build_provenance(item),
                        "why_included": explain_candidate(item),
                    }
                )

            bundles.append(
                {
                    "repo": current_repo,
                    "intent": repo_plan["intent"],
                    "focus": str(project_summary.get("focus") or ""),
                    "project_summary": str(project_summary.get("summary") or ""),
                    "retrieval_recipe": repo_plan["retrieval_recipe"],
                    "stage_context": {
                        "summary": context["summary_results"],
                        "symbol": context["symbol_results"],
                        "graph": context["graph_results"],
                        "body": context["body_results"],
                    },
                    "selected_context": selected_context,
                    "top_symbols": compact_candidates(symbol_candidates, limit=5),
                    "top_files": compact_candidates(file_candidates, limit=5),
                    "graph_neighborhoods": graph_neighborhoods,
                    "statement_slices": statement_slices,
                    "relevant_summaries": select_relevant_summaries_from_store(
                        project_summary,
                        metadata_store,
                        selected_context,
                    ),
                    "evidence": evidence[:limit],
                    "bundle_summary": summarize_bundle(selected_context, graph_neighborhoods, statement_slices, evidence),
                }
            )

        return {
            "task": task,
            "normalized_task": normalized_task,
            "refinement_hints": list(refinement_hints),
            "query_profile": plan["query_profile"],
            "bundles": bundles,
        }


def retrieve_iterative(
    search_root: Path,
    graph_root: Path,
    parsed_root: Path,
    task: str,
    *,
    repo_name: Optional[str] = None,
    limit: int = 8,
    prior_bundle: Optional[Dict[str, object]] = None,
    refinement_hints: Sequence[str] = (),
) -> Dict[str, object]:
    prior_hints = extract_prior_hints(prior_bundle)
    combined_hints = [*prior_hints, *refinement_hints]
    bundle = prepare_answer_bundle(
        search_root,
        graph_root,
        parsed_root,
        task,
        repo_name=repo_name,
        limit=limit,
        refinement_hints=combined_hints,
    )
    bundle["iteration"] = {
        "prior_hints": prior_hints,
        "applied_hints": combined_hints,
        "iteration_count": int((prior_bundle or {}).get("iteration", {}).get("iteration_count") or 0) + 1,
    }
    return bundle


def choose_recipe(query_profile: Dict[str, object]) -> List[str]:
    intent = str(query_profile.get("intent") or "exploration")
    recipe = ["summary_rollups", "symbol_localization"]
    if intent != "docs":
        recipe.append("graph_neighbors")
        recipe.append("body_hydration")
    recipe.append("answer_bundle")
    return recipe


def candidate_seed_types(query_profile: Dict[str, object]) -> List[str]:
    intent = str(query_profile.get("intent") or "exploration")
    if intent == "docs":
        return ["summary", "file", "doc"]
    if intent == "symbol":
        return ["summary", "symbol", "body"]
    return ["summary", "symbol", "graph"]


def stopping_criteria(query_profile: Dict[str, object], limit: int) -> Dict[str, object]:
    intent = str(query_profile.get("intent") or "exploration")
    return {
        "max_selected_context": limit,
        "stop_on_exact_symbol_hit": intent == "symbol",
        "stop_on_path_hit": intent == "docs",
        "max_graph_expansions": 16 if intent == "symbol" else 32,
    }


def recommended_edge_types(intent: str) -> List[str]:
    if intent == "symbol":
        return ["CALLS", "READS", "WRITES", "REFS", "USES"]
    if intent == "docs":
        return ["CONTAINS", "IMPORTS"]
    return ["CALLS", "IMPORTS", "IMPLEMENTS", "USES", "CONTAINS"]


def normalize_task(task: str, refinement_hints: Sequence[str]) -> str:
    hints = [hint.strip() for hint in refinement_hints if hint and hint.strip()]
    if not hints:
        return task
    return f"{task} {' '.join(hints)}".strip()


def build_provenance(item: Dict[str, object]) -> Dict[str, object]:
    return {
        "path": item.get("path"),
        "symbol_id": item.get("symbol_id"),
        "qualified_name": item.get("qualified_name"),
        "kind": item.get("kind"),
    }


def explain_candidate(item: Dict[str, object]) -> str:
    reasons = list(item.get("reasons", []))
    if not reasons:
        return "Included as a high-ranking retrieval candidate."
    return f"Included because of {', '.join(reasons)}."


def compact_candidates(candidates: Iterable[Dict[str, object]], *, limit: int) -> List[Dict[str, object]]:
    values = []
    for item in candidates:
        values.append(
            {
                "kind": item.get("kind"),
                "path": item.get("path"),
                "name": item.get("name"),
                "qualified_name": item.get("qualified_name"),
                "symbol_id": item.get("symbol_id"),
                "preview": item.get("preview"),
                "score": item.get("score"),
                "reasons": list(item.get("reasons", [])),
            }
        )
    return values[:limit]


def select_relevant_summaries(
    summaries: Dict[str, object],
    selected_context: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    selected_paths = {str(item.get("path")) for item in selected_context if item.get("path")}
    selected_symbol_ids = {str(item.get("symbol_id")) for item in selected_context if item.get("symbol_id")}
    selected_crates = {
        str(item.get("metadata", {}).get("crate") or "")
        for item in selected_context
        if item.get("metadata", {}).get("crate")
    }
    return {
        "project": summaries["project"],
        "packages": [
            item
            for item in summaries.get("packages", [])
            if str(item.get("package_name") or "") in selected_crates
        ][:5],
        "directories": [
            item
            for item in summaries["directories"]
            if any(path == item["path"] or path.startswith(f"{item['path']}/") for path in selected_paths)
        ][:5],
        "files": [item for item in summaries["files"] if item["path"] in selected_paths][:5],
        "symbols": [item for item in summaries["symbols"] if item["symbol_id"] in selected_symbol_ids][:5],
    }


def select_relevant_summaries_from_store(
    project_summary: Dict[str, object],
    metadata_store: object,
    selected_context: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    selected_paths = [str(item.get("path")) for item in selected_context if item.get("path")]
    selected_symbol_ids = [str(item.get("symbol_id")) for item in selected_context if item.get("symbol_id")]
    files: List[Dict[str, object]] = []
    symbols: List[Dict[str, object]] = []
    seen = set()

    for path in selected_paths:
        for summary in metadata_store.get_summary_by_path(path):
            key = str(summary.get("summary_id") or ("path", path))
            if key in seen:
                continue
            seen.add(key)
            files.append(summary)
            if len(files) >= 5:
                break
        if len(files) >= 5:
            break

    for symbol_id in selected_symbol_ids:
        for summary in metadata_store.get_summary_by_symbol(symbol_id):
            key = str(summary.get("summary_id") or ("symbol", symbol_id))
            if key in seen:
                continue
            seen.add(key)
            symbols.append(summary)
            if len(symbols) >= 5:
                break
        if len(symbols) >= 5:
            break

    return {
        "project": project_summary,
        "packages": [],
        "directories": [],
        "files": files,
        "symbols": symbols,
    }


def summarize_bundle(
    selected_context: Sequence[Dict[str, object]],
    graph_neighborhoods: Sequence[Dict[str, object]],
    statement_slices: Sequence[Dict[str, object]],
    evidence: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    return {
        "selected_context": len(selected_context),
        "graph_neighborhoods": len(graph_neighborhoods),
        "statement_slices": len(statement_slices),
        "evidence_items": len(evidence),
    }


def extract_prior_hints(prior_bundle: Optional[Dict[str, object]]) -> List[str]:
    if not prior_bundle:
        return []
    hints = []
    for bundle in prior_bundle.get("bundles", []):
        for item in bundle.get("top_symbols", [])[:2]:
            if item.get("qualified_name"):
                hints.append(str(item["qualified_name"]))
            elif item.get("name"):
                hints.append(str(item["name"]))
    deduped = []
    seen = set()
    for hint in hints:
        if hint in seen:
            continue
        seen.add(hint)
        deduped.append(hint)
    return deduped
