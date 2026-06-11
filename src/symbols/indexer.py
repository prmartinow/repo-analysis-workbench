from __future__ import annotations

import hashlib
import json
import re
import resource
import subprocess
import sys
import time
import tomllib
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

from common.inventory import is_generated_path
from parsers.rust import (
    ParsedRustFile,
    RustImport,
    RustSymbol,
    TextSpan,
    clean_rust_source_lines,
    parse_rust_file,
)
from parsers.rust_analyzer_backend import aggregate_rust_analyzer_probes, probe_rust_analyzer
from parsers.rustc_backend import aggregate_rustc_probes, probe_rust_ast
from parsers.tree_sitter_backend import aggregate_tree_sitter_probes, probe_tree_sitter


CALL_EXPR_RE = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"(?P<expr>(?:crate|super|Self|[A-Za-z_][A-Za-z0-9_]*)(?:::[A-Za-z_][A-Za-z0-9_]*)*)"
    r"\s*(?:::<[^()\n]*>)?\s*\("
)
FIELD_RE = re.compile(
    r"^\s*(?P<vis>pub(?:\([^)]*\))?\s+)?(?P<name>[a-z_][A-Za-z0-9_]*)\s*:\s*.+?(?:,\s*)?$"
)
GENERIC_ANGLE_RE = re.compile(r"<[^<>]*>")
LET_RE = re.compile(r"\blet\s+(?:mut\s+)?(?P<name>[a-z_][A-Za-z0-9_]*)\b")
PACKAGE_NAME_RE = re.compile(r'^\s*name\s*=\s*"([^"]+)"\s*$', re.M)
PACKAGE_BLOCK_RE = re.compile(r"\[package\](.*?)(?:\n\[|$)", re.S)
ASSIGN_RE = re.compile(r"^\s*(?P<lhs>[A-Za-z_][A-Za-z0-9_]*)\s*(?P<op>(?:[+\-*/%&|^]|<<|>>)?=)")
LEADING_CLOSE_RE = re.compile(r"^\s*(?P<braces>\}+)")
PATH_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"(?P<expr>(?:crate|super|Self|[A-Za-z_][A-Za-z0-9_]*)(?:::[A-Za-z_][A-Za-z0-9_]*)*)"
    r"(?![A-Za-z0-9_])"
)
IDENT_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_])(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?![A-Za-z0-9_])")
QUALIFIED_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"(?P<expr>(?:crate|super|Self|[A-Za-z_][A-Za-z0-9_]*)(?:::[A-Za-z_][A-Za-z0-9_]*)+)"
    r"(?![A-Za-z0-9_])"
)
METHOD_CALL_RE = re.compile(
    r"(?P<receiver>(?:self|Self|[A-Za-z_][A-Za-z0-9_]*))\s*\.\s*(?P<method>[A-Za-z_][A-Za-z0-9_]*)\s*(?:::<[^()\n]*>)?\s*\("
)
FIELD_ACCESS_RE = re.compile(
    r"(?P<receiver>(?:self|Self|[A-Za-z_][A-Za-z0-9_]*))\s*\.\s*(?P<field>[a-z_][A-Za-z0-9_]*)"
)
VARIANT_RE = re.compile(
    r"^\s*(?P<name>[A-Z][A-Za-z0-9_]*)\b(?:\s*(?:\(|\{|=|,|$).*)?$"
)

FUNCTION_LIKE_KINDS = {"function", "method"}
CACHE_SCHEMA_VERSION = "rust-file-cache-v1"
KEYWORDS = {
    "Self",
    "as",
    "async",
    "await",
    "break",
    "const",
    "continue",
    "crate",
    "dyn",
    "else",
    "enum",
    "extern",
    "false",
    "fn",
    "for",
    "if",
    "impl",
    "in",
    "let",
    "loop",
    "match",
    "mod",
    "move",
    "mut",
    "pub",
    "ref",
    "return",
    "self",
    "static",
    "struct",
    "super",
    "trait",
    "true",
    "type",
    "union",
    "unsafe",
    "use",
    "where",
    "while",
}
PRIMITIVE_TYPES = {
    "bool",
    "char",
    "f32",
    "f64",
    "i8",
    "i16",
    "i32",
    "i64",
    "i128",
    "isize",
    "str",
    "u8",
    "u16",
    "u32",
    "u64",
    "u128",
    "usize",
}


@dataclass
class ParsedFileContext:
    source_path: Path
    package_info: CargoPackageInfo
    workspace_index: WorkspaceIndex
    parsed: ParsedRustFile
    source: str
    source_lines: List[str]
    cleaned_lines: List[str]
    crate_root: str
    dependency_aliases: Dict[str, str]
    symbol_id_by_local: Dict[int, str]
    import_aliases: Dict[str, List[str]]
    compiler_probe: Dict[str, object]
    backend_probes: Dict[str, Dict[str, object]]
    primary_parser_backend: str
    parse_elapsed_ms: float
    cache_hit: bool = False


@dataclass(frozen=True)
class CargoPackageInfo:
    root: Path
    manifest_path: Path
    package_name: str
    crate_name: str
    crate_module: str
    dependency_aliases: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkspaceIndex:
    packages_by_root: Dict[Path, CargoPackageInfo]
    packages_by_name: Dict[str, CargoPackageInfo]
    packages_by_module: Dict[str, CargoPackageInfo]


def build_symbol_index(
    repo_name: str,
    repo_root: Path,
    raw_root: Path,
    path_prefixes: Sequence[str] = (),
    progress_callback: Callable[[Dict[str, object]], None] | None = None,
    cache_root: Path | None = None,
) -> Dict[str, object]:
    manifest = load_raw_manifest(raw_root, repo_name)
    parser_roots = manifest.get("parser_relevant_source_roots", [])
    workspace_index = build_workspace_index(repo_root)
    rust_files = discover_rust_files(repo_root, parser_roots, normalize_prefixes(path_prefixes))
    run_started = time.perf_counter()
    contexts: List[ParsedFileContext] = []
    backend_failures: DefaultDict[str, int] = defaultdict(int)
    slowest_files: List[Dict[str, object]] = []

    def emit_stage_progress(stage: str, **extra: object) -> None:
        if progress_callback is None:
            return
        progress_callback(
            {
                "event": "stage_progress",
                "repo": repo_name,
                "stage": stage,
                "elapsed_ms": round((time.perf_counter() - run_started) * 1000, 3),
                "rss_mb": current_rss_mb(),
                "contexts": len(contexts),
                **extra,
            }
        )

    if progress_callback is not None:
        progress_callback(
            {
                "event": "repo_scan_started",
                "repo": repo_name,
                "parser_roots": len(parser_roots),
                "rust_files_total": len(rust_files),
                "path_prefixes": list(normalize_prefixes(path_prefixes)),
                "rss_mb": current_rss_mb(),
                "elapsed_ms": 0.0,
            }
        )

    for index, path in enumerate(rust_files, start=1):
        file_started = time.perf_counter()
        context = load_or_parse_rust_source_file(repo_root, path, workspace_index, cache_root)
        contexts.append(context)
        for backend_name, probe in context.backend_probes.items():
            if probe.get("used") and not probe.get("parsed"):
                backend_failures[backend_name] += 1
        if context.compiler_probe.get("used") and not context.compiler_probe.get("parsed"):
            backend_failures["rustc_ast_probe"] += 1

        file_elapsed_ms = round((time.perf_counter() - file_started) * 1000, 3)
        slowest_files.append(
            {
                "path": context.parsed.path,
                "elapsed_ms": file_elapsed_ms,
                "primary_parser_backend": context.primary_parser_backend,
            }
        )
        slowest_files.sort(key=lambda item: item["elapsed_ms"], reverse=True)
        del slowest_files[10:]

        if progress_callback is not None:
            progress_callback(
                {
                    "event": "file_parsed",
                    "repo": repo_name,
                    "index": index,
                    "total": len(rust_files),
                    "path": context.parsed.path,
                    "crate": context.parsed.crate_name,
                    "module_path": context.parsed.module_path,
                    "elapsed_ms": round((time.perf_counter() - run_started) * 1000, 3),
                    "file_elapsed_ms": file_elapsed_ms,
                    "rss_mb": current_rss_mb(),
                    "primary_parser_backend": context.primary_parser_backend,
                    "cache_hit": context.cache_hit,
                    "backend_failures": dict(sorted(backend_failures.items())),
                    "slowest_files": list(slowest_files),
                }
            )

    emit_stage_progress("building_symbol_records")
    file_records: List[Dict[str, object]] = []
    symbol_records: List[Dict[str, object]] = []
    context_by_path = {context.parsed.path: context for context in contexts}

    for context in contexts:
        context.symbol_id_by_local = {
            symbol.local_id: symbol_stable_id(repo_name, context.parsed.path, symbol)
            for symbol in context.parsed.symbols
        }
        file_records.append(
            {
                "path": context.parsed.path,
                "crate": context.parsed.crate_name,
                "package_name": context.package_info.package_name,
                "module_path": context.parsed.module_path,
                "language": "Rust",
                "symbols": len(context.parsed.symbols),
                "imports": len(context.parsed.imports),
                "primary_parser_backend": context.primary_parser_backend,
                "content_hash": hashlib.sha1(context.source.encode("utf-8")).hexdigest(),
            }
        )
        for symbol in context.parsed.symbols:
            symbol_records.append(symbol_to_record(repo_name, context, symbol))

    emit_stage_progress("building_resolution_index", symbols=len(symbol_records))
    resolution_index = build_resolution_index(symbol_records)
    emit_stage_progress("building_import_records")
    import_records = build_import_records(repo_name, contexts, resolution_index)
    emit_stage_progress("resolving_impls", imports=len(import_records))
    resolve_impl_symbols(symbol_records, context_by_path, resolution_index)
    emit_stage_progress("resolving_trait_inheritance")
    resolve_trait_inheritance(symbol_records, context_by_path, resolution_index)
    emit_stage_progress("building_reference_records")
    reference_records = build_reference_records(repo_name, contexts, resolution_index)
    emit_stage_progress("building_statement_records", references=len(reference_records))
    statement_records = build_statement_records(repo_name, contexts, symbol_records, resolution_index)
    emit_stage_progress("enriching_symbol_semantics", statements=len(statement_records))
    enrich_symbol_semantics(symbol_records, reference_records, statement_records, resolution_index)
    emit_stage_progress("enriching_symbol_artifact_metadata")
    enrich_symbol_artifact_metadata(symbol_records, statement_records)
    emit_stage_progress("checking_duplicate_symbol_ids")
    duplicate_symbol_ids = find_duplicate_ids(symbol_records, "symbol_id")
    if duplicate_symbol_ids:
        raise ValueError(
            "Duplicate symbol_id values detected: "
            + ", ".join(sorted(duplicate_symbol_ids[:10]))
            + (" ..." if len(duplicate_symbol_ids) > 10 else "")
        )
    emit_stage_progress("aggregating_backend_probes")
    compiler_backends = {
        "rustc_ast_probe": aggregate_rustc_probes([context.compiler_probe for context in contexts]),
        "tree_sitter_rust": aggregate_tree_sitter_probes(
            [context.backend_probes["tree_sitter_rust"] for context in contexts]
        ),
        "rust_analyzer_lsp": aggregate_rust_analyzer_probes(
            [context.backend_probes["rust_analyzer_lsp"] for context in contexts]
        ),
    }

    emit_stage_progress("rolling_up_summary_counts")
    kind_counts = rollup_counts(item["kind"] for item in symbol_records)
    reference_kind_counts = rollup_counts(item["kind"] for item in reference_records)
    statement_kind_counts = rollup_counts(item["kind"] for item in statement_records)

    return {
        "schema_version": "0.6.0",
        "repo": repo_name,
        "generated_at": timestamp_now(),
        "parser": "rust-backend-fused-v2",
        "primary_parser_backends": rollup_counts(context.primary_parser_backend for context in contexts),
        "parser_backends": compiler_backends,
        "source_roots": parser_roots,
        "path_prefixes": list(normalize_prefixes(path_prefixes)),
        "files": file_records,
        "symbols": symbol_records,
        "imports": import_records,
        "references": reference_records,
        "statements": statement_records,
        "summary": {
            "rust_files": len(file_records),
            "symbols": len(symbol_records),
            "imports": len(import_records),
            "references": len(reference_records),
            "statements": len(statement_records),
            "tests": sum(1 for item in symbol_records if item.get("is_test")),
            "kind_counts": kind_counts,
            "reference_kind_counts": reference_kind_counts,
            "statement_kind_counts": statement_kind_counts,
        },
        "build_metrics": {
            "elapsed_ms": round((time.perf_counter() - run_started) * 1000, 3),
            "rss_mb": current_rss_mb(),
            "cache_hits": sum(1 for context in contexts if context.cache_hit),
            "cache_misses": sum(1 for context in contexts if not context.cache_hit),
            "backend_failures": dict(sorted(backend_failures.items())),
            "slowest_files": list(slowest_files),
        },
    }


def build_import_records(
    repo_name: str,
    contexts: Sequence[ParsedFileContext],
    resolution_index: Dict[str, object],
) -> List[Dict[str, object]]:
    import_records: List[Dict[str, object]] = []

    for context in contexts:
        context.import_aliases = {}
        for rust_import in context.parsed.imports:
            expanded_targets = expand_use_targets(rust_import.path)
            if not expanded_targets:
                expanded_targets = [(rust_import.path, None)]

            for expanded_target, alias in expanded_targets:
                normalized_target = normalize_path_expression(
                    expanded_target,
                    context.parsed.module_path,
                    context.crate_root,
                    context.dependency_aliases,
                    None,
                )
                alias_name = alias or normalized_target.split("::")[-1]
                if alias_name:
                    context.import_aliases.setdefault(alias_name, [])
                    if normalized_target not in context.import_aliases[alias_name]:
                        context.import_aliases[alias_name].append(normalized_target)

                resolved_symbol = resolve_expression(
                    expanded_target,
                    {
                        "crate": context.parsed.crate_name,
                        "module_path": context.parsed.module_path,
                        "path": context.parsed.path,
                        "container_qualified_name": rust_import.container_qualified_name,
                    },
                    resolution_index,
                    context.import_aliases,
                    context.dependency_aliases,
                    context.crate_root,
                    None,
                    None,
                )

                import_records.append(
                    {
                        "import_id": stable_id(
                            "imp",
                            repo_name,
                            context.parsed.path,
                            expanded_target,
                            str(rust_import.span.start_line),
                            str(rust_import.span.start_column),
                            alias or "",
                        ),
                        "repo": repo_name,
                        "path": context.parsed.path,
                        "crate": context.parsed.crate_name,
                        "module_path": rust_import.module_path,
                        "language": "Rust",
                        "visibility": rust_import.visibility,
                        "signature": rust_import.signature,
                        "raw_target": rust_import.path,
                        "target": expanded_target,
                        "normalized_target": normalized_target,
                        "alias": alias,
                        "span": span_to_dict(rust_import.span),
                        "container_symbol_id": context.symbol_id_by_local.get(rust_import.container_local_id),
                        "container_qualified_name": rust_import.container_qualified_name,
                        "target_symbol_id": resolved_symbol["target_symbol_id"],
                        "target_qualified_name": resolved_symbol["target_qualified_name"],
                        "target_kind": resolved_symbol["target_kind"],
                    }
                )

    return import_records


def build_reference_records(
    repo_name: str,
    contexts: Sequence[ParsedFileContext],
    resolution_index: Dict[str, object],
) -> List[Dict[str, object]]:
    reference_records: Dict[str, Dict[str, object]] = {}

    for context in contexts:
        symbol_records = {
            context.symbol_id_by_local[symbol.local_id]: symbol_to_record(repo_name, context, symbol)
            for symbol in context.parsed.symbols
        }
        for symbol in symbol_records.values():
            if symbol["kind"] not in FUNCTION_LIKE_KINDS and symbol["kind"] not in {"struct", "enum", "trait", "field"}:
                continue

            self_target = infer_self_target(symbol, resolution_index)

            for candidate, line_number, column in extract_signature_reference_candidates(
                symbol["signature"],
                symbol["name"],
                symbol["span"]["start_line"],
            ):
                resolved = resolve_expression(
                    candidate,
                    symbol,
                    resolution_index,
                    context.import_aliases,
                    context.dependency_aliases,
                    context.crate_root,
                    symbol["symbol_id"],
                    self_target,
                )
                add_reference_record(
                    reference_records,
                    repo_name,
                    symbol,
                    kind="use",
                    candidate=candidate,
                    line_number=line_number,
                    column=column,
                    resolved=resolved,
                )

            if symbol["kind"] not in FUNCTION_LIKE_KINDS:
                continue

            call_positions = set()
            for candidate, line_number, column in extract_body_call_candidates(
                context.cleaned_lines,
                symbol["span"],
            ):
                call_positions.add((candidate, line_number, column))
                resolved = resolve_expression(
                    candidate,
                    symbol,
                    resolution_index,
                    context.import_aliases,
                    context.dependency_aliases,
                    context.crate_root,
                    symbol["symbol_id"],
                    self_target,
                )
                add_reference_record(
                    reference_records,
                    repo_name,
                    symbol,
                    kind="call",
                    candidate=candidate,
                    line_number=line_number,
                    column=column,
                    resolved=resolved,
                )

            for candidate, line_number, column in extract_body_member_call_candidates(
                context.cleaned_lines,
                symbol["span"],
                symbol,
                resolution_index,
                context.import_aliases,
                context.dependency_aliases,
                context.crate_root,
                self_target,
            ):
                call_positions.add((candidate, line_number, column))
                resolved = resolve_expression(
                    candidate,
                    symbol,
                    resolution_index,
                    context.import_aliases,
                    context.dependency_aliases,
                    context.crate_root,
                    symbol["symbol_id"],
                    self_target,
                )
                add_reference_record(
                    reference_records,
                    repo_name,
                    symbol,
                    kind="call",
                    candidate=candidate,
                    line_number=line_number,
                    column=column,
                    resolved=resolved,
                )

            for candidate, line_number, column in extract_body_use_candidates(
                context.cleaned_lines,
                symbol["span"],
            ):
                if (candidate, line_number, column) in call_positions:
                    continue
                resolved = resolve_expression(
                    candidate,
                    symbol,
                    resolution_index,
                    context.import_aliases,
                    context.dependency_aliases,
                    context.crate_root,
                    symbol["symbol_id"],
                    self_target,
                )
                add_reference_record(
                    reference_records,
                    repo_name,
                    symbol,
                    kind="use",
                    candidate=candidate,
                    line_number=line_number,
                    column=column,
                    resolved=resolved,
                )

            for candidate, line_number, column in extract_body_field_use_candidates(
                context.cleaned_lines,
                symbol["span"],
                symbol,
                resolution_index,
                context.import_aliases,
                context.dependency_aliases,
                context.crate_root,
                self_target,
            ):
                if (candidate, line_number, column) in call_positions:
                    continue
                resolved = resolve_expression(
                    candidate,
                    symbol,
                    resolution_index,
                    context.import_aliases,
                    context.dependency_aliases,
                    context.crate_root,
                    symbol["symbol_id"],
                    self_target,
                )
                add_reference_record(
                    reference_records,
                    repo_name,
                    symbol,
                    kind="use",
                    candidate=candidate,
                    line_number=line_number,
                    column=column,
                    resolved=resolved,
                )

    return sorted(reference_records.values(), key=lambda item: (item["path"], item["span"]["start_line"], item["kind"], item["name"]))


def build_resolution_index(symbol_records: Sequence[Dict[str, object]]) -> Dict[str, object]:
    by_id: Dict[str, Dict[str, object]] = {}
    by_qname: Dict[str, Dict[str, object]] = {}
    by_name: DefaultDict[str, List[Dict[str, object]]] = defaultdict(list)
    locals_by_scope: DefaultDict[str, DefaultDict[str, List[Dict[str, object]]]] = defaultdict(lambda: defaultdict(list))

    for symbol in symbol_records:
        by_id[symbol["symbol_id"]] = symbol
        by_qname.setdefault(symbol["qualified_name"], symbol)
        by_name[symbol["name"]].append(symbol)
        if symbol["kind"] == "local" and symbol["scope_symbol_id"]:
            locals_by_scope[symbol["scope_symbol_id"]][symbol["name"]].append(symbol)

    return {
        "by_id": by_id,
        "by_qname": by_qname,
        "by_name": by_name,
        "locals_by_scope": locals_by_scope,
    }


def resolve_impl_symbols(
    symbol_records: Sequence[Dict[str, object]],
    context_by_path: Dict[str, ParsedFileContext],
    resolution_index: Dict[str, object],
) -> None:
    for symbol in symbol_records:
        if symbol["kind"] != "impl":
            continue

        context = context_by_path[symbol["path"]]
        impl_target = resolve_expression(
            symbol["impl_target"],
            symbol,
            resolution_index,
            context.import_aliases,
            context.dependency_aliases,
            context.crate_root,
            symbol["symbol_id"],
            None,
        )
        impl_trait = resolve_expression(
            symbol["impl_trait"],
            symbol,
            resolution_index,
            context.import_aliases,
            context.dependency_aliases,
            context.crate_root,
            symbol["symbol_id"],
            None,
        )

        symbol["resolved_impl_target_symbol_id"] = impl_target["target_symbol_id"]
        symbol["resolved_impl_target_qualified_name"] = impl_target["target_qualified_name"]
        symbol["resolved_impl_trait_symbol_id"] = impl_trait["target_symbol_id"]
        symbol["resolved_impl_trait_qualified_name"] = impl_trait["target_qualified_name"]


def enrich_symbol_semantics(
    symbol_records: Sequence[Dict[str, object]],
    reference_records: Sequence[Dict[str, object]],
    statement_records: Sequence[Dict[str, object]],
    resolution_index: Dict[str, object],
) -> None:
    calls_by_symbol: DefaultDict[str, List[Dict[str, object]]] = defaultdict(list)
    reads_by_symbol: DefaultDict[str, List[Dict[str, object]]] = defaultdict(list)
    writes_by_symbol: DefaultDict[str, List[Dict[str, object]]] = defaultdict(list)
    refs_by_symbol: DefaultDict[str, List[Dict[str, object]]] = defaultdict(list)

    for reference in reference_records:
        container_symbol_id = reference["container_symbol_id"]
        target = make_target_entry(
            reference["qualified_name_hint"],
            {
                "target_symbol_id": reference["target_symbol_id"],
                "target_qualified_name": reference["target_qualified_name"],
                "target_kind": reference["target_kind"],
                "qualified_name_hint": reference["qualified_name_hint"],
            },
        )
        append_unique_target(refs_by_symbol[container_symbol_id], target)
        if reference["kind"] == "call":
            append_unique_target(calls_by_symbol[container_symbol_id], target)

    for statement in statement_records:
        container_symbol_id = statement["container_symbol_id"]
        for target in statement.get("reads", []):
            append_unique_target(reads_by_symbol[container_symbol_id], dict(target))
            append_unique_target(refs_by_symbol[container_symbol_id], dict(target))
        for target in statement.get("writes", []):
            append_unique_target(writes_by_symbol[container_symbol_id], dict(target))
            append_unique_target(refs_by_symbol[container_symbol_id], dict(target))

    for symbol in symbol_records:
        if symbol["kind"] not in FUNCTION_LIKE_KINDS:
            symbol["semantic_summary"] = {
                "direct_calls": [],
                "transitive_calls": [],
                "reads": [],
                "writes": [],
                "references": [],
                "interprocedural_reads": [],
                "interprocedural_writes": [],
                "interprocedural_references": [],
            }
            continue

        direct_calls = list(calls_by_symbol.get(symbol["symbol_id"], []))
        symbol["semantic_summary"] = {
            "direct_calls": direct_calls,
            "transitive_calls": [],
            "reads": list(reads_by_symbol.get(symbol["symbol_id"], [])),
            "writes": list(writes_by_symbol.get(symbol["symbol_id"], [])),
            "references": list(refs_by_symbol.get(symbol["symbol_id"], [])),
            "interprocedural_reads": [],
            "interprocedural_writes": [],
            "interprocedural_references": [],
        }

    for symbol in symbol_records:
        if symbol["kind"] not in FUNCTION_LIKE_KINDS:
            continue
        symbol["semantic_summary"]["transitive_calls"] = compute_transitive_calls(
            symbol["semantic_summary"]["direct_calls"],
            resolution_index,
        )
        symbol["semantic_summary"]["interprocedural_reads"] = compute_interprocedural_targets(
            symbol["semantic_summary"]["direct_calls"],
            resolution_index,
            field_name="reads",
        )
        symbol["semantic_summary"]["interprocedural_writes"] = compute_interprocedural_targets(
            symbol["semantic_summary"]["direct_calls"],
            resolution_index,
            field_name="writes",
        )
        symbol["semantic_summary"]["interprocedural_references"] = compute_interprocedural_targets(
            symbol["semantic_summary"]["direct_calls"],
            resolution_index,
            field_name="references",
        )


def compute_transitive_calls(
    direct_calls: Sequence[Dict[str, object]],
    resolution_index: Dict[str, object],
    *,
    max_depth: int = 2,
) -> List[Dict[str, object]]:
    transitive: List[Dict[str, object]] = []
    seen = set()
    frontier = [dict(item) for item in direct_calls]
    depth = 0

    while frontier and depth < max_depth:
        next_frontier: List[Dict[str, object]] = []
        for item in frontier:
            target_symbol_id = item.get("target_symbol_id")
            if not target_symbol_id or target_symbol_id in seen:
                continue
            seen.add(target_symbol_id)
            target_symbol = resolution_index["by_id"].get(target_symbol_id)
            if not target_symbol:
                continue
            semantic_summary = target_symbol.get("semantic_summary") or {}
            for nested in semantic_summary.get("direct_calls", []):
                nested_copy = dict(nested)
                append_unique_target(transitive, nested_copy)
                next_frontier.append(nested_copy)
        frontier = next_frontier
        depth += 1

    return transitive


def compute_interprocedural_targets(
    direct_calls: Sequence[Dict[str, object]],
    resolution_index: Dict[str, object],
    *,
    field_name: str,
    max_depth: int = 3,
) -> List[Dict[str, object]]:
    aggregated: List[Dict[str, object]] = []
    seen_symbols = set()
    frontier = [item.get("target_symbol_id") for item in direct_calls if item.get("target_symbol_id")]
    depth = 0

    while frontier and depth < max_depth:
        next_frontier: List[str] = []
        for target_symbol_id in frontier:
            if not target_symbol_id or target_symbol_id in seen_symbols:
                continue
            seen_symbols.add(target_symbol_id)
            target_symbol = resolution_index["by_id"].get(target_symbol_id)
            if not target_symbol:
                continue
            semantic_summary = target_symbol.get("semantic_summary") or {}
            for target in semantic_summary.get(field_name, []):
                append_unique_target(aggregated, dict(target))
            for nested in semantic_summary.get("direct_calls", []):
                nested_symbol_id = nested.get("target_symbol_id")
                if nested_symbol_id and nested_symbol_id not in seen_symbols:
                    next_frontier.append(nested_symbol_id)
        frontier = next_frontier
        depth += 1

    return aggregated


def resolve_trait_inheritance(
    symbol_records: Sequence[Dict[str, object]],
    context_by_path: Dict[str, ParsedFileContext],
    resolution_index: Dict[str, object],
) -> None:
    for symbol in symbol_records:
        if symbol["kind"] != "trait":
            continue

        context = context_by_path[symbol["path"]]
        resolved_traits = []
        for parent in symbol.get("super_traits", []):
            resolved = resolve_expression(
                parent,
                symbol,
                resolution_index,
                context.import_aliases,
                context.dependency_aliases,
                context.crate_root,
                symbol["symbol_id"],
                None,
            )
            resolved_traits.append(
                {
                    "name": parent.split("::")[-1],
                    "qualified_name_hint": resolved["qualified_name_hint"] or parent,
                    "target_symbol_id": resolved["target_symbol_id"],
                    "target_qualified_name": resolved["target_qualified_name"] or parent,
                    "target_kind": resolved["target_kind"],
                }
            )
        symbol["resolved_super_traits"] = resolved_traits


def resolve_expression(
    expression: Optional[str],
    current_symbol: Dict[str, object],
    resolution_index: Dict[str, object],
    import_aliases: Dict[str, List[str]],
    dependency_aliases: Dict[str, str],
    crate_root: str,
    scope_symbol_id: Optional[str],
    self_target: Optional[str],
) -> Dict[str, Optional[str]]:
    if not expression:
        return empty_resolution()

    candidate = strip_expression_noise(expression)
    if not candidate or candidate in KEYWORDS or candidate in PRIMITIVE_TYPES:
        return empty_resolution()

    if scope_symbol_id and "::" not in candidate:
        local_matches = resolution_index["locals_by_scope"].get(scope_symbol_id, {}).get(candidate, [])
        if len(local_matches) == 1:
            local_symbol = local_matches[0]
            return {
                "target_symbol_id": local_symbol["symbol_id"],
                "target_qualified_name": local_symbol["qualified_name"],
                "target_kind": local_symbol["kind"],
                "qualified_name_hint": local_symbol["qualified_name"],
            }

    preferred_paths = expand_reference_candidates(
        candidate,
        current_symbol["module_path"],
        crate_root,
        import_aliases,
        dependency_aliases,
        self_target,
    )

    for preferred_path in preferred_paths:
        exact_match = resolution_index["by_qname"].get(preferred_path)
        if exact_match:
            return {
                "target_symbol_id": exact_match["symbol_id"],
                "target_qualified_name": exact_match["qualified_name"],
                "target_kind": exact_match["kind"],
                "qualified_name_hint": preferred_path,
            }

    simple_name = candidate.split("::")[-1]
    candidates = resolution_index["by_name"].get(simple_name, [])
    best_match = pick_best_symbol_candidate(
        candidates,
        preferred_paths,
        current_symbol,
        self_target,
        preferred_roots=preferred_resolution_roots(current_symbol, dependency_aliases, crate_root),
    )
    if best_match:
        return {
            "target_symbol_id": best_match["symbol_id"],
            "target_qualified_name": best_match["qualified_name"],
            "target_kind": best_match["kind"],
            "qualified_name_hint": best_match["qualified_name"],
        }

    return {
        "target_symbol_id": None,
        "target_qualified_name": preferred_paths[0] if preferred_paths else candidate,
        "target_kind": None,
        "qualified_name_hint": preferred_paths[0] if preferred_paths else candidate,
    }


def enrich_context_symbols(context: ParsedFileContext) -> None:
    normalize_method_qualified_names(context)
    extract_struct_fields(context)
    extract_enum_variants(context)
    extract_local_variables(context)


def normalize_method_qualified_names(context: ParsedFileContext) -> None:
    symbols_by_local = {symbol.local_id: symbol for symbol in context.parsed.symbols}

    for symbol in context.parsed.symbols:
        if symbol.kind != "method" or symbol.container_local_id is None:
            continue
        container = symbols_by_local.get(symbol.container_local_id)
        if not container:
            continue

        if container.kind == "impl" and container.impl_target:
            owner = normalize_path_expression(
                container.impl_target,
                symbol.module_path,
                context.crate_root,
                context.dependency_aliases,
                None,
            )
            if owner:
                symbol.qualified_name = f"{owner}::{symbol.name}"
        elif container.kind == "trait":
            symbol.qualified_name = f"{container.qualified_name}::{symbol.name}"


def extract_struct_fields(context: ParsedFileContext) -> None:
    next_local_id = next_symbol_local_id(context.parsed.symbols)
    for symbol in list(context.parsed.symbols):
        if symbol.kind != "struct":
            continue
        for line_number in range(symbol.span.start_line + 1, symbol.span.end_line):
            raw_line = context.source_lines[line_number - 1]
            cleaned_line = context.cleaned_lines[line_number - 1]
            stripped = cleaned_line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("//"):
                continue
            match = FIELD_RE.match(cleaned_line)
            if not match:
                continue
            field_name = match.group("name")
            context.parsed.symbols.append(
                RustSymbol(
                    local_id=next_local_id,
                    kind="field",
                    name=field_name,
                    qualified_name=f"{symbol.qualified_name}::{field_name}",
                    module_path=symbol.module_path,
                    span=TextSpan(
                        start_line=line_number,
                        start_column=raw_line.find(field_name) + 1,
                        end_line=line_number,
                        end_column=len(raw_line.rstrip()) + 1,
                    ),
                    signature=cleaned_line.strip(),
                    visibility=normalize_visibility(match.group("vis")),
                    docstring=None,
                    container_local_id=symbol.local_id,
                    container_qualified_name=symbol.qualified_name,
                    is_test=symbol.is_test,
                )
            )
            next_local_id += 1


def extract_enum_variants(context: ParsedFileContext) -> None:
    next_local_id = next_symbol_local_id(context.parsed.symbols)
    for symbol in list(context.parsed.symbols):
        if symbol.kind != "enum":
            continue
        for line_number in range(symbol.span.start_line + 1, symbol.span.end_line):
            raw_line = context.source_lines[line_number - 1]
            cleaned_line = context.cleaned_lines[line_number - 1]
            stripped = cleaned_line.strip()
            if not stripped or stripped.startswith("#") or stripped in {"{", "}"}:
                continue
            match = VARIANT_RE.match(cleaned_line)
            if not match:
                continue
            variant_name = match.group("name")
            context.parsed.symbols.append(
                RustSymbol(
                    local_id=next_local_id,
                    kind="variant",
                    name=variant_name,
                    qualified_name=f"{symbol.qualified_name}::{variant_name}",
                    module_path=symbol.module_path,
                    span=TextSpan(
                        start_line=line_number,
                        start_column=raw_line.find(variant_name) + 1,
                        end_line=line_number,
                        end_column=len(raw_line.rstrip()) + 1,
                    ),
                    signature=cleaned_line.strip().rstrip(","),
                    visibility="public",
                    docstring=None,
                    container_local_id=symbol.local_id,
                    container_qualified_name=symbol.qualified_name,
                    is_test=symbol.is_test,
                )
            )
            next_local_id += 1


def extract_local_variables(context: ParsedFileContext) -> None:
    next_local_id = next_symbol_local_id(context.parsed.symbols)
    for symbol in list(context.parsed.symbols):
        if symbol.kind not in FUNCTION_LIKE_KINDS:
            continue
        seen_locals: set[Tuple[int, str]] = set()
        for line_number in range(symbol.span.start_line, symbol.span.end_line + 1):
            raw_line = context.source_lines[line_number - 1]
            cleaned_line = context.cleaned_lines[line_number - 1]
            for match in LET_RE.finditer(cleaned_line):
                local_name = match.group("name")
                local_key = (line_number, local_name)
                if local_key in seen_locals:
                    continue
                seen_locals.add(local_key)
                context.parsed.symbols.append(
                    RustSymbol(
                        local_id=next_local_id,
                        kind="local",
                        name=local_name,
                        qualified_name=f"{symbol.qualified_name}::{local_name}@L{line_number}",
                        module_path=symbol.module_path,
                        span=TextSpan(
                            start_line=line_number,
                            start_column=match.start("name") + 1,
                            end_line=line_number,
                            end_column=match.end("name") + 1,
                        ),
                        signature=raw_line.strip(),
                        visibility="private",
                        docstring=None,
                        container_local_id=symbol.local_id,
                        container_qualified_name=symbol.qualified_name,
                        is_test=symbol.is_test,
                    )
                )
                next_local_id += 1


def add_reference_record(
    reference_records: Dict[str, Dict[str, object]],
    repo_name: str,
    symbol: Dict[str, object],
    kind: str,
    candidate: str,
    line_number: int,
    column: int,
    resolved: Dict[str, Optional[str]],
) -> None:
    qualified_name_hint = resolved["qualified_name_hint"] or candidate
    if qualified_name_hint == symbol["qualified_name"]:
        return

    reference_id = stable_id(
        "ref",
        repo_name,
        symbol["path"],
        symbol["symbol_id"],
        kind,
        qualified_name_hint,
        str(line_number),
    )
    reference_records[reference_id] = {
        "reference_id": reference_id,
        "repo": repo_name,
        "path": symbol["path"],
        "crate": symbol["crate"],
        "module_path": symbol["module_path"],
        "language": symbol["language"],
        "kind": kind,
        "name": candidate.split("::")[-1],
        "qualified_name_hint": qualified_name_hint,
        "span": {
            "start_line": line_number,
            "start_column": column,
            "end_line": line_number,
            "end_column": column + len(candidate),
        },
        "container_symbol_id": symbol["symbol_id"],
        "container_qualified_name": symbol["qualified_name"],
        "scope_symbol_id": symbol["symbol_id"] if symbol["kind"] in FUNCTION_LIKE_KINDS else symbol["scope_symbol_id"],
        "target_symbol_id": resolved["target_symbol_id"],
        "target_qualified_name": resolved["target_qualified_name"] or qualified_name_hint,
        "target_kind": resolved["target_kind"],
    }


def build_statement_records(
    repo_name: str,
    contexts: Sequence[ParsedFileContext],
    symbol_records: Sequence[Dict[str, object]],
    resolution_index: Dict[str, object],
) -> List[Dict[str, object]]:
    context_by_path = {context.parsed.path: context for context in contexts}
    symbol_records_by_id = {symbol["symbol_id"]: symbol for symbol in symbol_records}
    locals_by_scope_line_name: DefaultDict[str, Dict[Tuple[int, str], Dict[str, object]]] = defaultdict(dict)

    for symbol in symbol_records:
        if symbol["kind"] == "local" and symbol["scope_symbol_id"]:
            locals_by_scope_line_name[symbol["scope_symbol_id"]][(symbol["span"]["start_line"], symbol["name"])] = symbol

    statements: List[Dict[str, object]] = []
    for symbol in symbol_records:
        if symbol["kind"] not in FUNCTION_LIKE_KINDS:
            continue

        context = context_by_path[symbol["path"]]
        statements.extend(
            build_function_statement_records(
                repo_name,
                context,
                symbol,
                resolution_index,
                locals_by_scope_line_name.get(symbol["symbol_id"], {}),
                symbol_records_by_id,
            )
        )

    return sorted(
        statements,
        key=lambda item: (
            item["path"],
            item["span"]["start_line"],
            item["span"]["start_column"],
            item["statement_id"],
        ),
    )


def build_function_statement_records(
    repo_name: str,
    context: ParsedFileContext,
    symbol: Dict[str, object],
    resolution_index: Dict[str, object],
    locals_by_line_name: Dict[Tuple[int, str], Dict[str, object]],
    symbol_records_by_id: Dict[str, Dict[str, object]],
) -> List[Dict[str, object]]:
    statements: List[Dict[str, object]] = []
    body_depth = 1
    control_stack: List[Tuple[str, int]] = []
    previous_statement_id: Optional[str] = None
    self_target = infer_self_target(symbol, resolution_index)

    for line_number in range(symbol["span"]["start_line"], symbol["span"]["end_line"] + 1):
        fragment = statement_fragment_for_line(context.cleaned_lines[line_number - 1], symbol["span"], line_number)
        if fragment is None:
            continue

        leading_close_match = LEADING_CLOSE_RE.match(fragment)
        leading_closes = len(leading_close_match.group("braces")) if leading_close_match else 0
        effective_depth = max(body_depth - leading_closes, 1)
        while control_stack and effective_depth < control_stack[-1][1]:
            control_stack.pop()

        statement_text = collapse_whitespace(fragment).strip().rstrip(";")
        if not statement_text or statement_text in {"{", "}", "else", "else {"}:
            body_depth += fragment.count("{") - fragment.count("}")
            continue

        stripped_text = statement_text.lstrip("}")
        if not stripped_text:
            body_depth += fragment.count("{") - fragment.count("}")
            continue

        kind = classify_statement_kind(stripped_text)
        start_column = max(len(fragment) - len(fragment.lstrip()) + 1, 1)
        statement_id = stable_id(
            "stmt",
            repo_name,
            symbol["path"],
            symbol["symbol_id"],
            str(line_number),
            kind,
            stripped_text,
        )
        parent_statement_id = control_stack[-1][0] if control_stack else None

        definitions = collect_statement_definitions(locals_by_line_name, line_number, stripped_text, statement_id)
        writes = collect_statement_writes(
            stripped_text,
            symbol,
            resolution_index,
            context.import_aliases,
            context.dependency_aliases,
            context.crate_root,
            self_target,
        )
        calls, call_expressions = collect_statement_calls(
            stripped_text,
            symbol,
            resolution_index,
            context.import_aliases,
            context.dependency_aliases,
            context.crate_root,
            self_target,
        )
        reads = collect_statement_reads(
            stripped_text,
            symbol,
            resolution_index,
            context.import_aliases,
            context.dependency_aliases,
            context.crate_root,
            self_target,
            excluded_names={item["name"] for item in definitions + writes},
            excluded_expressions=call_expressions,
        )

        for definition in definitions:
            symbol_record = symbol_records_by_id.get(definition["target_symbol_id"] or "")
            if symbol_record is not None:
                symbol_record["statement_id"] = statement_id

        statement_record = {
            "statement_id": statement_id,
            "repo": repo_name,
            "path": symbol["path"],
            "crate": symbol["crate"],
            "module_path": symbol["module_path"],
            "language": symbol["language"],
            "kind": kind,
            "text": stripped_text,
            "span": {
                "start_line": line_number,
                "start_column": start_column,
                "end_line": line_number,
                "end_column": start_column + len(stripped_text),
            },
            "container_symbol_id": symbol["symbol_id"],
            "container_qualified_name": symbol["qualified_name"],
            "parent_statement_id": parent_statement_id,
            "previous_statement_id": previous_statement_id,
            "nesting_depth": max(len(control_stack), 0),
            "defines": definitions,
            "reads": reads,
            "writes": writes,
            "calls": calls,
        }
        statements.append(statement_record)
        previous_statement_id = statement_id

        if kind in {"if", "match", "for", "while", "loop"} and "{" in fragment:
            control_stack.append((statement_id, effective_depth + fragment.count("{")))

        body_depth += fragment.count("{") - fragment.count("}")

    return statements


def statement_fragment_for_line(cleaned_line: str, span: Dict[str, int], line_number: int) -> Optional[str]:
    fragment = cleaned_line
    if line_number == span["start_line"]:
        if "{" not in fragment:
            return None
        fragment = fragment.split("{", 1)[1]
    if line_number == span["end_line"] and "}" in fragment:
        fragment = fragment.rsplit("}", 1)[0]
    return fragment


def classify_statement_kind(statement_text: str) -> str:
    if statement_text.startswith("let "):
        return "let"
    if statement_text.startswith("if "):
        return "if"
    if statement_text.startswith("match "):
        return "match"
    if statement_text.startswith("for "):
        return "for"
    if statement_text.startswith("while "):
        return "while"
    if statement_text.startswith("loop"):
        return "loop"
    if statement_text.startswith("return"):
        return "return"
    if ASSIGN_RE.match(statement_text):
        return "assign"
    return "expr"


def collect_statement_definitions(
    locals_by_line_name: Dict[Tuple[int, str], Dict[str, object]],
    line_number: int,
    statement_text: str,
    statement_id: str,
) -> List[Dict[str, object]]:
    definitions: List[Dict[str, object]] = []
    for match in LET_RE.finditer(statement_text):
        name = match.group("name")
        local_symbol = locals_by_line_name.get((line_number, name))
        target = make_target_entry(
            name,
            {
                "target_symbol_id": local_symbol["symbol_id"] if local_symbol else None,
                "target_qualified_name": local_symbol["qualified_name"] if local_symbol else name,
                "target_kind": local_symbol["kind"] if local_symbol else "local",
                "qualified_name_hint": local_symbol["qualified_name"] if local_symbol else name,
            },
        )
        target["statement_id"] = statement_id
        append_unique_target(definitions, target)
    return definitions


def collect_statement_writes(
    statement_text: str,
    symbol: Dict[str, object],
    resolution_index: Dict[str, object],
    import_aliases: Dict[str, List[str]],
    dependency_aliases: Dict[str, str],
    crate_root: str,
    self_target: Optional[str],
) -> List[Dict[str, object]]:
    match = ASSIGN_RE.match(statement_text)
    writes: List[Dict[str, object]] = []
    if match:
        lhs = match.group("lhs")
        resolved = resolve_expression(
            lhs,
            symbol,
            resolution_index,
            import_aliases,
            dependency_aliases,
            crate_root,
            symbol["symbol_id"],
            self_target,
        )
        append_unique_target(writes, make_target_entry(lhs, resolved))

    member_match = re.match(
        r"^\s*(?P<receiver>(?:self|Self|[A-Za-z_][A-Za-z0-9_]*))\s*\.\s*(?P<field>[a-z_][A-Za-z0-9_]*)\s*(?:[+\-*/%&|^]|<<|>>)?=",
        statement_text,
    )
    if member_match:
        receiver = member_match.group("receiver")
        field = member_match.group("field")
        resolved = resolve_member_expression(
            receiver,
            field,
            symbol,
            resolution_index,
            import_aliases,
            dependency_aliases,
            crate_root,
            self_target,
        )
        append_unique_target(writes, make_target_entry(f"{receiver}.{field}", resolved))
    return writes


def collect_statement_calls(
    statement_text: str,
    symbol: Dict[str, object],
    resolution_index: Dict[str, object],
    import_aliases: Dict[str, List[str]],
    dependency_aliases: Dict[str, str],
    crate_root: str,
    self_target: Optional[str],
) -> Tuple[List[Dict[str, object]], List[str]]:
    calls: List[Dict[str, object]] = []
    expressions: List[str] = []
    for match in CALL_EXPR_RE.finditer(statement_text):
        expression = strip_expression_noise(match.group("expr"))
        if expression in KEYWORDS or expression in PRIMITIVE_TYPES:
            continue
        resolved = resolve_expression(
            expression,
            symbol,
            resolution_index,
            import_aliases,
            dependency_aliases,
            crate_root,
            symbol["symbol_id"],
            self_target,
        )
        append_unique_target(calls, make_target_entry(expression, resolved))
        expressions.append(expression)
    for match in METHOD_CALL_RE.finditer(statement_text):
        receiver = match.group("receiver")
        method = match.group("method")
        expression = f"{receiver}.{method}"
        resolved = resolve_member_expression(
            receiver,
            method,
            symbol,
            resolution_index,
            import_aliases,
            dependency_aliases,
            crate_root,
            self_target,
        )
        append_unique_target(calls, make_target_entry(expression, resolved))
        expressions.append(expression)
    return calls, expressions


def collect_statement_reads(
    statement_text: str,
    symbol: Dict[str, object],
    resolution_index: Dict[str, object],
    import_aliases: Dict[str, List[str]],
    dependency_aliases: Dict[str, str],
    crate_root: str,
    self_target: Optional[str],
    *,
    excluded_names: set[str],
    excluded_expressions: Sequence[str],
) -> List[Dict[str, object]]:
    reads: List[Dict[str, object]] = []
    qualified_matches = {
        strip_expression_noise(match.group("expr"))
        for match in QUALIFIED_PATH_RE.finditer(statement_text)
    }
    excluded = set(excluded_names) | set(excluded_expressions)

    for match in FIELD_ACCESS_RE.finditer(statement_text):
        receiver = match.group("receiver")
        field = match.group("field")
        expression = f"{receiver}.{field}"
        if expression in excluded:
            continue
        window = statement_text[match.end("field") : match.end("field") + 8]
        if "(" in window:
            continue
        resolved = resolve_member_expression(
            receiver,
            field,
            symbol,
            resolution_index,
            import_aliases,
            dependency_aliases,
            crate_root,
            self_target,
        )
        append_unique_target(reads, make_target_entry(expression, resolved))

    for expression in sorted(qualified_matches):
        if not expression or expression in excluded:
            continue
        resolved = resolve_expression(
            expression,
            symbol,
            resolution_index,
            import_aliases,
            dependency_aliases,
            crate_root,
            symbol["symbol_id"],
            self_target,
        )
        append_unique_target(reads, make_target_entry(expression, resolved))

    for match in IDENT_TOKEN_RE.finditer(statement_text):
        name = match.group("name")
        if name in excluded or name in KEYWORDS or name in PRIMITIVE_TYPES:
            continue
        if name[:1].isupper():
            resolved = resolve_expression(
                name,
                symbol,
                resolution_index,
                import_aliases,
                dependency_aliases,
                crate_root,
                symbol["symbol_id"],
                self_target,
            )
            append_unique_target(reads, make_target_entry(name, resolved))
            continue
        resolved = resolve_expression(
            name,
            symbol,
            resolution_index,
            import_aliases,
            dependency_aliases,
            crate_root,
            symbol["symbol_id"],
            self_target,
        )
        if resolved["target_symbol_id"] or resolved["target_qualified_name"] != name:
            append_unique_target(reads, make_target_entry(name, resolved))
    return reads


def resolve_member_expression(
    receiver: str,
    member: str,
    current_symbol: Dict[str, object],
    resolution_index: Dict[str, object],
    import_aliases: Dict[str, List[str]],
    dependency_aliases: Dict[str, str],
    crate_root: str,
    self_target: Optional[str],
) -> Dict[str, Optional[str]]:
    receiver_target: Optional[str] = None
    if receiver in {"self", "Self"} and self_target:
        receiver_target = self_target
    elif receiver[:1].isupper():
        resolved_receiver = resolve_expression(
            receiver,
            current_symbol,
            resolution_index,
            import_aliases,
            dependency_aliases,
            crate_root,
            current_symbol.get("symbol_id") or current_symbol.get("scope_symbol_id"),
            self_target,
        )
        receiver_target = resolved_receiver["target_qualified_name"] or resolved_receiver["qualified_name_hint"]

    if receiver_target:
        return resolve_expression(
            f"{receiver_target}::{member}",
            current_symbol,
            resolution_index,
            import_aliases,
            dependency_aliases,
            crate_root,
            current_symbol.get("symbol_id") or current_symbol.get("scope_symbol_id"),
            self_target,
        )

    return {
        "target_symbol_id": None,
        "target_qualified_name": None,
        "target_kind": None,
        "qualified_name_hint": f"{receiver}.{member}",
    }


def make_target_entry(expression: str, resolved: Dict[str, Optional[str]]) -> Dict[str, object]:
    qualified_name_hint = resolved["qualified_name_hint"] or expression
    return {
        "name": expression.split("::")[-1],
        "qualified_name_hint": qualified_name_hint,
        "target_symbol_id": resolved["target_symbol_id"],
        "target_qualified_name": resolved["target_qualified_name"] or qualified_name_hint,
        "target_kind": resolved["target_kind"],
    }


def append_unique_target(targets: List[Dict[str, object]], entry: Dict[str, object]) -> None:
    key = (
        entry.get("target_symbol_id"),
        entry.get("target_qualified_name"),
        entry.get("name"),
    )
    if any(
        (item.get("target_symbol_id"), item.get("target_qualified_name"), item.get("name")) == key
        for item in targets
    ):
        return
    targets.append(entry)


def discover_rust_files(repo_root: Path, parser_roots: Iterable[str], path_prefixes: Sequence[str]) -> List[Path]:
    candidates = []
    seen = set()

    for parser_root in parser_roots:
        absolute_root = repo_root / parser_root
        if not absolute_root.exists():
            continue
        if absolute_root.is_file():
            if absolute_root.suffix != ".rs":
                continue
            relative_path = absolute_root.relative_to(repo_root).as_posix()
            if relative_path in seen:
                continue
            if is_generated_path(relative_path):
                continue
            if path_prefixes and not matches_path_prefix(relative_path, path_prefixes):
                continue
            seen.add(relative_path)
            candidates.append(absolute_root)
            continue
        for path in sorted(absolute_root.rglob("*.rs")):
            relative_path = path.relative_to(repo_root).as_posix()
            if relative_path in seen:
                continue
            if is_generated_path(relative_path):
                continue
            if path_prefixes and not matches_path_prefix(relative_path, path_prefixes):
                continue
            seen.add(relative_path)
            candidates.append(path)

    return sorted(candidates, key=lambda item: item.relative_to(repo_root).as_posix())


def expand_reference_candidates(
    expression: str,
    module_path: str,
    crate_root: str,
    import_aliases: Dict[str, List[str]],
    dependency_aliases: Dict[str, str],
    self_target: Optional[str],
) -> List[str]:
    candidates: List[str] = []
    expr = strip_expression_noise(expression)
    if not expr:
        return candidates

    if expr == "Self" and self_target:
        return [self_target]

    if "::" not in expr:
        for alias_target in import_aliases.get(expr, []):
            candidates.append(alias_target)
        if expr in dependency_aliases:
            candidates.append(dependency_aliases[expr])
        if self_target and expr[:1].islower():
            candidates.append(f"{self_target}::{expr}")
        candidates.append(f"{module_path}::{expr}")
        if expr[:1].isupper():
            candidates.append(f"{crate_root}::{expr}")
        candidates.append(expr)
        return unique_values(candidates)

    if expr.startswith("Self::") and self_target:
        candidates.append(f"{self_target}{expr[4:]}")
    elif expr.startswith("crate::"):
        candidates.append(f"{crate_root}{expr[5:]}")
    elif expr.startswith("super::"):
        current = module_path
        remainder = expr
        while remainder.startswith("super::"):
            current = current.rsplit("::", 1)[0] if "::" in current else crate_root
            remainder = remainder[len("super::") :]
        candidates.append(f"{current}::{remainder}")

    first_segment, _, remainder = expr.partition("::")
    if first_segment in import_aliases:
        for alias_target in import_aliases[first_segment]:
            candidate = alias_target
            if remainder:
                candidate = f"{alias_target}::{remainder}"
            candidates.append(candidate)

    candidates.append(expr)
    return unique_values(
        normalize_path_expression(candidate, module_path, crate_root, dependency_aliases, self_target)
        for candidate in candidates
        if candidate
    )


def expand_use_targets(target: str) -> List[Tuple[str, Optional[str]]]:
    value = target.strip().rstrip(";")
    if not value:
        return []

    base, alias = split_top_level_alias(value)
    brace_open, brace_close = find_top_level_braces(base)
    if brace_open is None or brace_close is None:
        return [(collapse_whitespace(base), alias)]

    prefix = base[:brace_open].strip().rstrip(":")
    inner = base[brace_open + 1 : brace_close]
    suffix = base[brace_close + 1 :].strip()
    expanded: List[Tuple[str, Optional[str]]] = []

    for part in split_top_level(inner):
        if not part:
            continue
        if part == "self":
            candidate = prefix
        else:
            joiner = "::" if prefix else ""
            candidate = f"{prefix}{joiner}{part}"
        if suffix:
            candidate = f"{candidate}{suffix}"
        expanded.extend(expand_use_targets(candidate))

    return expanded


def extract_body_call_candidates(
    cleaned_lines: Sequence[str],
    span: Dict[str, int],
) -> List[Tuple[str, int, int]]:
    candidates: List[Tuple[str, int, int]] = []
    for line_number in range(span["start_line"], span["end_line"] + 1):
        cleaned_line = cleaned_lines[line_number - 1]
        for match in CALL_EXPR_RE.finditer(cleaned_line):
            expression = strip_expression_noise(match.group("expr"))
            if expression in KEYWORDS or expression in PRIMITIVE_TYPES:
                continue
            candidates.append((expression, line_number, match.start("expr") + 1))
    return unique_positioned_values(candidates)


def extract_body_member_call_candidates(
    cleaned_lines: Sequence[str],
    span: Dict[str, int],
    symbol: Dict[str, object],
    resolution_index: Dict[str, object],
    import_aliases: Dict[str, List[str]],
    dependency_aliases: Dict[str, str],
    crate_root: str,
    self_target: Optional[str],
) -> List[Tuple[str, int, int]]:
    candidates: List[Tuple[str, int, int]] = []
    for line_number in range(span["start_line"], span["end_line"] + 1):
        cleaned_line = cleaned_lines[line_number - 1]
        for match in METHOD_CALL_RE.finditer(cleaned_line):
            candidate = member_candidate_string(
                match.group("receiver"),
                match.group("method"),
                symbol,
                resolution_index,
                import_aliases,
                dependency_aliases,
                crate_root,
                self_target,
            )
            if candidate:
                candidates.append((candidate, line_number, match.start("method") + 1))
    return unique_positioned_values(candidates)


def extract_body_field_use_candidates(
    cleaned_lines: Sequence[str],
    span: Dict[str, int],
    symbol: Dict[str, object],
    resolution_index: Dict[str, object],
    import_aliases: Dict[str, List[str]],
    dependency_aliases: Dict[str, str],
    crate_root: str,
    self_target: Optional[str],
) -> List[Tuple[str, int, int]]:
    candidates: List[Tuple[str, int, int]] = []
    for line_number in range(span["start_line"], span["end_line"] + 1):
        cleaned_line = cleaned_lines[line_number - 1]
        for match in FIELD_ACCESS_RE.finditer(cleaned_line):
            window = cleaned_line[match.end("field") : match.end("field") + 8]
            if "(" in window:
                continue
            candidate = member_candidate_string(
                match.group("receiver"),
                match.group("field"),
                symbol,
                resolution_index,
                import_aliases,
                dependency_aliases,
                crate_root,
                self_target,
            )
            if candidate:
                candidates.append((candidate, line_number, match.start("field") + 1))
    return unique_positioned_values(candidates)


def member_candidate_string(
    receiver: str,
    member: str,
    symbol: Dict[str, object],
    resolution_index: Dict[str, object],
    import_aliases: Dict[str, List[str]],
    dependency_aliases: Dict[str, str],
    crate_root: str,
    self_target: Optional[str],
) -> str:
    resolved = resolve_member_expression(
        receiver,
        member,
        symbol,
        resolution_index,
        import_aliases,
        dependency_aliases,
        crate_root,
        self_target,
    )
    return resolved["target_qualified_name"] or resolved["qualified_name_hint"] or f"{receiver}.{member}"


def extract_body_use_candidates(
    cleaned_lines: Sequence[str],
    span: Dict[str, int],
) -> List[Tuple[str, int, int]]:
    candidates: List[Tuple[str, int, int]] = []
    for line_number in range(span["start_line"], span["end_line"] + 1):
        cleaned_line = cleaned_lines[line_number - 1]
        for match in QUALIFIED_PATH_RE.finditer(cleaned_line):
            expression = strip_expression_noise(match.group("expr"))
            if expression in KEYWORDS:
                continue
            candidates.append((expression, line_number, match.start("expr") + 1))
    return filter_shadowed_simple_tokens(unique_positioned_values(candidates))


def extract_signature_reference_candidates(
    signature: str,
    symbol_name: str,
    line_number: int,
) -> List[Tuple[str, int, int]]:
    candidates: List[Tuple[str, int, int]] = []
    for match in PATH_TOKEN_RE.finditer(signature):
        expression = strip_expression_noise(match.group("expr"))
        if not expression or expression == symbol_name:
            continue
        if expression in KEYWORDS or expression in PRIMITIVE_TYPES:
            continue
        if "::" not in expression and not expression[:1].isupper() and expression != "Self":
            continue
        candidates.append((expression, line_number, match.start("expr") + 1))
    return filter_shadowed_simple_tokens(unique_positioned_values(candidates))


def find_top_level_braces(value: str) -> Tuple[Optional[int], Optional[int]]:
    depth = 0
    open_index: Optional[int] = None
    for index, char in enumerate(value):
        if char == "{":
            if depth == 0:
                open_index = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and open_index is not None:
                return open_index, index
    return None, None


def infer_self_target(symbol: Dict[str, object], resolution_index: Dict[str, object]) -> Optional[str]:
    container_symbol_id = symbol.get("container_symbol_id")
    if not container_symbol_id:
        return None

    container_symbol = resolution_index["by_id"].get(container_symbol_id)
    if not container_symbol or container_symbol["kind"] != "impl":
        return None

    return (
        container_symbol.get("resolved_impl_target_qualified_name")
        or container_symbol.get("impl_target")
        or container_symbol.get("qualified_name")
    )


def load_cargo_package_name(manifest_path: Path) -> str:
    text = manifest_path.read_text(encoding="utf-8")
    block_match = PACKAGE_BLOCK_RE.search(text)
    if block_match:
        name_match = PACKAGE_NAME_RE.search(block_match.group(1))
        if name_match:
            return name_match.group(1)
    return manifest_path.parent.name


def load_raw_manifest(raw_root: Path, repo_name: str) -> Dict[str, object]:
    manifest_path = raw_root / repo_name / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing raw inventory manifest for {repo_name}: {manifest_path}. Run parse-repos first."
        )
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_workspace_index(repo_root: Path) -> WorkspaceIndex:
    metadata = load_cargo_metadata(repo_root)
    if metadata is not None:
        index = workspace_index_from_metadata(repo_root, metadata)
        if index.packages_by_root:
            return index

    return workspace_index_from_manifests(repo_root)


def workspace_index_from_manifests(repo_root: Path) -> WorkspaceIndex:
    package_toml_by_root: Dict[Path, Dict[str, object]] = {}
    packages_by_root: Dict[Path, CargoPackageInfo] = {}

    for manifest_path in sorted(repo_root.rglob("Cargo.toml")):
        relative_manifest = manifest_path.relative_to(repo_root).as_posix()
        if any(part in {"target", ".git", "node_modules"} for part in manifest_path.relative_to(repo_root).parts):
            continue
        with manifest_path.open("rb") as handle:
            cargo_data = tomllib.load(handle)
        package = cargo_data.get("package")
        if not isinstance(package, dict) or "name" not in package:
            continue

        package_name = str(package["name"])
        lib_table = cargo_data.get("lib", {}) if isinstance(cargo_data.get("lib"), dict) else {}
        crate_name = str(lib_table.get("name") or package_name)
        crate_module = crate_name.replace("-", "_")
        package_root = manifest_path.parent
        package_toml_by_root[package_root] = cargo_data
        packages_by_root[package_root] = CargoPackageInfo(
            root=package_root,
            manifest_path=manifest_path,
            package_name=package_name,
            crate_name=crate_name,
            crate_module=crate_module,
            dependency_aliases={},
        )

    packages_by_name = {package.package_name: package for package in packages_by_root.values()}
    packages_by_module = {package.crate_module: package for package in packages_by_root.values()}

    resolved_packages_by_root: Dict[Path, CargoPackageInfo] = {}
    for package_root, package_info in packages_by_root.items():
        cargo_data = package_toml_by_root[package_root]
        dependency_aliases = resolve_dependency_aliases(cargo_data, packages_by_name)
        resolved_packages_by_root[package_root] = CargoPackageInfo(
            root=package_info.root,
            manifest_path=package_info.manifest_path,
            package_name=package_info.package_name,
            crate_name=package_info.crate_name,
            crate_module=package_info.crate_module,
            dependency_aliases=dependency_aliases,
        )

    return WorkspaceIndex(
        packages_by_root=resolved_packages_by_root,
        packages_by_name={package.package_name: package for package in resolved_packages_by_root.values()},
        packages_by_module={package.crate_module: package for package in resolved_packages_by_root.values()},
    )


def load_cargo_metadata(repo_root: Path) -> Optional[Dict[str, object]]:
    manifest_path = repo_root / "Cargo.toml"
    if not manifest_path.exists():
        return None

    try:
        result = subprocess.run(
            [
                "cargo",
                "metadata",
                "--format-version",
                "1",
                "--no-deps",
                "--manifest-path",
                str(manifest_path),
            ],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def workspace_index_from_metadata(repo_root: Path, metadata: Dict[str, object]) -> WorkspaceIndex:
    packages_by_root: Dict[Path, CargoPackageInfo] = {}
    raw_packages: Dict[Path, Dict[str, object]] = {}

    for package in metadata.get("packages", []):
        if not isinstance(package, dict):
            continue
        manifest_value = package.get("manifest_path")
        if not isinstance(manifest_value, str):
            continue
        manifest_path = Path(manifest_value).resolve()
        if not manifest_path.is_relative_to(repo_root):
            continue

        package_root = manifest_path.parent
        package_name = str(package.get("name") or package_root.name)
        crate_name = package_name
        for target in package.get("targets", []):
            if not isinstance(target, dict):
                continue
            kinds = {str(kind) for kind in target.get("kind", [])}
            if kinds.intersection({"lib", "proc-macro"}):
                crate_name = str(target.get("name") or package_name)
                break

        packages_by_root[package_root] = CargoPackageInfo(
            root=package_root,
            manifest_path=manifest_path,
            package_name=package_name,
            crate_name=crate_name,
            crate_module=crate_name.replace("-", "_"),
            dependency_aliases={},
        )
        raw_packages[package_root] = package

    packages_by_name = {package.package_name: package for package in packages_by_root.values()}
    resolved_packages_by_root: Dict[Path, CargoPackageInfo] = {}
    for package_root, package_info in packages_by_root.items():
        dependency_aliases = resolve_dependency_aliases_from_metadata(raw_packages[package_root], packages_by_name)
        dependency_aliases.setdefault(package_info.crate_module, package_info.crate_module)
        resolved_packages_by_root[package_root] = CargoPackageInfo(
            root=package_info.root,
            manifest_path=package_info.manifest_path,
            package_name=package_info.package_name,
            crate_name=package_info.crate_name,
            crate_module=package_info.crate_module,
            dependency_aliases=dependency_aliases,
        )

    return WorkspaceIndex(
        packages_by_root=resolved_packages_by_root,
        packages_by_name={package.package_name: package for package in resolved_packages_by_root.values()},
        packages_by_module={package.crate_module: package for package in resolved_packages_by_root.values()},
    )


def resolve_dependency_aliases_from_metadata(
    package: Dict[str, object],
    packages_by_name: Dict[str, CargoPackageInfo],
) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    for dependency in package.get("dependencies", []):
        if not isinstance(dependency, dict):
            continue
        package_name = str(dependency.get("name") or "")
        package_info = packages_by_name.get(package_name)
        if package_info is None:
            continue
        alias = str(dependency.get("rename") or package_name)
        alias_module = alias.replace("-", "_")
        aliases[alias_module] = package_info.crate_module
        aliases.setdefault(package_info.crate_module, package_info.crate_module)
    return aliases


def resolve_dependency_aliases(
    cargo_data: Dict[str, object],
    packages_by_name: Dict[str, CargoPackageInfo],
) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    for dependency_table in iter_dependency_tables(cargo_data):
        for alias, spec in dependency_table.items():
            alias_module = str(alias).replace("-", "_")
            target_package_name = dependency_target_package_name(alias, spec)
            package_info = packages_by_name.get(target_package_name)
            if package_info is None:
                continue
            aliases[alias_module] = package_info.crate_module
            aliases.setdefault(package_info.crate_module, package_info.crate_module)
    return aliases


def iter_dependency_tables(cargo_data: Dict[str, object]) -> Iterable[Dict[str, object]]:
    for key, value in cargo_data.items():
        if key.endswith("dependencies") and isinstance(value, dict):
            yield value
        if key == "target" and isinstance(value, dict):
            for target_table in value.values():
                if not isinstance(target_table, dict):
                    continue
                for target_key, target_value in target_table.items():
                    if target_key.endswith("dependencies") and isinstance(target_value, dict):
                        yield target_value


def dependency_target_package_name(alias: object, spec: object) -> str:
    if isinstance(spec, dict):
        package_name = spec.get("package")
        if isinstance(package_name, str) and package_name.strip():
            return package_name
    return str(alias)


def matches_path_prefix(relative_path: str, path_prefixes: Sequence[str]) -> bool:
    return any(
        relative_path == prefix or relative_path.startswith(f"{prefix}/")
        for prefix in path_prefixes
    )


def nearest_cargo_package(file_path: Path, repo_root: Path, workspace_index: WorkspaceIndex) -> CargoPackageInfo:
    current = file_path.parent
    while True:
        package_info = workspace_index.packages_by_root.get(current)
        if package_info is not None:
            return package_info
        if current == repo_root:
            break
        current = current.parent
    fallback_name = repo_root.name
    return CargoPackageInfo(
        root=repo_root,
        manifest_path=repo_root / "Cargo.toml",
        package_name=fallback_name,
        crate_name=fallback_name,
        crate_module=fallback_name.replace("-", "_"),
        dependency_aliases={},
    )


def next_symbol_local_id(symbols: Sequence[RustSymbol]) -> int:
    return max((symbol.local_id for symbol in symbols), default=0) + 1


def normalize_prefixes(path_prefixes: Sequence[str]) -> Tuple[str, ...]:
    return tuple(
        sorted(
            {
                prefix.strip().lstrip("./").rstrip("/")
                for prefix in path_prefixes
                if prefix and prefix.strip().lstrip("./").rstrip("/")
            }
        )
    )


def normalize_path_expression(
    expression: str,
    module_path: str,
    crate_root: str,
    dependency_aliases: Dict[str, str],
    self_target: Optional[str],
) -> str:
    expr = strip_expression_noise(expression)
    if not expr:
        return ""

    if expr == "Self" and self_target:
        return self_target
    if expr.startswith("Self::") and self_target:
        return f"{self_target}{expr[4:]}"
    if expr.startswith("crate::"):
        return f"{crate_root}{expr[5:]}"
    if "::" not in expr and expr in dependency_aliases:
        return dependency_aliases[expr]
    if expr.startswith("super::"):
        current = module_path
        remainder = expr
        while remainder.startswith("super::"):
            current = current.rsplit("::", 1)[0] if "::" in current else crate_root
            remainder = remainder[len("super::") :]
        return f"{current}::{remainder}"

    first_segment, _, remainder = expr.partition("::")
    if first_segment in dependency_aliases:
        target = dependency_aliases[first_segment]
        return f"{target}::{remainder}" if remainder else target

    if "::" not in expr and expr[:1].isupper():
        return f"{module_path}::{expr}"
    return expr


def normalize_visibility(value: Optional[str]) -> str:
    return value.strip() if value else "private"


BACKEND_MERGEABLE_KINDS = {"module", "function", "method", "struct", "enum", "trait", "const", "static", "type"}


def merge_primary_backend_symbols(
    parsed: ParsedRustFile,
    source_lines: Sequence[str],
    tree_sitter_probe: Dict[str, object],
    rust_analyzer_probe: Dict[str, object],
) -> str:
    primary_backend = "rust-simple-v3"
    if merge_backend_symbols(
        parsed,
        source_lines,
        rust_analyzer_probe.get("document_symbols") or [],
        backend_name="rust_analyzer_lsp",
    ):
        primary_backend = "rust_analyzer_lsp"
    if merge_backend_symbols(
        parsed,
        source_lines,
        tree_sitter_probe.get("symbols") or [],
        backend_name="tree_sitter_rust",
    ) and primary_backend == "rust-simple-v3":
        primary_backend = "tree_sitter_rust"
    return primary_backend


def merge_backend_symbols(
    parsed: ParsedRustFile,
    source_lines: Sequence[str],
    backend_symbols: Sequence[Dict[str, object]],
    *,
    backend_name: str,
) -> bool:
    if not backend_symbols:
        return False

    symbols_by_qname = {symbol.qualified_name: symbol for symbol in parsed.symbols}
    next_local_id = next_symbol_local_id(parsed.symbols)
    used_backend = False

    for backend_symbol in sorted(
        backend_symbols,
        key=lambda item: (
            str(item.get("qualified_name") or "").count("::"),
            item.get("selection_range", {}).get("start_line", 0),
            item.get("selection_range", {}).get("start_column", 0),
            str(item.get("qualified_name") or ""),
        ),
    ):
        kind = str(backend_symbol.get("kind") or "")
        if kind not in BACKEND_MERGEABLE_KINDS:
            continue

        qualified_name = qualify_backend_symbol_name(parsed.module_path, str(backend_symbol.get("qualified_name") or ""))
        container_qualified_name = backend_symbol.get("container_qualified_name")
        if container_qualified_name:
            container_qualified_name = qualify_backend_symbol_name(parsed.module_path, str(container_qualified_name))

        existing = symbols_by_qname.get(qualified_name)
        if existing is None:
            existing = find_backend_symbol_match(
                parsed.symbols,
                kind,
                str(backend_symbol["name"]),
                container_qualified_name,
                backend_symbol=backend_symbol,
            )

        if existing is not None:
            if (
                kind in FUNCTION_LIKE_KINDS
                and existing.kind in FUNCTION_LIKE_KINDS
                and not container_qualified_name
                and existing.container_qualified_name
            ):
                qualified_name = existing.qualified_name
            update_symbol_from_backend(existing, backend_symbol, qualified_name)
            symbols_by_qname[existing.qualified_name] = existing
            used_backend = True
            continue

        container_local_id = None
        if container_qualified_name:
            container_symbol = symbols_by_qname.get(container_qualified_name)
            if container_symbol is not None:
                container_local_id = container_symbol.local_id

        parsed.symbols.append(
            RustSymbol(
                local_id=next_local_id,
                kind=kind,
                name=str(backend_symbol["name"]),
                qualified_name=qualified_name,
                module_path=backend_symbol_module_path(parsed.module_path, qualified_name, container_qualified_name),
                span=backend_range_to_span(backend_symbol),
                signature=str(backend_symbol.get("signature") or backend_signature(source_lines, backend_symbol)),
                visibility=infer_backend_visibility(source_lines, backend_symbol),
                docstring=None,
                container_local_id=container_local_id,
                container_qualified_name=container_qualified_name,
                attributes=(f"backend:{backend_name}",),
                is_test=is_backend_test_symbol(qualified_name),
            )
        )
        symbols_by_qname[qualified_name] = parsed.symbols[-1]
        next_local_id += 1
        used_backend = True

    return used_backend


def qualify_backend_symbol_name(module_path: str, backend_qualified_name: str) -> str:
    if not backend_qualified_name:
        return module_path
    crate_root = module_path.split("::")[0]
    if backend_qualified_name == "crate":
        return crate_root
    if backend_qualified_name.startswith("crate::"):
        return f"{crate_root}::{backend_qualified_name[7:]}"
    if backend_qualified_name == "self":
        return module_path
    if backend_qualified_name.startswith("self::"):
        return f"{module_path}::{backend_qualified_name[6:]}"
    if backend_qualified_name.startswith("super::"):
        current = module_path
        remainder = backend_qualified_name
        while remainder.startswith("super::"):
            current = current.rsplit("::", 1)[0] if "::" in current else crate_root
            remainder = remainder[len("super::") :]
        return f"{current}::{remainder}" if remainder else current
    if backend_qualified_name.startswith(f"{module_path}::") or backend_qualified_name == module_path:
        return backend_qualified_name
    return f"{module_path}::{backend_qualified_name}"


def backend_symbol_module_path(module_path: str, qualified_name: str, container_qualified_name: Optional[str]) -> str:
    if container_qualified_name:
        return container_qualified_name
    if "::" in qualified_name:
        return qualified_name.rsplit("::", 1)[0]
    return module_path


def find_backend_symbol_match(
    symbols: Sequence[RustSymbol],
    kind: str,
    name: str,
    container_qualified_name: Optional[str],
    *,
    backend_symbol: Optional[Dict[str, object]] = None,
) -> Optional[RustSymbol]:
    backend_span = backend_range_to_span(backend_symbol) if backend_symbol is not None else None
    for symbol in symbols:
        same_kind = symbol.kind == kind
        aliased_callable_kind = (
            kind in FUNCTION_LIKE_KINDS and symbol.kind in FUNCTION_LIKE_KINDS
        )
        if (not same_kind and not aliased_callable_kind) or symbol.name != name:
            continue
        if container_qualified_name and symbol.container_qualified_name != container_qualified_name:
            continue
        if backend_span is not None:
            same_start = (
                symbol.span.start_line == backend_span.start_line
                and symbol.span.start_column == backend_span.start_column
            )
            if not same_start and aliased_callable_kind:
                continue
        return symbol
    return None


def update_symbol_from_backend(symbol: RustSymbol, backend_symbol: Dict[str, object], qualified_name: str) -> None:
    backend_span = backend_range_to_span(backend_symbol)
    symbol.qualified_name = qualified_name
    if symbol.kind not in FUNCTION_LIKE_KINDS or symbol.container_qualified_name is None:
        symbol.module_path = backend_symbol_module_path(symbol.module_path, qualified_name, symbol.container_qualified_name)
    if backend_span.start_line <= symbol.span.start_line:
        symbol.span.start_line = backend_span.start_line
        symbol.span.start_column = backend_span.start_column
    if backend_span.end_line >= symbol.span.end_line:
        symbol.span.end_line = backend_span.end_line
        symbol.span.end_column = backend_span.end_column


def backend_range_to_span(backend_symbol: Dict[str, object]) -> TextSpan:
    selection_range = backend_symbol.get("selection_range", {})
    full_range = backend_symbol.get("range", selection_range)
    return TextSpan(
        start_line=int(selection_range.get("start_line", full_range.get("start_line", 1))),
        start_column=int(selection_range.get("start_column", full_range.get("start_column", 1))),
        end_line=int(full_range.get("end_line", selection_range.get("end_line", 1))),
        end_column=int(full_range.get("end_column", selection_range.get("end_column", 1))),
    )


def backend_signature(source_lines: Sequence[str], backend_symbol: Dict[str, object]) -> str:
    start_line = max(int(backend_symbol.get("selection_range", {}).get("start_line", 1)), 1)
    if not source_lines or start_line > len(source_lines):
        return str(backend_symbol.get("name") or "")
    return source_lines[start_line - 1].strip()


def infer_backend_visibility(source_lines: Sequence[str], backend_symbol: Dict[str, object]) -> str:
    signature = backend_signature(source_lines, backend_symbol)
    return "pub" if signature.lstrip().startswith("pub ") else "private"


def is_backend_test_symbol(qualified_name: str) -> bool:
    lowered = qualified_name.lower()
    return "::tests::" in lowered or lowered.endswith("::tests") or lowered.endswith("::smoke")


def parse_rust_source_file(
    repo_root: Path,
    source_path: Path,
    workspace_index: WorkspaceIndex,
) -> ParsedFileContext:
    started = time.perf_counter()
    package_info = nearest_cargo_package(source_path, repo_root, workspace_index)
    crate_root = package_info.root
    crate_name = package_info.package_name
    module_path = derive_module_path(package_info.crate_module, source_path, crate_root)
    relative_path = source_path.relative_to(repo_root).as_posix()
    source = source_path.read_text(encoding="utf-8")
    source_lines = source.splitlines()
    parsed = parse_rust_file(relative_path, source, crate_name, module_path)
    compiler_probe = probe_rust_ast(source_path)
    backend_probes = {
        "tree_sitter_rust": probe_tree_sitter(source_path, source),
        "rust_analyzer_lsp": probe_rust_analyzer(source_path, source, repo_root),
    }
    primary_parser_backend = merge_primary_backend_symbols(
        parsed,
        source_lines,
        backend_probes["tree_sitter_rust"],
        backend_probes["rust_analyzer_lsp"],
    )
    return ParsedFileContext(
        source_path=source_path,
        package_info=package_info,
        workspace_index=workspace_index,
        parsed=parsed,
        source=source,
        source_lines=source_lines,
        cleaned_lines=clean_rust_source_lines(source),
        crate_root=module_path.split("::")[0],
        dependency_aliases=package_info.dependency_aliases,
        symbol_id_by_local={},
        import_aliases={},
        compiler_probe=compiler_probe,
        backend_probes=backend_probes,
        primary_parser_backend=primary_parser_backend,
        parse_elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
        cache_hit=False,
    )


def current_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss_kb = usage.ru_maxrss
    if sys.platform == "darwin":
        return round(rss_kb / (1024 * 1024), 2)
    return round(rss_kb / 1024, 2)


def load_or_parse_rust_source_file(
    repo_root: Path,
    source_path: Path,
    workspace_index: WorkspaceIndex,
    cache_root: Path | None,
) -> ParsedFileContext:
    package_info = nearest_cargo_package(source_path, repo_root, workspace_index)
    source = source_path.read_text(encoding="utf-8")
    cache_path = file_cache_path(cache_root, repo_root, source_path) if cache_root is not None else None
    cache_key = build_file_cache_key(source_path, source, package_info.manifest_path)

    if cache_path is not None:
        cached = load_cached_file_context(
            cache_path,
            repo_root,
            source_path,
            workspace_index,
            package_info,
            source,
            cache_key,
        )
        if cached is not None:
            return cached

    context = parse_rust_source_file(repo_root, source_path, workspace_index)
    enrich_context_symbols(context)
    if cache_path is not None:
        write_cached_file_context(cache_path, context, cache_key)
    return context


def file_cache_path(cache_root: Path | None, repo_root: Path, source_path: Path) -> Path:
    assert cache_root is not None
    relative_path = source_path.relative_to(repo_root)
    return cache_root / "file-cache" / relative_path.with_suffix(relative_path.suffix + ".json")


def build_file_cache_key(source_path: Path, source: str, manifest_path: Path) -> Dict[str, object]:
    manifest_stat = manifest_path.stat()
    source_stat = source_path.stat()
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "source_sha1": hashlib.sha1(source.encode("utf-8")).hexdigest(),
        "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "manifest_path": str(manifest_path),
        "manifest_mtime_ns": manifest_stat.st_mtime_ns,
    }


def load_cached_file_context(
    cache_path: Path,
    repo_root: Path,
    source_path: Path,
    workspace_index: WorkspaceIndex,
    package_info: CargoPackageInfo,
    source: str,
    cache_key: Dict[str, object],
) -> ParsedFileContext | None:
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if payload.get("cache_key") != cache_key:
        return None
    parsed_payload = payload.get("parsed")
    if not isinstance(parsed_payload, dict):
        return None

    parsed = parsed_rust_file_from_payload(parsed_payload)
    return ParsedFileContext(
        source_path=source_path,
        package_info=package_info,
        workspace_index=workspace_index,
        parsed=parsed,
        source=source,
        source_lines=source.splitlines(),
        cleaned_lines=clean_rust_source_lines(source),
        crate_root=parsed.module_path.split("::")[0],
        dependency_aliases=package_info.dependency_aliases,
        symbol_id_by_local={},
        import_aliases={},
        compiler_probe=dict(payload.get("compiler_probe") or {}),
        backend_probes={key: dict(value) for key, value in dict(payload.get("backend_probes") or {}).items()},
        primary_parser_backend=str(payload.get("primary_parser_backend") or "rust-simple-v3"),
        parse_elapsed_ms=float(payload.get("parse_elapsed_ms") or 0.0),
        cache_hit=True,
    )


def write_cached_file_context(cache_path: Path, context: ParsedFileContext, cache_key: Dict[str, object]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_key": cache_key,
        "primary_parser_backend": context.primary_parser_backend,
        "parse_elapsed_ms": context.parse_elapsed_ms,
        "compiler_probe": context.compiler_probe,
        "backend_probes": context.backend_probes,
        "parsed": parsed_rust_file_to_payload(context.parsed),
    }
    cache_path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def parsed_rust_file_to_payload(parsed: ParsedRustFile) -> Dict[str, object]:
    return {
        "path": parsed.path,
        "crate_name": parsed.crate_name,
        "module_path": parsed.module_path,
        "symbols": [rust_symbol_to_payload(symbol) for symbol in parsed.symbols],
        "imports": [rust_import_to_payload(item) for item in parsed.imports],
    }


def parsed_rust_file_from_payload(payload: Dict[str, object]) -> ParsedRustFile:
    return ParsedRustFile(
        path=str(payload["path"]),
        crate_name=str(payload["crate_name"]),
        module_path=str(payload["module_path"]),
        symbols=[rust_symbol_from_payload(item) for item in payload.get("symbols", [])],
        imports=[rust_import_from_payload(item) for item in payload.get("imports", [])],
    )


def rust_symbol_to_payload(symbol: RustSymbol) -> Dict[str, object]:
    return {
        "local_id": symbol.local_id,
        "kind": symbol.kind,
        "name": symbol.name,
        "qualified_name": symbol.qualified_name,
        "module_path": symbol.module_path,
        "span": span_to_dict(symbol.span),
        "signature": symbol.signature,
        "visibility": symbol.visibility,
        "docstring": symbol.docstring,
        "container_local_id": symbol.container_local_id,
        "container_qualified_name": symbol.container_qualified_name,
        "attributes": list(symbol.attributes),
        "is_test": symbol.is_test,
        "impl_target": symbol.impl_target,
        "impl_trait": symbol.impl_trait,
        "super_traits": list(symbol.super_traits),
    }


def rust_symbol_from_payload(payload: Dict[str, object]) -> RustSymbol:
    return RustSymbol(
        local_id=int(payload["local_id"]),
        kind=str(payload["kind"]),
        name=str(payload["name"]),
        qualified_name=str(payload["qualified_name"]),
        module_path=str(payload["module_path"]),
        span=span_from_dict(dict(payload["span"])),
        signature=str(payload["signature"]),
        visibility=str(payload["visibility"]),
        docstring=payload.get("docstring"),
        container_local_id=payload.get("container_local_id"),
        container_qualified_name=payload.get("container_qualified_name"),
        attributes=tuple(payload.get("attributes", [])),
        is_test=bool(payload.get("is_test")),
        impl_target=payload.get("impl_target"),
        impl_trait=payload.get("impl_trait"),
        super_traits=tuple(payload.get("super_traits", [])),
    )


def rust_import_to_payload(item: RustImport) -> Dict[str, object]:
    return {
        "path": item.path,
        "module_path": item.module_path,
        "span": span_to_dict(item.span),
        "visibility": item.visibility,
        "signature": item.signature,
        "container_local_id": item.container_local_id,
        "container_qualified_name": item.container_qualified_name,
    }


def rust_import_from_payload(payload: Dict[str, object]) -> RustImport:
    return RustImport(
        path=str(payload["path"]),
        module_path=str(payload["module_path"]),
        span=span_from_dict(dict(payload["span"])),
        visibility=str(payload["visibility"]),
        signature=str(payload["signature"]),
        container_local_id=payload.get("container_local_id"),
        container_qualified_name=payload.get("container_qualified_name"),
    )


def pick_best_symbol_candidate(
    candidates: Sequence[Dict[str, object]],
    preferred_paths: Sequence[str],
    current_symbol: Dict[str, object],
    self_target: Optional[str],
    preferred_roots: Sequence[str] = (),
) -> Optional[Dict[str, object]]:
    best_candidate = None
    best_score = None
    tie = False
    preferred = set(preferred_paths)

    for candidate in candidates:
        score = 0
        candidate_root = candidate["qualified_name"].split("::")[0]
        if candidate["qualified_name"] in preferred:
            score += 100
        if candidate_root in preferred_roots:
            score += 40
        if candidate["crate"] == current_symbol["crate"]:
            score += 20
        if candidate["path"] == current_symbol["path"]:
            score += 15
        if candidate["module_path"] == current_symbol["module_path"]:
            score += 15
        if candidate["qualified_name"].startswith(f"{current_symbol['module_path']}::"):
            score += 10
        if current_symbol.get("container_qualified_name") and candidate["qualified_name"].startswith(
            f"{current_symbol['container_qualified_name']}::"
        ):
            score += 25
        if self_target and candidate["qualified_name"].startswith(f"{self_target}::"):
            score += 25
        if candidate["name"] == current_symbol.get("name"):
            score -= 5

        if best_score is None or score > best_score:
            best_candidate = candidate
            best_score = score
            tie = False
        elif score == best_score:
            tie = True

    if tie or best_score is None or best_score <= 0:
        return None
    return best_candidate


def preferred_resolution_roots(
    current_symbol: Dict[str, object],
    dependency_aliases: Dict[str, str],
    crate_root: str,
) -> List[str]:
    values = [
        str(current_symbol.get("module_path") or "").split("::")[0],
        str(current_symbol.get("crate") or "").replace("-", "_"),
        crate_root,
    ]
    values.extend(dependency_aliases.values())
    return [value for value in unique_values(values) if value]


def rollup_counts(values: Iterable[str]) -> List[Dict[str, object]]:
    counts: Dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return [
        {
            "kind": kind,
            "count": count,
        }
        for kind, count in sorted(counts.items(), key=lambda item: (item[0]))
    ]


def derive_module_path(crate_name: str, source_path: Path, crate_root: Path) -> str:
    source_root = crate_root / "src"
    base_name = crate_name.replace("-", "_")

    if source_path.is_relative_to(source_root):
        relative = source_path.relative_to(source_root)
    else:
        relative = source_path.relative_to(crate_root)

    parts = list(relative.parts)
    if not parts:
        return base_name

    filename = parts[-1]
    if filename in {"lib.rs", "main.rs", "mod.rs"}:
        parts = parts[:-1]
    else:
        parts[-1] = Path(filename).stem

    normalized_parts = [base_name]
    normalized_parts.extend(part.replace("-", "_") for part in parts)
    return "::".join(part for part in normalized_parts if part)


def empty_resolution() -> Dict[str, Optional[str]]:
    return {
        "target_symbol_id": None,
        "target_qualified_name": None,
        "target_kind": None,
        "qualified_name_hint": None,
    }


def span_to_dict(span: TextSpan) -> Dict[str, int]:
    return {
        "start_line": span.start_line,
        "start_column": span.start_column,
        "end_line": span.end_line,
        "end_column": span.end_column,
    }


def span_from_dict(payload: Dict[str, object]) -> TextSpan:
    return TextSpan(
        start_line=int(payload["start_line"]),
        start_column=int(payload["start_column"]),
        end_line=int(payload["end_line"]),
        end_column=int(payload["end_column"]),
    )


def split_top_level(value: str) -> List[str]:
    depth = 0
    current: List[str] = []
    items: List[str] = []

    for char in value:
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        elif char == "," and depth == 0:
            item = "".join(current).strip()
            if item:
                items.append(item)
            current = []
            continue
        current.append(char)

    tail = "".join(current).strip()
    if tail:
        items.append(tail)
    return items


def split_top_level_alias(value: str) -> Tuple[str, Optional[str]]:
    depth = 0
    for index in range(len(value) - 1):
        char = value[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        elif depth == 0 and value[index : index + 4] == " as ":
            return value[:index].strip(), value[index + 4 :].strip()
    return value, None


def stable_id(prefix: str, *parts: str) -> str:
    payload = "|".join(parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha1(payload).hexdigest()[:16]}"


def symbol_stable_id(repo_name: str, path: str, symbol: RustSymbol) -> str:
    return stable_id(
        "sym",
        repo_name,
        path,
        str(symbol.local_id),
        symbol.kind,
        symbol.qualified_name,
        str(symbol.span.start_line),
        str(symbol.span.start_column),
        str(symbol.span.end_line),
        str(symbol.span.end_column),
        collapse_whitespace(symbol.signature),
        symbol.visibility,
        ",".join(symbol.attributes),
        symbol.impl_target or "",
        symbol.impl_trait or "",
        ",".join(symbol.super_traits),
    )


def strip_expression_noise(expression: str) -> str:
    value = collapse_whitespace(expression)
    previous = None
    while previous != value:
        previous = value
        value = GENERIC_ANGLE_RE.sub("", value)

    value = value.replace("&", " ").replace("*", " ")
    value = re.sub(r"\b(?:dyn|impl|mut|ref)\b", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value.strip(",;(){}[] ")


def symbol_to_record(repo_name: str, context: ParsedFileContext, symbol: RustSymbol) -> Dict[str, object]:
    scope_symbol_id = context.symbol_id_by_local.get(symbol.container_local_id) if symbol.kind == "local" else None
    symbol_id = context.symbol_id_by_local[symbol.local_id]
    return {
        "symbol_id": symbol_id,
        "repo": repo_name,
        "path": context.parsed.path,
        "crate": context.parsed.crate_name,
        "package_name": context.package_info.package_name,
        "module_path": symbol.module_path,
        "language": "Rust",
        "kind": symbol.kind,
        "name": symbol.name,
        "qualified_name": symbol.qualified_name,
        "span": span_to_dict(symbol.span),
        "signature": symbol.signature,
        "docstring": symbol.docstring,
        "visibility": symbol.visibility,
        "container_symbol_id": context.symbol_id_by_local.get(symbol.container_local_id),
        "container_qualified_name": symbol.container_qualified_name,
        "statement_id": None,
        "scope_symbol_id": scope_symbol_id,
        "reference_target_symbol_id": None,
        "attributes": list(symbol.attributes),
        "is_test": symbol.is_test,
        "impl_target": symbol.impl_target,
        "impl_trait": symbol.impl_trait,
        "super_traits": list(symbol.super_traits),
        "resolved_impl_target_symbol_id": None,
        "resolved_impl_target_qualified_name": None,
        "resolved_impl_trait_symbol_id": None,
        "resolved_impl_trait_qualified_name": None,
        "resolved_super_traits": [],
        "summary_id": stable_id("sum", repo_name, "symbol", symbol_id),
        "normalized_body_hash": None,
        "semantic_summary": {
            "direct_calls": [],
            "transitive_calls": [],
            "reads": [],
            "writes": [],
            "references": [],
            "interprocedural_reads": [],
            "interprocedural_writes": [],
            "interprocedural_references": [],
        },
    }


def enrich_symbol_artifact_metadata(
    symbol_records: Sequence[Dict[str, object]],
    statement_records: Sequence[Dict[str, object]],
) -> None:
    statements_by_symbol: DefaultDict[str, List[Dict[str, object]]] = defaultdict(list)
    for statement in statement_records:
        statements_by_symbol[str(statement["container_symbol_id"])].append(statement)

    for symbol in symbol_records:
        chunks = [
            collapse_whitespace(str(statement.get("text") or ""))
            for statement in sorted(
                statements_by_symbol.get(symbol["symbol_id"], []),
                key=lambda item: (
                    int(item["span"]["start_line"]),
                    int(item["span"]["start_column"]),
                    str(item["statement_id"]),
                ),
            )
            if str(statement.get("text") or "").strip()
        ]
        normalized_body = "\n".join(chunks) if chunks else collapse_whitespace(str(symbol.get("signature") or ""))
        symbol["normalized_body_hash"] = hashlib.sha1(normalized_body.encode("utf-8")).hexdigest() if normalized_body else None


def timestamp_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def find_duplicate_ids(rows: Sequence[Dict[str, object]], field: str) -> List[str]:
    counts: DefaultDict[str, int] = defaultdict(int)
    for row in rows:
        value = str(row.get(field) or "")
        if value:
            counts[value] += 1
    return [value for value, count in counts.items() if count > 1]


def unique_positioned_values(values: Iterable[Tuple[str, int, int]]) -> List[Tuple[str, int, int]]:
    seen = set()
    ordered = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def unique_values(values: Iterable[str]) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def filter_shadowed_simple_tokens(values: List[Tuple[str, int, int]]) -> List[Tuple[str, int, int]]:
    qualified_by_line: Dict[int, List[str]] = defaultdict(list)
    for expression, line_number, _column in values:
        if "::" in expression:
            qualified_by_line[line_number].append(expression)

    filtered = []
    for expression, line_number, column in values:
        if "::" not in expression and any(path.endswith(f"::{expression}") for path in qualified_by_line[line_number]):
            continue
        filtered.append((expression, line_number, column))
    return filtered


def collapse_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()
