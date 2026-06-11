from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Callable, Dict, Iterable, List, Tuple

from common.text import path_terms
from graph.query import inspect_graph_backend_payload_uncached
from graph.store import write_graph_database
from symbols.indexer import stable_id, timestamp_now
from symbols.persistence import load_symbol_index, write_metadata_bundle


SCHEMA_VERSION = "0.2.0"


def build_summary_artifacts(
    repo_name: str,
    raw_root: Path,
    parsed_root: Path,
    graph_root: Path,
    progress_callback: Callable[[Dict[str, object]], None] | None = None,
) -> Dict[str, object]:
    started = time.perf_counter()

    def emit(event: str, **extra: object) -> None:
        if progress_callback is None:
            return
        progress_callback(
            {
                "event": event,
                "repo": repo_name,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                **extra,
            }
        )

    emit("build_started")
    manifest = load_json(raw_root / repo_name / "manifest.json")
    emit("loaded_manifest")
    repo_map = load_json(raw_root / repo_name / "repo_map.json")
    emit(
        "loaded_repo_map",
        files=len(repo_map.get("files", [])),
        directories=len(repo_map.get("directories", [])),
    )
    symbols = load_symbol_index(parsed_root, repo_name)
    emit(
        "loaded_symbols",
        files=len(symbols.get("files", [])),
        symbols=len(symbols.get("symbols", [])),
        statements=len(symbols.get("statements", [])),
    )
    graph = inspect_graph_backend_payload_uncached(graph_root, repo_name)["payload"]
    emit(
        "loaded_graph",
        nodes=len(graph.get("nodes", [])),
        edges=len(graph.get("edges", [])),
    )

    symbol_records = list(symbols.get("symbols", []))
    symbols_by_path: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for symbol in symbol_records:
        symbols_by_path[symbol["path"]].append(symbol)

    incoming_counts, outgoing_counts = edge_counts(graph)
    emit("building_project_summary")
    project_summary = build_project_summary(repo_name, manifest, symbols, graph)
    emit("building_package_summaries")
    package_summaries = build_package_summaries(symbols)
    emit("building_directory_summaries")
    directory_summaries = build_directory_summaries(repo_map, symbols)
    emit("building_file_summaries")
    file_summaries = build_file_summaries(symbols, symbols_by_path)
    emit("building_symbol_summaries")
    symbol_summaries = build_symbol_summaries(symbol_records, incoming_counts, outgoing_counts)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "repo": repo_name,
        "generated_at": timestamp_now(),
        "project": project_summary,
        "packages": package_summaries,
        "directories": directory_summaries,
        "files": file_summaries,
        "symbols": symbol_summaries,
        "summary": {
            "packages": len(package_summaries),
            "directories": len(directory_summaries),
            "files": len(file_summaries),
            "symbols": len(symbol_summaries),
        },
    }
    emit(
        "build_completed",
        packages=len(package_summaries),
        directories=len(directory_summaries),
        files=len(file_summaries),
        symbols=len(symbol_summaries),
    )
    return payload
def build_project_summary(
    repo_name: str,
    manifest: Dict[str, object],
    symbols: Dict[str, object],
    graph: Dict[str, object],
) -> Dict[str, object]:
    focus = infer_repo_focus(manifest)
    source_roots = list(manifest.get("parser_relevant_source_roots", []))
    language_mix = [f"{item['language']}:{item['files']}" for item in manifest.get("language_mix", [])[:5]]
    kind_counts = symbols.get("summary", {}).get("kind_counts", [])
    top_kinds = [f"{item['kind']}:{item['count']}" for item in kind_counts[:6]]
    summary = (
        f"{repo_name} is indexed as {focus}. "
        f"The current analysis slice covers {symbols['summary']['rust_files']} Rust files, "
        f"{symbols['summary']['symbols']} symbols, {symbols['summary']['imports']} imports, "
        f"{symbols['summary'].get('statements', 0)} statements, "
        f"and {graph['summary']['edges']} graph edges. "
        f"Indexed source roots: {', '.join(source_roots) or 'none detected'}."
    )
    return {
        "summary_id": stable_id("sum", repo_name, "project"),
        "repo": repo_name,
        "focus": focus,
        "analysis_surfaces": list(manifest.get("module_graph_seeds", {}).get("analysis_surfaces", [])),
        "parser_relevant_source_roots": source_roots,
        "build_commands": list(manifest.get("build_commands", [])),
        "test_commands": list(manifest.get("test_commands", [])),
        "language_mix": language_mix,
        "top_symbol_kinds": top_kinds,
        "summary": summary,
    }


def build_package_summaries(symbols: Dict[str, object]) -> List[Dict[str, object]]:
    rollups: Dict[str, Dict[str, object]] = defaultdict(
        lambda: {
            "files": set(),
            "module_paths": set(),
            "symbols": 0,
            "public_symbols": 0,
            "statements": 0,
            "top_symbol_kinds": Counter(),
        }
    )
    for file_record in symbols.get("files", []):
        package_name = str(file_record.get("package_name") or file_record.get("crate") or "")
        if not package_name:
            continue
        rollups[package_name]["files"].add(file_record["path"])
        if file_record.get("module_path"):
            rollups[package_name]["module_paths"].add(file_record["module_path"])

    for symbol in symbols.get("symbols", []):
        package_name = str(symbol.get("package_name") or symbol.get("crate") or "")
        if not package_name:
            continue
        rollups[package_name]["symbols"] += 1
        if str(symbol.get("visibility") or "").startswith("pub"):
            rollups[package_name]["public_symbols"] += 1
        rollups[package_name]["top_symbol_kinds"][symbol["kind"]] += 1

    for statement in symbols.get("statements", []):
        package_name = str(statement.get("crate") or "")
        if not package_name:
            continue
        rollups[package_name]["statements"] += 1

    summaries = []
    for package_name, rollup in sorted(rollups.items()):
        top_kinds = [kind for kind, _count in rollup["top_symbol_kinds"].most_common(4)]
        summaries.append(
            {
                "summary_id": stable_id("sum", "package", package_name),
                "package_name": package_name,
                "files": len(rollup["files"]),
                "symbols": rollup["symbols"],
                "public_symbols": rollup["public_symbols"],
                "statements": rollup["statements"],
                "top_symbol_kinds": top_kinds,
                "module_paths": sorted(rollup["module_paths"])[:8],
                "summary": (
                    f"{package_name} contains {len(rollup['files'])} files, "
                    f"{rollup['symbols']} indexed symbols, and {rollup['statements']} statements. "
                    f"Top symbol kinds: {', '.join(top_kinds) or 'none'}."
                ),
            }
        )
    return summaries


def build_directory_summaries(repo_map: Dict[str, object], symbols: Dict[str, object]) -> List[Dict[str, object]]:
    rollups: Dict[str, Dict[str, object]] = defaultdict(
        lambda: {
            "files": 0,
            "rust_files": 0,
            "symbols": 0,
            "statements": 0,
            "public_symbols": 0,
            "top_symbol_kinds": Counter(),
        }
    )
    for file_record in repo_map.get("files", []):
        for prefix in directory_prefixes(file_record["path"]):
            rollups[prefix]["files"] += 1
            if file_record.get("language") == "Rust":
                rollups[prefix]["rust_files"] += 1

    for symbol in symbols.get("symbols", []):
        for prefix in directory_prefixes(symbol["path"]):
            rollups[prefix]["symbols"] += 1
            if str(symbol.get("visibility") or "").startswith("pub"):
                rollups[prefix]["public_symbols"] += 1
            rollups[prefix]["top_symbol_kinds"][symbol["kind"]] += 1

    for statement in symbols.get("statements", []):
        for prefix in directory_prefixes(statement["path"]):
            rollups[prefix]["statements"] += 1

    summaries = []
    for directory in repo_map.get("directories", []):
        path = directory["path"]
        rollup = rollups.get(path, {})
        top_kinds = [kind for kind, _count in rollup.get("top_symbol_kinds", Counter()).most_common(4)]
        tags = path_tags(path)
        summaries.append(
            {
                "summary_id": stable_id("sum", "directory", path),
                "path": path,
                "depth": directory["depth"],
                "files": rollup.get("files", 0),
                "rust_files": rollup.get("rust_files", 0),
                "symbols": rollup.get("symbols", 0),
                "statements": rollup.get("statements", 0),
                "public_symbols": rollup.get("public_symbols", 0),
                "top_symbol_kinds": top_kinds,
                "tags": tags,
                "summary": (
                    f"{path} contains {rollup.get('files', 0)} files, "
                    f"{rollup.get('rust_files', 0)} Rust files, {rollup.get('symbols', 0)} indexed symbols, "
                    f"and {rollup.get('statements', 0)} statements."
                ),
            }
        )
    return summaries


def build_file_summaries(
    symbols: Dict[str, object],
    symbols_by_path: Dict[str, List[Dict[str, object]]],
) -> List[Dict[str, object]]:
    files = []
    file_records = {item["path"]: item for item in symbols.get("files", [])}
    statement_counts = Counter(statement["path"] for statement in symbols.get("statements", []))
    for path, file_record in sorted(file_records.items()):
        file_symbols = symbols_by_path.get(path, [])
        public_symbols = [symbol["qualified_name"] for symbol in file_symbols if str(symbol.get("visibility") or "").startswith("pub")]
        top_symbols = [symbol["qualified_name"] for symbol in file_symbols[:6]]
        tags = path_tags(path)
        files.append(
            {
                "summary_id": stable_id("sum", "file", path),
                "path": path,
                "crate": file_record.get("crate"),
                "package_name": file_record.get("package_name"),
                "module_path": file_record.get("module_path"),
                "language": file_record.get("language"),
                "symbols": len(file_symbols),
                "imports": file_record.get("imports", 0),
                "statements": statement_counts.get(path, 0),
                "public_symbols": public_symbols[:8],
                "top_symbols": top_symbols,
                "tags": tags,
                "summary": (
                    f"{path} defines {len(file_symbols)} symbols and {statement_counts.get(path, 0)} statements "
                    f"in crate {file_record.get('crate')}. "
                    f"Top symbols: {', '.join(top_symbols[:3]) or 'none'}."
                ),
            }
        )
    return files


def build_symbol_summaries(
    symbol_records: Iterable[Dict[str, object]],
    incoming_counts: Dict[str, Counter],
    outgoing_counts: Dict[str, Counter],
) -> List[Dict[str, object]]:
    summaries = []
    for symbol in symbol_records:
        incoming = incoming_counts.get(symbol["symbol_id"], Counter())
        outgoing = outgoing_counts.get(symbol["symbol_id"], Counter())
        summaries.append(
            {
                "summary_id": symbol.get("summary_id") or stable_id("sum", "symbol", symbol["symbol_id"]),
                "symbol_id": symbol["symbol_id"],
                "path": symbol["path"],
                "kind": symbol["kind"],
                "name": symbol["name"],
                "qualified_name": symbol["qualified_name"],
                "visibility": symbol["visibility"],
                "container_qualified_name": symbol["container_qualified_name"],
                "incoming_edges": dict(sorted(incoming.items())),
                "outgoing_edges": dict(sorted(outgoing.items())),
                "summary": (
                    f"{symbol['kind']} {symbol['qualified_name']} is defined in {symbol['path']}. "
                    f"Incoming edges: {format_edge_counts(incoming)}. "
                    f"Outgoing edges: {format_edge_counts(outgoing)}."
                ),
            }
        )
    return summaries


def sync_summary_state(
    parsed_root: Path,
    graph_root: Path,
    repo_name: str,
    payload: Dict[str, object],
    *,
    artifact_metadata: Dict[str, object] | None = None,
) -> None:
    symbol_payload = load_symbol_index(parsed_root, repo_name)
    write_metadata_bundle(
        parsed_root,
        repo_name,
        symbol_payload,
        summaries_payload=payload,
        artifact_metadata=artifact_metadata,
    )
    augment_graph_with_summaries(graph_root, repo_name, payload)


def augment_graph_with_summaries(
    graph_root: Path,
    repo_name: str,
    payload: Dict[str, object],
) -> None:
    graph = inspect_graph_backend_payload_uncached(graph_root, repo_name)["payload"]
    node_by_id = {node["node_id"]: node for node in graph.get("nodes", [])}
    edge_ids = {edge["edge_id"] for edge in graph.get("edges", [])}

    def append_summary_node(node_id: str, kind: str, title: str, path: str | None, symbol_id: str | None, summary: str) -> None:
        if node_id in node_by_id:
            return
        node = {
            "node_id": node_id,
            "kind": kind,
            "repo": repo_name,
            "path": path,
            "name": title,
            "qualified_name": symbol_id,
            "summary": summary,
        }
        graph.setdefault("nodes", []).append(node)
        node_by_id[node_id] = node

    def append_summary_edge(source_id: str, target_id: str, scope: str, path: str | None = None) -> None:
        edge = {
            "edge_id": stable_id("edge", repo_name, "SUMMARIZED_BY", source_id, target_id, scope),
            "type": "SUMMARIZED_BY",
            "from": source_id,
            "to": target_id,
            "metadata": {
                "scope": scope,
                "path": path,
            },
        }
        if edge["edge_id"] in edge_ids:
            return
        edge_ids.add(edge["edge_id"])
        graph.setdefault("edges", []).append(edge)

    repo_node_id = stable_id("repo", repo_name)
    project = payload.get("project") or {}
    project_summary_id = str(project.get("summary_id") or stable_id("sum", repo_name, "project"))
    append_summary_node(project_summary_id, "project_summary", repo_name, None, None, str(project.get("summary") or ""))
    append_summary_edge(repo_node_id, project_summary_id, "project")

    for package in payload.get("packages", []):
        summary_id = str(package.get("summary_id"))
        append_summary_node(summary_id, "package_summary", str(package.get("package_name") or summary_id), None, None, str(package.get("summary") or ""))
        package_node_id = stable_id("pkg", repo_name, str(package.get("package_name") or ""))
        if package_node_id in node_by_id:
            append_summary_edge(package_node_id, summary_id, "package")

    for directory in payload.get("directories", []):
        summary_id = str(directory.get("summary_id"))
        path = str(directory.get("path") or "")
        append_summary_node(summary_id, "directory_summary", path or ".", path or None, None, str(directory.get("summary") or ""))
        directory_node_id = stable_id("dir", repo_name, path or ".")
        if directory_node_id in node_by_id:
            append_summary_edge(directory_node_id, summary_id, "directory", path or None)

    for file_summary in payload.get("files", []):
        summary_id = str(file_summary.get("summary_id"))
        path = str(file_summary.get("path") or "")
        append_summary_node(summary_id, "file_summary", path, path or None, None, str(file_summary.get("summary") or ""))
        file_node_id = stable_id("file", repo_name, path)
        if file_node_id in node_by_id:
            append_summary_edge(file_node_id, summary_id, "file", path)

    for symbol_summary in payload.get("symbols", []):
        summary_id = str(symbol_summary.get("summary_id"))
        symbol_id = str(symbol_summary.get("symbol_id") or "")
        append_summary_node(
            summary_id,
            "symbol_summary",
            str(symbol_summary.get("qualified_name") or symbol_summary.get("name") or summary_id),
            symbol_summary.get("path"),
            symbol_id or None,
            str(symbol_summary.get("summary") or ""),
        )
        if symbol_id in node_by_id:
            append_summary_edge(symbol_id, summary_id, "symbol", symbol_summary.get("path"))

    edge_counts = Counter(edge["type"] for edge in graph.get("edges", []))
    graph["summary"] = {
        **graph.get("summary", {}),
        "nodes": len(graph.get("nodes", [])),
        "edges": len(graph.get("edges", [])),
        "edge_counts": [
            {"type": edge_type, "count": count}
            for edge_type, count in sorted(edge_counts.items())
        ],
    }
    write_graph_database(graph_root, repo_name, graph)


def edge_counts(graph: Dict[str, object]) -> Tuple[Dict[str, Counter], Dict[str, Counter]]:
    incoming: Dict[str, Counter] = defaultdict(Counter)
    outgoing: Dict[str, Counter] = defaultdict(Counter)
    for edge in graph.get("edges", []):
        incoming[edge["to"]][edge["type"]] += 1
        outgoing[edge["from"]][edge["type"]] += 1
    return incoming, outgoing


def format_edge_counts(counts: Counter) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{edge_type}:{count}" for edge_type, count in sorted(counts.items()))


def directory_prefixes(path: str) -> List[str]:
    parts = PurePosixPath(path).parts[:-1]
    prefixes = ["."]
    current: List[str] = []
    for part in parts:
        current.append(part)
        prefixes.append("/".join(current))
    return prefixes


def path_tags(path: str) -> List[str]:
    return path_terms(path)


def infer_repo_focus(manifest: Dict[str, object]) -> str:
    analysis_surfaces = list(manifest.get("module_graph_seeds", {}).get("analysis_surfaces", []))
    source_roots = list(manifest.get("parser_relevant_source_roots", []))
    language_mix = [str(item.get("language") or "").strip() for item in manifest.get("language_mix", [])]

    if analysis_surfaces:
        return f"a structure-first codebase centered on {', '.join(analysis_surfaces[:4])}"
    if source_roots:
        return f"a structure-first codebase rooted at {', '.join(source_roots[:4])}"

    primary_languages = [language for language in language_mix if language]
    if primary_languages:
        return f"a structure-first codebase built mostly in {', '.join(primary_languages[:3])}"
    return "a structure-first codebase"


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
