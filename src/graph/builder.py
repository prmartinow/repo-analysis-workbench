from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Dict, List

from symbols.indexer import stable_id, timestamp_now


def build_graph_artifact(symbol_index: Dict[str, object]) -> Dict[str, object]:
    repo_name = symbol_index["repo"]
    edge_counts: Dict[str, int] = {}
    file_node_ids: Dict[str, str] = {}
    directory_node_ids: Dict[str, str] = {}
    package_node_ids: Dict[str, str] = {}
    test_node_ids: Dict[str, str] = {}
    nodes: List[Dict[str, object]] = []
    edges: List[Dict[str, object]] = []
    node_ids = set()
    reference_nodes: Dict[str, str] = {}
    symbol_by_id = {symbol["symbol_id"]: symbol for symbol in symbol_index["symbols"]}
    symbol_by_qname = {symbol["qualified_name"]: symbol for symbol in symbol_index["symbols"]}
    local_crate_roots = {str(item.get("crate") or "").replace("-", "_") for item in symbol_index["files"]}
    local_package_names = {str(item.get("package_name") or "") for item in symbol_index["files"] if item.get("package_name")}

    repo_node_id = stable_id("repo", repo_name)
    append_node(
        nodes,
        node_ids,
        {
            "node_id": repo_node_id,
            "kind": "repository",
            "repo": repo_name,
            "name": repo_name,
        },
    )

    append_directory_node(nodes, node_ids, directory_node_ids, repo_name, ".")
    append_edge(
        edges,
        edge_counts,
        make_edge(
            "CONTAINS",
            repo_node_id,
            directory_node_ids["."],
            repo_name,
            path=".",
        ),
    )

    for package_name in sorted(local_package_names):
        package_node_id = stable_id("pkg", repo_name, package_name)
        package_node_ids[package_name] = package_node_id
        append_node(
            nodes,
            node_ids,
            {
                "node_id": package_node_id,
                "kind": "package",
                "repo": repo_name,
                "name": package_name,
                "qualified_name": package_name,
            },
        )
        append_edge(
            edges,
            edge_counts,
            make_edge("CONTAINS", repo_node_id, package_node_id, repo_name, package_name=package_name),
        )

    for file_record in symbol_index["files"]:
        node_id = stable_id("file", repo_name, file_record["path"])
        file_node_ids[file_record["path"]] = node_id
        ensure_directory_path(
            nodes,
            node_ids,
            edges,
            edge_counts,
            directory_node_ids,
            repo_name,
            repo_node_id,
            file_record["path"],
        )
        append_node(
            nodes,
            node_ids,
            {
                "node_id": node_id,
                "kind": "file",
                "repo": repo_name,
                "path": file_record["path"],
                "crate": file_record["crate"],
                "package_name": file_record.get("package_name"),
                "module_path": file_record["module_path"],
                "language": file_record["language"],
            },
        )
        append_edge(
            edges,
            edge_counts,
            make_edge(
                "CONTAINS",
                repo_node_id,
                node_id,
                repo_name,
                path=file_record["path"],
            ),
        )
        parent_directory = deepest_directory_prefix(file_record["path"])
        append_edge(
            edges,
            edge_counts,
            make_edge(
                "CONTAINS",
                directory_node_ids[parent_directory],
                node_id,
                repo_name,
                path=file_record["path"],
            ),
        )
        package_name = str(file_record.get("package_name") or "")
        if package_name and package_name in package_node_ids:
            append_edge(
                edges,
                edge_counts,
                make_edge(
                    "CONTAINS",
                    package_node_ids[package_name],
                    node_id,
                    repo_name,
                    path=file_record["path"],
                    package_name=package_name,
                ),
            )

    for symbol in symbol_index["symbols"]:
        append_node(
            nodes,
            node_ids,
            {
                "node_id": symbol["symbol_id"],
                "kind": symbol["kind"],
                "repo": repo_name,
                "path": symbol["path"],
                "name": symbol["name"],
                "qualified_name": symbol["qualified_name"],
                "crate": symbol["crate"],
                "package_name": symbol.get("package_name"),
                "module_path": symbol["module_path"],
                "language": symbol["language"],
                "visibility": symbol["visibility"],
                "is_test": symbol["is_test"],
            },
        )
        parent_node = symbol["container_symbol_id"] or file_node_ids[symbol["path"]]
        edge_type = "CONTAINS" if symbol["container_symbol_id"] else "DEFINES"
        append_edge(
            edges,
            edge_counts,
            make_edge(
                edge_type,
                parent_node,
                symbol["symbol_id"],
                repo_name,
                path=symbol["path"],
            ),
        )
        if not symbol["container_symbol_id"]:
            append_edge(
                edges,
                edge_counts,
                make_edge(
                    "DECLARES",
                    file_node_ids[symbol["path"]],
                    symbol["symbol_id"],
                    repo_name,
                    path=symbol["path"],
                ),
            )

        if symbol.get("is_test"):
            test_node_id = stable_id("test", repo_name, symbol["symbol_id"])
            test_node_ids[symbol["symbol_id"]] = test_node_id
            append_node(
                nodes,
                node_ids,
                {
                    "node_id": test_node_id,
                    "kind": "test",
                    "repo": repo_name,
                    "path": symbol["path"],
                    "name": symbol["name"],
                    "qualified_name": symbol["qualified_name"],
                },
            )
            append_edge(
                edges,
                edge_counts,
                make_edge(
                    "DECLARES",
                    symbol["symbol_id"],
                    test_node_id,
                    repo_name,
                    path=symbol["path"],
                ),
            )

        if symbol["kind"] == "impl":
            target_node_id = resolve_target_node(
                nodes,
                node_ids,
                reference_nodes,
                repo_name,
                symbol.get("resolved_impl_target_symbol_id"),
                symbol.get("resolved_impl_target_qualified_name") or symbol.get("impl_target"),
                "type_ref",
            )
            if target_node_id:
                append_edge(
                    edges,
                    edge_counts,
                    make_edge(
                        "REFERENCES",
                        symbol["symbol_id"],
                        target_node_id,
                        repo_name,
                        path=symbol["path"],
                        role="impl_target",
                    ),
                )

            trait_node_id = resolve_target_node(
                nodes,
                node_ids,
                reference_nodes,
                repo_name,
                symbol.get("resolved_impl_trait_symbol_id"),
                symbol.get("resolved_impl_trait_qualified_name") or symbol.get("impl_trait"),
                "trait_ref",
            )
            if trait_node_id:
                append_edge(
                    edges,
                    edge_counts,
                    make_edge(
                        "IMPLEMENTS",
                        symbol["symbol_id"],
                        trait_node_id,
                        repo_name,
                        path=symbol["path"],
                    ),
                )

        if symbol["kind"] == "trait":
            for parent in symbol.get("resolved_super_traits", []):
                parent_node_id = resolve_target_node(
                    nodes,
                    node_ids,
                    reference_nodes,
                    repo_name,
                    parent.get("target_symbol_id"),
                    parent.get("target_qualified_name") or parent.get("qualified_name_hint"),
                    "trait_ref",
                )
                if not parent_node_id:
                    continue
                append_edge(
                    edges,
                    edge_counts,
                    make_edge(
                        "INHERITS",
                        symbol["symbol_id"],
                        parent_node_id,
                        repo_name,
                        path=symbol["path"],
                    ),
                )

    for import_record in symbol_index["imports"]:
        target_qname = import_record.get("target_qualified_name") or import_record.get("normalized_target") or import_record.get("target")
        target_node_id = resolve_import_target_node(
            nodes,
            node_ids,
            reference_nodes,
            repo_name,
            import_record.get("target_symbol_id"),
            target_qname,
            local_crate_roots=local_crate_roots,
            local_package_names=local_package_names,
        )
        if not target_node_id:
            continue
        source_node_id = import_record["container_symbol_id"] or file_node_ids[import_record["path"]]
        append_edge(
            edges,
            edge_counts,
            make_edge(
                "IMPORTS",
                source_node_id,
                target_node_id,
                repo_name,
                path=import_record["path"],
                visibility=import_record["visibility"],
            ),
        )

    for reference in symbol_index.get("references", []):
        target_node_id = resolve_target_node(
            nodes,
            node_ids,
            reference_nodes,
            repo_name,
            reference.get("target_symbol_id"),
            reference.get("target_qualified_name") or reference.get("qualified_name_hint"),
            "symbol_ref",
        )
        if not target_node_id:
            continue
        source_node_id = reference["container_symbol_id"] or file_node_ids[reference["path"]]
        edge_type = "CALLS" if reference["kind"] == "call" else "USES"
        append_edge(
            edges,
            edge_counts,
            make_edge(
                edge_type,
                source_node_id,
                target_node_id,
                repo_name,
                path=reference["path"],
                line=reference["span"]["start_line"],
                kind=reference["kind"],
            ),
        )
        if reference["kind"] == "use":
            append_edge(
                edges,
                edge_counts,
                make_edge(
                    "USES_TYPE",
                    source_node_id,
                    target_node_id,
                    repo_name,
                    path=reference["path"],
                    line=reference["span"]["start_line"],
                    kind=reference["kind"],
                ),
            )
        source_symbol = symbol_by_id.get(reference["container_symbol_id"])
        if source_symbol and source_symbol.get("is_test"):
            append_edge(
                edges,
                edge_counts,
                make_edge(
                    "TESTS",
                    test_node_ids.get(source_symbol["symbol_id"], source_node_id),
                    target_node_id,
                    repo_name,
                    path=reference["path"],
                    line=reference["span"]["start_line"],
                    kind=reference["kind"],
                ),
            )

    append_symbol_semantic_edges(
        nodes,
        node_ids,
        edges,
        edge_counts,
        reference_nodes,
        repo_name,
        symbol_index.get("symbols", []),
    )
    append_override_edges(
        edges,
        edge_counts,
        repo_name,
        symbol_index.get("symbols", []),
        symbol_by_id,
        symbol_by_qname,
    )
    append_neighbor_edges(edges, edge_counts, repo_name, symbol_index.get("symbols", []))

    append_statement_graph(
        nodes,
        node_ids,
        edges,
        edge_counts,
        reference_nodes,
        file_node_ids,
        repo_name,
        symbol_index.get("statements", []),
    )
    edges = dedupe_edges(edges)
    edge_counts = rollup_edge_counts(edges)

    return {
        "schema_version": "0.4.0",
        "repo": repo_name,
        "generated_at": timestamp_now(),
        "nodes": nodes,
        "edges": edges,
        "summary": {
            "nodes": len(nodes),
            "edges": len(edges),
            "edge_counts": [
                {
                    "type": edge_type,
                    "count": count,
                }
                for edge_type, count in sorted(edge_counts.items())
            ],
        },
    }


def append_edge(edges: List[Dict[str, object]], edge_counts: Dict[str, int], edge: Dict[str, object]) -> None:
    edges.append(edge)
    edge_counts[edge["type"]] = edge_counts.get(edge["type"], 0) + 1


def append_node(nodes: List[Dict[str, object]], node_ids: set[str], node: Dict[str, object]) -> None:
    if node["node_id"] in node_ids:
        return
    node_ids.add(node["node_id"])
    nodes.append(node)


def ensure_reference_node(
    nodes: List[Dict[str, object]],
    node_ids: set[str],
    reference_nodes: Dict[str, str],
    repo_name: str,
    qualified_name: str,
    kind: str,
) -> str:
    node_id = reference_nodes.get(qualified_name)
    if node_id:
        return node_id

    node_id = stable_id("ref", repo_name, qualified_name)
    reference_nodes[qualified_name] = node_id
    append_node(
        nodes,
        node_ids,
        {
            "node_id": node_id,
            "kind": kind,
            "repo": repo_name,
            "name": qualified_name.split("::")[-1],
            "qualified_name": qualified_name,
        },
    )
    return node_id


def append_directory_node(
    nodes: List[Dict[str, object]],
    node_ids: set[str],
    directory_node_ids: Dict[str, str],
    repo_name: str,
    path: str,
) -> str:
    node_id = directory_node_ids.get(path)
    if node_id:
        return node_id
    node_id = stable_id("dir", repo_name, path)
    directory_node_ids[path] = node_id
    append_node(
        nodes,
        node_ids,
        {
            "node_id": node_id,
            "kind": "directory",
            "repo": repo_name,
            "path": path,
            "name": path.rsplit("/", 1)[-1] if path != "." else ".",
            "qualified_name": path,
        },
    )
    return node_id


def ensure_directory_path(
    nodes: List[Dict[str, object]],
    node_ids: set[str],
    edges: List[Dict[str, object]],
    edge_counts: Dict[str, int],
    directory_node_ids: Dict[str, str],
    repo_name: str,
    repo_node_id: str,
    file_path: str,
) -> None:
    prefixes = directory_prefixes(file_path)
    for prefix in prefixes:
        append_directory_node(nodes, node_ids, directory_node_ids, repo_name, prefix)
    for prefix in prefixes:
        parent = parent_directory(prefix)
        if parent is None:
            continue
        append_edge(
            edges,
            edge_counts,
            make_edge(
                "CONTAINS",
                directory_node_ids[parent],
                directory_node_ids[prefix],
                repo_name,
                path=prefix,
            ),
        )


def directory_prefixes(path: str) -> List[str]:
    parts = PurePosixPath(path).parts[:-1]
    prefixes = ["."]
    current: List[str] = []
    for part in parts:
        current.append(part)
        prefixes.append("/".join(current))
    return prefixes


def parent_directory(path: str) -> str | None:
    if path == ".":
        return None
    parent = str(PurePosixPath(path).parent)
    return "." if parent == "." or parent == "" else parent


def deepest_directory_prefix(path: str) -> str:
    prefixes = directory_prefixes(path)
    return prefixes[-1] if prefixes else "."


def make_edge(edge_type: str, source: str, target: str, repo_name: str, **metadata: object) -> Dict[str, object]:
    return {
        "edge_id": stable_id("edge", repo_name, edge_type, source, target, json.dumps(metadata, sort_keys=True)),
        "type": edge_type,
        "from": source,
        "to": target,
        "metadata": metadata,
    }


def dedupe_edges(edges: List[Dict[str, object]]) -> List[Dict[str, object]]:
    deduped: List[Dict[str, object]] = []
    seen = set()
    for edge in edges:
        if edge["edge_id"] in seen:
            continue
        seen.add(edge["edge_id"])
        deduped.append(edge)
    return deduped


def rollup_edge_counts(edges: List[Dict[str, object]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for edge in edges:
        counts[edge["type"]] = counts.get(edge["type"], 0) + 1
    return counts


def append_statement_graph(
    nodes: List[Dict[str, object]],
    node_ids: set[str],
    edges: List[Dict[str, object]],
    edge_counts: Dict[str, int],
    reference_nodes: Dict[str, str],
    file_node_ids: Dict[str, str],
    repo_name: str,
    statements: List[Dict[str, object]],
) -> None:
    statements_by_container: Dict[str, List[Dict[str, object]]] = {}
    for statement in statements:
        statements_by_container.setdefault(statement["container_symbol_id"], []).append(statement)
        append_node(
            nodes,
            node_ids,
            {
                "node_id": statement["statement_id"],
                "kind": "statement",
                "repo": repo_name,
                "path": statement["path"],
                "name": f"{statement['kind']}@L{statement['span']['start_line']}",
                "text": statement["text"],
                "container_symbol_id": statement["container_symbol_id"],
                "container_qualified_name": statement["container_qualified_name"],
            },
        )
        append_edge(
            edges,
            edge_counts,
            make_edge(
                "CONTAINS",
                statement["container_symbol_id"],
                statement["statement_id"],
                repo_name,
                path=statement["path"],
                line=statement["span"]["start_line"],
            ),
        )

        if statement.get("previous_statement_id"):
            append_edge(
                edges,
                edge_counts,
                make_edge(
                    "CONTROL_FLOW",
                    statement["previous_statement_id"],
                    statement["statement_id"],
                    repo_name,
                    path=statement["path"],
                    line=statement["span"]["start_line"],
                    relation="sequential",
                ),
            )

        if statement.get("parent_statement_id"):
            append_edge(
                edges,
                edge_counts,
                make_edge(
                    "DEPENDENCE",
                    statement["parent_statement_id"],
                    statement["statement_id"],
                    repo_name,
                    path=statement["path"],
                    line=statement["span"]["start_line"],
                    relation="control",
                ),
            )

        for edge_type, field_name in (("DEFINES", "defines"), ("READS", "reads"), ("WRITES", "writes"), ("REFS", "reads"), ("CALLS", "calls")):
            for target in statement.get(field_name, []):
                target_node_id = resolve_target_node(
                    nodes,
                    node_ids,
                    reference_nodes,
                    repo_name,
                    target.get("target_symbol_id"),
                    target.get("target_qualified_name") or target.get("qualified_name_hint"),
                    "symbol_ref",
                )
                if not target_node_id:
                    continue
                append_edge(
                    edges,
                    edge_counts,
                    make_edge(
                        edge_type,
                        statement["statement_id"],
                        target_node_id,
                        repo_name,
                        path=statement["path"],
                        line=statement["span"]["start_line"],
                        kind=statement["kind"],
                    ),
                )

    for container_symbol_id, container_statements in statements_by_container.items():
        last_write_by_target: Dict[str, str] = {}
        sorted_statements = sorted(
            container_statements,
            key=lambda item: (item["span"]["start_line"], item["span"]["start_column"], item["statement_id"]),
        )
        for statement in sorted_statements:
            for read in statement.get("reads", []):
                target_key = statement_target_key(read)
                if not target_key or target_key not in last_write_by_target:
                    continue
                append_edge(
                    edges,
                    edge_counts,
                    make_edge(
                        "DATA_FLOW",
                        last_write_by_target[target_key],
                        statement["statement_id"],
                        repo_name,
                        path=statement["path"],
                        line=statement["span"]["start_line"],
                        via=target_key,
                    ),
                )
            for write in list(statement.get("defines", [])) + list(statement.get("writes", [])):
                target_key = statement_target_key(write)
                if target_key:
                    last_write_by_target[target_key] = statement["statement_id"]


def append_symbol_semantic_edges(
    nodes: List[Dict[str, object]],
    node_ids: set[str],
    edges: List[Dict[str, object]],
    edge_counts: Dict[str, int],
    reference_nodes: Dict[str, str],
    repo_name: str,
    symbols: List[Dict[str, object]],
) -> None:
    emitted = set()
    for symbol in symbols:
        semantic_summary = symbol.get("semantic_summary") or {}
        for edge_type, field_name, semantic_level in (
            ("READS", "reads", "symbol"),
            ("WRITES", "writes", "symbol"),
            ("REFS", "references", "symbol"),
            ("READS", "interprocedural_reads", "interprocedural"),
            ("WRITES", "interprocedural_writes", "interprocedural"),
            ("REFS", "interprocedural_references", "interprocedural"),
            ("CALLS", "transitive_calls", "transitive"),
        ):
            for target in semantic_summary.get(field_name, []):
                target_node_id = resolve_target_node(
                    nodes,
                    node_ids,
                    reference_nodes,
                    repo_name,
                    target.get("target_symbol_id"),
                    target.get("target_qualified_name") or target.get("qualified_name_hint"),
                    "symbol_ref",
                )
                if not target_node_id:
                    continue
                edge_key = (symbol["symbol_id"], edge_type, semantic_level, target_node_id)
                if edge_key in emitted:
                    continue
                emitted.add(edge_key)
                append_edge(
                    edges,
                    edge_counts,
                    make_edge(
                        edge_type,
                        symbol["symbol_id"],
                        target_node_id,
                        repo_name,
                        path=symbol["path"],
                        semantic_level=semantic_level,
                    ),
                )


def append_override_edges(
    edges: List[Dict[str, object]],
    edge_counts: Dict[str, int],
    repo_name: str,
    symbols: List[Dict[str, object]],
    symbol_by_id: Dict[str, Dict[str, object]],
    symbol_by_qname: Dict[str, Dict[str, object]],
) -> None:
    for symbol in symbols:
        if symbol["kind"] not in {"fn", "method"}:
            continue
        container_symbol_id = symbol.get("container_symbol_id")
        container_symbol = symbol_by_id.get(container_symbol_id) if container_symbol_id else None
        if not container_symbol or container_symbol.get("kind") != "impl":
            continue
        trait_qname = container_symbol.get("resolved_impl_trait_qualified_name")
        if not trait_qname:
            continue
        target_symbol = symbol_by_qname.get(f"{trait_qname}::{symbol['name']}")
        if not target_symbol:
            continue
        append_edge(
            edges,
            edge_counts,
            make_edge(
                "OVERRIDES",
                symbol["symbol_id"],
                target_symbol["symbol_id"],
                repo_name,
                path=symbol["path"],
            ),
        )


def append_neighbor_edges(
    edges: List[Dict[str, object]],
    edge_counts: Dict[str, int],
    repo_name: str,
    symbols: List[Dict[str, object]],
) -> None:
    siblings: Dict[tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for symbol in symbols:
        key = (symbol["path"], symbol.get("container_symbol_id") or "")
        siblings[key].append(symbol)

    for group in siblings.values():
        ordered = sorted(
            group,
            key=lambda item: (
                int(item["span"]["start_line"]),
                int(item["span"]["start_column"]),
                str(item["symbol_id"]),
            ),
        )
        for previous, current in zip(ordered, ordered[1:]):
            for source, target in ((previous, current), (current, previous)):
                append_edge(
                    edges,
                    edge_counts,
                    make_edge(
                        "NEIGHBOR",
                        source["symbol_id"],
                        target["symbol_id"],
                        repo_name,
                        path=source["path"],
                    ),
                )


def statement_target_key(target: Dict[str, object]) -> str | None:
    return target.get("target_symbol_id") or target.get("target_qualified_name") or target.get("qualified_name_hint")


def resolve_target_node(
    nodes: List[Dict[str, object]],
    node_ids: set[str],
    reference_nodes: Dict[str, str],
    repo_name: str,
    target_symbol_id: str | None,
    target_qualified_name: str | None,
    fallback_kind: str,
) -> str | None:
    if target_symbol_id:
        return target_symbol_id
    if not target_qualified_name:
        return None
    return ensure_reference_node(nodes, node_ids, reference_nodes, repo_name, target_qualified_name, fallback_kind)


def resolve_import_target_node(
    nodes: List[Dict[str, object]],
    node_ids: set[str],
    reference_nodes: Dict[str, str],
    repo_name: str,
    target_symbol_id: str | None,
    target_qualified_name: str | None,
    *,
    local_crate_roots: set[str],
    local_package_names: set[str],
) -> str | None:
    if target_symbol_id:
        return target_symbol_id
    if not target_qualified_name:
        return None
    root = str(target_qualified_name).split("::", 1)[0].replace("-", "_")
    if root and root not in local_crate_roots and root not in local_package_names:
        node_id = stable_id("dep", repo_name, root)
        append_node(
            nodes,
            node_ids,
            {
                "node_id": node_id,
                "kind": "dependency",
                "repo": repo_name,
                "name": root,
                "qualified_name": root,
            },
        )
        return node_id
    return ensure_reference_node(nodes, node_ids, reference_nodes, repo_name, target_qualified_name, "module_ref")
