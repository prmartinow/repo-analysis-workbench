from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import ryugraph


EDGE_DEFAULTS = {
    "who_imports": ("IMPORTS",),
    "callers_of": ("CALLS",),
    "callees_of": ("CALLS",),
    "implements_of": ("IMPLEMENTS",),
    "inherits_of": ("INHERITS",),
    "reads_of": ("READS",),
    "writes_of": ("WRITES",),
    "refs_of": ("REFS", "REFERENCES"),
}
EDGE_DIRECTIONS = {
    "who_imports": "incoming",
    "callers_of": "incoming",
    "callees_of": "outgoing",
    "implements_of": "incoming",
    "inherits_of": "outgoing",
    "reads_of": "outgoing",
    "writes_of": "outgoing",
    "refs_of": "outgoing",
}


@dataclass(frozen=True)
class RyuGraphBackend:
    graph_root: Path
    repo_name: str

    @property
    def database_path(self) -> Path:
        return self.graph_root / self.repo_name / "graph.db"

    def artifact_fingerprint(self) -> str:
        snapshot = []
        if self.database_path.exists():
            snapshot.extend(snapshot_artifact(self.database_path))
        return hashlib.sha1(json.dumps(snapshot, sort_keys=True).encode("utf-8")).hexdigest()

    def execute(self, request: Dict[str, object]) -> Optional[Dict[str, object]]:
        if not self.database_path.exists():
            return None

        operation = str(request.get("operation") or "neighbors")
        limit = max(int(request.get("limit") or 20), 1)
        depth = max(int(request.get("depth") or 1), 0)
        direction = str(request.get("direction") or EDGE_DIRECTIONS.get(operation, "both"))
        edge_types = tuple(str(item) for item in (request.get("edge_types") or EDGE_DEFAULTS.get(operation, ())))
        node_kinds = tuple(str(item) for item in (request.get("node_kinds") or ()))

        conn = open_connection(str(self.database_path))
        seeds = resolve_seed_nodes(conn, request.get("seed") or request.get("query"), limit=max(limit, 10))
        payload = {
            "repo": self.repo_name,
            "operation": operation,
            "graph_backend": "ryugraph",
            "request": {
                "direction": direction,
                "edge_types": list(edge_types),
                "depth": depth,
                "limit": limit,
                "node_kinds": list(node_kinds),
                "seed": request.get("seed") or request.get("query"),
            },
            "seeds": [describe_node(seed) for seed in seeds],
        }

        if operation == "where_defined":
            payload["results"] = [describe_node(seed) for seed in seeds[:limit]]
            return payload
        if operation == "statement_slice":
            payload["results"] = build_statement_slice(
                conn,
                seeds,
                limit=limit,
                window=max(int(request.get("window") or 8), 1),
            )
            return payload
        if operation == "path_between":
            targets = resolve_seed_nodes(conn, request.get("target"), limit=max(limit, 10))
            payload["targets"] = [describe_node(target) for target in targets]
            payload["results"] = build_shortest_paths(
                conn,
                seeds,
                targets,
                edge_types=edge_types,
                direction=direction,
                node_kinds=node_kinds,
                limit=limit,
            )
            return payload
        if operation == "symbol_summary":
            payload["results"] = build_symbol_summaries(conn, seeds, limit=limit)
            return payload

        payload["results"] = build_neighbors(
            conn,
            seeds,
            direction=direction,
            edge_types=edge_types,
            node_kinds=node_kinds,
            depth=depth,
            limit=limit,
        )
        return payload

    def load_payload(self) -> Dict[str, object]:
        conn = open_connection(str(self.database_path))
        nodes = load_all_nodes(conn)
        edges = load_all_edges(conn)
        summary = metadata_dict(conn).get("summary_json")
        return {
            "payload": {
                "schema_version": metadata_dict(conn).get("schema_version"),
                "repo": metadata_dict(conn).get("repo") or self.repo_name,
                "generated_at": metadata_dict(conn).get("generated_at"),
                "summary": json.loads(summary or "{}"),
                "nodes": nodes,
                "edges": edges,
            }
        }


@lru_cache(maxsize=8)
def open_database(path: str, *, read_only: bool) -> ryugraph.Database:
    return ryugraph.Database(path, read_only=read_only)


def reset_database_cache() -> None:
    open_database.cache_clear()


def open_connection(path: str) -> ryugraph.Connection:
    return ryugraph.Connection(open_database(path, read_only=True))


def execute_rows(
    conn: ryugraph.Connection,
    query: str,
    parameters: Optional[Dict[str, object]] = None,
) -> List[Dict[str, object]]:
    result = conn.execute(query, parameters=parameters or {})
    return [dict(row) for row in result.rows_as_dict()]


def metadata_dict(conn: ryugraph.Connection) -> Dict[str, str]:
    rows = execute_rows(conn, "MATCH (m:GraphMetadata) RETURN m.key AS key, m.value AS value")
    return {str(row["key"]): str(row["value"]) for row in rows}


def load_edge_types(conn: ryugraph.Connection) -> List[str]:
    metadata = metadata_dict(conn)
    raw = metadata.get("edge_types_json") or "[]"
    try:
        values = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return [str(item) for item in values if item]


def resolve_seed_nodes(
    conn: ryugraph.Connection,
    seed: object,
    *,
    limit: int,
) -> List[Dict[str, object]]:
    if seed is None:
        return []
    if isinstance(seed, dict):
        if seed.get("node_id"):
            return load_nodes_by_ids(conn, [str(seed["node_id"])])
        if seed.get("symbol_id"):
            return load_nodes_by_ids(conn, [str(seed["symbol_id"])])
        for key in ("qualified_name", "name", "path"):
            if seed.get(key):
                return resolve_seed_nodes(conn, str(seed[key]), limit=limit)
        return []
    if not isinstance(seed, str):
        return []
    query = seed.strip()
    if not query:
        return []

    rows = execute_rows(
        conn,
        """
        MATCH (n:Node)
        WHERE n.node_id = $value OR n.qualified_name = $value OR n.name = $value OR n.path = $value
        RETURN n.node_id AS node_id, n.kind AS kind, n.repo AS repo, n.path AS path,
               n.name AS name, n.qualified_name AS qualified_name, n.metadata_json AS metadata_json
        LIMIT $limit
        """,
        {"value": query, "limit": limit},
    )
    if rows:
        return [inflate_node(row) for row in rows]

    lowered = query.lower()
    rows = execute_rows(
        conn,
        """
        MATCH (n:Node)
        WHERE LOWER(COALESCE(n.qualified_name, '')) = $value
           OR LOWER(COALESCE(n.name, '')) = $value
           OR LOWER(COALESCE(n.path, '')) = $value
        RETURN n.node_id AS node_id, n.kind AS kind, n.repo AS repo, n.path AS path,
               n.name AS name, n.qualified_name AS qualified_name, n.metadata_json AS metadata_json
        LIMIT $limit
        """,
        {"value": lowered, "limit": limit},
    )
    return [inflate_node(row) for row in rows]


def load_nodes_by_ids(conn: ryugraph.Connection, node_ids: Sequence[str]) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for node_id in node_ids:
        rows = execute_rows(
            conn,
            """
            MATCH (n:Node {node_id: $node_id})
            RETURN n.node_id AS node_id, n.kind AS kind, n.repo AS repo, n.path AS path,
                   n.name AS name, n.qualified_name AS qualified_name, n.metadata_json AS metadata_json
            """,
            {"node_id": node_id},
        )
        results.extend(inflate_node(row) for row in rows)
    return results


def build_neighbors(
    conn: ryugraph.Connection,
    seeds: Sequence[Dict[str, object]],
    *,
    direction: str,
    edge_types: Sequence[str],
    node_kinds: Sequence[str],
    depth: int,
    limit: int,
) -> List[Dict[str, object]]:
    allowed_node_kinds = set(node_kinds)
    queue = deque((str(seed["node_id"]), 0, "seed") for seed in seeds)
    seen_nodes = {str(seed["node_id"]) for seed in seeds}
    seen_results = set()
    results: List[Dict[str, object]] = []
    while queue and len(results) < limit:
        node_id, current_depth, arrived_via = queue.popleft()
        if current_depth >= max(depth, 1):
            continue
        for neighbor, edge_payload in query_adjacency(conn, node_id, direction=direction, edge_types=edge_types):
            if allowed_node_kinds and str(neighbor.get("kind") or "") not in allowed_node_kinds:
                continue
            key = (
                str(neighbor["node_id"]),
                str(edge_payload["type"]),
                str(edge_payload["direction"]),
                current_depth + 1,
            )
            if key in seen_results:
                continue
            seen_results.add(key)
            results.append(
                {
                    "depth": current_depth + 1,
                    "direction": str(edge_payload["direction"]),
                    "edge_type": str(edge_payload["type"]),
                    "arrived_via": arrived_via,
                    **describe_node(neighbor),
                    "edge": edge_payload,
                    "edge_metadata": dict(edge_payload["metadata"]),
                }
            )
            neighbor_id = str(neighbor["node_id"])
            if neighbor_id not in seen_nodes:
                seen_nodes.add(neighbor_id)
                queue.append((neighbor_id, current_depth + 1, str(edge_payload["type"])))
            if len(results) >= limit:
                break
    results.sort(
        key=lambda item: (
            int(item["depth"]),
            str(item["edge_type"]),
            str(item.get("path") or ""),
            str(item.get("qualified_name") or item.get("name") or ""),
        )
    )
    return results[:limit]


def build_statement_slice(
    conn: ryugraph.Connection,
    seeds: Sequence[Dict[str, object]],
    *,
    limit: int,
    window: int,
) -> List[Dict[str, object]]:
    statement_ids: List[str] = []
    seen = set()
    for seed in seeds:
        seed_id = str(seed["node_id"])
        if str(seed.get("kind") or "") == "statement":
            candidate_ids = [seed_id]
            candidate_ids.extend(
                item["node_id"]
                for item, _edge in query_adjacency(conn, seed_id, direction="outgoing", edge_types=("CONTROL_FLOW",))
            )
            candidate_ids.extend(
                item["node_id"]
                for item, _edge in query_adjacency(conn, seed_id, direction="incoming", edge_types=("CONTROL_FLOW",))
            )
        else:
            candidate_ids = [
                item["node_id"]
                for item, _edge in query_adjacency(conn, seed_id, direction="outgoing", edge_types=("CONTAINS",))
                if str(item.get("kind") or "") == "statement"
            ]
        ordered = sorted(
            (item for item in load_nodes_by_ids(conn, candidate_ids) if str(item.get("kind") or "") == "statement"),
            key=statement_sort_key,
        )
        for row in ordered[:window]:
            statement_id = str(row["node_id"])
            if statement_id in seen:
                continue
            seen.add(statement_id)
            statement_ids.append(statement_id)
            if len(statement_ids) >= limit:
                break
        if len(statement_ids) >= limit:
            break
    rows = {str(item["node_id"]): item for item in load_nodes_by_ids(conn, statement_ids)}
    return [hydrate_statement(conn, rows[item]) for item in statement_ids if item in rows][:limit]


def build_shortest_paths(
    conn: ryugraph.Connection,
    seeds: Sequence[Dict[str, object]],
    targets: Sequence[Dict[str, object]],
    *,
    edge_types: Sequence[str],
    direction: str,
    node_kinds: Sequence[str],
    limit: int,
) -> List[Dict[str, object]]:
    if not seeds or not targets:
        return []
    allowed_node_kinds = set(node_kinds)
    target_ids = {str(target["node_id"]) for target in targets}
    results = []
    for seed in seeds:
        queue = deque([str(seed["node_id"])])
        parents: Dict[str, Tuple[Optional[str], Optional[Dict[str, object]]]] = {str(seed["node_id"]): (None, None)}
        found_target: Optional[str] = None
        while queue and found_target is None:
            current = queue.popleft()
            for neighbor, edge_payload in query_adjacency(conn, current, direction=direction, edge_types=edge_types):
                neighbor_id = str(neighbor["node_id"])
                if neighbor_id in parents:
                    continue
                if allowed_node_kinds and str(neighbor.get("kind") or "") not in allowed_node_kinds and neighbor_id not in target_ids:
                    continue
                parents[neighbor_id] = (current, edge_payload)
                if neighbor_id in target_ids:
                    found_target = neighbor_id
                    break
                queue.append(neighbor_id)
        if found_target is None:
            continue
        ordered_ids: List[str] = []
        ordered_edges: List[Dict[str, object]] = []
        cursor = found_target
        while cursor is not None:
            parent_id, parent_edge = parents[cursor]
            ordered_ids.append(cursor)
            if parent_edge is not None:
                ordered_edges.append(parent_edge)
            cursor = parent_id
        ordered_ids.reverse()
        ordered_edges.reverse()
        node_rows = {str(item["node_id"]): item for item in load_nodes_by_ids(conn, ordered_ids)}
        results.append(
            {
                "source": describe_node(seed),
                "target": describe_node(node_rows[found_target]),
                "hop_count": len(ordered_edges),
                "nodes": [describe_node(node_rows[node_id]) for node_id in ordered_ids if node_id in node_rows],
                "edges": ordered_edges,
            }
        )
        if len(results) >= limit:
            break
    results.sort(
        key=lambda item: (
            int(item["hop_count"]),
            str(item["target"].get("path") or ""),
            str(item["target"].get("qualified_name") or item["target"].get("name") or ""),
        )
    )
    return results[:limit]


def build_symbol_summaries(
    conn: ryugraph.Connection,
    seeds: Sequence[Dict[str, object]],
    *,
    limit: int,
) -> List[Dict[str, object]]:
    all_edge_types = tuple(load_edge_types(conn))
    results = []
    for seed in seeds[:limit]:
        node_id = str(seed["node_id"])
        outgoing_pairs = list(query_adjacency(conn, node_id, direction="outgoing", edge_types=all_edge_types))
        incoming_pairs = list(query_adjacency(conn, node_id, direction="incoming", edge_types=all_edge_types))
        direct_calls = [describe_node(item) for item, edge in outgoing_pairs if str(edge["type"]) == "CALLS"]
        reads = [describe_node(item) for item, edge in outgoing_pairs if str(edge["type"]) == "READS"]
        writes = [describe_node(item) for item, edge in outgoing_pairs if str(edge["type"]) == "WRITES"]
        refs = [describe_node(item) for item, edge in outgoing_pairs if str(edge["type"]) in {"REFS", "REFERENCES"}]
        statement_rows = [
            hydrate_statement(conn, item)
            for item, edge in outgoing_pairs
            if str(edge["type"]) == "CONTAINS" and str(item.get("kind") or "") == "statement"
        ]
        results.append(
            {
                **describe_node(seed),
                "incoming_edge_counts": edge_counter(edge for _item, edge in incoming_pairs),
                "outgoing_edge_counts": edge_counter(edge for _item, edge in outgoing_pairs),
                "direct_calls": direct_calls,
                "reads": reads,
                "writes": writes,
                "references": refs,
                "defining_statements": statement_rows,
                "summary": summarize_symbol(describe_node(seed), direct_calls, reads, writes, refs, statement_rows),
            }
        )
    return results


def query_adjacency(
    conn: ryugraph.Connection,
    node_id: str,
    *,
    direction: str,
    edge_types: Sequence[str],
) -> Iterable[Tuple[Dict[str, object], Dict[str, object]]]:
    available_edge_types = set(load_edge_types(conn))
    requested_edge_types = tuple(edge_types) if edge_types else tuple(sorted(available_edge_types))
    requested_edge_types = tuple(edge_type for edge_type in requested_edge_types if edge_type in available_edge_types)
    for edge_type in requested_edge_types:
        if direction in {"outgoing", "both"}:
            rows = execute_rows(
                conn,
                f"""
                MATCH (a:Node {{node_id: $node_id}})-[e:{edge_type}]->(b:Node)
                RETURN b.node_id AS node_id, b.kind AS kind, b.repo AS repo, b.path AS path,
                       b.name AS name, b.qualified_name AS qualified_name, b.metadata_json AS metadata_json,
                       e.edge_id AS edge_id, e.path AS edge_path, e.metadata_json AS edge_metadata
                """,
                {"node_id": node_id},
            )
            for row in rows:
                yield inflate_node(row), {
                    "edge_id": row.get("edge_id"),
                    "type": edge_type,
                    "direction": "outgoing",
                    "metadata": parse_json_value(row.get("edge_metadata")),
                    "path": row.get("edge_path"),
                }
        if direction in {"incoming", "both"}:
            rows = execute_rows(
                conn,
                f"""
                MATCH (a:Node {{node_id: $node_id}})<-[e:{edge_type}]-(b:Node)
                RETURN b.node_id AS node_id, b.kind AS kind, b.repo AS repo, b.path AS path,
                       b.name AS name, b.qualified_name AS qualified_name, b.metadata_json AS metadata_json,
                       e.edge_id AS edge_id, e.path AS edge_path, e.metadata_json AS edge_metadata
                """,
                {"node_id": node_id},
            )
            for row in rows:
                yield inflate_node(row), {
                    "edge_id": row.get("edge_id"),
                    "type": edge_type,
                    "direction": "incoming",
                    "metadata": parse_json_value(row.get("edge_metadata")),
                    "path": row.get("edge_path"),
                }


def hydrate_statement(conn: ryugraph.Connection, row: Dict[str, object]) -> Dict[str, object]:
    metadata = dict(row.get("metadata") or {})
    described = {
        "statement_id": row["node_id"],
        "kind": row.get("kind"),
        "path": row.get("path"),
        "text": metadata.get("text") or "",
        "line": int((metadata.get("span") or {}).get("start_line") or 0),
        "container_symbol_id": metadata.get("container_symbol_id"),
        "container_qualified_name": metadata.get("container_qualified_name"),
    }
    reads = [describe_node(item) for item, edge in query_adjacency(conn, str(row["node_id"]), direction="outgoing", edge_types=("READS",))]
    writes = [describe_node(item) for item, edge in query_adjacency(conn, str(row["node_id"]), direction="outgoing", edge_types=("WRITES",))]
    refs = [
        describe_node(item)
        for item, edge in query_adjacency(conn, str(row["node_id"]), direction="outgoing", edge_types=("REFS", "REFERENCES"))
    ]
    calls = [describe_node(item) for item, edge in query_adjacency(conn, str(row["node_id"]), direction="outgoing", edge_types=("CALLS",))]
    control_predecessors = [
        describe_node(item)
        for item, edge in query_adjacency(conn, str(row["node_id"]), direction="incoming", edge_types=("CONTROL_FLOW",))
    ]
    control_successors = [
        describe_node(item)
        for item, edge in query_adjacency(conn, str(row["node_id"]), direction="outgoing", edge_types=("CONTROL_FLOW",))
    ]
    described["reads"] = reads
    described["writes"] = writes
    described["refs"] = refs
    described["calls"] = calls
    described["control_predecessors"] = control_predecessors
    described["control_successors"] = control_successors
    return described


def load_all_nodes(conn: ryugraph.Connection) -> List[Dict[str, object]]:
    rows = execute_rows(
        conn,
        """
        MATCH (n:Node)
        RETURN n.node_id AS node_id, n.kind AS kind, n.repo AS repo, n.path AS path,
               n.name AS name, n.qualified_name AS qualified_name, n.metadata_json AS metadata_json
        """,
    )
    return [
        {
            "node_id": row["node_id"],
            "kind": row["kind"],
            "repo": row["repo"],
            "path": row.get("path"),
            "name": row.get("name"),
            "qualified_name": row.get("qualified_name"),
            **parse_json_value(row.get("metadata_json")),
        }
        for row in rows
    ]


def load_all_edges(conn: ryugraph.Connection) -> List[Dict[str, object]]:
    edges: List[Dict[str, object]] = []
    for edge_type in load_edge_types(conn):
        rows = execute_rows(
            conn,
            f"""
            MATCH (a:Node)-[e:{edge_type}]->(b:Node)
            RETURN a.node_id AS from_id, b.node_id AS to_id,
                   e.edge_id AS edge_id, e.path AS edge_path, e.metadata_json AS edge_metadata
            """,
        )
        for row in rows:
            edges.append(
                {
                    "edge_id": row["edge_id"],
                    "type": edge_type,
                    "from": row["from_id"],
                    "to": row["to_id"],
                    "metadata": parse_json_value(row.get("edge_metadata")),
                    "path": row.get("edge_path"),
                }
            )
    return edges


def inflate_node(row: Dict[str, object]) -> Dict[str, object]:
    return {
        "node_id": row["node_id"],
        "kind": row["kind"],
        "repo": row.get("repo"),
        "path": row.get("path"),
        "name": row.get("name"),
        "qualified_name": row.get("qualified_name"),
        "metadata": parse_json_value(row.get("metadata_json")),
    }


def describe_node(node: Dict[str, object]) -> Dict[str, object]:
    metadata = dict(node.get("metadata") or {})
    return {
        "node_id": node["node_id"],
        "kind": node["kind"],
        "path": node.get("path"),
        "name": node.get("name"),
        "qualified_name": node.get("qualified_name"),
        "symbol_id": node["node_id"] if str(node["node_id"]).startswith("sym:") else None,
        "span": metadata.get("span"),
        "container_qualified_name": metadata.get("container_qualified_name"),
        "signature": metadata.get("signature"),
        "visibility": metadata.get("visibility"),
    }


def statement_sort_key(row: Dict[str, object]) -> Tuple[str, int, int, str]:
    metadata = dict(row.get("metadata") or {})
    span = metadata.get("span") or {}
    return (
        str(row.get("path") or ""),
        int(span.get("start_line") or 0),
        int(span.get("start_column") or 0),
        str(row.get("node_id") or ""),
    )


def edge_counter(edges: Iterable[Dict[str, object]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for edge in edges:
        edge_type = str(edge.get("type") or "")
        counts[edge_type] = counts.get(edge_type, 0) + 1
    return dict(sorted(counts.items()))


def summarize_symbol(
    node: Dict[str, object],
    direct_calls: Sequence[Dict[str, object]],
    reads: Sequence[Dict[str, object]],
    writes: Sequence[Dict[str, object]],
    refs: Sequence[Dict[str, object]],
    statements: Sequence[Dict[str, object]],
) -> str:
    return (
        f"{node.get('kind')} {node.get('qualified_name') or node.get('name')} in {node.get('path')}. "
        f"Calls={len(direct_calls)} Reads={len(reads)} Writes={len(writes)} Refs={len(refs)} Statements={len(statements)}."
    )


def parse_json_value(raw: object) -> Dict[str, object]:
    if raw in (None, ""):
        return {}
    try:
        value = json.loads(str(raw))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def snapshot_artifact(path: Path) -> List[Dict[str, object]]:
    if path.is_dir():
        rows: List[Dict[str, object]] = []
        for child in sorted(path.rglob("*")):
            if not child.is_file():
                continue
            stat = child.stat()
            rows.append(
                {
                    "path": str(child),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
        return rows
    stat = path.stat()
    return [
        {
            "path": str(path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    ]
