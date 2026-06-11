from __future__ import annotations

import csv
import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List

import ryugraph


NODE_TABLE = "Node"
METADATA_TABLE = "GraphMetadata"


def load_ryugraph_database(
    output_root: Path,
    repo_name: str,
    payload: Dict[str, object],
) -> Path:
    from backends.ryugraph.queries import reset_database_cache

    repo_output = output_root / repo_name
    repo_output.mkdir(parents=True, exist_ok=True)
    target = repo_output / "graph.db"
    reset_database_cache()
    remove_graph_database(target)

    db = ryugraph.Database(str(target))
    conn = ryugraph.Connection(db)
    edge_types = sorted({str(edge.get("type") or "") for edge in payload.get("edges", []) if edge.get("type")})

    conn.execute(
        """
        CREATE NODE TABLE Node(
            node_id STRING PRIMARY KEY,
            kind STRING,
            repo STRING,
            path STRING,
            name STRING,
            qualified_name STRING,
            metadata_json STRING
        )
        """
    )
    conn.execute(
        """
        CREATE NODE TABLE GraphMetadata(
            key STRING PRIMARY KEY,
            value STRING
        )
        """
    )
    for edge_type in edge_types:
        conn.execute(
            f"""
            CREATE REL TABLE {edge_type}(
                FROM Node TO Node,
                edge_id STRING,
                path STRING,
                metadata_json STRING
            )
            """
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_root = Path(tmpdir)
        node_csv = temp_root / "nodes.csv"
        metadata_csv = temp_root / "graph_metadata.csv"
        write_csv(
            node_csv,
            ["node_id", "kind", "repo", "path", "name", "qualified_name", "metadata_json"],
            dedupe_rows([flatten_node_row(row) for row in payload.get("nodes", [])], key="node_id"),
        )
        write_csv(
            metadata_csv,
            ["key", "value"],
            build_metadata_rows(payload, edge_types),
        )
        conn.execute(f"COPY {NODE_TABLE} FROM {quote_path(node_csv)} (HEADER=true)")
        conn.execute(f"COPY {METADATA_TABLE} FROM {quote_path(metadata_csv)} (HEADER=true)")

        flattened_edges = dedupe_rows([flatten_edge_row(row) for row in payload.get("edges", [])], key="edge_id")
        for edge_type in edge_types:
            edge_rows = [row for row in flattened_edges if str(row.get("type") or "") == edge_type]
            edge_csv = temp_root / f"{edge_type}.csv"
            write_csv(
                edge_csv,
                ["from", "to", "edge_id", "path", "metadata_json"],
                edge_rows,
            )
            conn.execute(f"COPY {edge_type} FROM {quote_path(edge_csv)} (HEADER=true)")

    for filename in ("graph.json", "ryugraph.json", "graph_manifest.json"):
        try:
            (repo_output / filename).unlink()
        except FileNotFoundError:
            pass
    reset_database_cache()
    return target


def remove_graph_database(target: Path) -> None:
    if not target.exists():
        return
    if target.is_dir():
        shutil.rmtree(target)
        return
    target.unlink()


def write_csv(path: Path, fieldnames: List[str], rows: Iterable[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: stringify_csv_value(row.get(key)) for key in fieldnames})


def stringify_csv_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def build_metadata_rows(payload: Dict[str, object], edge_types: List[str]) -> List[Dict[str, object]]:
    return [
        {"key": "schema_version", "value": str(payload.get("schema_version") or "")},
        {"key": "repo", "value": str(payload.get("repo") or "")},
        {"key": "generated_at", "value": str(payload.get("generated_at") or "")},
        {"key": "summary_json", "value": json.dumps(payload.get("summary", {}), sort_keys=True)},
        {"key": "graph_backend", "value": "ryugraph"},
        {"key": "edge_types_json", "value": json.dumps(edge_types)},
    ]


def flatten_node_row(row: Dict[str, object]) -> Dict[str, object]:
    metadata = dict(row)
    for key in ("node_id", "kind", "repo", "path", "name", "qualified_name"):
        metadata.pop(key, None)
    return {
        "node_id": row["node_id"],
        "kind": row["kind"],
        "repo": row["repo"],
        "path": row.get("path"),
        "name": row.get("name"),
        "qualified_name": row.get("qualified_name"),
        "metadata_json": json.dumps(metadata, sort_keys=True),
    }


def flatten_edge_row(row: Dict[str, object]) -> Dict[str, object]:
    metadata = dict(row.get("metadata") or {})
    return {
        "type": row["type"],
        "from": row["from"],
        "to": row["to"],
        "edge_id": row["edge_id"],
        "path": metadata.get("path"),
        "metadata_json": json.dumps(metadata, sort_keys=True),
    }


def dedupe_rows(rows: List[Dict[str, object]], *, key: str) -> List[Dict[str, object]]:
    deduped: List[Dict[str, object]] = []
    seen = set()
    for row in rows:
        row_key = row[key]
        if row_key in seen:
            continue
        seen.add(row_key)
        deduped.append(row)
    return deduped


def quote_path(path: Path) -> str:
    return json.dumps(str(path))
