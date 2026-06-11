from __future__ import annotations

import json
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import lmdb

from common.telemetry import increment_counter, trace_operation


def write_lmdb_metadata_bundle(
    output_root: Path,
    repo_name: str,
    payload: Dict[str, object],
    *,
    summaries_payload: Dict[str, object] | None = None,
    artifact_metadata: Dict[str, object] | None = None,
) -> None:
    repo_output = output_root / repo_name
    repo_output.mkdir(parents=True, exist_ok=True)
    target = repo_output / "metadata.lmdb"
    existing_artifact_metadata = load_lmdb_artifact_metadata(output_root, repo_name)
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    files_by_path = {str(row["path"]): dict(row) for row in payload.get("files", [])}
    symbols_by_id = {str(row["symbol_id"]): dict(row) for row in payload.get("symbols", [])}
    statements_by_symbol: Dict[str, List[Dict[str, object]]] = {}
    for row in payload.get("statements", []):
        symbol_id = str(row.get("container_symbol_id") or "")
        if not symbol_id:
            continue
        statements_by_symbol.setdefault(symbol_id, []).append(
            {
                "statement_id": row["statement_id"],
                "text": row["text"],
                "line": int(row["span"]["start_line"]),
            }
        )
    for rows in statements_by_symbol.values():
        rows.sort(key=lambda item: (int(item["line"]), str(item["statement_id"])))

    symbol_ids_by_qname: Dict[str, List[str]] = {}
    symbol_ids_by_name: Dict[str, List[str]] = {}
    for row in payload.get("symbols", []):
        symbol_id = str(row["symbol_id"])
        qname = str(row.get("qualified_name") or "")
        name = str(row.get("name") or "")
        if qname:
            symbol_ids_by_qname.setdefault(qname, []).append(symbol_id)
        if name:
            symbol_ids_by_name.setdefault(name, []).append(symbol_id)
    for ids in symbol_ids_by_qname.values():
        ids.sort()
    for ids in symbol_ids_by_name.values():
        ids.sort()

    summaries_by_id: Dict[str, Dict[str, object]] = {}
    summaries_by_path: Dict[str, List[Dict[str, object]]] = {}
    summaries_by_symbol: Dict[str, List[Dict[str, object]]] = {}
    if summaries_payload is not None:
        for key in ("project",):
            item = summaries_payload.get(key)
            if isinstance(item, dict) and item:
                summary_id = str(item.get("summary_id") or "")
                if summary_id:
                    summaries_by_id[summary_id] = dict(item)
        for collection_name in ("packages", "directories", "files", "symbols"):
            for item in summaries_payload.get(collection_name, []):
                summary_id = str(item.get("summary_id") or "")
                if not summary_id:
                    continue
                record = dict(item)
                summaries_by_id[summary_id] = record
                path = str(record.get("path") or "")
                symbol_id = str(record.get("symbol_id") or "")
                if path:
                    summaries_by_path.setdefault(path, []).append(record)
                if symbol_id:
                    summaries_by_symbol.setdefault(symbol_id, []).append(record)
        for rows in summaries_by_path.values():
            rows.sort(key=lambda item: str(item.get("summary_id") or ""))
        for rows in summaries_by_symbol.values():
            rows.sort(key=lambda item: str(item.get("summary_id") or ""))

    env = lmdb.open(
        str(target),
        map_size=1 << 36,
        subdir=True,
        create=True,
        max_dbs=16,
        lock=True,
        writemap=False,
        metasync=True,
        sync=True,
    )
    dbs = {
        "symbol_by_id": env.open_db(b"symbol_by_id"),
        "symbol_ids_by_qname": env.open_db(b"symbol_ids_by_qname"),
        "symbol_ids_by_name": env.open_db(b"symbol_ids_by_name"),
        "file_by_path": env.open_db(b"file_by_path"),
        "import_by_id": env.open_db(b"import_by_id"),
        "reference_by_id": env.open_db(b"reference_by_id"),
        "statement_by_id": env.open_db(b"statement_by_id"),
        "body_by_symbol_id": env.open_db(b"body_by_symbol_id"),
        "summary_by_id": env.open_db(b"summary_by_id"),
        "summary_by_path": env.open_db(b"summary_by_path"),
        "summary_by_symbol_id": env.open_db(b"summary_by_symbol_id"),
        "metadata": env.open_db(b"metadata"),
        "artifact_metadata": env.open_db(b"artifact_metadata"),
    }
    try:
        with env.begin(write=True) as txn:
            for key, value in (
                ("schema_version", str(payload["schema_version"])),
                ("repo", str(payload["repo"])),
                ("generated_at", str(payload["generated_at"])),
                ("parser", str(payload["parser"])),
                ("summary", payload.get("summary", {})),
                ("primary_parser_backends", payload.get("primary_parser_backends", [])),
                ("parser_backends", payload.get("parser_backends", {})),
                ("source_roots", payload.get("source_roots", [])),
                ("path_prefixes", payload.get("path_prefixes", [])),
                ("file_paths", sorted(files_by_path)),
                ("symbol_ids", sorted(symbols_by_id)),
                ("import_ids", sorted(str(row["import_id"]) for row in payload.get("imports", []))),
                ("reference_ids", sorted(str(row["reference_id"]) for row in payload.get("references", []))),
                ("statement_ids", sorted(str(row["statement_id"]) for row in payload.get("statements", []))),
                ("summary_bundle_payload", summaries_payload or {}),
            ):
                txn.put(encode_key(key), encode_value(value), db=dbs["metadata"])

            for path, row in sorted(files_by_path.items()):
                txn.put(encode_key(path), encode_value(row), db=dbs["file_by_path"])

            for symbol_id, row in sorted(symbols_by_id.items()):
                txn.put(encode_key(symbol_id), encode_value(row), db=dbs["symbol_by_id"])
                txn.put(
                    encode_key(symbol_id),
                    encode_value(
                        {
                            "symbol_id": symbol_id,
                            "path": row.get("path"),
                            "qualified_name": row.get("qualified_name"),
                            "signature": row.get("signature"),
                            "statements": statements_by_symbol.get(symbol_id, []),
                        }
                    ),
                    db=dbs["body_by_symbol_id"],
                )

            for row in payload.get("imports", []):
                txn.put(encode_key(str(row["import_id"])), encode_value(row), db=dbs["import_by_id"])

            for row in payload.get("references", []):
                txn.put(encode_key(str(row["reference_id"])), encode_value(row), db=dbs["reference_by_id"])

            for row in payload.get("statements", []):
                txn.put(encode_key(str(row["statement_id"])), encode_value(row), db=dbs["statement_by_id"])

            for qname, symbol_ids in sorted(symbol_ids_by_qname.items()):
                txn.put(encode_key(qname), encode_value(symbol_ids), db=dbs["symbol_ids_by_qname"])
            for name, symbol_ids in sorted(symbol_ids_by_name.items()):
                txn.put(encode_key(name), encode_value(symbol_ids), db=dbs["symbol_ids_by_name"])

            for summary_id, row in sorted(summaries_by_id.items()):
                txn.put(encode_key(summary_id), encode_value(row), db=dbs["summary_by_id"])
            for path, rows in sorted(summaries_by_path.items()):
                txn.put(encode_key(path), encode_value(rows), db=dbs["summary_by_path"])
            for symbol_id, rows in sorted(summaries_by_symbol.items()):
                txn.put(encode_key(symbol_id), encode_value(rows), db=dbs["summary_by_symbol_id"])

            merged_artifact_metadata = {**existing_artifact_metadata, **(artifact_metadata or {})}
            for key, value in sorted(merged_artifact_metadata.items()):
                txn.put(encode_key(key), encode_value(value), db=dbs["artifact_metadata"])
    finally:
        env.sync()
        env.close()


def write_metadata_bundle(
    output_root: Path,
    repo_name: str,
    payload: Dict[str, object],
    *,
    summaries_payload: Dict[str, object] | None = None,
    artifact_metadata: Dict[str, object] | None = None,
) -> None:
    write_lmdb_metadata_bundle(
        output_root,
        repo_name,
        payload,
        summaries_payload=summaries_payload,
        artifact_metadata=artifact_metadata,
    )


def load_lmdb_artifact_metadata(parsed_root: Path, repo_name: str) -> Dict[str, object]:
    lmdb_path = parsed_root / repo_name / "metadata.lmdb"
    if not lmdb_path.exists():
        return {}
    env = lmdb.open(
        str(lmdb_path),
        subdir=True,
        readonly=True,
        lock=False,
        max_dbs=17,
        readahead=True,
    )
    try:
        db = env.open_db(b"artifact_metadata")
        rows: Dict[str, object] = {}
        with env.begin(db=db, write=False) as txn:
            with txn.cursor() as cursor:
                for raw_key, raw_value in cursor:
                    rows[raw_key.decode("utf-8")] = decode_lmdb_json(raw_value)
        return rows
    finally:
        env.close()


def update_lmdb_artifact_metadata(parsed_root: Path, repo_name: str, updates: Dict[str, object]) -> None:
    if not updates:
        return
    lmdb_path = parsed_root / repo_name / "metadata.lmdb"
    if not lmdb_path.exists():
        raise FileNotFoundError(f"Missing metadata.lmdb for repo '{repo_name}' under {parsed_root / repo_name}")
    env = lmdb.open(
        str(lmdb_path),
        subdir=True,
        readonly=False,
        lock=True,
        create=False,
        max_dbs=17,
        readahead=False,
        map_size=1 << 36,
    )
    try:
        db = env.open_db(b"artifact_metadata")
        with env.begin(db=db, write=True) as txn:
            for key, value in sorted(updates.items()):
                txn.put(encode_key(key), encode_value(value), db=db)
    finally:
        env.sync()
        env.close()


def encode_key(value: str) -> bytes:
    return value.encode("utf-8")


def encode_value(value: object) -> bytes:
    return json.dumps(value, sort_keys=True).encode("utf-8")


def decode_lmdb_json(raw: bytes | None) -> object | None:
    if raw is None:
        return None
    return json.loads(raw.decode("utf-8"))


def load_symbol_index(parsed_root: Path, repo_name: str) -> Dict[str, object]:
    increment_counter("full_symbol_payload_loads")
    with trace_operation("load_symbol_index"):
        return _load_symbol_index_cached(str(parsed_root.resolve()), repo_name)


@lru_cache(maxsize=8)
def _load_symbol_index_cached(parsed_root: str, repo_name: str) -> Dict[str, object]:
    metadata_payload = load_symbol_index_from_metadata(Path(parsed_root), repo_name)
    if metadata_payload is not None:
        return metadata_payload
    raise FileNotFoundError(f"Missing metadata.lmdb for repo '{repo_name}' under {Path(parsed_root) / repo_name}")


def load_symbol_index_from_metadata(parsed_root: Path, repo_name: str) -> Dict[str, object] | None:
    lmdb_path = parsed_root / repo_name / "metadata.lmdb"
    if not lmdb_path.exists():
        return None

    env = lmdb.open(
        str(lmdb_path),
        subdir=True,
        readonly=True,
        lock=False,
        max_dbs=16,
        readahead=True,
    )
    try:
        metadata_db = env.open_db(b"metadata")
        file_db = env.open_db(b"file_by_path")
        symbol_db = env.open_db(b"symbol_by_id")
        import_db = env.open_db(b"import_by_id")
        reference_db = env.open_db(b"reference_by_id")
        statement_db = env.open_db(b"statement_by_id")
        with env.begin(write=False) as txn:
            schema_version = decode_lmdb_json(txn.get(encode_key("schema_version"), db=metadata_db))
            if schema_version is None:
                return None
            file_paths = decode_lmdb_json(txn.get(encode_key("file_paths"), db=metadata_db)) or []
            symbol_ids = decode_lmdb_json(txn.get(encode_key("symbol_ids"), db=metadata_db)) or []
            import_ids = decode_lmdb_json(txn.get(encode_key("import_ids"), db=metadata_db)) or []
            reference_ids = decode_lmdb_json(txn.get(encode_key("reference_ids"), db=metadata_db)) or []
            statement_ids = decode_lmdb_json(txn.get(encode_key("statement_ids"), db=metadata_db)) or []

            files = [
                decode_lmdb_json(txn.get(encode_key(str(path)), db=file_db))
                for path in file_paths
            ]
            symbols = [
                decode_lmdb_json(txn.get(encode_key(str(symbol_id)), db=symbol_db))
                for symbol_id in symbol_ids
            ]
            imports = [
                decode_lmdb_json(txn.get(encode_key(str(import_id)), db=import_db))
                for import_id in import_ids
            ]
            references = [
                decode_lmdb_json(txn.get(encode_key(str(reference_id)), db=reference_db))
                for reference_id in reference_ids
            ]
            statements = [
                decode_lmdb_json(txn.get(encode_key(str(statement_id)), db=statement_db))
                for statement_id in statement_ids
            ]

            return {
                "schema_version": schema_version,
                "repo": decode_lmdb_json(txn.get(encode_key("repo"), db=metadata_db)) or repo_name,
                "generated_at": decode_lmdb_json(txn.get(encode_key("generated_at"), db=metadata_db)),
                "parser": decode_lmdb_json(txn.get(encode_key("parser"), db=metadata_db)),
                "primary_parser_backends": decode_lmdb_json(txn.get(encode_key("primary_parser_backends"), db=metadata_db)) or [],
                "parser_backends": decode_lmdb_json(txn.get(encode_key("parser_backends"), db=metadata_db)) or {},
                "source_roots": decode_lmdb_json(txn.get(encode_key("source_roots"), db=metadata_db)) or [],
                "path_prefixes": decode_lmdb_json(txn.get(encode_key("path_prefixes"), db=metadata_db)) or [],
                "summary": decode_lmdb_json(txn.get(encode_key("summary"), db=metadata_db)) or {},
                "files": [row for row in files if isinstance(row, dict)],
                "symbols": [row for row in symbols if isinstance(row, dict)],
                "imports": [row for row in imports if isinstance(row, dict)],
                "references": [row for row in references if isinstance(row, dict)],
                "statements": [row for row in statements if isinstance(row, dict)],
            }
    finally:
        env.close()


def load_summary_bundle_from_metadata(parsed_root: Path, repo_name: str) -> Dict[str, object] | None:
    lmdb_path = parsed_root / repo_name / "metadata.lmdb"
    if not lmdb_path.exists():
        return None
    env = lmdb.open(
        str(lmdb_path),
        subdir=True,
        readonly=True,
        lock=False,
        max_dbs=16,
        readahead=True,
    )
    try:
        metadata_db = env.open_db(b"metadata")
        with env.begin(write=False) as txn:
            payload = decode_lmdb_json(txn.get(encode_key("summary_bundle_payload"), db=metadata_db))
            if isinstance(payload, dict) and payload:
                return payload
            return None
    finally:
        env.close()

