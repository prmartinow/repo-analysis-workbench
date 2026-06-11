from __future__ import annotations

import os
import re
import subprocess
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


ITEM_KIND_PATTERNS = {
    "const": re.compile(r"kind:\s+Const\("),
    "enum": re.compile(r"kind:\s+Enum\("),
    "extern_crate": re.compile(r"kind:\s+ExternCrate\("),
    "function": re.compile(r"kind:\s+Fn\("),
    "impl": re.compile(r"kind:\s+Impl\("),
    "module": re.compile(r"kind:\s+Mod\("),
    "static": re.compile(r"kind:\s+Static\("),
    "struct": re.compile(r"kind:\s+Struct\("),
    "trait": re.compile(r"kind:\s+Trait\("),
    "type": re.compile(r"kind:\s+TyAlias\("),
    "union": re.compile(r"kind:\s+Union\("),
    "use": re.compile(r"kind:\s+Use\("),
}
STATEMENT_KIND_PATTERNS = {
    "expr": re.compile(r"kind:\s+Expr\("),
    "local": re.compile(r"kind:\s+Local\("),
    "semi": re.compile(r"kind:\s+Semi\("),
}
CONTROL_KIND_PATTERNS = {
    "assign": re.compile(r"kind:\s+Assign\("),
    "assign_op": re.compile(r"kind:\s+AssignOp\("),
    "for": re.compile(r"kind:\s+ForLoop\("),
    "if": re.compile(r"kind:\s+If\("),
    "loop": re.compile(r"kind:\s+Loop\("),
    "match": re.compile(r"kind:\s+Match\("),
    "while": re.compile(r"kind:\s+While\("),
}


@lru_cache(maxsize=1)
def rustc_available() -> bool:
    try:
        result = subprocess.run(
            ["rustc", "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    return bool(result.stdout.strip())


def probe_rust_ast(path: Path) -> Dict[str, object]:
    started = time.perf_counter()
    if not rustc_available():
        return {
            "backend": "rustc-ast-tree",
            "available": False,
            "used": False,
            "parsed": False,
            "path": path.as_posix(),
            "latency_ms": 0.0,
            "item_counts": [],
            "statement_counts": [],
            "control_counts": [],
            "diagnostics": ["rustc not available"],
        }

    try:
        result = subprocess.run(
            [
                "rustc",
                "-Z",
                "unpretty=ast-tree",
                str(path),
                "--edition=2021",
            ],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "RUSTC_BOOTSTRAP": "1"},
        )
        parsed = True
        stdout = result.stdout
        diagnostics = []
    except subprocess.CalledProcessError as exc:
        parsed = False
        stdout = exc.stdout or ""
        diagnostics = [line for line in (exc.stderr or "").splitlines() if line.strip()][:20]

    latency_ms = round((time.perf_counter() - started) * 1000, 3)
    item_counts = count_matches(stdout, ITEM_KIND_PATTERNS)
    statement_counts = count_matches(stdout, STATEMENT_KIND_PATTERNS)
    control_counts = count_matches(stdout, CONTROL_KIND_PATTERNS)
    return {
        "backend": "rustc-ast-tree",
        "available": True,
        "used": True,
        "parsed": parsed,
        "path": path.as_posix(),
        "latency_ms": latency_ms,
        "item_counts": item_counts,
        "statement_counts": statement_counts,
        "control_counts": control_counts,
        "diagnostics": diagnostics,
    }


def aggregate_rustc_probes(file_probes: Sequence[Dict[str, object]]) -> Dict[str, object]:
    parsed_files = sum(1 for probe in file_probes if probe.get("parsed"))
    available = any(probe.get("available") for probe in file_probes) if file_probes else rustc_available()
    return {
        "backend": "rustc-ast-tree",
        "available": available,
        "used": bool(file_probes),
        "files": len(file_probes),
        "parsed_files": parsed_files,
        "item_counts": aggregate_counts(file_probes, "item_counts"),
        "statement_counts": aggregate_counts(file_probes, "statement_counts"),
        "control_counts": aggregate_counts(file_probes, "control_counts"),
        "samples": [
            {
                "path": probe["path"],
                "parsed": probe["parsed"],
                "latency_ms": probe["latency_ms"],
            }
            for probe in file_probes[:10]
        ],
    }


def count_matches(text: str, patterns: Dict[str, re.Pattern[str]]) -> List[Dict[str, object]]:
    return [
        {
            "kind": kind,
            "count": len(pattern.findall(text)),
        }
        for kind, pattern in sorted(patterns.items())
        if pattern.search(text)
    ]


def aggregate_counts(file_probes: Sequence[Dict[str, object]], field: str) -> List[Dict[str, object]]:
    counts: Dict[str, int] = {}
    for probe in file_probes:
        for item in probe.get(field, []):
            counts[item["kind"]] = counts.get(item["kind"], 0) + int(item["count"])
    return [
        {
            "kind": kind,
            "count": count,
        }
        for kind, count in sorted(counts.items())
    ]
