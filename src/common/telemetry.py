from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from typing import Dict, Iterator


LOGGER = logging.getLogger("repo_analysis.telemetry")
_LOCK = threading.Lock()
_COUNTERS: Dict[str, int] = {}
_TIMINGS: Dict[str, Dict[str, float]] = {}


def increment_counter(name: str, value: int = 1) -> None:
    with _LOCK:
        _COUNTERS[name] = _COUNTERS.get(name, 0) + value


@contextmanager
def trace_operation(name: str) -> Iterator[None]:
    started = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - started) * 1000
        with _LOCK:
            stats = _TIMINGS.setdefault(
                name,
                {
                    "count": 0.0,
                    "total_ms": 0.0,
                    "max_ms": 0.0,
                },
            )
            stats["count"] += 1
            stats["total_ms"] += elapsed_ms
            stats["max_ms"] = max(stats["max_ms"], elapsed_ms)
        LOGGER.debug("telemetry op=%s elapsed_ms=%.3f", name, elapsed_ms)


def snapshot_telemetry() -> Dict[str, object]:
    with _LOCK:
        return {
            "counters": dict(_COUNTERS),
            "timings": {
                name: {
                    "count": int(stats["count"]),
                    "total_ms": round(stats["total_ms"], 3),
                    "avg_ms": round(stats["total_ms"] / stats["count"], 3) if stats["count"] else 0.0,
                    "max_ms": round(stats["max_ms"], 3),
                }
                for name, stats in _TIMINGS.items()
            },
        }

def reset_telemetry() -> None:
    with _LOCK:
        _COUNTERS.clear()
        _TIMINGS.clear()
