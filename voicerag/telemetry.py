"""Latency instrumentation.

Every stage records a span; the recorder keeps a bounded ring of completed
requests so /v1/stats can report live P50/P70/P90/P99/P100 without a database.

`perf_counter` throughout — `time.time()` is wall-clock and can jump.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np

from .schemas import LatencyBreakdown, Span


@dataclass
class Timer:
    """Collects spans for one request."""

    spans: list[Span] = field(default_factory=list)
    _t0: float = field(default_factory=time.perf_counter)

    @contextmanager
    def span(self, name: str):
        start = time.perf_counter()
        ok = True
        detail = None
        try:
            yield
        except Exception as exc:  # noqa: BLE001
            ok = False
            detail = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self.spans.append(
                Span(
                    name=name,
                    duration_ms=round((time.perf_counter() - start) * 1000, 3),
                    ok=ok,
                    detail=detail,
                )
            )

    def record(self, name: str, duration_ms: float, ok: bool = True, detail: str | None = None):
        self.spans.append(
            Span(name=name, duration_ms=round(duration_ms, 3), ok=ok, detail=detail)
        )

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000

    def build(self, stt_ms: float | None = None) -> LatencyBreakdown:
        total = self.elapsed_ms
        core = sum(s.duration_ms for s in self.spans if not s.name.startswith("stt"))
        return LatencyBreakdown(
            spans=list(self.spans),
            total_ms=round(total, 3),
            stt_ms=round(stt_ms, 3) if stt_ms is not None else None,
            core_ms=round(core, 3),
        )


def percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "p50": round(float(np.percentile(arr, 50)), 2),
        "p70": round(float(np.percentile(arr, 70)), 2),
        "p90": round(float(np.percentile(arr, 90)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
        "p99": round(float(np.percentile(arr, 99)), 2),
        "p100": round(float(arr.max()), 2),
        "mean": round(float(arr.mean()), 2),
    }


class TelemetryStore:
    """Bounded, thread-safe ring of recent request latencies."""

    def __init__(self, maxlen: int = 5000):
        self._core: deque[float] = deque(maxlen=maxlen)
        self._total: deque[float] = deque(maxlen=maxlen)
        self._stt: deque[float] = deque(maxlen=maxlen)
        self._by_stage: dict[str, deque[float]] = {}
        self._by_mode: dict[str, deque[float]] = {}
        self._maxlen = maxlen
        self._lock = threading.Lock()

    def record(self, breakdown: LatencyBreakdown, mode: str = "fast") -> None:
        with self._lock:
            self._core.append(breakdown.core_ms)
            self._total.append(breakdown.total_ms)
            if breakdown.stt_ms is not None:
                self._stt.append(breakdown.stt_ms)
            self._by_mode.setdefault(mode, deque(maxlen=self._maxlen)).append(breakdown.core_ms)
            for span in breakdown.spans:
                self._by_stage.setdefault(span.name, deque(maxlen=self._maxlen)).append(
                    span.duration_ms
                )

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "core_pipeline_ms": percentiles(list(self._core)),
                "end_to_end_ms": percentiles(list(self._total)),
                "stt_ms": percentiles(list(self._stt)),
                "by_mode": {m: percentiles(list(v)) for m, v in self._by_mode.items()},
                "by_stage_ms": {k: percentiles(list(v)) for k, v in self._by_stage.items()},
            }

    def reset(self) -> None:
        with self._lock:
            self._core.clear()
            self._total.clear()
            self._stt.clear()
            self._by_stage.clear()
            self._by_mode.clear()
