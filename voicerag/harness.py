"""Execution harness.

The pipeline is not "call a model and hope". Every external or expensive step
runs through `run_stage`, which gives it:

  * a **deadline** carved out of a request-level budget, so a slow dependency
    degrades the response instead of blowing the latency target,
  * **bounded retries** with exponential backoff and full jitter, retrying only
    errors that are actually transient,
  * a **circuit breaker** per provider, so a dead STT endpoint fails in
    microseconds after the fourth failure instead of burning the budget on
    timeouts for every subsequent request,
  * a **typed fallback** — the value to degrade to when the stage can't
    complete — which is what lets the fast path stay under 200ms even when the
    optional LLM stage is unavailable.

`Budget` is threaded through the whole request so each stage knows how much
wall-clock it may actually consume, rather than each having an independent
timeout that can sum to far more than the target.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Sequence, TypeVar

from .telemetry import Timer

logger = logging.getLogger(__name__)

T = TypeVar("T")


class DeadlineExceeded(RuntimeError):
    """The remaining budget is too small to attempt this stage."""


class CircuitOpen(RuntimeError):
    """The provider is in a failing state and calls are being short-circuited."""


class StageFailed(RuntimeError):
    def __init__(self, stage: str, cause: BaseException):
        super().__init__(f"stage {stage!r} failed: {type(cause).__name__}: {cause}")
        self.stage = stage
        self.cause = cause


# --------------------------------------------------------------------------
@dataclass
class Budget:
    total_ms: float
    _start: float = field(default_factory=time.perf_counter)

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000

    @property
    def remaining_ms(self) -> float:
        return self.total_ms - self.elapsed_ms

    def remaining_s(self, reserve_ms: float = 0.0) -> float:
        return max(0.0, (self.remaining_ms - reserve_ms) / 1000.0)

    def exhausted(self, need_ms: float = 0.0) -> bool:
        return self.remaining_ms <= need_ms

    def child(self, ms: float) -> "Budget":
        return Budget(total_ms=min(ms, max(0.0, self.remaining_ms)))


# --------------------------------------------------------------------------
class CircuitBreaker:
    """Classic three-state breaker: closed -> open -> half-open."""

    def __init__(self, name: str, fail_threshold: int = 4, reset_after_s: float = 30.0):
        self.name = name
        self.fail_threshold = fail_threshold
        self.reset_after_s = reset_after_s
        self._failures = 0
        self._opened_at: float | None = None
        self._lock = asyncio.Lock()

    @property
    def state(self) -> str:
        if self._opened_at is None:
            return "closed"
        if (time.monotonic() - self._opened_at) >= self.reset_after_s:
            return "half_open"
        return "open"

    async def before(self) -> None:
        if self.state == "open":
            raise CircuitOpen(f"circuit {self.name!r} open; {self._failures} consecutive failures")

    async def on_success(self) -> None:
        async with self._lock:
            self._failures = 0
            self._opened_at = None

    async def on_failure(self) -> None:
        async with self._lock:
            self._failures += 1
            if self._failures >= self.fail_threshold:
                self._opened_at = time.monotonic()
                logger.warning("circuit %r opened after %d failures", self.name, self._failures)

    def snapshot(self) -> dict:
        return {"name": self.name, "state": self.state, "failures": self._failures}


# --------------------------------------------------------------------------
# Which exceptions are worth retrying. Retrying a 400 just wastes the budget.
# --------------------------------------------------------------------------
def default_is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError, OSError)):
        return True
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status is None:
        status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status in (408, 409, 429) or status >= 500
    name = type(exc).__name__
    return name in {
        "APIConnectionError", "APITimeoutError", "RateLimitError",
        "InternalServerError", "ReadTimeout", "ConnectTimeout", "PoolTimeout",
    }


class Harness:
    def __init__(
        self,
        *,
        max_retries: int = 2,
        breaker_fail_threshold: int = 4,
        breaker_reset_s: float = 30.0,
    ):
        self.max_retries = max_retries
        self._breakers: dict[str, CircuitBreaker] = {}
        self._breaker_cfg = (breaker_fail_threshold, breaker_reset_s)

    def breaker(self, name: str) -> CircuitBreaker:
        if name not in self._breakers:
            threshold, reset = self._breaker_cfg
            self._breakers[name] = CircuitBreaker(name, threshold, reset)
        return self._breakers[name]

    def breaker_states(self) -> list[dict]:
        return [b.snapshot() for b in self._breakers.values()]

    # ------------------------------------------------------------------
    async def run_stage(
        self,
        name: str,
        fn: Callable[[], Awaitable[T]],
        *,
        timer: Timer,
        budget: Budget,
        timeout_ms: float | None = None,
        retries: int | None = None,
        fallback: T | None = None,
        raise_on_fail: bool = False,
        breaker: str | None = None,
        is_retryable: Callable[[BaseException], bool] = default_is_retryable,
        min_budget_ms: float = 1.0,
    ) -> T:
        """Run one stage under the request budget. Returns `fallback` on failure
        unless `raise_on_fail`."""
        retries = self.max_retries if retries is None else retries
        cb = self.breaker(breaker) if breaker else None
        started = time.perf_counter()

        # Don't even start a stage we can't afford.
        if budget.exhausted(min_budget_ms):
            timer.record(name, (time.perf_counter() - started) * 1000, ok=False,
                         detail="skipped: budget exhausted")
            if raise_on_fail:
                raise DeadlineExceeded(f"no budget left for stage {name!r}")
            return fallback  # type: ignore[return-value]

        last_exc: BaseException | None = None
        for attempt in range(retries + 1):
            slice_ms = timeout_ms if timeout_ms is not None else budget.remaining_ms
            slice_s = max(0.001, min(slice_ms, max(0.0, budget.remaining_ms)) / 1000.0)
            try:
                if cb is not None:
                    await cb.before()
                result = await asyncio.wait_for(fn(), timeout=slice_s)
                if cb is not None:
                    await cb.on_success()
                timer.record(name, (time.perf_counter() - started) * 1000, ok=True)
                return result
            except CircuitOpen as exc:
                last_exc = exc
                break  # no point retrying an open circuit
            except Exception as exc:  # noqa: BLE001 - asyncio.TimeoutError included
                last_exc = exc
                if cb is not None:
                    await cb.on_failure()
                retryable = isinstance(exc, asyncio.TimeoutError) or is_retryable(exc)
                if attempt >= retries or not retryable:
                    break
                # Exponential backoff with full jitter, clamped to the budget.
                backoff = min(0.05 * (2**attempt), 0.5)
                delay = random.uniform(0, backoff)
                if budget.remaining_ms <= (delay * 1000 + min_budget_ms):
                    break
                logger.info("stage %s attempt %d failed (%s); retrying in %.0fms",
                            name, attempt + 1, type(exc).__name__, delay * 1000)
                await asyncio.sleep(delay)

        timer.record(
            name,
            (time.perf_counter() - started) * 1000,
            ok=False,
            detail=f"{type(last_exc).__name__}: {last_exc}" if last_exc else "failed",
        )
        if raise_on_fail:
            raise StageFailed(name, last_exc or RuntimeError("unknown"))
        logger.warning("stage %s degraded to fallback: %s", name, last_exc)
        return fallback  # type: ignore[return-value]

    # ------------------------------------------------------------------
    async def run_sync(
        self,
        name: str,
        fn: Callable[[], T],
        *,
        timer: Timer,
        budget: Budget,
        **kwargs: Any,
    ) -> T:
        """Run CPU-bound work (embedding, ANN search) in the default executor so
        the event loop keeps serving other requests."""

        async def _wrapped() -> T:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, fn)

        return await self.run_stage(name, _wrapped, timer=timer, budget=budget, **kwargs)


async def gather_stages(
    coros: Sequence[Awaitable[Any]], return_exceptions: bool = True
) -> list[Any]:
    return list(await asyncio.gather(*coros, return_exceptions=return_exceptions))
