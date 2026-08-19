"""Latency benchmark — P50 / P70 / P90 / P100 over real queries.

Two things this does deliberately, because both change the numbers a lot:

1. **The query embedding cache is disabled.** The service keeps an LRU cache of
   query vectors, which is correct in production but would turn a repeated
   benchmark query into a 0.2ms no-op and make the percentiles meaningless.
   Every measured request here pays full encoder cost.

2. **Warm-up requests are excluded.** The first few calls pay lazy kernel init
   and page-in costs. Including them would put a 400ms outlier in P100 that
   says nothing about steady-state behaviour.

Reported separately:
    core   — post-STT pipeline (guardrails + embed + retrieve + synthesise +
             verify). This is the number measured against the 200ms target.
    stages — per-stage percentiles, so a regression is attributable.

Usage:
    python scripts/bench_latency.py --n 300
    python scripts/bench_latency.py --n 60 --mode grounded   # needs ANTHROPIC_API_KEY
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voicerag.config import get_settings  # noqa: E402
from voicerag.embeddings import Encoder  # noqa: E402
from voicerag.pipeline import RagService  # noqa: E402
from voicerag.retrieval import Retriever  # noqa: E402
from voicerag.telemetry import percentiles  # noqa: E402


def load_queries(path: Path, n: int, seed: int) -> list[str]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    random.Random(seed).shuffle(rows)
    out: list[str] = []
    for row in rows:
        q = (row.get("query") or "").strip()
        if q:
            out.append(q)
        if len(out) >= n:
            break
    return out


async def run(args) -> int:
    settings = get_settings()
    queries_path = settings.corpus_dir / "queries.jsonl"
    if not queries_path.exists():
        print(f"missing {queries_path}; run scripts/prepare_corpus.py", file=sys.stderr)
        return 1

    print(f"loading encoder {settings.embed_model}…")
    encoder = Encoder(
        settings.embed_model,
        max_tokens=settings.embed_max_tokens,
        quantize=settings.embed_quantize,
        threads=settings.embed_threads,
    )
    retriever = Retriever.load(settings, encoder)
    service = RagService(settings, encoder, retriever, stt=None)

    queries = load_queries(queries_path, args.n + args.warmup, args.seed)
    if len(queries) < args.n + args.warmup:
        print(f"only {len(queries)} queries available", file=sys.stderr)

    warm, measured = queries[: args.warmup], queries[args.warmup : args.warmup + args.n]

    print(f"index: {retriever.n_chunks:,} chunks across {list(retriever.shards)}")
    print(f"warming up ({len(warm)} requests)…")
    encoder.warmup(3)
    for q in warm:
        await service.answer(q, mode=args.mode)

    # This is the load-bearing line: without it every repeated query is a cache
    # hit and the embed stage reads ~0.2ms instead of its real cost.
    encoder.cache_enabled = False
    service.telemetry.reset()

    print(f"measuring {len(measured)} requests (mode={args.mode}, query cache OFF)…")
    core: list[float] = []
    total: list[float] = []
    per_stage: dict[str, list[float]] = {}
    answered = 0
    abstained = 0
    t0 = time.perf_counter()

    for i, q in enumerate(measured, 1):
        resp = await service.answer(q, mode=args.mode)
        core.append(resp.latency.core_ms)
        total.append(resp.latency.total_ms)
        for span in resp.latency.spans:
            per_stage.setdefault(span.name, []).append(span.duration_ms)
        if resp.answered:
            answered += 1
        else:
            abstained += 1
        if i % 50 == 0:
            print(f"  {i}/{len(measured)}  running P50={statistics.median(core):.1f}ms")

    wall = time.perf_counter() - t0
    core_p = percentiles(core)
    total_p = percentiles(total)

    print("\n" + "=" * 66)
    print(f"LATENCY — mode={args.mode}, n={len(measured)}, wall={wall:.1f}s "
          f"({len(measured)/wall:.1f} req/s serial)")
    print("=" * 66)
    print(f"{'metric':<26}{'P50':>9}{'P70':>9}{'P90':>9}{'P99':>9}{'P100':>9}")
    print("-" * 66)
    for label, p in (("core pipeline (ms)", core_p), ("end-to-end (ms)", total_p)):
        print(f"{label:<26}{p['p50']:>9}{p['p70']:>9}{p['p90']:>9}{p['p99']:>9}{p['p100']:>9}")

    print(f"\n{'per-stage (ms)':<26}{'P50':>9}{'P70':>9}{'P90':>9}{'P99':>9}{'P100':>9}")
    print("-" * 66)
    for name, values in sorted(per_stage.items(), key=lambda kv: -statistics.median(kv[1])):
        p = percentiles(values)
        print(f"{name:<26}{p['p50']:>9}{p['p70']:>9}{p['p90']:>9}{p['p99']:>9}{p['p100']:>9}")

    under = sum(1 for c in core if c < settings.pipeline_budget_ms)
    print("-" * 66)
    print(f"under {settings.pipeline_budget_ms:.0f}ms target : {under}/{len(core)} "
          f"({100*under/max(1,len(core)):.1f}%)")
    print(f"answered / abstained      : {answered} / {abstained}")

    report = {
        "mode": args.mode,
        "n_requests": len(measured),
        "n_chunks": retriever.n_chunks,
        "strategies": list(retriever.shards),
        "embed_model": settings.embed_model,
        "query_cache_disabled": True,
        "target_core_ms": settings.pipeline_budget_ms,
        "core_pipeline_ms": core_p,
        "end_to_end_ms": total_p,
        "per_stage_ms": {k: percentiles(v) for k, v in per_stage.items()},
        "under_target_pct": round(100 * under / max(1, len(core)), 2),
        "answered": answered,
        "abstained": abstained,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport -> {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=300, help="measured requests")
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--mode", default="fast", choices=["fast", "grounded"])
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default="benchmarks/latency.json")
    return asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
