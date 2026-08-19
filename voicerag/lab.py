"""Chunking Lab — run one question through every strategy at once.

Chunking is the part of a RAG system nobody can see. Every team claims a
thoughtful strategy; the reader has to take it on faith. This runs the same
query against all eight indices simultaneously and returns what each one
actually retrieved, so the difference is visible rather than asserted.

It reads a separate index tree (`data/index_lab`) built over the same 300-query
subset that `scripts/bench_retrieval.py` scores, so the quality numbers shown
next to each strategy come from exactly the corpus being queried. The live
service keeps its own, larger index of only the two strategies the benchmark
justified shipping.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from .chunking import ALL_STRATEGIES
from .config import PROJECT_ROOT, Settings
from .embeddings import Encoder
from .index import StrategyShard

logger = logging.getLogger(__name__)

LAB_INDEX_DIR = PROJECT_ROOT / "data" / "index_lab"
BENCHMARK_FILE = PROJECT_ROOT / "benchmarks" / "retrieval.json"

# One line per strategy explaining what it is actually for, shown in the UI so
# the comparison reads as engineering rather than a leaderboard.
STRATEGY_NOTES = {
    "passage_atomic": "One chunk per passage — MS MARCO's natural unit.",
    "fixed_window": "180-token windows, 45 overlap. The naive baseline, kept to show its cost.",
    "recursive_structural": "Paragraph → sentence → word. Never splits mid-sentence.",
    "sentence_window": "Indexes one sentence, returns ±2 for context.",
    "semantic_drift": "Cuts where consecutive-sentence similarity drops.",
    "proposition": "Decomposes into atomic claims, one fact per chunk.",
    "hierarchical_parent_child": "Small children indexed, whole parent returned.",
    "late_chunking": "Pools token embeddings from one full-document pass.",
}


class ChunkingLab:
    def __init__(self, shards: dict[str, StrategyShard], encoder: Encoder, settings: Settings):
        self.shards = shards
        self.encoder = encoder
        self.settings = settings
        self.benchmark = self._load_benchmark()

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, settings: Settings, encoder: Encoder) -> "ChunkingLab | None":
        if not LAB_INDEX_DIR.exists():
            logger.info("chunking lab index not found at %s; lab disabled", LAB_INDEX_DIR)
            return None
        shards: dict[str, StrategyShard] = {}
        for strategy in ALL_STRATEGIES:
            if (LAB_INDEX_DIR / strategy / "chunks.jsonl").exists():
                try:
                    shards[strategy] = StrategyShard.load(
                        LAB_INDEX_DIR, strategy, ef_search=settings.hnsw_ef_search
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("lab: could not load %s (%s)", strategy, exc)
        if not shards:
            return None
        logger.info("chunking lab ready: %d strategies", len(shards))
        return cls(shards, encoder, settings)

    # ------------------------------------------------------------------
    @staticmethod
    def _load_benchmark() -> dict:
        if not BENCHMARK_FILE.exists():
            return {}
        try:
            data = json.loads(BENCHMARK_FILE.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
        out = {}
        for name, entry in (data.get("results") or {}).items():
            metrics = (entry.get("metrics") or {}).get("hybrid") or {}
            out[name] = {
                "recall_at_5": metrics.get("recall@5"),
                "mrr_at_10": metrics.get("mrr@10"),
                "ndcg_at_10": metrics.get("ndcg@10"),
                "chunks_per_doc": (entry.get("chunk_stats") or {}).get("chunks_per_doc"),
                "tokens_p50": (entry.get("chunk_stats") or {}).get("tokens_p50"),
            }
        return out

    # ------------------------------------------------------------------
    @property
    def info(self) -> dict:
        return {
            "available": True,
            "n_strategies": len(self.shards),
            "strategies": sorted(self.shards),
            "corpus_note": (
                "Lab runs over a 300-query / 5,967-passage subset — the same set "
                "scripts/bench_retrieval.py scores, so the metrics beside each "
                "strategy describe exactly this corpus."
            ),
            "shipped_live": self.settings.active_strategies,
            "benchmark": self.benchmark,
            "notes": STRATEGY_NOTES,
        }

    # ------------------------------------------------------------------
    def race(self, query: str, top_k: int = 3) -> dict:
        """Query every strategy independently and report what each retrieved."""
        t0 = time.perf_counter()
        query_vector = self.encoder.encode_query(query, use_cache=False)
        embed_ms = (time.perf_counter() - t0) * 1000

        results = []
        for strategy in sorted(self.shards):
            shard = self.shards[strategy]
            t1 = time.perf_counter()
            ids, sims = shard.dense.search(query_vector, max(top_k * 3, 12))
            search_ms = (time.perf_counter() - t1) * 1000

            hits, seen_docs = [], set()
            for row, sim in zip(ids, sims):
                if row < 0 or row >= len(shard.chunks):
                    continue
                chunk = shard.chunks[row]
                # Collapse to one hit per source passage so a strategy that
                # emits many chunks per document doesn't just fill the list
                # with neighbours of the same passage.
                if chunk.doc_id in seen_docs:
                    continue
                seen_docs.add(chunk.doc_id)
                hits.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "lang": chunk.lang,
                        "text": chunk.text,
                        "context_text": chunk.context_text,
                        "score": round(float(sim), 4),
                        "token_count": chunk.token_count,
                        "is_gold": chunk.is_selected,
                        "char_span": [chunk.char_start, chunk.char_end],
                    }
                )
                if len(hits) >= top_k:
                    break

            results.append(
                {
                    "strategy": strategy,
                    "note": STRATEGY_NOTES.get(strategy, ""),
                    "shipped_live": strategy in self.settings.active_strategies,
                    "n_chunks": len(shard.chunks),
                    "search_ms": round(search_ms, 2),
                    "top_score": hits[0]["score"] if hits else 0.0,
                    "found_gold": any(h["is_gold"] for h in hits),
                    "benchmark": self.benchmark.get(strategy, {}),
                    "hits": hits,
                }
            )

        # Rank by measured quality where we have it, else by this query's score.
        results.sort(
            key=lambda r: (
                r["benchmark"].get("ndcg_at_10") or 0.0,
                r["top_score"],
            ),
            reverse=True,
        )
        return {
            "query": query,
            "embed_ms": round(embed_ms, 2),
            "total_ms": round((time.perf_counter() - t0) * 1000, 2),
            "n_strategies": len(results),
            "results": results,
        }
