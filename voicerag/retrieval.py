"""Multi-strategy hybrid retrieval.

For each active chunking strategy we run a dense ANN search and a BM25 search,
then fuse every resulting ranking with RRF. Fusing across *strategies* as well
as modalities is the point: a sentence-window index nails precise factoid
matches, a parent-child index recovers context, and BM25 catches the rare
proper noun that a 384-dim embedding smooths away. Any one of them alone loses
a class of query the others handle.

After fusion: collapse parent-child siblings, suppress span-overlapping
duplicates, then MMR for diversity. The raw max cosine is carried out
separately because the out-of-domain guardrail needs a calibrated similarity,
not a fused rank score.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import Settings
from .embeddings import Encoder
from .index import (
    StrategyShard,
    mmr_select,
    reciprocal_rank_fusion,
    suppress_span_overlap,
)
from .schemas import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk] = field(default_factory=list)
    max_cosine: float = 0.0
    # Best raw BM25 score. Carried out separately from the fused rank score
    # because the out-of-domain rail needs an absolute lexical signal.
    max_sparse: float = 0.0
    max_fused: float = 0.0
    n_candidates: int = 0
    strategies_used: list[str] = field(default_factory=list)


class Retriever:
    def __init__(self, shards: dict[str, StrategyShard], encoder: Encoder, settings: Settings):
        self.shards = shards
        self.encoder = encoder
        self.settings = settings
        # chunk_id -> (strategy, row index within that shard)
        self._locate: dict[str, tuple[str, int]] = {}
        self._chunk_by_id: dict[str, Chunk] = {}
        for strategy, shard in shards.items():
            for row, chunk in enumerate(shard.chunks):
                self._locate[chunk.chunk_id] = (strategy, row)
                self._chunk_by_id[chunk.chunk_id] = chunk
        logger.info(
            "retriever ready: %d strategies, %d chunks total",
            len(shards),
            len(self._chunk_by_id),
        )

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, settings: Settings, encoder: Encoder) -> "Retriever":
        root: Path = settings.index_dir
        shards: dict[str, StrategyShard] = {}
        for strategy in settings.active_strategies:
            path = root / strategy
            if not (path / "chunks.jsonl").exists():
                logger.warning("index for strategy %r not found at %s; skipping", strategy, path)
                continue
            shards[strategy] = StrategyShard.load(
                root, strategy, ef_search=settings.hnsw_ef_search
            )
        if not shards:
            raise FileNotFoundError(
                f"no strategy indices found under {root}. Run scripts/build_index.py first."
            )
        return cls(shards, encoder, settings)

    @property
    def n_chunks(self) -> int:
        return len(self._chunk_by_id)

    # ------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        lang: str | None = None,
        query_vector: np.ndarray | None = None,
    ) -> RetrievalResult:
        cfg = self.settings
        top_k = top_k or cfg.final_top_k
        if query_vector is None:
            query_vector = self.encoder.encode_query(query)

        rankings: dict[str, list[str]] = {}
        dense_scores: dict[str, float] = {}
        sparse_scores: dict[str, float] = {}
        max_cosine = 0.0

        for strategy, shard in self.shards.items():
            # --- dense ---
            ids, sims = shard.dense.search(query_vector, cfg.dense_top_k)
            dense_ids: list[str] = []
            for row, sim in zip(ids, sims):
                if row < 0 or row >= len(shard.chunks):
                    continue
                chunk_id = shard.chunks[row].chunk_id
                dense_ids.append(chunk_id)
                sim_f = float(sim)
                if sim_f > dense_scores.get(chunk_id, -1.0):
                    dense_scores[chunk_id] = sim_f
                max_cosine = max(max_cosine, sim_f)
            if dense_ids:
                rankings[f"{strategy}::dense"] = dense_ids

            # --- sparse ---
            ids_s, scores_s = shard.sparse.search(query, cfg.sparse_top_k)
            sparse_ids: list[str] = []
            for row, score in zip(ids_s, scores_s):
                if row < 0 or row >= len(shard.chunks):
                    continue
                chunk_id = shard.chunks[row].chunk_id
                sparse_ids.append(chunk_id)
                sparse_scores[chunk_id] = max(sparse_scores.get(chunk_id, 0.0), float(score))
            if sparse_ids:
                rankings[f"{strategy}::sparse"] = sparse_ids

        max_sparse = max(sparse_scores.values()) if sparse_scores else 0.0

        if not rankings:
            return RetrievalResult(max_cosine=max_cosine, max_sparse=max_sparse)

        # Ranking design — all three decisions are measured, not assumed
        # (scripts/bench_retrieval.py + ablations over 300 labelled queries).
        #
        # 1. Dense is the primary ranker; BM25 augments recall rather than
        #    re-ranking:
        #       dense only                     nDCG@10 0.440
        #       dense-primary + BM25 appended  nDCG@10 0.438
        #       RRF dense-weighted 3:1         nDCG@10 0.383
        #       RRF equal weight               nDCG@10 0.348
        #       RRF equal + aggressive MMR     nDCG@10 0.332  <- first design
        #    E5 is a far stronger ranker than BM25 on natural-language
        #    questions, and equal-weight RRF let the weaker signal cost a
        #    quarter of the nDCG.
        #
        # 2. Across strategies, fuse by max score, not RRF:
        #       score-max   nDCG@10 0.443   R@5 0.800
        #       RRF         nDCG@10 0.436   R@5 0.787
        #       best single nDCG@10 0.441   R@5 0.800
        #    RRF rewards chunks ranked highly by *both* strategies and
        #    penalises one found by only one — but a passage the parent-child
        #    index alone surfaced is not less relevant for being missed by the
        #    atomic index. Score-max is the only fusion that beat the best
        #    single strategy, which is what justifies shipping an ensemble.
        #
        # Because each chunk belongs to exactly one strategy, taking the dense
        # cosine per chunk and sorting descending *is* score-max fusion.
        fused: dict[str, float] = dict(dense_scores)
        if not fused:
            fused = {
                chunk_id: score for chunk_id, score in sparse_scores.items()
            }
        if not fused:
            return RetrievalResult(max_cosine=max_cosine, max_sparse=max_sparse)

        # BM25 hits dense missed entirely, appended below every dense
        # candidate: keeps lexical recall for rare terms and proper nouns
        # without letting BM25 reorder anything dense already got right.
        if dense_scores:
            floor = min(dense_scores.values())
            for source, ids in rankings.items():
                if not source.endswith("::sparse"):
                    continue
                for rank, chunk_id in enumerate(ids):
                    if chunk_id not in fused:
                        fused[chunk_id] = floor * 0.5 * (1.0 - rank / (len(ids) + 1))

        # Optional metadata filter — the chunks carry `lang`, so this is a
        # cheap post-filter rather than a separate index per language.
        if lang:
            filtered = {
                cid: s
                for cid, s in fused.items()
                if self._chunk_by_id[cid].lang == lang
            }
            if filtered:
                fused = filtered

        # Per-chunk scores normalised to the top hit purely for display /
        # relative weighting. `max_fused` deliberately keeps the RAW top score:
        # an earlier version reported max(normalised), which is 1.0 by
        # construction, so the guardrail reading it could never fire.
        max_fused = max(fused.values())
        normalised = {cid: v / max_fused for cid, v in fused.items()}

        ordered = sorted(normalised, key=normalised.get, reverse=True)
        pool = ordered[: max(top_k * 4, 24)]

        # Collapse parent-child siblings: keep the best-scoring child per parent.
        collapsed: list[str] = []
        seen_parents: set[str] = set()
        for cid in pool:
            chunk = self._chunk_by_id[cid]
            if chunk.parent_id:
                if chunk.parent_id in seen_parents:
                    continue
                seen_parents.add(chunk.parent_id)
            collapsed.append(cid)

        vectors = self._vectors_for(collapsed)
        selected = mmr_select(
            collapsed, normalised, vectors, k=top_k * 2, lambda_=cfg.mmr_lambda
        )

        picked = suppress_span_overlap([self._chunk_by_id[c] for c in selected])[:top_k]

        results = [
            RetrievedChunk(
                chunk=chunk,
                score=normalised.get(chunk.chunk_id, 0.0),
                dense_score=dense_scores.get(chunk.chunk_id),
                sparse_score=sparse_scores.get(chunk.chunk_id),
                rank_sources=[
                    src for src, ids in rankings.items() if chunk.chunk_id in ids
                ],
            )
            for chunk in picked
        ]

        return RetrievalResult(
            chunks=results,
            max_cosine=max_cosine,
            max_sparse=max_sparse,
            max_fused=max_fused,
            n_candidates=len(fused),
            strategies_used=sorted({c.chunk.strategy for c in results}),
        )

    # ------------------------------------------------------------------
    def _vectors_for(self, chunk_ids: list[str]) -> dict[str, np.ndarray]:
        """Pull stored vectors for the candidate set, grouped per shard so each
        index is touched once."""
        by_strategy: dict[str, list[tuple[str, int]]] = {}
        for cid in chunk_ids:
            loc = self._locate.get(cid)
            if loc is None:
                continue
            strategy, row = loc
            by_strategy.setdefault(strategy, []).append((cid, row))

        out: dict[str, np.ndarray] = {}
        for strategy, pairs in by_strategy.items():
            shard = self.shards[strategy]
            try:
                vecs = shard.dense.get_vectors([row for _, row in pairs])
            except Exception:  # noqa: BLE001 - MMR degrades to pure relevance
                continue
            for (cid, _), vec in zip(pairs, vecs):
                out[cid] = vec
        return out
