"""Vector and lexical indices, plus rank fusion.

Dense  : hnswlib HNSW over cosine space, with an exact numpy fallback so the
         service still runs where hnswlib can't be built.
Sparse : BM25 held as a precomputed CSC weight matrix. Because the BM25 weight
         of a (doc, term) pair doesn't depend on the query, the whole scoring
         step collapses to "sum a few sparse columns" — a couple of ms over
         100k docs instead of a Python loop.
Fusion : Reciprocal Rank Fusion across every (strategy x modality) ranking.
         RRF is used deliberately: dense cosine and BM25 scores aren't on a
         comparable scale, and rank-based fusion sidesteps the normalisation
         problem entirely.
"""

from __future__ import annotations

import json
import logging
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import scipy.sparse as sp

from .schemas import Chunk

logger = logging.getLogger(__name__)

_TOKEN = re.compile(r"[\wऀ-෿؀-ۿ]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


# --------------------------------------------------------------------------
# Dense
# --------------------------------------------------------------------------
class DenseIndex:
    def __init__(self, dim: int, ef_search: int = 64, m: int = 16, ef_construction: int = 200):
        self.dim = dim
        self.ef_search = ef_search
        self.m = m
        self.ef_construction = ef_construction
        self._hnsw = None
        self._matrix: np.ndarray | None = None
        self.size = 0

    @property
    def backend(self) -> str:
        return "hnswlib" if self._hnsw is not None else "numpy-exact"

    def build(self, vectors: np.ndarray) -> None:
        vectors = np.ascontiguousarray(vectors.astype(np.float32))
        self.size = vectors.shape[0]
        try:
            import hnswlib

            index = hnswlib.Index(space="cosine", dim=self.dim)
            index.init_index(
                max_elements=max(1, self.size),
                ef_construction=self.ef_construction,
                M=self.m,
            )
            index.add_items(vectors, np.arange(self.size))
            index.set_ef(self.ef_search)
            self._hnsw = index
            logger.info("dense index: hnswlib, %d vectors", self.size)
        except Exception as exc:  # noqa: BLE001
            logger.warning("dense index: hnswlib unavailable (%s); exact numpy search", exc)
            self._matrix = vectors

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        k = min(k, max(1, self.size))
        if self._hnsw is not None:
            labels, distances = self._hnsw.knn_query(query.reshape(1, -1), k=k)
            # hnswlib cosine "distance" is 1 - cosine similarity
            return labels[0], (1.0 - distances[0])
        sims = self._matrix @ query.astype(np.float32)
        if k >= sims.shape[0]:
            idx = np.argsort(-sims)
        else:
            idx = np.argpartition(-sims, k)[:k]
            idx = idx[np.argsort(-sims[idx])]
        return idx, sims[idx]

    def get_vectors(self, ids: Sequence[int]) -> np.ndarray:
        """Fetch stored vectors for a handful of ids (used by MMR).

        Only ever called on the ~40 fused candidates, so pulling from the index
        is cheaper than keeping a full float32 copy of the corpus resident.
        """
        ids = list(ids)
        if not ids:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self._hnsw is not None:
            return np.asarray(self._hnsw.get_items(ids), dtype=np.float32)
        return self._matrix[ids]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._hnsw is not None:
            self._hnsw.save_index(str(path.with_suffix(".hnsw")))
            meta = {"backend": "hnswlib", "size": self.size, "dim": self.dim}
        else:
            np.save(path.with_suffix(".npy"), self._matrix)
            meta = {"backend": "numpy", "size": self.size, "dim": self.dim}
        path.with_suffix(".meta.json").write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, ef_search: int = 64) -> "DenseIndex":
        meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
        obj = cls(dim=meta["dim"], ef_search=ef_search)
        obj.size = meta["size"]
        if meta["backend"] == "hnswlib":
            import hnswlib

            index = hnswlib.Index(space="cosine", dim=meta["dim"])
            index.load_index(str(path.with_suffix(".hnsw")), max_elements=meta["size"])
            index.set_ef(ef_search)
            obj._hnsw = index
        else:
            obj._matrix = np.load(path.with_suffix(".npy"))
        return obj


# --------------------------------------------------------------------------
# Sparse (BM25)
# --------------------------------------------------------------------------
class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.vocab: dict[str, int] = {}
        self.weights: sp.csc_matrix | None = None
        self.size = 0

    def build(self, texts: Sequence[str]) -> None:
        self.size = len(texts)
        rows, cols, vals = [], [], []
        doc_lengths = np.zeros(self.size, dtype=np.float32)

        for doc_i, text in enumerate(texts):
            counts: dict[int, int] = {}
            tokens = tokenize(text)
            doc_lengths[doc_i] = len(tokens)
            for tok in tokens:
                tid = self.vocab.get(tok)
                if tid is None:
                    tid = len(self.vocab)
                    self.vocab[tok] = tid
                counts[tid] = counts.get(tid, 0) + 1
            for tid, tf in counts.items():
                rows.append(doc_i)
                cols.append(tid)
                vals.append(tf)

        n_terms = max(1, len(self.vocab))
        tf_matrix = sp.csr_matrix(
            (np.array(vals, dtype=np.float32), (rows, cols)),
            shape=(max(1, self.size), n_terms),
        )

        avgdl = float(doc_lengths.mean()) if self.size else 1.0
        avgdl = avgdl or 1.0
        df = np.asarray((tf_matrix > 0).sum(axis=0)).ravel()
        idf = np.log(1.0 + (self.size - df + 0.5) / (df + 0.5)).astype(np.float32)

        # Precompute the full BM25 weight for every stored (doc, term) pair.
        coo = tf_matrix.tocoo()
        tf = coo.data
        dl = doc_lengths[coo.row]
        denom = tf + self.k1 * (1.0 - self.b + self.b * dl / avgdl)
        weight = idf[coo.col] * (tf * (self.k1 + 1.0)) / np.maximum(denom, 1e-9)

        self.weights = sp.csc_matrix(
            (weight.astype(np.float32), (coo.row, coo.col)), shape=tf_matrix.shape
        )
        logger.info("bm25 index: %d docs, %d terms", self.size, len(self.vocab))

    def search(self, query: str, k: int) -> tuple[np.ndarray, np.ndarray]:
        if self.weights is None or self.size == 0:
            return np.array([], dtype=int), np.array([], dtype=np.float32)
        term_ids = [self.vocab[t] for t in tokenize(query) if t in self.vocab]
        if not term_ids:
            return np.array([], dtype=int), np.array([], dtype=np.float32)

        scores = np.zeros(self.size, dtype=np.float32)
        for tid in term_ids:
            col = self.weights.getcol(tid)
            scores[col.indices] += col.data

        nonzero = np.flatnonzero(scores)
        if nonzero.size == 0:
            return np.array([], dtype=int), np.array([], dtype=np.float32)
        k = min(k, nonzero.size)
        top = nonzero[np.argpartition(-scores[nonzero], k - 1)[:k]]
        top = top[np.argsort(-scores[top])]
        return top, scores[top]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(
                {"k1": self.k1, "b": self.b, "vocab": self.vocab,
                 "weights": self.weights, "size": self.size},
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        with path.open("rb") as fh:
            state = pickle.load(fh)
        obj = cls(k1=state["k1"], b=state["b"])
        obj.vocab = state["vocab"]
        obj.weights = state["weights"]
        obj.size = state["size"]
        return obj


# --------------------------------------------------------------------------
# Per-strategy shard
# --------------------------------------------------------------------------
@dataclass
class StrategyShard:
    strategy: str
    chunks: list[Chunk]
    dense: DenseIndex
    sparse: BM25Index

    def save(self, root: Path) -> None:
        out = root / self.strategy
        out.mkdir(parents=True, exist_ok=True)
        self.dense.save(out / "dense")
        self.sparse.save(out / "sparse.pkl")
        with (out / "chunks.jsonl").open("w", encoding="utf-8") as fh:
            for chunk in self.chunks:
                fh.write(chunk.model_dump_json() + "\n")

    @classmethod
    def load(cls, root: Path, strategy: str, ef_search: int = 64) -> "StrategyShard":
        src = root / strategy
        chunks = []
        with (src / "chunks.jsonl").open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    chunks.append(Chunk.model_validate_json(line))
        return cls(
            strategy=strategy,
            chunks=chunks,
            dense=DenseIndex.load(src / "dense", ef_search=ef_search),
            sparse=BM25Index.load(src / "sparse.pkl"),
        )


# --------------------------------------------------------------------------
# Rank fusion
# --------------------------------------------------------------------------
def reciprocal_rank_fusion(
    rankings: dict[str, Sequence[str]], k: int = 60, weights: dict[str, float] | None = None
) -> dict[str, float]:
    """RRF over named rankings of chunk ids. Returns id -> fused score."""
    fused: dict[str, float] = {}
    for source, ranked_ids in rankings.items():
        weight = (weights or {}).get(source, 1.0)
        for rank, chunk_id in enumerate(ranked_ids):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + weight / (k + rank + 1)
    return fused


def mmr_select(
    candidate_ids: Sequence[str],
    scores: dict[str, float],
    vectors: dict[str, np.ndarray],
    k: int,
    lambda_: float = 0.72,
) -> list[str]:
    """Maximal Marginal Relevance — trades relevance against redundancy.

    Overlapping chunking strategies produce near-duplicate hits by construction
    (a sentence window and its parent passage say the same thing). Without this,
    the top-k is often four restatements of one fact and no second fact.
    """
    if not candidate_ids:
        return []
    selected: list[str] = []
    remaining = list(candidate_ids)
    while remaining and len(selected) < k:
        best_id, best_val = None, -np.inf
        for cid in remaining:
            relevance = scores.get(cid, 0.0)
            if selected and cid in vectors:
                sims = [
                    float(np.dot(vectors[cid], vectors[s]))
                    for s in selected
                    if s in vectors
                ]
                redundancy = max(sims) if sims else 0.0
            else:
                redundancy = 0.0
            val = lambda_ * relevance - (1.0 - lambda_) * redundancy
            if val > best_val:
                best_id, best_val = cid, val
        if best_id is None:
            break
        selected.append(best_id)
        remaining.remove(best_id)
    return selected


def suppress_span_overlap(chunks: Iterable[Chunk], max_overlap: float = 0.6) -> list[Chunk]:
    """Drop chunks whose character span is mostly covered by an already-kept
    chunk from the same document. This is the overlap-handling counterpart to
    the deliberate overlap introduced by the fixed-window and recursive
    strategies: we want the overlap during matching, not in the final context.
    """
    kept: list[Chunk] = []
    for chunk in chunks:
        redundant = False
        for existing in kept:
            if existing.doc_id != chunk.doc_id:
                continue
            lo = max(existing.char_start, chunk.char_start)
            hi = min(existing.char_end, chunk.char_end)
            overlap = max(0, hi - lo)
            span = max(1, chunk.char_end - chunk.char_start)
            if overlap / span >= max_overlap:
                redundant = True
                break
        if not redundant:
            kept.append(chunk)
    return kept
