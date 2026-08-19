"""Chunking-strategy comparison on labelled MSMARCO-XI relevance judgements.

This is the evidence behind the chunking design. Each strategy is built into
its own index over the same passages, then scored on the same queries using the
dataset's `is_selected` labels as ground truth:

    Recall@k  — did any gold passage make the top k
    MRR@10    — how high the first gold passage ranked
    nDCG@10   — full graded ranking quality (binary relevance)

Also evaluated: dense-only vs BM25-only vs the hybrid the live retriever runs
(so the fusion earns its place rather than being assumed), and the
multi-strategy score-max ensemble the live service actually uses.

Evaluation runs on a query-aligned subsample: we take N queries and *all* of
their passages, so every query keeps its gold documents and the distractor pool
grows with N. Building eight indices over the full corpus takes far longer than
it takes to answer the question of which strategy wins.

Usage:
    python scripts/bench_retrieval.py --queries 300
    python scripts/bench_retrieval.py --queries 500 --strategies all
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# We batch explicitly; the tokenizers rayon pool otherwise oversubscribes
# the CPU against torch's own threads (confirmed via a stack sample).
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voicerag.chunking import ALL_STRATEGIES, Document, build_chunker, chunk_stats  # noqa: E402
from voicerag.config import get_settings  # noqa: E402
from voicerag.embeddings import Encoder  # noqa: E402
from voicerag.index import BM25Index, DenseIndex, StrategyShard  # noqa: E402
from voicerag.retrieval import Retriever  # noqa: E402

K_VALUES = (1, 5, 10)


def dcg(relevances: list[int]) -> float:
    return sum(r / np.log2(i + 2) for i, r in enumerate(relevances))


def score_ranking(ranked_doc_ids: list[str], gold: set[str], k: int = 10) -> dict:
    out: dict[str, float] = {}
    for kk in K_VALUES:
        out[f"recall@{kk}"] = 1.0 if gold & set(ranked_doc_ids[:kk]) else 0.0

    rr = 0.0
    for i, doc_id in enumerate(ranked_doc_ids[:k]):
        if doc_id in gold:
            rr = 1.0 / (i + 1)
            break
    out["mrr@10"] = rr

    rels = [1 if d in gold else 0 for d in ranked_doc_ids[:k]]
    ideal = [1] * min(len(gold), k)
    idcg = dcg(ideal)
    out["ndcg@10"] = (dcg(rels) / idcg) if idcg > 0 else 0.0
    return out


def dedupe_ranked_docs(chunk_docs: list[str], limit: int) -> list[str]:
    """Chunk-level rankings collapse to document level for scoring."""
    seen: list[str] = []
    for doc_id in chunk_docs:
        if doc_id not in seen:
            seen.append(doc_id)
        if len(seen) >= limit:
            break
    return seen


def main() -> int:
    settings = get_settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", type=int, default=300)
    ap.add_argument("--strategies", default="all")
    ap.add_argument("--corpus", default=str(settings.corpus_dir / "corpus.jsonl"))
    ap.add_argument("--queries-file", default=str(settings.corpus_dir / "queries.jsonl"))
    ap.add_argument("--out", default="benchmarks/retrieval.json")
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    strategies = ALL_STRATEGIES if args.strategies == "all" else [
        s.strip() for s in args.strategies.split(",") if s.strip()
    ]

    # ---- load the query-aligned subsample ----------------------------
    queries = []
    with Path(args.queries_file).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                queries.append(json.loads(line))
            if len(queries) >= args.queries:
                break
    query_ids = {q["query_id"] for q in queries}

    docs: list[Document] = []
    with Path(args.corpus).open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("query_id") in query_ids:
                docs.append(
                    Document(
                        doc_id=row["doc_id"],
                        text=row["text"],
                        lang=row.get("lang", "unknown"),
                        query_id=row.get("query_id"),
                        passage_idx=row.get("passage_idx"),
                        is_selected=bool(row.get("is_selected")),
                    )
                )

    print(f"eval set: {len(queries)} queries, {len(docs)} candidate passages")
    print(f"loading encoder {settings.embed_model}…")
    encoder = Encoder(
        settings.embed_model,
        max_tokens=settings.embed_max_tokens,
        quantize=settings.embed_quantize,
        threads=settings.embed_threads,
    )

    def token_len(text: str) -> int:
        return len(encoder.tokenizer.encode(text, add_special_tokens=False))

    # Pre-encode every query once and reuse across strategies — the comparison
    # is about the index, not about repeated encoder cost.
    print("encoding queries…")
    query_vectors = encoder.encode_queries([q["query"] for q in queries], use_cache=False)

    results: dict[str, dict] = {}
    shards: dict[str, StrategyShard] = {}

    for strategy in strategies:
        print(f"\n=== {strategy} ===")
        t0 = time.perf_counter()
        chunker = build_chunker(strategy, encoder=encoder, token_len=token_len)
        chunks = chunker.chunk(docs)
        if not chunks:
            print("  no chunks; skipping")
            continue
        stats = chunk_stats(chunks)

        vectors = chunker.embed_override(chunks, encoder)
        if vectors is None:
            vectors = encoder.encode_passages([c.text for c in chunks], batch_size=args.batch_size)
        dense = DenseIndex(dim=int(vectors.shape[1]), ef_search=settings.hnsw_ef_search)
        dense.build(np.asarray(vectors, dtype=np.float32))
        sparse = BM25Index()
        sparse.build([c.payload for c in chunks])
        shards[strategy] = StrategyShard(strategy, chunks, dense, sparse)
        build_s = time.perf_counter() - t0
        print(f"  {stats['n_chunks']} chunks ({stats['chunks_per_doc']}/doc, "
              f"p50={stats['tokens_p50']} tok) built in {build_s:.1f}s")

        # --- score dense-only, sparse-only, and their RRF fusion --------
        agg = {m: {f"recall@{k}": [] for k in K_VALUES} for m in ("dense", "sparse", "hybrid")}
        for m in agg:
            agg[m]["mrr@10"] = []
            agg[m]["ndcg@10"] = []
        latencies: list[float] = []

        single = Retriever({strategy: shards[strategy]}, encoder, settings)

        for q, qv in zip(queries, query_vectors):
            gold = set(q["gold_doc_ids"])

            ids_d, _ = dense.search(qv, 50)
            dense_docs = dedupe_ranked_docs([chunks[i].doc_id for i in ids_d if i < len(chunks)], 10)

            ids_s, _ = sparse.search(q["query"], 50)
            sparse_docs = dedupe_ranked_docs([chunks[i].doc_id for i in ids_s if i < len(chunks)], 10)

            t1 = time.perf_counter()
            res = single.retrieve(q["query"], top_k=10, query_vector=qv)
            latencies.append((time.perf_counter() - t1) * 1000)
            hybrid_docs = dedupe_ranked_docs([rc.chunk.doc_id for rc in res.chunks], 10)

            for name, ranked in (("dense", dense_docs), ("sparse", sparse_docs), ("hybrid", hybrid_docs)):
                for metric, value in score_ranking(ranked, gold).items():
                    agg[name][metric].append(value)

        results[strategy] = {
            "chunk_stats": stats,
            "build_seconds": round(build_s, 1),
            "retrieve_p50_ms": round(float(np.percentile(latencies, 50)), 2),
            "metrics": {
                mode: {k: round(float(np.mean(v)), 4) for k, v in metrics.items()}
                for mode, metrics in agg.items()
            },
        }
        h = results[strategy]["metrics"]["hybrid"]
        print(f"  hybrid  R@1={h['recall@1']:.3f}  R@5={h['recall@5']:.3f}  "
              f"MRR@10={h['mrr@10']:.3f}  nDCG@10={h['ndcg@10']:.3f}")

    # ---- the ensemble the live service uses ---------------------------
    ensemble_names = [s for s in settings.active_strategies if s in shards]
    if len(ensemble_names) > 1:
        print(f"\n=== score-max ensemble: {ensemble_names} ===")
        ens = Retriever({n: shards[n] for n in ensemble_names}, encoder, settings)
        agg = {f"recall@{k}": [] for k in K_VALUES}
        agg["mrr@10"], agg["ndcg@10"] = [], []
        lat: list[float] = []
        for q, qv in zip(queries, query_vectors):
            t1 = time.perf_counter()
            res = ens.retrieve(q["query"], top_k=10, query_vector=qv)
            lat.append((time.perf_counter() - t1) * 1000)
            ranked = dedupe_ranked_docs([rc.chunk.doc_id for rc in res.chunks], 10)
            for metric, value in score_ranking(ranked, set(q["gold_doc_ids"])).items():
                agg[metric].append(value)
        results["__ensemble__"] = {
            "strategies": ensemble_names,
            "retrieve_p50_ms": round(float(np.percentile(lat, 50)), 2),
            "metrics": {"hybrid": {k: round(float(np.mean(v)), 4) for k, v in agg.items()}},
        }
        e = results["__ensemble__"]["metrics"]["hybrid"]
        print(f"  R@1={e['recall@1']:.3f}  R@5={e['recall@5']:.3f}  "
              f"MRR@10={e['mrr@10']:.3f}  nDCG@10={e['ndcg@10']:.3f}")

    # ---- report -------------------------------------------------------
    print("\n" + "=" * 104)
    print(f"RETRIEVAL QUALITY — {len(queries)} queries, {len(docs)} passages "
          f"(hybrid = dense-primary + BM25 recall augmentation)")
    print("=" * 104)
    hdr = (f"{'strategy':<28}{'chunks':>8}{'/doc':>7}{'R@1':>8}{'R@5':>8}{'R@10':>8}"
           f"{'MRR@10':>9}{'nDCG@10':>9}{'dense R@5':>11}{'bm25 R@5':>10}{'ms':>7}")
    print(hdr)
    print("-" * 104)
    ordered = sorted(
        (k for k in results if k != "__ensemble__"),
        key=lambda k: -results[k]["metrics"]["hybrid"]["ndcg@10"],
    )
    for name in ordered:
        r = results[name]
        h, d, s = r["metrics"]["hybrid"], r["metrics"]["dense"], r["metrics"]["sparse"]
        cs = r["chunk_stats"]
        print(f"{name:<28}{cs['n_chunks']:>8}{cs['chunks_per_doc']:>7}"
              f"{h['recall@1']:>8.3f}{h['recall@5']:>8.3f}{h['recall@10']:>8.3f}"
              f"{h['mrr@10']:>9.3f}{h['ndcg@10']:>9.3f}"
              f"{d['recall@5']:>11.3f}{s['recall@5']:>10.3f}{r['retrieve_p50_ms']:>7.1f}")
    if "__ensemble__" in results:
        e = results["__ensemble__"]
        m = e["metrics"]["hybrid"]
        print("-" * 104)
        print(f"{'ensemble (live)':<28}{'-':>8}{'-':>7}"
              f"{m['recall@1']:>8.3f}{m['recall@5']:>8.3f}{m['recall@10']:>8.3f}"
              f"{m['mrr@10']:>9.3f}{m['ndcg@10']:>9.3f}{'-':>11}{'-':>10}"
              f"{e['retrieve_p50_ms']:>7.1f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"n_queries": len(queries), "n_documents": len(docs),
             "embed_model": settings.embed_model, "results": results},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nreport -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
