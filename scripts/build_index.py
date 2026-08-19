"""Build one dense + sparse index per chunking strategy.

Each strategy gets its own shard (chunks.jsonl + HNSW graph + BM25 matrix) so
the retriever can fuse across them at query time and the evaluator can score
them independently.

Usage:
    python scripts/build_index.py                       # the live strategies
    python scripts/build_index.py --strategies all      # all 8, for benchmarking
    python scripts/build_index.py --strategies fixed_window,late_chunking
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

from voicerag.chunking import (  # noqa: E402
    ALL_STRATEGIES,
    Document,
    build_chunker,
    chunk_stats,
)
from voicerag.config import get_settings  # noqa: E402
from voicerag.embeddings import Encoder  # noqa: E402
from voicerag.index import BM25Index, DenseIndex, StrategyShard  # noqa: E402


def load_documents(corpus_path: Path) -> list[Document]:
    docs: list[Document] = []
    with corpus_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
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
    return docs


def main() -> int:
    settings = get_settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default=str(settings.corpus_dir / "corpus.jsonl"))
    ap.add_argument("--out", default=str(settings.index_dir))
    ap.add_argument(
        "--strategies",
        default=",".join(settings.active_strategies),
        help=f"comma-separated, or 'all'. Known: {','.join(ALL_STRATEGIES)}",
    )
    ap.add_argument("--batch-size", type=int, default=settings.embed_batch_size)
    ap.add_argument("--no-quantize", action="store_true", help="build with fp32 encoder")
    args = ap.parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"corpus not found at {corpus_path}\nRun scripts/prepare_corpus.py first.", file=sys.stderr)
        return 1

    strategies = ALL_STRATEGIES if args.strategies == "all" else [
        s.strip() for s in args.strategies.split(",") if s.strip()
    ]
    unknown = [s for s in strategies if s not in ALL_STRATEGIES]
    if unknown:
        print(f"unknown strategies: {unknown}\nknown: {ALL_STRATEGIES}", file=sys.stderr)
        return 1

    print(f"loading corpus from {corpus_path}…")
    docs = load_documents(corpus_path)
    print(f"  {len(docs)} passages, {len({d.lang for d in docs})} languages")

    print(f"loading encoder {settings.embed_model}…")
    encoder = Encoder(
        settings.embed_model,
        max_tokens=settings.embed_max_tokens,
        quantize=not args.no_quantize,
        threads=settings.embed_threads,
        query_prefix=settings.embed_query_prefix,
        passage_prefix=settings.embed_passage_prefix,
    )
    # Use the real tokenizer for chunk sizing so token targets mean something
    # consistent across scripts.
    def token_len(text: str) -> int:
        return len(encoder.tokenizer.encode(text, add_special_tokens=False))

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "corpus": str(corpus_path),
        "n_documents": len(docs),
        "embed_model": settings.embed_model,
        "embed_dim": encoder.dim,
        "strategies": {},
    }

    for strategy in strategies:
        print(f"\n=== {strategy} ===")
        t0 = time.perf_counter()
        chunker = build_chunker(strategy, encoder=encoder, token_len=token_len)
        chunks = chunker.chunk(docs)
        t_chunk = time.perf_counter() - t0
        if not chunks:
            print("  produced no chunks; skipping")
            continue

        stats = chunk_stats(chunks)
        print(f"  chunked in {t_chunk:.1f}s -> {stats}")

        t1 = time.perf_counter()
        vectors = chunker.embed_override(chunks, encoder)
        if vectors is None:
            texts = [c.text for c in chunks]
            vectors = encoder.encode_passages(texts, batch_size=args.batch_size)
        else:
            print("  using chunker-supplied vectors (late chunking)")
        t_embed = time.perf_counter() - t1
        print(f"  embedded {len(chunks)} chunks in {t_embed:.1f}s "
              f"({len(chunks) / max(t_embed, 1e-6):.0f}/s)")

        t2 = time.perf_counter()
        dense = DenseIndex(
            dim=int(vectors.shape[1]),
            ef_search=settings.hnsw_ef_search,
            m=settings.hnsw_m,
            ef_construction=settings.hnsw_ef_construction,
        )
        dense.build(np.asarray(vectors, dtype=np.float32))

        sparse = BM25Index()
        # BM25 indexes the payload (context text where it differs), because
        # lexical match against the wider window is what catches rare terms
        # that the sentence itself paraphrases away.
        sparse.build([c.payload for c in chunks])
        t_index = time.perf_counter() - t2

        shard = StrategyShard(strategy=strategy, chunks=chunks, dense=dense, sparse=sparse)
        shard.save(out_root)
        print(f"  indexed in {t_index:.1f}s (dense={dense.backend}) -> {out_root / strategy}")

        manifest["strategies"][strategy] = {
            **stats,
            "chunk_seconds": round(t_chunk, 2),
            "embed_seconds": round(t_embed, 2),
            "index_seconds": round(t_index, 2),
            "dense_backend": dense.backend,
            "bm25_terms": len(sparse.vocab),
        }

    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nmanifest -> {out_root / 'manifest.json'}")
    print("\nchunking summary:")
    print(f"  {'strategy':<28} {'chunks':>9} {'per doc':>8} {'p50 tok':>8} {'p95 tok':>8}")
    for name, s in manifest["strategies"].items():
        print(f"  {name:<28} {s['n_chunks']:>9} {s['chunks_per_doc']:>8} "
              f"{s['tokens_p50']:>8} {s['tokens_p95']:>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
