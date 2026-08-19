"""Build the Chunking Lab index — all eight strategies over a small subset.

The live service ships only the two strategies the benchmark justified. The lab
needs all eight so the UI can race them against each other, but building eight
full-corpus indices costs ~1.5 hours and ~1.5 GB. So the lab runs over the same
300-query / ~6k-passage subset that `scripts/bench_retrieval.py` scores, which
keeps it to ~12 minutes and ~250 MB — and means the quality metrics shown next
to each strategy in the UI describe exactly the corpus being queried.

Usage:
    python scripts/build_lab_index.py                 # 300 queries, all 8
    python scripts/build_lab_index.py --queries 150   # smaller / faster
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voicerag.config import CORPUS_DIR  # noqa: E402

LAB_CORPUS = ROOT / "data" / "corpus_lab"
LAB_INDEX = ROOT / "data" / "index_lab"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", type=int, default=300, help="queries to carve out")
    ap.add_argument("--strategies", default="all")
    args = ap.parse_args()

    corpus_src = CORPUS_DIR / "corpus.jsonl"
    queries_src = CORPUS_DIR / "queries.jsonl"
    if not corpus_src.exists():
        print(f"missing {corpus_src}; run scripts/prepare_corpus.py first", file=sys.stderr)
        return 1

    LAB_CORPUS.mkdir(parents=True, exist_ok=True)

    queries = []
    with queries_src.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                queries.append(json.loads(line))
            if len(queries) >= args.queries:
                break
    query_ids = {q["query_id"] for q in queries}

    n_docs = 0
    with (LAB_CORPUS / "corpus.jsonl").open("w", encoding="utf-8") as out, \
         corpus_src.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip() and json.loads(line).get("query_id") in query_ids:
                out.write(line)
                n_docs += 1

    with (LAB_CORPUS / "queries.jsonl").open("w", encoding="utf-8") as out:
        for q in queries:
            out.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"lab corpus: {n_docs} passages from {len(queries)} queries")
    print(f"building {args.strategies} strategies -> {LAB_INDEX}\n")

    return subprocess.call([
        sys.executable, str(ROOT / "scripts" / "build_index.py"),
        "--corpus", str(LAB_CORPUS / "corpus.jsonl"),
        "--out", str(LAB_INDEX),
        "--strategies", args.strategies,
    ])


if __name__ == "__main__":
    raise SystemExit(main())
