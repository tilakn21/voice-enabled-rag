"""Build a working corpus + evaluation set from MSMARCO-XI.

The full dataset is 55.6 GB across 27 parquet files (one per language per
split). We pull the validation shard for the requested languages — ~460 MB
each, versus ~3.7 GB for a train shard — and slice the first N queries.

Each row gives us ten passages and an `is_selected` vector marking which ones
actually answer the query. That's a labelled relevance judgement per query,
which is what makes `bench_retrieval.py` a real measurement rather than a vibe
check.

Because every row carries both `Translated_passages` and `English_passages`,
one file yields a genuinely cross-lingual corpus: Hindi questions against a
pool containing both Hindi and English passages.

Usage:
    python scripts/prepare_corpus.py --languages hin --max-queries 1200
    python scripts/prepare_corpus.py --languages hin,tam --max-queries 800 --no-english
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from voicerag.config import CORPUS_DIR  # noqa: E402

REPO = "ai4bharat/MSMARCO-XI"

# The dataset exposes one `default` config; the real partitioning is by
# filename, e.g. validation/hinval.parquet, train/hintrain.parquet.
LANGUAGES = {
    "asm": "Assamese", "ben": "Bengali", "guj": "Gujarati", "hin": "Hindi",
    "kan": "Kannada", "mal": "Malayalam", "mar": "Marathi", "nep": "Nepali",
    "ori": "Odia", "pan": "Punjabi", "san": "Sanskrit", "tam": "Tamil",
    "tel": "Telugu", "urd": "Urdu",
}
# tel exists only in validation.
TRAIN_LANGUAGES = set(LANGUAGES) - {"tel"}

COLUMNS = ["query", "query_id", "target_lang", "passages", "Eng_Query", "Answer", "query_type"]


def shard_filename(lang: str, split: str) -> str:
    suffix = "val" if split == "validation" else "train"
    return f"{split}/{lang}{suffix}.parquet"


def text_key(text: str) -> str:
    return hashlib.blake2b(text.strip().encode("utf-8"), digest_size=12).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--languages", default="hin", help="comma-separated 3-letter codes, e.g. hin,tam")
    ap.add_argument("--split", default="validation", choices=["validation", "train"])
    ap.add_argument("--max-queries", type=int, default=1200, help="queries per language")
    ap.add_argument("--no-english", action="store_true", help="skip the parallel English passages")
    ap.add_argument("--out", default=str(CORPUS_DIR))
    ap.add_argument("--batch-size", type=int, default=512)
    args = ap.parse_args()

    langs = [l.strip() for l in args.languages.split(",") if l.strip()]
    for lang in langs:
        if lang not in LANGUAGES:
            ap.error(f"unknown language {lang!r}; choose from {sorted(LANGUAGES)}")
        if args.split == "train" and lang not in TRAIN_LANGUAGES:
            ap.error(f"{lang!r} has no train shard (validation only)")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = out_dir / "corpus.jsonl"
    queries_path = out_dir / "queries.jsonl"

    seen_text: dict[str, str] = {}  # text hash -> canonical doc_id
    n_docs = 0
    n_queries = 0
    per_lang: dict[str, dict[str, int]] = {}

    with corpus_path.open("w", encoding="utf-8") as corpus_fh, \
         queries_path.open("w", encoding="utf-8") as queries_fh:

        for lang in langs:
            filename = shard_filename(lang, args.split)
            print(f"[{lang}] downloading {filename} (cached after first run)…", flush=True)
            local = hf_hub_download(REPO, filename=filename, repo_type="dataset")

            pf = pq.ParquetFile(local)
            taken = 0
            lang_docs = 0

            for batch in pf.iter_batches(batch_size=args.batch_size, columns=COLUMNS):
                if taken >= args.max_queries:
                    break
                for row in batch.to_pylist():
                    if taken >= args.max_queries:
                        break

                    passages = row.get("passages") or {}
                    translated = passages.get("Translated_passages") or []
                    english = passages.get("English_passages") or []
                    selected = passages.get("is_selected") or []
                    query = (row.get("query") or "").strip()
                    if not query or not translated:
                        continue

                    target_lang = row.get("target_lang") or lang
                    query_id = int(row.get("query_id") or 0)
                    gold: list[str] = []

                    def emit(text: str, doc_lang: str, idx: int, is_gold: bool) -> None:
                        nonlocal n_docs, lang_docs
                        text = (text or "").strip()
                        if len(text) < 20:
                            return
                        key = text_key(text)
                        existing = seen_text.get(key)
                        doc_id = existing or f"{query_id}-{doc_lang}-{idx}"
                        if existing is None:
                            seen_text[key] = doc_id
                            corpus_fh.write(
                                json.dumps(
                                    {
                                        "doc_id": doc_id,
                                        "text": text,
                                        "lang": doc_lang,
                                        "query_id": query_id,
                                        "passage_idx": idx,
                                        "is_selected": is_gold,
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                            n_docs += 1
                            lang_docs += 1
                        if is_gold and doc_id not in gold:
                            gold.append(doc_id)

                    for idx, passage in enumerate(translated):
                        is_gold = bool(selected[idx]) if idx < len(selected) else False
                        emit(passage, target_lang, idx, is_gold)

                    if not args.no_english:
                        for idx, passage in enumerate(english):
                            is_gold = bool(selected[idx]) if idx < len(selected) else False
                            emit(passage, "eng_Latn", idx, is_gold)

                    if not gold:
                        # No labelled positive -> useless for retrieval eval,
                        # though its passages still enrich the corpus as
                        # distractors, which is why we emit before this check.
                        continue

                    queries_fh.write(
                        json.dumps(
                            {
                                "query_id": query_id,
                                "query": query,
                                "eng_query": (row.get("Eng_Query") or "").strip(),
                                "lang": target_lang,
                                "query_type": row.get("query_type"),
                                "reference_answer": (row.get("Answer") or "").strip(),
                                "gold_doc_ids": gold,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    taken += 1
                    n_queries += 1

            per_lang[lang] = {"queries": taken, "new_docs": lang_docs}
            print(f"[{lang}] {taken} queries, {lang_docs} new passages", flush=True)

    manifest = {
        "repo": REPO,
        "split": args.split,
        "languages": langs,
        "include_english": not args.no_english,
        "max_queries_per_language": args.max_queries,
        "n_documents": n_docs,
        "n_queries": n_queries,
        "per_language": per_lang,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\ncorpus  -> {corpus_path}  ({n_docs} passages)")
    print(f"queries -> {queries_path} ({n_queries} labelled queries)")
    print(f"manifest-> {out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
