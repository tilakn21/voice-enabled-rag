"""Calibrate the guardrails against the real index — and check they don't over-block.

This script exists because the first version of the out-of-domain rail was
wrong, and only measurement showed it.

The finding: **"can't answer" is two different problems.**

  off-topic     — the corpus has nothing on this subject. Dense retrieval
                  score separates these almost perfectly (AUC ~0.999).
  unanswerable  — private data, real-time state, or an action request. These
                  are *topically* well covered by a broad web corpus, so
                  retrieval works correctly and returns high-scoring passages.
                  Retrieval score is therefore a weak signal (AUC ~0.89), and
                  no amount of threshold tuning fixes that.

A BM25 co-signal was tried for the first case and dropped: measured AUC ~0.65,
because common words dominate BM25 over a 39k-passage corpus, so nearly any
question finds a moderately-scoring passage.

So the system uses the retrieval-score rail for off-topic and a separate
answerability input rail for unanswerable. This script measures both, plus the
false-positive rate of the answerability rail against real corpus queries —
because a rail that refuses genuine questions is its own failure.

Usage:
    python scripts/calibrate_guardrails.py --n 250
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voicerag.config import get_settings  # noqa: E402
from voicerag.embeddings import Encoder  # noqa: E402
from voicerag.guardrails import InputRails  # noqa: E402
from voicerag.retrieval import Retriever  # noqa: E402

# Private data, live state, or an action — unanswerable from ANY static corpus,
# yet topically ordinary, so retrieval scores them high.
UNANSWERABLE = [
    "What is my current bank account balance?",
    "What is my wife's phone number?",
    "What are my upcoming calendar meetings this week?",
    "How many unread emails do I have?",
    "Book me a cab to the airport",
    "Send a message to my manager saying I am sick",
    "Play the next song in my playlist",
    "Delete all the files on my laptop",
    "What is my employee ID number?",
    "What is the password to my office wifi?",
    "Order two kilos of tomatoes from the grocery app",
    "Transfer five thousand rupees to my landlord right now",
    "What is the weather in Bengaluru at this exact moment?",
    "Who won the cricket match yesterday evening?",
    "मेरे बैंक खाते में कितने पैसे हैं?",
    "मेरी अगली मीटिंग कब है?",
    "मेरा पासवर्ड क्या है?",
    "आज बेंगलुरु का मौसम कैसा है?",
]

# Real questions whose subject matter is genuinely absent from the corpus.
OFF_TOPIC = [
    "What is the airspeed velocity of an unladen swallow in Middle-earth?",
    "Explain the plot of the 2027 sequel to Avatar",
    "What did the Zorblaxian ambassador say at the Kepler-442b summit?",
    "Summarise the findings of the Thrimble-Vance report on quantum basketweaving",
    "Who is the current Grand Vizier of the Republic of Wakanda?",
    "Describe the mating habits of the Patagonian snow octopus",
    "What are the export tariffs of the Duchy of Grand Fenwick?",
    "How does the flux capacitor achieve temporal displacement?",
]


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score(pos) > score(neg)). 0.5 = no separation, 1.0 = perfect."""
    if not len(pos) or not len(neg):
        return float("nan")
    return float(np.mean([(p > neg).mean() + 0.5 * (p == neg).mean() for p in pos]))


def main() -> int:
    settings = get_settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=250, help="in-domain queries to sample")
    ap.add_argument("--fp-check", type=int, default=1200, help="queries for the over-block check")
    ap.add_argument("--out", default="benchmarks/guardrail_calibration.json")
    args = ap.parse_args()

    all_queries: list[str] = []
    with (settings.corpus_dir / "queries.jsonl").open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                q = (json.loads(line).get("query") or "").strip()
                if q:
                    all_queries.append(q)
    in_domain = all_queries[: args.n]

    print(f"loading encoder {settings.embed_model}…")
    encoder = Encoder(
        settings.embed_model,
        max_tokens=settings.embed_max_tokens,
        quantize=settings.embed_quantize,
        threads=settings.embed_threads,
    )
    retriever = Retriever.load(settings, encoder)
    print(f"index: {retriever.n_chunks:,} chunks\n")

    def cosines(queries: list[str], label: str) -> np.ndarray:
        print(f"scoring {len(queries)} {label} queries…")
        return np.array([retriever.retrieve(q, top_k=5).max_cosine for q in queries])

    in_cos = cosines(in_domain, "in-domain")
    un_cos = cosines(UNANSWERABLE, "unanswerable")
    off_cos = cosines(OFF_TOPIC, "off-topic")

    print("\n" + "=" * 78)
    print("SIGNAL: max cosine similarity to the best passage")
    print("=" * 78)
    print(f"{'population':<16}{'n':>5}{'min':>9}{'p05':>9}{'p50':>9}{'p95':>9}{'max':>9}{'AUC':>10}")
    print("-" * 78)
    for name, arr, show_auc in (
        ("in-domain", in_cos, False),
        ("unanswerable", un_cos, True),
        ("off-topic", off_cos, True),
    ):
        a = f"{auc(in_cos, arr):.3f}" if show_auc else "—"
        print(f"{name:<16}{len(arr):>5}{arr.min():>9.4f}{np.percentile(arr,5):>9.4f}"
              f"{np.percentile(arr,50):>9.4f}{np.percentile(arr,95):>9.4f}{arr.max():>9.4f}{a:>10}")
    print("-" * 78)
    print("AUC = P(in-domain scores higher). 1.0 is perfect separation, 0.5 is none.")

    # ---- pick the threshold on off-topic only -------------------------
    print("\n" + "=" * 78)
    print("THRESHOLD SWEEP (topic gate — judged on off-topic, per the note above)")
    print("=" * 78)
    print(f"{'threshold':>11}{'off-topic refused':>20}{'real questions kept':>22}")
    print("-" * 78)

    best = None
    for t in np.arange(0.78, 0.93, 0.005):
        refused_off = float((off_cos < t).mean())
        kept_real = float((in_cos >= t).mean())
        # Want both high; penalise letting an off-topic question through 3x.
        score = 3.0 * (1 - refused_off) + (1 - kept_real)
        if best is None or score < best[0]:
            best = (score, float(t), refused_off, kept_real)
        if abs(t * 1000 % 10) < 1e-6:  # print every 0.01
            print(f"{t:>11.3f}{refused_off:>19.1%}{kept_real:>22.1%}")

    _, thr, refused_off, kept_real = best
    print("-" * 78)
    print(f"{'BEST':>11}{'':>2}{thr:.3f}{refused_off:>17.1%}{kept_real:>22.1%}")

    # ---- does the answerability rail over-block real questions? -------
    print("\n" + "=" * 78)
    print("ANSWERABILITY RAIL (catches what retrieval score provably cannot)")
    print("=" * 78)
    rails = InputRails()

    def blocked_by_answerability(q: str) -> bool:
        outcome = rails.check(q)
        return any(
            r.name == "answerability" and r.verdict.value == "block"
            for r in outcome.report.results
        )

    caught = sum(blocked_by_answerability(q) for q in UNANSWERABLE)
    fp_sample = all_queries[: args.fp_check]
    false_pos = sum(blocked_by_answerability(q) for q in fp_sample)

    print(f"  unanswerable questions caught : {caught}/{len(UNANSWERABLE)} "
          f"({caught/len(UNANSWERABLE):.1%})")
    print(f"  real corpus questions blocked : {false_pos}/{len(fp_sample)} "
          f"({false_pos/max(1,len(fp_sample)):.2%})  <- false-positive rate")

    if false_pos:
        print("\n  examples wrongly blocked:")
        shown = 0
        for q in fp_sample:
            if blocked_by_answerability(q):
                print(f"    - {q[:90]}")
                shown += 1
                if shown >= 5:
                    break

    print("\n" + "=" * 78)
    print("Recommended settings (.env):")
    print(f"  IN_DOMAIN_COSINE_THRESHOLD={thr:.3f}")
    print(f"\nCombined: refuses {refused_off:.0%} of off-topic questions and "
          f"{caught/len(UNANSWERABLE):.0%} of unanswerable ones,")
    print(f"while still answering {kept_real:.0%} of real corpus questions.")

    report = {
        "signal": "max_cosine",
        "populations": {
            name: {
                "n": int(arr.size),
                "min": round(float(arr.min()), 4),
                "p05": round(float(np.percentile(arr, 5)), 4),
                "p50": round(float(np.percentile(arr, 50)), 4),
                "p95": round(float(np.percentile(arr, 95)), 4),
                "max": round(float(arr.max()), 4),
            }
            for name, arr in (("in_domain", in_cos), ("unanswerable", un_cos), ("off_topic", off_cos))
        },
        "auc_in_domain_vs_unanswerable": round(auc(in_cos, un_cos), 4),
        "auc_in_domain_vs_off_topic": round(auc(in_cos, off_cos), 4),
        "recommended_threshold": round(thr, 4),
        "off_topic_refused": round(refused_off, 4),
        "real_questions_kept": round(kept_real, 4),
        "answerability_rail": {
            "unanswerable_caught": caught,
            "unanswerable_total": len(UNANSWERABLE),
            "false_positive_rate": round(false_pos / max(1, len(fp_sample)), 5),
            "false_positive_sample_size": len(fp_sample),
        },
        "current_threshold": settings.in_domain_cosine_threshold,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
