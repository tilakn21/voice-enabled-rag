"""Guardrails.

Two banks of checks with different jobs:

Input rails run before retrieval and decide whether the question should be
answered at all — empty or garbled ASR output, prompt injection aimed at the
synthesis step, unsafe requests, and oversized inputs.

Output rails run after generation and decide whether the answer earned the
right to be shown — is every claim traceable to retrieved text, does every
citation point at a chunk we actually retrieved, and is the retrieval score
high enough that we're answering rather than pattern-matching noise.

The out-of-domain rail is the one that matters most for a demo: MSMARCO-XI is a
fixed corpus, so "who is the president of France" and "what is my bank balance"
both need to produce an honest refusal, not a fluent guess.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass

from .chunking import content_tokens as _content_tokens
from .schemas import GuardrailReport, RailVerdict, RetrievedChunk

_WORD = re.compile(r"[\wऀ-෿؀-ۿ]+", re.UNICODE)

# --------------------------------------------------------------------------
# Prompt injection. These target the LLM synthesis stage, which sees retrieved
# text and the user question in the same context.
# --------------------------------------------------------------------------
_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bignore\s+(all\s+|any\s+)?(previous|prior|above|earlier)\s+(instruction|prompt|rule|direction)",
        r"\bdisregard\s+(all\s+|the\s+)?(previous|prior|above|system)",
        r"\b(reveal|show|print|repeat|output|leak)\s+(me\s+)?(your|the)\s+(system\s+)?(prompt|instruction|rule)",
        r"\byou\s+are\s+now\s+(a|an|no longer)\b",
        r"\bact\s+as\s+(if\s+you\s+are\s+)?(a\s+|an\s+)?(dan|jailbroken|unrestricted|developer\s+mode)",
        r"\bforget\s+(everything|all|your\s+(instruction|rule|training))",
        r"<\s*/?\s*(system|assistant|human)\s*>",
        r"\[\s*(system|inst|/inst)\s*\]",
        r"\bnew\s+(system\s+)?(prompt|instruction)s?\s*[:=]",
        r"\boverride\s+(your\s+)?(safety|guardrail|restriction)",
    )
]

# --------------------------------------------------------------------------
# Unsafe-request heuristics. Deliberately narrow: this is a retrieval QA system
# over a public web-passage corpus, so the goal is refusing operational harm
# requests, not policing topics. Over-blocking a factual question is a real
# failure mode too, hence the requirement for an intent verb near the term.
# --------------------------------------------------------------------------
_HARM_INTENT = re.compile(
    r"\b(how\s+(to|do\s+i|can\s+i)|steps?\s+to|guide\s+to|instructions?\s+for|"
    r"teach\s+me|help\s+me|best\s+way\s+to|कैसे\s+बनाय|कैसे\s+बनाऊं)\b",
    re.IGNORECASE,
)
_HARM_OBJECT = re.compile(
    r"\b(bomb|explosive|ied|nerve\s+agent|sarin|ricin|anthrax|bioweapon|"
    r"meth(amphetamine)?|fentanyl|untraceable\s+(gun|poison)|ghost\s+gun|"
    r"child\s+(porn|sexual)|csam|"
    r"kill\s+(myself|yourself)|commit\s+suicide|end\s+my\s+life|"
    r"hack\s+into|ddos|ransomware|keylogger|steal\s+(credit\s+card|password|identity))\b",
    re.IGNORECASE,
)
_SELF_HARM = re.compile(
    r"\b(kill\s+myself|end\s+my\s+life|commit\s+suicide|want\s+to\s+die|self[-\s]?harm)\b",
    re.IGNORECASE,
)

# --------------------------------------------------------------------------
# Answerability.
#
# Measured finding (scripts/calibrate_guardrails.py): the retrieval-score rail
# separates genuinely off-topic questions almost perfectly (AUC 0.999) but is
# much weaker against *unanswerable* ones (AUC 0.89). The reason is structural,
# not a tuning problem — "what is my bank balance" is topically well covered by
# a broad web corpus, so it retrieves high-scoring passages about bank
# balances. No retrieval score can reject it, because retrieval is working
# correctly; the question is simply not answerable from any static corpus.
#
# So that class is caught here instead, as a property of the question: a
# first-person possessive paired with a personal-data noun, an explicit
# real-time reference, or an imperative action request.
# --------------------------------------------------------------------------
# NOTE ON \b AND INDIC SCRIPTS
# Devanagari vowel signs (े ी ा ु) are Unicode category Mn (combining marks),
# and Python's `\w` excludes them. So a word like "मेरे" ends in a NON-word
# character, and `\bमेरे\b` never matches — the trailing `\b` requires a word
# character on its left. Every Devanagari pattern here was silently failing
# until this was found (the rail caught 0/4 Hindi unanswerable questions).
# Devanagari is space-separated and these tokens are distinctive, so the
# boundary assertions are simply dropped for the Indic alternatives.
_PERSONAL_DATA = re.compile(
    r"\b(my|mine|our)\b[^?.!]{0,30}\b(account|balance|password|passcode|pin|email|inbox|"
    r"calendar|meeting|schedule|appointment|phone\s+number|contact|salary|payslip|"
    r"subscription|order|delivery|booking|ticket|playlist|files?|laptop|battery|"
    r"blood\s+test|prescription|employee\s+id|address|location|wifi)\b"
    # "how many unread emails do I have" — first person without a possessive
    r"|\bdo\s+i\s+have\b"
    r"|\bam\s+i\s+(scheduled|booked|registered)\b"
    r"|(मेरा|मेरी|मेरे|मुझे)[^?।!]{0,30}"
    r"(खाते|खाता|बैलेंस|पासवर्ड|मीटिंग|ईमेल|नंबर|पता|बैटरी|फोन|फ़ोन)",
    re.IGNORECASE | re.UNICODE,
)
_REALTIME = re.compile(
    r"\b(right\s+now|at\s+this\s+(exact\s+)?moment|as\s+of\s+(today|now)|"
    r"currently\s+(is|are)|yesterday|tomorrow|this\s+(week|morning|evening)|"
    r"latest\s+score|live\s+score)\b"
    # boundary assertions dropped for Devanagari — see the note above
    r"|(अभी|आज|कल)[^?।!]{0,20}(मौसम|स्कोर|मैच|कीमत)",
    re.IGNORECASE | re.UNICODE,
)
_ACTION_REQUEST = re.compile(
    r"^\s*(please\s+)?(send|book|order|buy|transfer|pay|delete|remove|play|call|"
    r"email|text|message|schedule|cancel|remind|set\s+a?\s*(reminder|alarm)|"
    r"turn\s+(on|off)|open|install|download)\b",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------
# PII — redacted before anything reaches a log line or a telemetry record.
# --------------------------------------------------------------------------
_PII_PATTERNS = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b")),
    ("PHONE", re.compile(r"\b(?:\+?\d{1,3}[\s-]?)?(?:\d[\s-]?){9,13}\d\b")),
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("AADHAAR", re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")),
    ("PAN", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")),
]

# Stop list lives in chunking.py so the rails and the generator agree on
# what counts as a content word.


def redact_pii(text: str) -> str:
    out = text
    for label, pattern in _PII_PATTERNS:
        out = pattern.sub(f"[{label}]", out)
    return out


def _char_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts: dict[str, int] = {}
    for ch in text:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


# --------------------------------------------------------------------------
@dataclass
class InputRailOutcome:
    report: GuardrailReport
    cleaned_query: str
    safe_to_answer: bool
    refusal_text: str | None = None


class InputRails:
    def __init__(self, min_chars: int = 3, max_chars: int = 800):
        self.min_chars = min_chars
        self.max_chars = max_chars

    def check(self, text: str) -> InputRailOutcome:
        report = GuardrailReport()
        cleaned = unicodedata.normalize("NFKC", (text or "")).strip()

        # ---- 1. transcript sanity -------------------------------------
        if len(cleaned) < self.min_chars:
            report.add(
                "transcript_sanity",
                RailVerdict.BLOCK,
                "Transcript is empty or too short to be a question.",
            )
            return InputRailOutcome(
                report, cleaned, False,
                "I didn't catch a question there — could you try again?",
            )

        if len(cleaned) > self.max_chars:
            cleaned = cleaned[: self.max_chars]
            report.add("length_cap", RailVerdict.WARN, f"Truncated to {self.max_chars} chars.")
        else:
            report.add("length_cap", RailVerdict.PASS)

        # ---- 2. gibberish / ASR failure -------------------------------
        tokens = _WORD.findall(cleaned)
        entropy = _char_entropy(cleaned)
        unique_ratio = len(set(t.lower() for t in tokens)) / max(1, len(tokens))
        if not tokens or (entropy < 2.0 and len(cleaned) > 12) or (len(tokens) > 6 and unique_ratio < 0.25):
            report.add(
                "asr_quality",
                RailVerdict.BLOCK,
                f"Low-information transcript (entropy={entropy:.2f}, unique={unique_ratio:.2f}).",
                score=entropy,
            )
            return InputRailOutcome(
                report, cleaned, False,
                "That came through garbled. Could you say it again?",
            )
        report.add("asr_quality", RailVerdict.PASS, score=round(entropy, 2))

        # ---- 3. prompt injection --------------------------------------
        for pattern in _INJECTION_PATTERNS:
            if pattern.search(cleaned):
                report.add(
                    "prompt_injection",
                    RailVerdict.BLOCK,
                    f"Matched injection pattern: {pattern.pattern[:52]}",
                )
                return InputRailOutcome(
                    report, cleaned, False,
                    "I can only answer questions about the indexed passages.",
                )
        report.add("prompt_injection", RailVerdict.PASS)

        # ---- 4. answerability -----------------------------------------
        # Retrieval scores cannot catch these (see the note above), so they are
        # rejected on the shape of the question instead.
        for rail_name, pattern, reason in (
            ("answerability", _PERSONAL_DATA, "asks about private user data"),
            ("answerability", _REALTIME, "asks for real-time information"),
            ("answerability", _ACTION_REQUEST, "is an action request, not a question"),
        ):
            if pattern.search(cleaned):
                report.add("answerability", RailVerdict.BLOCK, f"Question {reason}.")
                return InputRailOutcome(
                    report, cleaned, False,
                    "I can only answer questions from the indexed passages — I don't "
                    "have access to personal data, live information, or your accounts.",
                )
        report.add("answerability", RailVerdict.PASS)

        # ---- 5. unsafe intent -----------------------------------------
        if _SELF_HARM.search(cleaned):
            report.add("safety", RailVerdict.BLOCK, "Self-harm intent detected.")
            return InputRailOutcome(
                report, cleaned, False,
                "I can't help with that. If you're in distress, please reach out to a "
                "local crisis line — in India, iCall is on 9152987821 and Tele-MANAS on 14416.",
            )
        if _HARM_OBJECT.search(cleaned) and _HARM_INTENT.search(cleaned):
            report.add("safety", RailVerdict.BLOCK, "Operational harm request.")
            return InputRailOutcome(
                report, cleaned, False,
                "I can't help with that request.",
            )
        report.add("safety", RailVerdict.PASS)

        return InputRailOutcome(report, cleaned, True)


# --------------------------------------------------------------------------
@dataclass
class OutputRailOutcome:
    report: GuardrailReport
    groundedness: float
    confidence: float
    answered: bool
    abstain_reason: str | None = None


class OutputRails:
    def __init__(
        self,
        groundedness_threshold: float = 0.60,
        in_domain_cosine_threshold: float = 0.845,
    ):
        self.groundedness_threshold = groundedness_threshold
        self.in_domain_cosine_threshold = in_domain_cosine_threshold

    # ------------------------------------------------------------------
    def check_retrieval(
        self,
        report: GuardrailReport,
        max_fused: float,
        max_cosine: float,
        n_results: int,
        max_sparse: float = 0.0,
    ) -> tuple[bool, str | None]:
        """Out-of-domain gate. Runs *before* generation, so an off-corpus
        question never costs an LLM call.

        This is the *topic* gate, and dense cosine is the signal for it: measured
        AUC 0.999 separating in-corpus questions from genuinely off-topic ones.

        A BM25 co-signal was tried here and removed — measured AUC 0.65-0.68,
        i.e. close to useless. Common words dominate BM25 over a 39k-passage
        corpus, so almost any question finds a moderately-scoring passage. It
        is kept in `max_sparse` for telemetry, but nothing is gated on it.

        Questions that are unanswerable rather than off-topic (personal data,
        real-time, action requests) are handled by the answerability input
        rail, because retrieval scores provably cannot separate them.
        """
        if n_results == 0:
            report.add("retrieval_coverage", RailVerdict.BLOCK, "No candidates retrieved.")
            return False, "no_results"

        if max_cosine < self.in_domain_cosine_threshold:
            report.add(
                "out_of_domain",
                RailVerdict.BLOCK,
                f"Best passage similarity {max_cosine:.3f} < {self.in_domain_cosine_threshold:.3f}; "
                "no passage in the corpus is close enough to this question.",
                score=round(max_cosine, 4),
            )
            return False, "out_of_domain"

        report.add(
            "out_of_domain",
            RailVerdict.PASS,
            f"cosine={max_cosine:.3f}, bm25={max_sparse:.2f}",
            score=round(max_cosine, 4),
        )

        return True, None

    # ------------------------------------------------------------------
    def groundedness(self, answer: str, cited: list[RetrievedChunk]) -> float:
        """Fraction of the answer's content tokens (and bigrams) that appear in
        the cited context. Cheap, deterministic, and no second model call —
        which is what keeps it inside the latency budget."""
        answer_tokens = _content_tokens(answer)
        if not answer_tokens:
            return 0.0

        context_tokens = _content_tokens(" ".join(c.chunk.payload for c in cited))
        if not context_tokens:
            return 0.0
        context_unigrams = set(context_tokens)
        context_bigrams = set(zip(context_tokens, context_tokens[1:]))

        unigram = sum(1 for t in answer_tokens if t in context_unigrams) / len(answer_tokens)

        answer_bigrams = list(zip(answer_tokens, answer_tokens[1:]))
        if answer_bigrams and context_bigrams:
            bigram = sum(1 for b in answer_bigrams if b in context_bigrams) / len(answer_bigrams)
        else:
            # Single-token answer: no bigram evidence either way, so don't let
            # a zero drag the score down.
            bigram = unigram

        # Bigrams weighted lower: they're a stricter signal but noisier on
        # short answers and across morphologically rich Indic scripts.
        return round(0.65 * unigram + 0.35 * bigram, 4)

    # ------------------------------------------------------------------
    def check_answer(
        self,
        report: GuardrailReport,
        answer: str,
        cited: list[RetrievedChunk],
        retrieved_ids: set[str],
        max_fused: float,
    ) -> OutputRailOutcome:
        # ---- citation validity ----------------------------------------
        bad = [c.chunk.chunk_id for c in cited if c.chunk.chunk_id not in retrieved_ids]
        if bad:
            report.add(
                "citation_validity",
                RailVerdict.BLOCK,
                f"{len(bad)} citation(s) reference chunks that were not retrieved.",
            )
            return OutputRailOutcome(report, 0.0, 0.0, False, "invalid_citation")
        if not cited:
            report.add("citation_validity", RailVerdict.BLOCK, "Answer has no citations.")
            return OutputRailOutcome(report, 0.0, 0.0, False, "uncited")
        report.add("citation_validity", RailVerdict.PASS, f"{len(cited)} citation(s) verified.")

        # ---- groundedness ---------------------------------------------
        score = self.groundedness(answer, cited)
        if score < self.groundedness_threshold:
            report.add(
                "groundedness",
                RailVerdict.BLOCK,
                f"Only {score:.0%} of answer content is supported by cited passages "
                f"(need {self.groundedness_threshold:.0%}).",
                score=score,
            )
            return OutputRailOutcome(report, score, 0.0, False, "ungrounded")
        report.add("groundedness", RailVerdict.PASS, score=score)

        # ---- output safety --------------------------------------------
        if _HARM_OBJECT.search(answer) and _HARM_INTENT.search(answer):
            report.add("output_safety", RailVerdict.BLOCK, "Unsafe content in generated answer.")
            return OutputRailOutcome(report, score, 0.0, False, "unsafe_output")
        report.add("output_safety", RailVerdict.PASS)

        confidence = round(min(1.0, 0.5 * score + 0.5 * min(1.0, max_fused)), 4)
        return OutputRailOutcome(report, score, confidence, True, None)
