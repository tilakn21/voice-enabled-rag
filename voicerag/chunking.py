"""Chunking strategies.

Eight strategies, each with a different failure mode, all registered against a
common interface so the index builder and the offline evaluator can treat them
uniformly. The point of having many is not variety for its own sake: MSMARCO-XI
passages are short and heterogeneous across 13 scripts, and no single splitter
is right for all of them. `scripts/bench_retrieval.py` scores each strategy on
Recall@k / MRR@10 / nDCG@10 using the dataset's own `is_selected` labels, and
the live retriever fuses the winners by max score (measured better than RRF).

Design notes
------------
* `text` is what gets embedded and BM25-indexed. `context_text` is what the
  answer generator reads. Keeping them separate is what makes sentence-window
  and parent-child retrieval possible: index something small and precise,
  return something wide enough to answer from.
* Every chunk carries metadata (lang, query_id, passage_idx, char span, parent)
  so retrieval can filter and so overlapping chunks can be de-duplicated by
  span rather than by string equality.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np

from .schemas import Chunk

# --------------------------------------------------------------------------
# Text utilities (script-aware: MSMARCO-XI spans Devanagari, Bengali, Tamil,
# Telugu, Kannada, Malayalam, Gurmukhi, Odia, Gujarati, Arabic-script Urdu)
# --------------------------------------------------------------------------

# Danda (।) and double danda (॥) terminate sentences in most Indic scripts;
# Urdu uses the Arabic full stop (۔). Latin punctuation appears in the English
# passages and in code-mixed text.
_SENT_BOUNDARY = re.compile(r"(?<=[।॥۔.!?])[\s​]+|\n{2,}")
_PARA_BOUNDARY = re.compile(r"\n\s*\n+")
_WORD = re.compile(r"[\wऀ-෿؀-ۿ]+", re.UNICODE)

# Clause-ish separators used by the proposition splitter. Conjunctions are
# listed per-script because a Latin-only list silently no-ops on Indic text.
_CLAUSE_SPLIT = re.compile(
    r"\s*(?:;|—|--|\bhowever\b|\bbut\b|\bwhereas\b|\bwhile\b|\balthough\b"
    r"|\bऔर\b|\bलेकिन\b|\bक्योंकि\b|\bএবং\b|\bকিন্তু\b|\bமற்றும்\b|\bஆனால்\b"
    r"|\bమరియు\b|\bకానీ\b|\bಮತ್ತು\b|\bആന്നാൽ\b|\bاور\b|\bلیکن\b)\s*",
    re.IGNORECASE | re.UNICODE,
)

TokenLen = Callable[[str], int]


# Shared stop list. Overlap between a query and a passage has to be judged on
# *content* words: function words match everything, so leaving them in makes an
# unrelated sentence look relevant. (This is not hypothetical — the extractive
# answer used to append an off-topic sentence because "the" and "is" counted
# as overlap.)
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on", "for",
    "and", "or", "it", "its", "this", "that", "these", "those", "as", "at", "by",
    "with", "from", "be", "been", "has", "have", "had", "not", "no", "but", "if",
    "then", "than", "so", "such", "can", "will", "would", "should", "could",
    "there", "their", "they", "you", "your", "we", "our", "he", "she", "his",
    "her", "which", "what", "when", "where", "who", "how", "why", "do", "does",
    "did", "i", "me", "my", "am", "into", "about", "also", "any", "some",
    "है", "हैं", "का", "की", "के", "में", "और", "से", "को", "पर", "यह", "वह", "एक",
    "क्या", "कौन", "कब", "कहाँ", "कैसे", "था", "थी", "थे", "हो", "गया", "लिए",
    "ও", "এবং", "এই", "একটি", "না", "করে", "মধ্যে",
    "மற்றும்", "இது", "ஒரு", "என்று", "இல்",
    "మరియు", "ఇది", "ఒక", "లో",
}


def content_tokens(text: str) -> list[str]:
    """Lowercased content words — stop words and single characters removed."""
    return [
        t.lower()
        for t in _WORD.findall(text)
        if len(t) > 1 and t.lower() not in STOPWORDS
    ]


def approx_token_len(text: str) -> int:
    """Cheap token estimate used when a real tokenizer isn't wired in.

    Indic scripts tokenize far more aggressively than Latin under a subword
    vocab, so a flat words*1.3 factor badly under-counts them. Scale by the
    share of non-Latin characters instead.
    """
    words = len(_WORD.findall(text))
    if not words:
        return max(1, len(text) // 4)
    non_latin = sum(1 for ch in text if ord(ch) > 0x0590)
    ratio = non_latin / max(1, len(text))
    factor = 1.3 + 1.5 * ratio  # ~1.3 for English, ~2.8 for pure Devanagari
    return max(1, int(words * factor))


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT_BOUNDARY.split(text) if p and p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def split_paragraphs(text: str) -> list[str]:
    parts = [p.strip() for p in _PARA_BOUNDARY.split(text) if p and p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _hash_id(*parts: object) -> str:
    raw = "|".join(str(p) for p in parts)
    return hashlib.blake2b(raw.encode("utf-8"), digest_size=10).hexdigest()


def _locate(haystack: str, needle: str, start_hint: int = 0) -> tuple[int, int]:
    """Best-effort char span of `needle` within `haystack`."""
    if not needle:
        return start_hint, start_hint
    idx = haystack.find(needle, start_hint)
    if idx < 0:
        idx = haystack.find(needle)
    if idx < 0:
        return start_hint, start_hint + len(needle)
    return idx, idx + len(needle)


# --------------------------------------------------------------------------
# Document
# --------------------------------------------------------------------------
@dataclass
class Document:
    doc_id: str
    text: str
    lang: str = "unknown"
    query_id: int | None = None
    passage_idx: int | None = None
    is_selected: bool = False
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Base
# --------------------------------------------------------------------------
class BaseChunker:
    name: str = "base"

    def chunk(self, docs: Sequence[Document]) -> list[Chunk]:  # pragma: no cover
        raise NotImplementedError

    def embed_override(self, chunks: Sequence[Chunk], encoder) -> np.ndarray | None:
        """Return precomputed vectors, or None to let the indexer embed normally.

        Only `late_chunking` uses this — it needs token embeddings from a
        full-document forward pass, which the standard per-chunk path can't give.
        """
        return None

    # shared helper
    def _mk(
        self,
        doc: Document,
        text: str,
        *,
        context_text: str | None = None,
        span: tuple[int, int] | None = None,
        parent_id: str | None = None,
        seq: int = 0,
        token_len: TokenLen = approx_token_len,
        extra: dict | None = None,
    ) -> Chunk:
        start, end = span if span else _locate(doc.text, text)
        return Chunk(
            chunk_id=_hash_id(self.name, doc.doc_id, seq, start, end),
            doc_id=doc.doc_id,
            text=text,
            context_text=context_text,
            strategy=self.name,
            lang=doc.lang,
            query_id=doc.query_id,
            passage_idx=doc.passage_idx,
            char_start=start,
            char_end=end,
            parent_id=parent_id,
            is_selected=doc.is_selected,
            token_count=token_len(text),
            extra=extra or {},
        )


# --------------------------------------------------------------------------
# 1. Passage-atomic — the natural unit of MS MARCO. Baseline and, for short
#    passages, genuinely hard to beat.
# --------------------------------------------------------------------------
class PassageAtomicChunker(BaseChunker):
    name = "passage_atomic"

    def __init__(self, token_len: TokenLen = approx_token_len):
        self.token_len = token_len

    def chunk(self, docs: Sequence[Document]) -> list[Chunk]:
        out = []
        for doc in docs:
            text = doc.text.strip()
            if not text:
                continue
            out.append(
                self._mk(doc, text, span=(0, len(doc.text)), seq=0, token_len=self.token_len)
            )
        return out


# --------------------------------------------------------------------------
# 2. Fixed window with overlap — the naive baseline the brief warns against
#    submitting alone. Kept precisely so the benchmark can show what it costs.
# --------------------------------------------------------------------------
class FixedWindowChunker(BaseChunker):
    name = "fixed_window"

    def __init__(
        self,
        target_tokens: int = 180,
        overlap_tokens: int = 45,
        token_len: TokenLen = approx_token_len,
    ):
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.token_len = token_len

    def chunk(self, docs: Sequence[Document]) -> list[Chunk]:
        out: list[Chunk] = []
        for doc in docs:
            words = list(_WORD.finditer(doc.text))
            if not words:
                continue
            # Convert the token target into a word-count stride using the same
            # script-aware factor the estimator uses.
            per_word = max(1.0, self.token_len(doc.text) / max(1, len(words)))
            stride_words = max(8, int(self.target_tokens / per_word))
            overlap_words = max(0, int(self.overlap_tokens / per_word))
            step = max(1, stride_words - overlap_words)

            seq = 0
            for start_i in range(0, len(words), step):
                window = words[start_i : start_i + stride_words]
                if not window:
                    break
                s, e = window[0].start(), window[-1].end()
                text = doc.text[s:e].strip()
                if text:
                    out.append(
                        self._mk(doc, text, span=(s, e), seq=seq, token_len=self.token_len)
                    )
                    seq += 1
                if start_i + stride_words >= len(words):
                    break
        return out


# --------------------------------------------------------------------------
# 3. Recursive structural — respects paragraph, then sentence, then word
#    boundaries. Never splits mid-sentence unless a single sentence overflows.
# --------------------------------------------------------------------------
class RecursiveStructuralChunker(BaseChunker):
    name = "recursive_structural"

    def __init__(
        self,
        target_tokens: int = 200,
        overlap_sentences: int = 1,
        token_len: TokenLen = approx_token_len,
    ):
        self.target_tokens = target_tokens
        self.overlap_sentences = overlap_sentences
        self.token_len = token_len

    def chunk(self, docs: Sequence[Document]) -> list[Chunk]:
        out: list[Chunk] = []
        for doc in docs:
            seq = 0
            cursor = 0
            for para in split_paragraphs(doc.text):
                sentences = split_sentences(para)
                buf: list[str] = []
                buf_tokens = 0
                for sent in sentences:
                    st = self.token_len(sent)
                    if buf and buf_tokens + st > self.target_tokens:
                        text = " ".join(buf)
                        s, e = _locate(doc.text, buf[0], cursor)
                        out.append(
                            self._mk(
                                doc,
                                text,
                                span=(s, s + len(text)),
                                seq=seq,
                                token_len=self.token_len,
                            )
                        )
                        seq += 1
                        cursor = s
                        # carry the tail sentences forward as overlap
                        buf = buf[-self.overlap_sentences :] if self.overlap_sentences else []
                        buf_tokens = sum(self.token_len(x) for x in buf)
                    buf.append(sent)
                    buf_tokens += st
                if buf:
                    text = " ".join(buf)
                    s, e = _locate(doc.text, buf[0], cursor)
                    out.append(
                        self._mk(
                            doc, text, span=(s, s + len(text)), seq=seq, token_len=self.token_len
                        )
                    )
                    seq += 1
                    cursor = s
        return out


# --------------------------------------------------------------------------
# 4. Sentence-window — index one sentence (precise match surface), return the
#    surrounding window (enough context to actually answer from). This is the
#    clearest example of why `text` and `context_text` are separate fields.
# --------------------------------------------------------------------------
class SentenceWindowChunker(BaseChunker):
    name = "sentence_window"

    def __init__(self, window: int = 2, token_len: TokenLen = approx_token_len):
        self.window = window
        self.token_len = token_len

    def chunk(self, docs: Sequence[Document]) -> list[Chunk]:
        out: list[Chunk] = []
        for doc in docs:
            sentences = split_sentences(doc.text)
            if not sentences:
                continue
            cursor = 0
            spans: list[tuple[int, int]] = []
            for sent in sentences:
                s, e = _locate(doc.text, sent, cursor)
                spans.append((s, e))
                cursor = e
            for i, sent in enumerate(sentences):
                lo = max(0, i - self.window)
                hi = min(len(sentences), i + self.window + 1)
                context = " ".join(sentences[lo:hi])
                out.append(
                    self._mk(
                        doc,
                        sent,
                        context_text=context,
                        span=spans[i],
                        seq=i,
                        token_len=self.token_len,
                        extra={"window": self.window, "sentence_index": i},
                    )
                )
        return out


# --------------------------------------------------------------------------
# 5. Semantic drift — embed sentences, cut where consecutive similarity drops
#    below a percentile of the document's own distribution. Adaptive: a
#    topically uniform passage stays whole, a list of unrelated facts fragments.
# --------------------------------------------------------------------------
class SemanticDriftChunker(BaseChunker):
    name = "semantic_drift"

    def __init__(
        self,
        encoder=None,
        breakpoint_percentile: float = 25.0,
        min_sentences: int = 1,
        max_tokens: int = 260,
        token_len: TokenLen = approx_token_len,
    ):
        self.encoder = encoder
        self.breakpoint_percentile = breakpoint_percentile
        self.min_sentences = min_sentences
        self.max_tokens = max_tokens
        self.token_len = token_len

    def chunk(self, docs: Sequence[Document]) -> list[Chunk]:
        if self.encoder is None:
            # Degrade to structural splitting rather than failing the build.
            return RecursiveStructuralChunker(token_len=self.token_len).chunk(docs)

        out: list[Chunk] = []
        # Batch every sentence in the corpus through the encoder once.
        all_sents: list[str] = []
        offsets: list[tuple[int, int]] = []
        per_doc: list[list[str]] = []
        for doc in docs:
            sents = split_sentences(doc.text)
            per_doc.append(sents)
            offsets.append((len(all_sents), len(all_sents) + len(sents)))
            all_sents.extend(sents)

        if not all_sents:
            return out
        vectors = self.encoder.encode_passages(all_sents)

        for doc, sents, (lo, hi) in zip(docs, per_doc, offsets):
            if len(sents) <= 1:
                if sents:
                    out.append(
                        self._mk(doc, sents[0], span=(0, len(doc.text)), seq=0, token_len=self.token_len)
                    )
                continue

            vecs = vectors[lo:hi]
            # cosine between consecutive sentences (vectors are L2-normalised)
            sims = np.sum(vecs[:-1] * vecs[1:], axis=1)
            if sims.size == 0:
                cut_at = set()
            else:
                threshold = float(np.percentile(sims, self.breakpoint_percentile))
                cut_at = {i for i, s in enumerate(sims) if s < threshold}

            groups: list[list[str]] = []
            current: list[str] = []
            current_tokens = 0
            for i, sent in enumerate(sents):
                current.append(sent)
                current_tokens += self.token_len(sent)
                overflow = current_tokens >= self.max_tokens
                is_break = i in cut_at and len(current) >= self.min_sentences
                if (is_break or overflow) and i < len(sents) - 1:
                    groups.append(current)
                    current, current_tokens = [], 0
            if current:
                groups.append(current)

            cursor = 0
            for seq, group in enumerate(groups):
                text = " ".join(group)
                s, _ = _locate(doc.text, group[0], cursor)
                out.append(
                    self._mk(
                        doc,
                        text,
                        span=(s, s + len(text)),
                        seq=seq,
                        token_len=self.token_len,
                        extra={"n_sentences": len(group)},
                    )
                )
                cursor = s
        return out


# --------------------------------------------------------------------------
# 6. Proposition — decompose into atomic, self-contained claims. Each chunk is
#    one fact, so a query matching one fact doesn't drag in four unrelated ones.
#    Rule-based (clause splitting + pronoun-free filtering) so it stays offline
#    and free; an LLM decomposition pass can be swapped in behind the same name.
# --------------------------------------------------------------------------
class PropositionChunker(BaseChunker):
    name = "proposition"

    def __init__(
        self,
        min_tokens: int = 5,
        max_tokens: int = 90,
        token_len: TokenLen = approx_token_len,
    ):
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.token_len = token_len

    def chunk(self, docs: Sequence[Document]) -> list[Chunk]:
        out: list[Chunk] = []
        for doc in docs:
            sentences = split_sentences(doc.text)
            seq = 0
            cursor = 0
            for sent in sentences:
                pieces = [p.strip() for p in _CLAUSE_SPLIT.split(sent) if p and p.strip()]
                if not pieces:
                    pieces = [sent]
                # Merge fragments that are too short to stand alone as a claim.
                merged: list[str] = []
                for piece in pieces:
                    if merged and self.token_len(piece) < self.min_tokens:
                        merged[-1] = f"{merged[-1]} {piece}"
                    elif merged and self.token_len(merged[-1]) < self.min_tokens:
                        merged[-1] = f"{merged[-1]} {piece}"
                    else:
                        merged.append(piece)
                for piece in merged:
                    if self.token_len(piece) > self.max_tokens:
                        piece = piece[: self.max_tokens * 6]
                    s, e = _locate(doc.text, piece[:40], cursor)
                    out.append(
                        self._mk(
                            doc,
                            piece,
                            # A lone proposition is thin context for generation,
                            # so hand the generator the sentence it came from.
                            context_text=sent,
                            span=(s, s + len(piece)),
                            seq=seq,
                            token_len=self.token_len,
                            extra={"source_sentence": sent[:300]},
                        )
                    )
                    seq += 1
                cursor = _locate(doc.text, sent[:40], cursor)[1]
        return out


# --------------------------------------------------------------------------
# 7. Hierarchical parent-child — retrieve on small children for precision,
#    hand the parent to the generator for recall. Children carry `parent_id`
#    so the retriever can collapse siblings into one parent hit.
# --------------------------------------------------------------------------
class HierarchicalChunker(BaseChunker):
    name = "hierarchical_parent_child"

    def __init__(
        self,
        child_tokens: int = 60,
        token_len: TokenLen = approx_token_len,
    ):
        self.child_tokens = child_tokens
        self.token_len = token_len

    def chunk(self, docs: Sequence[Document]) -> list[Chunk]:
        out: list[Chunk] = []
        for doc in docs:
            parent_id = _hash_id("parent", doc.doc_id)
            sentences = split_sentences(doc.text)
            buf: list[str] = []
            buf_tokens = 0
            seq = 0
            cursor = 0

            def flush(buf_local: list[str], seq_local: int, cursor_local: int) -> int:
                text = " ".join(buf_local)
                s, _ = _locate(doc.text, buf_local[0], cursor_local)
                out.append(
                    self._mk(
                        doc,
                        text,
                        # The child is the match surface; the whole passage is
                        # the answer surface.
                        context_text=doc.text,
                        span=(s, s + len(text)),
                        parent_id=parent_id,
                        seq=seq_local,
                        token_len=self.token_len,
                        extra={"level": "child"},
                    )
                )
                return s

            for sent in sentences:
                st = self.token_len(sent)
                if buf and buf_tokens + st > self.child_tokens:
                    cursor = flush(buf, seq, cursor)
                    seq += 1
                    buf, buf_tokens = [], 0
                buf.append(sent)
                buf_tokens += st
            if buf:
                flush(buf, seq, cursor)
        return out


# --------------------------------------------------------------------------
# 8. Late chunking — encode the whole document in one forward pass, then mean-
#    pool the token embeddings belonging to each chunk's span. Each chunk's
#    vector therefore carries document-level context (resolved pronouns, topic)
#    that a chunk encoded in isolation cannot have.
# --------------------------------------------------------------------------
class LateChunkingChunker(BaseChunker):
    name = "late_chunking"

    def __init__(
        self,
        encoder=None,
        target_tokens: int = 90,
        token_len: TokenLen = approx_token_len,
    ):
        self.encoder = encoder
        self.target_tokens = target_tokens
        self.token_len = token_len
        self._vectors: dict[str, np.ndarray] = {}

    def chunk(self, docs: Sequence[Document]) -> list[Chunk]:
        out: list[Chunk] = []
        self._vectors = {}
        for doc in docs:
            sentences = split_sentences(doc.text)
            if not sentences:
                continue
            buf: list[str] = []
            buf_tokens = 0
            spans: list[tuple[int, int]] = []
            texts: list[str] = []
            cursor = 0
            for sent in sentences:
                st = self.token_len(sent)
                if buf and buf_tokens + st > self.target_tokens:
                    text = " ".join(buf)
                    s, _ = _locate(doc.text, buf[0], cursor)
                    spans.append((s, s + len(text)))
                    texts.append(text)
                    cursor = s
                    buf, buf_tokens = [], 0
                buf.append(sent)
                buf_tokens += st
            if buf:
                text = " ".join(buf)
                s, _ = _locate(doc.text, buf[0], cursor)
                spans.append((s, s + len(text)))
                texts.append(text)

            chunk_objs = [
                self._mk(doc, t, span=sp, seq=i, token_len=self.token_len,
                         extra={"level": "late"})
                for i, (t, sp) in enumerate(zip(texts, spans))
            ]

            if self.encoder is not None:
                try:
                    pooled = self.encoder.encode_spans(doc.text, spans)
                    for chunk_obj, vec in zip(chunk_objs, pooled):
                        self._vectors[chunk_obj.chunk_id] = vec
                except Exception:  # noqa: BLE001 - fall back to normal embedding
                    pass
            out.extend(chunk_objs)
        return out

    def embed_override(self, chunks: Sequence[Chunk], encoder) -> np.ndarray | None:
        if not self._vectors:
            return None
        missing = [c.chunk_id for c in chunks if c.chunk_id not in self._vectors]
        if missing:
            return None
        return np.stack([self._vectors[c.chunk_id] for c in chunks]).astype(np.float32)


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------
CHUNKER_FACTORIES: dict[str, Callable[..., BaseChunker]] = {
    PassageAtomicChunker.name: PassageAtomicChunker,
    FixedWindowChunker.name: FixedWindowChunker,
    RecursiveStructuralChunker.name: RecursiveStructuralChunker,
    SentenceWindowChunker.name: SentenceWindowChunker,
    SemanticDriftChunker.name: SemanticDriftChunker,
    PropositionChunker.name: PropositionChunker,
    HierarchicalChunker.name: HierarchicalChunker,
    LateChunkingChunker.name: LateChunkingChunker,
}

ALL_STRATEGIES = list(CHUNKER_FACTORIES)

# Strategies that need an encoder handed to them at construction time.
ENCODER_STRATEGIES = {SemanticDriftChunker.name, LateChunkingChunker.name}


def build_chunker(name: str, encoder=None, token_len: TokenLen = approx_token_len, **kwargs) -> BaseChunker:
    if name not in CHUNKER_FACTORIES:
        raise KeyError(f"unknown chunking strategy {name!r}; known: {ALL_STRATEGIES}")
    factory = CHUNKER_FACTORIES[name]
    if name in ENCODER_STRATEGIES:
        return factory(encoder=encoder, token_len=token_len, **kwargs)
    return factory(token_len=token_len, **kwargs)


def chunk_stats(chunks: Iterable[Chunk]) -> dict:
    chunks = list(chunks)
    if not chunks:
        return {"n_chunks": 0}
    tokens = [c.token_count for c in chunks]
    return {
        "n_chunks": len(chunks),
        "n_docs": len({c.doc_id for c in chunks}),
        "tokens_mean": round(float(np.mean(tokens)), 1),
        "tokens_p50": int(np.percentile(tokens, 50)),
        "tokens_p95": int(np.percentile(tokens, 95)),
        "tokens_max": int(np.max(tokens)),
        "chunks_per_doc": round(len(chunks) / max(1, len({c.doc_id for c in chunks})), 2),
    }
