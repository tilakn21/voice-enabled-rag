"""Answer synthesis — two paths behind one interface.

`ExtractiveSynthesizer` is the fast path and the default. It selects and
stitches the sentences from retrieved passages that actually answer the
question. No model call, ~2ms, and grounded by construction: every token it
emits came verbatim from a cited chunk. This is what keeps the end-to-end
pipeline inside the 200ms target.

`ClaudeSynthesizer` is the quality path. It runs a real tool-calling loop —
Claude can re-search the corpus with a reformulated query, pull the full parent
passage of a promising chunk, and must terminate by calling `submit_answer`
with a structured payload naming its citations. Making the final answer a tool
call rather than free text is what gives us schema-validated output out of an
agentic loop, so the output guardrails have typed fields to check rather than
prose to parse.

Both return an `AnswerPayload`, so the guardrail stage downstream is identical.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .chunking import content_tokens as _content_tokens, split_sentences
from .config import Settings
from .schemas import AnswerPayload, Citation, RetrievedChunk

logger = logging.getLogger(__name__)

ABSTAIN_TEXT = (
    "I don't have information about that in the indexed passages, so I'd rather "
    "not guess."
)


def _citation(rc: RetrievedChunk, snippet_len: int = 260) -> Citation:
    payload = rc.chunk.payload
    return Citation(
        chunk_id=rc.chunk.chunk_id,
        doc_id=rc.chunk.doc_id,
        strategy=rc.chunk.strategy,
        lang=rc.chunk.lang,
        score=round(rc.score, 4),
        snippet=payload[:snippet_len] + ("…" if len(payload) > snippet_len else ""),
    )


# --------------------------------------------------------------------------
# Fast path
# --------------------------------------------------------------------------
@dataclass
class _Candidate:
    text: str
    chunk: RetrievedChunk
    order: tuple[int, int]
    lexical: float
    prior: float
    dense: float = 0.0

    @property
    def score(self) -> float:
        return 0.55 * self.lexical + 0.25 * self.prior + 0.20 * self.dense


class ExtractiveSynthesizer:
    def __init__(
        self,
        encoder=None,
        max_sentences: int = 3,
        max_chars: int = 600,
        relevance_floor: float = 0.45,
        rerank_top_n: int = 8,
        rerank_max_tokens: int = 64,
    ):
        self.encoder = encoder
        self.max_sentences = max_sentences
        self.max_chars = max_chars
        # A follow-on sentence must score at least this fraction of the best
        # sentence's score to be included.
        self.relevance_floor = relevance_floor
        # Dense sentence reranking was the single largest cost in the pipeline
        # (P50 121ms of a 140ms budget) before these two bounds. Sentences are
        # short, so padding them out to the full 192-token passage window was
        # mostly wasted compute, and reranking 16 candidates bought nothing
        # over 8.
        self.rerank_top_n = rerank_top_n
        self.rerank_max_tokens = rerank_max_tokens

    def synthesize(
        self,
        query: str,
        retrieved: Sequence[RetrievedChunk],
        *,
        use_dense_rerank: bool = True,
        query_vector: np.ndarray | None = None,
    ) -> AnswerPayload:
        if not retrieved:
            return AnswerPayload(
                answer=ABSTAIN_TEXT, answered=False, abstain_reason="no_results"
            )

        query_tokens = set(_content_tokens(query))
        candidates: list[_Candidate] = []

        for ci, rc in enumerate(retrieved):
            sentences = split_sentences(rc.chunk.payload)
            for si, sent in enumerate(sentences):
                tokens = set(_content_tokens(sent))
                if not tokens:
                    continue
                # Coverage of the *query* matters more than coverage of the
                # sentence: a long sentence containing every query term is a
                # better answer than a short one that happens to be all-overlap.
                overlap = len(query_tokens & tokens)
                lexical = overlap / max(1, len(query_tokens))
                # Mild length preference — one-word sentences rarely answer anything.
                length_bonus = min(1.0, len(tokens) / 12.0)
                candidates.append(
                    _Candidate(
                        text=sent.strip(),
                        chunk=rc,
                        order=(ci, si),
                        lexical=lexical * (0.7 + 0.3 * length_bonus),
                        prior=rc.score,
                    )
                )

        if not candidates:
            return AnswerPayload(
                answer=ABSTAIN_TEXT, answered=False, abstain_reason="no_sentences"
            )

        # Dense rerank of the shortlist only — bounded work, and the caller
        # skips it when the remaining budget is thin.
        if use_dense_rerank and self.encoder is not None and len(candidates) > 1:
            shortlist = sorted(candidates, key=lambda c: c.score, reverse=True)[
                : self.rerank_top_n
            ]
            try:
                # The pipeline already encoded the query; re-encoding it here
                # was a second full forward pass for no new information.
                qv = query_vector if query_vector is not None else self.encoder.encode_query(query)
                sv = self.encoder.encode_passages(
                    [c.text for c in shortlist],
                    max_length=self.rerank_max_tokens,
                )
                for cand, vec in zip(shortlist, sv):
                    cand.dense = float(np.dot(qv, vec))
            except Exception as exc:  # noqa: BLE001
                logger.debug("dense sentence rerank skipped: %s", exc)

        ranked = sorted(candidates, key=lambda c: c.score, reverse=True)

        picked: list[_Candidate] = []
        seen: set[str] = set()
        total = 0
        best_score = ranked[0].score
        for cand in ranked:
            key = " ".join(sorted(set(_content_tokens(cand.text))))[:120]
            if key in seen:
                continue
            if total + len(cand.text) > self.max_chars and picked:
                continue
            # Every sentence after the first has to earn its place. Without
            # this the answer pads itself out to max_sentences with whatever
            # ranked next, which is how an unrelated passage ends up appended
            # to an otherwise correct answer.
            if picked and (
                cand.score < self.relevance_floor * best_score or cand.lexical <= 0.0
            ):
                continue
            picked.append(cand)
            seen.add(key)
            total += len(cand.text)
            if len(picked) >= self.max_sentences:
                break

        if not picked:
            return AnswerPayload(
                answer=ABSTAIN_TEXT, answered=False, abstain_reason="no_sentences"
            )

        # Restore reading order so the answer flows.
        picked.sort(key=lambda c: c.order)
        answer = " ".join(c.text for c in picked).strip()

        cited_ids: list[str] = []
        citations: list[Citation] = []
        cited_chunks: list[RetrievedChunk] = []
        for cand in picked:
            cid = cand.chunk.chunk.chunk_id
            if cid not in cited_ids:
                cited_ids.append(cid)
                citations.append(_citation(cand.chunk))
                cited_chunks.append(cand.chunk)

        return AnswerPayload(
            answer=answer,
            answered=True,
            citations=citations,
            cited_chunks=cited_chunks,
            groundedness=1.0,  # verified independently by the output rails
            confidence=float(np.mean([c.score for c in picked])),
        )


# --------------------------------------------------------------------------
# Quality path — Claude with tools
# --------------------------------------------------------------------------
SYSTEM_PROMPT = """You answer questions strictly from a retrieved passage corpus \
(MS MARCO, translated into Indic languages).

Rules:
- Use ONLY the retrieved passages. Never use outside knowledge, even if you are \
confident it is correct.
- If the passages do not contain the answer, say so via submit_answer with \
sufficient=false. An honest "not in the corpus" is a correct answer here; a \
plausible guess is a failure.
- Answer in the same language as the question.
- Be direct and brief — two sentences at most.
- You must finish by calling submit_answer exactly once. Every citation_id you \
pass must be a chunk_id that appeared in the context you were given or that a \
search_corpus call returned.

You may call search_corpus with a reformulated query if the initial passages \
look off-target, and expand_context to see the full passage a chunk came from."""


@dataclass
class _ToolState:
    retriever: object
    settings: Settings
    known: dict[str, RetrievedChunk] = field(default_factory=dict)
    submitted: dict | None = None
    tool_calls: list[str] = field(default_factory=list)

    def register(self, chunks: Sequence[RetrievedChunk]) -> None:
        for rc in chunks:
            self.known[rc.chunk.chunk_id] = rc


def _render_chunks(chunks: Sequence[RetrievedChunk]) -> str:
    lines = []
    for rc in chunks:
        lines.append(
            f"[chunk_id={rc.chunk.chunk_id}] (lang={rc.chunk.lang}, "
            f"strategy={rc.chunk.strategy}, score={rc.score:.3f})\n{rc.chunk.payload}"
        )
    return "\n\n".join(lines) if lines else "(no passages)"


class ClaudeSynthesizer:
    def __init__(self, settings: Settings, retriever, fallback: ExtractiveSynthesizer):
        self.settings = settings
        self.retriever = retriever
        self.fallback = fallback
        self._client = None
        self._fallbacks_supported = True

    # ------------------------------------------------------------------
    @property
    def available(self) -> bool:
        return bool(self.settings.anthropic_api_key)

    def _get_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(
                api_key=self.settings.anthropic_api_key,
                timeout=self.settings.anthropic_timeout_s,
                max_retries=1,  # the harness owns retry policy
            )
        return self._client

    # ------------------------------------------------------------------
    def synthesize(self, query: str, retrieved: Sequence[RetrievedChunk]) -> AnswerPayload:
        from anthropic import beta_tool

        state = _ToolState(retriever=self.retriever, settings=self.settings)
        state.register(retrieved)

        @beta_tool
        def search_corpus(query: str, top_k: int = 5) -> str:
            """Search the passage corpus again with a reformulated query.

            Use when the passages already shown don't cover the question — for
            example to try a synonym, a more specific entity, or the English
            form of an Indic-language term.

            Args:
                query: The reformulated search query.
                top_k: How many passages to return (1-10).
            """
            state.tool_calls.append("search_corpus")
            result = state.retriever.retrieve(query, top_k=max(1, min(top_k, 10)))
            state.register(result.chunks)
            return _render_chunks(result.chunks)

        @beta_tool
        def expand_context(chunk_id: str) -> str:
            """Return the full source passage a chunk was cut from.

            Use when a chunk looks relevant but is cut off mid-thought.

            Args:
                chunk_id: A chunk_id from the passages you have been shown.
            """
            state.tool_calls.append("expand_context")
            rc = state.known.get(chunk_id)
            if rc is None:
                return f"No chunk with id {chunk_id!r} is in scope."
            chunk = rc.chunk
            siblings = [
                c
                for c in state.retriever._chunk_by_id.values()  # noqa: SLF001
                if c.doc_id == chunk.doc_id
            ]
            widest = max(siblings, key=lambda c: len(c.payload), default=chunk)
            return f"[chunk_id={chunk.chunk_id}] full passage:\n{widest.payload}"

        @beta_tool
        def submit_answer(answer: str, citation_ids: list[str], sufficient: bool) -> str:
            """Submit the final answer. Call this exactly once, last.

            Args:
                answer: The answer, in the same language as the question. Two
                    sentences maximum. If sufficient is false, briefly say the
                    corpus does not cover the question.
                citation_ids: chunk_ids supporting the answer. Empty only when
                    sufficient is false.
                sufficient: Whether the retrieved passages actually answer the
                    question.
            """
            state.tool_calls.append("submit_answer")
            state.submitted = {
                "answer": answer,
                "citation_ids": list(citation_ids or []),
                "sufficient": bool(sufficient),
            }
            return "recorded"

        user_message = (
            f"Question: {query}\n\n"
            f"Retrieved passages:\n{_render_chunks(retrieved)}\n\n"
            "Answer using only these passages, then call submit_answer."
        )

        request: dict = {
            "model": self.settings.anthropic_model,
            "max_tokens": self.settings.anthropic_max_tokens,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_message}],
            "tools": [search_corpus, expand_context, submit_answer],
            # Adaptive thinking rather than disabled: with thinking off, Opus 5
            # can emit a tool call as plain text, which would silently skip
            # submit_answer. Low effort keeps the latency cost down.
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self.settings.anthropic_effort},
        }
        if self._fallbacks_supported:
            request["betas"] = ["server-side-fallback-2026-07-01"]
            request["fallbacks"] = "default"

        client = self._get_client()
        final = None
        try:
            final = self._drive(client, request, state)
        except Exception as exc:  # noqa: BLE001
            if self._fallbacks_supported and _looks_like_beta_rejection(exc):
                # Older API surface: drop the fallback opt-in and try once more.
                logger.warning("server-side fallbacks rejected (%s); retrying without", exc)
                self._fallbacks_supported = False
                request.pop("betas", None)
                request.pop("fallbacks", None)
                final = self._drive(client, request, state)
            else:
                raise

        if final is not None and getattr(final, "stop_reason", None) == "refusal":
            details = getattr(final, "stop_details", None)
            category = getattr(details, "category", None)
            logger.warning("model refused (category=%s)", category)
            return AnswerPayload(
                answer="I can't help with that request.",
                answered=False,
                abstain_reason=f"model_refusal:{category or 'unspecified'}",
            )

        if state.submitted is None:
            # The loop ended without a structured submission. Rather than
            # scraping prose, fall back to the deterministic extractive path.
            logger.warning("submit_answer was never called; using extractive fallback")
            payload = self.fallback.synthesize(query, retrieved)
            payload.abstain_reason = payload.abstain_reason or "llm_no_structured_output"
            return payload

        return self._to_payload(state)

    # ------------------------------------------------------------------
    def _drive(self, client, request: dict, state: _ToolState):
        runner = client.beta.messages.tool_runner(**request)
        final = None
        for i, message in enumerate(runner):
            final = message
            if state.submitted is not None:
                break
            if i + 1 >= self.settings.llm_max_tool_iterations:
                logger.warning("tool loop hit iteration cap (%d)", i + 1)
                break
        return final

    def _to_payload(self, state: _ToolState) -> AnswerPayload:
        submitted = state.submitted or {}
        if not submitted.get("sufficient", False):
            return AnswerPayload(
                answer=submitted.get("answer") or ABSTAIN_TEXT,
                answered=False,
                abstain_reason="model_insufficient_context",
            )

        citations: list[Citation] = []
        cited_chunks: list[RetrievedChunk] = []
        for cid in submitted.get("citation_ids", []):
            rc = state.known.get(cid)
            if rc is not None:
                citations.append(_citation(rc))
                cited_chunks.append(rc)
        # A model that answered but cited nothing valid gets caught by the
        # citation-validity rail; seeding with the top hit would launder that.

        return AnswerPayload(
            answer=(submitted.get("answer") or "").strip(),
            answered=True,
            citations=citations,
            cited_chunks=cited_chunks,
            confidence=0.0,  # set by the output rails
        )


def _looks_like_beta_rejection(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(
        marker in text
        for marker in ("beta", "fallback", "unsupported", "unrecognized", "invalid_request")
    )
