"""Answer synthesis — two paths behind one interface.

`ExtractiveSynthesizer` is the fast path and the default. It selects and
stitches the sentences from retrieved passages that actually answer the
question. No model call, ~2ms, and grounded by construction: every token it
emits came verbatim from a cited chunk. This is what keeps the end-to-end
pipeline inside the 200ms target.

`LLMSynthesizer` is the quality path. It runs a real tool-calling loop against
any OpenAI-compatible chat endpoint (Groq by default, but Together, Fireworks,
or a fully local Ollama / vLLM server work unchanged): the model can re-search
the corpus with a reformulated query, pull the full parent passage of a
promising chunk, and must terminate by calling `submit_answer` with a
structured payload naming its citations. Making the final answer a tool call
rather than free text is what gives us schema-validated output out of an
agentic loop, so the output guardrails have typed fields to check rather than
prose to parse.

Both return an `AnswerPayload`, so the guardrail stage downstream is identical.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Sequence

import httpx
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
# Quality path — an open-weight LLM with tools, over an OpenAI-compatible API
#
# Deliberately provider-agnostic rather than tied to one vendor: `llm_base_url`
# plus `llm_model` is all that changes between Groq, Together, Fireworks, and a
# fully local Ollama or vLLM server. That keeps the door open to running the
# whole system offline with no API key at all.
#
# Groq is the default because it is free to start, serves open-weight models
# (Llama, gpt-oss, Qwen), and is fast enough that the quality path stays
# interactive — its synthesis latency dominates this path, so a slow provider
# would make `grounded` mode unusable rather than merely slower.
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
- Be direct and brief - two sentences at most.
- You must finish by calling submit_answer exactly once. Every citation_id you \
pass must be a chunk_id that appeared in the context you were given or that a \
search_corpus call returned.

You may call search_corpus with a reformulated query if the initial passages \
look off-target, and expand_context to see the full passage a chunk came from."""


def _render_chunks(chunks: Sequence[RetrievedChunk]) -> str:
    lines = []
    for rc in chunks:
        lines.append(
            f"[chunk_id={rc.chunk.chunk_id}] (lang={rc.chunk.lang}, "
            f"strategy={rc.chunk.strategy}, score={rc.score:.3f})\n{rc.chunk.payload}"
        )
    return "\n\n".join(lines) if lines else "(no passages)"


# OpenAI-compatible function-tool schemas. `submit_answer` is the terminal tool:
# making the final answer a *tool call* rather than free text is what gives us
# schema-validated output out of an agentic loop, so the output guardrails get
# typed fields to check instead of prose to parse.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_corpus",
            "description": (
                "Search the passage corpus again with a reformulated query. Use "
                "when the passages already shown don't cover the question - for "
                "example to try a synonym, a more specific entity, or the "
                "English form of an Indic-language term."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The reformulated search query."},
                    "top_k": {
                        "type": "integer",
                        "description": "How many passages to return (1-10).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "expand_context",
            "description": (
                "Return the full source passage a chunk was cut from. Use when a "
                "chunk looks relevant but is cut off mid-thought."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_id": {
                        "type": "string",
                        "description": "A chunk_id from the passages you have been shown.",
                    }
                },
                "required": ["chunk_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": "Submit the final answer. Call this exactly once, last.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": (
                            "The answer, in the same language as the question. Two "
                            "sentences maximum. If sufficient is false, briefly say "
                            "the corpus does not cover the question."
                        ),
                    },
                    "citation_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "chunk_ids supporting the answer. Empty only when "
                            "sufficient is false."
                        ),
                    },
                    "sufficient": {
                        "type": "boolean",
                        "description": (
                            "Whether the retrieved passages actually answer the question."
                        ),
                    },
                },
                "required": ["answer", "citation_ids", "sufficient"],
            },
        },
    },
]


class LLMError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class LLMSynthesizer:
    """Tool-calling synthesis against any OpenAI-compatible chat endpoint."""

    def __init__(self, settings: Settings, retriever, fallback: ExtractiveSynthesizer):
        self.settings = settings
        self.retriever = retriever
        self.fallback = fallback

    # ------------------------------------------------------------------
    @property
    def available(self) -> bool:
        # A local server (Ollama, vLLM) needs no key, so a configured non-remote
        # base_url is enough on its own.
        if self.settings.llm_api_key:
            return True
        return self._is_local(self.settings.llm_base_url)

    @staticmethod
    def _is_local(url: str) -> bool:
        return any(h in url for h in ("localhost", "127.0.0.1", "0.0.0.0", "host.docker.internal"))

    @property
    def provider_label(self) -> str:
        host = self.settings.llm_base_url.split("//")[-1].split("/")[0]
        return f"{host}:{self.settings.llm_model}"

    # ------------------------------------------------------------------
    async def synthesize(self, query: str, retrieved: Sequence[RetrievedChunk]) -> AnswerPayload:
        cfg = self.settings
        known: dict[str, RetrievedChunk] = {rc.chunk.chunk_id: rc for rc in retrieved}
        submitted: dict | None = None
        tool_calls_made: list[str] = []

        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question: {query}\n\n"
                    f"Retrieved passages:\n{_render_chunks(retrieved)}\n\n"
                    "Answer using only these passages, then call submit_answer."
                ),
            },
        ]

        headers = {"Content-Type": "application/json"}
        if cfg.llm_api_key:
            headers["Authorization"] = f"Bearer {cfg.llm_api_key}"
        url = cfg.llm_base_url.rstrip("/") + "/chat/completions"

        async with httpx.AsyncClient(timeout=cfg.llm_timeout_s) as client:
            for _ in range(cfg.llm_max_tool_iterations):
                body = {
                    "model": cfg.llm_model,
                    "messages": messages,
                    "tools": TOOL_SCHEMAS,
                    "tool_choice": "auto",
                    # Low temperature: this is extraction, not creative writing.
                    "temperature": cfg.llm_temperature,
                    "max_tokens": cfg.llm_max_tokens,
                }
                resp = await client.post(url, headers=headers, json=body)
                if resp.status_code >= 400:
                    raise LLMError(
                        f"{self.provider_label} returned {resp.status_code}: {resp.text[:300]}",
                        status_code=resp.status_code,
                    )

                choice = (resp.json().get("choices") or [{}])[0]
                message = choice.get("message") or {}
                calls = message.get("tool_calls") or []

                if not calls:
                    # Model answered in prose instead of calling submit_answer.
                    break

                # Echo the assistant turn back before appending tool results -
                # OpenAI-compatible APIs reject tool results with no matching call.
                messages.append(
                    {
                        "role": "assistant",
                        "content": message.get("content") or "",
                        "tool_calls": calls,
                    }
                )

                for call in calls:
                    fn = (call.get("function") or {}).get("name") or ""
                    tool_calls_made.append(fn)
                    try:
                        args = json.loads((call.get("function") or {}).get("arguments") or "{}")
                    except json.JSONDecodeError:
                        args = {}

                    if fn == "search_corpus":
                        result = self.retriever.retrieve(
                            args.get("query") or query,
                            top_k=max(1, min(int(args.get("top_k") or 5), 10)),
                        )
                        for rc in result.chunks:
                            known[rc.chunk.chunk_id] = rc
                        content = _render_chunks(result.chunks)
                    elif fn == "expand_context":
                        rc = known.get(args.get("chunk_id") or "")
                        if rc is None:
                            content = f"No chunk with id {args.get('chunk_id')!r} is in scope."
                        else:
                            siblings = [
                                c
                                for c in self.retriever._chunk_by_id.values()  # noqa: SLF001
                                if c.doc_id == rc.chunk.doc_id
                            ]
                            widest = max(siblings, key=lambda c: len(c.payload), default=rc.chunk)
                            content = f"[chunk_id={rc.chunk.chunk_id}] full passage:\n{widest.payload}"
                    elif fn == "submit_answer":
                        submitted = {
                            "answer": args.get("answer") or "",
                            "citation_ids": list(args.get("citation_ids") or []),
                            "sufficient": bool(args.get("sufficient")),
                        }
                        content = "recorded"
                    else:
                        content = f"Unknown tool {fn!r}."

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.get("id") or fn,
                            "name": fn,
                            "content": content,
                        }
                    )

                if submitted is not None:
                    break

        if submitted is None:
            # No structured submission. Rather than scraping prose, fall back to
            # the deterministic extractive path, which is still correct.
            logger.warning(
                "submit_answer never called by %s (tools used: %s); extractive fallback",
                self.provider_label,
                tool_calls_made or "none",
            )
            payload = self.fallback.synthesize(query, retrieved)
            payload.abstain_reason = payload.abstain_reason or "llm_no_structured_output"
            return payload

        if not submitted["sufficient"]:
            return AnswerPayload(
                answer=submitted["answer"] or ABSTAIN_TEXT,
                answered=False,
                abstain_reason="model_insufficient_context",
            )

        citations: list[Citation] = []
        cited_chunks: list[RetrievedChunk] = []
        for cid in submitted["citation_ids"]:
            rc = known.get(cid)
            if rc is not None:
                citations.append(_citation(rc))
                cited_chunks.append(rc)
        # A model that answered but cited nothing valid is caught by the
        # citation-validity rail; seeding with the top hit would launder that.

        return AnswerPayload(
            answer=submitted["answer"].strip(),
            answered=True,
            citations=citations,
            cited_chunks=cited_chunks,
            confidence=0.0,  # set by the output rails
        )
