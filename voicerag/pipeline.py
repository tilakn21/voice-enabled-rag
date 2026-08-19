"""The orchestrated request pipeline.

  audio ─▶ STT ─▶ input rails ─▶ embed ─▶ retrieve ─▶ coverage rails
                                                          │
                                            ┌─────────────┴──────────────┐
                                            ▼                            ▼
                                   extractive (fast)        open LLM + tools (grounded)
                                            └─────────────┬──────────────┘
                                                          ▼
                                                   output rails ─▶ response

Everything after STT runs against one `Budget`. The 200ms target is measured on
that post-STT segment (`core_ms`) — STT is a network call whose duration scales
with how long the user spoke, so folding it in would measure the speaker, not
the pipeline. It's reported separately and in the end-to-end total.

Guardrails can terminate the pipeline at three points: bad input, retrieval
that doesn't clear the in-domain bar, and an answer that fails grounding. All
three produce an honest refusal rather than a degraded answer.
"""

from __future__ import annotations

import logging
import re
import uuid

from .config import Settings
from .embeddings import Encoder
from .generation import ABSTAIN_TEXT, ExtractiveSynthesizer, LLMSynthesizer
from .guardrails import InputRails, OutputRails, redact_pii
from .harness import Budget, Harness
from .retrieval import RetrievalResult, Retriever
from .schemas import AnswerPayload, GuardrailReport, QueryResponse
from .stt import BaseSTT, Transcript
from .telemetry import TelemetryStore, Timer

logger = logging.getLogger(__name__)

# Script detection is enough to spot cross-lingual answering here: the corpus
# is Devanagari-script Hindi plus Latin-script English, so comparing the
# question's script to the evidence's tells us whether the system bridged a
# language the asker could not read.
_SCRIPT_RANGES = (
    ("Deva", (0x0900, 0x097F)), ("Beng", (0x0980, 0x09FF)),
    ("Guru", (0x0A00, 0x0A7F)), ("Gujr", (0x0A80, 0x0AFF)),
    ("Orya", (0x0B00, 0x0B7F)), ("Taml", (0x0B80, 0x0BFF)),
    ("Telu", (0x0C00, 0x0C7F)), ("Knda", (0x0C80, 0x0CFF)),
    ("Mlym", (0x0D00, 0x0D7F)), ("Arab", (0x0600, 0x06FF)),
)


def detect_script(text: str) -> str:
    counts: dict[str, int] = {}
    latin = 0
    for ch in text:
        code = ord(ch)
        if not ch.isalpha():
            continue
        if code < 0x0250:
            latin += 1
            continue
        for name, (lo, hi) in _SCRIPT_RANGES:
            if lo <= code <= hi:
                counts[name] = counts.get(name, 0) + 1
                break
    if counts:
        best = max(counts, key=counts.get)
        # Code-mixed input is common in Indic speech; report the majority.
        return best if counts[best] >= latin else "Latn"
    return "Latn" if latin else "unknown"


class RagService:
    def __init__(
        self,
        settings: Settings,
        encoder: Encoder,
        retriever: Retriever,
        stt: BaseSTT | None = None,
    ):
        self.settings = settings
        self.encoder = encoder
        self.retriever = retriever
        self.stt = stt

        self.harness = Harness(
            max_retries=settings.stage_max_retries,
            breaker_fail_threshold=settings.breaker_fail_threshold,
            breaker_reset_s=settings.breaker_reset_s,
        )
        self.input_rails = InputRails(
            min_chars=settings.min_transcript_chars, max_chars=settings.max_query_chars
        )
        self.output_rails = OutputRails(
            groundedness_threshold=settings.groundedness_threshold,
            in_domain_cosine_threshold=settings.in_domain_cosine_threshold,
        )
        self.extractive = ExtractiveSynthesizer(encoder=encoder)
        self.llm = LLMSynthesizer(settings, retriever, self.extractive)
        self.telemetry = TelemetryStore(maxlen=settings.telemetry_ring_size)

    # ------------------------------------------------------------------
    @property
    def voice_enabled(self) -> bool:
        return self.stt is not None

    @property
    def grounded_enabled(self) -> bool:
        return self.llm.available

    # ------------------------------------------------------------------
    async def transcribe(self, audio: bytes, filename: str, content_type: str, timer: Timer):
        """STT under its own budget — deliberately outside the 200ms core."""
        if self.stt is None:
            raise RuntimeError("no STT provider configured")

        budget = Budget(total_ms=self.settings.stt_timeout_s * 1000)

        async def _call() -> Transcript:
            return await self.stt.transcribe(audio, filename, content_type)

        return await self.harness.run_stage(
            "stt",
            _call,
            timer=timer,
            budget=budget,
            retries=1,
            breaker=f"stt:{self.stt.name}",
            raise_on_fail=True,
        )

    # ------------------------------------------------------------------
    async def answer(
        self,
        query: str,
        *,
        mode: str | None = None,
        lang: str | None = None,
        top_k: int | None = None,
        budget_ms: float | None = None,
        timer: Timer | None = None,
        transcript: str | None = None,
        detected_language: str | None = None,
        stt_ms: float | None = None,
    ) -> QueryResponse:
        cfg = self.settings
        timer = timer or Timer()
        request_id = uuid.uuid4().hex[:12]
        mode = (mode or cfg.default_mode).lower()
        if mode == "grounded" and not self.grounded_enabled:
            logger.info("grounded mode requested but no LLM configured; using fast path")
            mode = "fast"

        default_budget = cfg.grounded_budget_ms if mode == "grounded" else cfg.pipeline_budget_ms
        # A caller-supplied budget is clamped to something sane: below ~15ms
        # not even the encoder fits, and an unbounded budget defeats the point.
        budget_total = default_budget if budget_ms is None else max(15.0, min(budget_ms, 60_000.0))
        budget = Budget(total_ms=budget_total)

        def respond(
            *,
            answer: str,
            answered: bool,
            report: GuardrailReport,
            abstain_reason: str | None = None,
            citations=None,
            groundedness: float = 0.0,
            confidence: float = 0.0,
            degraded: bool = False,
            degraded_reason: str | None = None,
        ) -> QueryResponse:
            cites = citations or []
            evidence_langs = sorted({c.lang for c in cites})
            q_script = detect_script(query)
            cross = any(
                q_script != "unknown"
                and not (lang.split("_")[-1] or "").startswith(q_script)
                for lang in evidence_langs
            )
            breakdown = timer.build(stt_ms=stt_ms)
            self.telemetry.record(breakdown, mode=mode)
            return QueryResponse(
                request_id=request_id,
                transcript=transcript,
                detected_language=detected_language,
                query=query,
                mode=mode,
                answer=answer,
                answered=answered,
                abstain_reason=abstain_reason,
                citations=cites,
                groundedness=groundedness,
                confidence=confidence,
                guardrails=report,
                latency=breakdown,
                degraded=degraded,
                degraded_reason=degraded_reason,
                cross_lingual=cross,
                query_script=q_script,
                evidence_languages=evidence_langs,
            )

        # ---- 1. input rails -------------------------------------------
        with timer.span("guard_input"):
            rails = self.input_rails.check(query)
        report = rails.report
        if not rails.safe_to_answer:
            logger.info("[%s] input blocked: %s", request_id, report.block_reason)
            return respond(
                answer=rails.refusal_text or ABSTAIN_TEXT,
                answered=False,
                report=report,
                abstain_reason="input_guardrail",
            )
        clean_query = rails.cleaned_query

        # ---- 2. embed --------------------------------------------------
        query_vector = await self.harness.run_sync(
            "embed_query",
            lambda: self.encoder.encode_query(clean_query),
            timer=timer,
            budget=budget,
            retries=0,
            raise_on_fail=True,
        )

        # ---- 3. retrieve -----------------------------------------------
        empty = RetrievalResult()
        retrieval: RetrievalResult = await self.harness.run_sync(
            "retrieve",
            lambda: self.retriever.retrieve(
                clean_query, top_k=top_k or cfg.final_top_k, lang=lang, query_vector=query_vector
            ),
            timer=timer,
            budget=budget,
            retries=0,
            fallback=empty,
        )

        # ---- 4. coverage / out-of-domain rails --------------------------
        with timer.span("guard_retrieval"):
            ok, reason = self.output_rails.check_retrieval(
                report,
                max_fused=retrieval.max_fused,
                max_cosine=retrieval.max_cosine,
                n_results=len(retrieval.chunks),
                max_sparse=retrieval.max_sparse,
            )
        if not ok:
            logger.info("[%s] abstaining (%s)", request_id, reason)
            return respond(
                answer=ABSTAIN_TEXT,
                answered=False,
                report=report,
                abstain_reason=reason,
            )

        # ---- 5. synthesis ----------------------------------------------
        degraded = False
        degraded_reason = None

        if mode == "grounded":
            payload: AnswerPayload = await self.harness.run_stage(
                "generate_llm",
                lambda: self.llm.synthesize(clean_query, retrieval.chunks),
                timer=timer,
                budget=budget,
                retries=1,
                breaker="llm",
                fallback=None,
            )
            if payload is None:
                # LLM unavailable, timed out, or circuit open — the extractive
                # path still produces a correct, grounded answer.
                degraded = True
                degraded_reason = "llm_unavailable_fell_back_to_extractive"
                with timer.span("generate_extractive_fallback"):
                    payload = self.extractive.synthesize(
                        clean_query, retrieval.chunks, query_vector=query_vector
                    )
        else:
            # Only pay for dense sentence reranking if the budget allows it.
            use_dense = budget.remaining_ms > 60.0
            if not use_dense:
                degraded = True
                degraded_reason = "skipped_dense_sentence_rerank_low_budget"
            with timer.span("generate_extractive"):
                payload = self.extractive.synthesize(
                    clean_query,
                    retrieval.chunks,
                    use_dense_rerank=use_dense,
                    query_vector=query_vector,
                )

        if not payload.answered:
            return respond(
                answer=payload.answer or ABSTAIN_TEXT,
                answered=False,
                report=report,
                abstain_reason=payload.abstain_reason or "insufficient_context",
                degraded=degraded,
                degraded_reason=degraded_reason,
            )

        # ---- 6. output rails --------------------------------------------
        # A citation is legitimate if it came from the initial retrieval *or*
        # from a search_corpus call the model made mid-loop. The synthesizer
        # reports exactly which chunks it cited, so the rail checks those.
        cited = payload.cited_chunks
        retrieved_ids = {rc.chunk.chunk_id for rc in retrieval.chunks}
        retrieved_ids |= {rc.chunk.chunk_id for rc in cited}

        with timer.span("guard_output"):
            outcome = self.output_rails.check_answer(
                report,
                answer=payload.answer,
                cited=cited,
                retrieved_ids=retrieved_ids,
                max_fused=retrieval.max_fused,
            )

        if not outcome.answered:
            logger.info("[%s] output blocked: %s", request_id, report.block_reason)
            return respond(
                answer=ABSTAIN_TEXT,
                answered=False,
                report=report,
                abstain_reason=outcome.abstain_reason,
                groundedness=outcome.groundedness,
                degraded=degraded,
                degraded_reason=degraded_reason,
            )

        logger.info(
            "[%s] answered mode=%s core=%.1fms grounded=%.2f q=%r",
            request_id,
            mode,
            timer.build().core_ms,
            outcome.groundedness,
            redact_pii(clean_query)[:60],
        )

        return respond(
            answer=payload.answer,
            answered=True,
            report=report,
            citations=payload.citations,
            groundedness=outcome.groundedness,
            confidence=outcome.confidence,
            degraded=degraded,
            degraded_reason=degraded_reason,
        )

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        return {
            "latency": self.telemetry.snapshot(),
            "circuit_breakers": self.harness.breaker_states(),
            "index": {
                "n_chunks": self.retriever.n_chunks,
                "strategies": {
                    name: {
                        "n_chunks": len(shard.chunks),
                        "dense_backend": shard.dense.backend,
                        "bm25_terms": len(shard.sparse.vocab),
                    }
                    for name, shard in self.retriever.shards.items()
                },
            },
            "config": {
                "target_core_ms": self.settings.pipeline_budget_ms,
                "embed_model": self.settings.embed_model,
                "llm_model": self.settings.llm_model,
                "llm_endpoint": self.settings.llm_base_url,
                "default_mode": self.settings.default_mode,
                "voice_enabled": self.voice_enabled,
                "grounded_enabled": self.grounded_enabled,
                "stt_provider": self.settings.stt_provider if self.voice_enabled else None,
            },
        }
