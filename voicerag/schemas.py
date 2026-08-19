"""Typed contracts for every stage boundary in the pipeline.

Every stage of the harness takes a validated model in and returns a validated
model out, so a malformed intermediate result fails loudly at the boundary that
produced it instead of surfacing as a confusing error three stages later.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
# Corpus / chunking
# --------------------------------------------------------------------------
class Chunk(BaseModel):
    """A retrievable unit. `text` is what gets embedded; `context_text` is what
    the answer generator actually sees (may be wider, e.g. sentence windows)."""

    chunk_id: str
    doc_id: str
    text: str
    context_text: str | None = None
    strategy: str
    # Metadata-aware retrieval: these travel with the chunk and are filterable.
    lang: str = "unknown"
    query_id: int | None = None
    passage_idx: int | None = None
    char_start: int = 0
    char_end: int = 0
    parent_id: str | None = None
    is_selected: bool = False
    token_count: int = 0
    extra: dict[str, Any] = Field(default_factory=dict)

    @property
    def payload(self) -> str:
        return self.context_text or self.text


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float
    dense_score: float | None = None
    sparse_score: float | None = None
    rank_sources: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Guardrails
# --------------------------------------------------------------------------
class RailVerdict(str, Enum):
    PASS = "pass"
    WARN = "warn"
    BLOCK = "block"


class RailResult(BaseModel):
    name: str
    verdict: RailVerdict
    detail: str = ""
    score: float | None = None


class GuardrailReport(BaseModel):
    results: list[RailResult] = Field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return any(r.verdict is RailVerdict.BLOCK for r in self.results)

    @property
    def block_reason(self) -> str | None:
        for r in self.results:
            if r.verdict is RailVerdict.BLOCK:
                return f"{r.name}: {r.detail}"
        return None

    def add(self, name: str, verdict: RailVerdict, detail: str = "", score: float | None = None):
        self.results.append(RailResult(name=name, verdict=verdict, detail=detail, score=score))
        return self


# --------------------------------------------------------------------------
# Telemetry
# --------------------------------------------------------------------------
class Span(BaseModel):
    name: str
    duration_ms: float
    ok: bool = True
    detail: str | None = None


class LatencyBreakdown(BaseModel):
    spans: list[Span] = Field(default_factory=list)
    total_ms: float = 0.0
    stt_ms: float | None = None
    # Post-STT pipeline time; this is the number measured against the 200ms target.
    core_ms: float = 0.0


# --------------------------------------------------------------------------
# Request / response
# --------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str
    mode: Literal["fast", "grounded"] | None = None
    lang: str | None = None
    top_k: int | None = None


class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    strategy: str
    lang: str
    score: float
    snippet: str


class AnswerPayload(BaseModel):
    """Structured output contract the generator must satisfy."""

    answer: str
    answered: bool
    abstain_reason: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    groundedness: float = 0.0
    confidence: float = 0.0
    # The actual chunk objects behind `citations`, carried internally so the
    # output rails can score groundedness against full text rather than the
    # truncated snippet. Excluded from serialisation — the API returns
    # `citations`, not this.
    cited_chunks: list[RetrievedChunk] = Field(default_factory=list, exclude=True)


class QueryResponse(BaseModel):
    request_id: str
    transcript: str | None = None
    detected_language: str | None = None
    query: str
    mode: str
    answer: str
    answered: bool
    abstain_reason: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    groundedness: float = 0.0
    confidence: float = 0.0
    guardrails: GuardrailReport = Field(default_factory=GuardrailReport)
    latency: LatencyBreakdown = Field(default_factory=LatencyBreakdown)
    degraded: bool = False
    degraded_reason: str | None = None
