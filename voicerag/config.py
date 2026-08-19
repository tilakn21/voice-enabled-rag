"""Central configuration. Everything tunable lives here or in the environment."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INDEX_DIR = DATA_DIR / "index"
CORPUS_DIR = DATA_DIR / "corpus"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=os.environ.get("VOICERAG_ENV_FILE", str(PROJECT_ROOT / ".env")),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---------- speech-to-text ----------
    # "sarvam" | "elevenlabs" | "none"
    stt_provider: str = "sarvam"
    sarvam_api_key: str | None = None
    sarvam_stt_url: str = "https://api.sarvam.ai/speech-to-text"
    sarvam_model: str = "saarika:v2.5"
    # "unknown" lets Saarika auto-detect the spoken language.
    sarvam_language_code: str = "unknown"
    elevenlabs_api_key: str | None = None
    elevenlabs_stt_url: str = "https://api.elevenlabs.io/v1/speech-to-text"
    elevenlabs_model: str = "scribe_v1"
    stt_timeout_s: float = 12.0

    # ---------- embeddings ----------
    # multilingual-e5-small: 384-dim, covers the Indic languages in MSMARCO-XI.
    embed_model: str = "intfloat/multilingual-e5-small"
    embed_dim: int = 384
    embed_max_tokens: int = 192
    embed_batch_size: int = 128
    # int8 dynamic quantisation of the Linear layers ~halves CPU latency.
    embed_quantize: bool = True
    embed_threads: int = 4
    embed_query_prefix: str = "query: "
    embed_passage_prefix: str = "passage: "

    # ---------- retrieval ----------
    # Which chunking strategies get a live index. All eight are implemented and
    # scored by scripts/bench_retrieval.py; these two are what the measurement
    # actually justifies shipping:
    #   hierarchical_parent_child  nDCG@10 0.349  <- best of eight
    #   passage_atomic             nDCG@10 0.333  (cheapest index, 1 chunk/doc)
    # Deliberately NOT shipped:
    #   sentence_window   nDCG@10 0.295 — worst of eight, and the largest index
    #                     (3.8 chunks/doc). Precision framing doesn't pay off
    #                     when relevance is judged at passage level.
    #   proposition       nDCG@10 0.307 — 4.6 chunks/doc for worse quality.
    #   semantic_drift    nDCG@10 0.324 — needs an extra full-corpus sentence
    #                     embedding pass, roughly doubling build time.
    #   late_chunking     nDCG@10 0.333 — ties passage_atomic at higher build
    #                     cost; worth revisiting on longer documents.
    active_strategies: list[str] = Field(
        default_factory=lambda: [
            "hierarchical_parent_child",
            "passage_atomic",
        ]
    )
    dense_top_k: int = 40
    sparse_top_k: int = 40
    rrf_k: int = 60
    final_top_k: int = 6
    # 0.72 cost ~0.10 nDCG@10 in ablation (diversity is the wrong objective
    # when the metric is "did you retrieve the one gold passage"). 0.9 keeps
    # near-duplicate suppression at no measured quality cost.
    mmr_lambda: float = 0.9
    hnsw_ef_search: int = 64
    hnsw_ef_construction: int = 200
    hnsw_m: int = 16

    # ---------- guardrails ----------
    # Topic out-of-domain gate: minimum cosine between the query and the best
    # passage. Set by scripts/calibrate_guardrails.py, which measures AUC 0.999
    # for this signal against genuinely off-topic questions. A BM25 co-signal
    # was tried and dropped (measured AUC 0.65). Questions that are
    # *unanswerable* rather than off-topic are caught by the answerability
    # input rail instead — see voicerag/guardrails.py for why.
    in_domain_cosine_threshold: float = 0.845
    # Fraction of answer content tokens that must be supported by cited context.
    groundedness_threshold: float = 0.60
    min_transcript_chars: int = 3
    max_query_chars: int = 800

    # ---------- generation ----------
    # "fast"    -> extractive synthesis only, no LLM call, sub-200ms budget
    # "grounded"-> adds the LLM tool-calling harness on top
    default_mode: str = "fast"
    # Open-weight LLM over an OpenAI-compatible chat API. Provider-agnostic on
    # purpose: base_url + model is the only thing that changes between Groq,
    # Together, Fireworks, and a fully local Ollama / vLLM server, so the
    # quality path can run with no vendor account at all.
    #   Groq (default, free tier, fast):
    #     https://api.groq.com/openai/v1   llama-3.3-70b-versatile
    #     ...also openai/gpt-oss-20b, openai/gpt-oss-120b, qwen/qwen3-32b
    #   Local, no key needed:
    #     http://localhost:11434/v1        llama3.1:8b        (Ollama)
    #     http://localhost:8000/v1         <served model>     (vLLM)
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "llama-3.3-70b-versatile"
    llm_api_key: str | None = None
    llm_max_tokens: int = 1024
    # Extraction, not creative writing - keep it near-deterministic.
    llm_temperature: float = 0.2
    llm_timeout_s: float = 30.0
    llm_max_tool_iterations: int = 4

    # ---------- harness ----------
    # Deadline for the whole post-STT pipeline. The orchestrator degrades
    # gracefully rather than overrunning it.
    pipeline_budget_ms: float = 200.0
    grounded_budget_ms: float = 25_000.0
    stage_max_retries: int = 2
    breaker_fail_threshold: int = 4
    breaker_reset_s: float = 30.0

    # ---------- serving ----------
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    telemetry_ring_size: int = 5000

    @property
    def index_dir(self) -> Path:
        return INDEX_DIR

    @property
    def corpus_dir(self) -> Path:
        return CORPUS_DIR


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
