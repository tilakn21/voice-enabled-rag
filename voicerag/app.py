"""FastAPI service.

Endpoints:
    POST /v1/query        text question -> grounded answer
    POST /v1/voice/query  audio upload  -> transcript + grounded answer
    GET  /v1/stats        live P50/P70/P90/P100 latency + index + breaker state
    GET  /v1/health       readiness
    GET  /                the demo UI

The heavy objects (encoder, indices) load once at startup, not per request —
loading a 118M-param model inside the request path would make the latency
numbers meaningless.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from .config import PROJECT_ROOT, get_settings
from .embeddings import Encoder
from .lab import ChunkingLab
from .pipeline import RagService
from .retrieval import Retriever
from .schemas import QueryRequest, QueryResponse
from .stt import build_stt
from .telemetry import Timer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)
logger = logging.getLogger("voicerag")

WEB_DIR = PROJECT_ROOT / "web"

_state: dict = {"service": None, "lab": None, "error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    try:
        logger.info("loading encoder %s…", settings.embed_model)
        t0 = time.perf_counter()
        encoder = Encoder(
            settings.embed_model,
            max_tokens=settings.embed_max_tokens,
            quantize=settings.embed_quantize,
            threads=settings.embed_threads,
            query_prefix=settings.embed_query_prefix,
            passage_prefix=settings.embed_passage_prefix,
        )
        logger.info("encoder ready in %.1fs", time.perf_counter() - t0)

        logger.info("loading indices from %s…", settings.index_dir)
        retriever = Retriever.load(settings, encoder)

        # Warm the model so request #1 isn't an outlier in the percentiles.
        encoder.warmup(3)
        retriever.retrieve("warmup", top_k=3)

        service = RagService(settings, encoder, retriever, stt=build_stt(settings))
        _state["service"] = service
        # Optional: the all-8-strategy comparison index. Absent in a minimal
        # deploy, in which case the lab tab reports itself unavailable.
        _state["lab"] = ChunkingLab.load(settings, encoder)
        logger.info(
            "ready | %d chunks across %s | voice=%s | grounded=%s",
            retriever.n_chunks,
            list(retriever.shards),
            service.voice_enabled,
            service.grounded_enabled,
        )
    except Exception as exc:  # noqa: BLE001
        _state["error"] = f"{type(exc).__name__}: {exc}"
        logger.exception("startup failed — service will report unhealthy")
    yield
    _state.clear()


app = FastAPI(
    title="Voice-Enabled RAG",
    description="Speech -> retrieval -> grounded answer over MSMARCO-XI.",
    version="1.0.0",
    lifespan=lifespan,
)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_service() -> RagService:
    service = _state.get("service")
    if service is None:
        raise HTTPException(
            status_code=503,
            detail=(
                _state.get("error")
                or "Service still starting. If this persists, build the index: "
                "python scripts/prepare_corpus.py && python scripts/build_index.py"
            ),
        )
    return service


# --------------------------------------------------------------------------
@app.get("/v1/health")
async def health():
    service = _state.get("service")
    if service is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": _state.get("error", "starting")},
        )
    lab = _state.get("lab")
    return {
        "status": "ok",
        "chunks": service.retriever.n_chunks,
        "strategies": list(service.retriever.shards),
        "voice_enabled": service.voice_enabled,
        "grounded_enabled": service.grounded_enabled,
        "llm_model": service.settings.llm_model if service.grounded_enabled else None,
        "lab_enabled": lab is not None,
        "lab_strategies": lab.info["n_strategies"] if lab else 0,
    }


@app.get("/v1/stats")
async def stats():
    return get_service().stats()


@app.post("/v1/stats/reset")
async def stats_reset():
    get_service().telemetry.reset()
    return {"status": "reset"}


@app.post("/v1/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    service = get_service()
    return await service.answer(
        request.query,
        mode=request.mode,
        lang=request.lang,
        top_k=request.top_k,
        budget_ms=request.budget_ms,
    )


# --------------------------------------------------------------------------
# Chunking Lab — the comparison the README can only describe
# --------------------------------------------------------------------------
def get_lab() -> ChunkingLab:
    lab = _state.get("lab")
    if lab is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Chunking lab index not built. Run:\n"
                "  python scripts/build_lab_index.py"
            ),
        )
    return lab


@app.get("/v1/lab/info")
async def lab_info():
    lab = _state.get("lab")
    if lab is None:
        return {"available": False, "reason": "lab index not built"}
    return lab.info


@app.post("/v1/lab/race")
async def lab_race(request: QueryRequest):
    """Run one query through every chunking strategy and return each result."""
    return get_lab().race(request.query, top_k=request.top_k or 3)


@app.post("/v1/voice/query", response_model=QueryResponse)
async def voice_query(
    audio: UploadFile = File(..., description="Recorded question (wav/webm/mp3/ogg)"),
    mode: str | None = Form(default=None),
    lang: str | None = Form(default=None),
    top_k: int | None = Form(default=None),
):
    service = get_service()
    if not service.voice_enabled:
        raise HTTPException(
            status_code=503,
            detail=(
                "No STT provider configured. Set SARVAM_API_KEY (or "
                "ELEVENLABS_API_KEY with STT_PROVIDER=elevenlabs), or use "
                "/v1/query with text."
            ),
        )

    payload = await audio.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Empty audio upload.")

    timer = Timer()
    try:
        transcript = await service.transcribe(
            payload, audio.filename or "audio.wav", audio.content_type or "audio/wav", timer
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("STT failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Speech-to-text failed: {exc}") from exc

    stt_ms = next((s.duration_ms for s in timer.spans if s.name == "stt"), None)

    return await service.answer(
        transcript.text,
        mode=mode,
        lang=lang,
        top_k=top_k,
        timer=timer,
        transcript=transcript.text,
        detected_language=transcript.language,
        stt_ms=stt_ms,
    )


# --------------------------------------------------------------------------
@app.get("/")
async def index_page():
    page = WEB_DIR / "index.html"
    if not page.exists():
        return JSONResponse({"service": "voice-rag", "docs": "/docs"})
    return FileResponse(page)


def main() -> None:
    import uvicorn

    cfg = get_settings()
    uvicorn.run("voicerag.app:app", host=cfg.host, port=cfg.port, log_level="info")


if __name__ == "__main__":
    main()
