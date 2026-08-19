# CPU-only image. torch comes from the CPU wheel index so we don't ship ~2.5GB
# of unused CUDA libraries.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

COPY voicerag/ ./voicerag/
COPY scripts/ ./scripts/
COPY web/ ./web/
COPY tests/ ./tests/

# Bake the encoder into the image so the first request doesn't wait on a
# 470MB download.
RUN python -c "\
from transformers import AutoModel, AutoTokenizer; \
m='intfloat/multilingual-e5-small'; \
AutoTokenizer.from_pretrained(m); AutoModel.from_pretrained(m)"

# Build the corpus + index at image build time. Adjust --max-queries to trade
# index size against build time; 1200 queries ≈ 39k passages ≈ 8 min on 4 cores.
ARG MAX_QUERIES=1200
ARG LANGUAGES=hin
RUN python scripts/prepare_corpus.py --languages ${LANGUAGES} --max-queries ${MAX_QUERIES} \
 && python scripts/build_index.py \
 && rm -rf /app/.hf/datasets /root/.cache/huggingface/datasets

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD curl -fsS http://localhost:8000/v1/health || exit 1

CMD ["python", "-m", "uvicorn", "voicerag.app:app", "--host", "0.0.0.0", "--port", "8000"]
