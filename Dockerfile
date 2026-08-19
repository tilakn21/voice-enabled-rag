# CPU-only image. torch comes from the CPU wheel index so we don't ship ~2.5GB
# of unused CUDA libraries.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf \
    TOKENIZERS_PARALLELISM=false \
    EMBED_THREADS=2

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
RUN chmod +x scripts/entrypoint.sh

# Bake the encoder weights in so the first request doesn't wait on a 470MB
# download. This is cheap (~2 min) and always worth doing.
RUN python -c "\
from transformers import AutoModel, AutoTokenizer; \
m='intfloat/multilingual-e5-small'; \
AutoTokenizer.from_pretrained(m); AutoModel.from_pretrained(m)"

# Corpus + index. Baking them in makes boot instant, but it downloads ~460MB
# and takes ~5-15 min depending on MAX_QUERIES, which some hosts' build
# timeouts won't tolerate. Set BUILD_INDEX=false to skip it — entrypoint.sh
# then builds on first boot instead.
ARG BUILD_INDEX=true
ARG MAX_QUERIES=400
ARG LANGUAGES=hin
RUN if [ "$BUILD_INDEX" = "true" ]; then \
      python scripts/prepare_corpus.py --languages "$LANGUAGES" --max-queries "$MAX_QUERIES" && \
      python scripts/build_index.py && \
      rm -rf /app/.hf/datasets /root/.cache/huggingface/datasets ; \
    else \
      echo "skipping build-time index; entrypoint will build on first boot" ; \
    fi

EXPOSE 8000
# start-period is generous because a boot-time index build can take minutes.
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
  CMD curl -fsS "http://localhost:${PORT:-8000}/v1/health" || exit 1

CMD ["./scripts/entrypoint.sh"]
