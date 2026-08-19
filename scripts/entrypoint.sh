#!/usr/bin/env sh
# Container entrypoint.
#
# Building the corpus + index at *image build* time is preferable (the first
# request pays nothing), but some hosts cap build minutes or run the build in a
# sandbox without network. So this checks at boot and builds if the index is
# missing, which means the same image works whether or not `BUILD_INDEX=true`
# succeeded at build time.
#
# Honours $PORT so it drops straight into Render / Railway / Fly / HF Spaces,
# which all inject it.
set -e

PORT="${PORT:-8000}"
INDEX_DIR="${INDEX_DIR:-/app/data/index}"
LANGUAGES="${LANGUAGES:-hin}"
MAX_QUERIES="${MAX_QUERIES:-400}"

if [ ! -d "$INDEX_DIR" ] || [ -z "$(ls -A "$INDEX_DIR" 2>/dev/null)" ]; then
  echo "[entrypoint] no index at $INDEX_DIR — building now"
  echo "[entrypoint] languages=$LANGUAGES max_queries=$MAX_QUERIES"
  echo "[entrypoint] first run downloads ~460MB from Hugging Face; expect several minutes"
  python scripts/prepare_corpus.py --languages "$LANGUAGES" --max-queries "$MAX_QUERIES"
  python scripts/build_index.py
  echo "[entrypoint] index ready"
else
  echo "[entrypoint] using prebuilt index at $INDEX_DIR"
fi

echo "[entrypoint] serving on 0.0.0.0:$PORT"
exec python -m uvicorn voicerag.app:app --host 0.0.0.0 --port "$PORT"
