#!/usr/bin/env bash
# One-command setup + run.
#   ./run.sh setup   -> venv, deps, corpus, index
#   ./run.sh serve   -> start the API + UI on :8000
#   ./run.sh bench   -> latency + retrieval + guardrail calibration
#   ./run.sh test    -> smoke test
set -euo pipefail

cd "$(dirname "$0")"
PY=".venv/bin/python"
PIP=".venv/bin/pip"

ensure_venv() {
  if [ ! -x "$PY" ]; then
    echo "==> creating .venv"
    python3 -m venv .venv
    "$PIP" install --upgrade pip
  fi
}

case "${1:-serve}" in
  setup)
    ensure_venv
    echo "==> installing dependencies"
    "$PIP" install -r requirements.txt
    echo "==> preparing corpus (downloads ~460MB on first run)"
    "$PY" scripts/prepare_corpus.py --languages "${LANGUAGES:-hin}" --max-queries "${MAX_QUERIES:-1200}"
    echo "==> building indices"
    "$PY" scripts/build_index.py
    echo "==> done. Start with: ./run.sh serve"
    ;;
  serve)
    ensure_venv
    if [ ! -d data/index ] || [ -z "$(ls -A data/index 2>/dev/null)" ]; then
      echo "No index found. Run: ./run.sh setup" >&2
      exit 1
    fi
    PORT="${PORT:-8000}"
    # Starting a second server while an old one still holds the port is the
    # classic "I redeployed but still see the old UI" trap: the new process
    # fails to bind and the browser keeps talking to the stale one.
    if pgrep -f "uvicorn voicerag.app" > /dev/null 2>&1; then
      echo "==> stopping stale voice-rag server(s):"
      pgrep -lf "uvicorn voicerag.app" | sed 's/^/    /'
      pkill -f "uvicorn voicerag.app" || true
      sleep 1
    fi
    if command -v lsof > /dev/null 2>&1 && lsof -i ":$PORT" -sTCP:LISTEN > /dev/null 2>&1; then
      echo "Port $PORT is in use by something else:" >&2
      lsof -i ":$PORT" -sTCP:LISTEN >&2
      echo "Free it, or run: PORT=8010 ./run.sh serve" >&2
      exit 1
    fi
    echo "==> http://localhost:$PORT   (hard-refresh the tab: Cmd/Ctrl+Shift+R)"
    exec "$PY" -m uvicorn voicerag.app:app --host "${HOST:-0.0.0.0}" --port "$PORT"
    ;;
  bench)
    ensure_venv
    "$PY" scripts/bench_latency.py --n "${N:-300}"
    "$PY" scripts/bench_retrieval.py --queries "${QUERIES:-300}" --strategies all
    "$PY" scripts/calibrate_guardrails.py --n "${N:-250}"
    ;;
  test)
    ensure_venv
    "$PY" tests/test_smoke.py
    ;;
  *)
    echo "usage: $0 {setup|serve|bench|test}" >&2
    exit 1
    ;;
esac
