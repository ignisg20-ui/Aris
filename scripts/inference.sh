#!/usr/bin/env bash
# Launch the FastAPI inference server.
set -euo pipefail

export ARIS_MODEL_CONFIG=${ARIS_MODEL_CONFIG:-llm/configs/1b.yaml}
export ARIS_LOG_LEVEL=${ARIS_LOG_LEVEL:-INFO}

exec python -m llm.api.server --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}"
