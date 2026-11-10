#!/usr/bin/env bash
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

PORT="${PORT:-8080}"
BIND="${BIND:-0.0.0.0}"
PREVIEW_SCHEME="${PREVIEW_SCHEME:-http}"
PREVIEW_HOST="${PREVIEW_HOST:-titan}"
BASE_URL="${BASE_URL:-${PREVIEW_SCHEME}://${PREVIEW_HOST}:${PORT}/}"

hugo server \
    --bind "$BIND" \
    --port "$PORT" \
    --baseURL "$BASE_URL" \
    --poll 100ms \
    --buildFuture \
    --buildDrafts \
    --disableFastRender \
    --printPathWarnings \
    --minify
