#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-8080}"
LOG_FILE="${LOG_FILE:-/tmp/3fmindset-hugo-preview.log}"
PID_FILE="${PID_FILE:-/tmp/3fmindset-hugo-preview.pid}"
PREVIEW_SCHEME="${PREVIEW_SCHEME:-http}"
PREVIEW_HOST="${PREVIEW_HOST:-titan}"
BASE_URL="${BASE_URL:-${PREVIEW_SCHEME}://${PREVIEW_HOST}:${PORT}/}"
RESTART="${RESTART:-0}"

cd "$REPO_ROOT"

if [[ -f "$PID_FILE" ]]; then
  existing_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "${existing_pid}" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    if [[ "$RESTART" == "1" ]]; then
      kill "$existing_pid"
      rm -f "$PID_FILE"
      sleep 2
    else
      echo "Hugo preview already running on PID $existing_pid"
      echo "Log: $LOG_FILE"
      exit 0
    fi
  fi
fi

if [[ "$RESTART" == "1" ]]; then
  pkill -f "hugo server" 2>/dev/null || true
  sleep 2
fi

# Force a fresh recursive file scan before startup so renamed or newly created
# content paths are visible on the first full rebuild.
find assets content layouts static themes -type f >/dev/null 2>&1 || true

# Start with a fresh log for each run so old rename events do not muddy diagnosis.
: >"$LOG_FILE"

nohup env PORT="$PORT" PREVIEW_SCHEME="$PREVIEW_SCHEME" PREVIEW_HOST="$PREVIEW_HOST" BASE_URL="$BASE_URL" bash ./preview.sh >"$LOG_FILE" 2>&1 &
echo $! >"$PID_FILE"

echo "Started Hugo preview"
echo "PID: $(cat "$PID_FILE")"
echo "Port: $PORT"
echo "Base URL: $BASE_URL"
echo "Log: $LOG_FILE"
