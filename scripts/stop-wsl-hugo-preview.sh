#!/usr/bin/env bash
set -euo pipefail

PID_FILE="${PID_FILE:-/tmp/3fmindset-hugo-preview.pid}"

if [[ ! -f "$PID_FILE" ]]; then
  echo "No PID file found"
  exit 0
fi

pid="$(cat "$PID_FILE" 2>/dev/null || true)"
if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
  kill "$pid"
  echo "Stopped Hugo preview PID $pid"
else
  echo "Preview process not running"
fi

pkill -f "hugo server" 2>/dev/null || true

rm -f "$PID_FILE"
