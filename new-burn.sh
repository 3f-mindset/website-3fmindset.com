#!/usr/bin/env bash
set -euo pipefail

if [[ "$(date +%u)" -eq 5 ]]; then
  DEFAULT_DATE="$(date +%F)"
else
  DEFAULT_DATE="$(date -d 'next friday' +%F)"
fi

exec "$(dirname "${BASH_SOURCE[0]}")/scripts/gutenberg-generate.sh" "$DEFAULT_DATE" "$@"
