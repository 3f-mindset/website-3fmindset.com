#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$REPO_ROOT/.env"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # The caller-owned root .env is sourced only in this process.
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

export CRUCIBLE_REPO_ROOT="$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
exec python3 -m crucible_cover.cli "$@"
