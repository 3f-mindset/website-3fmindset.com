#!/usr/bin/env bash
set -euo pipefail

exec "$(dirname "${BASH_SOURCE[0]}")/scripts/gutenberg-generate.sh" "$(date +%F)" "$@"
