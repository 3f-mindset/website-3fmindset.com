#!/usr/bin/env bash
set -euo pipefail

root="${1:-.}"
mapfile -d '' offending < <(find "$root/content" -type f -iname '*.png' -print0 2>/dev/null || true)

if ((${#offending[@]})); then
  printf 'PNG files are not allowed under content/; use JPEG instead:\n' >&2
  printf '  %s\n' "${offending[@]}" >&2
  exit 1
fi
