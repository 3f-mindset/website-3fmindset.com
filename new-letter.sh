#!/usr/bin/env bash
set -euo pipefail

# Usage: new-letter [--force] "Letter Title"
# Creates a new directory named YYYY-MM-DD under this script's folder,
# writes an index.md with frontmatter, and writes a cover.png placeholder.

FORCE=false
ARGS=()

for ARG in "$@"; do
  case "$ARG" in
    --force)
      FORCE=true
      ;;
    -h|--help)
      echo 'Usage: new-letter [--force] "Letter Title"'
      exit 0
      ;;
    *)
      ARGS+=("$ARG")
      ;;
  esac
done

TITLE="${ARGS[*]:-LETTER TITLE GOES HERE}"
SLUG="$(echo "$TITLE" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g' | sed -E 's/^-|-$//g')"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATE="$(date +%F)"
DIR="$SCRIPT_DIR/content/letters/$DATE-$SLUG"

mkdir -p "$DIR"

write_file() {
  local PATH_TO_WRITE="$1"

  if [[ -e "$PATH_TO_WRITE" && "$FORCE" != true ]]; then
    echo "Skipped existing: $PATH_TO_WRITE"
    return 0
  fi

  if [[ -d "$PATH_TO_WRITE" ]]; then
    echo "Cannot overwrite directory: $PATH_TO_WRITE" >&2
    return 1
  fi

  cat > "$PATH_TO_WRITE"

  if [[ "$FORCE" == true ]]; then
    echo "Wrote: $PATH_TO_WRITE"
  else
    echo "Created: $PATH_TO_WRITE"
  fi
}

write_file "$DIR/index.md" <<EOF
---
date: $DATE
slug: "$SLUG"
title: "$TITLE"
summary: "SUMMARY GOES HERE"


categories: letter


tags:
    - tag1
    - tag2

cover:
  image: "cover.png"
  relative: true

draft: false
---

For a long time, I thought ...
EOF

# Create a simple portrait placeholder image 720x480 if ImageMagick is available,
if [[ -e "$DIR/cover.png" && "$FORCE" != true ]]; then
  echo "Skipped existing: $DIR/cover.png"
elif [[ -d "$DIR/cover.png" ]]; then
  echo "Cannot overwrite directory: $DIR/cover.png" >&2
  exit 1
elif command -v convert >/dev/null 2>&1; then
  convert -size 720x480 xc:lightgray -gravity center -pointsize 72 -fill darkgray -annotate 0 "cover image" "$DIR/cover.png"
  if [[ "$FORCE" == true ]]; then
    echo "Wrote: $DIR/cover.png"
  else
    echo "Created: $DIR/cover.png"
  fi
else
  echo "Skipped cover image: ImageMagick convert is not available" >&2
fi


echo "Directory: $DIR"
if command -v tree >/dev/null 2>&1 && tree "$DIR" >/dev/null 2>&1; then
  tree "$DIR"
else
  find "$DIR" -maxdepth 1 -print | sort
fi
