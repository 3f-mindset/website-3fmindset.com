#!/usr/bin/env bash
set -euo pipefail

# Usage: new-post "Post Title"
# Creates a new directory named YYYY-MM-DD under this script's folder,
# writes an index.md with frontmatter, and writes a cover.png placeholder.

TITLE="${1:-POST TITLE GOES HERE}"
SLUG="$(echo "$TITLE" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g' | sed -E 's/^-|-$//g')"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$(date +%u)" -eq 5 ]]; then
  DATE="$(date +%F)"
else
  DATE="$(date -d 'next friday' +%F)"
fi
DIR="$SCRIPT_DIR/content/essays/$DATE-$SLUG"

mkdir -p "$DIR"

cat > "$DIR/index.md" <<EOF
---
date: $DATE
slug: "$SLUG"
title: "$TITLE"
summary: "SUMMARY GOES HERE"

series:
  - SteadyBurn

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
convert -size 720x480 xc:lightgray -gravity center -pointsize 72 -fill darkgray -annotate 0 "cover image" "$DIR/cover.png"


echo "Created: $DIR/index.md and $DIR/cover.png"
