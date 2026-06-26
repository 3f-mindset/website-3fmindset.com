#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/codex-generate-burn-file.sh --format markdown|svg --prompt-file FILE --output FILE [options]

Options:
  --input-file FILE       Optional source/context file appended to the prompt.
  --title TEXT            Optional burn title made available to the prompt.
  --slug TEXT             Optional burn slug made available to the prompt.
  --date YYYY-MM-DD       Optional burn date made available to the prompt.
  --model MODEL           Optional Codex model override.
  --force                 Overwrite an existing output file.
  --dry-run               Print the Codex command and assembled prompt path, but do not call Codex.
  -h, --help              Show this help.

Environment:
  CODEX_BIN               Codex executable. Defaults to "codex".
EOF
}

FORMAT=""
PROMPT_FILE=""
INPUT_FILE=""
OUTPUT_FILE=""
TITLE=""
SLUG=""
DATE_VALUE=""
MODEL=""
FORCE=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --format)
      FORMAT="${2:-}"
      shift 2
      ;;
    --prompt-file)
      PROMPT_FILE="${2:-}"
      shift 2
      ;;
    --input-file)
      INPUT_FILE="${2:-}"
      shift 2
      ;;
    --output)
      OUTPUT_FILE="${2:-}"
      shift 2
      ;;
    --title)
      TITLE="${2:-}"
      shift 2
      ;;
    --slug)
      SLUG="${2:-}"
      shift 2
      ;;
    --date)
      DATE_VALUE="${2:-}"
      shift 2
      ;;
    --model)
      MODEL="${2:-}"
      shift 2
      ;;
    --force)
      FORCE=true
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$FORMAT" != "markdown" && "$FORMAT" != "svg" ]]; then
  echo "--format must be either markdown or svg" >&2
  exit 2
fi

if [[ -z "$PROMPT_FILE" || ! -f "$PROMPT_FILE" ]]; then
  echo "--prompt-file is required and must exist" >&2
  exit 2
fi

if [[ -n "$INPUT_FILE" && ! -f "$INPUT_FILE" ]]; then
  echo "--input-file must exist when provided" >&2
  exit 2
fi

if [[ -z "$OUTPUT_FILE" ]]; then
  echo "--output is required" >&2
  exit 2
fi

if [[ -e "$OUTPUT_FILE" && "$FORCE" != true ]]; then
  echo "Output exists. Use --force to overwrite: $OUTPUT_FILE" >&2
  exit 1
fi

if [[ -d "$OUTPUT_FILE" ]]; then
  echo "Output path is a directory: $OUTPUT_FILE" >&2
  exit 1
fi

CODEX_BIN="${CODEX_BIN:-codex}"
if ! command -v "$CODEX_BIN" >/dev/null 2>&1; then
  echo "Codex CLI not found: $CODEX_BIN" >&2
  exit 127
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

ASSEMBLED_PROMPT="$TMP_DIR/prompt.md"
RAW_OUTPUT="$TMP_DIR/last-message.txt"
LOG_FILE="$TMP_DIR/codex.log"

{
  cat <<EOF
You are generating one file for the 3F Mindset SteadyBurn content pipeline.

Return only the requested file contents.
Do not include reasoning, explanation, status text, Markdown fences, XML fences, or surrounding commentary.
Do not edit files or run commands. Produce final text only.

Output format: $FORMAT
Title: $TITLE
Slug: $SLUG
Date: $DATE_VALUE

EOF

  if [[ "$FORMAT" == "svg" ]]; then
    cat <<'EOF'
SVG constraints:
- Return a complete SVG XML document starting with <svg.
- Do not wrap the SVG in a Markdown code block.
- Do not include prose before or after the XML.

EOF
  else
    cat <<'EOF'
Markdown constraints:
- Return Markdown only.
- Do not wrap the Markdown in a code block.
- Do not include prose about the task.

EOF
  fi

  cat <<EOF
Prompt template:
--- BEGIN PROMPT TEMPLATE ---
EOF
  cat "$PROMPT_FILE"
  cat <<EOF

--- END PROMPT TEMPLATE ---
EOF

  if [[ -n "$INPUT_FILE" ]]; then
    cat <<EOF

Input file: $INPUT_FILE
--- BEGIN INPUT FILE ---
EOF
    cat "$INPUT_FILE"
    cat <<'EOF'

--- END INPUT FILE ---
EOF
  fi
} > "$ASSEMBLED_PROMPT"

CODEX_ARGS=(
  --ask-for-approval never
  exec
  --cd "$(pwd)"
  --sandbox read-only
  --ephemeral
  --output-last-message "$RAW_OUTPUT"
)

if [[ -n "$MODEL" ]]; then
  CODEX_ARGS+=(--model "$MODEL")
fi

CODEX_ARGS+=(-)

if [[ "$DRY_RUN" == true ]]; then
  echo "Dry run. Assembled prompt: $ASSEMBLED_PROMPT"
  printf 'Command: %q' "$CODEX_BIN"
  printf ' %q' "${CODEX_ARGS[@]}"
  printf '\n'
  exit 0
fi

if ! "$CODEX_BIN" "${CODEX_ARGS[@]}" < "$ASSEMBLED_PROMPT" > "$LOG_FILE" 2>&1; then
  echo "Codex command failed. Last log lines:" >&2
  tail -n 40 "$LOG_FILE" >&2 || true
  exit 1
fi

if [[ ! -s "$RAW_OUTPUT" ]]; then
  echo "Codex returned an empty response. Log: $LOG_FILE" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_FILE")"
cp "$RAW_OUTPUT" "$OUTPUT_FILE"

if [[ "$FORMAT" == "svg" ]] && ! sed -n '1{/^[[:space:]]*<svg[[:space:]>]/q 0; q 1}' "$OUTPUT_FILE"; then
  echo "Generated SVG does not start with <svg: $OUTPUT_FILE" >&2
  exit 1
fi

if sed -n '1{/^[[:space:]]*```/q 0; q 1}' "$OUTPUT_FILE"; then
  echo "Generated output starts with a Markdown fence: $OUTPUT_FILE" >&2
  exit 1
fi

echo "Wrote: $OUTPUT_FILE"
