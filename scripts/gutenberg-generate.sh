#!/usr/bin/env bash
set -euo pipefail

DEFAULT_DATE="${1:?}"
shift
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
CALLER_DIR="$(pwd -P)"
if [[ "$CALLER_DIR" != "$SCRIPT_DIR" ]]; then
  printf 'Run this wrapper from the website repository root: %s\n' "$SCRIPT_DIR" >&2
  exit 2
fi

usage() {
  printf 'Usage: %s --source PATH [--source PATH ...] --brief TEXT --title TITLE [--date YYYY-MM-DD] [--slug SLUG] [--provider PROVIDER] [--model MODEL] [--provider-url URL] [--image-provider PROVIDER] [--image-model MODEL] [--image-provider-url URL] [--profile PROFILE] [--batch|--no-batch]\n' "${0##*/}" >&2
}

DATE="$DEFAULT_DATE"
SLUG=""
TITLE=""
BRIEF=""
PROVIDER=""
MODEL=""
IMAGE_PROVIDER=""
IMAGE_MODEL=""
PROVIDER_URL=""
IMAGE_PROVIDER_URL=""
PROFILE="steadyburn-profile-v1"
BATCH_MODE=""
SOURCES=()

while (($#)); do
  case "$1" in
    --source)
      (($# >= 2)) || { usage; exit 2; }
      SOURCES+=("$2")
      shift 2
      ;;
    --brief|--title|--date|--slug|--provider|--model|--image-provider|--image-model|--profile|--provider-url|--image-provider-url)
      (($# >= 2)) || { usage; exit 2; }
      option="$1"
      value="$2"
      shift 2
      case "$option" in
        --brief) BRIEF="$value" ;;
        --title) TITLE="$value" ;;
        --date) DATE="$value" ;;
        --slug) SLUG="$value" ;;
        --provider) PROVIDER="$value" ;;
        --model) MODEL="$value" ;;
        --image-provider) IMAGE_PROVIDER="$value" ;;
        --image-model) IMAGE_MODEL="$value" ;;
        --provider-url) PROVIDER_URL="$value" ;;
        --image-provider-url) IMAGE_PROVIDER_URL="$value" ;;
        --profile) PROFILE="$value" ;;
        *) ;;
      esac
      ;;
    --batch)
      BATCH_MODE=true
      shift
      ;;
    --no-batch)
      BATCH_MODE=false
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      usage
      exit 2
      ;;
  esac
done

if ((${#SOURCES[@]} == 0)) || [[ -z "$BRIEF" || -z "$TITLE" ]]; then
  printf 'Source, brief, and title are required.\n' >&2
  usage
  exit 2
fi

if [[ -z "$SLUG" ]]; then
  SLUG="$(printf '%s' "$TITLE" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-|-$//g')"
fi
if [[ ! "$DATE" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || ! date -d "$DATE" +%F >/dev/null 2>&1 || [[ "$(date -d "$DATE" +%F)" != "$DATE" ]]; then
  printf 'Invalid date: %s\n' "$DATE" >&2
  exit 2
fi
if [[ ! "$SLUG" =~ ^[a-z0-9]+(-[a-z0-9]+)*$ ]]; then
  printf 'Invalid slug: %s\n' "$SLUG" >&2
  exit 2
fi

for source in "${SOURCES[@]}"; do
  if [[ ! -f "$source" && ! -d "$source" ]]; then
    printf 'Source file or directory not found: %s\n' "$source" >&2
    exit 2
  fi
done

DESTINATION="content/letters/$DATE-$SLUG"
if [[ -e "$DESTINATION" || -L "$DESTINATION" ]]; then
  printf 'Refusing existing destination: %s\n' "$DESTINATION" >&2
  exit 2
fi

COMMAND=(uv run --no-project --with 'httpx>=0.27' --with 'PyYAML>=6' --with 'python-dotenv>=1.0,<2' --
  python -m gutenberg_ralph.interface generate
  --profile "$PROFILE"
  --output "$DESTINATION"
  --brief "$BRIEF"
  --title "$TITLE"
  --slug "$SLUG"
  --date "$DATE")
for option in provider model provider-url image-provider image-provider-url image-model; do
  case "$option" in
    provider) value="$PROVIDER" ;;
    model) value="$MODEL" ;;
    provider-url) value="$PROVIDER_URL" ;;
    image-provider) value="$IMAGE_PROVIDER" ;;
    image-provider-url) value="$IMAGE_PROVIDER_URL" ;;
    image-model) value="$IMAGE_MODEL" ;;
  esac
  if [[ -n "$value" ]]; then COMMAND+=("--$option" "$value"); fi
done
for source in "${SOURCES[@]}"; do COMMAND+=(--source "$source"); done
if [[ "$BATCH_MODE" == true ]]; then COMMAND+=(--batch); elif [[ "$BATCH_MODE" == false ]]; then COMMAND+=(--no-batch); fi

PYTHONPATH="$SCRIPT_DIR/tools/gutenberg/ralph${PYTHONPATH:+:$PYTHONPATH}" "${COMMAND[@]}"
printf 'Generated %s\nRun the website renderers as needed:\n  uv run scripts/build-the-burn-lesson-pdf.py --markdown-file %s/LESSON.md\n  uv run scripts/build-the-burn-worksheet-pdf.py %s/WORKSHEET.svg\n' "$DESTINATION" "$DESTINATION" "$DESTINATION"
