#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
TEMP="$(mktemp -d)"
trap 'rm -rf "$TEMP"' EXIT
mkdir -p "$TEMP/site/scripts" "$TEMP/site/tools/gutenberg" "$TEMP/site/content/letters" "$TEMP/bin" "$TEMP/sources"
cp "$ROOT/new-burn.sh" "$ROOT/new-letter.sh" "$ROOT/scripts/gutenberg-generate.sh" "$TEMP/site/scripts/"
cp "$ROOT/new-burn.sh" "$ROOT/new-letter.sh" "$TEMP/site/"
touch "$TEMP/sources/source.md"
cat > "$TEMP/bin/uv" <<'MOCK'
#!/usr/bin/env bash
printf '%s\n' "$@" > "$MOCK_ARGS"
printf '%s\n' "$PWD" > "$MOCK_CWD"
printf '%s\n' "$PYTHONPATH" > "$MOCK_PYTHONPATH"
MOCK
chmod +x "$TEMP/bin/uv"
export PATH="$TEMP/bin:$PATH" MOCK_ARGS="$TEMP/args" MOCK_CWD="$TEMP/cwd" MOCK_PYTHONPATH="$TEMP/pythonpath"
export OPENROUTER_API_KEY=test-openrouter-key OPENAI_API_KEY=test-openai-key
cd "$TEMP/site"

if ./new-letter.sh >/dev/null 2>&1; then
  printf 'missing required arguments unexpectedly succeeded\n' >&2
  exit 1
fi
[[ ! -e content/letters/$(date +%F)-example ]]

./new-letter.sh --source "$TEMP/sources/source.md" --brief 'A focused brief' --title Example \
  --date 2026-12-03 --slug example --provider openrouter --model text-model \
  --image-provider openai --image-model image-model
[[ "$(<"$MOCK_CWD")" == "$TEMP/site" ]]

assert_arg() {
  if ! python3 - "$MOCK_ARGS" "$1" "$2" <<'PY'
import pathlib, sys
args = pathlib.Path(sys.argv[1]).read_text().splitlines()
try:
    index = args.index(sys.argv[2])
except ValueError:
    raise SystemExit(1)
raise SystemExit(0 if args[index + 1] == sys.argv[3] else 1)
PY
  then
    printf 'missing or incorrect argument: %s %s\n' "$1" "$2" >&2
    exit 1
  fi
}
python3 - "$MOCK_ARGS" <<'PY'
import pathlib, sys
args = pathlib.Path(sys.argv[1]).read_text().splitlines()
assert args[:10] == ['run', '--no-project', '--with', 'httpx>=0.27', '--with', 'PyYAML>=6', '--', 'python', '-m', 'gutenberg_ralph.interface'], args
assert args[10] == 'generate', args
assert '--batch' not in args, args
PY
python3 - "$MOCK_PYTHONPATH" <<'PY'
import pathlib, sys
assert pathlib.Path(sys.argv[1]).read_text().strip().endswith("tools/gutenberg/ralph")
PY
assert_arg --output content/letters/2026-12-03-example
assert_arg --source "$TEMP/sources/source.md"
assert_arg --provider openrouter
assert_arg --model text-model
assert_arg --image-provider openai
assert_arg --image-model image-model
[[ ! -e content/letters/2026-12-03-example ]]

mkdir content/letters/2026-12-03-existing
if ./new-letter.sh --source "$TEMP/sources/source.md" --brief brief --title Existing \
  --date 2026-12-03 --slug existing --provider openrouter --model text-model \
  --image-provider openai --image-model image-model >/dev/null 2>&1; then
  printf 'existing destination unexpectedly succeeded\n' >&2
  exit 1
fi
[[ -d content/letters/2026-12-03-existing ]]

./new-letter.sh --source "$TEMP/sources/source.md" --brief brief --title Batched \
  --date 2026-12-04 --slug batched --provider openrouter --model text-model \
  --image-provider openai --image-model image-model --batch
python3 - "$MOCK_ARGS" <<'PY'
import pathlib, sys
assert '--batch' in pathlib.Path(sys.argv[1]).read_text().splitlines()
PY

if ./new-letter.sh --source "$TEMP/sources/source.md" --brief brief --title "Invalid Batch" \
  --date 2026-12-05 --slug invalid-batch --provider openai-compatible --model local-model \
  --image-provider openai-compatible --image-model local-image --batch >/dev/null 2>&1; then
  printf 'batch with non-OpenRouter provider unexpectedly succeeded\n' >&2
  exit 1
fi
[[ ! -e content/letters/2026-12-05-invalid-batch ]]

printf 'wrapper mock tests passed\n'
