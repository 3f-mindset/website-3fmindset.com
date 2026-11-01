#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
TEMP="$(mktemp -d)"
trap 'rm -rf "$TEMP"' EXIT
mkdir -p "$TEMP/site/scripts" "$TEMP/site/tools/gutenberg" "$TEMP/site/content/letters" "$TEMP/bin" "$TEMP/sources"
cp "$ROOT/new-burn.sh" "$ROOT/new-letter.sh" "$ROOT/scripts/gutenberg-generate.sh" "$TEMP/site/scripts/"
cp "$ROOT/new-burn.sh" "$ROOT/new-letter.sh" "$TEMP/site/"
touch "$TEMP/sources/source.md"
printf 'OPENROUTER_API_KEY=dotenv-openrouter-key\nOPENAI_API_KEY=dotenv-openai-key\nGUTENBERG_PROVIDER=openrouter\nGUTENBERG_MODEL=dotenv-text-model\nGUTENBERG_IMAGE_PROVIDER=openai\nGUTENBERG_IMAGE_MODEL=dotenv-image-model\nGUTENBERG_BATCH=true\n' > "$TEMP/site/.env"
cat > "$TEMP/bin/uv" <<'MOCK'
#!/usr/bin/env bash
printf '%s\n' "$@" > "$MOCK_ARGS"
printf '%s\n' "$PWD" > "$MOCK_CWD"
printf '%s\n' "$PYTHONPATH" > "$MOCK_PYTHONPATH"
MOCK
chmod +x "$TEMP/bin/uv"
export PATH="$TEMP/bin:$PATH" MOCK_ARGS="$TEMP/args" MOCK_CWD="$TEMP/cwd" MOCK_PYTHONPATH="$TEMP/pythonpath"
unset OPENROUTER_API_KEY OPENAI_API_KEY
cd "$TEMP/site"

if ./new-letter.sh >/dev/null 2>&1; then
  printf 'missing required arguments unexpectedly succeeded\n' >&2
  exit 1
fi
[[ ! -e content/letters/$(date +%F)-example ]]

./new-letter.sh --source "$TEMP/sources/source.md" --brief 'A focused brief' --title Example \
  --date 2026-12-03 --slug example
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
assert args[:12] == ['run', '--no-project', '--with', 'httpx>=0.27', '--with', 'PyYAML>=6', '--with', 'python-dotenv>=1.0,<2', '--', 'python', '-m', 'gutenberg_ralph.interface'], args
assert args[12] == 'generate', args
assert '--batch' not in args, args
assert '--no-batch' not in args, args
assert not any(option in args for option in ('--provider', '--model', '--provider-url', '--image-provider', '--image-provider-url', '--image-model')), args
PY
python3 - "$MOCK_PYTHONPATH" <<'PY'
import pathlib, sys
assert pathlib.Path(sys.argv[1]).read_text().strip().endswith("tools/gutenberg/ralph")
PY
assert_arg --output content/letters/2026-12-03-example
assert_arg --source "$TEMP/sources/source.md"
[[ ! -e content/letters/2026-12-03-example ]]

python3 - "$ROOT" <<'PY'
import subprocess, sys
revision = subprocess.check_output(['git', '-C', sys.argv[1] + '/tools/gutenberg', 'rev-parse', 'HEAD'], text=True).strip()
assert revision == '36babafd4efab0eb91ee1d2538ac3377a198e87e', revision
PY

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
  --provider-url https://text.example --image-provider openai --image-model image-model \
  --image-provider-url https://image.example --batch
python3 - "$MOCK_ARGS" <<'PY'
import pathlib, sys
args = pathlib.Path(sys.argv[1]).read_text().splitlines()
assert '--batch' in args, args
assert '--no-batch' not in args, args
for option, value in [('--provider', 'openrouter'), ('--model', 'text-model'), ('--provider-url', 'https://text.example'), ('--image-provider', 'openai'), ('--image-model', 'image-model'), ('--image-provider-url', 'https://image.example')]:
    assert args[args.index(option) + 1] == value, args
PY

./new-letter.sh --source "$TEMP/sources/source.md" --brief brief --title "Batch Override" \
  --date 2026-12-05 --slug batch-override --no-batch
python3 - "$MOCK_ARGS" <<'PY'
import pathlib, sys
args = pathlib.Path(sys.argv[1]).read_text().splitlines()
assert '--no-batch' in args, args
assert '--batch' not in args, args
PY

printf 'wrapper mock tests passed\n'
