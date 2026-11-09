#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LETTER_DIR="${PWD}"
CUSTOM_PROMPT=""
FORCE="0"
DRY_RUN="0"
LETTER_DIR_SET="0"

usage() {
  cat <<'USAGE'
Usage:
  ./generate-crucible-cover.sh [LETTER_DIRECTORY] [--prompt "scene direction"] [--force] [--dry-run]

If LETTER_DIRECTORY is omitted, the current directory must contain index.md.
The optional scene direction is combined with the article context and shared pulp style.
The script saves cover.png and adds relative cover frontmatter to index.md.
Existing covers are not replaced unless --force is given.
--dry-run prints the final prompt without calling OpenRouter or writing files.
USAGE
}

fail() {
  printf 'Error: %s\n' "$1" >&2
  exit 1
}

while (($#)); do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --prompt)
      (($# >= 2)) || fail "--prompt needs a value"
      [[ -z "$CUSTOM_PROMPT" ]] || fail "provide --prompt only once"
      CUSTOM_PROMPT="$2"
      shift 2
      ;;
    --force)
      FORCE="1"
      shift
      ;;
    --dry-run)
      DRY_RUN="1"
      shift
      ;;
    --*)
      fail "unknown option: $1"
      ;;
    *)
      if [[ "$LETTER_DIR_SET" == "0" ]]; then
        LETTER_DIR="$1"
        LETTER_DIR_SET="1"
      elif [[ -z "$CUSTOM_PROMPT" ]]; then
        CUSTOM_PROMPT="$1"
      else
        fail "unexpected extra argument: $1"
      fi
      shift
      ;;
  esac
done

ENV_FILE="$REPO_ROOT/.env"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # The caller-owned root .env is intentionally sourced only in this process.
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
elif [[ "$DRY_RUN" != "1" ]]; then
  fail "missing $ENV_FILE; add OPENROUTER_API_KEY there (do not commit the file)"
fi

export CRUCIBLE_LETTER_DIR="$LETTER_DIR"
export CRUCIBLE_COVER_PROMPT="$CUSTOM_PROMPT"
export CRUCIBLE_COVER_FORCE="$FORCE"
export CRUCIBLE_COVER_DRY_RUN="$DRY_RUN"

python3 - <<'PY'
import base64
import json
import os
import re
import struct
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path


STYLE_PROMPT = """Generate a wide 3:2 illustration. The illustration features a vintage style art with strong influences from the pulp magazine era characterized by:
1. Halftone Dot Pattern: The entire image uses a halftone dot effect, which gives it a textured, printed look, typical of old comic books and newspapers.
2. Bold, Black Linework: The lines outlining the objects, symbols, or elements in the scene are thick and prominent, providing a strong contrast against the softer colors and making the details stand out.
3. Limited, Muted Color Palette: The colors are desaturated and muted, mostly earthy tones like browns, beiges, and faded blues, which create a nostalgic and aged feel.
4. Cross-hatching and Shading: The shading on objects, symbols, and the background is done using cross-hatching, enhancing the vintage look and adding depth to the illustration.
5. Dynamic Composition: The arrangement of objects or symbols is energetic and slightly exaggerated, emphasizing movement or impact in the scene. Shapes and forms are expressive, capturing a moment of tension or drama.
6. Stylized Clouds and Background Elements: The clouds, abstract symbols, or environmental elements in the background are simplified and stylized, with the same halftone texture applied, ensuring consistency with the overall retro aesthetic.

These characteristics combine to give the image a distinct, old-timey, illustrative feel, akin to something from a 1940s-1950s comic book or an early 20th-century advertisement. No words in the image and no captions. Also exclude typography, labels, signage, letters, numbers, logos, or any readable marks. Avoid photography, modern photorealism, neon cyberpunk, solarpunk, heavy noir, intricate engraving, gloomy atmosphere, realistic environments, and UI mockups. Keep one central focal point, a simple secondary background, clear subject separation, bright readable lighting, and mostly midtones."""


def stop(message: str, code: int = 1) -> None:
    print(f"Error: {message}", file=sys.stderr)
    raise SystemExit(code)


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            name = stream.name
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def parse_frontmatter(text: str) -> tuple[str, str, str]:
    match = re.match(r"\A---\s*\n(.*?)\n---\s*\n", text, re.S)
    if not match:
        stop("index.md must start with YAML frontmatter delimited by ---")
    return match.group(1), text[match.end():], match.group(0)


def clean_excerpt(text: str, limit: int = 1400) -> str:
    paragraphs = re.split(r"\n\s*\n", text.strip())
    kept = []
    for paragraph in paragraphs:
        paragraph = re.sub(r"^#{1,6}\s+", "", paragraph.strip())
        paragraph = re.sub(r"\*\*([^*]+)\*\*|\*([^*]+)\*|`([^`]+)`", lambda m: next(v for v in m.groups() if v is not None), paragraph)
        paragraph = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", paragraph)
        if paragraph and not paragraph.startswith("{{<"):
            kept.append(paragraph)
        if len(" ".join(kept)) >= limit:
            break
    return " ".join(kept)[:limit]


def image_frontmatter(frontmatter: str) -> str:
    cover = re.search(r"(?ms)^cover:\s*\n((?:^[ \t]+.*(?:\n|$))*)", frontmatter)
    if cover:
        block = cover.group(1)
        image = re.search(r'(?m)^\s+image:\s*["\']?([^"\'\s]+)', block)
        relative = re.search(r"(?m)^\s+relative:\s*(true|false)\s*$", block)
        if not image or image.group(1) != "cover.png" or not relative or relative.group(1) != "true":
            stop("index.md has an incompatible cover declaration; refusing to replace it")
        return frontmatter

    draft = re.search(r"(?m)^draft:\s*false\s*$", frontmatter)
    if not draft:
        stop("index.md needs a `draft: false` field before cover metadata can be added")
    addition = 'cover:\n  image: "cover.png"\n  relative: true\n\n'
    return frontmatter[:draft.start()] + addition + frontmatter[draft.start():]


letter_arg = os.environ.get("CRUCIBLE_LETTER_DIR", ".")
letter_path = Path(letter_arg).expanduser()
if letter_path.is_file():
    if letter_path.name != "index.md":
        stop("letter path must be a directory or its index.md")
    letter_path = letter_path.parent
try:
    letter_path = letter_path.resolve(strict=True)
except FileNotFoundError:
    stop(f"letter directory does not exist: {letter_arg}")
if not letter_path.is_dir():
    stop("letter path must be a directory or its index.md")
index_path = letter_path / "index.md"
if not index_path.is_file():
    stop(f"letter directory does not contain index.md: {letter_path}")

original_index = index_path.read_text(encoding="utf-8")
frontmatter, body, delimiter = parse_frontmatter(original_index)
title_match = re.search(r"(?m)^title:\s*(.*)\s*$", frontmatter)
if not title_match:
    stop("index.md frontmatter is missing title")
title = title_match.group(1).strip().strip("\"'")
excerpt = clean_excerpt(body)
custom_prompt = os.environ.get("CRUCIBLE_COVER_PROMPT", "").strip()
article_context = f"Article title: {title}\nArticle context: {excerpt}"
scene = custom_prompt or "Choose one clear, article-specific object or action from the title and context. Do not make a generic cover."
combined_prompt = f"{STYLE_PROMPT}\n\n{article_context}\n\nScene direction: {scene}\n\nKeep the article-specific scene simple and readable. Do not include any on-image text or marks."

if os.environ.get("CRUCIBLE_COVER_DRY_RUN") == "1":
    print(combined_prompt)
    raise SystemExit(0)

api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
if not api_key:
    stop("OPENROUTER_API_KEY is missing from the repository-root .env")
output_path = letter_path / "cover.png"
force = os.environ.get("CRUCIBLE_COVER_FORCE") == "1"
if output_path.exists() and not force:
    stop(f"{output_path} already exists; pass --force to replace it")
new_frontmatter = image_frontmatter(frontmatter)
new_index = original_index if new_frontmatter == frontmatter else f"---\n{new_frontmatter}\n---\n{body}"

endpoint = "https://openrouter.ai/api/v1/images"
payload = {
    "model": "openai/gpt-image-1",
    "prompt": combined_prompt,
    "aspect_ratio": "3:2",
    "quality": "high",
    "n": 1,
}
request = urllib.request.Request(
    endpoint,
    data=json.dumps(payload).encode("utf-8"),
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://3fmindset.com/",
        "X-Title": "3F Mindset Crucible Cover Generator",
    },
    method="POST",
)
try:
    with urllib.request.urlopen(request, timeout=900) as response:
        result = json.loads(response.read())
except urllib.error.HTTPError as exc:
    detail = ""
    try:
        error_payload = json.loads(exc.read())
        detail = (error_payload.get("error") or {}).get("message", "")
    except Exception:
        pass
    stop(f"OpenRouter image request failed with HTTP {exc.code}" + (f": {detail[:300]}" if detail else ""))
except (urllib.error.URLError, TimeoutError) as exc:
    stop(f"OpenRouter image request failed: {type(exc).__name__}")

items = result.get("data") or []
if not items or not items[0].get("b64_json"):
    stop("OpenRouter response did not contain a base64 image")
try:
    image = base64.b64decode(items[0]["b64_json"], validate=True)
except (ValueError, TypeError):
    stop("OpenRouter returned invalid image data")
if not image.startswith(b"\x89PNG\r\n\x1a\n") or len(image) < 24:
    stop("OpenRouter response was not a valid PNG")
width, height = struct.unpack(">II", image[16:24])
if not width or not height or abs(width / height - 1.5) > 0.03:
    stop(f"expected a 3:2 PNG, got {width}x{height}; no files were written")

original_image = output_path.read_bytes() if output_path.exists() else None
try:
    atomic_write(output_path, image)
    if new_index != original_index:
        atomic_write(index_path, new_index.encode("utf-8"))
except Exception:
    if original_image is None:
        output_path.unlink(missing_ok=True)
    else:
        atomic_write(output_path, original_image)
    raise

usage = result.get("usage") or {}
cost = usage.get("cost")
print(f"Generated {output_path} ({width}x{height}, PNG) with {payload['model']} via OpenRouter.")
if cost is not None:
    print(f"Provider-reported image cost: ${float(cost):.4f}")
if new_index != original_index:
    print("Added relative cover metadata to index.md.")
PY
