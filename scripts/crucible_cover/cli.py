"""CLI composition root for the Crucible cover tool."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .adapters.filesystem import FileSystemArticleRepository
from .adapters.openrouter import OpenRouterImageGenerator
from .application import GenerateCover
from .domain import CoverRequest, CoverToolError


def _repo_root() -> Path:
    configured = os.environ.get("CRUCIBLE_REPO_ROOT")
    if configured:
        return Path(configured).resolve()
    return Path(__file__).resolve().parents[2]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate-crucible-cover.sh",
        description="Generate one article-specific Crucible pulp-style cover through OpenRouter.",
        epilog=(
            "If LETTER_DIRECTORY is omitted, the current directory must contain index.md. "
            "Use --dry-run to preview the combined prompt without calling OpenRouter."
        ),
    )
    parser.add_argument("letter_directory", nargs="?", help="letter bundle directory or its index.md")
    parser.add_argument("scene_direction", nargs="?", help="optional article-specific scene prompt")
    parser.add_argument("--prompt", dest="prompt_option", help="optional article-specific scene prompt")
    parser.add_argument("--force", action="store_true", help="replace an existing cover.jpg")
    parser.add_argument("--dry-run", action="store_true", help="print the combined prompt without generating an image")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.scene_direction and args.prompt_option:
        parser.error("provide a scene direction either positionally or with --prompt, not both")

    repo_root = _repo_root()
    if not args.dry_run:
        if not (repo_root / ".env").is_file():
            parser.error(f"missing {repo_root / '.env'}; add OPENROUTER_API_KEY there (do not commit the file)")
        api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            parser.error("OPENROUTER_API_KEY is missing from the repository-root .env")
    else:
        api_key = ""

    request = CoverRequest(
        letter_path=Path(args.letter_directory or Path.cwd()),
        scene_direction=args.prompt_option or args.scene_direction or "",
        force=args.force,
        dry_run=args.dry_run,
    )
    articles = FileSystemArticleRepository()
    images = OpenRouterImageGenerator(api_key) if not args.dry_run else None
    use_case = GenerateCover(articles, images)
    try:
        result = use_case.execute(request)
    except CoverToolError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(result.prompt)
        return 0

    assert result.output_path is not None
    assert result.image is not None
    assert result.width is not None and result.height is not None
    assert images is not None
    print(f"Generated {result.output_path} ({result.width}x{result.height}, JPEG) with {images.model} via OpenRouter.")
    if result.image.provider_cost is not None:
        print(f"Provider-reported image cost: ${result.image.provider_cost:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
