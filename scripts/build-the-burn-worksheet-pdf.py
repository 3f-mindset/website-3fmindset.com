#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "weasyprint>=62.0",
# ]
# ///

from __future__ import annotations

import argparse
import html
import subprocess
import sys
from pathlib import Path

from weasyprint import HTML


SVG_MARKER = "CHOICE"


PRINT_CSS = """
@page {
  size: letter landscape;
  margin: 0;
}

html,
body {
  height: 100%;
  margin: 0;
  padding: 0;
  width: 100%;
}

body {
  align-items: center;
  display: flex;
  justify-content: center;
}

img {
  display: block;
  height: 100vh;
  object-fit: contain;
  width: 100vw;
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render an SVG image as a full-page landscape PDF."
    )
    parser.add_argument(
        "svg_file",
        type=Path,
        nargs="?",
        help=(
            "SVG file to render. If omitted, the script searches uncommitted "
            "git changes for the SVG containing 'CHOICE'."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output PDF path. Defaults to <svg-file-stem>.pdf next to the SVG.",
    )
    return parser.parse_args()


def validate_input(svg_file: Path) -> Path:
    svg_file = svg_file.expanduser().resolve()

    if not svg_file.is_file():
        raise FileNotFoundError(f"SVG file not found: {svg_file}")
    if svg_file.suffix.lower() != ".svg":
        raise ValueError(f"Input file must be an SVG: {svg_file}")

    return svg_file


def git_changed_paths() -> list[Path]:
    changed = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=ACMRTUXB", "HEAD", "--"],
        check=True,
        capture_output=True,
        text=True,
    )
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        check=True,
        capture_output=True,
        text=True,
    )

    paths: list[Path] = []
    seen: set[str] = set()
    for line in [*changed.stdout.splitlines(), *untracked.stdout.splitlines()]:
        if not line or line in seen:
            continue
        seen.add(line)
        paths.append(Path(line))
    return paths


def find_uncommitted_choice_svg() -> Path:
    matches = []
    for path in git_changed_paths():
        if path.suffix.lower() != ".svg" or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if SVG_MARKER.lower() in text.lower():
            matches.append(path)

    if not matches:
        raise FileNotFoundError(
            f"No uncommitted SVG file contains {SVG_MARKER!r}."
        )
    if len(matches) > 1:
        formatted = "\n".join(f"  - {path}" for path in matches)
        raise ValueError(
            f"Multiple uncommitted SVG files contain {SVG_MARKER!r}:\n"
            f"{formatted}\nPass svg_file explicitly."
        )
    return matches[0]


def resolve_input(args: argparse.Namespace) -> Path:
    return args.svg_file if args.svg_file else find_uncommitted_choice_svg()


def default_output_path(svg_file: Path) -> Path:
    return svg_file.with_suffix(".pdf")


def svg_to_html(svg_file: Path) -> str:
    title = html.escape(svg_file.stem.replace("-", " ").replace("_", " ").title())
    svg_name = html.escape(svg_file.name, quote=True)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>{PRINT_CSS}</style>
</head>
<body>
  <img src="{svg_name}" alt="{title}">
</body>
</html>
"""


def render_svg_pdf(svg_file: Path, output_pdf: Path) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=svg_to_html(svg_file), base_url=str(svg_file.parent)).write_pdf(
        output_pdf
    )


def main() -> int:
    args = parse_args()

    try:
        svg_file = validate_input(resolve_input(args))
        output_pdf = (
            args.output.expanduser().resolve()
            if args.output
            else default_output_path(svg_file)
        )

        render_svg_pdf(svg_file, output_pdf)
        print(f"Wrote {output_pdf}")
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
