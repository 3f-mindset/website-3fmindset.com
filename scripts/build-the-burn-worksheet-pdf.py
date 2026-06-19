#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "cairosvg>=2.8.0",
#   "pillow>=10.0.0",
#   "weasyprint>=62.0",
# ]
# ///

from __future__ import annotations

import argparse
import html
import subprocess
import sys
from io import BytesIO
from pathlib import Path

import cairosvg
from PIL import Image
from weasyprint import HTML


JPEG_QUALITY = 75


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
            "SVG file to render. If omitted, the script renders every new or "
            "updated SVG in the working tree."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help=(
            "Output PDF path. Defaults to <svg-file-stem>.pdf next to the SVG. "
            "A JPEG copy is always written alongside the PDF."
        ),
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


def find_changed_svgs() -> list[Path]:
    matches = [
        path
        for path in git_changed_paths()
        if path.suffix.lower() == ".svg" and path.is_file()
    ]

    if not matches:
        raise FileNotFoundError("No new or updated SVG files found.")
    return sorted(matches)


def resolve_inputs(args: argparse.Namespace) -> list[Path]:
    return [args.svg_file] if args.svg_file else find_changed_svgs()


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


def render_svg_jpeg(svg_file: Path, output_jpeg: Path) -> None:
    output_jpeg.parent.mkdir(parents=True, exist_ok=True)
    png_bytes = cairosvg.svg2png(url=str(svg_file))

    with Image.open(BytesIO(png_bytes)) as image:
        rgba_image = image.convert("RGBA")
        flattened = Image.new("RGB", image.size, "white")
        flattened.paste(rgba_image, mask=rgba_image.getchannel("A"))
        flattened.save(
            output_jpeg,
            format="JPEG",
            quality=JPEG_QUALITY,
            optimize=True,
            progressive=True,
        )


def main() -> int:
    args = parse_args()

    try:
        svg_files = [validate_input(svg_file) for svg_file in resolve_inputs(args)]
        if args.output and len(svg_files) > 1:
            raise ValueError(
                "--output can only be used when rendering a single SVG file."
            )

        for svg_file in svg_files:
            output_pdf = (
                args.output.expanduser().resolve()
                if args.output
                else default_output_path(svg_file)
            )
            output_jpeg = output_pdf.with_suffix(".jpg")
            render_svg_pdf(svg_file, output_pdf)
            render_svg_jpeg(svg_file, output_jpeg)
            print(f"Wrote {output_pdf}")
            print(f"Wrote {output_jpeg}")
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
