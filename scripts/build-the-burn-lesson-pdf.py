#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "markdown>=3.6",
#   "pypdf>=5.0",
#   "weasyprint>=62.0",
# ]
# ///

from __future__ import annotations

import argparse
import html
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown
from pypdf import PdfReader, PdfWriter
from weasyprint import HTML


FRONT_MATTER_RE = re.compile(
    r"\A(?:---\s*\n.*?\n---\s*\n?|\+\+\+\s*\n.*?\n\+\+\+\s*\n?)",
    re.DOTALL,
)
DEFAULT_COVER_PDF = Path("static") / "worksheets" / "SB-cover.pdf"
MARKDOWN_SUFFIXES = {".md", ".markdown"}


PRINT_CSS = """
@page {
  size: letter;
  margin: 0.75in;
}

html {
  color: #1f2933;
  font-family: Georgia, "Times New Roman", serif;
  font-size: 12pt;
  line-height: 1.55;
}

body {
  margin: 0;
}

h1,
h2,
h3,
h4 {
  color: #111827;
  font-family: Arial, Helvetica, sans-serif;
  line-height: 1.2;
  margin: 1.4em 0 0.45em;
  page-break-after: avoid;
}

h1 {
  font-size: 24pt;
  margin-top: 0;
}

h2 {
  font-size: 17pt;
}

h3 {
  font-size: 13.5pt;
}

p,
ul,
ol,
blockquote,
pre,
table {
  margin: 0 0 0.9em;
}

li {
  margin: 0.2em 0;
}

a {
  color: #111827;
  text-decoration: underline;
}

blockquote {
  border-left: 3px solid #c9d1d9;
  color: #3b4652;
  padding-left: 0.9em;
}

code {
  background: #f2f4f7;
  border-radius: 3px;
  font-family: "Courier New", monospace;
  font-size: 0.9em;
  padding: 0.05em 0.25em;
}

pre {
  background: #f2f4f7;
  border-radius: 5px;
  font-size: 9.5pt;
  overflow-wrap: break-word;
  padding: 0.75em;
  white-space: pre-wrap;
}

pre code {
  background: transparent;
  padding: 0;
}

table {
  border-collapse: collapse;
  width: 100%;
}

th,
td {
  border: 1px solid #d8dee4;
  padding: 0.35em 0.5em;
  vertical-align: top;
}

img {
  height: auto;
  max-width: 100%;
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a PDF by placing a cover PDF before rendered markdown content."
        )
    )
    parser.add_argument(
        "--cover-pdf",
        type=Path,
        help=(
            "PDF to use as the cover page. Defaults to "
            "static/worksheets/SB-cover.pdf when markdown is auto-detected."
        ),
    )
    parser.add_argument(
        "--markdown-file",
        type=Path,
        help=(
            "Markdown file to render. If omitted, the script searches uncommitted "
            "git changes for new markdown files."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help=(
            "Output PDF path. Defaults to <markdown-file-stem>.pdf in the markdown "
            "file's directory. Only valid when rendering a single markdown file."
        ),
    )
    return parser.parse_args()


def validate_inputs(cover_pdf: Path, markdown_file: Path) -> tuple[Path, Path]:
    cover_pdf = cover_pdf.expanduser().resolve()
    markdown_file = markdown_file.expanduser().resolve()

    if not cover_pdf.is_file():
        raise FileNotFoundError(f"Cover PDF not found: {cover_pdf}")
    if cover_pdf.suffix.lower() != ".pdf":
        raise ValueError(f"Cover file must be a PDF: {cover_pdf}")

    if not markdown_file.is_file():
        raise FileNotFoundError(f"Markdown file not found: {markdown_file}")
    if markdown_file.suffix.lower() not in MARKDOWN_SUFFIXES:
        raise ValueError(f"Markdown file must end in .md or .markdown: {markdown_file}")

    return cover_pdf, markdown_file


def git_new_paths() -> list[Path]:
    changed = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=A", "HEAD", "--"],
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


def find_uncommitted_lesson_markdowns() -> list[Path]:
    matches: list[Path] = []
    for path in git_new_paths():
        if path.suffix.lower() not in MARKDOWN_SUFFIXES or not path.is_file():
            continue
        if path.name.lower() == "index.md":
            continue
        matches.append(path)

    if not matches:
        raise FileNotFoundError(
            "No uncommitted markdown files were found."
        )
    return matches


def resolve_inputs(args: argparse.Namespace) -> tuple[Path, list[Path]]:
    cover_pdf = args.cover_pdf if args.cover_pdf else DEFAULT_COVER_PDF
    markdown_files = (
        [args.markdown_file] if args.markdown_file else find_uncommitted_lesson_markdowns()
    )
    return cover_pdf, markdown_files


def strip_front_matter(markdown_text: str) -> str:
    return FRONT_MATTER_RE.sub("", markdown_text, count=1).lstrip()


def markdown_to_html(markdown_file: Path) -> str:
    markdown_text = strip_front_matter(markdown_file.read_text(encoding="utf-8"))
    body = markdown.markdown(
        markdown_text,
        extensions=[
            "markdown.extensions.extra",
            "markdown.extensions.sane_lists",
            "markdown.extensions.smarty",
            "markdown.extensions.toc",
        ],
        output_format="html5",
    )
    title = html.escape(markdown_file.stem.replace("-", " ").replace("_", " ").title())

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>{PRINT_CSS}</style>
</head>
<body>
{body}
</body>
</html>
"""


def render_markdown_pdf(markdown_file: Path, content_pdf: Path) -> None:
    html_text = markdown_to_html(markdown_file)
    HTML(string=html_text, base_url=str(markdown_file.parent)).write_pdf(content_pdf)


def merge_pdfs(cover_pdf: Path, content_pdf: Path, output_pdf: Path) -> None:
    writer = PdfWriter()

    for pdf_path in (cover_pdf, content_pdf):
        reader = PdfReader(str(pdf_path))
        for page in reader.pages:
            writer.add_page(page)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with output_pdf.open("wb") as output_file:
        writer.write(output_file)


def default_output_path(markdown_file: Path) -> Path:
    return markdown_file.with_suffix(".pdf")


def main() -> int:
    args = parse_args()

    try:
        cover_pdf_arg, markdown_file_args = resolve_inputs(args)
        cover_pdf = cover_pdf_arg.expanduser().resolve()
        markdown_files = [markdown_file.expanduser().resolve() for markdown_file in markdown_file_args]
        if args.output and len(markdown_files) > 1:
            raise ValueError(
                "--output can only be used when rendering a single markdown file."
            )

        for markdown_file in markdown_files:
            validated_cover_pdf, validated_markdown_file = validate_inputs(
                cover_pdf, markdown_file
            )
            output_pdf = (
                args.output.expanduser().resolve()
                if args.output
                else default_output_path(validated_markdown_file)
            )

            if output_pdf == validated_cover_pdf:
                raise ValueError("Output path would overwrite the cover PDF.")

            with tempfile.TemporaryDirectory() as temp_dir:
                content_pdf = Path(temp_dir) / "markdown-content.pdf"
                render_markdown_pdf(validated_markdown_file, content_pdf)
                merge_pdfs(validated_cover_pdf, content_pdf, output_pdf)

            print(f"Wrote {output_pdf}")
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
