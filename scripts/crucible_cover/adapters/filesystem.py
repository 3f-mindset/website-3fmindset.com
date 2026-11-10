"""Filesystem adapter for Hugo letter bundles."""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

from ..domain import ArticleContext, ArticleError, CoverConflictError, parse_article_markdown


class FileSystemArticleRepository:
    def load(self, letter_path: Path) -> ArticleContext:
        path = letter_path.expanduser()
        if path.is_file():
            if path.name != "index.md":
                raise ArticleError("letter path must be a directory or its index.md")
            path = path.parent
        try:
            directory = path.resolve(strict=True)
        except FileNotFoundError as exc:
            raise ArticleError(f"letter directory does not exist: {letter_path}") from exc
        if not directory.is_dir():
            raise ArticleError("letter path must be a directory or its index.md")
        index_path = directory / "index.md"
        if not index_path.is_file():
            raise ArticleError(f"letter directory does not contain index.md: {directory}")
        markdown = index_path.read_text(encoding="utf-8")
        return parse_article_markdown(directory, markdown)

    def validate_destination(self, article: ArticleContext, *, force: bool) -> None:
        self._assert_unchanged(article)
        self._updated_markdown(article)
        if article.cover_path.exists() and not force:
            raise CoverConflictError(
                f"{article.cover_path} already exists; pass --force to replace it"
            )

    def save_cover(
        self,
        article: ArticleContext,
        image: bytes,
        *,
        force: bool,
    ) -> Path:
        self._assert_unchanged(article)
        updated_markdown = self._updated_markdown(article)
        output = article.cover_path
        if output.exists() and not force:
            raise CoverConflictError(
                f"{output} already exists; pass --force to replace it"
            )

        previous_image = output.read_bytes() if output.exists() else None
        try:
            self._atomic_write(output, image)
            if updated_markdown != article.original_markdown:
                self._atomic_write(
                    article.index_path, updated_markdown.encode("utf-8")
                )
        except Exception:
            if previous_image is None:
                output.unlink(missing_ok=True)
            else:
                self._atomic_write(output, previous_image)
            raise
        return output

    @staticmethod
    def _assert_unchanged(article: ArticleContext) -> None:
        if article.index_path.read_text(encoding="utf-8") != article.original_markdown:
            raise CoverConflictError(
                "index.md changed while the cover was being prepared; reload and retry"
            )

    @staticmethod
    def _updated_markdown(article: ArticleContext) -> str:
        frontmatter = article.frontmatter
        cover = re.search(r"(?ms)^cover:\s*\n((?:^[ \t]+.*(?:\n|$))*)", frontmatter)
        if cover:
            block = cover.group(1)
            image = re.search(r'(?m)^\s+image:\s*["\']?([^"\'\s]+)', block)
            relative = re.search(r"(?m)^\s+relative:\s*(true|false)\s*$", block)
            if (
                not image
                or image.group(1) != "cover.jpg"
                or not relative
                or relative.group(1) != "true"
            ):
                raise ArticleError(
                    "index.md has an incompatible cover declaration; refusing to replace it"
                )
            return article.original_markdown

        draft = re.search(r"(?m)^draft:\s*false\s*$", frontmatter)
        if not draft:
            raise ArticleError(
                "index.md needs a `draft: false` field before cover metadata can be added"
            )
        addition = 'cover:\n  image: "cover.jpg"\n  relative: true\n\n'
        updated_frontmatter = (
            frontmatter[:draft.start()] + addition + frontmatter[draft.start():]
        )
        return f"---\n{updated_frontmatter}\n---\n\n{article.body}"

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        temp_name = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temp_name = stream.name
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, path)
        finally:
            if temp_name and os.path.exists(temp_name):
                os.unlink(temp_name)
