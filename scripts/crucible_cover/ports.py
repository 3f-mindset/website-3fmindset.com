"""Ports used by the application layer."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .domain import ArticleContext, GeneratedImage


class ArticleRepository(Protocol):
    def load(self, letter_path: Path) -> ArticleContext: ...

    def validate_destination(self, article: ArticleContext, *, force: bool) -> None: ...

    def save_cover(
        self,
        article: ArticleContext,
        image: bytes,
        *,
        force: bool,
    ) -> Path: ...


class ImageGenerator(Protocol):
    def generate(self, prompt: str) -> GeneratedImage: ...
