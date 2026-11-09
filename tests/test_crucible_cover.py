from __future__ import annotations

import base64
import contextlib
import io
import json
import os
import struct
import sys
import tempfile
import unittest
import urllib.request
import zlib
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from crucible_cover.adapters.filesystem import FileSystemArticleRepository
from crucible_cover.adapters.openrouter import OpenRouterImageGenerator
from crucible_cover.application import GenerateCover
from crucible_cover.cli import main
from crucible_cover.domain import (
    ArticleError,
    ArticleContext,
    CoverConflictError,
    CoverRequest,
    GeneratedImage,
    ImageValidationError,
    inspect_png,
    parse_article_markdown,
)


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(kind + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


def _small_png(width: int = 3, height: int = 2) -> bytes:
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    row = b"\x00" + (b"\x22\x44\x66" * width)
    pixels = zlib.compress(row * height)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", pixels)
        + _png_chunk(b"IEND", b"")
    )


def _article_markdown(title: str = "A clear day") -> str:
    return (
        "---\n"
        "date: 2026-01-04\n"
        'slug: "crucible-clear-day"\n'
        f'title: "{title}"\n'
        "draft: false\n"
        "categories: crucible\n"
        "tags:\n"
        "  - reflection\n"
        "---\n\n"
        "A quiet table holds a brass compass in a pool of light.\n\n"
        "The rest of the day waits nearby.\n"
    )


class _FakeRepository:
    def __init__(self, article: ArticleContext) -> None:
        self.article = article
        self.validated = False
        self.saved: bytes | None = None

    def load(self, letter_path: Path) -> ArticleContext:
        return self.article

    def validate_destination(self, article: ArticleContext, *, force: bool) -> None:
        self.validated = True

    def save_cover(self, article: ArticleContext, image: bytes, *, force: bool) -> Path:
        self.saved = image
        return article.cover_path


class _FakeImageGenerator:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> GeneratedImage:
        self.prompts.append(prompt)
        return GeneratedImage(_small_png(), 0.04)


class DomainTests(unittest.TestCase):
    def test_prompt_combines_style_article_context_and_scene(self) -> None:
        article = parse_article_markdown(Path("letter"), _article_markdown())
        from crucible_cover.domain import compose_prompt

        prompt = compose_prompt(article, "Keep the compass as the central focal point.")
        self.assertIn("halftone dot effect", prompt)
        self.assertIn("Article title: A clear day", prompt)
        self.assertIn("brass compass", prompt)
        self.assertIn("Keep the compass as the central focal point", prompt)
        self.assertIn("No words", prompt)

    def test_png_dimensions_must_be_three_by_two(self) -> None:
        self.assertEqual(inspect_png(_small_png()), (3, 2))
        with self.assertRaises(ImageValidationError):
            inspect_png(_small_png(width=2, height=2))


class ApplicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.article = parse_article_markdown(Path("letter"), _article_markdown())
        self.repository = _FakeRepository(self.article)
        self.generator = _FakeImageGenerator()
        self.use_case = GenerateCover(self.repository, self.generator)

    def test_dry_run_builds_prompt_without_provider_or_write(self) -> None:
        result = self.use_case.execute(
            CoverRequest(Path("letter"), "a focused scene", dry_run=True)
        )
        self.assertIn("a focused scene", result.prompt)
        self.assertIsNone(result.image)
        self.assertEqual(self.generator.prompts, [])
        self.assertFalse(self.repository.validated)
        self.assertIsNone(self.repository.saved)

    def test_generation_validates_then_saves_one_image(self) -> None:
        result = self.use_case.execute(CoverRequest(Path("letter")))
        self.assertTrue(self.repository.validated)
        self.assertEqual(len(self.generator.prompts), 1)
        self.assertEqual(self.repository.saved, _small_png())
        self.assertEqual((result.width, result.height), (3, 2))
        self.assertEqual(result.output_path, self.article.cover_path)


class FileSystemAdapterTests(unittest.TestCase):
    def test_saves_image_and_adds_relative_cover_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            letter = Path(temp)
            (letter / "index.md").write_text(_article_markdown(), encoding="utf-8")
            repository = FileSystemArticleRepository()
            article = repository.load(letter)
            repository.validate_destination(article, force=False)
            output = repository.save_cover(article, _small_png(), force=False)
            markdown = (letter / "index.md").read_text(encoding="utf-8")

            self.assertEqual(output.name, "cover.png")
            self.assertEqual(output.read_bytes(), _small_png())
            self.assertIn('cover:\n  image: "cover.png"\n  relative: true', markdown)
            self.assertIn("A quiet table holds a brass compass", markdown)

    def test_refuses_existing_cover_without_force(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            letter = Path(temp)
            (letter / "index.md").write_text(_article_markdown(), encoding="utf-8")
            (letter / "cover.png").write_bytes(_small_png())
            repository = FileSystemArticleRepository()
            article = repository.load(letter)
            with self.assertRaises(CoverConflictError):
                repository.validate_destination(article, force=False)

    def test_refuses_to_overwrite_incompatible_cover_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            letter = Path(temp)
            markdown = _article_markdown().replace(
                "draft: false\n",
                'cover:\n  image: "different.png"\n  relative: true\n\ndraft: false\n',
            )
            (letter / "index.md").write_text(markdown, encoding="utf-8")
            repository = FileSystemArticleRepository()
            article = repository.load(letter)
            with self.assertRaises(ArticleError):
                repository.validate_destination(article, force=False)

    def test_rejects_article_changed_during_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            letter = Path(temp)
            path = letter / "index.md"
            path.write_text(_article_markdown(), encoding="utf-8")
            repository = FileSystemArticleRepository()
            article = repository.load(letter)
            repository.validate_destination(article, force=False)
            path.write_text(path.read_text(encoding="utf-8") + "Changed.\n", encoding="utf-8")
            with self.assertRaises(CoverConflictError):
                repository.save_cover(article, _small_png(), force=False)
            self.assertFalse((letter / "cover.png").exists())


class OpenRouterAdapterTests(unittest.TestCase):
    def test_uses_openrouter_image_api_and_returns_usage(self) -> None:
        image = _small_png()
        response_body = json.dumps(
            {
                "data": [{"b64_json": base64.b64encode(image).decode("ascii")}],
                "usage": {"cost": 0.04},
            }
        ).encode("utf-8")

        class _Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return response_body

        adapter = OpenRouterImageGenerator("test-secret")
        with patch("urllib.request.urlopen", return_value=_Response()) as urlopen:
            result = adapter.generate("a short prompt")

        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://openrouter.ai/api/v1/images")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-secret")
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], "openai/gpt-image-2")
        self.assertEqual(payload["aspect_ratio"], "3:2")
        self.assertEqual(result.data, image)
        self.assertEqual(result.provider_cost, 0.04)


class CliTests(unittest.TestCase):
    def test_dry_run_works_without_provider_key(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            letter = Path(temp)
            (letter / "index.md").write_text(_article_markdown(), encoding="utf-8")
            output = io.StringIO()
            with patch.dict(os.environ, {"CRUCIBLE_REPO_ROOT": temp}, clear=False):
                with contextlib.redirect_stdout(output):
                    status = main([str(letter), "--prompt", "keep the compass central", "--dry-run"])
            self.assertEqual(status, 0)
            self.assertIn("keep the compass central", output.getvalue())


if __name__ == "__main__":
    unittest.main()
