"""Pure cover-domain rules: article context, prompt composition, and PNG checks."""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from pathlib import Path


class CoverToolError(Exception):
    """Base class for expected cover-generation failures."""


class ArticleError(CoverToolError):
    """The target letter is missing or has invalid cover metadata."""


class CoverConflictError(CoverToolError):
    """The target cover already exists or changed during generation."""


class ImageValidationError(CoverToolError):
    """The provider returned an unsupported or invalid image."""


class ProviderError(CoverToolError):
    """The image provider could not complete the request."""


STYLE_PROMPT = """Generate a wide 3:2 illustration. The illustration features a vintage style art with strong influences from the pulp magazine era characterized by:
1. Halftone Dot Pattern: The entire image uses a halftone dot effect, which gives it a textured, printed look, typical of old comic books and newspapers.
2. Bold, Black Linework: The lines outlining the objects, symbols, or elements in the scene are thick and prominent, providing a strong contrast against the softer colors and making the details stand out.
3. Limited, Muted Color Palette: The colors are desaturated and muted, mostly earthy tones like browns, beiges, and faded blues, which create a nostalgic and aged feel.
4. Cross-hatching and Shading: The shading on objects, symbols, and the background is done using cross-hatching, enhancing the vintage look and adding depth to the illustration.
5. Dynamic Composition: The arrangement of objects or symbols is energetic and slightly exaggerated, emphasizing movement or impact in the scene. Shapes and forms are expressive, capturing a moment of tension or drama.
6. Stylized Clouds and Background Elements: The clouds, abstract symbols, or environmental elements in the background are simplified and stylized, with the same halftone texture applied, ensuring consistency with the overall retro aesthetic.

These characteristics combine to give the image a distinct, old-timey, illustrative feel, akin to something from a 1940s-1950s comic book or an early 20th-century advertisement. The unmistakable primary/focal figure must be a clearly adult man. Any other people present must remain visually secondary and must not rival or displace him as the focal figure. No words in the image and no captions. Also exclude typography, labels, signage, letters, numbers, logos, or any readable marks. Avoid photography, modern photorealism, neon cyberpunk, solarpunk, heavy noir, intricate engraving, gloomy atmosphere, realistic environments, and UI mockups. Keep one central focal point, a simple secondary background, clear subject separation, bright readable lighting, and mostly midtones."""

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
TARGET_ASPECT_RATIO = 3 / 2
ASPECT_RATIO_TOLERANCE = 0.03


@dataclass(frozen=True)
class CoverRequest:
    letter_path: Path
    scene_direction: str = ""
    force: bool = False
    dry_run: bool = False


@dataclass(frozen=True)
class ArticleContext:
    letter_directory: Path
    index_path: Path
    title: str
    excerpt: str
    original_markdown: str
    frontmatter: str
    body: str

    @property
    def cover_path(self) -> Path:
        return self.letter_directory / "cover.jpg"


@dataclass(frozen=True)
class GeneratedImage:
    data: bytes
    provider_cost: float | None = None


@dataclass(frozen=True)
class CoverResult:
    prompt: str
    image: GeneratedImage | None
    width: int | None
    height: int | None
    output_path: Path | None


def parse_article_markdown(letter_path: Path, markdown: str) -> ArticleContext:
    """Extract only the publication context needed for a cover prompt."""
    match = re.match(r"\A---\s*\n(.*?)\n---\s*\n", markdown, re.S)
    if not match:
        raise ArticleError("index.md must start with YAML frontmatter delimited by ---")

    frontmatter = match.group(1)
    title_match = re.search(r"(?m)^title:\s*(.*)\s*$", frontmatter)
    if not title_match:
        raise ArticleError("index.md frontmatter is missing title")

    title = title_match.group(1).strip().strip("\"'")
    body = markdown[match.end():]
    excerpt = extract_excerpt(body)
    return ArticleContext(
        letter_directory=letter_path,
        index_path=letter_path / "index.md",
        title=title,
        excerpt=excerpt,
        original_markdown=markdown,
        frontmatter=frontmatter,
        body=body,
    )


def extract_excerpt(body: str, limit: int = 1400) -> str:
    paragraphs = re.split(r"\n\s*\n", body.strip())
    kept: list[str] = []
    for paragraph in paragraphs:
        paragraph = re.sub(r"^#{1,6}\s+", "", paragraph.strip())
        paragraph = re.sub(
            r"\*\*([^*]+)\*\*|\*([^*]+)\*|`([^`]+)`",
            lambda match: next(value for value in match.groups() if value is not None),
            paragraph,
        )
        paragraph = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", paragraph)
        if paragraph and not paragraph.startswith("{{<"):
            kept.append(paragraph)
        if len(" ".join(kept)) >= limit:
            break
    return " ".join(kept)[:limit]


def compose_prompt(article: ArticleContext, scene_direction: str = "") -> str:
    scene = scene_direction.strip() or (
        "Choose one clear, article-specific object or action from the title and context. "
        "Do not make a generic cover."
    )
    article_context = f"Article title: {article.title}\nArticle context: {article.excerpt}"
    return (
        f"{STYLE_PROMPT}\n\n{article_context}\n\n"
        f"Scene direction: {scene}\n\n"
        "Keep the article-specific scene simple and readable. "
        "Do not include any on-image text or marks."
    )


def inspect_png(data: bytes) -> tuple[int, int]:
    if not data.startswith(PNG_SIGNATURE) or len(data) < 24:
        raise ImageValidationError("OpenRouter response was not a valid PNG")
    width, height = struct.unpack(">II", data[16:24])
    if not width or not height:
        raise ImageValidationError("PNG has invalid dimensions")
    if abs(width / height - TARGET_ASPECT_RATIO) > ASPECT_RATIO_TOLERANCE:
        raise ImageValidationError(
            f"expected a 3:2 PNG, got {width}x{height}"
        )
    return width, height
