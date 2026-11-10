"""Application use case for generating and saving one letter cover."""

from __future__ import annotations

from io import BytesIO

from PIL import Image

from .domain import CoverRequest, CoverResult, CoverToolError, compose_prompt, inspect_png
from .ports import ArticleRepository, ImageGenerator


class GenerateCover:
    """Coordinate article loading, prompt composition, generation, and storage."""

    def __init__(
        self, articles: ArticleRepository, images: ImageGenerator | None
    ) -> None:
        self._articles = articles
        self._images = images

    def execute(self, request: CoverRequest) -> CoverResult:
        article = self._articles.load(request.letter_path)
        prompt = compose_prompt(article, request.scene_direction)
        if request.dry_run:
            return CoverResult(prompt, None, None, None, None)
        if self._images is None:
            raise CoverToolError("image generation is unavailable in this execution mode")

        self._articles.validate_destination(article, force=request.force)
        image = self._images.generate(prompt)
        width, height = inspect_png(image.data)
        source = Image.open(BytesIO(image.data)).convert("RGB")
        output = BytesIO()
        source.save(output, format="JPEG", quality=70, progressive=True, optimize=True, subsampling=2)
        output_path = self._articles.save_cover(article, output.getvalue(), force=request.force)
        return CoverResult(prompt, image, width, height, output_path)
