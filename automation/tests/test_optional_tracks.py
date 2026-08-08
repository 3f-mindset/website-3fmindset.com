import pytest

from burn_pipeline.application import BurnPipeline
from burn_pipeline.domain import PipelineSpec
from burn_pipeline.interface import enabled_steps


def test_optional_tracks_are_excluded_by_default() -> None:
    spec = PipelineSpec.model_validate(
        {
            "tracks": {"promo_assets": False, "landing_page": False},
            "steps": [
                {"id": "core", "format": "markdown", "prompt_file": "core.md", "output": "core.md"},
                {
                    "id": "promo",
                    "format": "markdown",
                    "prompt_file": "promo.md",
                    "output": "promo.md",
                    "tracks": ["promo_assets"],
                },
                {
                    "id": "landing",
                    "format": "html",
                    "prompt_file": "landing.md",
                    "output": "landing.html",
                    "tracks": ["landing_page"],
                },
            ],
        }
    )

    assert [step.id for step in enabled_steps(spec)] == ["core"]


def test_landing_track_requires_promo_assets() -> None:
    spec = PipelineSpec.model_validate(
        {
            "tracks": {"promo_assets": False, "landing_page": True},
            "steps": [
                {
                    "id": "promo",
                    "format": "markdown",
                    "prompt_file": "promo.md",
                    "output": "promo.md",
                    "tracks": ["promo_assets"],
                },
                {
                    "id": "landing",
                    "format": "html",
                    "prompt_file": "landing.md",
                    "output": "landing.html",
                    "depends_on": ["promo"],
                    "tracks": ["landing_page"],
                },
            ],
        }
    )
    pipeline = BurnPipeline(files=None, inference_factory=None)

    with pytest.raises(ValueError, match="landing -> promo"):
        pipeline.run_pipeline(spec, force=False)
