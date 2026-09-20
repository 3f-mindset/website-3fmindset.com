from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


ROLES = {"curriculum", "evidence", "voice", "examples", "constraints", "prior_artifact", "implementation_reference"}
STATES = {"draft", "queued", "running", "repairing", "passed", "failed", "promoted", "needs-review"}


@dataclass(frozen=True)
class Source:
    path: str
    alias: str
    role: str
    scope: str = "full"
    sha256: str = ""
    upstream_run: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QualityPolicy:
    required_criteria_pass: bool = True
    minimum_scores: dict[str, float] = field(default_factory=lambda: {"index.md": 70, "INSTRUCTIONS.md": 70})
    max_repairs: int = 3

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Intake:
    primary: Source
    supplements: tuple[Source, ...] = ()
    focus_prompt: str = ""
    specification: dict[str, Any] = field(default_factory=dict)
    tracks: tuple[str, ...] = ()
    quality: QualityPolicy = field(default_factory=QualityPolicy)

    def as_dict(self) -> dict[str, Any]:
        return {
            "primary": self.primary.as_dict(),
            "supplements": [source.as_dict() for source in self.supplements],
            "focus_prompt": self.focus_prompt,
            "specification": self.specification,
            "tracks": list(self.tracks),
            "quality": self.quality.as_dict(),
        }
