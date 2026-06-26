from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field


class OutputFormat(str, Enum):
    MARKDOWN = "markdown"
    SVG = "svg"


class ProviderKind(str, Enum):
    CODEX_CLI = "codex-cli"
    OPENAI = "openai"
    OPENAI_COMPATIBLE = "openai-compatible"


class BurnContext(BaseModel):
    title: str = ""
    slug: str = ""
    date: str = ""


class StepSpec(BaseModel):
    id: str
    format: OutputFormat
    prompt_file: Path
    output: Path
    inputs: list[Path] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)


class PipelineSpec(BaseModel):
    context: BurnContext = Field(default_factory=BurnContext)
    steps: list[StepSpec]


class ProviderConfig(BaseModel):
    kind: ProviderKind = ProviderKind.CODEX_CLI
    model: str | None = None
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    command: str = "codex"


class GenerateCommand(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    step: StepSpec
    context: BurnContext
    prompt_template: str
    named_inputs: dict[str, str] = Field(default_factory=dict)


class InferenceRequest(BaseModel):
    prompt: str
    output_format: OutputFormat
    model: str | None = None


class InferencePort(Protocol):
    def generate(self, request: InferenceRequest) -> str:
        """Return only generated file contents."""


class FileStorePort(Protocol):
    def read_text(self, path: Path) -> str:
        """Read UTF-8 text from a path."""

    def write_text(self, path: Path, content: str, force: bool) -> None:
        """Write UTF-8 text to a path."""

    def exists(self, path: Path) -> bool:
        """Return whether a path exists."""


def build_generation_prompt(command: GenerateCommand) -> str:
    lines = [
        "You are generating one file for the 3F Mindset SteadyBurn content pipeline.",
        "",
        "Return only the requested file contents.",
        "Do not include reasoning, explanation, status text, Markdown fences, XML fences, or surrounding commentary.",
        "Do not edit files or run commands. Produce final text only.",
        "",
        f"Output format: {command.step.format.value}",
        f"Title: {command.context.title}",
        f"Slug: {command.context.slug}",
        f"Date: {command.context.date}",
        "",
    ]

    if command.step.format == OutputFormat.SVG:
        lines.extend(
            [
                "SVG constraints:",
                "- Return a complete SVG XML document starting with <svg.",
                "- Do not wrap the SVG in a Markdown code block.",
                "- Do not include prose before or after the XML.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "Markdown constraints:",
                "- Return Markdown only.",
                "- Do not wrap the Markdown in a code block.",
                "- Do not include prose about the task.",
                "",
            ]
        )

    lines.extend(
        [
            "Prompt template:",
            "--- BEGIN PROMPT TEMPLATE ---",
            command.prompt_template,
            "--- END PROMPT TEMPLATE ---",
            "",
        ]
    )

    for name, content in command.named_inputs.items():
        lines.extend(
            [
                f"Input: {name}",
                "--- BEGIN INPUT ---",
                content,
                "--- END INPUT ---",
                "",
            ]
        )

    return "\n".join(lines)


def validate_generated_content(content: str, output_format: OutputFormat) -> None:
    if not content.strip():
        raise ValueError("Generated output is empty")

    if content.lstrip().startswith("```"):
        raise ValueError("Generated output starts with a Markdown fence")

    if output_format == OutputFormat.SVG and not content.lstrip().startswith("<svg"):
        raise ValueError("Generated SVG does not start with <svg")
