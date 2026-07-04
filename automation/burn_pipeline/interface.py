from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

from .application import BurnPipeline, step_from_paths
from .domain import BurnContext, GenerationModality, PipelineSpec, ProviderConfig, ProviderKind
from .infrastructure import (
    LocalFileStore,
    build_inference,
    load_model_registry,
    load_pipeline_spec,
    render_steadyburn_verb_index_markdown,
    scan_model_registry,
    scan_steadyburn_verb_index,
    write_model_registry,
)


DEFAULT_PROVIDER = ProviderKind.OPENAI_COMPATIBLE.value
DEFAULT_OPENAI_COMPATIBLE_BASE_URL = "http://localhost:11434"
DEFAULT_TEXT_MODEL = "active"


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cwd = Path(args.cwd).resolve()
    files = LocalFileStore(cwd)

    if args.command == "init-production":
        created = scaffold_production(
            files=files,
            cwd=cwd,
            title=args.title,
            slug=args.slug,
            date_value=args.date,
            target_dir=Path(args.target_dir) if args.target_dir else None,
            pipeline_file=Path(args.pipeline_file) if args.pipeline_file else None,
            force=args.force,
        )
        for path in created:
            print(f"Wrote: {path}")
        return

    if args.command == "sync-model-registry":
        registry = scan_model_registry(files)
        registry_path = Path(args.registry_file)
        write_model_registry(files, registry_path, registry, force=True)
        print(f"Wrote: {registry_path} ({len(registry.entries)} tracked model(s))")
        return

    if args.command == "sync-steadyburn-verb-index":
        index = scan_steadyburn_verb_index(files)
        json_path = Path(args.steadyburn_index_json)
        markdown_path = Path(args.steadyburn_index_markdown)
        files.write_text(
            json_path,
            json.dumps(index.model_dump(), indent=2, ensure_ascii=True) + "\n",
            force=True,
        )
        files.write_text(
            markdown_path,
            render_steadyburn_verb_index_markdown(index),
            force=True,
        )
        print(
            f"Wrote: {json_path} and {markdown_path} "
            f"({len(index.entries)} indexed SteadyBurn letter(s))"
        )
        return

    if args.command == "generate-step":
        step = step_from_paths(
            step_id=args.step_id,
            output_format=args.format,
            prompt_file=Path(args.prompt_file),
            output=Path(args.output),
            inputs=[Path(value) for value in args.input],
        )
        if args.dry_run:
            print_step_plan(step.id, [str(path.path) for path in step.inputs if path.path], str(step.output))
            return
        pipeline = build_pipeline(args, cwd, files)
        content = pipeline.generate_step(
            step=step,
            context=BurnContext(title=args.title, slug=args.slug, date=args.date),
            force=args.force,
            variables=parse_key_value_pairs(args.var),
        )
        print(f"Wrote: {step.output} ({len(content)} chars)")
        return

    if args.command == "run":
        spec = load_pipeline_spec(Path(args.pipeline))
        spec = apply_context_overrides(
            spec,
            title=args.title,
            slug=args.slug,
            date=args.date,
            target_dir=args.target_dir,
            variables=parse_key_value_pairs(args.var),
        )
        if args.dry_run:
            for step in spec.steps:
                print_step_plan(
                    step.id,
                    [describe_input(input_source) for input_source in step.inputs],
                    str(step.output),
                )
            return
        pipeline = build_pipeline(args, cwd, files, spec=spec)
        pipeline.run_pipeline(spec=spec, force=args.force)
        print(f"Wrote {len(spec.steps)} pipeline step output(s)")
        return

    if args.command == "render-prompt":
        spec = load_pipeline_spec(Path(args.pipeline))
        spec = apply_context_overrides(
            spec,
            title=args.title,
            slug=args.slug,
            date=args.date,
            target_dir=args.target_dir,
            variables=parse_key_value_pairs(args.var),
        )
        step = next((candidate for candidate in spec.steps if candidate.id == args.step_id), None)
        if step is None:
            parser.error(f"Unknown step id: {args.step_id}")
        pipeline = build_pipeline(args, cwd, files, spec=spec)
        prompt = pipeline.render_step_prompt(
            step=step,
            context=spec.context,
            variables=spec.variables,
            state={},
            steps_by_id={candidate.id: candidate for candidate in spec.steps},
        )
        print(prompt)
        return

    parser.error("No command provided")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="burn-pipeline",
        description="Generate SteadyBurn content artifacts with pluggable LLM providers.",
    )
    parser.add_argument("--cwd", default=".", help="Workspace root for relative paths.")
    parser.add_argument(
        "--provider",
        choices=[kind.value for kind in ProviderKind],
        default=DEFAULT_PROVIDER,
        help="Default text inference provider.",
    )
    parser.add_argument("--model", default=DEFAULT_TEXT_MODEL, help="Default text model.")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_OPENAI_COMPATIBLE_BASE_URL,
        help="Default text OpenAI-compatible API base URL.",
    )
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="API key environment variable.")
    parser.add_argument("--codex-bin", default="codex", help="Codex CLI executable when using codex-cli.")
    parser.add_argument(
        "--image-provider",
        choices=[kind.value for kind in ProviderKind],
        default=None,
        help="Default image inference provider.",
    )
    parser.add_argument("--image-model", default=None, help="Default image model.")
    parser.add_argument("--image-base-url", default=None, help="Default image OpenAI-compatible API base URL.")
    parser.add_argument("--image-api-key-env", default=None, help="Image API key environment variable.")
    parser.add_argument("--image-codex-bin", default=None, help="Codex CLI executable when using codex-cli for image.")
    parser.add_argument(
        "--audio-provider",
        choices=[kind.value for kind in ProviderKind],
        default=None,
        help="Default audio inference provider.",
    )
    parser.add_argument("--audio-model", default=None, help="Default audio model.")
    parser.add_argument("--audio-base-url", default=None, help="Default audio OpenAI-compatible API base URL.")
    parser.add_argument("--audio-api-key-env", default=None, help="Audio API key environment variable.")
    parser.add_argument("--audio-codex-bin", default=None, help="Codex CLI executable when using codex-cli for audio.")
    parser.add_argument(
        "--registry-file",
        default="automation/pipelines/model-registry.json",
        help="Path to the persisted developed-model registry.",
    )
    parser.add_argument(
        "--steadyburn-index-json",
        default="automation/pipelines/steadyburn-verb-index.json",
        help="Path to the SteadyBurn verb index JSON file.",
    )
    parser.add_argument(
        "--steadyburn-index-markdown",
        default="automation/pipelines/steadyburn-verb-index.md",
        help="Path to the SteadyBurn verb index Markdown file.",
    )
    parser.add_argument(
        "--var",
        action="append",
        default=[],
        help="Template variable override in key=value form. Repeatable.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    init_production = subparsers.add_parser(
        "init-production",
        help="Create a production brief and pipeline file for one letter.",
    )
    init_production.add_argument("--title", required=True)
    init_production.add_argument("--slug", default="")
    init_production.add_argument("--date", default="")
    init_production.add_argument("--target-dir", default="")
    init_production.add_argument("--pipeline-file", default="")
    init_production.add_argument("--force", action="store_true")

    step = subparsers.add_parser("generate-step", help="Generate one artifact.")
    add_context_args(step)
    step.add_argument("--step-id", default="manual")
    step.add_argument("--format", choices=["markdown", "svg", "html"], required=True)
    step.add_argument("--prompt-file", required=True)
    step.add_argument("--input", action="append", default=[], help="Input file. Repeatable.")
    step.add_argument("--output", required=True)
    step.add_argument("--force", action="store_true")
    step.add_argument("--dry-run", action="store_true")

    run = subparsers.add_parser("run", help="Run a TOML pipeline plan.")
    add_context_args(run)
    run.add_argument("--pipeline", required=True)
    run.add_argument("--target-dir", default="")
    run.add_argument("--force", action="store_true")
    run.add_argument("--dry-run", action="store_true")

    render_prompt = subparsers.add_parser(
        "render-prompt",
        help="Render one step prompt after template substitution.",
    )
    add_context_args(render_prompt)
    render_prompt.add_argument("--pipeline", required=True)
    render_prompt.add_argument("--step-id", required=True)
    render_prompt.add_argument("--target-dir", default="")

    subparsers.add_parser(
        "sync-model-registry",
        help="Scan existing productions and rebuild the developed-model registry.",
    )
    subparsers.add_parser(
        "sync-steadyburn-verb-index",
        help="Scan the SteadyBurn series and rebuild the numbered verb index.",
    )

    return parser


def add_context_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--title", default="")
    parser.add_argument("--slug", default="")
    parser.add_argument("--date", default="")


def apply_context_overrides(
    spec: PipelineSpec,
    *,
    title: str,
    slug: str,
    date: str,
    target_dir: str,
    variables: dict[str, str],
) -> PipelineSpec:
    values = spec.model_dump()
    if title:
        values["context"]["title"] = title
    if slug:
        values["context"]["slug"] = slug
    if date:
        values["context"]["date"] = date
    if target_dir:
        values["context"]["target_dir"] = target_dir
    values["variables"] = {**values.get("variables", {}), **variables}
    return PipelineSpec.model_validate(values)


def build_pipeline(
    args: argparse.Namespace,
    cwd: Path,
    files: LocalFileStore,
    spec: PipelineSpec | None = None,
) -> BurnPipeline:
    providers = resolve_provider_configs(args, spec=spec)
    registry_path = Path(args.registry_file)
    registry = load_model_registry(files, registry_path)
    return BurnPipeline(
        files=files,
        inference_factory=lambda provider: build_inference(provider, cwd),
        providers=providers,
        registry=registry,
        registry_path=registry_path,
    )


def resolve_provider_configs(
    args: argparse.Namespace,
    *,
    spec: PipelineSpec | None,
) -> dict[GenerationModality, ProviderConfig]:
    defaults = {
        GenerationModality.TEXT: ProviderConfig(
            kind=ProviderKind(args.provider),
            model=args.model,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
            command=args.codex_bin,
        )
    }
    if args.image_provider or args.image_model or args.image_base_url or args.image_api_key_env or args.image_codex_bin:
        defaults[GenerationModality.IMAGE] = ProviderConfig(
            kind=ProviderKind(args.image_provider or args.provider),
            model=args.image_model,
            base_url=args.image_base_url,
            api_key_env=args.image_api_key_env or args.api_key_env,
            command=args.image_codex_bin or args.codex_bin,
        )
    if args.audio_provider or args.audio_model or args.audio_base_url or args.audio_api_key_env or args.audio_codex_bin:
        defaults[GenerationModality.AUDIO] = ProviderConfig(
            kind=ProviderKind(args.audio_provider or args.provider),
            model=args.audio_model,
            base_url=args.audio_base_url,
            api_key_env=args.audio_api_key_env or args.api_key_env,
            command=args.audio_codex_bin or args.codex_bin,
        )
    if spec is None:
        return defaults

    resolved = dict(defaults)
    for modality, configured in spec.providers.items():
        resolved[modality] = merge_provider_config(defaults.get(modality), configured)
    if GenerationModality.IMAGE not in resolved and GenerationModality.TEXT in resolved:
        resolved[GenerationModality.IMAGE] = resolved[GenerationModality.TEXT]
    if GenerationModality.AUDIO not in resolved and GenerationModality.TEXT in resolved:
        resolved[GenerationModality.AUDIO] = resolved[GenerationModality.TEXT]
    return resolved


def merge_provider_config(
    base: ProviderConfig | None,
    override: ProviderConfig,
) -> ProviderConfig:
    if base is None:
        return override
    values = base.model_dump()
    for key, value in override.model_dump().items():
        if value is not None:
            values[key] = value
    return ProviderConfig.model_validate(values)


def print_step_plan(step_id: str, inputs: list[str], output: str) -> None:
    print(f"Step: {step_id}")
    if inputs:
        print(f"Inputs: {', '.join(inputs)}")
    else:
        print("Inputs: none")
    print(f"Output: {output}")


def parse_key_value_pairs(pairs: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Expected key=value, got: {pair}")
        key, value = pair.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def describe_input(input_source) -> str:
    if input_source.step:
        return f"step:{input_source.step}"
    if input_source.path:
        return str(input_source.path)
    return "unknown"


def scaffold_production(
    *,
    files: LocalFileStore,
    cwd: Path,
    title: str,
    slug: str,
    date_value: str,
    target_dir: Path | None,
    pipeline_file: Path | None,
    force: bool,
) -> list[Path]:
    resolved_slug = slug or slugify(title)
    resolved_date = date_value or default_letter_date().isoformat()
    target = target_dir or Path("content/letters") / f"{resolved_date}-{resolved_slug}"
    pipeline_path = pipeline_file or Path("automation/pipelines") / f"{resolved_date}-{resolved_slug}.toml"

    created: list[Path] = []
    files.write_text(
        target / "PRODUCTION_BRIEF.md",
        build_brief_template(title=title, slug=resolved_slug, date_value=resolved_date),
        force=force,
    )
    created.append((cwd / target / "PRODUCTION_BRIEF.md").resolve())
    files.write_text(
        target / "index.md",
        build_index_draft(title=title, slug=resolved_slug, date_value=resolved_date),
        force=force,
    )
    created.append((cwd / target / "index.md").resolve())
    files.write_text(
        pipeline_path,
        build_pipeline_template(
            title=title,
            slug=resolved_slug,
            date_value=resolved_date,
            target_dir=target,
        ),
        force=force,
    )
    created.append((cwd / pipeline_path).resolve())
    return created


def slugify(value: str) -> str:
    slug = []
    previous_dash = False
    for char in value.lower():
        if char.isalnum():
            slug.append(char)
            previous_dash = False
            continue
        if not previous_dash:
            slug.append("-")
            previous_dash = True
    return "".join(slug).strip("-") or "steadyburn-letter"


def default_letter_date() -> date:
    today = date.today()
    if today.weekday() == 4:
        return today
    days_until_friday = (4 - today.weekday()) % 7
    return today + timedelta(days=days_until_friday or 7)


def build_brief_template(*, title: str, slug: str, date_value: str) -> str:
    return f"""# Production Brief

- Title: {title}
- Slug: {slug}
- Date: {date_value}

## Core Tension

Write the main conflict, friction, or pattern this letter needs to confront.

## Reader

Describe who this is for, what season he is in, and what he tends to get wrong.

## Promise

State the change or payoff this production should deliver.

## Useful Scenes Or Stories

List concrete examples, images, memories, or situations worth using.

## Worksheet Outcome

Describe what the worksheet should help the reader see, name, decide, or do.

## Language To Use

Capture exact phrases, model names, Scripture, or lines that should shape the production.

## Language To Avoid

Call out tones, cliches, and angles that would weaken the piece.

## Constraints

Note any non-negotiables, required sections, deadlines, or asset constraints.

## Raw Notes

Paste unstructured inspiration, voice notes, transcripts, bullets, or fragments here.
"""


def build_index_draft(*, title: str, slug: str, date_value: str) -> str:
    return f"""---
date: {date_value}
slug: "{slug}"
title: "{title}"
summary: "SUMMARY GOES HERE"

series:
  - SteadyBurn

tags:
  - tag1
  - tag2

cover:
  image: "cover.png"
  relative: true

draft: false
---

{{{{< audio >}}}}

Write a rough public draft here, or leave this file as a placeholder and let the pipeline rebuild it later.
"""


def build_pipeline_template(*, title: str, slug: str, date_value: str, target_dir: Path) -> str:
    target_dir_string = target_dir.as_posix()
    return f"""[context]
title = "{escape_toml_string(title)}"
slug = "{escape_toml_string(slug)}"
date = "{date_value}"
target_dir = "{escape_toml_string(target_dir_string)}"

[providers.text]
kind = "openai-compatible"
base_url = "http://localhost:11434"
model = "active"

[providers.image]
kind = "openai-compatible"
base_url = "http://localhost:11434"
model = "IMAGE_MODEL_NAME"

[providers.audio]
kind = "openai-compatible"
base_url = "http://localhost:11434"
model = "AUDIO_MODEL_NAME"

[variables]
voice = "direct, pragmatic, masculine, concrete"
audience = "men who need structure, ownership, and steady action"

[[steps]]
id = "lesson"
format = "markdown"
prompt_file = "automation/prompts/burn/lesson.md"
output = "{target_dir_string}/LESSON.md"

[[steps.inputs]]
path = "{target_dir_string}/PRODUCTION_BRIEF.md"
alias = "brief"

[[steps.inputs]]
path = "{target_dir_string}/index.md"
alias = "draft_index"

[[steps]]
id = "instructions"
format = "markdown"
prompt_file = "automation/prompts/burn/instructions.md"
depends_on = ["lesson"]
output = "{target_dir_string}/INSTRUCTIONS.md"

[[steps.inputs]]
step = "lesson"
alias = "lesson"

[[steps.inputs]]
path = "{target_dir_string}/PRODUCTION_BRIEF.md"
alias = "brief"

[[steps]]
id = "context"
format = "markdown"
prompt_file = "automation/prompts/burn/context.md"
depends_on = ["lesson"]
output = "{target_dir_string}/CONTEXT.md"

[[steps.inputs]]
step = "lesson"
alias = "lesson"

[[steps.inputs]]
path = "{target_dir_string}/PRODUCTION_BRIEF.md"
alias = "brief"

[[steps]]
id = "gpt"
format = "markdown"
prompt_file = "automation/prompts/burn/gpt.md"
depends_on = ["context", "instructions"]
output = "{target_dir_string}/GPT.md"

[[steps.inputs]]
step = "context"
alias = "context"

[[steps.inputs]]
step = "instructions"
alias = "instructions"

[[steps]]
id = "worksheet"
format = "svg"
prompt_file = "automation/prompts/burn/worksheet-svg.md"
depends_on = ["lesson", "instructions"]
output = "{target_dir_string}/WORKSHEET.svg"

[[steps.inputs]]
step = "lesson"
alias = "lesson"

[[steps.inputs]]
step = "instructions"
alias = "instructions"

[[steps]]
id = "worksheet_masked"
format = "svg"
prompt_file = "automation/prompts/burn/worksheet-masked-svg.md"
depends_on = ["worksheet"]
output = "{target_dir_string}/WORKSHEET_MASKED.svg"

[[steps.inputs]]
step = "worksheet"
alias = "worksheet"

[[steps]]
id = "promo"
format = "markdown"
prompt_file = "automation/prompts/burn/promo-image.md"
depends_on = ["context", "worksheet_masked"]
output = "{target_dir_string}/PROMO_PROMPT.md"

[[steps.inputs]]
step = "context"
alias = "context"

[[steps.inputs]]
step = "worksheet_masked"
alias = "worksheet_masked"

[[steps]]
id = "banner"
format = "markdown"
prompt_file = "automation/prompts/burn/banner-image.md"
depends_on = ["context", "worksheet_masked", "promo"]
output = "{target_dir_string}/BANNER_PROMPT.md"

[[steps.inputs]]
step = "context"
alias = "context"

[[steps.inputs]]
step = "worksheet_masked"
alias = "worksheet_masked"

[[steps.inputs]]
step = "promo"
alias = "promo"

[[steps]]
id = "page_copy"
format = "markdown"
prompt_file = "automation/prompts/burn/page-copy.md"
depends_on = ["context", "promo", "banner"]
output = "{target_dir_string}/PAGE_COPY.md"

[[steps.inputs]]
step = "context"
alias = "context"

[[steps.inputs]]
step = "promo"
alias = "promo"

[[steps.inputs]]
step = "banner"
alias = "banner"

[[steps]]
id = "landing_page"
format = "html"
prompt_file = "automation/prompts/burn/landing-page-html.md"
depends_on = ["context", "banner", "page_copy"]
output = "{target_dir_string}/LANDING_PAGE.html"

[[steps.inputs]]
step = "context"
alias = "context"

[[steps.inputs]]
step = "banner"
alias = "banner"

[[steps.inputs]]
step = "page_copy"
alias = "page_copy"

[[steps]]
id = "worksheet_page"
format = "markdown"
prompt_file = "automation/prompts/burn/worksheet-page-prototype.md"
depends_on = ["context", "promo", "page_copy"]
output = "{target_dir_string}/WORKSHEET_PAGE.md"

[[steps.inputs]]
step = "context"
alias = "context"

[[steps.inputs]]
step = "promo"
alias = "promo"

[[steps.inputs]]
step = "page_copy"
alias = "page_copy"

[[steps]]
id = "index"
format = "markdown"
prompt_file = "automation/prompts/burn/index.md"
depends_on = ["lesson", "context"]
output = "{target_dir_string}/index.md"

[[steps.inputs]]
path = "{target_dir_string}/index.md"
alias = "draft_index"

[[steps.inputs]]
step = "lesson"
alias = "lesson"

[[steps.inputs]]
step = "context"
alias = "context"

[[steps.inputs]]
path = "{target_dir_string}/PRODUCTION_BRIEF.md"
alias = "brief"

[[steps]]
id = "newsletter_email"
format = "markdown"
prompt_file = "automation/prompts/burn/newsletter-email.md"
depends_on = ["index", "lesson", "instructions", "gpt", "context"]
output = "{target_dir_string}/NEWSLETTER_EMAIL.md"

[[steps.inputs]]
step = "index"
alias = "index"

[[steps.inputs]]
step = "lesson"
alias = "lesson"

[[steps.inputs]]
step = "instructions"
alias = "instructions"

[[steps.inputs]]
step = "gpt"
alias = "gpt"

[[steps.inputs]]
step = "context"
alias = "context"

[[steps]]
id = "community_post"
format = "markdown"
prompt_file = "automation/prompts/burn/community-post.md"
depends_on = ["index", "context"]
output = "{target_dir_string}/COMMUNITY_POST.md"

[[steps.inputs]]
step = "index"
alias = "index"

[[steps.inputs]]
step = "context"
alias = "context"
"""


def escape_toml_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


if __name__ == "__main__":
    main()
