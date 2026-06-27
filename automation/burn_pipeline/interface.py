from __future__ import annotations

import argparse
from pathlib import Path

from .application import BurnPipeline, step_from_paths
from .domain import BurnContext, PipelineSpec, ProviderConfig, ProviderKind
from .infrastructure import LocalFileStore, build_inference, load_pipeline_spec


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cwd = Path(args.cwd).resolve()
    files = LocalFileStore(cwd)

    if args.command == "generate-step":
        step = step_from_paths(
            step_id=args.step_id,
            output_format=args.format,
            prompt_file=Path(args.prompt_file),
            output=Path(args.output),
            inputs=[Path(value) for value in args.input],
        )
        if args.dry_run:
            print_step_plan(step.id, [str(path) for path in step.inputs], str(step.output))
            return
        pipeline = build_pipeline(args, cwd, files)
        content = pipeline.generate_step(
            step=step,
            context=BurnContext(title=args.title, slug=args.slug, date=args.date),
            force=args.force,
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
        )
        if args.dry_run:
            for step in spec.steps:
                print_step_plan(step.id, [str(path) for path in step.inputs], str(step.output))
            return
        pipeline = build_pipeline(args, cwd, files)
        pipeline.run_pipeline(spec=spec, force=args.force)
        print(f"Wrote {len(spec.steps)} pipeline step output(s)")
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
        default=ProviderKind.CODEX_CLI.value,
        help="Inference provider.",
    )
    parser.add_argument("--model", default=None, help="Model name for the selected provider.")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible API base URL.")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="API key environment variable.")
    parser.add_argument("--codex-bin", default="codex", help="Codex CLI executable.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    step = subparsers.add_parser("generate-step", help="Generate one artifact.")
    add_context_args(step)
    step.add_argument("--step-id", default="manual")
    step.add_argument("--format", choices=["markdown", "svg"], required=True)
    step.add_argument("--prompt-file", required=True)
    step.add_argument("--input", action="append", default=[], help="Input file. Repeatable.")
    step.add_argument("--output", required=True)
    step.add_argument("--force", action="store_true")
    step.add_argument("--dry-run", action="store_true")

    run = subparsers.add_parser("run", help="Run a TOML pipeline plan.")
    add_context_args(run)
    run.add_argument("--pipeline", required=True)
    run.add_argument("--force", action="store_true")
    run.add_argument("--dry-run", action="store_true")

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
) -> PipelineSpec:
    values = spec.model_dump()
    if title:
        values["context"]["title"] = title
    if slug:
        values["context"]["slug"] = slug
    if date:
        values["context"]["date"] = date
    return PipelineSpec.model_validate(values)


def build_pipeline(args: argparse.Namespace, cwd: Path, files: LocalFileStore) -> BurnPipeline:
    provider = ProviderConfig(
        kind=ProviderKind(args.provider),
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        command=args.codex_bin,
    )
    return BurnPipeline(files=files, inference=build_inference(provider, cwd))


def print_step_plan(step_id: str, inputs: list[str], output: str) -> None:
    print(f"Step: {step_id}")
    if inputs:
        print(f"Inputs: {', '.join(inputs)}")
    else:
        print("Inputs: none")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
