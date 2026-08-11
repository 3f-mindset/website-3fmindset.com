#!/usr/bin/env python
"""Run isolated OpenRouter comparisons of a content pipeline, one Git branch per model."""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
import yaml
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_PIPELINE = (
    "content/letters/2026-08-07-master-your-tasks-prioritization-and-time-management/pipeline.yaml"
)
DEFAULT_BRANCH_PREFIX = "content-model-compare"


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(git_output("rev-parse", "--show-toplevel"))
    pipeline = resolve_repo_path(repo_root, args.pipeline)
    pipeline_relative = pipeline.relative_to(repo_root)
    pipeline_data = load_pipeline(pipeline)
    target_dir = pipeline_data["context"]["target_dir"]
    models = expand_models(args.models)
    base_ref = git_output("rev-parse", "--verify", args.base_ref)

    if args.open_pr and not args.push:
        raise SystemExit("--open-pr requires --push")
    if not args.dry_run and not args.allow_dirty and git_output("status", "--porcelain"):
        raise SystemExit(
            "The source checkout has uncommitted changes. Commit the provider and harness first, "
            "or rerun with --allow-dirty after confirming --base-ref contains them."
        )
    if not args.dry_run and "OPENROUTER_API_KEY" not in __import__("os").environ:
        raise SystemExit("OPENROUTER_API_KEY must be set before running model comparisons.")

    worktree_root = Path(args.worktree_root) if args.worktree_root else repo_root.parent / f"{repo_root.name}-model-comparisons"
    steps = select_steps(pipeline_data, include_non_text=args.include_non_text)

    for model in models:
        branch = comparison_branch(args.branch_prefix, pipeline_data["context"].get("date", "undated"), model)
        worktree = worktree_root / branch.replace("/", "--")
        print(f"\nModel: {model}\nBranch: {branch}\nWorktree: {worktree}")
        if args.dry_run:
            print(f"Would run steps: {', '.join(steps)}")
            continue

        ensure_new_worktree(branch, worktree, base_ref)
        comparison_pipeline = write_openrouter_pipeline(worktree, pipeline_relative, model)
        run_pipeline_steps(worktree, comparison_pipeline, model, steps)
        write_run_manifest(worktree, target_dir, model, base_ref, pipeline_relative, steps)
        changed = git_output("status", "--porcelain", "--", target_dir, cwd=worktree)
        if not changed:
            print("No generated-content changes; no commit or PR was created.")
            continue
        if args.commit:
            run_git(worktree, "add", "--", target_dir)
            run_git(worktree, "commit", "-m", f"Compare {model} content output")
            if args.push:
                run_git(worktree, "push", "-u", "origin", branch)
            if args.open_pr:
                create_draft_pr(worktree, branch, model, target_dir)
        else:
            print("Generated changes are ready in the worktree. Re-run with --commit to prepare a PR.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True, help="OpenRouter model IDs; commas are also accepted.")
    parser.add_argument("--pipeline", default=DEFAULT_PIPELINE, help="Pipeline YAML to compare.")
    parser.add_argument("--base-ref", default="HEAD", help="Committed revision from which each comparison branch starts.")
    parser.add_argument("--branch-prefix", default=DEFAULT_BRANCH_PREFIX)
    parser.add_argument("--worktree-root", help="Directory that will contain the persistent comparison worktrees.")
    parser.add_argument("--include-non-text", action="store_true", help="Also run image/audio steps when the selected provider supports them.")
    parser.add_argument("--commit", action="store_true", help="Commit each model's generated changes on its comparison branch.")
    parser.add_argument("--push", action="store_true", help="Push committed comparison branches to origin.")
    parser.add_argument("--open-pr", action="store_true", help="Open a draft PR for every pushed comparison branch.")
    parser.add_argument("--dry-run", action="store_true", help="Show isolated branches and selected steps without creating worktrees or calling a model.")
    parser.add_argument("--allow-dirty", action="store_true", help="Permit execution from a dirty source checkout; changes are still excluded from worktrees.")
    return parser


def resolve_repo_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def load_pipeline(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"Pipeline not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if "context" not in data or "target_dir" not in data["context"]:
        raise SystemExit(f"Pipeline must define context.target_dir: {path}")
    return data


def expand_models(values: list[str]) -> list[str]:
    models = [model.strip() for value in values for model in value.split(",") if model.strip()]
    if not models:
        raise SystemExit("At least one model is required.")
    return list(dict.fromkeys(models))


def comparison_branch(prefix: str, date: str, model: str) -> str:
    safe_model = re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")[:48] or "model"
    digest = hashlib.sha256(model.encode("utf-8")).hexdigest()[:8]
    return f"{prefix.strip('/')}/{date}/{safe_model}-{digest}"


def select_steps(pipeline: dict, *, include_non_text: bool) -> list[str]:
    steps = pipeline.get("steps", [])
    selected = [step["id"] for step in steps if include_non_text or step.get("modality", "text") == "text"]
    if not selected:
        raise SystemExit("No matching pipeline steps were found.")
    return selected


def ensure_new_worktree(branch: str, worktree: Path, base_ref: str) -> None:
    if worktree.exists():
        raise SystemExit(f"Worktree already exists: {worktree}. Review it or choose a different --worktree-root.")
    existing = run(
        ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"],
        check=False,
    ).returncode == 0
    if existing:
        raise SystemExit(f"Branch already exists: {branch}. Review it or choose a different --branch-prefix.")
    worktree.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "worktree", "add", "-b", branch, str(worktree), base_ref])


def write_openrouter_pipeline(worktree: Path, source_relative: Path, model: str) -> Path:
    source = worktree / source_relative
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "providers" not in data or "text" not in data["providers"]:
        raise SystemExit(f"Pipeline has no [providers.text] block: {source_relative}")
    data["providers"]["text"] = {
        "kind": "openrouter",
        "model": model,
        "providerUrl": "https://openrouter.ai/api/v1",
        "timeout_seconds": 300,
        "retry_attempts": 4,
        "retry_wait_seconds": 75,
    }
    for step in data.get("steps", []):
        if step.get("modality", "text") == "text":
            step["model"] = model
    destination = worktree / "tmp" / "model-comparisons" / comparison_branch("run", "pipeline", model) / "pipeline.yaml"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=False), encoding="utf-8")
    return destination.relative_to(worktree)


def run_pipeline_steps(worktree: Path, pipeline: Path, model: str, steps: list[str]) -> None:
    for step in steps:
        print(f"Running {model}: {step}")
        run(
            [
                sys.executable,
                "scripts/burn-pipeline.py",
                "--provider",
                "openrouter",
                "--model",
                model,
                "run",
                "--pipeline",
                str(pipeline),
                "--step-id",
                step,
                "--force",
            ],
            cwd=worktree,
        )


def write_run_manifest(worktree: Path, target_dir: str, model: str, base_ref: str, pipeline: Path, steps: list[str]) -> None:
    manifest = worktree / target_dir / "MODEL_COMPARISON.md"
    manifest.write_text(
        "# Model Comparison Run\n\n"
        f"- Model: `{model}`\n"
        "- Provider: OpenRouter\n"
        "- Endpoint: `https://openrouter.ai/api/v1`\n"
        f"- Base revision: `{base_ref}`\n"
        f"- Source pipeline: `{pipeline.as_posix()}`\n"
        f"- Run at (UTC): `{datetime.now(timezone.utc).isoformat()}`\n"
        f"- Generated steps: {', '.join(f'`{step}`' for step in steps)}\n",
        encoding="utf-8",
    )


def create_draft_pr(worktree: Path, branch: str, model: str, target_dir: str) -> None:
    body = (
        f"OpenRouter model comparison for `{model}`.\n\n"
        f"Review the generated outputs and `{target_dir}/MODEL_COMPARISON.md`."
    )
    run(["gh", "pr", "create", "--draft", "--head", branch, "--title", f"Compare {model} content output", "--body", body], cwd=worktree)


def git_output(*args: str, cwd: Path | None = None, check: bool = True) -> str:
    completed = run(["git", *args], cwd=cwd, check=check, capture_output=True)
    return completed.stdout.strip()


def run(command: list[str], *, cwd: Path | None = None, check: bool = True, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, check=check, text=True, capture_output=capture_output)


def run_git(worktree: Path, *args: str) -> None:
    run(["git", *args], cwd=worktree)


if __name__ == "__main__":
    main()
