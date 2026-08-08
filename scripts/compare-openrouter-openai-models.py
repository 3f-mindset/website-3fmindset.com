#!/usr/bin/env python
"""Run an OpenRouter comparison of a content pipeline, one Git branch per OpenAI model."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_BRANCH_PREFIX = "content-model-compare"


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(git_output("rev-parse", "--show-toplevel"))
    pipeline = repo_root / args.pipeline
    spec = load_pipeline(pipeline)
    models = openai_models(args.models)
    base_ref = git_output("rev-parse", "--verify", args.base_ref)
    steps = text_steps(spec)
    target_dir = spec["context"]["target_dir"]
    worktree_root = Path(args.worktree_root) if args.worktree_root else repo_root.parent / f"{repo_root.name}-model-comparisons"

    if args.open_pr and not args.push:
        raise SystemExit("--open-pr requires --push")
    if not args.dry_run and git_output("status", "--porcelain"):
        raise SystemExit("Commit or stash source-checkout changes before running comparisons.")
    if not args.dry_run and not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY must be set before running comparisons.")

    for model in models:
        branch = comparison_branch(args.branch_prefix, spec["context"].get("date", "undated"), model)
        worktree = worktree_root / branch.replace("/", "--")
        print(f"\nModel: {model}\nBranch: {branch}\nWorktree: {worktree}")
        if args.dry_run:
            print(f"Would run text steps: {', '.join(steps)}")
            continue

        create_worktree(branch, worktree, base_ref)
        comparison_pipeline = write_openrouter_pipeline(worktree, args.pipeline, model)
        run_steps(worktree, comparison_pipeline, model, steps)
        write_manifest(worktree, target_dir, model, base_ref, args.pipeline, steps)
        if not git_output("status", "--porcelain", "--", target_dir, cwd=worktree):
            print("No generated-content changes; no commit or PR was created.")
            continue
        if not args.commit:
            print("Generated changes are ready in the worktree. Re-run with --commit to prepare a PR.")
            continue
        run_git(worktree, "add", "--", target_dir)
        run_git(worktree, "commit", "-m", f"Compare {model} content output")
        if args.push:
            run_git(worktree, "push", "-u", "origin", branch)
        if args.open_pr:
            print(f"Create a draft PR for {branch} against main after reviewing the generated diff.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True, help="OpenRouter OpenAI model IDs, e.g. openai/gpt-4.1.")
    parser.add_argument("--pipeline", required=True, help="Committed pipeline TOML to compare.")
    parser.add_argument("--base-ref", default="HEAD", help="Committed revision used for every comparison branch.")
    parser.add_argument("--branch-prefix", default=DEFAULT_BRANCH_PREFIX)
    parser.add_argument("--worktree-root", help="Persistent directory for comparison worktrees.")
    parser.add_argument("--commit", action="store_true", help="Commit each generated model comparison.")
    parser.add_argument("--push", action="store_true", help="Push committed branches to origin.")
    parser.add_argument("--open-pr", action="store_true", help="Print the draft-PR action after each pushed branch.")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def load_pipeline(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"Pipeline not found: {path}")
    spec = tomllib.loads(path.read_text(encoding="utf-8"))
    if "context" not in spec or not spec["context"].get("target_dir"):
        raise SystemExit("Pipeline must define context.target_dir.")
    return spec


def openai_models(values: list[str]) -> list[str]:
    models = [model.strip() for value in values for model in value.split(",") if model.strip()]
    invalid = [model for model in models if not model.startswith("openai/")]
    if invalid:
        raise SystemExit("This harness is restricted to OpenAI models via OpenRouter: " + ", ".join(invalid))
    return list(dict.fromkeys(models))


def text_steps(spec: dict) -> list[str]:
    steps = [step["id"] for step in spec.get("steps", []) if step.get("modality", "text") == "text"]
    if not steps:
        raise SystemExit("The pipeline has no text steps.")
    return steps


def comparison_branch(prefix: str, date: str, model: str) -> str:
    safe_model = re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")[:48]
    digest = hashlib.sha256(model.encode("utf-8")).hexdigest()[:8]
    return f"{prefix.strip('/')}/{date}/{safe_model}-{digest}"


def create_worktree(branch: str, worktree: Path, base_ref: str) -> None:
    if worktree.exists():
        raise SystemExit(f"Worktree already exists: {worktree}")
    exists = run(["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"], check=False).returncode == 0
    if exists:
        raise SystemExit(f"Branch already exists: {branch}")
    worktree.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "worktree", "add", "-b", branch, str(worktree), base_ref])


def write_openrouter_pipeline(worktree: Path, pipeline_path: str, model: str) -> Path:
    source = worktree / pipeline_path
    content = source.read_text(encoding="utf-8")
    provider = (
        "[providers.text]\n"
        'kind = "openrouter"\n'
        f'model = "{model}"\n'
        'base_url = "https://openrouter.ai/api/v1"\n'
        'api_key_env = "OPENROUTER_API_KEY"\n'
        "timeout_seconds = 300\nretry_attempts = 4\nretry_wait_seconds = 75\n\n"
    )
    replacement = re.compile(r"(?ms)^\[providers\.text\]\n.*?(?=^\[|\Z)").sub(provider, content, count=1)
    if replacement == content:
        raise SystemExit(f"Pipeline has no [providers.text] block: {pipeline_path}")
    destination = worktree / "tmp" / "model-comparisons" / hashlib.sha256(model.encode()).hexdigest()[:12] / "pipeline.toml"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(replacement, encoding="utf-8")
    return destination.relative_to(worktree)


def run_steps(worktree: Path, pipeline: Path, model: str, steps: list[str]) -> None:
    for step in steps:
        print(f"Running {model}: {step}")
        run([sys.executable, "scripts/burn-pipeline.py", "--provider", "openrouter", "--model", model, "run", "--pipeline", str(pipeline), "--step-id", step, "--force"], cwd=worktree)


def write_manifest(worktree: Path, target_dir: str, model: str, base_ref: str, pipeline: str, steps: list[str]) -> None:
    (worktree / target_dir / "MODEL_COMPARISON.md").write_text(
        "# Model Comparison Run\n\n"
        f"- Model: `{model}`\n- Provider: OpenRouter\n- Endpoint: `https://openrouter.ai/api/v1`\n"
        f"- Base revision: `{base_ref}`\n- Source pipeline: `{pipeline}`\n"
        f"- Run at (UTC): `{datetime.now(timezone.utc).isoformat()}`\n"
        f"- Generated steps: {', '.join(f'`{step}`' for step in steps)}\n",
        encoding="utf-8",
    )


def git_output(*args: str, cwd: Path | None = None) -> str:
    return run(["git", *args], cwd=cwd, capture_output=True).stdout.strip()


def run(command: list[str], *, cwd: Path | None = None, check: bool = True, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, check=check, text=True, capture_output=capture_output)


def run_git(worktree: Path, *args: str) -> None:
    run(["git", *args], cwd=worktree)


if __name__ == "__main__":
    main()
