#!/usr/bin/env python
"""Compare OpenRouter models end to end from one seed, with one draft PR per model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import yaml
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    args = build_parser().parse_args()
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY must be set.")
    root = Path(git_output("rev-parse", "--show-toplevel"))
    if git_output("status", "--porcelain"):
        raise SystemExit("Commit or stash source-checkout changes before running comparisons.")
    seed = Path(args.seed_file).resolve()
    if not seed.is_file():
        raise SystemExit(f"Seed file not found: {seed}")
    base_ref = git_output("rev-parse", "--verify", args.base_ref)
    models = list(dict.fromkeys(args.models))
    worktree_root = Path(args.worktree_root or root.parent / f"{root.name}-seed-comparisons")

    for model in models:
        branch = branch_name(args.branch_prefix, args.date, model)
        worktree = worktree_root / branch.replace("/", "--")
        print(f"\nModel: {model}\nBranch: {branch}\nWorktree: {worktree}")
        if args.dry_run:
            continue
        create_worktree(worktree, branch, base_ref)
        target = prepare_production(worktree, seed, args, model)
        usage_log = target / "OPENROUTER_USAGE.jsonl"
        environment = {**os.environ, "BURN_USAGE_LOG": str(usage_log)}
        generate_context(worktree, target, args, model, environment)
        pipeline = write_comparison_pipeline(worktree, target, model, args.image_model)
        run([sys.executable, "scripts/burn-pipeline.py", "run", "--pipeline", str(pipeline), "--force"], cwd=worktree, env=environment)
        write_manifest(target, model, base_ref, usage_log)
        run_git(worktree, "add", "--", target.relative_to(worktree))
        run_git(worktree, "commit", "-m", f"Compare {model} content output")
        run_git(worktree, "push", "-u", "origin", branch)
        run(["gh", "pr", "create", "--draft", "--base", args.base_branch, "--head", branch,
             "--title", f"Compare {model} content output",
             "--body", f"End-to-end OpenRouter comparison for `{model}`. Includes provider-reported usage and cost."], cwd=worktree)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-file", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--image-model", default="openai/gpt-5.4-image-2")
    parser.add_argument("--base-ref", default="HEAD")
    parser.add_argument("--base-branch", default="main")
    parser.add_argument("--branch-prefix", default="content-model-compare")
    parser.add_argument("--worktree-root")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def branch_name(prefix: str, date: str, model: str) -> str:
    safe = re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")[:48]
    return f"{prefix}/{date}/{safe}-{hashlib.sha256(model.encode()).hexdigest()[:8]}"


def create_worktree(worktree: Path, branch: str, base_ref: str) -> None:
    if worktree.exists():
        raise SystemExit(f"Worktree already exists: {worktree}")
    if run(["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"], check=False).returncode == 0:
        raise SystemExit(f"Branch already exists: {branch}")
    worktree.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "worktree", "add", "-b", branch, str(worktree), base_ref])


def prepare_production(worktree: Path, seed: Path, args: argparse.Namespace, model: str) -> Path:
    run([sys.executable, "scripts/burn-pipeline.py", "init-production", "--title", args.title,
         "--slug", args.slug, "--date", args.date, "--force"], cwd=worktree)
    target = worktree / "content" / "letters" / f"{args.date}-{args.slug}"
    shutil.copyfile(seed, target / "SEED.md")
    return target


def generate_context(worktree: Path, target: Path, args: argparse.Namespace, model: str, environment: dict[str, str]) -> None:
    relative = target.relative_to(worktree)
    run([sys.executable, "scripts/burn-pipeline.py", "--provider", "openrouter", "--model", model,
         "--registry-file", "tmp/model-comparisons/registry.json", "generate-step", "--step-id", "context",
         "--format", "markdown", "--prompt-file", "automation/prompts/burn/context.md",
         "--input", str(relative / "SEED.md"), "--output", str(relative / "CONTEXT.md"),
         "--title", args.title, "--slug", args.slug, "--date", args.date, "--force"], cwd=worktree, env=environment)


def write_comparison_pipeline(worktree: Path, target: Path, model: str, image_model: str) -> Path:
    source = target / "pipeline.yaml"
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "providers" not in data:
        raise SystemExit(f"Pipeline has no providers block: {source}")
    data["providers"]["text"] = {
        "kind": "openrouter",
        "providerUrl": "https://openrouter.ai/api/v1",
        "model": model,
    }
    data["providers"]["image"] = {
        "kind": "openrouter",
        "providerUrl": "https://openrouter.ai/api/v1",
        "model": image_model,
    }
    for step in data.get("steps", []):
        if step.get("modality", "text") == "text":
            step["model"] = model
        elif step.get("modality") == "image":
            step["model"] = image_model
    destination = worktree / "tmp" / "model-comparisons" / hashlib.sha256(model.encode()).hexdigest()[:12] / "pipeline.yaml"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=False), encoding="utf-8")
    return destination.relative_to(worktree)


def write_manifest(target: Path, model: str, base_ref: str, usage_log: Path) -> None:
    records = [json.loads(line) for line in usage_log.read_text(encoding="utf-8").splitlines()] if usage_log.exists() else []
    total = sum(float(record.get("usage", {}).get("cost", 0) or 0) for record in records)
    lines = ["# Model Comparison Run", "", f"- Model: `{model}`", "- Provider: OpenRouter",
             f"- Base revision: `{base_ref}`", f"- Run at (UTC): `{datetime.now(timezone.utc).isoformat()}`",
             f"- Provider-reported total cost: `${total:.6f}`", "", "## Usage Records", ""]
    for record in records:
        usage = record["usage"]
        lines.append(f"- `{record['endpoint']}` / `{record['model']}`: `${float(usage.get('cost', 0) or 0):.6f}` "
                     f"({usage.get('prompt_tokens', 0)} input, {usage.get('completion_tokens', 0)} output tokens)")
    (target / "MODEL_COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def git_output(*args: str) -> str:
    return run(["git", *args], capture_output=True).stdout.strip()


def run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None, check: bool = True, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, env=env, text=True, check=check, capture_output=capture_output)


def run_git(worktree: Path, *args: str) -> None:
    run(["git", *args], cwd=worktree)


if __name__ == "__main__":
    main()
