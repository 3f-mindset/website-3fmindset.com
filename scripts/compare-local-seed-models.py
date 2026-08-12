#!/usr/bin/env python3
"""Run the fixed editorial seed through the complete local text-model cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


def run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, cwd=cwd, env=env, check=True, text=True)


def git_output(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, check=True, text=True, capture_output=True).stdout.strip()


def safe_model_dir(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")[:80]


def branch_name(date: str, model: str) -> str:
    return f"local-content-compare/{date}/{safe_model_dir(model)}-{hashlib.sha256(model.encode()).hexdigest()[:8]}"


def configure_pipeline(source: Path, destination: Path, model: str, endpoint: str, image_model: str) -> None:
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    data["providers"]["text"] = {
        "kind": "openai-compatible", "providerUrl": endpoint, "model": model,
        "timeout_seconds": 900, "retry_attempts": 3, "retry_wait_seconds": 10,
    }
    data["providers"]["image"] = {
        "kind": "openai-compatible", "providerUrl": endpoint, "model": image_model,
        "timeout_seconds": 900, "retry_attempts": 3, "retry_wait_seconds": 10,
    }
    for step in data.get("steps", []):
        step["model"] = image_model if step.get("modality") == "image" else model
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=False), encoding="utf-8")


def copy_bundle(source: Path, target: Path, model: str, endpoint: str, generated_pipeline: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    names = (
        "SEED.md", "CONTEXT.md", "LESSON.md", "INSTRUCTIONS.md", "GPT.md",
        "WORKSHEET.svg", "WORKSHEET_MASKED.svg", "COVER_PROMPT.md", "cover.png",
        "index.md", "NEWSLETTER_EMAIL.md", "COMMUNITY_POST.md",
    )
    for name in names:
        source_file = source / name
        if source_file.exists():
            shutil.copy2(source_file, target / name)
    shutil.copy2(generated_pipeline, target / "pipeline.yaml")
    manifest = {
        "model": model,
        "provider": "local llama-swap",
        "endpoint": endpoint,
        "text_model": model,
        "image_model": "fixed local image model",
        "source": "fixed-seed editorial cross-run",
    }
    (target / "MODEL_COMPARISON.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (target / "MODEL_COMPARISON.md").write_text(
        "# Local editorial cross-run\n\n"
        f"- Text model: `{model}`\n"
        "- Provider: local llama-swap\n"
        f"- Endpoint: `{endpoint}`\n"
        "- Image model: fixed local image model\n"
        "- Workload: the fixed 2026-08-07 editorial seed and full text pipeline\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-file", required=True, type=Path)
    parser.add_argument("--research-root", required=True, type=Path)
    parser.add_argument("--title", required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--provider-url", default=os.environ.get("LOCAL_AI_BASE_URL", "http://titan:11434"))
    parser.add_argument("--image-model", default="unsloth-qwen-image-2512-gguf-qwen-image-2512-q4-k-m")
    parser.add_argument("--worktree-root", type=Path)
    args = parser.parse_args()

    root = Path(git_output(Path.cwd(), "rev-parse", "--show-toplevel"))
    # Worktrees are created from HEAD, so unrelated dirty files in the source
    # checkout are intentionally left untouched while the cross-run executes.
    seed = args.seed_file.resolve()
    research = args.research_root.resolve()
    if not seed.is_file():
        raise SystemExit(f"Seed file not found: {seed}")
    base_ref = git_output(root, "rev-parse", "--verify", "HEAD")
    worktree_root = (args.worktree_root or root.parent / f"{root.name}-local-seed-comparisons").resolve()
    worktree_root.mkdir(parents=True, exist_ok=True)
    output_root = research / "docs" / "model-comparisons"

    for model in dict.fromkeys(args.models):
        branch = branch_name(args.date, model)
        worktree = worktree_root / branch.replace("/", "--")
        if worktree.exists():
            raise SystemExit(f"Worktree already exists: {worktree}")
        run(["git", "worktree", "add", "-b", branch, str(worktree), base_ref], cwd=root)
        try:
            run([sys.executable, "scripts/burn-pipeline.py", "init-production", "--title", args.title,
                 "--slug", args.slug, "--date", args.date, "--force"], cwd=worktree)
            content_dir = worktree / "content" / "letters" / f"{args.date}-{args.slug}"
            shutil.copy2(seed, content_dir / "SEED.md")
            env = {**os.environ, "BURN_MAX_TOKENS": os.environ.get("BURN_MAX_TOKENS", "4096")}
            run([sys.executable, "scripts/burn-pipeline.py", "--provider", "openai-compatible",
                 "--provider-url", args.provider_url, "--model", model,
                 "generate-step", "--step-id", "context", "--format", "markdown",
                 "--prompt-file", "automation/prompts/burn/context.md",
                 "--input", str(content_dir.relative_to(worktree) / "SEED.md"),
                 "--output", str(content_dir.relative_to(worktree) / "CONTEXT.md"),
                 "--title", args.title, "--slug", args.slug, "--date", args.date, "--force"],
                cwd=worktree, env=env)
            generated_pipeline = worktree / "tmp" / "local-seed-comparisons" / f"{safe_model_dir(model)}.yaml"
            configure_pipeline(content_dir / "pipeline.yaml", generated_pipeline, model, args.provider_url, args.image_model)
            run([sys.executable, "scripts/burn-pipeline.py", "--provider", "openai-compatible",
                 "--provider-url", args.provider_url, "--model", model,
                 "--image-provider", "openai-compatible", "--image-provider-url", args.provider_url,
                 "--image-model", args.image_model, "run", "--pipeline", str(generated_pipeline), "--force"],
                cwd=worktree, env=env)
            target = output_root / safe_model_dir(model) / f"{args.date}-{args.slug}"
            copy_bundle(content_dir, target, model, args.provider_url, generated_pipeline)
            print(f"Published local editorial bundle: {model} -> {target}", flush=True)
        finally:
            run(["git", "worktree", "remove", "--force", str(worktree)], cwd=root)
            subprocess.run(["git", "branch", "-D", branch], cwd=root, check=False, text=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
