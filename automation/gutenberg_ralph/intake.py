from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml

from .domain import Intake, QualityPolicy, ROLES, Source


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    if path.is_file():
        hasher.update(path.read_bytes())
    else:
        for child in sorted(path.rglob("*")):
            if child.is_file() and ".ralph" not in child.parts:
                hasher.update(child.relative_to(path).as_posix().encode())
                hasher.update(child.read_bytes())
    return hasher.hexdigest()


def deterministic_alias(path: Path, root: Path | None = None) -> str:
    relative = path.resolve().relative_to((root or path.parent).resolve())
    alias = re.sub(r"[^a-zA-Z0-9]+", "-", relative.as_posix()).strip("-").lower()
    return alias or "source"


def markdown_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in {".md", ".markdown"} else []
    return sorted(item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in {".md", ".markdown"})


def source_for(path: Path, role: str, root: Path | None = None, scope: str = "full", upstream_run: str | None = None) -> Source:
    if role not in ROLES:
        raise ValueError(f"unknown supplement role: {role}; choose one of {sorted(ROLES)}")
    return Source(path=str(path), alias=deterministic_alias(path, root), role=role, scope=scope, sha256=digest(path), upstream_run=upstream_run)


def load_intake(primary: Path, *, supplements: list[tuple[Path, str]] | None = None, focus: str = "", specification: dict[str, Any] | None = None, tracks: list[str] | None = None, quality: QualityPolicy | None = None, upstream_run: str | None = None) -> Intake:
    primary = primary.resolve()
    if not primary.exists():
        raise FileNotFoundError(primary)
    primary_source = source_for(primary, "curriculum", primary.parent)
    supplement_sources: list[Source] = []
    for path, role in supplements or []:
        resolved = path.resolve()
        if not resolved.exists():
            raise FileNotFoundError(resolved)
        files = markdown_files(resolved)
        if resolved.is_file() or not files:
            supplement_sources.append(source_for(resolved, role, resolved.parent, upstream_run=upstream_run))
        else:
            root = resolved
            # Include the directory name so `index.md` from two supplements
            # cannot silently collide while remaining stable across runs.
            supplement_sources.extend(source_for(item, role, root.parent, upstream_run=upstream_run) for item in files)
    return Intake(primary_source, tuple(supplement_sources), focus, specification or {}, tuple(tracks or ()), quality or QualityPolicy())


def render_context(intake: Intake) -> str:
    sections = ["# Ralph Editorial Context", "", "## Primary context", "", f"Source: `{intake.primary.path}`", f"Alias: `{intake.primary.alias}`", ""]
    sections.append(read_material(Path(intake.primary.path)))
    if intake.focus_prompt:
        sections += ["", "## Focus prompt", "", intake.focus_prompt]
    if intake.supplements:
        sections += ["", "## Labeled supplements", ""]
        for source in intake.supplements:
            sections += [f"### {source.alias} ({source.role})", "", f"Source: `{source.path}`", f"Scope: {source.scope}", "", read_material(Path(source.path))]
    return "\n".join(sections).rstrip() + "\n"


def read_material(path: Path) -> str:
    files = markdown_files(path)
    if path.is_file():
        return path.read_text(encoding="utf-8")
    blocks = []
    for item in files:
        blocks.append(f"\n<!-- source: {item.relative_to(path).as_posix()} -->\n\n{item.read_text(encoding='utf-8')}")
    return "\n".join(blocks).lstrip()


def render_spec(intake: Intake) -> str:
    spec = intake.specification
    title = spec.get("title", "SteadyBurn weekly package")
    lines = [f"# {title}", "", "## Focus", "", intake.focus_prompt or "Use the primary context to produce the requested package.", "", "## Required outputs", "", "- `index.md`", "- `INSTRUCTIONS.md`", "- Complete artifact lineage and manifests", "", "## Selected tracks", "", *[f"- {track}" for track in intake.tracks or ("steadyburn",)], "", "## Editorial constraints", ""]
    for key, value in spec.items():
        if key != "title":
            lines.append(f"- **{key}:** {value}")
    return "\n".join(lines).rstrip() + "\n"


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def write_intake(package: Path, intake: Intake) -> None:
    package.mkdir(parents=True, exist_ok=True)
    (package / ".ralph").mkdir(exist_ok=True)
    (package / "CONTEXT.md").write_text(render_context(intake), encoding="utf-8")
    (package / "SPEC.md").write_text(render_spec(intake), encoding="utf-8")
    write_yaml(package / "spec.yaml", {"schema_version": "ralph-spec-v1", "focus_prompt": intake.focus_prompt, "primary": intake.primary.as_dict(), "supplements": [item.as_dict() for item in intake.supplements], "tracks": list(intake.tracks), "specification": intake.specification, "quality": intake.quality.as_dict()})
    (package / ".ralph" / "sources.json").write_text(json.dumps({"schema_version": "ralph-sources-v1", **intake.as_dict()}, indent=2) + "\n", encoding="utf-8")
    (package / ".ralph" / "state.json").write_text(json.dumps({"schema_version": "ralph-state-v1", "state": "draft", "repair_round": 0, "run_id": None, "events": [], "source_hash": digest(package / ".ralph" / "sources.json"), "spec_hash": digest(package / "spec.yaml")}, indent=2) + "\n", encoding="utf-8")
    (package / "README.md").write_text("# Gutenberg Ralph package\n\nThis package is staged and controller-managed. Generation and artifact linking are performed by the language-model-research operator.\n\n- `CONTEXT.md`: canonical editorial seed\n- `SPEC.md` / `spec.yaml`: human and machine-readable package contract\n- `.ralph/sources.json`: provenance and source hashes\n- `.ralph/state.json`: durable controller state\n", encoding="utf-8")
