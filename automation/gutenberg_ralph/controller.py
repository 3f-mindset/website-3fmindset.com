from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .domain import STATES
from .intake import digest


REQUIRED = ("README.md", "SPEC.md", "spec.yaml", "CONTEXT.md", ".ralph/sources.json", ".ralph/state.json")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_state(package: Path) -> dict[str, Any]:
    return json.loads((package / ".ralph" / "state.json").read_text(encoding="utf-8"))


def write_state(package: Path, state: dict[str, Any]) -> None:
    (package / ".ralph" / "state.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def transition(package: Path, target: str, **details: Any) -> dict[str, Any]:
    if target not in STATES:
        raise ValueError(f"invalid Ralph state: {target}")
    state = read_state(package)
    event = {"at": now(), "state": target, **details}
    state["state"] = target
    state.update({key: value for key, value in details.items() if key not in {"verification"}})
    state.setdefault("events", []).append(event)
    write_state(package, state)
    return state


def verify(package: Path) -> dict[str, Any]:
    missing = [item for item in REQUIRED if not (package / item).exists()]
    errors: list[str] = [f"missing required file: {item}" for item in missing]
    scores_path = next((package / item for item in (".ralph/scores.json", "CONTENT_SCORE.json", "content-score-report.json") if (package / item).exists()), None)
    scores: dict[str, Any] = {}
    if scores_path:
        try:
            scores = json.loads(scores_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"invalid score report: {exc}")
    quality = _quality(package)
    artifact_scores = scores.get("artifacts", scores.get("scores", {})) if scores else {}
    if not artifact_scores and isinstance(scores.get("reports"), list):
        artifact_scores = (scores["reports"][0] or {}).get("artifacts", {})
    for artifact, minimum in quality.get("minimum_scores", {}).items():
        value = _score_for(artifact_scores, artifact)
        if value is None:
            errors.append(f"missing score for {artifact}")
        elif value < minimum:
            errors.append(f"{artifact} score {value:g} is below {minimum:g}")
    lineage = package / ".ralph" / "lineage.json"
    manifest = package / ".ralph" / "manifest.json"
    if not lineage.exists():
        errors.append("missing complete lineage manifest: .ralph/lineage.json")
    if not manifest.exists():
        errors.append("missing artifact manifest: .ralph/manifest.json")
    elif not _manifest_is_complete(package, manifest):
        errors.append("artifact manifest is incomplete or points to missing files")
    report = {"passed": not errors, "errors": errors, "scores": scores, "checked_at": now()}
    (package / ".ralph" / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _manifest_is_complete(package: Path, path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, (list, dict)) or not artifacts:
        return False
    items = artifacts if isinstance(artifacts, list) else artifacts.values()
    for item in items:
        relative = item if isinstance(item, str) else item.get("path") if isinstance(item, dict) else None
        if not relative or not (package / relative).exists():
            return False
    return True


def _quality(package: Path) -> dict[str, Any]:
    import yaml
    spec_path = package / "spec.yaml"
    if not spec_path.exists():
        return {"minimum_scores": {"index.md": 70, "INSTRUCTIONS.md": 70}, "max_repairs": 3}
    raw = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
    return raw.get("quality", {"minimum_scores": {"index.md": 70, "INSTRUCTIONS.md": 70}})


def _score_for(scores: dict[str, Any], artifact: str) -> float | None:
    value = scores.get(artifact)
    if isinstance(value, dict):
        value = value.get("final_score", value.get("score"))
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def run_operator(package: Path, operator_command: str | None = None, *, dry_run: bool = False) -> dict[str, Any]:
    state = read_state(package)
    source_hash = digest(package / ".ralph" / "sources.json")
    spec_hash = digest(package / "spec.yaml")
    if state.get("source_hash") == source_hash and state.get("spec_hash") == spec_hash and state.get("state") in {"passed", "promoted"}:
        return {"skipped": True, "reason": "unchanged intake and specification"}
    transition(package, "queued", source_hash=source_hash, spec_hash=spec_hash)
    (package / ".ralph" / "job.json").write_text(json.dumps(job_payload(package), indent=2) + "\n", encoding="utf-8")
    if dry_run:
        return {"dry_run": True, "state": read_state(package)}
    command = operator_command or os.environ.get("GUTENBERG_RALPH_OPERATOR")
    if not command:
        transition(package, "failed", reason="no research operator configured")
        raise RuntimeError("configure GUTENBERG_RALPH_OPERATOR with the language-model-research editorial runner")
    transition(package, "running", operator=command)
    result = subprocess.run(command, shell=True, cwd=package, capture_output=True, text=True)
    (package / ".ralph" / "operator.stdout.log").write_text(result.stdout, encoding="utf-8")
    (package / ".ralph" / "operator.stderr.log").write_text(result.stderr, encoding="utf-8")
    if result.returncode:
        transition(package, "failed", returncode=result.returncode)
        raise RuntimeError(f"research operator failed with exit code {result.returncode}")
    report = verify(package)
    transition(package, "passed" if report["passed"] else "failed", verification=report)
    return report


def job_payload(package: Path) -> dict[str, Any]:
    """Return the stable boundary payload consumed by the research operator."""
    import yaml
    spec = yaml.safe_load((package / "spec.yaml").read_text(encoding="utf-8")) or {}
    sources = json.loads((package / ".ralph" / "sources.json").read_text(encoding="utf-8"))
    state = read_state(package)
    return {
        "contract": "gutenberg-ralph-editorial-v1",
        "package_root": str(package.resolve()),
        "primary": sources["primary"],
        "supplements": sources.get("supplements", []),
        "focus_prompt": spec.get("focus_prompt", ""),
        "specification": spec.get("specification", {}),
        "tracks": spec.get("tracks", []),
        "quality": spec.get("quality", {}),
        "idempotency": {"source_hash": state.get("source_hash"), "spec_hash": state.get("spec_hash")},
    }


def repair(package: Path, brief: str = "") -> dict[str, Any]:
    state = read_state(package)
    round_number = int(state.get("repair_round", 0)) + 1
    verification = verify(package)
    text = brief or "Repair every failed verification criterion and preserve lineage.\n\nFailed criteria:\n" + "\n".join(f"- {error}" for error in verification["errors"])
    (package / ".ralph" / "repair-brief.md").write_text("# Ralph repair brief\n\n" + text + "\n", encoding="utf-8")
    maximum = int(_quality(package).get("max_repairs", 3))
    if round_number > maximum:
        return transition(package, "needs-review", repair_round=round_number, reason="maximum repair rounds exceeded")
    return transition(package, "repairing", repair_round=round_number)


def promote(package: Path, destination: Path) -> dict[str, Any]:
    report = verify(package)
    if not report["passed"]:
        transition(package, "failed", verification=report)
        raise RuntimeError("promotion blocked by quality gate: " + "; ".join(report["errors"]))
    destination = destination.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError(f"promotion destination is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    for item in package.iterdir():
        if item.name == ".ralph":
            continue
        target = destination / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)
    transition(package, "promoted", destination=str(destination))
    return {"promoted": True, "destination": str(destination)}
