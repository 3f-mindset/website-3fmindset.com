from __future__ import annotations

import os
import json
import subprocess
import tempfile
import re
from pathlib import Path
from typing import Any

import httpx

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib

from .domain import (
    DevelopedModelEntry,
    InferenceRequest,
    ModelRegistry,
    PipelineSpec,
    ProviderConfig,
    SteadyBurnVerbIndex,
    SteadyBurnVerbIndexEntry,
    VerbTrackingStatus,
    extract_actionable_verb,
)


class LocalFileStore:
    def __init__(self, root: Path) -> None:
        self._root = root

    def read_text(self, path: Path) -> str:
        return self._resolve(path).read_text(encoding="utf-8")

    def write_text(self, path: Path, content: str, force: bool) -> None:
        resolved = self._resolve(path)
        if resolved.exists() and not force:
            raise FileExistsError(f"Output exists. Use --force to overwrite: {path}")
        if resolved.is_dir():
            raise IsADirectoryError(f"Output path is a directory: {path}")
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")

    def exists(self, path: Path) -> bool:
        return self._resolve(path).exists()

    def glob(self, pattern: str) -> list[Path]:
        base = self._root
        return [path.relative_to(base) for path in base.glob(pattern)]

    def _resolve(self, path: Path) -> Path:
        if path.is_absolute():
            return path
        return self._root / path


class CodexCliAgent:
    def __init__(self, command: str = "codex", model: str | None = None, cwd: Path | None = None) -> None:
        self._command = command
        self._model = model
        self._cwd = cwd or Path.cwd()

    def generate(self, request: InferenceRequest) -> str:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            output_file = tmp_dir / "last-message.txt"
            command = [
                self._command,
                "--ask-for-approval",
                "never",
                "exec",
                "--cd",
                str(self._cwd),
                "--sandbox",
                "read-only",
                "--ephemeral",
                "--output-last-message",
                str(output_file),
            ]
            model = request.model or self._model
            if model:
                command.extend(["--model", model])
            command.append("-")

            completed = subprocess.run(
                command,
                input=request.prompt,
                text=True,
                capture_output=True,
                cwd=self._cwd,
                check=False,
            )
            if completed.returncode != 0:
                stderr_tail = "\n".join(completed.stderr.splitlines()[-40:])
                stdout_tail = "\n".join(completed.stdout.splitlines()[-40:])
                raise RuntimeError(
                    "Codex CLI generation failed"
                    f"\nstdout:\n{stdout_tail}"
                    f"\nstderr:\n{stderr_tail}"
                )
            if not output_file.exists():
                raise RuntimeError("Codex CLI did not write the last-message output file")
            return output_file.read_text(encoding="utf-8")


class OpenAICompatibleLLM:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        model: str | None,
        timeout_seconds: float = 120.0,
    ) -> None:
        self._base_url = normalize_openai_base_url(base_url)
        self._api_key = api_key
        self._model = model
        self._timeout_seconds = timeout_seconds

    def generate(self, request: InferenceRequest) -> str:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        with httpx.Client(timeout=self._timeout_seconds) as client:
            model = request.model or self._model or self._resolve_model(client, headers)
            payload: dict[str, Any] = {
                "model": model,
                "messages": [{"role": "user", "content": request.prompt}],
                "temperature": 0,
            }
            response = client.post(
                f"{self._base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected LLM response shape: {data}") from exc
        if not isinstance(content, str):
            raise RuntimeError(f"Unexpected LLM content type: {type(content).__name__}")
        return content

    def _resolve_model(self, client: httpx.Client, headers: dict[str, str]) -> str:
        response = client.get(f"{self._base_url}/models", headers=headers)
        response.raise_for_status()
        data = response.json()
        models = data.get("data")
        if not isinstance(models, list) or not models:
            raise RuntimeError(f"Unable to resolve a default model from {self._base_url}/models: {data}")
        first = models[0]
        if not isinstance(first, dict) or not isinstance(first.get("id"), str) or not first["id"].strip():
            raise RuntimeError(f"Unexpected model entry from {self._base_url}/models: {first}")
        return first["id"].strip()


def normalize_openai_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        return normalized
    return f"{normalized}/v1"


def load_pipeline_spec(path: Path) -> PipelineSpec:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return PipelineSpec.model_validate(data)


def load_model_registry(files: LocalFileStore, registry_path: Path) -> ModelRegistry:
    if files.exists(registry_path):
        data = json.loads(files.read_text(registry_path))
        return ModelRegistry.model_validate(data)

    registry = scan_model_registry(files)
    write_model_registry(files, registry_path, registry, force=True)
    return registry


def write_model_registry(
    files: LocalFileStore,
    registry_path: Path,
    registry: ModelRegistry,
    *,
    force: bool,
) -> None:
    payload = registry.model_dump()
    files.write_text(
        registry_path,
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        force=force,
    )


def scan_model_registry(files: LocalFileStore) -> ModelRegistry:
    entries: list[DevelopedModelEntry] = []
    for path in sorted(files.glob("content/letters/**/*.md")):
        if path.name.upper() not in {"CONTEXT.MD", "GPT.MD", "GPT_PROMPT.MD", "SB19-CONTEXT.MD"} and "context" not in path.name.lower():
            continue
        content = files.read_text(path)
        verb = extract_actionable_verb(content)
        if not verb:
            continue
        date_value, slug = split_letter_folder(path)
        title = extract_title(content)
        entries.append(
            DevelopedModelEntry(
                verb=verb,
                title=title,
                slug=slug,
                date=date_value,
                source_path=str(path),
            )
        )
    steadyburn_index = scan_steadyburn_verb_index(files)
    for item in steadyburn_index.entries:
        if not item.verb:
            continue
        entries.append(
            DevelopedModelEntry(
                verb=item.verb,
                title=item.title,
                slug=item.slug,
                date=item.date,
                source_path=item.evidence_path,
            )
        )
    return ModelRegistry(entries=dedupe_registry_entries(entries))


def dedupe_registry_entries(entries: list[DevelopedModelEntry]) -> list[DevelopedModelEntry]:
    by_verb: dict[str, DevelopedModelEntry] = {}
    for entry in sorted(entries, key=lambda item: (item.date, item.slug, item.source_path)):
        by_verb[entry.verb.upper()] = entry
    return list(by_verb.values())


def split_letter_folder(path: Path) -> tuple[str, str]:
    parent_name = path.parent.name
    if len(parent_name) >= 11 and parent_name[4] == "-" and parent_name[7] == "-":
        return parent_name[:10], parent_name[11:]
    return "", parent_name


def extract_title(content: str) -> str:
    lines = content.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        normalized = stripped.lstrip("#").strip().lower()
        if normalized == "title":
            for candidate in lines[index + 1 :]:
                candidate_stripped = candidate.strip()
                if candidate_stripped:
                    return candidate_stripped
        if stripped.startswith("# "):
            title_candidate = stripped[2:].strip()
            if title_candidate.lower() not in {"title", "content crusher response"}:
                return title_candidate
    return ""


STEADYBURN_SERIES_PATTERN = re.compile(r"(?m)^series:\s*$\n\s*-\s*SteadyBurn\s*$")
FRONTMATTER_TITLE_PATTERN = re.compile(r'(?m)^title:\s*"?(.*?)"?\s*$')
FRONTMATTER_SLUG_PATTERN = re.compile(r'(?m)^slug:\s*"?(.*?)"?\s*$')
INLINE_ACTIONABLE_VERB_PATTERN = re.compile(
    r"(?im)actionable\s+verb\s*:\s*\**\s*([A-Z][A-Z0-9-]{2,})\s*\**"
)
UPPERCASE_FILENAME_TOKEN_PATTERN = re.compile(r"^([A-Z][A-Z0-9]{3,})(?:[-_].*)?$")
SERIES_TOKEN_PATTERN = re.compile(r"^SB\d+$")
GENERIC_FILENAME_TOKENS = {
    "LESSON",
    "INSTRUCTIONS",
    "CONTEXT",
    "WORKSHEET",
    "MASKED",
    "COPY",
    "PROMO",
    "CREATIVE",
    "COVER",
    "INDEX",
    "GPT",
    "PROMPT",
}


def scan_steadyburn_verb_index(files: LocalFileStore) -> SteadyBurnVerbIndex:
    folders = []
    for folder in sorted(files.glob("content/letters/*")):
        index_path = folder / "index.md"
        if not files.exists(index_path):
            continue
        text = files.read_text(index_path)
        if not STEADYBURN_SERIES_PATTERN.search(text):
            continue
        folders.append((folder, text))

    entries: list[SteadyBurnVerbIndexEntry] = []
    for sequence, (folder, index_text) in enumerate(folders, start=1):
        title_match = FRONTMATTER_TITLE_PATTERN.search(index_text)
        slug_match = FRONTMATTER_SLUG_PATTERN.search(index_text)
        verb, status, evidence_path, note = detect_folder_verb(files, folder)
        entries.append(
            SteadyBurnVerbIndexEntry(
                sequence=sequence,
                date=folder.name[:10],
                folder=folder.name,
                title=title_match.group(1) if title_match else folder.name,
                slug=slug_match.group(1) if slug_match else folder.name[11:],
                verb=verb,
                status=status,
                evidence_path=evidence_path,
                note=note,
            )
        )
    return SteadyBurnVerbIndex(entries=entries)


def detect_folder_verb(
    files: LocalFileStore,
    folder: Path,
) -> tuple[str, VerbTrackingStatus, str, str]:
    markdown_files = sorted(path for path in files.glob(f"{folder.as_posix()}/*.md"))
    for path in markdown_files:
        content = files.read_text(path)
        verb = extract_actionable_verb(content)
        if verb:
            return verb, VerbTrackingStatus.CONFIRMED, str(path), "Actionable VERB heading"

    for path in markdown_files:
        content = files.read_text(path)
        match = INLINE_ACTIONABLE_VERB_PATTERN.search(content)
        if match:
            return (
                match.group(1).upper(),
                VerbTrackingStatus.INFERRED,
                str(path),
                "Inline Actionable VERB reference",
            )

    for path in sorted(files.glob(f"{folder.as_posix()}/*")):
        if not path.is_file():
            continue
        token = extract_filename_verb_token(path)
        if token:
            return token, VerbTrackingStatus.INFERRED, str(path), "Filename token"

    return "", VerbTrackingStatus.MISSING, "", "No verb evidence found"


def extract_filename_verb_token(path: Path) -> str | None:
    match = UPPERCASE_FILENAME_TOKEN_PATTERN.match(path.stem)
    if not match:
        return None
    token = match.group(1).upper()
    if token in GENERIC_FILENAME_TOKENS:
        return None
    if SERIES_TOKEN_PATTERN.match(token):
        return None
    return token


def render_steadyburn_verb_index_markdown(index: SteadyBurnVerbIndex) -> str:
    lines = [
        "# SteadyBurn Verb Index",
        "",
        f"Total indexed letters: {len(index.entries)}",
        "",
        "| # | Date | Title | Slug | Verb | Status | Evidence | Note |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for entry in index.entries:
        verb = entry.verb or "TBD"
        evidence = entry.evidence_path or "-"
        note = entry.note or "-"
        title = entry.title.replace("|", "\\|")
        slug = entry.slug.replace("|", "\\|")
        lines.append(
            f"| {entry.sequence} | {entry.date} | {title} | `{slug}` | `{verb}` | {entry.status.value} | `{evidence}` | {note} |"
        )
    return "\n".join(lines) + "\n"


def build_inference(config: ProviderConfig, cwd: Path):
    if config.kind.value == "codex-cli":
        return CodexCliAgent(command=config.command, model=config.model, cwd=cwd)

    if config.kind.value == "openai":
        if not config.model:
            raise ValueError("--model or provider.model is required for openai")
        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise ValueError(f"{config.api_key_env} is required for openai")
        return OpenAICompatibleLLM(
            base_url=config.base_url or "https://api.openai.com/v1",
            api_key=api_key,
            model=config.model,
        )

    if config.kind.value == "openai-compatible":
        if not config.base_url:
            raise ValueError("--base-url or provider.base_url is required for openai-compatible")
        return OpenAICompatibleLLM(
            base_url=config.base_url,
            api_key=os.environ.get(config.api_key_env),
            model=config.model,
        )

    raise ValueError(f"Unsupported provider: {config.kind}")
