from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import httpx
import tomli

from .domain import InferenceRequest, PipelineSpec, ProviderConfig


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
        model: str,
        timeout_seconds: float = 120.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout_seconds = timeout_seconds

    def generate(self, request: InferenceRequest) -> str:
        model = request.model or self._model
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": request.prompt}],
            "temperature": 0,
        }
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        with httpx.Client(timeout=self._timeout_seconds) as client:
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


def load_pipeline_spec(path: Path) -> PipelineSpec:
    data = tomli.loads(path.read_text(encoding="utf-8"))
    return PipelineSpec.model_validate(data)


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
        if not config.model:
            raise ValueError("--model or provider.model is required for openai-compatible")
        return OpenAICompatibleLLM(
            base_url=config.base_url,
            api_key=os.environ.get(config.api_key_env),
            model=config.model,
        )

    raise ValueError(f"Unsupported provider: {config.kind}")
