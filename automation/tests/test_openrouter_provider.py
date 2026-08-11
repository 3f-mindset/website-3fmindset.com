from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from burn_pipeline.application import BurnPipeline
from burn_pipeline.domain import (
    BurnContext,
    GeneratedArtifact,
    GenerationModality,
    ProviderConfig,
    ProviderKind,
    StepSpec,
)
from burn_pipeline.infrastructure import ChatCompletionsAdapter, build_inference
from burn_pipeline.interface import build_parser, resolve_provider_configs


class OpenRouterProviderTests(unittest.TestCase):
    def test_step_model_overrides_modality_default(self) -> None:
        requests = []

        class Files:
            def read_text(self, path: Path) -> str:
                return "Write a short answer."

            def write_text(self, path: Path, content: str, force: bool) -> None:
                pass

            def write_bytes(self, path: Path, content: bytes, force: bool) -> None:
                pass

            def exists(self, path: Path) -> bool:
                return True

        class Inference:
            def generate(self, request):
                requests.append(request)
                return GeneratedArtifact(text="Generated answer")

        pipeline = BurnPipeline(
            files=Files(),
            inference_factory=lambda provider, modality: Inference(),
            providers={
                GenerationModality.TEXT: ProviderConfig(
                    kind=ProviderKind.OPENROUTER,
                    model="base-model",
                    providerUrl="https://example.test/v1",
                )
            },
        )
        step = StepSpec(
            id="lesson",
            format="markdown",
            prompt_file=Path("prompt.md"),
            output=Path("output.md"),
            model="specialist-model",
        )

        pipeline.generate_step(step, context=BurnContext(), force=True)

        self.assertEqual(requests[0].model, "specialist-model")

    def test_step_without_model_uses_provider_default(self) -> None:
        pipeline = BurnPipeline(
            files=None,
            inference_factory=None,
            providers={
                GenerationModality.TEXT: ProviderConfig(
                    kind=ProviderKind.OPENROUTER,
                    model="base-model",
                )
            },
        )
        step = StepSpec(
            id="lesson",
            format="markdown",
            prompt_file=Path("prompt.md"),
            output=Path("output.md"),
        )

        self.assertEqual(pipeline._model_for_step(step), "base-model")

    def test_provider_config_uses_generic_provider_url(self) -> None:
        config = ProviderConfig.model_validate(
            {"kind": "openrouter", "model": "openai/gpt-4.1", "providerUrl": "https://example.test/v1"}
        )
        self.assertEqual(config.provider_url, "https://example.test/v1")
        self.assertEqual(config.model_dump(by_alias=True)["providerUrl"], "https://example.test/v1")
        self.assertNotIn("base_url", config.model_dump())

    def test_generic_adapter_accepts_only_connection_values(self) -> None:
        adapter = ChatCompletionsAdapter(providerUrl="https://example.test/v1", apiKey="test-key", model="test-model")
        self.assertEqual(adapter._provider_url, "https://example.test/v1")
        self.assertEqual(adapter._api_key, "test-key")

    def test_provider_config_accepts_legacy_base_url(self) -> None:
        config = ProviderConfig.model_validate({"kind": "openrouter", "base_url": "https://example.test/v1"})
        self.assertEqual(config.provider_url, "https://example.test/v1")

    def test_cli_provider_uses_infrastructure_default(self) -> None:
        args = build_parser().parse_args(["--provider", "openrouter", "run", "--pipeline", "pipeline.yaml"])
        config = resolve_provider_configs(args, spec=None)[GenerationModality.TEXT]
        self.assertEqual(config.kind, ProviderKind.OPENROUTER)
        self.assertIsNone(config.provider_url)

    def test_openrouter_resolves_connection_before_constructing_generic_adapter(self) -> None:
        config = ProviderConfig(kind=ProviderKind.OPENROUTER, model="openai/gpt-4.1")
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True):
            inference = build_inference(config, Path.cwd())
        self.assertIsInstance(inference, ChatCompletionsAdapter)
        self.assertEqual(inference._provider_url, "https://openrouter.ai/api/v1")
        self.assertEqual(inference._api_key, "test-key")


if __name__ == "__main__":
    unittest.main()
