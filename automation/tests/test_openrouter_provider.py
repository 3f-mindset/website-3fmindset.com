from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from burn_pipeline.domain import GenerationModality, ProviderConfig, ProviderKind
from burn_pipeline.infrastructure import ChatCompletionsAdapter, build_inference
from burn_pipeline.interface import build_parser, resolve_provider_configs


class OpenRouterProviderTests(unittest.TestCase):
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
        args = build_parser().parse_args(["--provider", "openrouter", "run", "--pipeline", "pipeline.toml"])
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
