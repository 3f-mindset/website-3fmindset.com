from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from burn_pipeline.domain import GenerationModality, ProviderConfig, ProviderKind
from burn_pipeline.infrastructure import OpenAICompatibleLLM, build_inference
from burn_pipeline.interface import build_parser, resolve_provider_configs


class OpenRouterProviderTests(unittest.TestCase):
    def test_provider_config_applies_openrouter_defaults(self) -> None:
        config = ProviderConfig(kind=ProviderKind.OPENROUTER, model="openai/gpt-4.1")
        self.assertEqual(config.base_url, "https://openrouter.ai/api/v1")
        self.assertEqual(config.api_key_env, "OPENROUTER_API_KEY")

    def test_cli_provider_uses_openrouter_defaults(self) -> None:
        args = build_parser().parse_args(["--provider", "openrouter", "run", "--pipeline", "pipeline.toml"])
        config = resolve_provider_configs(args, spec=None)[GenerationModality.TEXT]
        self.assertEqual(config.kind, ProviderKind.OPENROUTER)
        self.assertEqual(config.base_url, "https://openrouter.ai/api/v1")
        self.assertEqual(config.api_key_env, "OPENROUTER_API_KEY")

    def test_openrouter_builds_compatible_text_client(self) -> None:
        config = ProviderConfig(kind=ProviderKind.OPENROUTER, model="openai/gpt-4.1")
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True):
            inference = build_inference(config, Path.cwd())
        self.assertIsInstance(inference, OpenAICompatibleLLM)
        self.assertEqual(inference._base_url, "https://openrouter.ai/api/v1")
        self.assertEqual(inference._api_key, "test-key")


if __name__ == "__main__":
    unittest.main()
