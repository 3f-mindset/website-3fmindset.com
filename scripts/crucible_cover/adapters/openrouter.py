"""OpenRouter image API adapter for OpenAI image models."""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request

from ..domain import GeneratedImage, ProviderError


class OpenRouterImageGenerator:
    def __init__(
        self,
        api_key: str,
        *,
        model: str = "openai/gpt-image-2",
        endpoint: str = "https://openrouter.ai/api/v1/images",
        timeout: float = 900,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._endpoint = endpoint
        self._timeout = timeout

    @property
    def model(self) -> str:
        return self._model

    def generate(self, prompt: str) -> GeneratedImage:
        payload = {
            "model": self._model,
            "prompt": prompt,
            "aspect_ratio": "3:2",
            "quality": "high",
            "n": 1,
        }
        request = urllib.request.Request(
            self._endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://3fmindset.com/",
                "X-Title": "3F Mindset Crucible Cover Generator",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                result = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = self._error_detail(exc)
            message = f"OpenRouter image request failed with HTTP {exc.code}"
            if detail:
                message += f": {detail[:300]}"
            raise ProviderError(message) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ProviderError(
                f"OpenRouter image request failed: {type(exc).__name__}"
            ) from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ProviderError("OpenRouter returned an unreadable response") from exc

        items = result.get("data") or []
        if not items or not items[0].get("b64_json"):
            raise ProviderError("OpenRouter response did not contain a base64 image")
        try:
            image = base64.b64decode(items[0]["b64_json"], validate=True)
        except (ValueError, TypeError) as exc:
            raise ProviderError("OpenRouter returned invalid image data") from exc

        cost = (result.get("usage") or {}).get("cost")
        try:
            parsed_cost = float(cost) if cost is not None else None
        except (TypeError, ValueError):
            parsed_cost = None
        return GeneratedImage(image, parsed_cost)

    @staticmethod
    def _error_detail(error: urllib.error.HTTPError) -> str:
        try:
            payload = json.loads(error.read())
            detail = (payload.get("error") or {}).get("message", "")
            return str(detail).replace("\n", " ")
        except Exception:
            return ""
