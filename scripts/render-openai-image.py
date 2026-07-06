#!/usr/bin/env python
from __future__ import annotations

import argparse
import base64
import re
from pathlib import Path

import httpx


SECTION_PATTERN = re.compile(r"(?ms)^# ([^\n]+)\n(.*?)(?=^# |\Z)")
DIMENSIONS_PATTERN = re.compile(r"(?im)^- Dimensions:\s*([0-9]+)\s*x\s*([0-9]+)px\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a markdown image brief through an OpenAI-compatible image endpoint.")
    parser.add_argument("--prompt-file", required=True, help="Markdown brief with # Image Prompt and optional sections.")
    parser.add_argument("--output", required=True, help="Output PNG or JPEG path.")
    parser.add_argument("--base-url", default="http://localhost:11434", help="OpenAI-compatible base URL.")
    parser.add_argument(
        "--model",
        default="unsloth-qwen-image-2512-gguf-qwen-image-2512-q4-k-m",
        help="Image model id.",
    )
    parser.add_argument("--size", default=None, help="Explicit size WIDTHxHEIGHT. Overrides dimensions in the brief.")
    parser.add_argument("--format", default="png", choices=["png", "jpeg"], help="Image output format.")
    parser.add_argument("--timeout", type=float, default=600.0, help="Request timeout in seconds.")
    return parser.parse_args()


def normalize_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        return normalized
    return f"{normalized}/v1"


def parse_sections(markdown: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    for match in SECTION_PATTERN.finditer(markdown):
        sections[match.group(1).strip().lower()] = match.group(2).strip()
    return sections


def parse_dimensions(markdown: str) -> str | None:
    match = DIMENSIONS_PATTERN.search(markdown)
    if not match:
        return None
    return f"{match.group(1)}x{match.group(2)}"


def clean_bullets(block: str) -> list[str]:
    items: list[str] = []
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("- "):
            items.append(stripped[2:].strip())
        else:
            items.append(stripped)
    return items


def build_prompt(markdown: str) -> tuple[str, str | None, str | None]:
    sections = parse_sections(markdown)
    image_prompt = sections.get("image prompt")
    if not image_prompt:
        raise ValueError("Prompt file is missing a '# Image Prompt' section")

    required_copy = sections.get("required on-image copy")
    negative_prompt = sections.get("negative prompt")

    prompt_parts = [image_prompt.strip()]
    if required_copy:
        copy_lines = clean_bullets(required_copy)
        prompt_parts.append("Render the following on-image copy exactly:")
        prompt_parts.extend(copy_lines)

    return "\n\n".join(prompt_parts), negative_prompt.strip() if negative_prompt else None, parse_dimensions(markdown)


def main() -> None:
    args = parse_args()
    prompt_path = Path(args.prompt_file)
    output_path = Path(args.output)

    markdown = prompt_path.read_text(encoding="utf-8")
    prompt, negative_prompt, inferred_size = build_prompt(markdown)
    size = args.size or inferred_size or "1024x1024"

    payload = {
        "model": args.model,
        "prompt": prompt,
        "size": size,
        "n": 1,
        "output_format": args.format,
    }
    if negative_prompt:
        payload["negative_prompt"] = negative_prompt

    headers = {"Content-Type": "application/json"}
    base_url = normalize_base_url(args.base_url)
    with httpx.Client(timeout=args.timeout) as client:
        response = client.post(f"{base_url}/images/generations", headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

    images = data.get("data")
    if not isinstance(images, list) or not images:
        raise RuntimeError(f"Unexpected image response shape: {data}")
    first = images[0]
    if not isinstance(first, dict) or not isinstance(first.get("b64_json"), str):
        raise RuntimeError(f"Image payload is missing b64_json: {first}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(base64.b64decode(first["b64_json"]))
    print(output_path)


if __name__ == "__main__":
    main()
