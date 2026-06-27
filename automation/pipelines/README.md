# Burn Pipeline

The content generation harness is a small Python package invoked through `uv run`.
Dependencies live in `pyproject.toml`; there is no `requirements.txt`.

## Onion Layout

- `burn_pipeline/domain.py`: core models, provider ports, prompt assembly, output validation.
- `burn_pipeline/application.py`: use cases for generating one step or running a pipeline.
- `burn_pipeline/infrastructure.py`: filesystem, Codex CLI, and OpenAI-compatible HTTP adapters.
- `burn_pipeline/interface.py`: CLI entrypoint used by `uv run burn-pipeline`.

The application layer depends only on domain ports. Provider details stay in the
infrastructure layer.

## Providers

Local Codex CLI:

```sh
uv run burn-pipeline --provider codex-cli generate-step \
  --format markdown \
  --prompt-file prompts/burn/poc-markdown.md \
  --output tmp/codex-burn-poc/POC.md \
  --force
```

OpenAI API:

```sh
OPENAI_API_KEY=... uv run burn-pipeline --provider openai --model gpt-4.1 run \
  --pipeline pipelines/burn-poc.toml \
  --force
```

OpenAI-compatible local/network server, such as vLLM or llama.cpp:

```sh
uv run burn-pipeline \
  --provider openai-compatible \
  --base-url http://localhost:8000/v1 \
  --model local-model-name \
  run --pipeline pipelines/burn-poc.toml --force
```

## Step Dependencies

Pipeline TOML files support multiple inputs per step and ordered dependencies:

```toml
[[steps]]
id = "c"
format = "markdown"
prompt_file = "prompts/burn/c.md"
depends_on = ["a", "b"]
inputs = ["tmp/a.md", "tmp/b.md"]
output = "tmp/c.md"
```

The `inputs` are read and appended to the prompt. `depends_on` is an ordering
guard so a later step cannot run before the step outputs it needs exist.
