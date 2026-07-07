# Burn Pipeline

The content generation harness is a small Python package invoked through `uv run`.
Dependencies live in `pyproject.toml`; there is no `requirements.txt`.
The primary execution path is `uv run burn-pipeline ...`, but a repo-local backup
harness also exists at `python scripts/burn-pipeline.py ...`.

## Onion Layout

- `burn_pipeline/domain.py`: core models, prompt rendering, and output validation.
- `burn_pipeline/application.py`: use cases for generating one step or running a full pipeline.
- `burn_pipeline/infrastructure.py`: filesystem, Codex CLI, and OpenAI-compatible HTTP adapters.
- `burn_pipeline/interface.py`: CLI entrypoint used by `uv run burn-pipeline`.

The application layer depends only on domain ports. Provider details stay in the
infrastructure layer.

## Providers

Default text provider: OpenAI-compatible inference at `http://localhost:11434`.
Default text model: `active`.
Default image provider: OpenAI-compatible inference at `http://localhost:11434`.
Default image model: `unsloth-qwen-image-2512-gguf-qwen-image-2512-q4-k-m`.

```sh
uv run burn-pipeline generate-step \
  --format markdown \
  --prompt-file automation/prompts/burn/poc-markdown.md \
  --output tmp/codex-burn-poc/POC.md \
  --force
```

OpenAI API:

```sh
OPENAI_API_KEY=... uv run burn-pipeline --provider openai --model gpt-4.1 run \
  --pipeline automation/pipelines/burn-poc.toml \
  --force
```

OpenAI-compatible local/network server, such as vLLM or llama.cpp:

```sh
uv run burn-pipeline \
  --base-url http://localhost:11434 \
  --model local-model-name \
  run --pipeline automation/pipelines/burn-poc.toml --force
```

Pipeline files can now declare separate providers for `text`, `image`, and `audio`.
That lets the writing steps use one model while future raster-generation or audio steps
use different models or endpoints.

Example:

```toml
[providers.text]
kind = "openai-compatible"
base_url = "http://localhost:11434"
model = "active"

[providers.image]
kind = "openai-compatible"
base_url = "http://localhost:11434"
model = "unsloth-qwen-image-2512-gguf-qwen-image-2512-q4-k-m"

[providers.audio]
kind = "openai-compatible"
base_url = "http://localhost:11434"
model = "AUDIO_MODEL_NAME"
```

Current SteadyBurn steps still run as `text` unless a future step explicitly sets another modality.

## Production Scaffold

Initialize a new production workspace before generation:

```sh
uv run burn-pipeline init-production \
  --title "Letter Title" \
  --date 2026-07-10
```

Backup path:

```sh
python scripts/burn-pipeline.py init-production \
  --title "Letter Title" \
  --date 2026-07-10
```

This creates:

- `content/letters/YYYY-MM-DD-slug/SEED.md`
- `content/letters/YYYY-MM-DD-slug/index.md`
- `content/letters/YYYY-MM-DD-slug/pipeline.toml`

Fill `SEED.md` with the inspiration, scenes, constraints, and worksheet intent.
That seed document is the first human input to the workflow.

To turn a real seed document into a final letter folder with generated `CONTEXT.md`:

```sh
uv run burn-pipeline seed-production \
  --seed-file path/to/seed.md \
  --date 2026-07-10
```

Backup path:

```sh
python scripts/burn-pipeline.py seed-production \
  --seed-file path/to/seed.md \
  --date 2026-07-10
```

That route generates `CONTEXT.md` first, derives the final title and slug from it,
then writes `SEED.md`, `CONTEXT.md`, `index.md`, and `pipeline.toml` into the final
letter folder. After that, `CONTEXT.md` becomes the primary source of truth for the
rest of the pipeline.

The generated pipeline now also includes a marketing-assets phase. That phase emits
prompt/spec files such as `PROMO_PROMPT.md`, which are intended to feed downstream
`png/jpeg` asset generation rather than pretending the text pipeline can write raster
images directly.

## Model Registry

The pipeline maintains `automation/pipelines/model-registry.json` as the registry of
previously developed worksheet models and actionable verbs.

It also maintains `automation/pipelines/steadyburn-verb-index.{json,md}` as the
numbered series-wide tracking artifact for every SteadyBurn letter, starting at 1.

Use it to prevent accidental reuse:

```sh
uv run burn-pipeline sync-model-registry
uv run burn-pipeline sync-steadyburn-verb-index
```

Backup path:

```sh
python scripts/burn-pipeline.py sync-model-registry
python scripts/burn-pipeline.py sync-steadyburn-verb-index
```

The model registry now merges confirmed `CONTEXT.md` verbs with verb evidence from the
series-wide SteadyBurn index, so older or inferred verbs can still be treated as used.
Normal pipeline runs also load this registry automatically before prompt rendering.
When `seed-production` generates `CONTEXT.md`, the newly generated `Actionable VERB`
is written back into the registry so future Content Crusher prompts can forbid it.

## Step Dependencies And Composable Inputs

Pipeline TOML files support ordered dependencies, direct file inputs, and references
to prior step outputs. Steps also support `modality = "text" | "image" | "audio"`,
though the current content pipeline defaults every step to `text`:

```toml
[[steps]]
id = "c"
format = "markdown"
prompt_file = "automation/prompts/burn/c.md"
depends_on = ["a", "b"]
modality = "text"
output = "tmp/c.md"

[[steps.inputs]]
step = "a"
alias = "first_pass"

[[steps.inputs]]
path = "tmp/b.md"
alias = "second_source"
```

Prompt templates can reference runtime values with `{{...}}`, for example:

- `{{context.title}}`
- `{{context.slug}}`
- `{{variables.voice}}`
- `{{current_step.output_path}}`

The `inputs` are read and appended to the prompt under their aliases. `depends_on`
remains the ordering guard so a later step cannot run before the required step outputs exist.

Use prompt rendering to inspect what one step will see before calling a model:

```sh
uv run burn-pipeline render-prompt \
  --pipeline content/letters/YYYY-MM-DD-slug/pipeline.toml \
  --step-id lesson
```

Backup path:

```sh
python scripts/burn-pipeline.py render-prompt \
  --pipeline content/letters/YYYY-MM-DD-slug/pipeline.toml \
  --step-id lesson
```
