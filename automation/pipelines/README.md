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
  --pipeline automation/pipelines/burn-poc.yaml \
  --force
```

OpenRouter (OpenAI-compatible API):

```sh
OPENROUTER_API_KEY=... uv run burn-pipeline --provider openrouter \
  --model openai/gpt-4.1 run \
  --pipeline automation/pipelines/burn-poc.yaml \
  --force
```

`openrouter` defaults to `https://openrouter.ai/api/v1` and `OPENROUTER_API_KEY`.
Use any OpenRouter model identifier (for example, `openai/gpt-4.1`).
Provider configuration uses the generic `providerUrl` field. The runtime selects the
provider's environment variable and passes only generic `providerUrl` and `apiKey`
values to the HTTP adapter.

The Make targets support the same provider:

```sh
OPENROUTER_API_KEY=... make burn-run BURN_PROVIDER=openrouter \
  CODEX_BURN_MODEL=openai/gpt-4.1
```

## Compare OpenAI Models Through OpenRouter

`scripts/compare-openrouter-openai-models.py` runs a specified production pipeline
once per `openai/...` model in isolated Git worktrees. Every worktree starts from the
same committed base and writes to the normal letter directory on its own branch, so
each PR is a direct generated-content diff rather than mixed alternatives in one
working directory. The selected pipeline must exist in the base revision.

Review the planned branches and text-only steps first:

```sh
python scripts/compare-openrouter-openai-models.py --dry-run \
  --pipeline content/letters/YYYY-MM-DD-slug/pipeline.yaml --models \
  openai/gpt-4.1 openai/gpt-4o
```

Run and commit a comparison branch per model:

```sh
OPENROUTER_API_KEY=... python scripts/compare-openrouter-openai-models.py \
  --pipeline content/letters/YYYY-MM-DD-slug/pipeline.yaml \
  --models openai/gpt-4.1 openai/gpt-4o --commit --push
```

The harness rejects non-OpenAI model IDs and excludes image/audio stages. Generated
outputs include `MODEL_COMPARISON.md`, recording the model, base revision, timestamp,
and pipeline stages. Worktrees remain in a sibling
`website-3fmindset.com-model-comparisons` directory for review and are never
overwritten.

## Representative Weekly Automation Cost

The following is a representative end-to-end run for the 2026-08-07 weekly letter,
"Master Your Tasks: Prioritization and Time Management." Each comparison generated
the context and the standard text pipeline, then rendered one hero cover with
`openai/gpt-5.4-image-2`. Costs are the provider-reported OpenRouter amounts recorded
at run time; they are examples, not price commitments.

| Writing model | Text generation | Hero cover | Recorded total |
| --- | ---: | ---: | ---: |
| `openai/gpt-4.1` | $0.184258 | $0.025406 | $0.209664 |
| `openai/gpt-5.4` | $0.461757 | $0.008912 | $0.470669 |
| `openai/gpt-5.5` | $1.218690 | $0.009472 | $1.228162 |
| `google/gemini-3.6-flash` | $0.478587 | $0.008112 | $0.486699 |
| `google/gemini-3.5-flash-lite` | $0.054309 | $0.014706 | $0.069015 |
| `deepseek/deepseek-v4-pro` | $0.016049 | $0.008840 | $0.024889 |
| `deepseek/deepseek-v4-flash` | $0.008122 | $0.008200 | $0.016322 |
| `deepseek/deepseek-v3.2` | $0.013471 | $0.009096 | $0.022567 |
| `openai/gpt-5.6-luna` | $0.018590 | $0.016416 | $0.035006 |

All rows include complete recorded text and hero-cover cost. Optional promo-image and
landing-page tracks were disabled, so enabling either will add model calls and cost.

OpenAI-compatible local/network server, such as vLLM or llama.cpp:

```sh
uv run burn-pipeline \
  --provider-url http://localhost:11434 \
  --model local-model-name \
  run --pipeline automation/pipelines/burn-poc.yaml --force
```

Pipeline files can now declare separate providers for `text`, `image`, and `audio`.
That lets the writing steps use one model while future raster-generation or audio steps
use different models or endpoints.

Each step may optionally override the model used for that step. Resolution is:

1. `model` on the step.
2. `model` on the provider for that step's modality.
3. The CLI default (`--model`, `--image-model`, or `--audio-model`).
4. The first model advertised by an OpenAI-compatible provider when no model is set.

For example, this keeps the text provider's model as the default while sending the
lesson and worksheet copy to a specialist model:

```yaml
providers:
  text:
    kind: openrouter
    providerUrl: https://openrouter.ai/api/v1
    model: openai/gpt-4.1

steps:
  - id: lesson
    format: markdown
    prompt_file: automation/prompts/burn/lesson.md
    output: tmp/lesson.md
    model: anthropic/claude-sonnet-4
  - id: instructions
    format: markdown
    prompt_file: automation/prompts/burn/instructions.md
    output: tmp/instructions.md
    depends_on:
      - lesson
```

Example:

```yaml
providers:
  text:
    kind: openai-compatible
    providerUrl: http://localhost:11434
    model: active
  image:
    kind: openai-compatible
    providerUrl: http://localhost:11434
    model: unsloth-qwen-image-2512-gguf-qwen-image-2512-q4-k-m
  audio:
    kind: openai-compatible
    providerUrl: http://localhost:11434
    model: AUDIO_MODEL_NAME
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
- `content/letters/YYYY-MM-DD-slug/pipeline.yaml`

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
then writes `SEED.md`, `CONTEXT.md`, `index.md`, and `pipeline.yaml` into the final
letter folder. After that, `CONTEXT.md` becomes the primary source of truth for the
rest of the pipeline.

The generated pipeline now also includes a marketing-assets phase. That phase emits
prompt/spec files such as `PROMO_PROMPT.md`, which are intended to feed downstream
`png/jpeg` asset generation rather than pretending the text pipeline can write raster
images directly.

Promo-image assets and landing-page assets are optional tracks and are disabled in new
production plans. Enable promo assets only with `--enable-track promo_assets`; enable
the landing-page track with both `--enable-track promo_assets --enable-track landing_page`.
The promo track waits for the final long-form letter and rendered cover image.

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

Pipeline YAML files support ordered dependencies, direct file inputs, and references
to prior step outputs. Steps also support `modality = "text" | "image" | "audio"`,
though the current content pipeline defaults every step to `text`:

```yaml
steps:
  - id: c
    format: markdown
    prompt_file: automation/prompts/burn/c.md
    depends_on:
      - a
      - b
    modality: text
    output: tmp/c.md
    inputs:
      - step: a
        alias: first_pass
      - path: tmp/b.md
        alias: second_source
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
  --pipeline content/letters/YYYY-MM-DD-slug/pipeline.yaml \
  --step-id lesson
```

Backup path:

```sh
python scripts/burn-pipeline.py render-prompt \
  --pipeline content/letters/YYYY-MM-DD-slug/pipeline.yaml \
  --step-id lesson
```
