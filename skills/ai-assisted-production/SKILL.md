---
name: ai-assisted-production
description: Generate or finish 3F Mindset SteadyBurn AI-assisted production artifacts from a letter draft or production folder. Use when creating LESSON.md, INSTRUCTIONS.md, CONTEXT.md, GPT.md, WORKSHEET.svg, WORKSHEET_MASKED.svg, PROMO_PROMPT.md, BANNER_PROMPT.md, PAGE_COPY.md, LANDING_PAGE.html, WORKSHEET_PAGE.md, NEWSLETTER_EMAIL.md, COMMUNITY_POST.md, PDFs/JPGs, or final Hugo index.md files under content/letters, or when running the local burn production scripts and pipeline.
---

# AI Assisted Production

## Overview

Use this skill to turn a SteadyBurn letter draft into a production-ready artifact set using this repo's existing scripts, pipeline files, and recent completed letters as examples.

Before generating content, read `references/steadyburn-production.md`. It defines the artifact set, dependency order, quality bar, and local commands.

## Workflow

1. Collect the user's inspiration first: tension, reader, promise, scenes, phrases, constraints, and desired worksheet outcome.
2. Identify the target letter folder under `content/letters/YYYY-MM-DD-slug`.
3. Initialize the production scaffold with `uv run burn-pipeline init-production --title "Letter Title" [--date YYYY-MM-DD]`, or use the backup harness `python scripts/burn-pipeline.py init-production --title "Letter Title" [--date YYYY-MM-DD]`, or update the existing folder if it is already present.
4. Write the collected inspiration into `PRODUCTION_BRIEF.md`. This file is the intake document the pipeline uses as the root input for later steps.
5. Sync the model registry with `uv run burn-pipeline sync-model-registry` or `python scripts/burn-pipeline.py sync-model-registry` when the repo has gained new historical productions or the registry may be stale.
6. Read the target `index.md` draft and inspect the latest completed production folders for style and file shape when the prompts need stronger examples.
7. Generate artifacts in dependency order:
   - `LESSON.md`
   - `INSTRUCTIONS.md`
   - `CONTEXT.md`
   - `WORKSHEET.svg`
   - `WORKSHEET_MASKED.svg`
   - marketing assets phase:
     - `PROMO_PROMPT.md`
     - `BANNER_PROMPT.md`
     - `PAGE_COPY.md`
     - `LANDING_PAGE.html`
     - `WORKSHEET_PAGE.md`
   - `GPT.md`
   - final `index.md`
   - `NEWSLETTER_EMAIL.md`
   - `COMMUNITY_POST.md`
8. Treat `CONTEXT.md` as the model-definition step. Its `Actionable VERB` must be new; the pipeline injects previously used verbs as forbidden constraints and then records the newly generated verb back into the registry after success.
9. Use the marketing-assets phase after Content Crusher so promotional creative prompt assets inherit the same topic, pain, promise, and worksheet silhouette before downstream `png/jpeg` rendering.
10. Render production derivatives:
   - lesson/instructions/context/GPT PDFs with `scripts/build-the-burn-lesson-pdf.py`
   - worksheet PDFs and JPGs with `scripts/build-the-burn-worksheet-pdf.py`
11. Validate that Markdown files contain plain Markdown only and SVG files start with `<svg`.

## Generation Routes

Prefer the pipeline route because it carries forward the output of one step into the next step through aliased inputs and step references.
It also uses the model registry to prevent reused worksheet verbs.

Use prompt rendering before generation when you need to inspect or debug a step:

```sh
uv run burn-pipeline render-prompt \
  --pipeline automation/pipelines/YYYY-MM-DD-slug.toml \
  --step-id lesson
```

Backup path:

```sh
python scripts/burn-pipeline.py render-prompt \
  --pipeline automation/pipelines/YYYY-MM-DD-slug.toml \
  --step-id lesson
```

Run the full pipeline when the brief and plan are ready:

```sh
uv run burn-pipeline run \
  --pipeline automation/pipelines/YYYY-MM-DD-slug.toml \
  --force
```

Backup path:

```sh
python scripts/burn-pipeline.py run \
  --pipeline automation/pipelines/YYYY-MM-DD-slug.toml \
  --force
```

Use the single-step route only for focused regeneration or debugging:

```sh
uv run burn-pipeline generate-step \
  --format markdown \
  --prompt-file automation/prompts/burn/lesson.md \
  --input content/letters/YYYY-MM-DD-slug/PRODUCTION_BRIEF.md \
  --input content/letters/YYYY-MM-DD-slug/index.md \
  --output content/letters/YYYY-MM-DD-slug/LESSON.md \
  --title "Letter Title" \
  --slug "slug" \
  --date YYYY-MM-DD \
  --force
```

The default text inference route is `openai-compatible` at `http://localhost:11434` with model `active`.
Pipeline files can also define separate `providers.text`, `providers.image`, and `providers.audio`
blocks so future image or audio steps can run on different models without changing the text flow.

## Guardrails

- Keep all generated files inside the target letter folder unless the user asks otherwise.
- Do not overwrite existing production files without checking whether the change is intentional.
- Keep `PRODUCTION_BRIEF.md` as the authoritative intake file for user inspiration. Update it when the user's direction changes instead of silently carrying separate hidden notes.
- Preserve Hugo frontmatter conventions in `index.md`: `date`, quoted `slug`, quoted `title`, `summary`, `series: SteadyBurn`, tags, cover block, and `draft: false`.
- Keep generated Markdown free of code fences around the whole file.
- Keep generated SVG as a complete XML document that begins with `<svg`.
- Treat the latest completed folders as stronger evidence than placeholder prompt templates.
