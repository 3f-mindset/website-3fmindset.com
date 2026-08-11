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
3. Use `uv run burn-pipeline init-production --title "Letter Title" [--date YYYY-MM-DD]` only when you need a blank scaffold with `SEED.md`, or use the backup harness `python scripts/burn-pipeline.py init-production ...`.
4. When you already have a real seed document, prefer `uv run burn-pipeline seed-production --seed-file path/to/seed.md [--date YYYY-MM-DD]`, or the backup harness `python scripts/burn-pipeline.py seed-production ...`.
5. Treat `SEED.md` as the human intake document and `CONTEXT.md` as the generated system-of-record document that controls slug, title, promise, and worksheet model.
6. Sync the model registry with `uv run burn-pipeline sync-model-registry` or `python scripts/burn-pipeline.py sync-model-registry` when the repo has gained new historical productions or the registry may be stale.
7. Read the target `index.md` draft and inspect the latest completed production folders for style and file shape when the prompts need stronger examples.
8. Generate artifacts in dependency order:
   - `CONTEXT.md` first when starting from a seed
   - `LESSON.md`
   - `INSTRUCTIONS.md`
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
9. Treat `CONTEXT.md` as the model-definition step and primary source of truth. Its `Actionable VERB` must be new; the pipeline injects previously used verbs as forbidden constraints and then records the newly generated verb back into the registry after success.
10. Use the marketing-assets phase after Content Crusher so promotional creative prompt assets inherit the same topic, pain, promise, and worksheet silhouette before downstream `png/jpeg` rendering.
11. Render production derivatives:

- lesson/instructions/context/GPT PDFs with `scripts/build-the-burn-lesson-pdf.py`
- worksheet PDFs and JPGs with `scripts/build-the-burn-worksheet-pdf.py`

12. Validate that Markdown files contain plain Markdown only and SVG files start with `<svg`.

## Generation Routes

Prefer the pipeline route because it carries forward the output of one step into the next step through aliased inputs and step references.
It also uses the model registry to prevent reused worksheet verbs.

Use prompt rendering before generation when you need to inspect or debug a step:

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

Run the full pipeline when the context-first letter folder is ready:

```sh
uv run burn-pipeline run \
  --pipeline content/letters/YYYY-MM-DD-slug/pipeline.yaml \
  --force
```

Backup path:

```sh
python scripts/burn-pipeline.py run \
  --pipeline content/letters/YYYY-MM-DD-slug/pipeline.yaml \
  --force
```

Use the single-step route only for focused regeneration or debugging:

```sh
uv run burn-pipeline generate-step \
  --format markdown \
  --prompt-file automation/prompts/burn/lesson.md \
  --input content/letters/YYYY-MM-DD-slug/CONTEXT.md \
  --input content/letters/YYYY-MM-DD-slug/SEED.md \
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
The default local image model is `unsloth-qwen-image-2512-gguf-qwen-image-2512-q4-k-m`.

## Guardrails

- Keep all generated files inside the target letter folder unless the user asks otherwise.
- Do not overwrite existing production files without checking whether the change is intentional.
- Keep `SEED.md` as the authoritative human intake file for user inspiration.
- Keep `CONTEXT.md` as the authoritative generated file for title, promise, slug, and worksheet model.
- Preserve Hugo frontmatter conventions in `index.md`: `date`, quoted `slug`, quoted `title`, `summary`, `series: SteadyBurn`, tags, cover block, and `draft: false`.
- Keep generated Markdown free of code fences around the whole file.
- Keep generated SVG as a complete XML document that begins with `<svg`.
- Treat the latest completed folders as stronger evidence than placeholder prompt templates.
