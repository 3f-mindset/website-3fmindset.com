# SteadyBurn Production Reference

## Source Material

Use these repo locations:

- `content/letters/`: SteadyBurn letter folders.
- `automation/prompts/burn/`: prompt templates for the generation harness. Many may be placeholders.
- `automation/pipelines/burn-all.template.toml`: canonical step order and file mapping.
- `automation/pipelines/YYYY-MM-DD-slug.toml`: per-letter pipeline plans created by `init-production`.
- `automation/pipelines/model-registry.json`: tracked registry of previously used worksheet models and actionable verbs.
- `automation/pipelines/steadyburn-verb-index.md`: numbered SteadyBurn series index starting at 1, including confirmed, inferred, and missing verb slots.
- `content/letters/YYYY-MM-DD-slug/PRODUCTION_BRIEF.md`: the intake file that stores the user's inspiration and becomes the root pipeline input.
- `scripts/codex-generate-burn-file.sh`: older single-file Codex generation harness.
- `scripts/build-the-burn-lesson-pdf.py`: renders Markdown artifacts to PDFs.
- `scripts/build-the-burn-worksheet-pdf.py`: renders worksheet SVGs to PDF and JPG.

Find the latest completed productions with:

```sh
find content/letters -maxdepth 2 -type f \
  \( -name 'LESSON.md' -o -name 'INSTRUCTIONS.md' -o -name 'CONTEXT.md' -o -name 'GPT.md' \) \
  | sort | tail -40
```

As of the current repo state, the most complete recent examples are:

- `content/letters/2026-06-26-your-day-is-training-you-even-when-you-are-not-paying-attention/`
- `content/letters/2026-06-19-the-people-shaping-your-future/`
- `content/letters/2026-06-12-your-space-is-training-you/`
- `content/letters/2026-05-29-what-breaks-first-when-life-gets-heavy/`

## Artifact Set

A completed AI-assisted production usually contains:

- `PRODUCTION_BRIEF.md`: structured intake document holding the user's inspiration, notes, and constraints.
- `index.md`: public Hugo letter with frontmatter, `{{< audio >}}`, body sections, and cover metadata.
- `LESSON.md`: core teaching script, usually headed `# THE LESSON`, `# THE SYSTEM`, and `# THE COMPONENT`.
- `INSTRUCTIONS.md`: guided worksheet process with an introduction and one section per model letter or step.
- `CONTEXT.md`: "CONTENT CRUSHER Response" style positioning document with title, subtitle, promise, motivations, challenges, frustrations, transformations, model, action steps, and FAQ.
- `PROMO_PROMPT.md`: downstream image-generation prompt for an evergreen square promo asset that will be rendered later as `png` or `jpeg`.
- `BANNER_PROMPT.md`: downstream image-generation prompt for a `4:3` landing-page hero banner that continues the same campaign style.
- `PAGE_COPY.md`: A/B squeeze-page copy paired with the banner and promo assets.
- `LANDING_PAGE.html`: responsive landing page HTML that combines the banner direction, page copy, and furnace/forge/flame visual language.
- `WORKSHEET_PAGE.md`: Hugo frontmatter prototype for the worksheet squeeze page in the existing `layout: worksheet` format.
- `NEWSLETTER_EMAIL.md`: student notification email built from the finished weekly letter and packet assets.
- `COMMUNITY_POST.md`: anticipation post for the group conversation, built from the essay introduction and weekly topic.
- `GPT.md` or `GPT_PROMPT.md`: system prompt for the worksheet coach.
- `WORKSHEET.svg`: landscape worksheet source.
- `WORKSHEET_MASKED.svg`: masked or student-facing worksheet variant.
- PDFs and JPGs generated from the Markdown/SVG sources.

Some older folders use week-specific names such as `SB19-lesson.md` or `FORGED.svg`; newer folders standardize on uppercase generic names.

## Dependency Order

Follow this order unless the user gives a narrower task:

1. Start from the target `index.md` draft or topic brief.
2. Write or revise `PRODUCTION_BRIEF.md` from the user's inspiration.
3. Generate `LESSON.md`.
4. Generate `INSTRUCTIONS.md` from `LESSON.md`.
5. Generate `CONTEXT.md` from `LESSON.md`.
6. Use the registry as a hard constraint during `CONTEXT.md` generation so the new `Actionable VERB` is not reused.
7. Record the new `Actionable VERB` back into the registry after `CONTEXT.md` is generated.
8. Generate `WORKSHEET.svg` from `LESSON.md` and `INSTRUCTIONS.md`.
9. Generate `WORKSHEET_MASKED.svg` from `WORKSHEET.svg`.
10. Generate `PROMO_PROMPT.md` in the marketing-assets phase from `CONTEXT.md` and `WORKSHEET_MASKED.svg`.
11. Generate `BANNER_PROMPT.md` in the marketing-assets phase from `CONTEXT.md`, `WORKSHEET_MASKED.svg`, and the promo brief so the landing-page hero continues the same campaign.
12. Generate `PAGE_COPY.md` in the marketing-assets phase from `CONTEXT.md`, the promo brief, and the banner brief.
13. Generate `LANDING_PAGE.html` in the marketing-assets phase from `CONTEXT.md`, the banner brief, and the page-copy brief.
14. Generate `WORKSHEET_PAGE.md` in the marketing-assets phase as the Hugo worksheet-page prototype from `CONTEXT.md`, the promo brief, and the page-copy brief.
15. Generate `GPT.md` from `CONTEXT.md` and `INSTRUCTIONS.md`.
16. Regenerate or refine final `index.md` from `LESSON.md` while preserving Hugo frontmatter.
17. Generate `NEWSLETTER_EMAIL.md` after the public letter so the email reflects the finished essay, packet, and weekly coach.
18. Generate `COMMUNITY_POST.md` after the public letter so the group teaser reflects the essay opening and the coming lesson.
19. Render PDFs/JPGs and downstream raster assets as needed.

## Content Shape

### `LESSON.md`

Use direct, masculine, reflective prose. Keep the language concrete and plain. Recent lessons usually include:

- `# THE LESSON`: the human problem, stakes, and behavior pattern.
- `# THE SYSTEM`: where the lesson fits in the 3F system, often Flame, Furnace, Forge, Anvil, or related model language.
- `# THE COMPONENT`: the named component or practice for the week.

Avoid hype, therapy softness, and abstract business language. Use short paragraphs, memorable lines, and occasional blockquotes.

### `INSTRUCTIONS.md`

Write this as a guided worksheet process. Recent examples:

- Start with a practical introduction that explains the worksheet's purpose.
- Use one section per model step, such as `# S: Survey the Full Day`.
- For each step include:
  - what the student is doing,
  - how to fill the section,
  - answers to avoid,
  - answers that work better,
  - a clear finish condition.

The tone should press toward action without shaming the student.

### `CONTEXT.md`

Use the "CONTENT CRUSHER Response" shape:

- `## Title`
- `## Subtitle`
- `## Promise`
- `## Motivations`
- `## Challenges & Sacrifices`
- `## Frustrations`
- `## Transformations`
- `## Choice`
- `## Actionable VERB`
- `## Model`
- `## Action Steps Summary`
- `## FAQ`

The `Actionable VERB` is usually a short memorable model name like `STRIKE`. The `Model` section expands each letter or step.

### `GPT.md`

Write a system prompt for a terse worksheet coach:

- Define the coach role and model steps.
- Set tone: direct, pragmatic, masculine, clear.
- Ask only one question at a time.
- Reject vague answers and require worksheet-ready responses.
- Include an opening message.
- Define step-by-step coaching rules for each model step.

### `index.md`

Preserve frontmatter and Hugo conventions:

```yaml
---
date: YYYY-MM-DD
slug: "slug"
title: "Title"
summary: "One-sentence summary."

series:
  - SteadyBurn

tags:
  - tag

cover:
  image: "cover.png"
  relative: true

draft: false
---
```

After frontmatter, include `{{< audio >}}`, then the public-facing letter. The public letter can differ from `LESSON.md`: it should read as an essay, not a worksheet script.

## Local Commands

Create a new letter folder:

```sh
uv run burn-pipeline init-production --title "Letter Title"
```

Backup path:

```sh
python scripts/burn-pipeline.py init-production --title "Letter Title"
```

Refresh the developed-model registry from historical productions:

```sh
uv run burn-pipeline sync-model-registry
```

Backup path:

```sh
python scripts/burn-pipeline.py sync-model-registry
```

Refresh the numbered SteadyBurn verb index:

```sh
uv run burn-pipeline sync-steadyburn-verb-index
```

Backup path:

```sh
python scripts/burn-pipeline.py sync-steadyburn-verb-index
```

Inspect the rendered prompt for one step:

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

Render a lesson-style PDF:

```sh
uv run scripts/build-the-burn-lesson-pdf.py \
  --markdown-file content/letters/YYYY-MM-DD-slug/LESSON.md \
  -o content/letters/YYYY-MM-DD-slug/LESSON.pdf
```

Render a worksheet PDF and JPG:

```sh
uv run scripts/build-the-burn-worksheet-pdf.py \
  content/letters/YYYY-MM-DD-slug/WORKSHEET.svg
```

Run the pipeline dry-run:

```sh
uv run burn-pipeline run \
  --pipeline automation/pipelines/YYYY-MM-DD-slug.toml \
  --dry-run
```

Backup path:

```sh
python scripts/burn-pipeline.py run \
  --pipeline automation/pipelines/YYYY-MM-DD-slug.toml \
  --dry-run
```

The default text inference route is `openai-compatible` at `http://localhost:11434` with model `active`.
Pipeline files may also include separate `providers.text`, `providers.image`, and `providers.audio`
blocks. Use that split when the local text model cannot generate raster images or audio.

## Validation Checklist

- The target folder has the expected Markdown and SVG source files.
- `PRODUCTION_BRIEF.md` reflects the latest user direction.
- `automation/pipelines/model-registry.json` includes the current set of developed verbs.
- `automation/pipelines/steadyburn-verb-index.md` numbers the SteadyBurn letters correctly and shows confirmed vs inferred vs missing verbs.
- Markdown source files do not start with a Markdown fence.
- SVG source files start with `<svg`.
- PDFs and JPGs exist for worksheet sources after rendering.
- Public `index.md` frontmatter is valid and still points to `cover.png`.
- Existing unrelated content is unchanged.
