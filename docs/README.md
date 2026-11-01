# Website documentation

This directory contains specifications and documentation for the Hugo website. Published content and its source files remain under `content/`.

SteadyBurn production and model assessment are maintained by 3F Gutenberg, pinned in this repository at `tools/gutenberg`. From the website root, `new-burn.sh` and `new-letter.sh` call that pinned generator to create a new bundle in `content/letters/`; provide `--source`, `--brief`, `--title`, `--provider`, `--model`, `--image-provider`, and `--image-model`. Provider keys belong in environment variables, and `--batch` is an explicit opt-in. Existing destinations are never overwritten.

After generation, render derivatives manually with `uv run scripts/build-the-burn-lesson-pdf.py --markdown-file content/letters/<date>-<slug>/LESSON.md` and `uv run scripts/build-the-burn-worksheet-pdf.py content/letters/<date>-<slug>/WORKSHEET.svg`. Review and commit approved bundles through a normal website PR; GitHub Pages only builds committed site content and does not generate content. See the [Gutenberg writer workflow](https://github.com/3f-mindset/3f-gutenberg/blob/main/WRITERS.md) and [SteadyBurn production contract](https://github.com/3f-mindset/3f-gutenberg/blob/main/specs/steadyburn-production-contract-v1.md).
