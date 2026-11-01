# 3F Mindset website

This repository contains the published Hugo site and the website-local tools used to author, preview, render, and publish its content.

## Local authoring and preview

- Install Hugo (extended version when needed) and run `hugo server -D` to preview drafts.
- Set up local Gutenberg settings with `cp tools/gutenberg/.env.example .env` and fill in the provider API keys plus `GUTENBERG_PROVIDER`, `GUTENBERG_MODEL`, `GUTENBERG_IMAGE_PROVIDER`, and `GUTENBERG_IMAGE_MODEL` as needed. The root `.gitignore` excludes `.env`; never commit real keys. The CLI reads this caller-root file, while process environment variables take precedence over `.env` and explicit wrapper flags take precedence over both.
- Use `new-letter.sh` or `new-burn.sh` to generate a bundle with the pinned Gutenberg submodule. Run from the repository root and provide a source, brief, and title. Provider/model and endpoint settings may come from `.env` or be supplied as explicit flags. `new-letter.sh` defaults to today's date and `new-burn.sh` to the next Friday. Both default the slug from the title, write to `content/letters/<date>-<slug>`, refuse existing destinations, and leave batching at its configured default unless overridden with `--batch` or `--no-batch`.
- For explicit overrides, run `./new-burn.sh --source sources/seed.md --brief "Focus on one concrete practice" --title "Example" --provider openrouter --model openai/gpt-6-luna --image-provider openai --image-model gpt-image-1`. Pass `--date` and `--slug` to override the defaults; repeat `--source` for additional files/directories.
- Use `scripts/start-wsl-hugo-preview.sh` and `scripts/stop-wsl-hugo-preview.sh` for the WSL preview server.
- After generation, render derivatives manually: `uv run scripts/build-the-burn-lesson-pdf.py --markdown-file content/letters/<date>-<slug>/LESSON.md` and `uv run scripts/build-the-burn-worksheet-pdf.py content/letters/<date>-<slug>/WORKSHEET.svg`. These website-owned scripts create the PDF/JPG outputs alongside their inputs.
- Build the production site with `hugo --minify`.

GitHub Pages publishes the Hugo site. SteadyBurn package production and assessment are owned by the [3F Gutenberg writer workflow](https://github.com/3f-mindset/3f-gutenberg/blob/main/WRITERS.md) and its [SteadyBurn production contract](https://github.com/3f-mindset/3f-gutenberg/blob/main/specs/steadyburn-production-contract-v1.md). Promote approved output into this repository explicitly for publication.
