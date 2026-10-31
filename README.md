# 3F Mindset website

This repository contains the published Hugo site and the website-local tools used to author, preview, render, and publish its content.

## Local authoring and preview

- Install Hugo (extended version when needed) and run `hugo server -D` to preview drafts.
- Use `new-letter.sh` to scaffold a newsletter, then edit its files under `content/letters/`.
- Use `scripts/start-wsl-hugo-preview.sh` and `scripts/stop-wsl-hugo-preview.sh` for the WSL preview server.
- Newsletter lesson and worksheet PDFs can be rendered with the scripts in `scripts/`.
- Build the production site with `hugo --minify`.

GitHub Pages publishes the Hugo site. SteadyBurn package production and assessment are owned by the [3F Gutenberg writer workflow](https://github.com/3f-mindset/3f-gutenberg/blob/main/WRITERS.md) and its [SteadyBurn production contract](https://github.com/3f-mindset/3f-gutenberg/blob/main/specs/steadyburn-production-contract-v1.md). Promote approved output into this repository explicitly for publication.
