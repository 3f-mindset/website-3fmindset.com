# Crucible Pulp-Style Cover Images

## Goal

Add one generated cover image to each of the 43 published Crucible journal-prompt essays. Start with the first essay, “The room inside an ordinary day” (introduced in PR #102), so the owner can review the visual direction before any other cover is generated. The remaining 42 images must stay unstarted until the owner confirms the first sample.

## User-visible behavior

- Each Crucible essay has its own article-specific wide cover image, saved as `cover.png` in its existing Hugo page bundle.
- A root-level `generate-crucible-cover.sh` accepts a letter directory (or uses the current directory when it contains `index.md`) and an optional `--prompt` scene direction. It combines the shared pulp style with context from that article and writes the generated image into the article directory. `--dry-run` previews the combined prompt without making a provider request.
- Each `index.md` references the image using the site's existing relative cover convention:

  ```yaml
  cover:
    image: "cover.png"
    relative: true
  ```

- The images share the requested vintage pulp-magazine illustration style, but each one depicts a distinct object or action grounded in its essay's main tension.
- Images contain no words, captions, typography, labels, signage, or readable marks.
- The root generation script and this spec can be reviewed in a tool PR. The first-cover review PR contains one image and its cover frontmatter; no other images are generated or added before explicit approval.

## Non-goals

- Do not change essay text, title, date, slug, category, links, or the vision exercise.
- Do not add cover images to non-Crucible letters or program pages.
- Do not make one generic image and reuse it across essays.
- Do not generate or commit audio, promo art, banners, thumbnails, or other marketing assets.
- Do not call OpenAI directly or use an OpenAI API key; image generation routes through OpenRouter using an OpenAI image model.
- Do not roll out the remaining 42 covers until the owner approves the first sample.

## Acceptance criteria

1. **Scope inventory.** Given the 43 Crucible essays on `main`, when the set is inventoried, then every essay is identified once in source-question order and the first review target is `content/letters/2026-01-04-crucible-what-has-been-taking-up-space/`.
2. **First sample asset.** Given the first-cover review change is prepared, when its cover is reviewed, then it contains one PNG cover for “The room inside an ordinary day,” stored as `cover.png` beside `index.md`, with relative cover frontmatter and no changes to article prose or unrelated files.
3. **Visual style.** Given the first image is generated, when it is reviewed at full size and thumbnail size, then it is a wide, object-centered vintage pulp illustration with visible halftone texture, bold black linework, cross-hatching, a limited muted earthy palette, a dynamic but readable composition, and no words or readable marks.
4. **Article relevance.** Given the first letter is about attention, crowding, and making room for what matters, when the image is inspected, then it shows one clear article-specific visual idea rather than an unrelated generic cover, a worksheet, or a document.
5. **Generation route and safety.** Given the image is requested, when the provider call is made, then it uses OpenRouter's image-generation endpoint with the OpenAI model `openai/gpt-image-1`, a 3:2 aspect ratio, and a PNG response. `OPENROUTER_API_KEY` is read from the caller's root `.env` in process and sent only as an Authorization header; it is never printed, logged, committed, or passed as a CLI argument.
6. **Root script.** Given an author runs `./generate-crucible-cover.sh <letter-directory> [--prompt "scene direction"]`, when the directory contains a valid `index.md`, then the script combines the shared style prompt with the article title/opening context and optional scene direction. With no directory argument, the current directory is accepted only if it contains `index.md`.
7. **Safe output.** Given OpenRouter returns a valid PNG, when the script completes, then it writes `cover.png` in the letter directory and inserts the relative `cover.image: "cover.png"` metadata without changing the essay text or other frontmatter. Existing images are not overwritten unless `--force` is supplied; an existing incompatible cover declaration is an error.
8. **Review gate.** Given the first-sample PR is open, when the owner has not yet confirmed the image, then no assets or metadata changes are made to the other 42 Crucible essays. After approval, a separate rollout plan can cover the remaining images.
9. **Hugo integration.** Given the first sample is built by Hugo, when the article renders, then the relative cover is found and displayed without altering existing page rendering behavior.
10. **Script validation.** Given no valid root `.env`, API key, article directory, or prompt can be resolved, when the script runs, then it exits with a clear non-secret error before writing files. `bash -n` passes and help, dry-run, and argument validation can be exercised without making a provider request.
11. **Tool PR scope.** Given the root script is ready but the image credential is unavailable, when its tool PR is opened, then it contains only this spec and the script; no cover generation or letter frontmatter change is claimed as complete.
12. **Architecture boundaries.** Given the generator is refactored, when its dependency direction is inspected, then domain and prompt/image rules are independent of file and network I/O; the application use case depends on ports; filesystem and OpenRouter adapters implement those ports; and the CLI is the outer composition root. The root Bash script only sources `.env` and delegates to the CLI.
13. **Behavior preservation.** Given a caller uses the existing path, prompt, `--force`, and `--dry-run` interface, when the refactored command runs, then prompt composition, output path, frontmatter changes, overwrite protection, provider parameters, errors, and secret handling remain unchanged. Offline tests use fake ports/adapters and make no paid API request.

## Shared style prompt

Use the owner's supplied prompt as the shared style direction for every image:

> Generate a wide illustration. The illustration features a vintage style art with strong influences from the pulp magazine era characterized by:
>
> 1. **Halftone Dot Pattern:** The entire image uses a halftone dot effect, which gives it a textured, printed look, typical of old comic books and newspapers.
> 2. **Bold, Black Linework:** The lines outlining the objects, symbols, or elements in the scene are thick and prominent, providing a strong contrast against the softer colors and making the details stand out.
> 3. **Limited, Muted Color Palette:** The colors are desaturated and muted, mostly earthy tones like browns, beiges, and faded blues, which create a nostalgic and aged feel.
> 4. **Cross-hatching and Shading:** The shading on objects, symbols, and the background is done using cross-hatching, enhancing the vintage look and adding depth to the illustration.
> 5. **Dynamic Composition:** The arrangement of objects or symbols is energetic and slightly exaggerated, emphasizing movement or impact in the scene. Shapes and forms are expressive, capturing a moment of tension or drama.
> 6. **Stylized Clouds and Background Elements:** The clouds, abstract symbols, or environmental elements in the background are simplified and stylized, with the same halftone texture applied, ensuring consistency with the overall retro aesthetic.
>
> These characteristics combine to give the image a distinct, old-timey, illustrative feel, akin to something from a 1940s-1950s comic book or an early 20th-century advertisement. NO WORDS IN THE IMAGE, NO CAPTIONS.

Add a concise scene description grounded in the individual essay. The scene must have one main focal point, simple readable shapes, a secondary uncluttered background, and no text-like marks. The first article's scene concept is a small brass compass in a clear pool of warm light on an ordinary table, with an old rotary phone and a clock at the outer edge suggesting competing demands. The compass is the single focal point; do not add paper or a worksheet.

## Constraints and assumptions

- `ASSUMPTION:` Use `openai/gpt-image-1` through OpenRouter's image API (`https://openrouter.ai/api/v1/images`) because this is an OpenAI image model served via OpenRouter.
- `ASSUMPTION:` Use a 3:2 composition, matching the dimensions of existing letter covers (typically 1536 × 1024), and save the returned PNG without destructive cropping. Verify the actual response dimensions before commit.
- `ASSUMPTION:` Use `quality: high` for the first review sample; record provider-reported usage/cost in the PR summary without including credentials.
- The required OpenRouter key must already be available in the caller root `.env`; if it is absent, stop and ask the owner to configure it. Do not search unrelated credential stores, create a key, or substitute another provider.
- The root Bash entrypoint sources only the root `.env` and delegates to `python3 -m crucible_cover.cli`; Python modules live under `scripts/crucible_cover/`. The key stays in the environment/header and never appears in command arguments or output.
- Keep Clean Architecture/Onion dependency direction: domain rules have no OpenRouter, filesystem, CLI, or environment dependencies; the application orchestrates ports; adapters implement those ports; and the CLI/composition root wires dependencies. Apply SOLID at this scale without adding external dependencies or generic frameworks.
- `ASSUMPTION:` The optional scene prompt supplements, rather than replaces, the article-derived title and opening context.
- Preserve all existing untracked files and local changes.

## Ordered implementation plan

1. Confirm the Crucible series inventory, existing cover/frontmatter convention, image-generation endpoint, and the first essay's story-specific visual subject.
2. Add this spec before generating any image.
3. Refactor the root entrypoint into a thin `.env`-loading wrapper and a small `scripts/crucible_cover/` package with domain, application, ports, filesystem/OpenRouter adapters, and a CLI composition root; validate the same behavior using offline fakes.
4. Open a tool PR for the spec and script. The root `.env` key is required before any live image request.
5. Once the key is available, generate only the first essay's cover through OpenRouter, save it as `cover.png`, and add relative cover frontmatter to the first essay in a separate first-sample review PR.
6. Inspect the PNG's format, dimensions, aspect ratio, absence of visible text, thumbnail readability, and article relevance; run Markdown validation and Hugo rendering/build checks when available.
7. Stop after sharing the first-sample PR; wait for explicit approval before generating or changing any of the remaining 42 essays.

## Rollback

If the first visual sample is rejected, close the sample PR and remove its new `cover.png` and cover frontmatter. Keep the root script/spec if the owner still wants the generation utility; no other Crucible letter is changed in the review phase.
