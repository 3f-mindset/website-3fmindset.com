# Crucible Pulp-Style Cover Images

## Goal

Add one generated cover image to each of the 43 published Crucible journal-prompt essays. The owner approved the first sample, “The room inside an ordinary day,” in merged PR #145. Generate the remaining 42 with the same tool/style and submit them in one reviewable rollout PR.

## User-visible behavior

- Each Crucible essay has its own article-specific wide cover image, saved as `cover.jpg` in its existing Hugo page bundle. Provider PNG responses are encoded as quality-70 progressive JPEGs before persistence.
- A root-level `generate-crucible-cover.sh` accepts a letter directory (or uses the current directory when it contains `index.md`) and an optional `--prompt` scene direction. It combines the shared pulp style with context from that article and writes the generated image into the article directory. `--dry-run` previews the combined prompt without making a provider request.
- Each `index.md` references the image using the site's existing relative cover convention:

  ```yaml
  cover:
    image: "cover.jpg"
    relative: true
  ```

- The images share the requested vintage pulp-magazine illustration style, but each one depicts a distinct object or action grounded in its essay's main tension.
- Images contain no words, captions, typography, labels, signage, or readable marks.
- The approved first cover from PR #145 remains unchanged. The rollout adds one distinct cover to each of the other 42 letters and leaves essay prose, titles, dates, slugs, categories, and links unchanged.

## Non-goals

- Do not change essay text, title, date, slug, category, links, or the vision exercise.
- Do not add cover images to non-Crucible letters or program pages.
- Do not make one generic image and reuse it across essays.
- Do not generate or commit audio, promo art, banners, thumbnails, or other marketing assets.
- Do not call OpenAI directly or use an OpenAI API key; image generation routes through OpenRouter using an OpenAI image model.
- Do not regenerate or replace the approved first cover during the rollout.

## Acceptance criteria

1. **Scope inventory.** Given the 43 Crucible essays on `main`, when the set is inventoried, then every essay is identified once in source-question order and the first review target is `content/letters/2026-01-04-crucible-what-has-been-taking-up-space/`.
2. **Approved sample preserved.** Given PR #145 is merged and approved, when the rollout is prepared, then the first cover image and its frontmatter remain unchanged; its served asset is `cover.jpg`.
3. **Visual style.** Given the first image is generated, when it is reviewed at full size and thumbnail size, then it is a wide, object-centered vintage pulp illustration with visible halftone texture, bold black linework, cross-hatching, a limited muted earthy palette, a dynamic but readable composition, and no words or readable marks.
4. **Article relevance.** Given the first letter is about attention, crowding, and making room for what matters, when the image is inspected, then it shows one clear article-specific visual idea rather than an unrelated generic cover, a worksheet, or a document.
5. **Generation route and safety.** Given the image is requested, when the provider call is made, then it uses OpenRouter's image-generation endpoint with the OpenAI model `openai/gpt-image-2`, a 3:2 aspect ratio, and a PNG response. `OPENROUTER_API_KEY` is read from the caller's root `.env` in process and sent only as an Authorization header; it is never printed, logged, committed, or passed as a CLI argument.
6. **Root script.** Given an author runs `./generate-crucible-cover.sh <letter-directory> [--prompt "scene direction"]`, when the directory contains a valid `index.md`, then the script combines the shared style prompt with the article title/opening context and optional scene direction. With no directory argument, the current directory is accepted only if it contains `index.md`.
7. **Safe output.** Given OpenRouter returns a valid PNG, when the script completes, then it writes `cover.jpg` in the letter directory and inserts the relative `cover.image: "cover.jpg"` metadata without changing the essay text or other frontmatter. Existing images are not overwritten unless `--force` is supplied; an existing incompatible cover declaration is an error.
8. **Rollout scope.** Given first-sample approval, when the remaining set is generated, then exactly one distinct `cover.jpg` and relative cover metadata are added to each of the other 42 Crucible letters, with no prose/date/slug/category/link changes.
9. **Hugo integration.** Given the first sample is built by Hugo, when the article renders, then the relative cover is found and displayed without altering existing page rendering behavior.
10. **Script validation.** Given no valid root `.env`, API key, article directory, or prompt can be resolved, when the script runs, then it exits with a clear non-secret error before writing files. `bash -n` passes and help, dry-run, and argument validation can be exercised without making a provider request.
11. **Rollout PR scope.** Given the remaining covers are generated, when the rollout PR is inspected, then it contains the 42 remaining cover images and matching frontmatter additions, plus this spec, with no unrelated files or changes to the approved first cover.
12. **Architecture boundaries.** Given the generator is refactored, when its dependency direction is inspected, then domain and prompt/image rules are independent of file and network I/O; the application use case depends on ports; filesystem and OpenRouter adapters implement those ports; and the CLI is the outer composition root. The root Bash script only sources `.env` and delegates to the CLI.
13. **Behavior preservation.** Given a caller uses the existing path, prompt, `--force`, and `--dry-run` interface, when the refactored command runs, then prompt composition, output path, frontmatter changes, overwrite protection, provider parameters, errors, and secret handling remain unchanged. Offline tests use fake ports/adapters and make no paid API request.
14. **Usage reporting.** Given each image response includes provider usage, when the rollout PR is prepared, then its summary reports the sum of captured provider-reported costs and clearly identifies any successful generations whose cost output was not retained. It does not present an estimate as an exact total or include credentials.

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

- `ASSUMPTION:` Use `openai/gpt-image-2` through OpenRouter's image API (`https://openrouter.ai/api/v1/images`) because the earlier `openai/gpt-image-1` request returned a model-not-found error and OpenRouter lists `openai/gpt-image-2` with the required 3:2 and PNG parameters.
- `ASSUMPTION:` Use a 3:2 composition, matching the dimensions of existing letter covers (typically 1536 × 1024), and save the returned PNG without destructive cropping. Verify the actual response dimensions before commit.
- `ASSUMPTION:` Use `quality: high` for the first review sample; record provider-reported usage/cost in the PR summary without including credentials.
- Record captured provider-reported generation costs in the rollout PR summary without including credentials; disclose any lost cost output rather than guessing an exact aggregate.
- `ASSUMPTION:` Group the remaining 42 images into one rollout PR to keep the approved visual system and all related article changes together.
- The required OpenRouter key must already be available in the caller root `.env`; if it is absent, stop and ask the owner to configure it. Do not search unrelated credential stores, create a key, or substitute another provider.
- The root Bash entrypoint sources only the root `.env` and delegates to `python3 -m crucible_cover.cli`; Python modules live under `scripts/crucible_cover/`. The key stays in the environment/header and never appears in command arguments or output.
- Keep Clean Architecture/Onion dependency direction: domain rules have no OpenRouter, filesystem, CLI, or environment dependencies; the application orchestrates ports; adapters implement those ports; and the CLI/composition root wires dependencies. Apply SOLID at this scale without adding external dependencies or generic frameworks.
- `ASSUMPTION:` The optional scene prompt supplements, rather than replaces, the article-derived title and opening context.
- Preserve all existing untracked files and local changes.

## Ordered implementation plan

1. Confirm the Crucible series inventory, existing cover/frontmatter convention, image-generation endpoint, and the first essay's story-specific visual subject.
2. Add this spec before generating any image.
3. Refactor the root entrypoint into a thin `.env`-loading wrapper and a small `scripts/crucible_cover/` package with domain, application, ports, filesystem/OpenRouter adapters, and a CLI composition root; validate the same behavior using offline fakes.
4. Confirm the approved first sample remains unchanged and the root `.env` contains the valid OpenRouter key.
5. Use `generate-crucible-cover.sh` on each remaining letter directory, deriving a unique scene from article context and using a custom scene prompt only where it improves relevance. Do not reuse images.
6. Inspect every PNG's format, dimensions, aspect ratio, absence of visible text, thumbnail readability, and article relevance; verify all 42 cover fields and ensure article text/metadata are otherwise unchanged.
7. Open one rollout PR for the 42 remaining images and frontmatter additions. Report the captured provider costs, identify any successful generations with uncaptured cost output, make no exact-cost estimate, and leave the PR open for review.

## Rollback

If a rollout image is unsuitable, remove or replace only that letter's new `cover.jpg` and cover frontmatter. Keep the approved first cover unchanged and preserve all essay content.
