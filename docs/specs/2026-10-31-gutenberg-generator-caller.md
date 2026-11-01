# Website caller for Gutenberg editorial generation

## Goal

Make the website repository a caller and publication owner for Gutenberg-generated SteadyBurn and future editorial bundles. Retire website-local placeholder content generation while keeping Hugo rendering, PDF/JPG conversion, review, and GitHub Pages deployment here.

## User-visible behavior

- The website pins an exact Gutenberg generator revision as a submodule under `tools/gutenberg`.
- A writer supplies source files, brief, title, date/slug, text provider/model and image provider/model; the wrapper invokes Gutenberg and writes the complete bundle directly into a new directory under `content/letters/`.
- The website retains its existing worksheet/lesson PDF/JPG renderers and runs them after Gutenberg generation.
- Existing website Pages build/deploy continues to use only committed site content. It does not fetch floating Gutenberg `main` or generate content dynamically during deployment.
- Existing published content remains unchanged by the integration PR; the site PR only changes future generation entry points and documentation.

## Non-goals

- Do not add generation prompts, provider adapters, model scoring, research dashboards, or a website-specific editorial pipeline.
- Do not remove/rewrite already published SteadyBurn letters, program pages, or curriculum.
- Do not add generated content to this PR. Generated output belongs in a later caller-owned content PR.
- Do not change Hugo layouts/theme, site rendering, Pages permissions/concurrency, or manual deployment behavior.
- Do not configure model-research batch mode in the Pages workflow; the site caller defaults to synchronous generation unless explicitly opted in.

## Acceptance criteria

1. **Pinned generator.** `.gitmodules` defines `tools/gutenberg` at an exact Gutenberg commit used by the wrapper; neither the wrapper nor the Pages workflow fetches a floating branch for generation.
2. **No local generation.** `new-burn.sh` and `new-letter.sh` become thin, safe Gutenberg callers. They do not create placeholder Markdown/SVG or placeholder cover images, and they do not contain editorial prompts/templates.
3. **Required inputs.** Before creating the output directory, the wrapper validates a caller-owned source file/directory, focus brief, title/date/slug, provider/model settings, API-key environment, and required image model settings. Missing inputs fail without mutating `content/`.
4. **Caller-owned output.** A successful invocation calls the pinned generator from the website working tree and writes the bundle to `content/letters/<date>-<slug>`. An existing/non-empty destination is refused; no existing content is overwritten.
5. **Rendering and publication.** The wrapper runs or clearly instructs the author to run the website-owned lesson/worksheet PDF/JPG rendering scripts after generation. Site publication remains a normal website PR; merging committed content triggers the existing Hugo/Pages workflow.
6. **No runtime fetch.** The Pages workflow still builds the already committed `content/` tree only. It does not invoke Gutenberg generation, call provider APIs, or select the latest Gutenberg revision. Its CI checkout initializes only the Hugo theme submodule, not the private generator submodule used by local caller scripts.
7. **Preservation.** Existing `content/**`, Hugo rendering inputs, `new-letter.sh` public destination conventions, and Pages workflow permissions/concurrency/manual dispatch remain intact except for the thin caller wrappers and submodule declaration. The pre-existing untracked edge-domain spec remains untouched.
8. **Verification.** `bash -n` passes for wrappers; mocked invocation tests prove caller-relative output, required inputs, non-overwrite behavior, pinned generator path and model flags. `hugo --minify` succeeds on the existing site without generated content or live provider calls.

## Constraints

- Gutenberg generator source is canonical at `https://github.com/3f-mindset/3f-gutenberg`; this website PR will pin the generator feature commit from Gutenberg PR #6.
- The website owns `content/`, frontmatter consumption, Hugo/theme/layout rendering, PDF/JPG conversion, and GitHub Pages deployment. The generator owns editorial prompts, dependency graph, text/image generation, and the content bundle manifest.
- API secrets are passed via environment variables and never written to generated files, command logs, or Git.
- The Pages workflow already checks out submodules recursively. It must remain a deploy-only renderer for committed content, not a generation runner.
- The current generator supports only the versioned SteadyBurn profile. The wrapper can accept `--profile` for future profile versions, but must not invent or fabricate an unsupported profile.

## Assumptions

- **ASSUMPTION:** The first wrapper revision requires explicit source/brief/title and provider/model settings; it does not silently choose a model. A future production default (such as GPT-6 Luna) can be configured separately once the owner finalizes that policy.
- **ASSUMPTION:** `new-burn.sh` keeps its familiar “next Friday” date default; `new-letter.sh` keeps the current-day default unless the caller supplies a date. Both remain wrappers over the same pinned Gutenberg generator.
- **ASSUMPTION:** Model-generated PDFs/JPGs remain website-side post-processing, not Gutenberg output. The standard bundle generation command does not run during Pages deployment.

## Plan and touched areas

1. Pin the Gutenberg generator source in `.gitmodules` at the reviewed feature commit.
2. Replace local scaffold/placeholder behavior in `new-burn.sh` and `new-letter.sh` with thin argument validation plus an invocation of the pinned Gutenberg CLI. Ensure checks happen before directory creation and refuse overwrites.
3. Keep website-owned PDF/JPG helper scripts unchanged; update `README.md`/`docs/README.md` with the generation handoff and manual render steps.
4. Add mocked wrapper tests and verify exact submodule SHA, protected content tree, workflow permissions/concurrency/manual dispatch, theme-only submodule checkout, and Hugo production build.

## Rollback

- Revert the website caller PR to restore the prior helper scripts; no generated content or existing page is changed by this migration PR.
- Keep Gutenberg source pinned at a known commit; do not remove the submodule until the caller PR has merged and reviewers accept its generated package format.
- If input/model/provider validation or Hugo build fails, do not commit generation outputs. Revert only the wrapper invocation change and retain the current publication workflow.
