# Crucible Program Content: Relative Internal Links

## Status

Completed from `main` at `5b27b2dc3788ea16dd0568e81f7ffde576156d7d`. Replaced six branch-pinned GitHub SVG links and 43 root-relative article links with browser-resolvable relative URLs. The two already-relative sibling-page links remain unchanged. A Hugo build produced 552 pages, and validation confirmed all 51 links resolve to the expected local SVG/article targets with link text and surrounding article copy preserved.

## Goal

Replace repository-specific GitHub links and site-root-absolute internal links in the Crucible program content with valid relative links to the same local worksheets and articles.

## User-visible behavior

- The six SVG worksheet links use relative URLs to the published `/worksheets/crucible-design/` assets (the source files live under Hugo's `static/` directory, which is omitted from their public URLs).
- The 43 links from the journal-prompts guide to Crucible letters use relative site URLs and continue to open the same rendered article routes.
- Existing source-relative links between the Crucible program articles remain unchanged and continue resolving.
- Link labels and surrounding text remain unchanged.

## Non-goals

- Do not change article or worksheet content, destinations, labels, titles, or frontmatter.
- Do not rewrite external links, social-sharing links, theme navigation, or files outside `content/program/Crucible/`, except for this spec.
- Do not point article links at GitHub, source Markdown URLs, or dated filesystem paths in the rendered website.

## Acceptance criteria

1. **SVG links.** Given the Crucible worksheet guide links to six SVGs, when updated, then each URL is relative from its rendered page and resolves to the corresponding `/worksheets/crucible-design/*.svg` asset; no worksheet link points to `github.com`.
2. **Article links.** Given the journal-prompts guide links to 43 Crucible letters, when updated, then each URL is relative from its rendered page, resolves to the same `/letters/<slug>/` route as before, and points to a slug present in the site.
3. **Existing local links.** Given the two links between sibling Crucible program articles are already source-relative, when the link inventory is reviewed, then they remain unchanged and Hugo renders them to the same sibling page routes.
4. **Scope and text.** Given all Markdown links in `content/program/Crucible/` are audited, when the changes are reviewed, then only the six GitHub SVG URLs and 42 root-relative article URLs are changed; labels and surrounding text are preserved.
5. **Build verification.** Given a Hugo build is rendered, when the final HTML is parsed, then each changed relative href resolves against the page URL to an existing local asset or article URL, and `hugo --minify` succeeds.

## Constraints and assumptions

- `ASSUMPTION:` “Relative” means browser-resolvable relative URLs from the rendered page, not links to source `.md` files. Crucible program pages render at `/program/crucible/<slug>/`; worksheet routes are under `/static/worksheets/`, and articles are under `/letters/`.
- `ASSUMPTION:` The six GitHub URLs are accidental branch-pinned links to assets maintained in this same website repository; local static assets are the intended destinations.
- `ASSUMPTION:` Keep valid external URLs unchanged; the requested correction concerns links whose target is local to this site.

## Ordered plan

1. Inventory every link in `content/program/Crucible/` and verify worksheet files and article slugs exist.
2. Update this spec before editing links.
3. Replace the six SVG GitHub URLs with `../../../worksheets/crucible-design/<filename>.svg` (Hugo publishes source `static/` files at the site root).
4. Replace the 42 `/letters/<slug>/` links with `../../../letters/<slug>/`; leave the two sibling-page source-relative links as they are.
5. Build Hugo, parse the rendered output links, resolve them relative to the rendered page URLs, and verify all target routes/assets exist.

## Rollback

Revert only the Markdown-link URL edits in the two Crucible program files. No source files or assets are changed or removed by this task.
