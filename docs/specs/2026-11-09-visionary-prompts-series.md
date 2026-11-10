# Rename Crucible Series to Visionary Prompts

## Goal

Rename the series taxonomy for the 43 Crucible journal-prompt essays to **Visionary Prompts** and update the site's main navigation to point to the renamed series archive.

## User-visible behavior

- All 43 journal-prompt essays are listed under the Hugo series term `Visionary Prompts`.
- The main navigation displays **Visionary Prompts** and links to `/series/visionary-prompts/`.
- Individual essay URLs, cover images, page titles, dates, content, and existing `[guide]` links remain unchanged.

## Non-goals

- Do not rename any article directory or URL slug containing `crucible`.
- Do not change the text of any essay, prompt, cover, or guide link.
- Do not change the other series values such as `SteadyBurn` or `marriage`.
- Do not add a redirect or alias for `/series/crucible/` unless separately requested.

## Acceptance criteria

1. **Series metadata.** Given the 43 Visionary Prompts essays, when their frontmatter is inspected, then each has `series: Visionary Prompts` and none remains assigned to the `crucible` series.
2. **Navigation.** Given the site menu is rendered, when a visitor selects **Visionary Prompts**, then it opens `/series/visionary-prompts/`.
3. **Scope preservation.** Given the metadata and menu are updated, when the diff is reviewed, then the only content changes are the 43 `series` values and the existing series menu label/URL; titles, dates, slugs, categories, tags, article bodies, covers, and guide links are unchanged.
4. **No collateral series changes.** Given all site letters are checked, when series metadata is enumerated, then other series (including SteadyBurn and marriage) retain their original values.
5. **Build.** Given the taxonomy/menu changes are present, when the production Hugo build runs, then `/series/visionary-prompts/` is generated and the site build succeeds.

## Constraints and assumptions

- `ASSUMPTION:` Replace the existing **Reflections** navigation item, which currently points to `/series/crucible/`, with the **Visionary Prompts** label and new series URL rather than adding a duplicate menu item.
- `ASSUMPTION:` The old `/series/crucible/` archive may stop resolving; no redirect is added unless requested.
- Keep the series spelling and capitalization exactly `Visionary Prompts`; Hugo generates the lowercase, hyphenated route.
- Preserve unrelated untracked files and local changes.

## Ordered plan

1. Count and inventory exactly 43 current `series: crucible` article bundles.
2. Replace only those 43 series values with `Visionary Prompts`.
3. Update the existing main-menu item label and URL in `config.yml`.
4. Verify exact count, route, menu target, untouched non-Crucible series, and diff scope; run the available Hugo validation.
5. Submit one focused PR for review.

## Rollback

Revert the focused PR to restore the 43 series values and the previous menu label/URL. No essay or cover content needs to be restored because this change does not alter it.
