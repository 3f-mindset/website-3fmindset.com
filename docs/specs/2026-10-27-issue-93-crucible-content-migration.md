# Issue #93 Crucible content migration

## Goal

Migrate the five comments on GitHub issue #93 into readable Markdown content under `content/program/Crucible/`, with one page per comment except that “Applying the nine Edges across the candidate life domains” becomes one page for each candidate domain.

## User-visible behavior

Readers can browse the complete substance of each issue comment as a separate Markdown page in the Crucible program content collection. Readers can also consult the nine-Edge matrix for a particular candidate domain on that domain's own page. The pages retain the source comment's headings, explanatory text, links, lists, and tables.

## Non-goals

- Do not rewrite, editorially revise, summarize, or omit any comment content.
- Do not add the pages to navigation, change site templates, or modify worksheet assets.
- Do not change issue #93 comments.
- Do not add content from issue descriptions or other comments/threads beyond the five current comments.

## Acceptance criteria

1. **Given** issue #93 has five comments, **when** the Crucible content is inspected, **then** there are Markdown pages for comments 1, 2, 3, and 5, each preserving the full comment body.
2. **Given** comment 4 contains eight candidate life domains, **when** it is migrated, **then** it has eight separate domain pages, each containing that domain's complete nine-Edge matrix and relevant introductory/usage guidance without combining matrices from other domains.
3. **Given** the pages are viewed as website content, **when** Markdown is rendered, **then** frontmatter matches existing program conventions and headings, links, lists, and tables remain valid Markdown.
4. **Given** a reader opens a migrated page, **when** they read its content, **then** the page contains no added source-attribution links or plain-text source-attribution lines; hyperlinks that were part of the original comment remain.
5. **Given** the work is added to PR #95, **when** the PR changes are inspected, **then** only the migration spec and these new content pages are added; existing unrelated untracked/working changes are preserved.

## Constraints and assumptions

- `ASSUMPTION:` Use `content/program/Crucible/` with capital C, as requested. The existing repository has no folder with this name; preserve the requested casing.
- `ASSUMPTION:` Use eight domain pages because comment 4's matrix contains eight candidate life domains.
- `ASSUMPTION:` “Each comment” means a page for each complete comment body, not merely the comment's main section or a summary.
- `ASSUMPTION:` For the matrix split, the first page includes comment-level introduction and the last page includes the final usage guidance; every domain's page includes its own full nine-row matrix and the original column explanations.
- `ASSUMPTION:` Use descriptive kebab-case page paths based on the original comment titles/domain names. Add ordinary Hugo frontmatter (`title`, `showToc`, `draft`, `slug`) modeled on the existing `content/program/curriculum/index.md` page.
- Number page filenames by source comment progression: comment 1 uses `01-`, comment 2 `02-`, comment 3 `03-`, all eight domain pages from comment 4 use `04-<domain>-`, and comment 5 uses `05-`. Keep each frontmatter slug aligned with its numbered filename.
- `ASSUMPTION:` Preserve the issue comments' practical guidance and domain/Edge examples, but omit references to the book, links to book material, and added source-attribution links or lines from the published pages.

## Ordered plan

1. Fetch the exact five comment bodies and retain their titles/order as provenance.
2. Add this spec and create the `Crucible` content directory.
3. Convert comments 1, 2, 3, and 5 into one numbered Markdown page apiece; split comment 4 into eight numbered `04-` pages aligned with its eight domain sections, preserving every matrix row and relevant shared guidance.
4. Check page count, title/frontmatter, absence of book references and added source-attribution links/lines, completeness of the practical guidance and matrices, Markdown table structure, and diff scope.
5. Run the repository's appropriate site/content validation, then commit and push the PR branch and verify PR #95 includes the changes.

## Rollback

Remove only the new `content/program/Crucible/` pages and this spec from the branch. Do not revert existing PR work or the pre-existing untracked file.
