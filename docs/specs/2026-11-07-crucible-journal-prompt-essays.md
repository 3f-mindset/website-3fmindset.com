# Crucible Journal Prompt Essays — Sections 1–3

## Goal

Turn every bullet prompt in sections 1–3 of `content/program/Crucible/06-journal-prompts-for-a-working-vision.md` into its own reader-facing blog essay and its own pull request. The essays should make a man consider the cost to his life of continuing to ignore that prompt and the possibilities that open when he gives it honest attention.

## User-visible behavior

Readers encounter one essay per source bullet: seven from section 1; six from section 2; six from longer-term vision; five from role; four from visible proof/action; nine from the individual Edge prompts; and six from “Then choose only the useful reflection” (43 total). Each essay title is an attention-catching, curiosity-generating adaptation of its source question or prompt, while honestly signaling its subject. Each essay is a distinct, reflective 5–10 minute read that helps readers see the stakes of avoidance, the possibilities of reflection, and their own answer without prescribing that answer.

Each essay is added as a dated letter/blog entry under `content/letters/` and is submitted in a PR containing exactly one essay file. Every source bullet must be represented once; no essay combines or splits a bullet.

## Non-goals

- Do not create essays for prompts in sections 4–10.
- Do not add the prompt document or the Crucible worksheets to the essays or modify existing content.
- Do not answer a reader's personal question for him, impose a life domain/role/Edge, or turn reflection into a scorecard.
- Do not fabricate personal anecdotes, client stories, research, statistics, or guarantees.
- Do not generate covers, audio, worksheets, or auxiliary production artifacts.
- Do not combine multiple essays into a single PR or put multiple essay files in a PR.

## Acceptance criteria

1. **Given** the source document is on `main`, **when** the series is inventoried, **then** exactly 43 source bullets from sections 1–3 are represented, with seven, six, six, five, four, nine, and six essays in their respective groups.
2. **Given** a source bullet is represented by an essay, **when** a reader sees its title, **then** the title is the text of the article's first internal `##` heading, moved into frontmatter and removed from the body; it is specific to the source prompt, compelling enough to invite attention, curiosity-generating, distinct from the other series titles, and free from clickbait, shame, or claims unsupported by the article.
3. **Given** a reader reads an essay, **when** he considers the source question, **then** the essay explores the quality of life that may result from continued avoidance and what attention/reflection could make possible, then leaves room for his own response.
4. **Given** an essay is reviewed, **when** its length and content are checked, **then** it is 800–1,300 words, coherent, specific, humane, and distinct from the other essays; it uses no fabricated first-person or empirical claims and contains no em dash glyphs (`—`) in its title or body.
5. **Given** an essay is added, **when** the PR is inspected, **then** it has standard letter frontmatter and its PR contains exactly one new essay; the first series PR may also include this related spec, and no PR combines essays.
6. **Given** all PRs are submitted, **when** their files are compared with the source inventory, **then** there is exactly one separate PR for each of the 43 bullets and none for prompts outside sections 1–3.

## Constraints and assumptions

- `ASSUMPTION:` Use the website's existing letter collection under `content/letters/YYYY-MM-DD-<slug>/index.md`; date the series `2026-11-07`, the current repository working date, and use a Hugo `slug`, `title`, `date`, `draft: false`, `categories: letter`, and relevant tags. Do not invent a cover asset.
- Article title is the text of its first internal `##` heading, moved to frontmatter; remove that heading from the body and remove any opening paragraph that merely restates the source prompt or meta-explains the exercise/prompt. Preserve independent narrative hooks; do not remove an opening paragraph solely because it is first.
- Preserve a compound source prompt as one article when it is represented by one source bullet.
- Titles and bodies contain no em dash glyphs (`—`); restructure sentences naturally instead of substituting punctuation mechanically.
- `ASSUMPTION:` If a bullet has no interrogative sentence (the vision-writing instruction in section 3), its opening instruction supplies the title concept.
- Each essay is 800–1,300 words and should be a separate, self-contained piece rather than a chapter that depends on another essay.
- Reuse the existing letter metadata and tone conventions while keeping the Crucible framework's agency-preserving, non-scoring framing.
- PRs target `main`, use a branch dedicated to one essay, and contain exactly one newly added `index.md`.

## Ordered implementation plan

1. Refresh `main`; verify the Crucible prompt page is merged and enumerate the 43 selected bullets.
2. Add this specification before drafting.
3. Draft one article per bullet using an attention-catching title grounded in the prompt, dated letter frontmatter, and the avoidance/possibility/reflection arc.
4. Verify title-to-source mapping, title distinction/appeal and honesty, article count and word lengths, non-repetition, editorial constraints, and Markdown/frontmatter.
5. Submit each article on its own branch and PR based on current `main`; include this related series spec only with the first essay. Verify each PR has one essay, its source mapping is unique, and no two essay PRs share a post.

## Rollback

Close any affected PRs and delete only the series branches and their individual essay files. Keep this spec as the rationale/history unless the series is abandoned before PR creation.
