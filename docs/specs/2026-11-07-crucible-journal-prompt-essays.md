# Crucible Journal Prompt Essays — Sections 1–3

## Goal

Turn every bullet prompt in sections 1–3 of `content/program/Crucible/06-journal-prompts-for-a-working-vision.md` into its own reader-facing blog essay and its own pull request. The essays should make a man consider the cost to his life of continuing to ignore that prompt and the possibilities that open when he gives it honest attention.

## User-visible behavior

Readers encounter one essay per source bullet: seven from section 1; six from section 2; six from longer-term vision; five from role; four from visible proof/action; nine from the individual Edge prompts; and six from “Then choose only the useful reflection” (43 total). Each essay title is the first internal heading promoted to the page title. Each essay is a distinct, reflective 5–10 minute read at about a fifth-grade reading level. Every essay stands alone and helps a reader understand why its specific question matters for the quality and direction of his life. It shows how considering the question can help him choose personal standards toward a vision and move from drift toward direction. The essays are backdated in source order, one per consecutive Sunday from 2026-01-11 through 2026-11-01.

Each essay is added as a dated Crucible blog entry under `content/letters/` and is submitted in a PR containing exactly one essay file. Every source bullet must be represented once; no essay combines or splits a bullet.

## Non-goals

- Do not create essays for prompts in sections 4–10.
- Do not add the prompt document or the Crucible worksheets to the essays or modify existing content.
- Do not answer a reader's personal question for him, impose a life domain/role/Edge, or turn reflection into a scorecard.
- Do not fabricate personal anecdotes, client stories, research, statistics, or guarantees.
- Do not generate covers, audio, worksheets, or auxiliary production artifacts.
- Do not combine multiple essays into a single PR or put multiple essay files in a PR.

## Acceptance criteria

1. **Given** the source document is on `main`, **when** the series is inventoried, **then** exactly 43 source bullets from sections 1–3 are represented, with seven, six, six, five, four, nine, and six essays in their respective groups.
2. **Given** a source bullet is represented by an essay, **when** a reader sees its title, **then** the page title is the title stored in frontmatter, specific to the source prompt, distinct from other series titles, and free from clickbait, shame, or unsupported claims. The original title heading is removed from the body; later section headings describe article sections and are not required to repeat the page title.
3. **Given** a reader has not read the worksheet or source prompt, **when** he begins an essay, **then** roughly the first two-thirds build the everyday stakes, value, and quality-of-life cost of drifting without quoting or asking the source question or assuming prior program knowledge.
4. **Given** the value and consequences have been established, **when** the essay reaches roughly its final third, **then** it directly asks the source-bullet question(s) faithfully and keeps all parts of one bullet together; punctuation may be normalized to avoid em dashes while preserving the question's wording and meaning.
5. **Given** a reader reaches the close, **when** the essay connects his answer to the earlier context, **then** it invites him to give the question space, consider his answer today, and reflect on what that answer and continued practice could create.
6. **Given** any single essay is read by itself, **when** the reader considers its subject, **then** it provides enough prompt-specific context to understand why the question helps set standards toward a vision and move from drift toward direction, without repetitive boilerplate or reliance on other materials.
7. **Given** an essay is reviewed, **when** its length and content are checked, **then** it is 800–1,300 words, coherent, specific, humane, and distinct from the other essays; it uses no fabricated first-person or empirical claims and contains no em dash glyphs (`—`) in its title or body.
8. **Given** the language is reviewed, **when** a fifth-grade reader encounters it, **then** sentences are clear, direct, and made mostly of familiar words, while the ideas remain thoughtful and adult.
5. **Given** an essay is added, **when** the PR is inspected, **then** it has standard letter frontmatter and its PR contains exactly one new essay; the first series PR may also include this related spec, and no PR combines essays.
6. **Given** all PRs are submitted, **when** their files are compared with the source inventory, **then** there is exactly one separate PR for each of the 43 bullets and none for prompts outside sections 1–3.
7. **Given** the 43 articles are dated, **when** their publication dates are reviewed in source order, **then** each date is a distinct Sunday, one week apart from 2026-01-11 through 2026-11-01, and the date matches its date-prefixed letter directory.
8. **Given** a reader encounters any one letter on its own, **when** he reads it without the worksheet or the other letters, **then** it makes clear what the specific question helps him examine, why that matters to the quality and direction of his life, and how reflection can reveal drift from or movement toward a vision; it provides sufficient context without relying on shared boilerplate or prior program knowledge.

## Constraints and assumptions

- Use the website's existing content collection under `content/letters/YYYY-MM-DD-<slug>/index.md`; use a Hugo `slug`, `title`, `date`, `draft: false`, `categories: crucible`, and relevant tags. Do not invent a cover asset.
- `ASSUMPTION:` Backdate in source-prompt order, assigning one consecutive Sunday per article from 2026-01-11 (first prompt) through 2026-11-01 (last prompt), the 43 Sundays preceding the current date 2026-11-08. Set the frontmatter date and directory date prefix to the same Sunday; these are historical publication dates, not future scheduled dates.
- The frontmatter title is the blog title; the original title heading is removed from the body. Later `##` headings may name article sections and need not match the page title. Remove any opening paragraph that merely restates the source prompt or meta-explains the exercise/prompt. Preserve independent narrative hooks; do not remove an opening paragraph solely because it is first.
- Preserve a compound source prompt as one article when it is represented by one source bullet.
- Each letter is independently understandable and earns the reader's attention by building the stakes before posing its source-bullet question. Do not assume the reader has seen the prompt or worksheet, and do not quote or directly ask the question until the value and consequences have been established.
- After building that context, directly invite the reader to consider and ask the source-bullet question(s) faithfully. Keep all subquestions together. If the source uses an em dash, normalize that punctuation (for example, with a comma, parentheses, or “such as”) while preserving the words and meaning. Close by tying his possible answer back to the earlier context and inviting him to give it time today and consider what continued practice could create.
- Explain how the specific question can help set standards toward a self-chosen vision and move from drift toward direction. Make this explanation concrete and unique to the question; do not paste the same boilerplate into every letter.
- Write at approximately a fifth-grade reading level with familiar words and clear, direct sentences; preserve thoughtful adult meaning rather than oversimplifying the subject.
- Titles and bodies contain no em dash glyphs (`—`); restructure sentences naturally instead of substituting punctuation mechanically.
- `ASSUMPTION:` If a bullet has no interrogative sentence (the vision-writing instruction in section 3), its opening instruction supplies the title concept.
- Each essay is 800–1,300 words and should be a separate, self-contained piece rather than a chapter that depends on another essay.
- Reuse the existing letter metadata and tone conventions while keeping the Crucible framework's agency-preserving, non-scoring framing.
- PRs target `main`, use a branch dedicated to one essay, and contain exactly one newly added `index.md`.

## Ordered implementation plan

1. Refresh `main`; verify the Crucible prompt page is merged and enumerate the 43 selected bullets.
2. Add this specification before drafting.
3. Draft one article per bullet using an attention-catching title grounded in the prompt, dated letter frontmatter, and the avoidance/possibility/reflection arc.
4. Verify title-to-source mapping, article arc and prompt placement, standalone context, fifth-grade readability, article count and word lengths, non-repetition, Sunday date mapping, editorial constraints, and Markdown/frontmatter.
5. Submit each article on its own branch and PR based on current `main`; include this related series spec only with the first essay. Verify each PR has one essay, its source mapping is unique, and no two essay PRs share a post.

## Rollback

Close any affected PRs and delete only the series branches and their individual essay files. Keep this spec as the rationale/history unless the series is abandoned before PR creation.
