Write the final public `index.md` for "{{context.title}}".

Use the appended inputs named `draft_index`, `lesson`, `context`, and `seed`.

Return the complete file, including frontmatter.

Preserve these frontmatter facts unless the draft is missing them:
- date `{{context.date}}`
- slug `{{context.slug}}`
- `series: SteadyBurn`
- `cover.image: "cover.png"`
- `draft: false`

Frontmatter content rules:
- Set the final frontmatter `title` from the `## Title` value in `CONTEXT.md`, not the scaffold placeholder title.
- Set the final frontmatter `summary` from the `## Promise` value in `CONTEXT.md`.
- If the Promise is too long for a summary line, compress it to one sharp sentence without changing the core promise.
- Do not leave the scaffold placeholder title in the final file.
- Set `tags` to at least 10 single-word tags that are directly related to the article's topic, tensions, themes, and transformation.
- Keep every tag to a single word only.
- Prefer plain lowercase tags.
- Do not use placeholder tags.
- Do not use multi-word tags, hyphenated phrases, or generic filler tags that could fit any article.

Keep `{{< audio >}}` immediately after the frontmatter.

This prompt acts like a ghostwriter for a book chapter, adapted here to write the public SteadyBurn letter.

The voice:
- Write to men who feel behind, incomplete, or as if something is missing in their lives.
- Provide steady guidance with firm accountability.
- Do not shame or attack.
- Apply pressure through clarity, responsibility, and grounded direction.
- Use a conversational but controlled voice.
- Address the reader as `you`.
- Reflect the internal tension of knowing more is possible and choosing to move toward it.

Core function:
- Treat the lesson and draft as complete in design but abbreviated in execution.
- Expand what is already present into a full public-facing editorial letter.
- Preserve the original argument and any existing substantive section headings if they belong in the public piece.
- Do not replace the central idea, reorder the core reasoning, or introduce unrelated arguments.

Before writing the letter body, silently choose one primary framework from this list. You may combine one secondary framework if it strengthens clarity or emotional impact. Do not combine more than two frameworks. The primary framework dominates the structure.

Approved framework codes:
- `[HJ]` Hero's Journey: problem -> resistance -> breakthrough -> transformation -> new identity
- `[BAB]` Before / After / Bridge: current struggle -> desired future -> path between them
- `[PCC]` Problem / Cause / Cure: define issue -> explain root cause -> provide solution
- `[PIA]` Principle / Insight / Application: core principle -> deeper understanding -> practical implementation
- `[MTM]` Myth / Truth / Method: false belief -> corrected truth -> actionable method
- `[BRR]` Breakdown / Realization / Reconstruction: collapse of ineffective thinking -> awareness -> rebuilding process
- `[QDF]` Question Driven Framework: chapter unfolds through progressive guiding questions
- `[3SS]` Three Step Shift: awareness -> reframing -> action
- `[EAF]` Emotional Arc Framework: fear -> hope -> confidence -> empowerment
- `[SBD]` Scientific Breakdown: research -> interpretation -> practical application
- `[HLS]` Habit Loop Structure: trigger -> behavior -> outcome -> redesign
- `[IBC]` Identity Based Change: identity shift produces behavioral change
- `[MCF]` Mentor Conversation Framework: direct coaching tone with guided progression
- `[CTF]` Chronological Transformation Framework: old mindset -> transition -> evolved mindset
- `[COF]` Contrarian Framework: unconventional truth -> logical defense -> implications
- `[TKF]` Toolkit Framework: practical concepts introduced progressively throughout the chapter
- `[LF]` Ladder Framework: foundational understanding -> development -> mastery
- `[MSE]` Mindset / Strategy / Execution: beliefs -> systems -> action
- `[REF]` Ripple Effect Framework: small internal change -> expanding external results
- `[OAF]` Obstacle to Advantage Framework: weakness or struggle transformed into strength

Mandatory writing rules:
1. Do not use autobiographical storytelling unless the source material already requires it.
2. Do not rely on personal anecdotes from the author.
3. Do not include journaling prompts, worksheets, or exercises outside the narrative flow.
4. Every section must advance transformation, clarity, or behavioral understanding.
5. Maintain conceptual continuity between sections.
6. Avoid repetitive motivational filler.
7. Use emotionally intelligent but intellectually disciplined language.
8. Create momentum toward internal change.
9. Use concrete observations instead of vague inspiration.
10. End with resolution, reframing, or forward movement, not summary alone.
11. Avoid building sentences around the word `because`; split or reorganize the argument into stronger declarations.

Expansion method:
- Deepen each section's opening by drawing out assumptions, implications, and context.
- Clarify and refine central claims through sustained reasoning and precise definitions.
- Extend examples into fuller illustrations, explaining what happens and why it matters.
- Layer analysis: move from example to pattern, from pattern to mechanism, from mechanism to implication.
- Strengthen transitions by making relationships between ideas clear and continuous.
- Introduce plausible opposing views when appropriate, represent them fairly, and reveal their limits through reasoning.
- Avoid redundancy. Each paragraph must add a new level of insight.

Public-letter constraints:
- Write an extra-long editorial essay length public letter, not an internal worksheet guide.
- Keep the piece public facing and readable as a standalone SteadyBurn essay.
- Avoid references to backstage production, prompts, or internal process.
- Avoid references to worksheets unless they belong naturally in the public letter.
- Write a sharp frontmatter summary line.

Strict rules:
- Do not use meta-structural language.
- Do not label parts of the argument.
- Avoid lead-in phrases such as `the point is` or `this shows`.
- Avoid `it's not X, but Y` constructions.
- Avoid unnecessary negation or forced contrast.
- Do not reference external sources or research.

Style:
- Medium to long sentences with clear rhythm.
- Conversational, steady, and firm tone.
- Frequent use of rhetorical questions for reflection.
- Concrete, visual language and grounded metaphors: fire, forge, pressure, building, weight, friction.
- Active phrasing that reinforces agency and responsibility.
- Fifth-grade reading level while maintaining depth.

Final requirement:
- Each expanded section should end with stronger clarity and authority than it began with.
- Treat `context` as the primary source of truth for final frontmatter title, summary, and transformation arc.
- Use `seed` only to sharpen specifics, scenes, and grounded details where it strengthens the public letter.
