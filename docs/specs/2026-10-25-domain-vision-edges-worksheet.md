# Domain Vision + Edges Snapshot Worksheet

## Goal

Create a paired planning sequence consistent with the existing season-design worksheets. The first snapshot helps you choose a handful of life domains, form a vision across selected domains, identify the role and action that serve the vision, and use the nine Edges as a reflective lens for coherence and balance. A follow-on weekly worksheet helps you turn the selected domains, roles, and Edges into a realistic week plan.

## User-visible behavior

You can use the domain-vision landscape worksheet to:

- Choose up to three active-focus domains from adaptable examples, while treating other domains as maintenance or deferred rather than unimportant.
- Describe a longer-term vision for each selected domain and connect that vision to a meaningful role and visible evidence/action.
- Reflect on which Edges are needed, overused, or worth developing in each focus area, without turning Edges into additional domains or a required scorecard.
- Check whether visions reinforce one another or depend on neglecting another domain.
- Name one aligned next action and a cycle/review point.

You can then use a companion weekly plan, modeled on `static/worksheets/season-design/03_weekly_container.svg`, to:

- Carry forward up to three active domains and choose a small, concrete weekly action (“strike”) for each, tied to the role/vision and a relevant Edge.
- Assign an effort intention that distinguishes establishing new ground from maintaining progress, without treating the rating as personal worth.
- Give actions visible time containers and protect capacity, restoration, and bounded relationship/community connection.
- Ask what each commitment is in service to, identify a boundary/refusal/defer decision, and set a review for proof and adjustment.

## Non-goals

- Do not reproduce the full multi-hour workshop agenda or facilitator guide in either worksheet.
- Do not duplicate the vision worksheet’s full nine-Edge exposition in the weekly sheet; reuse the names and have you select the relevant Edge(s) for the week.
- Do not prescribe a fixed life-domain taxonomy, a “correct” Edge for any domain, equal effort across domains, or universal numeric goals.
- Do not modify existing worksheet files or add the new worksheet to site navigation in this task.

## Acceptance criteria

1. **Given** the participant is beginning the exercise, **when** they open the worksheet, **then** they see a clear purpose and concise instructions distinguishing domains (where they are looking) from Edges (qualities of character they bring).
2. **Given** the participant is selecting domains, **when** they use the worksheet, **then** they can record up to three active-focus domains and see flexible examples including health/capacity, work/contribution, finances, relationships/community, family, meaning/values/growth, recreation/creativity, and home/daily life, with permission to rename/combine/replace them.
3. **Given** the participant has selected domains, **when** they complete the worksheet, **then** each of three focus areas has space for a vision, role, and visible evidence or next action.
4. **Given** the participant is reflecting through the Edges, **when** they complete the Edge section, **then** all nine are named—Direction, Craft, Alignment, Stewardship, Adaptation, Service, Legacy, Integration, Expression—and the prompts invite selection, balance, and noticing overuse without implying a score.
5. **Given** the participant has drafted visions, **when** they use the cross-domain check, **then** they can note whether the visions reinforce each other or create a tension/trade-off, and identify an integrating next action.
6. **Given** the worksheet is rendered or printed, **when** viewed at its native landscape letter ratio, **then** all content is within the 1100×850 canvas, legible, accessible via SVG title/description, and styled consistently with `static/worksheets/season-design/01_snapshot.svg` (white background, restrained black/gray palette, rounded boxes, writing lines).
7. **Given** you have completed the domain-vision worksheet, **when** you begin the weekly worksheet, **then** you can carry forward up to three focus domains and choose a weekly action for each that connects to its vision and role, with a place to identify a relevant Edge.
8. **Given** you have chosen weekly actions, **when** you complete the weekly plan, **then** you can assign effort intention (establish / maintain or equivalent), schedule visible time containers, and account for capacity/restoration and one bounded relationship/community action.
9. **Given** you review your week plan before accepting additional demands, **when** you use the alignment/refusal check, **then** you can state what your plan is in service to and record one boundary, refusal, or defer choice.
10. **Given** the week ends, **when** you use the review section, **then** you can record evidence, effort/capacity learning, and one adjustment for the next cycle. The weekly worksheet uses the same 1100×850 accessible, printable style as the source template `03_weekly_container.svg`.

## Constraints and assumptions

- `ASSUMPTION:` Each worksheet should be a single-page landscape SVG, 1100×850, matching the linked snapshot’s format and the season-design series.
- `ASSUMPTION:` The next sequence filename is `static/worksheets/season-design/06_domain_vision_edges.svg`.
- `ASSUMPTION:` Add the companion as `static/worksheets/season-design/07_weekly_domain_plan.svg` rather than modifying existing `03_weekly_container.svg`; the original remains unchanged and usable.
- `ASSUMPTION:` The weekly plan carries up to three chosen domains from the vision sheet; its Edge prompt is a reminder/selection, not a second nine-Edge lesson.
- The nine Edge names and framing come from issue #93’s latest comment and the linked 3F Mindset book chapters.
- Preserve participant agency: domains are examples; visions/goals are self-defined; Edges are a lens, not a scorecard.

## Ordered implementation plan

1. Update this specification (`docs/specs/2026-10-25-domain-vision-edges-worksheet.md`) to cover both the vision snapshot and its weekly follow-through. Verify goal, behavior, non-goals, G/W/T criteria, constraints, assumptions, and rollback.
2. Add `static/worksheets/season-design/06_domain_vision_edges.svg` for the domain-vision exercise and `static/worksheets/season-design/07_weekly_domain_plan.svg` for weekly follow-through. Reuse the relevant templates’ 1100×850 dimensions, accessibility pattern, typography palette, rounded boxes, and printable fill-in layout.
3. Validate both SVGs’ XML, dimensions, text/element bounds; render previews; run the Hugo build to confirm static assets do not break the site.

## Rollback

Delete the added worksheet SVG(s) from this PR; retain the spec as rationale/history unless the task is abandoned before implementation. Existing worksheets, especially `03_weekly_container.svg`, remain untouched.
