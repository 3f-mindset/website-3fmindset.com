# Domain Vision + Edges Snapshot Worksheet

## Goal

Create one printable worksheet snapshot, consistent with the existing season-design snapshot, that helps a participant complete the focused exercise described in website issue #93: choose a handful of life domains, form a vision across selected domains, identify the role and action that serve the vision, and use the nine Edges as a reflective lens for coherence and balance.

## User-visible behavior

The participant can use one landscape worksheet to:

- Choose up to three active-focus domains from adaptable examples, while treating other domains as maintenance or deferred rather than unimportant.
- Describe a longer-term vision for each selected domain and connect that vision to a meaningful role and visible evidence/action.
- Reflect on which Edges are needed, overused, or worth developing in each focus area, without turning Edges into additional domains or a required scorecard.
- Check whether visions reinforce one another or depend on neglecting another domain.
- Name one aligned next action and a cycle/review point.

## Non-goals

- Do not reproduce the full multi-hour workshop agenda or facilitator guide.
- Do not include the weekly effort-rating scale, relationship-specific exercise, or broader workshop activities except where needed to keep the vision/domain exercise coherent.
- Do not prescribe a fixed life-domain taxonomy, a “correct” Edge for any domain, equal effort across domains, or universal numeric goals.
- Do not modify existing worksheet files or add the new worksheet to site navigation in this task.

## Acceptance criteria

1. **Given** the participant is beginning the exercise, **when** they open the worksheet, **then** they see a clear purpose and concise instructions distinguishing domains (where they are looking) from Edges (qualities of character they bring).
2. **Given** the participant is selecting domains, **when** they use the worksheet, **then** they can record up to three active-focus domains and see flexible examples including health/capacity, work/contribution, finances, relationships/community, family, meaning/values/growth, recreation/creativity, and home/daily life, with permission to rename/combine/replace them.
3. **Given** the participant has selected domains, **when** they complete the worksheet, **then** each of three focus areas has space for a vision, role, and visible evidence or next action.
4. **Given** the participant is reflecting through the Edges, **when** they complete the Edge section, **then** all nine are named—Direction, Craft, Alignment, Stewardship, Adaptation, Service, Legacy, Integration, Expression—and the prompts invite selection, balance, and noticing overuse without implying a score.
5. **Given** the participant has drafted visions, **when** they use the cross-domain check, **then** they can note whether the visions reinforce each other or create a tension/trade-off, and identify an integrating next action.
6. **Given** the worksheet is rendered or printed, **when** viewed at its native landscape letter ratio, **then** all content is within the 1100×850 canvas, legible, accessible via SVG title/description, and styled consistently with `static/worksheets/season-design/01_snapshot.svg` (white background, restrained black/gray palette, rounded boxes, writing lines).

## Constraints and assumptions

- `ASSUMPTION:` The worksheet should be a single-page landscape SVG, 1100×850, matching the linked snapshot’s format and the season-design series.
- `ASSUMPTION:` The next sequence filename is `static/worksheets/season-design/06_domain_vision_edges.svg`.
- The nine Edge names and framing come from issue #93’s latest comment and the linked 3F Mindset book chapters.
- Preserve participant agency: domains are examples; visions/goals are self-defined; Edges are a lens, not a scorecard.

## Ordered implementation plan

1. Add this specification (`docs/specs/2026-10-25-domain-vision-edges-worksheet.md`). Verify it contains the goal, behavior, non-goals, G/W/T criteria, constraints, assumptions, and rollback.
2. Add only `static/worksheets/season-design/06_domain_vision_edges.svg`. Reuse the linked snapshot’s SVG dimensions, accessibility pattern, typography palette, rounded box treatment, and printable fill-in layout.
3. Validate XML/SVG syntax, dimensions and layout bounds; render a preview if a renderer is available; run the Hugo build to confirm static assets do not break the site.

## Rollback

Delete `static/worksheets/season-design/06_domain_vision_edges.svg`; retain the spec as the rationale/history unless the task is abandoned before implementation.
