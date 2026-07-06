Write `COVER_PROMPT.md` for "{{context.title}}".

Use the appended inputs named `context`, `lesson`, and `banner`.

Return Markdown only.

This file belongs to the marketing-assets phase.

Create a downstream image-generation prompt for an editorial `4:3` article cover image that fits the lesson directly rather than reusing the worksheet as the central subject.

The prompt you write must direct the image generator to create a cover asset that:
- uses a thematic central subject or decisive action pulled from the lesson's real tension
- will ultimately be rendered as a `png` or `jpeg`
- reads immediately at thumbnail size
- keeps one dominant focal point with strong subject separation
- avoids CTA framing and download language
- never uses a worksheet, paper artifact, page, document, blueprint sheet, or floating form layout as the centerpiece

Style requirements to include:
- clean, object-centered vintage pulp illustration style
- visual reference from pulp magazines, comic books, and illustrated advertisements from the 1940s-1950s
- visible halftone dot pattern for a printed look
- bold black linework with thick, prominent outlines
- cross-hatching and ink-based shading for depth
- limited, muted, slightly desaturated palette using earthy browns, beiges, faded blues, warm grays, ochres, and restrained reds
- aged, printed, hand-illustrated, nostalgic feeling rather than modern or painterly
- dynamic composition with tension, movement, and impact
- no photography
- no modern photorealism
- no neon cyberpunk aesthetics
- no explicit solarpunk influences
- no heavy noir mood
- no intricate engraving effects
- no gloomy atmosphere
- no realistic environments
- no UI mockups

Main subject requirements:
- choose one thematic subject, tool, or decisive action that fits the lesson's conflict about testing old beliefs and replacing them with proof
- prefer a single readable central scene over symbolic clutter
- if a person is used, keep the figure simple, readable, and fully subordinate to the clarity of the central action
- do not use the worksheet, paper artifact, page, document, form layout, masked worksheet render, or any document-like object as the centerpiece
- show the exact decisive moment clearly
- emphasize invention, action, readiness, pressure, and visual clarity

Composition requirements:
- explicitly specify aspect ratio `4:3`
- use a simple triangular arrangement between the main subject, primary tool or force, and support surface
- place the brightest glow or highest-contrast area at the center of the composition
- keep the background plain, solid-color, and minimally detailed whenever possible
- only include environmental detail when absolutely required for scene comprehension
- avoid elaborate settings, scenic vistas, cluttered workshops, decorative architecture, and complex background storytelling
- avoid crowded environments, excessive props, or visual noise
- keep the background secondary and free of competing focal points
- any smoke, heat, clouds, or environmental effects must be simplified and stylized

Lighting requirements:
- prioritize bright, readable illustration lighting over mood
- keep one clear center of brightness
- keep most of the image in midtones rather than deep shadow
- ensure the subject remains fully visible

Text requirements:
- no words
- no captions
- no typography
- no labels
- no signage
- no readable marks

Source material guidance:
- use the `Title`, `Promise`, `Choice`, `Actionable VERB`, and the strongest pressure image from the lesson
- use the banner brief only for broad campaign continuity, not for repeating the banner composition
- derive the central subject from the lesson's real conflict, such as false rules, testing assumptions, breaking inherited limits, evidence replacing fear, or a man confronting the mechanism of his old story

Output shape:
- `# Asset`
- `# Image Prompt`
- `# Negative Prompt`
- `# Required On-Image Copy`
- `# Visual Checklist`

Set the dimensions line to `1440px x 1080px`.
In `# Required On-Image Copy`, state clearly that there should be no on-image copy at all.
