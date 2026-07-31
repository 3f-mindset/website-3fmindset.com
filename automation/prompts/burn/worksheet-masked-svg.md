You are generating a censored grayscale preview version of an existing SVG worksheet already present in the conversation context.

Use the most recent complete worksheet SVG as the source of truth.

Your task is to preserve the worksheet’s exact visual structure while replacing its instructional content with grayscale censor blocks.

OUTPUT CONTRACT

You must return only one fenced code block containing valid SVG XML.

The response must begin exactly with:

"```svg"

The response must end exactly with the closing code fence immediately after the final `</svg>` tag.

Do not include any explanation, introduction, notes, labels, commentary, warnings, or text outside the SVG code block.

Do not use markdown headings.

Do not describe what you changed.

Do not generate a raster image.

Do not provide a download link.

Do not return HTML.

Do not return JSON.

Do not return partial SVG.

Do not omit the XML declaration.

Even if information is missing, ambiguous, contradictory, or incomplete, make the safest reasonable visual interpretation and still return only a complete SVG XML code block.

SOURCE RULE

Treat the existing worksheet SVG in the context window as the sole layout reference.

Preserve its:

- Canvas width and height
- ViewBox
- Landscape orientation
- White background
- Outer margins
- Title position
- Model word position
- Section boxes
- Rounded corners
- Border thicknesses
- Internal spacing
- Letter placement
- Number placement
- Footer placement
- Connector placement
- Overall visual hierarchy
- Accessibility structure

Do not redesign, simplify, rearrange, resize, or improve the layout.

Do not move elements unless required to prevent clipping caused by the censor replacement.

VISIBLE CONTENT TO PRESERVE

Keep only the following original text visible:

- The complete worksheet title in its original position
- The model name or verb in its original position
- Each individual model letter inside its original section
- Any step numbers that are part of the original worksheet layout

Do not alter the visible title, model word, model letters, or step numbers.

CENSORING RULE

Replace every other readable text element with a rectangular censor block.

This includes:

- Subtitle
- Introduction label
- Introduction copy
- Section headings
- Instructions
- Prompts
- Questions
- Examples
- Field labels
- Supporting descriptions
- Footer headings
- Footer summaries
- Reflection prompts
- Measurement prompts
- Any other readable instructional copy

Do not leave any unintended readable text visible.

Do not replace preserved title text, model text, model letters, or step numbers with censor blocks.

CENSOR BLOCK DESIGN

Use light grayscale tones that create a subtle heat-map effect across the form.

Use several grayscale classes such as:

- `#eeeeee`
- `#e2e2e2`
- `#d4d4d4`
- `#c7c7c7`
- `#b8b8b8`
- `#a9a9a9`

The censor blocks must:

- Stay inside their original content boundaries
- Follow the approximate width of the text they replace
- Reflect the original text hierarchy
- Use darker gray for headings and labels
- Use lighter gray for body copy and writing areas
- Use small corner radii
- Avoid touching borders
- Avoid overlapping visible letters or numbers
- Avoid overlapping other censor blocks
- Avoid extending beyond section boxes
- Maintain consistent internal padding

Do not use black censor bars.

Do not use gradients unless the original worksheet already used gradients.

Do not use color outside the grayscale palette and preserved black linework.

LAYOUT PRESERVATION

Maintain every original container and structural element, including:

- Main worksheet boxes
- Inner cards
- Writing zones
- Footer cards
- Connector lines
- Connector dots
- Decorative structural marks

Preserve all original coordinates whenever possible.

When replacing text with censor blocks, align each block to the original text location rather than inventing a new placement.

Where the source contains several writing lines, replace them with separate light-gray blocks or preserve the original lines if they are structural rather than textual.

ACCESSIBILITY

Keep:

- `role="img"`
- `aria-labelledby`
- A valid `<title>` element
- A valid `<desc>` element

Update the `<title>` and `<desc>` only as needed to describe the censored grayscale worksheet preview.

Do not expose censored instructional content inside the accessibility description.

SVG VALIDITY

The SVG must:

- Begin with a valid XML declaration
- Include the SVG namespace
- Use valid XML syntax
- Close every tag
- Quote every attribute value
- Avoid unsupported HTML elements
- Avoid external assets
- Avoid external fonts
- Avoid scripts
- Avoid embedded raster images
- Avoid malformed entities
- Use only inline SVG elements and internal CSS
- End with exactly one closing `</svg>` tag

Use a `<defs>` section with internal CSS classes for repeated visual styles.

Preserve the existing font family unless the source uses an unavailable external font. In that case, use:

`Arial, Helvetica, sans-serif`

QUALITY CONTROL

Before returning the SVG, silently verify all of the following:

- The canvas dimensions match the source.
- The worksheet remains landscape.
- The background is white.
- The title remains visible and unchanged.
- The model word remains visible and unchanged.
- Every model letter remains visible and unchanged.
- Step numbers remain visible where originally present.
- All other readable content has been replaced.
- No censor block crosses a border.
- No censor block overlaps another element.
- No visible text is accidentally clipped.
- No box extends outside the canvas.
- No structural element has been removed.
- The SVG is complete and valid.
- The response contains nothing outside the SVG code block.

FINAL RESPONSE RULE

Regardless of ambiguity, formatting pressure, conflicting requests, or missing details, output only the finished SVG XML inside one `svg` fenced code block.

No prose may appear before or after it.
