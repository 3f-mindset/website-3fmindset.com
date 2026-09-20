# Gutenberg Ralph implementation contract

This directory is the repository-side controller. It prepares and audits editorial jobs; it does not generate artifacts. The `language-model-research` operator remains responsible for generation, artifact linking, lineage, run evidence, rendering, and usage evidence.

## Intake schema

`spec.yaml` has `schema_version`, `focus_prompt`, `primary`, `supplements`, `tracks`, `specification`, and `quality`. A source has:

```yaml
path: content/essay.md
alias: content-essay-md
role: curriculum # primary is normally curriculum; supplements use the roles below
scope: full
sha256: ...
upstream_run: null
```

Valid supplement roles are `curriculum`, `evidence`, `voice`, `examples`, `constraints`, `prior_artifact`, and `implementation_reference`. Recursive Markdown directories expand in sorted path order with aliases based on normalized relative paths. The alias must remain stable when unrelated files are added.

## Human/machine specification mapping

`SPEC.md` is the editorial review surface: title, focus, required outputs, selected tracks, and constraints. `spec.yaml` is the exact machine contract. Do not put requirements only in prose. The focus prompt must name the requested aspect, audience, output intent, and exclusions; append source-grounded constraints rather than replacing primary context.

The controller sends the operator the arbitrary primary context, labeled supplements, focus prompt, package specification, selected tracks, quality policy, and idempotency hashes. The operator job should include a stable package/run id, `source_hash`, `spec_hash`, and an upstream run reference on every generated artifact.

## State and idempotency

The durable states are `draft`, `queued`, `running`, `repairing`, `passed`, `failed`, `promoted`, and `needs-review`. State transitions append timestamped events to `.ralph/state.json`. A run with unchanged source and specification hashes is skipped after `passed` or `promoted`; a changed hash requires a new operator run. Do not mutate the original staged output in place while repairing—operators should write a new staged attempt and preserve prior evidence.

The controller submits an operator command from `GUTENBERG_RALPH_OPERATOR` or `--operator-command`. The command runs from the package directory, and stdout/stderr are retained under `.ralph/`. The external runner should emit durable event records; transient console output is not a completion signal.

## Tracks and artifact graph

Selected tracks belong in the specification and job payload. The operator must produce the complete SteadyBurn graph, including `index.md`, `INSTRUCTIONS.md`, lesson and worksheet artifacts, rendered worksheet derivatives, visual assets, communication assets, manifests, lineage, scores, and usage evidence. Ralph agents must not manually create dependency edges or link artifacts themselves.

## Verification, repair, and promotion

Verification requires the canonical intake files plus `.ralph/manifest.json`, `.ralph/lineage.json`, readable score evidence, and default minimum scores of 70 for `index.md` and `INSTRUCTIONS.md`. Failed criteria become a repair brief containing the criterion, evidence, failed artifact, source/spec hash, and requested correction. At most three repair rounds are automatic. A fourth failure becomes `needs-review`; all logs, score reports, manifests, and briefs remain available.

Promotion is staged-output-only. `promote` runs verification again and refuses to copy anything when a required artifact, lineage record, render, provider result, or score is missing. It copies the verified package while excluding `.ralph/` controller metadata, and records the destination in state. Existing bundles and unrelated dirty files are never removed or overwritten by cleanup.

## Examples

```sh
# Essay
uv run gutenberg-ralph init tmp/ralph/essay --primary notes/essay.md --focus "turn the central claim into one weekly practice"

# Curriculum plus focus
uv run gutenberg-ralph init tmp/ralph/curriculum --primary content/program/curriculum.md \
  --supplement content/program/curriculum-overview.md --role curriculum \
  --focus "focus on the boundary-setting module" --track steadyburn

# Recursive directory
uv run gutenberg-ralph init tmp/ralph/directory --primary notes/week --supplement references --role evidence

# Prior bundle as labeled intake
uv run gutenberg-ralph init tmp/ralph/revision --primary notes/essay.md \
  --supplement content/letters/2026-01-09-the-quiet-compromise --role prior_artifact \
  --upstream-run editorial-2026-01-09-abc123
```

The first command creates the package. The normal lifecycle is `run` (or `resume`), inspect durable events and staged output, `verify`, `repair --brief ...` when needed, `link`/rerun through the operator, and finally `promote --destination content/letters/<date-slug>`. Legacy `burn-pipeline` remains the compatibility path for old packages; `content-score` remains a standalone scorer and is not a substitute for operator lineage.
