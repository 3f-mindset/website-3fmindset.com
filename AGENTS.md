# Agent guide

This repository is a Hugo site with four distinct concerns:

- `content/`, `layouts/`, `assets/`, and `static/` are published Hugo content and presentation.
- `automation/burn_pipeline/`, `automation/prompts/`, and `scripts/burn-pipeline.py` are the legacy burn tooling. Keep them functional for historical bundles.
- `automation/gutenberg_ralph/` is the Ralph controller and CLI. Ralph owns editorial intake, provenance, durable state, quality gates, repair briefs, and promotion.
- `operator/` is local harness material. The `language-model-research` submodule/operator is the generation authority when configured; do not reproduce its artifact dependency graph in this repository.

## Ralph mental model

An editorial package has one primary context, any number of labeled supplements, a focus prompt, a human-readable specification, a machine-readable specification, and a staged generated artifact graph. The operator generates and links SteadyBurn artifacts, rendered worksheet derivatives, visual and communication assets, manifests, lineage, scores, and usage evidence. Ralph observes durable operator evidence, applies the quality gate, requests repairs, and promotes only passing staged output.

Use `CONTEXT.md` as the canonical seed. `SEED.md` is accepted as an interchangeable legacy input. Preserve the broader source even when the package focuses on one aspect.

## Preparing inputs

Use the standalone essay as primary context when it contains the central claim and desired lesson. Use a full curriculum as primary when the request is curriculum-derived; put the requested aspect in the focus prompt and retain the remaining curriculum as a `curriculum` supplement. For a directory, sort recursive Markdown paths and use deterministic aliases. A prior generated bundle is a `prior_artifact` supplement, never an unquestioned source of truth. Mixed notes should be labeled individually as `evidence`, `voice`, `examples`, `constraints`, or `implementation_reference`.

Do not silently merge conflicting claims. Preserve source meaning, distinguish requirements from examples, mark uncertainty, and do not invent unsupported facts. Every source needs a path, deterministic alias, role, scope, SHA-256 hash, and upstream run reference when one exists.

Initialize a package with:

```sh
uv run gutenberg-ralph init tmp/ralph/week-01 --primary essay.md --focus "one focused aspect" --supplement curriculum.md --role curriculum
```

This creates `README.md`, `SPEC.md`, `spec.yaml`, `CONTEXT.md`, `.ralph/sources.json`, and `.ralph/state.json`. Run, resume, link, repair, verify, and promote with `gutenberg-ralph run`, `resume`, `link`, `repair`, `verify`, and `promote`. Configure the research operator with `GUTENBERG_RALPH_OPERATOR` or `--operator-command`; never manually recreate its dependency graph or bypass its operator.

## Quality and safety

The default gate requires all required criteria to pass and scores of at least 70 for `index.md` and `INSTRUCTIONS.md`. Missing artifacts, incomplete lineage, invalid manifests, render failures, provider failures, and scorer errors are repair conditions—not promotion conditions. No more than three repair rounds are automatic; the fourth transitions to `needs-review` with evidence preserved.

Never delete or overwrite an existing bundle as cleanup. Never discard dirty worktree changes. Promotion copies only a verified staged package and excludes controller metadata. Use Ralph for new weekly packages; use `burn-pipeline` for legacy stepwise generation and `content-score` for standalone scoring or compatibility checks.

### Intake checklist

- [ ] Primary source and focus are explicit.
- [ ] Every supplement has a role, alias, scope, and hash.
- [ ] Claims and uncertainty are preserved without invention.
- [ ] `CONTEXT.md`, `SPEC.md`, `spec.yaml`, and `.ralph/` metadata exist.
- [ ] The configured operator revision supports generalized editorial intake.

### Promotion checklist

- [ ] Operator evidence and upstream run reference are present.
- [ ] Complete artifact and lineage manifests exist.
- [ ] Rendered derivatives and required communication assets exist.
- [ ] Required scores and criteria pass.
- [ ] `gutenberg-ralph verify` passes before `promote`.
- [ ] Destination is reviewed and existing content is preserved.
