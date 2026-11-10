# Website content-workflow repository boundary and SteadyBurn migration

## Goal

Keep `website-3fmindset.com` focused on published content and the tooling needed to author, preview, render, and publish that content through GitHub Pages. Move the SteadyBurn production/orchestration and model-assessment responsibilities to the 3F Gutenberg project, using Gutenberg's Ralph/operator ownership model rather than copying the legacy burn pipeline.

## User-visible behavior

- Published Hugo content, assets, layouts, and website presentation remain in this repository and continue to build/deploy on GitHub Pages.
- Authors can still create and locally preview content with the retained website helpers.
- SteadyBurn production uses the Gutenberg writer/Wiggum workflow and its self-contained SteadyBurn production contract; the website repository no longer presents itself as the generation or model-assessment platform.
- Gutenberg takes custody of the legacy assessment evidence and the adopted SteadyBurn production policy; it does not receive Hugo site files or legacy pipeline implementation.

## Non-goals

- Do not change, rewrite, or delete published pages/media under `content/`, `static/`, `assets/`, `layouts/`, or the theme.
- Do not redesign Gutenberg's general lifecycle, operator API, repair policy, or Wiggum row state machine.
- Do not change model-provider execution or initialize/update the pinned operator submodule in this task.
- Do not make site content edits unrelated to preserving/rendering the existing site.
- Do not rewrite repository Git history.

## Acceptance criteria

1. **Website scope.** Given the repository after migration, when inspecting tracked top-level material, then published Hugo content/configuration/presentation, GitHub Pages workflows, and content-authoring/preview helpers remain; the legacy Python burn-generation and scoring packages, model-comparison implementation, and their packaging/CLI entry points do not.
2. **Content preservation.** Given the pre-migration website tree, when comparing the site-owned content/assets/presentation trees, then existing authored/published files remain byte-for-byte unchanged. Existing letter folders and their authoring/source files are not removed merely because they were used by an old pipeline.
3. **Evidence migration.** Given `docs/model-comparisons/`, when the website cleanup is complete, then all historical comparison inputs, outputs, reports, usage/cost data, and scores are preserved byte-for-byte in Gutenberg as clearly labeled historical run evidence. Executable report-generation/scoring utilities are excluded and listed in the migration manifest. Do not import website Git history or legacy pipeline implementation.
4. **Production-contract migration.** Given a writer starting a SteadyBurn package in Gutenberg, when following the Gutenberg writer workflow, then the production artifact set, dependencies, editorial/format requirements, publication conventions, workbook/render checks, and series model/verb constraints are available from Gutenberg-owned documentation/specification and do not require a file in this website repository.
5. **Assessment policy.** Given a Gutenberg SteadyBurn run, when model assessment is performed, then the legacy SteadyBurn rubric and its report semantics are represented in the Gutenberg-side contract and historical comparisons are retained. The assessment must not silently substitute tracker `model` metadata for the worksheet `Actionable VERB` registry.
6. **Pages workflow.** Given a push to the Pages deployment branch that changes any Hugo input (content, config, data, i18n, theme/modules, layouts, archetypes, assets, static files, or the Pages workflow), when the workflow runs, then it performs the existing lint/build/deploy behavior. Existing manual dispatch and deployment permissions/concurrency remain unchanged.
7. **Boundary documentation.** Given either repository's current README/writer instructions, when a user looks for SteadyBurn generation or assessment, then the website documentation directs them to Gutenberg and Gutenberg documentation treats website publication as an explicit downstream promotion, not a source generation dependency.
8. **Regression verification.** Hugo production build succeeds, retained authoring/preview helpers remain usable, Gutenberg tests pass, and migration checks show no lost site files or comparison evidence.

## Constraints

- Gutenberg `AGENTS.md` forbids adding Hugo `content/`, `layouts/`, `themes/`, `assets/`, or legacy `burn-pipeline` files. Its Ralph/operator boundary remains authoritative: Ralph owns intake/orchestration/gates/promotion; the operator owns generation, artifact linking, lineage, rendering, scoring evidence, and usage evidence.
- Gutenberg must not delete existing packages, runs, or dirty working-tree changes. Website's current untracked spec `docs/specs/2026-10-26-edge-domain-examples-traffic-light.md` is pre-existing user data and must be preserved.
- The legacy model-comparison corpus is about 43 MB; preserve all run/evaluation data without loss and avoid adding any individual GitHub-blocked file size. Report utilities are excluded as implementation, with exclusion paths documented in the manifest.
- Preserve website-local preview and newsletter scaffolding tools if they serve authoring/rendering rather than model generation/assessment.
- Retained Pages behavior must remain compatible with the existing Hugo theme/submodule setup.

## Assumptions

- **ASSUMPTION:** The phrase “migrate into Gutenberg now” means two linked repository changes/PRs are appropriate: one makes Gutenberg self-contained for SteadyBurn production and receives historical assessment evidence; the other removes obsolete pipeline/model-assessment ownership from this website repository. Do not merge either side in a way that leaves the production contract unavailable.
- **ASSUMPTION:** “Assume legacy policies carry over” means preserve the legacy SteadyBurn rubric and its documented interpretation as measurement, not an independent qualification gate; preserve the existing workbook/verb registry and series index as source data, including confirmed/inferred/missing status, without silently resolving historical ambiguity. The tracker/master-sheet model remains separate authoritative metadata.
- **ASSUMPTION:** Historical model-comparison artifacts are migrated into Gutenberg's `runs/` as immutable archival evidence. Do not import legacy implementation files or turn old comparisons into new live runs.
- **ASSUMPTION:** Site assets and prose that are already part of published letter bundles stay in this repository even if the old generator created them; future production is delegated to Gutenberg and published only by explicit promotion.
- **ASSUMPTION:** A Pages push filter should include all site-rendering inputs, not only `content/**`; this enables a deployment when configuration, layout, theme, or asset rendering behavior changes.

## Plan and touched areas

1. Add a Gutenberg-owned SteadyBurn contract and machine-readable legacy rubric/series registry material; update writer/Wiggum references so the workflow is self-contained. Add migration evidence under Gutenberg `runs/` without changing existing run/package records. Verify with Gutenberg tests and a content/evidence inventory comparison.
2. In this repository, remove only legacy generator/scorer/model-comparison implementation and packaging; keep site content, website rendering/publishing config, and relevant authoring/preview scripts. Replace old production docs with a short Gutenberg handoff. Preserve the existing untracked spec. Verify source-tree preservation and the Hugo build.
3. Expand the Pages workflow's push path filters to all Hugo rendering inputs while retaining job logic, permissions, concurrency, and manual dispatch. Validate the YAML and workflow diff.

## Rollback

- Each repository change is isolated on its own branch/PR and can be reverted independently.
- Before removing website-side files, confirm their policy/data/evidence counterparts are committed in the Gutenberg migration PR. Restore the website removal commit if Gutenberg migration verification or review fails.
- Do not delete or rewrite existing Gutenberg packages/runs. If evidence transfer validation fails, leave the website source evidence intact and do not merge the cleanup PR.
- For Pages workflow regressions, revert only the path-filter expansion; the prior workflow remains available.

## Open questions

- The Gutenberg operator submodule is uninitialized at its pinned revision in this checkout. This migration can specify the operator contract and test Ralph payloads with fakes, but end-to-end provider/render verification remains blocked until the operator is available.
- Registry conflict resolution for duplicate/variant actionable verbs remains unresolved; preserve the historical entries/status and require an explicit owner decision before new entries collide.
