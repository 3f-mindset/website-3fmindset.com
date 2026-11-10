# Steady Burn phase seeds consolidation

## Goal
Consolidate the seven open Steady Burn phase seed PRs (#32, #41, #47, #66, #67, #68, #69) into one reviewed change to `main`, so the `/program` content ships through one merge-triggered deployment rather than seven separate merges.

## User-visible behavior
The seven phase seed files remain available under `content/program/steady-burn/` at the paths and with the content currently proposed in their respective PRs. The seven source PRs are superseded by a single consolidated PR; only merging that PR to `main` triggers the intended publication.

## Non-goals
- Do not change seed wording, dates, front matter, navigation, templates, or other site content.
- Do not include unrelated open PR #95.
- Do not merge any of the seven source PRs individually.

## Acceptance criteria
- **Given** the current `main` branch and seven open phase PRs, **when** the consolidation branch is prepared, **then** it contains the exact seven phase seed files from PRs #32, #41, #47, #66, #67, #68, and #69, with no unrelated content changes.
- **Given** the consolidated branch, **when** the site's required lint/build checks run, **then** all required checks pass.
- **Given** the reviewed, passing consolidated PR, **when** merged to `main`, **then** it is one merge event covering all seven seed files and initiates one deployment workflow run.
- **Given** the consolidated PR is merged and verified on `main`, **when** source PRs are reviewed, **then** they are closed as superseded without individually merging them.

## Constraints
- Preserve `main` as the base and leave all unrelated changes untouched.
- Respect branch protection and required checks; do not bypass them.
- Record source PR references in the consolidated PR description.
- Rollback: revert the single consolidation merge commit on `main`; this removes the seven added seed files in one reversible change.

## Plan
1. Inspect repository guidance, workflows, and all seven PR diffs against current `main`.
2. Create a consolidation branch from fresh `main`, add exact seed contents, and this spec.
3. Run repository-prescribed lint/build validation, inspect the full diff, and open one PR.
4. Wait for required checks and merge that PR using the repository's allowed merge method.
5. Verify `main` contains all seven files and one merge-triggered deployment run; close the seven source PRs as superseded.

## Assumptions
- **ASSUMPTION:** All seven seeds are intended publication content under `/program`, as confirmed by the requester.
- **ASSUMPTION:** A single push/merge to `main` triggers the site's normal deployment workflow once; workflow configuration will be confirmed during recon.
- **ASSUMPTION:** The source PR seed files are authoritative; consolidation preserves their exact proposed content rather than attempting editorial changes.
