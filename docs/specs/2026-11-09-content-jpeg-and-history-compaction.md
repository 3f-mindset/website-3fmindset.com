# Content JPEG migration and Git history compaction

## Status

Content migration is complete on the isolated tree based on live `origin/main` (`23e981e1fddd822b9632f2d75fce2aaf4c0962df`). The Gutenberg JPEG generator changes are committed and pushed at `3f-gutenberg` commit `d2510a1d1111d5904d76a673158585811c2933c3` on `chore/content-jpeg-generator`; the website pins that commit. On 2026-11-09, `main` was lease-force-updated to one parentless commit whose tree contains this verified result. A fresh clone fetched both submodules, the minified Hugo build passed, and the Pages deployment completed successfully. The content-size totals and implementation gates are recorded below.

History archive preparation (2026-11-09): `/tmp/opencode/website-history-pre-compaction-20261109.bundle` was created before the cutover outside the repository, listed and verified, then mirror-cloned and `git fsck --full --connectivity-only` checked. The restored mirror's refs matched the source at archive time; the bundle lists 305 heads (including worktree refs) and is approximately 666 MiB. Before the follow-up status update, `/tmp/opencode/website-history-first-cutover-20261109.bundle` also archived the first parentless root and verified successfully. GitHub inventory at cutover: 17 remote branches, no tags, no open PRs, and `main` was not protected. Local Git pack size was 484.78 MiB; GitHub API reported `diskUsage: 680959` KiB. These are inventories, not a claim of immediate GitHub object-store reclamation.

Post-cutover branch review (2026-11-09): all 16 non-main advertised branches were retained and assessed as rebase candidates; none was rebased or deleted. Eight `agent/steady-burn-*` branches contain mostly phase-seed changes (phase 5–7 content already matches the new main tree; phase 1–4 also delete the routine-audit seed). `agent/qwen38-local-runner` is a separate, high-risk code/cleanup branch with 283 changed paths and substantial deletions; it needs focused review before any rebase. Seven `content/*` branches contain unpublished content bundles; the two `2026-08-14-systems-for-goals*` branches are alternatives, and `content/gemini-2-5-pro-master-your-tasks` is a model-specific variant of the standard master-your-tasks bundle. All branches have old merge bases rather than the new root. Keep them intact until their owners choose which work to replay; a rebase needs each branch's merge-base boundary and conflict review. No remote branch pruning was done.

## Goal

Reduce the size of published content images and the repository's Git history while preserving the current website and authoring behavior. Convert every PNG currently under `content/` to a quality-70 progressive JPEG, update all references and image-generation paths to use JPEG, then replace the current `main` history with one root commit containing the verified final tree.

## User-visible behavior

- Existing content images under `content/` are served under `.jpg` names and render in their current page locations.
- Existing image dimensions and visual composition are retained; JPEG encoding is lossy at quality 70.
- New generated content bundles use JPEG images and refer to those `.jpg` assets. PNG artifacts must not be left in generated bundles under `content/`.
- The site's publishing and preview behavior otherwise remains unchanged.
- After a separately approved cutover, `main` contains exactly one commit. Existing clones and branches based on the previous history must be replaced or deliberately migrated.

## Scope

- Convert all PNG files currently tracked under `content/` (planning estimate: 87 files, 322,855,323 bytes; verified live-main baseline: 130 files, 520,426,879 bytes), including cover images, diagrams, article illustrations, and promotional images.
- Update all references to those files, including Markdown image links, front matter, HTML, worksheets, and generator output/manifests.
- Update the canonical Gutenberg image-generation path and any website-owned scripts that persist generated image assets. Keep truly in-memory PNG intermediates only where they are not emitted as content files.
- Rewrite the current repository history so the final `main` tip is a single parentless commit; archive existing refs before cutover and review/prune obsolete website branches after archiving.

## Non-goals

- Do not convert PNGs outside `content/` (including `assets/images/burn.png`), SVGs, PDFs, or existing JPEGs.
- Do not change image dimensions, page design, image placement, editorial text, or Hugo's general image-processing policy except as required to keep the converted assets working.
- Do not remove or recompress audio or other non-image assets as part of this scope.
- Do not change deployment behavior, provider/model settings, or invoke live image-generation APIs for tests.
- Do not claim that a force-push immediately removes old objects from GitHub storage; server-side PR refs, caches, and garbage collection may retain them.

## Image encoding and migration requirements

- Encode to baseline JPEG using quality `70`, progressive scans, optimization, and 4:2:0 chroma subsampling (Pillow equivalent: `quality=70, progressive=True, optimize=True, subsampling=2`).
- Preserve source pixel dimensions. The current 87 content PNGs are opaque (85 RGB and 2 RGBA with fully opaque alpha), so conversion needs no transparency compositing policy for this baseline.
- Rename each converted asset from `.png` to `.jpg`; remove the replaced PNG from the final tree.
- Update every reference so no page, build, preview, or generated bundle relies on a removed `.png` path.
- Generated JPEG artifacts must use the same encoding settings. If the canonical generator returns a non-JPEG source image, conversion must occur before the artifact is written to its final content path.
- Add a regression guard that detects newly generated or committed `content/**/*.png` files. The guard should provide a useful file path and fail the relevant test/CI gate.
- Report actual byte totals before and after conversion. This worktree's verified 130-file PNG baseline was 520,426,879 bytes; the resulting JPEGs total 54,576,559 bytes (465,850,320 bytes / 89.51% smaller). No minimum savings threshold is imposed.

## History rewrite and cutover requirements

- The final `main` branch must contain exactly one commit with no parent and a tree matching the fully verified post-migration site.
- Before rewriting, create a Git bundle outside the working repository containing all local and fetched refs, verify that it can be listed/restored, and retain it as the rollback archive.
- Inventory remote branches, branch protections, open PRs, and tags. The current inventory includes numerous local and remote-tracking branches; do not assume rewriting `main` alone releases the old history while other refs retain it.
- Review obsolete website branches with the owner before pruning them. Archive them before removal. Do not silently overwrite branches belonging to active work or another workflow.
- Perform the history replacement only in an explicit, scheduled cutover after content/build verification and owner approval. Use a lease-protected force update for `main`; never force-push without the reviewed backup and coordination plan.
- After cutover, verify `main` is one parentless commit, the working tree and published site match the intended tree, and obsolete website refs have been pruned only as approved. Communicate clone/reset instructions.
- Keep the archive until the owner confirms the new history and deployment are healthy. Rollback is restoring the archived refs/tree and force-updating `main` back to the archived tip under the same coordination safeguards.

## Acceptance criteria

1. **Complete migration.** Given the current repository tree, when the conversion is complete, then every PNG under `content/` has a corresponding `.jpg` with the same dimensions, the old PNG is absent, and no unrelated content file has changed.
2. **Encoding.** Given any converted or newly generated persistent content image, when its JPEG metadata is inspected, then it is progressive, uses quality 70 and optimized output with 4:2:0 subsampling, and decodes successfully.
3. **Reference integrity.** Given all Markdown, front matter, HTML, worksheet, and generator references, when references are checked and Hugo builds, then there are no references to removed content PNG paths and every expected image renders.
4. **Generation behavior.** Given a mocked or fixture-based generation run with no provider calls, when a new bundle is produced, then its persistent raster images and manifest/references use `.jpg`, no PNG is emitted under the bundle, and an existing destination remains protected.
5. **Non-content PNG behavior.** Given PNGs outside `content/` and in-memory renderer intermediates, when this migration is applied, then they remain unchanged unless a documented content-generation requirement proves they are persistent content outputs.
6. **Visual and size review.** Given all converted covers and text-heavy diagrams, when representative samples are visually inspected (including small text, edges, and gradients), then they are acceptable at quality 70; the implementation reports aggregate original and JPEG byte sizes without imposing an unapproved minimum savings target.
7. **Regression protection.** Given a future content change that adds a PNG under `content/`, when the image-format guard runs, then it fails and identifies the offending path.
8. **Build and test gates.** Given the updated scripts, generator pin, references, and content, when shell syntax checks, wrapper/generator tests, Markdown lint, the PNG guard, and `hugo --minify` run, then all pass without live provider calls.
9. **Pre-cutover backup.** Given all repository refs before history replacement, when the archive is made, then the bundle is outside the repository, includes the inventoried refs, and passes a verification/listing check before any ref is rewritten or deleted.
10. **Single-commit result.** Given the reviewed final tree and approved cutover, when `main` is replaced, then it contains exactly one parentless commit whose tree equals the verified final tree; archived history remains recoverable, and only reviewed obsolete branches are pruned.
11. **Deployment and recovery.** Given the rewritten branch, when the Pages workflow and a fresh clone are checked, then deployment succeeds, the new clone contains the intended content, and documented recovery instructions identify the archived ref restoration path.

## Constraints

- Hugo content bundles and the local layouts/theme determine current image rendering; all references must be checked, not only `cover.image` front matter.
- The website delegates editorial image generation to the pinned Gutenberg project at `36babafd4efab0eb91ee1d2538ac3377a198e87e`. Its checkout was initialized in the isolated implementation worktree and the canonical image writer/tests were inspected.
- Website-local wrapper behavior and manual PDF/JPG rendering should remain safe and non-overwriting.
- Conversion must be deterministic where the chosen encoder permits it and must not require provider credentials.
- The live-main baseline has 130 content PNGs totaling 520,426,879 bytes; image inspection found 128 RGB and two fully opaque RGBA files. The repository's current Git object sizes include older revisions and refs, so before/after history measurements must use the same ref scope and clearly distinguish local object cleanup from remote hosting garbage collection.
- No force-push, branch deletion, or other history-rewriting operation occurs as part of implementing or reviewing this specification PR.

## Assumptions

- **ASSUMPTION:** “Everything inside `content/`” means every PNG in that tree, including article illustrations and promo/worksheet-related assets, rather than covers only.
- **ASSUMPTION:** The desired “70% quality progressive etc.” encoding means JPEG quality 70, progressive scans, optimization, and standard 4:2:0 chroma subsampling.
- **ASSUMPTION:** JPEG filenames use the conventional `.jpg` extension.
- **ASSUMPTION:** The requested single-commit result applies to the website repository's `main` branch, while old refs are first archived and obsolete website branches are pruned only after review; unrelated or active branches are not silently rewritten.
- **ASSUMPTION:** Image conversion and generator updates are prepared and tested before a separately coordinated history cutover; the full-history rewrite is not performed as part of the ordinary content migration commit.
- **ASSUMPTION:** Existing transparent images are not part of the present content PNG set. If new transparent content imagery becomes a requirement, its JPEG background/compositing policy must be specified before implementation.

## Open questions

- Which remote branches are active and must be preserved under the rewritten history rather than pruned? The implementation owner must review the branch inventory immediately before cutover.
- Does repository hosting retain old objects through open/closed PR refs after `main` and obsolete branches are rewritten? Confirm expected cleanup with the hosting provider; do not promise immediate server-side size reduction.
- Are there any external consumers or pinned clones that require a transition window or a scheduled announcement before `main` is force-updated?

## Ordered implementation plan

1. **Record baselines and inventory refs.** Confirm clean working tree and current `main` tip; inventory all content PNGs, reference paths, exact current/remote refs, tags, branch protection, PRs, and local/remote repository sizes. Record the pinned Gutenberg revision and inspect its canonical image writer and tests without changing behavior.
2. **Implement conversion and reference updates.** Add/use a deterministic conversion mechanism, convert only `content/**/*.png`, update all references and remove source PNGs. Capture per-file and aggregate sizes and inspect representative quality-sensitive assets.
3. **Update generator and local tooling.** Update the canonical Gutenberg artifact writer to emit `.jpg` with the specified settings and consistent references/manifests; update its tests. Bump the website's pinned generator revision. Update website wrappers/renderers only where they persist or assume PNG output, and add the content PNG regression guard.
4. **Verify the normal repository tree.** Run fixture-based generation, image integrity and reference checks, shell and generator tests, Markdown lint, the new PNG guard, visual review, and `hugo --minify`. Review all changes for scope and report actual content-size savings.
5. **Prepare the history archive.** Bundle all refs outside the repository and verify the archive. Review branches, open PRs and protections with the owner; decide the exact refs to prune and prepare clone/reset and rollback instructions. Do not rewrite refs during this step.
6. **Execute the approved history cutover.** After the verified content/generator changes are accepted and the owner explicitly authorizes the destructive operation, create a parentless commit from the final tree, lease-protect the update to `main`, and prune only approved obsolete website branches. Keep the archive intact.
7. **Post-cutover validation.** Verify the single-commit ancestry and exact tree, check hosting refs and repository size honestly, run a fresh clone/build and Pages deployment check, communicate migration instructions, and retain the archive until owner sign-off.

## Plan of touched areas

| Area | Expected files/paths (confirm generator internals during step 1) | Verification |
| --- | --- | --- |
| Existing image assets and references | `content/**/*.png`, matching `.jpg` replacements, content Markdown/HTML/front matter/worksheet references | File inventory, byte totals, reference scan, image decode/dimension check, visual review, Hugo build |
| Canonical generator | Gutenberg source image writer, manifest/reference generation, associated tests; website `tools/gutenberg` gitlink | Fixture/mock generation, unit tests, verify output encoding and paths, pin verification |
| Website scripts and regression protection | Relevant `new-*.sh`, `scripts/` generation/rendering helpers, `tests/`, CI workflow if needed, docs | Shell syntax, wrapper tests, PNG guard, no provider calls |
| Documentation | `README.md`, `docs/README.md`, this specification as needed to record implementation outcomes | Markdown lint and review against final workflow |
| Git history (separate cutover) | `main` ref and explicitly reviewed obsolete website branch refs; external Git bundle | Archive verification, ancestry/tree checks, fresh clone, deployment, rollback drill/instructions |

## Rollback

- **Before history cutover:** Revert the migration commit(s); restore PNGs and references from the reviewed source revision. Keep the prior generator pin and output behavior if the updated generator cannot pass its tests. No old history has been rewritten at this stage.
- **After history cutover:** Preserve the external Git bundle. If verification fails, restore the archived `main` and any pruned refs from the bundle and coordinate a lease-protected force update; notify clone owners. Do not delete the archive or run repository-wide garbage collection until the owner confirms deployment, clone migration, and recovery readiness.
