> Historical record. Use [the current documentation map](../README.md) for setup and current status. Results below belong to their recorded snapshot.

# Painted controls, foreground gutters, and reading axes

October 26, 2026. This revision addresses the eight screenshot defects from the main-engine gallery. These are inference/selection changes, using source relationships and source assets. No template IDs, human target boxes, prompt changes, or newly trained weights are used.

## Changes

| Screenshot | Root cause and repair |
| --- | --- |
| SALE TODAY title touching the pill | Infer a compact backing plus its single painted label even when the pill is an image without a group ID. Treat the panel/label as a control; leave a short-side-scaled gap from external copy, moving both control layers together. |
| Green Sale artwork through the copy lockup | For an edge image whose border is overwhelmingly plain light pixels and whose dark object is compact, reserve the object and the complete foreground lockup on opposite source-supported sides. Keep image aspect and source-relative foreground arrangement. Ordinary photos, gradients and textures do not qualify for this narrow detector. |
| BEERBRAND slightly high in its ribbon | Center actual Skia glyph ink inside the ribbon's common opaque light body. The high ribbon tails are excluded from the label rectangle. |
| Small MILANIO.IT text with no padding | Restore a source-scaled font floor, grow a compact control when room exists, and derive side padding from authored padding and font size. Preserve the image backing's aspect. |
| Kitchen card touching canvas edges | Apply safe gutters to a complete, originally inset foreground card. Scale the backing and contents uniformly. A large visual only qualifies as a backing if text is painted above it; foreground flower overlays do not become cards. Source bleed remains allowed. |
| Leaderboard product before logo | When a source logo precedes a hero vertically, preserve that reading order when both become a horizontal band. Repack the two whole units into available space before the copy. |
| Supporting copy touching the CTA | The control-clearance check handles external copy independently of its own label. An accidental source bounding-box overlap does not make external copy another CTA label. |
| Story content drifting off center | Source-centered primary units inherit one exact horizontal axis when the target retains a stacked arrangement. Keep independent units movable. Afterwards, separate source-disjoint text in source order so snapping cannot introduce a glyph collision. |

The shared implementation is `engine/layout_contracts.py`. `engine/resize.py` calls these contracts before ranking, applies them to late contrast candidates before rendering, and includes their defect count in the objective. The explicitly labeled raw pre-container ablation remains available for comparison. `scripts/layout_defects.py` exposes the same contracts as named diagnostic flags rather than hiding them behind an empty flag list.

## Important raster correction

The old text-versus-image occlusion gate compared a text envelope with an image's alpha bounding rectangle. An ornament's transparent interior could therefore disqualify a good candidate as 100% occluded. This particularly affected the large floral overlay after the SALE TODAY pill was moved to clear its title.

The image gate now intersects actual Skia glyph pixels with the later-painted image's alpha, mapping through fit, translation and rotation. It compares that coverage with coverage in the source so authored overlaps remain valid. Native shapes retain their geometric coverage path. Cached masks/coverage avoid rerasterizing identical pairs.

## Verification and review

Eleven focused regression tests cover control ownership/clearance, readable padded controls, ribbon cores, transparent-versus-opaque occlusion, shared axes, logo-before-hero order, source-ordered copy clearance, and peer-control clearance after readable growth. The existing engine tests also run.

`data/qwen_lora/visual_contracts_release/` contains fresh replays of the same saved decoded predictions: 89 held-out cases in four disjoint shards, 18 official formats, and the 12 reviewed Crello pairs under both adapters. This isolates repair and selector behavior; it is not a new model or prompt A/B experiment. All shards must have matching code hashes before aggregation. The compact Gemini packet contains the selected JSON and rendered JPEG for each of the 42 main-gallery outputs, plus paired worst-held-out renders.

Do not equate an empty flag list with aesthetic perfection. Source identity cases are preserved and may expose source-authored padding or centering issues under the stricter checks. Exact overlap, source-space relationships, raster contrast, and these visual contracts cover different failure modes. The isolated-object detector is deliberately limited to plain-background assets; it is not a general couch/person segmentation model. Review the actual output before broadening it.

## Repeatable commands

```sh
.venv/bin/python -m pytest -q
.venv/bin/python scripts/audit_inventory_fixes.py --kind official --output NEW_RUN/official
.venv/bin/python scripts/audit_inventory_fixes.py --kind crello --output NEW_RUN/crello
.venv/bin/python scripts/audit_inventory_fixes.py --kind crello --arm candidate --output NEW_RUN/crello_1p7b
# Four non-overlapping held-out shards: offsets/limits 0/23, 23/23, 46/23, 69/20.
.venv/bin/python scripts/audit_inventory_fixes.py --kind heldout --offset 0 --limit 23 --output NEW_RUN/heldout_a
.venv/bin/python scripts/finalize_visual_contract_audit.py NEW_RUN
.venv/bin/python scripts/evaluate_level3_reverse.py --output NEW_RUN/reverse_level3
.venv/bin/python scripts/package_visual_contract_comparisons.py --capture-before
.venv/bin/python scripts/package_inventory_review.py --run NEW_RUN
.venv/bin/python scripts/package_visual_contract_comparisons.py
```

## Follow-up: rotated ribbons, authored edges, and reverse resize

Rotated compact controls now center the painted glyph center in the backing's rotated coordinate system. A foreground asset whose visible source bottom meets the canvas bottom retains that attachment for aspect changes below 1.4×, unless the move newly covers readable text. Major reflows can drop that attachment. Source cards restore their internal source-relative offsets before applying gutters; a deterministic seed cannot put an icon into its own title while retaining the backing.

Background placement already ranks actual glyph-mask contrast and offers whole-lockup calm-cell moves. This revision does not add a general image-subject model or force copy into a fixed image region. Only plain-field isolated edge objects get the narrow reservation detector described above.

`tests/fixtures/level3_reverse/` adds four independently authored synthetic Level 3 sources (two 1456×180 leaderboards and two 320×1200 / 240×900 skyscrapers). All original Level 3 IDs, text, assets, groups, legal copy, rotated ribbon, and intentional missing asset are retained. They are test-only fixtures, not official samples or training data. `scripts/evaluate_level3_reverse.py` runs the installed 0.6B adapter with real inference on square and portrait targets for eight additional reviewed outputs. The viewer identifies them separately and includes source/output JSONs.

## Final audit provenance and limitations

The complete base replay lives in `data/qwen_lora/visual_contracts_verified/`. The final release repeats the dependent image/copy/control clearance chain three times so moving a ribbon away from a logo can be followed by copy clearance. `scripts/replay_visual_contract_delta.py` reruns **every** source with inferred controls. It certifies the exact previous module hash before reusing any zero-control source; its only extra operation there is the idempotent forward copy-clearance pass. Each result records whether it was re-executed. This is selective regression verification, not a claim that every final output was independently generated again.

The new reverse-source fixtures reveal further limitations: some selected layouts still have supporting-copy/control collisions or drifted axes. The main gallery and source/output comparison retain those flags. Source identity samples retain source-authored defects. The intentionally missing holiday asset is still present as a placeholder. These outputs are useful stress-test evidence and are **not** all aesthetically perfected.

## Recorded release results

- 132 existing and visual-contract tests pass (34 subtests); two additional replay-key regression tests pass.
- Held-out audit: 89 cases, no current defect flags or engine errors. The final clearance dispatch re-executes 13 controlled sources and certifies 76 unchanged sources.
- Reviewed Crello pairs: 12 under each adapter, no current defect flags or engine errors.
- Official matrix: 18 cases, four flagged outputs. Two are preserved identity sources. The resized leaderboard still has a tiny-ribbon readability flag; the dense skyscraper still has a control collision.
- New synthetic reverse suite: eight outputs from four extreme-source fixtures. Three remain flagged for collisions and/or axes. These are retained for review; the new source direction is not solved universally.

The replay harness also now queues predictions by target-pair order, rather than collapsing all pairs with equal source IDs and canvas size into one dictionary entry. Resume handling excludes completed pairs, and comparisons use the original pair ID. This fixes a misleading raw-MAE guard when distinct human target templates share a canvas size. Two tests cover equal-size pairs and checkpoint resume.
