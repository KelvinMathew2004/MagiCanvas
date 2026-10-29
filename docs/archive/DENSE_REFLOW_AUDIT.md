> Historical record. Use [the current documentation map](../README.md) for setup and current status. Results below belong to their recorded snapshot.

# Dense reflow audit decisions — October 27, 2026

Adapter publication is deferred at the user's request. No weights, adapter
configuration, training labels, model default or public repository were changed.
The design examples select development evidence only; production code does not
match template IDs, content strings, asset names or palette identifiers.

## What the evidence supports

- [x] Add measured container packing. Source stacks cannot always be scaled as
  rigid rectangles when the target is extremely short. `container_flow.py`
  measures wrapped text with the existing Skia/font path before allocating
  contiguous macro columns. Native text backings wrap around their measured
  children; source peer rows stay together and can wrap into a small grid.
- [x] Preserve photo aspect and source overlays. A source-adjacent photo/caption
  compound may stay stacked or become side by side, whichever safely fits and
  keeps more of the photo. The new flow does not create caption/photo overlays.
  Caption fonts use the existing source/semantic floor, not a universal 7.5px.
- [x] Add `qwen-macro-band-flow` as a separately identified candidate. Source
  mass weights the median predicted center of each functional group; tiny
  rules cannot dominate. Predicted group centers choose column/band membership
  and order, while measured packing determines child geometry. Both wide
  columns and tall/square bands are supported. No new prompt or retraining.
- [x] Keep raw Qwen and existing reflow alternatives available. The learned
  macro candidate reserves a slot when realizable, within the existing bounded
  eight-candidate macro render budget. Complete inventories and clean bounds
  alone do not establish that a proposed grouping can fit or looks good.
- [x] Permit an explicitly reported readability-only fallback to replace an
  already defective incumbent when its continuous quality/readability cost is
  lower. It cannot displace a clean layout or bypass collisions, clipping,
  inventory/bounds, padding or contrast checks. Any remaining flag is retained.
  Clean candidates rank ahead of readability-only candidates. No family-name
  veto or special proportional-layout ban was added.
- [x] Keep unresolved decorations out of the new measured family. Their lack
  of a proven flow parent is not permission to create arbitrary independent
  slots. Existing arrangement candidates retain them. Unsupported atomic
  lockups fit their font to their source line-count limit, retaining proportional
  tracking, rather than introducing awkward new wraps inside the atom.
- [x] Share the exact raster glyph-overflow assessment between the macro
  gate and batch audit. A readability warning must never mask simultaneous
  glyph clipping. Recenter control ink after final atomic font fitting.
- [x] Verify source immutability, exact inventory, photo aspect, caption contact,
  native label containment, wrapped peer integrity, model-driven macro order
  and non-bypassable safety gates with regression tests.
- [x] Preserve all voted campaign images and labels; new replays have separate
  code/input manifests and outputs. No human preference is inferred from flags.
- [ ] Final release: a full 89/309 sweep and the eventual adapter-required
  offline Docker build remain separate gates. No aesthetic perfection claim.

## Corrections and limits of the supplied audit

The diagnosed authored photo/caption sources are **900×1200**, rather than
1080×1350. The completed export has 23 dataset comparisons marked neither-good,
including 11 leaderboard comparisons. Those are **comparison pairs**, not 23
unique source/format tasks: deduplication yields 15 tasks, including five
leaderboards. Their before arm is the original campaign snapshot, so the
comparison includes earlier fixes; it is not an isolated patch ablation. These observations identify useful outliers, but do not prove
that candidate generation is the sole bottleneck or that ranking is solved.

The larger adapter's complete IDs do not prove its leaf geometry is good, nor
prove that source projection alone explains every learned-family loss. The new
macro family must earn selection under the same render/geometry gates. Model
win rate is reported honestly; it is not an objective to maximize at the expense
of design quality. The default remains the local 0.6B adapter pending stronger
end-to-end evidence. No new model inference is claimed for these replays.

The frozen SigLIP2 linear-head experiment remains 9/17 on validation, tied with
zero-shot. Its 39 test comparisons are still unscored; no head was promoted.
The small family-split validation sample and global low-resolution features
are plausible limitations, not established causal explanations. No additional
arbitrary labeling or unchanged head retraining was requested.

Equal-width columns and bounded band/grid sweeps trade generality for predictable
runtime. They do not optimize every possible composition. Very short canvases
can retain tiny supporting/legal text, and zero automated flags does not certify
balanced whitespace, legibility at normal viewing distance, or good reading order.

## Research basis and engineering interpretation

[W3C Flexbox](https://www.w3.org/TR/css-flexbox-1/) describes nested horizontal
and vertical flows, intrinsic sizing, wrapping and space distribution. Measuring
content before allocating lines/bands is applicable here. This implementation
is a bounded graphic-container realizer, not a CSS engine or proof of layout
optimality.

[LayoutNUWA](https://arxiv.org/abs/2309.09506) uses semantic layout-as-code and a
rendering stage. It supports investigating structured representations, but does
not show that our existing adapters will improve merely by changing inference
post-processing. That question is assessed by the saved-proposal replays.

[SigLIP 2](https://arxiv.org/abs/2502.14786) includes native-aspect/multiresolution
variants. That makes a native-detail ranker an option for a future controlled
experiment; it does not justify replacing the current backbone or discarding
reserved test labels after this small validation result.

## Evidence locations

- Fixed regression: `data/qwen_lora/regression_suite/` and the canonical
  Original/Resized viewer, with a code/input snapshot and prior-engine baseline.
- Four source-attachment tasks: `engine_upgrade/dense_attachments_verified/`.
- Both saved 20-case model arms: `engine_upgrade/dense_models_verified/`.
- Human-rejected tasks: `engine_upgrade/dense_reviewed_verified/`.

The last three paths are under `data/qwen_lora/`. They are development checks,
not independent new test sets. See ENGINE_UPGRADE_RESULTS.md for measured outcomes.
