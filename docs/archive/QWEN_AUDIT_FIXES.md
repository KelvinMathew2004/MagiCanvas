> Historical record. Use [the current documentation map](../README.md) for setup and current status. Results below belong to their recorded snapshot.

# Qwen audit changes — October 23, 2026

> Historical audit. The October 25 selector, container, training-data counts,
> and resolved mechanics failures are documented in [DAY4_5_AUDIT.md](DAY4_5_AUDIT.md).

The user-provided Gemini audit was checked against the current implementation.
The engine still uses the installed Qwen3-0.6B v2 adapter. No model upgrade or
SigLIP training is needed to exercise these changes.

## Runtime changes

- Contrast eligibility compares a Qwen candidate with the available heuristic
  alternatives: either its weak-text count is no worse, or its raster cost is
  within 1.25 of the best alternative. Invisible text is still rejected.
- Qwen is a candidate family: decoder-checked, source-relationship repaired,
  measured-font, bounded local-search, combined font-and-position, and up to
  four contrast-repair alternatives.
  The family prior no longer forces the single `qwen-proposal` candidate to win.
  Objective bands are reported separately for the family and global shortlist.
- Font calibration searches 0.70–1.45 of the predicted size in fixed model boxes,
  checks measured wrapping/height and source line count, scales letter spacing,
  and preserves clear source headline/body order where the bounds permit it.
  It is an alternative candidate; human training font labels stay unchanged.
- Joint column constraints propagate text moves through the full source-supported
  chain, move compact lockups as units, and align source peer rows only while the
  proposed layout still uses separate columns. Arbitrary nearest-image attachment
  and forced source-relative side-column placement were removed.
- Compact lockups use robust shared scale/translation with original internal
  offsets. Only explicit groups, contained superscripts, compact centered backing
  relationships, and adjacent thin rules qualify.
- Local Qwen optimization now gets a bounded search budget in a 3.5% translation
  region. It produces an additional candidate and preserves the original.
- Contrast moves sample backing pixels under actual glyph locations, search
  bounded local offsets and whole-lockup moves toward other canvas regions,
  preserve backing containment, and must pass geometry checks. A rerender must
  lower the full visual cost without reducing the moved text's measured contrast;
  otherwise the variant is rejected.
- Generation budgets estimate tokenized answer length, allow up to 4096 output
  tokens, and log token exhaustion/omissions. Complete objects from a truncated
  JSON array may be recovered. Large omissions reject the proposal; small
  omissions use matched source parents/peers where evidence exists.
- Edge-attached images retain proportions unless their source permits `cover`
  cropping. Intentional edge placement uses in-canvas frame coordinates.
- Dataset diagnostic renders use the same unadjusted wrapping option as the
  unadjusted comparison viewer. The starter renderer sources were not edited.

## Training and measurement

The source-capped v2 export uses at most 2 outgoing directions per source
template while connected components still determine leakage-safe splits. It
contains 2,013/309/89 train/validation/test directions over 295/35/21 families;
the earlier family-capped export contained 767/70/42. This sampling change
retains more examples but does not establish that the cap improves a trained
model.

The balanced v2 and experimental v3 datasets live in separate folders. V3 uses
`sqrt(width * height)` for font normalization and requires a new, explicitly
versioned adapter. The current v2 adapter/data remain usable.

Adapter evaluation reports raw model completeness and geometry before runtime
fallbacks, plus separate runtime metrics. Recovered missing IDs no longer inflate
the reported raw model validity. Review runs save actual decoded/model/engine
geometry in `candidate_outputs/*.json` as well as images and diagnostics.

The Colab recipe derives its masking example from the filtered training data,
saves/evaluates on matching steps, reloads the best validation checkpoint, and
writes the actual prompt-version marker into the exported adapter.

## Audit claims requiring qualification

- `qwen_seed.py` already called `_fit_text()`; accepting font size without any
  fitting was not the defect. The old fitter could shrink but never grow a
  model prediction, and the ranker explored no fitted alternative.
- Short-side font normalization is reversible. One normalized unit changes the
  pixel size by `min(width, height) / 1000`; banners do not inherently amplify
  this error. Geometric-mean normalization is an optional experiment, not a
  demonstrated fix for an encoding bug.
- Existing diagnostic values cannot prove visual quality, identify all semantic
  layout failures, or establish performance on an unspecified M5 configuration.
- Two missing proposals do not prove token truncation. New token/completeness
  diagnostics allow that hypothesis to be checked.
- The October 22 Gemini audit attributed large Qwen sibling ranking costs to
  `optimization.after`. That is not what the code does: `_objective_layout_cost`
  is recomputed for each candidate using the same overlap, bounds, text-fit, and
  tiny-font defect terms, then added to that candidate's raster cost. On the
  cited lockup cases, the relationship repair made some text exceed its box;
  the resulting text-fit penalty matched the reported cost increase. The raw
  decoder candidate did not have that defect. Flattening these costs would hide
  a real renderability problem. New diagnostics expose visual and layout-defect
  components separately.

The old 90-case review is historical evidence. Fresh candidate variants need
visual review before being used as training labels or claimed as improvements.

## Validation and remaining limits

- The 16 focused Qwen/ranking/panel checks pass, including source immutability,
  contain-versus-cover behavior, compact lockup geometry, complete column chains,
  font growth, prompt version round trips, omission anchoring, and token budgets.
- The full Python suite ran 47 checks: 44 passed. Three older failures were also
  reproduced using the pre-change engine snapshot: dense-leaderboard logo gutter
  balance, peer-button constraint repair, and supporting-copy role classification.
  They remain tracked; this is not a claim that the entire suite is green.
- The updated eight-case viewer is `data/qwen_lora/engine_review_v6_audit/review.html`.
  All eight outputs retain IDs, pass the runner's bounds checks, preserve required
  image proportions, and have no unintended text overlaps. Mean normalized box
  error on these inspected cases changed from 0.09498 to 0.08269. This selected
  regression set is not an independent estimate of generalization/visual appeal.
- Three inspected photo-background cases still choose heuristic layouts because
  the bounded Qwen contrast alternatives remain substantially worse than the
  available heuristics. Font-size accuracy is mixed. No claim is made that the
  new variants reproduce the reference designs or eliminate every aesthetic flaw.
- The follow-up review after the October 22 audit is
  `data/qwen_lora/engine_review_v7_audit/review.html`. It adds fitted-font plus
  local-position candidates, avoids rerunning relationship repair after font
  fitting, rejects contrast variants that fail the full-render monotonic check,
  and records visual versus layout-defect ranking costs separately. All eight
  cases pass ID/text-overlap checks; Qwen wins five. Mean geometry/font error is
  0.08222/0.18642 on this same inspected set. Visual quality remains mixed: one
  callout still crosses the hero illustration, which the text-to-text check does
  not cover.
- The prior family-capped v2/v3 exports have 767/70/42 directions across
  295/35/21 families; the newer v2 source-capped export has 2,013/309/89. Split
  isolation was checked. Equivalent v2/v3 reference font sizes differ by at most
  0.00144 pixels after serialization rounding.
- The installed v2 prompt is identical to the prior prompt in all eight inspected
  inputs. The existing adapter has not been retrained; changes to supervised
  reference-font labels/export policies require a separate training experiment.
  Do not confuse a new runtime candidate with updated model weights.

## October 23 follow-up: source-geometry repairs

- Aspect-clamped compact image ornaments authored in canvas corners now return
  to their proportionally scaled source corner when Qwen predicts a severe
  bar-shaped outlier. Other images retain model placement and source aspect.
- A non-background image/shape panel becomes a parent only when it is behind at
  least two source children, contains them in the source, and the raw model also
  placed them inside it. Their anchors are mapped into the clamped panel. The
  rule uses geometry, paint order, and element types; it embeds no IDs.
- A narrow image spanning a source canvas edge retains that target span with a
  `cover` frame, limited to source elements occupying at most 30% of the canvas.
  Relationship repair preserves the full-span frame while moving it with the
  source-evidenced lockup.
- Lockup repair retains source-scaled text frames instead of shrinking them
  independently after group transforms. Compact superscripts and overlapping
  backing strips may join a lockup based on source geometry and z-order.
- Source-side text clusters shift only when their proposed frames collide with
  a central source hero and a clear target slot exists. Source headers can move
  upward when they originally had space above that hero.
- Contrast repair tries 12–36 px local nudges and chooses the nearest viable
  move before using contrast margin as a tie-breaker.
- The dataset-only renderer can widen a tight frame by up to 10% when canvas
  bounds and neighboring text allow. Otherwise, it scales type metrics only in
  the in-memory diagnostic render. It does not edit exported JSON or the Adobe
  starter renderer.

## October 23 validation

- The fresh review is `data/qwen_lora/engine_review_v8_fixes/review.html`.
  Five of eight cases selected a Qwen-family candidate; mean normalized box
  error was 0.07535. Qwen-off and Qwen-on final outputs preserve every ID, stay
  on-canvas, keep contain-image ratios within 5%, create no unintended text
  overlaps, and report zero local renderer warnings. This small selected set
  cannot establish general visual quality; heuristic ranking still wins some
  photo-banner cases.
- The full suite ran with the copied Gemini snapshot excluded from test
  discovery: 44 passed and the same three layout-mechanics failures remain
  (logo gutter balance, peer-button row repair, supporting-copy role). The 16
  focused Qwen/ranking/panel checks pass.

## October 23: conditional source-stack candidate

- The analyzer already identifies artwork layers whose source boxes overlap
  substantially and have an explicit z-order. Qwen may still separate those
  layers while predicting target boxes. The runtime now detects that specific
  relationship break and adds a sibling that applies one source-proportional
  transform to only the broken stack, centered within Qwen's proposed envelope.
- The raw Qwen and normal repaired proposals remain candidates. The new sibling
  uses the standard geometry, text-overlap, rendering, contrast, and candidate
  ranking checks, and its measured layout-defect cost must be no higher than
  the ordinary Qwen proposal. Intact stacks do not trigger it. This is a
  conservative safeguard inspired by preserve-first approaches, not a claim
  that locking a collage is always the best reflow.
- The review builder adds a separate “Source-stack-preserved Qwen candidate”
  panel when the rule triggers. Focused Qwen pipeline tests cover a broken
  stack, an already-intact stack, and emission of the sibling from the runtime.

## October 23: v11 promotion and final training preflight

- The v11 source-stack sibling is now part of the active resize engine. The
  shared stack transform uses one uniform scale, preserving contain-image
  proportions; scaled text tracking follows its fitted font size.
- Superscript anchoring now tightens the hero text frame after moving the
  satellite, leaving a two-pixel raw-box gap while keeping measured glyph width.
- SigLIP may treat a Qwen busy-background penalty as within tolerance only when
  the Qwen candidate has no weak/invisible text, zero layout-defect cost, and
  strong contrast on every checked text. The lower-cost heuristic must have the
  same element geometry and reduce busyness only by shrinking display text.
- Contrast repair includes padded full-lockup text spans for source-backed
  cover strips and horizontal lockup placements at 17% and 21% of canvas width.
- The review builder now records renderer mode and displays it accurately. The
  final v11 review uses the unadjusted starter-renderer path for comparisons.
- These are inference-time changes. They do not alter the v2 JSONL prompt,
  answer schema, pair filters, or supervised target boxes, so existing prepared
  v2 train/validation/test files do not need regeneration for this engine update.
  The Colab response-only mask inspection remains in `QWEN_LORA_COLAB.md`.
- All 18 focused Qwen pipeline tests pass; see the final v11 Gemini review
  bundle for the regenerated visual evidence and current code snapshot. The
  eight-case batch does not trigger the stack-preservation sibling, so its real
  visual effect is covered by targeted tests but not demonstrated by this
  review sample.

## October 23: v12 source relationships and contrast repairs

- The active local adapter remains `models/qwen3-layout-lora-v2`. These fixes
  change inference only; prepared training JSONL and adapter weights are unchanged.
- Hero clearance excludes authored text/backdrop lockups and meaningful source
  overlap. Transparent image margins are checked against alpha pixels, so a
  bounding-box intersection alone cannot establish intentional layering. Top
  headers are excluded from side-column collision tests.
- Compact centered emblems can connect separated text, rules, ribbons, and
  icons through source proximity. Eligible groups occupy at most half the source
  canvas and exclude independent large visuals. One uniform transform preserves
  offsets and image proportions; major proposed boxes provide a robust scale,
  bounded to 0.85–1.25 times the change in canvas short side. The raw Qwen
  candidate remains available for genuine reflow. When a safe cohesive emblem
  candidate exists, ranking excludes severely fractured Qwen siblings using
  source-relative member offsets and non-text scale consistency.
- A centered primary illustration that was flush with the source bottom edge
  retains that anchor and source-relative height. Surrounding copy groups use
  source-side clearance without moving an isolated label out of its group.
  A narrow side slot can uniformly reduce the full callout column; merely
  refusing a shift must not leave copy overlapping a newly enlarged hero.
- Copy/button columns can fit beside a separate foreground image. For borders
  baked into a background photo, a text-free source render establishes a clean
  source color zone; target raster pixels locate a corresponding column. The
  entire copy/button column moves and scales together, including tracking.
- Contrast moves merge overlapping source groups. Marginal threshold misses
  remain diagnostic warnings but do not justify large group relocation. Safe,
  readable base Qwen candidates suppress contrast-only siblings. Repair render
  slots are shared across parents instead of being exhausted by one parent.
- A source-color column repair must improve each affected plain text's measured
  minimum contrast by at least 0.5, meet its source-derived floor (never below
  2.5), introduce no contrast regression, and add no weak text. This accommodates
  authored white-on-pink text whose contrast stays below the general threshold
  while removing the much worse white-on-photo-border collision. Warnings are
  retained; this is not a claim of accessibility compliance.
- The eight-case before/after viewer is
  `data/qwen_lora/engine_review_v12_source_relationships/comparison.html`.
  Saved pre-change candidates are rendered with the same unadjusted starter
  renderer and dataset fonts as current candidates. `before_after.json` records
  every case's geometry/font errors, unchanged raw proposals, and hard checks.
  No official `renderer/converter/` source is modified.
- The final controlled comparison replays the same verified v2 decoder proposals
  through the completed engine. Mean geometry MAE changed from 0.05663 to 0.04359;
  relative font error changed from 0.17360 to 0.13504. Seven cases improved in
  geometry MAE. The square illustration changed from 0.09844 to 0.10210 while
  gaining a bottom anchor and clear side columns. This repeatedly reviewed sample
  is insufficient to establish generalization. All eight preserve IDs and have
  zero unintended text overlaps and zero renderer warnings.
- All 27 focused pipeline tests pass. The full suite ran 64 tests: 61 passed,
  with the same two failures and one error in existing layout-mechanics tests
  (logo gutter balance, peer-button row repair, supporting-copy classification).
  The 22 viewer theme checks also pass.
