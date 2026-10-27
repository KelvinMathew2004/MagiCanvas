# Source-evidenced peer-row alignment — October 27, 2026

The existing equal-gap pass could compact an uneven-width row correctly but retain the first button's earlier X position. When font fitting or padding changed individual widths, the row's established source column drifted. Moving or resizing individual labels would treat the symptom and damage their control centering.

## Bounded change

`engine/resize.py::_even_source_peer_spacing` now anchors a compact row to a corroborated source column when it has no explicit source left-edge anchor. It uses the median peer-center axis, which preserves the source's middle axis even when outer controls have unequal widths.

- Existing source left-edge alignment takes priority.
- At least two independent copy groups must be explicitly center-aligned in the source and share the peer axis within 4 source pixels.
- Those groups must still share a target axis within max(2 pixels, 0.5% of canvas width). Reflows that split their columns do not imply a common target anchor.
- The existing equal-source-gap and horizontal-row gates remain. Wrapped rows, asymmetric source arrangements and rows with insufficient evidence are left alone.
- The final widths and gaps determine each relative center before translation. Each button/backplate and label move together; width, height, Y, font, tracking and internal offsets are untouched.
- The target anchor must fit inside the canvas. Existing pre-render text-overlap, feasibility and no-new-contact checks still govern admission. No selector weights, prompt contract or adapter defaults changed.

There is no template ID, label text, target-format lookup or design-specific coordinate in this rule. This is an extension of the existing peer-row mechanism, not a new hard-coded banner layout.

## Verification

The two reported cases use the same frozen Qwen predictions as the previous official checkpoint. Square row-axis error changes from 76.69 pixels to 0; the 1600×400 banner changes from 30.2806 to 0. Both retain 21/21 pixel clear gaps and the same selected candidate family. Only the six button/label X coordinates change materially. Three other official replays differ only by floating-point roundoff below 5e-13 pixels.

The 18-case official replay and full test-suite results are recorded under `data/qwen_lora/peer_axis_alignment/`. Tests include five unequal-width peers, unchanged label interiors, an intentionally asymmetric source, left-aligned text frames, explicit left-edge precedence, split target columns and wrapped rows. The full suite passed 277 tests and 34 subtests; all 18 official replays passed their automated checks.

The 89 held-out and eight authored sources were scanned using their dataset-rendering mode: none have the existing `peer_ids` groups that can reach the changed branch. Their earlier renders are retained, not presented as a newly executed 97-case model evaluation. This reachability check limits the regression surface; it does not demonstrate peer-layout generalization across those cohorts. The synthetic negative cases and the real official replays exercise the changed behavior.

The viewer shows original/resized pairs. Qwen remains enabled, while the selected candidate can still be deterministic. Passing these checks does not establish that every unseen source alignment is unambiguous.


## Reflowed copy-edge follow-up

The center-only rule deliberately rejected the 728×90 layout because its heading and body have unequal widths and therefore different centers. They do establish a shared target left edge, with both texts explicitly left-aligned. The row retained its earlier X and sat 15.4221 pixels to the left of that column.

When no shared center survives, the same source-corroborated group can now inherit a common target left or right edge. At least two target copy groups must have the same explicit text alignment and matching edge within the existing tolerance. Incidental matching edges of centered text do not count. The center branch still has priority, so the previous square and widescreen results are preserved. There is no target aspect-ratio check or template-specific placement.

The subsequent 18-case replay changes only the leaderboard materially: six control/label X values translate by 15.4221 pixels; width, height, fonts, tracking, Y and child offsets remain unchanged. Its left-axis error falls to zero, clear gaps remain 8.1/8.1 pixels, the selected candidate is unchanged and all 18 outputs pass automated checks. Other differences are sub-pixel floating-point roundoff. Full suite: 280 tests and 34 subtests passed. Added generic tests cover reflow into left and right columns and reject incidental edges of centered text.

Current evidence is under `data/qwen_lora/peer_edge_alignment/`. The earlier 89/8 source-reachability scans were repeated with dataset rendering enabled; neither cohort has the peer groups needed to enter this new branch. Their earlier outputs are retained, not claimed as freshly replayed model evaluations.
