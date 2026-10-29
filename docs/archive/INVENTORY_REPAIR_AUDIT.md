> Historical record. Use [the current documentation map](../README.md) for setup and current status. Results below belong to their recorded snapshot.

# Source-structure and extreme-resize repair audit

October 25, 2026. This audit addresses the attached inventory of 18 official outputs, 12 Crello pairs under both adapters, and the 89-case held-out prompt audit. It changes inference and verification; it does not train a new adapter or use target coordinates to place elements.

## Implemented causes and safeguards

| Reported issue | Current implementation | Verification |
| --- | --- | --- |
| Identity resize changes layout/fonts | Identity bypass preserves valid geometry exactly. Only already-invalid raw/rotated source groups are minimally bounded, as a group. No inference runs. | Identity independence test; all three official identity formats. |
| Alignment chains freeze a complete poster | Inferred alignment joins have compactness, hero-size and distinct atomic-group guards. Explicit groups and source-backed artwork remain atomic. | Separate logo/hero/copy test; container tests. |
| Translucent circles capture badges | Both container inference and source backing analysis exclude translucent elliptical decoration from parent inference. | Decoration-parent test; official core cases. |
| Source-centered brand stack drifts on a banner | A target that retains non-overlapping vertical header/copy bands inherits a common canvas axis; real inline reflows keep their own axes. This dispatch is source-structural, not a template ID. | Two brand-stack/inline regression tests; official matrix. |
| Almost-centered columns | Source-centered text and modest-aspect containers inherit the canvas axis; edge/midpoint snapping still permits genuine extreme reflow. | Existing axis tests and official renders. |
| Projected box/font scales disagree | The rigid alternative scales source text, tracking and frames together. The independently rendered raw alternative retains model typography. | Rigid font/tracking regression test. |
| Backplate labels drift | Source-centered single labels are aligned using painted ink. Horizontal native pills recover their source aspect when they collapse into square controls. | Backing/pill tests and official renders. |
| Badge/control collisions | Added source-disjoint control collision diagnostics/costs and peer control re-spacing; tall formats have one badge per row. | Disjoint-control test; tall badge-order test. |
| Quarter-turn side image leaves a gap | A side-anchored backdrop may use an unrotated uniform alternative when a quarter turn cannot touch the edge while satisfying both frame bounds. Asset pixels and aspect stay fixed. | Legal frame/aspect/edge backdrop test. |
| White rules/icons detach during relocation | Contrast groups expand over contained source attachments. Before relocation, small source-attached ornaments recover their offset and scale from nearby text. | Rigid relocation tests; current paired renders. |
| Callout markers swap / circles detach | Source marker/backplate/paragraph units remain atomic, and their actual source top-to-bottom order is preserved on each side. | Source callout inference and current Crello renders. |
| Skyscraper postage stamps | Added an independent vertical-stack candidate, vertical badge rows, target-size readability and underfill costs. | All official 160x600 cases; raw bounds/ID/peer-order test. |
| Banner islands and weak hierarchy | Banner columns fit raw frames before packing. Banner readability participates in selection even at 400px height. | Official banner/leaderboard renders and layout tests. |
| Circles become bars | Native source circles retain equal width/height before rendered selection. Quiet-region candidates require stronger background variation, excluding the supplied smooth gradient. | Circle aspect regression test; official matrix. |
| Cropped glyphs pass checks | Calibration and defect checks use the renderer's rounded element-surface dimensions. Frames can expand before type shrinks; left/top bearings inherited from the source use whole raster pixels. | Surface-overflow test and held-out raster audit. |
| Duplicate decorative elements | Decoration search now copies its occupancy list instead of mutating the caller and then returning the same elements for a second append. | Decoration mutation/duplicate-ID regression test. |
| Readable Qwen loses on excessive contrast floor | Contrast loss saturates at the readability threshold; a readable 6:1 result is not punished for failing to reproduce a 12:1 source. Severe regressions still gate/rank output. | Selector tests and controlled replays. |

## Decisions that differ from the pasted prescriptions

- The exported callout source has **02 above 01**, so the engine preserves that source order. Sorting the strings numerically would corrupt the source.
- The exported pillow circle is offset left in the source. It is not evidence for automatically centering every small circular decoration under a title. Backplate centering requires source containment/registration.
- Raster bounds are half-open: an ink right bound equal to the integer surface width is legal; the last painted pixel is width minus one. The previous extra-pixel allowance is removed without declaring this legal equality a clip.
- A source identity request can already include off-canvas decoration or a slightly off-canvas rotated ribbon. Keeping every such coordinate verbatim would violate the output contract; other identity geometry and fonts stay fixed.
- Internal image padding is distinct from painted overlap. Banner column generation now reserves raw frames; general hero occlusion still uses painted content to avoid treating empty alpha padding as a person/product.
- Rigid projection remains an alternative. Applying it indiscriminately to every deterministic reflow would recreate the full-poster collapse.

## Controlled evaluation

`scripts/audit_inventory_fixes.py` replays saved decoded coordinates through the current parser, geometry repair and normal rendered selector. This isolates engine changes from model sampling. It supports the 89 held-out cases, all 18 official format cases, and both saved 12-case adapter arms. Each run records its source case IDs and engine hashes and keeps before/after output JSON and rendered comparisons. No human target is supplied to the inference engine.

`pytest.ini` scopes normal test discovery to `tests/`; archival Gemini packet copies under `data/` are not executable test suites. This avoids duplicate-module collection errors without deleting review evidence.

Final run results are recorded below. Structural/raster passes are not a claim that every design is aesthetically perfect. The legacy source-axis metric also flags intentional reflows and compares frame axes, whereas label centering uses painted ink; those conservative flags are retained in the evidence.

## Verified replay results

The full initial engine replay covered all 89 held-out cases. A contrast-only change replaced the unit-gap gate inside `whole_lockup_contrast_variants`; reconstruction verifies the previous source hash exactly. All 31 cases that could dispatch that function were re-executed, and the other 58 reuse their unchanged dispatch results. Both reviewed adapter arms were replayed after that contrast fix. A subsequent brand-stack axis correction has a source-only dispatch that is disabled for every held-out and reviewed Crello source; the recorded predicate checks and reconstructed prior resize hash certify unchanged results. All 18 official formats were re-executed with that final correction.

| Cohort | Cases | Flagged final outputs | Before → after alignment incidents | Before → after glyph-fit incidents |
| --- | ---: | ---: | ---: | ---: |
| heldout | 89 | 0 | 158 → 138 | 7 → 0 |
| official | 18 | 0 | 34 → 41 | 0 → 0 |
| crello | 12 | 0 | 35 → 16 | 2 → 0 |
| crello_1p7b | 12 | 0 | 28 → 31 | 0 → 0 |

The frame-axis counter is intentionally not a clean-layout gate: it also counts intentional reflows and frame shifts used to center painted labels. Its remaining flags are exposed for review, not removed to manufacture a zero. See `data/qwen_lora/inventory_fixes_verified/index.html` and `summary.json`.

## Test results

Final code: `python -m pytest -q` — **121 passed**, including 34 subtests. The warnings are Pillow deprecations. `git diff --check` also passes.
