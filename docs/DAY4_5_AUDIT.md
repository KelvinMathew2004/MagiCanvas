# Candidate selector and training readiness audit — October 25, 2026

## Training assessment

The strict one-target-ID cap and family-disjoint splits are suitable A/B controls.
The installed 0.6B adapter remains the runtime baseline. Qwen3-1.7B and
Qwen2.5-1.5B-Instruct are reasonable lightweight candidates; their relative layout
quality and optimal training parameters have not been established by training.

`scripts/verify_qwen_training.py` checked both local tokenizers over all
1,122/309/89 train/validation/test rows. There are 1,122 unique training targets,
zero shared families across splits, and zero inference-prefix token mismatches.
The longest training sequences are 4,749 and 4,745 tokens. Fourteen examples
would have been silently discarded at 4,096 tokens, so the Colab recipe now uses
6,144 and asserts that all examples fit. It verifies response-only labels for
every training and validation example, fixes seeds, pins the requested model
revision, prevents repository remapping, and exports dependency/data hashes.
Three epochs, r=16, alpha=32, and 1e-4 learning rate remain initial settings,
not a claim of a measured optimum. The Colab GPU training itself is deferred.

The A/B reporter had a list/count variable collision that caused `len(int)`
after a completed engine run; this is fixed. Raw JSON completion is now a
non-regression condition so a baseline already at 100% does not make promotion
mathematically impossible. Geometry MAE and learned-family win rate must still
improve, and final hard checks must pass.

## Engine changes

- Rank by actual glyph overflow, unintended text overlap/hero occlusion,
  raw-frame bounds, image aspect integrity, source contrast loss, and broken
  source lockups. Nominal line height and background texture are not defects.
- Reserve a learned container/font-fitted candidate, a rigid whole-container
  calm-cell contrast alternative when needed and feasible, a genuine
  deterministic aspect reflow, and proportional-centered for aspect changes
  below 1.35. Rendered/eligible coverage is recorded for each applicable family.
- Use SigLIP only inside the measured defect tie band. A clean learned reflow
  cannot be vetoed by a heuristic template prior; a defective one gains no
  safety exemption from its model origin.
- Fix dense-logo gutter repair rollback on already-imperfect seeds, uniformly
  fit oversized transparent raw image frames before constraint solving, and
  avoid grouping supporting copy with translucent ellipse/line washes.
- Stack peer badges in actual narrow skyscraper slots, preserving larger rows
  when they fit. All three previously tracked mechanics failures now pass.
- Preserve the absence of optional `line_height` properties during projection.
- Remove the generic background rotation sweep that created raw boxes outside
  the canvas; keep assembled background frames within the target dimensions.
- Move text frames back into their available backing before font fitting. A
  right-aligned anchor outside its backing cannot be repaired by shrinking the
  font. Reject an infeasible binary-search lower bound instead of silently
  returning 1px. The cosmetics “Sale” fixed-geometry replay retains 125px; the
  fresh engine-selected layout uses 111.76px with no invisible/contrast flag.
- Test text collisions with intersecting rendered glyph masks. Preserve scaled
  source font bearings when checking overflow, while still checking canvas
  clipping and newly introduced overflow. This avoids treating normal ascender
  overhang or intersecting empty text envelopes as physical collisions.

These gates select a clean candidate when one is available. If every rendered
family has a defect, the engine still returns the least-defective layout and
reports the failure; it does not claim the layout passed or omit required IDs.
The adapter promotion gate requires 100% hard-check passes, so unresolved
outliers block promotion even when aggregate geometry improves.

The 89 test rows are frozen for reproducible comparisons, but they have also
been inspected for engine defect mining. Their scores are a development
benchmark, not a blind estimate of generalization to unseen judging designs.
Keep training and checkpoint decisions on the separate validation families.

Hero-occlusion checks use explicit source image elements and source overlap
relationships. They do not semantically segment furniture, faces, or other
subjects embedded in a full-canvas background image. Contrast relocation is a
measured container move, not an object-detection guarantee.

## Assessment of the supplied Gemini follow-up

Implemented the common hard gates, independent deterministic fallback, corrected
107-case coverage, and protection against catastrophic font fitting failure.
The proposed width-only font fallback was not adopted: ignoring height or
forcing a 6px minimum can create clipping or collisions in an impossible slot.
Tracing the actual Sale failure showed a misplaced right alignment anchor,
not unscaled tracking; `_set_font_size` already scales tracking proportionally.
Frame repair and a validated search interval address that cause without a
design-specific exception or a fixed font-size target.

Both A/B commands explicitly pair each adapter with its own base model. Omitting
the base paths, as in the supplied shorthand, can load incompatible weights.
No automatic Docker/default model switch or Colab training was performed.

## Validation and artifacts

The complete Python suite passes: **99 tests and 34 subtests**. This includes
new checks for true clipping vs template position, hard defects vs VLM score,
rigid calm-cell movement, duplicate IDs, and A/B summary aggregation.

The stress command is in `QWEN_LORA_COLAB.md`. There are three official hackathon
JSON files, so six requested formats produce **18 official combinations**, in
addition to **89 frozen Crello test pairs**. They are different sources of
coverage; the Crello designs are not described as official hackathon inputs.
The completed diagnostic run is `data/qwen_lora/day4_5_stress_v3/report.json`:
107/107 final JSON layouts with exact IDs, no engine errors, and 18/18 official
combinations passing. Six Crello rows were flagged, including the genuine Sale
collapse, one contrast regression, and font-envelope false positives. Those
findings led to the fitting and glyph-mask corrections above. V3's hard pass
count is 101/107 under its original metric definition.

The fresh verification run is `data/qwen_lora/day4_5_stress_v4/report.json`.
Its checkpoint manifest fingerprints engine code, input data, model settings,
and adapter files. Only flagged outliers are rendered for the viewer.

`mine_qwen_batch_defects.py` uses the same physical defect assessor, with
separate review triggers for raw completion, source-axis drift, and geometry
regression versus Qwen-off. It renders with unadjusted text widths and refuses
to resume incompatible historical metrics. For the next full test+validation
Qwen-on/off comparison, use a fresh output directory:

```bash
python3 scripts/mine_qwen_batch_defects.py --workers 2 --top 10 \
  --model-dir models/qwen3-0.6b --adapter-dir models/qwen3-layout-lora-v2 \
  --output data/qwen_lora/batch_defect_mining_v2
```

### Final v4 results

| Coverage | Cases | Hard checks passed | Flagged | Engine errors |
| --- | ---: | ---: | ---: | ---: |
| Crello test | 89 | 88 | 1 | 0 |
| Official samples × six formats | 18 | 18 | 0 | 0 |
| Total | 107 | 106 | 1 | 0 |

All 107 outputs preserve exact IDs. No image-aspect, new text-collision,
glyph-overflow, invisible-text, or structural-check failures remain in this run.
Raw JSON is complete in 107/107 cases; exact raw proposals pass in 106/107,
before runtime recovery. On the 89 Crello rows, post-constraint geometry MAE is
0.058159, relative font MAE is 0.170856, and Qwen-family win rate is 57/89
(64.04%). Official inputs have no dataset target geometry labels, so they are
excluded from those MAEs. The finished manifest matches the current runtime.

The remaining outlier is
`5e872fc04b3890eb0713a4df__to__5e202fd99fea0cc3744d32f5`: white “SALE TODAY”
crosses onto a pale region. Its lower-quintile glyph contrast drops from 3.135
to 2.687, below the source-relative floor. The current rigid-container search
does not resolve this case. It remains flagged and blocks a 100% promotion
gate; this result is not described as perfected.

The one-case review is
`data/qwen_lora/day4_5_stress_v4/outliers/review.html`. The repaired cosmetics
heading is saved separately as `data/qwen_lora/day4_5_stress_v4/sale_fix.png`.

The prior 398-case report remains an archival baseline; changed metric
definitions mean its raw defect-cost magnitudes are not directly comparable
to the new costs. The full 309-row validation split was not rerun through the
latest engine in this 107-case verification pass.

### Follow-up Gemini review: selector, contrast search and edge backdrops

The inferred `source_emblem_error` no longer contributes to physical
`layout_defect_cost`. It remains a separately reported source-lockup signal
for the selector. Previously it could make clean geometry exceed the `>1`
physical-defect threshold and block the learned-family preference. Actual
overflow, collisions, bounds, visibility and contrast gates remain active.

Calm-cell relocation now considers independent deterministic candidates as
well as the primary learned candidate. It searches a denser grid plus local
translations, rerenders each proposed sibling and checks its source-relative
contrast floor. If an inferred parent is too large, it can move a nested copy
container or a subgroup connected by source copy axes, explicit group IDs,
immediate text backings and compact paired ornaments. Explicit groups are
never split; the repair translates members rigidly without changing fonts.

Large lowest-layer images that cross a source edge and span the opposite
axis are recognized as canvas layers. They scale proportionally and retain
their authored side and rotation rather than shrinking into an icon slot.
Both raw and rotated output frames stay within canvas bounds. A rotated raw
frame may require a visual inset to meet both constraints; this is preferable
to distorting the image or introducing illegal negative coordinates.

The initial five-case replay passed its hard checks and improved the pink
sale title's lower-quintile contrast from 2.687 to 2.901. This is a structural
and legibility result, not proof of improved aesthetics or geometry on every
design: the selected repair uses a smaller existing typography candidate.
The original v4 report is retained as the baseline. Engine changes do not
modify the cleaned training data or the Colab recipe; training can proceed
independently, with adapter promotion still subject to evaluation.

## Qwen3 1.7B adapter A/B — October 25, 2026

The new `qwen3-layout-lora-v2-expanded-1.7b` adapter was evaluated with its
matching local `Qwen3-1.7B` base against the installed `qwen3-layout-lora-v2`
plus `Qwen3-0.6B` baseline. Both arms used the same 89 held-out test rows and
the full render-grounded engine. The A/B run stores a fingerprinted manifest
and per-arm checkpoints; 55 baseline results were reused only after verifying
the held-out input and core engine hashes against the prior stress run.

| Metric | 1.7B candidate | 0.6B baseline |
| --- | ---: | ---: |
| Raw JSON complete | 89/89 (100%) | 89/89 (100%) |
| Valid raw proposals | 89/89 (100%) | 88/89 (98.9%) |
| Final valid layouts | 89/89 (100%) | 89/89 (100%) |
| Hard-check pass rate | 89/89 (100%) | 89/89 (100%) |
| Post-constraint geometry MAE (lower is better) | 0.06347 | **0.06155** |
| Relative font-size MAE (lower is better) | 0.20039 | **0.17535** |
| Qwen-family win rate | 65.2% | **67.4%** |

The candidate improved raw proposal validity by one case, but did not improve
the post-constraint geometry MAE or Qwen-family win rate, and its font-size
error was higher. The promotion gate is false; keep the 0.6B adapter as the
default. No hard-check failures or engine errors occurred in either arm.

The paired top-10 review page is generated at
`data/qwen_lora/ab_qwen3_1.7b_vs_0.6b/ab_viewer/review.html`; the complete
machine-readable results are in the adjacent `report.json`. Re-run or resume
with the explicit base/adapter pairs:

```sh
.venv/bin/python scripts/evaluate_qwen_adapter.py \
  --input data/qwen_lora_expanded_v2/test.jsonl \
  --output data/qwen_lora/ab_qwen3_1.7b_vs_0.6b/report.json \
  --adapter-dir models/qwen3-layout-lora-v2-expanded-1.7b \
  --model-dir models/qwen3-1.7b \
  --compare-adapter-dir models/qwen3-layout-lora-v2 \
  --compare-model-dir models/qwen3-0.6b --engine --resume --review-top 10
```

The installed 0.6B baseline also completed the Day 5 engine stress batch:
89 held-out cases plus 18 official sample/format combinations (Story, Portrait,
Square, Widescreen, Leaderboard, and Skyscraper). All 107 outputs passed hard
checks; there were no flagged cases or engine errors. The official-format
portion had 18/18 valid layouts and 18/18 complete, valid Qwen proposals. The
report and outlier page are at
`data/qwen_lora/day4_5_stress_v6/report.json` and
`data/qwen_lora/day4_5_stress_v6/outliers/review.html`.

### Next actions

1. Keep Qwen3 0.6B v2 as the default; the tested 1.7B candidate did not pass the
   promotion gate. The held-out and official stress reports both have 100% hard
   check pass rates and no flagged layouts.
2. If comparing Qwen2.5 1.5B, train or attach its prompt-v2 adapter using the
   controlled recipe in `QWEN_LORA_COLAB.md`, then run the same 89-case A/B gate
   with each adapter paired to its own base model.
3. Use the paired A/B viewer to inspect any visual tradeoffs before considering
   a future adapter promotion; benchmark hard-check success alone does not prove
   visual preference.

Obsolete curated viewers and copied review code were archived before removal;
see `REVIEW_ARCHIVE.md` for the verified inventory and recovery location.

## Follow-up review of the 1.7B / 0.6B comparison — October 25, 2026

A second review identified two real risks in the post-processing path and made
three bounded changes. `project_container_tree()` now keeps the candidate's
predicted target font instead of replacing it with a source-scaled font. The
engine keeps clean `qwen-original`, `qwen-proposal`, and `qwen-font-fitted`
layouts as pre-container siblings so render ranking can reject a damaging
projection. Exact source same-row/same-column peers also get a dedicated center
snap when their target split is within 8% of that target axis; the general
source-axis clustering remains at 3%.

The claim that the selector bypasses Qwen is not supported as a general
diagnosis. In the pre-fix 89-case A/B report, Qwen-family layouts won 58/89
candidate cases and 60/89 baseline cases. The primary learned family was
eligible and rendered in 72/89 candidate cases and 70/89 baseline cases. The
seven byte-identical images in the old top-10 viewer are real; all seven have
the same winning candidate name in both arms, and those winners are
deterministic reflows. That shows the selector chose the same deterministic
layout for those difficult examples, not that the selector bypassed Qwen in
general.

Removing `not _feasible(...)` from `has_layout_defects` alone would not admit an
infeasible Qwen candidate: `_select_visual_candidate()` separately requires
`qwen_feasible` before a Qwen candidate can enter its safe pool. The
feasibility check includes hard bounds and unintended text overlap as well as
unit separation, so it remains a safety gate. We did not change this gate
without candidate-level evidence that its gap rule rejected otherwise clean
rendered layouts. The suggested 35% container-area cap and 25% background cap
are arbitrary cutoffs; the current tree already excludes full-canvas
backgrounds, and no evidence justified those global caps.

The rotated/bleed backdrop recommendation was also not applied. Current output
checks require both raw and rotated frames to remain inside the canvas, and the
review did not demonstrate a valid output that could safely violate that
contract. The exact source bleed case is included in the new review bundle for
independent inspection.

A post-change, targeted replay covers the old top-10 cases plus the kitchen
font-size and beer-mug peer examples (12 rows, both adapters). Both arms had
12/12 complete/valid proposals and hard-check passes, with zero flagged rows.
The 1.7B arm's mean final geometry MAE was 0.10882 versus 0.10993 for 0.6B;
font MAE was 0.23841 versus 0.21364, and Qwen-family winners were 4/12 versus
5/12. This is a deliberately difficult, curated subset and is not comparable
to the 89-row full-test averages. It does not establish a model promotion or a
general quality improvement. On the kitchen example, the baseline raw relative
font error was 0.03457 while the selected final font MAE remained 0.33815; the
projection overwrite is reduced, but the box/font fit remains an open
case-level issue.

The frozen pre-change 89-case report, post-change 12-case report, both viewers,
source snapshots, and review instructions are in
`data/gemini_review_qwen_projection_fixes_2026-10-25/`. The full 89-case A/B
was not rerun after these changes; the targeted report must not be presented as
a replacement for that baseline.

### Container and peer-alignment follow-up — October 25, 2026

The focused beer/kitchen replay exposed one structural miss: the two source
beer mugs shared a row, but a centered wheat emblem between them caused the
flat pairwise grouping to leave the mugs as separate roots. Container inference
now joins exact source-axis peers through a geometrically interposed element
when their boxes overlap the shared cross-axis span. This is content-agnostic
geometry, not an ornament-type rule. A lone vertically anchored container may
also recover its source top/bottom inset when target projection drifts by more
than 8% of target height. Horizontal edge anchoring remains model-proposed;
centered source groups still inherit the canvas center. Exact peer-axis repair
uses an 8% target-spread limit, while general axis clusters remain at 3%.

The final targeted A/B replay used two cases only: the kitchen banner font case
and the beer emblem/peer-row case. Both models produced complete, valid JSON in
2/2 rows, passed hard checks in 2/2, had zero flagged rows and zero engine
errors, and chose a Qwen-family winner in 2/2. The 1.7B candidate averaged
0.06719 geometry MAE and 0.30635 relative font MAE; the 0.6B baseline averaged
0.03495 and 0.31136. The 1.7B candidate therefore did not improve geometry on
this subset. On beer, its grouped Qwen result had 0.03427 geometry MAE and
0.24024 font MAE; the 0.6B result had 0.00715 and 0.28457. On kitchen, the
1.7B selector retained `qwen-pre-container-original` but its final geometry
and font MAE (0.10011 and 0.37247) remained worse than the 0.6B result (0.06275
and 0.33815). The font/container problem is still open there.

This two-row replay is a targeted regression check, not a representative model
comparison. The 89-case A/B remains the frozen pre-follow-up baseline; no full
89-row rerun was made after these final container and anchor changes. The
Gemini review packet now includes the final report, input, renderer outputs,
dataset previews, and exact source/target JSON for both pairs, alongside the
earlier full-set baseline.

### Expanded paired review and peer-snap crash fix — October 25, 2026

A 12-case paired replay was added for visual review, covering the earlier ten difficult pairs plus the kitchen and beer cases. The first pass exposed an `UnboundLocalError` in the rollback snapshot for a centered interposed support element: the snapshot referenced a Python comprehension variable that was not in scope. The snapshot now records the active alignment dimension explicitly. The single failing case was rerun with both adapters and passed hard checks; the other eleven successful results were retained because the corrected path was not reached on those rows. The combined report records this targeted recheck.

With the repaired row included, both arms have 12/12 complete valid proposals and 12/12 hard-check passes, with zero flags and zero engine errors. The 1.7B candidate mean geometry MAE is 0.10664 and relative font MAE is 0.21266; the 0.6B baseline is 0.10352 and 0.20456. Both have 5/12 Qwen-family wins. This curated difficult subset is for visual judgment, not a substitute for a fresh 89-case held-out evaluation or an adapter promotion gate. The self-contained paired viewer is in `data/gemini_review_qwen_projection_fixes_2026-10-25/review/expanded_12_current/review.html`.

## Selector and reflow fixes verified — October 25, 2026

Deterministic reflows now receive only glyph calibration, not container projection. A raw Qwen sibling uses decoded boxes, image aspect correction, canvas bounds and glyph fitting without constraint projection. Useful geometric repairs remain distinctly named competing siblings. It is reserved in render selection whenever its physical checks pass. Source-relative unit gaps remain a deterministic optimization aid; they no longer veto a physically valid Qwen candidate. IDs, raw/rotated bounds, image aspect, actual glyph collisions/clipping, explicit backing containment, severe source contrast loss and invisible text remain hard checks. Mild source contrast drops use a continuous cost. A microscopic clean Qwen layout cannot override a more readable clean leaderboard; a wide-banner reflow that enlarges every primary object with a substantial common gain can also win without a template-coordinate prior.

Container grouping excludes large uncontained bridges and ungrouped translucent decorations; compact interposed peer anchors remain supported. Small contained cards can grow to 76% of banner height, with proportional typography and a centered short-side position. Portrait-to-banner source edge restoration is disabled, while authored portrait-family edge anchors remain supported to retain the previously verified beer lockup. Leaderboards give the primary CTA its own column, retain a compact badge row, and compare readable type using a soft cost. The local viewer defaults Qwen on at startup when weights exist; Docker enables its baked local model. Early stopping checks actual expected IDs every 16 generated tokens.

The requested minimum-pixel contrast cutoff was not used: measured minima near 1.1 can coexist with lower-quintile glyph contrast near 2.7. Severe loss is gated using the robust lower quintile. Zetadot's 1.7B raw candidate actually measured 1.386 versus source 3.135, so its fallback is retained rather than labeling that as a 0.01 contrast miss. All raw proposals and gate diagnostics are available in the updated viewer.

The 12-case verification used fresh model generation for both adapters, then replayed the exact recorded coordinates through the final engine. The replay asserts unchanged raw geometry MAE and records code/prediction hashes; generation latency is excluded from replay runtimes. Every row is reevaluated, not patched into the old report.

| Current 12-case result | Qwen3 1.7B | Qwen3 0.6B |
| --- | ---: | ---: |
| Hard checks passed | 12/12 | 12/12 |
| Flagged outputs | 0 | 0 |
| Geometry MAE | 0.06644 | 0.07265 |
| Relative font MAE | 0.20468 | 0.19307 |
| Qwen-family winners | 9/12 | 10/12 |

The official check runs the unmodified `engine.main` entry function with all three samples across all six requested sizes, with Qwen disabled and enabled. Final current-engine replay of the recorded official proposals passes 18/18 hard checks; the deterministic comparison passes 18/18. The 728×90 advanced sample retains all of its long headline/body copy, badges and legal text, so some type remains small. The full 89-case held-out comparison has not been rerun.

Viewers: `data/gemini_review_qwen_projection_fixes_2026-10-25/review/expanded_12_current/review.html` and `review/official_18_current/review.html` within the same bundle. Complete reports and raw candidate diagnostics are in the bundle's `evidence/` directory.

The final full unit suite passes 110 tests, including raw-proposal preservation, physical backing containment, robust contrast eligibility, card sizing, grouping guards and readable alternative selection.
