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

### Next actions

1. Train both adapters using the same cleaned data, tokenizer audits, environment,
   and controlled settings in `QWEN_LORA_COLAB.md`; choose checkpoints on validation.
2. Run each explicit base-model/adapter A/B against the installed 0.6B baseline.
3. Continue generic contrast/container candidate work on the flagged outlier;
   do not promote a new adapter while hard-check failures remain.

Obsolete curated viewers and copied review code were archived before removal;
see `REVIEW_ARCHIVE.md` for the verified inventory and recovery location.
