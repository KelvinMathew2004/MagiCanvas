> Historical record. Use [the current documentation map](../README.md) for setup and current status. Results below belong to their recorded snapshot.

# Source-evidenced constraints and broader dataset review

October 26, 2026. Historical checkpoint before the shared source-attachment revision. See [SOURCE_ATTACHMENT_REVIEW.md](SOURCE_ATTACHMENT_REVIEW.md) for current results. Counts and limitations below describe this earlier run.

## Implemented general fixes

- **Painted artwork/ink safety:** new intersections with source-disjoint foreground artwork and foreground-image collisions participate in physical eligibility and diagnostics. Glyph/image coverage uses the renderer's text mask and the asset alpha, fit and rotation. A source-disjoint ungrouped cutout can be relocated outside copy; authored overlaps remain permitted.
- **Crop-aware negative space:** `engine/clean_zones.py` detects a connected, homogeneous source-color region only when source glyph contrast supports it. It projects the safe rectangle through source-frame → intrinsic-image → target-frame transforms, including actual center-cover cropping. Seam/low-p05 cases can generate a separately rendered `clean-zone-fitted` candidate. This is deliberately restricted to a single nonrotated photo; it is not general subject segmentation.
- **Contrast continuity:** a source contrast below 3.0 can vary by 0.40 without a relocation penalty if target contrast remains at least 1.75. Higher-contrast source text retains the severe-regression gate. Independent ungrouped corner marks are excluded from proximity-driven ornament attachment, and independent logo units are protected during calm-cell lockup moves.
- **Orientation-independent roles:** named/structurally supported logos, the largest substantive hero image, horizontal **or vertical** peer controls, low-opacity legal copy and small missing corner media are distinguished before reflow. Missing a file alone does not turn a genuine top logo or large hero into a footer.
- **Measured semantic fallback:** `engine/structural_reflow.py` adds an independent candidate family that packs atomic source units into measured header/hero/badge/copy/action/footer bands, then re-packs after fitting. It does not replace a clean learned candidate. Narrow last-resort grids use the semantic stack instead of placing products after CTAs.
- **Rotation-local font fitting:** same-angle backed labels are fitted in their own panel coordinates. Global axis-aligned clamping was shrinking correctly centered ribbon type. Shape-backed ribbons can reduce empty padding around measured glyphs while retaining their source angle and corner.
- **Final painted axes:** every rendered family, including pre-container Qwen variants, receives source-axis checks. A final glyph-aware pass follows font fitting, because changed glyph widths were reintroducing drift in four held-out cases. Backed copy inherits its parent-local alignment rather than being pulled onto the canvas axis. No geometry changes happen after render selection.
- **Completion diagnostics:** the output budget reserves capacity per expected element as well as measured serialized token cost, respects available context, and records expected count/stop reason. Early stopping requires all expected aliases. The eight original reverse-source predictions already had complete JSON in the retained evidence; empty fields in older snapshots alone were not proof of truncation.

## Evidence and disagreements with the pasted diagnosis

Raw target-coordinate MAE is reported but never used as an inference-time oracle. On one `RELY on EXPERTS` case, the supposedly minor failure measures approximately 1.4:1 over significant headline glyphs, versus about 3.4:1 in the source; the small word `on` has source contrast around 9:1. These are not all sub-0.4 dips in weak source text. Severe readability losses still justify repair even when a proposal is close to a human target's coordinates. Minor measured changes use the tolerance above.

Identity resizes still preserve authored geometry, except necessary frame legalization. A source-authored control defect that is unchanged internally under that rigid legalization is recorded in `source_authored_issues`, not mislabeled as a newly introduced resize defect. This is visible in the viewer and exported JSON. It is not a claim that the source was aesthetically perfect.

## Current verification

| Cohort | Outputs | New defect flags | Model provenance |
| --- | ---: | ---: | --- |
| Held-out regression | 89 | 0 | Same saved 0.6B predictions replayed through final engine |
| Official sample/format matrix | 18 | 0 | Same saved predictions; two identity sources retain disclosed authored issues |
| Reviewed Crello / 0.6B | 12 | 0 | Same saved predictions |
| Reviewed Crello / 1.7B | 12 | 0 | Same saved predictions |
| Synthetic reverse Level 3 | 8 | 0 | Same complete saved predictions; selected outputs use deterministic fallback |
| Additional validation-family sample | 12 | 3 | **Fresh local 0.6B inference**, 12/12 complete JSON |

The requested five-cohort regression covers 139 outputs with zero new flags and zero engine errors. The fresh validation sample is additional testing, not part of a claimed clean generalization benchmark. Seven of its twelve winners are Qwen-family candidates; three outputs retain flagged collage/panel/contrast failures. Those cases remain in the review. Training overlap with the installed historical adapter is not independently certified here.

The full suite passes **146 tests and 34 subtests**. Added checks cover crop mapping, clean-region seams, cutout occlusion, source-format-independent roles, foreground-image separation, contrast tolerance, authored identity issues, output-token capacity, stopping completeness, post-fit painted axes and score-independent dataset selection.

## Viewer expansion

The main gallery now adds two clearly separated sections:

1. **Twelve validation-family designs:** one source per family, outside the 89-case test families, chosen before inference using source complexity/aspect and the model's input-context contract. All received real inference with the installed 0.6B adapter. Source, human preview, raw decoded proposal and final selected output are shown; all failures remain visible.
2. **Twenty-four held-out source designs:** 13 families outside the previous main gallery's reviewed families, at most two sources per family, selected independently of defect costs, winners or target MAE. These reuse saved model predictions to isolate engine behavior. These particular novel families cover banner/landscape/square/portrait formats; tall coverage comes from the additional validation and synthetic reverse sections.

That is **36 additional dataset comparisons**. A learned-family winner is labeled separately from a deterministic fallback; an engine output is not presented as an unmodified model prediction. Local detailed viewers retain full panels. The portable Gemini review uses contact sheets to limit upload size.

The selection plans, cohort code hashes, raw proposal JSONs and selected JSONs are retained under `data/qwen_lora/source_evidence_final/`. Human target previews are used only for evaluation/review, never to place or rank inference candidates.

## Final visual review and diagnostic limits

The final contact-sheet review found an additional **unflagged** artwork/copy crowding problem in the fresh video-games design: the circular symbol intrudes into the copy column, unlike the source and human preview. The automated three-case failure count therefore understates the full visual failure mix. This is retained as a manual review note, without changing the automatic metrics or adding a design-specific placement rule. Paris retains source-like weak photo contrast; passing the source-regression gate does not establish ideal absolute readability.

A viewer-only bug was also corrected: raw proposal geometry must be merged with source text, assets and styles before rendering. Coordinate-only records produced blank raw panels. All 36 added comparisons now render the decoded predictions with their original paint fields; no fitting or coordinate repair is applied to these raw panels. The engine reports and selected outputs are unchanged. A dedicated viewer regression verifies paint-field preservation and that invalid raw coordinates remain unrepaired.

## Reproducing the review

```sh
.venv/bin/python -m pytest -q
# Existing frozen-prediction cohorts:
.venv/bin/python scripts/audit_inventory_fixes.py --kind official --output NEW_RUN/official
.venv/bin/python scripts/audit_inventory_fixes.py --kind crello --output NEW_RUN/crello
.venv/bin/python scripts/audit_inventory_fixes.py --kind crello --arm candidate --output NEW_RUN/crello_1p7b
# Held-out shards: offsets/limits 0/23, 23/23, 46/23, 69/20.
.venv/bin/python scripts/audit_inventory_fixes.py --kind heldout --offset 0 --limit 23 --output NEW_RUN/heldout_a
# Run the other three shards, then merge/render the worst-10 comparison:
.venv/bin/python scripts/finalize_visual_contract_audit.py NEW_RUN
# Fresh new-family validation sample (prepare before looking at outcomes):
.venv/bin/python scripts/evaluate_novel_validation.py --prepare --output NEW_RUN/novel_validation
.venv/bin/python scripts/evaluate_novel_validation.py --shard 0 --output NEW_RUN/novel_validation
.venv/bin/python scripts/evaluate_novel_validation.py --shard 1 --output NEW_RUN/novel_validation
.venv/bin/python scripts/evaluate_novel_validation.py --merge --output NEW_RUN/novel_validation
```

Zero current flags means the implemented raster/geometry contracts passed. It does not prove universal design quality. Further engine edits should follow recurring defects across different source families and include unaffected-source regression checks; isolated awkward compositions stay visible rather than acquiring template-specific rules.
