# Composite macro reflow and measured upgrades

October 26, 2026. Current implementation review. Final frozen run: `data/qwen_lora/macro_structure_final/`. Older source-attachment and visual-contract documents are historical checkpoints.

## What changed

`engine/macro_layout.py` adds source-derived composite blocks and a bounded new candidate family. Existing compact source lockups remain atomic, so an unbacked price emblem cannot be split into macro columns. Cards own all contained copy and nested controls, rather than only a short heading. A photo retains its overlay badges and width-matched adjoining caption bar. Thin dividers affiliate with adjacent content; full-span edge accent strips are handled separately and preserve their edge and thickness ratio.

Whitespace cuts produce rows, columns and 2×2 alternatives without cutting atomic cards. A separate media-column option handles photos that span several source text bands and prevent a legal whitespace cut. Within a band, measured rows pack locked atomic groups with explicit gaps. Fonts use renderer fitting inside their own frames. Existing content, IDs, font families, colors and asset bytes are preserved. No source template IDs, target coordinates or task-specific text dispatch drive these layouts.

The selector records content width/height utilization, content-frame area, minimum/median font size, relative font scale, attachment drift/orphans and margin imbalance. Frame subdivisions and source copy/art gaps have their own measured errors. Edge strips and backgrounds cannot inflate content occupancy. These are geometric/painted-bound metrics; the area figure is not a pixel-perfect subject segmentation or an aesthetic score.

New candidates are considered when a complex reflow has safety defects, severe aspect change with poor utilization/scale, or broken attachments. They must pass the existing rendered contrast, glyph, ID and geometry checks. A clean incumbent requires a clean sibling that improves the measured quality cost by at least 15% or two points, whichever is larger. Existing candidates are not globally forced to fill every blank region. This is a measured regression policy, not a promise of zero regressions on arbitrary designs.

Three smaller source-evidenced siblings address the screenshots: restore a source-supported gap beside independent artwork; center a nearly centered lockup whose source elements shared that axis; and recover painted text anchors inside rectangular subpanels detected from sparse divider-line assets. Subpanel fitting uses the asset's own pixel structure, not a VLM or a design-name rule. Every moved sibling is rendered and checked before selection; a new candidate without a successful diagnostic render cannot replace the incumbent. Intentional zero-gap source contact does not acquire an invented separation.

## Concrete root bugs corrected

- A short heading above long body copy was being mistaken for a one-label control because the detector discarded strings longer than 80 characters before counting children. It now counts all contained copy before classifying the panel. This fixes the Winter Edit heading being centered over its own body.
- A nested CTA label intersecting its source parent card was falsely counted as a new control collision across group IDs. An actual source-supported text/backing containment is permitted; a source-disjoint heading entering that card is still a collision.
- Quality diagnostics tolerate authored zero-size rules without dividing by zero. Existing output legalization continues to handle degenerate geometry.

## What I agree with, and what I defer

The useful immediate proposals are composite grouping, macro layouts, explicit utilization/readability metrics and before/after regression comparisons. Qwen's original v2 contract and installed adapter remain unchanged. Small missing-ID recovery already existed, so it was incorrect to say those seven raw failures automatically excluded every repaired Qwen candidate. Raw missing IDs remain reported as raw failures; recovery is never relabeled as successful model completion. Complete JSON with missing members is not, by itself, evidence that the output-token budget truncated the response.

I do not recommend adding a production VLM under the present deadline. The observed grouping, containment and scale bugs already have measurable evidence. A VLM would require another locally packaged model, runtime/quality evaluation and reliable offline behavior; it would not provide ID, clipping or overlap guarantees. An offline critic can later help rank aesthetically weak but mechanically valid layouts. No VLM, saliency model, new crop API, preference-head training or model-contract migration is introduced in this revision.

Blank space can be intentional: for example, a central emblem against a photographic scene does not need to occupy 85% of banner width. A universal utilization floor would damage valid source compositions. Focal-point cropping is deferred rather than adding unvalidated subject guesses. The current changes preserve media groups and aspect ratios; they do not claim reliable general subject detection.

## Verification and remaining limitations

The final audit uses exactly the earlier model coordinates, including incomplete raw proposals on the authored fixtures. It reruns the current decoder, recovery, repair, renderer and selector, then compares complete design hashes against the previous frozen outputs. `scripts/build_structure_comparisons.py` includes every changed result, not just favorable examples. Unchanged canonical hashes mean identical design values; that is distinct from claiming every exported JPEG byte was separately compared.

The 89 test pairs, 18 official formats, paired twelve-case adapter cohorts, twelve additional validation families and eight legacy reverse cases remain separate from the eight new authored tasks. No source or target fixtures were tuned after seeing model results. The fixed baseline was retained; the two price-emblem regressions found during visual audit were addressed by a general compact-lockup ownership rule, not a template exclusion. There are no human resized targets for the authored tasks, so target geometry/font MAEs stay null. Model completion and learned-family selection rates are reported separately from engine safety improvements.

The proposed full ratio cross-product and round-trip benchmark is a useful next stress test, not something this fixed-case audit establishes. No result here should be cited as a completed 89×ratio sweep or a round-trip consistency guarantee. Native 90px leaderboards remain a difficult legibility limit; zero current flags does not establish aesthetic perfection.

## Final results

The targeted regression suite passed **170 tests and 34 subtests**. All cohorts used the same frozen engine hashes and exact earlier predictions, with no new model inference or training.

| Cohort | Outputs | Previously flagged → now flagged | Selected geometry MAE, previous → current |
| --- | ---: | ---: | ---: |
| Held-out test | 89 | 0 → 0 | 0.04444 → 0.04470 |
| Official samples × formats | 18 | 0 → 0 | No human resized target |
| Curated Crello / 0.6B | 12 | 0 → 0 | 0.06808 → 0.06958 |
| Curated Crello / 1.7B | 12 | 0 → 0 | 0.06996 → 0.06886 |
| Additional validation | 12 | 0 → 0 | 0.04031 → 0.03899 |
| Legacy reverse fixtures | 8 | 0 → 0 | No human resized target |
| New authored compositions | 8 | 6 → 0 | No human resized target |

The eight new authored tasks also passed in the Qwen-disabled arm (eight additional outputs). All eight selected model-ON outputs came from deterministic candidates: raw JSON completion remains 8/8, but complete valid decoded IDs remain only 1/8. This revision improves engine handling; it does not show that the model learned novel layouts. Held-out learned-family selection was 52/89, separate from the 89/89 mechanical pass rate.

Across the 159 primary outputs, 128 retain identical canonical design hashes; 16 differ only within 1e-8 numeric tolerance and have identical matched review-render pixels; 15 have substantive changes. The comparison gallery retains all 31 differing outputs. Pixel hashes refer to matched review renders at a maximum dimension of 1100px, not a full native-resolution pixel audit. Substantive changes were reviewed against their sources: the eight authored tasks, three screenshot repairs, three adapter/cohort instances of Kitchen centering, and one RELY contrast-cell selection change. Price emblems remain intact after the compact-lockup guard.

Reference MAE did not improve in every cohort. Kitchen centering and the RELY cell change raise held-out MAE slightly, while additional-validation MAE improves. Those differences are disclosed rather than describing this as a universal quality gain. Source-authored defects and intentional contacts remain separate from newly introduced defects. Weak native-size supporting-copy readability, generous whitespace in the After Dark story, and subjective hierarchy/scale are still review limitations. The full log-spaced ratio sweep, 309-row validation sweep and round-trip benchmark were not run in this revision.

## Reproduction

```sh
.venv/bin/python -m pytest -q
.venv/bin/python scripts/replay_structure_audit.py --prior BASELINE/novel_designs/report.json --output NEW_RUN/novel_designs --off
.venv/bin/python scripts/replay_structure_audit.py --prior BASELINE/novel_validation/report.json --output NEW_RUN/novel_validation
# Replay the other cohorts; split held-out offsets/limits into 0/18, 18/18, 36/18, 54/18, 72/17.
.venv/bin/python scripts/finalize_visual_contract_audit.py NEW_RUN --baseline BASELINE/heldout/report.json
.venv/bin/python scripts/build_structure_comparisons.py --run NEW_RUN --prior BASELINE --output NEW_RUN/all_changes
```
