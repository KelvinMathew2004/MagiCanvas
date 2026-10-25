# Engine deletion audit

Audited commit `af6bc5b` against its parent `d646694`, then exercised the replacement passes and restored two omitted safeguards.

## Size and scope

The two engine files had **1,181 deleted lines and 544 added lines: a net reduction of 637 lines**. This was a replacement of the alignment/refinement path, not deletion of thousands of lines from unrelated engine subsystems. The full commit's deletion count also includes eleven changed test lines.

Seven removed helpers accounted for 528 lines:

| Removed helper | Replacement / assessment |
|---|---|
| `_source_alignment_variant` | Source container projection and shared-axis clustering. Its many choices of text/rule/hero anchor were folded into structural inference. |
| `_source_rule_span_variant` | Source-relative container dimensions and source-axis snapping. Source imperfections outside the new thresholds are intentionally retained. |
| `_source_panel_center_variant` | Parent/container relationships and canvas center constraints. Its special treatment of duplicated outlines is no longer an independent override. |
| `_source_side_flourish_center_variant` | Exact source offsets and inferred axes. Rome-specific mini-rule and separator branches were intentionally retired. |
| `_source_centered_copy_group_variant` | Functional copy containers and the canvas parent axis. |
| `_source_panel_child_gutters_variant` | Rigid parent/child transforms and backing-aware font calibration. Audit found that the later snap pass still needed an explicit padding-preservation guard; restored generically. |
| `_source_panel_margin_variant` | Whole-container bounds and source-aware canvas gutters. Its raster-panel size thresholds are not retained as separate policy. |

Other deleted lines were primarily the old `relationship_variant` and `font_variant` implementations, repeated sibling-candidate creation, repeated feasibility gates, and post-selection patch branches. Those functions still exist with replacement implementations. The optimizer, objective/energy calculations, collision checks, candidate reflow families, background handling, rotation candidates, Qwen decoding, and renderer were not deleted wholesale. Compatibility helpers such as `repair_relationships` and superscript anchoring also remain available.

## Useful behavior that was omitted

It would be incorrect to describe every deleted line as useless or every old behavior as automatically covered by the new tree.

### 1. Parent padding could be lost during snapping

A source label inset 2px from its backing's left edge fell inside the 4px source-axis threshold. The new snap pass placed both edges on their median and reduced the inset to zero. The old gutter repair protected this relationship later in the pipeline.

Restored a generic backing constraint: a snap cannot consume a positive source inset or worsen existing containment. This works for any inferred backing and does not recognize particular text, assets, ornament types, or design IDs. A regression test reproduces the zero-padding defect before the fix and passes after it.

### 2. Authored corner crops were pulled fully onto the canvas

The removed relationship code preserved deliberate source cropping on corner artwork. The replacement's whole-container clamp instead moved a source image from `(-20, -10)` to `(0, 0)`, exposing artwork that had intentionally been outside the canvas.

Restored source crop fractions as a parent constraint for axis-aligned raster containers with a deliberate single-edge crop on each axis. The artwork and its children receive one shared transform. Interior objects gain no new bleed permission, and the existing source-bleed validator remains active. This is bounded corner-crop preservation, not unrestricted off-canvas placement. A second regression test failed before the fix and passes after it.

## Intentional behavior changes

Some old policies are deliberately not equivalent: independently growing fonts by up to 45%, manually repairing a source's imperfect Rome alignment, and separate outline/ornament-specific overrides. Font fitting now uses rendered glyph probes, widening before shrinking, with source hierarchy supplied by container transforms. The audit does not establish identical output for every historical source design or a universal minimum readable font size in physically crowded targets.

## Validation

- **58 focused tests pass**, including the two new regressions and the existing structural, Qwen pipeline, font, and selection checks.
- **Eight controlled before/after replays** use the same saved V65 inputs. All sixteen renders have no asset warnings, unintended text overlaps, or glyph-box overflow.
- Four replays are unchanged. The four affected cases have maximum coordinate changes of approximately 2.44px, 0.27px, 6.38px, and 2.89px respectively; their before/after images were visually inspected.
- The wider mechanics suite remains at 22 passes, two failures, and one error: the same dense-leaderboard gutter, peer-row, and subheading-role problems already reproduced on the V64 baseline.
- Python compilation and `git diff --check` pass.

[Controlled before/after replay viewer](/Users/kelvinmathew/Documents/MagiCanvas/data/qwen_lora/engine_deletion_audit/review.html) · [Replay measurements](/Users/kelvinmathew/Documents/MagiCanvas/data/qwen_lora/engine_deletion_audit/replay_results.json)

These replays isolate the restored constraints; no fresh Qwen inference or full candidate-ranking rerun was performed for this audit. V65 remains the last full model-and-ranker review. The conclusion is that the reduction was largely justified, **but two useful safeguards needed restoration**, which is now done in the shared structural pass.
