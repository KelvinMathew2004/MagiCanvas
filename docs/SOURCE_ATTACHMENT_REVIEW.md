# Source attachments, selector integrity and new-composition testing

October 26, 2026. Historical checkpoint. Read [MACRO_STRUCTURE_REVIEW.md](MACRO_STRUCTURE_REVIEW.md) for the current revision. Earlier source-evidence and visual-contract documents are historical checkpoints.

## General repairs

`engine/source_structure.py` builds a shared, source-only functional graph from enclosing panels, explicit groups, compact painted lockups, nearby aligned copy, superscripts and attached ornaments. It does not dispatch on template IDs or sample text. Independent source-attached proportional and flow candidates compete with Qwen and other deterministic reflows. They preserve internal uniform transforms, source edge contact and parent containment; they do not reconstruct a human target.

The same disjoint groups now drive contrast relocation and attachment scoring. Contrast repair can translate a complete lockup but cannot separate its headline, rules and footer or drag an independent logo. Whole-group bounds correction follows font calibration. Compact single-card compositions may use more of the target short side. Clipped decorative assets retain their source edge; if revealing hidden artwork causes a new glyph collision, a legal smaller frame preserves the visible source fraction. This is not asset editing and cannot exactly preserve every original crop without crop metadata.

Primary-axis checks use whole groups rather than independently centering every leaf of a price/suffix or emblem. Measured text fitting handles rotated captions in their own local coordinates. CTA/copy alignment is applied only to action controls and refuses moves that create new ink collisions; badge peers retain their separate positions. Identity outputs can clear unrelated text touching a CTA while preserving authored artwork overlaps. Clean identity designs remain unchanged.

Selection no longer forces a Qwen family over a materially better measured candidate. Qwen preference and SigLIP apply only inside the global defect-cost tie band. Minor contrast changes remain soft; IDs, bounds, actual clipping and new glyph collisions remain safety checks. Target-coordinate MAE never participates in inference-time placement or selection.

## Final frozen-code verification

Run directory: `data/qwen_lora/source_attachment_verified/`.

| Cohort | Cases | New automatic defect flags | Prediction provenance |
| --- | ---: | ---: | --- |
| Held-out test | 89 | 0 | Identical saved 0.6B proposals, current repairs and selector |
| Official samples × six formats | 18 | 0 | Identical saved proposals; two identity outputs retain disclosed source-authored issues |
| Reviewed Crello / 0.6B | 12 | 0 | Identical saved proposals |
| Reviewed Crello / 1.7B | 12 | 0 | Identical saved proposals |
| Additional validation families | 12 | 0 | Replay of the earlier fresh local inference, not newly sampled predictions |
| Legacy reverse Level 3 fixtures | 8 | 0 | Replay; all winners are deterministic fallbacks |

That is 151 replayed outputs with zero new detected defects and zero engine errors. The validation sample had three automatically flagged failures at the previous checkpoint, plus manually observed video-game-art/copy crowding. All are now clean under the implemented checks. Manual contact-sheet review confirms the origami card/footer separation, edge decorations, Kenner callouts, Pygmalion panels and video-game copy clearance improved. Paris still has weak absolute photo contrast; a source-relative gate is not an ideal-readability guarantee.

On the 89 held-out cases, selected normalized geometry MAE is **0.04444**, versus **0.04491** for valid raw proposals. Qwen-family candidates win **52/89**. On the curated twelve-case arms, repairs increase target-coordinate error from 0.05636 to 0.06808 (0.6B) and from 0.05760 to 0.06996 (1.7B), while satisfying current physical contracts. These results do not justify claiming every repair improves fidelity, nor promoting a model. Geometry proximity and raster quality remain separately reported.

The separate fine-grained source-axis diagnostic still reports **102 relation deviations across 44/89 selected outputs**, down from 158 across 48/89 decoded proposals. Its 0.4% coordinate threshold is stricter than the physical selection gates and can include intended reflow. These counts are retained, not relabeled as perfect alignment. Text-fit issues in that same diagnostic drop from seven to zero; measured stretch, overlap and bounds counts are zero at both decoded and selected stages. This is a comparison of available decoded coordinates, not token-level unconstrained model output.

The full suite passes **160 tests and 34 subtests**. New regression tests cover parent containment, shared contrast groups, local rotated-caption fitting, edge legality, card scale, badge/CTA isolation and cost-tie-only Qwen preference. The raw viewer also verifies tracking scales with predicted font size without repairing invalid raw coordinates.

## New designs, rather than renamed Level 3 examples

`tests/fixtures/novel_level3_sources/` contains four independently authored compositions:

- Field Notes: maker event, 1440×240 source → square and portrait.
- Slow Morning: coffee tasting, 320×1200 source → square and widescreen.
- After Dark: music event, 900×900 source → leaderboard and story.
- North Collective: winter editorial, 840×1120 source → widescreen and square.

Each has 21–23 elements, new text, a different composition and photograph, multiple containers, feature/action information, and neutral element/group IDs. They are not cloned official element sets or engine-generated source resizes. Source photographs are unmodified existing Crello training-split assets, so this tests new compositions and source formats, not unseen-asset generalization. These sources are evaluation fixtures only, never exported as training records.

Source render QA precedes inference, then file hashes freeze the sources. The saved source QA also discloses static control-style warnings, a nested panel/CTA intersection and a one-pixel right glyph-bearing warning in the winter source. These are not silently removed from the evidence; the new resize failures include additional major collisions and scale problems. The evaluator records code and adapter hashes. The gallery shows source, raw decoded Qwen, selected Qwen-ON engine output and Qwen-OFF output, keeping failures visible. An engine result that wins via deterministic fallback is explicitly labeled. No human resized target exists, so all target geometry/font MAEs are null. Results and visual review notes are in the packet's `evidence/novel_designs_summary.json`.

**New-composition result: 8/8 JSON completions, 1/8 complete/valid decoded ID sets, 0/8 selected Qwen-family winners, 6/8 selected outputs flagged, zero engine exceptions.** The ON and OFF arms both have six flagged outputs. Even the two gate-passing outputs have visible tiny-content and disconnected-group problems. These new tests do **not** establish robust generalization: the familiar regression suite is considerably easier for the current engine. The eight sources and outputs remain unchanged and fully visible. Recurring weaknesses are multi-band structure fragmentation, photo/caption association and scale allocation at extreme aspect changes; they need another cross-family root-cause iteration rather than renamed examples or hidden failures.

The older reverse fixtures that reuse the Solstice elements remain a labeled regression archive, not evidence of a new-design capability. The new-composition gallery replaces them in the primary review cards.

## Reproduction

```sh
# Rebuild authored source files only before starting a new frozen run.
.venv/bin/python scripts/build_novel_level3_designs.py
.venv/bin/python scripts/evaluate_novel_designs.py --prepare --output NEW_RUN/novel_designs
.venv/bin/python scripts/evaluate_novel_designs.py --output NEW_RUN/novel_designs
# Re-render a frozen run without generating another model prediction:
.venv/bin/python scripts/evaluate_novel_designs.py --viewer-only --output NEW_RUN/novel_designs
# Current regression audit: see scripts/audit_inventory_fixes.py and
# scripts/replay_validation_sample.py; merge the four held-out shards with:
.venv/bin/python scripts/finalize_visual_contract_audit.py NEW_RUN
.venv/bin/python -m pytest -q
```

Future changes should target repeated failures across different families, accompanied by unaffected-design regression checks. Zero automatic flags does not establish aesthetic perfection or universal adaptation.
