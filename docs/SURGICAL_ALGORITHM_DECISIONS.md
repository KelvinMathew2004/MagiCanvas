# Surgical algorithm exploration and decision ledger

Reviewed October 28, 2026. This records all six suggestions in the latest Gemini
attachment, related earlier experiments, implementation limits and the evidence
needed to revisit a decision. **No new production layout rule or model default
was enabled in this follow-up.** Rejection here means “not justified by this
experiment,” not “the idea can never work.”

The earlier, broader ledger is [UPGRADE_EXPLORATION_AND_EVALUATION.md](archive/UPGRADE_EXPLORATION_AND_EVALUATION.md);
its machine-readable index is `config/upgrade_experiments.json`. Completed
model/grammar/sampling pilots are in [UPGRADE_PILOT_RESULTS.md](archive/UPGRADE_PILOT_RESULTS.md),
and the six larger algorithm arms are in [TRAINING_FREE_SCALE_RESULTS.md](archive/TRAINING_FREE_SCALE_RESULTS.md).
Those documents retain unfinished research rather than treating everything as
evaluated. The current attachment's SHA-256 and executed code hashes are in
`data/upgrade_exploration/surgical_followup_v1/manifest.json`.

## Controls and meaning of the measurements

- 134 frozen source/target engine tasks, including all 89 held-out reference
  pairs (83 distinct requests). Qwen remains enabled; a deterministic winner is
  named honestly. This is a development analysis, not new inference or unseen
  validation. The 59-case public regression snapshot remains current.
- Candidate generation/filtering never reads human target coordinates. Reference
  geometry/font error is calculated afterward. No new human labels or training.
- Font experiments edit saved final proposals to isolate the hypothesis. Unsafe
  proposals are deliberately retained in **research** reports to expose failures;
  they do not become production winners. This is not a full re-optimization of
  every original shortlist.
- The topology experiment filters the **14 earlier changed retrieval/grouping
  outcomes**, not every generated candidate. The gated critic uses all 178
  previously consumed paired labels and cached fixed scores, not a fresh
  full-engine or runtime A/B. These distinctions prevent inflated claims.
- All 36 research proposals were rendered and inspected directly, including
  ten raw-font repairs. Native contrast audits cover those ten. Public cards
  display Original/Resized; current/proposed resized images can be switched.

## Decisions at a glance

| ID | Proposal | What was implemented/tested | Decision |
|---|---|---|---|
| S01 | Line-count inverse fit + 18% width slack | Source-width font goal and a separate raw-model font goal; bounded measured width search, preserved anchors/tracking/backings | Keep experimental; existing calibration covers most opportunities, small additional gains do not justify a general switch |
| S01b | Secondary type within .75–1.25 of headline scale | Tight-block type-scale band across 134 outputs | Reject as a universal rule: 9 new flagged layouts |
| S02 | Dual raw/rotated envelope clamp | Mathematical prototype, 500 randomized rotations, all 134 outputs, micro-banner counterexample | Formula is sound for independent rectangles; do not mutate verified final groups blindly |
| S03 | Containment + exact sibling order + occupancy activation | Literal and explicit-owner interpretations on current outputs and 14 earlier alternatives | Do not promote: retained fractured lockups and worsened reference error |
| S04 | Zero-size quarantine and tiny-banner floor relaxation | Existing source no-paint rules checked; three measured-grid scale variants and dual clamp on the failing micro-banner | Preserve no-paint quarantine; reject lowering readability just to clear flags |
| S05 | Elongated-only/top-two/0.15-margin tiled SigLIP | Fixed gated policy on all 178 recorded preferences; mixed-score counterexample | Reject this threshold policy; no <15% latency guarantee established |
| S06 | Clean training targets + line-count annotations | All 1,231 unique train/validation targets: geometry/line sidecars; 32 native target audits | Adopt offline screening/sidecars; no automatic dataset deletion or prompt-schema change |
| X01 | Per-text glyph ownership audit (additional idea) | Native isolated masks + leave-one-out renders for 111 text elements in the same 32 targets | Adopt as an offline validation tool; much stronger visibility evidence than a shared mask |
| X02 | Tolerance-aware boundary triage (additional idea) | Separate strict-negative numerical counts from >.01px physical excess | Adopt in the research report; preserve historical metric results and their provenance |
| X03 | Uncertainty-aware selective crop scoring | Analytic cost/calibration review, not a new trained confidence model | Defer until new preferences and a measured engine-level budget justify it |

## S01 — Font and line-count recovery

**Agree with the direction.** Intrinsic text capacity depends on font, width,
tracking, breaks and language. A change in line count is discrete, so a smooth
root-finding method is not automatically appropriate. The bounded experiment
uses a finite width search and actual renderer-backed measurement. W3C text
layout distinguishes forced and soft line breaks; the source's soft line count
is useful evidence, not an invariant across arbitrary aspect changes.
[CSS Text](https://www.w3.org/TR/css-text-4/),
[SciPy root-finding preconditions](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html).

The code already has:

- `engine/resize.py::_calibrate_text_boxes`: source line-count preference,
  empty-space widening, backing/neighbour limits, height release and measured
  fitting before shrinking.
- `_fit_text`: its upper bound is `desired`; it does **not** arbitrarily inflate
  beyond that desired size. Other layout passes determine that desired size.
- `_set_font_size`: scales tracking with font size.
- Local rotated-control fitting and painted label centering.

The 134 outputs contain **814 measured text occurrences**; 23 have more lines
than source. Raw model font values are available for 739; 178 selected sizes
are less than 75% of their raw values. This diagnoses a size change, not its
cause: raw type may be too large for its box or available slot.

On the 89 reference pairs, of 307 text occurrences, the selected line count
matches the reference in **305**, and source matches in **303**. Thus large
font MAE is not evidence that extra wrapping is the dominant remaining cause.
Line counts use the same source styles plus reference geometry/font fields;
they are not OCR annotations from dataset thumbnails.

Two independent width-slack tests preserve source line count, alignment anchor,
explicit breaks and tracking, expand no more than 18%, respect source backings
and avoid new contacts. They only attempt upward recovery up to 1.25× the
current size, and do not shorten copy:

1. Source-width-derived proportional font goal: **no additional feasible edit**.
2. Raw-Qwen font goal: **18 labels / ten tasks** change. Four proposals acquire
   primary-axis or control-padding flags. Native source-floor contrast does not
   regress in any of the ten, but contrast cleanliness does not remove those
   geometric failures. Six are geometrically clean, with mostly small gains.

Across the 89 reference pairs, the unselected raw-font proposals change geometry
MAE **.044773 → .044747** and font MAE **.168213 → .168189**. These tiny averages
include the unsafe proposals and are not evidence for production promotion.
Several affected tasks are supplemental designs without reference targets.

The .75–1.25 headline-scale band changes **48 elements in 14 tasks**, including
making some CTA/day labels almost unreadable and enlarging copy into collisions.
**Nine tasks acquire new flags.** Font MAE improves only .168213 → .167926;
geometry is unchanged because only fonts/tracking change. This is insufficient
to justify tying all captions, badges and legal copy to one headline scale.

**Decision:** keep bounded raw-font recovery as a candidate experiment, not a
mandatory final pass. The universal band is rejected. A useful next experiment
would apply capacity recovery **before** the final shared-axis/ownership passes,
retain the incumbent and verify selection end-to-end on new families. Do not
expand thresholds just to obtain a positive result.

Evidence: `diagnostic_summary.json`, `diagnostics/`, `font_native.json`,
`reference_comparison.json`, and `raw_font_review_0..3.jpg` under the local
follow-up folder. Prototype: `experiments/layout_upgrade/surgical.py`.

## S02 — Boundary clamp: real issue versus numerical residue

The cited held-out image has `y = -2.842170943040401e-14`, rotation 0°, and
height 400 on a 400px canvas. The historical `measure()` flags any negative
left/top, but tolerates 1e-4px on right/bottom. This is a numerical counting
asymmetry, **not a rotated-envelope leak**. No render benefit follows from
shrinking that image. Historical evidence is retained, not silently recomputed
with different tolerances.

Across all 134 outputs there are two strictly positive envelope violations:
that roundoff residue and a **6.95px title excess** on the known 320×50 conference
task. `_bounded_qwen_elements` already uses both raw and rotated envelopes in
the proposal path. `_physical_layout_safe` checks both with tolerance.

The proposed mathematical clamp was implemented independently and tested on
500 random positions, sizes and angles. It preserves image aspect ratio and
fits both envelopes within numerical precision. But moving/scaling individual
children can detach them from their parent or create new text collisions.
On the failing micro-banner, dual-clamping the title removes the structural
boundary flag while leaving text/control overlap and glyph overflow.

**Decision:** no unconditional last-output clamp. The engine deliberately emits
the exact layout it rendered and judged; a post-verification mutation breaks
that contract. Any future repair belongs **before** verification, moves an
owned group coherently and must handle intended source bleed separately.
Raw and transformed boxes are genuinely different geometric quantities;
[CSS Transforms](https://www.w3.org/TR/css-transforms-1/) documents that
distinction, but does not establish a poster-specific repair policy.

## S03 — Topology gate for retrieval and loose grouping

**Agree that stronger ownership is needed; disagree that the suggested
invariants are sufficient.** A full photograph often encloses over 60% of
many unrelated foreground objects. Conversely, a 60% overlap test says nothing
about the distance between two detached parts of an emblem. Exact ordering
along the old axis can also veto a legitimate row-to-column reflow.

We tested the literal 60% containment/order rule, then a stronger interpretation
limited to the smallest explicit, lower-layer owner rather than every photo.
Even current production passes only **103/134** literal graphs and **113/134**
owner-aware graphs. That warrants false-veto investigation, not blanket gating.

Of the 14 previously changed retrieval/loose-grouping proposals, the literal
gate plus contrast/25%-occupancy trigger admits seven; owner-aware admits eight.
The contrast trigger **does not bypass containment/order safety**. Keeping the
earlier proxy-cost ordering among survivors (never using reference MAE to pick)
gives 89-pair geometry MAE **.044773 → .047176**, about **5.37% worse**; font MAE
**.168213 → .188615**. The beer emblem still separates vertically, and the
ceramics text/rules still split across a banner while passing these tests.
The medical frame does gain contrast/size in one alternative. Useful isolated
wins coexist with regressions; they do not justify global activation.

**Decision:** retain existing ownership contracts; do not add this as a universal
gate or activate retrieval globally. Revisit with edge identity, relative
attachment distance, painted containment and explicit reflow permission, not
just order and occupancy. Much of that information already exists as soft
relationship costs; its false positives need calibration before hard gating.
Evidence: `topology.json` and `reference_comparison.json`.

## S04 — Degenerate objects and the 320×50 micro-banner

`engine/nested_structure.py::unpainted_ids` already distinguishes exact zero
dimensions, zero opacity, whitespace-only text and wholly off-canvas source
paint. Shared glyph-overflow checks use that set. Inert IDs are preserved.
Making an authored zero-width object 1px wide is **not** equivalent: it may
create visible paint, alter relationships or change wrapping. Small positive
text is not automatically exempted.

The supplied conference task also contains a long title, six speaker cards,
mixed scripts, malformed style runs and authored inert objects. Its remaining
failure is not explained by a zero-height divider alone. Three existing
measured-grid variants at scale caps 1, .75 and .5 still report control
readability/glyph overflow; their median fonts are about **1.42, 1.42 and 1.04px**.
Those are not usable results even if a relaxed threshold could label them clean.
Dual-envelope-only still overlaps copy and controls.

**Decision:** do not lower floors or exempt painted glyphs to turn this case
green. Preserve the flag. A measured local packing repair may still help, but
this experiment does not prove impossibility or solve arbitrary dense strips.
A future capacity report should explain preserved IDs and actual readability
tradeoffs; optional omission/condensation belongs to a separately authorized
product contract, not automatic Stage-1 behavior. Evidence: `micro_strip.json`.

## S05 — Selective tiled SigLIP

Fixed, predeclared gates: aspect elongation ≥2.2, global top-two margin ≤.15.
Applied to the 178 existing human pairs, this activates only **13** comparisons.
Correct choices: **107/178**, versus **109/178 global** and **115/178 full tiled**.
These are consumed retrospective labels; no threshold search was performed.

Two complications prevent the claimed speed/quality guarantee:

1. The tiled and global scores have different numerical distributions. If
   two rescored finalists are inserted into a larger global-score list, an
   unrescored third candidate can win spuriously. A counterexample is tested.
   A safe implementation would restrict the decision to the finalist pair or
   calibrate scores, and still retain the original cost/glyph gates.
2. The earlier eight engine winner changes were not limited to elongated
   canvases. Cheap gating cannot be assumed to preserve all their gains. The
   earlier measured median replay ratio was ~1.39× under non-isolated work;
   a cited single 3.1× example is not the overall latency distribution.

**Decision:** do not replace the optional fixed tiled mode with these thresholds.
Neither <15% overhead nor preserved winner gains has been demonstrated.
An uncertainty-aware sequential scorer remains research: calibrate on new
families, compare full-engine choices and measure isolated wall time/memory on
the intended hardware. More threshold searches on these same 178 labels would
overfit the development set. Evidence: `gated_critic.json`.

## S06 and X01 — Target-data auditing before further training

**Agree strongly with auditing labels.** The export contains 1,122 train pairs
with 1,122 unique targets, plus 309 validation pairs with 109 distinct targets:
**1,231 unique targets** in total. All now have geometry and rendered-line-count
sidecars. The train/validation files and the v2 prompt schema were not rewritten.

The first screening pass finds **632 targets with glyph-overflow warnings** and
**971 with off-canvas source geometry**, often cropped/oversized background
photos or source preview/export differences. These are *screening warnings*,
not 971 invalid human designs. An all-target drop rule would discard most of
the corpus without establishing that the actual target text is unusable.

A predeclared 32-target native subset combines deterministic hash sampling
with a flagged-geometry stratum. Its shared all-text mask marks one target as
possibly invisible and 16 as low-contrast. The additional ownership audit renders
each text in isolation and removes it from the full design to measure its own
visible contribution. Across **111 text elements** (including three rotated):

- zero have an empty isolated glyph mask;
- zero have no measurable visible contribution;
- ten have lower-quintile contrast below 1.15 under the existing engine formula.

Low contrast and total absence of paint are different failure classes. An
all-text mask inside overlapping text boxes can attribute another label's ink
to the wrong text. The new `scripts/audit_target_glyph_ownership.py` addresses
that diagnostic weakness; it does not silently change the production metric.
Contrast here uses the engine's legacy gamma convention, **not** a WCAG
conformance calculation. W3C contrast guidance also distinguishes incidental
text and logotypes, so its thresholds are not an automatic training-data
deletion rule or a substitute for the challenge's own requirements.
[W3C contrast guidance](https://www.w3.org/WAI/WCAG21/Understanding/contrast-minimum).

**Decision:** adopt sidecars and ownership-aware offline auditing. Defer clean
dataset generation until actual converter clipping, occlusion and faint but
intentional art are separated. Full native auditing of all 1,231 targets remains
unfinished; 32 is explicitly a sample. Adding `lines` to the current inference
prompt is a contract change and should accompany matched training/evaluation,
not be described as free compatibility with the existing adapter.

## What is retained for future work

- [x] All six suggestions read, mapped to current code and given a concrete test
  or a documented integration reason.
- [x] Experimental font slack, hierarchy, dual-envelope, topology and selective
  critic implementations; six property/counterexample tests.
- [x] Full unique-target geometry/line sidecars and native 32-target audit.
- [x] Per-element isolated/leave-one-out glyph audit, kept offline.
- [x] Original/Resized review cards and raw/intermediate JSON evidence.
- [ ] New-family full-engine candidate A/B for raw-font recovery before axes.
- [ ] Full native target audit and reviewed converter-defect quarantine policy.
- [ ] Calibrated attachment-distance/painted-topology graph and new-label critic
  uncertainty study. Neither is implemented as a production hard gate.

No new model training, adapter upload or user setup is required for this work.
We did not promote a weak change merely to report an engine improvement.

Verification: **404 tests and 34 subtests passed** after the new experiments
and six property/counterexample tests. `refresh_regression_viewer.py --check`
reports CURRENT. Engine and renderer files were not changed in this follow-up;
the existing 59-case production snapshot is retained, not relabeled as fresh
inference or a newly rendered regression run.

## Reproduce and locate evidence

Local evidence root: `data/upgrade_exploration/surgical_followup_v1/`.
Research viewer: `http://127.0.0.1:8766/review/surgical_followup/index.html`.
Production viewer remains `http://127.0.0.1:8766/review/main_engine_qwen_outputs/index.html`.

```sh
.venv/bin/python scripts/explore_surgical_upgrades.py --phase diagnostics
.venv/bin/python scripts/explore_surgical_upgrades.py --phase policies
.venv/bin/python scripts/explore_surgical_upgrades.py --phase targets --native-targets 32
.venv/bin/python scripts/audit_target_glyph_ownership.py --workers 2
.venv/bin/python scripts/report_surgical_upgrades.py
.venv/bin/python -m pytest -q tests/test_surgical_experiments.py
```

These are pinned development snapshots; use a new root/version before changing
the hypotheses. Do not relabel old native audits as freshly rendered. Full
datasets, fonts, weights and complete target sidecars stay local; the Gemini
packet includes aggregate findings, relevant code and per-case review evidence.
