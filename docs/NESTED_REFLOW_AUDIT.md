# Nested reflow audit and implementation decisions

October 28, 2026. Development experiment; not an unseen benchmark.

The attached review is treated as a proposal, not proof. The model adapter and
raw decoded predictions stay fixed. A deterministic winner is still a
Qwen-enabled run. No training, adapter upload, or production model promotion
is part of this change.

## Checklist and decisions

- [x] Separate regional backing scope from atomic ownership. An ungrouped
  rectangular region containing multiple independently grouped cards no longer
  welds all their children into one rigid object. Its paint wraps the realized
  children with bounded source-derived padding. Single cards may still contain
  a nested CTA. No named design, content string or template ID is consulted.
- [x] Preserve authored source geometry for grouping. Measured text-frame
  tightening must not erase the baseline/column evidence used to discover rows.
- [x] Infer repeated two-field rows only with at least three consistent,
  disjoint pairs, matching type scale, shared axes and regular row pitch.
  Rows containing a third/fourth field are excluded: a real table needs a
  table model, not overlapping two-field guesses. Affiliated translucent
  hairline rules stay with their rows; translucency
  alone is not evidence of ambient paint.
- [x] Reflow repeated explicit card/control groups using bounded grid options.
  Cards with a photo above copy measure wrapping before packing; overlays retain
  their existing arrangement. Choose retained type scale and media prominence,
  rather than the shortest possible thumbnail row.
- [x] Realize two-field rows with a measured trailing field and wrapping leading
  field. Keep every original ID and string; never rewrite a price or URL.
- [x] Place independent artwork in conservatively quantized free rectangles
  after primary flow. No free rectangle means reject that candidate, not overlap.
- [x] Bound isolated controls by authored type hierarchy. A slot's available
  width is a limit, not an instruction to stretch a CTA across the whole column.
- [x] Handle exact no-paint source layers separately: zero area, opacity zero,
  whitespace-only text, or boxes entirely outside the source viewport. Keep IDs
  and immutable styles in output with minimal bounded geometry; exclude source layers without a viewport footprint from visual flow
  and ink diagnostics. A one-pixel visible box is **not** exempt.
- [x] Fit actual rendered glyph widths, including integer raster quantization.
  Unbreakable tokens use the full allocated width before type shrinks; tracking
  scales with font size. Visible speaker-name clipping remains a defect.
- [x] Verify source-unsupported image overlap is already a physical defect in
  the selector objective and contract checks. Keep that gate; the menu failure
  needed clean row-aware candidates, not another duplicate image-overlap rule.
- [x] Exempt attached photo captions from the control hierarchy cap and its
  prominence penalty. Their full-width backing belongs to the photo, even when
  a generic role classifier mistakes the caption for an action.
- [x] Add viewer filters for changed outputs, flagged outputs and new flag types;
  show target dimensions and previous/current winners. Only Original and Resized
  appear. Native-size images and raw/intermediate audit JSON remain linked.
- [ ] Promote the structural ranker. Deferred: the relevant small campaign
  validation slice did not establish a reliable win over the incumbent, and
  zero flags do not capture all aesthetic failures. No case-specific whitelist.
- [ ] Switch to 1.7B by default or retrain. Deferred pending a broader quality
  and runtime gate. The earlier 13-task experiment showed 10 identical final
  outputs and higher generation cost; better raw completion alone is insufficient.

## Research and limits

The useful analogy with CSS is intrinsic sizing plus ownership, not a universal
30-line algorithm that guarantees good graphic design. The W3C
[Flexbox specification](https://www.w3.org/TR/css-flexbox-1/) constrains flex base
sizes by content and min/max sizes; the
[Grid specification](https://www.w3.org/TR/css-grid-2/) measures intrinsic
contributions before distributing tracks. This implementation applies that
principle to measured text/cards, without claiming CSS or a research paper proves
our heuristic grouping thresholds correct.

The proposed blanket 22% area / four-child cap was not adopted: it would split
legitimate multi-part cards and emblems. Regional scope and repeated source
ownership provide narrower evidence. The claim that all conference overflow
came from inert boxes was also incorrect: the original selected output clipped
a visible long speaker name. Actual raster measurements remain the gate.

Preserving all content in a 320×50 output with six speaker portraits and long
names leaves extremely small type even with collision-free placement. A layout
can pass geometric checks and still be visually weak. Full 89/309 release
benchmarks are not newly claimed by the 59-case development replay.

## Verification

Frozen baseline: `data/qwen_lora/regression_suite/bbfa0e1465407168`.
Current results, exact snapshot hashes and before/after changes are recorded in
`review/regression_suite/run_manifest.json` and `summary.json` in the Gemini
packet. The baseline uses identical saved Qwen predictions and adapter.
Final counts and inspected limitations are appended after the completed replay.

## Visual acceptance notes

Native-size comparisons caught two regressions that automated flags did not:
a full-width photo caption narrowed under the action-size cap, and an invoice
with four fields was interpreted as overlapping two-field rows. The cap now
exempts source-attached photo captions, and the row detector rejects baselines
with additional fields. Both fixes have explicit regression tests.

Do not read an unflagged stress render as perfect: product names/prices remain
small at 300×250 and 160×600, especially with five cards and a long headline.
The conference skyscraper retains every speaker but its exceptionally long
name is tiny; substituted fixture fonts also lack some non-Latin glyphs.
The menu's fields and leaders stay together, but the last row can flow to the
next column/band. Some older independent designs still have excessive spacing
or large supporting copy. Those are recorded limitations, not training wins.

The final release keeps source styles immutable, including opacity. Zero-area
or wholly off-viewport source artifacts receive minimal bounded geometry while
retaining their IDs. A rotated box outside the raw canvas is not assumed to be
inert: its rotated pixels may still enter the viewport.

## Final frozen-cohort results

Release snapshot: `31e2ab98f1909007`, compared with `bbfa0e1465407168` using
identical decoded Qwen predictions, adapter and target canvases. Qwen remained
enabled throughout; most improvements selected deterministic measured candidates.

- 59 conversions completed: fixed 38 plus 21 supplemental.
- 13 changed outputs; 46 rendered output JPEGs are byte-identical. Earlier
  replay-only floating-point drift was below 1e-6 px and is absent in this run.
- All 18 official and 12 core dataset renders are byte-identical. Of the eight
  older alternate conversions, five are unchanged and three were visually
  inspected after changes. No new automated flag types appeared.
- Flagged outputs: **7 → 3**. Product-card compact/skyscraper conversions,
  conference skyscraper and menu banner/skyscraper improve. Readability remains
  limited by dense content and substituted fonts; no claim of visual perfection.
- Remaining flagged outputs: supplied conference 320×50 (six types of defects),
  festival 1920×120 (control collision), festival 160×600 (control readability and
  text/control clearance). Those remain visible in the Flagged outputs filter.
- **357 tests + 34 subtests pass.** `refresh_regression_viewer.py --check` reports
  CURRENT. This is cached-proposal engine regression testing, not a latency
  benchmark, fresh inference run, model-training result, or full 89/309 release gate.

Reproduce the exact identity/grouping audit with `scripts/audit_nested_reflow.py`;
its report is included as `evidence/nested_reflow_summary.json` in the packet.
The stale 1.7B alternatives are hidden rather than relabelled as current after
an engine change. No default-model switch or structural-ranker promotion occurred.

The final viewer refresh also fixes per-card source dimensions: the previous
publisher accidentally reused the last case's dimensions in every caption.
Images and layout geometry were unaffected. The completed cohort is replayed
again under the corrected publisher snapshot; it is not fresh model inference.
