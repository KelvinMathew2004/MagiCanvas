# Additional industry-algorithm research

October 29, 2026. Production defaults are unchanged. This follow-up tests two
distinct ideas across **134 frozen requests**, using the same recorded Qwen
proposals and verified current baseline. These consumed development cohorts
include 59 regression tasks and all 89 reference pairs, which map to 83 unique
requests. Reference coordinates are used only for reporting. No training,
fresh inference, new model download or user action was required.

## What the public sources actually establish

| Primary source | Established mechanism | What it does not establish |
|---|---|---|
| [Adobe InDesign composition documentation](https://helpx.adobe.com/indesign/desktop/format-and-style-text/composition-and-text-wrapping/set-text-composition.html) | Paragraph Composer considers multiple lines together; Single-line Composer finalizes each line separately. Both respect composition settings | That Express uses the same composer, or that balanced lines are always preferable for a poster |
| [Chrome balanced-wrapping explanation](https://developer.chrome.com/docs/css-ui/css-text-wrap-balance) | Search for a smaller effective line width without increasing line count. The article also explains limitations inside cards | A drop-in replacement for our supplied renderer, or permission to insert newlines/change text content |
| [WebKit typography explanation](https://webkit.org/blog/16547/better-typography-with-text-wrap-pretty/) | Distinguishes balanced line lengths from paragraph-wide typographic optimization and discusses performance limits | That a shorter final line is necessarily a defect in an authored composition |
| [Canva engineering: responsive design system](https://www.canva.dev/blog/engineering/adding-responsiveness-to-canvas-design-system/) | Easel uses composable responsive UI primitives, spacing tokens and breakpoints | Magic Resize's artwork retargeting algorithm. Product UI layout is a different problem |
| [Canva element-positioning API](https://www.canva.dev/docs/apps/positioning-elements/) | Default placement centers and scales added elements; apps can control placement | The private inference, solver or candidate-selection implementation behind resizing |
| [CSS Flexbox specification](https://www.w3.org/TR/css-flexbox-1/) | Flexible sizing, wrapping and alignment inside declared containers | How to infer correct containers from a flat artistic poster, or a reason to stretch images |
| [ReverseORC](https://arxiv.org/abs/2202.11523) | Infers responsive OR constraints from multiple observed sizes | Recovery of the intended layout from a single source image, without additional supervision |
| [Adobe cropping research](https://research.adobe.com/publication/automatic-image-cropping-using-visual-composition-boundary-simplicity-and-content-preservation-models/) | Scores composition, boundary simplicity and content preservation together | That skin-color or texture heuristics reliably detect arbitrary subjects |

The sources justify experiments, not a claim that MagiCanvas reproduces Canva
or Express internals. No proprietary implementation was available for review.

## Newly executed experiments

Fixed settings: `config/industry_followup.json`. Implementations are isolated in
`experiments/layout_upgrade/industry.py`; production imports neither arm.

**Paragraph balancing:** sweep 24 narrower widths, down to 70% of the current
allocation. Retain actual font, tracking, text content, alignment anchor, height
and measured line count. Require at least 25% reduction in squared line-width
variation. Skip authored newlines, controls and rotated text. Use the normal
glyph, overflow, source-contact and contrast checks. This is a width-search
approximation inspired by browser balancing, not Adobe's Paragraph Composer or
a full Knuth–Plass implementation. It preserves the renderer's existing greedy
word-break contract; no protected text is rewritten.

**Intact-group neighborhood:** at most 12 deterministic proposals per request,
using adjacent whole-group swaps and ±8% two-sided allocation changes. Children
move/scale rigidly and font tracking follows scale. Overlapping groups, regional
backings and ambient/edge art are not independently moved. Rank with the fixed
source-only quality cost, then require geometry/ink and rendered checks. This
tests a bounded local-search move vocabulary without importing another engine
or relying on unavailable critic weights. It is not a reproduction of that
engine's complete learned search.

| Arm | All-request result | Reference result | Production decision |
|---|---|---|---|
| Paragraph balancing | 13 proposals; four rejected on the unchanged quality margin; nine research outputs, zero new automated flags | All 89 reference comparisons unchanged: geometry MAE .0447733435, relative font MAE .168213152 | **Explored, rejected as a default.** Some headline wraps improve, but evidence is concentrated in a small development stratum, with no reference-cohort benefit. Narrower copy can lose intended column rhythm |
| Intact-group neighborhood | 621 proposals; 612 rejected on quality margin, eight on geometry/ink checks; one research output, zero new flags | All 89 reference comparisons unchanged | **Explored, rejected for production.** The survivor enlarges the logo and shrinks the corner ribbon; no convincing general improvement in the dense composition |

Nine plus one means **ten arm/request comparisons**, covering nine distinct
requests. Only the best geometry-qualified proposal per arm is fully rendered;
621 proposals do not mean 621 full renders. All proposed transformations,
screening reasons and post-realization geometry remain in audit JSON.

Native-resolution before/after checks on all ten research outputs found no new
invisible-text or source-floor contrast failures. These checks use a shared
all-text mask and retain documented rotation/color exclusions and ownership
limitations. Zero new flags does not establish visual quality. All ten changes
were directly inspected, including full-size small-type details.

The paragraph arm retains actual fonts and line counts on every changed text.
Rag variation improves on 13 eligible text elements across the nine outputs.
For example, two-line headline variation falls .2040 → .000382 in one banner;
that is a typographic metric improvement, not proof the whole design is better.
The normal candidate realizer also shifts a complete three-control row by
0.536 pixels in one case, including both backplates and labels;
those secondary changes are recorded explicitly rather than attributed to
paragraph balancing alone. Existing sparse compositions, tiny micro-format
copy, missing-asset placeholders and imperfect card proportions remain.

## Remaining mechanisms: already covered or not justified as new ports

| Mechanism | Current disposition and evidence |
|---|---|
| Ordered unequal-width row-break DP | Already implemented in `container_flow.ordered_peer_rows`; exact contiguous partitions, edge gaps and no control stretching. Existing tests cover unequal widths. A duplicate port adds no distinct capability |
| Balanced measured columns | Already implemented in `container_flow.measured_columns`; DP uses real wrapped heights. Different objectives need a separate measured failure, not a renamed copy |
| Guillotine rectangle packing | Earlier CP-SAT/free-rectangle experiments already tested alternative macro packing. A greedy split can orphan captions or invert reading order. Not newly executed here; retained as a lower-priority distinct family, not declared disproven |
| Row/column/grid OR alternatives | Existing measured sibling grids, dual granularity and macro reflows already explore this family. Earlier 134-request results record the failures of blanket regrouping |
| Cassowary-style linear constraints | Shared axes, insets and the earlier LP implement a relevant linear subset. A linear solver cannot choose the correct non-overlap topology or repair an incorrect inferred parent tree. Not claimed as a full Cassowary implementation |
| Random hill climbing with a learned critic | Move vocabulary tested above with a bounded deterministic quality proxy. Original critic weights are unavailable; synthetic-damage critic results did not establish human-preference quality. The complete learned system remains untested |
| Saliency/skin/detail-based subject preservation | Earlier cheap subject heuristics were rejected visually. Existing clean-zone checks measure text-background suitability, not semantic object identity. Not relabeled as a couch/face detector |
| Learned composition-aware focal cropping | Credible, but needs a verified subject/composition model, packaged weights and separate crop-quality/latency evaluation. Not implemented or dismissed as impossible; defer while source-image model training is in progress |
| Seam carving / nonuniform asset retargeting | Conflicts with protected asset geometry and the user's strict aspect-preservation preference. Do not apply to artwork; a different authorized image-editing mode would be required |
| Paragraph-wide Knuth–Plass composition | Renderer contract permits width/font changes, not custom break arrays or content rewriting. Requires an explicit renderer/converter contract and parity tests; not silently simulated by inserting newlines |
| Coarse-to-fine render quotas | Earlier 134-request funnel lost an independent rendered family on two tasks for only 2.76% render savings. Explicitly rejected; an incumbent-retaining policy would be a new experiment |
| Shared growth / exact order statistics | Earlier 134-request experiments are explicitly **explored and rejected for production** in `ALGORITHM_FOLLOWUP.md`; code and counterexamples retained |

The local public-repository branch/blob audit is retained privately in the
ignored research evidence. All **34 local remote-branch tips in nine repositories**
still match the previous inventory. This follow-up rechecked the distinct search
and packing implementations against current code; it did not re-run every historic
engine, fetch new remote tips, import third-party runtime code or borrow trained
weights. Historical static coverage must not be confused with runtime evidence.

## Reproduce and inspect

```sh
.venv/bin/python -m pytest -q tests/test_industry_followup.py
.venv/bin/python scripts/evaluate_industry_followup.py --workers 2
.venv/bin/python scripts/report_industry_followup.py
.venv/bin/python scripts/refresh_regression_viewer.py --check
.venv/bin/python scripts/package_regression_review.py
```

Evidence: `data/upgrade_exploration/industry_followup_20261029/`.
Research viewer: `http://127.0.0.1:8766/review/industry_followup/index.html`.
It shows Original/Resized only, with a dropdown for experimental versus current
resized output. The canonical production viewer remains unchanged and current.
No new inference-latency or Mac/Docker speed claim is made.

Validation: **464 tests and 34 subtests passed**, including seven new research
invariant tests. The production code/input snapshot still matches the canonical
59-case viewer. The known flagged dense 320×50 conference output remains.

The useful next evidence is the fresh matching-base adapter comparison already
prepared. These negative results do not prove the current engine globally
optimal; they show that these particular no-training changes lack sufficient
evidence to replace its current behavior.
