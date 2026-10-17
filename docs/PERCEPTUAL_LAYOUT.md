# Perceptual layout heuristics

October 16, 2026 follow-up to the second Gemini review. These are model-free
heuristics inside `engine/resize.py`, not a reproduction of a learned design
system or a claim to match Adobe's reference outputs.

## Implemented changes and limits

| Suggestion | Implementation | Limit |
| --- | --- | --- |
| Background saliency | Render existing full-bleed image backgrounds once, downsample to 96 × 96, measure edge density and color variance in nine cells. Generate up to three copy/action placement candidates near calmer cells; add a soft text/background busyness cost. | This measures busyness, not semantic saliency. It cannot identify a face/product, choose a semantic crop, or change asset pixels. Flat backgrounds need no extra seeds. Missing-asset warnings, rendering errors and >16 MP canvases skip the optional map. Actual candidate contrast checks remain active. |
| Icon/caption relationships | Infer soft links only for at least two similarly sized, tightly aligned repeated icon/caption pairs. Add a candidate with paired feature cells, caption widening, distance and neighborhood-exclusivity costs. | No output `group_id` is added. Explicit groups remain authoritative. Ambiguous proximity can be wrong; inference is deliberately narrow. This is not a general pricing-table/card parser. |
| Balanced typography | Cache per-line measured widths and word counts. Score uneven wraps and severe one-word final lines; explore modest width/font reductions for standalone headings/subheadings. Short captions also receive a rag cost. | Preserve content and explicit authored newlines. The renderer's greedy wrapping remains unchanged; this is not Knuth–Plass. Glyph metrics remain approximate for unknown fonts. |
| Small-format typography | Before packing, compute one continuous hierarchy exponent `gamma = min(1, .65 + .7 * sqrt(target_area/source_area))`. Font targets use source median × packing scale × (source font/source median)^gamma. Small fonts shrink less relative to large headings. | Font fit and all-content preservation still take precedence. No impossible 11–14 px guarantee. Rigid illustration groups retain one common scale; their internal fonts are not independently distorted. |
| Alignment synchronization | Split and inline-banner copy candidates align standalone text left on a common axis and tighten line extents. Their `text_align` can adapt to left; stacked/preservation candidates retain source justification. | Font family, content, colors and explicit group internals remain fixed. Alignment is a schema-supported layout property, not a text rewrite. All layout families remain candidates, not mandatory formats. |
| Whitespace distribution | Add a low-weight penalty for large uncovered central regions rather than relying only on overall outer bounds and center of mass. | No uniform-occupancy target. Intentional negative space remains valid; this weak signal does not replace a designer's composition judgment. |

Button sizing also measures its desired font before calculating padding/width,
and prefers a single line when the available dimensions permit it. An authored
multi-line label is left intact. Readability and space constraints can still
require wrapping in extreme targets.

## Scoring and diagnostics

New soft energy weights: rag 5, pairing affinity 18, central whitespace 5,
background busyness 7. Hard bounds, group preservation and required separation
are unchanged. Background analysis and affiliation inference happen before
seed generation. Optimizer refinement and rendered diagnostics then select among
feasible candidates. Diagnostics expose the nine-cell map, inferred links,
hierarchy exponent and individual energy terms; none of these become official
output schema fields.

The visual ranker now receives up to five candidates from distinct layout
families within a bounded energy band; this avoids comparing only proportional
variants when a split or banner layout is viable. It applies up to two bounded
role-aware refinements, comparing each variant with its parent against the
original render and inferred relationship cues; it stops when the parent stays
best. Source-identified corner
callouts stay pinned to their corresponding target corner, and banner CTAs
reserve extra clearance from hero artwork. These are additional soft/layout
rules, not proof that the ranker understands every asset semantically.
When the source-to-target aspect-ratio shift is at least 1.6×, proportional
scaling remains a generated fallback but is excluded from the rendered judging
shortlist when any feasible reflow exists. This prevents the perceptual model
from selecting a scaled source composition that fails to use the new canvas.

Use the existing inspection command:

```sh
python scripts/inspect_layout.py input/chameleon_sample_level2_core.json
```

Generated observations are under `output/inspection/` (ignored by Git). The
repeated-feature example found three icon/caption links, selected its paired
candidate, passed hard placement constraints and produced no runner warnings.
The official sample banner selected an inline candidate with a shared left text
axis and a single-line CTA. This is local evidence, not hidden-set validation.
The existing regression contract's allowed layout-property list now includes
`text_align`; content/font-family/style-run preservation remains required.

## Research context

- [O'Donovan, Agarwala and Hertzmann — Learning Layouts for Single-Page Graphic Designs](https://www.dgp.toronto.edu/~donovan/layout/) describes a learned energy-based layout approach with alignment/importance/segmentation analysis. Our hand-weighted heuristics are not its learned model.
- [LayoutNUWA](https://arxiv.org/abs/2309.09506) is an LLM/code-generation method. It is not a lightweight model-free algorithm implemented by the changes above.

The supplied feedback's exact numeric rules and formulas are proposals; citing
research does not establish that these parameters reproduce those papers or
will improve every design. Varied visual evaluation is still needed.

## Mechanical follow-up: quiet regions, caption tolerance and text anchors

The third review identified useful interactions; the following changes were
made without introducing a model:

- Quiet-region seeds now rebuild header/footer bands, a compact copy/action
  cluster and a complementary artwork region. Side or stacked placement follows
  the requested quiet cell and canvas proportions. They no longer translate the
  widely separated copy and action bounds from a split seed. Hard projection
  still validates the resulting candidate; the region preference is soft.
- Repeated captions receive proportional width padding (`0.15 × font_size`,
  within the existing widening cap). This reduces accidental wrapping near
  floating-point thresholds; it does not guarantee that every caption fits one
  line. Shared feature-cell dimensions are stored in a separate `fit_size` hint.
  `bounds` continues to describe the actual items instead of an enlarged virtual
  envelope. Explicit source groups remain unchanged.
- After standalone text reflow, left-aligned widths are tightened only if the
  measured line count and height remain unchanged and the line extents still
  fit. Center-aligned text retains its horizontal center; right-aligned text
  retains its right edge. Thus a width-only optimizer proposal no longer drifts
  the text anchor or leaves avoidable trailing width on left-aligned copy.

For the supplied core design, quiet cells 0, 2 and 8 produced zero copy/art
bounding-box intersection and passed hard placement checks before projection.
The ordinary runner also produced all three official sample JSON/PNG pairs
without structural warnings. These observations do not establish hidden-case
quality or amount to exhaustive testing.

An optional vision layer remains a future experiment. Any subject-map or
perceptual tie-breaker must be evaluated for accuracy, correlation with layout
judgment, actual CPU latency, offline packaging and approved-model compliance.
The proposed approximately 40 ms runtime and equivalence to Adobe's evaluator
are unverified. No weights, downloads, inference runtime, or model hooks were
added in this follow-up.
