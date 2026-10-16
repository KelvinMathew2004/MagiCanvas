# Adobe challenge requirements audit

Reviewed against the supplied **Final Chameleon Challenge.pdf**, particularly
sections 2-5 and 7, plus the repo's `README.md`, `RULES.md`, `FAQ.md`, and
`chameleon_design_schema_v0.9.json`. This is a team audit, not Adobe certification
or an official score. It describes the implementation as of October 16, 2026.

## Stage-one technical contract

| Requirement | Status | Evidence / scope |
| --- | --- | --- |
| Headless engine receives source design and target canvas automatically | Implemented | Existing `engine.main` calls `resize(source, target_canvas)` without UI interaction. |
| Return the same design structure with updated layout properties | Implemented and locally tested | Official v0.9 canvas/elements output; structural/schema regression checks. |
| Same element IDs, count, content, fonts, and original asset references | Implemented and locally tested | Deep copies; stable source order; no invented or deleted elements; text/font-family/style-run preservation checks. |
| Preserve image proportions and designer grouping | Implemented with limitations | Non-background images and artwork groups scale proportionally. Simple contained backing/label pairs are elastic. Inferred relationships are heuristic, not semantic ground truth. |
| Keep elements in bounds | Implemented and locally tested | Rotated bounds are considered. Tests cover samples and synthetic groups; this does not prove every unseen case. |
| Reflow, not merely stretch a flattened design | Implemented | Multiple layout candidates, typography fitting, unit placement and geometric ranking. No flattened-image resize substitutes for output elements. |
| Render actual assets/fonts into a PNG as well as JSON | Implemented | Starter real renderer, local preview, and previously verified Docker outputs. The runner's existing fallback remains unchanged; do not submit placeholder-only renders. |
| Run in a container without runtime internet | Rebuilt and verified October 16 | New Kiwi dependency installed at build time; linux/amd64 `--network none` run produced all three sample JSON/PNG pairs. Rebuild on the team's submission machine before code freeze. |
| No regenerated/redrawn/replaced asset pixels | Implemented | Uses the given asset files and existing renderer. No generative image/model integration. |
| Missing assets handled without a crash | Implemented by starter renderer | Intentional missing holiday badge remains a labeled placeholder. Do not fabricate it. |
| General solution, not sample-specific logic | Implemented approach; broader validation needed | No ID/asset-name/sample-size lookup layouts. Hidden-case quality is unproven. |
| Short README describing approach and how to run | Implemented | Root README, IMPLEMENTATION.md, viewer docs, and archived original starter instructions. |

The above establishes a functioning technical submission pipeline. It is not a
claim that the design-quality goals below are fully achieved.

## Design-quality goals that remain partial

The brief does not require a particular algorithm, AI model, solver, or training
method. It describes desired outcomes. Current geometric heuristics address
parts of them, but there is no complete implementation of the following:

| Brief goal | Current gap / useful next work |
| --- | --- |
| All text readable, with enough contrast | Font metrics/fitting and tiny-text warnings exist. Competitive candidates now receive rendered local contrast, busyness and visible-text checks. This is a diagnostic heuristic, skips rotated text/oversized canvases, and does not repair every contrast issue. Some extreme targets have tiny copy. |
| No unintended collisions; intentional layering preserved | Cassowary enforces bounds and separation for feasible candidate topologies; atomic groups and source artwork layering are exempt. Dense impossible targets use a reported bounded fallback. There is no comprehensive visual-overlap classifier. |
| Intentional hierarchy, alignment, spacing, and balance | Multi-start energy refinement, typography/affiliation/central-whitespace costs,
background-guided candidates and deterministic raster reranking exist, but they are not a learned aesthetic judge and have no demonstrated match to Adobe's evaluator. More varied visual review and engine iteration are needed. |
| Good results across many designs, element counts and extreme shapes | Tests cover samples, six target formats, and synthetic edge cases. They do not establish generalization to the hidden set. Unknown fonts, rich typography and complex scripts need more coverage. |

A model is **optional**. Installing or training one is not an outstanding
stage-one requirement. If one is added, the offline/approved-model/size rules
must be rechecked, and fine-tuned adapter weights must be publicly accessible.
No local model code was transferred from Project Chameleon.

## Viewer and finale status

- The main brief (3.3 and 5.1, and its embedded submission FAQ) puts the full
  Express add-on in the finale. The standalone repo FAQ Q3/Q5 additionally calls
  a stage-one viewer add-on required but unscored. These supplied texts differ;
  confirm the latest official submission checklist with organizers.
- A local comparison viewer and separately packaged minimal Express viewer are
  present. The Express viewer loads generated output JSON/PNGs or comparison
  bundles and can request flattened PNG insertion through the SDK.
- **Still needed:** live Express sideloading and insertion verification. A build
  alone does not prove host acceptance, usability, or submission completeness.
- **Not built for the finale:** a full Express add-on running the engine against
  live editable designs, the short pitch, and the brief demo video. A standalone
  web preview and flattened PNG insertion are not that full finale experience.
- Suggested extras such as user locks, multiple options, and copy rewriting are
  not mandatory stage-one features. Preserve text and font families under the
  current stage-one rules. Do not introduce asset generation under the supplied
  explicit ban.

## Starter-renderer compatibility caveats

The JSON contract is broader than a guarantee that the starter rasterizes every
optional field perfectly. Inspection found two concrete cases to validate with
organizers before relying on them in custom designs:

- `renderer/converter/colorutil.py` parses comma-separated `rgb(...)` and
  `rgba(...)`. Hex/named shape colors fall back to the renderer's default color.
- `renderer/converter/shapes.py` draws rectangle/ellipse fills and line strokes,
  but does not draw rectangle/ellipse outlines from `stroke_color`/`stroke_width`.

The engine preserves these properties; that is different from implementing their
rasterization. The supplied renderer is unchanged. Coordinate any compatibility
fix against the grading renderer rather than silently replacing it with a
separate implementation. The sample assets use the supported color notation.

## Items the supplied documents leave unresolved

- Approved-model list and exact model-size limit, if the team chooses a model.
- Complete reference-machine CPU/RAM/disk/GPU specification.
- Final schema/version and target-format catalogue, if organizers update the
  provisional materials. Viewer presets are conveniences, not a claimed full
  official catalogue; the engine accepts target dimensions directly.
- Final submission destination/form, deadline/code-freeze details, and the
  stage-one viewer wording above.
- The brand kit describes its assets/fonts as stand-ins. Confirm whether the
  organizers have supplied a final replacement kit before freezing a submission.

These are organizer confirmations, not details the team should invent. This
review used the supplied files; it does not claim to have checked later private
announcements or obtained organizer approval.

## PNG resolution findings

- Viewer download uses the renderer's PNG bytes directly; it does not JPEG-
  compress the file or export a screenshot of the on-screen thumbnail.
- The supplied task targets produce 1080 x 1350, 1600 x 400, and 728 x 90 PNGs.
  The last format really is 90 pixels high. Zooming it cannot create detail.
- The starter renderer defaults to a 2048px maximum dimension. The local viewer
  now passes its existing `max_size` option explicitly to preserve the source/
  target's native dimensions, including custom canvases larger than 2048px.
  Neither the official converter nor submission runner was changed. Their
  existing default preview cap still applies to the normal runner.
- Original raster assets also impose limits: the supplied gradient is 1200 x
  1500, product is 800 x 800, and logo is 900 x 300. Upsizing existing raster art
  does not reveal new detail. Text/shape rasterization is also finite at the
  requested output dimensions; this implementation does not invent extra pixels
  through AI enhancement.
- The lightbox displays the actual PNG dimensions and offers fit/100% views.
  Native export dimensions are regression-tested above and below the cap.
