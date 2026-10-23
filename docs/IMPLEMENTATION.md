# MagiCanvas implementation

## Current model path (October 22)

The engine now has an optional Qwen3-0.6B v2 LoRA proposal path, enabled with
`MAGICANVAS_QWEN_ENABLED=1`. `engine/qwen_refine.py` supplies compact source
lockups, joint column/peer-row constraints, measured-font alternatives, and
bounded local/contrast repair candidates. `engine/resize.py` compares those
candidates using feasibility, relative raster defects, and optional SigLIP
scores. The current SigLIP ranker is zero-shot; its fine-tuned preference head
has not been trained/installed. See [QWEN_AUDIT_FIXES.md](QWEN_AUDIT_FIXES.md) for
current behavior and limitations. The sections below describe the evolution
from the original October 16 deterministic baseline; its historical statements
about a future model decision are superseded by this paragraph.

## Scope and submission boundary

The scored engine is `engine/resize.py`. The existing function signature, runner,
JSON schema, assets, renderer, Dockerfile, input tasks, and container
entry point are preserved. The original Project Chameleon repo is unchanged.

The port adapts the prototype's analysis, grouped layout units, candidate search,
text fitting, and geometric ranking into Python. It does **not** copy its React
editor, internal v1 JSON schema, HTTP worker, AI provider integrations, prompts,
training data, model weights, or model dependencies.

## Approach

1. Infer backgrounds and coarse layout roles from element geometry, size, type,
   stacking order, and typography. IDs and asset names are not semantic labels.
2. Treat explicit `group_id` relationships as atomic units. Infer a relationship
   only for tightly contained text/backplate pairs. General proximity or alignment
   does not turn a whole design into one group.
3. Preserve artwork proportions with a common transform for the entire group.
   Simple backing/label units may widen together and refit their labels.
4. Classify the selected target into a format bucket (banner, wide, square,
   portrait or story), while retaining a separate near-source `preserve` signal.
   Generate a proportional preservation candidate and bucket-appropriate reflow
   candidates with different copy/art allocations and typography scales. Fonts
   can grow as well as shrink. Support multiple visuals and rows of badges.
5. Measure supplied local brand fonts with Pillow, using a wrap calculation based
   on the vendored renderer. Font/style lookup is cached. Unknown fonts have an
   approximate fallback. Actual rendering remains the judge of appearance.
6. Rank candidates by text fit, readability, unwanted collisions, in-bounds
   rotated geometry, composition, and use of space. Decorations remain visible;
   inferred full-bleed backgrounds fill the target canvas using existing pixels.

The ranker is a heuristic baseline, **not the official scorer or a trained
model**. It runs without a model or network. Model selection/training remains a
separate future decision once the organizers publish the approved list and size
limit; the supplied documents do not finalize those numbers.

## Constraint and energy pipeline (October 16 upgrade)

All scored logic remains in `engine/resize.py`; the public `resize()` signature,
runner, renderer and schema are unchanged. `kiwisolver==1.5.1` is the only new
runtime dependency. It installs at container build time and needs no model or
network service.

- Infer logo, hero/art, headline, kicker, subheading/body, CTA, badge, footer and
  decoration hypotheses from geometry and typography. Confidence is diagnostic,
  not calibrated probability. Repeated compact pills are distinguished from
  larger CTA backplates. Explicit artwork groups remain atomic.
- Compute target/source aspect-ratio delta and classify preservation, banner,
  landscape, square, portrait or story flow. Generate proportional, split,
  copy-first, hero-first, inline banner, semantic-stack and fallback-grid seeds.
  Alternative visual shares avoid committing to one guessed arrangement.
- Cassowary solves required combined raw/rotated-box bounds and separation constraints for
  each seed, with soft position/alignment preferences. Non-overlap is a
  disjunction: the seed chooses a separating axis/direction first. This is not
  a globally optimal mixed-integer solver. Infeasible topologies are rejected.
- Refine up to ten feasible starts using 36 bounded, reproducibly seeded
  annealing steps each. Moves translate units, scale whole artwork groups
  uniformly, or reflow standalone text with font fitting. Keep the best feasible
  state, so refinement cannot worsen that seed's energy. No per-frame group
  member jitter or independent deformation of an illustration.
- Energy weights: boundary 1500, overlap 1000, text overflow 500, canvas use 35,
  hierarchy 30, visual balance 18, source fidelity 18, alignment 16,
  readability 13, copy spacing 8. Terms are normalized heuristics, not learned
  design preferences or Adobe's scoring function.
- Render up to five candidates from distinct layout families with the existing renderer. A
  text-free *diagnostic* rendition measures backing contrast and busyness;
  comparison with the full rendition flags invisible text. Structurally
  different candidates inside a bounded energy band reach the visual ranker,
  rather than only near-identical geometry winners. Up to two bounded,
  role-aware refinement passes compare each variant with its parent using the
  original-image and relationship cues, stopping when the parent remains best.
  The returned document always contains every
  original element. Diagnostic renders are temporary.
- The raster check skips rotated text and canvases above 16 megapixels; if it
  cannot render, retain the best geometry candidate and report the reason.
  Colors, original assets and content are never rewritten to improve a score.

Deliberate differences from the proposed architecture: no LLM token generation,
no model downloads at runtime, no training, no SciPy, and no claim of an 80 ms
runtime. A lightweight SigLIP vision model is used only as an optional, offline
perceptual judge over already-valid candidate renders (see "SigLIP perceptual
re-rank" below). It sees the original render, a compact inferred
role/relationship summary, and each candidate; it never generates content or
alters geometry.
Seeded annealing suits discontinuous wrapping and collision topology;
Cassowary handles each linear placement problem. Intentional source artwork
layering and within-group overlap are permitted. Arbitrary overlapping text
panels or a malformed group still require better relationship inference; blanket
zero-IoU for every element would break legitimate illustrations and buttons.
Dense small banners can retain all content yet remain hard to read.

### SigLIP perceptual re-rank (optional learned tie-breaker)

Both the output JSON and the rendered PNG are scored, and the PNG is judged by an
AI evaluator. To better approximate that judgment than the deterministic
contrast/busyness heuristics alone, the 86M-parameter SigLIP 2 Base model
(`google/siglip2-base-patch16-224`) ranks the final, already-valid candidate
renders. The target bucket and estimated content density condition format-aware
image/text prompts; additional cues cover inferred logo, copy, CTA, artwork,
grouped-piece, badge-row, and corner-callout relationships. A
restrained image-embedding similarity to the original render encourages visual
identity continuity without requiring the adapted composition to copy the
source. It lives in `engine/vision.py` and is invoked from `_visual_rerank` /
`_siglip_rerank` in `engine/resize.py`.

- **Scope is deliberately narrow.** The model only ranks the bounded visual
  shortlist of up to five structurally distinct candidates that already passed
  every hard constraint (bounds, non-overlap, element preservation). It cannot
  add, remove, move, or restyle elements, and it never touches asset pixels — so the disqualifying
  "no regeneration" rule and the v0.9 contract are untouched. Its influence is
  bounded: SigLIP may only choose among candidates whose combined heuristic
  cost (layout energy + legibility/busyness cost) is within a finite band of the
  best. A small capped geometry penalty is combined with the model score, so
  a weak visual-score difference cannot overrule a much better-balanced layout.
  When the aspect-ratio shift is at least 1.6×, proportional scaling remains a
  fallback but is left out of visual ranking whenever a feasible reflow exists;
  large format changes should use the newly available canvas axis.
  Banner composition and its spacing/centering terms are active only in the
  banner bucket; regular landscape formats use editorial and split candidates
  without inheriting the compact-banner layout rules.
- **Offline by construction.** Weights are baked into the Docker image at *build*
  time by `scripts/fetch_siglip.py` into `models/siglip` and loaded with
  `local_files_only=True`; nothing is fetched at runtime, so `--network none`
  scoring is satisfied. CPU-only torch is installed from the PyTorch CPU wheel
  index in the Dockerfile to avoid the large CUDA build. Fine-tuning is not used,
  so there are no private adapter weights to publish; the base model is a public
  Hugging Face repo.
- **Graceful degradation.** If torch/transformers or the baked weights are
  absent, `engine/vision.py` returns `None` and the engine falls back to the
  deterministic ranking verbatim. The engine therefore still runs with no model
  present (as in the local test environment). Inference runs in CPU
  eval/inference mode, preserving the determinism contract.
- **Diagnostics.** The inspection report's `visual.perceptual` block records the
  per-candidate SigLIP and adjusted scores, the bounded candidate band, the selected index, and whether
  the learned judge overrode the pure-heuristic winner. None of this enters the
  official output schema.

If the organizers publish an approved-model list or a size limit, re-check SigLIP
against it before freezing a submission.


### Inspect a decision

```sh
python scripts/inspect_layout.py input/chameleon_sample_level3_advanced.json
```

Writes JSON, PNG and a diagnostics report in `output/inspection/`. The report
includes role hypotheses, aspect delta/flow, seed counts, energy before/after,
winning family, constraint status, raster-check findings and measured timings.
It does not add fields to the official output schema. This script is a local
team tool, outside the container entry point.

### Follow-up review hardening

See [docs/GEMINI_ENGINE_REVIEW.md](docs/GEMINI_ENGINE_REVIEW.md) for the assessment
of the supplied feedback and concrete execution findings. Placement now respects
both raw and rotated bounds, tries alternate separating topologies, and repairs
up to three rejected seeds for 60 iterations each. Repair explicitly penalizes
missing separation gaps; infeasible repairs cannot enter ranking. Implicit
backplate inference and elastic labels require a strict one-shape/one-text pair.
Optional raster diagnostics isolate ordinary renderer exceptions. No starter
runner or renderer behavior was changed.

### Typography, relationships and background cues

The subsequent perceptual pass adds line-balance scoring, continuous font
hierarchy compression, left-aligned split/banner copy candidates, conservative
repeated icon/caption affinities, central-void scoring, and optional background
busyness-guided candidates. These are soft heuristics, not semantic image
understanding. Quiet candidates repack copy and art in complementary regions;
text reflow preserves alignment anchors, and feature fitting uses a separate
size hint with proportional caption padding. Details, weights, preserved
properties and research caveats:
[docs/PERCEPTUAL_LAYOUT.md](docs/PERCEPTUAL_LAYOUT.md).

### Source alignment and visual hierarchy

A follow-up fixes discarded slot centering, adds source-centered axial
preferences, separates lower badge strips from CTAs and top-corner callouts,
and retains small footer images at their source corner. See
[docs/LEVEL3_LAYOUT_REVIEW.md](docs/LEVEL3_LAYOUT_REVIEW.md) for root causes,
weights, before/after observations and limits.

## Preserved properties and limits

Every ID appears exactly once, in source order. The engine deep-copies output,
leaves the input untouched, and preserves text content, font family, colors,
opacity, group IDs, asset references, style runs, and unrelated properties.
Geometry, font size, letter spacing, standalone copy alignment, corners,
strokes, and full-bleed background fit may adapt. Non-background images retain their source box aspect ratio.
Rotated bounds, not only unrotated rectangles, are considered when fitting.

The engine has no editor-specific minimum size or aspect-ratio cap. It supports
the official 728 x 90 target. Keeping all of the long-copy sample's content at
that resolution necessarily produces very small text. No text is silently
removed, rewritten, made transparent, or replaced. The viewer reports small or
unfitted text rather than claiming it is legible. Rich typography measurement is
conservative; unknown fonts and complex scripts need further work. This is an
initial deterministic port, not a guarantee of designer-quality results for all
unseen inputs.

## Run

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m engine.main
python -m unittest discover -s tests -v
```

Tests also validate the official schema when `jsonschema` is installed in the
development environment. It is not an engine dependency.

```sh
docker build --platform linux/amd64 -t magicanvas-engine .
docker run --rm --network none \
  -v "$PWD/input:/app/input:ro" -v "$PWD/output:/app/output" \
  magicanvas-engine
```

## Viewer requirements

The supplied documents disagree on stage-one viewer scope. The main brief,
sections 3.3 and 5.1, places the full Express add-on in the finale. The repo
`FAQ.md` Q3/Q5 describes a viewer add-on as required but unscored in stage one.
An ordinary web editor would not satisfy an Express add-on requirement.

Accordingly, `viewer/` is a small, separate development preview for official v0.9
JSON, and `viewer/express/` is a separate minimal Adobe Express viewer add-on.
They are outside the Dockerfile's COPY paths and do not participate in engine
execution or scoring. See `viewer/README.md` for startup and packaging. Confirm
the conflicting stage-one viewer wording with organizers before submission.

## Validation completed in this workspace

The October 16 comparison reports are in `output/optimizer-report.json` and
`output/optimizer-review.png` (generated, Git-ignored). All three winning layouts
passed the engine's hard placement constraints and used raster reranking.
Energy improved from its starting value in 8/10, 9/10 and 10/10 refinement
starts respectively. Warm local engine calls measured 0.61–0.77 s total,
including raster reranking; geometry measured 0.05–0.26 s. These are measurements
on this machine, not performance promises. The 728 × 90 sample remains dense
and small despite retaining all content.

- Six regression tests passed, including all three supplied designs across six
  target formats (18 sample/format combinations), plus synthetic grouped art,
  button fitting, long words, opaque properties, zero-size shapes, invalid
  targets, determinism, and input immutability. Schema validation was enabled.
- `python -m engine.main` rendered all three sample tasks with the real renderer
  and no structural self-check warnings. Source/adapted previews were inspected.
- The existing Dockerfile built for `linux/amd64`. A container run with
  `--network none` and read-only input produced all three JSON/PNG output pairs
  successfully. The October 16 rebuild includes Kiwi 1.5.1 while keeping the
  original Dockerfile and entry point. The offline runner reported 3.11 s,
  1.18 s and 1.07 s for the three `resize()` calls (including diagnostic reranking, excluding
  the runner's subsequent final PNG render).
- The local viewer API returned real source/output previews and rejected a
  malformed task. Browser checks confirmed sample selection, format selection,
  and committing a custom width with Enter.
- Adobe's tooling built and packaged `viewer/express/dist.zip`; its manifest and
  bundled files were checked. No certificate trust was changed and no add-on was
  sideloaded into a live Express account. Host insertion remains unverified.

The intentional Level 3 missing asset still renders as the starter's placeholder;
no replacement asset was added. Generated outputs, dependencies, and the Express
ZIP are ignored by Git; source files and the Express lockfile are available to
reproduce them.


## Starter instructions and port provenance

The starter's module instructions and complete `resize()` argument/return/rules
contract are restored verbatim in `engine/resize.py`. The complete original file
is archived below so even comments specific to the removed baseline remain
available. This archived code is reference material, not the active engine.

The original implementation was an intentionally naive baseline: it multiplied
positions, widths and heights by independent horizontal/vertical factors. Its
own comments explicitly required replacing that implementation, not merely
tuning its constants. Keeping that final stretch after reflow would distort
artwork and undo the layout. The current implementation preserves its public
signature, output shape, input non-mutation, element/property preservation, and
1px font-size floor. Its proportional fallback is a different, aspect-preserving
candidate; it does not reproduce the original anisotropic stretch.

Project Chameleon's reusable deterministic ideas were selectively translated
into Python, not merged file-for-file or copied in full:

| Prototype source | Current implementation |
| --- | --- |
| `src/engine/analysis.ts`: role/relationship inference | `_analyze`, explicit `group_id` units and conservative backing/label inference |
| `src/engine/layout.ts`: proportional/portrait/landscape candidates | `_proportional`, `_reflow`, `_visuals`, candidate generation inside `resize` |
| `src/engine/layout.ts`: wrapped text and linked label fitting | `_font`, `_advances`, `_text`, `_fit_text`, label branch of `_unit` |
| `src/engine/layout.ts`: grouped transforms and flow spacing | `_scale`, `_unit`, `_copy`, `_row` |
| `src/engine/score.ts`: candidate ranking | `_score`, using schema-v0.9 geometry and text properties |
| `src/engine/adapt.ts`: pipeline orchestration | `resize`, preserving the starter's exact public API |

The prototype's separate collision-repair pass was not ported verbatim; this
version generates separated layout slots and penalizes conflicts during ranking.
Chameleon-only roles, text-flow links, custom anchors, decorative-image flags,
editor state, and adaptation reports are not imported into the official format.
The official schema does not provide those custom annotations. Group IDs are
supported directly; further relationships are inferred internally. Local-model
inference/training/provider code is excluded as requested. This port therefore
reuses the deterministic approach, not every feature of the old engine.

### Original starter file (verbatim historical reference)

```python
"""
THIS IS THE FILE YOU IMPLEMENT.

Everything else in this repo (input loading, output writing, rendering,
the Dockerfile) is plumbing so you can focus on exactly one function:
`resize()`. See RULES.md and the design schema for the full contract.
"""

from typing import Any


def resize(source: dict[str, Any], target_canvas: dict[str, Any]) -> dict[str, Any]:
    """
    Args:
        source: {"canvas": {"width": int, "height": int}, "elements": [...]}
                 — the design at its original size. See
                 chameleon_design_schema_v0.9.json for the full element spec
                 (text / image / shape, position, size, rotation, opacity,
                 z_index, group_id, and type-specific fields).
        target_canvas: {"width": int, "height": int} — the shape you're
                 resizing into.

    Returns:
        A design dict with the SAME shape as `source`:
        {"canvas": target_canvas, "elements": [...]}

        Hard requirements (see RULES.md for the full list):
        - Every element `id` in `source` must appear exactly once in your
          output. Do not add, remove, or invent elements.
        - Do not regenerate or redraw any asset's pixel content — you are
          repositioning and resizing the elements you were given, nothing
          more. This is a disqualifying rule, not a style preference.
        - Non-background image elements should be kept in proportion
          (matching the source aspect ratio within ~5%) unless they are a
          full-bleed / `fit: "cover"` element, which is expected to crop.
        - `font_size` may change to help text fit. `font_family` and
          `content` should not, for this stage of scoring (see RULES.md,
          "Adaptive typography and copy").
        - Nothing should end up outside `target_canvas`'s bounds.

    This is where your actual approach goes: constraint solving, a learned
    layout model, heuristics, whatever you choose. Latency is not a scoring
    concern (see RULES.md) — spend your effort on quality, not speed.
    """

    # ------------------------------------------------------------------
    # Naive baseline so this file runs out of the box. This is a uniform
    # stretch — literally the thing the brief tells you NOT to do ("a
    # naive program stretches the picture; a good one moves the
    # furniture"). It's here so `python -m engine.main` produces valid,
    # schema-correct output on day one. Replace the body of this function;
    # do not just tune these three lines.
    # ------------------------------------------------------------------
    scale_x = target_canvas["width"] / source["canvas"]["width"]
    scale_y = target_canvas["height"] / source["canvas"]["height"]

    out_elements = []
    for el in source["elements"]:
        el = dict(el)  # shallow copy — don't mutate the input
        el["x"] = el["x"] * scale_x
        el["y"] = el["y"] * scale_y
        el["width"] = el["width"] * scale_x
        el["height"] = el["height"] * scale_y
        if el["type"] == "text":
            # font_size scaling isn't required to be uniform — this is
            # just a starting point. Floored at 1px: the schema requires
            # font_size >= 1, and an extreme downscale (e.g. a tall poster
            # squeezed into a short banner) can otherwise compute < 1.
            el["font_size"] = max(1.0, el["font_size"] * min(scale_x, scale_y))
        out_elements.append(el)

    return {
        "canvas": {"width": target_canvas["width"], "height": target_canvas["height"]},
        "elements": out_elements,
    }
```

## Wide-format and vision preprocessing follow-up

See [docs/WIDE_FORMAT_VERIFICATION.md](docs/WIDE_FORMAT_VERIFICATION.md) for
logo visibility, editorial copy/CTA columns, badge/hero hierarchy, bounded
callout sizing, square letterboxing, and reproducible five-format checks.
