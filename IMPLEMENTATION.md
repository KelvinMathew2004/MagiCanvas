# MagiCanvas implementation

## Scope and submission boundary

The scored engine is `engine/resize.py`. The existing function signature, runner,
JSON schema, assets, renderer, requirements, Dockerfile, input tasks, and container
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
4. Generate a proportional preservation candidate and portrait/landscape reflow
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

## Preserved properties and limits

Every ID appears exactly once, in source order. The engine deep-copies output,
leaves the input untouched, and preserves text content, font family, colors,
opacity, group IDs, asset references, style runs, and unrelated properties.
Geometry, font size, letter spacing, corners, strokes, and full-bleed background
fit may adapt. Non-background images retain their source box aspect ratio.
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

- Six regression tests passed, including all three supplied designs across six
  target formats (18 sample/format combinations), plus synthetic grouped art,
  button fitting, long words, opaque properties, zero-size shapes, invalid
  targets, determinism, and input immutability. Schema validation was enabled.
- `python -m engine.main` rendered all three sample tasks with the real renderer
  and no structural self-check warnings. Source/adapted previews were inspected.
- The existing Dockerfile built for `linux/amd64`. A container run with
  `--network none` and read-only input produced all three JSON/PNG output pairs
  successfully. The image uses the original requirements and packaging.
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
