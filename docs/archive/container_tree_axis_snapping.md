> Historical record. Use [the current documentation map](../README.md) for setup and current status. Results below belong to their recorded snapshot.

# Container trees, source axes, and rendered font calibration

The engine treats a model proposal as a set of proposed functional container placements. Source geometry supplies the internal layout. This replaces the active ornament-specific alignment candidate family and the post-selection leaf patches. The exact candidate geometry rendered by the verifier is delivered without a later alignment rewrite.

## 1. Functional container projection

`engine/qwen_refine.py::build_container_tree` builds a canvas root, functional containers, nested backing containers, and atomic element leaves. It reconstructs the source prompt's geometric same-column, same-row, backing, and overlap relationships without the prompt's token-budget truncation. Explicit `group_id` is strong evidence. Nearby shared source axes provide additional coupling evidence. A full-canvas background cannot connect every foreground element into a giant group.

Crossing memberships are merged; strict subsets stay nested. Unrelated source regions remain separate roots. The tree and its inferred relationships are included in the Qwen diagnostics, rather than added to the output design schema.

`project_container_tree` infers each root's proposed translation and scale from robust weighted medians of substantial source/proposal correspondences. Thin geometry does not estimate the transform. Every child receives the common source-to-target transform, including its source offsets, rotation, text alignment, font hierarchy, and proportional tracking. The resulting container box is recorded in diagnostics.

Images always scale uniformly. Native shape backplates can absorb bounded extra horizontal room while the internal copy remains rigid. This is deliberately a limited semi-elastic mode; it does not distort a raster frame, stretch an image, invent a new font, or rewrite copy. A single source-centered root inherits the canvas parent axis. Boundaries are handled for the complete root rather than clamping each child separately. Deliberate single-edge source crops on axis-aligned raster corner containers retain their crop fractions through the same shared transform.

Candidate families can still offer different top-level reflows. Every final family passes through the shared structural projection before visual ranking. Raw model proposals remain diagnostic evidence; they cannot bypass the structural passes by winning a perceptual tie-break. If no projected candidate is feasible, the engine retains a complete projected layout with an explicit fallback reason instead of silently returning fragmented raw coordinates.

## 2. Universal source-axis clustering

`snap_source_axes` clusters X-left, X-center, X-right, Y-top, Y-center, and Y-bottom. A cluster's source-axis diameter must be at most 4 source pixels, and its proposed target-axis diameter must be at most 3% of the relevant target canvas dimension. These are engine thresholds, not claims about proprietary products or research implementations.

Clusters use bounded diameter, not transitive neighbor chaining. Elements on a clearly different reflowed row or column do not snap. An accepted target axis is the exact median of the proposed axes. Competing left/center/right constraints are resolved deterministically by support, source spread, and authored text alignment; a later conflicting constraint cannot repeatedly move an already assigned element. Rotated items can share a center axis, but their unrotated box edges are not treated as visible alignment edges.

Each snap must remain on canvas, preserve positive source padding within inferred backings, and avoid introducing unintended rendered text overlap. A rejected or incompatible constraint is not advertised as a successful snap. This is a geometric constraint system, not a guarantee that every inferred group or every imperfect source design has a unique correct alignment.

## 3. Renderer-based font and box calibration

The font lookup combines the optional local `data/crello_fonts.pickle` with `brand_kit/fonts/`, with brand faces taking priority for matching family names. No font download is required. Existing Crello aliases are retained. The actual Skia converter and text renderer are used for pixel-bound probes, including styled text, tracking, glyph overhang, and rotated visible bounds.

The renderer has two different measurement stages: its converter wraps using fractional-size glyph advances, while its text renderer draws at integer font sizes. A safe width must satisfy both stages. The layout measurement now accounts for both; a narrower drawn width cannot conceal an earlier wrap decision.

For a tight text frame, calibration attempts, in order:

1. Widen the frame into empty space while preserving its authored left/center/right anchor.
2. Expand available height if rendered glyphs require it.
3. Reduce font size only when the available space remains insufficient.

The smallest source backing limits expansion; unrelated neighboring content is an obstacle. Intentional source overlap is not treated as empty space. Every font size adjustment scales tracking by `new_font / old_font`, including negative tracking. The renderer's pixel bounds are used for final overlap checks so touching text frames are not automatically mistaken for touching glyphs. A local font/renderer failure is exposed as a metric fallback in the calibration notes; it is not claimed to provide exact renderer parity.

## Verification and limits

The V65 viewer compares the dataset preview, local target JSON render, and final Qwen-off/on results for the same eight regression designs. The engine does not read paired target JSON while projecting or selecting a layout. Paired targets are evaluation evidence only.

The focused suite covers source immutability, background exclusion, nested backing membership, rigid transforms under thin-rule outliers, whole-container canvas bounds, proportional image scaling, rotated source typography, exact median snapping, rejection of distant/reflowed axes, non-transitive clusters, shape elasticity, widening before shrinking, negative tracking, and fractional-wrap/integer-draw behavior.

The broader layout mechanics suite currently has three known baseline problems: dense leaderboard visible gutter balance, peer button row repair, and subheading role detection. These were reproduced with the V64 code snapshot before this structural upgrade; they are recorded separately from the focused structural suite. Do not report the entire repository suite as passing.

The productivity signature remains source-authored two-color text over a black/white wave asset. Structure and glyph fitting do not repair poor source/background contrast. Group inference also cannot recover semantic grouping with certainty from geometry alone; explicit source groups remain stronger evidence than inferred proximity.

## What the cited systems actually establish

- [O'Donovan, Agarwala, and Hertzmann, Learning Layouts for Single-Page Graphic Designs (2014)](https://research.adobe.com/publication/learning-layouts-for-single-page-graphic-designs/) presents a learned energy-based layout model. It supports reasoning about design relationships and optimization; it does not establish that every professional engine always reflows exactly five or six rigid containers.
- [Adobe XD responsive resize and constraints](https://helpx.adobe.com/uk/xd/desktop/design/layout/using-responsive-resize.html) documents inference from grouping, layout, and parent-edge proximity. [XD group API documentation](https://developer.adobe.com/xd/uxp/develop/reference/group) documents responsive and content-aware group behavior. These are public XD mechanisms, not proof of Adobe Express's private implementation.
- [Canva Design Editing API](https://www.canva.dev/docs/apps/design-editing/) and [positioning documentation](https://www.canva.dev/docs/apps/positioning-elements/) expose a document tree, groups, and box geometry. Their public documentation does not disclose the complete automatic resize optimizer or a universal snapping percentage.
- [PLAYGROUND (October 2026)](https://arxiv.org/abs/2610.18865) uses progressive layout generation with render-grounded planning. It supports rendered feedback during generation rather than relying on a single flat prediction.
- [DesignAsCode (February 2026; revised August 2026)](https://arxiv.org/abs/2602.17690) uses semantic hierarchy, HTML/CSS implementation, and visual reflection. It supports hierarchical planning and iterative verification, without prescribing this engine's exact two-level grouping or 4-pixel/3% snapping constants.

## Follow-up deletion audit

[The deletion audit](engine_deletion_audit.md) found and restored two omitted protections: source backing padding during snapping and deliberate corner crop fractions. The follow-up focused suite has 58 passing tests. The controlled replay evidence is distinct from the original V65 full model-and-ranker render.
