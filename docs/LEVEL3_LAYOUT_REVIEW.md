# Level 3 layout hierarchy and alignment review

October 16, 2026. The supplied feedback was checked against actual source
geometry and before/after renders in four formats. Fixes are general rules in
`engine/resize.py`; element IDs, asset filenames, sample names and target sizes
are not used to choose a layout.

## What was wrong and what changed

1. **Stack placement discarded horizontal centering.** `_visuals()` computed
   padding, but `_move()` then anchored its content bounds to a column's left
   edge. Stack placement now explicitly centers actual content extents within
   its slot; split artwork is centered within its own column. This was a direct
   seed-placement error, in addition to the feedback's optimizer diagnosis.
2. **Center-of-mass balance was insufficient.** Units centered in their source
   (logo, hero, headline, subheading/body and CTA) now receive a separate axial
   cost in vertical/square flows. Focal logo/hero horizontal optimizer proposals
   target the center axis instead of arbitrary text-box edges. Wide/asymmetric
   layouts retain their separate flow rules; arbitrary elements are not forced
   into a centered composition.
3. **Every tag was promoted to the header.** Promotion now requires source
   placement in the upper half. Source-lower tags remain in the lower action
   zone across reflow candidates. In vertical/square formats, badges and CTA
   use distinct rows, with the badge strip above the CTA. Banner candidates keep
   lower tags below copy or in the lower action band instead of above headlines.
4. **Corner ribbons were mixed with badge peers.** Rotated labels in a source
   top corner become separate corner callouts. They retain a common group
   transform, stay in their top corner, and are separated from main content by
   the hard placement gate. They do not contribute balancing mass that would
   pull a centered hero away from its axis.
5. **Peer detection ignored typography and rotation.** Badge peers now require
   similar source height, font size, y-band and rotation. Verified strips share
   a row-fitting scale; soft costs preserve their order, row alignment, spacing
   and relationship to the CTA. No output `group_id` is invented. A row can still
   wrap or shrink when the available space genuinely cannot accommodate it.
6. **Footer images lost their corner relationship.** Small source-corner footer
   images carry a left/right anchor. Footer packing separates them from centered
   legal copy; optimization retains the edge anchor. An additional extent bug
   used a row's right coordinate as its width, leaving corner items too far
   inward. Packing now uses `right - left`. The intentionally missing asset
   remains the renderer's placeholder; its pixels/file were not replaced.

New soft energy weights: axial alignment 160, badge flow 45, footer anchor 100.
These are hand-tuned preferences, not learned weights or Adobe's evaluator.
Hard bounds, required separation, image proportions and explicit groups remain
part of candidate validation. Existing labels retain their source fill colors;
color alone is not treated as proof that an element is the primary CTA.

## Rendered observations

Generated JSON, PNG and diagnostics are in `output/level3-review/before/` and
`output/level3-review/after/` (ignored by Git). The four target formats are used
for evaluation only, not for engine lookups.

- Story: the hero center moved from approximately 27.5% of canvas width to 50%;
  the logo moved from 28.6% to 50%.
- Square: the hero center moved from approximately 15.5% to 50%; the logo moved
  from 23.6% to 50%.
- Source-lower badges now appear together above the CTA in story/square; the
  rotated callout remains at the top rather than joining their strip.
- The missing-asset image's right edge is at the right content margin rather
  than below the CTA at the canvas center.
- All four final candidates passed hard placement checks. The ordinary runner
  also rendered all three official tasks without structural warnings.

These observations address the reported failures. They do not establish
professional quality on every unseen design, and very dense leaderboard copy
remains small. The cited research's guarantee that four changes will produce
"designer-grade" outputs is not supported by these local examples alone.
