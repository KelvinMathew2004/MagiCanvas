# Wide-format layout and vision-input follow-up

Implemented from the supplied five-point request, October 16, 2026. General
rules live in `engine/resize.py` and `engine/vision.py`; no sample ID, filename,
or catalogue-size lookup drives layout decisions.

## Changes

1. **Brand visibility.** A single small image can be recognized using source
   canvas area when there is no second image for comparison. The warmup logo
   was already caught by another header heuristic; its main failure was the
   tiny top-strip budget. Wide logo slots now use most of the column height,
   with proportional scaling capped at 1.8 × the current layout scale. A soft
   visibility cost discourages geometrically valid specks.
2. **Editorial columns.** Wide reflow variants and moderate-landscape split
   candidates keep source-upper tags with the copy rather than in the logo
   sidebar. The copy column packs logo where appropriate, kicker badges,
   heading/body, lower badges and CTA in reading order with a common left axis.
   Alternate hero sides and copy/art allocations remain candidates. Inline
   banners retain a separate CTA column. Group internals remain intact.
3. **Hierarchy and prominence.** Multiple secondary badges and a primary CTA
   no longer become one undifferentiated row. Badges use a subordinate scale
   and a separate strip. Hero prominence compares target and source area share;
   a target below 45% of the source share receives a soft penalty outside
   ultra-wide formats. Visual review showed the initial weight still selected
   a small-product stack over a valid split, so its weight was increased to 140.
   An avoidable-wrap cost (weight 25) discourages narrow multi-line headings
   when their current row has room for fewer lines. This prevents repeated box
   tightening from collapsing a short banner headline into three lines.
   Column alignment scores whole badge bands instead of incorrectly penalizing
   every badge for being to the right of the column's left edge. Its weight is
   120; brand visibility is 45.
4. **Corner callouts.** Callout scale uses the larger of short-side scaling and
   0.85 × area scaling, bounded by 32% of target width and 38% of target height.
   Its source top corner is retained with a small inset. Split candidates
   reserve actual callout clearance in the affected column; hard separation
   still rejects collisions. Tiny canvases can still have small callout text.
5. **Vision geometry.** `_letterbox_square()` preserves render proportions and
   pixels on a square RGB canvas padded with the sampled border median color.
   Only that diagnostic copy is passed to SigLIP's square processor. Original
   design JSON, assets and exported render geometry are unchanged. Candidate
   images share one batched forward pass instead of re-encoding identical
   prompts for each image. This avoids
   geometric squashing; it does not establish aesthetic-scoring accuracy.

## Reproduce verification

Host tests and model-free format matrix:

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/verify_layout_formats.py --output output/format-verification-host
```

Offline matrix with the actual downloaded vision model:

```sh
docker build --platform linux/amd64 -t my-chameleon-engine .
docker run --rm --platform linux/amd64 --network none \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 \
  -v "$PWD/input:/app/input:ro" \
  -v "$PWD/output:/app/output" \
  -v "$PWD/scripts:/app/scripts:ro" \
  --entrypoint python my-chameleon-engine -u \
  scripts/verify_layout_formats.py --require-vision \
  --output /app/output/format-verification-vision-final
```

The development verification script is mounted separately. The normal scored
container entry point remains `python -m engine.main` and does not require this
script or tests. The script checks all three supplied designs in Story, Square,
Widescreen, Web Banner and Leaderboard formats: exact IDs/order, fixed-property
preservation, image proportions, raw/rotated bounds, placement constraints, text
fit, source immutability, plus repeat-output determinism on the core banner.
It saves actual PNGs, official-format JSON and separate diagnostics/summary.

## Recorded checks

- 20 host regression tests passed, including nine new layout/preprocessing
  cases and the existing renderer-export checks. The model-absent test now
  explicitly simulates absence instead of depending on machine setup.
- All 15 host combinations passed, with repeat-output determinism confirmed.
- The public model's safetensors file is 812,672,320 bytes (about 813 MB), not
  the earlier README estimate of 370 MB. Loading and inference succeeded with
  networking disabled. Cold loading measured about 7.5 seconds in this
  linux/amd64 container on the local machine; this is not a latency guarantee.
- Final offline vision matrix: **15/15 passed**, repeat-output determinism
  confirmed, and SigLIP used in **13/15** cases. The two unchanged square cases
  had fewer than two competitive candidates and correctly skipped reranking.
  Results are recorded in
  `output/format-verification-vision-final/summary.json`.
  The earlier `output/format-verification-vision/` folder was an interrupted
  intermediate run before final ranking adjustments; do not use it as the
  verification summary.

Weights are baked at `/app/models/siglip` inside `my-chameleon-engine`.
They are not installed into the host `.venv` by Docker. The localhost viewer
uses the host environment; geometry changes apply there after restart, while
SigLIP requires local weights/dependencies or a container-backed viewer setup.

No custom model training, asset regeneration, content rewriting, renderer
modification or schema changes were performed.

Measured batched model-assisted resize times on this machine ranged from 7.99 to 11.4 seconds with two CPU threads in linux/amd64 Docker. These include geometry and rendered diagnostics, and are not a cross-machine latency promise.

The ordinary Docker entry point also completed all three supplied tasks offline, writing JSON/PNG pairs under `output/vision-container/` without structural warnings. Measured `resize()` times were 15.65 s for the first cold-model task, then 9.18 s and 8.77 s.
