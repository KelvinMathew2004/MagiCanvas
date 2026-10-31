# Docker rendering parity investigation — 2026-10-31

The README submission image was reproduced offline on the official level-three
728×90 task. It completed in 1534.81 seconds under emulated linux/amd64 on the
local Apple machine (pause time may be included in wall time). The adapter
safetensors SHA-256 is identical to the native champion:
`6ebbbe8d20e88504a1ecbb9fb6d5c9466df1c111bde84f099a6f788089d3c63b`.
The engine/resize.py hash also matches. The adapter is not absent from this image.

## Controlled evidence

A second Docker run replayed the exact saved native prediction, removing fresh
model generation from the comparison. Source and raw model elements match.
Native selected `deterministic-aspect-reflow-dense-short-banner`. Docker selected
`macro-band-reflow-3x1-0-left`. Native macro upgrade was not needed. Docker's
initial `dense-short-banner` had glyph_overflow, which activated the macro
fallback. Thus rendering/fit/gate differences can change the layout even when
model coordinates are frozen. Fresh CPU generation may add further differences.

All seven brand font-file hashes match. Source text measurements differ: the
wrapped subhead natural width is 828.498 natively versus 801 in Docker; the first
badge label is 120.645 versus 127. Native uses Skia 144.0.post2, Pillow 12.3.0,
Transformers 5.20.1; Docker uses Skia 144.1, Pillow 13.0.0, Transformers 5.22.1.
Requirements currently allow version drift. Platform rasterization also differs.

An isolated explicit-Skia-linear-metrics probe reduced advance differences to
less than 0.01 pixels for the eight source text elements. It changed no native
advances in this probe. However full frozen replay with linear metrics still
selected a Docker macro fallback; width parity alone does not establish glyph
raster, eligibility or final-output parity. The native one-case output stayed
unchanged. This prototype is NOT a validated fix and is not enabled in production.

## Decision

Do not patch layout templates or weaken clipping gates to match a screenshot.
Next isolate installed-renderer revision, Skia font metrics and pixel bounds
using identical frozen candidates, then validate any portable font policy in
both Linux and native environments before changing defaults. Pin the tested
runtime dependency set once established. Frozen engine replay is not generation
latency evidence and cached review images are not Docker deployment validation.

Evidence is local in `data/docker_reproduction_20261031/`: fresh Docker PNG/JSON,
frozen-prediction report, native/Docker text metrics, font hashes, linear probes,
linear end-to-end ablations and logs. Docker parity remains unresolved.

## Confirmed clipping cause and narrow fix

The initial Linux dense-banner candidate failed solely on `legal_text`: at font
size 3.2427 the renderer returned local ink top -1, while the larger source face
had a nonnegative top bearing. That is intrinsic hinted font geometry, not
canvas clipping. The text was inside the 90px canvas. Re-running calibration
could not remove it: translating the text box also translates the glyph, and
shrinking could preserve the negative bearing. The defect cost became 83.33,
forcing the macro fallback. Matching Pillow/Skia versions alone did not fix it.

`_source_ink_overhang` now measures a target-size, unconstrained, left-aligned
font probe when a new negative bearing appears. Only measured intrinsic
bearings are allowed. Center/right alignment overflow cannot become an
intrinsic allowance. Actual canvas overflow, right/bottom overflow and ink
collisions are still checked. No blanket pixel tolerance or task/content
special case was added; supplied renderer and runner remain unchanged.

Three focused tests exercise intrinsic top bearing, rejection of centered-frame
overflow, and preservation of canvas-overflow rejection. Layout mechanics plus
those tests passed (38 tests, 5 subtests); visual/copy-axis contracts plus those
tests passed (19 tests). The fixed Docker frozen-prediction replay now selects
`dense-short-banner`, with no flags, instead of the wrong macro fallback. A
fresh-generation Docker run and 59-case native regression replay are in progress.
Rebuild using the existing README command to include the engine change; no
adapter re-upload is required. Do not claim pixel-identical cross-platform
outputs or final deployment verification until the remaining checks finish.

### Regression refinement

The first broader prototype also allowed new intrinsic horizontal bearings.
It changed two supplemental Circulation layouts on native Mac. That horizontal
extension was removed: the final engine change only adds measured target-size
**vertical** bearing allowance. Horizontal source-bearing policy is unchanged.
An additional test locks that restriction. The final 59-case regression replay
and a rebuilt Linux frozen-prediction replay are running against this narrower
version; earlier prototype results are not final release evidence.
