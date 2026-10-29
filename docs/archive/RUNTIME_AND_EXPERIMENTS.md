> Historical record. Use [the current documentation map](../README.md) for setup and current status. Results below belong to their recorded snapshot.

# Runtime and experimental layout features

The production runtime detects usable acceleration: CUDA when available,
native Apple Metal (MPS) on supported Macs, otherwise CPU. Native MPS uses
FP16 and merges the adapter into the in-memory base model. Saved weights are
not changed. ARM64 Linux in Docker on a Mac does not expose Metal.
`MAGICANVAS_QWEN_MERGE_ADAPTER=0` disables the merge for comparison.

Eight fresh conversions on the available M1 Pro took 303.35 seconds combined
versus 331.74 seconds without merging (8.6% less); all eight final JSON outputs
were identical. This is not an M5 Pro or Docker measurement. The interactive
preview separately caches exact repeated requests, at most eight / 32 MiB.
Cache hits are not generation-latency evidence.

Source-derived macro retrieval with Hungarian assignment and alternate grouping
are available as experiments, disabled by default. The local derived layout
library is not distributed. The ten-feature human-preference ranker improved
aggregate validation, but only tied on the actual campaign subset, so it is
not deployed. Qwen3-0.6B remains the default; the prior 1.7B comparison did not
justify promotion. Model publication remains deferred.

## Expanded skyscraper testing

The unified viewer adds four supplied complex designs as explicitly disclosed
asset-backed test copies. Original missing assets and invented fonts are replaced
with local illustrative art and supplied Solstice fonts. Geometry and text are
preserved. The festival duplicate ID is renamed only in its derived fixture;
the original is still tested for rejection. These are user-authored stress
inputs, not confirmed hidden evaluation designs.

Each copy is tested at its requested target and 160×600. Four dense portrait
validation designs and an existing portrait alternate also target 160×600.
Selection uses source complexity and distinct dataset families, never output
quality. The fixed 38 and eight prior fresh conversions remain in the same
viewer. All 59 cases keep Qwen enabled and display Original and Resized only.

Rebuild local fixture assets with `scripts/prepare_user_design_stress.py`, then
run `scripts/evaluate_skyscraper_stress.py` for fresh model inference. The frozen
regression refresh reuses those predictions; it is not another inference run.
Assets, predictions, fonts and weights remain local. Review current residual
flags in the viewer and its summary; zero flags do not guarantee good design.

Competitor research documents and audit scripts are local-only and ignored.
Removing them from the index does not erase any earlier Git history.
