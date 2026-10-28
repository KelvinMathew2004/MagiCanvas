# Engine upgrade checklist — reviewed October 27, 2026

Latest October 28 work: [runtime, experimental features and skyscraper testing](RUNTIME_AND_EXPERIMENTS.md). The unified viewer retains the fixed 38, eight fresh independent conversions, and adds 13 supplied/skyscraper stress conversions. Historical results below retain their original scope.
Earlier dated results below retain their original scope.

Source: the supplied “Detailed Analysis of the Latest Code & 38-Case Regression
Suite” review. Items below distinguish verified existing work, changes to
implement, experiments, and actions requiring human labels or publication.
No design IDs or reference target geometry may drive engine behavior.

## Evidence and research

- [x] Check every recommendation against current code rather than assuming it is absent.
- [x] Confirm Qwen already uses native MPS/FP16; Docker already enables Qwen.
- [x] Confirm token budgets are element/token-aware (up to 4096), and missing-ID
  recovery already accepts limited omissions. Seven/eight incomplete raw ID
  inventories do **not** prove all seven were rejected before selection.
- [x] Confirm peer spacing runs immediately before final rasterization, not only
  as an optional candidate. Recheck actual gaps before altering this path.
- [x] Confirm deprecated Pillow `getdata()` calls are already removed. Remaining
  Python per-pixel work is a separate performance opportunity.
- [x] Research hardware/dtype and renderer thread-safety before changing defaults.
- [x] Profile current engine phases and compare measured timings; do not promise
  the review's unmeasured 6–12s, 45%, 5×–8×, or 60% improvements.

## Learned proposals and completion

- [x] Run **fresh** paired generation on eight dense alternates plus twelve
  complex dataset cases, with the same prompt contract and sufficient output
  budget, `repetition_penalty=1.0`, and Qwen enabled in both arms.
- [x] Record raw completion, exact ID inventory, missing IDs, prompt/output
  tokens, device/dtype, generation time, and repaired/final defects separately.
- [x] Strengthen omission completion: accept at least 85% measured source ID
  coverage and at most three omissions; use an evidenced parent/peer first,
  then deterministic free-space placement for unattached elements. Preserve raw
  invalidity in reporting and identify the repaired candidate's provenance.
- [x] Compare both fresh model arms through the same final engine/selector.
  Keep the existing default unless the measured promotion gate passes.
- [x] Conditional only: prepare targeted dense-example SFT continuation **if**
  fresh evidence supports it. Preserve response-only loss, target uniqueness,
  family-disjoint evaluation, and v2 compatibility. Do not train on this review
  cohort or blindly duplicate capped targets to oversample examples.

## General container realization

- [x] Verify final peer-row gaps and retain the existing safe final equalization
  path. Avoid adding redundant early leaf mutations that could undo ownership.
- [x] Offer gated repeated-sibling row/grid/column layouts within the allocated
  macro box; choose by achievable scale, preserve atomic attachments, source
  order and spacing. No unconditional universal 12px floor: it is infeasible
  for legal copy and small 90px/160px formats.
- [x] Preserve source-attached/macro ownership, aspect ratios, intentional
  layering and existing candidate-family quotas.

## Performance without hiding defects

- [x] Add shared capability-aware device/dtype/thread policy and diagnostics.
  Keep safe CPU dtype defaults and opt-in overrides; do not force CPU BF16 or
  assume Docker can use the host's Metal GPU.
- [x] Stop on syntactically closed JSON; preserve genuine missing-ID metadata.
  Retain token-aware headroom instead of the proposed conflicting 1400 cap.
- [x] Vectorize glyph-mask contrast, exact order-statistic percentiles, luma
  roughness and edge density with NumPy while preserving metric semantics.
- [x] Cache text-free backing renders by the complete non-text render signature.
- [x] Reuse **exact** duplicate rendered documents while keeping all candidate
  names/family eligibility. Reject approximate 0.5%-geometry deduplication:
  it can merge visibly different fonts, tracking, colors, crop, layer order or
  the small axis corrections this project is trying to preserve.
- [x] Add isolated, bounded render concurrency only after checking renderer
  ownership. Never concurrently mutate a shared SkCanvas/SkSurface or model.
  Keep deterministic ordering and serial fallback; benchmark before promotion.
- [x] Retain final glyph/geometry safety checks; existing diagnostics already
  render at max 640px, so do not silently lower resolution to hide small flaws.

## Preference head (human review required)

- [x] Prepare a same-renderer preference page with explicit choices and export
  of real labels. Include flag-free tiny-content/balanced-reflow pairs.
- [x] Verify renderer compatibility/SSIM provenance and connected-family split.
- [x] Import all 168 campaign choices, retaining the unchanged first 44 and
  verifying image hashes/side mappings. Six of eight consistency repeats agree;
  quarantine two ambiguous originals without inventing replacement votes.
  Seven unsafe preferred layouts remain diagnostic-only. Combined eligible
  corpus: 178 pairs, 64 family components (45 train, seven validation, 12 test).
- [x] Complete the 160-comparison queue plus eight disclosed repeats: 56 new
  Level 3-style comparisons, 104 dataset comparisons, all six formats, Qwen on.
- [x] Bind feature caches to image/model bytes and use family-disjoint folds.
  Test embeddings/pair scores are deferred until validation passes. No pilot
  consumes reserved test comparisons or writes under models/.
- [x] Train the complete reviewed corpus with the predeclared head configuration:
  122 train pairs, 17 validation pairs; best epoch seven. Validation is 9/17
  for both combined head and zero-shot baseline. No head saved or deployed;
  the 39 test pairs were not scored because validation failed.
- [ ] **Gated:** a ranking head must beat zero-shot on validation AND independent
  test, then pass engine regression. Collecting more arbitrary pairs alone is
  not established as useful; investigate native detail/features/candidate limits.
- [x] Fix generic growth of source-attached backing panels: retain their shared
  media edge and grow away from it only while the target still has that contact
  and there is safe canvas room. Detached reflows/intentional overlays unchanged.
  Two labelled widescreen outliers now select a cleaner larger reflow; the two
  leaderboard cases remain flagged for control readability/padding.
- [x] Regression after attachment fix: no winner flips/new flags; 36/38 pixels
  identical, two caption shifts <0.35 px visually inspected; 311 tests pass.
- [x] Add bounded, request-owned asset lookup memoization after profiling;
  verify parallel-request isolation, error reset and newly available assets.
  Frozen 38-case winners, JSON geometry and pixels remain identical.

## Submission and release

- [x] Prepare a pinned, adapter-required offline Docker path with local artifact
  verification. Keep adapter-free development mode explicit and separate.
- [x] Verify base/adapter compatibility, prompt marker and hashes; fail clearly
  rather than silently using the base model in the adapter-required build.
- [ ] **Waiting for user repository/publication:** publish winning adapter and
  pin its public commit. User selected local preparation, no uploads now.
- [x] Update README with local build, offline runtime, limits and measured scope.
- [x] Run tests and refresh the fixed 38-case Qwen-enabled Original/Resized viewer;
  inspect changed winners and newly flagged outputs against the frozen baseline.
- [x] Update full/compact Gemini packets with checklist, code, images, timings,
  A/B evidence and remaining limitations. No full 89/309 run claimed unless run.

## Primary research and decisions

- [PyTorch MPS notes](https://docs.pytorch.org/docs/main/notes/mps.html): moving
  models and tensors to an available MPS device is required; Linux Docker does
  not inherit a macOS Metal device.
- [PyTorch CPU BF16](https://pytorch.org/blog/empowering-pytorch-on-intel-xeon-scalable-processors-with-bfloat16/):
  acceleration depends on supported CPU kernels/ISA; not a portable blanket win.
- [PyTorch thread configuration](https://docs.pytorch.org/docs/stable/generated/torch.set_num_threads.html):
  thread settings must precede computation and account for concurrent workers.
- [Skia maintainers on surfaces/canvases](https://groups.google.com/g/skia-discuss/c/yqF1rwi98pg):
  SkSurface/SkCanvas are not shared mutable thread-safe objects. GIL release alone
  does not establish safety of this renderer's Python caches or FontManager.

The review's hardware speed estimates, “all failed because IDs were dropped”
claim, missing-final-peer-pass claim, and deprecated-API count are hypotheses
or outdated observations, not accepted measurements. Their underlying goals
are retained above and will be tested.

## Current measured disposition

Fresh 20-case A/B completed: exact raw IDs 12/20 (0.6B) vs 20/20 (1.7B);
learned-family winners 5/20 in both; final dataset geometry MAE 0.03780 vs
0.03556, final font MAE 0.10578 vs 0.12191. The narrow geometry/ID gate
passes, the additional font non-regression gate fails. No default promotion
or Qwen retraining. The 38-case frozen regression retains identical geometry,
winners and pixels in all 38 cases, with no new automated flags. All four
realized source-evidenced horizontal peer rows have equal clear gaps to
floating-point precision (maximum gap difference <2e-13 px).

296 tests and 34 subtests passed. Docker artifact verification passed locally;
the Docker daemon is unavailable, so a build/offline container smoke test is
still pending. A public repository and pinned commit are also still pending.
Full 89/309 release sweeps remain future checks, not completed items here.

## October 27 follow-up: dense container packing

The current decision record is [DENSE_REFLOW_AUDIT.md](DENSE_REFLOW_AUDIT.md).
Earlier counts/results above retain their historical snapshot scope.

- [x] Add measured text/native-backing and bounded contiguous-column packing.
- [x] Reorient source-adjacent photo/caption compounds without new overlaps.
- [x] Preserve peer membership and offer wrapped control rows/grids.
- [x] Add separately audited Qwen-driven macro columns and tall/square bands.
- [x] Allow only a measured readability-only fallback, retaining its flags;
  bounds, inventory, clipping, overlap, padding and contrast remain gates.
- [x] Guard unresolved decorations and atomic source line counts after the
  broader human-rejected replay exposed visually worse low-cost alternatives.
- [x] Leave model/head training and adapter publication unchanged/deferred.
- [x] Record final guarded-code regression, 20-case arms, four attachment tasks
  and deduplicated human-rejected task results; inspect changed outputs.
- [x] Refresh canonical viewer, preview process and compact Gemini packet.

Final guarded-code verification: 321 tests / 34 subtests; 38/38 frozen pixel
buffers identical, zero winner flips/new flags. Both diagnosed leaderboards
improve (8.0%→97.5% width use; 1.125→5.4px median type). Fifteen deduplicated
human-rejected tasks were replayed; many remain aesthetically weak. The two
saved 20-case model arms keep their previous metrics and five learned-family
wins each. No learned macro candidate won; no model/head promotion or retraining.
