# Engine upgrade implementation and evidence — October 27, 2026

Work is tracked in ENGINE_UPGRADE_CHECKLIST.md. This document records actual
measurements rather than the supplied review's speed estimates.

## Accepted changes

- Capability-aware Qwen runtime selection, explicit dtype/device/thread knobs,
  CUDA support, MPS disable override, and safe CPU precision defaults. Qwen was
  already MPS-enabled; accelerated vision remains opt-in for reproducibility.
- Syntactically closed JSON stops generation even when IDs are missing. It is
  still invalid as a raw inventory; no repaired output is credited to the model.
  Existing token-aware budgets are retained; repetition penalty is explicitly 1.
- Bounded omission completion: >=85% supplied IDs, at most three missing; source
  parent/group first, free rectangles otherwise. Only original model anchors
  participate, so recovered guesses cannot form an anchor chain. A distinctly
  named completed candidate is evaluated by the normal renderer/selector.
- Repeated sibling row/grid/column candidates inside the proposed macro box,
  preserving atomic cards and internal source offsets. A shared source
  typographic scale and repeated structure are required; headline/body/footer
  text is not treated as interchangeable menu items. No universal 12px minimum.
- Exact full-document/backing raster reuse, with candidate family identities
  retained. Signatures include canvas, style, tracking, color, layer order,
  assets and diagnostic mode, not just approximate geometry.
- NumPy glyph statistics preserve the old kth-order quantile, alpha and mask
  semantics. Source and candidate measurements use the same vectorized path.
  The historical selector gamma is 2.0, not standards-based WCAG 2.4. It is
  deliberately unchanged in this performance pass; a gamma migration needs a
  separately calibrated source-floor/contrast ablation.
- Optional process-isolated render concurrency; no shared renderer/canvas
  threads. Defaults remain serial, with fallback and deterministic result order.
- Fail-closed adapter-required submission Docker stage plus a verified local
  adapter context. Development base-only target is explicit. No public upload.

## Measured renderer microbenchmark

Same sixteen requested documents, actual PNG pixel hashes checked across modes:

| Mode | Wall time | Unique raster calls | Cache reuse |
|---|---:|---:|---:|
| Serial cache | 1.18s | 7 | 9/16 |
| Two isolated processes | 1.97s | 7 | 9/16 |

All output pixels matched. This small render-only workload was measured while
other development work was active, so it is not an end-to-end speed claim or
proof of optimal CPU worker count. The two-process startup cost lost here;
default concurrency stays one. `render_seconds` in cache diagnostics is an
aggregate per-request/queue latency; use benchmark `wall_seconds` for timing
comparisons. More representative end-to-end profiling is recorded separately.

## Human preference preparation

Sixty training-family comparisons are prepared. Forty-six pass preview/renderer
SSIM and text-region gates, fourteen remain review-only. None overlap the
withheld test/validation IDs checked during this audit. The user subsequently supplied 60 real choices, all hash-verified; 46 meet
fidelity gates, while 14 remain review-only. No ranker weights were trained. Both images use the
same local renderer. The candidate is a controlled foreground shrink, explicitly
labelled as a synthetic comparison rather than a fresh engine output. This is a
useful initial batch, not a complete distribution of alignment/attachment flaws.

Review `data/siglip_human_review/review.html`, export real JSONL preferences, and
pass them explicitly to train_siglip_ranker.py. It supports either preference,
excludes ties, verifies image hashes, splits connected families, and preserves
the incumbent unless its held-out accuracy promotion gate passes.

## Release preparation and limits

The local 0.6B adapter was checked against the base config and all 28 q-projection
LoRA input widths. The Docker daemon is unavailable (`docker.sock` missing), so
no image build or offline container execution is claimed. The generated ignored
context is `data/submission_context/`. It includes weights for local testing and
must not be added to Git. Final public adapter distribution is still pending
user repository/publication, as requested.

A conditional dense-continuation curriculum was validated over 1,122 existing
unique-target training pairs; 71 have 15–25 elements. Sampling weights do not
create duplicate target records. This prepares an option, not a training run;
no solver-generated labels or evaluation examples were added to training.

## Fresh model comparison and promotion decision

Same 20 source/target tasks: eight authored alternates and twelve dataset pairs.
Both use prompt v2, sufficient token-aware budgets, repetition penalty 1,
deterministic generation, MPS/FP16, and the same final engine/selector. Actual
raw predictions are retained before recovery. The 1.7B run used a later
qwen_seed.py hash containing only a guarded adapter-required loading check;
that mode was off for both experiments. Generation settings and inference
behavior otherwise match. Artifacts retain both code hashes rather than
pretending the entire files were identical.

| Measurement | 0.6B | 1.7B |
|---|---:|---:|
| Decoder-complete JSON | 20/20 | 20/20 |
| Exact raw ID inventory | 12/20 | 20/20 |
| Exact raw IDs on eight dense alternates | 1/8 | 8/8 |
| Final learned-family winners | 5/20 | 5/20 |
| Final automatically flagged outputs | 0/20 | 0/20 |
| Final dataset geometry MAE (12 pairs) | 0.03780 | 0.03556 |
| Final dataset relative font MAE | 0.10578 | 0.12191 |
| Raw dataset relative font MAE | 0.46186 | 0.11609 |
| Total measured generation seconds (20 tasks) | 447.22 | 819.74 |

JSON completion follows the existing decoder's delimiter-normalization behavior;
this run did not archive literal response strings for an independent syntax
audit. Exact raw ID inventories use the saved model-elements snapshot before
any omitted-ID recovery. No recovered ID is counted as a model prediction.

The narrow ID/geometry/flag/win-rate development gate passes, but final font
error regresses ~15.2%. Therefore the added font non-regression gate fails and
**0.6B remains the default**. Four winner names change and six output documents
change across model arms. One of eight authored layouts becomes a learned
winner under 1.7B; seven still select deterministic layouts despite complete
IDs. Missing IDs alone did not explain the weak novel-design results.

The six changed outputs were inspected. 1.7B gives a stronger, larger hero in
After Dark/story and more coherent placement in Origami. Other results remain
mixed: Drinkware's bottom rules stay detached from the copy, and the reverse
Rome resize uses a smaller lockup. Minor artwork/source-supported overlaps
persist in the pillow design. Neither zero flags nor lower average geometry
MAE establishes aesthetic perfection. No production VLM was added.

## Regression, gaps and tests

All 38 cases were replayed against frozen prior-engine outputs with exactly
the same saved model proposals and Qwen enabled. Every selected winner,
geometry and final PNG pixel buffer is identical (38/38). This verifies that
the new optional candidate paths and vectorization did not change the frozen
incumbents; it does not demonstrate that their remaining visual flaws improved.
Four realized source-evidenced horizontal peer rows were measured after final
control padding. Maximum clear-gap asymmetry is 1.71e-13 px. No redundant peer
repair was added in response to the review's outdated missing-pass claim.

Full suite: **296 passed, 34 subtests passed**. There is no new 89/309 release
sweep, unseen-family quality claim, adapter promotion or Qwen training run.

## Performance conclusion

A cold, profiled Level 3 leaderboard replay took 35.12s, including profiler and
import overhead, excluding fresh token generation. Geometry polishing took
7.95s cumulatively, objective scoring 6.84s and visual reranking 4.70s; these
nested cumulative timings must not be summed. Repeated asset-bound lookup and
geometry cost evaluation are plausible next optimization targets. Rendering
is not established as the dominant bottleneck. The speculative 6–12s end-to-end
claim is unsupported. Safe exact reuse/vectorization is implemented; process
render parallelism stays opt-in because it lost the measured microbenchmark.

## Preference correction following user review

The user correctly noted that every original batch A image was the local human
target and every B image a controlled shrink. Those labels measure occupancy,
not general design quality. They remain preserved in approvals_occupancy.jsonl,
with 60 matching hashes and 46 eligible pairs. The replacement batch has 21 comparisons across seven training families,
prepared from eight actual Qwen-enabled source runs (one yielded no distinct
safe alternatives). It compares actual generated realizations, including
winners and differing alternatives; no test/validation template IDs overlap.
Neither side is a human reference. A/B positions are randomized and ties/neither
are supported. All 21 have no renderer warnings. A selected realization still
can contain flaws: the page requests actual human preferences, not approvals of
the engine’s automated judgment. Latest skip decisions revoke earlier approvals,
and training rejects stale image hashes. No approvals are
synthesized, no evaluation examples are used as training labels, and no head
is promoted from the narrow occupancy-only batch.

