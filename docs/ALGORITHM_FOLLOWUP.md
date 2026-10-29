# Training-free follow-up: intact growth and rendering work

October 29, 2026. Preregistered settings are in
`config/algorithm_followup.json`. The completed comparison covers all **134 distinct
requests**, covering the current 59-case development suite plus all 89 reference
pairs (83 unique requests). These are consumed development cohorts, not unseen
validation. Qwen remains enabled with the same stored decoded predictions.
No challenger is automatically activated.

| Arm | Distinction from earlier experiments | Required evidence |
|---|---|---|
| Shared-scale container growth | Earlier L-BFGS polish translated blocks; this LP also enlarges all intact blocks by one shared factor, up to 1.25, with at most 5% translation per axis | No new defects, intact children and aspects, native contrast checks, better source-only proxy, review every replacement |
| Family-quota render funnel | Reduce the input pool to 12 candidates while reserving raw/repaired learned paths and independent deterministic families; use true geometric defects before rasterization | Render savings plus unchanged winner/quality and family coverage; no claim that geometry alone predicts contrast |
| Exact order-statistic selection | Replace full sorting with selection of the same minimum/5th/20th-percentile ranks | Exact values on every live glyph patch, paired timing and unchanged outputs; retain sorting if measured benefit is absent |

The growth experiment first maximizes shared scale with a linear program, then
minimizes absolute translations at that scale. Each source-owned container is a
rigid variable: font size and tracking scale together, image aspect and child
offsets stay coherent. Existing non-overlap directions, confirmed source/target
axes and existing edge attachments constrain the solve. This retains topology;
it does **not** invent a better row/column arrangement or establish that the
inferred source tree is correct. Native rendered checks remain necessary.
[SciPy linear-programming contract](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html).

The funnel is a separate speed hypothesis. Its geometry score excludes target
reference coordinates, contrast measurements and heuristic template similarity.
An unrendered clean-looking box may have poor glyph contrast; quotas cannot
guarantee winner recall. Both arms retain full source/intermediate records and
measure actual render counts. Replay times include instrumentation and concurrent
work, so they are not isolated M5 Pro or Docker generation benchmarks.

The statistics experiment uses NumPy's selection operation: requested ranks have
the same value as a fully sorted array, while other positions need not be sorted.
No percentile interpolation, gamma change, threshold change or source-floor
recalibration is involved. Full statistic calls alternate execution order on the
same real glyph patches; they return the existing implementation to avoid
changing the baseline during profiling.
[NumPy partition semantics](https://numpy.org/doc/stable/reference/generated/numpy.partition.html).

## Reproduce

```bash
.venv/bin/python -m pytest -q tests/test_scale_followup.py
.venv/bin/python scripts/evaluate_scale_followup.py --workers 2
.venv/bin/python scripts/report_scale_followup.py
```

Local results: `data/upgrade_exploration/scale_followup_20261029/`.
Research viewer: `http://127.0.0.1:8766/review/scale_followup/index.html`.
It is separate from the unchanged production viewer. Public cards show only
Original/Resized, with a selector for the experimental resized image or current
production. Rejected proposals remain explicitly labeled.

## Decision

**Explored and rejected for production: all three arms.** This means rejected
under the measured release criteria, not discarded as universally useless.
The implementations and negative evidence remain available for future work.
No production change was made. All 134 requests were
replayed twice through the complete current engine, with identical recorded
Qwen proposals, source assets and selector settings. The fresh baseline matches
all 59 previously cached development-suite outputs. The production viewer remains
current. All 89 reference pairs are covered through the request-alias map; six
reference pairs share requests. The 309 validation pairs are not described as
replayed because their exact stored raw proposals are unavailable.

| Arm | Result | Decision |
|---|---|---|
| Intact shared-scale growth | 19 research replacements, no new automated flag types. Reference geometry MAE .044773 → .044312; font MAE .168213 → .171249. Of 89 reference comparisons, geometry improved in 12 and worsened in 7 | Rejected for production. Larger content helps some underfilled lockups but over-enlarges others and reduces edge breathing room; retain research code only |
| 12-candidate quota funnel | Five materially changed outputs. Input candidates 2,098 → 1,652, but diagnostic renders only 3,762 → 3,658 (2.76% fewer). Two requests lost the rendered deterministic-aspect family. Geometry MAE .044773 → .044355; font MAE .168213 → .167403 | Reject default. Small average gains and render savings do not justify lost independent-path coverage or removal of the current incumbent |
| Exact rank selection | Every statistic matched in 32,366 live calls. Paired full-statistic time totaled 8.614s sorted / 8.394s partitioned, ~2.6% less for this kernel | Rejected for production at present: about .221s total observed saving across 268 instrumented replays, too small/noisy for a useful end-to-end speed claim |

The full comparison took **268 model-enabled engine replays**, plus eligibility
screening and source-owned growth verification. No new token generation occurs:
this isolates engine algorithms, not inference latency. Shared-scale growth was
rejected for geometry on 53 requests, had insufficient quality margin on 29,
failed its LP on 20, found no material headroom on nine, skipped three identity
requests and failed final raster/geometry verification on one. The remaining 19
were retained for research review. Those counts are applicability/rejection
results, not 134 successful transformations.

Direct inspection covered **all 24 materially changed comparisons** and three
explicitly rejected counterexamples. Native-resolution source-floor checks on
the 24 changes found no new invisible-text or source-floor contrast failures.
These use the legacy shared text mask: rotation/color exclusions and ownership
limitations remain, so they do not establish aesthetic correctness.

Observed visual tradeoffs:

- Some framed copy and emblems gain useful prominence while preserving their
  child geometry; the algorithm does not repair an existing bad arrangement.
- Uniform growth can push small captions close to the canvas boundary or make
  an already readable emblem visually dominant. A zero-collision result does
  not identify the intended amount of negative space.
- The quota policy can shift a complete lockup to a different background zone,
  and small center corrections appear in other outputs. It is not guaranteed
  to preserve the original winner merely because raw learned candidates remain.
- The existing dense 320×50 conference failure remains. None of these arms
  establishes a readable solution under the fixed-content contract.

Exploratory family-cluster bootstrap intervals for mean geometry delta cross
zero: growth **[-.004104, +.002352]**, funnel **[-.001037, +.0000024]**, from
2,000 resamples of the 21 already-consumed reference families. These are
descriptive uncertainty estimates, not independent significance or sealed-test
claims. No threshold was adjusted after inspecting these results.

The funnel's median replay time was 10.166s → 9.168s with alternating order and
paired statistic instrumentation in both arms. Concurrent workers, model warmup
and optional macro reranking limit attribution. The measured render count is
more reliable than extrapolating that time difference to Colab, M5 Pro or Docker.

For a next attempt, retain the verified incumbent outside the funnel, reserve
the exact aspect-family predicate rather than only approximate name families,
and prune only after measuring winner recall. That is a **new unrun policy**,
not a retrospectively tuned positive result. Growth needs a source-relative
margin and prominence contract with new-family evidence before activation.
Both directions remain documented rather than silently discarded.

Validation: **457 tests and 34 subtests passed**. No engine, renderer, selector
default, inference setting or training-bundle change was made in this follow-up.
All code, fixed settings, intermediate JSON, rejected reasons and visual notes
are retained locally and summarized in the refreshed Gemini review packet.
