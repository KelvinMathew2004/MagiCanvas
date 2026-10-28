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


## October 27 follow-up: human labels and exact lookup cache

The second batch's 21 exports are hash-verified (18 incumbent preferences, two
alternative preferences, one tie). Its old 17/3 two-way split is historical;
production now reserves three family folds and joins declared ancestry.

The user then supplied 44 campaign choices, all bound to the current images
and randomized-side mapping. Ten reject both layouts; one is an acceptable tie.
Two opposite-side repeats agree with the originals and are excluded from
training. Seven preferred layouts fail automated physical eligibility and are
retained for constraint/repair investigation, not silently accepted or discarded
as human feedback. Twenty-four pairs are eligible. The combined corpus has 90
eligible preferences in 38 family components: 69 train pairs/29 families,
15 validation pairs/six families, six test pairs/three families.

A non-production frozen-feature pilot selects epoch eight and predicts 8/15
validation preferences correctly, versus zero-shot 7/15. One additional correct
choice is weak evidence; it also misses the >60% release threshold. The head
stays in ignored data/siglip_ranker_experiments, outside models/. Test preferences
are not scored or used for fitting, thresholds or epoch selection. No head is
promoted and no Qwen adapter retrained. Current test-family coverage is below
the five-family release minimum; more labels are necessary but not a guarantee
of better unseen layouts. Occupancy-only labels still dominate part of the
training distribution, so later ranking A/B must report actual-engine strata.

The first broad campaign targets 160 distinct comparisons plus eight disclosed
repeats. Twelve new 20–22-element sources cover six themes, two palettes and
three geometries, alongside family-separated dataset training inputs. Palette
siblings share ancestry. Full evidence records the generating engine snapshot;
that job started before the asset cache edit and is not relabelled as a later
snapshot. The current main regression viewer separately records current code.

Profiling one Level 3 leaderboard request found 54,437 repeated asset-root
lookups for four assets. A request-owned cache reduces rediscovery to four;
its ContextVar is isolated between simultaneous calls and reset even on error.
It caches neither geometry nor missing assets across requests. Warm diagnostic
timing pairs were noisy under concurrent jobs; no portable speedup is claimed.
The new replay 3ce220948faf736e versus 3e769527fb4f81df preserved every winner,
document and rendered pixel buffer (38/38); no new automated flags. This is an
exact performance refactor, not evidence of improved visual quality. The four
final horizontal peer rows retain clear-gap asymmetry <=1.71e-13 px.

See PREFERENCE_CAMPAIGN_PLAN.md for researched next experiments: candidate-pool
ceiling, image-plus-geometry ranking, informative comparisons and bounded
native-aspect encoder evaluation. There is no justified blanket model upgrade,
full-backbone training or production VLM change from these results.

Final follow-up suite: **303 passed, 34 subtests passed**. Interactive preview
restarted with Qwen enabled after the engine cache edit; the canonical 38-case
viewer and audit retain only Original and Resized as their public panels.

Production-toggle microbenchmark: 54,437 actual filesystem discoveries become
307 per complete frozen replay (four distinct roots inside the adaptation;
303 lookups outside its deliberately bounded lifetime). Two alternating warm
baseline runs took 11.15/9.09s versus cached 8.40/7.96s. Documents and pixels
match. This approximately 19% average reduction is a one-case observation
excluding token generation/imports under concurrent work, not a general SLA.

Seven human-preferred rejected alternatives were reinspected without changing
engine gates. Four have new hero/caption-backing collisions, two of those also
have caption-ink/photo contact; another dataset example has text/art occlusion.
Two more have small glyph-fit costs. These are recorded in
preferred_constraint_conflicts.json. Photo/caption attachment realization is a
concrete next repair experiment before a larger model or wholesale gate removal.

The first queue is complete: **160 distinct image pairs + eight consistency
repeats**, including 56 comparisons from all twelve new authored sources and
104 dataset comparisons, across 50 family components and all six formats.
153 pairs have at least one physically eligible alternative; seven are explicitly
diagnostic-only. All 168 image hashes/paths pass integrity checks, there are no
Qwen validation/test-family leaks, and all 20 generating source-file hashes
match the archived snapshot. The browser retained 44 submitted choices after
refresh and was left at comparison 45/168, without creating any labels.

## Complete 168-slot human review and attachment repair

All 168 image hashes/randomized-side mappings match, including all previous
44 choices unchanged. The export records 91 incumbent choices, 36 alternatives
and 41 skips (including repeats). Six/eight repeats agree. Two inconsistent
originals are quarantined; repeated votes never inflate training counts. After
skips, seven physically ineligible positives, repeats and the two ambiguous
originals are excluded, this batch supplies 112 eligible preferences. Combined
with the earlier 46 occupancy and 20 engine preferences, the corpus has **178
pairs / 64 family components**: 122 train/45 families, 17 validation/seven
families, 39 test/12 families. Label files and immutable training-input copies
remain local; no approvals are synthesized.

The fixed-configuration frozen SigLIP2 head selects epoch seven and predicts
**9/17 validation choices**, identical to zero-shot 9/17. It fails both the
>60% threshold and the baseline-improvement gate. Test examples are not even
prepared/scored until validation passes; all 39 remain reserved. No candidate
head was saved, no production weights changed and no Qwen adapter was retrained.
See preference_complete_training.json for exact input/code/model hashes.

The recurring photo/caption collision is caused by expanding a text backing
around its center. `_attached_growth` now preserves source-evidenced contact on
any of the four sides of a substantial foreground image, growing away from it.
It applies only when unrotated target elements still share that contact. It
leaves deliberate source overlaps and detached reflows alone, and refuses
growth that cannot retain the edge inside canvas rather than create a new
collision. No design ID, caption wording or named ornament is used.

In a same-candidate ablation, four source-disjoint photo/backing collisions
become zero. This does not make all four candidates acceptable: the requested
readable caption cannot fit in those particular allocations, so existing
readability checks continue rejecting them. The complete-engine replay finds
a different, clean macro reflow for both widescreen inputs, with a much larger
photo and its caption correctly attached. Those outputs were visually inspected.
The two leaderboard outputs remain byte-identical to their prior tiny layouts,
with control_padding/control_readability failures. This is a remaining candidate
capacity problem, not solved by training the preference head or hiding flags.

Regression run `8cce61ab2f2644f1` against `3ce220948faf736e` keeps every winner
and adds no flags. 36/38 pixel buffers are identical; two existing Field Notes
caption/backing pairs move by 0.272 and 0.340 px to restore their contact edge.
Those changes were inspected. Source/Resized panels and audit JSON are refreshed
in the unified viewer; the four additional diagnosed inputs are linked from it.
**311 tests and 34 subtests pass**. No fresh 89/309 release sweep is claimed.

## October 27: guarded measured-container follow-up

See DENSE_REFLOW_AUDIT.md for the supplied audit's checklist, primary references
and accepted/rejected details. The implementation adds measured text/native
backing realization, bounded contiguous column partitions, attached photo/caption
orientation choices, wrapped peer controls and separately identified Qwen-driven
macro bands. Qwen remains enabled; deterministic wins retain their actual names.
No adapter or prompt contract changed, no weights were trained or published.

The broader replay caught two weaknesses before finalizing: unresolved ornaments
could get invented flow slots, and atomic tracked lettering could acquire extra
lines. Conservative guards retain those arrangements. A second check exposed
that a readability flag could mask simultaneous glyph overflow in the macro
check. The macro gate and batch audit now share the exact glyph-overflow helper;
all flag types accumulate. Label ink is centered again after final atomic font
fitting. Readability-only compromise candidates never bypass glyph overflow,
collisions, inventory/bounds, padding or contrast, and retain any remaining flag.

Final fixed regression `1f26c6e5d74e8077` versus `8cce61ab2f2644f1` preserves all
38 winners and output geometry, with no new flags. The final test run has
**321 passed tests and 34 passed subtests**. The preview process was restarted.
Two additional 22-element, 900×1200-to-728×90 attachment tasks now select
`macro-measured-columns-6-1.0` instead of the tiny proportional fallback. Both
have no automated flags. Their two widescreen companions retain their clean
macro reflow. These four outputs were visually inspected; small legal/stamp
annotations and the equal-width column composition still limit aesthetics.

The 23 dataset neither-good comparison votes deduplicate to 15 source/format
tasks (five leaderboards, three portraits, three stories, two squares, one
skyscraper and one widescreen). Replays preserve the original voted images and
labels; the additional viewer contains Original and Resized only. Its comparison
baseline is the campaign's generating snapshot, so differences include earlier
engine work; it is not an isolated ablation of this patch. Zero flags on those
15 outputs does not overturn the human neither-good judgments. In particular,
tracked display typography still has awkward wrapping/reading order in an
extreme banner, and sparse or weak compositions remain in other unchanged cases.
Do not call those designs perfected or relabel them as approved training pairs.

No new full 89/309 release sweep, model training, production ranking-head
promotion, public adapter publication or offline Docker build is claimed.

Final pixel comparison confirms **38/38 identical rendered buffers** against the
start-of-change snapshot; four existing peer rows still have equal clear gaps
(maximum difference <2e-13 px). In both diagnosed leaderboards, content width
use grows from 8.0% to 97.5%, median font from 1.125px to 5.4px and headline from
2.7px to 8.87px. This measures the controlled four-task replay, not a universal
readability guarantee. The 15-task replay has two material geometry differences
and one changed winner versus its older campaign snapshot; no automated flags.

The final replay of both saved 20-case model arms retains five learned-family
wins per arm and the previous dataset geometry/font metrics. There is one
rendered Qwen macro-band candidate per arm, rejected for control readability;
none wins. Merely adding macro realization therefore does **not** resolve the
seven/eight dense authored learned-family losses. Poor/incompatible coarse
bands still need better proposals or container allocation; we do not force them
to win or promote 1.7B on valid IDs alone. These are replay results using prior
fresh model outputs; the stored generation seconds belong to that earlier
inference, not this change's runtime measurement.
