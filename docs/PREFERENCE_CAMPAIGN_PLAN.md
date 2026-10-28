# Human preference campaign and next priorities — October 27, 2026

## Current result

All 168 slots are reviewed and image-bound. Six/eight repeat judgments agree;
two ambiguous originals are quarantined. The merged corpus has 178 eligible
pairs in 64 independent family components: 45 train, seven validation, 12 test.
The complete fixed-configuration head matches zero-shot at 9/17 validation
preferences, so no head is saved/promoted and 39 test comparisons stay unscored.
The below first-44 counts and pilot describe the earlier checkpoint.

A generic media/backing contact-preserving growth fix improves two reviewed
widescreen layouts. Two leaderboard cases remain too small. The unified 38
regression has no new flags/winner flips; only two subpixel caption changes.

## Earlier label checkpoints

The user supplied 60 occupancy comparisons and then 21 actual-engine choices.
All 81 image hashes were verified. Forty-six occupancy pairs pass fidelity
checks; the second batch has 18 incumbent preferences, two alternative
preferences and one tie. Those 20 preferences span only seven families.
They are useful feedback, not enough evidence for a general ranking model.

The first 44 campaign labels also arrived: all hashes/side mappings match.
Twenty-four safe preferences remain after ten neither-good, one acceptable tie,
two consistent repeats and seven unsafe-preferred diagnostic choices. Combined
corpus: 90 eligible pairs / 38 family components, with 29 train, six validation
and three test families. A pilot trained only on train/validation scored 8/15
versus zero-shot 7/15 validation preferences. No test scores were computed and
no production head was installed. This weak result argues for broader, more
informative labels and measured ranking experiments, not immediate deployment.

These comparisons train a small ranking head over frozen visual features.
They do not directly supervise Qwen's JSON generator. More labels can improve
selection only when a better candidate exists and survives the constraints.
They cannot by themselves create a missing reflow or fix renderer geometry.

## Larger first round

The new two-panel queue is `review/preferences_campaign/review.html` in the
local Gemini packet. It is checkpointed and publishes completed pairs while
generation continues. Reviewed image pairs and side randomization stay fixed.
The first round is complete: 160 distinct comparisons plus eight disclosed
consistency repeats, not 168 independent training observations. It contains
56 comparisons from all twelve new authored sources and 104 from dataset
sources, spanning 50 family components. All image hashes/paths and 160 unique
image-pair signatures were verified; no Qwen validation/test family leaks were
found. The exact generating source snapshot is archived separately. Further rounds should
target model disagreements after fitting, rather than add arbitrary volume.

- Dataset source families come only from the model-training split; Qwen
  validation/test families and previous preference templates are excluded.
- New authored inputs have 20–22 elements, captions, cards, composite controls,
  long copy, edge strips, rotated tags and hero photos/products. Twelve
  variations cover six themes, two palettes and three source geometries.
  Existing official and alternate evaluation fixtures are unchanged. Related
  authored variations share family ancestry rather than being split as if
  independent designs.
- Targets include Story, Portrait, Square, Widescreen, Leaderboard and
  Skyscraper. Wide/tall source designs also convert into square/story outputs.
- Alternatives are actual rendered engine realizations from Qwen-enabled
  runs. Some comparisons are between two alternatives, not always the winner.
- Some extreme resizes have no constraint-passing alternative. Their limited
  diagnostic comparisons are marked separately; labels help failure mining.
  A preferred hard-ineligible layout cannot become a positive training example.
- Acceptable ties and “neither is good” are separate review outcomes. Neither
  fabricates a preferred image. Consistency repeats do not inflate training.

## Training and release gates

`train_siglip_ranker.py` now joins both shared template IDs and declared family
ancestry before splitting. Sampling only a few edges from a family must not
let related compositions leak across folds. Default production preparation
reserves independent training, validation and test families. Validation chooses
the epoch; test performance is measured afterward. At least 20 train, five
validation and five test families are required. A small pilot must write outside
`models/` and cannot pass the production gate.

The frozen feature cache binds actual model-weight/config hashes and image
bytes. Repeated comparisons can reuse embeddings without silently reusing a
different encoder. Explicit human choices, current image hashes, fidelity and
preferred-layout safety remain mandatory. The incumbent production head is
preserved unless the candidate beats the existing zero-shot baseline on both
validation and independent test comparisons. Test sets are development evidence,
not proof of performance on Adobe's hidden designs.

Before any head promotion, replay the fixed 38 and full 89-case suite with the
same Qwen predictions and candidate sets; inspect changed winners. Keep exact
ID, aspect, bounds, glyph clipping and unintended overlap gates authoritative.
Report results by aspect bucket and family, not only a global average.

## Highest-value proposals after this round

1. **Repair measured attachment conflicts before changing model size.** Four
   human-preferred authored alternatives have new photo/caption-backing
   collisions; two also report caption ink contacting the photo. Another
   dataset alternative has text/art occlusion, and two have smaller glyph-fit
   costs without those collision flags. No human preference is overridden, but
   the head cannot select these through current safety gates. The first contact-preserving growth repair is implemented and verified on
   the fixed cohort and four labelled outliers. Readability still prevents
   the two leaderboard reflows from winning; continue allocation/attachment
   repair rather than only ranking their existing candidates. Further changes
   need their own regression replay;
   do not simply remove the gates. See preferred_constraint_conflicts.json.
2. **Measure the candidate ceiling.** Persist the rendered candidate pool and
   compare the chosen result, the best available safe layout, raw Qwen and
   repaired Qwen. For dataset experiments, human target geometry can report an
   offline oracle ceiling; it must never choose the production output. This
   separates generator, repair and selector failures before another training run.
3. **Compare image-only with image-plus-geometry ranking.** Already measured
   containment, axis error, typography, contrast, attachment and occupancy
   features carry detail that a 224-pixel semantic encoder may lose. Fit a
   small regularized preference model using the same labels and frozen family
   split; admit it only after an independent measured win. This is a proposed
   offline experiment, not a currently enabled production feature.
4. **Use informative next queries.** After the first usable head, select close
   or disagreeing candidates across new families/aspects, with per-family caps
   and a few repeats. Active preference-learning research motivates this
   approach; it does not guarantee a specific improvement on graphic layouts.
5. **Evaluate native-aspect vision afterward.** Google's SigLIP2 NaFlex supports
   variable resolution/aspect-ratio handling. This merits a bounded encoder A/B
   for extreme banners, with latency and fine-text checks. It does not guarantee
   pixel-level alignment, and no backbone swap has been made.
6. **Keep Qwen training conditional.** The recent 1.7B run fixed ID coverage but
   worsened final font error. Retraining the generator needs evidence of raw
   topology failure, clean target/family caps and a versioned group contract;
   preference-head labels alone are not a drop-in Qwen SFT dataset.

## Performance status

Safe parts of the Gemini checklist are implemented: exact raster reuse,
vectorized glyph statistics, capability-aware device/dtype/thread settings,
syntax-aware stopping and token headroom. Two-process rendering is available
but remains opt-in: the measured small workload was slower than serial.
Blanket CPU BF16, approximate geometry deduplication and a 1400-token cap were
not adopted. No verified 6–12 second end-to-end runtime is claimed.

Profiling identified repeated asset-root discovery. The diagnostic case made
54,437 calls for four assets. The original diagnostic whole-call
memoizer resolved four roots. The bounded production cache, which deliberately
excludes work outside an adaptation, reduced actual discovery to 307 across
that complete replay; 303 occur outside the request cache. Documents/pixels
remained identical. Two warm uncached replays took 11.15/9.09s, versus
8.40/7.96s cached, excluding imports and generation. Warm timing pairs were noisy under concurrent work, so reducing
lookup count is verified while a portable speedup remains unproven. A bounded
request-owned cache passed all 38 frozen document/winner/pixel regression
checks. It resets between requests, including after failure, and isolates
parallel requests. It caches no geometry or missing-asset result globally.

## Primary references checked

- [Active preference ranking, Maystre and Grossglauser, ICML 2017](https://proceedings.mlr.press/v70/maystre17a.html)
- [Active preference learning for language models, Muldrew et al., ICML 2024](https://proceedings.mlr.press/v235/muldrew24a.html)
- [Official Transformers SigLIP2 documentation](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/siglip2.md)
- [Google's SigLIP2 NaFlex model card](https://huggingface.co/google/siglip2-base-patch16-naflex)

The proposed ranking experiments are engineering inferences from those sources
and local failure evidence. No source establishes that collecting a particular
number of labels will make this engine the challenge winner.
