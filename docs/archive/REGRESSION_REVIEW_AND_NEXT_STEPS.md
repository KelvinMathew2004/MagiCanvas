> Historical record. Use [the current documentation map](../README.md) for setup and current status. Results below belong to their recorded snapshot.

# Shared copy axes and the regression workflow

Latest October 28 work: [runtime, experimental features and skyscraper testing](RUNTIME_AND_EXPERIMENTS.md). The unified viewer retains the fixed 38, eight fresh independent conversions, and adds 13 supplied/skyscraper stress conversions. Historical results below retain their original scope.
Earlier dated results below retain their original scope.

**Latest:** see [ENGINE_UPGRADE_RESULTS.md](ENGINE_UPGRADE_RESULTS.md) and
[ENGINE_UPGRADE_CHECKLIST.md](ENGINE_UPGRADE_CHECKLIST.md) for the fresh 20-case
A/B, the current 296-test/38-case verification, preference-label status and
submission preparation. The copy-axis checkpoint below is historical evidence.

## Why the reported leaderboard drifted

This was an engine repair problem, not evidence that the adapter needs another
training run. In the recorded Level 2 → 728×90 case, the initial editorial seed
placed the kicker backing, headline and body at X = 167.5703. Independent
annealing moved the body to 175.6523. `_repair_copy_hierarchy` moved the headline
to the body's new edge but did not own the kicker. The later source-axis
projection used their former centered relationship even though this candidate
had reflowed into a left-aligned editorial column. The final headline/body
frame lefts became 184.5320 and 166.7725; the kicker was at 167.3264.

The same seed had a separate inline CTA column. It should remain separate;
aligning every button to every nearby text block would break that topology.
The original source also has authored body/CTA contact, which is a different
issue from an introduced resize misalignment.

## Bounded correction

`source_structure.copy_axis_contracts` remembers an axis a candidate already
declared: at least two distinct copy units with explicit matching text
alignment, shared target left/center/right axis within 0.5px, a vertical order,
and a bounded gap. An already aligned kicker immediately above can join as a
whole backing/label unit. Peer rows, offset text, incidental matching edges of
centered text, unrelated corner badges and inline CTA columns do not qualify.

The optimizer projects proposed moves onto these existing column axes before
scoring. The final renderer path restores them after source-axis snapping,
text fitting and artwork separation. Only X changes; children move rigidly.
New text collisions, new non-layered unit contacts and canvas violations reject
the entire contract. This preserves candidate topology rather than imposing a
new heuristic template or making Qwen win artificially.

Source-attached and macro-locked candidates are excluded because their existing
container realization already owns internal positions.

The same code is used across eligible candidate families; no source IDs, strings,
ornament types or target-format coordinates enter the rule. This is deliberately
a narrow preservation pass: it does not straighten arbitrary messy columns
that were never aligned, solve every near-alignment, or guarantee aesthetics.

## One current viewer

Canonical URL:
`http://127.0.0.1:8766/review/main_engine_qwen_outputs/index.html`.
It redirects to the single completed `review/regression_suite/index.html` and
retains the optional group filter. Clear the filter for all 38 conversions.
There are exactly two displayed images per case: Original and Resized.

`config/regression_review.json` freezes all 18 official conversions across
square, portrait, story, 1600×400 web banner, 728×90 leaderboard and 160×600
skyscraper; eight conversions of the four independently authored alternate
sources; and twelve dataset cases (eight test, four validation). Dataset
selection used source element/text counts and family diversity before reading
output scores: 11–20 elements, at least three text elements, nine families.
These are similar-complexity checks, not exact replicas of the official schema.

```bash
.venv/bin/python scripts/refresh_regression_viewer.py --workers 2
.venv/bin/python scripts/refresh_regression_viewer.py --check
.venv/bin/python scripts/package_regression_review.py
```

The first command replays the same saved decoded predictions with Qwen enabled,
through the normal parser, repair, candidate generation, renderer and selector.
No model-off arm is generated. Original raw completion/validity metadata is
retained separately: rebuilding JSON for replay is not credited as model
completion. Official and authored fixtures have no resized human target, so
geometry/font MAE is null. Dataset gold is used for reporting, not engine input.
Two isolated CPU workers accelerate independent cases; this is not concurrent
LLM generation or a token-speed benchmark. Runtime measurements exclude model
sampling and should not be advertised as end-to-end inference speed.

Runs use code/input/adapter fingerprints and publish only after all 38 engine
runs and image renders complete. Failures leave the previous complete viewer
available. `--check` detects common code/input changes; model weight statistics
are recorded alongside hashes of the actual adapter and cached prediction
reports. Repeatedly reviewed data is a development regression set, not a blind
proof of generalization. The wider 89-case test and 309-case validation cohorts
remain appropriate release checks; this change does not claim both were rerun.

For this particular change, `--compare-before` disables only candidate-axis
contracts in the baseline. Both arms have identical current engine code,
selector, adapter and raw predictions. This option is a focused ablation,
not a generic substitute for freezing the previous engine for future changes.

## What to do next, in priority order

1. **Keep the pipeline measurable before more training.** Inspect winner flips,
   new flags and actual rendered axes in this fixed viewer, then validate larger
   batches by defect type and aspect bucket. Extend metrics for column-axis
   error, orphaned attachments and usable occupancy before adding more layouts.
   Prefer candidate-owned constraints over additional ornament-specific repairs.
2. **Keep the 0.6B adapter for now.** The existing paired 89-case A/B in
   `DAY4_5_AUDIT.md` did not promote 1.7B: geometry MAE 0.06347 vs 0.06155 and
   relative font MAE 0.20039 vs 0.17535 were worse, despite slightly better raw
   validity. Those are historical checkpoints, not new scores for this change.
   A fresh fair A/B after freezing the selector is useful; model size alone is
   not a reason to switch. Current post-processing drift cannot be trained away
   by repeating the same adapter recipe. The dense authored fixtures are a
   separate warning: seven of their eight saved 0.6B proposals omit 1–3 IDs,
   despite complete JSON syntax. Before retraining or switching defaults, run
   fresh 0.6B/1.7B inference on this same dense cohort with identical prompt,
   budgets and selector. Report ID coverage, raw topology, final defects and
   full runtime separately. A conditional dense-input model policy is an
   experiment to evaluate, not an implemented or proven improvement. The
   current alignment correction does not fix these missing-ID/raw-reflow issues.
3. **We already run SigLIP 2.** `fetch_siglip.py` downloads Google's
   `siglip2-base-patch16-224`; the `models/siglip` directory name is historical.
   No active `siglip-ranker.pt` or human approval file was found in this audit.
   The existing `train_siglip_ranker.py` trains only a small preference head over
   frozen embeddings; the backbone stays frozen. Collect diverse same-renderer
   human preferences first, including near-aligned vs aligned candidates and
   attachment failures. Roughly 100–300 reviewed comparisons is a practical
   starting target, not a guarantee or mandatory threshold. The script's minimum
   of 20 is only a software gate; use family-disjoint validation and demand a
   measured win without weakening geometry safety checks.
4. **Then consider a NaFlex encoder A/B.** Extreme banners lose tiny-text detail
   in the current 224-square, letterboxed semantic encoder. Google's NaFlex
   variant supports aspect-ratio/resolution adaptability. This motivates an
   offline experiment, not an assumption that it understands pixel alignment.
   Keep it a bounded semantic tie-breaker; geometry and glyph checks still own
   safety. A new backbone needs compatibility, runtime and preference-label
   checks before promotion.
5. **Only retrain Qwen when failure mining points at model output.** If complete
   raw proposals consistently have poor topology across untouched families,
   build a versioned macro-decision/group contract and regenerate training
   labels with target/family leakage caps. Keep the current v2 adapter supported.
   Compare raw and repaired defects separately, use response-only loss and
   evaluate on unseen families. No new training, model download, production VLM
   or adapter promotion was performed for this fix. An optional offline VLM can
   help label failures, but is not required to resolve the current axis bug.

Primary references checked October 27, 2026:
[Google SigLIP2 NaFlex model card](https://huggingface.co/google/siglip2-base-patch16-naflex),
[SigLIP2 release and NaFlex explanation](https://huggingface.co/blog/siglip2),
[Qwen3 1.7B model card](https://huggingface.co/Qwen/Qwen3-1.7B).
The recommendations above combine those capabilities with local project
measurements; the model cards do not promise improved graphic-layout ranking.

## Verified checkpoint — October 27

The final suite passed 285 tests and 34 subtests. The 38-case same-prediction
comparison added no automated flag types; all 38 outputs currently have zero
automated flags. Five official-format renders changed, and the other 33 PNGs
were pixel-identical in a separate render comparison. Three candidate names
changed; one dataset winner flip produced identical pixels. All alternate and
dataset renders remained pixel-identical. The reported leaderboard's headline/
body left-axis error fell from 17.7595px to zero; its kicker is within 0.348px.

The changed official widescreen and skyscraper results were inspected visually,
as was the reported leaderboard. A 0.0071px geometry change in the dense Level 3
leaderboard is retained in the evidence rather than hidden by rounding. Current
evidence is `data/qwen_lora/regression_suite/a3250424f7a5b396/` and
`data/qwen_lora/shared_column_audit/`. This is a focused development regression
check, not a new 89/309 full-split evaluation or proof that tiny/underutilized
alternate layouts are solved.
