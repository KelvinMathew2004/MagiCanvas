# Recovered 0.6B adapter: completed A/B

October 29, 2026. **Keep `qwen3-layout-lora-v2` as production default.**
The recovered checkpoint is integrated as the selectable profile
`qwen3_0p6b_audited_candidate`, but it fails the promotion gates. Lower training
loss and several improved layouts do not outweigh inventory and final-layout
regressions.

[Full 148-task comparison](http://127.0.0.1:8766/review/trained_adapters/index.html)
· [Production regression viewer](http://127.0.0.1:8766/review/main_engine_qwen_outputs/index.html).
Both show Original and Resized; the A/B selector switches only Resized.

## Completed comparison

Both arms use the same Qwen3-0.6B base, prompt v2, greedy inference, renderer,
selector and engine snapshot. Qwen stays enabled; deterministic winners are
reported honestly. Each arm covers **89 reference pairs and 59 review tasks**.
Three exact-size identity tasks bypass generation, leaving 145 generation tasks.
These are consumed development cohorts, not an unseen test.

| Metric | Existing adapter | Recovered adapter |
|---|---:|---:|
| Exact raw IDs, all generation tasks | 122/145 | 115/145 |
| Complete JSON, same tasks | 145/145 | 145/145 |
| Exact raw IDs, 89 reference pairs | 88/89 | 88/89 |
| Raw box MAE, 87 common-valid reference pairs | .045335 | .047144 |
| Raw relative font MAE, same 87 pairs | .265078 | .259187 |
| Final box MAE, all 89 references | .044782 | .049229 |
| Final relative font MAE, all 89 references | .167238 | .181275 |
| Learned-family winners, all 148 tasks | 63 | 61 |
| Flagged final outputs, all tasks | 1 | 1 |
| Exact final IDs | 148/148 | 148/148 |

Final reference box error rises **9.9%**, font error **8.4%**. Box error improves
on 21 reference pairs, worsens on 25 and is unchanged on 43. Raw font error's
mean improves 2.2%, but its median worsens from .14151 to .17516. The two raw-ID
failures concern different reference tasks, so equal 88/89 totals conceal a lost
previously valid proposal. On 35 dense tasks, exact raw IDs improve 14 → 16,
but previously valid dense inputs also regress. This is not a safe replacement.

The official subset is particularly weak: exact raw IDs fall **15/15 → 5/15**;
on the full 59-case review subset, **34/56 → 27/56**. The 12 review dataset pairs
have slightly better box MAE (.03780 → .03599) but worse font MAE
(.10578 → .12121). No 309-case fresh validation benchmark is claimed here.

Family-cluster bootstrap over 21 reference families gives an after-minus-before
box delta of +.004447, descriptive 95% interval [.000037, .011060]; font delta
+.014037, interval [−.008647, .040510]. These repeated development measurements
are not a claim of unseen-test significance.

## Before versus after engine processing

The independent stage audit retains literal decoded boxes, missing IDs and fonts;
it does not fill, snap, clamp or font-fit the raw arm. Source tracking scales with
the predicted font. Counts below are **cases with a measured defect**, not the
number of offending elements.

| Measured category | Existing raw | New raw | Existing final | New final |
|---|---:|---:|---:|---:|
| Source box-aspect deviations | 130 | 130 | 0 | 0 |
| Label/primary-axis misalignment | 66 | 45 | 0 | 0 |
| Unintended overlap | 104 | 72 | 1 | 1 |
| Glyph overflow | 95 | 93 | 1 | 1 |
| Out-of-bounds boxes | 27 | 24 | 1 | 1 |

Denominator: 145 non-identity tasks per arm. Box-aspect deviations use the existing
geometry contracts, not a claim that every `contain` image visibly stretches.
Misalignment covers the existing label/primary-axis contracts, not every aesthetic
relationship. Final means the **selected engine layout**, which can be a
deterministic alternative rather than a repaired Qwen proposal.

Fewer raw overlaps can come from omitted text. On the **109 tasks where both raw
inventories are complete**, old/new raw misalignment is 40/32, overlap 70/54,
glyph overflow 71/73 and out-of-bounds 18/19. Thus improved alignment/overlap
coexists with worse text fitting and inventory reliability. Independent native
contrast counts are diagnostic shared-mask measurements; rotated/unknown-color
text remains explicitly unmeasured. They do not override production gates.

## Visual review and remaining defects

All **61 changed selected layouts**, including all 15 changed review outputs,
were inspected. Ten largest reference geometry regressions and five largest
improvements were also compared with native human-target JSON renders. Exact
before/after document hashes accompany the assistant's notes; these are not
human preference labels. Browser control was disabled in this session, so
rendered artifacts were inspected directly; browser interaction is not claimed.

- Gains include clearer Origami attachment, larger beer emblems, a stronger
  independent story, and improved Eco-Friendship/Ceramics typography.
- Losses include smaller official typography, a shrunken Rome/church lockup,
  separated illustration/caption stacks, and RELY frames detached from their red
  decorative panels. Some have zero automated flags.
- A larger Kitchen card is readable but overshoots the human target. Its new raw
  box MAE worsens .07286 → .14505; the selector chooses a split, and final font
  error increases .25261 → .88428. Raw font improvement alone is insufficient.
- The same **320×50 supplied conference** remains flagged in both arms: tiny
  cards and cramped/overflowing controls. It is unchanged, not fixed or hidden.

## Recovery and reproducibility

The local package is `models/qwen3-0.6b-latest/final`: completed checkpoint
210/210, epoch 3, best recorded validation loss .7931865454; 392 finite tensors,
10,092,544 LoRA parameters, rank 16 / alpha 32. The tokenizer matches the locally
prepared 0.6B bundle, and matching-base PEFT load plus finite CPU forward passed.
The original final training manifest and full Colab environment were not supplied;
no exact training-bundle identity or original base revision is invented.

```bash
.venv/bin/python scripts/check_model_profiles.py --profile qwen3_0p6b_audited_candidate
.venv/bin/python scripts/evaluate_trained_layouts.py \
  --candidate-adapter models/qwen3-0.6b-latest/final
```

Completed snapshot: `data/training_next/evaluation/9cce0817eef84ff1/`.
`paired_analysis.json`, `stage_defect_audit.json`, `visual_review.json` and
`native_raster_audit.json` retain results; `manifest.json` hashes engine, inputs,
assets, weights and evaluators. The first interrupted collection supplied 48
existing-adapter and 33 recovered-adapter decoded predictions. Compatibility was
verified and every one was re-realized under the corrected engine; old final
outputs were never copied. Remaining predictions were freshly generated.

Independent MPS processes overlapped on this local M1 Pro. The baseline was
suspended for a single-writer handoff; one task includes that waiting in its wall
time. See `timing_limitations.json`. This study establishes **no generation
speedup or Linux/Docker latency claim**.

A conference zero-span source spacer exposed an exception after successful Qwen
decoding. The adopted generic correction records realization failure and retains
raw evidence plus the existing fallback gates. Required loading/generation still
fails closed. An initial denominator-guard experiment was rejected after a poor
conference winner. Corrected fixed replay `a8e413ab4a4d1ca4` has all 59 geometries,
winners and Original/Resized JPEGs identical to `f013e16cf8704707`, with no new
flags. Current tests: **485 passed and 34 subtests passed**.

## Next training decision

Do **not** repeat the completed 0.6B epochs. Keep this adapter for research, not
production. The separately prepared 0.8B and VL-2B challengers remain pending;
[TRAINING_CONTINUATION.md](TRAINING_CONTINUATION.md) runs only those and saves
frequent recovery checkpoints. Publication remains pending.

The local prepared corpus has complete answers in 1,043 training and 309
validation rows, but zero native shape elements and only 34 training rows with
18+ elements. The final comparison misses 71 element instances with the old
adapter versus 135 with the recovered one, mostly text. That suggests a useful
**separately versioned training-only native-shape/control augmentation study**,
not another arbitrary spacing patch. It does not prove causality or reconstruct
the missing original training provenance. Keep official, reviewed alternate and
evaluation sources out of training, and require these same two promotion gates
for each challenger.
