> Historical record. Use [the current documentation map](../README.md) for setup and current status. Results below belong to their recorded snapshot.

# No-training upgrades: expanded paired evaluation

October 28, 2026. No new model training, human labels, uploads or user setup was
required. These are development experiments on previously used families, not a
new blind benchmark. Production Qwen stays enabled with its existing adapter.

## Scope and controls

- **134 distinct source/target resize tasks**, comprising the 59-case regression
  cohort and all 89 held-out dataset pairs after deduplicating identical engine
  requests. The 89 reference pairs represent **83 unique requests / 21 families**.
  Six duplicate requests share an engine output but retain separate reference
  comparisons. The alias map proves all 89 are covered; an earlier manifest's
  counter of 83 referred to representative IDs and is corrected in coverage.json.
- Exact saved decoded Qwen predictions, current source assets/fonts and one
  fixed production baseline. No Qwen-off arm, inference-speed claim or reference
  coordinates in candidate generation/selection. The recorded old-v2 prompt arm
  was selected before testing; no best-prompt oracle was used.
- Each layout algorithm competes independently with the incumbent retained.
  Complete atomic groups move together. Existing geometry, glyph, ownership and
  contrast checks remain, as do prominence and material-improvement guards.
- All changed outputs were rendered for inspection. Native-resolution glyph
  contrast/visibility audits cover **23 changed tasks across the arms**; no new
  source-floor contrast regressions or invisible-text instances were found there.
  Rotated/unmeasurable text exclusions remain explicit. Automated cleanliness is
  not aesthetic correctness.
- The 309 validation cases were **not** replayed. Their historical archive lacks
  the decoded proposals required for the same controlled experiment. We did not
  substitute old final layouts and call that a current Qwen replay.

## Layout algorithms

| Independent arm | Candidates generated / rendered | Final tasks changed | Geometry MAE on all 89 reference pairs | Decision |
|---|---:|---:|---:|---|
| Current production | — | — | 0.044773 | Retain |
| Bounded rigid-block L-BFGS-B polish | 40 / 32 | 0 | 0.044773 | Experimental; no change met the replacement margin |
| Maximal rectangles + distance-transform whitespace | 6 / 4 | 0 | 0.044773 | Experimental; few applicable collisions and no qualifying replacement |
| Nearby background-aware copy movement | 30 / 19 | 2 | 0.044773 | Reject default: moved CTAs away from a centered lockup |
| Expanded human-layout retrieval | 75 / 38 | 4 | 0.045344 | Reject default: mixed visuals; geometry and type error worsened overall |
| Tight and loose grouping alternatives | 102 / 60 | 10 | 0.050131 | Reject default: larger content sometimes fractures emblems and text stacks |
| Measured CP-SAT packing, 100ms solve budget | 74 / 33 | 3 | 0.046133 | Experimental; bounding-box feasibility did not preserve visual attachments |

"134 tasks" includes eligibility checks, not 134 applicable transformations per
algorithm. Generated/rendered counts above make the actual coverage explicit.
Render checks, measurement and Python overhead are additional to the CP-SAT
solve budget. Time-bounded solver results need not be bit-deterministic.

The retrieval experiment expanded the library from **134 to 267 layouts**,
covering **105 training-connected families**. Default, tight and loose source
partitions were considered, with at most two templates per family/aspect bucket.
Held-out and preference-connected families remain excluded. Only normalized
macro geometry/roles are stored; no borrowed pixels or copy. Extreme-ratio
coverage remains sparse: only three leaderboard-near and two skyscraper-near
entries. The library remains local and is excluded from the Gemini upload.

These failures reveal a metric limitation: width/height utilization can improve
when a compact emblem is broken apart. A no-collision packing is not equivalent
to a coherent graphic. Looser grouping separated parts of the beer emblem and
robot illustration; packing stranded a heading/decorative rules; calm movement
shifted a CTA independently of its centered heading. These examples are audit
observations, never template/content-specific engine conditions.

## Fixed critics and selectors

All **178 existing hash-verified human preferences** were evaluated with fixed
critics, no fitting or prompt/weight sweep:

| Fixed critic | All preferences | Actual campaign | Controlled shrink |
|---|---:|---:|---:|
| Global SigLIP | 109/178 | 76/132 | 33/46 |
| Half global, half three native-aspect crops | 115/178 | 80/132 | 35/46 |
| NaFlex, existing 256-patch contract | 105/178 | 74/132 | 31/46 |

Tiling wins 12 previously missed comparisons and loses six previous correct
ones; pair-level exact sign probability is 0.238. NaFlex wins 27 and loses 31
(0.694). These are descriptive, exploratory comparisons; family dependence and
multiple comparisons limit inference. All folds are now consumed. None is
reported as sealed test performance, and no label is relabeled or invented.

The tiled critic was then tested **inside the complete real engine**, with its
contextual prompts, source similarity, physical gates and coarse cost band.
All 83 distinct requests / 89 reference pairs were replayed. Eight final outputs
changed; five reference geometries improved, two worsened, and one had no
material geometry difference. Mean MAE fell **2.31%**, 0.044773 → 0.043741.
No new automated flags or audited native contrast/visibility failures appeared.
The largest gain is concentrated in one family. The exploratory family-cluster
bootstrap interval for the mean delta is approximately **[-0.002897, +0.000011]**;
it still crosses zero. Non-isolated replay-time ratios had a median around 1.39×;
this is not a clean latency benchmark or a prediction for the judge's Mac.

**Integration decision:** add `MAGICANVAS_SIGLIP_TILED=1` as an optional runtime
mode, **off by default**. It changes only the existing coarse visual tie-break.
The blend and crop locations are fixed, no additional weights are required,
and crop scoring falls back to global scoring on failure. A global-image-trained
local preference head disables tiling; more than 32 candidates also uses the
global path. The integrated scorer exactly matched the prototype's native
scores on a real paired render. The 59-case default regression is replayed
separately after this addition; its provenance is in the current viewer.

Release verification: **398 tests and 34 subtests passed**. The final default
replay, `9a62eb1431f30e97`, preserves all **59 output JSONs and rendered JPEG
bytes** against baseline `8eae9718c383ccc6`. No winners or automated flags
changed. One previously flagged 320×50 conference task remains unresolved.
The snapshot check reports CURRENT; enabling the optional tiled mode correctly
marks the default renders stale. The interactive preview server was restarted.

The previously fitted ten-feature structural head was also tested without
retraining. It compared the incumbent against clean finalists of the six new
algorithm arms—not every original engine shortlist entry. It had alternatives
on 16/134 tasks and chose seven changes. Reference MAE rose to **0.048060**
(about **7.34% worse**). It repeated some grouping/utilization failures visible
in the proposed layouts. It remains unintegrated.

## What requires no training, and what remains separate

The algorithms, expanded geometry library, fixed critics, use of the existing
structural head and speculative decoding do not require new training or labels.
A native 1.7B-target / 0.6B-draft probe used the existing matching adapters on
three actual resizes, with four draft tokens per verification step. Target-only
versus assisted times were **8.98/11.74, 41.58/57.71 and 26.08/37.58 seconds**.
All six outputs were valid. Exact target token parity held on **two of three**
cases, not all three; the different studio output needs numerical/decoding
investigation before a correctness claim. The cause was not established. Both
models were resident in both arms, order was alternated, and the native runs
overlapped CPU regression work; this is a feasibility probe, not a controlled
M5 Pro/Docker speed benchmark. No assisted decoder was promoted.

Quantized GGUF/ONNX/MLX backends still require implementation and
backend-parity/performance evaluation; they are not claimed complete here.
Semantic DINO/vision-language subject grounding is also not established by the
low-level color/texture calm-zone experiment. Stage 2 interface/export features
remain a separate product workstream, not measured layout-quality replacements.

A different generator base, macro-output contract, CVAE/diffusion generator,
or DPO/GRPO tuning needs matched training/data and an evaluation plan. A LoRA
does not transfer to a different base just because both models run locally.
No new training job is needed to use the current engine or optional tiling mode.

## Review and reproduction

- Current production: `http://127.0.0.1:8766/review/main_engine_qwen_outputs/index.html`
- Changed algorithm outputs: `http://127.0.0.1:8766/review/training_free_scale/index.html`
- Changed selector outputs: `http://127.0.0.1:8766/review/training_free_selectors/index.html`

Public cards show Original/Resized only. A dropdown switches the resized image
between current production and the experimental result; raw/intermediate
geometry stays in audit JSON. Direct inspection used local rendered images;
desktop browser automation was unavailable in this session.

Complete local evidence: `data/upgrade_exploration/training_free_scale_v1/`.
The original experiment hashes refer to the pre-integration engine; the new
optional wrapper preserves the default global path. Historical experiment
renders are not relabeled as new inference after integration.

Use a new output root after code changes:

```sh
.venv/bin/python scripts/evaluate_training_free_upgrades.py --root data/upgrade_exploration/my_scale_run --phase baseline --workers 2
.venv/bin/python scripts/build_expanded_layout_library.py --output data/upgrade_exploration/my_scale_run/expanded_library.json
.venv/bin/python scripts/evaluate_training_free_upgrades.py --root data/upgrade_exploration/my_scale_run --phase algorithms --workers 2 --arms micro-polish whitespace calm-zone retrieval dual-granularity cpsat
.venv/bin/python scripts/evaluate_tiled_selector.py --root data/upgrade_exploration/my_scale_run
```

SciPy and solver pilot dependencies are pinned under
`experiments/layout_upgrade/requirements.txt`, isolated in
`data/upgrade_exploration/deps`. They are not new production dependencies.
Datasets, caches, fonts, source-level library provenance and weights stay local.
