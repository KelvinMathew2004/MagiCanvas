> Historical record. Use [the current documentation map](../README.md) for setup and current status. Results below belong to their recorded snapshot.

# Upgrade implementation pilots — October 28, 2026

These are executed local experiments, kept outside the production model path.
The original model/solver/critic renders record the earlier engine snapshot;
the follow-up below records the current engine. The small model pilots include one
same-size identity control; that control does not establish generator quality
because the final engine intentionally preserves the original.

## Implemented

* Pinned downloads of Qwen2.5-Coder-1.5B-Instruct, Qwen3.5-2B,
  Qwen3-VL-2B-Instruct and SigLIP2 base NaFlex. Revisions and local paths are in
  `config/upgrade_pilot.json`. Weights, caches and experiment dependencies remain
  ignored by Git and excluded from the audit ZIP.
* A shared normalized-v2 experimental provider interface for the two trained
  Qwen3 packages, untuned Coder and native multimodal Transformers providers.
  Images, when enabled, are source images, never resized human references.
  No unrelated LoRA is attached to a new base.
* XGrammar tuple schemas force the exact ordered alias inventory, required
  text scale fields, finite positive dimensions and normalized numeric domains.
  They **do not** enforce non-overlap, aspect ratios, containment or typography.
  Literal inventory completeness and usable numeric geometry are separate
  metrics; missing font fields must not be reported as omitted IDs.
* An opt-in CP-SAT candidate family packs measured atomic groups, with bounded
  wall time, optional size modes, non-overlapping macro boxes and margins.
  It keeps source children together and passes through the existing raster
  selector and safety gates. No production solver dependency was added.
* Validation-only critic comparison: unchanged global SigLIP, a predeclared
  50% global / 50% three-tile score, and native-aspect NaFlex (256 patches).
  No training, prompt sweep, test-label access or automatic promotion.
* Fresh generator checkpoints and separate unchanged-selector replays, with
  literal responses, raw coordinates, repair/selection diagnostics, hashes,
  runtime profile and model provenance. Public pages show Original/Resized.

## Adopted repair and bounded sampling follow-up

The traced rotated-label failure is now repaired in `engine/resize.py` using
the source-owned panel's local capacity. It first tries the current font in
the available local width, then fits down to the existing readability floor.
It preserves source line count and proportional tracking, leaves the panel
unchanged and uses the existing painted-label centering. Unfittable controls
remain visible and flagged; no clipping/contrast gate was relaxed. This applies
to detected owned controls, not particular content, IDs or ornament names.

In the recorded independent studio proposal, the earlier 61.21px font became
8.99px within its small rotated panel (existing floor 7.5px). The realized
candidate's glyph-overflow count fell from one to zero. Other contacts remain,
so it still does not win. This is an intermediate repair improvement, not a
claimed final aesthetic improvement.

The full 59-case frozen replay against snapshot `2707354c9629c6c6` produced
**59/59 identical selected JSONs and JPEG bytes**, zero winner changes and
zero new flag types. One previously flagged 320×50 supplied conference task
remains flagged. Current comparison snapshot: `8eae9718c383ccc6`. It reuses
the exact completed current-engine outputs from `bda550b29f1131e7` after
validating code/input fingerprints; the manifest discloses that reuse. This
is not fresh inference or a speed benchmark. Tests: **391 passed, 34 subtests**.

An optional experimental `--sample-seed` now supports recorded sampling with
temperature 0.7 and top-p 0.9. Four genuine difficult resizes were compared
using the same trained 0.6B adapter: official complex → skyscraper, independent
studio → square, supplied conference → skyscraper, and independent design →
widescreen. The grammar produced complete usable inventories on all twelve
decodes (greedy plus seeds 17 and 31). The normal engine selected the same
final layout for each of a task's three decodes. All eight additional realized
learned candidates still failed the existing physical gate, so none qualified
for the merged-pool selector experiment. This does **not** demonstrate that
ranking a pool of clean sampled layouts was tested; the proposal-quality
prerequisite failed. The complete single-arm selections and rejected stage
geometry are retained for audit.

Recorded generation totals for three decodes versus one were 84.6/29.7,
119.7/38.5, 190.3/58.3 and 107.7/31.2 seconds. Runs overlapped CPU work on an
M1 Pro; these are not controlled M5 Pro latency measurements. Production stays
single-decode. No larger model, new solver, critic, training or public adapter
upload was promoted. The current Original/Resized follow-up is at
`http://127.0.0.1:8766/review/upgrade_repair_followup/index.html`.

Direct raster inspection still finds a large unused upper region in the studio
square, dense headline/card type in the conference skyscraper, and tiny legal
copy in the official skyscraper. The supplied 320×50 conference output still
has collisions, clipping and structural flags. These are remaining limitations,
including in outputs without automated flags. Browser automation was disabled
in this desktop session; inspection used the actual local rendered images and
HTTP checks, rather than claiming an interactive browser test.

### Training worth considering later

No training is required to use this repair. The next useful training experiment
would use a matching new-base adapter with actual layout reflow examples,
especially dense and reverse-aspect tasks; the existing LoRA cannot be moved
to a different architecture. A source-image-conditioned VLM needs task-matched
training before a fair comparison. Before spending that time, create a genuinely
fresh family-held-out evaluation and more hard human preference pairs: the
small current validation subset cannot establish critic gains. A new schema
or macro-decision output requires its own dataset and adapter, while retaining
the current v2 model path as a control. None of those training jobs has started.

## Completed critic result

| Critic | All validation pairs | Actual campaign | Controlled shrink |
|---|---:|---:|---:|
| Global current | 9/17 | 5/8 | 4/9 |
| Global + three tiles | 13/17 | 6/8 | 7/9 |
| NaFlex | 11/17 | 5/8 | 6/9 |

Tiling is the strongest next critic experiment. The actual campaign improvement
is only one vote out of eight; these seven validation families cannot establish
a reliable production gain. NaFlex did not improve the real campaign subset.
The test split was not consulted. No head was trained or promoted. Cached
preferences and current prompt framing are a small, repeatedly used development
sample, not a new independent benchmark.

## Completed solver result

The corrected, paired six-case replay produced **zero changed final outputs**
and no new automated flags. Its control reproduced the published baseline for
all six tasks. The solver found bounded feasible packings for three complex
cases at both grouping granularities; some packings failed the shared label
axis/readability or primary-axis checks, and the rest did not earn replacement.
No inference is included in this frozen-proposal result. A solved rectangular
packing is insufficient evidence of a good graphic design; the solver stays
experimental. The earlier wrapper failures and wrong asset-mode run are retained
as failed evidence, superseded by `solver_pilot_v3` and the final
`solver_pilot_v4`. The latter reuses the existing painted-label centering
invariant after group placement. It removes label-axis warnings from the two
clean alternate-design packings, but their quality costs remain worse than
the incumbent. Solver time is bounded; text measurement and rendering are
additional costs, and wall-time-bounded CP-SAT is not a bit-determinism guarantee.

## Interpreting generator experiments

The native SDPA six-case comparison is complete. Both 0.6B arms emitted every
literal ID. Usable positive geometry and required type-scale fields improved
from **3/6 to 6/6** with the grammar. The trained 1.7B grammar arm also completed
6/6. None of the runs needed output truncation repair to earn those counts.

| Native arm, same six cases | Raw stretch | Raw alignment | Raw overlap contacts | Raw bounds | Raw glyph overflow |
|---|---:|---:|---:|---:|---:|
| Trained 0.6B free | 21 | 16 | 144 | 2 | 16 |
| Trained 0.6B grammar | 21 | 16 | 104 | 6 | 22 |
| Trained 1.7B grammar | 21 | 11 | 23 | 1 | 27 |

These are instance counts, not affected-output counts. Stretch counts distinct
image/decorative-art IDs once and excludes source full-bleed backgrounds;
the two existing aspect checks can otherwise report the same asset. Overlap contact classes may count the
same contact separately, so do not add columns into a single quality score.
Raw contrast regressions against the source floor were 10 / 8 / 0 respectively;
native raster exclusions (rotated or unmeasurable type) are explicit in JSON.
Raw missing/invalid layers remain absent; their contrast cannot be measured.

The final selected six 0.6B outputs are **byte-identical JSON and JPEGs** between
free and grammar arms. The 1.7B final outputs are identical on five of six
tasks. The only changed task is the church lockup: its reference geometry MAE
is 0.02043 for 0.6B and 0.02115 for 1.7B, while both final font MAEs are 0.16686.
I inspected both native renders; this is a small placement difference, not a
convincing aesthetic win. Raw geometry MAE on that task is 0.01420 versus
0.02924. Reference metrics are null for official/authored tasks.

All selected pilot outputs have zero measured flag types and zero native
source-floor contrast regressions. That is **not** aesthetic perfection:
the complex skyscrapers remain dense, source controls remain visible in the
identity control, and selector fallbacks dominate the difficult inputs. The
1.7B arm reduced raw overlaps but did not improve the final complex designs.

Native generation totals were about 138 / 191 / 271 seconds respectively.
They overlapped some CPU auditing and have different output lengths; they are
not controlled end-to-end latency measurements. The grammar's per-token CPU
mask transfer adds overhead on MPS. Bigger and stricter is not automatically a
better final quality/runtime trade-off.

### Where repair helps and hurts

Additional traced replays capture the parsed/aspect/font-repaired seed and the
realized Qwen candidate **before** selection. Tracing reproduces the prior
selected output exactly for every task. Geometry remains in per-case audit JSON;
the public page still shows only Original/Resized.

| Trained grammar arm | Stage | Tasks with this stage | Stretch | Alignment | Overlap contacts | Bounds | Glyph overflow |
|---|---|---:|---:|---:|---:|---:|---:|
| 0.6B | Parsed/aspect/font repair | 5 | 0 | 13 | 134 | 3 | 4 |
| 0.6B | Realized learned candidate | 5 | 0 | 2 | 113 | 0 | 1 |
| 1.7B | Parsed/aspect/font repair | 5 | 0 | 10 | 77 | 3 | 4 |
| 1.7B | Realized learned candidate | 5 | 0 | 0 | 6 | 0 | 1 |

The sixth identity task bypasses these stages. The free 0.6B arm only has four
parsed seeds and three of the tracked realized candidates; missing stages are
explicitly absent rather than scored as clean. Do not compare unequal stage
coverage as an accuracy improvement. Intermediate-stage contrast was not measured;
raw/final contrast is audited separately at native resolution.

This establishes two concrete points. Aspect repair works, and source/container
realization removes much of the 1.7B alignment/contact error. However, font/box
repair can increase contacts relative to raw coordinates, and a later container
candidate can introduce glyph overflow after an earlier stage had none. In the
independent studio design the 0.6B parsed seed has zero overflow and the realized
candidate has one overflowing text frame. It is rejected rather than silently
selected. A follow-up should measure font growth, parent capacity and attachment
constraints jointly. Do not weaken final clipping gates or add a rule matching
that design. This is a more actionable next engine target than changing the
model on the strength of schema completion alone.

## Untuned alternative models

Coder-1.5B, Qwen3.5-2B and Qwen3-VL-2B all completed the three-case constrained
smoke test (four, 35 and 20 source elements). Qwen3-VL was tested with and
without source pixels using the same prompt contract and native attention.
Both arms produced identical decoded geometry and final rendered images on
all three tasks. Image conditioning added 237–253 prompt tokens, confirming
the source pixels were actually included. This is a negative three-task
result, not a general claim that vision is useless.

The twenty-element reverse-ratio design exposes the failure clearly: all three
untuned bases copied essentially the same normalized source coordinates,
stretching the hero allocation and colliding headline, description and cards.
The grammar guarantees the schema, not a different topology. I inspected the
raw rendered failures and the final outputs. The selector chose existing
structural reflows for both actual resize tasks; the identity control stayed
identity. These bases need matching layout training before a fair trained
comparison. Merely moving the existing LoRA across them is invalid.

Qwen3.5-2B also uses fallback PyTorch gated-delta/causal-convolution kernels on
this Mac. No unsupported CUDA kernel was installed to hide that limitation.
The grammar and generator now share the union of tokenizer, model-generation
and nested text-model EOS tokens. The earlier EOS mismatch caused unnecessary
post-JSON generation and time-limit stops; only the corrected `new_model_native_v3`
run is used for the final multimodal comparison. Earlier runs remain historical.

### Earlier diagnostic runs

The initial eager-attention six-case 0.6B comparison emitted every literal ID
in both arms. Unconstrained output had missing type-scale fields or degenerate
geometry on four tasks. The grammar made the numeric contract complete, but
both arms selected the same six final layouts. The final selected outputs still
show aesthetic limitations despite zero automated flags. Raw stretching,
alignment defects and overflow persist: valid JSON is not a layout solution.

The initial 1.7B reference-attention run was stopped after a dense-case timeout.
It is **not** the native optimized runtime result. Native SDPA repetitions and
the new-model smoke tests are recorded separately. Model time limits are checked
between generation steps; a long prefill can overshoot the requested wall time.
Generation overlapped some CPU repair experiments, so the recorded seconds are
diagnostic, not an isolated latency benchmark or a prediction for an M5 Pro.

The new base models are **untuned** feasibility probes. The existing 0.6B and
1.7B adapters are separately trained packages; a raw base losing to them does
not show that a matched new-base adapter would lose after training.

## Reproduce locally

Use the existing project environment and its compatible PyTorch/Transformers.
Pilot dependencies stay separate; requirements are pinned in
`experiments/layout_upgrade/requirements.txt`. In this run Torchvision 0.29.2
was verified compatible with the installed PyTorch 2.14.1.

```sh
.venv/bin/python -m pip install --no-deps --target data/upgrade_exploration/deps -r experiments/layout_upgrade/requirements.txt
.venv/bin/python scripts/fetch_upgrade_pilot.py
.venv/bin/python scripts/run_upgrade_model_pilot.py --arm qwen06 --grammar --output data/upgrade_exploration/my_native_pilot
.venv/bin/python scripts/replay_upgrade_pilots.py --root data/upgrade_exploration/my_native_pilot --arm qwen06_text_grammar
.venv/bin/python scripts/replay_upgrade_pilots.py --root data/upgrade_exploration/my_solver_pilot --arm cpsat
.venv/bin/python scripts/evaluate_critic_pilot.py --arm tiled
.venv/bin/python scripts/audit_upgrade_rasters.py --root data/upgrade_exploration/my_native_pilot --arm qwen06_text_grammar
.venv/bin/python scripts/build_upgrade_pilot_review.py
```

Run only one generator process at a time on a 16GB Apple Silicon machine.
`--arm coder15`, `--arm qwen35`, or `--arm qwen_vl --image` select a different
native provider, all retaining the v2 output contract. Their corresponding
adapter is absent by design in this feasibility stage. The production
`config/layout_model_profiles.json` remains restricted to verified matched
trained packages. A VLM provider is experimental, not a production environment
variable swap.

The machine used here is an M1 Pro with 16GB unified memory, not the judge's
M5 Pro. No MLX/quantization speedup, complete 89/309 sweep, new adapter training,
public publication or production quality improvement is claimed by this pilot.

## Evidence and review

The complete local evidence lives under `data/upgrade_exploration/` and the
two-panel review is at
`http://127.0.0.1:8766/review/upgrade_pilots/index.html`.
The consolidated `report.json` contains per-stage flaw instance counts and
model-specific provenance. Separate native-size raw/final glyph-contrast audits
record coverage, exclusions and source floors. Native contrast is evaluated for
each final arm; the original selector-replay raw metrics retain their truthful
unmeasured marker, with the additional raster result alongside. Only dataset
records with actual human resized references have geometry/font MAE. The
official/authored references are null. Rendering was inspected from local image
files because browser automation is disabled in the desktop settings.

## Decisions and next experiments

1. Keep the trained 0.6B production package. Grammar remains opt-in until a
   larger same-model run demonstrates final-layout benefit worth its overhead.
   The 1.7B package remains swappable, but this pilot does not justify making it
   the default: five outputs are unchanged and the one changed output is not
   clearly better.
2. Tiled current SigLIP is the leading critic candidate. Collect new difficult
   actual-engine comparisons before fitting or promoting it. The existing eight
   campaign validation votes are now consumed evidence; do not tune to them.
   NaFlex is deprioritized because its campaign accuracy did not improve.
3. For a new-base training pilot, prioritize source-image-aware Qwen3-VL-2B
   with its own adapter and measured output contract. The existing adapter
   cannot transfer. Include dense/reverse-ratio layouts and keep target IDs and
   families disjoint. Compare against the trained 1.7B package on new families.
   The untuned image arm's identical geometry proves a need for task-specific
   learning, not that its future trained quality is known.
4. Qwen3.5-2B is a secondary candidate until its native backend is faster. A
   Mac-specific MLX implementation or CUDA-only kernel result must be evaluated
   separately; Linux ARM64 does not expose Apple Metal. Coder is a useful
   smaller text-provider control, but has not shown an immediate layout benefit.
5. Retain CP-SAT as an opt-in research candidate. Rectangular feasibility and
   an exact inventory alone are not enough: readable child realization,
   hierarchy and source-supported attachments still dominate. Do not weaken
   current raster gates to manufacture a solver win.

No full 89/309 model sweep is warranted as a promotion claim from these small
non-winning pilots. A future trained candidate must pass those broader checks,
then the frozen 59-case regression and changed-winner visual review before
activation. This experiment implements and evaluates the leading proposals;
it does not assert that every researched algorithm was adopted.
