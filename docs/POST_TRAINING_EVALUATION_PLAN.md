# Internal plan after uniform challenger training

October 29, 2026. Gemini's suggestions and current decisions; not extra Colab
instructions. Production stays unchanged.

## Data and experiment identity

New hashed `audited-uniform-v1` bundle: 1,043 screened train / 309 validation
rows, unit weights, one sampler visit per row. Retain the 79 exclusions, source
PNGs, modern projection targeting and response-only loss. The earlier A/B does
not prove that weighting caused omissions; original training provenance is
incomplete and no isolated weighting ablation established causality. Comparing
the challengers with a differently trained champion is a capability experiment,
not a pure architecture ablation.

Prioritize VL-2B for its source-image input, then 0.8B whose hybrid reference
kernels were slow in the observed T4 run. This does not establish superior
quality, speed or T4 fit. Require actual CUDA preflight. Match each adapter with
its own pinned base; LoRA weights do not transfer between architectures.

Current user priority is **VL-2B only**; 0.8B is deferred. The portable bundle
still contains both datasets for reproducibility, but the guide and notebook
explicitly select VL-2B, so no bundle/hash change or re-upload is needed.

## What the champion's history actually establishes

The installed `models/qwen3-layout-lora-v2/adapter_config.json` records the
Unsloth Qwen3-0.6B 4-bit base alias, rank 16, alpha 32, no dropout, and seven
standard attention/MLP projection targets. It does not record the exact data
hash, optimizer state, full training arguments or original base revision. Its
README is a generated model-card skeleton with missing training-data details.

Git recovery now establishes more than metadata: commit `ae0fe91` added the
40,422,168-byte adapter weight file, and its SHA-256 is identical to the current
champion (`6ebbbe8d20e88504a1ecbb9fb6d5c9466df1c111bde84f099a6f788089d3c63b`).
The adapter config and prompt-version marker also match byte for byte. Exact
notebooks and exporters from that commit and its parent `210a68b` are recovered
in [champion_training_history](archive/champion_training_history/README.md),
with per-file hashes. No weights are redistributed in that documentation.

Both historical notebooks specify Unsloth, response-only masking, three epochs,
1e-4 learning rate, microbatch 1 / accumulation 16, 4096-token maximum and 8-bit
AdamW. They differ in data policy and checkpoint selection: the parent notebook
uses `data/qwen_lora/`, epoch saves and a target-font-fitting exporter. The
adapter-addition version preserves reference fonts, introduces source caps and
best-validation step checkpoints, but calls source-capped data the **next
experiment**. That distinction prevents us from assigning the newer data policy
to the champion simply because both appear in the same commit. Neither version
saves an exact uploaded-dataset hash or package lock with the adapter. Gemini's
assertion that weighting definitely caused the regression remains unproven.

The retained source-capped export has 2,013 rows but only 938 unique target IDs
(derived from pair endpoints), with up to 16 rows per target. The uniform VLM
export has 1,043 rows and 1,043 unique targets. Therefore a larger row count does
not itself establish greater target diversity; restoring repeats would undo the
explicit one-pair-per-target policy without a controlled ablation.

The uniform VLM bundle keeps response-only supervision, three epochs, rank
16/alpha 32 and 1e-4 learning rate, while removing dense repetition and quality
downweighting. It retains image-conditioned processing and architecture-aware
targets. It does not transfer the rejected adapter, fabricate an exact champion
recipe, add native-shape examples, restore excluded targets or expand to 2,013
rows. The native-shape/domain gap remains an experimental limitation shared by
the Crello-only corpus. No untested augmentation or optimizer swap should be
introduced just before this run; gains still require the frozen-selector A/B.

## First: frozen-selector model comparison

- Verify final weights, processor/tokenizer, pinned base, bundle hash, best
  validation checkpoint and actual training environment.
- Extend the evaluator to select either completed challenger without requiring
  another 0.6B training run. Collect fresh inference for the champion and each
  challenger on the same 89 references + 59 review tasks, Qwen enabled.
- Hold prompt contract, renderer and selector fixed. The VLM gets source JSON
  and source render only; no human-target image or target line-count leakage.
- Report exact IDs versus JSON closure, omissions by source role/type and
  density, common-valid raw box/font errors, final errors and learned winners.
- Inspect all changed review outputs, new flags and reference outliers. Keep
  Original/Resized public panels and intermediates in audit JSON. Refresh the
  canonical viewer and Gemini packet. Zero flags is not aesthetic approval.
- Do not promote if previously valid dense inventories, final errors or new
  defects regress. Do not tune promotion gates after seeing the result.

## Then: independent engine ablations

### Frame-unpinned inner reflow

Consider, not integrated. Confirm on actual sparse-case composite graphs that
an enclosing low-information layer is the articulation bridge. A one-block
count alone does not establish this cause. Test removal of that grouping edge
while preserving atomic child lockups and source attachments. Never stretch
artwork/images; an elastic backing/border primitive requires explicit type and
source evidence. Add a candidate, retain the incumbent, require normal gates
and complete 89+59 regression/visual comparison. Read prior grouping-split
counterexamples before another broad sweep.

### Contrast/busyness mutation acceptance

Consider separately after model-only evaluation. Seeing the source does not
guarantee subject detection or good placement. Compare pre-container versus
projected candidates with native glyph masks and source contrast floors.
Reject a blind minimum-pixel monotonicity veto: antialiasing, legitimate font
growth and changed mask area can make minimum contrast/busyness noisy. Keep
ownership uncertainty explicit and measure meaningful regressions, runtime
overhead and actual visual gains before adding render checks to every pass.

### Group-box-only measured layout

Potentially useful, partly implemented already. Audit `project_container_tree`,
`container_flow` and current Qwen container-locked candidates first. Identify
whether source grouping, predicted macro allocation or leaf jitter actually
caused the failure. Add an opt-in measured-group candidate only for a distinct
missing behavior. Ignore tiny rules in macro placement, preserve rigid
subgroups, measure backed labels and reject unsafe inter-group resolution.
Retain raw and existing projected candidates for attribution.

Run the three ablations independently with frozen model predictions. Do not
silently add them before the model comparison. More Qwen winners alone is not
better output. Adopt only generic source-evidenced gains without regressions;
record rejected changes too.

## Remaining limits and submission

The 320×50 supplied conference remains defective; sparse proportional fallbacks
and lost attachments can pass automated checks. Rejected heuristics do not
prove a Pareto frontier or that only three algorithms remain. Numeric negative
zero in an older report is not a visible bounds defect; use current raster and
geometry evidence. No guarantee of perfect core/output follows from test counts.

After selection, user-approved public Hugging Face publication and pinned
revisions remain pending. Verify the final offline `linux/amd64` Docker build
and `--network none` run, CPU memory and evaluator operation limits. Recording
this plan performs no training, publication or selector change.
