> Historical record. Use [the current documentation map](../README.md) for setup and current status. Results below belong to their recorded snapshot.

# MagiCanvas

MagiCanvas adapts editable Chameleon designs to new canvas sizes, producing JSON and a PNG using the supplied renderer. It combines model proposals with deterministic reflow and render-based checks. It preserves element IDs, text and original image assets; image cropping can change as a design is resized.

## Quick start

Requires Python 3.10+ and network access to install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m engine.main
```

The runner reads `input/*.json` and writes JSON and PNG to `output/`. The three official sample inputs are included. The deterministic engine runs without model weights. To use the interactive local preview:

```bash
python viewer/server.py
# Open http://127.0.0.1:5050
```

Restart the preview server after changing engine code. The preview enables Qwen when its default local base weights are present; set `MAGICANVAS_QWEN_ENABLED=0` before starting it to disable Qwen. See [viewer instructions](viewer/README.md).

## Offline Docker evaluation

The build needs internet access to install dependencies and fetch public Qwen3-0.6B and SigLIP2 base weights. The resulting container uses local files without network access:

```bash
docker build --platform linux/amd64 --target runtime -t magicanvas .
docker run --rm --network none \
  -v "$(pwd)/input:/app/input" -v "$(pwd)/output:/app/output" magicanvas
```

The explicit `runtime` development target runs the **base Qwen model without a LoRA adapter**. The default/final `submission` target fails closed unless a public adapter repository and full 40-character pinned commit are supplied. It verifies base/LoRA hidden-width compatibility and the prompt marker during the build and requires the adapter at runtime. The fine-tuned adapter is not included in Git and will be published to Hugging Face for final submission; no public adapter repository has been configured yet. Local adapter verification has passed; Docker build/offline execution is still unverified because the development Docker daemon is unavailable. Once the public artifact is available, bake it in at build time (a revision pins the artifact):

```bash
docker build --platform linux/amd64 --target submission \
  --build-arg QWEN_ADAPTER_REPO=OWNER/ADAPTER_REPO \
  --build-arg QWEN_ADAPTER_REVISION=REVISION -t magicanvas .
```

For deterministic-only comparison add `-e MAGICANVAS_QWEN_ENABLED=0` to `docker run`. Adapter-free and fine-tuned output can differ. Model loading falls back to deterministic candidates if the model or adapter cannot be loaded; inspect diagnostics when evaluating model performance.

## Architecture and trade-offs

The engine infers source relationships and functional containers, then generates learned Qwen proposals and deterministic candidates including proportional, semantic/editorial and, when appropriate, macro-band reflows. Source relationships guide repair and attachment preservation. A renderer measures text fit, overlaps, contrast and other defects before selection; SigLIP2 is an optional bounded tie-breaker. The checks are safeguards, not a guarantee of a flawless or feasible layout for every dense or extreme format. Existing intentional overlaps can be preserved. The engine preserves artwork proportions; native panels, backgrounds and dividers may resize as layout surfaces.

Ungrouped ambient paint is separated from functional containers using effective layer and fill opacity. Additional intrinsic row/column candidates move atomic cards and media attachments together; measured occupancy, readability and primary-content prominence govern acceptance. This is a bounded layout search, not a full CSS/Figma implementation. See the [research and regression decisions](docs/UNIVERSAL_REFLOW_RESEARCH.md).

The Qwen base and compatible LoRA are selected independently through `MAGICANVAS_QWEN_DIR` and `MAGICANVAS_QWEN_ADAPTER_DIR` before starting the process. The adapter must match the base and prompt contract; this is not a promise that arbitrary models work. The default local adapter is `models/qwen3-layout-lora-v2` when present. For example:

```bash
MAGICANVAS_QWEN_ENABLED=1 \
MAGICANVAS_QWEN_DIR=models/qwen3-1.7b \
MAGICANVAS_QWEN_ADAPTER_DIR=models/qwen3-layout-lora-v2-expanded-1.7b \
python -m engine.main
```

Qwen admits prompts up to 6,000 tokens only when an inventory-sized JSON output reserve fits within 8,192 combined tokens (or the model's smaller limit). Dense prompts crossing the former 4,096-token boundary use a 900-token relationship budget with the same pair-tuple schema. Output remains bounded at 4,096 tokens. It stops early on completed or repetitive output. A skipped or invalid proposal does not prevent deterministic fallback. The preview separately limits upload size and canvas dimensions. See [rules](RULES.md), [FAQ](FAQ.md), and [implementation notes](docs/IMPLEMENTATION.md) for the full contract.

## Formats and testing

The viewer offers square 1080×1080, portrait 1080×1350, story 1080×1920, widescreen 1920×1080, web banner 1600×400, leaderboard 728×90, skyscraper 160×600, and custom dimensions. A historical official-sample stress run covered the three supplied levels across six of these formats: square, portrait, story, web banner, leaderboard and skyscraper. The 1920×1080 preset was not part of that 18-case run. Dataset tests and locally generated review images are not included in Git.

```bash
python -m pytest -q
```

Historical audits, limitations and replay instructions are in [rigid artwork review](docs/RIGID_ART_REVIEW.md), [macro structure review](docs/MACRO_STRUCTURE_REVIEW.md), and [model evaluation and training](docs/QWEN_LORA_COLAB.md). Prior replay results are checkpoints, not evidence that a later code change has passed the same run.

## Current visual regression review

The single local review viewer combines the **fixed 38 cases (18 official, 8 original alternates, 12 complex dataset cases) plus 8 fresh independent conversions and 13 supplied/skyscraper stress conversions**. Every case has Qwen enabled and shows only Original and Resized. After major engine changes, refresh all cases together and rebuild the Gemini packet:

```bash
.venv/bin/python scripts/refresh_regression_viewer.py --workers 2
.venv/bin/python scripts/refresh_regression_viewer.py --check
.venv/bin/python scripts/package_regression_review.py
# Existing local review server: http://127.0.0.1:8766/review/main_engine_qwen_outputs/index.html
```

This development workflow needs the local Crello assets/fonts, weights and saved prediction reports named in `config/regression_review.json`; these large research artifacts are not bundled with Git. It replays exact recorded Qwen predictions to isolate engine changes, and does not measure fresh generation speed. Incomplete runs do not replace the current viewer. See [diagnosis, regression workflow and next steps](docs/REGRESSION_REVIEW_AND_NEXT_STEPS.md). The existing 1.7B A/B did not justify promotion; no additional training is needed to fix the current post-processing alignment defect. The visual backbone is SigLIP2; the human-labelled structural-ranker pilot is recorded separately and is not deployed.

## Data and artifacts

`brand_kit/`, the renderer, schemas, sample inputs, source code and tests are included. Downloaded models, adapters, Crello data, generated reviews and outputs stay local and are ignored by Git; they are not needed for deterministic use of the official samples. Local dataset tools use CyberAgent's [Crello dataset](https://huggingface.co/datasets/cyberagent/crello); see [dataset and training notes](docs/CRELLO_DATA_AND_TRAINING.md) for attribution and sharing considerations.

## Current upgrade experiments and human preference labels

[Implementation checklist](docs/ENGINE_UPGRADE_CHECKLIST.md) records the accepted,
already implemented, experimental and human-gated recommendations. Fresh dense
A/B generation is separate from cached engine regression replay. No public
adapter upload or ranking-head training is performed without its prerequisites.

To prepare the selected adapter privately for an offline local Docker test:

```bash
docker build --platform linux/amd64 --target runtime -t magicanvas:base .
.venv/bin/python scripts/prepare_submission_context.py
docker build --platform linux/amd64 -t magicanvas:submission data/submission_context
docker run --rm --network none -v "$PWD/input:/app/input" -v "$PWD/output:/app/output" magicanvas:submission
```

The ignored local context contains weights; do not commit or upload it. This is
local testing only: final rules still require publicly accessible weights. The
public `submission` target supports `QWEN_MODEL`, `QWEN_MODEL_REVISION`,
`QWEN_ADAPTER_REPO` and `QWEN_ADAPTER_REVISION`; base and adapter must match.

The first occupancy-only review is preserved, but does not establish general
ranking quality. The current [preference campaign plan](docs/PREFERENCE_CAMPAIGN_PLAN.md)
uses randomized comparisons of actual Qwen-enabled candidates, including new
20–22-element authored designs, six themes, and all six tested target formats.
Related palette variants stay in the same family. Review the live queue at
`http://127.0.0.1:8766/review/preferences_campaign/review.html`; export labels
from the page. Original/resized regression inputs are excluded from preference
training. Acceptable ties, “neither is good” and repeated consistency checks
are kept separate from training preferences.

```bash
.venv/bin/python scripts/import_preference_labels.py --labels /path/to/siglip-campaign-preferences-v3.jsonl
.venv/bin/python scripts/train_siglip_ranker.py --manifest data/siglip_reviewed_corpus/manifest.jsonl --approvals data/siglip_reviewed_corpus/approvals.jsonl --check-only
```

Import validates exact image hashes and randomized-side mapping. It combines
eligible real labels locally, without inventing approvals. Training uses frozen
SigLIP2 features and independent family folds: at least 20 train, five validation
and five test families. Validation selects the epoch; test is evaluated only
after fitting. Promotion requires beating zero-shot on both. A diagnostic
`--pilot --output data/siglip_ranker_experiments/pilot.pt` never consumes test
labels or writes production weights. The completed 168-slot export yields 178 eligible preferences across all
three campaigns. The complete training run scores 9/17 validation preferences,
equal to zero-shot, so it fails promotion. No ranking head is saved or deployed;
that image head never consumed the reserved test pairs. A subsequent structural
ranker study evaluated 37 verified test pairs after excluding five exact-pixel
reconstruction failures across the folds. Those test outcomes are now consumed
and must not be reused as a sealed promotion set. The earlier 44-label pilot (8/15
versus 7/15) is historical.

Request-local asset discovery is cached exactly and reset between adaptations,
including errors; it does not cache geometry across requests. Set
`MAGICANVAS_ASSET_LOOKUP_CACHE=0` for a diagnostic uncached comparison. All 38
frozen regression documents, winners and rendered pixel buffers remained
identical after enabling it. Reduced filesystem lookup counts are verified;
no portable end-to-end speedup is promised.

Hardware overrides are opt-in: `MAGICANVAS_QWEN_DEVICE=auto|cpu|mps|cuda`,
`MAGICANVAS_QWEN_DTYPE=auto|float32|float16|bfloat16`,
`MAGICANVAS_VISION_DEVICE=cpu|auto|mps|cuda`, `MAGICANVAS_CPU_THREADS=N` and
`MAGICANVAS_RENDER_WORKERS=1|2`. Vision stays on CPU and renderer concurrency
stays at one by default until local measurements justify changing them. Linux
Docker uses CPU, not the macOS host's Metal GPU. No fixed speedup is promised.

The generic attachment fix keeps a growing text backing outside adjacent media.
Measured container packing now reflows text and native backings before allocating
columns, keeps peer badges together, and can place a source-adjacent caption
beside its photo. `qwen-macro-band-flow` offers model-driven macro grouping with
measured child geometry; raw Qwen and existing candidates remain available.
Unsupported decorative groups retain their arrangement. Raster clipping remains
a hard gate even when a separate readability warning is present.

The guarded 38-case replay preserves all selected output geometry and winners,
with no new automated flags. Both diagnosed dense leaderboards now select a
larger measured layout. Additional replays cover 15 distinct human-rejected
source/format tasks; many still look weak despite passing automated checks.
**321 tests and 34 subtests pass.** These development checks are not a new full
89/309 release sweep, fresh model inference, or evidence of aesthetic perfection.
See [audit decisions](docs/DENSE_REFLOW_AUDIT.md),
[attachment examples](http://127.0.0.1:8766/review/attachment_growth/index.html),
and [reviewed outliers](http://127.0.0.1:8766/review/reviewed_outliers/index.html).
Adapter publication and the final adapter-required Docker test remain deferred.

## Measured retrieval, ranking and preview experiments

See [runtime and experiment results](docs/RUNTIME_AND_EXPERIMENTS.md).
The local optional human-zone library, Hungarian matching and dual-granularity
macro alternatives are available with `MAGICANVAS_EXPERIMENTAL_MACRO_LIBRARY=1`;
they must earn selection through existing safety/improvement gates. Build the
training-derived library locally with `scripts/build_layout_library.py` (requires
the local corpus; the geometry library is not redistributed). These experimental
families are off by default until they demonstrate a useful quality gain.

A ten-feature human-calibrated ranker improved aggregate pair accuracy but is
**not deployed**: its actual-engine validation subset only tied the baseline.
No Qwen retraining or model-default change is implied by that pilot.

The interactive preview caches exact repeated requests (at most eight / 32 MiB)
and labels cache hits. New designs/formats still run Qwen. One actual local HTTP
check took 42.12 seconds fresh versus 9 ms for the identical cached repeat; this
is not a fresh-generation speed claim. Set `MAGICANVAS_PREVIEW_CACHE=0` for speed
benchmarks. Native Metal automatically merges LoRA in memory to remove per-token adapter
dispatch without editing saved weights. `MAGICANVAS_QWEN_MERGE_ADAPTER=0` keeps the
unmerged comparison path; CPU/CUDA defaults are unchanged.
Restart the preview server after engine/font/weight changes.

## Native Apple Silicon / CUDA / CPU runtime

No M5-specific flag is required for a native macOS run. Actual Torch backend
capabilities select Metal/MPS FP16 with inference-only LoRA merging, CUDA BF16
when supported (FP16 otherwise), or portable CPU FP32. ARM64 alone does not
imply GPU availability. Check the detected configuration without loading weights:

```bash
.venv/bin/python scripts/runtime_diagnostics.py
.venv/bin/python viewer/server.py
# http://127.0.0.1:5050 — fresh results show the actual device and phase timings
```

On the available M1 Pro, eight fresh new-design conversions took 303.35 seconds
combined versus 331.74 seconds unmerged (about 8.6% less); all eight final JSON
outputs were unchanged, although five raw completions differed. This is one
development comparison, **not an M5 Pro measurement** or universal optimum.

Linux Docker on an M5 Pro cannot access Metal/MPS. The required Linux/amd64
container uses the portable CPU path on that host; native macOS and emulated
Docker timings must be reported separately. See [runtime, experiments and skyscraper test scope](docs/RUNTIME_AND_EXPERIMENTS.md).

## Dense prompts and the optional 1.7B model

See [capacity and model decisions](docs/PROMPT_CAPACITY_AND_MODEL_DECISIONS.md).
The 1.7B base and its matching local adapter are supported through the same
interface; a larger model still shares the engine's context and safety limits.
To select it explicitly for native preview after stopping the running server:

```bash
MAGICANVAS_QWEN_ENABLED=1 \
MAGICANVAS_QWEN_DIR=models/qwen3-1.7b \
MAGICANVAS_QWEN_ADAPTER_DIR=models/qwen3-layout-lora-v2-expanded-1.7b \
.venv/bin/python viewer/server.py --port 5050
```

Both directories must exist locally. This does not download or publish weights.
The production default remains subject to measured quality and runtime gates.

The latest [nested reflow audit](docs/NESTED_REFLOW_AUDIT.md) documents regional
backing scopes, repeated row/card ownership and measured container sizing.
The unified viewer can filter changed outputs, current flags and newly added
flag types while retaining just Original and Resized panels. These are frozen
Qwen-enabled development replays; inspect native-size renders and audit JSON
before treating an unflagged output as visually good.

The [Gemini follow-up decisions](docs/GEMINI_FOLLOWUP_DECISIONS.md) cover all 25
review proposals, including measured variable-width peer wrapping, button/caption
ownership and alignment-safe axis snapping. Larger model/ranker/crop experiments
remain gated; the review packet records accepted changes and residual defects.

The [upgrade exploration and evaluation plan](docs/UPGRADE_EXPLORATION_AND_EVALUATION.md)
covers the subsequent model, decoding, solver, runtime and product suggestions,
including a current primary-source model search. It distinguishes implemented
features from proposed experiments and records explicit comparison/promotion
gates. The default engine is unchanged by that research. Model metadata can be
audited without loading weights using `scripts/probe_upgrade_readiness.py`;
`--refresh-hub` fetches public metadata only.

Model swapping selects a **base model and its matching adapter together**. The
0.6B LoRA cannot be reused on the 1.7B or a different architecture. Both existing
packages can be checked offline with `.venv/bin/python scripts/check_model_profiles.py`;
`config/layout_model_profiles.json` is their audit manifest, not an active runtime
selector. The existing environment settings above select a package at process
startup, so restart the preview server when changing models. The research plan
documents broader provider support, pending provenance pins and training needed
for a different base.

Executed opt-in model/algorithm pilots are documented in
[upgrade pilot results](docs/UPGRADE_PILOT_RESULTS.md) and the
[Original/Resized pilot viewer](http://127.0.0.1:8766/review/upgrade_pilots/index.html).
They include pinned Coder/Qwen3.5/Qwen3-VL/NaFlex downloads, exact-inventory
grammar decoding, bounded CP-SAT group packing and a tiled critic comparison.
New bases are untuned; existing trained adapters are never attached across
architectures. Pilot dependencies and weights stay local, and none of these
experiments automatically changes the production model or safety gates.

The follow-up adopted a source-owned rotated-label fit, with 391 passing tests
and 34 passing subtests. All 59 frozen regression outputs remain byte-identical
to the prior engine; the measured improvement is in a previously rejected
intermediate proposal. Three decodes per difficult task brought no final gain
in a four-task pilot and remain opt-in research. See the results document for
remaining defects, exact scope and training options; no new training is needed
to run the current engine.

The [expanded no-training comparison](docs/TRAINING_FREE_SCALE_RESULTS.md)
evaluates 134 distinct resize tasks, covering all 89 dataset test pairs, and
178 existing human preferences. It adds optional `MAGICANVAS_SIGLIP_TILED=1`
scoring with the existing local vision weights; the default remains global
scoring. The measured gain is small and concentrated in a few families, so it
is not an automatic upgrade. No new training, labels or downloads are needed
to use that option; restart the preview server after changing it. New
Original/Resized comparisons are in the
[algorithm viewer](http://127.0.0.1:8766/review/training_free_scale/index.html) and
[selector viewer](http://127.0.0.1:8766/review/training_free_selectors/index.html).

The [surgical algorithm decision ledger](docs/SURGICAL_ALGORITHM_DECISIONS.md)
records the later font, topology, boundary, micro-banner, selective-critic and
training-target investigations, including negative results and unfinished work.
Its [research viewer](http://127.0.0.1:8766/review/surgical_followup/index.html)
shows experimental proposals separately from the unchanged production outputs.
