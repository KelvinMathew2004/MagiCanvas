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

Qwen skips prompts exceeding 4,096 tokens, bounds output at 4,096 tokens and uses at most 8,192 combined tokens (or the model's smaller limit). It stops early on completed or repetitive output. A skipped or invalid proposal does not prevent deterministic fallback. The preview separately limits upload size and canvas dimensions. See [rules](RULES.md), [FAQ](FAQ.md), and [implementation notes](docs/IMPLEMENTATION.md) for the full contract.

## Formats and testing

The viewer offers square 1080×1080, portrait 1080×1350, story 1080×1920, widescreen 1920×1080, web banner 1600×400, leaderboard 728×90, skyscraper 160×600, and custom dimensions. A historical official-sample stress run covered the three supplied levels across six of these formats: square, portrait, story, web banner, leaderboard and skyscraper. The 1920×1080 preset was not part of that 18-case run. Dataset tests and locally generated review images are not included in Git.

```bash
python -m pytest -q
```

Historical audits, limitations and replay instructions are in [rigid artwork review](docs/RIGID_ART_REVIEW.md), [macro structure review](docs/MACRO_STRUCTURE_REVIEW.md), and [model evaluation and training](docs/QWEN_LORA_COLAB.md). Prior replay results are checkpoints, not evidence that a later code change has passed the same run.

## Current visual regression review

The single local review viewer combines **18 official conversions, 8 independent alternate-design conversions, and 12 complex dataset cases**. Every case has Qwen enabled and shows only Original and Resized. After major engine changes, refresh all cases together and rebuild the Gemini packet:

```bash
.venv/bin/python scripts/refresh_regression_viewer.py --workers 2
.venv/bin/python scripts/refresh_regression_viewer.py --check
.venv/bin/python scripts/package_regression_review.py
# Existing local review server: http://127.0.0.1:8766/review/main_engine_qwen_outputs/index.html
```

This development workflow needs the local Crello assets/fonts, weights and saved prediction reports named in `config/regression_review.json`; these large research artifacts are not bundled with Git. It replays exact recorded Qwen predictions to isolate engine changes, and does not measure fresh generation speed. Incomplete runs do not replace the current viewer. See [diagnosis, regression workflow and next steps](docs/REGRESSION_REVIEW_AND_NEXT_STEPS.md). The existing 1.7B A/B did not justify promotion; no additional training is needed to fix the current post-processing alignment defect. The visual backbone already is SigLIP2; preference-head training needs reviewed labels first.

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
the 39 reserved test pairs remain unscored. The earlier 44-label pilot (8/15
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
