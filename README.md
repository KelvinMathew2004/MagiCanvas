# MagiCanvas

MagiCanvas resizes editable Chameleon designs. The engine returns both an
updated JSON design and a PNG rendered with the supplied Adobe starter renderer.
It preserves the original elements, IDs, text, and assets; it does not generate
or replace image content.

## Run locally

Requirements: Python 3.10+ and internet access for installing packages.

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m engine.main
```

The runner reads `input/*.json` and writes each result as JSON and PNG under
`output/`. The three supplied sample tasks are ready to run. The deterministic
layout engine works without a model; optional model weights must already be
available locally.

## Build and run the judging container

Docker is the recommended run path for a reproducible, offline model setup. The
build step needs internet because it installs dependencies and downloads the
SigLIP and Qwen base weights. Docker enables its baked Qwen model by default. The
engine makes no network requests at runtime.

```bash
docker build --platform linux/amd64 -t magicanvas .
docker run --rm --network none \
  -v "$(pwd)/input:/app/input" \
  -v "$(pwd)/output:/app/output" \
  magicanvas
```

To compare the judging container with Qwen disabled:

```bash
docker run --rm --network none \
  -e MAGICANVAS_QWEN_ENABLED=0 \
  -v "$(pwd)/input:/app/input" \
  -v "$(pwd)/output:/app/output" \
  magicanvas
```

Direct local CLI runs opt into Qwen explicitly. Pair the 1.7B candidate with its matching base model:

```bash
MAGICANVAS_QWEN_ENABLED=1 \
MAGICANVAS_QWEN_DIR=models/qwen3-1.7b \
MAGICANVAS_QWEN_ADAPTER_DIR=models/qwen3-layout-lora-v2-expanded-1.7b \
python -m engine.main
```

The standard local default remains `models/qwen3-layout-lora-v2` until the
1.7B candidate passes a fresh held-out A/B evaluation. The installed 1.7B
adapter is kept separately at `models/qwen3-layout-lora-v2-expanded-1.7b/`.

SigLIP breaks close ties after exact structural and raster defect checks when
its build-time weights are present; otherwise ranking is deterministic. Qwen can use a
configured Qwen checkpoint and optional public Hugging Face LoRA adapter. To
bake an adapter into the image, pass its repository at build time:

```bash
docker build --platform linux/amd64 \
  --build-arg QWEN_ADAPTER_REPO=OWNER/ADAPTER_REPO \
  --build-arg QWEN_ADAPTER_REVISION=REVISION \
  -t magicanvas .
```

For the task contract, constraints, and model/runtime caveats, see
[RULES.md](RULES.md), [FAQ.md](FAQ.md), and
[docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md). Adobe Q&A notes, including model and
evaluation guidance, are in [docs/ADOBE_ORGANIZER_QA.md](docs/ADOBE_ORGANIZER_QA.md).
Qwen currently limits input to 4096 tokens, output to 4096 tokens, and combined
context to 8192 tokens (or a smaller model limit). Invalid proposals fall back
to other engine candidates. Current implementation, verification, and training
readiness are documented in [docs/DAY4_5_AUDIT.md](docs/DAY4_5_AUDIT.md).

## Current layout and evaluation approach

The engine retains deterministic reflows and a bounded, font-fitted raw Qwen
proposal alongside source-container projection and supported shared-axis snaps.
Physical geometry/raster checks gate selection; heuristic unit gaps do not veto Qwen.
Font calibration uses rendered glyphs and the available local font bundle,
widens into available space before shrinking, and scales tracking with fonts.
Learned container candidates compete with independent deterministic aspect
reflows and proportional preservation. Hard defects apply to every family;
matching a preferred heuristic template is not a final selection requirement.

Evaluation runs batches and renders only flagged outliers for review. The current
stress suite covers 89 frozen Crello test pairs and the three official hackathon
inputs across six formats (107 cases). The earlier 398-case test/validation
mining report remains a historical baseline. Commands and the response-masked
Colab training recipe are in [docs/QWEN_LORA_COLAB.md](docs/QWEN_LORA_COLAB.md).

The cleaned training split has 1,122 unique target IDs, with 309 validation and
89 test rows and no family overlap between splits. Both larger base models are
available locally for later A/B training: Qwen3-1.7B and Qwen2.5-1.5B-Instruct.
The installed Qwen3-1.7B adapter was evaluated but did not pass the original
89-case promotion gate. Keep the existing 0.6B adapter as the default; the 1.5B
base remains available for a future controlled comparison.
The training recipe uses 6,144 tokens so complex examples are retained.

The latest selector fixes have a 12-case paired review with raw proposal
ablations and a separate 18-format official Qwen on/off comparison. See
[the current review packet](data/gemini_review_qwen_projection_fixes_2026-10-25/README_FOR_GEMINI.md)
and [the verification audit](docs/DAY4_5_AUDIT.md). These curated checks do not
replace a fresh full 89-case evaluation. The local viewer enables Qwen at
startup when local weights exist; an explicit `MAGICANVAS_QWEN_ENABLED=0` is respected.

Archived review snapshots and recovery details are listed in
[docs/REVIEW_ARCHIVE.md](docs/REVIEW_ARCHIVE.md).

The input/output contract is [chameleon_design_schema_v0.9.json](chameleon_design_schema_v0.9.json).
The original starter instructions are preserved in
[docs/STARTER_README.md](docs/STARTER_README.md).

## Development

```bash
python -m unittest discover -s tests -v
python -m engine.main
```

Interactive local preview: `.venv/bin/python viewer/server.py`, then open
http://127.0.0.1:5050. See [viewer/README.md](viewer/README.md) for the optional
Adobe Express viewer.

## Dataset attribution

Local data tools use CyberAgent's [Crello dataset](https://huggingface.co/datasets/cyberagent/crello).
Credit CyberAgent and the dataset authors when discussing this work. Follow the
dataset card and VistaCreate terms; source designs may contain third-party
rights. Downloaded and converted dataset files stay local and are not included
in this repository. See [docs/CRELLO_DATA_AND_TRAINING.md](docs/CRELLO_DATA_AND_TRAINING.md)
before sharing data or training artifacts.

Current engine repair evidence and repeatable replay commands are documented in [docs/INVENTORY_REPAIR_AUDIT.md](docs/INVENTORY_REPAIR_AUDIT.md).

The October 26 visual-contract revision adds source-inferred control padding, painted ribbon-body centering, foreground-card gutters, brand reading order, and stacked primary axes. Text/image occlusion now measures glyph pixels against image alpha instead of transparent image envelopes. See [docs/VISUAL_CONTRACT_REPAIRS.md](docs/VISUAL_CONTRACT_REPAIRS.md) for scope, replay commands, and limitations.

The earlier [source-evidence review](docs/SOURCE_EVIDENCE_REVIEW.md) documents crop-aware negative-space candidates, orientation-independent roles, measured semantic fallback, final glyph-axis calibration, and the expanded dataset viewers. That checkpoint covered 139 replayed outputs and 12 freshly inferred designs. The current verification is linked below.

The current [source-attachment review](docs/SOURCE_ATTACHMENT_REVIEW.md) records shared functional groups, cost-tie-only Qwen selection, rotated-caption calibration and 151 frozen-proposal regression outputs. Four independently authored designs now add eight actual model ON/OFF tests with different content, photographs, source formats and neutral IDs; the older Solstice-element variants remain regression fixtures. These new fixtures are never training data, and the viewer distinguishes raw predictions, learned-family winners and deterministic fallback.

The latest [macro-structure review](docs/MACRO_STRUCTURE_REVIEW.md) covers multi-child cards, photo/caption/badge ownership, edge-strip handling, measured macro layouts, subpanel text axes and conservative incumbent replacement. The model contract and adapter are unchanged; improved engine outputs are not credited as improved raw model predictions.
