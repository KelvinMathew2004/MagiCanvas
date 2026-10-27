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

The build needs internet access to install dependencies and fetch public Qwen3-0.6B and SigLIP base weights. The resulting container uses local files without network access:

```bash
docker build --platform linux/amd64 -t magicanvas .
docker run --rm --network none \
  -v "$(pwd)/input:/app/input" -v "$(pwd)/output:/app/output" magicanvas
```

The default Docker build runs the **base Qwen model without a LoRA adapter**. The fine-tuned adapter is not included in Git and will be published to Hugging Face for final submission; no public adapter repository has been configured yet. Once available, bake it in at build time (a revision pins the artifact):

```bash
docker build --platform linux/amd64 \
  --build-arg QWEN_ADAPTER_REPO=OWNER/ADAPTER_REPO \
  --build-arg QWEN_ADAPTER_REVISION=REVISION -t magicanvas .
```

For deterministic-only comparison add `-e MAGICANVAS_QWEN_ENABLED=0` to `docker run`. Adapter-free and fine-tuned output can differ. Model loading falls back to deterministic candidates if the model or adapter cannot be loaded; inspect diagnostics when evaluating model performance.

## Architecture and trade-offs

The engine infers source relationships and functional containers, then generates learned Qwen proposals and deterministic candidates including proportional, semantic/editorial and, when appropriate, macro-band reflows. Source relationships guide repair and attachment preservation. A renderer measures text fit, overlaps, contrast and other defects before selection; SigLIP is an optional bounded tie-breaker. The checks are safeguards, not a guarantee of a flawless or feasible layout for every dense or extreme format. Existing intentional overlaps can be preserved. The engine preserves artwork proportions; native panels, backgrounds and dividers may resize as layout surfaces.

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

## Data and artifacts

`brand_kit/`, the renderer, schemas, sample inputs, source code and tests are included. Downloaded models, adapters, Crello data, generated reviews and outputs stay local and are ignored by Git; they are not needed for deterministic use of the official samples. Local dataset tools use CyberAgent's [Crello dataset](https://huggingface.co/datasets/cyberagent/crello); see [dataset and training notes](docs/CRELLO_DATA_AND_TRAINING.md) for attribution and sharing considerations.
