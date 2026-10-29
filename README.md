# MagiCanvas

Resizes editable Chameleon designs into new formats using **Generate → Repair → Verify**: Qwen proposes a layout; source containers and measured reflows offer alternatives; the supplied renderer checks text fit, collisions, contrast and artwork proportions before selection. IDs, copy, fonts and original assets are preserved. No external API or image generation is used.

## Run locally

Requires Python 3.10+. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/fetch_qwen.py --model Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca
python scripts/fetch_siglip.py
MAGICANVAS_QWEN_ENABLED=1 python -m engine.main
```

Put `{ "source": { "canvas": ..., "elements": [...] }, "target_canvas": ... }` tasks in `input/`. Each produces `output/<task>_output.json` and `_output.png`. Three official samples and their assets/fonts are included. The missing holiday badge in Level 3 is intentional.

**LoRA publication to Hugging Face is pending.** With `models/qwen3-layout-lora-v2/` installed, the command uses our fine-tuned adapter. A fresh clone without it uses the pretrained base and can produce different results. Final submission will require the published, pinned adapter; it must not silently fall back to the base. Training datasets and weights are not stored in Git.

## Docker: evaluator run

The currently runnable feedback build uses the public **pretrained base**:

```bash
docker build --platform linux/amd64 --target runtime -t magicanvas .
docker run --rm --network none \
  -v "$PWD/input:/app/input" -v "$PWD/output:/app/output" magicanvas
```

All dependencies and model weights download at **build time**; runtime is offline. For a faithful local test with our installed adapter, or the final public adapter-required build, use the short [evaluation instructions](docs/EVALUATION.md). The default `submission` Docker stage intentionally refuses to build until the public adapter repository and pinned commit are supplied. Final offline Docker verification remains pending; the local Docker daemon was unavailable.

## Model operation limits

| Item | Current behavior |
|---|---|
| Layout model | Qwen3-0.6B + matching v2 LoRA; non-thinking generation |
| Visual tie-breaker | SigLIP2 Base, ~375M total parameters; cannot override hard defect gates |
| Context | At most 8,192 combined tokens; prompt ≤6,000 with inventory-sized output reserve; output ≤4,096 |
| Dense inputs | Bounded relationship compaction and missing-ID recovery; no guarantee all dense/extreme layouts are readable |
| Runtime | Local weights only. No inference-time downloads or cloud services |
| Hardware | Native Mac: Metal FP16. CUDA: BF16/FP16. CPU: FP32. Linux/amd64 Docker on Mac uses CPU, **not Metal** |
| Model swapping | Set `MAGICANVAS_QWEN_DIR` and `MAGICANVAS_QWEN_ADAPTER_DIR` to a compatible base/adapter pair, then restart. A LoRA is not portable between different bases |
| Official limits | Brief names M5 Pro, but exact RAM/model-size/approved-model limits remain to be confirmed. No hard stage-one time limit is specified |

## Preview and verification

```bash
python viewer/server.py --port 5050
# Open http://127.0.0.1:5050
python -m pytest -q
```

Tested formats: square **1080×1080**, portrait **1080×1350**, story **1080×1920**, web banner **1600×400**, leaderboard **728×90**, skyscraper **160×600**. The preview also offers widescreen **1920×1080** and custom sizes. Its upload limits are development guards, separate from the engine.

The local [unified regression viewer](http://127.0.0.1:8766/review/main_engine_qwen_outputs/index.html) shows 59 official, independent alternate and complex Crello/stress conversions, Original/Resized only. It replays recorded Qwen predictions through current code; it is **not fresh inference or an unseen benchmark**. One known dense 320×50 conference output remains flagged. Zero flags does not mean a perfect design.

[Express add-on](viewer/README.md) · [Architecture](docs/IMPLEMENTATION.md) · [Training recipe](docs/TRAINING_RUNBOOK.md) · [Six model candidates](docs/MODEL_TRAINING_CANDIDATES.md) · [Evaluation and review refresh](docs/EVALUATION.md) · [Research decisions and evidence](docs/README.md) · [Official rules](RULES.md)
