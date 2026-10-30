# MagiCanvas

An offline design-resizing engine for the Chameleon challenge. **Generate → Repair → Verify:** a small Qwen LoRA proposes layouts; source-aware structural reflows provide alternatives; rendered text and geometry checks select the result. Element IDs, copy, font families and original assets are preserved. No cloud inference or image generation.

## Docker — stage-one evaluation

**Public LoRA upload is pending.** The final build requires the selected adapter's public Hugging Face repository and full commit SHA; replace both placeholders below after publication. It fails rather than silently evaluating the pretrained base without our adapter.

```bash
docker build --platform linux/amd64 --target submission \
  --build-arg QWEN_MODEL_REVISION=c1899de289a04d12100db370d81485cdf75e47ca \
  --build-arg QWEN_ADAPTER_REPO=OWNER/REPOSITORY \
  --build-arg QWEN_ADAPTER_REVISION=FULL_40_CHARACTER_COMMIT \
  -t magicanvas .
docker run --rm --network none \
  -v "$PWD/input:/app/input" -v "$PWD/output:/app/output" magicanvas
```

Run from the repository root. Each `input/*.json` task contains `source` and `target_canvas`; results are `output/<task>_output.json` and `_output.png`. Dependencies and weights download at **build time only**. The supplied runner, renderer, schema, official samples and brand kit are unchanged. Level 3's missing holiday asset is intentional.

Until publication, the [private local-adapter Docker instructions](docs/EVALUATION.md) reproduce the trained engine from an existing checkout. A final offline Docker build/run is still to be verified; do not treat the base-only development image as the submitted model.

## Optional: native Python and preview

Requires Python 3.10+ and the selected adapter installed at `models/qwen3-layout-lora-v2/`. This path does not require Docker.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/fetch_qwen.py --model Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca
python scripts/fetch_siglip.py
export MAGICANVAS_QWEN_ENABLED=1 MAGICANVAS_REQUIRE_QWEN_ADAPTER=1
python -m engine.main
# Optional interactive preview:
python viewer/server.py --port 5050
```

Open **http://127.0.0.1:5050** for the preview. Adapter setup and verification: [EVALUATION.md](docs/EVALUATION.md).

## Model operation limits

| Item | Current implementation |
|---|---|
| Generator | Lightweight **Qwen3-0.6B** plus its matching v2 LoRA; greedy, non-thinking inference |
| Visual ranking | Local SigLIP2 Base (~375M parameters), bounded tie-breaker after defect checks |
| Token budget | Combined context ≤8,192; prompt ≤6,000 with an inventory-sized output reserve; output ≤4,096 |
| Hardware | Designed for local Apple Silicon, including M5 Pro: native macOS uses Metal FP16. CUDA uses BF16/FP16; CPU uses FP32. **Linux/amd64 Docker on Mac uses CPU, not Metal.** Speed varies with design density; no M5 Pro Docker latency guarantee is claimed |
| Offline behavior | Weights loaded locally; no runtime downloads or API calls. Dense/extreme layouts can still lose readability |
| Swapping models | Set `MAGICANVAS_QWEN_DIR` and `MAGICANVAS_QWEN_ADAPTER_DIR` to a compatible base/adapter pair and restart. Supported text-model loading includes Qwen3-1.7B and Qwen2.5-1.5B-Instruct; each needs its own adapter. The VL-2B image-aware provider remains experimental, not the production default |

The brief leaves exact RAM, permitted-model and model-size limits to organizer confirmation. Tests cover **1080×1080 square, 1080×1350 portrait, 1080×1920 story, 1600×400 web banner, 728×90 leaderboard and 160×600 skyscraper**; preview also supports 1920×1080 and custom sizes.

## Implemented algorithms

- Source relationship graphs: nested containers, backing ownership, attachments and repeated sibling groups.
- Learned proposals plus measured stack/grid, macro-band, editorial/banner and source-proportional candidate reflows; bounded missing-ID recovery.
- Renderer-measured text fitting and tracking, shared-axis snapping, control padding and equal edge-to-edge peer gaps.
- Proportional artwork scaling, visible-alpha geometry, source edge anchors and safe margins.
- Deterministic placement constraints and whole-lockup contrast/clean-zone repairs that respect source-supported overlaps.
- Render-grounded ID/bounds, glyph-fit, collision and contrast checks; candidate-family quotas and bounded perceptual ranking.

[Architecture](docs/IMPLEMENTATION.md) · [Starter-file integrity audit](docs/STARTER_INTEGRITY.md) · [Evaluation](docs/EVALUATION.md) · [Training](docs/TRAINING_RUNBOOK.md) · [Official rules](RULES.md). Historical experiments are retained in a local, Git-ignored archive and are not current performance claims.
