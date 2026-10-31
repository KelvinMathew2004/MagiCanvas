# Preview tools

The scored engine is headless. These tools are for local review and the Express add-on.

## Local layout preview

From the repository root with dependencies and local model package installed:

```bash
.venv/bin/python viewer/server.py --port 5050
```

Open http://127.0.0.1:5050. Select an included sample or upload a Chameleon task/design JSON; choose a target size and click **Adapt layout**. The two panels show the original and returned design through the real renderer. Download the output JSON/PNG or Express comparison bundle. Restart after engine, model or font changes.

- Qwen is enabled when local weights are available. The adapter must match the selected base. Missing required weights should be diagnosed before judging model quality.
- First runs include model loading. Exact repeated requests can use an explicitly labelled cache hit; a hit is not fresh generation. Set `MAGICANVAS_PREVIEW_CACHE=0` for timings.
- Upload guards: 4 MB JSON, 1–500 elements, dimensions 1–12,000 px, at most 40 million pixels. These apply to the development preview, not a promised engine capacity.
- Demo assets/fonts resolve from `brand_kit/`. Local dataset previews use their matching asset roots and `data/crello_fonts.pickle`. No fonts or weights download during adaptation.
- Rendering errors appear explicitly. The Level 3 holiday badge is intentionally missing; its placeholder should remain.

## Unified development regression viewer

http://127.0.0.1:8766/review/main_engine_qwen_outputs/index.html

This is the existing static review server, separate from port 5050. It shows all 59 official/alternate/Crello/stress conversions with Original/Resized panels, group filters and audit details. Refresh instructions and evidence limits are in [evaluation](../docs/EVALUATION.md). It replays recorded Qwen predictions; it does not measure fresh generation speed.

## Adobe Express add-on

```bash
cd viewer/express
npm ci
npm run build
npm run start
```

Enable Adobe Express Add-on Development and test the HTTPS URL printed by the CLI (normally https://localhost:5241). Load a comparison bundle from the preview or matching output JSON/PNG. Inserting the preview adds a flattened PNG, not editable scene elements. Run `npm run package` to package it.

The Express host requires a live Adobe/browser session; the engine remains offline. A successful build is not an in-Express acceptance test. The live sideload/insertion check remains to be performed.
