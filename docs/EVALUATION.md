# Evaluation and local review

For stage-one setup, start with the Docker section of the root [README](../README.md).
The optional Python/preview commands there are development conveniences, not
the container evaluation entry point. [STARTER_INTEGRITY.md](STARTER_INTEGRITY.md)
records the exact comparison against the organizer's starter files.

## Final public submission — pinned champion adapter

The selected champion LoRA is published at `NeoliteNexus/magicanvas-qwen3-0.6b-lora`, commit `09216ebcf0260889f84d21a723ce4ecacdc58e5b`. The base is pinned too.

```bash
docker build --platform linux/amd64 --target submission \
  --build-arg QWEN_MODEL=Qwen/Qwen3-0.6B \
  --build-arg QWEN_MODEL_REVISION=c1899de289a04d12100db370d81485cdf75e47ca \
  --build-arg QWEN_ADAPTER_REPO=NeoliteNexus/magicanvas-qwen3-0.6b-lora \
  --build-arg QWEN_ADAPTER_REVISION=09216ebcf0260889f84d21a723ce4ecacdc58e5b \
  -t magicanvas .
docker run --rm --network none \
  -v "$PWD/input:/app/input" -v "$PWD/output:/app/output" magicanvas
```

This is the challenge's input/output mount convention. The submission stage verifies adapter compatibility and fails closed if the required adapter cannot load. Complete an offline run after the build to verify the final image.

## Same behavior with the existing private local adapter

From a checkout that already has `models/qwen3-layout-lora-v2` and the matching base:

```bash
docker build --platform linux/amd64 --target runtime \
  --build-arg QWEN_MODEL_REVISION=c1899de289a04d12100db370d81485cdf75e47ca \
  -t magicanvas:base .
.venv/bin/python scripts/prepare_submission_context.py
docker build --platform linux/amd64 -t magicanvas:local data/submission_context
docker run --rm --network none \
  -v "$PWD/input:/app/input" -v "$PWD/output:/app/output" magicanvas:local
```

The generated context contains local adapter weights and is ignored by Git. This private feedback path does not satisfy the final public-weight requirement. Do not distribute the context inadvertently.

## Model / hardware diagnostics

```bash
.venv/bin/python scripts/check_model_profiles.py
.venv/bin/python scripts/runtime_diagnostics.py
```

The engine reads local weights only. Native Mac Metal timings and Linux/amd64 CPU timings are different measurements. The reference is M5 Pro; exact memory/approved-model limits are not specified in the supplied PDF. Keep IDs, text, font families and original assets. The renderer's Level 3 missing-asset placeholder is intentional.

## Refresh all current review cases

Needs the local Crello data/fonts, base/adapter and frozen prediction reports; these are not distributed in Git or the Gemini ZIP.

```bash
.venv/bin/python scripts/refresh_regression_viewer.py --workers 2
.venv/bin/python scripts/refresh_regression_viewer.py --check
.venv/bin/python scripts/package_regression_review.py
```

Open http://127.0.0.1:8766/review/main_engine_qwen_outputs/index.html using the existing static review server. This single viewer includes all 18 official conversions, eight independent alternate conversions, 12 complex dataset conversions and 21 supplemental stress conversions. Qwen remains enabled; deterministic winners are named honestly. The replay isolates engine changes with exact saved predictions and does not measure model generation speed. Restart `viewer/server.py` after engine, font or model edits.

The wider 89 reference pairs and 309 validation rows complement the 59-case reviewed development suite. They have been used repeatedly and are not a sealed blind test. New adapter evaluation uses fresh inference: follow [TRAINING_RUNBOOK.md](TRAINING_RUNBOOK.md). Inspect winner changes and new defects; automated zero-flag results are not aesthetic approval.

The recovered completed 0.6B export can be evaluated independently of the two
pending Colab challengers:

```bash
.venv/bin/python scripts/evaluate_trained_layouts.py \
  --candidate-adapter models/qwen3-0.6b-latest/final
```

This compares 89 reference pairs plus 59 review tasks, keeps production
unchanged, and writes a separate Original/Resized model-comparison viewer.
Recovery provenance, interrupted-run continuation and measured results are in
[RECOVERED_ADAPTER_EVALUATION.md](RECOVERED_ADAPTER_EVALUATION.md).
