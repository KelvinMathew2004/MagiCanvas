# Documentation map

## Evaluator entry points

- [Project README](../README.md): Docker command, input/output contract, model limits and algorithm summary.
- [EVALUATION.md](EVALUATION.md): pinned public build, private local-adapter build and verification commands.
- [IMPLEMENTATION.md](IMPLEMENTATION.md): current architecture and model-swapping boundary.
- [STARTER_INTEGRITY.md](STARTER_INTEGRITY.md): exact comparison with the organizer's starter files.

## Training and development references

These documents describe experiments and development cohorts, not hidden-test
scores or promises about the submitted model.

- [TRAINING_RUNBOOK.md](TRAINING_RUNBOOK.md) and [TRAINING_CONTINUATION.md](TRAINING_CONTINUATION.md): current VL-2B-only Colab instructions.
- [VLM_TRAINING_VERIFICATION.md](VLM_TRAINING_VERIFICATION.md): image/masking checks and remaining training limits.
- [MODEL_TRAINING_CANDIDATES.md](MODEL_TRAINING_CANDIDATES.md), [RECOVERED_ADAPTER_EVALUATION.md](RECOVERED_ADAPTER_EVALUATION.md) and [POST_TRAINING_EVALUATION_PLAN.md](POST_TRAINING_EVALUATION_PLAN.md): challenger research, rejected retraining and promotion gates.
- [SURGICAL_ALGORITHM_DECISIONS.md](SURGICAL_ALGORITHM_DECISIONS.md), [ALGORITHM_FOLLOWUP.md](ALGORITHM_FOLLOWUP.md) and [INDUSTRY_ALGORITHM_DECISIONS.md](INDUSTRY_ALGORITHM_DECISIONS.md): dated algorithm experiments, including rejected changes.
- [CRELLO_DATA_AND_TRAINING.md](CRELLO_DATA_AND_TRAINING.md): dataset provenance and sharing boundaries.
- [REVIEW_ARCHIVE.md](REVIEW_ARCHIVE.md): local artifact recovery inventory.

## Local research archive

`docs/archive/` contains superseded reports and notebooks. It is retained locally,
ignored by Git and not part of the evaluator-facing checkout. Older numbers
are dated experiment records, not current release claims. Training data, weights
and review renders under `data/` also stay local.

Production remains Qwen3-0.6B with the existing v2 LoRA. The recovered 0.6B
retraining was not promoted. VL-2B is an experiment, not the installed default.
Public adapter publication and final offline Docker verification remain pending.
