# Documentation map

Start with the root [README](../README.md). Current operational documents:

- [EVALUATION.md](EVALUATION.md): exact local/private/public Docker paths, release blockers and unified viewer refresh.
- [IMPLEMENTATION.md](IMPLEMENTATION.md): current Generate → Repair → Verify architecture and compatible model swapping.
- [MODEL_TRAINING_CANDIDATES.md](MODEL_TRAINING_CANDIDATES.md): six references/challengers, modality differences and current evidence.
- [TRAINING_RUNBOOK.md](TRAINING_RUNBOOK.md): ordered local preparation, Colab training and fresh model evaluation.
- [TRAINING_CONTINUATION.md](TRAINING_CONTINUATION.md): skip the completed 0.6B and resume the remaining challengers with frequent recovery saves.
- [RECOVERED_ADAPTER_EVALUATION.md](RECOVERED_ADAPTER_EVALUATION.md): recovered 0.6B package, fresh champion comparison and promotion evidence.
- [POST_TRAINING_EVALUATION_PLAN.md](POST_TRAINING_EVALUATION_PLAN.md): internal challenger gates and considered engine ablations after training.
- [VLM_TRAINING_VERIFICATION.md](VLM_TRAINING_VERIFICATION.md): full-bundle image/masking checks, historical weight identity and remaining training limits.
- [CRELLO_DATA_AND_TRAINING.md](CRELLO_DATA_AND_TRAINING.md): dataset provenance and sharing boundaries.
- [SURGICAL_ALGORITHM_DECISIONS.md](SURGICAL_ALGORITHM_DECISIONS.md): latest detailed research decisions, including negative results and remaining defects.
- [ALGORITHM_FOLLOWUP.md](ALGORITHM_FOLLOWUP.md): intact-container growth, render quotas and exact-statistic performance evaluation.
- [INDUSTRY_ALGORITHM_DECISIONS.md](INDUSTRY_ALGORITHM_DECISIONS.md): verified industry mechanisms, paragraph balancing and whole-group search; measured decisions and untested limits.
- [REVIEW_ARCHIVE.md](REVIEW_ARCHIVE.md): recoverable cleanup inventory.

- [Recovered champion training scripts](archive/champion_training_history/README.md): exact historical notebooks/exporters and a byte-identical Git weight match.

## Research history

Historical notes are preserved in [archive/](archive/) for evidence and future work. They are dated experiments, not current setup instructions. Numeric test counts, timings and model claims belong to their recorded snapshots.

- [Overall proposal ledger](archive/UPGRADE_EXPLORATION_AND_EVALUATION.md): all model/algorithm suggestions and what remains untested.
- [Model and decoding pilots](archive/UPGRADE_PILOT_RESULTS.md): matching-base comparisons, grammar, sampling and runtime limits.
- [134-task no-training study](archive/TRAINING_FREE_SCALE_RESULTS.md): retrieval, grouping, packing and visual tie-break experiments; no universal promotion.
- [Implementation checklist](archive/ENGINE_UPGRADE_CHECKLIST.md): accepted versus experimental versus human-gated work.
- [Previous complete project narrative](archive/PROJECT_HISTORY_2026-10-28.md): retained history, superseded by the concise root README.

## Current state

The production default remains Qwen3-0.6B with the existing v2 LoRA. The recovered completed 0.6B adapter was evaluated on 89+59 tasks and was not promoted: final reference box/font errors worsened. Read RECOVERED_ADAPTER_EVALUATION.md for gains, failures and provenance limits. The 59-case current-code replay retains one flagged dense micro-banner; no automatic flag is a guarantee of visual perfection. Training corpus, base models, adapters and required regression caches remain local. Hugging Face publication and a successful final adapter-required offline Docker run are pending.
