# Gemini review: expanded Qwen v2 adapter

Please independently review the latest eight-case held-out visual output and the included Qwen inference, refinement, constraint, training, and evaluation code. Focus on whether the expanded adapter is being used as intended, whether the eight examples support the claims in `evidence/review_summary.json`, and whether any implementation or evaluation issue should be fixed before a full 89-example A/B run.

Open `review/review.html` first. It is self-contained with the selected dataset previews and generated candidate images. The candidates were rendered locally on 8 stratified examples from the 89-row test split. Qwen-on uses `models/qwen3-layout-lora-v2-expanded`; Qwen-off is the deterministic engine baseline.

The sample viewer is qualitative and does not establish held-out aggregate superiority. Do not treat the included test output as passing: three layout-mechanics tests currently fail, detailed in `evidence/test_run.txt`. Adapter weights are intentionally omitted; `code/adapter/adapter_config.json` and its README describe the candidate.
