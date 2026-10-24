# Expanded Qwen v2 held-out review

Open `review.html` in a browser. This viewer was generated with `models/qwen3-layout-lora-v2-expanded` and prompt version 2. It shows eight examples selected across aspect-ratio buckets from the 89-row held-out test split.

The page compares the original dataset preview, local target render, raw Qwen proposal, intermediate repairs, Qwen-off engine output, and final Qwen-on engine output. The Qwen-off path is the deterministic engine baseline; Qwen-on loads the expanded adapter.

`review_summary.json` reports aggregate metrics for these eight examples. This is a visual spot check, not the full held-out adapter evaluation. The model directory remains a candidate; the repository default adapter was not changed by this review.
